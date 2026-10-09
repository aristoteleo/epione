"""ArchR-style marker-feature and peak-to-gene heatmaps.

Ports of ArchR's ``plotMarkerHeatmap`` (``markerHeatmap``) and
``plotPeak2GeneHeatmap`` (``peak2GeneHeatmap``) — ArchR (MIT), Granja et al.
*Nat Genet* 2021. The matrix computations, binary row sorting, ``hclust``
column ordering, ``kmeans`` grouping and ``nPlot`` subsampling follow the R
source line by line (R's RNG / ``hclust`` / Hartigan-Wong ``kmeans`` are
re-implemented in :mod:`._heatmap_rcompat`), so ``return_matrix=True``
reproduces ArchR's ``returnMatrix = TRUE`` output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._heatmap_rcompat import (
    RRandom,
    r_dist,
    r_hclust,
    r_kmeans,
    r_col_means,
    r_kmeans_init,
    r_reorder_dendrogram,
    r_row_means,
)

# ---------------------------------------------------------------------------
# ArchR palettes (ArchRPalettes, in the order paletteContinuous interpolates)
# ---------------------------------------------------------------------------

ARCHR_PALETTES: Dict[str, List[str]] = {
    "solarExtra": ["#3361A5", "#248AF3", "#14B3FF", "#88CEEF", "#C1D5DC",
                   "#EAD397", "#FDB31A", "#E42A2A", "#A31D1D"],
    "blueYellow": ["#352A86", "#343DAE", "#0262E0", "#1389D2", "#2DB7A3",
                   "#A5BE6A", "#F8BA43", "#F6DA23", "#F8FA0D"],
}
# stallion in gtools::mixedsort(names) order, as used by paletteDiscrete().
ARCHR_STALLION_MIXEDSORT = [
    "#D51F26", "#272E6A", "#208A42", "#89288F", "#F47D2B", "#FEE500",
    "#8A9FD1", "#C06CAB", "#D8A767", "#90D5E4", "#89C75F", "#F37B7D",
    "#9983BD", "#D24B27", "#3BBCA8", "#6E4B9E", "#0C727C", "#7E1416",
    "#E6C2DC", "#3D3D3D",
]


def archr_cmap(name: str = "solarExtra", n: int = 256, reverse: bool = False):
    """``paletteContinuous(set = name)`` as a matplotlib colormap."""
    import matplotlib.colors as mcolors
    cols = list(ARCHR_PALETTES[name])
    if reverse:
        cols = cols[::-1]
    return mcolors.LinearSegmentedColormap.from_list(name, cols, N=n)


def _mixedsort_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def palette_discrete(values: Sequence[str]) -> Dict[str, str]:
    """``paletteDiscrete(values, set = "stallion")``."""
    import matplotlib.colors as mcolors
    vals = sorted(dict.fromkeys(str(v) for v in values), key=_mixedsort_key)
    pal = ARCHR_STALLION_MIXEDSORT
    if len(vals) > len(pal):
        cmap = mcolors.LinearSegmentedColormap.from_list("stallion", pal, N=len(vals))
        cols = [mcolors.to_hex(cmap(i)) for i in range(len(vals))]
    else:
        cols = pal[: len(vals)]
    return dict(zip(vals, cols))


# ---------------------------------------------------------------------------
# R-semantics helpers
# ---------------------------------------------------------------------------

def _row_vars(m: np.ndarray) -> np.ndarray:
    """``matrixStats::rowVars`` (1.x): double sums with the ``refine`` pass."""
    m = np.asarray(m, dtype=np.float64)
    k = m.shape[1]
    if k <= 1:
        return np.full(m.shape[0], np.nan)
    s = np.zeros(m.shape[0])
    for j in range(k):
        s += m[:, j]
    mu = s / k
    s = np.zeros(m.shape[0])
    for j in range(k):
        s += m[:, j] - mu
    mu = mu + s / k
    s2 = np.zeros(m.shape[0])
    for j in range(k):
        v = m[:, j] - mu
        s2 += v * v
    return s2 / (k - 1)


def _row_sds(m: np.ndarray) -> np.ndarray:
    """``matrixStats::rowSds``."""
    return np.sqrt(_row_vars(m))


def _r_log2(x: np.ndarray) -> np.ndarray:
    """``log2`` via the C library (numpy's SIMD log2 differs in the last ulp)."""
    import math
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    out = np.fromiter((math.log2(v) if v > 0 else (-np.inf if v == 0 else np.nan)
                       for v in flat), dtype=np.float64, count=flat.size)
    return out.reshape(x.shape)


def _r_col_sums(m: np.ndarray) -> np.ndarray:
    """``colSums`` (long-double accumulation)."""
    return np.asarray(m, dtype=np.float64).astype(np.longdouble).sum(axis=0).astype(np.float64)


def _row_zscores(m: np.ndarray) -> np.ndarray:
    """ArchR ``.rowZscores(limit = FALSE)``."""
    m = np.asarray(m, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (m - r_row_means(m)[:, None]) / _row_sds(m)[:, None]


def _r_order_logical(lmat: np.ndarray) -> np.ndarray:
    """``do.call(order, c(as.data.frame(lmat), decreasing = TRUE))`` (0-based).

    Per key R ranks TRUE < FALSE < NA (NA last); ``radix`` order is stable.
    ``lmat`` is float with 1/0/NaN.
    """
    keys = np.where(np.isnan(lmat), 2, np.where(lmat > 0, 0, 1)).astype(np.int8)
    # np.lexsort sorts by last key first and is stable.
    return np.lexsort(keys[:, ::-1].T)


def _r_order_decreasing(v: np.ndarray) -> np.ndarray:
    """``order(v, decreasing = TRUE)`` with NA last, stable."""
    v = np.asarray(v, dtype=np.float64)
    nan = np.isnan(v)
    idx = np.arange(len(v))
    key = np.where(nan, np.inf, -v)
    return idx[np.lexsort((idx, key))]


def _binary_sort(m: np.ndarray, lmat: Optional[np.ndarray] = None, *,
                 cut_off: float = 1.0, scale: bool = False,
                 cluster_cols: bool = True):
    """ArchR ``.binarySort``. Returns ``(row_idx, col_idx, hclust_or_None)``."""
    m = np.asarray(m, dtype=np.float64)
    if lmat is None:
        lm = _row_zscores(m) if scale else m
        lmat = (lm >= cut_off).astype(np.float64)
    else:
        lmat = np.asarray(lmat, dtype=np.float64)
    if cluster_cols:
        hc = r_hclust(r_dist(m.T), "complete")
        col_idx = hc.order0
    else:
        hc = None
        col_idx = np.arange(m.shape[1])
    row_idx = _r_order_logical(lmat[:, col_idx])
    return row_idx, col_idx, hc


# ---------------------------------------------------------------------------
# cutOff expressions (R syntax, three-valued logic)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""
    \s*(?:
      (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
     |(?P<name>[A-Za-z.][A-Za-z0-9._]*)
     |(?P<op>&&|\|\||<=|>=|==|!=|[<>&|!+\-*/^(),])
    )""", re.VERBOSE)


def _tokenize(expr: str):
    pos, out = 0, []
    expr = expr.strip()
    while pos < len(expr):
        mt = _TOKEN_RE.match(expr, pos)
        if not mt or mt.end() == pos:
            raise ValueError(f"cannot parse cutOff near {expr[pos:]!r}")
        pos = mt.end()
        kind = mt.lastgroup
        out.append((kind, mt.group(kind)))
    return out


def _na_and(a, b):
    out = np.where((a == 0) | (b == 0), 0.0, np.where(np.isnan(a) | np.isnan(b), np.nan, 1.0))
    return out


def _na_or(a, b):
    return np.where((a == 1) | (b == 1), 1.0, np.where(np.isnan(a) | np.isnan(b), np.nan, 0.0))


_FUNCS = {
    "abs": np.abs, "log2": np.log2, "log10": np.log10, "log": np.log,
    "exp": np.exp, "sqrt": np.sqrt,
    "is.na": lambda x: np.isnan(x).astype(np.float64),
}


class _CutoffParser:
    def __init__(self, tokens, env):
        self.t, self.i, self.env = tokens, 0, env

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, val=None):
        tok = self.peek()
        if val is not None and tok[1] != val:
            raise ValueError(f"expected {val!r}, got {tok[1]!r}")
        self.i += 1
        return tok

    def parse(self):
        v = self.or_()
        if self.i != len(self.t):
            raise ValueError(f"unexpected token {self.peek()[1]!r}")
        return v

    def or_(self):
        v = self.and_()
        while self.peek()[1] in ("|", "||"):
            self.take()
            v = _na_or(v, self.and_())
        return v

    def and_(self):
        v = self.not_()
        while self.peek()[1] in ("&", "&&"):
            self.take()
            v = _na_and(v, self.not_())
        return v

    def not_(self):
        if self.peek()[1] == "!":
            self.take()
            v = self.not_()
            return np.where(np.isnan(v), np.nan, np.where(v != 0, 0.0, 1.0))
        return self.cmp()

    def cmp(self):
        v = self.arith()
        op = self.peek()[1]
        if op in ("<", ">", "<=", ">=", "==", "!="):
            self.take()
            w = self.arith()
            with np.errstate(invalid="ignore"):
                r = {"<": np.less, ">": np.greater, "<=": np.less_equal,
                     ">=": np.greater_equal, "==": np.equal, "!=": np.not_equal}[op](v, w)
            return np.where(np.isnan(v) | np.isnan(w), np.nan, r.astype(np.float64))
        return v

    def arith(self):
        v = self.term()
        while self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            w = self.term()
            v = v + w if op == "+" else v - w
        return v

    def term(self):
        v = self.unary()
        while self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            w = self.unary()
            with np.errstate(divide="ignore", invalid="ignore"):
                v = v * w if op == "*" else v / w
        return v

    def unary(self):
        if self.peek()[1] in ("-", "+"):
            op = self.take()[1]
            v = self.unary()
            return -v if op == "-" else v
        return self.power()

    def power(self):
        v = self.atom()
        if self.peek()[1] == "^":
            self.take()
            v = np.power(v, self.unary())
        return v

    def atom(self):
        kind, val = self.take()
        if kind == "num":
            return np.float64(val)
        if kind == "name":
            if val in ("TRUE", "T"):
                return np.float64(1.0)
            if val in ("FALSE", "F"):
                return np.float64(0.0)
            if self.peek()[1] == "(":
                self.take("(")
                arg = self.or_()
                self.take(")")
                if val not in _FUNCS:
                    raise ValueError(f"unsupported function {val!r} in cutOff")
                with np.errstate(divide="ignore", invalid="ignore"):
                    return _FUNCS[val](arg)
            if val not in self.env:
                raise KeyError(f"cutOff refers to unknown assay {val!r}; "
                               f"available: {sorted(self.env)}")
            return np.asarray(self.env[val], dtype=np.float64)
        if val == "(":
            v = self.or_()
            self.take(")")
            return v
        raise ValueError(f"unexpected token {val!r}")


def eval_cutoff(expr: str, assays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Evaluate an ArchR ``cutOff`` string (R syntax) over assay matrices.

    Returns a float matrix of 1 (TRUE), 0 (FALSE) and NaN (NA) following R's
    three-valued logic (``NA & FALSE`` is FALSE, ``NA | TRUE`` is TRUE).
    """
    return _CutoffParser(_tokenize(expr), dict(assays)).parse()


# ---------------------------------------------------------------------------
# Marker container
# ---------------------------------------------------------------------------

_LONG_TO_ARCHR = {
    "mean_fg": "Mean", "mean_bg": "MeanBGD", "log2_fc": "Log2FC",
    "fdr": "FDR", "p_value": "Pval", "pct_fg": "PctFG", "pct_bg": "PctBGD",
    "z": "Z", "U": "U",
}
_PEAK_RE = re.compile(r"^(?P<chr>[^:\s]+)[:_-](?P<start>\d+)[-_](?P<end>\d+)$")


@dataclass
class MarkerAssays:
    """ArchR ``seMarker`` equivalent: ``assays`` are features × groups."""

    assays: Dict[str, pd.DataFrame]
    row_data: pd.DataFrame
    use_matrix: Optional[str] = None

    @property
    def groups(self) -> List[str]:
        return list(next(iter(self.assays.values())).columns)

    @property
    def n_features(self) -> int:
        return self.row_data.shape[0]


def as_marker_assays(markers, use_matrix: Optional[str] = None) -> MarkerAssays:
    """Normalise marker results into :class:`MarkerAssays`.

    Accepts a :class:`MarkerAssays`, an AnnData whose ``uns['markers']`` holds
    :func:`epione.tl.find_marker_features` output, the long-form DataFrame
    itself (columns ``group, feature, mean_fg, log2_fc, fdr, ...``) or a dict
    ``{'assays': {name: features × groups}, 'row_data': df, 'use_matrix': str}``.
    """
    if isinstance(markers, MarkerAssays):
        if use_matrix is not None:
            markers.use_matrix = use_matrix
        return markers
    try:
        from anndata import AnnData
        if isinstance(markers, AnnData):
            markers = markers.uns["markers"]
    except ImportError:  # pragma: no cover
        pass
    if isinstance(markers, Mapping):
        assays = {k: pd.DataFrame(v) for k, v in markers["assays"].items()}
        rd = markers.get("row_data")
        if rd is None:
            rd = pd.DataFrame({"name": list(next(iter(assays.values())).index)})
        return MarkerAssays(assays, pd.DataFrame(rd).reset_index(drop=True),
                            use_matrix or markers.get("use_matrix"))
    if isinstance(markers, pd.DataFrame) and {"group", "feature"} <= set(markers.columns):
        df = markers
        features = pd.unique(df["feature"])
        groups = list(pd.unique(df["group"].astype(str)))
        assays = {}
        for col, name in _LONG_TO_ARCHR.items():
            if col in df.columns:
                w = df.pivot(index="feature", columns="group", values=col)
                w.columns = w.columns.astype(str)
                assays[name] = w.reindex(index=features, columns=groups).astype(np.float64)
        parsed = pd.Series(features).astype(str).str.extract(_PEAK_RE)
        if parsed.notna().all().all():
            rd = pd.DataFrame({"seqnames": parsed["chr"],
                               "start": parsed["start"].astype(int),
                               "end": parsed["end"].astype(int)})
        else:
            rd = pd.DataFrame({"name": pd.Series(features).astype(str)})
        return MarkerAssays(assays, rd, use_matrix)
    raise TypeError("unsupported marker input; see as_marker_assays() docstring")


def _feature_labels(rd: pd.DataFrame) -> np.ndarray:
    if "name" not in rd.columns:
        return (rd["seqnames"].astype(str) + ":" + rd["start"].astype(str) + "-"
                + rd["end"].astype(str)).to_numpy(dtype=object)
    name = rd["name"].astype(str)
    if name.duplicated().any():
        return (rd["seqnames"].astype(str) + ":" + name).to_numpy(dtype=object)
    return name.to_numpy(dtype=object)


# ---------------------------------------------------------------------------
# plotMarkerHeatmap
# ---------------------------------------------------------------------------

@dataclass
class MarkerHeatmapResult:
    """Everything ``plotMarkerHeatmap`` computes before drawing."""

    matrix: pd.DataFrame                 # == returnMatrix=TRUE output
    pass_matrix: pd.DataFrame            # cutOff evaluation for kept rows
    label_markers: List[Optional[str]]   # labelMarkers vector
    label_index: np.ndarray              # mn (0-based rows/cols of `matrix`)
    row_order: np.ndarray                # display order of `matrix` rows (0-based)
    col_order: np.ndarray                # display order of `matrix` columns (0-based)
    transpose: bool
    pal: List[str]
    use_matrix: Optional[str]
    top_features: Dict[str, List[str]] = field(default_factory=dict)
    col_hclust: Any = None
    row_hclust: Any = None


def marker_heatmap_matrix(
    markers,
    cut_off: str = "FDR <= 0.01 & Log2FC >= 0.5",
    log2_norm: bool = True,
    scale_to: float = 1e4,
    scale_rows: bool = True,
    plot_log2fc: bool = False,
    limits: Tuple[float, float] = (-2, 2),
    grep_exclude: Optional[str] = None,
    pal: Optional[Sequence[str]] = None,
    binary_cluster_rows: bool = True,
    cluster_cols: bool = True,
    subset_markers: Optional[Sequence[int]] = None,
    label_markers: Optional[Sequence[str]] = None,
    n_label: int = 15,
    n_print: int = 15,
    transpose: bool = False,
    invert: bool = False,
    use_matrix: Optional[str] = None,
) -> MarkerHeatmapResult:
    """Compute ArchR ``plotMarkerHeatmap`` (matrix, ordering, labels).

    Arguments mirror ArchR (snake_case). ``subset_markers`` are 1-based row
    indices into the marker table, as in R. The returned
    :attr:`MarkerHeatmapResult.matrix` equals ArchR's ``returnMatrix = TRUE``.
    """
    ma = as_marker_assays(markers, use_matrix)
    groups = ma.groups
    assays = {k: v.to_numpy(dtype=np.float64) for k, v in ma.assays.items()}
    pass_mat = eval_cutoff(cut_off, assays)
    if pass_mat.ndim == 0:
        pass_mat = np.full((ma.n_features, len(groups)), float(pass_mat))

    if len(groups) <= 2 and not plot_log2fc:
        raise ValueError("Must use plot_log2fc=True when there are <= 2 groups!")
    if plot_log2fc:
        mat = assays["Log2FC"].copy()
    else:
        mat = assays["Mean"].copy()
        if log2_norm:
            mat = _r_log2(mat / _r_col_sums(mat) * scale_to + 1)
        if scale_rows:
            with np.errstate(divide="ignore", invalid="ignore"):
                mat = (mat - r_row_means(mat)[:, None]) / _row_sds(mat)[:, None]
    lo, hi = min(limits), max(limits)
    mat = np.where(mat > hi, hi, mat)
    mat = np.where(mat < lo, lo, mat)

    npass = np.nansum(pass_mat, axis=1)
    if mat.shape[1] == 1:
        idx = np.where(npass > 0)[0]
    else:
        rv = _row_vars(mat)
        idx = np.where((npass > 0) & (rv != 0) & ~np.isnan(rv))[0]
    if subset_markers is not None:
        sm = np.asarray(subset_markers, dtype=np.int64)
        if np.any((sm < 1) | (sm > mat.shape[0])):
            raise ValueError("subset_markers are outside the boundaries of the marker table.")
        idx = sm - 1
    mat = mat[idx]
    pass_mat = pass_mat[idx]
    if mat.shape[0] == 0:
        raise ValueError("No Makers Found!")

    rn = _feature_labels(ma.row_data.iloc[idx])
    if grep_exclude is not None:
        keep = np.array([re.search(grep_exclude, str(r)) is None for r in rn])
        mat, rn_mat = mat[keep], rn[keep]
    else:
        rn_mat = rn
    if mat.shape[0] == 0:
        raise ValueError("No Makers Found!")

    with np.errstate(divide="ignore", invalid="ignore"):
        spmat = pass_mat / pass_mat.sum(axis=1, keepdims=True)

    def _rn(i):
        return rn_mat[i] if i < len(rn_mat) else None

    top_features = {}
    if subset_markers is None and (ma.use_matrix or "") == "GeneScoreMatrix":
        for x, g in enumerate(groups):
            top_features[g] = [_rn(i) for i in _r_order_decreasing(spmat[:, x])[:n_print]]
    if label_markers is None:
        labs = []
        for x in range(spmat.shape[1]):
            labs.extend(_rn(i) for i in _r_order_decreasing(spmat[:, x])[:n_label])
        label_markers = list(dict.fromkeys(labs))
    else:
        label_markers = list(label_markers)

    if mat.shape[1] == 1:
        binary_cluster_rows = False

    # passMat[rownames(mat), ] — first row matching each name.
    first = {}
    for i, r in enumerate(rn):
        first.setdefault(r, i)
    lmat = pass_mat[[first[r] for r in rn_mat]]

    col_hc = None
    if binary_cluster_rows:
        row_idx, col_idx, col_hc = _binary_sort(-mat if invert else mat, lmat=lmat,
                                                cluster_cols=cluster_cols)
        mat = mat[row_idx]
        rn_mat = rn_mat[row_idx]
        cluster_rows = False
        cluster_cols_flag = col_hc is not None
    else:
        cluster_rows = True
        cluster_cols_flag = True

    if pal is None:
        um = (ma.use_matrix or "")
        pal = ARCHR_PALETTES["blueYellow" if um.lower() == "genescorematrix" else "solarExtra"]
    pal = list(pal)
    if invert:
        pal = pal[::-1]

    out = pd.DataFrame(mat, index=rn_mat, columns=groups)
    pm = pd.DataFrame(pass_mat, index=rn, columns=groups)
    lower = [str(x).lower() for x in out.index]
    pos = {}
    for i, v in enumerate(lower):
        pos.setdefault(v, i)
    mn = np.array([pos[str(l).lower()] for l in label_markers
                   if l is not None and str(l).lower() in pos], dtype=np.int64)

    # Display ordering, as ComplexHeatmap would draw it.
    shown = out.T if transpose else out
    smat = shown.to_numpy()
    s = (smat - smat.min()) / (smat.max() - smat.min()) if smat.max() > smat.min() else smat * 0
    row_hc = None
    if transpose:
        row_order = np.arange(shown.shape[0])
        if cluster_rows:
            col_hc_t = r_hclust(r_dist(s.T), "ward.D2")
            col_order = r_reorder_dendrogram(col_hc_t, -r_col_means(s))
            col_hc = col_hc_t
        else:
            col_order = np.arange(shown.shape[1])
            col_hc = None
    else:
        if cluster_rows:
            row_hc = r_hclust(r_dist(s), "ward.D2")
            row_order = r_reorder_dendrogram(row_hc, -r_row_means(s))
            col_hc = r_hclust(r_dist(s.T), "ward.D2")
            col_order = r_reorder_dendrogram(col_hc, -r_col_means(s))
        else:
            row_order = np.arange(shown.shape[0])
            col_order = col_hc.order0 if (cluster_cols_flag and col_hc is not None) \
                else np.arange(shown.shape[1])

    return MarkerHeatmapResult(
        matrix=shown, pass_matrix=pm, label_markers=label_markers, label_index=mn,
        row_order=np.asarray(row_order), col_order=np.asarray(col_order),
        transpose=transpose, pal=pal, use_matrix=ma.use_matrix,
        top_features=top_features, col_hclust=col_hc, row_hclust=row_hc,
    )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _dendro_segments(hc, order0):
    """Segments (x, y) for an R hclust drawn in leaf order ``order0``."""
    pos = {leaf: i for i, leaf in enumerate(order0)}
    xy = {}
    segs = []
    for i, (a, b) in enumerate(hc.merge, start=1):
        h = hc.height[i - 1]
        pts = []
        for ref in (a, b):
            if ref < 0:
                pts.append((pos[-ref - 1], 0.0))
            else:
                pts.append(xy[ref])
        (xa, ya), (xb, yb) = pts
        segs += [((xa, ya), (xa, h)), ((xb, yb), (xb, h)), ((xa, h), (xb, h))]
        xy[i] = ((xa + xb) / 2.0, h)
    return segs


def _place_labels(targets: np.ndarray, n: int, min_gap: float) -> np.ndarray:
    """Spread label positions (anno_mark-like) avoiding overlap."""
    if len(targets) == 0:
        return targets
    order = np.argsort(targets, kind="stable")
    t = targets[order].astype(float)
    p = t.copy()
    p[0] = max(p[0], -0.5)
    for i in range(1, len(p)):
        p[i] = max(p[i], p[i - 1] + min_gap)
    p[-1] = min(p[-1], n - 0.5)
    for i in range(len(p) - 2, -1, -1):
        p[i] = min(p[i], p[i + 1] - min_gap)
    out = np.empty_like(p)
    out[order] = p
    return out


def _draw_heatmap(ax, mat, cmap, vmin, vmax):
    return ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap,
                     vmin=vmin, vmax=vmax)


def marker_heatmap(
    markers,
    cut_off: str = "FDR <= 0.01 & Log2FC >= 0.5",
    log2_norm: bool = True,
    scale_to: float = 1e4,
    scale_rows: bool = True,
    plot_log2fc: bool = False,
    limits: Tuple[float, float] = (-2, 2),
    grep_exclude: Optional[str] = None,
    pal: Optional[Sequence[str]] = None,
    binary_cluster_rows: bool = True,
    cluster_cols: bool = True,
    subset_markers: Optional[Sequence[int]] = None,
    label_markers: Optional[Sequence[str]] = None,
    n_label: int = 15,
    n_print: int = 15,
    label_rows: bool = False,
    return_matrix: bool = False,
    transpose: bool = False,
    invert: bool = False,
    use_matrix: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show: bool = True,
):
    """ArchR ``plotMarkerHeatmap`` — heatmap of marker features per group.

    Arguments
    ---------
    markers
        Marker statistics: output of :func:`epione.tl.find_marker_features`
        (long DataFrame or the AnnData holding it), a :class:`MarkerAssays`,
        or ``{'assays': {'Mean': ..., 'Log2FC': ..., 'FDR': ...}, 'row_data': ...}``
        with features × groups tables (ArchR ``getMarkerFeatures`` layout).
    cut_off
        R-syntax logical over assay names, e.g. ``"FDR <= 0.01 & Log2FC >= 1.25"``.
    log2_norm, scale_to
        Column-normalise ``Mean`` to ``scale_to`` then ``log2(x + 1)``.
    scale_rows
        Plot row z-scores. ``plot_log2fc`` plots ``Log2FC`` instead of ``Mean``.
    limits
        Clip values to this range.
    grep_exclude
        Regex of feature names to drop.
    binary_cluster_rows, cluster_cols
        ArchR binary sorting of rows / ``hclust`` of columns.
    subset_markers
        1-based rows of the marker table to plot (ignores ``cut_off``).
    label_markers, n_label
        Features to annotate; default top ``n_label`` per group.
    return_matrix
        Return the matrix (identical to ArchR ``returnMatrix = TRUE``).
    transpose, invert
        As in ArchR.

    Returns
    -------
    ``pandas.DataFrame`` when ``return_matrix=True``, otherwise
    ``(fig, MarkerHeatmapResult)``.
    """
    res = marker_heatmap_matrix(
        markers, cut_off=cut_off, log2_norm=log2_norm, scale_to=scale_to,
        scale_rows=scale_rows, plot_log2fc=plot_log2fc, limits=limits,
        grep_exclude=grep_exclude, pal=pal, binary_cluster_rows=binary_cluster_rows,
        cluster_cols=cluster_cols, subset_markers=subset_markers,
        label_markers=label_markers, n_label=n_label, n_print=n_print,
        transpose=transpose, invert=invert, use_matrix=use_matrix,
    )
    if return_matrix:
        return res.matrix
    fig = plot_marker_heatmap_result(res, label_rows=label_rows, figsize=figsize,
                                     title=title)
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return fig, res


def plot_marker_heatmap_result(res: MarkerHeatmapResult, label_rows: bool = False,
                               figsize=None, title: Optional[str] = None):
    """Draw a :class:`MarkerHeatmapResult` (ComplexHeatmap-like layout)."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection

    shown = res.matrix
    mat = shown.to_numpy()[np.ix_(res.row_order, res.col_order)]
    rows = np.asarray(shown.index)[res.row_order]
    cols = np.asarray(shown.columns)[res.col_order]
    cmap = mcolors.LinearSegmentedColormap.from_list("archr", res.pal, N=256)
    nr, nc = mat.shape
    lab_names = np.asarray(shown.columns if res.transpose else shown.index)
    order_axis = res.col_order if res.transpose else res.row_order
    inv = np.empty(len(order_axis), dtype=np.int64)
    inv[order_axis] = np.arange(len(order_axis))
    lab_pos = inv[res.label_index] if len(res.label_index) else np.array([])
    lab_txt = lab_names[res.label_index] if len(res.label_index) else np.array([])

    if figsize is None:
        figsize = (max(6, 0.12 * nc + 3), 7) if res.transpose else (max(4.5, 0.35 * nc + 3), 8)
    fig = plt.figure(figsize=figsize)
    dend_h = 0.12
    if res.transpose:
        ax = fig.add_axes([0.12, 0.30, 0.80, 0.50])
        ax_lab = fig.add_axes([0.12, 0.80, 0.80, 0.14])
        ax_d = fig.add_axes([0.02, 0.30, 0.08, 0.50])
        ax_d.axis("off")
    else:
        ax = fig.add_axes([0.10, 0.18, 0.55, 0.62])
        ax_d = fig.add_axes([0.10, 0.80, 0.55, dend_h], sharex=ax)
        ax_lab = fig.add_axes([0.65, 0.18, 0.25, 0.62])
    lo, hi = float(np.nanmin(mat)), float(np.nanmax(mat))
    im = _draw_heatmap(ax, mat, cmap, lo, hi)
    if res.transpose:
        ax.set_yticks(range(nr))
        ax.set_yticklabels(rows, fontsize=9)
        ax.set_xticks([])
        if label_rows:
            ax.set_xticks(range(nc))
            ax.set_xticklabels(cols, rotation=90, fontsize=6)
    else:
        ax.set_xticks(range(nc))
        ax.set_xticklabels(cols, rotation=90, fontsize=9)
        ax.set_yticks([])
        if label_rows:
            ax.set_yticks(range(nr))
            ax.set_yticklabels(rows, fontsize=6)
    hc = res.col_hclust if not res.transpose else None
    if hc is not None:
        segs = _dendro_segments(hc, res.col_order)
        lc = LineCollection(segs, colors="black", linewidths=0.8)
        ax_d.add_collection(lc)
        ax_d.set_ylim(0, max(hc.height) * 1.05)
        ax_d.set_xlim(-0.5, nc - 0.5)
    ax_d.axis("off")

    ax_lab.axis("off")
    if res.transpose:
        ax_lab.set_xlim(ax.get_xlim())
        ax_lab.set_ylim(0, 1)
    else:
        ax_lab.set_ylim(ax.get_ylim())
        ax_lab.set_xlim(0, 1)
    if len(lab_pos):
        n_axis = nc if res.transpose else nr
        gap = min(n_axis / 40.0, n_axis / (len(lab_pos) + 1))
        fs = float(np.clip(450.0 / max(len(lab_pos), 1), 3.0, 7.0))
        placed = _place_labels(lab_pos.astype(float), n_axis, gap)
        for tgt, p, txt in zip(lab_pos, placed, lab_txt):
            if res.transpose:
                ax_lab.plot([tgt, tgt, p], [0, 0.25, 0.45], color="black", lw=0.5,
                            transform=ax_lab.get_xaxis_transform())
                ax_lab.text(p, 0.5, txt, rotation=90, va="bottom", ha="center",
                            fontsize=fs, transform=ax_lab.get_xaxis_transform())
            else:
                ax_lab.plot([0, 0.15, 0.3], [tgt, tgt, p], color="black", lw=0.5,
                            transform=ax_lab.get_yaxis_transform())
                ax_lab.text(0.32, p, txt, va="center", ha="left", fontsize=fs,
                            transform=ax_lab.get_yaxis_transform())
    cax = fig.add_axes([0.30, 0.06, 0.25, 0.02])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[lo, hi])
    cb.ax.set_xticklabels([f"{lo:.2f}", f"{hi:.2f}"])
    kind = "Column" if res.transpose else "Row"
    n_feat = nc if res.transpose else nr
    cb.set_label(f"{kind} Z-Scores | {n_feat} features | {res.use_matrix or ''}", fontsize=8)
    if title:
        fig.suptitle(title)
    return fig


# ---------------------------------------------------------------------------
# plotPeak2GeneHeatmap
# ---------------------------------------------------------------------------

@dataclass
class Peak2GeneHeatmapResult:
    """ArchR ``plotPeak2GeneHeatmap(returnMatrices = TRUE)`` output."""

    atac: pd.DataFrame           # links × KNN groups (z-scores), rows in kDF order
    rna: pd.DataFrame
    kmeans_id: np.ndarray        # kDF[,3] (1-based, ordered by binary sort)
    col_groups: pd.Series        # majority `group_by` label per KNN column (colOrder)
    links: pd.DataFrame          # p2g rows in kDF order (incl. `peak`, `gene`)
    kmeans_cluster: np.ndarray   # raw kmeans labels (1-based) for all filtered links
    kdf: pd.DataFrame            # columns k, idx (1-based), k_ordered


def _archr_quantiles(v: np.ndarray) -> np.ndarray:
    """ArchR ``.getQuantiles``: ``trunc(rank(v)) / length(v)``."""
    from scipy.stats import rankdata
    return np.trunc(rankdata(v, method="average")) / len(v)


def _knn_majority(knn_index: Sequence[Sequence[int]], cell_groups: Sequence[str]) -> List[str]:
    """``names(sort(table(groups[KNN]), decreasing = TRUE))[1]`` per KNN group."""
    cg = np.asarray(cell_groups).astype(str)
    out = []
    for idx in knn_index:
        vals, counts = np.unique(cg[np.asarray(idx, dtype=np.int64)], return_counts=True)
        # np.unique sorts codepoint-wise (C locale); argmax returns the first max,
        # matching R's stable decreasing sort of the alphabetically-ordered table.
        out.append(vals[int(np.argmax(counts))])
    return out


def peak2gene_heatmap_matrices(
    links: pd.DataFrame,
    atac_knn: np.ndarray,
    rna_knn: np.ndarray,
    *,
    knn_groups: Optional[Sequence[str]] = None,
    knn_index: Optional[Sequence[Sequence[int]]] = None,
    cell_groups: Optional[Sequence[str]] = None,
    peak_names: Optional[Sequence[str]] = None,
    gene_names: Optional[Sequence[str]] = None,
    cor_cutoff: float = 0.45,
    fdr_cutoff: float = 1e-4,
    var_cutoff_atac: Optional[float] = 0.25,
    var_cutoff_rna: Optional[float] = 0.25,
    k: int = 25,
    n_plot: int = 25000,
    seed: Optional[int] = 1,
    rng_kind: str = "L'Ecuyer-CMRG",
    kmeans_cluster: Optional[np.ndarray] = None,
) -> Peak2GeneHeatmapResult:
    """Compute ArchR ``plotPeak2GeneHeatmap(returnMatrices = TRUE)``.

    Arguments
    ---------
    links
        Peak-to-gene table with ArchR columns ``idxATAC, idxRNA`` (1-based rows
        of ``atac_knn`` / ``rna_knn``), ``Correlation, FDR, VarQATAC, VarQRNA``.
    atac_knn, rna_knn
        Log2-normalised KNN-group matrices (features × KNN groups), i.e. the
        ``seATAC`` / ``seRNA`` assays of ``addPeak2GeneLinks``.
    knn_groups
        Label of each KNN group; or give ``knn_index`` (0-based cell indices
        per group) + ``cell_groups`` to take the majority label like ArchR.
    k, n_plot, seed
        kmeans groups, max links shown, ``set.seed`` value. kmeans is R's
        Hartigan-Wong seeded with R's own RNG, so results match R.
    rng_kind
        R ``RNGkind`` of the reference session. ArchR's
        ``addArchRThreads(threads > 1)`` switches R to ``"L'Ecuyer-CMRG"``
        (the default here); use ``"Mersenne-Twister"`` for single-threaded /
        plain R sessions.
    kmeans_cluster
        Optional 1-based kmeans labels to inject (e.g. from R); the RNG still
        advances through kmeans' initial-centre draw so ``n_plot`` subsampling
        stays in sync with R.
    """
    p2g = links.copy()
    with np.errstate(invalid="ignore"):
        keep = (p2g["Correlation"] >= cor_cutoff) & (p2g["FDR"] <= fdr_cutoff)
    p2g = p2g[keep.fillna(False).to_numpy()]
    if var_cutoff_atac is not None:
        p2g = p2g[(p2g["VarQATAC"] > var_cutoff_atac).fillna(False).to_numpy()]
    if var_cutoff_rna is not None:
        p2g = p2g[(p2g["VarQRNA"] > var_cutoff_rna).fillna(False).to_numpy()]
    if len(p2g) == 0:
        raise ValueError("No peak2genelinks found with your cutoffs!")
    p2g = p2g.reset_index(drop=True)
    ia = p2g["idxATAC"].to_numpy(dtype=np.int64) - 1
    ir = p2g["idxRNA"].to_numpy(dtype=np.int64) - 1
    if peak_names is not None:
        p2g["peak"] = np.asarray(peak_names, dtype=object)[ia]
    if gene_names is not None:
        p2g["gene"] = np.asarray(gene_names, dtype=object)[ir]
    mATAC = _row_zscores(np.asarray(atac_knn, dtype=np.float64)[ia])
    mRNA = _row_zscores(np.asarray(rna_knn, dtype=np.float64)[ir])
    n_k = mATAC.shape[1]
    kcols = np.array([f"K_{i + 1}" for i in range(n_k)])

    if knn_groups is None:
        if knn_index is None or cell_groups is None:
            raise ValueError("give knn_groups, or knn_index + cell_groups")
        knn_groups = _knn_majority(knn_index, cell_groups)
    knn_groups = np.asarray(knn_groups).astype(str)

    rng = RRandom(seed if seed is not None else int(np.random.randint(1, 2**31 - 1)), rng_kind)
    m = mATAC.shape[0]
    if kmeans_cluster is None:
        cl = r_kmeans(mATAC, k, rng=rng).cluster
    else:
        r_kmeans_init(np.ascontiguousarray(mATAC), k, rng)  # advance RNG like kmeans()
        cl = np.asarray(kmeans_cluster, dtype=np.int64)

    if m > n_plot:
        levels = np.unique(cl)
        counts = np.array([(cl == lv).sum() for lv in levels])
        nPK = n_plot * counts / len(cl)
        ks, idxs = [], []
        for x, lv in enumerate(levels):
            members = np.where(cl == lv)[0] + 1
            take = rng.sample(members, int(np.floor(nPK[x])))
            ks.append(np.full(len(take), x + 1, dtype=np.int64))
            idxs.append(np.asarray(take, dtype=np.int64))
        kk = np.concatenate(ks)
        kidx = np.concatenate(idxs)
    else:
        kk = cl.copy()
        kidx = np.arange(1, m + 1)

    # t(.groupMeans(t(mATAC[kDF$idx, ]), kDF$k)): groups (first-appearance order) × K
    sub = mATAC[kidx - 1]
    uniq = list(dict.fromkeys(kk.tolist()))
    gm = np.vstack([r_col_means(sub[kk == g]) for g in uniq])
    row_idx, col_idx, _ = _binary_sort(gm, cut_off=1.0, cluster_cols=True)
    row_order = [uniq[i] for i in row_idx]
    newlab = {g: i + 1 for i, g in enumerate(row_order)}
    k3 = np.array([newlab[g] for g in kk], dtype=np.int64)

    col_order = kcols[col_idx]
    rows = [f"P2G_{i}" for i in kidx]
    atac = pd.DataFrame(mATAC[kidx - 1][:, col_idx], index=rows, columns=col_order)
    rna = pd.DataFrame(mRNA[kidx - 1][:, col_idx], index=rows, columns=col_order)
    p2g.index = [f"P2G_{i + 1}" for i in range(len(p2g))]
    return Peak2GeneHeatmapResult(
        atac=atac, rna=rna, kmeans_id=k3,
        col_groups=pd.Series(knn_groups[col_idx], index=col_order, name="groupBy"),
        links=p2g.iloc[kidx - 1],
        kmeans_cluster=cl,
        kdf=pd.DataFrame({"k": kk, "idx": kidx, "k_ordered": k3}),
    )


def prepare_peak2gene_heatmap(
    adata,
    gene_X,
    *,
    links: Optional[pd.DataFrame] = None,
    key: str = "peak_to_gene",
    gene_names: Optional[Sequence[str]] = None,
    use_rep: Optional[str] = None,
    n_metacells: Optional[int] = None,
    k_neighbors: Optional[int] = None,
    seed: Optional[int] = None,
    scale_to: float = 1e4,
    log2_norm: bool = True,
) -> Dict[str, Any]:
    """Rebuild ArchR-style heatmap inputs from :func:`epione.tl.peak_to_gene`.

    Recreates the metacells used by ``peak_to_gene`` (same parameters/seed),
    aggregates the peak (``adata.X``) and gene (``gene_X``, cells × genes)
    matrices, column-normalises to ``scale_to`` and log2-transforms like
    ``addPeak2GeneLinks``, and computes ArchR's ``VarQATAC`` / ``VarQRNA``.
    Returns keyword arguments for :func:`peak2gene_heatmap_matrices`.
    """
    from ..tl._peak_to_gene import _build_metacell_indices, _aggregate

    params = dict(adata.uns.get(key + "_params", {}))
    links = adata.uns[key] if links is None else links
    use_rep = use_rep or params.get("use_rep", "X_iterative_lsi")
    n_metacells = n_metacells or params.get("n_metacells", 500)
    k_neighbors = k_neighbors or params.get("k_neighbors", 100)
    seed = params.get("seed", 1) if seed is None else seed
    if hasattr(gene_X, "var_names"):  # AnnData passed as `rna=` to peak_to_gene
        if gene_names is None:
            gene_names = gene_X.var_names
        gene_X = gene_X.X
    if gene_names is None:
        raise ValueError("`gene_names` is required when `gene_X` is a matrix")
    emb = np.asarray(adata.obsm[use_rep])
    mc = _build_metacell_indices(emb, n_metacells, k_neighbors, seed)
    peak_mc = _aggregate(adata.X, mc).astype(np.float64)
    gene_mc = _aggregate(gene_X, mc).astype(np.float64)

    def _norm(a):
        a = a / a.sum(axis=1, keepdims=True) * scale_to
        return _r_log2(a + 1) if log2_norm else a

    atac_knn = _norm(peak_mc).T
    rna_knn = _norm(gene_mc).T
    var_a = _archr_quantiles(_row_vars(atac_knn))
    var_r = _archr_quantiles(_row_vars(rna_knn))
    pnames = np.asarray(adata.var_names)
    gnames = np.asarray(gene_names)
    pidx = pd.Series(np.arange(len(pnames)), index=pnames)
    gidx = pd.Series(np.arange(len(gnames)), index=gnames)
    lk = links[links["peak"].isin(pidx.index) & links["gene"].isin(gidx.index)]
    ia = pidx[lk["peak"].to_numpy()].to_numpy()
    ir = gidx[lk["gene"].to_numpy()].to_numpy()
    arch = pd.DataFrame({
        "idxATAC": ia + 1, "idxRNA": ir + 1,
        "Correlation": lk["correlation"].to_numpy(dtype=np.float64),
        "FDR": lk["fdr"].to_numpy(dtype=np.float64),
        "VarQATAC": var_a[ia], "VarQRNA": var_r[ir],
    })
    return dict(links=arch, atac_knn=atac_knn, rna_knn=rna_knn,
                knn_index=[np.asarray(r) for r in mc],
                peak_names=pnames, gene_names=gnames)


def peak2gene_heatmap(
    links: Optional[pd.DataFrame] = None,
    atac_knn: Optional[np.ndarray] = None,
    rna_knn: Optional[np.ndarray] = None,
    *,
    cor_cutoff: float = 0.45,
    fdr_cutoff: float = 1e-4,
    var_cutoff_atac: Optional[float] = 0.25,
    var_cutoff_rna: Optional[float] = 0.25,
    k: int = 25,
    n_plot: int = 25000,
    limits_atac: Tuple[float, float] = (-2, 2),
    limits_rna: Tuple[float, float] = (-2, 2),
    pal_group: Optional[Mapping[str, str]] = None,
    pal_atac: Optional[Sequence[str]] = None,
    pal_rna: Optional[Sequence[str]] = None,
    return_matrices: bool = False,
    seed: Optional[int] = 1,
    figsize: Tuple[float, float] = (10, 8),
    show: bool = True,
    **kwargs,
):
    """ArchR ``plotPeak2GeneHeatmap`` — side-by-side ATAC / RNA heatmaps of
    peak-to-gene links, rows grouped by kmeans and columns by KNN group.

    Pass ArchR-style ``links`` + ``atac_knn`` / ``rna_knn`` (+ ``knn_groups``
    or ``knn_index`` & ``cell_groups``), or the dict from
    :func:`prepare_peak2gene_heatmap` via ``**``. ``return_matrices=True``
    returns a :class:`Peak2GeneHeatmapResult` equal to ArchR's
    ``returnMatrices = TRUE`` output.
    """
    res = peak2gene_heatmap_matrices(
        links, atac_knn, rna_knn, cor_cutoff=cor_cutoff, fdr_cutoff=fdr_cutoff,
        var_cutoff_atac=var_cutoff_atac, var_cutoff_rna=var_cutoff_rna,
        k=k, n_plot=n_plot, seed=seed, **kwargs,
    )
    if return_matrices:
        return res
    fig = plot_peak2gene_heatmap_result(
        res, limits_atac=limits_atac, limits_rna=limits_rna, pal_group=pal_group,
        pal_atac=pal_atac, pal_rna=pal_rna, figsize=figsize,
        all_groups=kwargs.get("cell_groups"),
    )
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return fig, res


def plot_peak2gene_heatmap_result(res: Peak2GeneHeatmapResult, limits_atac=(-2, 2),
                                  limits_rna=(-2, 2), pal_group=None, pal_atac=None,
                                  pal_rna=None, figsize=(10, 8), all_groups=None):
    """Draw a :class:`Peak2GeneHeatmapResult` like ArchR (rows split by kmeans)."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    order = np.argsort(res.kmeans_id, kind="stable")
    kid = res.kmeans_id[order]
    groups = res.col_groups.to_numpy()
    pal = palette_discrete(all_groups if all_groups is not None else groups)
    if pal_group:
        pal.update({k: v for k, v in pal_group.items() if k in pal})
    fig = plt.figure(figsize=figsize)
    specs = [("ATAC", res.atac, limits_atac, pal_atac or ARCHR_PALETTES["solarExtra"], 0.06),
             ("RNA", res.rna, limits_rna, pal_rna or ARCHR_PALETTES["blueYellow"], 0.52)]
    bounds = np.r_[0, np.cumsum(np.unique(kid, return_counts=True)[1])]
    for name, df, lim, colors, x0 in specs:
        mat = np.clip(df.to_numpy()[order], min(lim), max(lim))
        cmap = mcolors.LinearSegmentedColormap.from_list(name, list(colors), N=256)
        ax_top = fig.add_axes([x0, 0.855, 0.40, 0.025])
        ax = fig.add_axes([x0, 0.12, 0.40, 0.73], sharex=ax_top)
        ax_top.imshow(np.array([[mcolors.to_rgb(pal[g]) for g in groups]]), aspect="auto",
                      interpolation="nearest")
        ax_top.axis("off")
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap,
                       vmin=min(lim), vmax=max(lim))
        for b in bounds[1:-1]:
            ax.axhline(b - 0.5, color="white", lw=1.2)
        ax.set_xticks([])
        if name == "ATAC":
            mids = (bounds[:-1] + bounds[1:]) / 2 - 0.5
            ax.set_yticks(mids)
            ax.set_yticklabels([str(i) for i in np.unique(kid)], fontsize=7)
        else:
            ax.set_yticks([])
        ax_top.set_title(f"{name} Z-Scores | {df.shape[0]} P2GLinks", fontsize=9)
        cax = fig.add_axes([x0 + 0.12, 0.06, 0.16, 0.015])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[min(lim), max(lim)])
        cb.ax.tick_params(labelsize=7)
    handles = [Patch(color=pal[g], label=g) for g in sorted(set(groups), key=_mixedsort_key)]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 11), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.985))
    return fig
