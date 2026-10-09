"""Peak-annotation hypergeometric enrichment (ArchR ``peakAnnoEnrichment``).

Port of ArchR (MIT) ``AnnotationPeaks.R``: ``addPeakAnnotations``,
``addArchRAnnotations``, ``peakAnnoEnrichment``, ``customEnrichment`` and the
internal ``.computeEnrichment``. Numerics follow R exactly, including the
``round(4)`` on ``mlog10p`` and the ``mlog10Padj = mlog10p - log10(9)``
correction (ArchR uses the column count of its summary table, which is 9).

Annotation matrices follow the Epione convention used by
:func:`epione.tl.add_motif_matrix`: ``adata.varm[key]`` is a sparse boolean
``n_peaks x n_annotations`` matrix and ``adata.uns[key + '_names']`` holds the
annotation names.
"""
from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import sparse

from ..utils import console


ENRICH_ASSAYS = (
    "mlog10Padj", "mlog10p", "Enrichment", "BackgroundProporition",
    "nBackground", "BackgroundFrequency", "CompareProportion", "nCompare",
    "CompareFrequency", "feature",
)

ARCHR_ANNO_URLS = {
    ("archr", "hg19"): "https://jeffgranja.s3.amazonaws.com/ArchR/Annotations/ArchR-Hg19-v1.Anno",
    ("archr", "hg38"): "https://jeffgranja.s3.amazonaws.com/ArchR/Annotations/ArchR-Hg38-v1.Anno",
    ("lola", "hg19"): "https://jeffgranja.s3.amazonaws.com/ArchR/Annotations/LOLA-Hg19-v1.Anno",
    ("lola", "hg38"): "https://jeffgranja.s3.amazonaws.com/ArchR/Annotations/LOLA-Hg38-v1.Anno",
    ("lola", "mm9"): "https://jeffgranja.s3.amazonaws.com/ArchR/Annotations/LOLA-Mm9-v1.Anno",
    ("lola", "mm10"): "https://jeffgranja.s3.amazonaws.com/ArchR/Annotations/LOLA-Mm10-v1.Anno",
}

_MARKER_ALIASES = {
    "fdr": "FDR", "log2_fc": "Log2FC", "p_value": "Pval", "mean_fg": "Mean",
    "mean_bg": "MeanBGD",
}


# ---------------------------------------------------------------------------
# R-compatible numerics
# ---------------------------------------------------------------------------

_LN_2PI = math.log(2.0 * math.pi)
_LN_SQRT_2PI = 0.5 * _LN_2PI
_S0, _S1, _S2, _S3, _S4 = 1 / 12, 1 / 360, 1 / 1260, 1 / 1680, 1 / 1188


def _stirlerr(n: float) -> float:
    """log(n!) - log(sqrt(2*pi*n)*(n/e)^n) (Loader 2000, as in R nmath)."""
    if n <= 15.0:
        return math.lgamma(n + 1.0) - (n + 0.5) * math.log(n) + n - _LN_SQRT_2PI
    nn = n * n
    if n > 500:
        return (_S0 - _S1 / nn) / n
    if n > 80:
        return (_S0 - (_S1 - _S2 / nn) / nn) / n
    if n > 35:
        return (_S0 - (_S1 - (_S2 - _S3 / nn) / nn) / nn) / n
    return (_S0 - (_S1 - (_S2 - (_S3 - _S4 / nn) / nn) / nn) / nn) / n


def _bd0(x: float, np_: float) -> float:
    if abs(x - np_) < 0.1 * (x + np_):
        v = (x - np_) / (x + np_)
        s = (x - np_) * v
        if abs(s) < 2.2250738585072014e-308:
            return s
        ej = 2 * x * v
        v = v * v
        for j in range(1, 1000):
            ej *= v
            s1 = s + ej / (2 * j + 1)
            if s1 == s:
                return s1
            s = s1
    return x * math.log(x / np_) + np_ - x


def _log_dbinom_raw(x: float, n: float, p: float, q: float) -> float:
    if p == 0:
        return 0.0 if x == 0 else -math.inf
    if q == 0:
        return 0.0 if x == n else -math.inf
    if x == 0:
        if n == 0:
            return 0.0
        return -_bd0(n, n * q) - n * p if p < 0.1 else n * math.log(q)
    if x == n:
        return -_bd0(n, n * p) - n * q if q < 0.1 else n * math.log(p)
    if x < 0 or x > n:
        return -math.inf
    lc = (_stirlerr(n) - _stirlerr(x) - _stirlerr(n - x)
          - _bd0(x, n * p) - _bd0(n - x, n * q))
    lf = _LN_2PI + math.log(x) + math.log1p(-x / n)
    return lc - 0.5 * lf


def _log_dhyper(x: float, r: float, b: float, n: float) -> float:
    if x < 0 or n < x or r < x or n - x > b:
        return -math.inf
    if n == 0:
        return 0.0 if x == 0 else -math.inf
    p = n / (r + b)
    q = (r + b - n) / (r + b)
    return (_log_dbinom_raw(x, r, p, q) + _log_dbinom_raw(n - x, b, p, q)
            - _log_dbinom_raw(n, r + b, p, q))


def _log_pdhyper(x: float, NR: float, NB: float, n: float) -> float:
    total = np.longdouble(0.0)
    term = np.longdouble(1.0)
    eps = np.longdouble(np.finfo(np.float64).eps)
    while x > 0 and term >= eps * total:
        term *= np.longdouble(x * (NB - n + x)) / np.longdouble(n + 1 - x) / np.longdouble(NR + 1 - x)
        total += term
        x -= 1
    return math.log1p(float(total))


def _log1mexp(x: float) -> float:
    return math.log(-math.expm1(x)) if x > -math.log(2.0) else math.log1p(-math.exp(x))


def _phyper_log_upper(q: float, NR: float, NB: float, n: float) -> float:
    """``phyper(q, NR, NB, n, lower.tail = FALSE, log.p = TRUE)`` (R nmath)."""
    x = math.floor(q + 1e-7)
    lower = False
    if x * (NR + NB) > n * NR:
        NR, NB = NB, NR
        x = n - x - 1
        lower = True
    if x < 0 or x < n - NB:
        return -math.inf if lower else 0.0
    if x >= NR or x >= n:
        return 0.0 if lower else -math.inf
    d = _log_dhyper(x, NR, NB, n)
    if d == -math.inf:
        return -math.inf if lower else 0.0
    pd_ = _log_pdhyper(x, NR, NB, n)
    return d + pd_ if lower else _log1mexp(d + pd_)


def _r_round(x, digits: int = 4) -> np.ndarray:
    """R >= 4.0 ``round(x, digits)``: nearest of floor/ceil candidates, ties to even."""
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    ok = np.isfinite(x) & (x != 0)
    ax = np.abs(x[ok])
    p10 = 10.0 ** digits
    i10 = np.floor(ax * p10)
    xd = i10 / p10
    xu = np.ceil(ax * p10) / p10
    du, dd = xu - ax, ax - xd
    r = np.where((du < dd) | ((du == dd) & (np.fmod(i10, 2.0) == 1.0)), xu, xd)
    with np.errstate(divide="ignore"):
        r = np.where(np.log10(ax) + digits > 15, ax, r)
    out[ok] = np.sign(x[ok]) * r
    return out


def _r_order_desc(v) -> np.ndarray:
    """``order(v, decreasing = TRUE)``; stable for ties, NA last."""
    v = np.asarray(v, dtype=np.float64)
    key = np.where(np.isnan(v), np.inf, -v)
    return np.argsort(key, kind="stable")


# ---------------------------------------------------------------------------
# Core enrichment
# ---------------------------------------------------------------------------

def _compute_enrichment(matches: sparse.csr_matrix, names: np.ndarray,
                        compare: np.ndarray, background: np.ndarray) -> pd.DataFrame:
    """ArchR ``.computeEnrichment``; ``compare``/``background`` are 0-based row indices."""
    m = matches
    cmp_tot = np.asarray(m[compare].sum(axis=0)).ravel().astype(np.float64)
    if background.size == m.shape[0] and np.array_equal(background, np.arange(m.shape[0])):
        bg_tot = np.asarray(m.sum(axis=0)).ravel().astype(np.float64)
    else:
        counts = np.bincount(background, minlength=m.shape[0]).astype(np.float64)
        bg_tot = np.asarray(m.T @ counts).ravel()
    n_cmp = float(compare.size)
    n_bg = float(background.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        cmp_prop = cmp_tot / n_cmp
        bg_prop = bg_tot / n_bg
        enrich = cmp_prop / bg_prop
    raw = np.array([
        -_phyper_log_upper(cf - 1, bf, n_bg - bf, n_cmp) / math.log(10)
        for cf, bf in zip(cmp_tot, bg_tot)
    ])
    mlog10p = _r_round(raw, 4)
    df = pd.DataFrame({
        "feature": names,
        "CompareFrequency": cmp_tot,
        "nCompare": n_cmp,
        "CompareProportion": cmp_prop,
        "BackgroundFrequency": bg_tot,
        "nBackground": n_bg,
        "BackgroundProporition": bg_prop,
        "Enrichment": enrich,
        "mlog10p": mlog10p,
    }, index=pd.Index(names, name=None))
    df["mlog10Padj"] = np.maximum(df["mlog10p"].to_numpy() - math.log10(9), 0.0)
    return df.iloc[_r_order_desc(df["mlog10p"].to_numpy())]


def _as_csr_bool(M) -> sparse.csr_matrix:
    if sparse.issparse(M):
        M = sparse.csr_matrix(M)
    else:
        M = sparse.csr_matrix(np.asarray(M))
    M = M.astype(bool).astype(np.float64)
    M.eliminate_zeros()
    return M


def _resolve_matches(adata, annotation, matches, annotation_names, peak_names):
    if matches is None:
        if adata is None:
            raise ValueError("Pass `adata` (with adata.varm[annotation]) or `matches`.")
        if annotation not in adata.varm:
            raise KeyError(f"adata.varm[{annotation!r}] not found — run "
                           "epi.tl.add_motif_matrix / add_peak_annotations first.")
        matches = adata.varm[annotation]
        if annotation_names is None:
            annotation_names = adata.uns.get(f"{annotation}_names")
        if peak_names is None:
            peak_names = adata.var_names
    M = _as_csr_bool(matches)
    if annotation_names is None:
        annotation_names = [f"{annotation}_{i + 1}" for i in range(M.shape[1])]
    annotation_names = np.asarray(annotation_names, dtype=object).astype(str)
    if peak_names is None:
        raise ValueError("`peak_names` is required when passing a bare `matches` matrix.")
    peak_names = pd.Index(np.asarray(peak_names).astype(str))
    if len(peak_names) != M.shape[0] or len(annotation_names) != M.shape[1]:
        raise ValueError("matches shape does not match peak_names / annotation_names.")
    return M, annotation_names, peak_names


def _markers_to_assays(markers, adata=None) -> Dict[str, pd.DataFrame]:
    if isinstance(markers, str):
        if adata is None:
            raise ValueError("markers given as a uns key but adata is None")
        markers = adata.uns[markers]
    if isinstance(markers, Mapping):
        assays = {k: (v if isinstance(v, pd.DataFrame) else pd.DataFrame(v))
                  for k, v in markers.items()}
        ref = next(iter(assays.values()))
        for k, v in assays.items():
            if v.shape != ref.shape:
                raise ValueError(f"marker assay {k!r} has a different shape")
        return assays
    if isinstance(markers, pd.DataFrame) and {"group", "feature"}.issubset(markers.columns):
        long = markers
        groups = list(pd.unique(long["group"].astype(str)))
        feats = list(pd.unique(long["feature"].astype(str)))
        assays = {}
        for col in long.columns:
            if col in ("group", "feature") or not np.issubdtype(long[col].dtype, np.number):
                continue
            wide = long.pivot(index="feature", columns="group", values=col)
            wide.columns = wide.columns.astype(str)
            wide = wide.reindex(index=feats, columns=groups)
            assays[col] = wide
            if col in _MARKER_ALIASES:
                assays[_MARKER_ALIASES[col]] = wide
        if "mean_fg" in long.columns and "mean_bg" in long.columns:
            assays["MeanDiff"] = assays["mean_fg"] - assays["mean_bg"]
        return assays
    raise TypeError("markers must be a dict of (peaks x groups) DataFrames, the long "
                    "table from epi.tl.find_marker_features, or an adata.uns key.")


def _eval_cutoff(cutoff, assays: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    ref = next(iter(assays.values()))
    if callable(cutoff):
        res = cutoff(assays)
    elif isinstance(cutoff, pd.DataFrame):
        res = cutoff
    else:
        expr = str(cutoff)
        if re.search(r"[A-Za-z_][A-Za-z0-9_.]*\s*\(", expr) or "__" in expr or "@" in expr:
            raise ValueError("cutoff may only contain assay names, numbers, "
                             "comparisons and & | ! operators")
        expr = re.sub(r"!(?!=)", "~", expr)
        local = {k: v.astype(float) for k, v in assays.items()
                 if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k)}
        res = pd.eval(expr, local_dict=local, engine="python", parser="pandas")
    res = pd.DataFrame(res).reindex(index=ref.index, columns=ref.columns)
    return res.fillna(False).astype(bool)


def _assemble(enrich_list: Dict[str, pd.DataFrame], names: np.ndarray) -> Dict[str, pd.DataFrame]:
    out = {}
    for assay in ENRICH_ASSAYS:
        out[assay] = pd.DataFrame(
            {g: df.loc[names, assay].to_numpy() for g, df in enrich_list.items()},
            index=pd.Index(names),
        )
    return out


def peak_anno_enrichment(
    markers,
    adata=None,
    *,
    annotation: str = "motif",
    matches=None,
    annotation_names: Optional[Sequence[str]] = None,
    peak_names: Optional[Sequence[str]] = None,
    cutoff: Union[str, Callable, pd.DataFrame] = "FDR <= 0.1 & Log2FC >= 0.5",
    background: str = "all",
    bg_peaks: Optional[np.ndarray] = None,
    bg_key: str = "bg_peaks",
    key_added: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Hypergeometric enrichment of a peak annotation within marker peaks.

    Port of ArchR ``peakAnnoEnrichment``. For each marker group the peaks
    passing ``cutoff`` form the foreground; the test is
    ``phyper(k - 1, K, N - K, n, lower.tail = FALSE)`` against all peaks
    (``background='all'``) or against the foreground plus its GC/accessibility
    matched background peaks (``background='bgdPeaks'``).

    Arguments:
        markers: marker statistics, one of (i) a dict ``{assay: DataFrame}``
            with peaks as rows and groups as columns (ArchR ``seMarker``
            assays such as ``FDR``, ``Log2FC``, ``MeanDiff``); (ii) the long
            table returned by :func:`epione.tl.find_marker_features` (its
            ``fdr``/``log2_fc``/``p_value`` columns are also exposed as
            ``FDR``/``Log2FC``/``Pval``); (iii) an ``adata.uns`` key holding (ii).
        adata: peak AnnData holding the annotation in ``adata.varm[annotation]``.
        annotation: ``adata.varm`` key of the peak x annotation matrix
            (``'motif'`` from :func:`add_motif_matrix`, or a key from
            :func:`add_peak_annotations` / :func:`add_archr_annotations`).
        matches: optional explicit peak x annotation matrix (overrides adata).
        annotation_names: names of the annotation columns.
        peak_names: row names of ``matches``; must contain every marker peak.
        cutoff: R-style expression over assay names, e.g.
            ``"FDR <= 0.1 & Log2FC >= 0.5"``; or a callable
            ``f(assays) -> bool DataFrame``; or a bool DataFrame. NA -> False.
        background: ``'all'`` or ``'bgdPeaks'`` (aliases ``'bgd'``,
            ``'background'``, ``'backgroundpeaks'``).
        bg_peaks: ``n_peaks x n_iter`` 0-based background peak indices
            (default ``adata.varm[bg_key]`` from :func:`add_background_peaks`).
            Rows are indexed by foreground position in marker order, as ArchR does.
        key_added: if given, the result is stored in ``adata.uns[key_added]``.
        verbose: print progress.

    Returns:
        dict ``{assay: DataFrame}`` (annotations x groups) with the ArchR
        assays ``mlog10Padj, mlog10p, Enrichment, BackgroundProporition,
        nBackground, BackgroundFrequency, CompareProportion, nCompare,
        CompareFrequency, feature`` (ArchR's spelling of
        ``BackgroundProporition`` is kept for compatibility).
    """
    assays = _markers_to_assays(markers, adata)
    M, names, pnames = _resolve_matches(adata, annotation, matches, annotation_names, peak_names)
    ref = next(iter(assays.values()))
    marker_peaks = pd.Index(ref.index.astype(str))
    pos = pnames.get_indexer(marker_peaks)
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} marker peaks are not in the annotation matrix.")
    M = M[pos]
    pass_mat = _eval_cutoff(cutoff, assays)

    method = "bgd" if str(background).lower() in ("backgroundpeaks", "bgdpeaks", "background", "bgd") else "all"
    if method == "bgd":
        if bg_peaks is None:
            if adata is None or bg_key not in adata.varm:
                raise KeyError(f"background='bgdPeaks' needs bg_peaks or adata.varm[{bg_key!r}]")
            bg_peaks = adata.varm[bg_key]
        bg_peaks = np.asarray(bg_peaks, dtype=np.int64)

    enrich_list = {}
    all_rows = np.arange(M.shape[0])
    for gi, g in enumerate(ref.columns):
        if verbose:
            console.info(f"[peak_anno_enrichment] {gi + 1}/{ref.shape[1]} group {g}")
        idx = np.flatnonzero(pass_mat.iloc[:, gi].to_numpy())
        if method == "bgd":
            bg = np.concatenate([idx, bg_peaks[idx].ravel(order="F")])
        else:
            bg = all_rows
        enrich_list[str(g)] = _compute_enrichment(M, names, idx, bg)
    out = _assemble(enrich_list, names)
    if key_added is not None and adata is not None:
        adata.uns[key_added] = out
    return out


def custom_enrichment(
    ranges,
    adata=None,
    *,
    annotation: str = "motif",
    matches=None,
    annotation_names: Optional[Sequence[str]] = None,
    peak_names: Optional[Sequence[str]] = None,
    bg_peaks: Optional[np.ndarray] = None,
    bg_key: Optional[str] = None,
) -> pd.DataFrame:
    """Hypergeometric enrichment of an annotation in peaks overlapping ``ranges``.

    Port of ArchR ``customEnrichment``. ``ranges`` is a DataFrame with
    ``chrom/start/end`` (or ``seqnames``) columns, closed intervals. With
    ``bg_peaks`` (or ``bg_key``) the background is the foreground plus its
    matched background peaks; otherwise all peaks. Returns the summary table
    sorted by ``mlog10p`` (decreasing), as ArchR does.
    """
    M, names, pnames = _resolve_matches(adata, annotation, matches, annotation_names, peak_names)
    chrom, start, end = _parse_peak_coords(pnames, adata if matches is None else None)
    r = _region_frame(ranges)
    hit = _overlap_any(chrom, start, end, r)
    idx = np.flatnonzero(hit)
    if bg_key is not None and bg_peaks is None:
        bg_peaks = adata.varm[bg_key]
    if bg_peaks is not None:
        bg_peaks = np.asarray(bg_peaks, dtype=np.int64)
        bg = np.concatenate([idx, bg_peaks[idx].ravel(order="F")])
    else:
        bg = np.arange(M.shape[0])
    return _compute_enrichment(M, names, idx, bg)


# ---------------------------------------------------------------------------
# Peak annotations from region sets
# ---------------------------------------------------------------------------

_COORD_RE = re.compile(r"^(.+?)[:_-](\d+)[-_](\d+)$")


def _parse_peak_coords(peak_names, adata=None):
    if adata is not None and {"chrom", "start", "end"}.issubset(adata.var.columns):
        v = adata.var
        return (v["chrom"].astype(str).to_numpy(), v["start"].to_numpy(np.int64),
                v["end"].to_numpy(np.int64))
    chrom, start, end = [], [], []
    for p in peak_names:
        m = _COORD_RE.match(str(p))
        if m is None:
            raise ValueError(f"cannot parse peak coordinate from {p!r} "
                             "(expected chrom:start-end)")
        chrom.append(m.group(1)); start.append(int(m.group(2))); end.append(int(m.group(3)))
    return np.asarray(chrom), np.asarray(start, np.int64), np.asarray(end, np.int64)


def _region_frame(region) -> pd.DataFrame:
    if isinstance(region, (str, os.PathLike)):
        df = pd.read_csv(region, sep=r"\s+", header=None, comment="#", usecols=[0, 1, 2],
                         engine="python")
        df.columns = ["chrom", "start", "end"]
    elif isinstance(region, pd.DataFrame):
        cols = {c.lower(): c for c in region.columns}
        ck = cols.get("chrom") or cols.get("seqnames") or cols.get("chromosome") or region.columns[0]
        sk = cols.get("start") or region.columns[1]
        ek = cols.get("end") or region.columns[2]
        df = pd.DataFrame({"chrom": region[ck].astype(str).to_numpy(),
                           "start": region[sk].to_numpy(), "end": region[ek].to_numpy()})
    else:
        raise TypeError("regions must be BED paths or DataFrames with chrom/start/end")
    df["chrom"] = df["chrom"].astype(str)
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    return df


def _overlap_any(chrom, start, end, regions: pd.DataFrame) -> np.ndarray:
    """Peaks overlapping any region (closed intervals, ignore strand)."""
    hit = np.zeros(len(chrom), dtype=bool)
    for c, sub in regions.groupby("chrom", sort=False):
        pk = np.flatnonzero(chrom == c)
        if pk.size == 0:
            continue
        o = np.argsort(sub["start"].to_numpy(), kind="stable")
        rs = sub["start"].to_numpy()[o]
        re_max = np.maximum.accumulate(sub["end"].to_numpy()[o])
        j = np.searchsorted(rs, end[pk], side="right")
        ok = j > 0
        hit[pk[ok]] = re_max[j[ok] - 1] >= start[pk[ok]]
    return hit


def _store(adata, key, M, names, info=None):
    adata.varm[key] = sparse.csr_matrix(M, dtype=bool)
    adata.uns[f"{key}_names"] = np.asarray(names, dtype=object)
    if info is not None:
        adata.uns[f"{key}_info"] = info


def add_peak_annotations(
    adata,
    regions,
    *,
    key_added: str = "Region",
    verbose: bool = True,
):
    """Binary peak x region-set overlap matrix (ArchR ``addPeakAnnotations``).

    Arguments:
        adata: peak AnnData (``var_names`` ``chrom:start-end`` or
            ``adata.var[['chrom','start','end']]``).
        regions: a dict ``{name: BED path | DataFrame}``, a list of those
            (named ``Region_1..n``), or a single DataFrame (named ``region``).
            BED columns 1-3 are used as-is (ArchR reads them with
            ``makeGRangesFromDataFrame`` without a 0/1-based shift) and all
            intervals are treated as closed.
        key_added: stored as ``adata.varm[key_added]`` and
            ``adata.uns[key_added + '_names']``.

    Region sets with no overlapping peak are dropped, as in ArchR.
    """
    if isinstance(regions, pd.DataFrame):
        regions = {"region": regions}
    elif not isinstance(regions, Mapping):
        regions = {f"Region_{i + 1}": r for i, r in enumerate(regions)}
    names = [str(k) for k in regions]
    if len(set(names)) != len(names):
        raise ValueError("Found duplicated region names! Please make unique!")
    chrom, start, end = _parse_peak_coords(adata.var_names, adata)
    cols = [_overlap_any(chrom, start, end, _region_frame(r)) for r in regions.values()]
    M = sparse.csr_matrix(np.column_stack(cols)) if cols else sparse.csr_matrix((len(chrom), 0))
    n_o = np.asarray(M.sum(axis=0)).ravel()
    if (n_o == 0).all():
        raise ValueError("No Overlaps Found between regions and peak Matrix!")
    keep = n_o > 0
    if verbose and (~keep).any():
        console.info(f"[add_peak_annotations] dropping region sets with 0 overlaps: "
                     f"{', '.join(np.asarray(names)[~keep])}")
    _store(adata, key_added, M[:, np.flatnonzero(keep)], np.asarray(names)[keep])
    return adata


def _download_anno(db: str, genome: str, cache_dir: Optional[str]) -> str:
    key = (db.lower(), genome.lower())
    if key not in ARCHR_ANNO_URLS:
        raise ValueError(f"No ArchR annotation database for db={db!r}, genome={genome!r}; "
                         f"available: {sorted(ARCHR_ANNO_URLS)}")
    url = ARCHR_ANNO_URLS[key]
    cache_dir = cache_dir or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "epione", "archr_annotations")
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(dest):
        import requests
        console.info(f"[add_archr_annotations] downloading {url}")
        tmp = dest + ".part"
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        os.replace(tmp, dest)
    return dest


def add_archr_annotations(
    adata,
    *,
    db: str = "ArchR",
    collection: str = "EncodeTFBS",
    genome: str = "hg19",
    key_added: Optional[str] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
):
    """Peak overlap with ArchR's curated region databases (``addArchRAnnotations``).

    Arguments:
        adata: peak AnnData.
        db: ``'ArchR'`` (collections ``ATAC``, ``EncodeTFBS``,
            ``CistromeTFBS``, ``Codex``), ``'LOLA'`` (``EncodeTFBS``,
            ``CistromeTFBS``, ``CistromeEpigenome``, ``Codex``,
            ``SheffieldDnase``), or a path to an ``.Anno`` (ArchRAnno HDF5) file.
        collection: collection within the database (case-insensitive).
        genome: ``hg19``/``hg38`` (ArchR) or also ``mm9``/``mm10`` (LOLA).
        key_added: ``adata.varm`` key (default: ``collection``).
        cache_dir: download cache (default ``~/.cache/epione/archr_annotations``).

    The database is downloaded once from the same URLs ArchR uses. Stores the
    sparse match matrix in ``adata.varm[key]``, names in
    ``adata.uns[key + '_names']`` and the per-region metadata table in
    ``adata.uns[key + '_info']``.
    """
    import h5py

    path = db if os.path.exists(str(db)) else _download_anno(db, genome, cache_dir)
    chrom, start, end = _parse_peak_coords(adata.var_names, adata)
    with h5py.File(path, "r") as f:
        cls = f["Class"][()]
        cls = cls[0] if np.ndim(cls) else cls
        if (cls.decode() if isinstance(cls, bytes) else str(cls)) != "ArchRAnno":
            raise ValueError("Not Valid ArchRAnno!")
        groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        match = [g for g in groups if g.lower() == collection.lower()]
        if not match:
            raise ValueError(f"Not Valid Collection in ArchRAnno Database! Options: {groups}")
        grp = f[match[0]]
        info = pd.DataFrame(grp["Info"][()])
        for c in info.columns:
            if info[c].dtype == object:
                info[c] = info[c].map(lambda s: s.decode() if isinstance(s, bytes) else s)
        rows, cols = [], []
        max_id = 0
        for c in pd.unique(chrom):
            if c not in grp:
                raise ValueError(f"Error! Chromosome {c} not in AnnoFile!")
            g = grp[c]
            lengths = np.asarray(g["IDLengths"][()]).ravel()
            if lengths.sum() == 0:
                continue
            ids = np.repeat(np.asarray(g["IDValues"][()]).ravel(), lengths).astype(np.int64)
            rng = np.asarray(g["Ranges"][()])
            if rng.shape[0] != 2:
                rng = rng.T
            rs = rng[0].astype(np.int64)
            re_ = rs + rng[1].astype(np.int64) - 1
            max_id = max(max_id, int(ids.max()))
            pk = np.flatnonzero(chrom == c)
            ps, pe = start[pk], end[pk]
            o = np.lexsort((rs, ids))
            ids_s, rs_s, re_s = ids[o], rs[o], re_[o]
            bounds = np.flatnonzero(np.diff(ids_s)) + 1
            for a, b in zip(np.r_[0, bounds], np.r_[bounds, ids_s.size]):
                emax = np.maximum.accumulate(re_s[a:b])
                j = np.searchsorted(rs_s[a:b], pe, side="right")
                ok = j > 0
                h = np.zeros(pk.size, dtype=bool)
                h[ok] = emax[j[ok] - 1] >= ps[ok]
                hp = pk[h]
                rows.append(hp)
                cols.append(np.full(hp.size, ids_s[a] - 1))
    rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
    n_ann = max(max_id, len(info))
    M = sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(len(chrom), n_ann))
    M.sum_duplicates()
    M.data[:] = 1.0
    names = info["name"].astype(str).to_numpy()
    n_o = np.asarray(M.sum(axis=0)).ravel()
    if (n_o == 0).all():
        raise ValueError("No Overlaps Found! Please check your peakSet and genome!")
    keep = np.flatnonzero(n_o > 0)
    if verbose:
        console.info(f"[add_archr_annotations] {collection}: {keep.size}/{n_ann} annotations overlap peaks")
    info = info.iloc[keep].set_index("name")
    _store(adata, key_added or match[0], M[:, keep], names[keep], info)
    return adata
