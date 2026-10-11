"""Topic binarization, imputed accessibility and DARs — pycisTopic ports (SCENIC+ 1.0).

* :func:`binarize_topics` — ``pycisTopic.topic_binarization.binarize_topics``
  (``otsu``, ``yen``, ``li``, ``ntop``; optional ``smooth_topics``).
* :func:`impute_accessibility`, :func:`normalize_scores`,
  :func:`find_highly_variable_features`, :func:`find_diff_features` —
  ``pycisTopic.diff_features``.

The Wilcoxon rank-sum tests of :func:`find_diff_features` are vectorised: each
region is ranked once over all cells and every one-vs-rest contrast reads its
rank sum from that ranking, instead of re-ranking per contrast and per region
in Ray workers. Rank sums are exact (half-integers), so the p-values equal
``scipy.stats.ranksums``'s.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import special

from ._np126 import argsort_compat, log1p_compat, log_compat
from ._topics import TopicModel

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - numba ships with scanpy
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


# ---------------------------------------------------------------------------
# Topic binarization
# ---------------------------------------------------------------------------

def _smooth_topics_distributions(dist: pd.DataFrame, compat: str = "numpy126") -> pd.DataFrame:
    """``smooth_topics_distributions``: x * (log x - mean(log x)) per row."""
    x = dist.to_numpy()
    lx = log_compat(x + 1e-100, compat)
    out = x * (lx - np.sum(lx, axis=1, keepdims=True) / x.shape[1])
    return pd.DataFrame(out, index=dist.index, columns=dist.columns)


def _histogram_and_bin_centers(array: np.ndarray, nbins: int = 100):
    hist, bin_edges = np.histogram(array.ravel(), bins=nbins, range=None)
    return hist, (bin_edges[:-1] + bin_edges[1:]) / 2.0


def _threshold_otsu(array: np.ndarray, nbins: int = 100) -> float:
    hist, bin_centers = _histogram_and_bin_centers(array, nbins)
    hist = hist.astype(float)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    return bin_centers[:-1][np.argmax(variance12)]


def _threshold_yen(array: np.ndarray, nbins: int = 100) -> float:
    hist, bin_centers = _histogram_and_bin_centers(array, nbins)
    pmf = hist.astype(np.float32) / hist.sum()
    P1 = np.cumsum(pmf)
    P1_sq = np.cumsum(pmf ** 2)
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) * (P1[:-1] * (1.0 - P1[:-1])) ** 2)
    return bin_centers[crit.argmax()]


def _cross_entropy(array: np.ndarray, threshold: float, nbins: int = 100,
                   compat: str = "numpy126") -> float:
    hist, bin_centers = _histogram_and_bin_centers(array, nbins=nbins)
    t = np.flatnonzero(bin_centers > threshold)[0]
    m0a = np.sum(hist[:t])
    m0b = np.sum(hist[t:])
    m1a = np.sum(hist[:t] * bin_centers[:t])
    m1b = np.sum(hist[t:] * bin_centers[t:])
    mua = m1a / m0a
    mub = m1b / m0b
    lg = log_compat(np.array([mua, mub]), compat)
    return -m1a * lg[0] - m1b * lg[1]


def _sort_desc_index(values: np.ndarray, compat: str) -> np.ndarray:
    """Row order of ``DataFrame.sort_values(col, ascending=False)`` (pandas nargsort)."""
    idx = np.arange(len(values))[::-1]
    return idx[argsort_compat(np.ascontiguousarray(values[::-1]), compat)][::-1]


def binarize_topics(
    model: TopicModel,
    target: str = "region",
    method: str = "otsu",
    smooth_topics: bool = True,
    ntop: int = 2000,
    predefined_thr: Optional[dict] = None,
    nbins: int = 100,
    compat: str = "numpy126",
) -> dict:
    """Binarize topic distributions; port of ``pycisTopic.binarize_topics``.

    Parameters
    ----------
    model
        A :class:`TopicModel`.
    target
        ``"region"`` (topic-region distributions) or ``"cell"`` (cell-topic).
    method
        ``"otsu"``, ``"yen"``, ``"li"`` or ``"ntop"``.
    smooth_topics
        Down-weight features that are likely in many topics before thresholding.
    ntop
        Number of top features kept per topic with ``method="ntop"``.
    predefined_thr
        ``{"TopicN": threshold}`` overriding the method for some topics.
    nbins
        Histogram bins for ``otsu``/``yen``/``li``.
    compat
        pycisTopic smooths with ``np.log`` and orders the selected features
        with pandas' quicksort; both depend on the numpy build.
        ``"numpy126"`` (default) reproduces numpy 1.26 on AVX-512 CPUs, the
        SCENIC+ reference stack, on any machine; ``"native"`` uses the
        installed numpy.

    Returns
    -------
    ``{"Topic1": DataFrame, ...}``: the selected features of each topic with
    their (smoothed) score, highest first.
    """
    if target == "region":
        dist = model.topic_region
    elif target == "cell":
        dist = model.cell_topic.T
    else:
        raise ValueError("target must be 'region' or 'cell'")
    if smooth_topics:
        dist = _smooth_topics_distributions(dist, compat)
    values = dist.to_numpy()
    out = {}
    for i in range(values.shape[1]):
        name = "Topic" + str(i + 1)
        col = dist.columns[i]
        l = np.asarray(values[:, i])
        l_norm = (l - np.min(l)) / np.ptp(l)
        if isinstance(predefined_thr, dict) and name in predefined_thr:
            thr = predefined_thr[name]
        elif method == "otsu":
            thr = _threshold_otsu(l_norm, nbins=nbins)
        elif method == "yen":
            thr = _threshold_yen(l_norm, nbins=nbins)
        elif method == "li":
            thresholds = np.arange(np.min(l_norm) + 0.01, np.max(l_norm) - 0.01, 0.01)
            entropies = [_cross_entropy(l_norm, t, nbins=nbins, compat=compat) for t in thresholds]
            thr = thresholds[np.argmin(entropies)]
        elif method == "ntop":
            # value at position ``ntop`` of the descending order (ties irrelevant)
            thr = float(-np.partition(-l_norm, ntop)[ntop])
        else:
            raise ValueError("method must be 'otsu', 'yen', 'li' or 'ntop'")
        sel = np.flatnonzero(l_norm > thr)
        sub = values[sel, i]
        order = _sort_desc_index(sub, compat)
        out[name] = pd.DataFrame({col: sub[order]}, index=dist.index[sel[order]])
    return out


# ---------------------------------------------------------------------------
# Imputed accessibility
# ---------------------------------------------------------------------------

@dataclass
class ImputedFeatures:
    """Regions × cells matrix (mirror of ``pycisTopic.CistopicImputedFeatures``)."""

    mtx: np.ndarray
    feature_names: list
    cell_names: list

    def __repr__(self) -> str:
        return (f"ImputedFeatures(features={len(self.feature_names)}, "
                f"cells={len(self.cell_names)}, dtype={self.mtx.dtype})")

    def subset(self, cells: Optional[Sequence[str]] = None,
               features: Optional[Sequence[str]] = None) -> "ImputedFeatures":
        """Select cells / features (in the given order) and drop all-zero rows."""
        mtx, cell_names, feature_names = self.mtx, list(self.cell_names), list(self.feature_names)
        if cells is not None:
            pos = {c: i for i, c in enumerate(cell_names)}
            ci = np.array([pos[c] for c in cells], dtype=np.int64)
            mtx = mtx[:, ci]
            cell_names = [cell_names[i] for i in ci]
        if features is not None:
            pos = {f: i for i, f in enumerate(feature_names)}
            fi = np.array([pos[f] for f in features], dtype=np.int64)
            mtx = mtx[fi, :]
            feature_names = [feature_names[i] for i in fi]
        keep = np.flatnonzero(np.count_nonzero(mtx, axis=1))
        if keep.size != mtx.shape[0]:
            mtx = mtx[keep, :]
            feature_names = [feature_names[i] for i in keep]
        return ImputedFeatures(mtx, feature_names, cell_names)

    def to_anndata(self):
        """Cells × features AnnData view of the matrix."""
        import anndata as ad

        a = ad.AnnData(np.asarray(self.mtx).T)
        a.obs_names = list(self.cell_names)
        a.var_names = list(self.feature_names)
        return a


def impute_accessibility(
    model: TopicModel,
    cells: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
    scale_factor: Union[int, float] = 10 ** 6,
    chunk_size: int = 20000,
) -> ImputedFeatures:
    """Region accessibility imputed from the topic model.

    Port of ``pycisTopic.diff_features.impute_accessibility``: the float32
    product ``topic_region @ cell_topic`` per chunk of regions, scaled by
    ``scale_factor`` and truncated to int32 when ``scale_factor`` is an
    integer other than 1; all-zero regions are dropped. The float32 product is
    computed by the BLAS ``sgemm`` kernel, so exact integer equality with
    pycisTopic holds when both run on the same BLAS build and CPU family.
    """
    cell_topic = model.cell_topic
    topic_region = model.topic_region
    cell_names = list(cell_topic.columns) if cells is None else list(cells)
    region_names = list(topic_region.index) if regions is None else list(regions)
    if cells is not None:
        cell_topic = cell_topic.loc[:, cell_names]
    if regions is not None:
        topic_region = topic_region.loc[region_names]
    ct = cell_topic.to_numpy().astype(np.float32)
    tr = topic_region.to_numpy().astype(np.float32)
    to_int = isinstance(scale_factor, (int, np.integer)) and scale_factor != 1
    out = np.empty((tr.shape[0], ct.shape[1]), dtype=np.int32 if to_int else np.float32)
    kept = []
    end = 0
    for start in range(0, tr.shape[0], chunk_size):
        chunk = tr[start:start + chunk_size] @ ct
        if to_int:
            chunk *= np.float32(scale_factor)
            chunk = chunk.astype(np.int32)
        keep = np.flatnonzero(np.count_nonzero(chunk, axis=1))
        out[end:end + keep.size] = chunk[keep]
        kept.extend(region_names[start + j] for j in keep)
        end += keep.size
    return ImputedFeatures(out[:end], kept, cell_names)


def normalize_scores(imputed: ImputedFeatures, scale_factor: Union[int, float] = 10 ** 4,
                     compat: str = "numpy126") -> ImputedFeatures:
    """``log1p(x / (column sum / scale_factor))``; port of ``normalize_scores``.

    ``compat="numpy126"`` evaluates ``log1p`` as numpy 1.26 does on AVX-512
    CPUs (SVML), the SCENIC+ reference stack; ``"native"`` uses the installed
    numpy.
    """
    mtx = imputed.mtx.toarray() if sp.issparse(imputed.mtx) else imputed.mtx
    normalized = mtx / (np.sum(mtx, axis=0) / scale_factor)
    log1p_compat(normalized, compat, out=normalized)
    return ImputedFeatures(normalized, list(imputed.feature_names), list(imputed.cell_names))


def find_highly_variable_features(
    normalized: Union[ImputedFeatures, pd.DataFrame],
    min_disp: float = 0.05,
    min_mean: float = 0.0125,
    max_disp: float = np.inf,
    max_mean: float = 3,
    n_bins: int = 20,
    n_top_features: Optional[int] = None,
    return_table: bool = False,
):
    """Seurat-style highly variable regions; port of ``find_highly_variable_features``.

    Returns the list of variable feature names (and the per-feature table of
    means / dispersions when ``return_table``).
    """
    if isinstance(normalized, pd.DataFrame):
        mat = normalized.values
        features = normalized.index.tolist()
    else:
        mat = normalized.mtx
        features = list(normalized.feature_names)
    if sp.issparse(mat):
        from sklearn.utils.sparsefuncs import mean_variance_axis

        mean, var = mean_variance_axis(mat, axis=1)
    else:
        mean = np.mean(mat, axis=1, dtype=np.float32)
        var = np.var(mat, axis=1, dtype=np.float32)
    mean[mean == 0] = 1e-12
    dispersion = var / mean
    dispersion[dispersion == 0] = np.nan
    dispersion = np.log(dispersion)
    df = pd.DataFrame()
    df["means"] = mean
    df["dispersions"] = dispersion
    df["mean_bin"] = pd.cut(df["means"], bins=n_bins)
    disp_grouped = df.groupby("mean_bin", observed=False)["dispersions"]
    disp_mean_bin = disp_grouped.mean()
    disp_std_bin = disp_grouped.std(ddof=1)
    one_feature_per_bin = disp_std_bin.isnull()
    disp_std_bin[one_feature_per_bin.values] = disp_mean_bin[one_feature_per_bin.values].values
    disp_mean_bin[one_feature_per_bin.values] = 0
    df["dispersions_norm"] = (
        df["dispersions"].values - disp_mean_bin[df["mean_bin"].values].values
    ) / disp_std_bin[df["mean_bin"].values].values
    dispersion_norm = df["dispersions_norm"].values.astype("float32")
    if n_top_features is not None:
        dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
        dispersion_norm[::-1].sort()
        disp_cut_off = dispersion_norm[n_top_features - 1]
        subset = np.nan_to_num(df["dispersions_norm"].values) >= disp_cut_off
    else:
        dispersion_norm[np.isnan(dispersion_norm)] = 0
        subset = np.logical_and.reduce((mean > min_mean, mean < max_mean,
                                        dispersion_norm > min_disp, dispersion_norm < max_disp))
    df["highly_variable"] = subset
    var_features = [features[i] for i in np.flatnonzero(subset)]
    if return_table:
        df.index = features
        return var_features, df
    return var_features


# ---------------------------------------------------------------------------
# Differentially accessible regions
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _rank_rows(mat):
    """Average ranks (1-based, ties averaged) of every row, as ``scipy.stats.rankdata``."""
    n_rows, n = mat.shape
    out = np.empty((n_rows, n), dtype=np.float64)
    for r in prange(n_rows):
        row = mat[r]
        order = np.argsort(row, kind="mergesort")
        i = 0
        while i < n:
            j = i
            v = row[order[i]]
            while j + 1 < n and row[order[j + 1]] == v:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for t in range(i, j + 1):
                out[r, order[t]] = avg
            i = j + 1
    return out


@njit(cache=True)
def _log2_fc(fg_mean, bg_mean):
    """``np.log2((fg + 1e-12) / (bg + 1e-12))`` evaluated in numba, as pycisTopic does."""
    out = np.empty(fg_mean.shape[0])
    for i in range(fg_mean.shape[0]):
        out[i] = np.log2((fg_mean[i] + 10 ** -12) / (bg_mean[i] + 10 ** -12))
    return out


def _p_adjust_bh(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


def find_diff_features(
    imputed: ImputedFeatures,
    groups: pd.Series,
    var_features: Optional[Sequence[str]] = None,
    contrasts: Optional[list] = None,
    adjpval_thr: float = 0.05,
    log2fc_thr: float = np.log2(1.5),
    chunk_size: int = 20000,
) -> dict:
    """Differentially accessible regions; port of ``pycisTopic.find_diff_features``.

    Parameters
    ----------
    imputed
        Imputed accessibility (:func:`impute_accessibility`).
    groups
        Cell → group label (cells missing from ``imputed`` or with NaN labels
        are ignored).
    var_features
        Regions to test (e.g. :func:`find_highly_variable_features`).
    contrasts
        ``[[[fg groups], [bg groups]], ...]``; default: each group vs the rest.
    adjpval_thr, log2fc_thr
        Thresholds on the BH-adjusted Wilcoxon p-value and the log2 fold change.

    Returns
    -------
    ``{contrast name: DataFrame}`` with ``Log2FC``, ``Adjusted_pval`` and
    ``Contrast`` per region, sorted by decreasing ``Log2FC``.
    """
    cell_set = set(map(str, imputed.cell_names))
    groups = groups[[c in cell_set for c in map(str, groups.index)]].dropna()
    if contrasts is None:
        levels = sorted(set(groups.tolist()))
        contrasts = [[[x], levels[:levels.index(x)] + levels[levels.index(x) + 1:]]
                     for x in levels]
        names = levels
    else:
        names = ["_".join(c[0]) + "_VS_" + "_".join(c[1]) for c in contrasts]
    sub = imputed.subset(features=var_features)
    mat = sub.mtx
    col = {c: i for i, c in enumerate(map(str, sub.cell_names))}
    labels = groups.to_numpy()
    cell_pos = np.array([col[str(c)] for c in groups.index], dtype=np.int64)

    fg_idx, bg_idx = [], []
    for fg, bg in contrasts:
        fg_idx.append(cell_pos[np.isin(labels, fg)])
        bg_idx.append(cell_pos[np.isin(labels, bg)])

    n_feat = mat.shape[0]
    pvals = np.empty((len(contrasts), n_feat))
    log2fc = np.empty((len(contrasts), n_feat))
    # contrasts sharing the same tested cells share one ranking
    union_keys = [tuple(np.sort(np.concatenate([f, b]))) for f, b in zip(fg_idx, bg_idx)]
    for start in range(0, n_feat, chunk_size):
        block = mat[start:start + chunk_size]
        ranks_by_union = {}
        for c, (f, b) in enumerate(zip(fg_idx, bg_idx)):
            key = union_keys[c]
            if key not in ranks_by_union:
                cols = np.asarray(key, dtype=np.int64)
                ranks = _rank_rows(np.ascontiguousarray(block[:, cols]))
                ranks_by_union[key] = (cols, ranks)
            cols, ranks = ranks_by_union[key]
            pos = np.searchsorted(cols, f)
            n1, n2 = f.size, b.size
            s = ranks[:, pos].sum(axis=1)
            expected = n1 * (n1 + n2 + 1) / 2.0
            z = (s - expected) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
            pvals[c, start:start + block.shape[0]] = 2 * special.ndtr(-np.abs(z))
            fg_mean = block[:, f].sum(axis=1, dtype=np.int64) / n1
            bg_mean = block[:, b].sum(axis=1, dtype=np.int64) / n2
            log2fc[c, start:start + block.shape[0]] = _log2_fc(fg_mean, bg_mean)

    features = list(sub.feature_names)
    out = {}
    for c, name in enumerate(names):
        adj = _p_adjust_bh(pvals[c])
        df = pd.DataFrame({"Log2FC": log2fc[c], "Adjusted_pval": adj,
                           "Contrast": [name] * n_feat}, index=features)
        df = df.loc[df["Adjusted_pval"] <= adjpval_thr]
        df = df.loc[df["Log2FC"] >= log2fc_thr]
        df = df.sort_values(["Log2FC", "Adjusted_pval"], ascending=[False, True], kind="stable")
        out[name] = df
    return out


# ---------------------------------------------------------------------------
# Region sets
# ---------------------------------------------------------------------------

def write_region_sets(folder, region_sets: dict) -> None:
    """Write ``{collection: {name: region names}}`` as BED files.

    Each set becomes ``folder/<collection>/<name>.bed`` (``chrom, start,
    end``), sorted by chromosome name, start and end — the input layout of
    pycistarget's cisTarget/DEM runs.
    """
    from pathlib import Path

    for collection, sets in region_sets.items():
        out = Path(folder) / collection
        out.mkdir(parents=True, exist_ok=True)
        for name, regions in sets.items():
            df = pd.DataFrame([(r.split(":")[0], *map(int, r.split(":")[1].split("-")))
                               for r in regions], columns=["Chromosome", "Start", "End"])
            df = df.sort_values(["Chromosome", "Start", "End"])
            df.to_csv(out / f"{name}.bed", sep="\t", header=False, index=False)
