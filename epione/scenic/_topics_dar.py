"""Topic binarization, imputed accessibility and DARs (SCENIC+ 1.0, pycisTopic behaviour).

* :func:`binarize_topics` — same results as ``pycisTopic.binarize_topics``
  (``otsu``, ``yen``, ``li``, ``ntop``; optional ``smooth_topics``).
* :func:`impute_accessibility`, :func:`normalize_scores`,
  :func:`find_highly_variable_features`, :func:`find_diff_features` — same
  results as the functions of ``pycisTopic.diff_features``.

pycisTopic is distributed under a non-commercial academic license; nothing here
is copied from it. The behaviour was reimplemented from the published methods
(Otsu and Yen thresholds in the histogram form of scikit-image, BSD-3; Li & Lee
cross-entropy; Seurat dispersion selection as in scanpy, BSD-3) and checked
against pycisTopic's outputs.

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


def _hist_centres(values: np.ndarray, nbins: int):
    """Counts and bin midpoints of an ``nbins`` histogram over the data range."""
    counts, edges = np.histogram(values.ravel(), bins=nbins, range=None)
    return counts, (edges[:-1] + edges[1:]) / 2.0


def _threshold_otsu(values: np.ndarray, nbins: int = 100) -> float:
    """Otsu's threshold on a histogram (the histogram form of scikit-image, BSD-3)."""
    counts, centres = _hist_centres(values, nbins)
    counts = counts.astype(float)
    mass = counts * centres
    below_n = np.cumsum(counts)
    above_n = np.cumsum(counts[::-1])[::-1]
    below_mu = np.cumsum(mass) / below_n
    above_mu = (np.cumsum(mass[::-1]) / above_n[::-1])[::-1]
    between = below_n[:-1] * above_n[1:] * (below_mu[:-1] - above_mu[1:]) ** 2
    return centres[:-1][np.argmax(between)]


def _threshold_yen(values: np.ndarray, nbins: int = 100) -> float:
    """Yen's threshold (histogram form of scikit-image, BSD-3; float32 pmf)."""
    counts, centres = _hist_centres(values, nbins)
    pmf = counts.astype(np.float32) / counts.sum()
    cum = np.cumsum(pmf)
    cum_sq = np.cumsum(pmf ** 2)
    rev_cum_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    score = np.log(((cum_sq[:-1] * rev_cum_sq[1:]) ** -1) * (cum[:-1] * (1.0 - cum[:-1])) ** 2)
    return centres[score.argmax()]


def _li_cross_entropy(values: np.ndarray, threshold: float, nbins: int = 100,
                      compat: str = "numpy126") -> float:
    """Li & Lee (1993) minimum cross-entropy criterion for one candidate threshold.

    The histogram is split at the first bin whose midpoint exceeds
    ``threshold``; with first moments ``m_b``, ``m_f`` and means ``mu_b``,
    ``mu_f`` of the two parts the criterion is ``-m_b log mu_b - m_f log mu_f``.
    """
    counts, centres = _hist_centres(values, nbins)
    cut = np.flatnonzero(centres > threshold)[0]
    n_b, n_f = np.sum(counts[:cut]), np.sum(counts[cut:])
    m_b = np.sum(counts[:cut] * centres[:cut])
    m_f = np.sum(counts[cut:] * centres[cut:])
    log_mu = log_compat(np.array([m_b / n_b, m_f / n_f]), compat)
    return -m_b * log_mu[0] - m_f * log_mu[1]


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
    """Binarize topic distributions; reproduces ``pycisTopic.binarize_topics``.

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
        topic = f"Topic{i + 1}"
        score = np.asarray(values[:, i])
        scaled = (score - np.min(score)) / np.ptp(score)   # min-max to [0, 1]
        if isinstance(predefined_thr, dict) and topic in predefined_thr:
            cutoff = predefined_thr[topic]
        elif method == "otsu":
            cutoff = _threshold_otsu(scaled, nbins=nbins)
        elif method == "yen":
            cutoff = _threshold_yen(scaled, nbins=nbins)
        elif method == "li":
            # candidate cut-offs every 0.01 strictly inside the range
            grid = np.arange(np.min(scaled) + 0.01, np.max(scaled) - 0.01, 0.01)
            crit = [_li_cross_entropy(scaled, g, nbins=nbins, compat=compat) for g in grid]
            cutoff = grid[np.argmin(crit)]
        elif method == "ntop":
            # the (ntop+1)-th largest value; features strictly above it are kept
            cutoff = float(-np.partition(-scaled, ntop)[ntop])
        else:
            raise ValueError("method must be 'otsu', 'yen', 'li' or 'ntop'")
        keep = np.flatnonzero(scaled > cutoff)
        kept = values[keep, i]
        order = _sort_desc_index(kept, compat)
        out[topic] = pd.DataFrame({dist.columns[i]: kept[order]}, index=dist.index[keep[order]])
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

    Reproduces ``pycisTopic.diff_features.impute_accessibility``: the float32
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
    """``log1p(x / (column sum / scale_factor))``; reproduces ``normalize_scores``.

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
    """Seurat-style highly variable regions; reproduces ``find_highly_variable_features``.

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
        # float32 accumulation, as the reference
        mean = np.mean(mat, axis=1, dtype=np.float32)
        var = np.var(mat, axis=1, dtype=np.float32)
    # Seurat "mean.var.plot" selection (as in scanpy's ``flavor="seurat"``,
    # BSD-3): log dispersion z-scored within 20 equal-width bins of the mean
    mean[mean == 0] = 1e-12
    disp = var / mean
    disp[disp == 0] = np.nan
    disp = np.log(disp)
    table = pd.DataFrame({"means": mean, "dispersions": disp})
    table["mean_bin"] = pd.cut(table["means"], bins=n_bins)
    by_bin = table.groupby("mean_bin", observed=False)["dispersions"]
    bin_mean = by_bin.mean()
    bin_sd = by_bin.std(ddof=1)
    # a bin holding a single feature has no spread: give it a normalised dispersion of 1
    lonely = bin_sd.isnull().values
    bin_sd[lonely] = bin_mean[lonely].values
    bin_mean[lonely] = 0
    bins = table["mean_bin"].values
    table["dispersions_norm"] = ((table["dispersions"].values - bin_mean[bins].values)
                                 / bin_sd[bins].values)
    norm32 = table["dispersions_norm"].values.astype("float32")
    if n_top_features is not None:
        ranked = norm32[~np.isnan(norm32)]
        ranked[::-1].sort()
        floor = ranked[n_top_features - 1]
        selected = np.nan_to_num(table["dispersions_norm"].values) >= floor
    else:
        norm32[np.isnan(norm32)] = 0
        selected = ((mean > min_mean) & (mean < max_mean) & (norm32 > min_disp)
                    & (norm32 < max_disp))
    table["highly_variable"] = selected
    var_features = [features[i] for i in np.flatnonzero(selected)]
    if return_table:
        table.index = features
        return var_features, table
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


def _p_adjust_bh(pvals) -> np.ndarray:
    """Benjamini-Hochberg q-values (step-up, capped at 1)."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    desc = pvals.argsort()[::-1]                     # largest p first
    scale = float(n) / np.arange(n, 0, -1)           # n / rank
    q_desc = np.minimum(1, np.minimum.accumulate(scale * pvals[desc]))
    q = np.empty(n)
    q[desc] = q_desc
    return q


def find_diff_features(
    imputed: ImputedFeatures,
    groups: pd.Series,
    var_features: Optional[Sequence[str]] = None,
    contrasts: Optional[list] = None,
    adjpval_thr: float = 0.05,
    log2fc_thr: float = np.log2(1.5),
    chunk_size: int = 20000,
) -> dict:
    """Differentially accessible regions; reproduces ``pycisTopic.find_diff_features``.

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
