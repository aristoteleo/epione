"""Cicero: cis-co-accessibility networks from single-cell chromatin accessibility.

Port of the R package **cicero** for Monocle 3 (Pliner et al., *Mol Cell* 2018;
cole-trapnell-lab/cicero-release, branch ``monocle3``, v1.3.9) into epione,
following the omicverse-rebuildr protocol: the R source is the specification,
parity is gated against R on pre-registered fixtures (``tests/parity/cicero``).

Pipeline (R name -> Python name, both in :mod:`epione.tl`)::

    make_cicero_cds              -> make_cicero_cds          kNN "bagging" of cells into aggregates
    estimate_distance_parameter  -> estimate_distance_parameter
    generate_cicero_models       -> generate_cicero_models   windowed graphical lasso
    assemble_connections         -> assemble_connections     Peak1 / Peak2 / coaccess table
    run_cicero                   -> run_cicero               the three steps above
    generate_ccans               -> generate_ccans           Louvain modules of co-accessible peaks
    annotate_cds_by_site         -> annotate_cds_by_site
    build_gene_activity_matrix   -> build_gene_activity_matrix
    normalize_gene_activities    -> normalize_gene_activities

Data model: R works on a ``cell_data_set`` (peaks x cells); here the input is
an AnnData with cells in ``obs`` and peaks in ``var`` (names ``chr_start_end``,
``chr:start-end`` or ``chr-start-end``). ``make_cicero_cds`` returns an AnnData
with one row per aggregated cell bin.

Randomness. R draws from its global Mersenne-Twister stream (``sample()`` in
``make_cicero_cds`` and ``estimate_distance_parameter``). With ``rng="R"``
(default) these draws come from :class:`epione.tl._rrng.RRandom`, a bit-exact
emulation of R's generator: ``random_state=s`` reproduces ``set.seed(s)``
called right before the R function, and passing one ``RRandom`` object to
several calls continues a single stream exactly like an R session does.
``rng="numpy"`` uses :func:`numpy.random.default_rng` instead (results then
agree with R only in distribution).
"""
from __future__ import annotations

import math
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from numba import njit

from ._glasso import _glasso_core
from ._rrng import RRandom

__all__ = [
    "make_cicero_cds", "estimate_distance_parameter", "generate_cicero_models",
    "assemble_connections", "run_cicero", "generate_ccans", "find_overlapping_ccans",
    "annotate_cds_by_site", "build_gene_activity_matrix", "normalize_gene_activities",
    "generate_windows", "cicero_connections_to_celloracle", "CiceroModel",
]

_LD = np.longdouble


# ============================================================================ helpers
def _log(silent: bool, *msg):
    if not silent:
        print(*msg, flush=True)


def _r_mean(x) -> float:
    """R's ``mean()`` of a double vector: long-double sum, then a second
    long-double pass that adds the mean residual (``summary.c``)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return float("nan")
    xl = x.astype(_LD)
    s = np.cumsum(xl)[-1] / n          # cumsum: strictly sequential accumulation
    if np.isfinite(s):
        t = np.cumsum(xl - s)[-1]
        s = s + t / n
    return float(s)


def _split_peak_names(names) -> pd.DataFrame:
    """``split_peak_names``: split on the last two of ``:``, ``-`` or ``_``."""
    out = []
    for nm in names:
        parts = re.split(r"[:_-]", str(nm)[::-1], maxsplit=2)
        parts = [p[::-1] for p in parts] + [""] * (3 - len(parts))
        out.append((parts[2], parts[1], parts[0]))
    return pd.DataFrame(out, columns=["chr", "bp1", "bp2"])


def _peak_coords(var: pd.DataFrame, var_names) -> pd.DataFrame:
    """chr / bp1 / bp2 as R gets them: from ``var`` if all three exist,
    otherwise parsed from the peak names (``df_for_coords``)."""
    if all(c in var.columns for c in ("chr", "bp1", "bp2")):
        df = var[["chr", "bp1", "bp2"]].copy()
        df.index = range(len(df))
    else:
        df = _split_peak_names(var_names)
    df["chr"] = df["chr"].astype(str)
    df["bp1"] = pd.to_numeric(df["bp1"]).astype(np.float64)
    df["bp2"] = pd.to_numeric(df["bp2"]).astype(np.float64)
    return df


def _reduced_coordinates(adata: AnnData, reduced_coordinates) -> np.ndarray:
    if isinstance(reduced_coordinates, str):
        rc = adata.obsm[reduced_coordinates]
        rc = rc.to_numpy() if isinstance(rc, pd.DataFrame) else np.asarray(rc)
    elif isinstance(reduced_coordinates, pd.DataFrame):
        if set(reduced_coordinates.index) != set(adata.obs_names):
            raise ValueError("row names of reduced_coordinates must match adata.obs_names")
        rc = reduced_coordinates.loc[adata.obs_names].to_numpy()
    else:
        rc = np.asarray(reduced_coordinates)
    rc = np.asarray(rc, dtype=np.float64)
    if rc.shape[0] != adata.n_obs:
        raise ValueError("reduced_coordinates must have one row per cell")
    return rc


def _make_rng(random_state, rng: str):
    if isinstance(random_state, RRandom):
        return "R", random_state
    if isinstance(random_state, np.random.Generator):
        return "numpy", random_state
    if rng == "R":
        return "R", RRandom(0 if random_state is None else int(random_state))
    if rng == "numpy":
        return "numpy", np.random.default_rng(random_state)
    raise ValueError("rng must be 'R' or 'numpy'")


def _draw_index(kind, gen, n: int) -> int:
    """0-based equivalent of R's ``sample(seq_len(n), 1)``."""
    if kind == "R":
        return int(gen.unif_index(n))
    return int(gen.integers(n))


def _knn_index(coords: np.ndarray, k: int) -> np.ndarray:
    """``FNN::knn.index(coords, k)``: k nearest neighbours of every point,
    excluding the point itself, as 0-based indices ordered by distance."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    idx = np.asarray(idx).reshape(coords.shape[0], k + 1)
    out = np.empty((coords.shape[0], k), dtype=np.int64)
    rows = np.arange(coords.shape[0])
    is_self = idx == rows[:, None]
    has_self = is_self.any(1)
    for i in range(coords.shape[0]):
        if has_self[i]:
            out[i] = idx[i][~is_self[i]][:k]
        else:  # duplicated coordinates: FNN drops the first hit
            out[i] = idx[i][1:]
    return out


def _sparse_x(adata: AnnData, layer: Optional[str]):
    X = adata.layers[layer] if layer is not None else adata.X
    return sp.csr_matrix(X, dtype=np.float64) if not sp.issparse(X) else X.tocsr().astype(np.float64)


def _estimate_sf(counts_by_cell: sp.csr_matrix, round_exprs: bool = True):
    """monocle3 ``estimate_sf_sparse`` (method ``mean-geometric-mean-total``);
    ``counts_by_cell`` is cells x features. Returns None when a cell has zero
    total (monocle3 warns and leaves the CDS unchanged)."""
    M = counts_by_cell
    if round_exprs:
        M = M.copy()
        M.data = np.round(M.data)                      # R round(): half to even
    cell_total = np.asarray(M.sum(axis=1)).ravel()
    if np.any(cell_total == 0):
        return None
    logs = np.array([math.log(v) for v in cell_total])  # libm log, as R
    sfs = cell_total / math.exp(_r_mean(logs))
    sfs[np.isnan(sfs)] = 1.0
    return sfs


# ============================================================================ make_cicero_cds
def make_cicero_cds(
    adata: AnnData,
    reduced_coordinates: Union[str, np.ndarray, pd.DataFrame] = "X_umap",
    k: int = 50,
    summary_stats: Optional[Sequence[str]] = None,
    size_factor_normalize: bool = True,
    silent: bool = False,
    return_agg_info: bool = False,
    *,
    random_state: Union[int, RRandom, np.random.Generator, None] = 0,
    rng: str = "R",
    chosen: Optional[Sequence[int]] = None,
    layer: Optional[str] = None,
):
    """Aggregate similar cells into overlapping bins -- R ``make_cicero_cds``.

    A kNN graph (``k - 1`` neighbours, Euclidean, on ``reduced_coordinates``)
    is built; cells are visited in random order and a cell's neighbourhood
    (itself + its neighbours) becomes a bin unless it shares ``>= 0.9 k``
    cells with an already accepted bin (at most 5000 candidates are tried,
    as in R). Accessibility is summed per bin, ``detect_genes(min_expr=0.1)``
    and ``estimate_size_factors`` are applied, and (by default) each bin is
    divided by its size factor.

    Parameters
    ----------
    adata
        Cells x peaks AnnData (counts in ``X`` or ``layer``; the Cicero
        vignette binarises them first). Peak names like ``chr1_100_600``.
    reduced_coordinates
        ``adata.obsm`` key, array (cells x d) or DataFrame indexed by cell.
        Cicero recommends a 2-D UMAP / tSNE.
    k
        Cells per bin (R default 50).
    summary_stats
        Numeric ``adata.obs`` columns averaged per bin (``mean_<col>``).
    size_factor_normalize
        Divide each bin by its size factor (R default ``TRUE``).
    silent
        Suppress the overlap QC message.
    return_agg_info
        Also return the cell -> bin assignment table (R ``return_agg_info``).
    random_state, rng
        ``rng="R"`` (default): ``random_state=s`` reproduces R's
        ``set.seed(s); make_cicero_cds(...)``; an :class:`RRandom` continues a
        stream. ``rng="numpy"``: numpy Generator (distributional agreement).
    chosen
        Inject the accepted bins (0-based indices of the seed cells, in R's
        order, e.g. from R's ``return_agg_info``). Skips the random draws.
    layer
        Use ``adata.layers[layer]`` instead of ``X``.

    Returns
    -------
    AnnData of bins x peaks (``X`` = aggregated, normalised accessibility,
    CSR float64) with ``obs['agg_cell']``, ``obs['num_genes_expressed']``,
    ``obs['Size_Factor']`` and ``var[['chr', 'bp1', 'bp2',
    'num_cells_expressed']]``. With ``return_agg_info=True`` a tuple
    ``(cicero_adata, agg_map)``.
    """
    if not (isinstance(k, (int, np.integer)) and k > 1):
        raise ValueError("k must be an integer > 1")
    if summary_stats is not None:
        summary_stats = [summary_stats] if isinstance(summary_stats, str) else list(summary_stats)
        miss = [c for c in summary_stats if c not in adata.obs.columns]
        if miss:
            raise ValueError(f"summary_stats missing from adata.obs: {miss}")
        for c in summary_stats:
            if not pd.api.types.is_numeric_dtype(adata.obs[c]):
                raise ValueError("All columns in summary_stats must be numeric")
    coords = _reduced_coordinates(adata, reduced_coordinates)
    n = adata.n_obs
    nn_map = np.concatenate([_knn_index(coords, k - 1), np.arange(n)[:, None]], axis=1)  # + self (R cbind)

    if chosen is None:
        kind, gen = _make_rng(random_state, rng)
        good = list(range(n))
        c0 = _draw_index(kind, gen, len(good))
        chosen_l = [good.pop(c0)]
        # inverted index cell -> accepted bins; shared(cand, bin) = |cells(cand) & cells(bin)|
        # (R: 2k - length(union(...)), each row of nn_map has k distinct cells)
        cell_bins: List[List[int]] = [[] for _ in range(n)]
        for c in nn_map[chosen_l[0]]:
            cell_bins[c].append(0)
        thr = 0.9 * k
        it = 0
        while len(good) > 0 and it < 5000:
            it += 1
            ch = _draw_index(kind, gen, len(good))
            cand = good.pop(ch)
            hits = [b for c in nn_map[cand] for b in cell_bins[c]]
            mx = np.bincount(hits).max() if hits else 0
            if mx < thr:
                b = len(chosen_l)
                chosen_l.append(cand)
                for c in nn_map[cand]:
                    cell_bins[c].append(b)
        chosen_arr = np.asarray(chosen_l, dtype=np.int64)
    else:
        chosen_arr = np.asarray(list(chosen), dtype=np.int64)

    cell_sample = nn_map[chosen_arr]                       # bins x k
    nb = len(chosen_arr)
    mask = sp.csr_matrix((np.ones(nb * k), (np.repeat(np.arange(nb), k), cell_sample.ravel())),
                         shape=(nb, n))
    mask.data[:] = 1.0                                     # (duplicates impossible; be safe)
    if not silent and nb > 1:
        C = (mask @ mask.T).toarray()
        iu = np.triu_indices(nb, 1)
        shared = C[iu]
        msg = ("Overlap QC metrics:\nCells per bin: {}\nMaximum shared cells bin-bin: {}\n"
               "Mean shared cells bin-bin: {}\nMedian shared cells bin-bin: {}").format(
            k, int(shared.max()), _fmt(shared.mean()), _fmt(float(np.median(shared))))
        print(msg, flush=True)
        if shared.mean() / k > 0.1:
            warnings.warn("On average, more than 10% of cells are shared between paired bins.")

    X = _sparse_x(adata, layer)
    agg = (mask @ X).tocsr()                               # bins x peaks (R: exprs %*% mask)
    agg.sort_indices()
    agg_names = np.array([f"agg_{c + 1}" for c in chosen_arr])

    obs = pd.DataFrame({"agg_cell": agg_names}, index=agg_names)
    if summary_stats:
        sub_obs = adata.obs[summary_stats].to_numpy(dtype=np.float64)
        for j, c in enumerate(summary_stats):
            obs["mean_" + c] = [_r_mean(sub_obs[row, j]) for row in cell_sample]
    var = adata.var.copy()
    coords_df = _peak_coords(adata.var, adata.var_names)
    if not all(c in var.columns for c in ("chr", "bp1", "bp2")):
        for c in ("chr", "bp1", "bp2"):
            var[c] = coords_df[c].to_numpy()
    # detect_genes(min_expr = 0.1) on the aggregated counts
    var["num_cells_expressed"] = np.asarray((agg > 0.1).sum(axis=0)).ravel()
    obs["num_genes_expressed"] = np.asarray((agg > 0.1).sum(axis=1)).ravel()
    sf = _estimate_sf(agg)
    if sf is None:
        raise ValueError("an aggregated bin has zero total accessibility; size factors are undefined "
                         "(monocle3::estimate_size_factors fails the same way)")
    obs["Size_Factor"] = sf
    out_x = agg
    if size_factor_normalize:
        out_x = _row_divide(agg, sf)
        sf2 = _estimate_sf(out_x)                          # new_cell_data_set() re-estimates on the normalised data
        if sf2 is not None:
            obs["Size_Factor"] = sf2
    cds = AnnData(X=out_x, obs=obs, var=var)
    cds.uns["cicero"] = {"k": int(k), "chosen": chosen_arr, "size_factor_normalize": bool(size_factor_normalize),
                         "agg_size_factor": sf}
    if return_agg_info:
        # reshape2::melt(mask) keeps TRUE entries column by column (bin by bin), cells in input order
        rows, cols = [], []
        for b in range(nb):
            cells = np.sort(cell_sample[b])
            rows.append(cells)
            cols.append(np.full(len(cells), b))
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        agg_map = pd.DataFrame({"cell": adata.obs_names.to_numpy()[rows], "agg_cell": agg_names[cols]})
        return cds, agg_map
    return cds


def _fmt(x: float) -> str:
    return format(float(x), ".15g")


def _row_divide(M: sp.csr_matrix, d: np.ndarray) -> sp.csr_matrix:
    """``M[i, :] / d[i]`` elementwise (true division, as R)."""
    M = M.tocsr(copy=True)
    rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    M.data = M.data / d[rows]
    return M


# ============================================================================ windows
def _read_genomic_coords(genomic_coords) -> pd.DataFrame:
    if genomic_coords is None:
        raise ValueError("genomic_coords is required: a DataFrame (chromosome, length), a dict, "
                         "or a path to a two-column chrom.sizes file")
    if isinstance(genomic_coords, (str, bytes)) or hasattr(genomic_coords, "__fspath__"):
        df = pd.read_csv(genomic_coords, sep=r"\s+", header=None, usecols=[0, 1])
    elif isinstance(genomic_coords, dict):
        df = pd.DataFrame(list(genomic_coords.items()))
    else:
        df = pd.DataFrame(genomic_coords).iloc[:, :2].copy()
    df.columns = ["V1", "V2"]
    df["V1"] = df["V1"].astype(str)
    return df


def generate_windows(window: float, genomic_coords) -> pd.DataFrame:
    """Overlapping genomic windows of size ``window`` every ``window / 2`` bp
    (R ``generate_windows``; chromosomes in sorted order as ``plyr::ddply``
    returns them -- byte order, i.e. R's C-locale collation)."""
    chr_maxes = _read_genomic_coords(genomic_coords)
    out = []
    by = window / 2
    for chrom in sorted(chr_maxes["V1"].unique()):
        length = float(chr_maxes.loc[chr_maxes["V1"] == chrom, "V2"].iloc[0])
        nseq = int(math.floor((length - 1) / by + 1e-10))      # R seq(from = 1, to, by)
        starts = 1 + np.arange(nseq + 1) * by
        for s in starts:
            out.append((chrom, int(s), int(s + window - 1)))
    df = pd.DataFrame(out, columns=["chrom", "start", "end"])
    df["name"] = df["chrom"] + "_" + df["start"].astype(str) + "_" + df["end"].astype(str)
    return df


class _WindowIndex:
    """Peak lookup for ``get_genomic_range`` on a cicero AnnData."""

    def __init__(self, cds: AnnData):
        coords = _peak_coords(cds.var, cds.var_names)
        self.chr = coords["chr"].str.replace("chr", "", regex=False).to_numpy()
        self.bp1 = coords["bp1"].to_numpy()
        self.bp2 = coords["bp2"].to_numpy()
        self.by_chr: Dict[str, np.ndarray] = {}
        for c in pd.unique(self.chr):
            self.by_chr[c] = np.flatnonzero(self.chr == c)
        X = cds.X
        self.Xc = X.tocsc() if sp.issparse(X) else sp.csc_matrix(X)   # bins x peaks, column access
        self.names = cds.var_names.to_numpy()

    def peaks(self, chrom: str, start: int, end: int) -> np.ndarray:
        idx = self.by_chr.get(chrom.replace("chr", ""))
        if idx is None:
            return np.empty(0, dtype=np.int64)
        b1 = self.bp1[idx]
        b2 = self.bp2[idx]
        keep = ((b1 < end) & (b1 > start)) | ((b2 < end) & (b2 > start))
        return idx[keep]

    def values(self, peaks: np.ndarray) -> np.ndarray:
        """Dense bins x peaks block (R: t(as.matrix(exprs(win_range))))."""
        return self.Xc[:, peaks].toarray()

    def mean_bp(self, peaks: np.ndarray) -> np.ndarray:
        return (self.bp1[peaks] + self.bp2[peaks]) / 2


def _r_cov(v: np.ndarray, chunk: int = 256) -> np.ndarray:
    """``stats::cov`` of the columns of ``v`` (n x p) exactly as R's C code:
    two-pass long-double means, double products, long-double sequential sums."""
    v = np.asarray(v, dtype=np.float64)
    n, p = v.shape
    vl = v.astype(_LD)
    tmp = np.cumsum(vl, axis=0)[-1] / n
    tmp = tmp + np.cumsum(vl - tmp, axis=0)[-1] / n
    xm = tmp.astype(np.float64)
    d = v - xm
    acc = np.zeros((1, p, p), dtype=_LD)
    for a in range(0, n, chunk):
        blk = d[a:a + chunk]
        prods = (blk[:, :, None] * blk[:, None, :]).astype(_LD)      # double products, exact in ld
        acc = np.add.reduce(np.concatenate([acc, prods], axis=0), axis=0, keepdims=True)  # sequential
    return (acc[0] / (n - 1)).astype(np.float64)


@njit(cache=True, nogil=True)
def _rho_mat(mean_bp, distance_parameter, s):
    """``get_rho_mat(calc_dist_matrix(...))``: (1 - (1000 / d)^s) * dp, floored at 0."""
    n = mean_bp.shape[0]
    out = np.empty((n, n), dtype=np.float64).T
    for j in range(n):
        for i in range(n):
            d = abs(mean_bp[i] - mean_bp[j])
            if d == 0.0:
                out[i, j] = 0.0
                continue
            x = 1000.0 / d
            p = 1.0 if x == 1.0 else math.pow(x, s)
            v = (1.0 - p) * distance_parameter
            if not math.isfinite(v) or v < 0.0:
                v = 0.0
            out[i, j] = v
    return out


def _dist_mat(mean_bp: np.ndarray) -> np.ndarray:
    return np.abs(mean_bp[:, None] - mean_bp[None, :])


def _window_cov(index: _WindowIndex, peaks: np.ndarray) -> np.ndarray:
    S = _r_cov(index.values(peaks))
    S[np.diag_indices_from(S)] += 1e-4
    return np.asfortranarray(S)


def _find_distance_parameter(dist_mat, S, mean_bp, maxit, null_rho, s, distance_constraint,
                             distance_parameter_convergence):
    if (dist_mat > distance_constraint).sum() / 2 < 1:
        return "No long edges"
    found = False
    starting_max = 2.0
    distance_parameter = 2.0
    distance_parameter_max = 2.0
    distance_parameter_min = 0.0
    it = 0
    long_mask = dist_mat > distance_constraint
    big_entries = int(long_mask.sum())
    n = S.shape[0]
    while (not found) and it < maxit:
        rho = _rho_mat(mean_bp, distance_parameter, s)
        _, wi, _, _ = _glasso_core(S, rho, 1, 1.0e-4, 10000)
        if ((np.count_nonzero(wi[long_mask]) / big_entries) > 0.05) or \
                (np.count_nonzero(wi == 0) / (n ** 2) < 0.2):
            longs_zero = False
        else:
            longs_zero = True
        if (not longs_zero) or (distance_parameter == 0):
            distance_parameter_min = distance_parameter
        else:
            distance_parameter_max = distance_parameter
        new_distance_parameter = (distance_parameter_min + distance_parameter_max) / 2
        if new_distance_parameter == starting_max:
            new_distance_parameter = 2 * starting_max
            starting_max = new_distance_parameter
        if distance_parameter_convergence > abs(distance_parameter - new_distance_parameter):
            found = True
        else:
            distance_parameter = new_distance_parameter
        it += 1
    if maxit == it:
        warnings.warn("maximum iterations hit")
    return distance_parameter


# ============================================================================ estimate_distance_parameter
def estimate_distance_parameter(
    cds: AnnData,
    window: float = 500000,
    maxit: int = 100,
    s: float = 0.75,
    sample_num: int = 100,
    distance_constraint: int = 250000,
    distance_parameter_convergence: float = 1e-22,
    max_elements: int = 200,
    genomic_coords=None,
    max_sample_windows: int = 500,
    *,
    random_state: Union[int, RRandom, np.random.Generator, None] = 0,
    rng: str = "R",
    sample_windows: Optional[Sequence[int]] = None,
    return_details: bool = False,
):
    """Distance-penalty scaling parameter on random windows -- R ``estimate_distance_parameter``.

    Random windows (with replacement) are drawn until ``sample_num`` of them
    yield a value or ``max_sample_windows`` were tried. In each window the
    smallest ``distance_parameter`` is found by bisection such that at most
    5 % of site pairs farther apart than ``distance_constraint`` stay non-zero
    in the glasso inverse and at least 20 % of all entries are zero.
    Windows with <= 1 or > ``max_elements`` sites, or without long pairs,
    are skipped.

    Parameters follow R. ``random_state`` / ``rng`` as in
    :func:`make_cicero_cds` (``random_state=s`` == ``set.seed(s)`` right
    before the R call). ``sample_windows`` injects the sequence of sampled
    window numbers (1-based, as R's ``sample()`` returns) instead of drawing.

    Returns
    -------
    ``numpy.ndarray`` of distance parameters (R returns a numeric vector).
    With ``return_details=True`` also a DataFrame with one row per sampled
    window (``win``, ``n_peaks``, ``status``, ``dp``).
    """
    if not (0 < s < 1):
        raise ValueError("s must be in (0, 1)")
    if not distance_constraint < window:
        raise ValueError("distance_constraint must be smaller than window")
    grs = generate_windows(window, genomic_coords)
    index = _WindowIndex(cds)
    kind = gen = None
    if sample_windows is None:
        kind, gen = _make_rng(random_state, rng)
    dps: List[float] = []
    rows = []
    it = 0
    while sample_num > len(dps) and it < max_sample_windows:
        if sample_windows is not None and it >= len(sample_windows):
            break
        it += 1
        win = int(sample_windows[it - 1]) if sample_windows is not None else _draw_index(kind, gen, len(grs)) + 1
        w = grs.iloc[win - 1]
        peaks = index.peaks(w.chrom, w.start, w.end)
        npk = len(peaks)
        status, dp = "ok", np.nan
        if npk <= 1:
            status = "le1"
        elif npk > max_elements:
            status = "gt_max"
        else:
            mbp = index.mean_bp(peaks)
            res = _find_distance_parameter(_dist_mat(mbp), _window_cov(index, peaks), mbp, maxit, 0, s,
                                           distance_constraint, distance_parameter_convergence)
            if isinstance(res, str):
                status = res
            else:
                dp = res
                dps.append(res)
        rows.append((it, win, npk, status, dp))
    if len(dps) < sample_num:
        warnings.warn(f"Could not calculate sample_num distance_parameters ({len(dps)} were calculated) "
                      "- see documentation details")
    if len(dps) == 0:
        raise RuntimeError("No distance_parameters calculated")
    out = np.asarray(dps, dtype=np.float64)
    if return_details:
        return out, pd.DataFrame(rows, columns=["it", "win", "n_peaks", "status", "dp"])
    return out


# ============================================================================ generate_cicero_models
@dataclass
class CiceroModel:
    """Glasso fit of one window (R: the ``glasso`` list with named ``w`` / ``wi``)."""
    peaks: np.ndarray               #: peak names (row/column names of w, wi)
    w: np.ndarray                   #: regularised covariance
    wi: Optional[np.ndarray]        #: regularised inverse covariance (None if not kept)
    niter: int
    peak_index: np.ndarray          #: positions of the peaks in the cicero AnnData's var


def _fit_window(index: _WindowIndex, w, distance_parameter, s, max_elements, keep_wi):
    peaks = index.peaks(w.chrom, w.start, w.end)
    if len(peaks) <= 1:
        return "Zero or one element in range"
    if len(peaks) > max_elements:
        return "Too many elements in range"
    mbp = index.mean_bp(peaks)
    rho = _rho_mat(mbp, distance_parameter, s)
    S = _window_cov(index, peaks)
    W, WI, niter, _ = _glasso_core(S, rho, 1, 1.0e-4, 10000)
    return CiceroModel(index.names[peaks], np.ascontiguousarray(W),
                       np.ascontiguousarray(WI) if keep_wi else None, int(niter), peaks)


def generate_cicero_models(
    cds: AnnData,
    distance_parameter: float,
    s: float = 0.75,
    window: float = 500000,
    max_elements: int = 200,
    genomic_coords=None,
    *,
    n_jobs: int = 1,
    keep_wi: bool = True,
) -> Dict[str, Union[CiceroModel, str]]:
    """Graphical lasso in every overlapping window -- R ``generate_cicero_models``.

    For each window the sample covariance of its sites across the cicero
    bins (+1e-4 on the diagonal) is regularised by glasso with the
    distance-scaled penalty ``rho_ij = distance_parameter * (1 - (1000 /
    d_ij)^s)`` (0 for ``d_ij <= 1000``).

    Returns
    -------
    dict ``window name -> CiceroModel`` or the reason the window was skipped
    (``"Zero or one element in range"`` / ``"Too many elements in range"``),
    in R's window order. ``n_jobs > 1`` fits windows in threads (the numba
    solver releases the GIL); results are identical.
    """
    if not (0 < s < 1):
        raise ValueError("s must be in (0, 1)")
    grs = generate_windows(window, genomic_coords)
    index = _WindowIndex(cds)
    items = list(grs.itertuples(index=False))

    def fit(w):
        return _fit_window(index, w, float(distance_parameter), s, max_elements, keep_wi)

    if n_jobs is not None and n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            res = list(ex.map(fit, items))
    else:
        res = [fit(w) for w in items]
    return dict(zip(grs["name"], res))


# ============================================================================ assemble_connections
def _cov2cor(V: np.ndarray) -> np.ndarray:
    Is = np.sqrt(1.0 / np.diag(V))
    r = (Is[:, None] * V) * Is[None, :]
    np.fill_diagonal(r, 1.0)
    return r


def _r_mean_groups(vals_ld_sorted: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """R ``mean()`` per group of a value vector sorted by group (sequential long double)."""
    out = np.empty(len(starts), dtype=np.float64)
    one = counts == 1
    out[one] = vals_ld_sorted[starts[one]].astype(np.float64)
    for c in np.unique(counts[~one]):
        sel = np.flatnonzero(counts == c)
        blk = vals_ld_sorted[starts[sel][:, None] + np.arange(c)[None, :]]     # g x c, long double
        s = blk[:, 0].copy()
        for j in range(1, c):
            s = s + blk[:, j]
        s = s / c
        t = (blk[:, 0] - s)
        for j in range(1, c):
            t = t + (blk[:, j] - s)
        fin = np.isfinite(s)
        s = np.where(fin, s + t / c, s)
        out[sel] = s.astype(np.float64)
    return out


def assemble_connections(cicero_model_list: Dict[str, Union[CiceroModel, str]],
                         silent: bool = False) -> pd.DataFrame:
    """Combine window models into one co-accessibility table -- R ``assemble_connections``.

    Each model's covariance is converted to correlations (``cov2cor``). Pairs
    seen in several overlapping windows are reconciled: the mean if all
    values have the same sign (zeros count as either), ``NaN`` otherwise.

    Returns
    -------
    DataFrame with columns ``Peak1``, ``Peak2``, ``coaccess`` (both
    orientations of every pair, no self pairs), ordered like R's output
    (``Peak1`` in byte order, then ``Peak2`` by first appearance).
    """
    models = [m for m in cicero_model_list.values() if isinstance(m, CiceroModel)]
    other = [m for m in cicero_model_list.values() if isinstance(m, str)]
    if not silent:
        print(f"Successful cicero models:  {len(models)}")
        print("Other models: ")
        if other:
            print(pd.Series(other).value_counts().sort_index().to_string())
        print(f"Models with errors:  0", flush=True)
    if not models:
        return pd.DataFrame({"Peak1": pd.Series(dtype=str), "Peak2": pd.Series(dtype=str),
                             "coaccess": pd.Series(dtype=float)})
    # global peak ids in order of first appearance (= R's factor levels of Var2)
    name_to_id: Dict[str, int] = {}
    names: List[str] = []
    keys_i, keys_j, vals = [], [], []
    for m in models:
        ids = np.empty(len(m.peaks), dtype=np.int64)
        for t, nm in enumerate(m.peaks):
            g = name_to_id.get(nm)
            if g is None:
                g = len(names)
                name_to_id[nm] = g
                names.append(nm)
            ids[t] = g
        cors = _cov2cor(m.w)
        p = len(ids)
        keys_i.append(np.repeat(ids, p))      # Var1 = row
        keys_j.append(np.tile(ids, p))        # Var2 = column
        vals.append(cors.ravel())             # row-major: (row, col) pairs
    I = np.concatenate(keys_i)
    J = np.concatenate(keys_j)
    V = np.concatenate(vals)
    npk = len(names)
    key = I * npk + J
    order = np.argsort(key, kind="stable")     # keeps window order inside a pair
    key_s = key[order]
    V_s = V[order]
    starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
    counts = np.diff(np.r_[starts, len(key_s)])
    # reconcile(): same sign (or zero) -> mean, else NA
    pos = np.add.reduceat((V_s >= 0).astype(np.int64), starts)
    neg = np.add.reduceat((V_s <= 0).astype(np.int64), starts)
    mean = _r_mean_groups(V_s.astype(_LD), starts, counts)
    co = np.where((pos == counts) | (neg == counts), mean, np.nan)
    gi = key_s[starts] // npk
    gj = key_s[starts] % npk
    keep = gi != gj
    gi, gj, co = gi[keep], gj[keep], co[keep]
    # R order: setkey(Var1 = character, Var2 = factor in appearance order)
    names_arr = np.array(names, dtype=object)
    rank1 = np.empty(npk, dtype=np.int64)
    rank1[np.argsort(np.array([n.encode() for n in names], dtype=object), kind="stable")] = np.arange(npk)
    o = np.lexsort((gj, rank1[gi]))
    cat = pd.CategoricalDtype(categories=names_arr)
    return pd.DataFrame({
        "Peak1": pd.Categorical.from_codes(gi[o], dtype=cat),
        "Peak2": pd.Categorical.from_codes(gj[o], dtype=cat),
        "coaccess": co[o],
    })


# ============================================================================ run_cicero
def run_cicero(
    cds: AnnData,
    genomic_coords,
    window: float = 500000,
    silent: bool = False,
    sample_num: int = 100,
    *,
    random_state: Union[int, RRandom, np.random.Generator, None] = 0,
    rng: str = "R",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run Cicero -- R ``run_cicero``.

    ``estimate_distance_parameter`` (``maxit=100``, ``distance_constraint =
    250000``, ``distance_parameter_convergence = 1e-22``) -> mean ->
    ``generate_cicero_models`` -> ``assemble_connections``.

    Parameters
    ----------
    cds
        Output of :func:`make_cicero_cds`.
    genomic_coords
        Chromosome lengths: DataFrame (name, length), dict or chrom.sizes path.
        Only chromosomes listed here are scanned.
    window, silent, sample_num
        As in R (500 kb windows overlapping by 250 kb, 100 sampled windows).
    random_state, rng
        Seed of the window sampling (``random_state=s`` == ``set.seed(s)``
        before ``run_cicero`` in R).
    n_jobs
        Threads for the window models (identical results).

    Returns
    -------
    DataFrame ``Peak1``, ``Peak2``, ``coaccess`` -- the format CellOracle's
    base-GRN step reads (see :func:`cicero_connections_to_celloracle`).
    """
    _log(silent, "Starting Cicero")
    _log(silent, "Calculating distance_parameter value")
    dps = estimate_distance_parameter(cds, window=window, maxit=100, sample_num=sample_num,
                                      distance_constraint=250000, distance_parameter_convergence=1e-22,
                                      genomic_coords=genomic_coords, random_state=random_state, rng=rng)
    mean_dp = _r_mean(dps)
    _log(silent, "Running models")
    models = generate_cicero_models(cds, distance_parameter=mean_dp, window=window,
                                    genomic_coords=genomic_coords, n_jobs=n_jobs, keep_wi=False)
    _log(silent, "Assembling connections")
    cons = assemble_connections(models, silent=silent)
    _log(silent, "Done")
    cons.attrs["distance_parameters"] = dps
    cons.attrs["mean_distance_parameter"] = mean_dp
    return cons


def cicero_connections_to_celloracle(conns: pd.DataFrame, path: Optional[str] = None,
                                     drop_na: bool = False) -> pd.DataFrame:
    """Cicero table in the layout CellOracle's base-GRN step reads.

    Peak ids are rewritten to ``chr_start_end`` (CellOracle splits on ``_``);
    columns are ``Peak1``, ``Peak2``, ``coaccess``. With ``path`` the table is
    written as CSV with an index column (``pd.read_csv(path, index_col=0)``
    in the CellOracle tutorial).
    """
    def norm(x):
        c = _split_peak_names(pd.Series(x).astype(str))
        return (c["chr"] + "_" + c["bp1"] + "_" + c["bp2"]).to_numpy()

    p1 = conns["Peak1"]
    p2 = conns["Peak2"]
    if isinstance(p1.dtype, pd.CategoricalDtype) and p1.cat.categories.equals(p2.cat.categories):
        cats = pd.Index(norm(p1.cat.categories))
        out = pd.DataFrame({"Peak1": cats[p1.cat.codes], "Peak2": cats[p2.cat.codes],
                            "coaccess": conns["coaccess"].to_numpy()})
    else:
        out = pd.DataFrame({"Peak1": norm(p1), "Peak2": norm(p2), "coaccess": conns["coaccess"].to_numpy()})
    if drop_na:
        out = out[out["coaccess"].notna()].reset_index(drop=True)
    if path is not None:
        out.to_csv(path)
    return out


# ============================================================================ CCANs
def _ccan_graph(conns: pd.DataFrame, coaccess_cutoff: float):
    import igraph as ig
    co = conns["coaccess"].to_numpy(dtype=np.float64)
    keep = ~np.isnan(co) & (co > coaccess_cutoff)
    if keep.sum() == 0:
        raise ValueError("No connections for graph")
    p1 = np.asarray(conns["Peak1"].astype(str))[keep]
    p2 = np.asarray(conns["Peak2"].astype(str))[keep]
    w = co[keep]
    peaks = pd.unique(np.concatenate([p1, p2]))      # make_sparse_matrix vertex order
    pid = pd.Index(peaks)
    i = pid.get_indexer(p1)
    j = pid.get_indexer(p2)
    a, b = np.minimum(i, j), np.maximum(i, j)
    e = pd.DataFrame({"a": a, "b": b, "w": w}).drop_duplicates(["a", "b"], keep="first")
    e = e[e.a != e.b]
    g = ig.Graph(n=len(peaks), edges=list(zip(e.a.to_numpy(), e.b.to_numpy())), directed=False)
    g.es["weight"] = e.w.to_numpy()
    g.vs["name"] = list(peaks)
    return g


def _louvain(g):
    return g.community_multilevel(weights="weight")


def _number_of_ccans(conns, cutoff):
    cl = _louvain(_ccan_graph(conns, cutoff))
    return int((np.asarray(cl.sizes()) > 2).sum())


def _r_round(x: float, digits: int) -> float:
    return round(float(x), digits)


def _find_ccan_cutoff(conns: pd.DataFrame, tolerance_digits: int) -> float:
    co = conns["coaccess"].to_numpy(dtype=np.float64)
    conns = conns[~(co <= 0)]                         # R keeps NA rows (dropped later in the graph)
    tolerance = 10 ** -tolerance_digits
    bottom = 0.0
    top = float(np.nanmax(conns["coaccess"].to_numpy(dtype=np.float64)))
    while (top - bottom) > tolerance:
        test_val = bottom + _r_round((top - bottom) / 2, tolerance_digits + 1)
        n1 = _number_of_ccans(conns, test_val)
        next_step = test_val
        while True:
            next_step = next_step + (top - bottom) / 10
            n2 = _number_of_ccans(conns, next_step)
            if n2 != n1:
                break
        if n1 > n2:
            top = test_val
        else:
            bottom = test_val
    return _r_round((top + bottom) / 2, tolerance_digits)


def generate_ccans(connections_df: pd.DataFrame, coaccess_cutoff_override: Optional[float] = None,
                   tolerance_digits: int = 2, *, random_state: Optional[int] = 0,
                   silent: bool = False) -> pd.DataFrame:
    """Cis-co-accessibility networks -- R ``generate_ccans``.

    Edges with ``coaccess > cutoff`` form a graph; Louvain communities with
    more than two peaks are CCANs. Without ``coaccess_cutoff_override`` the
    cutoff is found by the same bisection on the number of CCANs as R
    (``find_ccan_cutoff``).

    ``random_state`` seeds python-igraph's Louvain (R draws from its own RNG,
    so memberships agree with R in distribution, see the parity report).

    Returns
    -------
    DataFrame ``Peak``, ``CCAN`` (peaks outside CCANs are not returned);
    ``.attrs['coaccess_cutoff']`` holds the cutoff used.
    """
    for c in ("Peak1", "Peak2", "coaccess"):
        if c not in connections_df.columns:
            raise ValueError(f"connections_df must have a column {c!r}")
    if coaccess_cutoff_override is not None and not (0 <= coaccess_cutoff_override <= 1):
        raise ValueError("coaccess_cutoff_override must be between 0 and 1 (or NULL)")
    import random

    import igraph as ig
    state = random.Random(random_state) if random_state is not None else random.Random()
    ig.set_random_number_generator(state)
    try:
        if coaccess_cutoff_override is not None:
            cutoff = float(coaccess_cutoff_override)
        else:
            cutoff = _find_ccan_cutoff(connections_df, tolerance_digits)
        _log(silent, f"Coaccessibility cutoff used: {cutoff}")
        g = _ccan_graph(connections_df, cutoff)
        cl = _louvain(g)
    finally:
        ig.set_random_number_generator(random)
    memb = np.asarray(cl.membership) + 1
    sizes = np.bincount(memb)
    keep = sizes[memb] > 2
    df = pd.DataFrame({"Peak": np.asarray(g.vs["name"], dtype=object)[keep], "CCAN": memb[keep]})
    df.index = df["Peak"].to_numpy()
    df.attrs["coaccess_cutoff"] = cutoff
    return df


def find_overlapping_ccans(ccan_assignments: pd.DataFrame, min_overlap: int = 1) -> pd.DataFrame:
    """CCANs whose genomic spans overlap -- R ``find_overlapping_ccans``.

    Returns a reciprocal DataFrame ``CCAN1``, ``CCAN2``.
    """
    df = ccan_assignments.iloc[:, :2].copy()
    df.columns = ["Peak", "CCAN"]
    c = _split_peak_names(df["Peak"])
    df["chr"] = c["chr"].to_numpy()
    df["bp1"] = c["bp1"].astype(float).to_numpy()
    df["bp2"] = c["bp2"].astype(float).to_numpy()
    info = df.groupby("CCAN", sort=True).agg(chr=("chr", "first"), bp1=("bp1", "min"), bp2=("bp2", "max"))
    rows = []
    for a, ra in info.iterrows():
        for b, rb in info.iterrows():
            if a == b or ra.chr != rb.chr:
                continue
            ov = min(ra.bp2, rb.bp2) - max(ra.bp1, rb.bp1) + 1
            if ov >= min_overlap:
                rows.append((a, b))
    return pd.DataFrame(rows, columns=["CCAN1", "CCAN2"]).drop_duplicates().reset_index(drop=True)


# ============================================================================ gene activity
def annotate_cds_by_site(adata: AnnData, feature_data, verbose: bool = False, maxgap: int = 0,
                         all: bool = False, header: bool = False, inplace: bool = True):
    """Annotate peaks with overlapping features -- R ``annotate_cds_by_site``.

    ``feature_data`` is a bed-like DataFrame (or path): chromosome, start,
    end (1-based closed, as ``GenomicRanges``) and extra columns, which are
    copied to ``adata.var`` together with ``overlap`` (bp overlap; 0 when
    the hit comes only from ``maxgap``; NaN when no feature hits). With
    ``all=False`` the largest overlap wins (first on ties), with
    ``all=True`` all hits are collapsed with ``,``.
    """
    if isinstance(feature_data, (str, bytes)) or hasattr(feature_data, "__fspath__"):
        fd = pd.read_csv(feature_data, sep="\t", header=0 if header else None)
    else:
        fd = pd.DataFrame(feature_data).copy()
    cols = list(fd.columns)
    extra = cols[3:]
    f_chr = fd.iloc[:, 0].astype(str).to_numpy()
    f_s = fd.iloc[:, 1].to_numpy(dtype=np.float64)
    f_e = fd.iloc[:, 2].to_numpy(dtype=np.float64)
    pc = _split_peak_names(adata.var_names)
    p_chr = pc["chr"].to_numpy()
    p_s = pc["bp1"].astype(float).to_numpy()
    p_e = pc["bp2"].astype(float).to_numpy()
    if maxgap == "nearest":
        raise NotImplementedError("maxgap='nearest' is not ported")
    hits_q, hits_s = [], []
    for c in pd.unique(p_chr):
        qi = np.flatnonzero(p_chr == c)
        si = np.flatnonzero(f_chr == c)
        if len(si) == 0:
            continue
        o = np.argsort(f_s[si], kind="stable")
        si_sorted = si[o]
        fs = f_s[si_sorted]
        for q in qi:
            # candidate features: start <= q.end + maxgap + 1 ; then filter end >= q.start - maxgap - 1
            hi = np.searchsorted(fs, p_e[q] + maxgap + 1, side="right")
            cand = si_sorted[:hi]
            cand = cand[f_e[cand] >= p_s[q] - maxgap - 1]
            if len(cand):
                # findOverlaps (GNCList on the subject) reports a query's hits in NCList
                # traversal order: by feature start, then end (decreasing), then index
                cand = cand[np.lexsort((cand, -f_e[cand], f_s[cand]))]
                hits_q.append(np.full(len(cand), q))
                hits_s.append(cand)
    if hits_q:
        hq = np.concatenate(hits_q)
        hs = np.concatenate(hits_s)
    else:
        hq = hs = np.empty(0, dtype=np.int64)
    width = np.maximum(0, np.minimum(p_e[hq], f_e[hs]) - np.maximum(p_s[hq], f_s[hs]) + 1)
    ol = pd.DataFrame({"row": hq, "overlap": width})
    for c in extra:
        ol[c] = fd[c].to_numpy()[hs]
    if all:
        agg = {"overlap": lambda x: ",".join(_r_str(v) for v in x)}
        for c in extra:
            agg[c] = lambda x: ",".join(_r_str(v) for v in x)
        ol = ol.groupby("row", sort=True).agg(agg)
    else:
        o = np.argsort(-ol["overlap"].to_numpy(), kind="stable")
        ol = ol.iloc[o]
        ol = ol[~ol["row"].duplicated()].set_index("row")
    var = adata.var.copy()
    newcols = ["overlap"] + extra
    for c in newcols:
        col = pd.Series(np.nan, index=range(adata.n_vars), dtype=object if (all or c not in ("overlap",)) else float)
        col.loc[ol.index] = ol[c].to_numpy()
        var[c] = col.to_numpy()
    if inplace:
        adata.var = var
        return None
    out = adata.copy()
    out.var = var
    return out


def _r_str(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def build_gene_activity_matrix(adata: AnnData, cicero_cons_info: pd.DataFrame, site_weights=None,
                               dist_thresh: float = 250000, coaccess_cutoff: float = 0.25,
                               gene_col: str = "gene", layer: Optional[str] = None) -> AnnData:
    """Unnormalised Cicero gene activity -- R ``build_gene_activity_matrix``.

    A gene's activity is the accessibility of its promoter peaks plus the
    accessibility of distal peaks linked to them (``coaccess >=
    coaccess_cutoff``, distance < ``dist_thresh``), weighted by the
    co-accessibility (row-normalised, capped at 1).

    Parameters
    ----------
    adata
        Cells x peaks AnnData (binary accessibility in the vignette) whose
        ``var[gene_col]`` names the gene of promoter peaks and is NaN for
        distal peaks (see :func:`annotate_cds_by_site`).
    cicero_cons_info
        Connections (``Peak1``, ``Peak2``, ``coaccess``; optional ``dist``).

    Returns
    -------
    AnnData cells x genes (genes in byte-sorted order, as R's factor levels
    in the C locale). R returns the transpose (genes x cells).
    """
    if gene_col not in adata.var.columns:
        raise ValueError(f"adata.var must have a column {gene_col!r} (gene name of promoter peaks, NaN elsewhere)")
    for c in ("Peak1", "Peak2", "coaccess"):
        if c not in cicero_cons_info.columns:
            raise ValueError(f"cicero_cons_info must have a column {c!r}")
    acc = _sparse_x(adata, layer).T.tocsr()                       # peaks x cells
    peak_names = adata.var_names.to_numpy().astype(str)
    pidx = pd.Index(peak_names)
    if site_weights is None:
        sw = pd.Series(1.0, index=peak_names)
    else:
        sw = pd.Series(site_weights, index=peak_names) if not isinstance(site_weights, pd.Series) else site_weights
    gene = adata.var[gene_col]
    ppt = pd.DataFrame({"peak": peak_names, "gene": gene.to_numpy()})
    ppt = ppt[gene.notna().to_numpy()]
    ppt["gene"] = ppt["gene"].astype(str)
    prom_unique = pd.unique(ppt["peak"].to_numpy())
    prom_set = set(prom_unique)

    con = cicero_cons_info
    P1 = con["Peak1"].astype(str).to_numpy()
    P2 = con["Peak2"].astype(str).to_numpy()
    co = con["coaccess"].to_numpy(dtype=np.float64)
    if "dist" in con.columns:
        dist = con["dist"].to_numpy(dtype=np.float64)
    else:
        c1 = _split_peak_names(P1)
        c2 = _split_peak_names(P2)
        b1 = np.round((c1["bp2"].astype(np.int64).to_numpy() + c1["bp1"].astype(np.int64).to_numpy()) / 2)
        b2 = np.round((c2["bp2"].astype(np.int64).to_numpy() + c2["bp1"].astype(np.int64).to_numpy()) / 2)
        dist = np.abs(b2 - b1)
    in1 = pd.Series(P1).isin(prom_set).to_numpy()
    in2 = pd.Series(P2).isin(prom_set).to_numpy()
    with np.errstate(invalid="ignore"):
        sel = (in1 | in2) & (co >= coaccess_cutoff) & (dist < dist_thresh)
    nn = pd.DataFrame({"Peak1": P1[sel], "Peak2": P2[sel], "coaccess": co[sel]}).drop_duplicates()
    nn = pd.concat([nn, pd.DataFrame({"Peak1": prom_unique, "Peak2": prom_unique, "coaccess": 0.0})],
                   ignore_index=True)
    # make_sparse_matrix: symmetric, first occurrence of an (unordered) pair wins
    mpeaks = pd.unique(np.concatenate([nn["Peak1"].to_numpy(), nn["Peak2"].to_numpy()]))
    mid = pd.Index(mpeaks)
    i = mid.get_indexer(nn["Peak1"].to_numpy())
    j = mid.get_indexer(nn["Peak2"].to_numpy())
    cx = nn["coaccess"].to_numpy(dtype=np.float64)
    e = pd.DataFrame({"a": np.r_[i, j], "b": np.r_[j, i], "x": np.r_[cx, cx]})
    e = e.drop_duplicates(["a", "b"], keep="first")
    Smat = sp.csr_matrix((e["x"].to_numpy(dtype=np.float64), (e["a"].to_numpy(), e["b"].to_numpy())),
                         shape=(len(mpeaks), len(mpeaks)))
    prom_rows = mid.get_indexer(prom_unique)
    promoter_conn = Smat[prom_rows]                                # promoters x all
    promoter_safe = [p for p in prom_unique if p in pidx]
    in_acc = pd.Index(mpeaks).isin(pidx)
    distal_safe = [p for p, ok in zip(mpeaks, in_acc) if ok and p not in set(promoter_safe)]
    prom_acc = acc[pidx.get_indexer(promoter_safe)]
    distal_acc = acc[pidx.get_indexer(distal_safe)]
    dcols = mid.get_indexer(distal_safe)
    wd = sw.reindex(distal_safe).to_numpy(dtype=np.float64)
    pc_d = promoter_conn[:, dcols].tocsr()
    tl = np.asarray((pc_d @ sp.diags(wd)).sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        tl = 1.0 / tl
    tl[~np.isfinite(tl)] = 0.0
    scaled = (sp.diags(tl) @ pc_d @ sp.diags(wd)).tocsr()
    scaled.data[scaled.data > 1] = 1.0
    distal_scores = (scaled @ distal_acc).tocsr()                  # promoters (all unique) x cells
    order_rows = pd.Index(prom_unique).get_indexer(promoter_safe)
    distal_scores = distal_scores[order_rows]
    prom_scores = (distal_scores + prom_acc).tocsr()               # promoter_safe x cells
    genes = np.array(sorted(pd.unique(ppt["gene"]), key=lambda s: s.encode()), dtype=object)
    peaks_lv = np.array(sorted(pd.unique(ppt["peak"]), key=lambda s: s.encode()), dtype=object)
    gi = pd.Index(genes).get_indexer(ppt["gene"].to_numpy())
    pj = pd.Index(peaks_lv).get_indexer(ppt["peak"].to_numpy())
    pg = sp.csr_matrix((np.ones(len(gi)), (gi, pj)), shape=(len(genes), len(peaks_lv)))
    pg = pg[:, pd.Index(peaks_lv).get_indexer(promoter_safe)]
    ga = (pg @ prom_scores).T.tocsr()                              # cells x genes
    return AnnData(X=ga, obs=pd.DataFrame(index=adata.obs_names.copy()), var=pd.DataFrame(index=genes))


def normalize_gene_activities(activity_matrices, cell_num_genes):
    """Normalise Cicero gene activities -- R ``normalize_gene_activities``.

    ``activity_matrices`` is one AnnData (cells x genes, output of
    :func:`build_gene_activity_matrix`) or a list of them (normalised
    together, with a per-matrix group term in the size model).
    ``cell_num_genes`` maps cell names to the number of accessible sites
    (``adata.obs['num_genes_expressed']`` of the peak AnnData).

    As in R, the inputs must not contain all-zero genes or cells (the
    vignette removes them first); R errors on them, so does this port.
    """
    single = isinstance(activity_matrices, AnnData)
    mats = [activity_matrices] if single else list(activity_matrices)
    blocks, cells, groups = [], [], []
    for g, a in enumerate(mats):
        blocks.append(sp.csr_matrix(a.X, dtype=np.float64).T.tocsc())   # genes x cells
        cells.extend(a.obs_names)
        groups.extend([g + 1] * a.n_obs)
    genes = mats[0].var_names
    for a in mats[1:]:
        if not a.var_names.equals(genes):
            raise ValueError("all activity matrices must have the same genes in the same order")
    scores = sp.hstack(blocks).tocsc()
    rs = np.asarray(scores.sum(axis=1)).ravel()
    cs = np.asarray(scores.sum(axis=0)).ravel()
    if np.any(rs == 0) or np.any(cs == 0):
        raise ValueError("activity matrices contain all-zero genes or cells; remove them first "
                         "(R's normalize_gene_activities errors on them as well)")
    cng = pd.Series(cell_num_genes)
    total_sites = cng.reindex(pd.Index(cells).astype(str)).to_numpy(dtype=np.float64)
    if np.isnan(total_sites).any():
        raise ValueError("cell_num_genes is missing cells of the activity matrices")
    y = np.log(cs)
    xs = np.log(total_sites)
    if single:
        A = np.column_stack([np.ones_like(xs), xs])
    else:
        grp = np.asarray(groups)
        lv = np.unique(grp)
        cols = [np.ones_like(xs), xs]
        dums = [(grp == g).astype(float) for g in lv[1:]]
        cols += dums
        cols += [xs * d for d in dums]
        A = np.column_stack(cols)
    Q, R = np.linalg.qr(A)
    beta = np.linalg.solve(R, Q.T @ y)
    fitted = np.exp(A @ beta)
    lf = np.log(fitted)
    size_factors = lf / _r_mean(lf)
    inv = 1.0 / size_factors
    S = scores.tocsc(copy=True)
    S.data = S.data * np.repeat(inv, np.diff(S.indptr))
    S.data = np.minimum(1e9, np.exp(S.data) - 1)
    tot = np.asarray(S.sum(axis=0)).ravel()
    sc = 1.0 / tot
    S.data = S.data * np.repeat(sc, np.diff(S.indptr))
    S = S.T.tocsr()                                                  # cells x genes
    outs = []
    off = 0
    for a in mats:
        blk = S[off:off + a.n_obs]
        off += a.n_obs
        outs.append(AnnData(X=blk, obs=pd.DataFrame(index=a.obs_names.copy()), var=pd.DataFrame(index=genes.copy())))
    return outs[0] if single else outs
