"""Project bulk ATAC-seq samples into a single-cell embedding.

``project_bulk_atac`` ports ArchR's ``projectBulkATAC`` (Granja et al.,
*Nat Genet* 2021). Each bulk sample is down-sampled into pseudo-cells at
single-cell depth, the pseudo-cells are projected into the LSI space of the
single-cell data with the stored LSI model, and from there into the UMAP.

``deconvolve_bulk`` is the complementary reference-mapping view: it explains
each bulk profile as a non-negative mixture of cell-group pseudobulk profiles
(non-negative least squares).

Portions of this module are derived from ArchR
(https://github.com/GreenleafLab/ArchR), which is distributed under the MIT
License:

    Copyright (c) 2020 Jeffrey Granja, Ryan Corces

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ..utils import console


# ---------------------------------------------------------------------------
# LSI model
# ---------------------------------------------------------------------------

@dataclass
class LSIModel:
    """Everything needed to project new profiles into a fitted LSI space.

    Two flavours are supported:

    ``flavor="archr"``
        ArchR's ``.computeLSI`` / ``.projectLSI``. ``loadings`` is the left
        singular matrix ``svd$u`` (features x dims), ``feature_sums`` is
        ``rowSm`` and ``n_cells`` is ``nCol``. Coordinates are
        ``X_norm @ U`` and ``scale_dims`` z-scores each *cell* across dims
        (ArchR ``.scaleDims``).

    ``flavor="epione"``
        :func:`epione.tl.iterative_lsi`. ``loadings`` are the right singular
        vectors of the kept dims, coordinates are ``X_norm @ V / s`` and
        ``scale_dims`` z-scores each *dimension* with the mean and SD of the
        fitted cells. Build it with :meth:`LSIModel.from_adata`.

    Attributes
    ----------
    features
        DataFrame with ``chrom``, ``start``, ``end`` for the model features,
        row-aligned with ``loadings``.
    loadings
        ``(n_features, n_dims)``.
    singular_values
        ``(n_dims,)``.
    feature_sums
        Per-feature accessibility totals over the fitted cells (binarised if
        ``binarize``), the IDF denominator.
    n_cells
        Number of fitted cells, the IDF numerator.
    flavor
        ``"archr"`` or ``"epione"``.
    lsi_method
        ArchR ``LSIMethod``: 1 = tf-logidf, 2 = log(tf-idf), 3 = logtf-logidf.
        Ignored for ``flavor="epione"``.
    scale_to
        Multiplier inside the log.
    binarize
        Binarise counts before projecting.
    scale_dims
        Apply the flavour's dimension scaling.
    dims
        Indices (0-based) of the dims the embedding uses, after the depth
        correlation filter. ``None`` means all.
    scale_mean, scale_sd
        Epione flavour only: per-dim mean and SD used for scaling.
    seed
        ArchR flavour only: ``LSI$seed``. ArchR's ``.projectLSI`` calls
        ``set.seed(seed)``, which affects the RNG state of later samples in
        :func:`project_bulk_atac` with ``rng="archr"``.
    closed
        Whether ``features`` coordinates are closed intervals (ArchR's
        GRanges) or half-open (BED / Epione).
    """

    features: pd.DataFrame
    loadings: np.ndarray
    singular_values: np.ndarray
    feature_sums: np.ndarray
    n_cells: int
    flavor: Literal["archr", "epione"] = "epione"
    lsi_method: int = 2
    scale_to: float = 10000.0
    binarize: bool = True
    scale_dims: bool = True
    dims: Optional[np.ndarray] = None
    scale_mean: Optional[np.ndarray] = None
    scale_sd: Optional[np.ndarray] = None
    seed: int = 1
    closed: bool = False

    def __post_init__(self):
        self.loadings = np.asarray(self.loadings, dtype=np.float64)
        self.singular_values = np.asarray(self.singular_values, dtype=np.float64)
        self.feature_sums = np.asarray(self.feature_sums, dtype=np.float64)
        if self.loadings.shape[0] != len(self.features):
            raise ValueError("loadings and features must have the same number of rows")
        if self.feature_sums.shape[0] != len(self.features):
            raise ValueError("feature_sums and features must have the same length")
        if self.flavor not in ("archr", "epione"):
            raise ValueError(f"unknown flavor {self.flavor!r}")
        if self.dims is not None:
            self.dims = np.asarray(self.dims, dtype=np.int64)

    @property
    def mean_depth(self) -> int:
        """Accessible model features per fitted cell, rounded.

        ArchR's ``depthN = round(sum(rowSm / nCol))``, the depth at which
        bulk samples are down-sampled.
        """
        return int(_r_round(float(np.sum(self.feature_sums / self.n_cells))))

    # -- construction ------------------------------------------------------

    @classmethod
    def from_adata(cls, adata: AnnData, key: str = "X_iterative_lsi",
                   closed: Optional[bool] = None) -> "LSIModel":
        """Model of a fit made by :func:`epione.tl.iterative_lsi`.

        Needs the model fields that ``iterative_lsi`` stores since this
        release (``feature_sums``, ``n_cells_fit``, ``scale_mean``,
        ``scale_sd``). For older objects they are recomputed from
        ``adata.X``. That is exact when the final round was fitted on all
        cells (the default) and the matrix is unchanged.

        ``closed`` gives the coordinate convention of ``adata.var``. By
        default it is detected: Epione tile names such as ``chr1:0-499`` are
        closed intervals, peak names such as ``chr1:100-600`` half-open.
        """
        if key not in adata.uns or key + "_loadings" not in adata.varm:
            raise KeyError(
                f"adata.uns[{key!r}] / adata.varm[{key + '_loadings'!r}] not found; "
                "run epione.tl.iterative_lsi first"
            )
        info = adata.uns[key]
        params = dict(info.get("params", {}))
        feat_mask = np.asarray(info["features_final"]).astype(bool)
        kept = np.asarray(info["kept_dims"], dtype=np.int64)
        s = np.asarray(info["stdev"], dtype=np.float64)[kept]
        loadings = np.asarray(adata.varm[key + "_loadings"], dtype=np.float64)[feat_mask]
        features = _var_regions(adata, mask=feat_mask)
        if closed is None:
            closed = _looks_closed(features)
        binarize = bool(params.get("binarize", True))
        scale_to = float(params.get("scale_to", 10000.0))
        scale_dims = bool(params.get("scale_dims", True))

        if "feature_sums" in info and "n_cells_fit" in info:
            feature_sums = np.asarray(info["feature_sums"], dtype=np.float64)
            n_cells = int(info["n_cells_fit"])
            scale_mean = np.asarray(info["scale_mean"], dtype=np.float64)
            scale_sd = np.asarray(info["scale_sd"], dtype=np.float64)
        else:
            warnings.warn(
                f"adata.uns[{key!r}] predates stored LSI models; recomputing the "
                "IDF and dim scaling from adata.X. Re-run iterative_lsi to store them.",
                stacklevel=2,
            )
            X = _to_csr(adata.X)[:, np.where(feat_mask)[0]]
            if binarize:
                X = _binarize_csr(X)
            feature_sums = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
            n_cells = X.shape[0]
            tmp = cls(features=features, loadings=loadings, singular_values=s,
                      feature_sums=feature_sums, n_cells=n_cells, flavor="epione",
                      scale_to=scale_to, binarize=False, scale_dims=False)
            U = tmp._raw_coords(X)
            scale_mean = U.mean(axis=0)
            scale_sd = U.std(axis=0, ddof=1)
        return cls(
            features=features, loadings=loadings, singular_values=s,
            feature_sums=feature_sums, n_cells=n_cells, flavor="epione",
            scale_to=scale_to, binarize=binarize, scale_dims=scale_dims,
            dims=None, scale_mean=scale_mean, scale_sd=scale_sd,
            seed=int(params.get("seed") or 0),
            closed=bool(closed),
        )

    # -- projection --------------------------------------------------------

    def _raw_coords(self, X) -> np.ndarray:
        """Normalise ``X`` (cells x model features) and project, unscaled."""
        X = _to_csr(X).astype(np.float64)
        if self.binarize:
            X = _binarize_csr(X)
        row = np.asarray(X.sum(axis=1)).ravel()
        if self.flavor == "archr":
            if np.any(row == 0):
                raise ValueError("some profiles have no reads in the model features")
            tf = sp.diags(1.0 / row) @ X
            if self.lsi_method == 1:
                idf = np.log(1.0 + self.n_cells / self.feature_sums)
                Xn = tf @ sp.diags(idf)
            elif self.lsi_method == 2:
                idf = self.n_cells / self.feature_sums
                Xn = (tf @ sp.diags(idf)).tocsr()
                Xn.data = np.log(Xn.data * self.scale_to + 1.0)
            elif self.lsi_method == 3:
                tf = tf.tocsr()
                tf.data = np.log(tf.data + 1.0)
                idf = np.log(1.0 + self.n_cells / self.feature_sums)
                Xn = tf @ sp.diags(idf)
            else:
                raise ValueError(f"unknown ArchR LSIMethod {self.lsi_method}")
            # ArchR: t(mat) %*% u %*% diag(1/d), then %*% diag(d).
            d = self.singular_values
            V = np.asarray(Xn @ self.loadings) @ np.diag(1.0 / d)
            return V @ np.diag(d)
        # Epione flavour: same normalisation as iterative_lsi's _tf_logidf.
        fs = np.where(self.feature_sums > 0, self.feature_sums, 1.0)
        idf = np.log(1.0 + self.n_cells / fs)
        row_safe = np.where(row > 0, row, 1.0)
        Xn = (sp.diags(1.0 / row_safe) @ X @ sp.diags(idf)).tocsr()
        Xn.data = np.log1p(Xn.data * self.scale_to)
        s = np.where(self.singular_values > 0, self.singular_values, 1.0)
        return np.asarray(Xn @ self.loadings) / s[None, :]

    def transform(self, X, scale: Optional[bool] = None) -> np.ndarray:
        """Project profiles (cells x model features) into the LSI space.

        Returns the coordinates of the embedding dims (``dims``), scaled as
        the single-cell embedding was when ``scale`` (default
        ``scale_dims``).
        """
        scale = self.scale_dims if scale is None else scale
        U = self._raw_coords(X)
        if self.flavor == "archr":
            if scale:
                U = _row_zscores(U)
            return U if self.dims is None else U[:, self.dims]
        if self.dims is not None:
            U = U[:, self.dims]
        if scale:
            sd = np.where(self.scale_sd > 0, self.scale_sd, 1.0)
            U = (U - self.scale_mean[None, :]) / sd[None, :]
        return U


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r_round(x: float) -> float:
    """R's round(): IEC 60559 round-half-to-even, like numpy."""
    return float(np.round(x))


def _row_zscores(m: np.ndarray) -> np.ndarray:
    """ArchR ``.rowZscores``: centre and scale each row (SD with n - 1)."""
    mu = m.mean(axis=1, keepdims=True)
    sd = m.std(axis=1, ddof=1, keepdims=True)
    return (m - mu) / sd


def _to_csr(X) -> sp.csr_matrix:
    if type(X).__name__ in ("BackedArray", "_SubsetBackedArray",
                            "TransformedBackedArray", "ScaledBackedArray"):
        X = sp.vstack([c for _, _, c in X.chunked()])
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(np.asarray(X))


def _binarize_csr(X: sp.csr_matrix) -> sp.csr_matrix:
    X = X.copy()
    X.data = (X.data > 0).astype(X.dtype)
    X.eliminate_zeros()
    return X


def _parse_region_names(names: Sequence[str]) -> pd.DataFrame:
    s = pd.Series(np.asarray(names, dtype=str))
    parts = s.str.extract(r"^(?P<chrom>[^:]+)[:_-](?P<start>\d+)[-_](?P<end>\d+)$")
    if parts.isna().any().any():
        bad = s[parts.isna().any(axis=1)].iloc[0]
        raise ValueError(f"cannot parse region name {bad!r}; expected 'chr:start-end'")
    return pd.DataFrame({
        "chrom": parts["chrom"].to_numpy(),
        "start": parts["start"].astype(np.int64).to_numpy(),
        "end": parts["end"].astype(np.int64).to_numpy(),
    })


def _var_regions(adata: AnnData, mask: Optional[np.ndarray] = None) -> pd.DataFrame:
    v = adata.var if mask is None else adata.var.iloc[np.where(mask)[0]]
    if {"chrom", "start", "end"}.issubset(v.columns):
        return pd.DataFrame({
            "chrom": v["chrom"].astype(str).to_numpy(),
            "start": v["start"].astype(np.int64).to_numpy(),
            "end": v["end"].astype(np.int64).to_numpy(),
        })
    return _parse_region_names(v.index)


def _looks_closed(regions: pd.DataFrame) -> bool:
    """True when region widths look like ``end = start + bin - 1`` (e.g. 499)."""
    w = (regions["end"] - regions["start"]).to_numpy()
    if w.size == 0:
        return False
    return bool(np.mean((w + 1) % 10 == 0) > 0.9 and np.mean(w % 10 == 0) < 0.1)


def _overlap_pairs(peaks: pd.DataFrame, features: pd.DataFrame, closed: bool):
    """All (peak, feature) overlapping pairs, sorted by peak then feature.

    ``features`` must be non-overlapping within each chromosome (tiles or
    merged peaks). Intervals are closed ``[start, end]`` when ``closed``, else
    half-open ``[start, end)``.
    """
    q_idx, s_idx = [], []
    f_chrom = features["chrom"].to_numpy()
    p_chrom = peaks["chrom"].to_numpy()
    for chrom in pd.unique(p_chrom):
        fi = np.where(f_chrom == chrom)[0]
        if fi.size == 0:
            continue
        fs = features["start"].to_numpy()[fi]
        fe = features["end"].to_numpy()[fi]
        order = np.argsort(fs, kind="stable")
        fi, fs, fe = fi[order], fs[order], fe[order]
        if fi.size > 1 and np.any(fs[1:] < (fe[:-1] + (1 if closed else 0))):
            raise ValueError(f"model features overlap each other on {chrom}")
        pi = np.where(p_chrom == chrom)[0]
        ps = peaks["start"].to_numpy()[pi]
        pe = peaks["end"].to_numpy()[pi]
        if closed:
            lo = np.searchsorted(fe, ps, side="left")    # first feature with end >= start
            hi = np.searchsorted(fs, pe, side="right")   # features with start <= end
        else:
            lo = np.searchsorted(fe, ps, side="right")   # first feature with end > start
            hi = np.searchsorted(fs, pe, side="left")    # features with start < end
        n = np.maximum(hi - lo, 0)
        if n.sum() == 0:
            continue
        rep_p = np.repeat(pi, n)
        offs = np.concatenate([np.arange(k) for k in n]) if n.size else np.empty(0, int)
        rep_f = fi[np.repeat(lo, n) + offs]
        q_idx.append(rep_p)
        s_idx.append(rep_f)
    if not q_idx:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    q = np.concatenate(q_idx)
    s = np.concatenate(s_idx)
    o = np.lexsort((s, q))
    return q[o], s[o]


def bulk_to_features(
    bulk_counts,
    bulk_regions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    closed: bool = False,
    return_overlap: bool = False,
):
    """Map bulk peak counts onto model features, as ArchR does.

    Bulk peaks that overlap no feature are dropped and the rest are ordered
    by decreasing total counts (stable). Each feature takes the counts of the
    first overlapping peak in that order, i.e. the highest-count one; features
    without an overlapping peak get zeros.

    Returns a ``(n_features, n_samples)`` CSC matrix (and the fraction of
    features covered when ``return_overlap``).
    """
    C = sp.csr_matrix(bulk_counts) if not sp.issparse(bulk_counts) else bulk_counts.tocsr()
    if C.shape[0] != len(bulk_regions):
        raise ValueError("bulk_counts rows must match bulk_regions")
    q, s = _overlap_pairs(bulk_regions.reset_index(drop=True),
                          features.reset_index(drop=True), closed=closed)
    if q.size == 0:
        raise ValueError(
            "no overlaps between the bulk regions and the model features; "
            "the data sets are probably incompatible (genome build, chromosome names)"
        )
    # subsetByOverlaps, then order by decreasing rowSums (R order is stable).
    hit = np.unique(q)
    tot = np.asarray(C[hit].sum(axis=1)).ravel()
    order = hit[np.argsort(-tot, kind="stable")]
    rank = np.empty(C.shape[0], dtype=np.int64)
    rank[order] = np.arange(order.size)
    # findOverlaps on the reordered peaks: pairs sorted by (rank, feature).
    o = np.lexsort((s, rank[q]))
    q, s = q[o], s[o]
    _, first = np.unique(s, return_index=True)
    keep = np.sort(first)  # !duplicated(subjectHits), in pair order
    q_keep, s_keep = q[keep], s[keep]
    M = sp.lil_matrix((len(features), C.shape[1]), dtype=C.dtype)
    M[s_keep] = C[q_keep]
    M = M.tocsc()
    frac = np.unique(s).size / len(features)
    return (M, frac) if return_overlap else M


# -- R sampling (bit-exact sample(x, size, replace = TRUE)) -----------------

try:
    import numba as _nb

    @_nb.njit(cache=True)
    def _mt_next(mt, mti):
        N, M = 624, 397
        if mti[0] >= N:
            for kk in range(N):
                y = (mt[kk] & np.uint64(0x80000000)) | (mt[(kk + 1) % N] & np.uint64(0x7FFFFFFF))
                v = mt[(kk + M) % N] ^ (y >> np.uint64(1))
                if y & np.uint64(1):
                    v ^= np.uint64(0x9908B0DF)
                mt[kk] = v
            mti[0] = 0
        y = mt[mti[0]]
        mti[0] += 1
        y ^= y >> np.uint64(11)
        y ^= (y << np.uint64(7)) & np.uint64(0x9D2C5680)
        y ^= (y << np.uint64(15)) & np.uint64(0xEFC60000)
        y ^= y >> np.uint64(18)
        return float(y & np.uint64(0xFFFFFFFF)) * 2.3283064365386963e-10

    @_nb.njit(cache=True)
    def _unif_rand(mt, mti):
        v = _mt_next(mt, mti)
        i2 = 2.328306437080797e-10
        if v <= 0.0:
            return 0.5 * i2
        if 1.0 - v <= 0.0:
            return 1.0 - 0.5 * i2
        return v

    @_nb.njit(cache=True)
    def _r_sample_replace(mt, mti, n, size):
        out = np.empty(size, dtype=np.int64)
        bits = int(math.ceil(math.log2(n))) if n > 1 else 0
        for i in range(size):
            if n <= 1:
                out[i] = 1
                continue
            while True:
                v = 0
                k = 0
                while k <= bits:
                    v1 = int(math.floor(_unif_rand(mt, mti) * 65536))
                    v = 65536 * v + v1
                    k += 16
                dv = v & ((1 << bits) - 1)
                if n > dv:
                    out[i] = dv + 1
                    break
        return out

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is a hard dependency via scanpy
    _HAVE_NUMBA = False


class _RSampler:
    """R's Mersenne-Twister after ``set.seed(seed)``, for ``sample(replace=TRUE)``."""

    def __init__(self, seed: int):
        from ._r_rng import RRandom
        r = RRandom(seed)
        self._mt = r._mt.copy()
        self._mti = np.array([r._mti], dtype=np.int64)

    def sample_replace(self, n: int, size: int) -> np.ndarray:
        """1-based indices, like ``sample.int(n, size, replace = TRUE)``."""
        if _HAVE_NUMBA:
            return _r_sample_replace(self._mt, self._mti, int(n), int(size))
        from ._r_rng import RRandom  # slow fallback
        r = RRandom.__new__(RRandom)
        r.kind = "Mersenne-Twister"
        r._mt, r._mti = self._mt, int(self._mti[0])
        out = r.sample_int(int(n), int(size), replace=True)
        self._mti[0] = r._mti
        return out


_RATIOS = (2.0, 1.5, 1.0, 0.5, 0.25)


def _simulate_cells_archr(counts: np.ndarray, depth: int, n2: int, sampler: _RSampler,
                          ratios=_RATIOS) -> sp.csr_matrix:
    """ArchR's pseudo-cell simulation for one sample, bit-exact.

    Returns pseudo-cells x features counts (duplicates summed, as
    ``Matrix::sparseMatrix`` does).
    """
    counts = np.asarray(counts, dtype=np.int64)
    pool = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    rows, cols = [], []
    for y, ratio in enumerate(ratios):
        nrow = int(math.ceil(ratio * depth))
        draw = sampler.sample_replace(pool.size, nrow * n2) - 1
        feat = pool[draw]
        # matrix(..., ncol = n2) fills column-major: draw k -> cell k // nrow.
        cell = np.arange(feat.size, dtype=np.int64) // nrow + y * n2
        rows.append(cell)
        cols.append(feat)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    M = sp.coo_matrix((np.ones(r.size, dtype=np.float64), (r, c)),
                      shape=(n2 * len(ratios), counts.size)).tocsr()
    M.sum_duplicates()
    return M


def _simulate_cells_numpy(counts: np.ndarray, depth: int, n2: int,
                          rng: np.random.Generator, ratios=_RATIOS) -> sp.csr_matrix:
    """Same distribution as ArchR's sampling (multinomial), numpy RNG."""
    counts = np.asarray(counts, dtype=np.float64)
    p = counts / counts.sum()
    nz = np.where(p > 0)[0]
    blocks = []
    for ratio in ratios:
        nrow = int(math.ceil(ratio * depth))
        m = rng.multinomial(nrow, p[nz], size=n2)
        blk = sp.csr_matrix(m.astype(np.float64))
        blocks.append(blk)
    M = sp.vstack(blocks).tocsr()
    out = sp.csr_matrix((M.data, nz[M.indices], M.indptr), shape=(M.shape[0], counts.size))
    out.eliminate_zeros()
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _read_bulk(bulk, regions):
    """Normalise the bulk input to (counts regions x samples, regions, names)."""
    if isinstance(bulk, AnnData):
        # samples x regions, as everywhere else in Epione
        counts = bulk.X.T
        regs = _var_regions(bulk)
        names = list(map(str, bulk.obs_names))
    elif isinstance(bulk, pd.DataFrame):
        cols = [c for c in bulk.columns if str(c).lower() in ("chr", "chrom", "seqnames")]
        if regions is None and cols and {"start", "end"} <= {str(c).lower() for c in bulk.columns}:
            lower = {str(c).lower(): c for c in bulk.columns}
            regs = pd.DataFrame({
                "chrom": bulk[cols[0]].astype(str).to_numpy(),
                "start": bulk[lower["start"]].astype(np.int64).to_numpy(),
                "end": bulk[lower["end"]].astype(np.int64).to_numpy(),
            })
            data = bulk.drop(columns=[cols[0], lower["start"], lower["end"]])
        else:
            regs = _parse_region_names(bulk.index) if regions is None else regions
            data = bulk
        counts = sp.csr_matrix(data.to_numpy())
        names = list(map(str, data.columns))
    else:
        if regions is None:
            raise ValueError("pass `regions` with a plain count matrix")
        counts = bulk
        regs = regions
        names = [f"bulk{i + 1}" for i in range(counts.shape[1])]
    regs = regs.reset_index(drop=True)
    if counts.shape[0] != len(regs):
        raise ValueError("bulk counts and regions disagree in length")
    if sp.issparse(counts):
        counts = counts.tocsr()
        if counts.dtype.kind == "f" and np.any(counts.data != np.round(counts.data)):
            raise ValueError("bulk counts must be integers")
        counts = counts.astype(np.int64)
    else:
        counts = np.asarray(counts)
        if counts.dtype.kind == "f" and np.any(counts != np.round(counts)):
            raise ValueError("bulk counts must be integers")
        counts = sp.csr_matrix(counts.astype(np.int64))
    return counts, regs, names


def project_bulk_atac(
    adata: AnnData,
    bulk,
    *,
    regions: Optional[pd.DataFrame] = None,
    lsi_key: str = "X_iterative_lsi",
    model: Optional[LSIModel] = None,
    n_cells: int = 250,
    embedding: Optional[str] = "X_umap",
    umap_model=None,
    rng: Literal["numpy", "archr"] = "numpy",
    seed: int = 1,
    groupby: Optional[str] = None,
    n_neighbors: int = 10,
    force: bool = False,
    verbose: bool = True,
) -> AnnData:
    """Project bulk ATAC-seq samples into a single-cell embedding.

    Port of ArchR's ``projectBulkATAC``. Each bulk sample is turned into
    ``n_cells`` pseudo-cells: its reads are sampled with replacement at
    2, 1.5, 1, 0.5 and 0.25 times the mean single-cell depth (accessible LSI
    features per cell), ``n_cells / 5`` pseudo-cells per depth. The
    pseudo-cells are projected into the single-cell LSI space with the stored
    LSI model and, if ``embedding`` is given, into the UMAP.

    Parameters
    ----------
    adata
        Single-cell AnnData with an LSI fit (:func:`epione.tl.iterative_lsi`),
        or any AnnData when ``model`` is given.
    bulk
        Bulk counts: an AnnData (samples x regions), a DataFrame with
        ``chrom/start/end`` columns plus one column per sample (or regions as
        ``chr:start-end`` index), or a regions x samples matrix with
        ``regions``. Counts must be integers.
    regions
        Regions (``chrom``, ``start``, ``end``) for a plain matrix.
    lsi_key
        Key of the LSI fit, used when ``model`` is None.
    model
        An :class:`LSIModel`; defaults to ``LSIModel.from_adata(adata, lsi_key)``.
    n_cells
        Pseudo-cells per sample (rounded up to a multiple of 5).
    embedding
        ``adata.obsm`` key of a UMAP to project into, or None to stop at the
        LSI. See Notes.
    umap_model
        A fitted ``umap.UMAP`` whose input is the model's scaled LSI. If None
        and ``embedding`` is set, one is fitted on ``adata`` with the
        embedding as initialisation.
    rng
        ``"numpy"``: multinomial sampling with ``numpy.random.default_rng``.
        ``"archr"``: reproduce ArchR's draws bit-exactly, assuming
        ``set.seed(seed)`` was called right before ``projectBulkATAC`` with
        one thread. ArchR re-seeds with the LSI seed inside its projection,
        so every sample after the first is drawn from that same RNG state;
        this is reproduced too.
    seed
        Seed for the sampling.
    groupby
        Optional ``adata.obs`` column. Each pseudo-cell gets the majority
        label of its ``n_neighbors`` nearest single cells in the scaled LSI
        space (``obs['nearest_<groupby>']``), and
        ``uns['group_fractions']`` summarises the labels per sample.
    force
        Continue when fewer than 25% of model features are covered by bulk
        regions (ArchR errors in that case unless ``force``).

    Returns
    -------
    AnnData of pseudo-cells with ``obs['sample']`` and ``obs['depth_ratio']``,
    ``obsm['X_lsi']`` (the model's scaled LSI dims) and, with an embedding,
    ``obsm[embedding]``. ``uns`` holds the overlap ratio, the sampling depth,
    the single-cell coordinates under the fitted UMAP model
    (``uns['reference_umap']``) and the re-projection correlation.

    Notes
    -----
    ArchR projects through the saved ``uwot`` model. Epione's UMAPs come
    from scanpy, which keeps no model, so a ``umap-learn`` model is fitted on
    the same LSI with the existing embedding as initialisation. The
    single-cell layout under that model (``uns['reference_umap']``) is close
    to, but not identical with, ``adata.obsm[embedding]``; plot pseudo-cells
    against ``uns['reference_umap']``. As in ArchR, a warning is raised when a
    re-projection of up to 5,000 single cells correlates below 0.8 with their
    coordinates.

    Examples
    --------
    >>> proj = epi.tl.project_bulk_atac(tiles, bulk, groupby="Clusters")
    >>> proj.uns["group_fractions"]
    """
    model = model if model is not None else LSIModel.from_adata(adata, lsi_key)
    counts, regs, names = _read_bulk(bulk, regions)
    M, frac = bulk_to_features(counts, regs, model.features, closed=model.closed,
                               return_overlap=True)
    if verbose:
        console.info(f"[project_bulk_atac] overlap ratio of LSI features = {frac:.4f}")
    if frac < 0.25:
        msg = (f"only {frac:.1%} of the LSI features are covered by the bulk regions")
        if not force:
            raise ValueError(msg + "; set force=True to continue")
        warnings.warn(msg + "; continuing because force=True", stacklevel=2)

    depth = model.mean_depth
    n_rep = len(_RATIOS)
    n2 = int(math.ceil(n_cells / n_rep))
    M = M.tocsc()
    sims, sample_lab, ratio_lab = [], [], []
    np_rng = np.random.default_rng(seed)
    sampler = _RSampler(seed) if rng == "archr" else None
    for j, name in enumerate(names):
        if verbose:
            console.info(f"[project_bulk_atac] sample {j + 1}/{len(names)}: {name}")
        cvec = M[:, j].toarray().ravel()
        if cvec.sum() == 0:
            raise ValueError(f"bulk sample {name!r} has no reads in the LSI features")
        if rng == "archr":
            if j > 0:
                sampler = _RSampler(model.seed)  # .projectLSI's set.seed(LSI$seed)
            sim = _simulate_cells_archr(cvec, depth, n2, sampler)
        elif rng == "numpy":
            sim = _simulate_cells_numpy(cvec, depth, n2, np_rng)
        else:
            raise ValueError("rng must be 'numpy' or 'archr'")
        sims.append(sim)
        sample_lab += [name] * sim.shape[0]
        ratio_lab += list(np.repeat(_RATIOS, n2))
    S = sp.vstack(sims).tocsr()
    lsi = model.transform(S) if model.flavor == "epione" else None
    if model.flavor == "archr":
        raw = model._raw_coords(S)
        if embedding is None:
            lsi = _row_zscores(raw) if model.scale_dims else raw
        else:
            lsi = _row_zscores(raw)  # projectBulkATAC always scales before the UMAP
        if model.dims is not None:
            lsi = lsi[:, model.dims]

    obs_names = [f"{s}#{i + 1}" for s, i in zip(sample_lab, _per_sample_index(sample_lab))]
    out = AnnData(
        X=sp.csr_matrix((S.shape[0], 0), dtype=np.float32),
        obs=pd.DataFrame({"sample": pd.Categorical(sample_lab, categories=names),
                          "depth_ratio": np.asarray(ratio_lab, dtype=np.float64)},
                         index=obs_names),
    )
    out.obsm["X_lsi"] = lsi.astype(np.float64)
    out.uns["project_bulk_atac"] = {
        "overlap_ratio": float(frac), "depth": int(depth), "n_cells_per_depth": int(n2),
        "depth_ratios": np.asarray(_RATIOS), "rng": rng, "seed": int(seed),
        "lsi_flavor": model.flavor,
    }
    out.uns["simulated_counts_nnz"] = int(S.nnz)

    if groupby is not None or embedding is not None:
        ref = _reference_lsi(adata, model, lsi_key)
    if groupby is not None:
        _nearest_groups(out, adata, ref, groupby, n_neighbors)

    if embedding is not None:
        _project_umap(out, adata, ref, embedding, umap_model, seed, verbose)
    return out


def _per_sample_index(labels):
    seen = {}
    out = []
    for lab in labels:
        k = seen.get(lab, 0)
        out.append(k)
        seen[lab] = k + 1
    return out


def _reference_lsi(adata: AnnData, model: LSIModel, lsi_key: str) -> np.ndarray:
    """Single-cell coordinates in the same space as the projected pseudo-cells.

    For an Epione model this is the stored ``iterative_lsi`` embedding. For an
    ArchR model, ``adata.obsm[lsi_key]`` must hold ArchR's scaled LSI of the
    embedding dims (``getReducedDims(..., scaleDims = TRUE)``).
    """
    if lsi_key not in adata.obsm:
        raise KeyError(f"adata.obsm[{lsi_key!r}] not found")
    return np.asarray(adata.obsm[lsi_key], dtype=np.float64)


def _nearest_groups(out: AnnData, adata: AnnData, ref: np.ndarray, groupby: str, k: int):
    from sklearn.neighbors import NearestNeighbors
    labels = adata.obs[groupby].astype(str).to_numpy()
    nn = NearestNeighbors(n_neighbors=k).fit(ref)
    _, idx = nn.kneighbors(out.obsm["X_lsi"])
    lab = labels[idx]
    maj = []
    for row in lab:
        vals, cnt = np.unique(row, return_counts=True)
        maj.append(vals[np.argmax(cnt)])
    cats = list(pd.Categorical(adata.obs[groupby]).categories.astype(str))
    out.obs[f"nearest_{groupby}"] = pd.Categorical(maj, categories=cats)
    tab = pd.crosstab(out.obs["sample"], out.obs[f"nearest_{groupby}"], normalize="index")
    out.uns["group_fractions"] = tab.reindex(columns=cats, fill_value=0.0)


def _project_umap(out, adata, ref, embedding, umap_model, seed, verbose):
    import umap as _umap
    if umap_model is None:
        if embedding not in adata.obsm:
            raise KeyError(f"adata.obsm[{embedding!r}] not found")
        nb = adata.uns.get("neighbors", {}).get("params", {})
        up = adata.uns.get("umap", {}).get("params", {})
        n_neighbors = int(nb.get("n_neighbors", 30))
        metric = str(nb.get("metric", "cosine"))
        kw = {}
        if "a" in up and "b" in up:
            kw.update(a=float(up["a"]), b=float(up["b"]))
        else:
            kw.update(min_dist=float(up.get("min_dist", 0.5)))
        init = np.asarray(adata.obsm[embedding], dtype=np.float64)
        if verbose:
            console.info(f"[project_bulk_atac] fitting umap-learn (n_neighbors={n_neighbors}, "
                         f"metric={metric}) initialised from adata.obsm[{embedding!r}]")
        umap_model = _umap.UMAP(n_neighbors=n_neighbors, metric=metric, init=init,
                                random_state=seed, transform_seed=seed, **kw)
        umap_model.fit(ref)
    reference = np.asarray(umap_model.embedding_, dtype=np.float64)
    rs = np.random.default_rng(seed)
    idx = np.sort(rs.choice(ref.shape[0], size=min(ref.shape[0], 5000), replace=False))
    re = umap_model.transform(ref[idx])
    c = [float(np.corrcoef(re[:, i], reference[idx, i])[0, 1]) for i in range(2)]
    if min(c) < 0.8:
        warnings.warn(
            f"re-projection correlation is {min(c):.3f} (< 0.8); the projection may not be "
            "accurate because of limited heterogeneity in the single-cell data",
            stacklevel=3,
        )
    out.obsm[embedding] = np.asarray(umap_model.transform(out.obsm["X_lsi"]), dtype=np.float64)
    out.uns["reference_umap"] = reference
    out.uns["reference_obs_names"] = np.asarray(adata.obs_names, dtype=str)
    out.uns["project_bulk_atac"]["reprojection_correlation"] = np.asarray(c)


def deconvolve_bulk(
    adata: AnnData,
    bulk,
    groupby: str,
    *,
    regions: Optional[pd.DataFrame] = None,
    layer: Optional[str] = None,
    n_features: int = 5000,
    min_cells: int = 20,
    weight: Literal["reads", "cells"] = "cells",
) -> pd.DataFrame:
    """Estimate cell-group composition of bulk ATAC samples by NNLS.

    Each bulk profile ``b`` is modelled as a non-negative combination of
    cell-group pseudobulk profiles ``P`` (``b ~ P w``, ``w >= 0``), the
    reference-based least-squares model most bulk ATAC deconvolution methods
    build on (e.g. EPIC-ATAC adds marker weighting and an "other cells"
    term). Profiles are summed single-cell counts per group, both sides are
    scaled to sum to one over the selected features, and ``w`` is solved with
    ``scipy.optimize.nnls``.

    Parameters
    ----------
    adata
        Single-cell counts (cells x regions, e.g. peaks or tiles) with
        ``adata.obs[groupby]``.
    bulk, regions
        As in :func:`project_bulk_atac`. Bulk regions are mapped onto
        ``adata.var`` regions with the same rule.
    groupby
        Cell-group column.
    layer
        Counts layer; default ``adata.X``.
    n_features
        Use the ``n_features`` features whose log pseudobulk accessibility
        varies most across groups, among those covered by the bulk regions.
    min_cells
        Groups with fewer cells are dropped.
    weight
        ``"cells"`` divides read weights by each group's mean reads per cell
        in the selected features, so fractions estimate cell proportions;
        ``"reads"`` returns read-contribution fractions.

    Returns
    -------
    DataFrame, samples x groups, rows summing to one, with an extra
    ``residual_r2`` column (fraction of the bulk profile's variance explained).
    """
    from scipy.optimize import nnls

    counts, regs, names = _read_bulk(bulk, regions)
    feats = _var_regions(adata)
    B = bulk_to_features(counts, regs, feats, closed=_looks_closed(feats)).toarray().astype(np.float64)
    X = _to_csr(adata.layers[layer] if layer is not None else adata.X).astype(np.float64)
    groups = pd.Categorical(adata.obs[groupby])
    keep_groups = [g for g in groups.categories if (groups == g).sum() >= min_cells]
    if len(keep_groups) < 2:
        raise ValueError("need at least two groups with enough cells")
    P = np.zeros((X.shape[1], len(keep_groups)))
    n_per = np.zeros(len(keep_groups))
    for k, g in enumerate(keep_groups):
        m = np.asarray(groups == g)
        P[:, k] = np.asarray(X[m].sum(axis=0)).ravel()
        n_per[k] = m.sum()
    covered = B.sum(axis=1) > 0
    cpm = P / P.sum(axis=0, keepdims=True) * 1e6
    lv = np.log1p(cpm).var(axis=1)
    lv[~covered] = -np.inf
    nf = int(min(n_features, covered.sum()))
    sel = np.argsort(-lv, kind="stable")[:nf]
    Ps = P[sel]
    per_cell = Ps.sum(axis=0) / n_per  # mean reads per cell in the selected features
    Pn = Ps / Ps.sum(axis=0, keepdims=True)
    rows = []
    for j, name in enumerate(names):
        b = B[sel, j]
        if b.sum() == 0:
            raise ValueError(f"bulk sample {name!r} has no reads in the selected features")
        bn = b / b.sum()
        w, _ = nnls(Pn, bn)
        fitted = Pn @ w
        ss_res = float(((bn - fitted) ** 2).sum())
        ss_tot = float(((bn - bn.mean()) ** 2).sum())
        if weight == "cells":
            w = w / per_cell
        elif weight != "reads":
            raise ValueError("weight must be 'cells' or 'reads'")
        w = w / w.sum() if w.sum() > 0 else w
        rows.append(list(w) + [1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan])
    return pd.DataFrame(rows, index=names, columns=[str(g) for g in keep_groups] + ["residual_r2"])
