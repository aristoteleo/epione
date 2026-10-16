"""Exact, multi-threaded replacements for the SciPy sparse operations that
dominate ``iterative_lsi`` (column subsetting, ``A.T @ Q``, per-cluster
column sums, TF-logIDF).

Every function returns *bit-identical* results to the SciPy expression it
replaces (same entries, same order, same float32 operations); only the
work is split over threads (numba ``prange`` or GIL-releasing NumPy ufuncs
on row blocks).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp

_K: dict = {}


def _kernels():
    if _K:
        return _K
    import numba as nb
    jit = nb.njit(nogil=True, cache=False)

    @jit
    def col_subset_count(r0, r1, rows, indptr, indices, colmap, nnz):
        for o in range(r0, r1):
            i = rows[o]
            c = 0
            for p in range(indptr[i], indptr[i + 1]):
                if colmap[indices[p]] >= 0:
                    c += 1
            nnz[o] = c

    @jit
    def col_subset_fill(r0, r1, rows, indptr, indices, data, ones, colmap,
                        o_indptr, o_indices, o_data):
        for o in range(r0, r1):
            i = rows[o]
            q = o_indptr[o]
            for p in range(indptr[i], indptr[i + 1]):
                j = colmap[indices[p]]
                if j >= 0:
                    o_indices[q] = j
                    o_data[q] = 1.0 if ones else data[p]
                    q += 1

    @jit
    def t_matmul(c0, c1, indptr, indices, data, Q, out):
        n = indptr.shape[0] - 1
        k = Q.shape[1]
        for i in range(n):
            a0 = indptr[i]
            a1 = indptr[i + 1]
            p = a0 + np.searchsorted(indices[a0:a1], c0)
            while p < a1 and indices[p] < c1:
                f = indices[p]
                v = data[p]
                for j in range(k):
                    out[f, j] += v * Q[i, j]
                p += 1

    @jit
    def col_counts(indices, n_cols):
        out = np.zeros(n_cols, dtype=np.int64)
        for p in range(indices.shape[0]):
            out[indices[p]] += 1
        return out

    @jit
    def group_col_sums(indptr, indices, data, ones, codes, out):
        n = indptr.shape[0] - 1
        for i in range(n):
            g = codes[i]
            for p in range(indptr[i], indptr[i + 1]):
                out[g, indices[p]] += 1.0 if ones else data[p]

    _K.update(col_subset_count=col_subset_count, col_subset_fill=col_subset_fill,
              group_col_sums=group_col_sums, t_matmul=t_matmul,
              col_counts=col_counts)
    return _K


def _blocks(indptr, n_jobs, n_blocks=None):
    from ..pp._fragment_io import resolve_n_jobs
    nt = resolve_n_jobs(n_jobs)
    n = len(indptr) - 1
    nb_ = max(1, min(n, n_blocks or 4 * nt))
    # balance by nnz
    edges = np.searchsorted(indptr, np.linspace(0, indptr[-1], nb_ + 1)[1:-1])
    edges = np.unique(np.r_[0, np.minimum(edges, n), n])
    return nt, [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _pool_map(nt, fn, blocks):
    if not blocks:
        return []
    fn(blocks[0][0], blocks[0][0])                 # JIT outside the pool
    with ThreadPoolExecutor(max_workers=nt) as pool:
        return list(pool.map(lambda ab: fn(*ab), blocks))


def _idx_dtype(*vals):
    return np.int32 if max(vals) < np.iinfo(np.int32).max else np.int64


def col_counts(indices: np.ndarray, n_cols: int) -> np.ndarray:
    """``np.bincount(indices, minlength=n_cols)`` without the int64 copy
    of ``indices`` that ``bincount`` makes."""
    return _kernels()["col_counts"](indices, int(n_cols))


class BinaryCSR:
    """Structure of a CSR matrix whose stored values are all 1.0 (float32):
    the binarised tile matrix without its data array. Every derived matrix
    / statistic equals the one computed from the materialised float32
    0/1 matrix (sums of ones are exact integers below 2**24)."""

    dtype = np.dtype(np.float32)

    def __init__(self, indptr, indices, shape):
        self.indptr = indptr
        self.indices = indices
        self.shape = tuple(shape)

    @property
    def nnz(self):
        return int(self.indptr[-1])

    def col_sums(self) -> np.ndarray:
        return col_counts(self.indices, self.shape[1]).astype(np.float32)

    def row_sums(self) -> np.ndarray:
        return np.diff(self.indptr).astype(np.float32)

    def tocsr(self) -> sp.csr_matrix:
        return sp.csr_matrix((np.ones(self.nnz, dtype=np.float32), self.indices, self.indptr),
                             shape=self.shape)


def csr_select(X, cols: np.ndarray, rows=None, n_jobs=None) -> sp.csr_matrix:
    """``X[rows][:, cols]`` (``rows`` in the given order, ``None`` = all) for
    sorted unique ``cols``; ``X`` a canonical CSR matrix or a
    :class:`BinaryCSR`. Same entries in the same order as SciPy's fancy
    indexing."""
    k = _kernels()
    ones = isinstance(X, BinaryCSR)
    data_in = np.zeros(1, dtype=np.float32) if ones else X.data
    out_dt = np.float32 if ones else X.dtype
    colmap = np.full(X.shape[1], -1, dtype=np.int64)
    colmap[cols] = np.arange(len(cols))
    rows = (np.arange(X.shape[0], dtype=np.int64) if rows is None
            else np.asarray(rows, dtype=np.int64))
    n = len(rows)
    sub_ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.diff(X.indptr)[rows], out=sub_ptr[1:])
    nt, blocks = _blocks(sub_ptr, n_jobs)
    nnz = np.zeros(n, dtype=np.int64)
    _pool_map(nt, lambda a, b: k["col_subset_count"](a, b, rows, X.indptr, X.indices,
                                                     colmap, nnz), blocks)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(nnz, out=indptr[1:])
    dt = _idx_dtype(indptr[-1], len(cols), n)
    indices = np.empty(indptr[-1], dtype=dt)
    data = np.empty(indptr[-1], dtype=out_dt)
    _pool_map(nt, lambda a, b: k["col_subset_fill"](a, b, rows, X.indptr, X.indices, data_in,
                                                    ones, colmap, indptr, indices, data), blocks)
    out = sp.csr_matrix((data, indices, indptr.astype(dt, copy=False)), shape=(n, len(cols)))
    out.has_sorted_indices = True
    return out


def csr_select_columns(X, cols: np.ndarray, n_jobs=None) -> sp.csr_matrix:
    """``X[:, cols]`` for sorted unique ``cols`` (canonical CSR in/out)."""
    return csr_select(X, cols, rows=None, n_jobs=n_jobs)


def csr_t_matmul(X, Q: np.ndarray, n_jobs=None) -> np.ndarray:
    """``X.T @ Q`` for canonical CSR ``X`` and dense ``Q``, bit-identical to
    SciPy (``csc_matvecs``: each output row sums over rows of ``X`` in
    increasing order), split over column ranges on ``n_jobs`` threads."""
    from ..pp._fragment_io import resolve_n_jobs
    k = _kernels()
    nt = resolve_n_jobs(n_jobs)
    n, m = X.shape
    dt = np.result_type(X.dtype, Q.dtype)
    Q = np.ascontiguousarray(Q, dtype=dt)
    out = np.zeros((m, Q.shape[1]), dtype=dt)
    colw = getattr(X, "_col_nnz_cum", None)
    if colw is None:
        colw = np.cumsum(col_counts(X.indices, m))
        try:
            X._col_nnz_cum = colw
        except AttributeError:
            pass
    nb_ = max(1, min(m, 4 * nt))
    edges = np.searchsorted(colw, np.linspace(0, colw[-1] if m else 0, nb_ + 1)[1:-1])
    edges = np.unique(np.r_[0, np.minimum(edges, m), m])
    blocks = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    data = X.data if X.data.dtype == dt else X.data.astype(dt)
    _pool_map(nt, lambda a, b: k["t_matmul"](a, b, X.indptr, X.indices, data, Q, out), blocks)
    return out


def group_col_sums(X: sp.csr_matrix, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """``out[g] = X[codes == g].sum(axis=0)`` for binary float32 ``X``
    (exact: sums of 0/1 values are integers < 2**24)."""
    ones = isinstance(X, BinaryCSR)
    out = np.zeros((n_groups, X.shape[1]), dtype=np.float32 if ones else X.dtype)
    _kernels()["group_col_sums"](X.indptr, X.indices,
                                 np.zeros(1, np.float32) if ones else X.data, ones,
                                 np.asarray(codes, dtype=np.int64), out)
    return out


def tf_logidf_data(X: sp.csr_matrix, inv_row: np.ndarray, idf: np.ndarray,
                   scale_to: float, n_jobs=None) -> sp.csr_matrix:
    """``log1p(X.multiply(inv_row[:, None]).multiply(idf) * scale_to)`` with
    float32 ``inv_row`` / ``idf``, computed on row blocks in threads."""
    from ..pp._fragment_io import resolve_n_jobs
    n = X.shape[0]
    out = np.empty(X.nnz, dtype=np.float32)
    nt = resolve_n_jobs(n_jobs)
    edges = np.linspace(0, n, max(1, min(n, 4 * nt)) + 1).astype(np.int64)

    def run(j):
        r0, r1 = edges[j], edges[j + 1]
        a, b = X.indptr[r0], X.indptr[r1]
        rows = np.repeat(inv_row[r0:r1], np.diff(X.indptr[r0:r1 + 1]))
        d = X.data[a:b] * rows
        d = d * idf[X.indices[a:b]]
        out[a:b] = np.log1p(d * scale_to).astype(np.float32)

    with ThreadPoolExecutor(max_workers=nt) as pool:
        list(pool.map(run, range(len(edges) - 1)))
    res = sp.csr_matrix((out, X.indices, X.indptr), shape=X.shape, copy=False)
    res.has_sorted_indices = bool(X.has_sorted_indices)
    return res
