"""Matrix-free TF-logIDF operator for :func:`epione.tl.iterative_lsi`.

The TF-logIDF matrix of a cell x feature subset of a binary tile matrix is
never materialised. :class:`TfidfStream` keeps only per-row / per-column
statistics (inverse row sums, IDF) of the subset and, for every product with
a dense block, rebuilds row blocks of the subset from the *input* CSR
structure (column re-numbering by a numba kernel, TF-logIDF values by the
same NumPy expression as :func:`epione.tl._lsi_kernels.tf_logidf_data`).
Memory is bounded by a few block buffers, independent of the matrix size --
the same idea as rapids-singlecell's chunked sparse PCA (stream CSR row
batches, apply the transform per batch, accumulate small dense results).

Products are bit-identical to the materialised matrix:

* ``A @ Q``: each row block is multiplied by SciPy's own ``csr @ dense``
  kernel, which computes every output row independently;
* ``A.T @ Q``: row blocks are visited in increasing row order and each one
  is applied by the column-partitioned numba kernel of
  :func:`~epione.tl._lsi_kernels.csr_t_matmul`, so every output entry sums
  its terms in increasing row order, exactly as SciPy's ``csc_matvecs``.

:class:`TorchTfidfStream` is the device twin used by ``device="cuda"``: the
subset's row blocks (int32 indices, float32 values) are kept on the device
while they fit the memory budget and streamed host -> device for every
product otherwise, so device memory stays bounded whatever the input size.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from ._lsi_kernels import BinaryCSR, _kernels

#: target nnz of one host block (per thread); ~2M nnz = ~24 MB of scratch
_BLOCK_NNZ = int(os.environ.get("EPIONE_LSI_BLOCK_NNZ", 1 << 21))
#: cap (GiB) of the device bytes the LSI operand may keep resident when
#: ``EPIONE_GPU_MEMORY_GB`` is not set
_GPU_CAP_GB = 12.0


def tf_logidf_params(col: np.ndarray, row: np.ndarray, n_cells: int):
    """``(inv_row, idf)`` (float32) of :func:`_tf_logidf` for float32 column /
    row sums -- the very expressions used there."""
    col_safe = np.where(col > 0, col, 1.0)
    idf = np.log(1.0 + n_cells / col_safe)
    row_safe = np.where(row > 0, row, 1.0)
    return (1.0 / row_safe).astype(np.float32), idf.astype(np.float32)


def _resolve_threads(n_jobs) -> int:
    from ..pp._fragment_io import resolve_n_jobs
    return resolve_n_jobs(n_jobs)


def _ranges(cum: np.ndarray, target: int, min_parts: int = 1) -> List[Tuple[int, int]]:
    """Split ``range(len(cum) - 1)`` into contiguous ranges of about
    ``target`` weight, where ``cum`` is the cumulative weight (len n + 1)."""
    n = len(cum) - 1
    if n <= 0:
        return []
    total = int(cum[-1])
    parts = max(min_parts, -(-total // max(int(target), 1)), 1)
    parts = min(parts, n)
    edges = np.searchsorted(cum, np.linspace(0, total, parts + 1)[1:-1], side="left")
    edges = np.unique(np.r_[0, np.clip(edges, 0, n), n])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


class TfidfStream:
    """TF-logIDF of ``X[rows][:, cols]`` (``X`` binary) as a linear operator.

    Parameters
    ----------
    X
        :class:`BinaryCSR` (sorted indices) -- the binarised input.
    cols
        Sorted unique selected feature indices.
    rows
        Selected cell indices in output order (``None`` = all cells).
    scale_to
        Multiplier inside ``log1p``.
    n_jobs
        Threads.
    """

    __array_ufunc__ = None          # ndarray @ op -> op.__rmatmul__
    ndim = 2
    dtype = np.dtype(np.float32)

    def __init__(self, X: BinaryCSR, cols: np.ndarray, rows=None,
                 scale_to: float = 10000.0, n_jobs=None,
                 block_nnz: Optional[int] = None):
        k = _kernels()
        self.X = X
        self.nt = _resolve_threads(n_jobs)
        self.scale_to = scale_to
        cols = np.asarray(cols)
        n_vars = X.shape[1]
        self.colmap = np.full(n_vars, -1, dtype=np.int32 if len(cols) < 2**31 else np.int64)
        self.colmap[cols] = np.arange(len(cols), dtype=self.colmap.dtype)
        self.all_rows = rows is None
        self.rows = (np.arange(X.shape[0], dtype=np.int64) if rows is None
                     else np.asarray(rows, dtype=np.int64))
        n = len(self.rows)
        m = len(cols)
        self.shape = (n, m)

        # counting pass (row nnz + column counts), parallel over row ranges
        # weighted by the *input* nnz they scan
        base_w = np.diff(X.indptr)[self.rows] if not self.all_rows else np.diff(X.indptr)
        base_cum = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(base_w, out=base_cum[1:])
        del base_w
        self._scan_cum = base_cum
        row_nnz = np.zeros(n, dtype=np.int64)
        parts = _ranges(base_cum, max(int(base_cum[-1]) // (4 * self.nt), 1), 4 * self.nt)
        col_cnt = np.zeros(m, dtype=np.int64)
        if parts:
            k["sub_counts"](0, 0, self.rows, X.indptr, X.indices, self.colmap,
                            row_nnz, col_cnt)                       # JIT
            n_buf = min(self.nt, len(parts))
            bufs = [np.zeros(m, dtype=np.int64) for _ in range(n_buf)]
            groups = [parts[i::n_buf] for i in range(n_buf)]

            def count(g):
                for a, b in groups[g]:
                    k["sub_counts"](a, b, self.rows, X.indptr, X.indices,
                                    self.colmap, row_nnz, bufs[g])

            with ThreadPoolExecutor(max_workers=n_buf) as pool:
                list(pool.map(count, range(n_buf)))
            for b in bufs:
                col_cnt += b
            del bufs
        self.sub_ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(row_nnz, out=self.sub_ptr[1:])
        del row_nnz
        self.nnz = int(self.sub_ptr[-1])
        self.col_cnt = col_cnt
        # float32 sums of 0/1 values: exact integers (as SciPy's .sum())
        self.row_sums = np.diff(self.sub_ptr).astype(np.float32)
        self.col_sums = col_cnt.astype(np.float32)
        self.inv_row, self.idf = tf_logidf_params(self.col_sums, self.row_sums, n)

        bn = int(block_nnz or _BLOCK_NNZ)
        # output-row blocks of bounded *subset* nnz and bounded scan work
        w = self.sub_ptr + base_cum                      # scan + write cost
        self.blocks = _ranges(w, 2 * bn, 1)
        self._block_nnz = bn
        col_cum = np.zeros(m + 1, dtype=np.int64)
        np.cumsum(col_cnt, out=col_cum[1:])
        self._col_ranges = _ranges(col_cum, max(self.nnz // (4 * self.nt), 1), 4 * self.nt) \
            or [(0, m)]
        self._T = None

    # -- block construction -------------------------------------------------

    def fill(self, a: int, b: int, indices: np.ndarray, data: np.ndarray, q0: int = 0):
        """Write the compact column indices and TF-logIDF values of output
        rows ``a:b`` to ``indices[q0:]`` / ``data[q0:]``."""
        k = _kernels()
        s0, s1 = int(self.sub_ptr[a]), int(self.sub_ptr[b])
        k["sub_fill"](a, b, self.rows, self.X.indptr, self.X.indices, self.colmap,
                      indices, q0)
        nloc = s1 - s0
        ind = indices[q0:q0 + nloc]
        # == tf_logidf_data on the materialised subset (data = 1.0f, so
        # data * inv_row == inv_row exactly)
        d = np.repeat(self.inv_row[a:b], np.diff(self.sub_ptr[a:b + 1]))
        g = self.idf[ind]
        np.multiply(d, g, out=d)
        del g
        st = self.scale_to
        if np.result_type(d, st) == np.float32:
            np.multiply(d, st, out=d)
            np.log1p(d, out=data[q0:q0 + nloc])
        else:
            data[q0:q0 + nloc] = np.log1p(d * st).astype(np.float32)

    def block(self, a: int, b: int, idx_dtype=np.int32):
        """CSR pieces ``(indptr, indices, data)`` of output rows ``a:b``."""
        s0, s1 = int(self.sub_ptr[a]), int(self.sub_ptr[b])
        indices = np.empty(s1 - s0, dtype=idx_dtype)
        data = np.empty(s1 - s0, dtype=np.float32)
        self.fill(a, b, indices, data)
        indptr = (self.sub_ptr[a:b + 1] - s0)
        if idx_dtype == np.int32 and s1 - s0 < 2**31:
            indptr = indptr.astype(np.int32)
        return indptr, indices, data

    def build_range(self, a: int, b: int, pool=None, idx_dtype=np.int32):
        """:meth:`block` of rows ``a:b`` built on all threads."""
        s0, s1 = int(self.sub_ptr[a]), int(self.sub_ptr[b])
        indices = np.empty(s1 - s0, dtype=idx_dtype)
        data = np.empty(s1 - s0, dtype=np.float32)
        sub = self._sub_blocks(a, b)
        if len(sub) <= 1 or pool is None:
            for x, y in sub:
                self.fill(x, y, indices, data, int(self.sub_ptr[x]) - s0)
        else:
            list(pool.map(lambda xy: self.fill(xy[0], xy[1], indices, data,
                                               int(self.sub_ptr[xy[0]]) - s0), sub))
        indptr = self.sub_ptr[a:b + 1] - s0
        if idx_dtype == np.int32 and s1 - s0 < 2**31:
            indptr = indptr.astype(np.int32)
        return indptr, indices, data

    def _sub_blocks(self, a, b):
        return [(x, y) for x, y in self.blocks if x >= a and y <= b] or [(a, b)]

    def _super_blocks(self, max_nnz: int):
        """Contiguous groups of :attr:`blocks` with at most ``max_nnz`` nnz."""
        out, cur, cur_nnz = [], None, 0
        for a, b in self.blocks:
            nz = int(self.sub_ptr[b] - self.sub_ptr[a])
            if cur is not None and cur_nnz + nz > max_nnz:
                out.append(cur)
                cur, cur_nnz = None, 0
            cur = (a, b) if cur is None else (cur[0], b)
            cur_nnz += nz
        if cur is not None:
            out.append(cur)
        return out

    # -- products -------------------------------------------------------------

    def matmul(self, Q: np.ndarray) -> np.ndarray:
        """``A @ Q`` (bit-identical to SciPy on the materialised matrix)."""
        Q = np.asarray(Q)
        vec = Q.ndim == 1
        if vec:
            Q = Q[:, None]
        dt = np.result_type(self.dtype, Q.dtype)
        out = np.empty((self.shape[0], Q.shape[1]), dtype=dt)

        def run(ab):
            a, b = ab
            ptr, ind, dat = self.block(a, b)
            blk = sp.csr_matrix((dat, ind, ptr), shape=(b - a, self.shape[1]), copy=False)
            out[a:b] = blk @ Q

        if self.nt <= 1 or len(self.blocks) <= 1:
            for ab in self.blocks:
                run(ab)
        else:
            with ThreadPoolExecutor(max_workers=self.nt) as pool:
                list(pool.map(run, self.blocks))
        if not self.blocks:
            out[:] = 0
        return out[:, 0] if vec else out

    def tmatmul(self, Q: np.ndarray) -> np.ndarray:
        """``A.T @ Q`` (bit-identical to SciPy's ``csc_matvecs``)."""
        k = _kernels()
        Q = np.asarray(Q)
        vec = Q.ndim == 1
        if vec:
            Q = Q[:, None]
        dt = np.result_type(self.dtype, Q.dtype)
        Q = np.ascontiguousarray(Q, dtype=dt)
        out = np.zeros((self.shape[1], Q.shape[1]), dtype=dt)
        k["t_matmul"](0, 0, np.zeros(2, np.int64), np.zeros(0, np.int32),
                      np.zeros(0, dt), Q[:1], out)          # JIT
        nt = self.nt
        with ThreadPoolExecutor(max_workers=nt) as pool:
            for a, b in self._super_blocks(max(nt, 1) * self._block_nnz):
                ptr, ind, dat = self.build_range(a, b, pool, idx_dtype=np.int32)
                ptr = ptr.astype(np.int64, copy=False)
                if dat.dtype != dt:
                    dat = dat.astype(dt)
                Qb = Q[a:b]
                if nt <= 1 or len(self._col_ranges) <= 1:
                    for c0, c1 in self._col_ranges:
                        k["t_matmul"](c0, c1, ptr, ind, dat, Qb, out)
                else:
                    list(pool.map(lambda cc: k["t_matmul"](cc[0], cc[1], ptr, ind, dat,
                                                           Qb, out), self._col_ranges))
                del ptr, ind, dat
        return out[:, 0] if vec else out

    def __matmul__(self, Q):
        return self.matmul(Q)

    def __rmatmul__(self, Y):
        # Y @ A == (A.T @ Y.T).T  (SciPy's __rmatmul__ on a CSR matrix)
        return self.tmatmul(np.asarray(Y).T).T

    @property
    def T(self):
        if self._T is None:
            self._T = _StreamT(self)
        return self._T

    def transpose(self, *args, **kwargs):
        return self.T

    def materialize(self) -> sp.csr_matrix:
        """The TF-logIDF matrix itself (tests / small inputs only)."""
        ptr, ind, dat = self.build_range(0, self.shape[0], None,
                                         idx_dtype=np.int64 if self.nnz >= 2**31 else np.int32)
        out = sp.csr_matrix((dat, ind, ptr), shape=self.shape)
        out.has_sorted_indices = True
        return out


class _StreamT:
    """Lazy transpose of a :class:`TfidfStream`."""

    __array_ufunc__ = None
    ndim = 2
    dtype = np.dtype(np.float32)

    def __init__(self, base: TfidfStream):
        self._base = base
        self.shape = (base.shape[1], base.shape[0])

    @property
    def T(self):
        return self._base

    def transpose(self, *args, **kwargs):
        return self._base

    def __matmul__(self, Q):
        return self._base.tmatmul(Q)

    def __rmatmul__(self, Y):
        # Y @ A.T == (A @ Y.T).T
        return self._base.matmul(np.asarray(Y).T).T


def randomized_svd_stream(op: TfidfStream, n_components: int, random_state: int = 0):
    """sklearn's randomized SVD (``n_iter="auto"``) on the operator; the same
    calls :func:`sklearn.utils.extmath.randomized_svd` makes after its input
    check (which only accepts arrays / sparse matrices)."""
    from sklearn.utils.extmath import _randomized_svd
    return _randomized_svd(op, n_components=n_components, random_state=random_state,
                           n_iter="auto")


# ---------------------------------------------------------------------------
# torch (GPU) twin
# ---------------------------------------------------------------------------

def gpu_budget(dev) -> int:
    """Device bytes the LSI operand may keep resident: ``EPIONE_GPU_MEMORY_GB``
    when set, else ``min(memory_budget(dev), 12 GiB)`` -- the operand is
    streamed beyond that, so the GPU is never filled on a shared card."""
    from ..core._gpu import capped_budget
    return capped_budget(dev, _GPU_CAP_GB)


class TorchTfidfStream:
    """Device operator over the row blocks of a :class:`TfidfStream`.

    Row blocks (int32 CSR + their device-side transposes) stay resident while
    they fit ``budget`` bytes; the remaining blocks are rebuilt on the host
    (bounded buffers) and uploaded for every product. Exposes ``shape``,
    ``dtype`` (torch), ``matmul(D)`` (``A @ D``) and ``rmatmul(D)``
    (``A.T @ D``) for :func:`epione.core._gpu_ops.randomized_svd`.
    """

    def __init__(self, stream: TfidfStream, dev, budget: Optional[int] = None):
        from ..core._gpu import get_torch
        torch = get_torch()
        self.torch = torch
        self.s = stream
        self.dev = dev
        self.dtype = torch.float32
        self.shape = stream.shape
        self.budget = int(gpu_budget(dev) if budget is None else budget)
        # A device block costs 8 B/nnz (float32 values + int32 indices) and
        # as much again for its transpose; building the transpose (torch's
        # CSR -> CSC goes through int64 COO indices and a sort) needs up to
        # ~48 B/nnz of scratch. Blocks are sized so that this transient stays
        # a fraction of the budget.
        scratch = 48
        max_blk = int(min(2**31 - 1, max(self.budget // (4 * scratch), 1 << 16)))
        self.blocks = stream._super_blocks(max_blk)
        self._res = {}
        used = 0
        self._pool = ThreadPoolExecutor(max_workers=stream.nt)
        for a, b in self.blocks:
            nz = int(stream.sub_ptr[b] - stream.sub_ptr[a])
            need = nz * 16 + (b - a + self.shape[1] + 2) * 8
            if used + need + nz * scratch > self.budget:
                break
            Xd = self._upload(a, b)
            XTd = _transpose(Xd)
            self._res[(a, b)] = (Xd, XTd)
            used += need
        self.resident = len(self._res) == len(self.blocks)
        from ..utils import console
        console.level2(
            f"[iterative_lsi]   device operand: {stream.nnz:,} nnz in {len(self.blocks)} "
            f"block(s), {len(self._res)} resident ({used / 2**30:.2f} GiB of "
            f"{self.budget / 2**30:.2f} GiB budget), rest streamed per product")

    def _upload(self, a, b):
        from ..core._gpu import _csr_tensor
        torch = self.torch
        ptr, ind, dat = self.s.build_range(a, b, self._pool, idx_dtype=np.int32)
        crow = torch.from_numpy(ptr.astype(np.int32, copy=False)).to(self.dev)
        col = torch.from_numpy(ind).to(self.dev)
        val = torch.from_numpy(dat).to(self.dev)
        del ptr, ind, dat
        return _csr_tensor(crow, col, val, (b - a, self.shape[1]), self.dev)

    def _iter(self, transposed: bool):
        for a, b in self.blocks:
            r = self._res.get((a, b))
            if r is not None:
                yield a, b, (r[1] if transposed else r[0])
            else:
                Xd = self._upload(a, b)
                yield a, b, (_transpose(Xd) if transposed else Xd)
                del Xd

    def matmul(self, D):
        out = self.torch.empty((self.shape[0], D.shape[1]), dtype=D.dtype, device=self.dev)
        for a, b, Xd in self._iter(False):
            out[a:b] = Xd @ D
        return out

    def rmatmul(self, D):
        out = self.torch.zeros((self.shape[1], D.shape[1]), dtype=D.dtype, device=self.dev)
        for a, b, XTd in self._iter(True):
            out += XTd @ D[a:b]
        return out

    def close(self):
        self._res.clear()
        self._pool.shutdown(wait=False)


def _transpose(T):
    from ..core._gpu import _transpose_csr
    return _transpose_csr(T)
