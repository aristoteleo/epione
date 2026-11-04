"""CUDA path of BiRMT-2's matrix-free level-0 fit (``epi.tl.birmt(device='cuda')``).

Only the sparse products move to the GPU; every dense step (Sinkhorn
updates and stopping rule, the subsample Gram matrix and ``α``, the block
Gram-Schmidt / QR, the Rayleigh-Ritz eigenproblem, shrinkage, the
out-of-sample extension) runs on the host with exactly the CPU code, in
float64. The sparse products are the CPU kernels' pattern sums
(:mod:`epione.tl._birmt2` ``pat_sum`` / ``pat_mm`` and their remapped
``_r`` twins), runtime-compiled with NVRTC (:mod:`epione.core._cuda_rt`).
Each row is cut into segments of ``_SEG`` = 1,024 stored entries; a warp
adds a segment's terms in the stored (CPU) order and the segment sums of a
row are then added in order. The result is deterministic and independent of
the device budget / block layout; rows with at most 1,024 entries give the
CPU's bits, longer rows (cells of the tile matrix) differ from the CPU sum
only by that regrouping (~1e-16 relative), so the MP rank, ``α`` and the
embedding agree with the CPU to float64 round-off.

Memory. The device keeps the int32 index arrays of as many row blocks of the
pattern (CSR, cells) and of its transpose (CSC, features) as fit the device
budget (``EPIONE_GPU_MEMORY_GB`` / ``gpu_memory_gb``; default
``min(12 GiB, half the free memory)``) next to the dense working set (one
``n_features x krylov_block`` float64 block, the cell-side Krylov blocks and
small output tiles). The remaining blocks are streamed host -> device for
every product from the host's own arrays, page-locked in place (no copy),
with the copy of block ``k+1`` overlapping the kernels on block ``k`` (two
staging slots). Outputs of feature-side products that go back to the host
(``rmatmat``) are written tile by tile into the (page-locked) host result.
The host holds exactly the arrays of the CPU path (the compact pattern or the
memory-lean remapped view of the input, and the transpose).
"""
from __future__ import annotations

import ctypes
from typing import List, Optional, Tuple

import numpy as np

from ..core import console

_SRC = r"""
// Pattern products out[i, c] = sum_{p in row i} X[col(p), c] in a fixed,
// layout-independent order: every row is cut into segments of S stored
// entries (counted from the row start); one warp sums a segment in the
// stored order, s = ((0 + v_1) + v_2) + ..., exactly like the sequential
// numba kernels, and the segment sums of a row are then added in order. A
// row with at most S entries therefore gives the CPU's bits; a longer row
// differs from the CPU sum only by the regrouping at segment boundaries.
// Dropped entries (column remap < 0) contribute +0.0 (a sum that starts at
// +0.0 never becomes -0.0, so +0.0 leaves it unchanged).
#define FULL 0xffffffffu
#define DEF(NAME, IT, REMAP)                                                         \
extern "C" __global__ void NAME(const long long* __restrict__ indptr,                \
    const long long* __restrict__ segptr, const IT* __restrict__ idx,                \
    const int* __restrict__ remap, const double* __restrict__ X,                     \
    double* __restrict__ part, long long ra, long long re, long long g0,             \
    long long nseg, long long p0, int b, int S) {                                    \
  const long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;       \
  const int lane = threadIdx.x & 31;                                                 \
  const int nch = (b + 31) >> 5;                                                     \
  const long long sg = w / nch;                                                      \
  if (sg >= nseg) return;                                                            \
  const int c = (int)(w - sg * nch) * 32 + lane;                                     \
  const long long g = g0 + sg;                                                       \
  long long lo = ra, hi = re - 1;            /* row i: segptr[i] <= g < segptr[i+1] */ \
  while (lo < hi) {                                                                  \
    const long long mid = (lo + hi + 1) >> 1;                                        \
    if (segptr[mid] <= g) lo = mid; else hi = mid - 1;                               \
  }                                                                                  \
  const long long i = lo;                                                            \
  const long long a = indptr[i] - p0 + (g - segptr[i]) * (long long)S;               \
  const long long e = min(a + (long long)S, indptr[i + 1] - p0);                     \
  if (b == 1) {                                                                      \
    double s = 0.0;                                                                  \
    for (long long base = a; base < e; base += 32) {                                 \
      const long long p = base + lane;                                               \
      double v = 0.0;                                                                \
      if (p < e) {                                                                   \
        long long j = (long long)idx[p];                                             \
        if (REMAP) j = (long long)remap[j];                                          \
        if (j >= 0) v = X[j];                                                        \
      }                                                                              \
      const int cnt = (int)min((long long)32, e - base);                             \
      if (cnt == 32) {                                                               \
        _Pragma("unroll")                                                            \
        for (int k = 0; k < 32; ++k) s += __shfl_sync(FULL, v, k);                   \
      } else {                                                                       \
        for (int k = 0; k < cnt; ++k) s += __shfl_sync(FULL, v, k);                  \
      }                                                                              \
    }                                                                                \
    if (lane == 0) part[sg] = s;                                                     \
    return;                                                                          \
  }                                                                                  \
  const bool on = c < b;                                                             \
  double s = 0.0;                                                                    \
  for (long long base = a; base < e; base += 32) {                                   \
    const long long p = base + lane;                                                 \
    long long jl = -1;                                                               \
    if (p < e) {                                                                     \
      jl = (long long)idx[p];                                                        \
      if (REMAP) jl = (long long)remap[jl];                                          \
    }                                                                                \
    const int cnt = (int)min((long long)32, e - base);                               \
    if (cnt == 32) {                                                                 \
      double v[32];                                                                  \
      _Pragma("unroll")                                                              \
      for (int k = 0; k < 32; ++k) {                                                 \
        const long long j = __shfl_sync(FULL, jl, k);                                \
        v[k] = (on && j >= 0) ? X[j * b + c] : 0.0;                                  \
      }                                                                              \
      _Pragma("unroll")                                                              \
      for (int k = 0; k < 32; ++k) s += v[k];                                        \
    } else {                                                                         \
      for (int k = 0; k < cnt; ++k) {                                                \
        const long long j = __shfl_sync(FULL, jl, k);                                \
        s += (on && j >= 0) ? X[j * b + c] : 0.0;                                    \
      }                                                                              \
    }                                                                                \
  }                                                                                  \
  if (on) part[sg * b + c] = s;                                                      \
}
DEF(pmm_i32, int, 0)
DEF(pmm_i64, long long, 0)
DEF(pmm_i32_r, int, 1)
DEF(pmm_i64_r, long long, 1)

// out[i, c] = sum of the row's segment sums, in order
extern "C" __global__ void seg_combine(const long long* __restrict__ segptr,
    const double* __restrict__ part, double* __restrict__ out, long long ra,
    long long nr, long long g0, long long out_r0, int b) {
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long i = ra + t / b;
  if (t >= nr * (long long)b) return;
  const int c = (int)(t % b);
  double s = 0.0;
  for (long long g = segptr[i]; g < segptr[i + 1]; ++g) s += part[(g - g0) * b + c];
  out[(i - out_r0) * (long long)b + c] = s;
}
"""
_NAMES = ["pmm_i32", "pmm_i64", "pmm_i32_r", "pmm_i64_r", "seg_combine"]
_THREADS = 256
_GPU_CAP_GB = 12.0
# stored entries per row block (streaming granularity) and an optional cap on
# the resident index bytes (tests force streaming with it)
_BLOCK_NNZ = (1 << 22, 1 << 26)
_RESIDENT_CAP: Optional[int] = None
#: stored entries per summation segment (rows up to this length are summed in
#: exactly the CPU order)
_SEG = 1024


def _ptr(t) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(t.data_ptr()) if t is not None else 0)


def gpu_unavailable_reason(dev) -> Optional[str]:
    """``None`` when the CUDA kernels can run on ``dev``, else why not."""
    from ..core import _cuda_rt
    if not _cuda_rt.available(dev):
        return f"CUDA runtime compilation unavailable ({_cuda_rt.last_error()})"
    return None


class _Orientation:
    """One orientation (CSR of ``Y`` or of ``Yᵀ``) of the binary pattern as
    row blocks of at most ``blk_nnz`` stored entries, each either resident on
    the device (int32 indices) or streamed from the host array."""

    def __init__(self, indptr, indices, n_rows: int, remap, dev, blk_nnz: int):
        import torch
        self.torch = torch
        self.dev = dev
        self.n_rows = int(n_rows)
        self.h_ind = indices
        self.itemsize = int(indices.dtype.itemsize)
        if indices.dtype not in (np.int32, np.int64):
            raise TypeError(f"unsupported index dtype {indices.dtype}")
        ip = np.asarray(indptr, dtype=np.int64)
        self.indptr = torch.from_numpy(ip).to(dev)
        self.S = int(_SEG)
        sp_h = np.zeros(len(ip), dtype=np.int64)
        np.cumsum((np.diff(ip) + self.S - 1) // self.S, out=sp_h[1:])
        self.segptr_h = sp_h
        self.segptr = torch.from_numpy(sp_h).to(dev)
        self.remap = None if remap is None else torch.from_numpy(
            np.asarray(remap, dtype=np.int32)).to(dev)
        from ..core._gpu import csr_row_blocks
        self.blocks: List[Tuple[int, int, int, int]] = []
        for s in csr_row_blocks(ip, blk_nnz):
            self.blocks.append((s.start, s.stop, int(ip[s.start]), int(ip[s.stop])))
        self.resident = {}
        self.nnz = int(ip[-1])
        self._reg = None

    def block_nnz(self, k):
        return self.blocks[k][3] - self.blocks[k][2]

    def kernel(self, resident: bool):
        if self.remap is None:
            return "pmm_i32" if (resident or self.itemsize == 4) else "pmm_i64"
        return "pmm_i32_r" if (resident or self.itemsize == 4) else "pmm_i64_r"

    def make_resident(self, k):
        torch = self.torch
        _, _, p0, p1 = self.blocks[k]
        h = torch.from_numpy(self.h_ind[p0:p1])
        self.resident[k] = h.to(self.dev).to(torch.int32) if self.itemsize == 8 else h.to(self.dev)

    def register_tail(self, cuda_rt):
        """Page-lock the host index range of the streamed blocks (in place)."""
        streamed = [k for k in range(len(self.blocks)) if k not in self.resident]
        if not streamed:
            return
        p0 = min(self.blocks[k][2] for k in streamed)
        p1 = max(self.blocks[k][3] for k in streamed)
        ptr = int(self.h_ind.ctypes.data) + p0 * self.itemsize
        if cuda_rt.host_register(ptr, (p1 - p0) * self.itemsize):
            self._reg = ptr

    def close(self, cuda_rt):
        if self._reg is not None:
            cuda_rt.host_unregister(self._reg)
            self._reg = None
        self.resident.clear()
        self.indptr = self.remap = self.segptr = None


class CudaBinaryBiwhitened:
    """GPU twin of :class:`epione.tl._birmt2.BinaryBiwhitened` (same
    ``matmat`` / ``rmatmat`` results up to the segment regrouping, see the
    module docstring) that also runs the Sinkhorn passes. Host arrays in,
    host arrays out."""

    def __init__(self, Y, Yt, dev, *, budget: Optional[int] = None, block: int = 16,
                 k_max: int = 128):
        import torch
        from ..core import _cuda_rt
        from ..core._gpu import capped_budget
        self.torch = torch
        self.rt = _cuda_rt
        self.dev = dev
        self.mod = _cuda_rt.Module.get(_SRC, _NAMES, dev)
        self.shape = tuple(Y.shape)
        m, n = self.shape
        self.remap = getattr(Y, "remap", None)
        self.budget = int(capped_budget(dev, _GPU_CAP_GB) if budget is None else budget)
        torch.cuda.reset_peak_memory_stats(dev)
        self._base_alloc = torch.cuda.memory_allocated(dev)
        isz = max(int(Y.indices.dtype.itemsize), int(Yt.indices.dtype.itemsize))
        # dense working set: one feature-side block (n x block), cell-side
        # Krylov blocks, the Rayleigh-Ritz operand (m x L) and output tiles
        L = int(2 * k_max + 2 * block)
        self._tile = int(min(64 << 20, max(self.budget // 32, 1 << 20)))
        work = 8 * (n * block + 4 * m * block + m * L + 2 * (m + n)) + 3 * self._tile + (16 << 20)
        fixed = 16 * (m + n + 2) + (0 if self.remap is None else 4 * len(self.remap))
        blk = int(min(max(self.budget // 48 // isz, _BLOCK_NNZ[0]), _BLOCK_NNZ[1]))
        slots = 2 * blk * isz
        room = self.budget - work - fixed - slots
        if _RESIDENT_CAP is not None:
            room = min(room, int(_RESIDENT_CAP))
        if room < -slots // 2:
            raise MemoryError(f"GPU budget {self.budget / 2**30:.2f} GiB is below the "
                              f"BiRMT-2 working set ({(work + fixed + slots) / 2**30:.2f} GiB)")
        self.R = _Orientation(Y.indptr, Y.indices, m, self.remap, dev, blk)
        self.C = _Orientation(Yt.indptr, Yt.indices, n, None, dev, blk)
        # resident blocks, alternating the two orientations (each Sinkhorn
        # iteration and each Krylov step reads both once)
        used = 0
        order = [(o, k) for kk in range(max(len(self.R.blocks), len(self.C.blocks)))
                 for o in (self.R, self.C) if kk < len(o.blocks) for k in (kk,)]
        for o, k in order:
            need = 4 * o.block_nnz(k)
            if used + need > room:
                continue
            o.make_resident(k)
            used += need
        self.R.register_tail(_cuda_rt)
        self.C.register_tail(_cuda_rt)
        self._slots = [torch.empty(blk * isz, dtype=torch.uint8, device=dev) for _ in range(2)]
        self._part = torch.empty(self._tile // 8, dtype=torch.float64, device=dev)
        self._copy = torch.cuda.Stream(dev)
        self._ready = [torch.cuda.Event() for _ in range(2)]
        self._free = [torch.cuda.Event() for _ in range(2)]
        self._used = [False, False]
        self.sx = self.sy = None
        self._sy_d = None
        nres = len(self.R.resident) + len(self.C.resident)
        ntot = len(self.R.blocks) + len(self.C.blocks)
        self.resident_bytes = used
        console.level2(
            f"[birmt2] GPU operand: {self.R.nnz + self.C.nnz:,} indices in {ntot} blocks, "
            f"{nres} resident ({used / 2**30:.2f} GiB of {self.budget / 2**30:.2f} GiB budget), "
            f"rest streamed (page-locked: R={self.R._reg is not None} C={self.C._reg is not None})")

    # -- core product -------------------------------------------------------
    def _launch(self, o: _Orientation, name, idx_t, X, out, r0, r1, p0, out_r0, b):
        """``out[i - out_r0] = (P_o @ X)[i]`` for rows ``r0 <= i < r1`` of a block
        whose stored entries start at ``p0`` (``idx_t``)."""
        sp_h = o.segptr_h
        cap = max(1, self._part.numel() // b)
        a = r0
        while a < r1:
            # rows a:e whose segments fit the partial-sum buffer
            e = int(np.searchsorted(sp_h, sp_h[a] + cap, side="right")) - 1
            e = min(max(e, a + 1), r1)
            g0, nseg = int(sp_h[a]), int(sp_h[e] - sp_h[a])
            if nseg > cap:
                raise RuntimeError("segment buffer too small")   # one row > cap segments
            if nseg:
                nch = (b + 31) // 32
                grid = (nseg * nch * 32 + _THREADS - 1) // _THREADS
                self.mod.launch(name, grid, _THREADS, [
                    _ptr(o.indptr), _ptr(o.segptr), _ptr(idx_t), _ptr(o.remap), _ptr(X),
                    _ptr(self._part), ctypes.c_longlong(a), ctypes.c_longlong(e),
                    ctypes.c_longlong(g0), ctypes.c_longlong(nseg), ctypes.c_longlong(p0),
                    ctypes.c_int(b), ctypes.c_int(o.S)])
            nt = (e - a) * b
            self.mod.launch("seg_combine", (nt + _THREADS - 1) // _THREADS, _THREADS, [
                _ptr(o.segptr), _ptr(self._part), _ptr(out), ctypes.c_longlong(a),
                ctypes.c_longlong(e - a), ctypes.c_longlong(g0), ctypes.c_longlong(out_r0),
                ctypes.c_int(b)])
            a = e

    def _blocks(self, o: _Orientation):
        """Yield ``(k, idx_tensor, kernel)`` for every block of ``o``, resident
        first; streamed blocks arrive through the two staging slots with the
        next copy overlapping the current kernel. The caller must enqueue its
        work on the block before advancing the generator."""
        torch = self.torch
        comp = torch.cuda.current_stream(self.dev)
        todo = [k for k in range(len(o.blocks)) if k not in o.resident]
        isz = o.itemsize
        dt = torch.int32 if isz == 4 else torch.int64

        def issue(t):
            s = t % 2
            _, _, p0, p1 = o.blocks[todo[t]]
            if self._used[s]:
                self._copy.wait_event(self._free[s])
            self.rt.memcpy_htod_async(int(self._slots[s].data_ptr()),
                                      int(o.h_ind.ctypes.data) + p0 * isz, (p1 - p0) * isz,
                                      self._copy)
            self._ready[s].record(self._copy)

        # the first two copies run while the resident blocks are computed
        for t in range(min(2, len(todo))):
            issue(t)
        for k, t in o.resident.items():
            yield k, t, o.kernel(True)
        for t, k in enumerate(todo):
            s = t % 2
            comp.wait_event(self._ready[s])
            nnz = o.block_nnz(k)
            yield k, self._slots[s][:nnz * isz].view(dt), o.kernel(False)
            self._free[s].record(comp)
            self._used[s] = True
            if t + 2 < len(todo):
                issue(t + 2)

    def _apply(self, o: _Orientation, X, out):
        """``out = P_o @ X`` on the device (out rows = rows of ``o``)."""
        b = int(X.shape[1])
        for k, idx_t, name in self._blocks(o):
            r0, r1, p0, _ = o.blocks[k]
            self._launch(o, name, idx_t, X, out, r0, r1, p0, 0, b)
        return out

    def _apply_to_host(self, o: _Orientation, X, out_h, row_scale=None):
        """``out_h = P_o @ X`` streamed to the host in row tiles (no device
        array of the full output); optional per-row scale (a device column,
        applied to each tile before the copy: the same IEEE products as
        scaling on the host).
        ``out_h`` is page-locked for the duration, so the tile copies are
        asynchronous DMA on the compute stream."""
        torch = self.torch
        b = int(X.shape[1])
        assert out_h.flags.c_contiguous and out_h.dtype == np.float64
        rows_tile = max(1, self._tile // (8 * b))
        tiles = [torch.empty((rows_tile, b), dtype=torch.float64, device=self.dev) for _ in range(2)]
        comp = torch.cuda.current_stream(self.dev)
        base = int(out_h.ctypes.data)
        reg = self.rt.host_register(base, out_h.nbytes)
        done = [torch.cuda.Event(), torch.cuda.Event()]
        used = [False, False]
        t = 0
        try:
            for k, idx_t, name in self._blocks(o):
                r0, r1, p0, _ = o.blocks[k]
                for a in range(r0, r1, rows_tile):
                    e = min(a + rows_tile, r1)
                    s = t % 2
                    t += 1
                    if not reg and used[s]:
                        done[s].synchronize()
                    self._launch(o, name, idx_t, X, tiles[s], a, e, p0, a, b)
                    if row_scale is not None:
                        tiles[s][:e - a].mul_(row_scale[a:e])
                    if reg:
                        self.rt.memcpy_dtoh_async(base + a * b * 8, int(tiles[s].data_ptr()),
                                                  (e - a) * b * 8, comp)
                    else:
                        out_h[a:e] = tiles[s][:e - a].cpu().numpy()
                    done[s].record(comp)
                    used[s] = True
            comp.synchronize()
        finally:
            if reg:
                comp.synchronize()
                self.rt.host_unregister(base)
        return out_h

    def _dev(self, A):
        return self.torch.from_numpy(np.ascontiguousarray(A, dtype=np.float64)).to(self.dev)

    # -- Sinkhorn -----------------------------------------------------------
    def row_sums(self, v: np.ndarray) -> np.ndarray:
        out = self.torch.empty((self.shape[0], 1), dtype=self.torch.float64, device=self.dev)
        return self._apply(self.R, self._dev(v[:, None]), out)[:, 0].cpu().numpy()

    def col_sums(self, v: np.ndarray) -> np.ndarray:
        out = self.torch.empty((self.shape[1], 1), dtype=self.torch.float64, device=self.dev)
        return self._apply(self.C, self._dev(v[:, None]), out)[:, 0].cpu().numpy()

    def sinkhorn(self, *, tol: float = 1e-6, max_iter: int = 10000):
        """:func:`epione.tl._birmt2.sinkhorn_binary` with the pattern sums on
        the device (the updates and the stopping rule are the host code)."""
        m, n = self.shape
        x = np.ones(m)
        it = 0
        cs = self.col_sums(x)
        for it in range(1, max_iter + 1):
            y = m / cs
            x = n / self.row_sums(y)
            cs = self.col_sums(x)
            if np.max(np.abs(y * cs / m - 1.0)) < tol:
                break
        else:
            console.warn(f"[birmt2] Sinkhorn did not reach tol={tol} in {max_iter} iterations")
        return x, y, it

    def set_scaling(self, x, y):
        self.sx = np.sqrt(np.asarray(x, np.float64))
        self.sy = np.sqrt(np.asarray(y, np.float64))
        self._sy_d = self._dev(self.sy[:, None])

    # -- BinaryBiwhitened interface ------------------------------------------
    def matmat(self, X):
        """``Ŷ X`` (host in / out)."""
        X = np.ascontiguousarray(np.asarray(X, np.float64) * self.sy[:, None])
        Xd = self._dev(X)
        del X
        out = self.torch.empty((self.shape[0], Xd.shape[1]), dtype=self.torch.float64,
                               device=self.dev)
        self._apply(self.R, Xd, out)
        del Xd
        out = out.cpu().numpy()
        out *= self.sx[:, None]
        return out

    def rmatmat(self, X):
        """``Ŷᵀ X`` (host in / out; the n x b output is written to the host
        tile by tile)."""
        X = np.ascontiguousarray(np.asarray(X, np.float64) * self.sx[:, None])
        Xd = self._dev(X)
        del X
        out = np.empty((self.shape[1], Xd.shape[1]))
        self._apply_to_host(self.C, Xd, out, row_scale=self._sy_d)
        return out

    def normal_matmat(self, Z):
        """``Ŷ (Ŷᵀ Z)`` = ``matmat(rmatmat(Z))`` with the feature-side
        intermediate kept on the device (the same operations)."""
        torch = self.torch
        X = np.ascontiguousarray(np.asarray(Z, np.float64) * self.sx[:, None])
        Xd = self._dev(X)
        T = torch.empty((self.shape[1], Xd.shape[1]), dtype=torch.float64, device=self.dev)
        self._apply(self.C, Xd, T)
        T.mul_(self._sy_d)            # rmatmat: out *= sy
        T.mul_(self._sy_d)            # matmat: X * sy
        out = torch.empty((self.shape[0], Xd.shape[1]), dtype=torch.float64, device=self.dev)
        del Xd
        self._apply(self.R, T, out)
        del T
        out = out.cpu().numpy()
        out *= self.sx[:, None]
        return out

    def peak_bytes(self) -> int:
        return int(self.torch.cuda.max_memory_allocated(self.dev) - self._base_alloc)

    def close(self):
        if getattr(self, "R", None) is None:
            return
        self.torch.cuda.synchronize(self.dev)
        self.R.close(self.rt)
        self.C.close(self.rt)
        self.R = self.C = None
        self._slots = self._part = None
        self._sy_d = None
        from ..core._gpu import empty_cache
        empty_cache(self.dev)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
