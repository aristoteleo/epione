"""BiRMT-2: fast biwhitened random-matrix embedding with cell-score optimal
shrinkage and Markov-time-1 clustering for scATAC-seq.

BiRMT-2 keeps the estimator chain of BiRMT v1 (:mod:`epione.tl._birmt`) and
changes three things (default ``epi.tl.birmt(version=2)``):

1. **Matrix-free biwhitening and block Krylov SVD.** For the binary cells x
   peaks matrix ``Y`` the biwhitened matrix is ``Ŷ = D(√x) Y D(√y)`` with the
   Sinkhorn scalings of Landa, Zhang & Kluger (2022, Algorithm 2). Products
   with ``Ŷ`` and ``Ŷᵀ`` are taken on the CSR / CSC index arrays of ``Y``
   (numba, no value arrays). The noise scale ``α`` is v1's Eq. (24) estimate
   on 2,000 random cells (the subsample Gram matrix is formed by a pattern
   kernel). The singular values are computed by randomized block Krylov
   Rayleigh-Ritz (Musco & Musco 2015), which resolves the values just above
   the MP edge ``√α(√m+√n)`` more accurately than v1's subspace iteration at
   a fraction of the cost (on PBMC 3k v1 reported rank 19 while ARPACK gives
   30 values above the edge; components that close to the edge receive
   near-zero weight under either shrinker below).
2. **Cell-score optimal shrinkage** (:func:`score_shrinkage`). In the
   spiked model of Gavish & Donoho (2017) / Benaych-Georges & Nadakuditi
   (2012) a singular value ``y`` above the edge has true value ``x(y)`` and
   singular-vector cosines ``c`` (cell side) and ``c̃`` (feature side). The
   Frobenius shrinker ``x c c̃`` is optimal for the whole matrix; a cell
   embedding only needs the cell scores ``x u_true``, whose MSE-optimal
   scalar estimate from ``u_emp`` is ``x c``. The embedding is
   ``Z = U diag(x c)`` divided per cell by ``√x_i · depth_i`` (v1 step 2).
3. **Markov-time-1 clustering.** One Leiden run (igraph, modularity, to
   convergence) on v1's doubly-stochastic Gaussian kernel ``W`` (Landa,
   Coifman & Kluger 2021; exact kNN, Sinkhorn capped at 1,000 iterations).
   Because ``W`` has unit node strengths, resolution 1 is the linearised
   Markov stability at Markov time 1 (Delvenne, Yaliraki & Barahona 2010;
   Lambiotte et al. 2014). It replaces v1's 250-run resolution scan, which
   was the runtime bottleneck and under-clustered; the scan stays available
   (``cluster_method='markov_stability'``).

Optional components that were evaluated on the development data and did not
help (kept for reproducibility, off by default):

* ``tile_sizes``: multiscale tile pyramid on the peak matrix. Level ``l``
  aggregates peaks into genomic tiles of size ``T_l`` by midpoint
  (``Y_l = Y_0 A_l``, Haar approximation coefficients of the peak track;
  Mallat 1989), binarised (default) or as counts with the quadratic variance
  biwhitening ``V̂ = α[(1-β)Y + βY²]`` of Landa et al. 2022 Eq. (23), ``β``
  chosen on a 2,000-cell submatrix by the KS distance to MP (their Sec. 5.1).
  Each level is denoised separately (stacking before denoising would give
  correlated noise) and the per-level coordinates are concatenated with
  weights ``‖S̃_l‖_F``.
* ``seq_lambda``: DNA-sequence prior. A peak-peak cosine kNN graph of
  canonical k-mer profiles (GC regressed out) gives a normalised Laplacian
  ``L``; the denoised level-0 matrix is regularised on it,
  ``X* = argmin_X ½‖X - X̂‖² + (λ/2) tr(X L Xᵀ) = X̂(I+λL)⁻¹`` (Kalofolias et
  al. 2014; the V-step of GRALS, Rao et al. 2015), i.e. the cell coordinates
  become ``U S̃ (Vᵀ(I+λL)⁻²V)^½``.

References
----------
Landa B, Zhang TTCK, Kluger Y (2022) Biwhitening reveals the rank of a count
matrix. SIAM J Math Data Sci 4(4):1420-1446.
Gavish M, Donoho DL (2017) Optimal shrinkage of singular values. IEEE Trans
Inf Theory 63(4):2137-2152.
Benaych-Georges F, Nadakuditi RR (2012) The singular values and vectors of
low rank perturbations of large rectangular random matrices. J Multivar Anal
111:120-135.
Musco C, Musco C (2015) Randomized block Krylov methods for stronger and
faster approximate singular value decomposition. NeurIPS 28.
Landa B, Coifman RR, Kluger Y (2021) Doubly stochastic normalization of the
Gaussian kernel is robust to heteroskedastic noise. SIAM J Math Data Sci
3(1):388-413.
Delvenne J-C, Yaliraki SN, Barahona M (2010) Stability of graph communities
across time scales. PNAS 107(29):12755-12760.
Lambiotte R, Delvenne J-C, Barahona M (2014) Random walks, Markov processes
and the multiscale modular organization of complex networks. IEEE Trans Netw
Sci Eng 1(2):76-90.
Traag VA, Waltman L, van Eck NJ (2019) From Louvain to Leiden: guaranteeing
well-connected communities. Sci Rep 9:5233.
Mallat SG (1989) A theory for multiresolution signal decomposition: the
wavelet representation. IEEE Trans Pattern Anal Mach Intell 11(7):674-693.
Kalofolias V, Bresson X, Bronstein M, Vandergheynst P (2014) Matrix completion
on graphs. arXiv:1408.1717.
Rao N, Yu H-F, Ravikumar P, Dhillon IS (2015) Collaborative filtering with
graph information: consistency and scalable methods. NeurIPS 28.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from ..core import console
from . import _birmt as B1


# ---------------------------------------------------------------------------
# fast matrix-free kernels on the binary pattern (numba, cached)
# ---------------------------------------------------------------------------
# For a binary Y the biwhitened matrix is Ŷ = D(√x) Y D(√y): products with Ŷ
# and Ŷᵀ are computed from the sparsity pattern of Y (CSR and CSC index arrays,
# no value arrays) with the scalings applied to the dense operands, which
# halves the memory traffic of every pass and needs no reduction buffers.

_FK: dict = {}


def _fast_kernels():
    if _FK:
        return _FK
    import numba as nb

    @nb.njit(parallel=True, cache=True)
    def pat_sum(indptr, indices, v):
        n = indptr.shape[0] - 1
        out = np.empty(n, dtype=np.float64)
        for i in nb.prange(n):
            s = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                s += v[indices[p]]
            out[i] = s
        return out

    @nb.njit(parallel=True, cache=True)
    def pat_mm(indptr, indices, Xin, out):
        # out[i, :] = sum_{p in row i} Xin[indices[p], :]
        n = indptr.shape[0] - 1
        b = Xin.shape[1]
        for i in nb.prange(n):
            for c in range(b):
                out[i, c] = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                for c in range(b):
                    out[i, c] += Xin[j, c]

    @nb.njit(cache=True)
    def pat_transpose(indptr, indices, n_cols):
        # CSR pattern -> CSR pattern of the transpose (counting sort; rows sorted)
        n_rows = indptr.shape[0] - 1
        cnt = np.zeros(n_cols + 1, dtype=np.int64)
        for p in range(indices.shape[0]):
            cnt[indices[p] + 1] += 1
        tp = np.cumsum(cnt)
        pos = tp[:-1].copy()
        ti = np.empty(indices.shape[0], dtype=np.int32)
        for i in range(n_rows):
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                ti[pos[j]] = i
                pos[j] += 1
        return tp, ti

    @nb.njit(parallel=True, cache=True)
    def pat_gram(rp, ri, cp, ci, w_row, w_col, n_rows):
        # G = S S^T for S = D(w_row) P D(w_col), P a binary pattern given as CSR (rp, ri)
        # and CSC (cp, ci) restricted to the same n_rows rows
        G = np.zeros((n_rows, n_rows))
        for i in nb.prange(n_rows):
            for p in range(rp[i], rp[i + 1]):
                j = ri[p]
                wj = w_col[j] * w_col[j]
                for q in range(cp[j], cp[j + 1]):
                    G[i, ci[q]] += wj
            for k in range(n_rows):
                G[i, k] *= w_row[i] * w_row[k]
        return G

    @nb.njit(parallel=True, cache=True)
    def pat_filter(indptr, indices, remap):
        # keep the entries whose column maps to >= 0, relabelled (rows keep their order)
        n = indptr.shape[0] - 1
        cnt = np.zeros(n + 1, dtype=np.int64)
        for i in nb.prange(n):
            c = 0
            for p in range(indptr[i], indptr[i + 1]):
                if remap[indices[p]] >= 0:
                    c += 1
            cnt[i + 1] = c
        newp = np.cumsum(cnt)
        newi = np.empty(newp[-1], dtype=np.int32)
        for i in nb.prange(n):
            q = newp[i]
            for p in range(indptr[i], indptr[i + 1]):
                j = remap[indices[p]]
                if j >= 0:
                    newi[q] = j
                    q += 1
        return newp, newi

    @nb.njit(parallel=True, cache=True)
    def pat_transpose_par(indptr, indices, n_cols, n_chunks):
        # pat_transpose on n_chunks contiguous row ranges: per-chunk column
        # counts give every (chunk, column) its write offset, so each column
        # lists its rows in increasing order -- the same arrays as the
        # sequential counting sort.
        n_rows = indptr.shape[0] - 1
        bounds = np.empty(n_chunks + 1, dtype=np.int64)
        tot = indptr[n_rows]
        r = 0
        for c in range(n_chunks + 1):
            target = (tot * c) // n_chunks
            while r < n_rows and indptr[r] < target:
                r += 1
            bounds[c] = r
        bounds[n_chunks] = n_rows
        cnt = np.zeros((n_chunks, n_cols), dtype=np.int64)
        for c in nb.prange(n_chunks):
            for i in range(bounds[c], bounds[c + 1]):
                for p in range(indptr[i], indptr[i + 1]):
                    cnt[c, indices[p]] += 1
        tp = np.zeros(n_cols + 1, dtype=np.int64)
        for j in range(n_cols):
            tp[j + 1] = tp[j]
            for c in range(n_chunks):
                tp[j + 1] += cnt[c, j]
        # cnt[c, j] <- write position of chunk c in column j
        for j in nb.prange(n_cols):
            pos = tp[j]
            for c in range(n_chunks):
                k = cnt[c, j]
                cnt[c, j] = pos
                pos += k
        ti = np.empty(indices.shape[0], dtype=np.int32)
        for c in nb.prange(n_chunks):
            for i in range(bounds[c], bounds[c + 1]):
                for p in range(indptr[i], indptr[i + 1]):
                    j = indices[p]
                    ti[cnt[c, j]] = i
                    cnt[c, j] += 1
        return tp, ti

    @nb.njit(parallel=True, cache=True)
    def pat_select(indptr, indices, rows, remap, use_remap):
        # rows `rows` (in that order) with columns relabelled by remap (>= 0
        # kept), entry order preserved -- SciPy's Y[rows][:, cols] pattern
        n = rows.shape[0]
        cnt = np.zeros(n + 1, dtype=np.int64)
        for o in nb.prange(n):
            i = rows[o]
            if use_remap:
                c = 0
                for p in range(indptr[i], indptr[i + 1]):
                    if remap[indices[p]] >= 0:
                        c += 1
                cnt[o + 1] = c
            else:
                cnt[o + 1] = indptr[i + 1] - indptr[i]
        newp = np.cumsum(cnt)
        newi = np.empty(newp[-1], dtype=np.int32)
        for o in nb.prange(n):
            i = rows[o]
            q = newp[o]
            for p in range(indptr[i], indptr[i + 1]):
                j = remap[indices[p]] if use_remap else indices[p]
                if j >= 0:
                    newi[q] = j
                    q += 1
        return newp, newi

    @nb.njit(parallel=True, cache=True)
    def col_count_par(indices, n_cols, n_chunks):
        n = indices.shape[0]
        part = np.zeros((n_chunks, n_cols), dtype=np.int64)
        for c in nb.prange(n_chunks):
            a = (n * c) // n_chunks
            b = (n * (c + 1)) // n_chunks
            for p in range(a, b):
                part[c, indices[p]] += 1
        out = np.zeros(n_cols, dtype=np.int64)
        for j in nb.prange(n_cols):
            s = 0
            for c in range(n_chunks):
                s += part[c, j]
            out[j] = s
        return out

    @nb.njit(parallel=True, cache=True)
    def pat_sum_r(indptr, indices, remap, v):
        # pat_sum over the entries whose column maps to >= 0 (relabelled)
        n = indptr.shape[0] - 1
        out = np.empty(n, dtype=np.float64)
        for i in nb.prange(n):
            s = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                j = remap[indices[p]]
                if j >= 0:
                    s += v[j]
            out[i] = s
        return out

    @nb.njit(parallel=True, cache=True)
    def pat_mm_r(indptr, indices, remap, Xin, out):
        n = indptr.shape[0] - 1
        b = Xin.shape[1]
        for i in nb.prange(n):
            for c in range(b):
                out[i, c] = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                j = remap[indices[p]]
                if j >= 0:
                    for c in range(b):
                        out[i, c] += Xin[j, c]

    @nb.njit(parallel=True, cache=True)
    def row_nnz_r(indptr, indices, remap):
        n = indptr.shape[0] - 1
        out = np.empty(n, dtype=np.int64)
        for i in nb.prange(n):
            c = 0
            for p in range(indptr[i], indptr[i + 1]):
                if remap[indices[p]] >= 0:
                    c += 1
            out[i] = c
        return out

    @nb.njit(parallel=True, cache=True)
    def pat_transpose_r(indptr, indices, remap, n_cols, n_chunks):
        # pat_transpose_par of the remapped pattern (columns remap >= 0)
        n_rows = indptr.shape[0] - 1
        bounds = np.empty(n_chunks + 1, dtype=np.int64)
        tot = indptr[n_rows]
        r = 0
        for c in range(n_chunks + 1):
            target = (tot * c) // n_chunks
            while r < n_rows and indptr[r] < target:
                r += 1
            bounds[c] = r
        bounds[n_chunks] = n_rows
        cnt = np.zeros((n_chunks, n_cols), dtype=np.int64)
        for c in nb.prange(n_chunks):
            for i in range(bounds[c], bounds[c + 1]):
                for p in range(indptr[i], indptr[i + 1]):
                    j = remap[indices[p]]
                    if j >= 0:
                        cnt[c, j] += 1
        tp = np.zeros(n_cols + 1, dtype=np.int64)
        for j in range(n_cols):
            tp[j + 1] = tp[j]
            for c in range(n_chunks):
                tp[j + 1] += cnt[c, j]
        for j in nb.prange(n_cols):
            pos = tp[j]
            for c in range(n_chunks):
                k = cnt[c, j]
                cnt[c, j] = pos
                pos += k
        ti = np.empty(tp[n_cols], dtype=np.int32)
        for c in nb.prange(n_chunks):
            for i in range(bounds[c], bounds[c + 1]):
                for p in range(indptr[i], indptr[i + 1]):
                    j = remap[indices[p]]
                    if j >= 0:
                        ti[cnt[c, j]] = i
                        cnt[c, j] += 1
        return tp, ti

    _FK.update(pat_sum=pat_sum, pat_mm=pat_mm, pat_transpose=pat_transpose, pat_gram=pat_gram,
               pat_filter=pat_filter, pat_transpose_par=pat_transpose_par, pat_select=pat_select,
               col_count_par=col_count_par, pat_sum_r=pat_sum_r, pat_mm_r=pat_mm_r,
               row_nnz_r=row_nnz_r, pat_transpose_r=pat_transpose_r)
    return _FK


def _n_chunks() -> int:
    import numba as nb
    return max(1, min(16, nb.get_num_threads()))


class Pattern:
    """Sparsity pattern of a binary matrix without a value array: CSR
    ``indptr`` / ``indices`` and, optionally, a column ``remap`` (an entry in
    column ``c`` of the stored arrays is column ``remap[c]`` of the pattern,
    dropped when ``remap[c] < 0``). The remapped form views the input
    matrix's own index arrays (the min-cells feature filter costs nothing);
    every BiRMT-2 kernel reads only these arrays, in the same entry order as
    the compacted 0/1 matrix, so all sums are bit-identical to it."""

    def __init__(self, indptr, indices, shape, remap=None):
        self.indptr = indptr
        self.indices = indices
        self.shape = tuple(int(v) for v in shape)
        self.remap = remap
        self._deg = None

    def row_nnz(self) -> np.ndarray:
        if self._deg is None:
            if self.remap is None:
                self._deg = np.diff(self.indptr)
            else:
                self._deg = _fast_kernels()["row_nnz_r"](self.indptr, self.indices, self.remap)
        return self._deg

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1]) if self.remap is None else int(self.row_nnz().sum())

    def transpose(self) -> "Pattern":
        K = _fast_kernels()
        if self.remap is None:
            tp, ti = K["pat_transpose_par"](self.indptr, self.indices, self.shape[1], _n_chunks())
        else:
            tp, ti = K["pat_transpose_r"](self.indptr, self.indices, self.remap, self.shape[1],
                                          _n_chunks())
        return Pattern(tp, ti, self.shape[::-1])

    def take(self, rows=None, cols=None) -> "Pattern":
        """Compact pattern of ``Y[rows][:, cols]`` (``cols`` sorted; ``None`` = all)."""
        rows = (np.arange(self.shape[0], dtype=np.int64) if rows is None
                else np.asarray(rows, dtype=np.int64))
        if cols is None and self.remap is None:
            remap, use = np.zeros(1, np.int64), False
            ncol = self.shape[1]
        else:
            if cols is None:
                sel = np.arange(self.shape[1], dtype=np.int64)
            else:
                sel = np.full(self.shape[1], -1, np.int64)
                sel[np.asarray(cols)] = np.arange(len(cols))
            if self.remap is None:
                remap = sel
            else:
                remap = np.where(self.remap >= 0, sel[np.maximum(self.remap, 0)], -1)
            use = True
            ncol = self.shape[1] if cols is None else len(cols)
        newp, newi = _fast_kernels()["pat_select"](self.indptr, self.indices, rows, remap, use)
        return Pattern(newp, newi, (len(rows), ncol))

    def col_counts(self) -> np.ndarray:
        if self.remap is not None:
            return self.take().col_counts()
        return _fast_kernels()["col_count_par"](self.indices, self.shape[1], _n_chunks())

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """``Y @ v`` (float64; = SciPy's float32-ones CSR times ``v``)."""
        v = np.ascontiguousarray(v, dtype=np.float64)
        K = _fast_kernels()
        if self.remap is None:
            return K["pat_sum"](self.indptr, self.indices, v)
        return K["pat_sum_r"](self.indptr, self.indices, self.remap, v)

    def matmat(self, V: np.ndarray) -> np.ndarray:
        V = np.ascontiguousarray(V, dtype=np.float64)
        out = np.empty((self.shape[0], V.shape[1]))
        K = _fast_kernels()
        if self.remap is None:
            K["pat_mm"](self.indptr, self.indices, V, out)
        else:
            K["pat_mm_r"](self.indptr, self.indices, self.remap, V, out)
        return out

    def tocsr(self) -> sp.csr_matrix:
        """float32 0/1 ``csr_matrix`` (only for the optional tile / sequence /
        ``svd='v1'`` paths)."""
        P = self.take() if self.remap is not None else self
        return sp.csr_matrix((np.ones(P.nnz, dtype=np.float32), P.indices, P.indptr),
                             shape=P.shape)

    def toarray(self):
        return self.tocsr().toarray()

    def __matmul__(self, other):
        return self.tocsr() @ other


def as_pattern(Y) -> Pattern:
    """Pattern of a binary ``csr_matrix`` (or a :class:`Pattern`)."""
    if isinstance(Y, Pattern):
        return Y
    Y = sp.csr_matrix(Y)
    ind = Y.indices if Y.indices.dtype == np.int32 else Y.indices.astype(np.int32)
    return Pattern(Y.indptr, ind, Y.shape)


class BinaryBiwhitened:
    """Matrix-free operator ``Ŷ = D(√x) Y D(√y)`` for a binary sparse ``Y``
    stored as CSR and CSC index arrays (``int32``/``int64``), with
    ``matmat`` (Ŷ X) and ``rmatmat`` (Ŷᵀ X) on ``n_jobs`` numba threads."""

    def __init__(self, Y: sp.csr_matrix, x: np.ndarray, y: np.ndarray, Yt: Optional[sp.csr_matrix] = None):
        self.shape = Y.shape
        self.rp, self.ri = Y.indptr, Y.indices
        self.remap = Y.remap if isinstance(Y, Pattern) else None
        if Yt is None:
            Yt = Y.transpose() if isinstance(Y, Pattern) else Y.T.tocsr()
        self.cp, self.ci = Yt.indptr, Yt.indices
        self.sx = np.sqrt(np.asarray(x, np.float64))
        self.sy = np.sqrt(np.asarray(y, np.float64))
        self.k = _fast_kernels()

    def matmat(self, X):
        X = np.ascontiguousarray(np.asarray(X, np.float64) * self.sy[:, None])
        out = np.empty((self.shape[0], X.shape[1]))
        if self.remap is None:
            self.k["pat_mm"](self.rp, self.ri, X, out)
        else:
            self.k["pat_mm_r"](self.rp, self.ri, self.remap, X, out)
        out *= self.sx[:, None]
        return out

    def rmatmat(self, X):
        X = np.ascontiguousarray(np.asarray(X, np.float64) * self.sx[:, None])
        out = np.empty((self.shape[1], X.shape[1]))
        self.k["pat_mm"](self.cp, self.ci, X, out)
        out *= self.sy[:, None]
        return out

    def rows(self, idx):
        """Dense rows ``Ŷ[idx]`` (float64)."""
        out = np.zeros((len(idx), self.shape[1]))
        for a, i in enumerate(idx):
            cols = self.ri[self.rp[i]:self.rp[i + 1]]
            if self.remap is not None:
                cols = self.remap[cols]
                cols = cols[cols >= 0]
            out[a, cols] = self.sx[i] * self.sy[cols]
        return out


def sinkhorn_binary(Y: sp.csr_matrix, Yt: sp.csr_matrix, *, tol: float = 1e-6,
                    max_iter: int = 10000) -> tuple[np.ndarray, np.ndarray, int]:
    """Sinkhorn-Knopp scaling of a binary ``m x n`` matrix to row sums ``n``
    and column sums ``m`` (Landa et al. 2022, Algorithm 2), two pattern passes
    per iteration; same fixed point and stopping rule as
    :func:`epione.tl._birmt.sinkhorn_scale` (relative column error < tol after
    the row update)."""
    k = _fast_kernels()
    m, n = Y.shape
    x = np.ones(m)
    it = 0
    remap = getattr(Y, "remap", None)
    cs = k["pat_sum"](Yt.indptr, Yt.indices, x)
    for it in range(1, max_iter + 1):
        y = m / cs
        if remap is None:
            x = n / k["pat_sum"](Y.indptr, Y.indices, y)
        else:
            x = n / k["pat_sum_r"](Y.indptr, Y.indices, remap, y)
        cs = k["pat_sum"](Yt.indptr, Yt.indices, x)
        if np.max(np.abs(y * cs / m - 1.0)) < tol:
            break
    else:
        console.warn(f"[birmt2] Sinkhorn did not reach tol={tol} in {max_iter} iterations")
    return x, y, it


def block_krylov_svd(op, n_vectors: int, *, block: int, seed: int = 0):
    """Randomized block Krylov SVD (Musco & Musco 2015, Algorithm 2).

    Builds the Krylov space ``[ŶΩ, (ŶŶᵀ)ŶΩ, …, (ŶŶᵀ)^q ŶΩ]`` with block size
    ``block`` and ``q = ⌈n_vectors/block⌉`` (re-orthonormalised at every
    step), then Rayleigh-Ritz on it. Returns ``U, s`` (cell side) for the
    ``q·block + block`` Ritz pairs, descending. Cost: ``(3q+2)·block``
    operand columns of sparse products, versus ``(2·n_iter+2)(k+p)`` for
    subspace iteration; the Krylov space also resolves the singular values
    just above a dense noise bulk far better at equal cost (Musco & Musco
    2015, Thm 1: gap-independent (1+ε) error with q = O(log n / √ε)).
    """
    m, n = op.shape
    rng = np.random.default_rng(seed)
    q = int(np.ceil(n_vectors / block))
    Om = rng.standard_normal((n, block))
    Z, _ = np.linalg.qr(op.matmat(Om))
    blocks = [Z]
    for _ in range(q):
        Z = op.matmat(op.rmatmat(Z))
        # orthogonalise against the previous blocks (block Gram-Schmidt x2)
        for _ in range(2):
            for Bk in blocks:
                Z -= Bk @ (Bk.T @ Z)
        Z, _ = np.linalg.qr(Z)
        blocks.append(Z)
    Q = np.hstack(blocks)
    Q, _ = np.linalg.qr(Q)
    W = op.rmatmat(Q)                        # n x L  (Ŷᵀ Q)
    G = W.T @ W                              # = Qᵀ Ŷ Ŷᵀ Q
    ev, R = np.linalg.eigh((G + G.T) / 2)
    o = np.argsort(-ev)
    ev, R = ev[o], R[:, o]
    s = np.sqrt(np.clip(ev, 0, None))
    U = Q @ R
    return U, s


# ---------------------------------------------------------------------------
# peak coordinates and the tile pyramid
# ---------------------------------------------------------------------------

def peak_coordinates(adata: AnnData):
    """``(chrom, start, end)`` arrays from ``var`` columns
    (``chrom``/``chromStart``/``chromEnd``) or from ``chr:start-end`` names."""
    v = adata.var
    for c, s, e in (("chrom", "chromStart", "chromEnd"), ("chrom", "start", "end")):
        if c in v and s in v and e in v:
            return (v[c].astype(str).to_numpy(), v[s].to_numpy(np.int64),
                    v[e].to_numpy(np.int64))
    import pandas as pd
    co = pd.Series(adata.var_names.astype(str)).str.extract(r"^(.+?)[:_-](\d+)[-_](\d+)$")
    if co.isna().any().any():
        raise ValueError("cannot parse peak coordinates from var_names; add var "
                         "columns chrom/chromStart/chromEnd")
    return co[0].to_numpy(), co[1].astype(np.int64).to_numpy(), co[2].astype(np.int64).to_numpy()


def tile_map(chrom: np.ndarray, mid: np.ndarray, size: int) -> sp.csr_matrix:
    """0/1 peak-to-tile matrix (peaks x tiles) for tiles of ``size`` bp."""
    _, cc = np.unique(chrom, return_inverse=True)
    key = cc.astype(np.int64) * (1 << 40) + (np.asarray(mid, np.int64) // int(size))
    _, inv = np.unique(key, return_inverse=True)
    n = len(mid)
    return sp.csr_matrix((np.ones(n, np.float32), (np.arange(n), inv)), shape=(n, inv.max() + 1))


# ---------------------------------------------------------------------------
# quadratic-variance biwhitening with KS selection of beta (Landa et al. Sec 5.1)
# ---------------------------------------------------------------------------

def mp_cdf_table(gamma: float, n_grid: int = 4000):
    """Grid ``(t, F(t))`` of the MP CDF with ratio ``gamma`` and variance 1."""
    g = float(gamma)
    lo, hi = (1 - np.sqrt(g)) ** 2, (1 + np.sqrt(g)) ** 2
    t = np.linspace(lo, hi, n_grid)
    dens = np.sqrt(np.maximum((hi - t) * (t - lo), 0.0)) / (2 * np.pi * g * np.maximum(t, 1e-300))
    F = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) * 0.5 * np.diff(t))])
    F /= F[-1]
    return t, F


def ks_to_mp(ev: np.ndarray, gamma: float) -> float:
    """KS distance between the empirical distribution of ``ev`` and MP(γ, 1)."""
    ev = np.sort(np.asarray(ev, np.float64))
    t, F = mp_cdf_table(gamma)
    Fm = np.interp(ev, t, F, left=0.0, right=1.0)
    k = len(ev)
    hi = np.arange(1, k + 1) / k
    lo = np.arange(0, k) / k
    return float(max(np.max(np.abs(hi - Fm)), np.max(np.abs(lo - Fm))))


def _qv_sinkhorn(Y: sp.csr_matrix, beta: float, tol: float, max_iter: int):
    V = Y.copy()
    V.data = (1.0 - beta) * Y.data + beta * Y.data ** 2
    return B1.sinkhorn_scale(V, tol=tol, max_iter=max_iter)


def biwhiten_qv(
    Y, *, betas: Optional[Sequence[float]] = None, n_subsample: int = 2000,
    tol: float = 1e-6, max_iter: int = 10000, seed: int = 0, n_jobs: Optional[int] = None,
) -> dict:
    """Biwhiten a nonnegative count matrix with ``V̂ = α[(1-β)Y + βY²]``.

    For each ``β`` in ``betas`` (default ``0, 0.1, …, 1``): Sinkhorn scaling
    of ``(1-β)Y + βY²`` (Landa et al. 2022 Eq. (17)/(23)), ``α`` by Eq. (24) on
    the Gram matrix of ``n_subsample`` random rows of the smaller side, and the
    KS distance (26) of the scaled spectrum to MP. The ``β`` with the smallest
    KS distance is returned (first on ties). Binary input: ``V̂ ∝ Y`` for every
    ``β`` and only ``β = 0`` is evaluated (identical to v1 :func:`biwhiten`).
    """
    nt = B1._set_threads(n_jobs)
    Y = sp.csr_matrix(Y, dtype=np.float64)
    Y.eliminate_zeros()
    m, n = Y.shape
    binary = bool(np.all(Y.data == 1.0))
    if betas is None:
        betas = np.round(np.linspace(0, 1, 11), 4)
    betas = [0.0] if binary else [float(b) for b in betas]
    rng = np.random.default_rng(seed)
    small_is_rows = m <= n
    a = m if small_is_rows else n
    b = n if small_is_rows else m
    idx = np.sort(rng.choice(a, size=min(a, int(n_subsample)), replace=False))
    def _scaled(Ym, beta):
        x, y, it = _qv_sinkhorn(Ym, beta, tol, max_iter)
        data = np.empty(Ym.nnz, dtype=np.float64)
        B1._kernels()["scaled_data"](Ym.indptr, Ym.indices, Ym.data, np.sqrt(x), np.sqrt(y), data)
        return x, y, it, sp.csr_matrix((data, Ym.indices, Ym.indptr), shape=Ym.shape)

    def _alpha_ks(Yw, rows, bb):
        S = Yw[rows] if small_is_rows else Yw.T.tocsr()[rows]
        ev = np.linalg.eigvalsh((S @ S.T).toarray() / bb)
        g = len(rows) / bb
        alpha = float(np.median(ev) / B1.mp_median(g))
        return alpha, ks_to_mp(ev / alpha, g)

    table = []
    if len(betas) > 1:
        # beta scan on the submatrix of the n_subsample sampled rows of the smaller side
        # (Landa et al. 2022 Sec. 5.1 select (alpha, beta) on a submatrix)
        Ys = Y[idx] if small_is_rows else Y.T.tocsr()[idx]
        cnz = np.flatnonzero(np.bincount(Ys.indices, minlength=Ys.shape[1]) > 0)
        Ys = Ys[:, cnz].tocsr()
        Ys = Ys[np.diff(Ys.indptr) > 0].tocsr()
        sel = None
        for beta in betas:
            _, _, _, Yws = _scaled(Ys, beta)
            Gs = Yws @ Yws.T
            ev = np.linalg.eigvalsh(Gs.toarray() / Ys.shape[1])
            gs = Ys.shape[0] / Ys.shape[1]
            al = float(np.median(ev) / B1.mp_median(gs))
            ks = ks_to_mp(ev / al, gs)
            table.append((beta, al, ks))
            if sel is None or ks < sel[0] - 1e-12:
                sel = (ks, beta)
        beta_sel = sel[1]
    else:
        beta_sel = betas[0]
    x, y, it, Yw = _scaled(Y, beta_sel)
    alpha, ks = _alpha_ks(Yw, idx, b)
    if not table:
        table.append((beta_sel, alpha, ks))
    best = (ks, beta_sel, alpha, x, y, it, Yw)
    ks, beta, alpha, x, y, it, Yw = best
    edge = float(np.sqrt(alpha) * (np.sqrt(m) + np.sqrt(n)))
    return dict(Yw=Yw.astype(np.float32), x=x, y=y, alpha=alpha, beta=beta, ks=ks, edge=edge,
                sinkhorn_iter=it, table=np.asarray(table), n_threads=nt)


def score_shrinkage(s: np.ndarray, m: int, n: int, sigma: float = 1.0) -> np.ndarray:
    """Optimal shrinkage for the *cell-score* matrix ``U_true T`` (rows = the
    ``m`` cells) rather than the full matrix.

    In the spiked model of Gavish & Donoho (2017) (Benaych-Georges &
    Nadakuditi 2012), an empirical singular value ``y`` (natural noise
    scale) above the bulk edge corresponds to the true value
    ``x(y) = √(((y²-β-1) + √((y²-β-1)² - 4β))/2)``, and the empirical singular
    vectors have cosines ``c(x) = √((x⁴-β)/(x⁴+βx²))`` (side of length
    ``min(m, n)``) and ``c̃(x) = √((x⁴-β)/(x⁴+x²))`` (side of length
    ``max(m, n)``) with the true ones. The Frobenius shrinker is ``x c c̃``.
    The MSE-optimal scalar ``w`` for estimating the true cell scores
    ``x u_true`` from ``u_emp`` is ``w = x <u_emp, u_true> = x c_cells``, i.e.
    the Frobenius shrinker without the feature-side cosine (which only
    measures how well the feature loadings, unused by a cell embedding, are
    estimated). Returned on the scale of ``s`` (Lemma 14 rescaling).
    """
    s = np.asarray(s, dtype=np.float64)
    N, M = max(m, n), min(m, n)
    beta = M / N
    c = sigma * np.sqrt(N)
    y = s / c
    out = np.zeros_like(s)
    above = y > 1 + np.sqrt(beta)
    ya = y[above]
    t = ya ** 2 - beta - 1
    x = np.sqrt(np.maximum((t + np.sqrt(np.maximum(t ** 2 - 4 * beta, 0.0))) / 2, 0.0))
    if m <= n:      # cells on the short side
        cos = np.sqrt(np.maximum((x ** 4 - beta) / (x ** 4 + beta * x ** 2), 0.0))
    else:
        cos = np.sqrt(np.maximum((x ** 4 - beta) / (x ** 4 + x ** 2), 0.0))
    out[above] = c * x * cos
    return out


# ---------------------------------------------------------------------------
# one level: biwhiten -> MP rank -> GD shrinkage -> depth-normalised coords
# ---------------------------------------------------------------------------

def denoise_level(
    Y: sp.csr_matrix, *, k_max: int = 128, shrinkage: str = "frobenius",
    row_norm: str = "depth", seed: int = 0, n_jobs: Optional[int] = None,
    betas: Optional[Sequence[float]] = None,
) -> dict:
    """Biwhiten (quadratic variance, KS-selected β), MP rank, GD shrinkage.

    Returns ``Z`` (depth-normalised denoised coordinates, cells x kept), the
    SVD factors, the shrunk singular values and diagnostics."""
    nt = B1._set_threads(n_jobs)
    bw = biwhiten_qv(Y, betas=betas, seed=seed, n_jobs=nt)
    U, s, Vt, r, capped = B1.mp_rank_svd(bw["Yw"], bw["edge"], k_max=k_max, seed=seed, n_jobs=nt)
    m, n = Y.shape
    if r < 1:
        return dict(Z=None, rank=0, alpha=bw["alpha"], beta=bw["beta"], ks=bw["ks"])
    if shrinkage == "score":
        s_sh = score_shrinkage(s[:r], m, n, sigma=np.sqrt(bw["alpha"]))
    else:
        s_sh = B1.shrink_singular_values(s[:r], m, n, sigma=np.sqrt(bw["alpha"]), kind=shrinkage)
    keep = s_sh > 0
    if not keep.any():
        keep[0] = True
        s_sh[0] = s[0]
    Uk, sk, Vk = U[:, :r][:, keep], s_sh[keep], Vt[:r][keep]
    depth = np.asarray(Y.sum(1)).ravel()
    if row_norm == "depth":
        rs = 1.0 / (np.sqrt(bw["x"]) * depth)
    elif row_norm == "none":
        rs = np.ones(m)
    else:
        raise ValueError(f"unknown row_norm {row_norm!r}")
    return dict(U=Uk, s=sk, Vt=Vk, row_scale=rs, rank=int(r), capped=bool(capped),
                alpha=float(bw["alpha"]), beta=float(bw["beta"]), ks=float(bw["ks"]),
                singular_values=s, n_features=int(n))


def _fast_denoise_fit(
    Y: sp.csr_matrix, *, k_max: int = 128, shrinkage: str = "score", block: int = 16,
    n_subsample: int = 2000, tol: float = 1e-6, seed: int = 0, n_jobs: Optional[int] = None,
) -> dict:
    """Level-0 BiRMT on a binary matrix with the matrix-free kernels.

    Same estimator as v1 steps 1-2 (Sinkhorn biwhitening, α by Landa et al.
    Eq. (24) on ``n_subsample`` random cells, MP rank, Gavish-Donoho-type
    shrinkage, depth row normalisation); the SVD is a block Krylov
    Rayleigh-Ritz (:func:`block_krylov_svd`) whose size starts from a rank
    pre-estimate (the number of subsample eigenvalues above the subsample MP
    edge, which is a lower bound in expectation for the full matrix) and is
    doubled while the MP rank comes within one block of the Krylov
    dimension.
    """
    import time
    tt = [time.time()]
    nt = B1._set_threads(n_jobs)
    m, n = Y.shape
    # pattern only: int32 indices of Y and of its transpose, no value arrays
    Y = as_pattern(Y)
    Yt = Y.transpose()
    tt.append(time.time())
    x, y, it = sinkhorn_binary(Y, Yt, tol=tol)
    tt.append(time.time())
    op = BinaryBiwhitened(Y, x, y, Yt)
    # alpha (Eq. 24) and the rank pre-estimate on a cell subsample
    rng = np.random.default_rng(seed)
    K = _fast_kernels()
    if m <= n:
        idx = np.sort(rng.choice(m, size=min(m, int(n_subsample)), replace=False))
        Sr = Y.take(idx)
        w_row, w_col, bdim = op.sx[idx], op.sy, n
    else:
        idx = np.sort(rng.choice(n, size=min(n, int(n_subsample)), replace=False))
        Sr = Yt.take(idx)
        w_row, w_col, bdim = op.sy[idx], op.sx, m
    tt.append(time.time())
    sc_p, sc_i = K["pat_transpose"](Sr.indptr, Sr.indices, Sr.shape[1])
    G = K["pat_gram"](Sr.indptr, Sr.indices, sc_p, sc_i, w_row, w_col, Sr.shape[0])
    tt.append(time.time())
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1, user_api="blas"):
        # a 2,000 x 2,000 symmetric eigenproblem: single-threaded LAPACK is
        # fast and avoids multi-threaded dsyevd stalls on oversubscribed nodes
        ev = np.linalg.eigvalsh((G + G.T) / (2 * bdim))
    tt.append(time.time())
    g = len(idx) / bdim
    alpha = float(np.median(ev) / B1.mp_median(g))
    r0 = int(np.sum(ev / alpha > (1 + np.sqrt(g)) ** 2))
    edge = float(np.sqrt(alpha) * (np.sqrt(m) + np.sqrt(n)))
    nv = int(min(max(48, 2 * r0 + block), 2 * k_max))
    tt.append(time.time())
    while True:
        U, sv = block_krylov_svd(op, nv, block=block, seed=seed)
        r = int(np.sum(sv > edge))
        if r <= nv - block or nv >= 2 * k_max:
            break
        nv = int(min(2 * nv, 2 * k_max))
    tt.append(time.time())
    console.info("[birmt2] level-0 timings (s) transpose/sinkhorn/subsample/gram/eig/-/krylov: " + " ".join(f"{b - a:.1f}" for a, b in zip(tt[:-1], tt[1:])))
    capped = r > k_max
    r = min(r, k_max)
    if r < 1:
        raise ValueError("no singular value above the MP edge: no detectable signal")
    if shrinkage == "score":
        s_sh = score_shrinkage(sv[:r], m, n, sigma=np.sqrt(alpha))
    else:
        s_sh = B1.shrink_singular_values(sv[:r], m, n, sigma=np.sqrt(alpha), kind=shrinkage)
    keep = s_sh > 0
    if not keep.any():
        keep[0] = True
        s_sh[0] = sv[0]
    depth = Y.row_nnz().astype(np.float64)
    return dict(U=U[:, :r][:, keep], s=s_sh[keep], s_raw=sv[:r][keep], row_scale=1.0 / (op.sx * depth), rank=r,
                capped=bool(capped), alpha=alpha, beta=0.0, ks=float(ks_to_mp(ev / alpha, g)),
                rank_pre=r0, krylov_dim=int(U.shape[1]), sinkhorn_iter=int(it),
                singular_values=sv, n_features=int(n), op=op)


def fast_denoise_binary(
    Y: sp.csr_matrix, *, k_max: int = 128, shrinkage: str = "score", block: int = 16,
    n_subsample: int = 2000, tol: float = 1e-6, seed: int = 0, n_jobs: Optional[int] = None,
) -> dict:
    """Level-0 BiRMT on a binary matrix (see :func:`_fast_denoise_fit`).

    Sinkhorn scaling of a binary ``m x n`` matrix to row sums ``n`` and
    column sums ``m`` requires every row to have at least ``n/m`` non-zeros
    (a row with ``d`` entries, each at most its column total ``m``, sums to at
    most ``d m``). Rows below that bound (in practice near-empty barcodes or
    spots, e.g. a few open tiles out of 50,000) make the scaling diverge. They
    are left out of the fit and placed by the out-of-sample (Nyström)
    extension of the fitted factorisation: ``x_i = n / Σ_j Y_ij y_j`` from the
    row constraint, ``u_i = √x_i (Y_i ∘ √y) V / s``. With no such row this is
    exactly :func:`_fast_denoise_fit`.
    """
    m, n = Y.shape
    depth = as_pattern(Y).row_nnz()
    low = depth < (n / m)
    if not low.any():
        return _fast_denoise_fit(Y, k_max=k_max, shrinkage=shrinkage, block=block,
                                 n_subsample=n_subsample, tol=tol, seed=seed, n_jobs=n_jobs)
    fit_rows = np.flatnonzero(~low)
    console.warn(f"[birmt2] {int(low.sum()):,} rows have fewer than n/m = {n / m:.1f} non-zeros "
                 "(Sinkhorn-infeasible); fitted out of sample")
    Y = as_pattern(Y)
    Yf = Y.take(fit_rows)
    cols = np.flatnonzero(Yf.col_counts() > 0)
    Yf = Yf.take(None, cols)
    lv = _fast_denoise_fit(Yf, k_max=k_max, shrinkage=shrinkage, block=block,
                           n_subsample=n_subsample, tol=tol, seed=seed, n_jobs=n_jobs)
    del Yf
    op = lv["op"]
    V = op.rmatmat(lv["U"]) / lv["s_raw"][None, :]            # right singular vectors (cols)
    Yl = Y.take(np.flatnonzero(low), cols)
    sy = op.sy
    rs = Yl.matvec(sy ** 2)                                    # Σ_j Y_ij y_j
    xl = len(cols) / np.maximum(rs, 1e-300)                    # row target = number of columns
    Ul = (np.sqrt(xl)[:, None] * Yl.matmat(sy[:, None] * V)) / lv["s_raw"][None, :]
    U = np.empty((m, lv["U"].shape[1]))
    U[fit_rows] = lv["U"]
    U[low] = Ul
    rsc = np.empty(m)
    rsc[fit_rows] = lv["row_scale"]
    dl = np.maximum(Yl.row_nnz().astype(np.float64), 1.0)
    rsc[low] = 1.0 / (np.sqrt(xl) * dl)
    lv.update(U=U, row_scale=rsc, n_out_of_sample=int(low.sum()))
    return lv


# ---------------------------------------------------------------------------
# sequence prior: k-mer graph + Laplacian (Tikhonov) regularisation
# ---------------------------------------------------------------------------

_LUT = np.full(256, 4, dtype=np.uint8)
for _c, _v in zip(b"ACGTacgt", (0, 1, 2, 3, 0, 1, 2, 3)):
    _LUT[_c] = _v


def _fetch_codes(fasta: str, chrom, start, end) -> np.ndarray:
    """Integer-coded sequences (A0 C1 G2 T3 N4), all of length end-start."""
    import pyfaidx
    fa = pyfaidx.Fasta(fasta, as_raw=True, sequence_always_upper=True)
    w = int(end[0] - start[0])
    out = np.full((len(chrom), w), 4, dtype=np.uint8)
    for i, (c, s, e) in enumerate(zip(chrom, start, end)):
        if c not in fa:
            continue
        L = len(fa[c])
        s0, e0 = max(int(s), 0), min(int(e), L)
        if e0 <= s0:
            continue
        seq = fa[c][s0:e0]
        out[i, s0 - int(s):s0 - int(s) + len(seq)] = _LUT[np.frombuffer(seq.encode(), np.uint8)]
    return out


def kmer_profiles(codes: np.ndarray, k: int = 6, chunk: int = 20000) -> tuple[sp.csr_matrix, np.ndarray]:
    """Canonical k-mer counts (peaks x canonical k-mers) and GC fraction."""
    n, w = codes.shape
    allk = np.arange(4 ** k)
    rcall = np.zeros_like(allk)
    t = allk.copy()
    for j in range(k):
        rcall = rcall * 4 + (3 - t % 4)
        t //= 4
    canon = np.unique(np.minimum(allk, rcall))
    pos = np.full(4 ** k, -1, np.int64)
    pos[canon] = np.arange(len(canon))
    nwin = w - k + 1
    blocks = []
    for a in range(0, n, chunk):
        cb = codes[a:a + chunk]
        valid = cb < 4
        c = np.where(valid, cb, 0).astype(np.int64)
        fw = np.zeros((cb.shape[0], nwin), np.int64)
        rc = np.zeros((cb.shape[0], nwin), np.int64)
        ok = np.ones((cb.shape[0], nwin), bool)
        for j in range(k):
            fw = fw * 4 + c[:, j:j + nwin]
            rc = rc + (3 - c[:, j:j + nwin]) * (4 ** j)
            ok &= valid[:, j:j + nwin]
        can = np.minimum(fw, rc)
        rows = np.repeat(np.arange(cb.shape[0]), nwin)[ok.ravel()]
        cols = pos[can.ravel()[ok.ravel()]]
        Kb = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(cb.shape[0], len(canon)))
        Kb.sum_duplicates()
        blocks.append(Kb)
    K = sp.vstack(blocks).tocsr()
    valid = codes < 4
    gc = ((codes == 1) | (codes == 2)).sum(1) / np.maximum(valid.sum(1), 1)
    return K, gc


def sequence_graph(
    chrom, start, end, fasta: str, *, k: int = 6, width: int = 500, n_pcs: int = 50,
    n_neighbors: int = 15, gc_regress: bool = True, seed: int = 0, n_jobs: Optional[int] = None,
) -> tuple[sp.csr_matrix, dict]:
    """Normalised Laplacian of the cosine kNN graph of peak k-mer profiles."""
    nt = B1._set_threads(n_jobs)
    mid = (np.asarray(start) + np.asarray(end)) // 2
    s = mid - width // 2
    codes = _fetch_codes(fasta, chrom, s, s + width)
    K, gc = kmer_profiles(codes, k)
    tot = np.asarray(K.sum(1)).ravel()
    F = K.toarray().astype(np.float32)
    F /= np.maximum(tot, 1.0)[:, None]
    F = np.log(F + 1.0 / K.shape[1])
    F -= F.mean(0)
    F /= np.maximum(F.std(0), 1e-12)
    if gc_regress:
        g = (gc - gc.mean())[:, None].astype(np.float32)
        beta = (g.T @ F) / max(float((g.T @ g)[0, 0]), 1e-12)
        F -= g @ beta
    import sklearn.utils.extmath
    Up, sp_, _ = sklearn.utils.extmath.randomized_svd(F, n_components=min(n_pcs, F.shape[1] - 1),
                                                      n_iter=4, random_state=seed)
    P = (Up * sp_).astype(np.float32)
    P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)   # cosine
    bad = tot == 0
    idx, _ = B1._knn(P, n_neighbors, seed, nt)
    n = P.shape[0]
    A = sp.csr_matrix((np.ones(idx.size, np.float32), (np.repeat(np.arange(n), idx.shape[1]), idx.ravel())),
                      shape=(n, n))
    A = A.maximum(A.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    if bad.any():   # peaks without sequence: isolate
        D = sp.diags((~bad).astype(np.float32))
        A = (D @ A @ D).tocsr()
        A.eliminate_zeros()
    d = np.asarray(A.sum(1)).ravel()
    dinv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-300)), 0.0)
    Lsym = sp.identity(n, format="csr") - sp.diags(dinv) @ A @ sp.diags(dinv)
    return Lsym.tocsr(), dict(gc=gc, n_no_sequence=int(bad.sum()), mean_degree=float(d.mean()))


def tikhonov_transform(Vt: np.ndarray, L: sp.csr_matrix, lam: float, tol: float = 1e-8):
    """``M = (Vᵀ (I+λL)⁻² V)^½`` for orthonormal rows ``Vt`` (r x n)."""
    from scipy.sparse.linalg import cg
    r, n = Vt.shape
    F = (sp.identity(n, format="csr") + lam * L).tocsr()
    W = np.empty((n, r))
    for j in range(r):
        w, info = cg(F, Vt[j].astype(np.float64), rtol=tol, maxiter=2000)
        W[:, j] = w
    G = W.T @ W
    ev, Q = np.linalg.eigh((G + G.T) / 2)
    return (Q * np.sqrt(np.clip(ev, 0, None))) @ Q.T, W


# ---------------------------------------------------------------------------
# public entry
# ---------------------------------------------------------------------------

def _filter_columns(Y, min_cells: int):
    """Binary pattern (:class:`Pattern`) restricted to columns open in
    >= min_cells rows; ``Y`` is any CSR structure (values ignored)."""
    cn = _fast_kernels()["col_count_par"](Y.indices, Y.shape[1], _n_chunks())
    feats = np.flatnonzero(cn >= min_cells)
    remap = np.full(Y.shape[1], -1, np.int64)
    remap[feats] = np.arange(len(feats))
    # A view of Y's own index arrays with the column relabelling (no copy,
    # +4 B per non-zero overall for the transpose) when memory is tight;
    # otherwise a compact int32 copy (+8 B per non-zero, ~1.5x faster
    # products). Bit-identical results either way.
    view = Pattern(Y.indptr, Y.indices, (Y.shape[0], len(feats)), remap=remap)
    if _compact_pattern_fits(int(Y.indptr[-1]), Y.shape[0]):
        newp, newi = _fast_kernels()["pat_filter"](Y.indptr, Y.indices, remap)
        return Pattern(newp, newi, (Y.shape[0], len(feats))), feats
    console.level2("[birmt2] memory-lean pattern view of the input (no compact copy)")
    return view, feats


def _compact_pattern_fits(nnz: int, n_rows: int) -> bool:
    """Whether the compact pattern copy (+ its transpose, ~9 B per non-zero)
    fits comfortably next to the current RSS within the process memory
    limit (RAM / cgroup, e.g. Slurm --mem). ``EPIONE_BIRMT_PATTERN`` =
    ``compact`` / ``view`` forces either."""
    import os
    mode = os.environ.get("EPIONE_BIRMT_PATTERN", "auto")
    if mode in ("compact", "view"):
        return mode == "compact"
    from ..pp._fragment_io import memory_limit_bytes
    rss = 0
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
    except OSError:
        pass
    need = 9 * nnz + 16 * n_rows + (4 << 30)
    return rss + need <= 0.9 * memory_limit_bytes()


def leiden_markov(W: sp.csr_matrix, resolution: float = 1.0, seed: int = 0) -> np.ndarray:
    """One Leiden run (igraph C implementation, modularity objective,
    iterated to convergence) on the doubly-stochastic kernel ``W``.

    Every node of ``W`` has strength 1 (uniform stationary distribution), so
    the RB-configuration/modularity quality with resolution ``γ`` is the
    linearised Markov stability at Markov time ``1/γ`` (Lambiotte, Delvenne &
    Barahona 2014); ``γ = 1`` is one step of the doubly-stochastic random walk,
    the natural time scale of the kernel (Delvenne et al. 2010: modularity is
    stability at t = 1)."""
    import random
    import igraph as ig
    U = sp.triu(W, k=1).tocoo()
    g = ig.Graph(n=W.shape[0], edges=np.column_stack([U.row, U.col]), directed=False)
    random.seed(int(seed))
    ig.set_random_number_generator(random)
    part = g.community_leiden(objective_function="modularity", weights=U.data.astype(float),
                              resolution=float(resolution), n_iterations=-1)
    return np.asarray(part.membership, dtype=np.int32)


def birmt2(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
    min_cells: Union[int, Literal["auto"]] = "auto",
    shrinkage: Literal["frobenius", "operator", "hard", "none", "score"] = "score",
    tile_sizes: Sequence[int] = (),
    tile_binarize: bool = True,
    level_weight: Literal["energy", "equal"] = "energy",
    genome: Optional[str] = None,
    seq_lambda: float = 0.0,
    kmer_k: int = 6,
    kmer_width: int = 500,
    kmer_neighbors: int = 15,
    kmer_pcs: int = 50,
    gc_regress: bool = True,
    k_max: int = 128,
    svd: Literal["krylov", "v1"] = "krylov",
    krylov_block: int = 16,
    n_neighbors: int = 30,
    kernel_max_iter: int = 1000,
    cluster: bool = True,
    cluster_method: Literal["markov1", "markov_stability"] = "markov1",
    resolution: float = 1.0,
    resolutions: Optional[Sequence[float]] = None,
    n_seeds: int = 10,
    key_added: str = "X_birmt",
    cluster_key: str = "birmt",
    seed: int = 0,
    n_jobs: Optional[int] = None,
) -> AnnData:
    """BiRMT-2 embedding and clustering (see module docstring). Called by
    ``epi.tl.birmt(version=2)``.

    Parameters (beyond v1)
    ----------------------
    shrinkage
        ``'score'`` (cell-score optimal shrinker, :func:`score_shrinkage`)
        or a v1 Gavish-Donoho shrinker.
    tile_sizes, tile_binarize, level_weight
        Multiscale tile pyramid (empty = peaks only). ``tile_binarize``: a
        tile is open in a cell if any of its peaks is (binary level, same
        estimator and fast kernels as level 0); ``False``: peak counts per
        tile with the quadratic-variance biwhitening.
    genome, seq_lambda, kmer_*, gc_regress
        Sequence prior (``seq_lambda = 0`` disables it).
    svd
        ``'krylov'``: matrix-free block Krylov (:func:`fast_denoise_binary`);
        ``'v1'``: v1's randomized subspace iteration.
    cluster_method, resolution
        ``'markov1'``: one Leiden run on ``W`` at resolution ``resolution``
        (Markov time ``1/resolution``; :func:`leiden_markov`);
        ``'markov_stability'``: the v1 scan over ``resolutions`` x ``n_seeds``.
    """
    import time
    t0 = time.time()
    nt = B1._set_threads(n_jobs)
    X = adata.layers[layer] if layer is not None else adata.X
    if type(X).__name__ in ("BackedArray", "_SubsetBackedArray", "TransformedBackedArray",
                            "ScaledBackedArray"):
        # anndataoom backed matrix: materialise chunk by chunk (as iterative_lsi does)
        parts = [chunk for _, _, chunk in X.chunked()]
        X = sp.vstack([sp.csr_matrix(c) for c in parts]).tocsr()
    X = sp.csr_matrix(X)
    if X.data.size and X.data.min() <= 0:
        X = X.copy(); X.data[X.data <= 0] = 0; X.eliminate_zeros()
    # the binary matrix is X's sparsity pattern: no value array is built
    n_cells = X.shape[0]
    mc = int(max(5, np.ceil(2 * np.log(n_cells)))) if min_cells == "auto" else int(min_cells)
    Y0, feats = _filter_columns(X, mc)
    if np.any(Y0.row_nnz() == 0):
        raise ValueError("some cells have no retained feature")
    need_coords = len(tile_sizes) > 0 or seq_lambda > 0
    if need_coords:
        chrom, start, end = peak_coordinates(adata)
        chrom, start, end = chrom[feats], start[feats], end[feats]
        mid = (start + end) // 2
    console.info(f"[birmt2] {n_cells:,} cells x {len(feats):,} peaks (min_cells={mc}); "
                 f"shrinkage={shrinkage} tiles={list(tile_sizes)} seq_lambda={seq_lambda}")
    if svd == "krylov":
        lv0 = fast_denoise_binary(Y0, k_max=k_max, shrinkage=shrinkage, block=krylov_block,
                                  seed=seed, n_jobs=nt)
    else:
        lv0 = denoise_level(Y0.tocsr(), k_max=k_max, shrinkage=shrinkage, seed=seed, n_jobs=nt)
    M = None
    seq_info = {}
    if seq_lambda > 0:
        if genome is None:
            raise ValueError("seq_lambda > 0 needs genome= (FASTA path)")
        if "Vt" in lv0:
            Vt = lv0["Vt"]
        else:   # right singular vectors of the kept components: Ŷᵀ U / s
            sv = lv0["singular_values"][:lv0["rank"]][:lv0["U"].shape[1]]
            V = lv0["op"].rmatmat(lv0["U"]) / np.maximum(sv, 1e-300)[None, :]
            Vt, _ = np.linalg.qr(V)
            Vt = Vt.T
        L, sinfo = sequence_graph(chrom, start, end, genome, k=kmer_k, width=kmer_width,
                                  n_pcs=kmer_pcs, n_neighbors=kmer_neighbors,
                                  gc_regress=gc_regress, seed=seed, n_jobs=nt)
        M, _ = tikhonov_transform(Vt, L, float(seq_lambda))
        seq_info = dict(n_no_sequence=sinfo["n_no_sequence"], mean_degree=sinfo["mean_degree"],
                        M_diag=np.diag(M).copy())
    lv0.pop("op", None)
    levels = []
    Z0 = lv0["U"] * lv0["s"][None, :]
    if M is not None:
        Z0 = Z0 @ M
    levels.append(dict(name="peaks", Z=Z0 * lv0["row_scale"][:, None], energy=float(np.linalg.norm(Z0)),
                       info={k: lv0[k] for k in ("rank", "alpha", "beta", "ks", "n_features")}))
    for T in tile_sizes:
        A = tile_map(chrom, mid, int(T))
        Yt = (Y0.tocsr() @ A).tocsr()
        cn = np.bincount(Yt.indices, minlength=Yt.shape[1])
        Yt = Yt[:, np.flatnonzero(cn >= mc)].tocsr()
        if tile_binarize:
            Yt = sp.csr_matrix((np.ones(Yt.nnz, np.float32), Yt.indices, Yt.indptr), shape=Yt.shape)
            try:
                lv = fast_denoise_binary(Yt, k_max=k_max, shrinkage=shrinkage, block=krylov_block,
                                         seed=seed, n_jobs=nt)
                lv.pop("op", None)
            except ValueError:
                lv = {}
        else:
            lv = denoise_level(Yt, k_max=k_max, shrinkage=shrinkage, seed=seed, n_jobs=nt)
        if lv.get("rank", 0) < 1:
            console.warn(f"[birmt2] tile level {T} bp: no signal above the MP edge; skipped")
            continue
        Zl = lv["U"] * lv["s"][None, :]
        levels.append(dict(name=f"tile_{int(T)}", Z=Zl * lv["row_scale"][:, None],
                           energy=float(np.linalg.norm(Zl)),
                           info={k: lv[k] for k in ("rank", "alpha", "beta", "ks", "n_features")}))
    parts = []
    for lv in levels:
        nz = np.linalg.norm(lv["Z"])
        w = lv["energy"] if level_weight == "energy" else 1.0
        parts.append(lv["Z"] * (w / max(nz, 1e-300)))
    Z = np.hstack(parts) if len(parts) > 1 else levels[0]["Z"]
    t1 = time.time()
    for lv in levels:
        console.info(f"[birmt2] level {lv['name']}: " + " ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in lv["info"].items())
            + f" ({t1 - t0:.1f}s)")
    adata.obsm[key_added] = np.asarray(Z, dtype=np.float32)
    info = dict(version=2, levels=[dict(name=lv["name"], **lv["info"], energy=lv["energy"]) for lv in levels],
                n_dims=int(Z.shape[1]), min_cells=mc, seq=seq_info, time_embedding_s=float(t1 - t0),
                rank_pre=lv0.get("rank_pre"), krylov_dim=lv0.get("krylov_dim"),
                params=dict(shrinkage=shrinkage, tile_sizes=list(map(int, tile_sizes)), tile_binarize=tile_binarize,
                            level_weight=level_weight, seq_lambda=float(seq_lambda), kmer_k=kmer_k,
                            kmer_width=kmer_width, kmer_neighbors=kmer_neighbors, kmer_pcs=kmer_pcs,
                            gc_regress=gc_regress, k_max=k_max, svd=svd, krylov_block=krylov_block,
                            n_neighbors=n_neighbors, cluster_method=cluster_method,
                            resolution=resolution, seed=seed))
    if cluster:
        import pandas as pd
        W, kinfo = ds_kernel_fast(Z, n_neighbors=n_neighbors, max_iter=kernel_max_iter, n_jobs=nt)
        adata.obsp[key_added + "_W"] = W
        info["epsilon"] = float(kinfo["epsilon"])
        info["kernel_sinkhorn_iter"] = int(kinfo["sinkhorn_iter"])
        if cluster_method == "markov1":
            labels = leiden_markov(W, resolution=resolution, seed=seed)
            info["resolution"] = float(resolution)
        elif cluster_method == "markov_stability":
            ms = B1.markov_stability(W, resolutions=resolutions, n_seeds=n_seeds, n_jobs=nt)
            labels = ms["labels"]
            info["resolution"] = ms["resolution"]
            info["markov_stability"] = {c: ms["table"][c].to_numpy() for c in ms["table"].columns}
        else:
            raise ValueError(f"unknown cluster_method {cluster_method!r}")
        adata.obs[cluster_key] = pd.Categorical(labels.astype(str))
        console.info(f"[birmt2] clusters={len(np.unique(labels))} ({time.time() - t1:.1f}s)")
    info["time_total_s"] = float(time.time() - t0)
    adata.uns[key_added] = info
    return adata


def ds_kernel_fast(Z: np.ndarray, *, n_neighbors: int = 30, tol: float = 1e-6,
                   max_iter: int = 1000, n_jobs: Optional[int] = None):
    """v1's doubly-stochastic Gaussian kernel (:func:`epione.tl._birmt.ds_kernel`)
    with an exact kNN search by sklearn's tree/brute selection (same neighbour
    sets as v1's brute force) and the symmetric Sinkhorn capped at
    ``max_iter`` iterations (``tol`` on the row sums)."""
    from sklearn.neighbors import NearestNeighbors
    nt = B1._set_threads(n_jobs)
    n = Z.shape[0]
    k = min(n_neighbors, n - 1)
    Zf = np.ascontiguousarray(Z, dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=nt).fit(Zf)
    dist, idx = nn.kneighbors(Zf)
    self_first = idx[:, 0] == np.arange(n)
    idx = np.where(self_first[:, None], idx[:, 1:], idx[:, :-1]).astype(np.int64)
    dist = np.where(self_first[:, None], dist[:, 1:], dist[:, :-1]).astype(np.float64)
    eps = float(np.median(dist[:, -1] ** 2))
    if not eps > 0:
        eps = float(np.mean(dist ** 2)) or 1.0
    rows = np.repeat(np.arange(n), idx.shape[1])
    D2 = sp.csr_matrix((dist.ravel() ** 2 + 1.0, (rows, idx.ravel())), shape=(n, n))
    D2 = D2.maximum(D2.T).tocsr()
    D2.setdiag(0)
    D2.eliminate_zeros()
    D2.data -= 1.0
    D2.sort_indices()
    logk = -D2.data / eps
    kk = B1._kernels()
    u = np.zeros(n)
    lse = np.empty(n)
    err = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        kk["row_lse"](D2.indptr, D2.indices, logk, u, lse)
        err = np.max(np.abs(np.expm1(u + lse)))
        if err < tol:
            break
        u = 0.5 * (u - lse)
    rr = np.repeat(np.arange(n), np.diff(D2.indptr))
    W = sp.csr_matrix((np.exp(logk + u[rr] + u[D2.indices]), D2.indices.copy(), D2.indptr.copy()),
                      shape=(n, n))
    W = ((W + W.T) * 0.5).tocsr()
    return W, dict(epsilon=eps, sinkhorn_iter=it, sinkhorn_err=float(err))
