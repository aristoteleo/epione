"""OT-batch: per-batch biwhitening + unbalanced entropic optimal-transport
barycenter alignment for scATAC-seq batch correction.

The method has two parts.

**A. Per-batch biwhitened joint embedding** (Landa, Zhang & Kluger 2022).
The binarised cells x features matrix ``Y`` is split by batch. For every
batch ``b`` the Sinkhorn-Knopp algorithm finds positive scalings ``x_b``
(cells) and ``y_b`` (features open in at least ``max(5, ⌈2 ln n_b⌉)`` cells
of the batch; the others get ``y = 0``) such that ``D(x_b) Y_b D(y_b)`` has
row sums ``m_b`` (retained features) and column sums ``n_b`` (cells): with
the Poisson/quadratic variance estimator ``V̂ = Y`` (for binary data every
estimator of their Eq. (23) is ``∝ Y``) this gives unit average noise
variance in every row and column of the batch (their Algorithms 1-2). The
remaining noise scale ``α_b`` is set by their Eq. (24) (median eigenvalue of
a row sub-sample's Gram matrix ``Ŷ_S Ŷ_Sᵀ / m_b`` divided by the
Marchenko-Pastur median), and

    Ŷ_b = D(x_b)^½ Y_b D(y_b)^½ / √α_b .

Each batch is thus normalised to its *own* noise structure (depth, feature
detection, noise level) before a joint basis is formed. The stacked ``Ŷ``
is factorised by one randomized SVD (Halko et al. 2011) of size
``k_max + 1``; the rank ``r`` is the number of singular values above the MP
edge ``√n + √m`` (capped at ``k_max``) and the retained singular values are
shrunk with the Frobenius-optimal shrinker of Gavish & Donoho (2017,
Eq. (7)). The leading singular pair of a Sinkhorn-scaled nonnegative
matrix is its Perron (scaling) component, ``≈ (√x, √y)``, which carries the
per-cell scaling and no cell-type contrast; with ``drop_first`` it is
removed. The coordinates ``Z = U S̃`` are optionally centred and
L2-normalised per cell (``row_norm='l2'``).

**B. Unbalanced entropic OT barycenter alignment** (Cuturi 2013; Cuturi &
Doucet 2014; Chizat, Peyré, Schmitzer & Vialard 2018; Peyré & Cuturi 2019).
Let ``ẑ_i = z_i / ‖z_i‖`` and ``μ_b = Σ_{i∈b} n_b⁻¹ δ_{ẑ_i}`` be the
empirical measure of batch ``b`` on the unit sphere, and
``ν = Σ_k w_k δ_{y_k}`` a free-support barycenter with ``K`` support points
(landmarks; default ``min(round(n/30), 100)``, initialised by k-means on a
batch-balanced sub-sample). One correction round consists of
``n_bary_iter`` alternations of

1. for every batch, the semi-unbalanced entropic plan (every cell keeps its
   mass, the batch's landmark marginal is relaxed towards ``w``)

       π_b = argmin_{π1 = a} ⟨C_b, π⟩ + ε KL(π | a⊗w) + ρ KL(πᵀ1 | w),
       C_b[i, k] = ‖ẑ_i - y_k‖² = 2(1 - cos(ẑ_i, y_k)),

   solved by the log-domain generalised Sinkhorn iterations of Chizat et al.
   (2018, Alg. 2), warm-started from the previous dual potential:

       f_i ← -ε log Σ_k w_k exp((g_k - C_ik)/ε)
       g_k ← -λ ε log Σ_i a_i exp((f_i - C_ik)/ε),  λ = ρ/(ρ+ε)

   (``ρ = ∞``: balanced, every batch must use the landmarks in the
   proportions ``w``; ``ρ → 0``: plain entropic soft assignment). The dual
   potential ``g`` acts as a per-batch, per-landmark bias that raises the
   attractiveness of under-used landmarks only as far as the KL budget
   allows, so batches with different cell-type composition are not forced
   onto each other;
2. the free-support barycenter update (Cuturi & Doucet 2014, Alg. 2; on the
   sphere): ``y_k ∝ Σ_b n_b⁻¹ Σ_{i∈b} R_ik ẑ_i`` with the soft assignments
   ``R_ik = π_b[i,k] / Σ_k' π_b[i,k']``, and ``w_k = B⁻¹ Σ_b n_b⁻¹ Σ_i R_ik``;

followed by the correction, always computed from the *original*
coordinates ``z⁰`` (so it does not drift over rounds): with ``S_bk =
Σ_{i∈b} R_ik``, batch-wise barycentric images ``m_bk = Σ_{i∈b} R_ik z⁰_i /
S_bk`` and their batch-balanced mean ``ȳ_k = Σ_b (S_bk/n_b) m_bk /
Σ_b (S_bk/n_b)``,

    z_i = z⁰_i + Σ_k R_ik (ȳ_k - m_{b(i)k}) · S_{b(i)k} / (S_{b(i)k} + λ_r) ,

i.e. every landmark moves each batch's local barycentric image onto the
barycenter, with a ridge shrinkage ``λ_r`` (``ridge``, in cells) of offsets
that are estimated from few cells. Locally this is a translation, so
within-batch neighbourhoods are kept. Rounds repeat (plans recomputed on the
corrected, re-normalised coordinates) until the RMS change is below ``tol``
or ``n_rounds`` is reached. Every plan is ``n_b x K`` (O(nK) time and
memory per Sinkhorn iteration).

References
----------
Landa B, Zhang TTCK, Kluger Y (2022) Biwhitening reveals the rank of a count
matrix. SIAM J Math Data Sci 4(4):1420-1446.
Gavish M, Donoho DL (2017) Optimal shrinkage of singular values. IEEE Trans
Inf Theory 63(4):2137-2152.
Cuturi M (2013) Sinkhorn distances: lightspeed computation of optimal
transport. NeurIPS 26.
Cuturi M, Doucet A (2014) Fast computation of Wasserstein barycenters. ICML.
Chizat L, Peyré G, Schmitzer B, Vialard F-X (2018) Scaling algorithms for
unbalanced optimal transport problems. Math Comp 87(314):2563-2609.
Peyré G, Cuturi M (2019) Computational optimal transport. Found Trends Mach
Learn 11(5-6):355-607.
Halko N, Martinsson PG, Tropp JA (2011) Finding structure with randomness.
SIAM Rev 53(2):217-288.
"""
from __future__ import annotations

import time
from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ..core import console


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------

_K: dict = {}


def _kernels():
    if _K:
        return _K
    import numba as nb

    @nb.njit(parallel=True, cache=False)
    def seg_sum(indptr, indices, data, v):
        n = indptr.shape[0] - 1
        out = np.empty(n, dtype=np.float64)
        for i in nb.prange(n):
            s = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                s += data[p] * v[indices[p]]
            out[i] = s
        return out

    @nb.njit(parallel=True, cache=False)
    def col_sum(indptr, indices, data, v, m, nchunk):
        # out_j = Σ_i data_ij v_i  (column sums of D(v) A) without a transpose
        n = indptr.shape[0] - 1
        step = (n + nchunk - 1) // nchunk
        acc = np.zeros((nchunk, m))
        for c in nb.prange(nchunk):
            for i in range(c * step, min(n, (c + 1) * step)):
                vi = v[i]
                for p in range(indptr[i], indptr[i + 1]):
                    acc[c, indices[p]] += data[p] * vi
        out = np.zeros(m)
        for c in range(nchunk):
            out += acc[c]
        return out

    @nb.njit(parallel=True, cache=False)
    def col_sum_bin(indptr, indices, v, m, nchunk):
        # col_sum for a 0/1 matrix (data == 1 exactly): same chunking and
        # summation order as col_sum, without reading the data array
        n = indptr.shape[0] - 1
        step = (n + nchunk - 1) // nchunk
        acc = np.zeros((nchunk, m))
        for c in nb.prange(nchunk):
            for i in range(c * step, min(n, (c + 1) * step)):
                vi = v[i]
                for p in range(indptr[i], indptr[i + 1]):
                    acc[c, indices[p]] += vi
        out = np.zeros(m)
        for c in range(nchunk):
            out += acc[c]
        return out

    @nb.njit(parallel=True, cache=False)
    def seg_sum_bin(indptr, indices, v):
        n = indptr.shape[0] - 1
        out = np.empty(n, dtype=np.float64)
        for i in nb.prange(n):
            s = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                s += v[indices[p]]
            out[i] = s
        return out

    @nb.njit(parallel=True, cache=False)
    def scaled_data(indptr, indices, data, sx, sy, out):
        n = indptr.shape[0] - 1
        for i in nb.prange(n):
            for p in range(indptr[i], indptr[i + 1]):
                out[p] = data[p] * sx[i] * sy[indices[p]]

    @nb.njit(parallel=True, cache=False)
    def row_lse(C, g, logw, inv_eps, out):
        # out_i = log Σ_k exp(logw_k + (g_k - C_ik)/ε)
        n, K = C.shape
        for i in nb.prange(n):
            m = -np.inf
            for k in range(K):
                v = logw[k] + (g[k] - C[i, k]) * inv_eps
                if v > m:
                    m = v
            s = 0.0
            for k in range(K):
                s += np.exp(logw[k] + (g[k] - C[i, k]) * inv_eps - m)
            out[i] = m + np.log(s)

    @nb.njit(parallel=True, cache=False)
    def col_lse(C, f, loga, inv_eps, nchunk):
        # out_k = log Σ_i exp(loga_i + (f_i - C_ik)/ε), chunked over rows
        n, K = C.shape
        step = (n + nchunk - 1) // nchunk
        M = np.full((nchunk, K), -np.inf)
        S = np.zeros((nchunk, K))
        for c in nb.prange(nchunk):
            r0 = c * step
            r1 = min(n, r0 + step)
            for i in range(r0, r1):
                base = loga[i] + f[i] * inv_eps
                for k in range(K):
                    v = base - C[i, k] * inv_eps
                    mk = M[c, k]
                    if v > mk:
                        S[c, k] = S[c, k] * np.exp(mk - v) + 1.0
                        M[c, k] = v
                    else:
                        S[c, k] += np.exp(v - mk)
        out = np.empty(K)
        for k in range(K):
            m = -np.inf
            for c in range(nchunk):
                if M[c, k] > m:
                    m = M[c, k]
            if m == -np.inf:
                out[k] = -np.inf
                continue
            s = 0.0
            for c in range(nchunk):
                if S[c, k] > 0:
                    s += S[c, k] * np.exp(M[c, k] - m)
            out[k] = m + np.log(s)
        return out

    @nb.njit(parallel=True, cache=False)
    def plan_stats(C, f, g, loga, logw, inv_eps, Z, Y, out_rowmass, out_disp_w):
        # π_ik = exp(loga_i + logw_k + (f_i + g_k - C_ik)/ε)
        # out_rowmass_i = Σ_k π_ik ; out_disp_w = π (dense float32, row-normalised later)
        n, K = C.shape
        for i in nb.prange(n):
            s = 0.0
            for k in range(K):
                p = np.exp(loga[i] + logw[k] + (f[i] + g[k] - C[i, k]) * inv_eps)
                out_disp_w[i, k] = p
                s += p
            out_rowmass[i] = s

    @nb.njit(parallel=True, cache=False)
    def gram(indptr, indices, data, m):
        # G = S Sᵀ for a CSR S (rows x m), dense output, one scattered row per thread
        n = indptr.shape[0] - 1
        G = np.zeros((n, n))
        for i in nb.prange(n):
            buf = np.zeros(m)
            for p in range(indptr[i], indptr[i + 1]):
                buf[indices[p]] = data[p]
            for j in range(i, n):
                s = 0.0
                for p in range(indptr[j], indptr[j + 1]):
                    s += buf[indices[p]] * data[p]
                G[i, j] = s
                G[j, i] = s
        return G

    @nb.njit(parallel=True, cache=False)
    def select_count(indptr, indices, data, perm, colmap):
        n = perm.shape[0]
        cnt = np.zeros(n + 1, dtype=np.int64)
        for r in nb.prange(n):
            i = perm[r]
            c = 0
            for p in range(indptr[i], indptr[i + 1]):
                if data[p] > 0 and colmap[indices[p]] >= 0:
                    c += 1
            cnt[r + 1] = c
        return cnt

    @nb.njit(parallel=True, cache=False)
    def select_fill(indptr, indices, data, perm, colmap, out_ptr, out_idx):
        n = perm.shape[0]
        for r in nb.prange(n):
            i = perm[r]
            q = out_ptr[r]
            for p in range(indptr[i], indptr[i + 1]):
                if data[p] > 0:
                    j = colmap[indices[p]]
                    if j >= 0:
                        out_idx[q] = j
                        q += 1

    @nb.njit(parallel=True, cache=False)
    def pos_col_count(indptr, indices, data, m, nchunk):
        # number of rows with data > 0 per column (integer, exact)
        n = indptr.shape[0] - 1
        step = (n + nchunk - 1) // nchunk
        acc = np.zeros((nchunk, m), dtype=np.int64)
        for c in nb.prange(nchunk):
            for i in range(c * step, min(n, (c + 1) * step)):
                for p in range(indptr[i], indptr[i + 1]):
                    if data[p] > 0:
                        acc[c, indices[p]] += 1
        return acc.sum(axis=0)

    _K.update(pos_col_count=pos_col_count, col_sum_bin=col_sum_bin, seg_sum_bin=seg_sum_bin, select_count=select_count, select_fill=select_fill,
              gram=gram, col_sum=col_sum, seg_sum=seg_sum, scaled_data=scaled_data, row_lse=row_lse,
              col_lse=col_lse, plan_stats=plan_stats)
    return _K


def _set_threads(n_jobs):
    from ..pp._fragment_io import resolve_n_jobs
    nt = resolve_n_jobs(n_jobs)
    try:
        import numba
        numba.set_num_threads(min(nt, numba.config.NUMBA_NUM_THREADS))
    except Exception:  # pragma: no cover
        pass
    return nt


# ---------------------------------------------------------------------------
# Part A: per-batch biwhitening
# ---------------------------------------------------------------------------

def sinkhorn_scale(A: sp.csr_matrix, *, tol: float = 1e-6, max_iter: int = 10000):
    """Sinkhorn-Knopp scaling of a nonnegative sparse ``m x n`` matrix without
    empty rows/columns to row sums ``n`` and column sums ``m`` (Landa et al.
    2022, Algorithm 2). Returns ``x`` (m,), ``y`` (n,), iterations."""
    k = _kernels()
    A = sp.csr_matrix(A, dtype=np.float64)
    m, n = A.shape
    ip, ix, iv = A.indptr, A.indices, A.data
    try:
        import numba
        nchunk = 2 * numba.get_num_threads()
    except Exception:  # pragma: no cover
        nchunk = 8
    if np.any(np.diff(ip) == 0) or np.any(np.bincount(ix, minlength=n) == 0):
        raise ValueError("matrix has an empty row or column")
    x = np.ones(m)
    y = np.ones(n)
    it = 0
    for it in range(1, max_iter + 1):
        y = m / k["col_sum"](ip, ix, iv, x, n, nchunk)
        x = n / k["seg_sum"](ip, ix, iv, y)
        cs = y * k["col_sum"](ip, ix, iv, x, n, nchunk)
        if np.max(np.abs(cs / m - 1.0)) < tol:
            break
    else:
        console.warn(f"[ot_batch] biwhitening Sinkhorn did not reach tol={tol} in {max_iter} iterations")
    return x, y, it


def mp_median(gamma: float) -> float:
    """Median of the Marchenko-Pastur law with ratio ``gamma`` in (0, 1] and
    unit variance (Landa et al. 2022 Eq. (25))."""
    from scipy.integrate import quad
    from scipy.optimize import brentq
    g = float(gamma)
    if not 0 < g <= 1:
        raise ValueError("gamma must be in (0, 1]")
    lo, hi = (1 - np.sqrt(g)) ** 2, (1 + np.sqrt(g)) ** 2

    def dens(t):
        return np.sqrt(max((hi - t) * (t - lo), 0.0)) / (2 * np.pi * g * t)

    return brentq(lambda t: quad(dens, lo, t, limit=200)[0] - 0.5, lo, hi, xtol=1e-12)


def shrink_frobenius(s: np.ndarray, m: int, n: int, sigma: float = 1.0) -> np.ndarray:
    """Gavish-Donoho (2017) Frobenius-optimal shrinker, Eq. (7), for noise of
    entry s.d. ``sigma`` (rescaled by their Lemma 14)."""
    s = np.asarray(s, dtype=np.float64)
    N, M = max(m, n), min(m, n)
    beta = M / N
    c = sigma * np.sqrt(N)
    y = s / c
    out = np.zeros_like(s)
    above = y > 1 + np.sqrt(beta)
    ya = y[above]
    out[above] = c * np.sqrt(np.maximum((ya ** 2 - beta - 1) ** 2 - 4 * beta, 0.0)) / ya
    return out


def _alpha_mp(Yw: sp.csr_matrix, n_subsample: int, rng) -> float:
    """Noise variance by the MP-median rule (Landa et al. 2022 Eq. (24))."""
    A = Yw if Yw.shape[0] <= Yw.shape[1] else Yw.T.tocsr()
    a, b = A.shape
    idx = np.sort(rng.choice(a, size=min(a, int(n_subsample)), replace=False))
    S = sp.csr_matrix(A[idx], dtype=np.float64)
    G = _kernels()["gram"](S.indptr, S.indices, S.data, S.shape[1])
    ev = np.linalg.eigvalsh(G / b)
    return float(np.median(ev) / mp_median(len(idx) / b))


def biwhiten_by_batch(
    Y: sp.csr_matrix, codes: np.ndarray, *, per_batch: bool = True,
    n_subsample: int = 2000, tol: float = 1e-4, max_iter: int = 10000, seed: int = 0,
) -> tuple[sp.csr_matrix, np.ndarray, list]:
    """Biwhiten the binary CSR ``Y`` batch by batch (``per_batch=True``) or
    globally. Within a block, features open in fewer than
    ``max(5, ⌈2 ln n_b⌉)`` of its ``n_b`` cells get zero weight (the sparse
    regime in which the MP edge holds needs column degrees growing like
    ``log n``; it also keeps the Sinkhorn iteration well conditioned).
    Returns ``Ŷ`` (float32 CSR, rows in the original order, every block
    divided by ``√α_b``), the per-cell row scaling ``x`` and per-batch
    diagnostics."""
    k = _kernels()
    rng = np.random.default_rng(seed)
    n, m = Y.shape
    groups = [np.flatnonzero(codes == b) for b in np.unique(codes)] if per_batch else [np.arange(n)]
    data = np.empty(Y.nnz, dtype=np.float32)
    xs = np.empty(n)
    info = []
    try:
        import numba
        nchunk = 2 * numba.get_num_threads()
    except Exception:  # pragma: no cover
        nchunk = 8
    for rows in groups:
        contiguous = rows[-1] - rows[0] + 1 == len(rows)
        if contiguous:
            # zero-copy view of the block (cells are sorted by batch)
            a0, a1 = int(Y.indptr[rows[0]]), int(Y.indptr[rows[-1] + 1])
            ip = (Y.indptr[rows[0]:rows[-1] + 2] - a0).astype(Y.indptr.dtype)
            ix, iv = Y.indices[a0:a1], Y.data[a0:a1]
            Yb = sp.csr_matrix((iv, ix, ip), shape=(len(rows), m), copy=False)
        else:
            Yb = Y[rows]
            ip, ix, iv = Yb.indptr, Yb.indices, Yb.data
        nb_, mcb = len(rows), int(max(5, np.ceil(2 * np.log(len(rows)))))
        cnt_b = np.bincount(ix, minlength=m)
        # every cell must keep >= 1 feature: if a cell's features are all below
        # the floor, the floor of this batch is lowered to the largest value
        # that keeps every cell (= min over cells of the cell's most frequent
        # feature count)
        row_max = np.maximum.reduceat(cnt_b[ix], ip[:-1].astype(np.int64)) if len(ix) else np.zeros(nb_)
        mcb = int(min(mcb, row_max.min()))
        present = cnt_b >= mcb
        mp_ = int(present.sum())
        # Cells with deg_i * n_b < m_b cannot reach the row target m_b (each column
        # carries at most n_b): the scaling problem has no solution with them
        # (Sinkhorn diverges). They are excluded from the Sinkhorn iteration and
        # row-scaled afterwards with the batch's converged feature scaling
        # (out-of-sample cells); a no-op whenever every cell is feasible.
        deg = k["seg_sum_bin"](ip, ix, present.astype(np.float64))
        ins = deg * nb_ >= mp_
        n_in = int(ins.sum())
        if n_in == 0:
            raise ValueError("no cell of the batch satisfies the biwhitening feasibility condition")
        # masked Sinkhorn (features absent from the block get y = 0): row sums
        # = #present features, column sums of present features = #in-sample cells
        # (Y is 0/1: the *_bin kernels skip the data array; the column sums of
        # D(x)Y computed for the convergence check are reused by the next
        # iteration -- both exact, results unchanged)
        x = ins.astype(np.float64)
        y = np.zeros(m)
        it = 0
        raw = k["col_sum_bin"](ip, ix, x, m, nchunk)
        for it in range(1, max_iter + 1):
            y = np.where(present, n_in / np.maximum(raw, 1e-300), 0.0)
            rs = k["seg_sum_bin"](ip, ix, y)
            if np.any(rs[ins] <= 0):
                raise ValueError(f"{int((rs[ins] <= 0).sum())} cells have no feature open in "
                                 f">= {mcb} cells of their batch")
            x = np.where(ins, mp_ / np.maximum(rs, 1e-300), 0.0)
            raw = k["col_sum_bin"](ip, ix, x, m, nchunk)
            cs = y * raw
            if np.max(np.abs(cs[present] / n_in - 1.0)) < tol:
                break
        else:
            console.warn(f"[ot_batch] biwhitening Sinkhorn did not reach tol={tol} in {max_iter} iterations")
        if n_in < nb_:
            rs = k["seg_sum_bin"](ip, ix, y)
            x = np.where(ins, x, mp_ / np.maximum(rs, 1e-300))
        d = np.empty(Yb.nnz)
        k["scaled_data"](ip, ix, iv, np.sqrt(x), np.sqrt(y), d)
        # noise scale (Eq. 24) on the block restricted to its present features
        if nb_ <= mp_:
            idx = np.sort(rng.choice(nb_, size=min(nb_, int(n_subsample)), replace=False))
            S = sp.csr_matrix((d, ix, ip), shape=Yb.shape)[idx]
            G = k["gram"](S.indptr, S.indices, S.data, m)
            alpha = float(np.median(np.linalg.eigvalsh(G / mp_)) / mp_median(len(idx) / mp_))
        else:
            Wb = sp.csr_matrix((d, ix, ip), shape=Yb.shape)[:, np.flatnonzero(present)]
            alpha = _alpha_mp(Wb.tocsr(), n_subsample, rng)
        d *= 1.0 / np.sqrt(alpha)
        xs[rows] = x / alpha
        if contiguous:
            data[Y.indptr[rows[0]]:Y.indptr[rows[-1] + 1]] = d
        else:
            for j, r in enumerate(rows):
                data[Y.indptr[r]:Y.indptr[r + 1]] = d[ip[j]:ip[j + 1]]
        info.append(dict(n_cells=int(nb_), n_features=mp_, alpha=float(alpha),
                         sinkhorn_iter=int(it), feature_floor=int(mcb),
                         n_out_of_sample=int(nb_ - n_in)))
    Yw = sp.csr_matrix((data, Y.indices.copy(), Y.indptr.copy()), shape=Y.shape)
    Yw.eliminate_zeros()
    return Yw, xs, info


def _randomized_svd(X, k, seed, nt, n_iter=6, n_oversamples=20):
    import sklearn.utils.extmath
    from ._iterative_lsi import _ThreadedCSR
    X = sp.csr_matrix(X)
    X.sort_indices()
    k = int(min(k, min(X.shape) - 1))
    return sklearn.utils.extmath.randomized_svd(
        _ThreadedCSR.wrap(X, nt), n_components=k, n_oversamples=int(n_oversamples),
        n_iter=n_iter, random_state=seed)


def biwhitened_embedding(
    X, batch: np.ndarray, *, per_batch: bool = True, min_cells: Union[int, str] = "auto",
    k_max: int = 64, row_norm: Literal["l2", "depth", "none"] = "l2",
    shrink: bool = True, n_features: Optional[int] = None, svd_iter: int = 6,
    n_subsample: int = 2000, drop_first: bool = False, center: bool = False,
    seed: int = 0, n_jobs: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """Part A: per-batch biwhitened, MP-rank, Gavish-Donoho-shrunk joint
    embedding of a cells x features count matrix (binarised).

    ``n_features`` keeps only the most frequently open features (after the
    ``min_cells`` filter; ``None`` = all); ``svd_iter`` is the number of
    power iterations of the randomized SVD; ``n_subsample`` the row
    sub-sample of the noise-scale estimate."""
    nt = _set_threads(n_jobs)
    tm = {}
    _t = time.time()
    X = sp.csr_matrix(X)
    if not X.has_sorted_indices:
        X = X.copy()
        X.sort_indices()
    n, m0 = X.shape
    kk = _kernels()
    mc = int(max(5, np.ceil(2 * np.log(n)))) if min_cells == "auto" else int(min_cells)
    try:
        import numba
        _nch = 2 * numba.get_num_threads()
    except Exception:  # pragma: no cover
        _nch = 8
    colnnz = kk["pos_col_count"](X.indptr, X.indices, X.data, m0, _nch)
    feats = np.flatnonzero(colnnz >= mc)
    if n_features is not None and len(feats) > n_features:
        order = np.argsort(-colnnz[feats], kind="stable")[:int(n_features)]
        feats = np.sort(feats[order])
    codes = np.unique(np.asarray(batch), return_inverse=True)[1]
    # binarise, keep `feats`, and sort cells by batch (contiguous blocks) in one
    # pass; the cell order is restored at the end
    perm = np.argsort(codes, kind="stable")
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    colmap = np.full(m0, -1, dtype=np.int64)
    colmap[feats] = np.arange(len(feats))
    cnt = kk["select_count"](X.indptr, X.indices, X.data, perm, colmap)
    ptr = np.cumsum(cnt)
    itype = np.int32 if ptr[-1] < np.iinfo(np.int32).max else np.int64
    ptr = ptr.astype(itype)
    idx = np.empty(int(ptr[-1]), dtype=itype)
    kk["select_fill"](X.indptr, X.indices, X.data, perm, colmap, ptr, idx)
    Y = sp.csr_matrix((np.ones(len(idx), dtype=np.float32), idx, ptr), shape=(n, len(feats)))
    Y.has_sorted_indices = True
    codes = codes[perm]
    depth = np.diff(Y.indptr).astype(np.float64)
    if np.any(depth == 0):
        raise ValueError(f"{int((depth == 0).sum())} cells have no retained feature")
    tm["filter"] = time.time() - _t; _t = time.time()
    Yw, xs, binfo = biwhiten_by_batch(Y, codes, per_batch=per_batch, seed=seed,
                                      n_subsample=n_subsample)
    tm["biwhiten"] = time.time() - _t; _t = time.time()
    m = Y.shape[1]
    edge = np.sqrt(n) + np.sqrt(m)
    # one randomized SVD of size k_max + 1 (no restart: the cost is linear in k)
    U, s, Vt = _randomized_svd(Yw, k_max + 1, seed, nt, n_iter=svd_iter)
    r = int(np.sum(s > edge))
    r = max(1, min(r, k_max))
    s_sh = shrink_frobenius(s[:r], n, m, 1.0) if shrink else s[:r].copy()
    keep = s_sh > 0
    if not keep.any():
        keep[0] = True
        s_sh[0] = s[0]
    if drop_first:
        keep[0] = False
        if not keep.any():
            raise ValueError("only the leading (Perron) component is above the MP edge")
    Z = U[:, :r][:, keep] * s_sh[keep][None, :]
    if center:
        Z = Z - Z.mean(0, keepdims=True)
    if row_norm == "depth":
        Z = Z / (np.sqrt(xs) * depth)[:, None]
    elif row_norm == "l2":
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-300)
    elif row_norm != "none":
        raise ValueError(f"unknown row_norm {row_norm!r}")
    Z = Z[inv]
    tm["svd"] = time.time() - _t
    info = dict(rank=int(r), n_kept=int(keep.sum()), mp_edge=float(edge),
                singular_values=s, shrunk=s_sh, n_features=int(m), min_cells=mc,
                batches=binfo, timing=tm)
    return Z.astype(np.float64), info


# ---------------------------------------------------------------------------
# Part B: unbalanced entropic OT barycenter alignment
# ---------------------------------------------------------------------------

def _sqdist(Z, Y):
    zz = np.einsum("ij,ij->i", Z, Z)
    yy = np.einsum("ij,ij->i", Y, Y)
    C = zz[:, None] + yy[None, :] - 2.0 * (Z @ Y.T)
    np.maximum(C, 0.0, out=C)
    return C


def unbalanced_sinkhorn(
    C: np.ndarray, loga: np.ndarray, logw: np.ndarray, eps: float,
    rho_a: float, rho_w: float, *, n_iter: int = 1000, tol: float = 1e-5,
    g0: Optional[np.ndarray] = None, nchunk: int = 64,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Log-domain generalised Sinkhorn (Chizat et al. 2018, Alg. 2) for
    ``min ⟨C,π⟩ + ε KL(π|a⊗w) + ρ_a KL(π1|a) + ρ_w KL(πᵀ1|w)``.
    ``rho = inf`` imposes the marginal exactly. Returns the dual potentials
    ``f, g`` and the iteration count; ``π = exp(log a + log w + (f+g-C)/ε)``."""
    k = _kernels()
    n, K = C.shape
    la = 1.0 if not np.isfinite(rho_a) else rho_a / (rho_a + eps)
    lw = 1.0 if not np.isfinite(rho_w) else rho_w / (rho_w + eps)
    inv = 1.0 / eps
    C = np.ascontiguousarray(C, dtype=np.float64)
    g = np.zeros(K) if g0 is None else g0.copy()
    f = np.zeros(n)
    lse_r = np.empty(n)
    it = 0
    for it in range(1, n_iter + 1):
        k["row_lse"](C, g, logw, inv, lse_r)
        f = -la * eps * lse_r
        lse_c = k["col_lse"](C, f, loga, inv, nchunk)
        g_new = -lw * eps * lse_c
        g_new[~np.isfinite(g_new)] = 0.0
        if np.max(np.abs(g_new - g)) < tol * eps * 10 and it > 1:
            g = g_new
            break
        g = g_new
    k["row_lse"](C, g, logw, inv, lse_r)
    f = -la * eps * lse_r
    return f, g, it


def _l2n(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-300)


def ot_align(
    Z: np.ndarray, batch: np.ndarray, *, n_landmarks: Optional[int] = None,
    eps: float = 0.1, rho: float = 1.0, n_rounds: int = 10, n_bary_iter: int = 5,
    ridge: float = 1.0, tol: float = 1e-4, sinkhorn_iter: int = 200,
    sinkhorn_tol: float = 1e-4, seed: int = 0, n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Part B. Align the batches of ``Z`` (cells x dims) to an unbalanced
    entropic Wasserstein barycenter on the unit sphere and correct every cell
    by ridge-shrunk barycentric offsets (see module docstring).

    ``n_landmarks`` defaults to ``min(round(n / 30), 100)``; ``eps`` is the
    entropic regularisation in units of the squared chordal distance
    ``‖ẑ - ŷ‖² = 2(1 - cos)``; ``rho`` the KL penalty on each batch's
    landmark marginal (``inf`` = balanced); ``ridge`` the shrinkage ``λ``
    (in cells) of the offsets.
    """
    nt = _set_threads(n_jobs)
    k = _kernels()
    rng = np.random.default_rng(seed)
    Zo = np.ascontiguousarray(Z, dtype=np.float64)
    n, d = Zo.shape
    codes = np.unique(np.asarray(batch), return_inverse=True)[1]
    B = int(codes.max()) + 1
    groups = [np.flatnonzero(codes == b) for b in range(B)]
    nb = np.array([len(g) for g in groups], dtype=np.float64)
    K = int(min(round(n / 30), 100)) if n_landmarks is None else int(n_landmarks)
    K = max(2, min(K, n))
    # spherical k-means initialisation on a batch-balanced sub-sample
    from sklearn.cluster import KMeans
    Zn = _l2n(Zo)
    per = max(1, int(np.ceil(50 * K / B)))
    sub = np.concatenate([rng.choice(g, size=min(len(g), per), replace=False) for g in groups])
    km = KMeans(n_clusters=K, n_init=1, max_iter=25, random_state=seed).fit(Zn[sub])
    Yl = _l2n(km.cluster_centers_)
    w = np.full(K, 1.0 / K)
    g_prev = [None] * B
    Zc = Zo.copy()
    hist = []
    R = [None] * B
    for rnd in range(n_rounds):
        Zn = _l2n(Zc)
        for _ in range(n_bary_iter):
            num = np.zeros((K, d))
            mass = np.zeros((B, K))
            for b, rows in enumerate(groups):
                Cb = _sqdist(Zn[rows], Yl)
                loga = np.full(len(rows), -np.log(len(rows)))
                logw = np.log(np.maximum(w, 1e-300))
                f, g, it = unbalanced_sinkhorn(Cb, loga, logw, eps, np.inf, rho,
                                               n_iter=sinkhorn_iter, tol=sinkhorn_tol,
                                               g0=g_prev[b], nchunk=4 * nt)
                g_prev[b] = g
                P = np.empty_like(Cb)
                rm = np.empty(len(rows))
                k["plan_stats"](Cb, f, g, loga, logw, 1.0 / eps, Zn, Yl, rm, P)
                P /= np.maximum(rm, 1e-300)[:, None]       # soft assignment R_ik
                R[b] = P
                sb = P.sum(0) / len(rows)                  # batch landmark proportions
                mass[b] = sb
                num += (P.T @ Zn[rows]) / len(rows)        # batch weight 1/B
            Yl = _l2n(num)
            w = mass.mean(0)
            w = np.maximum(w, 1e-12)
            w /= w.sum()
        # correction from the original coordinates (no drift accumulation)
        S = np.zeros((B, K))
        M = np.zeros((B, K, d))
        for b, rows in enumerate(groups):
            S[b] = R[b].sum(0)
            M[b] = R[b].T @ Zo[rows]
        Mb = M / np.maximum(S, 1e-300)[:, :, None]              # m_bk
        q = S / nb[:, None]                                      # batch-normalised mass
        Ybar = (q[:, :, None] * Mb).sum(0) / np.maximum(q.sum(0), 1e-300)[:, None]
        shrink = S / (S + ridge)
        Znew = Zo.copy()
        for b, rows in enumerate(groups):
            D = (Ybar - Mb[b]) * shrink[b][:, None]
            D[S[b] <= 1e-12] = 0.0
            Znew[rows] += R[b] @ D
        change = float(np.sqrt(np.mean(np.sum((Znew - Zc) ** 2, 1))))
        Zc = Znew
        hist.append(dict(round=rnd + 1, rms_change=change))
        if verbose:
            console.info(f"[ot_batch] round {rnd + 1}: rms change {change:.4g}")
        if change < tol:
            break
    return Zc, dict(n_landmarks=K, history=hist, landmarks=Yl, weights=w, n_rounds_run=len(hist))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ot_batch(
    adata: AnnData,
    batch_key: str,
    *,
    layer: Optional[str] = None,
    use_rep: Optional[str] = None,
    per_batch_biwhitening: bool = True,
    min_cells: Union[int, str] = "auto",
    k_max: int = 50,
    row_norm: Literal["l2", "depth", "none"] = "l2",
    shrink: bool = True,
    n_features: Optional[int] = None,
    svd_iter: int = 3,
    n_subsample: int = 500,
    drop_first: bool = True,
    center: bool = True,
    n_landmarks: Optional[int] = None,
    eps: float = 0.05,
    rho: float = 0.01,
    n_rounds: int = 10,
    n_bary_iter: int = 2,
    ridge: float = 1.0,
    key_added: str = "X_ot_batch",
    seed: int = 0,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> AnnData:
    """Batch correction by per-batch biwhitening and unbalanced entropic OT
    barycenter alignment (see the module docstring for formulas/references).

    Parameters
    ----------
    adata
        Cells x features accessibility counts in ``.X`` (or ``layer``).
    batch_key
        ``obs`` column with the batch labels.
    use_rep
        If given, skip part A and align ``obsm[use_rep]`` directly.
    per_batch_biwhitening
        Biwhiten each batch separately (default) or the whole matrix once.
    min_cells
        Drop features open in fewer cells (``'auto'`` = ``max(5, ⌈2 ln n⌉)``).
    k_max
        Rank cap of the MP-rank SVD.
    row_norm
        Row normalisation of the part-A embedding.
    n_features
        Keep only the most frequently open features (``None`` = all).
    svd_iter, n_subsample
        Power iterations of the randomized SVD; row sub-sample of the
        noise-scale (MP-median) estimate.
    drop_first, center
        Remove the leading (Perron/scaling) component; centre the
        coordinates before the row normalisation.
    n_landmarks, eps, rho, n_rounds, n_bary_iter, ridge
        Part B: barycenter support size ``K`` (default
        ``min(round(n/30), 100)``), entropic regularisation ``ε`` (squared
        chordal distance units), KL penalty ``ρ`` on each batch's landmark
        marginal (``inf`` = balanced), maximum number of correction rounds,
        barycenter updates per round and offset ridge ``λ`` (cells).

    Returns
    -------
    ``adata`` with ``obsm[key_added]`` (corrected embedding),
    ``obsm[key_added + '_uncorrected']`` (part-A embedding) and
    ``uns[key_added]`` (diagnostics and parameters).
    """
    t0 = time.time()
    if batch_key not in adata.obs:
        raise KeyError(f"batch_key {batch_key!r} not in adata.obs")
    batch = adata.obs[batch_key].astype(str).to_numpy()
    if use_rep is None:
        X = adata.layers[layer] if layer is not None else adata.X
        Z, ainfo = biwhitened_embedding(X, batch, per_batch=per_batch_biwhitening,
                                        min_cells=min_cells, k_max=k_max, row_norm=row_norm,
                                        shrink=shrink, n_features=n_features,
                                        svd_iter=svd_iter, n_subsample=n_subsample,
                                        drop_first=drop_first, center=center,
                                        seed=seed, n_jobs=n_jobs)
        if verbose:
            console.info(f"[ot_batch] biwhitened embedding: rank {ainfo['rank']}, "
                         f"{ainfo['n_features']:,} features ({time.time() - t0:.1f}s)")
    else:
        Z = np.asarray(adata.obsm[use_rep], dtype=np.float64)
        ainfo = {}
    t1 = time.time()
    Zc, binfo = ot_align(Z, batch, n_landmarks=n_landmarks, eps=eps, rho=rho,
                         n_rounds=n_rounds, n_bary_iter=n_bary_iter, ridge=ridge,
                         seed=seed, n_jobs=n_jobs, verbose=verbose)
    adata.obsm[key_added + "_uncorrected"] = Z.astype(np.float32)
    adata.obsm[key_added] = Zc.astype(np.float32)
    adata.uns[key_added] = dict(
        embedding={kk: v for kk, v in ainfo.items() if kk != "batches"},
        batches=pd.DataFrame(ainfo.get("batches", [])),
        alignment=dict(n_landmarks=binfo["n_landmarks"], n_rounds_run=binfo["n_rounds_run"],
                       rms_change=np.array([h["rms_change"] for h in binfo["history"]])),
        params=dict(batch_key=batch_key, layer=str(layer), use_rep=str(use_rep),
                    per_batch_biwhitening=per_batch_biwhitening, min_cells=str(min_cells),
                    k_max=k_max, row_norm=row_norm, shrink=shrink,
                    n_features=str(n_features), svd_iter=svd_iter, n_subsample=n_subsample,
                    drop_first=drop_first, center=center, n_landmarks=str(n_landmarks), eps=eps, rho=rho, n_rounds=n_rounds,
                    n_bary_iter=n_bary_iter, ridge=ridge, seed=seed),
        time_embedding_s=float(t1 - t0), time_alignment_s=float(time.time() - t1),
    )
    return adata
