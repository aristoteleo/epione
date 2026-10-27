"""AEGLE: Adaptive Edge-aware Graph-regularised Latent Estimation of spatial domains.

An Epione-native spatial-domain method for spot-level spatial ATAC (optionally with RNA from
the same spots). All steps are closed form or EM; no neural network.

1. **Biwhitening.** The binary spots x peaks matrix ``Y`` is scaled to
   ``Ŷ = D(√x) Y D(√y)`` with the Sinkhorn scalings of Landa, Zhang & Kluger (2022) (as in
   BiRMT-2, :mod:`epione.tl._birmt2`), so that the noise ``E = Ŷ - E[Ŷ]`` has (approximately)
   independent entries of common variance ``α`` (their Eq. (24) estimate). RNA counts use the
   Poisson variant (variance = mean).
2. **Graph-Tikhonov smoothing before the rank decision.** With the normalised Laplacian
   ``L = I - D^{-1/2} A D^{-1/2}`` of the spot graph, ``S = (I + λL)^{-1}`` is the minimiser of
   ``‖X - Ŷ‖² + λ tr(Xᵀ L X)`` (graph-Tikhonov / graph low-pass filter; cf. GraphPCA, Yang et
   al. 2024, Genome Biol 25:287). ``S`` is applied matrix-free by conjugate gradients inside a
   randomized block-Krylov SVD (Musco & Musco 2015) of ``S Ŷ``. The columns of ``S E`` are
   i.i.d. with covariance ``α S²``, so the sample-covariance spectrum of the noise follows the
   *generalised* Marchenko-Pastur law with population spectrum ``H = eig(α S²)`` (Silverstein
   & Bai 1995; Bai & Silverstein 2010, Ch. 6), whose right edge is
   ``x(m*)`` with ``x(m) = -1/m + γ ∫ t dH(t)/(1 + t m)`` and ``x'(m*) = 0`` on
   ``(-1/max t, 0)`` (``γ = n_spots / n_peaks``). Components of ``S Ŷ`` above
   ``√(n_peaks · x(m*))`` are kept. ``H`` is obtained from the Laplacian spectrum, exactly
   (dense ``eigvalsh``) for small graphs and by stochastic Lanczos quadrature
   (Ubaru, Chen & Saad 2017) otherwise. Without smoothing (``λ = 0``) this is the plain MP
   edge ``√α(√n_spots + √n_peaks)`` of BiRMT.
3. **Edge-aware weights.** Optionally (``bilateral=True``) grid edges are re-weighted by
   ``exp(-‖z_i - z_j‖² / (2τ²))`` of a pilot embedding (τ² = median over edges), so that
   smoothing does not cross domain boundaries (bilateral filtering; Tomasi & Manduchi 1998).
4. **Spatial mixture model.** A Gaussian mixture with a common full covariance (mclust
   ``EEE``) whose E-step carries a Potts / hidden-Markov-random-field prior in mean-field form,
   ``r_ik ∝ π_k N(z_i | μ_k, Σ) exp(β Σ_j w_ij r_jk)`` (Zhang, Brady & Smith 2001; the spatial
   prior of BayesSpace, Zhao et al. 2021, and BASS, Li & Zhou 2022, at GMM cost).
5. **Label-free hyper-parameter choice** (``select=True``): the counts are split by binomial
   thinning (data thinning; Neufeld, Dharamshi, Gao & Witten 2024, JMLR 25:57) into a
   training and a test half; each candidate ``(λ, β)`` is fitted on the training half and
   scored by the held-out multinomial log-likelihood of the test counts under the domains'
   accessibility profiles; the best candidate is refitted on all counts (λ rescaled for the
   doubled depth, see ``aegle`` docstring).
6. **ATAC + RNA.** Each modality is processed by steps 1-2 and the two embeddings are
   concatenated, each block scaled to unit Frobenius norm times the square root of its
   signal excess ``Σ_k (s_k² / edge² - 1)_+``.
"""
from __future__ import annotations

import time
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..core import console

__all__ = ["aegle", "aegle_embedding", "potts_gmm", "generalized_mp_edge", "laplacian_spectrum"]


# ---------------------------------------------------------------------------
# graph and Tikhonov operator
# ---------------------------------------------------------------------------

def _sym_norm(A: sp.spmatrix):
    A = sp.csr_matrix(A, dtype=np.float64)
    A = ((A + A.T) * 0.5).tocsr()
    d = np.asarray(A.sum(1)).ravel()
    di = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-300)), 0.0)
    return (sp.diags(di) @ A @ sp.diags(di)).tocsr()


class Tikhonov:
    """``S = (I + λ L)^{-1}``, ``L = I - P``, ``P = D^{-1/2} A D^{-1/2}``; ``apply`` solves
    ``(I + λL) X = B`` column-wise by conjugate gradients (SPD, condition number ≤ 1 + 2λ)."""

    def __init__(self, A: sp.spmatrix, lam: float, tol: float = 1e-6, max_iter: int = 200):
        self.P = _sym_norm(A)
        self.lam = float(lam)
        self.tol, self.max_iter = tol, max_iter
        self.n_matvec = 0

    def _op(self, X):
        return (1.0 + self.lam) * X - self.lam * (self.P @ X)

    def apply(self, B):
        B = np.asarray(B, dtype=np.float64)
        if self.lam == 0:
            return B.copy()
        one = B.ndim == 1
        if one:
            B = B[:, None]
        X = B / (1.0 + self.lam)             # Jacobi start
        R = B - self._op(X)
        Pd = R.copy()
        rr = np.einsum("ij,ij->j", R, R)
        b2 = np.maximum(np.einsum("ij,ij->j", B, B), 1e-300)
        for _ in range(self.max_iter):
            if np.all(rr <= (self.tol ** 2) * b2):
                break
            Q = self._op(Pd)
            self.n_matvec += 1
            a = rr / np.maximum(np.einsum("ij,ij->j", Pd, Q), 1e-300)
            X += Pd * a
            R -= Q * a
            rr_new = np.einsum("ij,ij->j", R, R)
            Pd = R + Pd * (rr_new / np.maximum(rr, 1e-300))
            rr = rr_new
        return X[:, 0] if one else X


def laplacian_spectrum(A: sp.spmatrix, *, exact_max: int = 6000, n_probe: int = 24,
                       n_steps: int = 60, seed: int = 0):
    """Eigenvalues ``μ`` and weights ``w`` (summing to 1) of the spectral distribution of the
    normalised Laplacian ``L = I - D^{-1/2} A D^{-1/2}``: exact for ``n <= exact_max``,
    otherwise stochastic Lanczos quadrature with ``n_probe`` Rademacher probes and
    ``n_steps`` Lanczos steps (full reorthogonalisation; Ubaru, Chen & Saad 2017)."""
    P = _sym_norm(A)
    n = P.shape[0]
    if n <= exact_max:
        mu = 1.0 - np.linalg.eigvalsh(P.toarray())
        return np.clip(mu, 0.0, 2.0), np.full(n, 1.0 / n)
    rng = np.random.default_rng(seed)
    nodes, weights = [], []
    for _ in range(n_probe):
        v = rng.choice([-1.0, 1.0], size=n)
        v /= np.linalg.norm(v)
        V = np.zeros((n, n_steps))
        al, be = np.zeros(n_steps), np.zeros(n_steps)
        V[:, 0] = v
        k_used = n_steps
        for k in range(n_steps):
            w = P @ V[:, k]
            al[k] = V[:, k] @ w
            w -= V[:, :k + 1] @ (V[:, :k + 1].T @ w)       # full reorthogonalisation
            w -= V[:, :k + 1] @ (V[:, :k + 1].T @ w)
            if k + 1 < n_steps:
                be[k] = np.linalg.norm(w)
                if be[k] < 1e-10:
                    k_used = k + 1
                    break
                V[:, k + 1] = w / be[k]
        T = np.diag(al[:k_used]) + np.diag(be[:k_used - 1], 1) + np.diag(be[:k_used - 1], -1)
        th, ev = np.linalg.eigh(T)
        nodes.append(1.0 - th)
        weights.append(ev[0] ** 2 / n_probe)
    mu = np.clip(np.concatenate(nodes), 0.0, 2.0)
    w = np.concatenate(weights)
    return mu, w / w.sum()


def generalized_mp_edge(t: np.ndarray, w: np.ndarray, gamma: float) -> float:
    """Right edge of the limiting spectral distribution of ``(1/N) X Xᵀ`` where the ``N``
    columns of the ``p x N`` matrix ``X`` are i.i.d. with population covariance spectrum
    ``H = Σ_k w_k δ_{t_k}`` and ``γ = p / N`` (Silverstein & Choi 1995: the edge is
    ``x(m*)``, ``x(m) = -1/m + γ Σ_k w_k t_k / (1 + t_k m)``, where ``x'(m*) = 0`` on
    ``(-1/max t, 0)``). For ``H = δ_α`` it returns ``α (1 + √γ)²``."""
    t = np.asarray(t, np.float64)
    w = np.asarray(w, np.float64)
    w = w / w.sum()
    tmax = t.max()

    def dx(m):
        return 1.0 / m ** 2 - gamma * np.sum(w * t ** 2 / (1.0 + t * m) ** 2)

    lo, hi = -1.0 / tmax * (1 - 1e-12), -1e-12 / tmax
    # dx(lo) < 0 (pole at -1/tmax if the top mass is positive), dx(hi) > 0
    if dx(lo) > 0:     # numerically flat top: fall back to the support's upper end
        lo = -1.0 / tmax * (1 - 1e-6)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dx(mid) > 0:
            hi = mid
        else:
            lo = mid
    m = 0.5 * (lo + hi)
    return float(-1.0 / m + gamma * np.sum(w * t / (1.0 + t * m)))


# ---------------------------------------------------------------------------
# biwhitening (binary ATAC and Poisson RNA)
# ---------------------------------------------------------------------------

class _Biwhitened:
    """Matrix-free ``Ŷ = D(√x) Y D(√y)`` for a sparse non-negative ``Y`` (CSR) with ``α``."""

    def __init__(self, Y, x, y, alpha, depth, binary_op=None):
        self.Y = Y
        self.shape = Y.shape
        self.sx, self.sy = np.sqrt(x), np.sqrt(y)
        self.alpha = float(alpha)
        self.depth = depth
        self.bop = binary_op
        if binary_op is None:
            self.Yt = Y.T.tocsr()

    def matmat(self, X):
        if self.bop is not None:
            return self.bop.matmat(X)
        return self.sx[:, None] * (self.Y @ (self.sy[:, None] * X))

    def rmatmat(self, X):
        if self.bop is not None:
            return self.bop.rmatmat(X)
        return self.sy[:, None] * (self.Yt @ (self.sx[:, None] * X))


def _alpha_mp(rows_dense: np.ndarray, n_cols: int) -> float:
    from ._birmt import mp_median
    G = rows_dense @ rows_dense.T / n_cols
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1, user_api="blas"):
        ev = np.linalg.eigvalsh((G + G.T) / 2)
    g = rows_dense.shape[0] / n_cols
    return float(np.median(ev) / mp_median(min(g, 1.0)))


def biwhiten_binary(X: sp.spmatrix, *, n_subsample: int = 2000, seed: int = 0) -> _Biwhitened:
    """BiRMT-2 biwhitening of the binarised matrix (numba pattern kernels); rows too sparse
    for the Sinkhorn constraint get ``x_i`` from the row equation (out-of-sample)."""
    from . import _birmt2 as B2
    Y = sp.csr_matrix(X)
    Yb = sp.csr_matrix((np.ones(Y.nnz, np.float32), Y.indices, Y.indptr), shape=Y.shape)
    Yb.eliminate_zeros()
    m, n = Yb.shape
    depth = np.diff(Yb.indptr).astype(np.float64)
    if np.any(depth == 0):
        raise ValueError("some spots have no open feature")
    low = depth < n / m * 1.05
    fit = np.flatnonzero(~low)
    Yf = Yb[fit] if low.any() else Yb
    P = B2.as_pattern(Yf)
    xf, y, _ = B2.sinkhorn_binary(P, P.transpose())
    x = np.empty(m)
    x[fit] = xf
    if low.any():
        x[low] = n / np.maximum(Yb[low] @ y, 1e-300)
    Pf = B2.as_pattern(Yb)
    bop = B2.BinaryBiwhitened(Pf, x, y, Pf.transpose())
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(fit, size=min(len(fit), n_subsample), replace=False))
    alpha = _alpha_mp(bop.rows(idx), n)
    return _Biwhitened(Yb, x, y, alpha, depth, binary_op=bop)


def biwhiten_poisson(X: sp.spmatrix, *, min_spots: int = 10, n_subsample: int = 2000,
                     seed: int = 0, tol: float = 1e-6, max_iter: int = 5000):
    """Poisson biwhitening of a count matrix (Landa et al. 2022, Algorithm 1 with the
    variance estimate ``V = Y``; ``α`` by their Eq. (24)). Genes detected in fewer than
    ``min_spots`` spots are dropped. Returns the operator and the kept gene index."""
    Y = sp.csr_matrix(X, dtype=np.float64)
    keep = np.flatnonzero(np.asarray((Y > 0).sum(0)).ravel() >= min_spots)
    Y = Y[:, keep].tocsr()
    m, n = Y.shape
    Yt = Y.T.tocsr()
    x = np.ones(m)
    for _ in range(max_iter):
        y = m / np.maximum(Yt @ x, 1e-300)
        x = n / np.maximum(Y @ y, 1e-300)
        if np.max(np.abs(y * (Yt @ x) / m - 1.0)) < tol:
            break
    depth = np.asarray(Y.sum(1)).ravel()
    op = _Biwhitened(Y, x, y, 1.0, depth)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(m, size=min(m, n_subsample), replace=False))
    rows = (sp.diags(op.sx[idx]) @ Y[idx] @ sp.diags(op.sy)).toarray()
    op.alpha = _alpha_mp(rows, n)
    return op, keep


# ---------------------------------------------------------------------------
# smoothed spectral embedding with the generalised-MP edge
# ---------------------------------------------------------------------------

class _SmoothedOp:
    def __init__(self, bw, S):
        self.bw, self.S = bw, S
        self.shape = bw.shape

    def matmat(self, X):
        return self.S.apply(self.bw.matmat(X))

    def rmatmat(self, X):
        return self.bw.rmatmat(self.S.apply(X))


def _edge_bilateral(A: sp.spmatrix, Z: np.ndarray, scale: float = 1.0):
    A = sp.coo_matrix(A)
    m = A.row != A.col
    r, c = A.row[m], A.col[m]
    d2 = np.sum((Z[r] - Z[c]) ** 2, axis=1)
    tau2 = max(np.median(d2), 1e-300) * scale
    w = np.exp(-d2 / (2 * tau2))
    W = sp.csr_matrix((w, (r, c)), shape=A.shape)
    return ((W + W.T) * 0.5).tocsr()


def smoothed_embedding(bw, A, lam: float, *, spectrum=None, k_max: int = 64, block: int = 16,
                       weight: Literal["excess", "raw", "sqrt_excess"] = "excess",
                       row_norm: Literal["none", "l2", "depth"] = "none", seed: int = 0):
    """Block-Krylov SVD of ``S Ŷ`` and the generalised-MP rank decision (module step 2).

    Returns ``dict(Z, rank, edge, s, excess)``: ``Z = U_r diag(w_r)`` with
    ``w = √(s² - edge²)`` (``weight="excess"``; the eigenvalue excess, which is the debiased
    signal variance under the spiked model at first order) or ``w = s`` (``"raw"``)."""
    from ._birmt2 import block_krylov_svd
    m, n = bw.shape
    S = Tikhonov(A, lam)
    op = _SmoothedOp(bw, S)
    if spectrum is None:
        spectrum = laplacian_spectrum(A, seed=seed) if lam > 0 else (np.zeros(1), np.ones(1))
    mu, wmu = spectrum
    t = bw.alpha / (1.0 + lam * mu) ** 2
    if m <= n:
        lam_edge = generalized_mp_edge(t, wmu, m / n)      # eigenvalue edge of (1/n) N Nᵀ
        edge = float(np.sqrt(n * lam_edge))
    else:  # spots > features: the companion matrix has the same non-zero spectrum
        lam_edge = generalized_mp_edge(t, wmu, m / n)
        edge = float(np.sqrt(n * lam_edge))
    nv = 48
    while True:
        U, s = block_krylov_svd(op, nv, block=block, seed=seed)
        r = int(np.sum(s > edge))
        if r <= nv - block or nv >= 2 * k_max:
            break
        nv = min(2 * nv, 2 * k_max)
    r = max(1, min(r, k_max))
    s_r = s[:r]
    if weight == "raw":
        w = s_r
    elif weight == "sqrt_excess":
        w = np.sqrt(np.sqrt(np.maximum(s_r ** 2 - edge ** 2, 0.0)) * s_r)
    else:
        w = np.sqrt(np.maximum(s_r ** 2 - edge ** 2, 0.0))
    Z = U[:, :r] * w[None, :]
    if row_norm == "l2":
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-300)
    elif row_norm == "depth":
        Z = Z / S.apply(bw.sx * bw.depth / np.mean(bw.sx * bw.depth))[:, None]
    excess = float(np.sum(np.maximum(s_r ** 2 / edge ** 2 - 1.0, 0.0)))
    return dict(Z=Z, rank=r, edge=edge, s=s, excess=excess, n_matvec=S.n_matvec)


# ---------------------------------------------------------------------------
# Potts-prior Gaussian mixture (mean-field EM)
# ---------------------------------------------------------------------------

def _row_stochastic(A):
    A = sp.csr_matrix(A, dtype=np.float64)
    d = np.asarray(A.sum(1)).ravel()
    return (sp.diags(np.where(d > 0, 1.0 / np.maximum(d, 1e-300), 0.0)) @ A).tocsr()


def _log_gauss_shared(Z, mu, Sigma):
    d = Z.shape[1]
    Lc = np.linalg.cholesky(Sigma)
    from scipy.linalg import solve_triangular
    Zw = solve_triangular(Lc, Z.T, lower=True).T
    Mw = solve_triangular(Lc, mu.T, lower=True).T
    maha = (Zw ** 2).sum(1)[:, None] + (Mw ** 2).sum(1)[None, :] - 2.0 * Zw @ Mw.T
    logdet = 2 * np.sum(np.log(np.diag(Lc)))
    return -0.5 * (np.maximum(maha, 0.0) + logdet + d * np.log(2 * np.pi))


def potts_gmm(Z: np.ndarray, K: int, A: Optional[sp.spmatrix] = None, *, beta: float = 0.0,
              n_init: int = 10, max_iter: int = 200, n_mf: int = 3, tol: float = 1e-6,
              seed: int = 0, reg: float = 1e-6):
    """Gaussian mixture with one shared full covariance (mclust ``EEE``) and a mean-field
    Potts prior: ``r_ik ∝ π_k N(z_i|μ_k,Σ) exp(β Σ_j a_ij r_jk)`` with ``a`` the
    row-stochastic spot graph (``n_mf`` mean-field sweeps per E-step). ``beta=0`` is a plain
    EEE GMM. Initialised by k-means (``n_init`` restarts); returns ``(labels, resp, info)``."""
    from sklearn.cluster import KMeans
    Z = np.asarray(Z, np.float64)
    n, d = Z.shape
    Pm = _row_stochastic(A) if (A is not None and beta > 0) else None
    lab0 = KMeans(n_clusters=K, n_init=n_init, random_state=seed).fit_predict(Z)
    R = np.eye(K)[lab0]
    ZtZ = Z.T @ Z
    prev = -np.inf
    ll = prev
    for it in range(max_iter):
        Nk = R.sum(0) + 1e-10
        pi = Nk / n
        mu = (R.T @ Z) / Nk[:, None]
        Sig = (ZtZ - mu.T @ (Nk[:, None] * mu)) / n
        Sig += np.eye(d) * reg * max(np.trace(Sig) / d, 1e-12)
        logp = _log_gauss_shared(Z, mu, Sig) + np.log(pi)[None, :]
        if Pm is not None:
            Rm = R
            for _ in range(n_mf):
                field = logp + beta * (Pm @ Rm)
                field -= field.max(1, keepdims=True)
                Rm = np.exp(field)
                Rm /= Rm.sum(1, keepdims=True)
            R = Rm
            mx = logp.max(1, keepdims=True)
            ll = float(np.sum(mx[:, 0] + np.log(np.exp(logp - mx).sum(1))))
        else:
            mx = logp.max(1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(logp - mx).sum(1))
            R = np.exp(logp - lse[:, None])
            ll = float(lse.sum())
        if abs(ll - prev) < tol * abs(ll):
            break
        prev = ll
    lab = R.argmax(1)
    return lab, R, dict(n_iter=it + 1, loglik=ll)


# ---------------------------------------------------------------------------
# count-splitting selection
# ---------------------------------------------------------------------------

def _thin(X: sp.csr_matrix, eps: float, rng):
    X = sp.csr_matrix(X)
    v = X.data
    if np.allclose(v, np.round(v)):
        tr = rng.binomial(v.astype(np.int64), eps).astype(np.float64)
    else:   # non-integer values: thin the pattern
        tr = (rng.random(v.size) < eps).astype(np.float64)
        v = np.ones_like(v)
    Xtr = sp.csr_matrix((tr, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    Xte = sp.csr_matrix((v - tr, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    Xtr.eliminate_zeros(); Xte.eliminate_zeros()
    return Xtr, Xte


def heldout_loglik(Xtr: sp.csr_matrix, Xte: sp.csr_matrix, lab: np.ndarray, pseudo: float = 0.5) -> float:
    """Mean held-out multinomial log-likelihood per test count of ``Xte`` under the
    domain accessibility profiles estimated from ``Xtr`` (labels ``lab``)."""
    lab = np.asarray(lab)
    ks, inv = np.unique(lab, return_inverse=True)
    M = sp.csr_matrix((np.ones(len(lab)), (inv, np.arange(len(lab)))), shape=(len(ks), len(lab)))
    C = np.asarray((M @ Xtr).todense())
    Pk = (C + pseudo) / (C.sum(1, keepdims=True) + pseudo * C.shape[1])
    logP = np.log(Pk)
    Xte = sp.csr_matrix(Xte)
    rows = np.repeat(np.arange(Xte.shape[0]), np.diff(Xte.indptr))
    tot = float(np.sum(Xte.data * logP[inv[rows], Xte.indices]))
    return tot / max(Xte.data.sum(), 1.0)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def _embed_all(mods, A, lam, spectrum, *, bilateral, bilateral_scale, weight, row_norm, fusion,
               k_max, seed, hops):
    """mods: list of biwhitened operators. Returns (Z, info, A_used)."""
    A_used = A
    infos = []
    if bilateral:
        pil = [smoothed_embedding(bw, A, lam, spectrum=spectrum, k_max=k_max, weight=weight,
                                  row_norm=row_norm, seed=seed) for bw in mods]
        Zp = np.hstack([p["Z"] / max(np.linalg.norm(p["Z"]), 1e-300) for p in pil])
        A_used = _edge_bilateral(A, Zp, bilateral_scale)
        spectrum = laplacian_spectrum(A_used, seed=seed) if lam > 0 else spectrum
    blocks = []
    for bw in mods:
        e = smoothed_embedding(bw, A_used, lam, spectrum=spectrum, k_max=k_max, weight=weight,
                               row_norm=row_norm, seed=seed)
        infos.append(dict(rank=e["rank"], edge=e["edge"], excess=e["excess"], alpha=bw.alpha))
        Zb = e["Z"] / max(np.linalg.norm(e["Z"]), 1e-300)
        if len(mods) > 1 and fusion == "excess":
            Zb = Zb * np.sqrt(e["excess"])
        blocks.append(Zb)
    Z = np.hstack(blocks)
    if hops > 0:
        Pm = _row_stochastic(A_used)
        parts, cur = [Z], Z
        for _ in range(hops):
            cur = Pm @ cur
            parts.append(cur)
        Z = np.hstack(parts)
    return Z, infos, A_used


def aegle_embedding(adata, *, rna=None, lam: float = 4.0, graph_key: str = "spatial",
                    layer: Optional[str] = None, bilateral: bool = False, bilateral_scale: float = 1.0,
                    weight: str = "excess", row_norm: str = "none", fusion: str = "excess",
                    hops: int = 0, k_max: int = 64, seed: int = 0, key_added: str = "X_aegle"):
    """AEGLE embedding only (module steps 1-3, 6); stored in ``obsm[key_added]``."""
    A = _get_graph(adata, graph_key)
    X = adata.layers[layer] if layer is not None else adata.X
    mods = [biwhiten_binary(X, seed=seed)]
    if rna is not None:
        mods.append(biwhiten_poisson(rna.X, seed=seed)[0])
    spectrum = laplacian_spectrum(A, seed=seed) if lam > 0 else None
    Z, infos, _ = _embed_all(mods, A, lam, spectrum, bilateral=bilateral, bilateral_scale=bilateral_scale,
                             weight=weight, row_norm=row_norm, fusion=fusion, k_max=k_max, seed=seed,
                             hops=hops)
    adata.obsm[key_added] = Z.astype(np.float32)
    adata.uns[key_added] = dict(lam=lam, modalities=infos)
    return Z


def _get_graph(adata, graph_key):
    k = f"{graph_key}_connectivities"
    if k not in adata.obsp:
        from ._spatial import spatial_neighbors
        coord = "grid" if {"array_row", "array_col"} <= set(adata.obs.columns) else "generic"
        spatial_neighbors(adata, coord_type=coord, n_neighs=4 if coord == "grid" else 6, key_added=graph_key)
    return adata.obsp[k]


def aegle(
    adata,
    n_domains: int,
    *,
    rna=None,
    lam: float | Sequence[float] = 8.0,
    beta: float | Sequence[float] = 2.0,
    select: bool = False,
    lam_grid: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0),
    beta_grid: Sequence[float] = (0.0, 1.0, 2.0),
    thin_eps: float = 0.5,
    graph_key: str = "spatial",
    layer: Optional[str] = None,
    bilateral: bool = True,
    bilateral_scale: float = 1.0,
    weight: str = "excess",
    row_norm: str = "depth",
    fusion: str = "excess",
    hops: int = 1,
    k_max: int = 64,
    n_init: int = 10,
    seed: int = 0,
    key_added: str = "domain",
    rep_key: str = "X_aegle",
):
    """AEGLE spatial domains (see the module docstring for the method and references).

    Arguments:
        adata: spots x peaks AnnData (counts or binary in ``X`` / ``layer``) with
            ``obsm['spatial']`` (and ``obs['array_row'/'array_col']`` for a DBiT grid). The
            spot graph ``obsp[f'{graph_key}_connectivities']`` is built (4-neighbour grid)
            if absent.
        n_domains: number of domains ``K``.
        rna: optional AnnData of RNA counts for the same spots (same order): ATAC + RNA mode.
        lam: Tikhonov strength λ (cut-off of the graph low-pass filter ``1/(1+λμ)``).
        beta: Potts prior weight β of the mixture model.
        select: choose ``(λ, β)`` from ``lam_grid`` x ``beta_grid`` by count splitting
            (``thin_eps`` of the counts for training); λ is refitted on all counts as
            ``λ_sel · thin_eps`` (Tikhonov's optimal λ scales with the noise variance, which
            halves when the depth doubles).
        bilateral: re-weight grid edges by a pilot embedding (edge-aware smoothing).
        hops: append ``hops`` neighbour-averaged copies of the embedding (BANKSY/CellCharter).

    Returns:
        ``obs[key_added]`` (categorical, 1..K); embedding in ``obsm[rep_key]``; parameters,
        ranks and timings in ``uns[rep_key]``.
    """
    t0 = time.time()
    A = _get_graph(adata, graph_key)
    X = sp.csr_matrix(adata.layers[layer] if layer is not None else adata.X)
    Xr = sp.csr_matrix(rna.X) if rna is not None else None
    opts = dict(bilateral=bilateral, bilateral_scale=bilateral_scale, weight=weight, row_norm=row_norm,
                fusion=fusion, k_max=k_max, seed=seed, hops=hops)
    sel_table = None
    lam_f, beta_f = float(np.atleast_1d(lam)[0]), float(np.atleast_1d(beta)[0])
    spectrum = laplacian_spectrum(A, seed=seed)
    if select:
        rng = np.random.default_rng(seed)
        Xtr, Xte = _thin(X, thin_eps, rng)
        keep = np.diff(Xtr.indptr) > 0
        if not keep.all():   # spots empty after thinning: give them one count back
            raise ValueError("a spot has no training count after thinning; lower thin_eps")
        mods = [biwhiten_binary(Xtr, seed=seed)]
        if Xr is not None:
            Rtr, _ = _thin(Xr, thin_eps, rng)
            mods.append(biwhiten_poisson(Rtr, seed=seed)[0])
        rows = []
        for lm in lam_grid:
            Z, infos, A_used = _embed_all(mods, A, float(lm), spectrum, **opts)
            for b in beta_grid:
                lab, _, _ = potts_gmm(Z, n_domains, A_used, beta=float(b), n_init=n_init, seed=seed)
                rows.append(dict(lam=float(lm), beta=float(b), loglik=heldout_loglik(Xtr, Xte, lab),
                                 rank=infos[0]["rank"]))
        sel_table = pd.DataFrame(rows)
        best = sel_table.loc[sel_table["loglik"].idxmax()]
        lam_f, beta_f = float(best["lam"]) * thin_eps, float(best["beta"])
    mods = [biwhiten_binary(X, seed=seed)]
    if Xr is not None:
        mods.append(biwhiten_poisson(Xr, seed=seed)[0])
    Z, infos, A_used = _embed_all(mods, A, lam_f, spectrum, **opts)
    lab, R, ginfo = potts_gmm(Z, n_domains, A_used, beta=beta_f, n_init=n_init, seed=seed)
    adata.obsm[rep_key] = Z.astype(np.float32)
    adata.obs[key_added] = pd.Categorical(lab + 1, categories=list(range(1, n_domains + 1)))
    adata.uns[rep_key] = dict(lam=lam_f, beta=beta_f, modalities=infos, gmm=ginfo,
                              selection=None if sel_table is None else sel_table.to_dict("list"),
                              time_s=float(time.time() - t0))
    console.info(f"[aegle] K={n_domains} lam={lam_f:g} beta={beta_f:g} ranks="
                 f"{[i['rank'] for i in infos]} ({time.time() - t0:.1f}s)")
    return adata.obs[key_added]
