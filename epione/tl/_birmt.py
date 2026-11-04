"""BiRMT: biwhitened random-matrix embedding + doubly-stochastic diffusion
map + Markov-stability clustering for scATAC-seq.

The method chains four published, parameter-light estimators:

1. **Biwhitening and Marchenko-Pastur rank** (Landa, Zhang & Kluger 2022).
   For a count matrix ``Y`` (cells x features) whose entries are independent
   with a quadratic variance function, find positive scalings ``x`` (cells)
   and ``y`` (features) with the Sinkhorn-Knopp algorithm such that
   ``D(x) Var D(y)`` has row sums ``n`` and column sums ``m`` (unit average
   noise variance in every row and column; their Algorithm 1 and 2). The
   noise of ``Ŷ = D(x)^½ Y D(y)^½`` then follows the Marchenko-Pastur (MP)
   law with variance 1 and the rank is the number of singular values above
   ``√m + √n``. For a *binary* matrix every quadratic variance estimator of
   their Eq. (23), ``α[(1-β)Y + βY²]``, equals ``αY``: biwhitening is the
   Sinkhorn scaling of ``Y`` itself, and the only free quantity is the global
   noise scale ``α``, which is set by their Eq. (24) (median eigenvalue
   matched to the MP median; Gavish & Donoho 2014, Eq. (4)/(51) of 2017).
   The edge becomes ``√α (√m + √n)``.
2. **Optimal singular-value shrinkage** (Gavish & Donoho 2017). With
   ``N = max(m, n)``, ``β = min(m, n)/N`` and ``σ = √α``, each singular value
   ``s`` is mapped to ``σ√N · η(s / (σ√N))`` with the Frobenius-loss optimal
   shrinker ``η(y) = √((y² - β - 1)² - 4β) / y`` for ``y > 1 + √β`` (0
   otherwise; their Eq. (7) and Lemma 14). Alternatives: operator-loss
   shrinker (their Eq. (8)) or the optimal hard threshold
   ``λ*(β)σ√N`` of Gavish & Donoho 2014, Eq. (11).
   The cell coordinates ``U S̃`` are taken back out of the row scaling and
   divided by the cell's total count (``row_norm='depth'``), i.e. each cell
   becomes its denoised accessibility profile in the biwhitened feature
   metric, the analogue of the per-cell normalisation used before the
   kernel in Landa, Coifman & Kluger 2021 (Sec. 3.2).
3. **Doubly-stochastic Gaussian kernel and diffusion map** (Landa, Coifman &
   Kluger 2021; Coifman & Lafon 2006). Gaussian kernel
   ``K_ij = exp(-‖z_i - z_j‖²/ε)`` on the symmetrised kNN graph with zero
   main diagonal, scaled by the symmetric Sinkhorn-Knopp algorithm (their
   Algorithm 1, run in the log domain) to ``W = D(d) K D(d)`` with unit row
   and column sums -- the normalisation that is robust to heteroskedastic
   noise (low-depth cells). ``W`` is a symmetric Markov matrix with uniform
   stationary distribution, so its eigenvectors give diffusion coordinates
   ``Ψ_t(i) = (λ_k^t φ_k(i))_k`` whose Euclidean distance is the diffusion
   distance at time ``t``.
   *Bandwidth:* ``ε`` = median over cells of the squared distance to the
   ``n_neighbors``-th neighbour. *Number of coordinates:* the MP rank ``r``
   (non-trivial eigenvectors, the constant one dropped). *Diffusion time
   (spectral-gap rule):* with ``λ_1 ≥ … ≥ λ_r`` the non-trivial eigenvalues,
   ``q = argmax_k (λ_k - λ_{k+1})`` (first on ties) and
   ``t = max(1, ⌈log δ / log(λ_{q+1}/λ_q)⌉)`` (capped at ``t_max``): the
   smallest integer time at which the component just after the largest
   spectral gap is attenuated by the factor ``δ`` relative to the one just
   before it (Coifman & Lafon's δ-truncation applied at the gap).
4. **Resolution by Markov stability** (Delvenne, Yaliraki & Barahona 2010;
   Lambiotte, Delvenne & Barahona 2014). Because ``W`` is doubly stochastic
   (every node has strength 1, the stationary distribution is uniform),
   Leiden with the RB-configuration quality and resolution ``γ`` on ``W``
   optimises the linearised Markov stability at Markov time ``1/γ``. A
   log-spaced grid of resolutions is scanned with ``n_seeds`` Leiden seeds
   each; the selected resolution minimises the mean pairwise normalised
   variation of information ``NVI = VI / H(P, Q)`` across seeds among grid
   points whose partitions have at least two clusters (ties within
   ``1e-9`` -> the larger resolution, i.e. the finer partition). The
   returned partition is the seed partition with the smallest mean NVI to
   the others (first on ties).

Output (frozen default, selected on non-benchmark development data before
any benchmark run): ``obsm['X_birmt']`` holds the step-2 coordinates (the
denoised biwhitened profiles); the step-3 doubly-stochastic kernel is the
Markov chain used by step 4, and the step-3 diffusion coordinates are stored
in ``obsm['X_birmt_diffmap']`` (``embedding='diffusion'`` exports them as the
main embedding instead). On the development sets the diffusion coordinates
scored lower than the step-2 coordinates with the benchmark's evaluation.

References
----------
Landa B, Zhang TTCK, Kluger Y (2022) Biwhitening reveals the rank of a count
matrix. SIAM J Math Data Sci 4(4):1420-1446.
Gavish M, Donoho DL (2014) The optimal hard threshold for singular values is
4/√3. IEEE Trans Inf Theory 60(8):5040-5053.
Gavish M, Donoho DL (2017) Optimal shrinkage of singular values. IEEE Trans
Inf Theory 63(4):2137-2152.
Landa B, Coifman RR, Kluger Y (2021) Doubly stochastic normalization of the
Gaussian kernel is robust to heteroskedastic noise. SIAM J Math Data Sci
3(1):388-413.
Coifman RR, Lafon S (2006) Diffusion maps. Appl Comput Harmon Anal
21(1):5-30.
Delvenne J-C, Yaliraki SN, Barahona M (2010) Stability of graph communities
across time scales. PNAS 107(29):12755-12760.
Lambiotte R, Delvenne J-C, Barahona M (2014) Random walks, Markov processes
and the multiscale modular organization of complex networks. IEEE Trans Netw
Sci Eng 1(2):76-90.
Knight PA (2008) The Sinkhorn-Knopp algorithm: convergence and applications.
SIAM J Matrix Anal Appl 30(1):261-275.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from ..core import console


# ---------------------------------------------------------------------------
# numba kernels (binary structure only: the matrix values are all 1)
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
    def scaled_data(indptr, indices, data, sx, sy, out):
        n = indptr.shape[0] - 1
        for i in nb.prange(n):
            for p in range(indptr[i], indptr[i + 1]):
                out[p] = data[p] * sx[i] * sy[indices[p]]

    @nb.njit(parallel=True, cache=False)
    def row_lse(indptr, indices, logk, u, out):
        n = indptr.shape[0] - 1
        for i in nb.prange(n):
            a, b = indptr[i], indptr[i + 1]
            m = -np.inf
            for p in range(a, b):
                v = logk[p] + u[indices[p]]
                if v > m:
                    m = v
            s = 0.0
            for p in range(a, b):
                s += np.exp(logk[p] + u[indices[p]] - m)
            out[i] = m + np.log(s)

    _K.update(seg_sum=seg_sum, scaled_data=scaled_data, row_lse=row_lse)
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
# Step 1: biwhitening + MP rank
# ---------------------------------------------------------------------------

def sinkhorn_scale(
    A: sp.csr_matrix, *, tol: float = 1e-6, max_iter: int = 10000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sinkhorn-Knopp scaling of a nonnegative sparse ``m x n`` matrix to row
    sums ``n`` and column sums ``m`` (Landa et al. 2022, Algorithm 2).

    Returns ``x`` (m,), ``y`` (n,) and the number of iterations. Converged
    when every row and column sum is within relative ``tol`` of its target.
    """
    k = _kernels()
    A = sp.csr_matrix(A, dtype=np.float64)
    A.eliminate_zeros()
    m, n = A.shape
    At = A.T.tocsr()
    ip, ix, iv = A.indptr, A.indices, A.data
    tp, tx, tv = At.indptr, At.indices, At.data
    if np.any(np.diff(ip) == 0) or np.any(np.diff(tp) == 0):
        raise ValueError("matrix has an empty row or column; filter it first")
    x = np.ones(m)
    y = np.ones(n)
    it = 0
    for it in range(1, max_iter + 1):
        y = m / k["seg_sum"](tp, tx, tv, x)
        x = n / k["seg_sum"](ip, ix, iv, y)
        # rows are exact after the x update; check the columns
        cs = y * k["seg_sum"](tp, tx, tv, x)
        if np.max(np.abs(cs / m - 1.0)) < tol:
            break
    else:
        console.warn(f"[birmt] Sinkhorn (biwhitening) did not reach tol={tol} "
                     f"in {max_iter} iterations")
    return x, y, it


def mp_median(gamma: float) -> float:
    """Median of the Marchenko-Pastur law with ratio ``gamma`` in (0, 1] and
    variance 1 (Landa et al. 2022 Eq. (25); Gavish & Donoho 2014)."""
    from scipy.integrate import quad
    from scipy.optimize import brentq
    g = float(gamma)
    if not 0 < g <= 1:
        raise ValueError("gamma must be in (0, 1]")
    lo, hi = (1 - np.sqrt(g)) ** 2, (1 + np.sqrt(g)) ** 2

    def dens(t):
        if t <= 0.0:          # gamma = 1: the lower edge is 0 (density -> 0/0)
            return 0.0
        return np.sqrt(max((hi - t) * (t - lo), 0.0)) / (2 * np.pi * g * t)

    def cdf(t):
        # F(lo) = 0 exactly; never hand quad an empty interval (some SciPy
        # versions evaluate the integrand at the endpoint, NaN at t = 0)
        if t <= lo:
            return 0.0
        return quad(dens, lo, t, limit=200)[0]

    return brentq(lambda t: cdf(t) - 0.5, lo, hi, xtol=1e-12)


def optimal_hard_threshold_coef(beta: float) -> float:
    """λ*(β) of Gavish & Donoho 2014, Eq. (11) (known noise level)."""
    b = float(beta)
    return float(np.sqrt(2 * (b + 1) + 8 * b / ((b + 1) + np.sqrt(b * b + 14 * b + 1))))


def shrink_singular_values(
    s: np.ndarray, m: int, n: int, sigma: float = 1.0,
    kind: Literal["frobenius", "operator", "hard", "none"] = "frobenius",
) -> np.ndarray:
    """Gavish-Donoho singular-value shrinkage for an ``m x n`` matrix with
    i.i.d.-like noise of entry standard deviation ``sigma``.

    ``frobenius``: Eq. (7) of Gavish & Donoho 2017; ``operator``: Eq. (8);
    ``hard``: keep ``s > λ*(β) σ √N`` unchanged (2014); ``none``: keep
    ``s`` above the MP edge unchanged. All are rescaled from the paper's
    natural noise level ``1/√N`` by Lemma 14 (``η_c(y) = c η(y/c)``).
    """
    s = np.asarray(s, dtype=np.float64)
    N, M = max(m, n), min(m, n)
    beta = M / N
    c = sigma * np.sqrt(N)
    y = s / c
    edge = 1 + np.sqrt(beta)
    above = y > edge
    out = np.zeros_like(s)
    if kind == "frobenius":
        ya = y[above]
        out[above] = c * np.sqrt(np.maximum((ya ** 2 - beta - 1) ** 2 - 4 * beta, 0.0)) / ya
    elif kind == "operator":
        ya = y[above]
        t = ya ** 2 - beta - 1
        out[above] = c * np.sqrt(np.maximum(t + np.sqrt(np.maximum(t ** 2 - 4 * beta, 0.0)), 0.0) / 2)
    elif kind == "hard":
        keep = y > optimal_hard_threshold_coef(beta)
        out[keep] = s[keep]
    elif kind == "none":
        out[above] = s[above]
    else:
        raise ValueError(f"unknown shrinkage {kind!r}")
    return out


def _threaded(X, nt):
    from ._iterative_lsi import _ThreadedCSR
    X = sp.csr_matrix(X)
    X.sort_indices()
    return _ThreadedCSR.wrap(X, nt)


def _randomized_svd(X, k, seed, nt, n_iter=6):
    import sklearn.utils.extmath
    k = int(min(k, min(X.shape) - 1))
    return sklearn.utils.extmath.randomized_svd(
        _threaded(X, nt), n_components=k, n_oversamples=max(10, k),
        n_iter=n_iter, random_state=seed)


def biwhiten(
    Y, *, noise_scale: Union[Literal["mp_median"], float] = "mp_median",
    n_subsample: int = 2000, tol: float = 1e-6, max_iter: int = 10000,
    seed: int = 0, n_jobs: Optional[int] = None,
) -> dict:
    """Biwhiten a nonnegative cells x features count matrix (Landa et al.
    2022, Algorithm 1 with the Poisson variance estimator ``V̂ = Y``; for a
    binary matrix this is also the general quadratic-variance estimator of
    their Eq. (23) up to the global scale ``alpha``).

    Returns a dict with the biwhitened CSR matrix ``Yw`` (float32,
    ``√x_i Y_ij √y_j``), the scalings ``x``, ``y`` (of ``V̂``), the global
    noise variance ``alpha`` (Eq. (24): median eigenvalue of
    ``Yw_S Yw_Sᵀ / n`` over ``n_subsample`` random rows of the smaller side,
    divided by the MP median; or the given float) and the MP edge
    ``edge = √alpha (√m + √n)`` for the singular values of ``Yw``.
    """
    nt = _set_threads(n_jobs)
    Y = sp.csr_matrix(Y, dtype=np.float64)
    Y.eliminate_zeros()
    m, n = Y.shape
    x, y, it = sinkhorn_scale(Y, tol=tol, max_iter=max_iter)
    data = np.empty(Y.nnz, dtype=np.float64)
    _kernels()["scaled_data"](Y.indptr, Y.indices, Y.data, np.sqrt(x), np.sqrt(y), data)
    Yw = sp.csr_matrix((data.astype(np.float32), Y.indices, Y.indptr), shape=(m, n))
    if noise_scale == "mp_median":
        rng = np.random.default_rng(seed)
        A = Yw if m <= n else Yw.T.tocsr()
        a, b = A.shape                        # a <= b
        idx = np.sort(rng.choice(a, size=min(a, int(n_subsample)), replace=False))
        S = A[idx].astype(np.float64)
        G = (S @ S.T).toarray() / b
        ev = np.linalg.eigvalsh(G)
        alpha = float(np.median(ev) / mp_median(len(idx) / b))
    else:
        alpha = float(noise_scale)
    edge = float(np.sqrt(alpha) * (np.sqrt(m) + np.sqrt(n)))
    return dict(Yw=Yw, x=x, y=y, alpha=alpha, edge=edge, sinkhorn_iter=it, n_threads=nt)


def mp_rank_svd(
    Yw, edge: float, *, k_start: int = 48, k_max: int = 128, seed: int = 0,
    n_jobs: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """Randomized SVD of ``Yw`` with ``k`` doubled from ``k_start`` until a
    computed singular value falls below ``edge`` (or ``k_max`` is reached).
    Returns ``U, s, Vt`` (all computed components), the rank (number of
    singular values above ``edge``) and whether the rank hit ``k_max``."""
    nt = _set_threads(n_jobs)
    k = min(k_start, k_max)
    while True:
        U, s, Vt = _randomized_svd(Yw, k + 1, seed, nt)
        r = int(np.sum(s > edge))
        if r <= k or k >= k_max:
            break
        k = min(2 * k, k_max)
    capped = r > k_max or (r >= k_max)
    r = min(r, k_max)
    return U, s, Vt, r, capped


# ---------------------------------------------------------------------------
# Step 3: doubly-stochastic kernel + diffusion map
# ---------------------------------------------------------------------------

def _knn(Z: np.ndarray, k: int, seed: int, nt: int, exact_max: int = 50000):
    """Euclidean kNN (self excluded): exact brute force up to ``exact_max``
    cells, pynndescent above."""
    n = Z.shape[0]
    k = min(k, n - 1)
    if n <= exact_max:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=nt).fit(Z)
        dist, idx = nn.kneighbors(Z)
    else:
        from pynndescent import NNDescent
        index = NNDescent(Z, n_neighbors=k + 1, metric="euclidean", random_state=seed,
                          n_jobs=nt, low_memory=True, compressed=False)
        idx, dist = index.neighbor_graph
    # drop self (first column when present, otherwise the farthest)
    self_first = idx[:, 0] == np.arange(n)
    out_i = np.where(self_first[:, None], idx[:, 1:], idx[:, :-1])
    out_d = np.where(self_first[:, None], dist[:, 1:], dist[:, :-1])
    return out_i.astype(np.int64), out_d.astype(np.float64)


def ds_kernel(
    Z: np.ndarray, *, n_neighbors: int = 30, tol: float = 1e-8, max_iter: int = 20000,
    normalization: Literal["doubly", "row"] = "doubly",
    seed: int = 0, n_jobs: Optional[int] = None,
) -> tuple[sp.csr_matrix, dict]:
    """Sparse doubly-stochastic Gaussian kernel (Landa, Coifman & Kluger 2021).

    kNN graph (Euclidean, ``n_neighbors``), symmetrised by union, kernel
    ``exp(-d²/ε)`` with ``ε`` = median squared distance to the
    ``n_neighbors``-th neighbour, zero main diagonal, symmetric Sinkhorn
    (Knight's ``d <- √(d / Kd)`` iteration, log domain) to unit row sums.
    ``normalization='row'`` (ablation) returns the symmetrised row-stochastic
    kernel ``(D⁻¹K + KD⁻¹)/2`` instead (Eq. (2) of Landa et al. 2021).
    """
    nt = _set_threads(n_jobs)
    n = Z.shape[0]
    idx, dist = _knn(np.ascontiguousarray(Z, dtype=np.float32), n_neighbors, seed, nt)
    eps = float(np.median(dist[:, -1] ** 2))
    if not eps > 0:
        eps = float(np.mean(dist ** 2)) or 1.0
    rows = np.repeat(np.arange(n), idx.shape[1])
    # store d² + 1 so that exact duplicates (d = 0) survive the sparse
    # symmetrisation; the offset is removed afterwards
    D2 = sp.csr_matrix((dist.ravel() ** 2 + 1.0, (rows, idx.ravel())), shape=(n, n))
    D2 = D2.maximum(D2.T).tocsr()           # union; distances are symmetric
    D2.setdiag(0)
    D2.eliminate_zeros()                    # zero main diagonal
    D2.data -= 1.0
    D2.sort_indices()
    logk = -D2.data / eps
    k = _kernels()
    u = np.zeros(n)
    lse = np.empty(n)
    if normalization == "row":
        k["row_lse"](D2.indptr, D2.indices, logk, u, lse)
        rr = np.repeat(np.arange(n), np.diff(D2.indptr))
        P = sp.csr_matrix((np.exp(logk - lse[rr]), D2.indices.copy(), D2.indptr.copy()), shape=(n, n))
        return ((P + P.T) * 0.5).tocsr(), dict(epsilon=eps, sinkhorn_iter=0, sinkhorn_err=0.0,
                                                n_neighbors=int(idx.shape[1]))
    if normalization != "doubly":
        raise ValueError(f"unknown normalization {normalization!r}")
    it = 0
    for it in range(1, max_iter + 1):
        k["row_lse"](D2.indptr, D2.indices, logk, u, lse)
        # row sums of D K D are exp(u_i + lse_i)
        err = np.max(np.abs(np.expm1(u + lse)))
        if err < tol:
            break
        u = 0.5 * (u - lse)
    else:
        console.warn(f"[birmt] symmetric Sinkhorn did not reach tol={tol} in {max_iter} iterations")
    rr = np.repeat(np.arange(n), np.diff(D2.indptr))
    W = sp.csr_matrix((np.exp(logk + u[rr] + u[D2.indices]), D2.indices.copy(), D2.indptr.copy()),
                      shape=(n, n))
    W = ((W + W.T) * 0.5).tocsr()           # remove round-off asymmetry
    return W, dict(epsilon=eps, sinkhorn_iter=it, sinkhorn_err=float(err),
                   n_neighbors=int(idx.shape[1]))


def diffusion_time_gap(lam: np.ndarray, delta: float, t_max: int = 100) -> tuple[int, int]:
    """Spectral-gap rule for the diffusion time (see module docstring).

    ``lam``: non-trivial eigenvalues, descending. Returns ``(t, q)`` with
    ``q`` the 1-based index of the largest gap ``λ_q - λ_{q+1}``.
    """
    lam = np.asarray(lam, dtype=np.float64)
    if lam.size < 2:
        return 1, int(lam.size)
    gaps = lam[:-1] - lam[1:]
    q = int(np.argmax(gaps))                # 0-based index of λ_q
    a, b = lam[q], lam[q + 1]
    if b <= 0 or a <= 0:
        return 1, q + 1
    ratio = b / a
    if ratio >= 1:
        return 1, q + 1
    t = int(np.ceil(np.log(delta) / np.log(ratio)))
    return int(min(max(t, 1), t_max)), q + 1


def diffusion_map(
    W: sp.csr_matrix, n_components: int, *, t: Union[int, Literal["gap"]] = "gap",
    delta: float = 0.25, t_max: int = 100, seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Diffusion coordinates of a symmetric doubly-stochastic ``W``.

    The ``n_components + 1`` algebraically largest eigenpairs are computed;
    the eigenvector closest to the constant vector (λ = 1) is dropped and the
    rest are returned as ``√n λ_k^t φ_k``.
    """
    from scipy.sparse.linalg import eigsh
    n = W.shape[0]
    kk = int(min(n_components + 1, n - 2))
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n)
    lam, phi = eigsh(W, k=kk, which="LA", v0=v0, tol=1e-10)
    o = np.argsort(-lam)
    lam, phi = lam[o], phi[:, o]
    const = int(np.argmax(np.abs(phi.T @ np.full(n, 1 / np.sqrt(n)))))
    keep = np.setdiff1d(np.arange(kk), [const])
    lam, phi = lam[keep], phi[:, keep]
    q = None
    if t == "gap":
        t_used, q = diffusion_time_gap(lam, delta, t_max)
    else:
        t_used = int(t)
    lam_pos = np.clip(lam, 0.0, None)
    coords = np.sqrt(n) * phi * (lam_pos ** t_used)[None, :]
    # deterministic signs: largest-|entry| positive
    sgn = np.sign(phi[np.argmax(np.abs(phi), axis=0), np.arange(phi.shape[1])])
    coords *= sgn[None, :]
    return coords, dict(eigenvalues=lam, t=int(t_used), gap_index=q)


# ---------------------------------------------------------------------------
# Step 4: Markov stability resolution
# ---------------------------------------------------------------------------

def _entropy_counts(c):
    p = c[c > 0] / c.sum()
    return float(-(p * np.log(p)).sum())


def nvi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised variation of information ``VI/H(a,b)`` (0 if identical)."""
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    joint = np.unique(a.astype(np.int64) * (b.max() + 1) + b, return_counts=True)[1]
    hab = _entropy_counts(joint)
    if hab == 0:
        return 0.0
    ha = _entropy_counts(np.bincount(a))
    hb = _entropy_counts(np.bincount(b))
    return float((2 * hab - ha - hb) / hab)


_G = {}


def _leiden_worker(args):
    gamma, seed = args
    import leidenalg
    g = _G["g"]
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, weights="weight",
        resolution_parameter=float(gamma), seed=int(seed), n_iterations=-1)
    return np.asarray(part.membership, dtype=np.int32)


def _graph_from_adjacency(W):
    import igraph as ig
    U = sp.triu(W, k=1).tocoo()
    g = ig.Graph(n=W.shape[0], edges=np.column_stack([U.row, U.col]).tolist(), directed=False)
    g.es["weight"] = U.data.astype(float).tolist()
    return g


def markov_stability(
    W: sp.csr_matrix, *, resolutions: Optional[Sequence[float]] = None,
    n_seeds: int = 10, n_jobs: Optional[int] = None, tie_tol: float = 1e-9,
) -> dict:
    """Scan Leiden resolutions (Markov time ``1/γ``) on ``W`` and select the
    most seed-stable partition (minimum mean pairwise NVI, ≥ 2 clusters;
    ties -> larger ``γ``). Returns labels and the scan table."""
    import pandas as pd
    if resolutions is None:
        resolutions = default_resolutions()
    resolutions = np.asarray(sorted(float(r) for r in resolutions))
    nt = _set_threads(n_jobs)
    _G["g"] = _graph_from_adjacency(W)
    jobs = [(g, s) for g in resolutions for s in range(n_seeds)]
    if nt > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=nt, mp_context=mp.get_context("fork")) as ex:
            parts = list(ex.map(_leiden_worker, jobs, chunksize=1))
    else:
        parts = [_leiden_worker(j) for j in jobs]
    _G.clear()
    rows, best = [], None
    for gi, gamma in enumerate(resolutions):
        P = parts[gi * n_seeds:(gi + 1) * n_seeds]
        M = np.zeros((n_seeds, n_seeds))
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                M[i, j] = M[j, i] = nvi(P[i], P[j])
        mean_nvi = float(M[np.triu_indices(n_seeds, 1)].mean()) if n_seeds > 1 else 0.0
        medoid = int(np.argmin(M.sum(1)))
        ncl = [int(np.unique(p).size) for p in P]
        rows.append(dict(resolution=float(gamma), markov_time=1.0 / float(gamma),
                         mean_nvi=mean_nvi, n_clusters_median=float(np.median(ncl)),
                         n_clusters_medoid=ncl[medoid]))
        if np.median(ncl) >= 2:
            if best is None or mean_nvi < best[0] - tie_tol or abs(mean_nvi - best[0]) <= tie_tol:
                best = (mean_nvi, gi, P[medoid])
    table = pd.DataFrame(rows)
    if best is None:
        gi = int(np.argmax(resolutions))
        labels = parts[gi * n_seeds]
    else:
        _, gi, labels = best
    return dict(labels=labels, resolution=float(resolutions[gi]), table=table)


def default_resolutions() -> np.ndarray:
    """Pre-registered scan grid: 25 log-spaced resolutions in [0.1, 10]."""
    return np.round(np.geomspace(0.1, 10.0, 25), 6)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def birmt(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
    min_cells: Union[int, Literal["auto"]] = "auto",
    noise_scale: Union[Literal["mp_median"], float] = "mp_median",
    k_max: int = 128,
    shrinkage: Optional[Literal["frobenius", "operator", "hard", "none", "score"]] = None,
    row_norm: Literal["depth", "l2", "none"] = "depth",
    embedding: Literal["shrunk", "diffusion"] = "shrunk",
    diffusion: bool = True,
    n_neighbors: int = 30,
    kernel_normalization: Literal["doubly", "row"] = "doubly",
    t: Union[int, Literal["gap"]] = "gap",
    delta: float = 0.25,
    t_max: int = 100,
    cluster: Optional[bool] = None,
    resolutions: Optional[Sequence[float]] = None,
    n_seeds: int = 10,
    key_added: str = "X_birmt",
    cluster_key: str = "birmt",
    seed: int = 0,
    n_jobs: Optional[int] = None,
    version: Literal[1, 2] = 2,
    device: str = "cpu",
    **v2_options,
) -> AnnData:
    """BiRMT embedding and clustering of a cells x features accessibility
    matrix (binarised). See the module docstring for the method and refs.

    **Default (``version=2``): BiRMT-2**, Epione's default cell embedding
    (:func:`epione.tl._birmt2.birmt2`: matrix-free biwhitening, block
    Krylov SVD, cell-score optimal shrinkage), stored in
    ``obsm['X_birmt']``; cluster it with the usual
    ``epi.pp.neighbors(adata, use_rep='X_birmt')`` + ``epi.tl.clusters``.
    For ``version=2`` the label-free Markov-time-1 partition is off by
    default (``cluster=True`` turns it on); ``v2_options`` are passed to
    :func:`~epione.tl._birmt2.birmt2`. ``version=1`` reproduces BiRMT v1
    exactly (its defaults, including ``cluster=True``, are unchanged).

    Parameters
    ----------
    adata
        AnnData with counts in ``.X`` (or ``layer``); binarised internally.
    min_cells
        Features open in fewer cells are dropped before biwhitening.
        ``'auto'`` = ``max(5, ⌈2 ln(n_cells)⌉)``: the sparse-matrix regime
        where the largest noise singular value still sits at the MP edge
        needs expected column degrees growing like ``log n``.
    noise_scale
        ``'mp_median'`` (Landa et al. 2022 Eq. (24)) or a fixed noise
        variance (``1.0`` = their Algorithm 1 for Poisson data).
    k_max
        Largest rank considered (randomized SVD size cap).
    shrinkage
        Gavish-Donoho shrinker for the retained singular values (``None`` =
        ``'frobenius'`` for v1 and the BiRMT-2 default for ``version=2``).
    row_norm
        ``'depth'``: undo the row scaling and divide by the cell's total
        count (denoised profile); ``'l2'``: unit-norm rows; ``'none'``.
    embedding
        What is stored in ``obsm[key_added]``: ``'shrunk'`` (step-2
        coordinates; the frozen default, chosen on the development sets) or
        ``'diffusion'`` (step-3 diffusion coordinates).
    diffusion
        Compute the diffusion map of step 3 (stored in
        ``obsm[key_added + '_diffmap']``); required for
        ``embedding='diffusion'``. The doubly-stochastic kernel itself is
        always built: it is the Markov chain of step 4.
    n_neighbors
        kNN size of the sparse Gaussian kernel (bandwidth uses the
        ``n_neighbors``-th neighbour).
    kernel_normalization
        ``'doubly'`` (Landa et al. 2021) or ``'row'`` (ablation).
    t, delta, t_max
        Diffusion time: ``'gap'`` (spectral-gap rule with attenuation
        ``delta``, capped at ``t_max``) or a fixed integer.
    cluster, resolutions, n_seeds
        Step 4 (Markov-stability resolution selection); grid defaults to
        :func:`default_resolutions`. ``cluster=None``: ``True`` for
        ``version=1``, ``False`` for ``version=2``.
    device
        ``version=2`` only: ``'cpu'`` (default), ``'cuda'`` / ``'cuda:N'``
        or ``'auto'`` -- GPU sparse products for the biwhitening and the
        Krylov SVD (float64; same rank, equal to the CPU result up to
        round-off)
        (:func:`epione.tl._birmt2.birmt2`; ``gpu_memory_gb`` sets the device
        budget). Falls back to the CPU when CUDA is unavailable.

    Returns
    -------
    ``adata`` with ``obsm[key_added]`` (see ``embedding``),
    ``obsm[key_added + '_diffmap']`` (step-3 diffusion coordinates, if
    ``diffusion``), ``obsp[key_added + '_W']`` (doubly-stochastic kernel),
    ``obs[cluster_key]`` (if ``cluster``) and ``uns[key_added]`` with all
    diagnostics (rank, noise scale, singular values, ε, eigenvalues, t,
    Markov-stability table) and the parameters.
    """
    if version == 2:
        from ._birmt2 import birmt2
        if cluster is None:
            cluster = False
        if shrinkage is not None:
            v2_options["shrinkage"] = shrinkage
        return birmt2(adata, layer=layer, min_cells=min_cells, k_max=k_max, n_neighbors=n_neighbors,
                      cluster=cluster, resolutions=resolutions, n_seeds=n_seeds, key_added=key_added,
                      cluster_key=cluster_key, seed=seed, n_jobs=n_jobs, device=device, **v2_options)
    if version != 1:
        raise ValueError(f"unknown BiRMT version {version!r}")
    if v2_options:
        raise TypeError(f"unexpected arguments for version=1: {sorted(v2_options)}")
    if device is not None and str(device).lower() != "cpu":
        raise ValueError("device= is only supported by version=2 (BiRMT-2)")
    if shrinkage is None:
        shrinkage = "frobenius"
    if cluster is None:
        cluster = True
    import time
    t0 = time.time()
    nt = _set_threads(n_jobs)
    X = adata.layers[layer] if layer is not None else adata.X
    X = sp.csr_matrix(X)
    Y = sp.csr_matrix((np.ones(X.nnz, dtype=np.float32), X.indices, X.indptr), shape=X.shape)
    Y.data[X.data <= 0] = 0
    Y.eliminate_zeros()
    n_cells = Y.shape[0]
    if min_cells == "auto":
        min_cells_v = int(max(5, np.ceil(2 * np.log(n_cells))))
    else:
        min_cells_v = int(min_cells)
    colnnz = np.bincount(Y.indices, minlength=Y.shape[1])
    feats = np.flatnonzero(colnnz >= min_cells_v)
    Y = Y[:, feats].tocsr()
    depth = np.diff(Y.indptr).astype(np.float64)
    if np.any(depth == 0):
        raise ValueError(f"{int((depth == 0).sum())} cells have no feature open in "
                         f">= {min_cells_v} cells")
    console.info(f"[birmt] {n_cells:,} cells x {len(feats):,} features (min_cells={min_cells_v})")

    bw = biwhiten(Y, noise_scale=noise_scale, seed=seed, n_jobs=nt)
    U, s, Vt, r, capped = mp_rank_svd(bw["Yw"], bw["edge"], k_max=k_max, seed=seed, n_jobs=nt)
    if r < 1:
        raise ValueError("no singular value above the MP edge: no detectable signal")
    if capped:
        console.warn(f"[birmt] MP rank reached k_max={k_max}; increase k_max")
    m, n = Y.shape
    s_sh = shrink_singular_values(s[:r], m, n, sigma=np.sqrt(bw["alpha"]), kind=shrinkage)
    keep = s_sh > 0
    if not keep.any():
        keep[0] = True
        s_sh[0] = s[0]
    Z = U[:, :r][:, keep] * s_sh[keep][None, :]
    if row_norm == "depth":
        Z = Z / (np.sqrt(bw["x"]) * depth)[:, None]
    elif row_norm == "l2":
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-300)
    elif row_norm != "none":
        raise ValueError(f"unknown row_norm {row_norm!r}")
    t1 = time.time()
    console.info(f"[birmt] alpha={bw['alpha']:.4f} edge={bw['edge']:.1f} rank={r} "
                 f"kept={int(keep.sum())} ({t1 - t0:.1f}s)")

    if embedding == "diffusion" and not diffusion:
        raise ValueError("embedding='diffusion' needs diffusion=True")
    W, kinfo = ds_kernel(Z, n_neighbors=n_neighbors, normalization=kernel_normalization,
                         seed=seed, n_jobs=nt)
    dinfo = {}
    dmap = None
    if diffusion:
        n_diff = int(max(2, keep.sum()))
        dmap, dinfo = diffusion_map(W, n_diff, t=t, delta=delta, t_max=t_max, seed=seed)
    emb = dmap if embedding == "diffusion" else Z
    t2 = time.time()
    console.info(f"[birmt] eps={kinfo['epsilon']:.4g} t={dinfo.get('t')} "
                 f"gap_index={dinfo.get('gap_index')} ({t2 - t1:.1f}s)")

    adata.obsm[key_added] = np.asarray(emb, dtype=np.float32)
    if dmap is not None:
        adata.obsm[key_added + "_diffmap"] = np.asarray(dmap, dtype=np.float32)
    adata.obsp[key_added + "_W"] = W
    info = dict(
        rank=int(r), rank_capped=bool(capped), n_kept=int(keep.sum()),
        alpha=float(bw["alpha"]), mp_edge=float(bw["edge"]),
        singular_values=np.asarray(s, dtype=np.float64),
        shrunk_singular_values=np.asarray(s_sh, dtype=np.float64),
        n_features_used=int(len(feats)), min_cells=int(min_cells_v),
        biwhiten_sinkhorn_iter=int(bw["sinkhorn_iter"]),
        epsilon=float(kinfo["epsilon"]), kernel_sinkhorn_iter=int(kinfo["sinkhorn_iter"]),
        eigenvalues=np.asarray(dinfo.get("eigenvalues", []), dtype=np.float64),
        t=dinfo.get("t"), gap_index=dinfo.get("gap_index"),
        params=dict(layer=layer, min_cells=str(min_cells), noise_scale=str(noise_scale),
                    k_max=k_max, shrinkage=shrinkage, row_norm=row_norm,
                    embedding=embedding, diffusion=diffusion, n_neighbors=n_neighbors,
                    kernel_normalization=kernel_normalization, t=str(t), delta=delta,
                    t_max=t_max, cluster=cluster, n_seeds=n_seeds, seed=seed),
        time_embedding_s=float(t2 - t0),
    )
    if cluster:
        ms = markov_stability(W, resolutions=resolutions, n_seeds=n_seeds, n_jobs=nt)
        import pandas as pd
        adata.obs[cluster_key] = pd.Categorical(ms["labels"].astype(str))
        info["resolution"] = ms["resolution"]
        info["markov_stability"] = {c: ms["table"][c].to_numpy() for c in ms["table"].columns}
        info["time_total_s"] = float(time.time() - t0)
        console.info(f"[birmt] Markov stability: resolution={ms['resolution']:.3g} "
                     f"clusters={adata.obs[cluster_key].nunique()} ({time.time() - t2:.1f}s)")
    adata.uns[key_added] = info
    return adata
