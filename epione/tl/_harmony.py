"""Harmony batch correction — a port of R ``harmony`` 1.2.4 as run by ArchR.

``epi.tl.harmony`` reproduces ``set.seed(seed); ArchR::addHarmony(proj,
reducedDims=..., groupBy=...)``, i.e. ``harmony::RunHarmony`` (the C++
``harmony`` class from harmony 1.2.4, Korsunsky et al. 2019, GPL-3) with
R's ``stats::kmeans`` (Hartigan–Wong, ``nstart=10``) centroid
initialisation and Armadillo's R-RNG-backed ``shuffle`` for the block
update order. Both random streams are emulated with :class:`RRandom`, so
the same seed gives the same embedding as R.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._rrng import RRandom

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba ships with scanpy
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


def _console(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  └─ [harmony] {msg}", flush=True)


# ---------------------------------------------------------------------------
# stats::kmeans, Hartigan–Wong (R 4.4.3 src/library/stats/src/kmns.f, AS 136)
# ---------------------------------------------------------------------------

_BIG = float(np.float32(1e30))  # Fortran ``DATA BIG /1.E30/`` (a REAL literal)


@njit(cache=True)
def _optra(A, C, ic1, ic2, nc, an1, an2, ncp, D, itran, live, indx):
    M, N = A.shape
    K = C.shape[0]
    for L in range(K):
        if itran[L] == 1:
            live[L] = M + 1
    for i in range(M):
        I = i + 1
        indx += 1
        l1 = ic1[i]
        l2 = ic2[i]
        ll = l2
        if nc[l1] != 1:
            if ncp[l1] != 0:
                de = 0.0
                for j in range(N):
                    df = A[i, j] - C[l1, j]
                    de = de + df * df
                D[i] = de * an1[l1]
            da = 0.0
            for j in range(N):
                db = A[i, j] - C[l2, j]
                da = da + db * db
            r2 = da * an2[l2]
            for L in range(K):
                if (I >= live[l1] and I >= live[L]) or L == l1 or L == ll:
                    continue
                rr = r2 / an2[L]
                dc = 0.0
                skip = False
                for j in range(N):
                    dd = A[i, j] - C[L, j]
                    dc = dc + dd * dd
                    if dc >= rr:
                        skip = True
                        break
                if skip:
                    continue
                r2 = dc * an2[L]
                l2 = L
            if r2 >= D[i]:
                ic2[i] = l2
            else:
                indx = 0
                live[l1] = M + I
                live[l2] = M + I
                ncp[l1] = I
                ncp[l2] = I
                al1 = float(nc[l1])
                alw = al1 - 1.0
                al2 = float(nc[l2])
                alt = al2 + 1.0
                for j in range(N):
                    C[l1, j] = (C[l1, j] * al1 - A[i, j]) / alw
                    C[l2, j] = (C[l2, j] * al2 + A[i, j]) / alt
                nc[l1] -= 1
                nc[l2] += 1
                an2[l1] = alw / al1
                an1[l1] = _BIG
                if alw > 1.0:
                    an1[l1] = alw / (alw - 1.0)
                an1[l2] = alt / al2
                an2[l2] = alt / (alt + 1.0)
                ic1[i] = l2
                ic2[i] = l1
        if indx == M:
            return indx
    for L in range(K):
        itran[L] = 0
        live[L] = live[L] - M
    return indx


@njit(cache=True)
def _qtran(A, C, ic1, ic2, nc, an1, an2, ncp, D, itran, indx, imaxqtr):
    M, N = A.shape
    icoun = 0
    istep = 0
    while True:
        for i in range(M):
            icoun += 1
            istep += 1
            if istep >= imaxqtr:
                return indx, -1
            l1 = ic1[i]
            l2 = ic2[i]
            if nc[l1] != 1:
                if istep <= ncp[l1]:
                    da = 0.0
                    for j in range(N):
                        db = A[i, j] - C[l1, j]
                        da = da + db * db
                    D[i] = da * an1[l1]
                if istep < ncp[l1] or istep < ncp[l2]:
                    r2 = D[i] / an2[l2]
                    dd = 0.0
                    moved = True
                    for j in range(N):
                        de = A[i, j] - C[l2, j]
                        dd = dd + de * de
                        if dd >= r2:
                            moved = False
                            break
                    if moved:
                        icoun = 0
                        indx = 0
                        itran[l1] = 1
                        itran[l2] = 1
                        ncp[l1] = istep + M
                        ncp[l2] = istep + M
                        al1 = float(nc[l1])
                        alw = al1 - 1.0
                        al2 = float(nc[l2])
                        alt = al2 + 1.0
                        for j in range(N):
                            C[l1, j] = (C[l1, j] * al1 - A[i, j]) / alw
                            C[l2, j] = (C[l2, j] * al2 + A[i, j]) / alt
                        nc[l1] -= 1
                        nc[l2] += 1
                        an2[l1] = alw / al1
                        an1[l1] = _BIG
                        if alw > 1.0:
                            an1[l1] = alw / (alw - 1.0)
                        an1[l2] = alt / al2
                        an2[l2] = alt / (alt + 1.0)
                        ic1[i] = l2
                        ic2[i] = l1
            if icoun == M:
                return indx, imaxqtr


@njit(cache=True)
def _kmns(A, C, iter_max, imaxqtr):
    """Returns (centers, cluster, wss, ifault); ``C`` is overwritten."""
    M, N = A.shape
    K = C.shape[0]
    ic1 = np.zeros(M, np.int64)
    ic2 = np.zeros(M, np.int64)
    nc = np.zeros(K, np.int64)
    an1 = np.zeros(K)
    an2 = np.zeros(K)
    ncp = np.zeros(K, np.int64)
    D = np.zeros(M)
    itran = np.zeros(K, np.int64)
    live = np.zeros(K, np.int64)
    wss = np.zeros(K)
    dt = np.zeros(2)
    for i in range(M):
        ic1[i] = 0
        ic2[i] = 1
        for il in range(2):
            dt[il] = 0.0
            for j in range(N):
                da = A[i, j] - C[il, j]
                dt[il] = dt[il] + da * da
        if dt[0] > dt[1]:
            ic1[i] = 1
            ic2[i] = 0
            temp = dt[0]
            dt[0] = dt[1]
            dt[1] = temp
        for L in range(2, K):
            db = 0.0
            far = False
            for j in range(N):
                dc = A[i, j] - C[L, j]
                db = db + dc * dc
                if db >= dt[1]:
                    far = True
                    break
            if far:
                continue
            if db >= dt[0]:
                dt[1] = db
                ic2[i] = L
            else:
                dt[1] = dt[0]
                ic2[i] = ic1[i]
                dt[0] = db
                ic1[i] = L
    for L in range(K):
        nc[L] = 0
        for j in range(N):
            C[L, j] = 0.0
    for i in range(M):
        L = ic1[i]
        nc[L] += 1
        for j in range(N):
            C[L, j] = C[L, j] + A[i, j]
    for L in range(K):
        if nc[L] == 0:
            return C, ic1, wss, 1
        aa = float(nc[L])
        for j in range(N):
            C[L, j] = C[L, j] / aa
        an2[L] = aa / (aa + 1.0)
        an1[L] = _BIG
        if aa > 1.0:
            an1[L] = aa / (aa - 1.0)
        itran[L] = 1
        ncp[L] = -1
    indx = 0
    ifault = 2
    for ij in range(iter_max):
        indx = _optra(A, C, ic1, ic2, nc, an1, an2, ncp, D, itran, live, indx)
        if indx == M:
            ifault = 0
            break
        indx, imaxqtr = _qtran(A, C, ic1, ic2, nc, an1, an2, ncp, D, itran, indx, imaxqtr)
        if imaxqtr < 0:
            ifault = 4
            break
        if K == 2:
            ifault = 0
            break
        for L in range(K):
            ncp[L] = 0
    for L in range(K):
        wss[L] = 0.0
        for j in range(N):
            C[L, j] = 0.0
    for i in range(M):
        ii = ic1[i]
        for j in range(N):
            C[ii, j] = C[ii, j] + A[i, j]
    for j in range(N):
        for L in range(K):
            C[L, j] = C[L, j] / float(nc[L])
        for i in range(M):
            ii = ic1[i]
            da = A[i, j] - C[ii, j]
            wss[ii] = wss[ii] + da * da
    return C, ic1, wss, ifault


def _unique_rows_first(x: np.ndarray) -> np.ndarray:
    _, first = np.unique(x, axis=0, return_index=True)
    return x[np.sort(first)]


def r_kmeans(x: np.ndarray, k: int, rng: RRandom, iter_max: int = 25,
             nstart: int = 10) -> np.ndarray:
    """``stats::kmeans(x, centers=k, iter.max, nstart)`` (Hartigan–Wong).

    Consumes ``rng`` exactly as R does and returns the ``k × p`` centers.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    m = x.shape[0]
    imaxqtr = min(2147483647, 50 * m)
    if nstart == 1:
        centers = x[rng.sample_int(m, k) - 1]
        cn = None
    if nstart >= 2 or len(_unique_rows_first(centers)) < k:
        cn = _unique_rows_first(x)
        mm = cn.shape[0]
        if mm < k:
            raise ValueError("more cluster centers than distinct data points.")
        centers = cn[rng.sample_int(mm, k) - 1]

    def do_one(c0):
        c, _, wss, ifault = _kmns(x, np.ascontiguousarray(c0.copy()), int(iter_max), imaxqtr)
        if ifault == 1:
            raise RuntimeError("empty cluster: try a better set of initial centers")
        return c.copy(), wss

    best_c, wss = do_one(centers)
    best = _rsum(wss)
    if nstart >= 2 and cn is not None:
        for _ in range(1, nstart):
            centers = cn[rng.sample_int(mm, k) - 1]
            c, wss = do_one(centers)
            z = _rsum(wss)
            if z < best:
                best_c, best = c, z
    return best_c


def _rsum(v: np.ndarray):
    acc = np.longdouble(0.0)
    for val in v:
        acc += np.longdouble(val)
    return acc


# ---------------------------------------------------------------------------
# harmony 1.2.4 C++ core (src/harmony.cpp)
# ---------------------------------------------------------------------------

_F = np.float32


def _l2_normalise_cols(X: np.ndarray) -> np.ndarray:
    nrm = np.sqrt(np.einsum("ij,ij->j", X, X))
    nrm[nrm == 0] = 1.0
    return X / nrm


def _l1_normalise_cols(X: np.ndarray) -> np.ndarray:
    nrm = np.abs(X).sum(axis=0)
    nrm[nrm == 0] = 1.0
    return X / nrm


class _Harmony:
    window_size = 3

    def __init__(self, Z, Phi, sigma, theta, lambda_vec, alpha, max_iter_kmeans,
                 epsilon_kmeans, epsilon_harmony, K, block_size, rng):
        self.d, self.N = Z.shape
        self.B = Phi.shape[0]
        self.K = int(K)
        self.Z_orig = Z
        self.Z_cos = _l2_normalise_cols(Z)
        self.Z_corr = np.zeros_like(Z)
        self.Phi = Phi
        self.Pr_b = Phi.sum(axis=1) / self.N
        self.index = [np.flatnonzero(Phi[b] > 0) for b in range(self.B)]
        self.epsilon_kmeans = _F(epsilon_kmeans)
        self.epsilon_harmony = _F(epsilon_harmony)
        self.lambda_estimation = lambda_vec[0] == -1
        self.lambda_vec = lambda_vec
        self.sigma = sigma
        self.theta = theta
        self.alpha = float(_F(alpha))
        self.max_iter_kmeans = int(max_iter_kmeans)
        if self.N < 6:
            raise ValueError("Refusing to run with less than 6 cells")
        self.block_size = _F(0.2) if self.N < 40 else _F(block_size)
        self.Phi_moe = np.vstack([np.ones((1, self.N)), Phi])
        self.rng = rng
        self.objective_kmeans, self.objective_harmony = [], []
        self.objective_kmeans_dist, self.objective_kmeans_entropy = [], []
        self.objective_kmeans_cross, self.kmeans_rounds = [], []

    # -- init -----------------------------------------------------------
    def init_cluster(self, Y=None):
        if Y is None:
            Y = r_kmeans(self.Z_cos.T, self.K, self.rng, iter_max=25, nstart=10).T
        self.Y = _l2_normalise_cols(np.asarray(Y, dtype=np.float64))
        self.dist_mat = 2.0 * (1.0 - self.Y.T @ self.Z_cos)
        R = np.exp(-self.dist_mat / self.sigma[:, None])
        self.R = R / R.sum(axis=0)
        self.E = self.R.sum(axis=1)[:, None] * self.Pr_b[None, :]
        self.O = self.R @ self.Phi.T
        self._compute_objective()
        self.objective_harmony.append(self.objective_kmeans[-1])
        self.dist_mat = 2.0 * (1.0 - self.Y.T @ self.Z_cos)

    def _compute_objective(self):
        norm_const = _F(2000) / _F(self.N)
        kmeans_error = _F(np.sum(self.R * self.dist_mat))
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = self.R * np.log(self.R)
        ent[~np.isfinite(ent)] = 0.0
        entropy = _F(np.sum(ent * self.sigma[:, None]))
        cross = _F(np.sum((self.R * self.sigma[:, None])
                          * ((self.theta[None, :] * np.log((self.O + self.E) / self.E)) @ self.Phi)))
        self.objective_kmeans.append(_F(_F(_F(kmeans_error + entropy) + cross) * norm_const))
        self.objective_kmeans_dist.append(_F(kmeans_error * norm_const))
        self.objective_kmeans_entropy.append(_F(entropy * norm_const))
        self.objective_kmeans_cross.append(_F(cross * norm_const))

    def _check_convergence(self, kind: int) -> bool:
        if kind == 0:
            obj_old, obj_new = _F(0), _F(0)
            ok = self.objective_kmeans
            for i in range(self.window_size):
                obj_old = _F(obj_old + ok[-2 - i])
                obj_new = _F(obj_new + ok[-1 - i])
            return bool(_F(_F(obj_old - obj_new) / abs(obj_old)) < self.epsilon_kmeans)
        oh = self.objective_harmony
        obj_old, obj_new = oh[-2], oh[-1]
        return bool(_F(_F(obj_old - obj_new) / abs(obj_old)) < self.epsilon_harmony)

    # -- clustering -----------------------------------------------------
    def cluster(self, orders=None):
        it = 0
        while it < self.max_iter_kmeans:
            self.Y = _l2_normalise_cols(self.Z_cos @ self.R.T)
            self.dist_mat = 2.0 * (1.0 - self.Y.T @ self.Z_cos)
            self._update_R(None if orders is None else orders.pop(0))
            self._compute_objective()
            if it > self.window_size and self._check_convergence(0):
                it += 1
                break
            it += 1
        self.kmeans_rounds.append(it)
        self.objective_harmony.append(self.objective_kmeans[-1])

    def _draw_order(self):
        # arma::shuffle with RcppArmadillo's R RNG: sort (int) runif(0, RAND_MAX).
        vals = self.rng.runif(self.N, 0.0, 2147483647.0).astype(np.int64)
        return np.argsort(vals, kind="stable")

    def _update_R(self, order=None):
        order = self._draw_order() if order is None else order
        scale = np.exp(-self.dist_mat / self.sigma[:, None])
        scale = _l1_normalise_cols(scale)
        n_blocks = int(np.ceil(_F(1.0 / float(self.block_size))))
        cells_per_block = int(_F(self.N) * self.block_size)
        R, E, O, Phi = self.R, self.E, self.O, self.Phi
        for i in range(n_blocks):
            lo = i * cells_per_block
            hi = self.N if i == n_blocks - 1 else (i + 1) * cells_per_block
            cells = order[lo:hi]
            Rc = R[:, cells]
            Pc = Phi[:, cells]
            E -= Rc.sum(axis=1)[:, None] * self.Pr_b[None, :]
            O -= Rc @ Pc.T
            Rc = scale[:, cells] * (np.power(E / (O + E), self.theta[None, :]) @ Pc)
            Rc = _l1_normalise_cols(Rc)
            E += Rc.sum(axis=1)[:, None] * self.Pr_b[None, :]
            O += Rc @ Pc.T
            R[:, cells] = Rc

    # -- correction -----------------------------------------------------
    def moe_correct_ridge(self):
        B = self.B
        lam = np.diag(self.lambda_vec) if not self.lambda_estimation else None
        Z_corr = self.Z_orig.copy()
        for k in range(self.K):
            if self.lambda_estimation:
                lam = np.diag(np.concatenate([[0.0], self.E[k] * self.alpha]))
            rk = self.R[k]
            Phi_Rk = self.Phi_moe * rk[None, :]
            inv_cov = np.linalg.inv(Phi_Rk @ self.Phi_moe.T + lam)
            Z_tmp = self.Z_orig * rk[None, :]
            W = np.outer(inv_cov[:, 0], Z_tmp.sum(axis=1))
            for b in range(B):
                W += np.outer(inv_cov[:, b + 1], Z_tmp[:, self.index[b]].sum(axis=1))
            W[0, :] = 0.0
            Z_corr -= W.T @ Phi_Rk
        self.Z_corr = Z_corr
        self.Z_cos = _l2_normalise_cols(Z_corr)


def _design(meta: pd.DataFrame, vars_use: Sequence[str]):
    blocks, b_vec, levels = [], [], []
    for v in vars_use:
        col = meta[v]
        if isinstance(col.dtype, pd.CategoricalDtype):
            cat = pd.Categorical(col).remove_unused_categories()
        else:
            cat = pd.Categorical(col.astype(str), categories=sorted(col.astype(str).unique()))
        codes = np.asarray(cat.codes)
        onehot = np.zeros((len(cat.categories), len(codes)))
        onehot[codes, np.arange(len(codes))] = 1.0
        blocks.append(onehot)
        b_vec.append(len(cat.categories))
        levels.append([f"{v}={c}" for c in cat.categories])
    return np.vstack(blocks), np.asarray(b_vec), sum(levels, [])


def run_harmony(
    data_mat: np.ndarray,
    meta_data: pd.DataFrame,
    vars_use: Union[str, Sequence[str]],
    *,
    theta: Optional[Union[float, Sequence[float]]] = None,
    sigma: Union[float, Sequence[float]] = 0.1,
    lambda_: Optional[Union[float, Sequence[float]]] = 1.0,
    nclust: Optional[int] = None,
    max_iter: int = 10,
    early_stop: bool = True,
    alpha: float = 0.2,
    tau: float = 0.0,
    block_size: float = 0.05,
    max_iter_cluster: int = 20,
    epsilon_cluster: float = 1e-3,
    epsilon_harmony: float = 1e-2,
    seed: Optional[int] = 1,
    rng_state: Optional[Sequence[int]] = None,
    init_centers: Optional[np.ndarray] = None,
    block_orders: Optional[list] = None,
    return_object: bool = False,
    verbose: bool = True,
):
    """Matrix-level ``harmony::RunHarmony`` (cells × dims in, cells × dims out).

    ``seed`` reproduces ``set.seed(seed)`` before the R call; ``rng_state``
    resumes an exported ``.Random.seed`` instead. ``init_centers``
    (``nclust × dims``) and ``block_orders`` (one 0-based permutation per
    clustering round) inject R's random draws directly — used by the parity
    tests to check the deterministic part independently of the RNG.
    """
    X = np.asarray(data_mat, dtype=np.float64)
    N = X.shape[0]
    if isinstance(vars_use, str):
        vars_use = [vars_use]
    if len(meta_data) != N:
        raise ValueError("number of labels do not correspond to number of samples in data matrix")
    if nclust is None:
        nclust = int(min(round(N / 30), 100))  # Python and R both round half to even
    theta_in = np.repeat(2.0, len(vars_use)) if theta is None else np.atleast_1d(np.asarray(theta, float))
    if len(theta_in) != len(vars_use):
        raise ValueError("Please specify theta for each variable")
    sigma = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    if sigma.size == 1 and nclust > 1:
        sigma = np.repeat(sigma, nclust)
    Phi, B_vec, levels = _design(meta_data, vars_use)
    N_b = Phi.sum(axis=1)
    if lambda_ is None:
        lambda_vec = np.array([-1.0])
    else:
        lam = np.atleast_1d(np.asarray(lambda_, dtype=np.float64))
        if not np.all(lam > 0):
            raise ValueError("Provided lambdas must be positive")
        if lam.size == 1:
            lambda_vec = np.concatenate([[0.0], np.repeat(lam[0], B_vec.sum())])
        elif lam.size == len(vars_use):
            lambda_vec = np.concatenate([[0.0], np.repeat(lam, B_vec)])
        else:
            raise ValueError("number of lambdas must match number of covariates")
    theta_vec = np.repeat(theta_in, B_vec)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_vec = theta_vec * (1.0 - np.exp(-(N_b / (nclust * float(tau))) ** 2))
    eps_h = epsilon_harmony if early_stop else -np.inf
    rng = RRandom.from_state(rng_state) if rng_state is not None else RRandom(seed)

    ho = _Harmony(X.T.copy(), Phi, sigma, theta_vec, lambda_vec, alpha,
                  max_iter_cluster, epsilon_cluster, eps_h, nclust, block_size, rng)
    _console(f"{N} cells, {X.shape[1]} dims, {Phi.shape[0]} batch levels, K={nclust}", verbose)
    ho.init_cluster(None if init_centers is None else np.asarray(init_centers, float).T)
    orders = None if block_orders is None else list(block_orders)
    n_iter = 0
    for it in range(1, max_iter + 1):
        ho.cluster(orders)
        ho.moe_correct_ridge()
        n_iter = it
        if ho._check_convergence(1):
            _console(f"converged after {it} iterations", verbose)
            break
    else:
        _console(f"stopped after {max_iter} iterations", verbose)
    ho.n_iter = n_iter
    ho.levels = levels
    if return_object:
        return ho
    return ho.Z_corr.T.copy()


def harmony(
    adata,
    batch_key: Union[str, Sequence[str]] = "Sample",
    *,
    use_rep: str = "X_birmt",
    key_added: str = "X_harmony",
    theta: Optional[Union[float, Sequence[float]]] = None,
    sigma: float = 0.1,
    lambda_: Optional[Union[float, Sequence[float]]] = 1.0,
    nclust: Optional[int] = None,
    max_iter: int = 10,
    early_stop: bool = True,
    seed: Optional[int] = 1,
    verbose: bool = True,
    **options,
):
    """Harmony batch correction of a low-dimensional embedding (ArchR ``addHarmony``).

    Port of R ``harmony`` 1.2.4 (``RunHarmony``) with ArchR's call pattern:
    the embedding in ``adata.obsm[use_rep]`` (cells × dims, e.g. the
    iterative-LSI output — ArchR passes the depth-filtered, scaled LSI from
    ``getReducedDims``) is corrected for the categorical covariate(s)
    ``batch_key`` and written to ``adata.obsm[key_added]``. With the same
    ``seed`` the result matches ``set.seed(seed); addHarmony(...)`` in R:
    the k-means initialisation (``stats::kmeans``, Hartigan–Wong,
    ``nstart = 10``) and the randomised block updates replay R's
    Mersenne-Twister stream.

    Arguments:
        adata: AnnData with the embedding in ``obsm[use_rep]``.
        batch_key: ``obs`` column(s) to integrate over (ArchR ``groupBy``).
        use_rep: ``obsm`` key of the input embedding.
        key_added: ``obsm`` key for the corrected embedding.
        theta: diversity penalty per covariate (default 2).
        sigma: soft-clustering width (default 0.1).
        lambda_: ridge penalty per covariate; ``None`` estimates it per
            cluster as ``alpha * E`` (harmony >= 1.0 automatic mode).
        nclust: number of clusters (default ``min(round(N / 30), 100)``).
        max_iter: maximum harmony rounds (default 10).
        early_stop: stop when the objective improves by < ``epsilon_harmony``.
        seed: R ``set.seed`` value to replay (default 1).
        verbose: print progress.
        **options: harmony's ``.options`` (``alpha``, ``tau``,
            ``block_size``, ``max_iter_cluster``, ``epsilon_cluster``,
            ``epsilon_harmony``) and parity hooks (``rng_state``,
            ``init_centers``, ``block_orders``).

    Returns:
        ``None``; sets ``adata.obsm[key_added]`` and
        ``adata.uns['harmony']`` (parameters, objective trace, iterations).
    """
    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm[{use_rep!r}] not found — run epi.tl.birmt (default embedding, obsm['X_birmt']) or pass use_rep='X_iterative_lsi' after epi.tl.iterative_lsi")
    keys = [batch_key] if isinstance(batch_key, str) else list(batch_key)
    missing = [k for k in keys if k not in adata.obs]
    if missing:
        raise KeyError(f"batch_key column(s) not in adata.obs: {missing}")
    ho = run_harmony(
        np.asarray(adata.obsm[use_rep]), adata.obs[keys], keys,
        theta=theta, sigma=sigma, lambda_=lambda_, nclust=nclust,
        max_iter=max_iter, early_stop=early_stop, seed=seed,
        return_object=True, verbose=verbose, **options,
    )
    adata.obsm[key_added] = ho.Z_corr.T.copy()
    adata.uns["harmony"] = {
        "params": {"batch_key": keys, "use_rep": use_rep, "theta": theta,
                   "sigma": sigma, "lambda": lambda_, "nclust": ho.K,
                   "max_iter": max_iter, "early_stop": early_stop, "seed": seed},
        "n_iter": ho.n_iter,
        "objective_harmony": np.asarray(ho.objective_harmony, dtype=np.float64),
        "kmeans_rounds": np.asarray(ho.kmeans_rounds),
        "batch_levels": ho.levels,
    }
