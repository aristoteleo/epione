"""Tests for epione.tl.birmt (biwhitening / MP rank / shrinkage / DS kernel /
diffusion time / Markov stability).

Run with:  pytest tests/test_birmt.py -q
"""
import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def _planted_poisson(m=500, n=1500, r=8, seed=0):
    rng = np.random.default_rng(seed)
    # heteroskedastic, strictly positive rank-r intensity
    A = rng.gamma(2.0, 1.0, size=(m, r)) * rng.uniform(0.3, 3.0, size=(m, 1))
    B = rng.gamma(0.7, 1.0, size=(n, r)) * rng.uniform(0.3, 3.0, size=(n, 1))
    X = A @ B.T
    X *= 2.0 / X.mean()
    return sp.csr_matrix(rng.poisson(X).astype(np.float64)), X


def test_sinkhorn_scales_to_targets():
    from epione.tl._birmt import sinkhorn_scale
    Y, _ = _planted_poisson(m=200, n=400, r=3, seed=1)
    x, y, _ = sinkhorn_scale(Y, tol=1e-10)
    S = sp.diags(x) @ Y @ sp.diags(y)
    np.testing.assert_allclose(np.asarray(S.sum(1)).ravel(), 400, rtol=1e-8)
    np.testing.assert_allclose(np.asarray(S.sum(0)).ravel(), 200, rtol=1e-8)


def test_mp_median_known_values():
    from epione.tl._birmt import mp_median
    # gamma = 1: Gavish & Donoho 2014 tabulate omega(1) = 2.858 with
    # omega = lambda*(1) / sqrt(mu_1), lambda*(1) = 4/sqrt(3)  ->  mu_1 = 0.6528
    assert abs(mp_median(1.0) - (4 / np.sqrt(3) / 2.858) ** 2) < 1e-3
    # small gamma -> concentrates at 1
    assert abs(mp_median(1e-4) - 1.0) < 0.02


def test_mp_rank_recovers_planted_rank_poisson():
    """Algorithm 1 of Landa et al. (Poisson, alpha = 1) recovers r = 8."""
    from epione.tl._birmt import biwhiten, mp_rank_svd
    Y, X = _planted_poisson(m=500, n=1500, r=8, seed=0)
    bw = biwhiten(Y, noise_scale=1.0, n_jobs=2)
    _, s, _, r, _ = mp_rank_svd(bw["Yw"], bw["edge"], k_start=16, k_max=64, n_jobs=2)
    assert r == 8, (r, s[:12], bw["edge"])
    # the adaptive noise scale (Eq. 24) is ~1 for Poisson data
    bw2 = biwhiten(Y, noise_scale="mp_median", n_jobs=2)
    assert 0.9 < bw2["alpha"] < 1.1
    _, _, _, r2, _ = mp_rank_svd(bw2["Yw"], bw2["edge"], k_start=16, k_max=64, n_jobs=2)
    assert r2 == 8


def test_mp_rank_recovers_planted_rank_binary():
    """Bernoulli data: biwhitening of Y with the adaptive noise scale."""
    from epione.tl._birmt import biwhiten, mp_rank_svd
    rng = np.random.default_rng(3)
    m, n, r = 600, 3000, 6
    A = rng.gamma(1.0, 1.0, size=(m, r))
    B = rng.gamma(0.5, 1.0, size=(n, r))
    P = A @ B.T
    P = np.clip(P * (0.08 / P.mean()), 0, 0.9)
    Y = sp.csr_matrix((rng.random((m, n)) < P).astype(np.float64))
    bw = biwhiten(Y, n_jobs=2)
    assert bw["alpha"] < 1.0           # Bernoulli variance p(1-p) < p
    _, s, _, rk, _ = mp_rank_svd(bw["Yw"], bw["edge"], k_start=16, k_max=64, n_jobs=2)
    assert rk == r, (rk, s[:10], bw["edge"])


def test_shrinkers():
    from epione.tl._birmt import shrink_singular_values, optimal_hard_threshold_coef
    m, n = 100, 400
    beta = 0.25
    edge = (1 + np.sqrt(beta)) * np.sqrt(n)
    s = np.array([0.5 * edge, 0.99 * edge, 1.5 * edge, 5 * edge])
    f = shrink_singular_values(s, m, n, kind="frobenius")
    assert f[0] == 0 and f[1] == 0 and 0 < f[2] < s[2] and 0 < f[3] < s[3]
    # Frobenius shrinker formula at y = 3 (natural scale): sqrt((9-1.25)^2-1)/3
    y = 3.0
    got = shrink_singular_values(np.array([y * np.sqrt(n)]), m, n, kind="frobenius")[0] / np.sqrt(n)
    assert abs(got - np.sqrt((y * y - beta - 1) ** 2 - 4 * beta) / y) < 1e-12
    # 4/sqrt(3) for square matrices
    assert abs(optimal_hard_threshold_coef(1.0) - 4 / np.sqrt(3)) < 1e-12
    # operator shrinker >= Frobenius shrinker
    o = shrink_singular_values(s, m, n, kind="operator")
    assert np.all(o >= f - 1e-12)


def test_ds_kernel_doubly_stochastic_and_symmetric():
    from epione.tl._birmt import ds_kernel
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(0, 1, (150, 5)), rng.normal(6, 3, (150, 5))])
    W, info = ds_kernel(Z, n_neighbors=15, n_jobs=2)
    assert abs(W - W.T).max() < 1e-12
    np.testing.assert_allclose(np.asarray(W.sum(1)).ravel(), 1.0, atol=1e-6)
    assert W.diagonal().max() == 0
    assert info["epsilon"] > 0


def test_diffusion_time_gap_rule():
    from epione.tl._birmt import diffusion_time_gap
    lam = np.array([0.99, 0.98, 0.97, 0.60, 0.55, 0.50])
    t, q = diffusion_time_gap(lam, delta=0.25)
    assert q == 3
    assert t == int(np.ceil(np.log(0.25) / np.log(0.60 / 0.97)))
    assert (0.60 / 0.97) ** t <= 0.25 < (0.60 / 0.97) ** (t - 1)


def test_nvi():
    from epione.tl._birmt import nvi
    a = np.array([0, 0, 1, 1, 2, 2])
    assert nvi(a, a) == 0
    assert nvi(a, np.array([5, 5, 3, 3, 9, 9])) == 0
    assert 0 < nvi(a, np.array([0, 1, 0, 1, 0, 1])) <= 1


def _blocks_atac(n_per=150, groups=4, n_features=3000, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(groups), n_per)
    base = rng.uniform(0.01, 0.08, size=n_features)
    P = np.tile(base, (groups, 1))
    for g, fs in enumerate(np.array_split(rng.permutation(n_features), groups)):
        P[g, fs[: len(fs) // 2]] += 0.25
    depth = rng.lognormal(0, 0.4, size=len(labels))
    prob = np.clip(P[labels] * depth[:, None], 0, 0.95)
    X = sp.csr_matrix((rng.random(prob.shape) < prob).astype(np.int32))
    return AnnData(X=X, obs={"group": labels.astype(str)})


def test_birmt_end_to_end_recovers_groups():
    import epione as epi
    from sklearn.metrics import adjusted_rand_score
    ad = _blocks_atac()
    epi.tl.birmt(ad, n_seeds=4, resolutions=[0.3, 0.5, 1.0, 2.0], n_jobs=1)
    assert ad.obsm["X_birmt"].shape[0] == ad.n_obs
    info = ad.uns["X_birmt"]
    assert info["rank"] >= 3          # 4 groups -> rank >= 3 (+ mean)
    assert info["t"] >= 1
    ari = adjusted_rand_score(ad.obs["group"], ad.obs["birmt"])
    assert ari > 0.95, ari
