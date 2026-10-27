"""Synthetic tests of AEGLE (epione.tl.aegle): the maths of each step and planted-domain recovery."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

anndata = pytest.importorskip("anndata")

from epione.tl import _aegle as AE  # noqa: E402


def _grid(g):
    idx = np.arange(g * g).reshape(g, g)
    r, c = [], []
    for i in range(g):
        for j in range(g):
            if i + 1 < g:
                r.append(idx[i, j]); c.append(idx[i + 1, j])
            if j + 1 < g:
                r.append(idx[i, j]); c.append(idx[i, j + 1])
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(g * g, g * g))
    return (A + A.T).tocsr()


def _planted(g=30, n_peaks=3000, depth=0.03, seed=0):
    """Three horizontal bands plus a thin (2-row) layer; Bernoulli peaks with band-specific
    accessible sets and per-spot depth variation."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(g), g)
    cols = np.tile(np.arange(g), g)
    lab = np.where(rows < g // 3, 0, np.where(rows < 2 * g // 3, 1, 2))
    lab[(rows >= g // 2) & (rows < g // 2 + 2)] = 3
    base = rng.random(n_peaks) * depth
    prof = np.tile(base, (4, 1))
    for k in range(4):
        sel = rng.choice(n_peaks, n_peaks // 6, replace=False)
        prof[k, sel] *= 4.0
    scale = rng.lognormal(0, 0.3, size=g * g)
    P = np.clip(prof[lab] * scale[:, None], 0, 1)
    X = sp.csr_matrix((rng.random(P.shape) < P).astype(np.float32))
    a = anndata.AnnData(X, obs=pd.DataFrame({"array_row": rows, "array_col": cols}))
    a.obsm["spatial"] = np.c_[cols, -rows].astype(float)
    return a, lab


def test_tikhonov_solves_system():
    A = _grid(12)
    S = AE.Tikhonov(A, 3.0, tol=1e-10)
    B = np.random.default_rng(0).standard_normal((144, 3))
    X = S.apply(B)
    P = AE._sym_norm(A)
    np.testing.assert_allclose(4.0 * X - 3.0 * (P @ X), B, atol=1e-7)


def test_generalized_mp_edge_reduces_to_mp():
    g = 0.3
    assert AE.generalized_mp_edge(np.array([2.0]), np.array([1.0]), g) == pytest.approx(2.0 * (1 + np.sqrt(g)) ** 2, rel=1e-6)


def test_generalized_mp_edge_matches_simulation():
    A = _grid(20)
    m, n = 400, 2000
    mu, w = AE.laplacian_spectrum(A)
    rng = np.random.default_rng(1)
    for lam in (2.0, 8.0):
        S = AE.Tikhonov(A, lam, tol=1e-10)
        top = np.linalg.svd(S.apply(rng.standard_normal((m, n))), compute_uv=False)[0]
        edge = np.sqrt(n * AE.generalized_mp_edge(1 / (1 + lam * mu) ** 2, w, m / n))
        assert abs(top - edge) / edge < 0.05
        assert edge < np.sqrt(m) + np.sqrt(n)        # smoothing lowers the noise edge


def test_slq_spectrum_close_to_exact():
    A = _grid(25)
    mu, w = AE.laplacian_spectrum(A)
    mu2, w2 = AE.laplacian_spectrum(A, exact_max=10, n_probe=30, n_steps=50)
    for lam in (1.0, 4.0):
        e1 = AE.generalized_mp_edge(1 / (1 + lam * mu) ** 2, w, 0.2)
        e2 = AE.generalized_mp_edge(1 / (1 + lam * mu2) ** 2, w2, 0.2)
        assert abs(e1 - e2) / e1 < 0.02


def test_potts_gmm_recovers_mixture():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(m, 1.0, size=(200, 2)) for m in (0, 6, 12)])
    lab, R, info = AE.potts_gmm(Z, 3, seed=0)
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(np.repeat([0, 1, 2], 200), lab) > 0.98
    np.testing.assert_allclose(R.sum(1), 1.0)


def test_heldout_loglik_prefers_true_partition():
    a, lab = _planted(seed=2)
    Xtr, Xte = AE._thin(a.X, 0.5, np.random.default_rng(0))
    good = AE.heldout_loglik(Xtr, Xte, lab)
    bad = AE.heldout_loglik(Xtr, Xte, np.random.default_rng(1).permutation(lab))
    assert good > bad


def test_aegle_recovers_planted_domains_and_thin_layer():
    from sklearn.metrics import adjusted_rand_score
    a, lab = _planted(seed=3)
    AE.aegle(a, 4, lam=4.0, beta=2.0, bilateral=True, hops=1, row_norm="depth", seed=0)
    pred = a.obs["domain"].to_numpy(int)
    assert adjusted_rand_score(lab, pred) > 0.8
    thin = lab == 3
    # the 2-row layer is found as its own domain for most of its spots
    top = np.bincount(pred[thin]).argmax()
    assert np.mean(pred[thin] == top) > 0.7 and np.mean(pred[~thin] == top) < 0.05
    assert a.uns["X_aegle"]["modalities"][0]["rank"] >= 3


def test_spatial_domains_entry_point_and_selection():
    import epione as epi
    a, lab = _planted(g=24, n_peaks=2000, seed=4)
    epi.tl.spatial_domains(a, 4, method="aegle", select=True, lam_grid=(2.0, 8.0), beta_grid=(0.0, 2.0),
                           key_added="dom")
    assert a.obs["dom"].nunique() <= 4
    sel = a.uns["X_aegle"]["selection"]
    assert len(sel["loglik"]) == 4
    with pytest.raises(TypeError):
        epi.tl.spatial_domains(a, 4, method="kmeans", use_rep="X_aegle", use_pca=False, select=True)


def test_spatial_domains_default_is_aegle():
    import epione as epi
    a, lab = _planted(g=20, n_peaks=1500, seed=5)
    epi.tl.spatial_domains(a, 4, key_added="dom")           # no method: AEGLE on adata.X
    assert "X_aegle" in a.uns and a.obs["dom"].nunique() <= 4
