"""Synthetic tests of epi.tl.ot_batch (per-batch biwhitening + unbalanced
entropic OT barycenter alignment)."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from epione.tl import _ot_batch as ob


def test_sinkhorn_scale_marginals():
    rng = np.random.default_rng(0)
    A = sp.csr_matrix((rng.random((60, 40)) < 0.3).astype(float))
    A = A[np.diff(A.indptr) > 0]
    A = A[:, np.flatnonzero(np.bincount(A.indices, minlength=40) > 0)].tocsr()
    x, y, _ = ob.sinkhorn_scale(A, tol=1e-10)
    S = sp.diags(x) @ A @ sp.diags(y)
    m, n = A.shape
    np.testing.assert_allclose(np.asarray(S.sum(1)).ravel(), n, rtol=1e-8)
    np.testing.assert_allclose(np.asarray(S.sum(0)).ravel(), m, rtol=1e-8)


def test_mp_median_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for g in (1.0, 0.25):
        n = 600
        p = int(n / g)
        G = rng.standard_normal((n, p))
        ev = np.linalg.eigvalsh(G @ G.T / p)
        assert abs(np.median(ev) - ob.mp_median(g)) < 0.02


def test_balanced_sinkhorn_marginals():
    rng = np.random.default_rng(1)
    C = rng.random((50, 7))
    loga = np.full(50, -np.log(50))
    w = rng.random(7); w /= w.sum()
    f, g, _ = ob.unbalanced_sinkhorn(C, loga, np.log(w), 0.05, np.inf, np.inf, n_iter=5000, tol=1e-9)
    P = np.exp(loga[:, None] + np.log(w)[None, :] + (f[:, None] + g[None, :] - C) / 0.05)
    np.testing.assert_allclose(P.sum(1), 1 / 50, rtol=1e-6)
    np.testing.assert_allclose(P.sum(0), w, rtol=1e-4)


def test_unbalanced_sinkhorn_destroys_far_mass():
    # one target far away: with a small marginal penalty it receives ~no mass
    C = np.array([[0.0, 50.0], [0.0, 50.0], [0.1, 50.0]])
    loga = np.full(3, -np.log(3))
    logw = np.log([0.5, 0.5])
    f, g, _ = ob.unbalanced_sinkhorn(C, loga, logw, 0.05, 0.5, 0.5, n_iter=5000, tol=1e-9)
    P = np.exp(loga[:, None] + logw[None, :] + (f[:, None] + g[None, :] - C) / 0.05)
    assert P[:, 1].sum() < 1e-6 * P[:, 0].sum()


def _two_batches(rng, shift, drop_cluster=None, n=300, d=8):
    centers = rng.normal(size=(4, d)) * 6
    Z, lab, bat = [], [], []
    for b in range(2):
        for c in range(4):
            if b == 1 and c == drop_cluster:
                continue
            pts = centers[c] + rng.normal(size=(n, d))
            if b == 1:
                pts = pts + shift
            Z.append(pts); lab += [c] * n; bat += [b] * n
    return np.vstack(Z), np.array(lab), np.array(bat)


def _knn_batch_mixing(Z, bat, k=15):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[1][:, 1:]
    return float(np.mean(bat[idx] != bat[:, None]))


def _knn_label_purity(Z, lab, k=15):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[1][:, 1:]
    return float(np.mean(lab[idx] == lab[:, None]))


def _on_sphere(Z):
    Z = Z + 20 * np.r_[1.0, np.zeros(Z.shape[1] - 1)]
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def test_ot_align_removes_planted_shift():
    rng = np.random.default_rng(2)
    shift = np.r_[4.0, 0, 0, 3.0, np.zeros(4)]
    Z, lab, bat = _two_batches(rng, shift)
    Z = _on_sphere(Z)
    before = _knn_batch_mixing(Z, bat)
    Zc, info = ob.ot_align(Z, bat.astype(str), n_landmarks=20, eps=0.02, rho=np.inf,
                           seed=0, verbose=False)
    after = _knn_batch_mixing(Zc, bat)
    assert before < 0.1 and after > 0.45
    assert _knn_label_purity(Zc, lab) > 0.99
    for c in range(4):
        d0 = Z[(lab == c) & (bat == 0)].mean(0) - Z[(lab == c) & (bat == 1)].mean(0)
        d1 = Zc[(lab == c) & (bat == 0)].mean(0) - Zc[(lab == c) & (bat == 1)].mean(0)
        assert np.linalg.norm(d1) < 0.1 * np.linalg.norm(d0)


def test_ot_align_keeps_batch_specific_cluster():
    # batch 1 lacks cluster 1: with a relaxed landmark marginal it is neither
    # pulled onto other clusters nor moved, while the shared clusters align
    rng = np.random.default_rng(5)
    shift = np.r_[4.0, 0, 0, 3.0, np.zeros(4)]
    Z, lab, bat = _two_batches(rng, shift, drop_cluster=1)
    Z = _on_sphere(Z)
    Zc, _ = ob.ot_align(Z, bat.astype(str), n_landmarks=20, eps=0.02, rho=0.1,
                        seed=0, verbose=False)
    assert _knn_label_purity(Zc, lab) > 0.99
    m1 = lab == 1
    span = np.linalg.norm(Z[(lab == 0) & (bat == 0)].mean(0) - Z[(lab == 0) & (bat == 1)].mean(0))
    assert np.linalg.norm(Zc[m1].mean(0) - Z[m1].mean(0)) < 0.2 * span
    for c in (0, 2, 3):
        d1 = Zc[(lab == c) & (bat == 0)].mean(0) - Zc[(lab == c) & (bat == 1)].mean(0)
        assert np.linalg.norm(d1) < 0.2 * span
    # balanced OT (rho = inf) does force the missing type onto the others
    Zb, _ = ob.ot_align(Z, bat.astype(str), n_landmarks=20, eps=0.02, rho=np.inf,
                        seed=0, verbose=False)
    assert np.linalg.norm(Zb[m1].mean(0) - Z[m1].mean(0)) > np.linalg.norm(Zc[m1].mean(0) - Z[m1].mean(0))


def test_biwhitened_embedding_rank_and_api():
    import anndata as ad
    rng = np.random.default_rng(4)
    n, m, r = 600, 800, 3
    lab = rng.integers(0, r, n)
    R = (rng.random((r, m)) < 0.15).astype(float)
    P = 0.03 + 0.5 * R[lab]
    bat = np.repeat(["a", "b"], n // 2)
    P[bat == "b"] *= 0.5                       # batch b: half the depth
    X = sp.csr_matrix((rng.random((n, m)) < P).astype(np.float32))
    Z, info = ob.biwhitened_embedding(X, bat, min_cells=1)
    assert 2 <= info["rank"] <= 8
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0)
    a = ad.AnnData(X=X, obs=pd.DataFrame({"batch": bat}, index=[str(i) for i in range(n)]))
    ob.ot_batch(a, "batch", min_cells=1, n_landmarks=16, n_rounds=2, verbose=False)
    assert a.obsm["X_ot_batch"].shape[0] == n
    assert np.isfinite(a.obsm["X_ot_batch"]).all()
    assert "params" in a.uns["X_ot_batch"]


def test_biwhiten_keeps_cell_with_only_rare_features():
    # one cell whose only open features are rare within its batch must not crash
    rng = np.random.default_rng(6)
    n, m = 400, 300
    X = (rng.random((n, m)) < 0.2).astype(np.float32)
    X[0] = 0
    X[0, -3:] = 1                      # features open in only this cell ...
    X[1:, -3:] = 0
    X[1:3, -3:] = 1                    # ... and two others (below the floor)
    bat = np.repeat(["a", "b"], n // 2)
    Z, info = ob.biwhitened_embedding(sp.csr_matrix(X), bat, min_cells=1)
    assert np.isfinite(Z).all()
    assert info["batches"][0]["feature_floor"] <= 3


def test_biwhiten_handles_infeasible_low_degree_cell():
    # a cell with 2 open features among many cannot meet the Sinkhorn row target;
    # it is scaled out of sample instead of making the iteration diverge
    rng = np.random.default_rng(7)
    n, m = 300, 2000
    X = (rng.random((n, m)) < 0.15).astype(np.float32)
    X[0] = 0
    X[0, :2] = 1
    bat = np.repeat(["a", "b"], n // 2)
    Z, info = ob.biwhitened_embedding(sp.csr_matrix(X), bat, min_cells=1)
    assert np.isfinite(Z).all()
    assert info["batches"][0]["n_out_of_sample"] == 1
    assert info["batches"][1]["n_out_of_sample"] == 0
