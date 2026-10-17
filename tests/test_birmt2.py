"""Tests for BiRMT-2 (epione.tl._birmt2): tile pyramid, quadratic-variance
biwhitening with KS-selected beta, k-mer profiles, Laplacian prior, and the
reduction to v1.

Run with:  pytest tests/test_birmt2.py -q
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def _peaks_adata(m=300, n_per=40, n_chrom=3, r=4, seed=0, p=0.04):
    """Binary cells x peaks with a planted rank-r cell-type structure and
    peak coordinates; peaks in 10 kb neighbourhoods share loadings."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, r, size=m)
    names, chroms, starts = [], [], []
    for c in range(n_chrom):
        pos = np.sort(rng.choice(np.arange(1, 2000) * 1000, size=n_per * 10, replace=False))
        for s in pos:
            names.append(f"chr{c + 1}:{s}-{s + 500}")
            chroms.append(f"chr{c + 1}")
            starts.append(s)
    n = len(names)
    starts = np.asarray(starts)
    block = (starts // 50000) + 100 * np.asarray([int(c[3:]) for c in chroms])
    _, block = np.unique(block, return_inverse=True)
    btype = rng.integers(0, r, size=block.max() + 1)
    P = np.full((m, n), p)
    P[labels[:, None] == btype[block][None, :]] = 4 * p
    Y = (rng.random((m, n)) < P).astype(np.float32)
    var = pd.DataFrame(index=names)
    return AnnData(sp.csr_matrix(Y), var=var), labels


def test_tile_map_aggregates_by_midpoint():
    from epione.tl._birmt2 import tile_map
    chrom = np.array(["chr1", "chr1", "chr1", "chr2"])
    mid = np.array([100, 4999, 5001, 100])
    A = tile_map(chrom, mid, 5000).toarray()
    assert A.shape == (4, 3)
    assert (A.sum(1) == 1).all()
    assert A[0].argmax() == A[1].argmax() != A[2].argmax()
    assert A[3].argmax() not in (A[0].argmax(), A[2].argmax())


def test_peak_coordinates_from_names():
    from epione.tl._birmt2 import peak_coordinates
    a = AnnData(sp.csr_matrix((2, 2)), var=pd.DataFrame(index=["chr1:10-20", "chrX:5-9"]))
    c, s, e = peak_coordinates(a)
    assert list(c) == ["chr1", "chrX"] and list(s) == [10, 5] and list(e) == [20, 9]


def test_ks_to_mp_small_for_pure_noise():
    from epione.tl._birmt2 import ks_to_mp
    rng = np.random.default_rng(0)
    m, n = 400, 2000
    E = rng.standard_normal((m, n))
    ev = np.linalg.eigvalsh(E @ E.T / n)
    assert ks_to_mp(ev, m / n) < 0.05
    assert ks_to_mp(2 * ev, m / n) > 0.3


def test_biwhiten_qv_selects_quadratic_beta_for_negative_binomial():
    """NB counts have Var = mu + mu^2/k: a quadratic (beta > 0) variance
    function must fit the MP law better than Poisson (beta = 0)."""
    from epione.tl._birmt2 import biwhiten_qv
    from epione.tl._birmt import mp_rank_svd
    rng = np.random.default_rng(1)
    m, n, r = 400, 1500, 3
    A = rng.gamma(2.0, 1.0, size=(m, r))
    B = rng.gamma(1.0, 1.0, size=(n, r))
    mu = A @ B.T
    mu *= 3.0 / mu.mean()
    k = 1.0
    Y = rng.negative_binomial(k, k / (k + mu)).astype(np.float64)
    Y = sp.csr_matrix(Y)
    bw = biwhiten_qv(Y, n_subsample=400)
    tab = bw["table"]
    assert bw["beta"] > 0.2, tab
    assert bw["ks"] < tab[tab[:, 0] == 0.0, 2][0]
    _, _, _, rank, _ = mp_rank_svd(bw["Yw"], bw["edge"], k_start=16, k_max=32, n_jobs=1)
    assert rank == r


def test_biwhiten_qv_binary_equals_v1():
    from epione.tl._birmt2 import biwhiten_qv
    from epione.tl._birmt import biwhiten
    a, _ = _peaks_adata(seed=2)
    Y = a.X.tocsr()
    Y = Y[:, np.asarray(Y.sum(0)).ravel() > 0]
    b1 = biwhiten(Y, n_jobs=1)
    b2 = biwhiten_qv(Y, n_jobs=1)
    assert b2["beta"] == 0.0
    np.testing.assert_allclose(b2["alpha"], b1["alpha"], rtol=1e-6)
    np.testing.assert_allclose(b2["Yw"].toarray(), b1["Yw"].toarray(), rtol=1e-5)


def test_kmer_profiles_canonical_counts():
    from epione.tl._birmt2 import kmer_profiles, _LUT
    seqs = [b"ACGTAC", b"GTACGT", b"AAAANA"]
    codes = np.stack([_LUT[np.frombuffer(s, np.uint8)] for s in seqs])
    K, gc = kmer_profiles(codes, k=3)
    assert K.shape[1] == 32                      # canonical 3-mers
    # reverse complements give identical canonical profiles
    np.testing.assert_array_equal(K[0].toarray(), K[1].toarray())
    assert K[0].sum() == 4
    assert K[2].sum() == 2                       # windows containing N are skipped
    np.testing.assert_allclose(gc, [0.5, 0.5, 0.0])


def test_tikhonov_identity_at_zero_and_damps_rough_components():
    from epione.tl._birmt2 import tikhonov_transform
    n = 200
    # path graph: smooth vector = low graph frequency
    A = sp.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], format="csr")
    d = np.asarray(A.sum(1)).ravel()
    L = sp.identity(n) - sp.diags(1 / np.sqrt(d)) @ A @ sp.diags(1 / np.sqrt(d))
    x = np.linspace(0, 1, n)
    v1 = np.cos(np.pi * x); v1 /= np.linalg.norm(v1)
    v2 = (-1.0) ** np.arange(n); v2 -= v2 @ v1 * v1; v2 /= np.linalg.norm(v2)
    Vt = np.stack([v1, v2])
    M0, _ = tikhonov_transform(Vt, L, 0.0)
    np.testing.assert_allclose(M0, np.eye(2), atol=1e-6)
    M, _ = tikhonov_transform(Vt, L.tocsr(), 5.0)
    assert M[0, 0] > 0.9 and M[1, 1] < 0.2


def test_birmt2_reduces_to_v1_and_tiles_run():
    import epione as epi
    a, labels = _peaks_adata(seed=3)
    b = a.copy()
    epi.tl.birmt(a, cluster=False, n_jobs=1, version=1)
    epi.tl.birmt(b, cluster=False, n_jobs=1, version=2, shrinkage="frobenius", svd="v1")
    np.testing.assert_allclose(np.abs(b.obsm["X_birmt"]), np.abs(a.obsm["X_birmt"]), rtol=1e-4, atol=1e-7)
    c = a.copy()
    epi.tl.birmt(c, cluster=False, n_jobs=1, version=2, tile_sizes=[10000, 50000])
    info = c.uns["X_birmt"]
    assert [lv["name"] for lv in info["levels"]][0] == "peaks"
    assert c.obsm["X_birmt"].shape[1] >= a.obsm["X_birmt"].shape[1]
    # planted types are separated in the multiscale embedding
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    km = KMeans(4, n_init=10, random_state=0).fit_predict(c.obsm["X_birmt"])
    assert adjusted_rand_score(labels, km) > 0.9


def test_version1_rejects_v2_options():
    import pytest
    import epione as epi
    a, _ = _peaks_adata(seed=4)
    with pytest.raises(TypeError):
        epi.tl.birmt(a, cluster=False, tile_sizes=[1000], version=1)


def test_score_shrinkage_is_frobenius_without_feature_cosine():
    """x c c~ equals the GD Frobenius shrinker; the cell-score shrinker x c
    is larger (c~ < 1) and tends to the true value for strong spikes."""
    from epione.tl._birmt2 import score_shrinkage
    from epione.tl._birmt import shrink_singular_values
    m, n = 500, 5000
    beta = m / n
    s = np.sqrt(n) * np.array([1.35, 1.5, 2.0, 4.0, 50.0])
    fro = shrink_singular_values(s, m, n, kind="frobenius")
    sc = score_shrinkage(s, m, n)
    y = s / np.sqrt(n)
    t = y ** 2 - beta - 1
    x = np.sqrt((t + np.sqrt(t ** 2 - 4 * beta)) / 2)
    ct = np.sqrt((x ** 4 - beta) / (x ** 4 + x ** 2))
    np.testing.assert_allclose(fro, sc * ct, rtol=1e-10)
    assert (sc >= fro).all()
    assert abs(sc[-1] / (np.sqrt(n) * x[-1]) - 1) < 1e-3
    assert score_shrinkage(np.array([0.5 * np.sqrt(n)]), m, n)[0] == 0.0


def _binary(m=300, n=2000, r=5, seed=0, scale=0.05):
    rng = np.random.default_rng(seed)
    A = rng.gamma(1.0, 1.0, size=(m, r))
    B = rng.gamma(0.5, 1.0, size=(n, r))
    P = A @ B.T
    P *= scale / P.mean()
    Y = (rng.random((m, n)) < np.minimum(P, 1)).astype(np.float32)
    Y = Y[:, Y.sum(0) > 0]
    Y = Y[Y.sum(1) > 0]
    return sp.csr_matrix(Y)


def test_fast_kernels_match_scipy():
    from epione.tl._birmt2 import _fast_kernels, _filter_columns, BinaryBiwhitened
    Y = _binary(seed=5)
    K = _fast_kernels()
    tp, ti = K["pat_transpose"](Y.indptr, Y.indices, Y.shape[1])
    Yt = sp.csr_matrix((np.ones(len(ti)), ti, tp), shape=Y.shape[::-1])
    np.testing.assert_array_equal(Yt.toarray(), Y.T.toarray())
    Z, feats = _filter_columns(Y, 3)
    np.testing.assert_array_equal(Z.toarray(), Y[:, np.asarray(Y.sum(0)).ravel() >= 3].toarray())
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0.5, 2, Y.shape[0]), rng.uniform(0.5, 2, Y.shape[1])
    op = BinaryBiwhitened(Y, x, y)
    D = np.sqrt(x)[:, None] * Y.toarray() * np.sqrt(y)[None, :]
    V = rng.standard_normal((Y.shape[1], 3))
    U = rng.standard_normal((Y.shape[0], 3))
    np.testing.assert_allclose(op.matmat(V), D @ V, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(op.rmatmat(U), D.T @ U, rtol=1e-10, atol=1e-10)
    idx = np.array([0, 7, 9])
    S = Y[idx]
    sp_, si = K["pat_transpose"](S.indptr, S.indices, S.shape[1])
    G = K["pat_gram"](S.indptr, S.indices, sp_, si, np.sqrt(x[idx]), np.sqrt(y), 3)
    np.testing.assert_allclose(G, D[idx] @ D[idx].T, rtol=1e-10)


def test_sinkhorn_binary_matches_v1():
    from epione.tl._birmt2 import sinkhorn_binary
    from epione.tl._birmt import sinkhorn_scale
    Y = _binary(seed=6)
    x1, y1, _ = sinkhorn_scale(Y, tol=1e-10)
    x2, y2, _ = sinkhorn_binary(Y, Y.T.tocsr(), tol=1e-10)
    np.testing.assert_allclose(x2, x1, rtol=1e-7)
    np.testing.assert_allclose(y2, y1, rtol=1e-7)


def test_block_krylov_matches_exact_svd():
    from epione.tl._birmt2 import BinaryBiwhitened, block_krylov_svd, sinkhorn_binary
    Y = _binary(m=400, n=3000, r=6, seed=7, scale=0.15)
    x, y, _ = sinkhorn_binary(Y, Y.T.tocsr(), tol=1e-10)
    op = BinaryBiwhitened(Y, x, y)
    D = np.sqrt(x)[:, None] * Y.toarray() * np.sqrt(y)[None, :]
    s_exact = np.linalg.svd(D, compute_uv=False)
    U, s = block_krylov_svd(op, 48, block=16, seed=0)
    edge = np.sqrt(Y.shape[0]) + np.sqrt(Y.shape[1])      # MP edge, unit noise
    top = s_exact > 1.2 * edge                             # well-separated signal values
    assert top.sum() >= 3
    np.testing.assert_allclose(s[:top.sum()], s_exact[top], rtol=1e-4)
    # Ritz values are lower bounds of the true singular values
    assert np.all(s[:30] <= s_exact[:30] * (1 + 1e-10))


def test_fast_denoise_recovers_planted_rank_and_clusters():
    import epione as epi
    from sklearn.metrics import adjusted_rand_score
    a, labels = _peaks_adata(m=400, seed=8)
    epi.tl.birmt(a, version=2, n_jobs=1, cluster=True)
    info = a.uns["X_birmt"]
    assert info["levels"][0]["rank"] >= 3
    assert adjusted_rand_score(labels, a.obs["birmt"].astype(str)) > 0.9
    assert "X_birmt_W" in a.obsp


def test_default_embedding_path():
    """Default pipeline: epi.tl.birmt() -> BiRMT-2 embedding in obsm['X_birmt']
    (no label-free partition), then Epione's neighbors + Leiden, and the
    helpers whose use_rep defaults to 'X_birmt'."""
    import epione as epi
    from sklearn.metrics import adjusted_rand_score
    a, labels = _peaks_adata(m=400, seed=9)
    epi.tl.birmt(a, n_jobs=1)
    assert a.uns["X_birmt"]["version"] == 2
    assert "birmt" not in a.obs and "X_birmt_W" not in a.obsp
    epi.pp.neighbors(a, use_rep="X_birmt", n_neighbors=15)
    epi.tl.clusters(a)
    key = a.obs.columns[-1]
    from sklearn.metrics import homogeneity_score
    # Leiden at resolution 1 may split planted types, but must not mix them
    assert homogeneity_score(labels, a.obs[key].astype(str)) > 0.9
    import inspect
    for f in (epi.tl.harmony, epi.tl.add_impute_weights, epi.tl.peak_to_gene, epi.tl.coaccessibility,
              epi.tl.add_trajectory, epi.tl.correlate_matrices):
        assert inspect.signature(f).parameters["use_rep"].default == "X_birmt", f
    a.obs["batch"] = np.where(np.arange(a.n_obs) % 2, "b1", "b2")
    epi.tl.harmony(a, batch_key="batch", verbose=False)
    assert a.obsm["X_harmony"].shape == a.obsm["X_birmt"].shape


def test_version1_still_default_clusters():
    import epione as epi
    a, _ = _peaks_adata(m=300, seed=10)
    epi.tl.birmt(a, version=1, n_seeds=2, resolutions=[0.5, 1.0], n_jobs=1)
    assert "birmt" in a.obs and "rank" in a.uns["X_birmt"]


def test_sinkhorn_infeasible_rows_fitted_out_of_sample():
    """A row with fewer than n/m non-zeros makes the Sinkhorn scaling
    infeasible; it is placed by the Nystrom extension and the other rows
    keep the embedding of the fit without it."""
    from epione.tl._birmt2 import fast_denoise_binary
    Y = _binary(m=200, n=4000, r=5, seed=11, scale=0.1).tolil()
    Y[0, :] = 0
    Y[0, [3, 7]] = 1                     # 2 non-zeros < n/m = 20
    Y = sp.csr_matrix(Y, dtype=np.float32)
    lv = fast_denoise_binary(Y, n_jobs=1)
    assert lv["n_out_of_sample"] == 1
    Z = lv["U"] * lv["s"] * lv["row_scale"][:, None]
    assert np.isfinite(Z).all()
    ref = fast_denoise_binary(Y[1:], n_jobs=1)
    Zr = ref["U"] * ref["s"] * ref["row_scale"][:, None]
    np.testing.assert_allclose(np.abs(Z[1:]), np.abs(Zr), rtol=1e-6, atol=1e-9)
