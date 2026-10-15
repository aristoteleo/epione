"""CPU-vs-GPU equivalence of the torch backend.

Every public function with a ``device=`` path is compared with its NumPy /
SciPy implementation (``device="cpu"``) twice: on CUDA (skipped when no GPU is
visible, as on CI) and through the internal ``device=tdev`` hook, which
runs the very same torch code on the CPU so that CI checks it too.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from epione.core import _gpu  # noqa: E402

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture(params=["torch-cpu", pytest.param("cuda", marks=_CUDA)])
def tdev(request):
    return request.param


# ---------------------------------------------------------------------------
# torch helpers on the CPU
# ---------------------------------------------------------------------------

def _cpu():
    return _gpu.torch_device(_gpu.resolve_device("torch-cpu"))


def _dev(name):
    return _gpu.torch_device(_gpu.resolve_device(name))


@pytest.fixture
def X_sparse():
    rng = np.random.default_rng(0)
    X = sp.random(300, 120, density=0.08, format="csr", random_state=2, dtype=np.float64)
    X.data = rng.integers(1, 5, X.nnz).astype(np.float64)
    return X


def test_csr_round_trip(X_sparse):
    T = _gpu.to_torch_csr(X_sparse, _cpu())
    assert T.layout == torch.sparse_csr and T.dtype == torch.float64
    assert T.crow_indices().dtype == torch.int32
    back = _gpu.csr_from_torch(T)
    assert (back != X_sparse).nnz == 0
    ints = X_sparse.astype(np.int32)
    assert _gpu.to_torch_csr(ints, _cpu()).dtype == torch.float32
    rows = _gpu.csr_row_ids(T).numpy()
    assert np.array_equal(rows, np.repeat(np.arange(300), np.diff(X_sparse.indptr)))


def test_power_iterations_use_scipy_lu(tdev, monkeypatch):
    """The LU normaliser is scipy's ``lu(permute_l=True)``, exactly as in sklearn."""
    import scipy.linalg
    from epione.core._gpu_ops import randomized_svd
    calls = []
    orig = scipy.linalg.lu

    def spy(*a, **k):
        calls.append(k.get("permute_l"))
        return orig(*a, **k)
    monkeypatch.setattr(scipy.linalg, "lu", spy)
    X = np.random.default_rng(0).normal(size=(300, 12))
    randomized_svd(sp.csr_matrix(X), 3, dev=_dev(tdev), random_state=0, n_iter=4)
    assert len(calls) == 8 and all(calls)


@pytest.mark.parametrize("budget", [None, 4096, 100_000])   # resident / streamed / X only
def test_sparse_operand_matches_scipy(X_sparse, budget, tdev):
    dev = _dev(tdev)
    op = _gpu.SparseOperand(X_sparse, dev, torch.float64, budget=budget)
    assert op.resident == (budget is None)
    if budget == 100_000:
        assert op._Xd is not None and op._XTd is None
    if budget == 4096:
        assert op._Xd is None
    D = np.random.default_rng(1).normal(size=(120, 5))
    E = np.random.default_rng(2).normal(size=(300, 4))
    np.testing.assert_allclose(op.matmul(torch.as_tensor(D, device=dev)).cpu().numpy(), X_sparse @ D, rtol=1e-12)
    np.testing.assert_allclose(op.rmatmul(torch.as_tensor(E, device=dev)).cpu().numpy(), X_sparse.T @ E, rtol=1e-12)
    np.testing.assert_allclose((op.T @ torch.as_tensor(E, device=dev)).cpu().numpy(), X_sparse.T @ E, rtol=1e-12)


@pytest.mark.parametrize("budget", [None, 4096, 100_000])
def test_sparse_operand_value_fn(X_sparse, budget, tdev):
    dev = _dev(tdev)
    w = np.arange(1, 121, dtype=np.float64)
    r = np.arange(1, 301, dtype=np.float64)
    wt, rt = torch.as_tensor(w, device=dev), torch.as_tensor(r, device=dev)
    fn = lambda v, i, j: torch.log1p(v * rt[i] / wt[j])  # noqa: E731
    op = _gpu.SparseOperand(X_sparse, dev, torch.float64, budget=budget, value_fn=fn)
    Y = X_sparse.tocoo()
    ref = sp.csr_matrix((np.log1p(Y.data * r[Y.row] / w[Y.col]), (Y.row, Y.col)), shape=Y.shape)
    D = np.random.default_rng(3).normal(size=(120, 3))
    E = np.random.default_rng(4).normal(size=(300, 3))
    np.testing.assert_allclose(op.matmul(torch.as_tensor(D, device=dev)).cpu().numpy(), ref @ D, rtol=1e-12)
    np.testing.assert_allclose(op.rmatmul(torch.as_tensor(E, device=dev)).cpu().numpy(), ref.T @ E, rtol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(400, 90), (90, 400)])
def test_randomized_svd_matches_sklearn(dtype, shape, tdev):
    from sklearn.utils.extmath import randomized_svd as sk
    from epione.core._gpu_ops import randomized_svd
    X = sp.random(*shape, density=0.2, format="csr", random_state=5).astype(dtype)
    U0, s0, V0 = sk(X, 10, random_state=3)
    U1, s1, V1 = randomized_svd(X, 10, dev=_dev(tdev), random_state=3)
    assert U1.dtype == dtype
    tol = 1e-3 if dtype == np.float32 else 1e-8
    np.testing.assert_allclose(s1, s0, rtol=tol)
    # same sign convention -> columns agree directly
    np.testing.assert_allclose(U1[:, :5], U0[:, :5], atol=tol * 10)
    np.testing.assert_allclose(V1[:5], V0[:5], atol=tol * 10)


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan", "correlation"])
def test_knn_matches_sklearn(metric, tdev):
    from sklearn.neighbors import NearestNeighbors
    from epione.core._gpu_ops import knn
    X = np.random.default_rng(6).normal(size=(500, 12)).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=10, algorithm="brute", metric=metric).fit(X)
    d0, i0 = nn.kneighbors(X)
    i1, d1 = knn(X, 10, dev=_dev(tdev), metric=metric, budget=50_000)   # forces several blocks
    jac = np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(i0, i1)])
    assert jac > 0.999
    assert np.all(i1[:, 0] == np.arange(500)) and np.all(d1[:, 0] == 0)
    np.testing.assert_allclose(np.sort(d1, 1)[:, 1:], np.sort(d0, 1)[:, 1:], atol=1e-5)
    with pytest.raises(NotImplementedError):
        knn(X, 5, dev=_cpu(), metric="hamming")


# ---------------------------------------------------------------------------
# public functions: device=tdev vs device="cpu"
# ---------------------------------------------------------------------------

@pytest.fixture
def peaks():
    import anndata as ad
    rng = np.random.default_rng(0)
    n, p = 400, 1500
    lam = rng.gamma(0.3, 1.0, size=(1, p)) * rng.gamma(2, 1, size=(n, 1)) * 0.3
    X = sp.csr_matrix(rng.poisson(lam).astype(np.int32))
    a = ad.AnnData(X)
    a.var_names = [f"chr1:{i * 600}-{i * 600 + 500}" for i in range(p)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.varm["motif"] = sp.csr_matrix(rng.random((p, 30)) < 0.08)
    a.varm["bg_peaks"] = rng.integers(0, p, size=(p, 15))
    return a


def _abs_cos(A, B):
    return np.abs((A * B).sum(0) / np.linalg.norm(A, axis=0) / np.linalg.norm(B, axis=0))


def test_lsi_torch_matches_cpu(peaks, tdev):
    import epione as epi
    a, b = peaks.copy(), peaks.copy()
    epi.tl.lsi(a, n_components=10, random_state=0)
    epi.tl.lsi(b, n_components=10, random_state=0, device=tdev)
    assert _abs_cos(a.obsm["X_lsi"], b.obsm["X_lsi"]).min() > 0.9999


def test_iterative_lsi_one_round_torch_matches_cpu(peaks, tdev):
    import epione as epi
    a, b = peaks.copy(), peaks.copy()
    kw = dict(iterations=1, n_components=8, var_features=1000, total_features=1500,
              seed=1, verbose=False, sample_cells_pre=None)
    epi.tl.iterative_lsi(a, **kw)
    epi.tl.iterative_lsi(b, device=tdev, **kw)
    assert list(a.uns["X_iterative_lsi"]["kept_dims"]) == list(b.uns["X_iterative_lsi"]["kept_dims"])
    assert _abs_cos(a.obsm["X_iterative_lsi"], b.obsm["X_iterative_lsi"]).min() > 0.999


def test_iterative_lsi_two_rounds_torch_runs(peaks, tdev):
    pytest.importorskip("igraph")
    import epione as epi
    b = peaks.copy()
    epi.tl.iterative_lsi(b, iterations=2, n_components=8, var_features=500,
                         total_features=1500, seed=1, verbose=False,
                         device=tdev, n_neighbors=10)
    assert b.obsm["X_iterative_lsi"].shape[0] == b.n_obs


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_neighbors_torch_matches_cpu(peaks, metric, tdev):
    import epione as epi
    a, b = peaks.copy(), peaks.copy()
    emb = np.random.default_rng(1).normal(size=(peaks.n_obs, 10)).astype(np.float32)
    a.obsm["E"] = emb; b.obsm["E"] = emb
    epi.pp.neighbors(a, n_neighbors=12, use_rep="E", metric=metric, method="gauss")
    epi.pp.neighbors(b, n_neighbors=12, use_rep="E", metric=metric, method="gauss",
                     device=tdev)
    A, B = a.obsp["distances"], b.obsp["distances"]
    jac = np.mean([len(set(A[i].indices) & set(B[i].indices)) /
                   len(set(A[i].indices) | set(B[i].indices)) for i in range(A.shape[0])])
    assert jac > 0.999
    np.testing.assert_allclose(np.sort(A.toarray(), 1), np.sort(B.toarray(), 1), atol=1e-5)


@pytest.mark.parametrize("method", ["analytical", "sample"])
def test_chromvar_torch_matches_cpu(peaks, method, tdev):
    import epione as epi
    a, b = peaks.copy(), peaks.copy()
    epi.tl.compute_deviations(a, method=method, verbose=False)
    epi.tl.compute_deviations(b, method=method, verbose=False, device=tdev,
                              batch_cells=97)
    Za, Zb = a.obsm["motif_deviations"], b.obsm["motif_deviations"]
    assert np.array_equal(np.isnan(Za), np.isnan(Zb))
    np.testing.assert_allclose(Zb, Za, atol=1e-4, rtol=1e-4, equal_nan=True)
    np.testing.assert_allclose(b.obsm["motif_deviations_raw"], a.obsm["motif_deviations_raw"],
                               atol=1e-5, equal_nan=True)


def test_peak_to_gene_torch_matches_cpu(peaks, tdev):
    import pandas as pd
    import epione as epi
    a = peaks.copy()
    rng = np.random.default_rng(3)
    a.obsm["E"] = rng.normal(size=(a.n_obs, 8))
    G = rng.poisson(2, size=(a.n_obs, 40)).astype(np.float32)
    genes = pd.DataFrame({"gene_name": [f"g{i}" for i in range(40)], "chrom": "chr1",
                          "start": np.arange(40) * 20000, "end": np.arange(40) * 20000 + 5000,
                          "strand": "+"})
    kw = dict(rna=G, gene_names=list(genes.gene_name), gene_annotation=genes, use_rep="E",
              n_metacells=60, k_neighbors=20, max_distance=50_000, verbose=False, inplace=False)
    r0 = epi.tl.peak_to_gene(a, **kw)
    r1 = epi.tl.peak_to_gene(a, device=tdev, **kw)
    assert list(r0["peak"]) == list(r1["peak"]) and list(r0["gene"]) == list(r1["gene"])
    np.testing.assert_allclose(r1["correlation"], r0["correlation"], atol=1e-5)


@pytest.mark.parametrize("delta_scale", [1.0, 1e9])   # 1e9: every window rescored exactly
def test_moods_exact_scan(tdev, delta_scale, monkeypatch):
    import epione.tl._motif_scan_torch as mst
    monkeypatch.setattr(mst, "_DELTA_SCALE", delta_scale)
    MOODS = pytest.importorskip("MOODS")
    import MOODS.scan
    import MOODS.tools
    from epione.tl._motif_scan_torch import moods_any_hit
    rng = np.random.default_rng(0)
    bg = MOODS.tools.flat_bg(4)
    mats, th = [], []
    for L in [5, 6, 7, 9, 12, 18]:
        c = rng.integers(0, 20, size=(4, L)).astype(float)
        mats.append(MOODS.tools.log_odds(tuple(map(tuple, c.tolist())), bg, 1e-4))
    n = len(mats)
    mats = mats + [MOODS.tools.reverse_complement(m) for m in mats]
    th = [MOODS.tools.threshold_from_p(m, bg, 2e-3) for m in mats[:n]] * 2
    seqs = []
    for k in range(400):
        s = list(rng.choice(list("ACGTacgt"), int(rng.integers(3, 120))))
        if k % 4 == 0 and len(s) > 2:
            s[int(rng.integers(0, len(s)))] = "N"
        seqs.append("".join(s))
    sc = MOODS.scan.Scanner(7)
    sc.set_motifs(matrices=mats, bg=bg, thresholds=th)
    ref = np.zeros((len(seqs), n), np.uint8)
    for i, s in enumerate(seqs):
        r = sc.scan(s)
        for j in range(n):
            ref[i, j] = int(bool(r[j]) or bool(r[j + n]))
    got = moods_any_hit(seqs, mats, th, bg, n, dev=_dev(tdev), batch_size=37)
    assert ref.sum() > 50
    assert np.array_equal(ref, got)




@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_neighbors_umap_connectivities_torch_match_cpu(peaks, metric, tdev):
    pytest.importorskip("umap")
    import epione as epi
    a, b = peaks.copy(), peaks.copy()
    emb = (np.random.default_rng(2).normal(size=(peaks.n_obs, 10)) * 50).astype(np.float32)
    a.obsm["E"] = emb; b.obsm["E"] = emb
    epi.pp.neighbors(a, n_neighbors=12, use_rep="E", metric=metric)
    epi.pp.neighbors(b, n_neighbors=12, use_rep="E", metric=metric, device=tdev)
    np.testing.assert_allclose(a.obsp["connectivities"].toarray(),
                               b.obsp["connectivities"].toarray(), atol=1e-4)
