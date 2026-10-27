"""Spatial-domain method ports: COSMOS, PRESENT, INSTINCT, CANDIES (fast synthetic tests).

Parity with the upstream implementations is recorded in tests/parity/spatial_ports/
(test_spatial_ports_parity.py, skipped unless EPIONE_SPATIAL_PARITY_DIR is set).
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")
ad = pytest.importorskip("anndata")

import epione as epi  # noqa: E402
from epione.tl import _gnn  # noqa: E402


def _grid(n_side=8):
    r, c = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
    return np.c_[c.ravel(), r.ravel()].astype(float)


def _knn_edges(xy, k=4):
    from sklearn.neighbors import kneighbors_graph

    g = kneighbors_graph(xy, k).tocoo()
    return torch.from_numpy(np.vstack([g.row, g.col]).astype(np.int64))


def _synthetic(n_side=8, n_peaks=120, n_genes=60, seed=0, domains=2):
    rng = np.random.default_rng(seed)
    xy = _grid(n_side)
    lab = (xy[:, 0] >= n_side / 2).astype(int) if domains == 2 else (xy[:, 0] * domains // n_side).astype(int)
    rate_p = np.where(lab[:, None] == (np.arange(n_peaks) % domains)[None, :], 2.0, 0.3)
    rate_g = np.where(lab[:, None] == (np.arange(n_genes) % domains)[None, :], 6.0, 1.0)
    atac = ad.AnnData(sp.csr_matrix(rng.poisson(rate_p).astype(np.float32)))
    rna = ad.AnnData(sp.csr_matrix(rng.poisson(rate_g).astype(np.float32)))
    names = [f"s{i}" for i in range(xy.shape[0])]
    for a in (atac, rna):
        a.obs_names = names
        a.obsm["spatial"] = xy.copy()
        a.obs["truth"] = lab.astype(str)
    atac.var_names = [f"chr1:{1000 + 600 * i}-{1500 + 600 * i}" for i in range(n_peaks)]
    rna.var_names = [f"g{i}" for i in range(n_genes)]
    return atac, rna


# ---------------------------------------------------------------- layers
def test_gcnconv_matches_dense_formula():
    xy = _grid(6)
    ei = _knn_edges(xy, 3)
    n = xy.shape[0]
    x = torch.randn(n, 5)
    conv = _gnn.GCNConv(5, 3)
    A = torch.zeros(n, n)
    A[ei[1], ei[0]] = 1.0  # message j -> i lands in row i
    A = A + torch.eye(n)
    deg = A.sum(1)  # in-degree incl. self loop (PyG: degree of the target)
    Dm = torch.diag(deg.pow(-0.5))
    want = Dm @ A @ Dm @ (x @ conv.lin.weight.T) + conv.bias
    assert torch.allclose(conv(x, ei), want, atol=1e-5)
    # sparse-adjacency path: symmetric normalised adjacency + I, renormalised (CANDIES' ATAC AE)
    S = (A + A.T).clamp(max=1).to_sparse()
    out = conv(x, S)
    S2 = S.to_dense() + torch.eye(n)
    d2 = torch.diag(S2.sum(0).pow(-0.5))
    assert torch.allclose(out, d2 @ S2 @ d2 @ (x @ conv.lin.weight.T) + conv.bias, atol=1e-5)


def test_gatconv_attention_is_a_softmax_per_target():
    xy = _grid(5)
    ei = _knn_edges(xy, 4)
    conv = _gnn.GATConv(6, 4, dropout=0.0).eval()
    out, (eidx, alpha) = conv(torch.randn(xy.shape[0], 6), ei, return_attention_weights=True)
    sums = _gnn.scatter_sum(alpha, eidx[1], xy.shape[0])
    assert out.shape == (xy.shape[0], 4)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_layer_init_consumes_rng_like_pyg():
    torch.manual_seed(0)
    a = _gnn.GCNConv(7, 3)
    torch.manual_seed(0)
    _ = torch.empty(3, 7).uniform_()  # Linear() at construction
    w = torch.empty(3, 7)
    bound = (6.0 / 10) ** 0.5
    w.uniform_(-bound, bound)  # GCNConv.reset_parameters() redraws
    assert torch.equal(a.lin.weight.data, w)


# ---------------------------------------------------------------- COSMOS
def test_cosmos_runs_and_is_seeded():
    atac, rna = _synthetic()
    kw = dict(n_neighbors=4, z_dim=8, wnn_epoch=5, total_epoch=12, min_stop=2, random_seed=3, device="cpu",
              spatial_regularization_strength=0.01, edge_subset_sz=2000)
    pytest.importorskip("scanpy")
    e1 = epi.tl.cosmos(rna, atac, **kw).copy()
    e2 = epi.tl.cosmos(rna.copy(), atac.copy(), **kw)
    assert e1.shape == (atac.n_obs, 8)
    assert np.allclose(e1, e2)
    w = atac.obsm["X_cosmos_weights"]
    assert w.shape == (atac.n_obs, 2) and np.allclose(w.sum(1), 1, atol=1e-5)


# ---------------------------------------------------------------- PRESENT
def test_present_rna_atac_and_partition_fallback(monkeypatch):
    pytest.importorskip("skmisc")
    from epione.tl import _present as P

    atac, rna = _synthetic(n_side=10, n_peaks=150, n_genes=80)
    out = epi.tl.present(rna, atac, num_hvg=40, epochs=2, batch_size=40, nclusters=None, device="cpu",
                         random_seed=0, d_lat=12)
    assert out.obsm["embeddings"].shape == (atac.n_obs, 12)
    assert atac.obsm["X_present"].shape == (atac.n_obs, 12)

    monkeypatch.setattr(P, "_metis", lambda ei, n, parts, coords=None: (
        torch.from_numpy(P._coordinate_bisection(np.asarray(coords, float), parts)), "coordinate_bisection"))
    out2 = epi.tl.present(adata_atac=atac.copy(), epochs=1, batch_size=40, nclusters=None, device="cpu", d_lat=12)
    assert out2.uns["present"]["partitioner"] == "coordinate_bisection"


def test_coordinate_bisection_is_balanced():
    from epione.tl._present import _coordinate_bisection

    lab = _coordinate_bisection(_grid(10), 7)
    counts = np.bincount(lab)
    assert counts.size == 7 and counts.max() - counts.min() <= 1


def test_cluster_batches_cover_every_node_once():
    from epione.tl import _present as P

    xy = _grid(9)
    from sklearn.neighbors import kneighbors_graph

    g = P._geometric_data(kneighbors_graph(xy, 6, include_self=False).T.copy(), torch.arange(xy.shape[0]))
    cb = P._ClusterBatches(g, 6, coords=xy)
    seen = torch.cat([b.node_index for b in cb.loader(batch_size=2, shuffle=True)])
    assert sorted(seen.tolist()) == list(range(xy.shape[0]))


# ---------------------------------------------------------------- INSTINCT
def test_peak_sets_alignment_merges_and_sums():
    from epione.tl._instinct import peak_sets_alignment

    a = ad.AnnData(sp.csr_matrix(np.array([[1, 2, 3], [0, 1, 0]], dtype=np.int64)))
    a.var_names = ["chr1:100-200", "chr1:150-260", "chr2:10-60"]
    b = ad.AnnData(sp.csr_matrix(np.array([[5, 7], [1, 0]], dtype=np.int64)))
    b.var_names = ["chr1:120-180", "chr3:5-50"]
    out = peak_sets_alignment([a, b])
    assert list(out[0].var_names) == ["chr1:100-260"]  # chr2 / chr3 not shared
    assert out[0].X.toarray().ravel().tolist() == [3, 1]
    assert out[1].X.toarray().ravel().tolist() == [5, 1]


def test_instinct_two_slices():
    s1, _ = _synthetic(seed=1)
    s2, _ = _synthetic(seed=2)
    concat = epi.tl.instinct([s1, s2], ["a", "b"], n_pcs=10, training_steps=(3, 3), k=5, hidden_dims_G=(8,),
                             latent_dim=4, hidden_dims_D=(8,), n_domains=2, device="cpu")
    assert concat.obsm["X_instinct"].shape == (s1.n_obs + s2.n_obs, 4)
    assert s1.obsm["X_instinct"].shape == (s1.n_obs, 4)
    assert set(concat.obs["slice_name"]) == {"a", "b"}


# ---------------------------------------------------------------- CANDIES
def test_candies_small_cpu():
    pytest.importorskip("skmisc")
    pytest.importorskip("mclust_py")
    atac, rna = _synthetic(n_side=8, n_peaks=200, n_genes=80)
    emb = epi.tl.candies(rna, atac, rna_min_cells=1, n_top_genes=40, n_lsi=6, zinb_epochs=3, atac_epochs=3,
                         latent_dim=8, diff_batch_size=32, diff_hidden_size=8, diff_epochs=2, diffusion_step=5,
                         diff_depth=1, diff_heads=2, integration_epochs=3, n_domains=2, device="cpu")
    assert emb.shape == (atac.n_obs, 64)  # upstream integration output is 64-d (train_and_infer default)
    assert atac.obs["candies_domain"].nunique() == 2
    assert not torch.are_deterministic_algorithms_enabled()


# ---------------------------------------------------------------- dispatcher
def test_spatial_domains_dispatch_errors():
    atac, rna = _synthetic()
    with pytest.raises(ValueError, match="spatial methods"):
        epi.tl.spatial_domains(atac, 2, method="nope")
    with pytest.raises(ValueError, match="rna="):
        epi.tl.spatial_domains(atac, 2, method="cosmos")
    with pytest.raises(ValueError, match="several slices"):
        epi.tl.spatial_domains(atac, 2, method="instinct")


def test_spatial_domains_cosmos_end_to_end():
    pytest.importorskip("mclust_py")
    atac, rna = _synthetic()
    epi.tl.spatial_domains(atac, 2, method="cosmos", rna=rna, n_neighbors=4, z_dim=8, wnn_epoch=5,
                           total_epoch=12, min_stop=2, device="cpu", edge_subset_sz=2000)
    assert atac.obs["domain"].nunique() == 2
    assert atac.uns["spatial_domains"]["domain"]["method"] == "cosmos"


# ---------------------------------------------------------------- MISO / PRAGA
def test_miso_two_layers_with_interactions():
    atac, rna = _synthetic()
    emb, lab = epi.tl.miso([epi.tl.miso_features(rna, "rna", min_cells=1), epi.tl.miso_features(atac, "atac", min_cells=1)],
                           n_clusters=2, n_pcs=16, latent=4, epochs=20, device="cpu")
    assert emb.shape == (atac.n_obs, 4 * 3)  # two layers + one interaction block
    assert len(set(lab)) == 2


def test_praga_runs_on_cpu(monkeypatch):
    pytest.importorskip("skmisc")
    pytest.importorskip("mclust_py")
    from epione.tl import _praga

    monkeypatch.setitem(_praga._SCHEDULES, "Spatial-epigenome-transcriptome", (110, 1e-3, 5e-2, 2, 3))  # small lr: random features diverge at 1e-2
    atac, rna = _synthetic(n_side=9, n_peaks=200, n_genes=300)
    rna.obsm["feat"] = np.random.default_rng(0).normal(size=(rna.n_obs, 10))
    atac.obsm["feat"] = np.random.default_rng(1).normal(size=(atac.n_obs, 10))
    emb = epi.tl.praga(rna, atac, n_clusters=2, init_k=3, knn_k=5, preprocess=False, device="cpu")
    assert emb.shape == (rna.n_obs, 64)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-4)
    assert atac.obs["praga_domain"].nunique() == 2


# ---------------------------------------------------------------- SpaMosaic
def test_spamosaic_two_bridge_batches_and_dispatch():
    pytest.importorskip("skmisc")
    pytest.importorskip("mclust_py")
    from epione.tl._spamosaic import spamosaic, spamosaic_preprocess

    batches = {"rna": [], "atac": []}
    for b in range(2):
        atac, rna = _synthetic(seed=b)
        for x in (atac, rna):
            x.obs_names = [f"{o}_b{b}" for o in x.obs_names]
            x.obs["src"] = f"b{b}"
        batches["atac"].append(atac)
        batches["rna"].append(rna)
    spamosaic_preprocess(batches["rna"], "rna", n_features=40, n_comps=8, batch_key="src")
    spamosaic_preprocess(batches["atac"], "atac", n_features=100, n_comps=8, batch_key="src")
    finals = spamosaic(batches, batch_key="src", intra_knns=4, radius_cutoff=1.5, n_epochs=5, n_clusters=2, device="cpu")
    assert [f.n_obs for f in finals] == [64, 64]
    assert finals[0].obsm["X_spamosaic"].shape == (64, 32)
    assert not torch.are_deterministic_algorithms_enabled()
    atac, rna = _synthetic()
    epi.tl.spatial_domains(atac, 2, method="spamosaic", rna=rna, n_hvg=40, n_peak=100, lsi_comps=8, intra_knns=4,
                           radius_cutoff=1.5, n_epochs=5, device="cpu")
    assert atac.obs["domain"].nunique() == 2
