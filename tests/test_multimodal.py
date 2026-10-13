"""Multimodal tools: combined_dims, wnn, tsne, multivi and read_10x_multiome.

Small synthetic data only; the parity checks against ArchR / Seurat live in
tests/parity/multimodal and need EPIONE_MULTIMODAL_PARITY_DIR.
"""
from __future__ import annotations

import gzip
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp
from anndata import AnnData

import epione as epi
from epione.tl import _multimodal as MM


def _two_modalities(n=300, seed=0):
    """Two groups of cells: modality A separates sub-types in group 0, B in group 1."""
    rng = np.random.default_rng(seed)
    group = np.repeat([0, 1], n // 2)
    sub = rng.integers(0, 3, n)
    centers = rng.normal(0, 4, (3, 10))
    a = rng.normal(0, 1, (n, 10))
    b = rng.normal(0, 1, (n, 10))
    a[group == 0] += centers[sub[group == 0]]
    b[group == 1] += centers[sub[group == 1]]
    obs = pd.DataFrame({"group": group, "sub": sub}, index=[f"c{i}" for i in range(n)])
    ad = AnnData(np.zeros((n, 1)), obs=obs)
    ad.obsm["X_a"], ad.obsm["X_b"] = a, b
    return ad


# ------------------------------------------------------------ combined_dims
def test_combined_dims_formula_and_kept_dims():
    rng = np.random.default_rng(1)
    ad = AnnData(np.zeros((50, 1)))
    ad.obsm["r1"] = rng.normal(size=(50, 6))
    ad.obsm["r2"] = rng.normal(size=(50, 4)) * 3
    out = epi.tl.combined_dims(ad, ["r1", "r2"], dim_weights=[1.0, 0.5], dims_to_use=[3, 1, 2, 3],
                               cor_to_depth={"r1": [0.9, 0, 0, 0, 0, 0]}, cor_cutoff=0.75)
    # r1 keeps dims 3, 2 (1 is depth-correlated; order follows dims_to_use); r2 keeps 3, 1, 2.
    assert ad.uns["X_combined"]["dims_kept"] == {"r1": [3, 2], "r2": [3, 1, 2]}
    b1 = ad.obsm["r1"][:, [2, 1]]
    b2 = ad.obsm["r2"][:, [2, 0, 1]]
    exp = np.hstack([b1 / np.sqrt(b1.var(axis=0, ddof=1).sum()), 0.5 * b2 / np.sqrt(b2.var(axis=0, ddof=1).sum())])
    np.testing.assert_allclose(out, exp, rtol=0, atol=1e-12)
    # each block carries total variance weight**2
    assert np.isclose(out[:, :2].var(axis=0, ddof=1).sum(), 1.0)
    assert np.isclose(out[:, 2:].var(axis=0, ddof=1).sum(), 0.25)


def test_combined_dims_scale_dims_and_depth_key():
    rng = np.random.default_rng(2)
    ad = AnnData(np.zeros((40, 1)), obs=pd.DataFrame({"n_frag": rng.integers(100, 10000, 40)}))
    depth = np.log10(ad.obs["n_frag"].to_numpy() + 1)
    m = rng.normal(size=(40, 5))
    m[:, 0] = depth * 10 + rng.normal(scale=0.01, size=40)
    ad.obsm["r"] = m
    epi.tl.combined_dims(ad, ["r"], scale_dims=False, depth_key="n_frag")
    assert 1 not in ad.uns["X_combined"]["dims_kept"]["r"]
    out = epi.tl.combined_dims(ad, ["r"], scale_dims=True, cor_cutoff=None, key_added="z")
    z = MM._row_zscores(m)
    np.testing.assert_allclose(out, z / np.sqrt(z.var(axis=0, ddof=1).sum()), atol=1e-12)


# --------------------------------------------------------------------- wnn
def test_bandwidth_matches_brute_force():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 4))
    k = 5
    from sklearn.neighbors import NearestNeighbors

    idx = NearestNeighbors(n_neighbors=k).fit(x).kneighbors(x)[1]
    got = MM._bandwidth(x, idx, n_select=4)
    sets = [set(r) for r in idx]
    for i in range(60):
        cand = []
        for j in range(60):
            if j == i:
                continue
            inter = len(sets[i] & sets[j])
            if inter:
                cand.append((inter / (2 * k - inter), -np.linalg.norm(x[i] - x[j]), j))
        cand.sort()
        exp = np.mean([-c[1] for c in cand[:4]])
        assert np.isclose(got[i], exp)


def test_wnn_weights_follow_informative_modality():
    import scanpy as sc

    ad = _two_modalities()
    w, idx = epi.tl.wnn(ad, reps=["X_a", "X_b"], modality_names=["A", "B"], n_neighbors=15, knn_range=60)
    g = ad.obs["group"].to_numpy()
    assert w.shape == (ad.n_obs, 2) and np.allclose(w.sum(axis=1), 1)
    assert w[g == 0, 0].mean() > 0.6 and w[g == 1, 1].mean() > 0.6
    assert (idx[:, 0] == np.arange(ad.n_obs)).all()
    assert "wnn_weight_A" in ad.obs and ad.obsp["wnn_connectivities"].shape == (ad.n_obs, ad.n_obs)
    # neighbours stay within the true sub-type far more often than chance
    same = (ad.obs["sub"].to_numpy()[idx[:, 1:]] == ad.obs["sub"].to_numpy()[:, None]).mean()
    assert same > 0.8
    sc.tl.leiden(ad, neighbors_key="wnn", flavor="igraph", n_iterations=2, directed=False)
    sc.tl.umap(ad, neighbors_key="wnn")
    assert ad.obsm["X_umap"].shape == (ad.n_obs, 2)


def test_wnn_needs_two_reps():
    ad = _two_modalities(60)
    with pytest.raises(ValueError):
        epi.tl.wnn(ad, reps=["X_a"])


# -------------------------------------------------------------------- tsne
def test_tsne_writes_obsm():
    pytest.importorskip("openTSNE")
    ad = _two_modalities(120)
    ad.obsm["X_lsi"] = ad.obsm["X_a"]
    emb = epi.tl.tsne(ad, dims=range(2, 11), perplexity=10, n_iter=100, n_jobs=1)
    assert emb.shape == (120, 2) and np.isfinite(emb).all()
    assert ad.obsm["X_tsne"] is emb or np.array_equal(ad.obsm["X_tsne"], emb)
    emb2 = epi.tl.tsne(ad, dims=range(2, 11), perplexity=10, n_iter=100, n_jobs=1, key_added="X_tsne2")
    np.testing.assert_allclose(emb, emb2)


# ----------------------------------------------------------------- multivi
def test_multivi_small_cpu():
    pytest.importorskip("scvi")
    rng = np.random.default_rng(0)
    n = 80
    rna = AnnData(sp.csr_matrix(rng.poisson(1.0, (n, 30)).astype(np.float32)),
                  obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]))
    atac = AnnData(sp.csr_matrix((rng.random((n, 40)) < 0.2).astype(np.float32)), obs=rna.obs.copy())
    epi.tl.multivi(rna, atac, n_latent=4, max_epochs=2, accelerator="cpu")
    assert rna.obsm["X_multivi"].shape == (n, 4)
    np.testing.assert_array_equal(rna.obsm["X_multivi"], atac.obsm["X_multivi"])


def test_multivi_rejects_unpaired_order():
    pytest.importorskip("scvi")
    rna = AnnData(np.ones((3, 2)), obs=pd.DataFrame(index=["a", "b", "c"]))
    atac = AnnData(np.ones((3, 2)), obs=pd.DataFrame(index=["a", "c", "b"]))
    with pytest.raises(ValueError):
        epi.tl.multivi(rna, atac)


# --------------------------------------------------------------- reader
def _fake_10x(tmp_path, prefix=""):
    import h5py

    rng = np.random.default_rng(0)
    genes = [("ENSG1", "GeneA"), ("ENSG2", "GeneB"), ("ENSG3", "GeneA")]
    peaks = ["chr1:100-600", "chr2:1000-1500"]
    bcs = ["AAAC-1", "AAAG-1", "AAAT-1", "AACC-1"]
    X = sp.csc_matrix(rng.poisson(2, (len(genes) + len(peaks), len(bcs))).astype(np.int32))
    with h5py.File(tmp_path / f"{prefix}filtered_feature_bc_matrix.h5", "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = X.data, X.indices, X.indptr
        g["shape"] = np.array(X.shape, dtype=np.int32)
        g["barcodes"] = np.array(bcs, dtype="S")
        fe = g.create_group("features")
        fe["id"] = np.array([x[0] for x in genes] + peaks, dtype="S")
        fe["name"] = np.array([x[1] for x in genes] + peaks, dtype="S")
        fe["feature_type"] = np.array(["Gene Expression"] * 3 + ["Peaks"] * 2, dtype="S")
        fe["genome"] = np.array(["GRCh38"] * 5, dtype="S")
        fe["interval"] = np.array(["chr1:1-2"] * 3 + ["chr1:100-600", "chr2:1000-1500"], dtype="S")
        fe["_all_tag_keys"] = np.array(["genome", "interval"], dtype="S")
    pd.DataFrame({"barcode": bcs, "is_cell": 1, "atac_fragments": [10, 20, 30, 40]}).to_csv(
        tmp_path / f"{prefix}per_barcode_metrics.csv", index=False)
    (tmp_path / f"{prefix}atac_fragments.tsv.gz").write_bytes(b"")
    return X.T.tocsr(), genes, peaks, bcs


def test_read_10x_multiome_h5_dir(tmp_path):
    X, genes, peaks, bcs = _fake_10x(tmp_path, prefix="pbmc_")
    rna, atac = epi.io.read_10x_multiome(tmp_path)
    assert list(rna.obs_names) == bcs and rna.n_vars == 3 and atac.n_vars == 2
    assert rna.var_names.is_unique
    np.testing.assert_array_equal(rna.X.toarray(), X[:, :3].toarray())
    np.testing.assert_array_equal(atac.X.toarray(), X[:, 3:].toarray())
    assert atac.var["chromStart"].tolist() == [100, 1000] and atac.var["chrom"].tolist() == ["chr1", "chr2"]
    assert rna.obs["atac_fragments"].tolist() == [10, 20, 30, 40]
    assert atac.uns["files"]["fragments"].endswith("pbmc_atac_fragments.tsv.gz")


def test_read_10x_multiome_mtx_folder_and_mudata(tmp_path):
    X, genes, peaks, bcs = _fake_10x(tmp_path)
    folder = tmp_path / "filtered_feature_bc_matrix"
    folder.mkdir()
    with gzip.open(folder / "matrix.mtx.gz", "wb") as f:
        scipy.io.mmwrite(f, sp.coo_matrix(X.T))
    with gzip.open(folder / "barcodes.tsv.gz", "wt") as f:
        f.write("\n".join(bcs) + "\n")
    with gzip.open(folder / "features.tsv.gz", "wt") as f:
        for gid, name in genes:
            f.write(f"{gid}\t{name}\tGene Expression\tGRCh38\tchr1\t1\t2\n")
        for p in peaks:
            c, se = p.split(":")
            s, e = se.split("-")
            f.write(f"{p}\t{p}\tPeaks\tGRCh38\t{c}\t{s}\t{e}\n")
    rna, atac = epi.io.read_10x_multiome(folder)
    np.testing.assert_array_equal(atac.X.toarray(), X[:, 3:].toarray())
    assert atac.uns["files"]["fragments"].endswith("atac_fragments.tsv.gz")
    md = epi.io.read_10x_multiome(tmp_path / "filtered_feature_bc_matrix.h5", as_mudata=True)
    assert set(md.mod) == {"rna", "atac"} and md.n_obs == 4


def test_read_10x_multiome_backed(tmp_path):
    pytest.importorskip("mudataoom")
    _fake_10x(tmp_path)
    md = epi.io.read_10x_multiome(tmp_path, backed=tmp_path / "pbmc.h5mu")
    assert (tmp_path / "pbmc.h5mu").exists()
    assert md.n_obs == 4 and set(md.mod) == {"rna", "atac"}


def test_epi_io_works_before_core_is_imported():
    code = "import epione as epi; epi.io.read_gtf; epi.io.read_10x_multiome"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def test_translate_benchmark_plot_accepts_custom_metrics():
    import matplotlib

    matplotlib.use("Agg")
    from epione.translate import plot_benchmark, summarize

    spec = {"ARI": ("clustering", True), "silhouette": ("structure", True), "kNN_purity": ("structure", True)}
    rows = []
    for meth, vals in {"A": (0.8, 0.3, 0.9), "B": (0.6, 0.1, 0.8)}.items():
        for (m, (c, hib)), v in zip(spec.items(), vals):
            rows.append({"task": "joint", "method": meth, "metric": m, "category": c, "higher_is_better": hib, "value": v})
    df = pd.DataFrame(rows)
    s = summarize(df, metrics=spec)
    assert s.loc[("joint", "A"), ("score", "overall")] == 1.0
    assert ("score", "structure") in s.columns
    fig = plot_benchmark(df, metrics=spec)
    assert fig is not None
