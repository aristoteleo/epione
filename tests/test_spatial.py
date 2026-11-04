"""Spatial ATAC: readers, spot graph, smoothing, domains, SpatialGlue port."""
import gzip

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import epione as epi


def _grid(n=10):
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return r.ravel(), c.ravel()


def _barcodes(n):
    rng = np.random.default_rng(0)
    seen = set()
    while len(seen) < n:
        seen.add("".join(rng.choice(list("ACGT"), 16)))
    return sorted(seen)


def test_read_positions_spaceranger_and_whitelist(tmp_path):
    r, c = _grid(4)
    bc = _barcodes(16)
    p = tmp_path / "tissue_positions_list.csv.gz"
    with gzip.open(p, "wt") as fh:
        for i, b in enumerate(bc):
            fh.write(f"{b},{int(i % 3 != 0)},{r[i]},{c[i]},{r[i] * 10.0},{c[i] * 10.0}\n")
        fh.write("\n")
    pos = epi.io.read_spatial_positions(p)
    assert pos.shape == (16, 5) and pos["in_tissue"].dtype == bool
    assert pos.loc[bc[5], "pxl_col_in_fullres"] == c[5] * 10.0

    w = tmp_path / "barcodes.txt"
    w.write_text("barcode\tx\ty\n" + "".join(f"{b}-1\t{c[i]}\t{r[i]}\n" for i, b in enumerate(bc)))
    pos2 = epi.io.read_spatial_positions(w)
    assert list(pos2.index) == bc  # GEM-well suffix removed
    assert (pos2["array_col"].to_numpy() == c).all() and pos2["in_tissue"].all()


def test_add_spatial_matches_suffix_and_masks(tmp_path):
    r, c = _grid(4)
    bc = _barcodes(16)
    pos = pd.DataFrame({"in_tissue": [i % 2 == 0 for i in range(16)], "array_row": r, "array_col": c,
                        "pxl_row_in_fullres": r * 5.0, "pxl_col_in_fullres": c * 5.0}, index=bc)
    a = ad.AnnData(np.ones((16, 2), dtype=np.float32))
    a.obs_names = [f"{b}-1" for b in bc]
    out = epi.io.add_spatial(a, pos, tissue_only=True, library_id="s1")
    assert out.n_obs == 8 and out.obs_names[0].endswith("-1")
    np.testing.assert_allclose(out.obsm["spatial"][:, 0], c[::2] * 5.0)
    assert "s1" in out.uns["spatial"]


def test_read_spatial_atac_rna_tables(tmp_path):
    r, c = _grid(3)
    bc = _barcodes(9)
    pos = tmp_path / "pos.csv"
    pos.write_text("".join(f"{b},1,{r[i]},{c[i]},{r[i]},{c[i]}\n" for i, b in enumerate(bc)))
    rna = tmp_path / "rna.tsv"
    rna.write_text("\t" + "\t".join(bc[::-1]) + "\nGeneA\t" + "\t".join("1" * 9) + "\nGeneB\t"
                   + "\t".join("0" * 9) + "\n")
    atac = ad.AnnData(sp.csr_matrix(np.eye(9, 4, dtype=np.float32)))
    atac.obs_names = [f"{b}-1" for b in bc]
    m = epi.io.read_spatial_atac_rna(None, rna, pos, atac=atac)
    assert m.n_obs == 9 and m["rna"].n_vars == 2
    assert list(m["rna"].obs_names) == list(m["atac"].obs_names)
    assert m["atac"].obs_names[0].endswith("-1")
    assert m.obsm["spatial"].shape == (9, 2)


def test_spatial_neighbors_grid_and_smooth():
    r, c = _grid(5)
    a = ad.AnnData(np.zeros((25, 1), dtype=np.float32))
    a.obs["array_row"], a.obs["array_col"] = r, c
    a.obsm["spatial"] = np.c_[c, r].astype(float)
    epi.tl.spatial_neighbors(a, coord_type="grid", n_neighs=4)
    deg = np.asarray(a.obsp["spatial_connectivities"].sum(1)).ravel()
    assert deg.max() == 4 and deg.min() == 2  # interior / corner
    epi.tl.spatial_neighbors(a, coord_type="grid", n_neighs=8, key_added="g8")
    assert np.asarray(a.obsp["g8_connectivities"].sum(1)).max() == 8
    a.obsm["X"] = np.arange(25, dtype=float)[:, None]
    s = epi.tl.spatial_smooth(a, use_rep="X", lam=1.0)
    assert np.isclose(s[12, 0], np.mean([7, 11, 13, 17]))
    assert np.allclose(epi.tl.spatial_smooth(a, use_rep="X", lam=0.0), a.obsm["X"])


def test_moran_and_domains():
    r, c = _grid(10)
    a = ad.AnnData(np.zeros((100, 1), dtype=np.float32))
    a.obs["array_row"], a.obs["array_col"] = r, c
    a.obsm["spatial"] = np.c_[c, r].astype(float)
    epi.tl.spatial_neighbors(a, coord_type="grid", n_neighs=4)
    rng = np.random.default_rng(0)
    left = (c < 5).astype(float)
    a.obsm["f"] = np.c_[left, rng.normal(size=100)]
    I = epi.tl.spatial_moran(a, use_rep="f").to_numpy()
    assert I[0] > 0.8 and abs(I[1]) < 0.3
    a.obsm["emb"] = np.c_[left * 5 + rng.normal(scale=0.1, size=100), rng.normal(scale=0.1, size=100)]
    lab = epi.tl.spatial_domains(a, 2, use_rep="emb", method="kmeans", use_pca=False)
    from sklearn.metrics import adjusted_rand_score

    assert adjusted_rand_score(left, lab) == 1.0


def test_spatial_plot_runs():
    import matplotlib

    matplotlib.use("Agg")
    r, c = _grid(6)
    a = ad.AnnData(np.random.default_rng(0).poisson(1, (36, 3)).astype(np.float32))
    a.var_names = ["g1", "g2", "g3"]
    a.obs["array_row"], a.obs["array_col"] = r, c
    a.obs["n_fragment"] = np.arange(36) + 1
    a.obs["dom"] = pd.Categorical(np.where(c < 3, "a", "b"))
    a.obsm["spatial"] = np.c_[c, r].astype(float) * 50
    axes = epi.pl.spatial(a, ["dom", "g1"], show=False)
    assert len(axes) == 2
    axes = epi.pl.spatial_qc(a, show=False)
    assert len(axes) == 1 and "log10(1+n_fragment)" not in a.obs


def test_spatialglue_small_deterministic():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    r, c = _grid(8)
    n = 64
    base = (c >= 4).astype(float)
    a1 = ad.AnnData(np.zeros((n, 1), dtype=np.float32))
    a2 = ad.AnnData(np.zeros((n, 1), dtype=np.float32))
    for a, d in ((a1, 10), (a2, 12)):
        a.obsm["feat"] = (base[:, None] * rng.normal(size=d) + rng.normal(scale=0.3, size=(n, d))).astype(np.float32)
        a.obsm["spatial"] = np.c_[c, r].astype(float)
    o1 = epi.tl.spatialglue((a1, a2), epochs=20, device="cpu", verbose=False, random_seed=1)
    o2 = epi.tl.spatialglue((a1, a2), epochs=20, device="cpu", verbose=False, random_seed=1)
    assert o1["SpatialGlue"].shape == (n, 64)
    np.testing.assert_allclose(np.linalg.norm(o1["SpatialGlue"], axis=1), 1, atol=1e-5)
    np.testing.assert_array_equal(o1["SpatialGlue"], o2["SpatialGlue"])
    assert o1["alpha"].shape == (n, 2)
    np.testing.assert_allclose(o1["alpha"].sum(1), 1, atol=1e-5)
