"""Tests for epione.tl.project_bulk_atac / deconvolve_bulk / LSIModel.

Synthetic data only, no downloads. Parity with ArchR's projectBulkATAC is in
``tests/parity/bulk_projection`` (``EPIONE_BULKPROJ_PARITY_DIR``).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import epione as epi
from epione.tl._bulk_projection import (
    LSIModel, bulk_to_features, _RSampler, _simulate_cells_archr, _row_zscores,
)
from epione.tl._r_rng import RRandom


def _tiles(n_feat, chrom="chr1", width=500, closed=True):
    start = np.arange(n_feat) * width
    end = start + (width - 1 if closed else width)
    names = [f"{chrom}:{s}-{e}" for s, e in zip(start, end)]
    return names, start, end


def _synthetic(n_cells=600, n_feat=3000, n_groups=3, seed=0, mean_open=400):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_groups), n_cells // n_groups)
    sets = np.array_split(np.arange(n_feat), n_groups)
    rows, cols = [], []
    for i, g in enumerate(labels):
        k = rng.integers(int(mean_open * 0.6), int(mean_open * 1.4))
        idx = np.unique(np.concatenate([rng.choice(n_feat, k // 2, replace=False),
                                        rng.choice(sets[g], k // 2, replace=False)]))
        rows += [i] * len(idx)
        cols += idx.tolist()
    X = sp.csr_matrix((np.ones(len(rows), np.int32), (rows, cols)), shape=(n_cells, n_feat))
    names, _, _ = _tiles(n_feat)
    ad = AnnData(X=X, obs=pd.DataFrame({"group": pd.Categorical([f"g{g}" for g in labels])},
                                       index=[f"c{i}" for i in range(n_cells)]))
    ad.var_names = names
    ad.obs["n_fragment"] = np.asarray(X.sum(axis=1)).ravel()
    return ad, sets


@pytest.fixture(scope="module")
def fitted():
    ad, sets = _synthetic()
    epi.tl.iterative_lsi(ad, n_components=15, iterations=2, var_features=2000,
                         total_features=3000, resolution=0.3, n_neighbors=15,
                         sample_cells_pre=None, depth_col="n_fragment", seed=0,
                         verbose=False)
    return ad, sets


def test_iterative_lsi_stores_model_fields(fitted):
    ad, _ = fitted
    info = ad.uns["X_iterative_lsi"]
    for k in ("feature_sums", "n_cells_fit", "scale_mean", "scale_sd"):
        assert k in info
    assert info["n_cells_fit"] == ad.n_obs
    assert len(info["feature_sums"]) == int(np.asarray(info["features_final"]).sum())


def test_model_reprojects_training_cells(fitted):
    ad, _ = fitted
    m = LSIModel.from_adata(ad)
    assert m.closed  # chr1:0-499 style tiles
    feat = np.asarray(ad.uns["X_iterative_lsi"]["features_final"]).astype(bool)
    proj = m.transform(ad.X[:, np.where(feat)[0]])
    emb = ad.obsm["X_iterative_lsi"]
    for j in range(emb.shape[1]):
        assert np.corrcoef(proj[:, j], emb[:, j])[0, 1] > 0.995


def test_model_fallback_matches_stored(fitted):
    ad, _ = fitted
    old = ad.copy()
    for k in ("feature_sums", "n_cells_fit", "scale_mean", "scale_sd"):
        del old.uns["X_iterative_lsi"][k]
    with pytest.warns(UserWarning, match="predates"):
        m_old = LSIModel.from_adata(old)
    m_new = LSIModel.from_adata(ad)
    np.testing.assert_allclose(m_old.feature_sums, m_new.feature_sums)
    assert m_old.n_cells == m_new.n_cells


def test_bulk_to_features_archr_rules():
    feats = pd.DataFrame({"chrom": ["chr1"] * 4, "start": [0, 500, 1000, 1500],
                          "end": [499, 999, 1499, 1999]})
    regs = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1", "chr2"],
                         "start": [400, 450, 1100, 0], "end": [600, 520, 1200, 100]})
    # peak0 (total 3) and peak1 (total 9) both overlap features 0 and 1;
    # the higher-count peak1 wins both. Feature 3 has no overlap -> zeros.
    C = np.array([[1, 2], [4, 5], [7, 0], [9, 9]])
    M, frac = bulk_to_features(C, regs, feats, closed=True, return_overlap=True)
    M = M.toarray()
    np.testing.assert_array_equal(M[0], [4, 5])
    np.testing.assert_array_equal(M[1], [4, 5])
    np.testing.assert_array_equal(M[2], [7, 0])
    np.testing.assert_array_equal(M[3], [0, 0])
    assert frac == 0.75


def test_bulk_to_features_half_open_boundary():
    feats = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 500], "end": [500, 1000]})
    regs = pd.DataFrame({"chrom": ["chr1"], "start": [500], "end": [600]})
    M = bulk_to_features(np.array([[3]]), regs, feats, closed=False).toarray()
    np.testing.assert_array_equal(M[:, 0], [0, 3])  # [500,600) does not touch [0,500)


def test_r_sampler_matches_r_rng():
    for seed, n in [(1, 10007), (99, 3), (5, 2 ** 18)]:
        a = _RSampler(seed).sample_replace(n, 3000)
        b = RRandom(seed).sample_int(n, 3000, replace=True)
        np.testing.assert_array_equal(a, b)


def test_simulate_cells_archr_layout():
    counts = np.array([0, 3, 1, 0, 2])
    depth, n2 = 4, 2
    sim = _simulate_cells_archr(counts, depth, n2, _RSampler(7))
    # replay the same draws by hand: R's matrix(..., ncol = n2) is column-major
    pool = np.repeat(np.arange(5), counts)
    r = RRandom(7)
    exp = np.zeros((n2 * 5, 5))
    for y, ratio in enumerate([2, 1.5, 1, 0.5, 0.25]):
        nrow = int(np.ceil(ratio * depth))
        draw = pool[r.sample_int(pool.size, nrow * n2, replace=True) - 1]
        for k, f in enumerate(draw):
            exp[y * n2 + k // nrow, f] += 1
    np.testing.assert_array_equal(sim.toarray(), exp)


def test_archr_flavour_formulas():
    rng = np.random.default_rng(0)
    nf, k = 50, 4
    feats = pd.DataFrame({"chrom": ["chr1"] * nf, "start": np.arange(nf) * 500,
                          "end": np.arange(nf) * 500 + 499})
    U = np.linalg.qr(rng.normal(size=(nf, k)))[0]
    d = np.array([5.0, 3.0, 2.0, 1.0])
    rs = rng.integers(1, 30, nf).astype(float)
    X = sp.csr_matrix((rng.random((6, nf)) < 0.3).astype(float))
    X[0, 0] = 1
    for method in (1, 2, 3):
        m = LSIModel(features=feats, loadings=U, singular_values=d, feature_sums=rs,
                     n_cells=100, flavor="archr", lsi_method=method, closed=True)
        x = X.toarray()
        tf = x / x.sum(axis=1, keepdims=True)
        if method == 1:
            xn = tf * np.log(1 + 100 / rs)
        elif method == 2:
            xn = np.log(tf * (100 / rs) * 10000 + 1)
        else:
            xn = np.log(tf + 1) * np.log(1 + 100 / rs)
        np.testing.assert_allclose(m._raw_coords(X), xn @ U, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(m.transform(X), _row_zscores(xn @ U), atol=1e-12)


def _pseudobulk(ad, groups, regions=None):
    cols = {}
    for g in groups:
        cols[g] = np.asarray(ad.X[(ad.obs["group"] == g).to_numpy()].sum(axis=0)).ravel()
    df = pd.DataFrame(cols, index=ad.var_names)
    return df


def test_project_bulk_atac_groups(fitted):
    ad, _ = fitted
    bulk = _pseudobulk(ad, ["g0", "g1", "g2"])
    out = epi.tl.project_bulk_atac(ad, bulk, n_cells=100, groupby="group", embedding=None,
                                   verbose=False)
    assert out.n_obs == 300
    assert out.obsm["X_lsi"].shape[1] == ad.obsm["X_iterative_lsi"].shape[1]
    fr = out.uns["group_fractions"]
    for g in ["g0", "g1", "g2"]:
        assert fr.loc[g].idxmax() == g
        assert fr.loc[g, g] > 0.9


def test_project_bulk_atac_umap(fitted):
    import scanpy as sc
    ad, _ = fitted
    tmp = ad.copy()
    sc.pp.neighbors(tmp, use_rep="X_iterative_lsi", n_neighbors=15, metric="cosine")
    sc.tl.umap(tmp, min_dist=0.5, random_state=0)
    bulk = _pseudobulk(tmp, ["g1"])
    out = epi.tl.project_bulk_atac(tmp, bulk, n_cells=50, embedding="X_umap", verbose=False)
    assert out.obsm["X_umap"].shape == (50, 2)
    assert out.uns["reference_umap"].shape == (tmp.n_obs, 2)
    assert min(out.uns["project_bulk_atac"]["reprojection_correlation"]) > 0.8


def test_project_bulk_atac_low_overlap_needs_force(fitted):
    ad, _ = fitted
    bulk = pd.DataFrame({"s": [5, 5]}, index=["chr1:0-499", "chr1:500-999"])
    with pytest.raises(ValueError, match="force=True"):
        epi.tl.project_bulk_atac(ad, bulk, embedding=None, verbose=False)


def test_deconvolve_bulk_recovers_mixture():
    ad, _ = _synthetic(n_cells=900, seed=1)
    rng = np.random.default_rng(0)
    groups = ad.obs["group"].astype(str).to_numpy()
    ref_mask = np.zeros(ad.n_obs, bool)
    for g in np.unique(groups):
        ii = np.where(groups == g)[0]
        ref_mask[rng.permutation(ii)[: len(ii) // 2]] = True
    ref = ad[ref_mask].copy()
    truth, cols = [], []
    for _ in range(8):
        p = rng.dirichlet(np.ones(3))
        n = rng.multinomial(300, p)
        rows = np.concatenate([rng.choice(np.where((~ref_mask) & (groups == f"g{k}"))[0], n[k])
                               for k in range(3) if n[k] > 0])
        cols.append(np.asarray(ad.X[rows].sum(axis=0)).ravel())
        truth.append(n / n.sum())
    bulk = pd.DataFrame(np.vstack(cols).T, index=ad.var_names)
    est = epi.tl.deconvolve_bulk(ref, bulk, "group", n_features=1500)
    E = est[["g0", "g1", "g2"]].to_numpy()
    T = np.vstack(truth)
    np.testing.assert_allclose(E.sum(axis=1), 1.0)
    assert np.abs(E - T).mean() < 0.05


@pytest.mark.skipif(not os.environ.get("EPIONE_BULKPROJ_PARITY_DIR"),
                    reason="set EPIONE_BULKPROJ_PARITY_DIR to run the ArchR parity check")
def test_parity_with_archr():
    p = os.environ["EPIONE_BULKPROJ_PARITY_DIR"]
    script = Path(__file__).parent / "parity" / "bulk_projection" / "parity_check.py"
    r = subprocess.run([sys.executable, str(script), p], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ALL PASS" in r.stdout, r.stdout[-2000:]
