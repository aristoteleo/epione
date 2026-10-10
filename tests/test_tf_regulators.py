"""Tests for epione.tl.var_deviations / correlate_matrices /
positive_tf_regulators (ArchR getVarDeviations / correlateMatrices ports)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse, stats

import epione as epi
from epione.tl._r_rng import RRandom
from epione.tl._tf_regulators import _combine_variances, _cor_test, _knn_groups, _match_names


# ---- R RNG emulation (values printed by R 4.4.3) ---------------------------
def test_r_rng_matches_r():
    assert RRandom(1).sample_int(10660, 8).tolist() == [1017, 8004, 4775, 10369, 9725, 10539, 8462, 4050]
    assert RRandom(42).sample_int(37, 6).tolist() == [37, 1, 25, 10, 18, 26]
    r = RRandom(1)
    assert [round(r.unif_rand(), 15) for _ in range(3)] == [0.265508663142100, 0.372123899636790, 0.572853363351896]
    assert RRandom(7).sample_int(5000, 600, replace=True)[594:].tolist() == [1042, 4434, 337, 1689, 1345, 2908]


def test_r_rng_lecuyer_matches_r():
    k = "L'Ecuyer-CMRG"
    assert RRandom(1, k).sample_int(10660, 8).tolist() == [10511, 5942, 6139, 6985, 10001, 4668, 8753, 6102]
    r = RRandom(1, k)
    assert [round(r.unif_rand(), 15) for _ in range(3)] == [0.677532828628744, 0.427345722887644, 0.910380530487548]
    assert RRandom(123, k).sample_int(10**6, 5).tolist() == [484301, 761774, 495014, 339328, 213374]
    assert RRandom(9, k).sample_int(50, 60, replace=True)[54:].tolist() == [25, 25, 37, 46, 13, 48]


# ---- var_deviations --------------------------------------------------------
def _toy_motif_adata(n=300, m=20, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, m)) * np.linspace(0.2, 3, m)
    a = ad.AnnData(X=np.zeros((n, 1)))
    a.obs["Sample"] = np.repeat(["s1", "s2", "s3"], n // 3)
    a.obsm["motif_deviations"] = z
    a.uns["motif_deviations_names"] = np.array([f"TF{i}_{i}" for i in range(m)])
    return a, z


def test_var_deviations_equals_pooled_variance():
    a, z = _toy_motif_adata()
    df = epi.tl.var_deviations(a, verbose=False)
    ref = pd.Series(z.var(axis=0, ddof=1), index=a.uns["motif_deviations_names"])
    np.testing.assert_allclose(df.set_index("name").combinedVars, ref.loc[df.name], rtol=1e-12)
    assert df["rank"].tolist() == list(range(1, 21))
    assert df["name"].iloc[0] == "TF19_19"
    assert "motif_deviations_var" in a.uns
    no_sample = epi.tl.var_deviations(a, sample_key=None, verbose=False)
    np.testing.assert_allclose(no_sample.combinedVars, df.combinedVars, rtol=1e-12)


def test_combine_variances_single_cell_sample_is_zeroed():
    means = np.array([[1.0, 3.0]]); var = np.array([[2.0, np.nan]]); ns = np.array([5.0, 1.0])
    cv, cm = _combine_variances(means, var, ns)
    assert np.isfinite(cv).all() and np.isclose(cm[0], (5 + 3) / 6)


# ---- correlate_matrices pieces ---------------------------------------------
def test_match_names_rules():
    out = _match_names(["Gata1_383", "nfe2-l2", "Sox9.2", "HOX10"], ["underscore", "dash", "dot", "numeric"])
    assert out.tolist() == ["GATA", "NFE", "SOX", "HOX"]
    assert _match_names(["Gata1_383"], ["underscore"]).tolist() == ["GATA1"]


def test_cor_test_matches_scipy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=60); y = x + rng.normal(size=60)
    r, p = _cor_test(x, y)
    rs, ps = stats.pearsonr(x, y)
    assert np.isclose(r, rs) and np.isclose(p, ps, rtol=1e-10)
    assert np.isnan(_cor_test(np.ones(60), y)[0])
    assert np.isnan(_cor_test(x[:10], y[:10])[0])  # <= min observations


def test_knn_groups_respect_overlap_cutoff():
    rng = np.random.default_rng(0)
    rd = rng.normal(size=(500, 5))
    g = _knn_groups(rd, np.arange(0, 500, 5), k=20, overlap_cutoff=0.5)
    assert g.shape[1] == 20 and g.shape[0] >= 1
    sets = [set(x) for x in g]
    for i in range(len(sets)):
        for j in range(i):
            assert len(sets[i] & sets[j]) <= 10
    assert 0 in sets[0]  # self is the first neighbour of the first seed


def _toy_pair(n=600, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 4))
    genes = np.exp(latent @ rng.normal(size=(4, 6)) * 0.5)
    tfz = np.column_stack([np.log1p(genes[:, 0]), -np.log1p(genes[:, 1]),
                           rng.normal(size=n), np.log1p(genes[:, 3])]) + rng.normal(scale=0.1, size=(n, 4))
    a = ad.AnnData(X=sparse.csr_matrix(genes))
    a.var_names = ["GATA1", "SPI1", "CEBPA", "EBF1", "PAX5", "TCF7"]
    a.obs["Clusters"] = np.where(latent[:, 0] > 0, "C1", "C2")
    a.obsm["X_lsi"] = latent
    a.obsm["motif_deviations"] = tfz
    a.uns["motif_deviations_names"] = np.array(["GATA1_383", "SPI1_322", "CEBPA_155", "EBF1_67"])
    return a


def test_correlate_matrices_recovers_signal():
    a = _toy_pair()
    df = epi.tl.correlate_matrices(a, key2="motif_deviations", use_rep="X_lsi", k=20,
                                   knn_iteration=200, cor_cutoff=None, verbose=False)
    assert len(df) == 4
    s = df.set_index("matrix1_name")
    assert s.loc["GATA1", "cor"] > 0.8 and s.loc["SPI1", "cor"] < -0.8
    assert s.loc["GATA1", "padj"] < 1e-5
    assert np.all((df.padj >= df.pval) | df.pval.isna())
    # deterministic + R RNG reproducible
    df2 = epi.tl.correlate_matrices(a, key2="motif_deviations", use_rep="X_lsi", k=20,
                                    knn_iteration=200, cor_cutoff=None, verbose=False)
    pd.testing.assert_frame_equal(df, df2)
    np.testing.assert_array_equal(df.attrs["seed_cells"], RRandom(1, "L'Ecuyer-CMRG").sample_int(600, 200))
    dfr = epi.tl.correlate_matrices(a, key2="motif_deviations", use_rep="X_lsi", k=20, rng="R",
                                    knn_iteration=200, cor_cutoff=None, verbose=False)
    np.testing.assert_array_equal(dfr.attrs["seed_cells"], RRandom(1).sample_int(600, 200))
    with pytest.raises(ValueError):
        epi.tl.correlate_matrices(a, key2="motif_deviations", use_rep="X_lsi", rng="bogus", verbose=False)


def test_positive_tf_regulators_toy():
    a = _toy_pair()
    tf = epi.tl.positive_tf_regulators(a, use_rep="X_lsi", k=20, knn_iteration=200,
                                       cor_cutoff=None, verbose=False, delta_quantile=0.0)
    assert set(tf.columns) >= {"cor", "padj", "maxDelta", "TFRegulator"}
    assert "GATA1" in set(tf.loc[tf.TFRegulator == "YES", "GeneScoreMatrix_name"])
    assert "SPI1" not in set(tf.loc[tf.TFRegulator == "YES", "GeneScoreMatrix_name"])
    gm = epi.tl.group_means(a, "Clusters", key="motif_deviations")
    np.testing.assert_allclose(tf.set_index("MotifMatrix_name").maxDelta.loc[gm.index],
                               gm.max(axis=1) - gm.min(axis=1))
    assert "positive_tf_regulators" in a.uns


def test_plots_smoke():
    import matplotlib
    matplotlib.use("Agg")
    a, _ = _toy_motif_adata()
    epi.tl.var_deviations(a, verbose=False)
    fig, ax = epi.pl.var_dev(a, n=5, show=False)
    assert ax.get_ylabel() == "Variability"
    b = _toy_pair()
    epi.tl.positive_tf_regulators(b, use_rep="X_lsi", k=20, knn_iteration=200, cor_cutoff=None, verbose=False)
    fig, ax = epi.pl.tf_regulators(b, show=False)
    assert ax.get_xlabel() == "Correlation To Gene Score"


# ---- parity against ArchR 1.0.3 (gates in tests/parity/var_dev_correlate) ---
REF = Path(os.environ.get("EPIONE_ARCHR_PARITY_DIR", "")) / "var_dev_correlate"


@pytest.mark.skipif(not os.environ.get("EPIONE_ARCHR_PARITY_DIR")
                    or not (REF / "correlateMatrices_GSM_cisbp.tsv").exists(),
                    reason="EPIONE_ARCHR_PARITY_DIR with ArchR reference outputs not available")
def test_parity_with_archr():
    # Load by path under a unique name: several parity folders ship a
    # ``py_parity.py`` and a plain import would reuse whichever loaded first.
    spec = importlib.util.spec_from_file_location(
        "py_parity_var_dev_correlate", Path(__file__).parent / "parity" / "var_dev_correlate" / "py_parity.py")
    _pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_pp)
    run = _pp.run

    m = run(REF)
    for tag in ("MotifMatrix", "cisbpMatrix"):
        assert m[f"vardev_{tag}_combinedVars_maxabs"] <= 1e-8
        assert m[f"vardev_{tag}_combinedMeans_maxabs"] <= 1e-8
        assert m[f"vardev_{tag}_rank_mismatches"] == 0
    assert m["seed_cells_archr_rng_mismatches"] == 0
    assert m["archr_rng_cor_maxabs"] <= 1e-8
    assert m["knn_group_set_mismatches"] == 0
    assert m["cor_nan_mismatch"] == 0 and m["cor_pairs_merged"] == m["cor_pairs_r"]
    assert m["cor_maxabs"] <= 1e-8
    assert m["nlog10_pval_maxabs"] <= 1e-6 and m["nlog10_padj_maxabs"] <= 1e-6
    assert m["groupSE_maxabs"] <= 1e-8 and m["maxDelta_maxabs"] <= 1e-8
    assert m["tf_yes_identical_archr_rng"] and m["tf_table_order_identical_archr_rng"]
    assert m["free_numpy_spearman_cor"] >= 0.90 and m["free_numpy_spearman_nlog10_padj"] >= 0.90
    assert m["free_numpy_tf_jaccard"] >= 0.80
