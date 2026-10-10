"""Tests for epione.tl.add_module_score / confusion_matrix and epione.pl.plot_groups
(ports of ArchR addModuleScore / confusionMatrix / plotGroups)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import anndata as ad

import epione as epi
from epione.core._r_random import RRandom
from epione.pl._groups import _bw_nrd0, _mixedsort, _r_density, _r_quantile7, group_plot_data


# --------------------------------------------------------------------------- R RNG

def test_r_random_matches_r():
    # set.seed(1); runif(5)  /  set.seed(1); sample(10)  /  set.seed(42); sample(1000, 7)
    np.testing.assert_array_equal(
        RRandom(1).runif(5),
        [0.26550866314209998, 0.37212389963679016, 0.57285336335189641, 0.90820778999477625, 0.2016819310374558],
    )
    np.testing.assert_array_equal(RRandom(1).sample_int(10) + 1, [9, 4, 7, 1, 2, 5, 3, 10, 6, 8])
    np.testing.assert_array_equal(RRandom(42).sample_int(1000, 7) + 1, [561, 997, 321, 153, 74, 228, 146])
    r = RRandom(7)  # set.seed(7); sample(1e5, 5); sample(3, 2); sample(70000, 3, replace = TRUE)
    np.testing.assert_array_equal(r.sample_int(100000, 5) + 1, [91603, 4572, 51906, 63706, 95624])
    np.testing.assert_array_equal(r.sample_int(3, 2) + 1, [3, 2])
    np.testing.assert_array_equal(r.sample_int(70000, 3, replace=True) + 1, [36744, 64602, 41094])
    np.testing.assert_array_equal(RRandom(-3).sample_int(5) + 1, [2, 4, 5, 3, 1])


# --------------------------------------------------------------------------- R numerics

def test_density_bw_quantile_mixedsort_match_r():
    x = np.array([1, 2, 2.5, 3, 5, 8, 8.2])
    _, y = _r_density(x, 0.7, 0.0, 9.0, 512)
    np.testing.assert_allclose(
        y[[0, 99, 255, 399, 511]],
        [0.030867858969757783, 0.1841236518354662, 0.072508268220343799, 0.052256579274800957, 0.071719888418847022],
        rtol=0, atol=1e-12,
    )
    assert _bw_nrd0(x) == pytest.approx(1.7688648770957192, abs=1e-14)
    np.testing.assert_allclose(_r_quantile7(x, [0.002, 0.25, 0.5, 0.998]), [1.012, 2.25, 3, 8.1976], atol=1e-14)
    assert _mixedsort(["C10", "C2", "C1", "B", "C11", "A3", "A10"]) == ["A3", "A10", "B", "C1", "C2", "C10", "C11"]


# --------------------------------------------------------------------------- confusion matrix

def test_confusion_matrix_order_and_counts():
    i = ["b", "a", "b", "c", "a", "b"]
    j = ["x", "y", "y", "x", "y", "z"]
    cm = epi.tl.confusion_matrix(i, j)
    assert list(cm.index) == ["b", "a", "c"]
    assert list(cm.columns) == ["x", "y", "z"]
    np.testing.assert_array_equal(cm.values, [[1, 1, 1], [0, 2, 0], [1, 0, 0]])
    assert cm.values.sum() == len(i)


# --------------------------------------------------------------------------- module score

def _toy(n_cells=60, n_genes=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(rng.gamma(1.0, 1.0, size=n_genes), size=(n_cells, n_genes)).astype(float)
    X[:, :5] = 0  # zero-signal features -> bin 0
    obs = pd.DataFrame({"Sample": np.repeat(["s1", "s2"], n_cells // 2)}, index=[f"c{k}" for k in range(n_cells)])
    var = pd.DataFrame(index=[f"G{k}" for k in range(n_genes)])
    return ad.AnnData(sparse.csr_matrix(X), obs=obs, var=var)


def test_module_score_injected_background_formula():
    a = _toy()
    X = a.X.toarray()
    bgd = {"Module.A": ["G10", "G11", "G12", "G13"]}
    s = epi.tl.add_module_score(a, {"A": ["G20", "g21"]}, n_bin=5, n_bgd=2, background=bgd, verbose=False)
    exp = X[:, [20, 21]].mean(1) - X[:, [10, 11, 12, 13]].mean(1)
    np.testing.assert_allclose(s["Module.A"].values, exp, atol=1e-12)
    assert "Module.A" in a.obs and a.uns["module_score"]["background"]["Module.A"] == bgd["Module.A"]


def test_module_score_duplicate_rows_are_zeroed():
    # ArchR's .getMatFromArrow only fills the first occurrence of a repeated feature.
    a = _toy()
    X = a.X.toarray()
    s = epi.tl.add_module_score(a, {"A": ["G20", "G21"]}, n_bin=5, n_bgd=2,
                                background={"Module.A": ["G20", "G30", "G30", "G31"]}, inplace=False, verbose=False)
    exp = X[:, [20, 21]].mean(1) - (X[:, 30] + X[:, 31]) / 4
    np.testing.assert_allclose(s["Module.A"].values, exp, atol=1e-12)


def test_module_score_seeded_draws_reproducible_and_binning():
    a = _toy()
    feats = {"A": ["G20", "G21"], "B": ["G40"]}
    s1 = epi.tl.add_module_score(a, feats, n_bin=5, n_bgd=3, seed=3, inplace=False, verbose=False)
    s2 = epi.tl.add_module_score(a, feats, n_bin=5, n_bgd=3, seed=3, inplace=False, verbose=False)
    pd.testing.assert_frame_equal(s1, s2)
    assert list(s1.columns) == ["Module.A", "Module.B"]
    unnamed = epi.tl.add_module_score(a, [["G20", "G21"]], n_bin=5, n_bgd=3, inplace=False, verbose=False)
    assert list(unnamed.columns) == ["Module1"]
    with pytest.raises(ValueError, match="nBgd must be lower than"):
        epi.tl.add_module_score(a, feats, n_bin=5, n_bgd=500, inplace=False, verbose=False)
    with pytest.raises(KeyError):
        epi.tl.add_module_score(a, {"A": ["NOPE"]}, n_bin=5, n_bgd=2, inplace=False, verbose=False)


def test_module_score_bin_indexing_flag():
    a = _toy()
    epi.tl.add_module_score(a, {"A": ["G100"]}, n_bin=5, n_bgd=3, seed=1, archr_bin_indexing=False, verbose=False)
    own = a.uns["module_score"]["background"]["Module.A"]
    epi.tl.add_module_score(a, {"A": ["G100"]}, n_bin=5, n_bgd=3, seed=1, archr_bin_indexing=True, verbose=False)
    archr = a.uns["module_score"]["background"]["Module.A"]
    sums = np.asarray(a.X.sum(0)).ravel()
    order = np.argsort(sums, kind="stable")
    rank = {a.var_names[k]: r for r, k in enumerate(order)}
    # ArchR's positional lookup draws from a lower-signal bin than the module gene's own bin.
    assert np.mean([rank[g] for g in archr]) < np.mean([rank[g] for g in own])


def test_module_score_sample_key_sums():
    a = _toy()
    from epione.tl._module_score import _feature_sums
    X = a.X.toarray()
    s = _feature_sums(a, a.X, "Sample")
    np.testing.assert_allclose(s, X[:30].sum(0) + X[30:].sum(0))


# --------------------------------------------------------------------------- plot groups

def test_plot_groups_data_and_figures():
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(1)
    n = 300
    obs = pd.DataFrame({"grp": rng.choice(["C1", "C2", "C10"], n), "score": rng.normal(size=n)},
                       index=[f"c{k}" for k in range(n)])
    a = ad.AnnData(sparse.csr_matrix(rng.poisson(1, (n, 4)).astype(float)), obs=obs,
                   var=pd.DataFrame(index=["GA", "GB", "GC", "GD"]))
    d = group_plot_data(a, "grp", "score", max_cells=50, seed=1)
    assert d["levels"] == ["C10", "C2", "C1"]          # ridges: rev(mixedsort)
    assert len(d["y"]) == 150
    raw = np.asarray(a.obs.loc[d["cells"], "score"])
    lo, hi = _r_quantile7(raw, [0.002, 0.998])
    np.testing.assert_allclose(d["y"], np.clip(raw, lo, hi))
    v = group_plot_data(a, "grp", "gb", plot_as="violin", max_cells=None)
    assert v["levels"] == ["C1", "C2", "C10"]
    assert set(v["box"].columns) >= {"ymin", "lower", "middle", "upper", "ymax"}
    assert v["violin"]["violinwidth"].max() == pytest.approx(1.0)
    fig = epi.pl.plot_groups(a, "grp", "score")
    figs = epi.pl.plot_groups(a, "grp", ["GA", "GB"], plot_as="violin")
    assert set(figs) == {"GA", "GB"}
    fig2 = epi.pl.plot_confusion_matrix(epi.tl.confusion_matrix(a.obs["grp"], a.obs["grp"]))
    assert fig is not None and fig2 is not None


# --------------------------------------------------------------------------- ArchR parity

_PARITY = Path(os.environ.get("EPIONE_ARCHR_PARITY_DIR", "")) / "module_score_groups"


@pytest.mark.skipif(not os.environ.get("EPIONE_ARCHR_PARITY_DIR") or not (_PARITY / "ref").exists(),
                    reason="EPIONE_ARCHR_PARITY_DIR with ArchR reference outputs not available")
def test_parity_with_archr():
    # Load by path under a unique name: several parity folders ship a
    # ``py_parity.py`` and a plain import would reuse whichever loaded first.
    spec = importlib.util.spec_from_file_location(
        "py_parity_module_score_groups", Path(__file__).parent / "parity" / "module_score_groups" / "py_parity.py")
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)

    _REF = _PARITY / "ref"
    a = pp.load_gene_scores(_PARITY / "tables")
    cm = pp.confusion_parity(a, _REF)
    assert cm["same_order"] and cm["max_abs"] == 0
    tag = "official_B_nbin25_nbgd100_seed1"
    r = pp._scores(_REF, tag).loc[a.obs_names]
    free = epi.tl.add_module_score(a, {"B": pp.MODULES["B"]}, n_bin=25, n_bgd=100, seed=1,
                                   sample_key="Sample", inplace=False, verbose=False)
    assert np.abs(free.values - r.values).max() <= 1e-8
    d = group_plot_data(a, "Clusters", "CD34", plot_as="violin", max_cells=1000, seed=1)
    assert list(d["cells"]) == (_REF / "pg_cells_clusters_seed1.txt").read_text().split()
    l2 = pd.read_csv(_REF / "pg_gs_CD34_violin_layer2.tsv", sep="\t")
    cols = ["ymin", "lower", "middle", "upper", "ymax"]
    assert np.abs(d["box"][cols].values - l2[cols].values).max() <= 1e-8
