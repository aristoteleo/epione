"""Doublet detection: ArchR port, filter_doublets and the pyscdblfinder wrappers.

Synthetic, CPU-only, no downloads. Parity with ArchR and the real-data benchmark
live in tests/parity/doublets/.
"""
import sys
import types

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import epione as epi
from epione.pp import _doublets as D


def _synthetic(n_per_type=250, n_types=3, n_feat=6000, n_dbl=60, seed=0):
    """Cells x tiles with cell-type specific open tiles, plus heterotypic doublets."""
    rng = np.random.default_rng(seed)
    prof = rng.random((n_types, n_feat)) < 0.02
    for t in range(n_types):
        prof[t, rng.choice(n_feat, 400, replace=False)] = True
    rows, labels = [], []
    for t in range(n_types):
        p = np.where(prof[t], 0.35, 0.004)
        rows.append(rng.random((n_per_type, n_feat)) < p)
        labels += [f"t{t}"] * n_per_type
    singlets = np.vstack(rows)
    a = rng.integers(0, n_per_type, n_dbl)
    b = rng.integers(n_per_type, 2 * n_per_type, n_dbl)
    doublets = singlets[a] | singlets[b]
    X = sp.csr_matrix(np.vstack([singlets, doublets]).astype(np.float32))
    obs = pd.DataFrame({"type": labels + ["dbl"] * n_dbl,
                        "is_dbl": [False] * len(labels) + [True] * n_dbl},
                       index=[f"S#c{i}" for i in range(X.shape[0])])
    obs["n_fragment"] = np.asarray(X.sum(axis=1)).ravel() * 1.5
    return AnnData(X, obs=obs)


# --------------------------------------------------------------------------- LSI

def test_project_lsi_reproduces_fit():
    """Projecting the fitted cells reproduces the fitted embedding (ArchR's own check)."""
    ad = _synthetic(n_dbl=0)
    mat = sp.csc_matrix(ad.X.T)
    lsi = D._compute_lsi(mat, n_dims=10)
    keep, proj = D._project_lsi(mat, lsi)
    np.testing.assert_array_equal(keep, lsi["cells"])
    np.testing.assert_allclose(proj, lsi["mat_svd"], rtol=1e-6, atol=1e-8)


def test_binom_scores_match_archr_formula():
    from scipy.stats import binom
    counts = np.array([0, 3, 10, 50, 2], dtype=float)
    score, enrich = D._binom_scores(counts, n_cells=5)
    scale = 10000 / 5
    p = binom.sf(np.round(counts * scale) - 1, np.round(counts.sum() * scale), 1e-4)
    np.testing.assert_allclose(score, -np.log10(np.maximum(np.minimum(p * 5, 1), 4.940656e-324)))
    np.testing.assert_allclose(enrich, 10000 * counts / counts.sum())


def test_archr_doublets_ranks_doublets_high():
    from sklearn.metrics import roc_auc_score
    ad = _synthetic()
    epi.pp.archr_doublets(ad, var_features=1500, total_features=6000, dims=15, n_sample=200,
                          force=True, seed=1)
    for c in ("archr_doublet_score", "archr_doublet_enrichment"):
        assert c in ad.obs and np.isfinite(ad.obs[c]).all()
    auc = roc_auc_score(ad.obs["is_dbl"], ad.obs["archr_doublet_enrichment"])
    assert auc > 0.85, auc
    assert set(ad.uns["archr_doublet"]) == {"0"}


def test_archr_doublets_per_sample_and_not_inplace():
    ad = _synthetic(n_per_type=150, n_dbl=30)
    ad.obs["sample"] = np.where(np.arange(ad.n_obs) % 2 == 0, "A", "B")
    df = epi.pp.archr_doublets(ad, sample_key="sample", var_features=1500, total_features=6000,
                               dims=10, n_sample=100, force=True, inplace=False)
    assert list(df.columns) == ["archr_doublet_score", "archr_doublet_enrichment"]
    assert (df.index == ad.obs_names).all()
    assert "archr_doublet_score" not in ad.obs


# --------------------------------------------------------------------------- filter

def test_archr_filter_rule():
    """Stable descending order, cutEnrich, head(n * ratio * n / 1e5) truncated."""
    n = 1000  # 1000 * 1000 / 1e5 = 10 cells per sample at most
    enrich = np.r_[np.linspace(5, 0, n), np.linspace(0.5, 3, n)]
    groups = np.r_[["A"] * n, ["B"] * n]
    enrich[[3, 4]] = enrich[2]  # tie: keep original order
    m = D._archr_filter_mask(enrich, np.zeros(2 * n), groups, 1.0, -np.inf, 1.0)
    assert m[:n].sum() == 10 and m[:10].all()
    top_b = n + np.argsort(-enrich[n:], kind="stable")[:10]
    assert m[n:].sum() == 10 and m[top_b].all()
    # 0.5 ratio: int(0.5 * 1000 * 0.01) = 5
    assert D._archr_filter_mask(enrich, np.zeros(2 * n), groups, 1.0, None, 0.5)[:n].sum() == 5


def test_filter_doublets_modes():
    ad = _synthetic(n_per_type=50, n_dbl=10)
    ad.obs["s"] = np.linspace(0, 1, ad.n_obs)
    ad.obs["amulet_q"] = np.linspace(1, 0, ad.n_obs)
    ad.obs["cls"] = np.where(ad.obs["is_dbl"], "doublet", "singlet")
    a = epi.pp.filter_doublets(ad, method="threshold", score_key="s", threshold=0.9, inplace=False)
    assert a.n_obs == (ad.obs["s"] < 0.9).sum()
    b = epi.pp.filter_doublets(ad, method="threshold", score_key="amulet_q", threshold=0.01,
                               inplace=False)
    assert b.n_obs == (ad.obs["amulet_q"] > 0.01).sum()
    c = ad.copy()
    epi.pp.filter_doublets(c, method="class", score_key="cls")
    assert c.n_obs == 150 and not c.obs["is_dbl"].any()
    # key_added marks the cells removed by the last call made on ``ad`` (the q-value one)
    assert ad.obs["doublet_removed"].sum() == (ad.obs["amulet_q"] <= 0.01).sum()
    with pytest.raises(KeyError):
        epi.pp.filter_doublets(ad.copy(), method="archr")


# --------------------------------------------------------------------------- wrappers

@pytest.fixture
def fake_pyscdbl(monkeypatch):
    calls = {}
    mod = types.ModuleType("pyscdblfinder")
    mod.__version__ = "0.3.0"

    class Res:
        def __init__(self, n):
            self.table = pd.DataFrame({"score": np.linspace(0, 1, n),
                                       "class": ["singlet"] * (n - 1) + ["doublet"]})
            self.n_real, self.score_threshold = n, 0.9

        def real_cells(self):
            return self.table.iloc[: self.n_real]

    def sc_dbl_finder(counts, *, aggregate_features=False, n_features=1352, processing="default",
                      dbr=None, random_state=0, artificial_doublets=None):
        calls.setdefault("sdf", []).append(dict(shape=counts.shape, agg=aggregate_features,
                                                nf=n_features, proc=processing))
        return Res(counts.shape[1])

    def amulet(fragments, *, barcodes=None, regions_to_exclude=None):
        calls["amulet"] = dict(fragments=fragments, barcodes=list(barcodes),
                               rte=regions_to_exclude)
        b = list(barcodes)
        return pd.DataFrame({"nAbove2": np.arange(len(b)), "p.value": np.linspace(1, 0.001, len(b)),
                             "q.value": np.linspace(1, 0.001, len(b))}, index=b)

    def clamulet(fragments, *, barcodes=None):
        b = list(barcodes)
        return pd.DataFrame({"score": np.linspace(0, 1, len(b))}, index=b[::-1])

    mod.sc_dbl_finder, mod.amulet, mod.clamulet = sc_dbl_finder, amulet, clamulet
    monkeypatch.setitem(sys.modules, "pyscdblfinder", mod)
    return calls


def test_scdblfinder_wrapper(fake_pyscdbl):
    ad = _synthetic(n_per_type=20, n_dbl=4)
    ad.obs["sample"] = ["A"] * 30 + ["B"] * (ad.n_obs - 30)
    epi.pp.scdblfinder(ad, sample_key="sample")
    sdf = fake_pyscdbl["sdf"]
    assert [c["shape"] for c in sdf] == [(ad.n_vars, 30), (ad.n_vars, ad.n_obs - 30)]
    assert all(c["agg"] and c["nf"] == 25 and c["proc"] == "normFeatures" for c in sdf)
    assert ad.obs["scDblFinder_class"].tolist().count("doublet") == 2
    assert set(ad.uns["scDblFinder"]["threshold"]) == {"A", "B"}
    with pytest.raises(TypeError, match="does not accept"):
        epi.pp.scdblfinder(ad, not_an_argument=1)


def test_amulet_and_clamulet_wrappers(fake_pyscdbl):
    ad = _synthetic(n_per_type=10, n_dbl=2)
    epi.pp.amulet(ad, "frags.tsv.gz", regions_to_exclude="rep.bed")
    call = fake_pyscdbl["amulet"]
    assert call["fragments"] == "frags.tsv.gz" and call["rte"] == "rep.bed"
    assert call["barcodes"][0] == "c0"  # "S#c0" -> fragment barcode "c0"
    assert (ad.obs["amulet_class"] == "doublet").sum() == (ad.obs["amulet_q"] < 0.01).sum()
    ad.uns["files"] = {"fragments": "from_uns.tsv.gz"}
    epi.pp.clamulet(ad)
    np.testing.assert_allclose(ad.obs["clamulet_score"], np.linspace(1, 0, ad.n_obs))  # reindexed
    epi.pp.scdblfinder(ad)
    epi.pp.combine_doublet_scores(ad)
    from scipy.stats import chi2
    p = np.maximum(np.c_[ad.obs["amulet_p"], 1 - ad.obs["scDblFinder_score"]], 0.001)
    np.testing.assert_allclose(ad.obs["doublet_combined_p"], chi2.sf(-2 * np.log(p).sum(1), 4))


def test_wrapper_version_gate(monkeypatch):
    old = types.ModuleType("pyscdblfinder")
    old.__version__ = "0.2.1"
    monkeypatch.setitem(sys.modules, "pyscdblfinder", old)
    with pytest.raises(ImportError, match="0.3.0"):
        epi.pp.scdblfinder(_synthetic(n_per_type=5, n_dbl=1))


# --------------------------------------------------------------------------- parity (opt-in)

def _parity():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "doublets_parity_check", Path(__file__).parent / "parity/doublets/parity_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not __import__("os").environ.get("EPIONE_DOUBLETS_PARITY_DIR"),
                    reason="needs EPIONE_DOUBLETS_PARITY_DIR (ArchR references)")
def test_gate_f1_filter_matches_archr_exactly():
    pc = _parity()
    res = pc.gate_f1(pc.parity_dir())
    assert res["identical"], res
