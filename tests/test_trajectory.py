"""Tests for the ArchR trajectory port (epione.tl.*trajectory*, epione.pl.plot_trajectory*)."""
from __future__ import annotations

import importlib.util
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import epione as epi
from epione.tl._trajectory import (
    _bh, _center_roll_mean, _get_quantiles, _nknots_smspl, _smooth_spline,
)

# stats::smooth.spline(x, y, df = 250, spar) fitted values from R 4.4.3 at the
# listed 0-based indices of fit$y (x/y generated below).
SMOOTH_SPLINE_REF = {
    "n30": {"n": 30, "spar": 1, "ties": False, "nx": 30, "idx": [0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 21, 23, 25, 27, 29], "fit": [0.80850815752892646, 0.75634478377484959, 0.69140543858680514, 0.61683265794968045, 0.5326810643225306, 0.43791978643966434, 0.33169536299924574, 0.15080862306107343, 0.017096393136490532, -0.1246630693279155, -0.27104612357319458, -0.41875274051813821, -0.56628746825721199, -0.71426741642344793, -0.86417984463717734], "sum": 3.0542782776769837},
    "n120t": {"n": 120, "spar": 0.5, "ties": True, "nx": 31, "idx": [0, 2, 4, 6, 9, 11, 13, 15, 17, 19, 21, 24, 26, 28, 30], "fit": [-0.010686769761158595, 0.38851846874850676, 0.71701091319584731, 0.93304567172853536, 0.97221945103432883, 0.81062376645703804, 0.52292916569565873, 0.1504307648046313, -0.25424486278403824, -0.60701786784087719, -0.85476674491319016, -0.98110737373969581, -0.86387955758525126, -0.60527931895661957, -0.26449024050997261], "sum": 0.28532348940801522},
    "n1000": {"n": 1000, "spar": 1, "ties": False, "nx": 1000, "idx": [0, 71, 143, 214, 285, 357, 428, 499, 571, 642, 714, 785, 856, 928, 999], "fit": [0.033670614825057725, 0.2122033819910277, 0.46521400458855466, 0.71789610269127857, 0.91207854804960631, 0.98806519958717431, 0.90099325719055467, 0.64279477786341344, 0.23860413912238737, -0.22533832109681451, -0.65305749407333447, -0.91820203597293581, -0.94417481118605673, -0.7210564002063552, -0.35941782196308919], "sum": 101.77286455578009},
    "n4000": {"n": 4000, "spar": 0.7, "ties": False, "nx": 4000, "idx": [0, 286, 571, 857, 1143, 1428, 1714, 2000, 2285, 2571, 2856, 3142, 3428, 3713, 3999], "fit": [-0.0014504065451011478, 0.1949157413092639, 0.46206539837103161, 0.72708792845287085, 0.92697309176005815, 1.0035420978736471, 0.9154905076765868, 0.65168539490541011, 0.2457714804091784, -0.22902897138291137, -0.66147839843463663, -0.93916421444484588, -0.97169016046573198, -0.73183848420677033, -0.26033555445551015], "sum": 406.67910772837013},
}


@pytest.mark.parametrize("case", list(SMOOTH_SPLINE_REF))
def test_smooth_spline_matches_r(case):
    ref = SMOOTH_SPLINE_REF[case]
    i = np.arange(1, ref["n"] + 1, dtype=float)
    x = (i / ref["n"]) ** 1.3 * 3
    if ref["ties"]:
        x = np.round(x, 1)
    y = np.sin(2 * x) + 0.3 * np.cos(37 * i) + 0.01 * i / ref["n"]
    ux, fit = _smooth_spline(x, y, spar=ref["spar"])
    assert len(fit) == ref["nx"]
    np.testing.assert_allclose(fit[ref["idx"]], ref["fit"], rtol=0, atol=1e-8)
    assert abs(fit.sum() - ref["sum"]) < 1e-8
    assert np.all(np.diff(ux) > 0)


def test_nknots_matches_r():
    # stats:::.nknots.smspl
    assert [_nknots_smspl(n) for n in (10, 49, 50, 100, 199, 500, 1000, 3199, 3200, 5000)] == \
        [10, 49, 49, 62, 99, 118, 144, 199, 200, 204]


def test_get_quantiles_and_roll_mean():
    np.testing.assert_allclose(_get_quantiles([3, 1, 3, 2]), [0.75, 0.25, 0.75, 0.5])
    np.testing.assert_allclose(_center_roll_mean(np.arange(1, 11.0), 3),
                               [2, 2, 3, 4, 5, 6, 7, 8, 9, 9])
    np.testing.assert_allclose(_center_roll_mean(np.arange(1, 11.0), 4),
                               [2.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 8.5, 8.5])


def test_bh_matches_r_semantics():
    p = np.array([0.01, 0.04, np.nan, 0.03, 0.5])
    # p.adjust(c(0.01, 0.04, NA, 0.03, 0.5), "fdr")
    np.testing.assert_allclose(_bh(p), [0.04, 0.16 / 3, np.nan, 0.16 / 3, 0.5])


def _toy(n_per=150, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0, 0], [4, 1, 0], [8, 0, 1], [0, 8, 8]])
    X = np.vstack([c + rng.normal(scale=0.8, size=(n_per, 3)) for c in centers])
    groups = np.repeat(["A", "B", "C", "D"], n_per)
    genes = np.abs(rng.normal(size=(X.shape[0], 20))) * 5
    genes[:, 0] += X[:, 0].clip(0) * 3           # increases along A -> C
    genes[:, 1] += (8 - X[:, 0]).clip(0) * 3     # decreases
    ad = AnnData(X=genes, obs=pd.DataFrame({"cl": groups}, index=[f"c{i}" for i in range(len(groups))]),
                 var=pd.DataFrame(index=[f"chr1:G{i}" for i in range(20)]))
    ad.obsm["X_lsi"] = X
    ad.obsm["X_umap"] = X[:, :2]
    return ad


def test_add_trajectory_orders_groups():
    ad = _toy()
    df = epi.tl.add_trajectory(ad, ["A", "B", "C"], groupby="cl", use_rep="X_lsi",
                               return_df=True, verbose=False)
    pt = ad.obs["Trajectory"]
    assert pt[ad.obs.cl == "D"].isna().all()
    assert (pt.dropna() > 0).all() and (pt.dropna() <= 100).all()
    med = pt.groupby(ad.obs.cl, observed=True).median()
    assert med["A"] < med["B"] < med["C"]
    assert set(df.columns) == {"Distance", "DistanceIdx", "Trajectory"}
    assert ad.uns["Trajectory_spline"].shape[1] == 3
    with pytest.raises(ValueError):
        epi.tl.add_trajectory(ad, ["A", "B"], groupby="cl", use_rep="X_lsi", verbose=False)


def test_get_trajectory_heatmap_correlate():
    ad = _toy()
    epi.tl.add_trajectory(ad, ["A", "B", "C"], groupby="cl", use_rep="X_lsi", verbose=False)
    tr = epi.tl.get_trajectory(ad, trajectory_label="cl", smooth_window=5)
    assert tr.shape == (100, 20)
    assert tr.obs_names[0] == "T.0_1" and tr.obs_names[-1] == "T.99_100"
    assert {"mat", "smoothMat"} <= set(tr.layers)
    assert tr.obs["label"].iloc[0] == "A" and tr.obs["label"].iloc[-1] == "C"
    np.testing.assert_allclose(np.nansum(2 ** tr.layers["mat"] - 1, axis=1)[:3], 10000, rtol=1e-10)

    tr2 = epi.tl.get_trajectory(ad, group_every=5, smooth_window=None)
    assert tr2.shape[0] == 20 and "smoothMat" not in tr2.layers

    hm = epi.tl.trajectory_heatmap_matrix(tr, var_cutoff=0.5)
    assert hm.shape == (10, 100)
    assert hm.values.max() <= 1.5 and hm.values.min() >= -1.5
    assert np.all(np.diff(np.argmax(hm.values, axis=1)) >= 0)
    assert hm.index[0] == "chr1:G1" and hm.index[-1] == "chr1:G0"

    tr_b = tr.copy()
    tr_b.var_names = [f"z:G{i}_motif{i}" for i in range(20)]
    tr_b.var["name"] = [f"G{i}_motif{i}" for i in range(20)]
    tr.var["name"] = [f"G{i}" for i in range(20)]
    cor = epi.tl.correlate_trajectories(tr, tr_b, name_key1="name", name_key2="name",
                                        var_cutoff1=0, var_cutoff2=0)
    allm = cor["allMappings"]
    assert len(allm) == 20
    np.testing.assert_allclose(allm["Correlation"].values, 1.0, atol=1e-12)
    assert len(cor["correlatedMappings"]) == 20


def test_plots_smoke():
    ad = _toy()
    epi.tl.add_trajectory(ad, ["A", "B", "C"], groupby="cl", use_rep="X_lsi", verbose=False)
    fig, axes = epi.pl.plot_trajectory(ad, basis="X_umap", show=False)
    assert len(axes) == 2
    fig, axes = epi.pl.plot_trajectory(ad, color_by="chr1:G0", basis="X_umap", show=False)
    tr = epi.tl.get_trajectory(ad, trajectory_label="cl")
    fig, mat = epi.pl.plot_trajectory_heatmap(tr, var_cutoff=0.5, color_columns=True,
                                              label_markers=["chr1:G0"], show=False)
    assert mat.shape[0] == 10


# ---------------------------------------------------------------------------
# Parity against ArchR 1.0.3 on the Hematopoiesis tutorial project
# ---------------------------------------------------------------------------
_PARITY_ROOT = os.environ.get("EPIONE_ARCHR_PARITY_DIR")
REF = pathlib.Path(_PARITY_ROOT) / "trajectory" if _PARITY_ROOT else None
TABLES = REF / "tables" if REF is not None else None


@pytest.mark.skipif(REF is None or not (REF / "trajectory_lsi.tsv").exists()
                    or not (TABLES / "GeneScoreMatrix.mtx").exists(),
                    reason="set EPIONE_ARCHR_PARITY_DIR to the ArchR reference outputs")
def test_parity_with_archr():
    path = pathlib.Path(__file__).parent / "parity" / "trajectory" / "parity_trajectory.py"
    spec = importlib.util.spec_from_file_location("parity_trajectory", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = mod.run(str(REF), str(TABLES))
    failed = {k: v for k, v in res.items() if not mod.passed(k, v)}
    assert not failed, failed
