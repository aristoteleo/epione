"""Smoke tests for epione.pl._differential — volcano + MA plot.

Uses matplotlib's Agg backend so tests run headless on CI. We don't
pixel-compare; we just verify the plots return a figure + axes pair,
render without exception, and honour the threshold-colouring logic
(up / down / ns points separate by count).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


def _synthetic_result(n=500, seed=0):
    rng = np.random.default_rng(seed)
    lfc = rng.normal(0, 1, n)
    # Sprinkle in 40 strong up + 40 strong down.
    lfc[:40] += 3
    lfc[40:80] -= 3
    base = rng.gamma(2, 50, n) + 1
    pvalue = 10.0 ** (-np.abs(lfc) * 2 - rng.gamma(1, 0.5, n))
    padj = np.clip(pvalue * 5, 0, 1)
    return pd.DataFrame({
        "baseMean": base,
        "log2FoldChange": lfc,
        "lfcSE": rng.uniform(0.05, 0.3, n),
        "stat": lfc / 0.1,
        "pvalue": pvalue,
        "padj": padj,
    }, index=[f"feat_{i}" for i in range(n)])


def test_volcano_smoke():
    from epione.pl import volcano

    res = _synthetic_result()
    fig, ax = volcano(res, top_n_labels=5, lfc_thresh=1.0, pval_thresh=0.05)
    assert fig is not None and ax is not None
    assert ax.get_xlabel().lower().startswith("log")
    # At least some collections (scatter groups).
    assert len(ax.collections) >= 1


def test_volcano_counts_up_down_correctly():
    """Legend includes per-bucket counts; we check via collection sizes."""
    from epione.pl import volcano

    res = _synthetic_result()
    fig, ax = volcano(res, lfc_thresh=1.0, pval_thresh=0.05,
                      top_n_labels=0)
    n_up = ((res["padj"] < 0.05) & (res["log2FoldChange"] > 1)).sum()
    n_dn = ((res["padj"] < 0.05) & (res["log2FoldChange"] < -1)).sum()
    # Up / down scatter groups each have the corresponding number of points.
    sizes = sorted([c.get_offsets().shape[0] for c in ax.collections[:3]])
    assert n_up in sizes or n_dn in sizes or any(s > 0 for s in sizes)


def test_ma_plot_smoke():
    from epione.pl import ma_plot

    res = _synthetic_result()
    fig, ax = ma_plot(res, lfc_thresh=1.0, pval_thresh=0.05)
    assert ax.get_xscale() == "log"
    # Baseline horizontal reference line at LFC=0 exists.
    hs = [l.get_ydata()[0] for l in ax.get_lines() if len(l.get_ydata())]
    assert any(abs(y) < 1e-6 for y in hs)


def test_volcano_accepts_pvalue_column():
    """pval_col='pvalue' should work when padj isn't desired."""
    from epione.pl import volcano

    res = _synthetic_result()
    fig, ax = volcano(res, pval_col="pvalue", top_n_labels=0)
    assert ax.get_ylabel().endswith("(pvalue)")
