"""Tests for epione.tl.differential_peaks — dual-backend DE analysis.

Uses a synthetic (samples × features) count matrix where we plant
N_UP features with 3× mean in the 'trt' condition and N_DN features
with 1/3× mean, everything else null. Both backends should:

- return the unified result schema
- recover the planted direction (log2FoldChange sign matches)
- rank the planted features above the nulls by padj.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epione.tl import differential_peaks


CANONICAL_COLS = ["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]


def _make_planted_counts(n_features=1500, n_per_group=4,
                          n_up=30, n_dn=30, seed=0):
    rng = np.random.default_rng(seed)
    n_samples = n_per_group * 2
    base = rng.gamma(2.0, 10.0, n_features) + 5
    fold = np.ones(n_features)
    fold[:n_up] = 3.0
    fold[n_up:n_up + n_dn] = 1.0 / 3.0

    metadata = pd.DataFrame({
        "condition": ["ctrl"] * n_per_group + ["trt"] * n_per_group,
    }, index=[f"S{i+1}" for i in range(n_samples)])

    counts = np.zeros((n_samples, n_features), dtype=int)
    for i, cond in enumerate(metadata["condition"]):
        mu = base * (fold if cond == "trt" else 1.0)
        counts[i] = rng.poisson(mu)

    counts_df = pd.DataFrame(
        counts, index=metadata.index,
        columns=[f"feat_{j}" for j in range(n_features)],
    )
    return counts_df, metadata, (n_up, n_dn)


@pytest.mark.parametrize("backend", ["pydeseq2", "edgepy"])
def test_differential_peaks_schema_and_direction(backend):
    """Each backend returns the canonical schema and gets the LFC sign
    right on planted features."""
    if backend == "edgepy":
        pytest.importorskip("inmoose")
        pytest.importorskip("patsy")
    else:
        pytest.importorskip("pydeseq2")

    counts, meta, (n_up, n_dn) = _make_planted_counts()
    res = differential_peaks(
        counts=counts, metadata=meta,
        design="~condition",
        contrast=("condition", "trt", "ctrl"),
        backend=backend,
        min_count=10, min_samples=1,
        quiet=True,
    )
    # Schema.
    assert list(res.columns) == CANONICAL_COLS, \
        f"{backend} returned columns {list(res.columns)}"

    # Direction: planted up features should have positive LFC on average.
    up_lfc = res.loc[[f"feat_{i}" for i in range(n_up)], "log2FoldChange"]
    dn_lfc = res.loc[[f"feat_{i + n_up}" for i in range(n_dn)], "log2FoldChange"]
    assert up_lfc.mean() > 0.8, f"up mean LFC {up_lfc.mean():.3f} < 0.8"
    assert dn_lfc.mean() < -0.8, f"dn mean LFC {dn_lfc.mean():.3f} > -0.8"


@pytest.mark.parametrize("backend", ["pydeseq2", "edgepy"])
def test_differential_peaks_planted_features_rank_top(backend):
    """Planted features should dominate the padj-sorted head over random nulls."""
    if backend == "edgepy":
        pytest.importorskip("inmoose")
        pytest.importorskip("patsy")
    else:
        pytest.importorskip("pydeseq2")

    counts, meta, (n_up, n_dn) = _make_planted_counts()
    res = differential_peaks(
        counts=counts, metadata=meta,
        design="~condition",
        contrast=("condition", "trt", "ctrl"),
        backend=backend,
        min_count=10, min_samples=1, quiet=True,
    )
    n_planted = n_up + n_dn
    top = res.sort_values("padj").head(n_planted).index
    planted = {f"feat_{i}" for i in range(n_planted)}
    recall = len(planted & set(top)) / n_planted
    assert recall >= 0.8, f"{backend} only recovered {recall:.1%} of planted features in top-{n_planted}"


def test_differential_peaks_rejects_missing_contrast():
    counts, meta, _ = _make_planted_counts(n_features=100)
    with pytest.raises(ValueError, match="contrast"):
        differential_peaks(counts=counts, metadata=meta,
                           design="~condition")


def test_differential_peaks_rejects_duplicate_feature_names():
    counts, meta, _ = _make_planted_counts(n_features=100)
    counts.columns = ["feat_0"] * counts.shape[1]
    with pytest.raises(ValueError, match="duplicate"):
        differential_peaks(counts=counts, metadata=meta,
                           design="~condition",
                           contrast=("condition", "trt", "ctrl"))


def test_differential_peaks_min_count_filter_drops_low_features():
    counts, meta, _ = _make_planted_counts(n_features=100)
    # Add 50 all-zero features; they must not appear in the result.
    for j in range(50):
        counts[f"zero_{j}"] = 0
    res = differential_peaks(counts=counts, metadata=meta,
                             design="~condition",
                             contrast=("condition", "trt", "ctrl"),
                             backend="pydeseq2",
                             min_count=10, min_samples=1, quiet=True)
    assert not any(str(f).startswith("zero_") for f in res.index)
