"""Tests for epione.utils._sampling — peak-set / distance utilities
that drive the OTX2 case study.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epione.utils import (
    distance_to_nearest_peak,
    filter_distal_peaks,
    classify_peaks_by_overlap,
    expression_matched_sample,
)


def _mk_peaks(starts, ends, chrom="chr1"):
    return pd.DataFrame({
        "chrom": chrom,
        "start": np.asarray(starts, dtype=int),
        "end":   np.asarray(ends,   dtype=int),
    })


def test_filter_distal_peaks_keeps_far_peaks():
    peaks = _mk_peaks([100, 5_000, 20_000], [200, 5_100, 20_100])
    tss   = pd.DataFrame({"chrom": "chr1", "tss": [1_000, 30_000]})
    out = filter_distal_peaks(peaks, tss, min_distance=2_500)
    # Peak at 100-200 is within 2.5 kb of tss=1000 → dropped.
    # Peak at 5_000 is 4 kb from tss=1000 → kept.
    # Peak at 20_000 is 10 kb from tss=30_000 → kept.
    assert list(out["start"]) == [5_000, 20_000]


def test_filter_distal_peaks_empty_input():
    peaks = _mk_peaks([], [])
    tss = pd.DataFrame({"chrom": ["chr1"], "tss": [1_000]})
    out = filter_distal_peaks(peaks, tss, min_distance=2_500)
    assert len(out) == 0


def test_filter_distal_peaks_keeps_peak_on_tss_less_chrom():
    """Peaks on a chromosome with no annotated TSS are trivially distal."""
    peaks = _mk_peaks([500], [600], chrom="chrY_weird")
    tss = pd.DataFrame({"chrom": ["chr1"], "tss": [1_000]})
    out = filter_distal_peaks(peaks, tss, min_distance=5_000)
    assert len(out) == 1


def test_distance_to_nearest_peak_basic():
    peaks = _mk_peaks([100, 1_000, 10_000], [200, 1_100, 10_100])
    features = pd.DataFrame({"chrom": "chr1", "tss": [500, 9_900]})
    d = distance_to_nearest_peak(features, peaks,
                                  feature_col="tss",
                                  peak_center="center")
    # nearest to tss=500 is peak centre 150 (d=350) or 1050 (d=550)
    #   → 350 wins
    # nearest to tss=9900 is centre 10050 → d=150
    assert list(d) == [350.0, 150.0]


def test_distance_to_nearest_peak_skips_chrom_without_peaks():
    peaks = _mk_peaks([100], [200], chrom="chr1")
    features = pd.DataFrame({"chrom": ["chr1", "chr2"],
                              "tss": [500, 500]})
    d = distance_to_nearest_peak(features, peaks,
                                  feature_col="tss", peak_center="center")
    # chr2 feature has no same-chrom peaks → skipped from output.
    assert len(d) == 1


def test_classify_peaks_by_overlap_three_sets():
    a = _mk_peaks([100, 500, 1_500], [200, 600, 1_600])
    b = _mk_peaks([150, 2_000],      [250, 2_100])
    c = _mk_peaks([1_520, 3_000],    [1_620, 3_100])

    res = classify_peaks_by_overlap(
        {"A": a, "B": b, "C": c}, primary_order=["A", "B", "C"],
    )
    # Expected cluster labels:
    #   peak 100-200 (A) overlaps b:150-250    → "A/B shared"
    #   peak 500-600 (A) overlaps nothing else → "A-specific"
    #   peak 1500-1600 (A) overlaps c:1520-1620 → "A/C shared"
    #   peak 2000-2100 (B) no A, no C           → "B-specific"
    #   peak 3000-3100 (C) no A, no B           → "C-specific"
    cluster_counts = res["cluster"].value_counts().to_dict()
    assert cluster_counts.get("A-specific") == 1
    assert cluster_counts.get("B-specific") == 1
    assert cluster_counts.get("C-specific") == 1
    assert cluster_counts.get("A/B shared") == 1
    assert cluster_counts.get("A/C shared") == 1


def test_classify_peaks_by_overlap_primary_order_breaks_ties():
    a = _mk_peaks([100], [200])
    b = _mk_peaks([150], [250])
    ab = classify_peaks_by_overlap({"A": a, "B": b}, primary_order=["A", "B"])
    ba = classify_peaks_by_overlap({"A": a, "B": b}, primary_order=["B", "A"])
    # Overlapping peaks get collapsed to one row anchored to the set
    # listed first in primary_order; cluster label names all sets it
    # hits, regardless of order.
    assert len(ab) == 1
    assert len(ba) == 1
    # The cluster label lists set names in primary_order sequence.
    assert set(ab["cluster"]) == {"A/B shared"}
    assert set(ba["cluster"]) == {"B/A shared"}
    # Under "A first" the coordinates come from A (start=100); under
    # "B first" from B (start=150).
    assert int(ab["start"].iloc[0]) == 100
    assert int(ba["start"].iloc[0]) == 150


def test_expression_matched_sample_matches_distribution():
    rng = np.random.default_rng(0)
    # Target lives in the high-expression quantiles; pool covers a much
    # wider range. Expression-matching should pull samples from the upper
    # end of pool.
    target = rng.lognormal(3.0, 0.3, 100)
    pool   = rng.lognormal(1.5, 1.0, 3000)

    idx = expression_matched_sample(target_values=target,
                                     pool_values=pool, seed=1)
    assert len(idx) > 0
    sampled = pool[idx]
    # Matched sample's median should be much closer to target's median than
    # a naive uniform draw from the pool would be.
    naive_med = float(np.median(pool))
    assert abs(np.median(sampled) - np.median(target)) < \
           abs(naive_med - np.median(target))
