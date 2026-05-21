"""Tests for the no-replicate (Poisson) differential backend and the two
region x sample quantification builders — count_reads_in_peaks and
peak_signal_matrix.

These cover the path used when a ChIP-seq / ATAC-seq comparison has only
one sample per condition (no dispersion can be estimated) and the input
is raw BAMs or per-sample narrowPeak files rather than a ready count
matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epione.tl import (
    differential_peaks,
    count_reads_in_peaks,
    peak_signal_matrix,
)

CANONICAL_COLS = ["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]


# --------------------------------------------------------------------------
# Poisson backend + replicate guard
# --------------------------------------------------------------------------

def _make_one_vs_one(n_features=1200, n_up=40, n_dn=40, seed=0):
    """One sample per condition; plant n_up regions 4x up and n_dn 4x down
    in the 'trt' sample. Region means are large enough to clear min_count."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(3.0, 40.0, n_features) + 20      # mean ~140 reads/region
    fold = np.ones(n_features)
    fold[:n_up] = 4.0
    fold[n_up:n_up + n_dn] = 0.25
    ctrl = rng.poisson(base)
    trt = rng.poisson(base * fold)
    counts = pd.DataFrame(
        np.vstack([ctrl, trt]),
        index=["ctrl", "trt"],
        columns=[f"region_{j}" for j in range(n_features)],
    )
    meta = pd.DataFrame({"condition": ["ctrl", "trt"]}, index=["ctrl", "trt"])
    return counts, meta, (n_up, n_dn)


def test_poisson_backend_schema_and_direction():
    counts, meta, (n_up, n_dn) = _make_one_vs_one()
    res = differential_peaks(
        counts=counts, metadata=meta,
        contrast=("condition", "trt", "ctrl"), backend="poisson",
        min_count=5, min_samples=1, quiet=True,
    )
    assert list(res.columns) == CANONICAL_COLS
    up = res.loc[[f"region_{i}" for i in range(n_up)], "log2FoldChange"]
    dn = res.loc[[f"region_{i + n_up}" for i in range(n_dn)], "log2FoldChange"]
    assert up.mean() > 0.8, f"up mean LFC {up.mean():.3f}"
    assert dn.mean() < -0.8, f"dn mean LFC {dn.mean():.3f}"


def test_poisson_backend_ranks_planted_top():
    counts, meta, (n_up, n_dn) = _make_one_vs_one()
    res = differential_peaks(
        counts=counts, metadata=meta,
        contrast=("condition", "trt", "ctrl"), backend="poisson",
        min_count=5, min_samples=1, quiet=True,
    )
    n_planted = n_up + n_dn
    top = set(res.sort_values("padj").head(n_planted).index)
    planted = {f"region_{i}" for i in range(n_planted)}
    recall = len(planted & top) / n_planted
    assert recall >= 0.7, f"poisson recovered only {recall:.0%} of planted"


def test_poisson_backend_one_sided():
    """alternative='less' flags only the planted-down regions; the
    planted-up regions stay non-significant under a depletion-only test."""
    counts, meta, (n_up, n_dn) = _make_one_vs_one()
    res = differential_peaks(
        counts=counts, metadata=meta,
        contrast=("condition", "trt", "ctrl"), backend="poisson",
        alternative="less", min_count=5, min_samples=1, quiet=True,
    )
    dn = res.loc[[f"region_{i + n_up}" for i in range(n_dn)], "padj"]
    up = res.loc[[f"region_{i}" for i in range(n_up)], "padj"]
    assert (dn < 0.05).mean() > 0.7, "planted-down not caught by 'less'"
    assert (up > 0.5).mean() > 0.7, "planted-up wrongly flagged by 'less'"


def test_poisson_backend_rejects_bad_alternative():
    counts, meta, _ = _make_one_vs_one(n_features=120)
    with pytest.raises(ValueError, match="alternative"):
        differential_peaks(
            counts=counts, metadata=meta,
            contrast=("condition", "trt", "ctrl"), backend="poisson",
            alternative="lower", quiet=True,
        )


def test_count_backends_reject_no_replicate_design():
    """pydeseq2 / edgepy must refuse a one-vs-one design and name the
    poisson backend in the error."""
    counts, meta, _ = _make_one_vs_one(n_features=120)
    for backend in ("pydeseq2", "edgepy"):
        with pytest.raises(ValueError, match="poisson"):
            differential_peaks(
                counts=counts, metadata=meta,
                contrast=("condition", "trt", "ctrl"), backend=backend,
                quiet=True,
            )


def test_poisson_backend_pools_replicates():
    """With >1 sample per condition the poisson backend still runs (pools
    them as added depth) and recovers direction."""
    rng = np.random.default_rng(1)
    n = 300
    base = rng.gamma(3.0, 40.0, n) + 20
    fold = np.ones(n); fold[:20] = 4.0
    rows, idx, cond = [], [], []
    for k in range(2):
        rows.append(rng.poisson(base)); idx.append(f"c{k}"); cond.append("ctrl")
    for k in range(2):
        rows.append(rng.poisson(base * fold)); idx.append(f"t{k}"); cond.append("trt")
    counts = pd.DataFrame(np.vstack(rows), index=idx,
                          columns=[f"r{j}" for j in range(n)])
    meta = pd.DataFrame({"condition": cond}, index=idx)
    res = differential_peaks(
        counts=counts, metadata=meta,
        contrast=("condition", "trt", "ctrl"), backend="poisson",
        min_count=5, min_samples=1, quiet=True,
    )
    up = res.loc[[f"r{i}" for i in range(20)], "log2FoldChange"]
    assert up.mean() > 0.8


# --------------------------------------------------------------------------
# count_reads_in_peaks
# --------------------------------------------------------------------------

def _write_bam(path, reads, refs):
    """reads: list of (chrom, start, length). refs: {name: length}."""
    import pysam
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": k, "LN": v} for k, v in refs.items()]}
    names = list(refs)
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for i, (chrom, start, length) in enumerate(sorted(reads)):
            a = pysam.AlignedSegment()
            a.query_name = f"read{i}"
            a.query_sequence = "A" * length
            a.flag = 0
            a.reference_id = names.index(chrom)
            a.reference_start = start
            a.mapping_quality = 30
            a.cigar = [(0, length)]
            a.query_qualities = pysam.qualitystring_to_array("I" * length)
            bam.write(a)
    pysam.index(str(path))


def test_count_reads_in_peaks_matches_known_layout(tmp_path):
    import pysam  # noqa: F401  (skip cleanly if pysam absent)
    refs = {"chr1": 10_000}
    peaks = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"],
        "start": [100, 500, 5000],
        "end":   [200, 600, 5100],
    }, index=["p1", "p2", "p3"])
    # sample A: 5 reads in p1, 2 in p2, 0 in p3
    reads_a = ([("chr1", 110, 50)] * 5 + [("chr1", 520, 50)] * 2)
    # sample B: 1 read in p1, 0 in p2, 4 in p3
    reads_b = ([("chr1", 150, 50)] * 1 + [("chr1", 5010, 50)] * 4)
    _write_bam(tmp_path / "A.bam", reads_a, refs)
    _write_bam(tmp_path / "B.bam", reads_b, refs)

    counts = count_reads_in_peaks(
        {"A": tmp_path / "A.bam", "B": tmp_path / "B.bam"}, peaks)
    assert list(counts.index) == ["p1", "p2", "p3"]
    assert list(counts.columns) == ["A", "B"]
    assert counts.loc["p1", "A"] == 5
    assert counts.loc["p2", "A"] == 2
    assert counts.loc["p3", "A"] == 0
    assert counts.loc["p1", "B"] == 1
    assert counts.loc["p3", "B"] == 4


def test_count_reads_in_peaks_skips_unknown_chrom(tmp_path):
    import pysam  # noqa: F401
    refs = {"chr1": 10_000}
    _write_bam(tmp_path / "A.bam", [("chr1", 110, 50)] * 3, refs)
    peaks = pd.DataFrame({
        "chrom": ["chr1", "chrZ"], "start": [100, 100], "end": [200, 200],
    }, index=["p1", "absent"])
    counts = count_reads_in_peaks({"A": tmp_path / "A.bam"}, peaks)
    assert counts.loc["p1", "A"] == 3
    assert counts.loc["absent", "A"] == 0   # chrom not in BAM -> 0, no crash


# --------------------------------------------------------------------------
# peak_signal_matrix
# --------------------------------------------------------------------------

def _write_narrowpeak(path, rows):
    """rows: list of (chrom, start, end, signalValue)."""
    with open(path, "w") as fh:
        for i, (c, s, e, sig) in enumerate(rows):
            fh.write(f"{c}\t{s}\t{e}\tpeak_{i}\t100\t.\t{sig}\t5.0\t4.0\t"
                     f"{(e - s) // 2}\n")


def test_peak_signal_matrix_consensus_and_signal(tmp_path):
    # s1 and s2 share an overlapping peak around chr1:1000-1300;
    # s1 has a private peak at chr1:5000; s2 a private one at chr2:200.
    _write_narrowpeak(tmp_path / "s1.narrowPeak", [
        ("chr1", 1000, 1200, 8.0),
        ("chr1", 5000, 5200, 3.0),
    ])
    _write_narrowpeak(tmp_path / "s2.narrowPeak", [
        ("chr1", 1100, 1300, 12.0),
        ("chr2", 200, 400, 6.0),
    ])
    signal, regions = peak_signal_matrix({
        "s1": tmp_path / "s1.narrowPeak",
        "s2": tmp_path / "s2.narrowPeak",
    })
    assert list(signal.columns) == ["s1", "s2"]
    assert len(signal) == len(regions) == 3        # merged shared peak -> 3
    assert (signal.index == regions.index).all()

    # the shared region spans 1000-1300 and carries each sample's signal
    shared = regions.index[(regions.chrom == "chr1") &
                           (regions.start == 1000) & (regions.end == 1300)]
    assert len(shared) == 1
    rid = shared[0]
    assert signal.loc[rid, "s1"] == 8.0
    assert signal.loc[rid, "s2"] == 12.0

    # a region a sample did not call gets 0
    priv2 = regions.index[regions.chrom == "chr2"][0]
    assert signal.loc[priv2, "s1"] == 0.0
    assert signal.loc[priv2, "s2"] == 6.0


def test_peak_signal_matrix_agg_modes(tmp_path):
    # two peaks of s1 both inside one consensus region created by s2's wide peak
    _write_narrowpeak(tmp_path / "wide.narrowPeak", [("chr1", 1000, 2000, 1.0)])
    _write_narrowpeak(tmp_path / "two.narrowPeak", [
        ("chr1", 1050, 1150, 4.0),
        ("chr1", 1700, 1800, 10.0),
    ])
    files = {"wide": tmp_path / "wide.narrowPeak",
             "two": tmp_path / "two.narrowPeak"}
    smax, _ = peak_signal_matrix(files, agg="max")
    ssum, _ = peak_signal_matrix(files, agg="sum")
    assert len(smax) == 1                          # all merged into one region
    assert smax.loc[smax.index[0], "two"] == 10.0
    assert ssum.loc[ssum.index[0], "two"] == 14.0


def _write_macs2_xls(path, rows):
    """rows: list of (chrom, start, end, pileup, fold_enrichment)."""
    with open(path, "w") as fh:
        fh.write("# MACS2 peaks.xls — synthetic\n\n")
        fh.write("chr\tstart\tend\tlength\tabs_summit\tpileup\t"
                 "-log10(pvalue)\tfold_enrichment\t-log10(qvalue)\tname\n")
        for i, (c, s, e, pileup, fe) in enumerate(rows):
            fh.write(f"{c}\t{s}\t{e}\t{e - s}\t{(s + e) // 2}\t{pileup}\t"
                     f"5.0\t{fe}\t4.0\tpeak_{i}\n")


def test_peak_signal_matrix_reads_macs2_xls_pileup(tmp_path):
    """peak_signal_matrix auto-detects MACS2 _peaks.xls and can read the
    pileup (read-coverage) column."""
    _write_macs2_xls(tmp_path / "s1_peaks.xls", [
        ("chr1", 1000, 1200, 40.0, 8.0),
        ("chr1", 5000, 5200, 12.0, 3.0),
    ])
    _write_macs2_xls(tmp_path / "s2_peaks.xls", [
        ("chr1", 1100, 1300, 60.0, 12.0),
    ])
    signal, regions = peak_signal_matrix(
        {"s1": tmp_path / "s1_peaks.xls", "s2": tmp_path / "s2_peaks.xls"},
        value_col="pileup")
    assert list(signal.columns) == ["s1", "s2"]
    shared = regions.index[(regions.chrom == "chr1") &
                           (regions.start == 1000)][0]
    assert signal.loc[shared, "s1"] == 40.0
    assert signal.loc[shared, "s2"] == 60.0


def test_peak_signal_matrix_normalize_equalises_library_size(tmp_path):
    """normalize=True scales every sample's column to the same total."""
    _write_narrowpeak(tmp_path / "deep.narrowPeak", [
        ("chr1", 1000, 1200, 100.0), ("chr2", 1000, 1200, 100.0)])
    _write_narrowpeak(tmp_path / "shallow.narrowPeak", [
        ("chr1", 1000, 1200, 1.0), ("chr2", 1000, 1200, 1.0)])
    sig, _ = peak_signal_matrix(
        {"deep": tmp_path / "deep.narrowPeak",
         "shallow": tmp_path / "shallow.narrowPeak"}, normalize=True)
    assert np.isclose(sig["deep"].sum(), sig["shallow"].sum())
    assert np.isclose(sig["deep"].sum(), 1e6)


def test_peak_signal_matrix_rejects_bad_value_col(tmp_path):
    _write_narrowpeak(tmp_path / "s.narrowPeak", [("chr1", 10, 20, 1.0)])
    with pytest.raises(ValueError, match="value_col"):
        peak_signal_matrix({"s": tmp_path / "s.narrowPeak"},
                           value_col="not_a_column")
