"""Tests for epione.bulk.bigwig — compute_matrix / compute_matrix_region.

Writes a tiny synthetic bigwig via pyBigWig.addEntries, then checks
that the signal-extraction routines see the signal we put in.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("pyBigWig")


def _write_synthetic_bigwig(path, chrom_size=200_000, seed=0):
    import pyBigWig
    rng = np.random.default_rng(seed)
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([("chr1", chrom_size)])
    # Three peaks at 10k, 50k, 100k — height 10 each, width 1 kb.
    # Flanking baseline = 1.0 everywhere else.
    step = 100
    starts = np.arange(0, chrom_size, step, dtype=np.int64)
    ends = starts + step
    vals = np.ones(len(starts), dtype=np.float32)
    for centre in (10_000, 50_000, 100_000):
        in_peak = (starts >= centre - 500) & (starts < centre + 500)
        vals[in_peak] = 10.0
    # Add small uniform noise so the test isn't trivially zeroed.
    vals += rng.uniform(0, 0.1, len(vals)).astype(np.float32)
    bw.addEntries(["chr1"] * len(starts), starts.tolist(),
                   ends=ends.tolist(), values=vals.tolist())
    bw.close()


def _tiny_gtf_df():
    """A minimal GTF-ish DataFrame with 3 'genes', one per synthetic peak."""
    return pd.DataFrame({
        "seqname": ["chr1"] * 3,
        "start":   [9_500, 49_500, 99_500],
        "end":     [10_500, 50_500, 100_500],
        "strand":  ["+", "+", "-"],
        "gene_id": ["A", "B", "C"],
        "feature": ["transcript"] * 3,
    })


def test_compute_matrix_region_extracts_peak_signal(tmp_path):
    """Peak regions should have mean signal >> baseline (~1.0)."""
    import epione.bulk as ebk

    bw_path = tmp_path / "toy.bw"
    _write_synthetic_bigwig(bw_path)
    obj = ebk.bigwig({"toy": str(bw_path)})
    obj.read()

    gtf = _tiny_gtf_df()
    # anchor='center' + small flanks → each 'gene' is centred on a peak.
    ad = obj.compute_matrix_region(
        "toy", region=gtf,
        nbins=10, upstream=500, downstream=500,
        anchor="center", sort=False, n_jobs=1,
    )
    assert ad.shape == (3, 10)
    X = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)
    # Each row's max bin should be near 10 (planted peak height).
    max_per_row = X.max(axis=1)
    assert np.all(max_per_row > 5.0), f"max per row: {max_per_row}"


def test_compute_matrix_vs_compute_matrix_region_equivalence(tmp_path):
    """For symmetric upstream==downstream windows and anchor='5p',
    compute_matrix_region should reproduce compute_matrix."""
    import epione.bulk as ebk

    bw_path = tmp_path / "toy.bw"
    _write_synthetic_bigwig(bw_path)
    obj = ebk.bigwig({"toy": str(bw_path)})
    obj.read()
    obj.gtf = _tiny_gtf_df()
    obj.gtf["gene_name"] = obj.gtf["gene_id"]

    # compute_matrix returns a (tss, tes, body) tuple of AnnData. We
    # compare the tss block against compute_matrix_region(anchor='5p').
    ad_tss, _ad_tes, _ad_body = obj.compute_matrix(
        "toy", nbins=10, upstream=500, downstream=500,
    )
    ad_cm = ad_tss
    ad_reg = obj.compute_matrix_region(
        "toy", region=obj.gtf,
        nbins=10, upstream=500, downstream=500,
        anchor="5p", sort=False, n_jobs=1,
    )
    # Match by gene_id (order might differ between the two paths).
    common = sorted(set(ad_cm.obs_names) & set(ad_reg.obs_names))
    assert len(common) >= 1
    a = ad_cm[common].X
    b = ad_reg[common].X
    a = a.toarray() if hasattr(a, "toarray") else np.asarray(a)
    b = b.toarray() if hasattr(b, "toarray") else np.asarray(b)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4,
                                err_msg="compute_matrix vs compute_matrix_region disagree")


def test_compute_matrix_region_multi_anchor_returns_dict(tmp_path):
    import epione.bulk as ebk

    bw_path = tmp_path / "toy.bw"
    _write_synthetic_bigwig(bw_path)
    obj = ebk.bigwig({"toy": str(bw_path)})
    obj.read()
    gtf = _tiny_gtf_df()

    out = obj.compute_matrix_region(
        "toy", region=gtf,
        nbins=8, upstream=400, downstream=400,
        anchor=["5p", "3p", "center"], sort=False, n_jobs=1,
    )
    assert isinstance(out, dict)
    assert set(out.keys()) == {"5p", "3p", "center"}
    for anchor, ad in out.items():
        assert ad.shape == (3, 8), f"{anchor} shape={ad.shape}"
