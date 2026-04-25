"""Smoke tests for the bulk Hi-C pipeline (v0.4 layout):

  * :func:`epione.upstream.pairs_to_cool`
  * :func:`epione.bulk.hic.balance_cool`
  * :func:`epione.bulk.hic.plot_contact_matrix` /
    :func:`epione.bulk.hic.plot_decay_curve` /
    :func:`epione.bulk.hic.plot_coverage`

Skips :func:`epione.upstream.pairs_from_bam` because it requires a
real Hi-C BAM. Everything else works on a synthetic 4DN-format
``.pairs`` with a loop-like diagonal motif.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("cooler")
pytest.importorskip("matplotlib")


def _make_synthetic_pairs(tmp_path, n_pairs=20_000, seed=0):
    """Write a 4DN-compatible .pairs file + chrom.sizes for a 2-Mb genome.

    Contacts follow a power-law decay with distance so the ICE balance
    + log-scale heatmap pick up structure rather than noise.
    """
    rng = np.random.default_rng(seed)
    chrom = "chr_synt"
    size = 2_000_000
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text(f"{chrom}\t{size}\n")

    # Power-law interaction distances (median ~10 kb).
    d = rng.exponential(10_000, n_pairs).astype(int)
    pos1 = rng.integers(0, size - d.max() - 1, n_pairs)
    pos2 = pos1 + d
    # Keep within chromosome.
    mask = pos2 < size
    pos1, pos2 = pos1[mask], pos2[mask]

    pairs = tmp_path / "synt.pairs"
    with pairs.open("w") as fh:
        fh.write("## pairs format v1.0\n")
        fh.write(f"#chromsize: {chrom} {size}\n")
        fh.write("#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2\n")
        for i, (p1, p2) in enumerate(zip(pos1, pos2)):
            fh.write(f"r{i}\t{chrom}\t{p1}\t{chrom}\t{p2}\t+\t+\n")
    # bgzip for cooler cload pairs (pysam is already an epione dep, so
    # we avoid needing the bgzip CLI on PATH).
    import pysam
    pairs_gz = tmp_path / "synt.pairs.gz"
    pysam.tabix_compress(str(pairs), str(pairs_gz), force=True)
    return pairs_gz, sizes


def test_pairs_to_cool_builds_matrix(tmp_path):
    import epione as epi

    pairs_gz, sizes = _make_synthetic_pairs(tmp_path)
    out_cool = tmp_path / "synt.cool"
    res = epi.upstream.pairs_to_cool(
        pairs_path=pairs_gz, chrom_sizes=sizes,
        out_cool=out_cool, binsize=50_000,
    )
    assert res == out_cool
    assert out_cool.exists()

    import cooler
    clr = cooler.Cooler(str(out_cool))
    # 2 Mb / 50 kb = 40 bins
    assert clr.bins()[:].shape[0] == 40
    mat = clr.matrix(balance=False).fetch("chr_synt")
    assert mat.sum() > 0


def test_balance_cool_writes_weight_column(tmp_path):
    import epione as epi

    pairs_gz, sizes = _make_synthetic_pairs(tmp_path)
    out_cool = tmp_path / "synt.cool"
    epi.upstream.pairs_to_cool(pairs_gz, sizes, out_cool, binsize=50_000)

    stats = epi.bulk.hic.balance_cool(
        out_cool, mad_max=10, min_nnz=1, ignore_diags=0, max_iters=50,
    )
    assert "converged" in stats
    assert "n_masked" in stats

    import cooler
    clr = cooler.Cooler(str(out_cool))
    bins = clr.bins()[:]
    assert "weight" in bins.columns
    balanced = clr.matrix(balance=True).fetch("chr_synt")
    # Balanced matrix has NaNs where weights are NaN — ensure some finite entries.
    finite = balanced[np.isfinite(balanced)]
    assert finite.size > 0


def test_balance_cool_idempotent(tmp_path):
    import epione as epi

    pairs_gz, sizes = _make_synthetic_pairs(tmp_path)
    out_cool = tmp_path / "synt.cool"
    epi.upstream.pairs_to_cool(pairs_gz, sizes, out_cool, binsize=50_000)

    epi.bulk.hic.balance_cool(out_cool, mad_max=10, min_nnz=1, ignore_diags=0)
    res2 = epi.bulk.hic.balance_cool(out_cool, mad_max=10, min_nnz=1, ignore_diags=0)
    assert res2["reused_existing"]


def test_plot_decay_curve(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import epione as epi

    pairs_gz, sizes = _make_synthetic_pairs(tmp_path)
    out_cool = tmp_path / "synt.cool"
    epi.upstream.pairs_to_cool(pairs_gz, sizes, out_cool, binsize=50_000)

    fig, ax, df = epi.bulk.hic.plot_decay_curve(
        out_cool, balance=False, figsize=(5, 3),
    )
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    # Power-law decay in synthetic data should give monotone-ish decline.
    assert df["mean_contact"].iloc[0] >= df["mean_contact"].iloc[-1]
    plt.close(fig)


def test_plot_coverage_two_panels_after_balance(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import epione as epi

    pairs_gz, sizes = _make_synthetic_pairs(tmp_path)
    out_cool = tmp_path / "synt.cool"
    epi.upstream.pairs_to_cool(pairs_gz, sizes, out_cool, binsize=50_000)
    epi.bulk.hic.balance_cool(out_cool, mad_max=10, min_nnz=1, ignore_diags=0)

    fig, axes = epi.bulk.hic.plot_coverage(out_cool, balance=False, figsize=(8, 3))
    assert len(axes) == 2  # coverage + weight panels after balance
    plt.close(fig)


def test_plot_contact_matrix_renders(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import epione as epi

    pairs_gz, sizes = _make_synthetic_pairs(tmp_path)
    out_cool = tmp_path / "synt.cool"
    epi.upstream.pairs_to_cool(pairs_gz, sizes, out_cool, binsize=50_000)

    fig, ax, img = epi.bulk.hic.plot_contact_matrix(
        out_cool, region="chr_synt", balance=False,
        figsize=(4, 3.5), title="synt",
    )
    assert fig is not None and ax is not None and img is not None
    assert ax.get_title() == "synt"
    plt.close(fig)
