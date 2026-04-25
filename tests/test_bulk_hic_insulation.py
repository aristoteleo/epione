"""Smoke tests for ``epione.bulk.hic.insulation`` / ``tad_boundaries``
on a synthetic 4-Mb cool with three injected TAD-like blocks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("cooler")
pytest.importorskip("cooltools")
pytest.importorskip("matplotlib")


def _make_cool_with_tads(tmp_path, seed=0):
    """Synthetic 4-Mb single-chrom cool with three TAD blocks injected.

    Within-block contacts fire ~10x more often than between-block, so
    insulation troughs should appear around the two block boundaries.
    """
    import pysam
    import epione as epi

    rng = np.random.default_rng(seed)
    chrom = "chrTAD"
    size = 4_000_000
    block_edges = [0, 1_300_000, 2_600_000, size]  # 3 blocks
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text(f"{chrom}\t{size}\n")

    pairs = tmp_path / "synt.pairs"
    n_within_per_block = 25_000
    n_between = 4_000
    rid = 0
    with pairs.open("w") as fh:
        fh.write("## pairs format v1.0\n")
        fh.write(f"#chromsize: {chrom} {size}\n")
        fh.write(
            "#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2\n"
        )
        for blk_start, blk_end in zip(block_edges[:-1], block_edges[1:]):
            for _ in range(n_within_per_block):
                p1 = rng.integers(blk_start, blk_end)
                p2 = rng.integers(blk_start, blk_end)
                if p1 > p2:
                    p1, p2 = p2, p1
                fh.write(f"r{rid}\t{chrom}\t{p1}\t{chrom}\t{p2}\t+\t+\n")
                rid += 1
        # Sparse between-block background (uniform)
        for _ in range(n_between):
            p1 = rng.integers(0, size)
            p2 = rng.integers(0, size)
            if p1 > p2:
                p1, p2 = p2, p1
            fh.write(f"r{rid}\t{chrom}\t{p1}\t{chrom}\t{p2}\t+\t+\n")
            rid += 1

    pairs_gz = tmp_path / "synt.pairs.gz"
    pysam.tabix_compress(str(pairs), str(pairs_gz), force=True)
    cool = tmp_path / "synt.cool"
    epi.upstream.pairs_to_cool(pairs_gz, sizes, cool, binsize=20_000)
    epi.bulk.hic.balance_cool(cool, mad_max=10, min_nnz=1, ignore_diags=0)
    return cool, block_edges


def test_insulation_returns_per_bin_score(tmp_path):
    import epione as epi
    cool, _ = _make_cool_with_tads(tmp_path)
    df = epi.bulk.hic.insulation(
        cool, window_bp=200_000, chromosomes=["chrTAD"],
        ignore_diags=0,
    )
    score_cols = [c for c in df.columns
                  if c.startswith("log2_insulation_score_")]
    assert len(score_cols) == 1
    assert (df["chrom"] == "chrTAD").all()
    finite_score = df[score_cols[0]].dropna()
    assert finite_score.size > 0


def test_tad_boundaries_recovers_block_edges(tmp_path):
    """Injected boundaries at 1.3 Mb / 2.6 Mb should be among the
    strongest insulation dips."""
    import epione as epi
    cool, edges = _make_cool_with_tads(tmp_path)
    df = epi.bulk.hic.insulation(
        cool, window_bp=200_000, chromosomes=["chrTAD"],
        ignore_diags=0,
    )
    boundaries = epi.bulk.hic.tad_boundaries(df, window_bp=200_000)
    assert len(boundaries) > 0
    # Sort by strength and pick top 5
    top = boundaries.sort_values(
        "boundary_strength", ascending=False
    ).head(5)
    top_centres = ((top["start"] + top["end"]) / 2).to_numpy()
    # Each injected edge (1.3 Mb, 2.6 Mb) must have a top-5 boundary
    # within ±400 kb (synthetic data has ~one bin worth of jitter).
    for edge in (1_300_000, 2_600_000):
        nearest = np.min(np.abs(top_centres - edge))
        assert nearest < 400_000, (
            f"no top-5 boundary within 400 kb of injected edge {edge}: "
            f"top centres = {top_centres.tolist()}"
        )


def test_plot_insulation_renders(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import epione as epi
    cool, _ = _make_cool_with_tads(tmp_path)
    df = epi.bulk.hic.insulation(
        cool, window_bp=200_000, chromosomes=["chrTAD"],
        ignore_diags=0,
    )
    fig, ax = epi.pl.plot_insulation(
        df, chromosome="chrTAD", window_bp=200_000,
        figsize=(6, 1.5),
    )
    assert fig is not None
    plt.close(fig)
