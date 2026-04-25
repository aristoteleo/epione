"""Smoke tests for ``epione.bulk.hic.loops`` and ``pileup`` on a small
synthetic cool with injected dot-like signal.

cooltools' HICCUPS-style dot finder needs reasonably populated matrices
to fire (4-kernel BH-FDR over ``lambda_bin``); we don't assert that any
specific loop is recovered here — we test the API contract (shapes,
column names, dtypes) and that ``pileup`` averages a known feature set
correctly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("cooler")
pytest.importorskip("cooltools")
pytest.importorskip("matplotlib")


def _make_cool_with_dots(tmp_path, seed=0):
    """Synthetic 6-Mb cool with two pairs of strong dots injected.

    Background = uniform cis contacts. On top, inject extra contacts
    between two anchor pairs (dot1 = 1.5/3.5 Mb, dot2 = 2.0/4.5 Mb)
    so the (i, j) cell in the contact matrix lights up.
    """
    import pysam
    import epione as epi

    rng = np.random.default_rng(seed)
    chrom = "chrL"
    size = 6_000_000
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text(f"{chrom}\t{size}\n")

    pairs = tmp_path / "synt.pairs"
    n_bg = 100_000
    n_dot = 6_000
    rid = 0
    dot_anchors = [(1_500_000, 3_500_000), (2_000_000, 4_500_000)]
    anchor_window = 30_000  # ±30 kb around each anchor
    with pairs.open("w") as fh:
        fh.write("## pairs format v1.0\n")
        fh.write(f"#chromsize: {chrom} {size}\n")
        fh.write(
            "#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2\n"
        )
        # Uniform cis background
        for _ in range(n_bg):
            p1 = rng.integers(0, size)
            p2 = rng.integers(0, size)
            if p1 > p2:
                p1, p2 = p2, p1
            fh.write(f"r{rid}\t{chrom}\t{p1}\t{chrom}\t{p2}\t+\t+\n")
            rid += 1
        # Dot anchors
        for a, b in dot_anchors:
            for _ in range(n_dot):
                p1 = a + rng.integers(-anchor_window, anchor_window)
                p2 = b + rng.integers(-anchor_window, anchor_window)
                if p1 > p2:
                    p1, p2 = p2, p1
                fh.write(f"r{rid}\t{chrom}\t{p1}\t{chrom}\t{p2}\t+\t+\n")
                rid += 1

    pairs_gz = tmp_path / "synt.pairs.gz"
    pysam.tabix_compress(str(pairs), str(pairs_gz), force=True)
    cool = tmp_path / "synt.cool"
    epi.upstream.pairs_to_cool(pairs_gz, sizes, cool, binsize=20_000)
    epi.bulk.hic.balance_cool(cool, mad_max=10, min_nnz=1, ignore_diags=0)
    return cool, dot_anchors


def test_loops_returns_bedpe_shape(tmp_path):
    """``loops()`` runs end-to-end and returns a BEDPE-shaped DataFrame
    (may be empty on tiny synthetic data — we test the contract)."""
    import epione as epi
    cool, _ = _make_cool_with_dots(tmp_path)
    df = epi.bulk.hic.loops(
        cool, chromosomes=["chrL"],
        max_loci_separation=5_000_000,
        fdr=0.2,
        clustering_radius=40_000,
    )
    assert isinstance(df, pd.DataFrame)
    for col in ("chrom1", "start1", "end1", "chrom2", "start2", "end2"):
        assert col in df.columns, f"missing BEDPE column {col!r}"
    if len(df):
        # When loops are called, anchors must satisfy chrom1 == chrom2
        # and start1 < start2 (cis only, upper triangle).
        assert (df["chrom1"] == df["chrom2"]).all()
        assert (df["start1"] <= df["start2"]).all()


def test_pileup_returns_square_matrix(tmp_path):
    """``pileup()`` over a 2-feature BEDPE must return a square 2-D
    array of side ``2*flank/binsize + 1``."""
    import epione as epi
    cool, anchors = _make_cool_with_dots(tmp_path)
    flank = 100_000
    binsize = 20_000
    feats = pd.DataFrame({
        "chrom1": ["chrL"] * len(anchors),
        "start1": [a[0] for a in anchors],
        "end1":   [a[0] + binsize for a in anchors],
        "chrom2": ["chrL"] * len(anchors),
        "start2": [a[1] for a in anchors],
        "end2":   [a[1] + binsize for a in anchors],
    })
    apa = epi.bulk.hic.pileup(
        cool, feats, flank=flank, chromosomes=["chrL"],
    )
    assert apa.ndim == 2
    assert apa.shape[0] == apa.shape[1]
    expected_side = (2 * flank) // binsize + 1
    assert apa.shape[0] == expected_side, (
        f"expected side {expected_side}, got {apa.shape[0]}"
    )
    finite = apa[np.isfinite(apa)]
    assert finite.size > 0


def test_plot_apa_renders(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import epione as epi
    cool, anchors = _make_cool_with_dots(tmp_path)
    feats = pd.DataFrame({
        "chrom1": ["chrL"] * len(anchors),
        "start1": [a[0] for a in anchors],
        "end1":   [a[0] + 20_000 for a in anchors],
        "chrom2": ["chrL"] * len(anchors),
        "start2": [a[1] for a in anchors],
        "end2":   [a[1] + 20_000 for a in anchors],
    })
    apa = epi.bulk.hic.pileup(
        cool, feats, flank=80_000, chromosomes=["chrL"],
    )
    fig, ax, img = epi.pl.plot_apa(
        apa, flank=80_000, binsize=20_000, figsize=(3.5, 3.2),
    )
    assert fig is not None
    plt.close(fig)


def test_plot_loops_renders_without_calls(tmp_path):
    """``plot_loops`` should render fine even if ``loops_df`` is empty
    (no dots called) — useful when a region is loop-free."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import epione as epi
    cool, _ = _make_cool_with_dots(tmp_path)
    empty = pd.DataFrame(columns=[
        "chrom1", "start1", "end1", "chrom2", "start2", "end2",
    ])
    fig, ax = epi.pl.plot_loops(
        cool, region="chrL:1,000,000-5,000,000",
        loops_df=empty, figsize=(4, 3.6),
    )
    assert fig is not None
    plt.close(fig)
