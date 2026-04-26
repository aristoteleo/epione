"""Tests for ``epione.single.hic`` Phase 2 — Droplet Hi-C demux,
pseudobulk, and celltype × celltype correlation.

Builds a synthetic multi-cell pairs file where each line encodes its
cell barcode in the readID suffix (Chang 2024 convention). Three
"celltypes" with deliberately different intra-chromosomal contact
profiles are simulated; demux + pseudobulk should recover them and
:func:`cluster_correlation` should rank within-celltype replicates
above between-celltype.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("cooler")
pytest.importorskip("cooltools")
pytest.importorskip("matplotlib")
pytest.importorskip("pysam")


def _write_synthetic_droplet_pairs(
    tmp_path: Path,
    seed: int = 0,
):
    """Write a Droplet-Hi-C-style multi-cell pairs file.

    Three celltypes A, B, C; two cells per celltype (6 cells total).
    Each celltype emits contacts biased toward a different anchor
    region on the same chromosome — so per-celltype pseudobulk maps
    differ in a way correlation should resolve.
    """
    import pysam

    rng = np.random.default_rng(seed)
    chrom = "chr1"
    size = 4_000_000
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text(f"{chrom}\t{size}\n")

    # Two cells per celltype, three celltypes. Barcodes are single
    # underscore-free tokens so the default ``_``-split barcode
    # extractor pulls them off readID cleanly.
    cells = {
        "Acell1": "A", "Acell2": "A",
        "Bcell1": "B", "Bcell2": "B",
        "Ccell1": "C", "Ccell2": "C",
    }
    # Celltype-specific anchor windows on chr1
    anchors = {
        "A": (   200_000, 1_000_000),
        "B": ( 1_500_000, 2_300_000),
        "C": ( 2_800_000, 3_700_000),
    }
    n_per_cell = 4_000

    pairs_plain = tmp_path / "droplet.pairs"
    rid = 0
    with pairs_plain.open("w") as fh:
        fh.write("## pairs format v1.0\n")
        fh.write(f"#chromsize: {chrom} {size}\n")
        fh.write(
            "#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2\n"
        )
        for cell, ct in cells.items():
            anc_lo, anc_hi = anchors[ct]
            for _ in range(n_per_cell):
                # Half within celltype anchor, half uniform background.
                if rng.random() < 0.7:
                    p1 = rng.integers(anc_lo, anc_hi)
                    p2 = rng.integers(anc_lo, anc_hi)
                else:
                    p1 = rng.integers(0, size)
                    p2 = rng.integers(0, size)
                if p1 > p2:
                    p1, p2 = p2, p1
                # readID = "cellname_BARCODE" — Chang convention; we
                # use the cell name as the "barcode" string here.
                read_id = f"r{rid}_{cell}"
                fh.write(
                    f"{read_id}\t{chrom}\t{p1}\t{chrom}\t{p2}\t+\t+\n"
                )
                rid += 1

    pairs_gz = tmp_path / "droplet.pairs.gz"
    pysam.tabix_compress(str(pairs_plain), str(pairs_gz), force=True)

    barcode_to_celltype = {cell: ct for cell, ct in cells.items()}
    return pairs_gz, sizes, barcode_to_celltype, cells


def test_demux_pairs_by_barcode_round_trips(tmp_path):
    """Every input line must end up in exactly one output file."""
    import epione as epi
    pairs, sizes, bc2ct, cells = _write_synthetic_droplet_pairs(tmp_path)

    out_dir = tmp_path / "demux"
    paths = epi.single.hic.demux_pairs_by_barcode(
        pairs, bc2ct, out_dir,
    )
    # 3 celltypes
    assert set(paths.keys()) == {"A", "B", "C"}
    for p in paths.values():
        assert p.exists() and str(p).endswith(".pairs.gz")

    # Body line count across outputs == input body line count
    def _body_count(p):
        n = 0
        with gzip.open(p, "rt") as fh:
            for line in fh:
                if not line.startswith("#"):
                    n += 1
        return n

    in_count = _body_count(pairs)
    out_count = sum(_body_count(p) for p in paths.values())
    assert out_count == in_count, (
        f"line count mismatch: in={in_count}, out={out_count}"
    )


def test_demux_drops_unassigned_barcodes(tmp_path):
    """Reads whose barcode isn't in the mapping must be dropped (default)."""
    import epione as epi
    pairs, sizes, bc2ct, cells = _write_synthetic_droplet_pairs(tmp_path)

    # Map only celltype A cells; everything else should be dropped.
    partial = {bc: ct for bc, ct in bc2ct.items() if ct == "A"}
    out_dir = tmp_path / "demux_partial"
    paths = epi.single.hic.demux_pairs_by_barcode(
        pairs, partial, out_dir, drop_unassigned=True,
    )
    assert set(paths.keys()) == {"A"}


def test_pseudobulk_by_celltype_builds_balanced_cools(tmp_path):
    """End-to-end: 3 celltypes → 3 balanced cools."""
    import cooler
    import epione as epi

    pairs, sizes, bc2ct, _ = _write_synthetic_droplet_pairs(tmp_path)
    out_dir = tmp_path / "pseudobulk"
    cools = epi.single.hic.pseudobulk_by_celltype(
        pairs, bc2ct, sizes, out_dir, binsize=100_000, balance=True,
    )
    assert set(cools.keys()) == {"A", "B", "C"}
    for ct, c in cools.items():
        clr = cooler.Cooler(str(c))
        assert "weight" in clr.bins().columns, (
            f"{ct}: cool was not ICE-balanced"
        )
        # And pixel count > 0 — the demuxed pairs landed.
        assert len(clr.pixels()[:]) > 0


def test_cluster_correlation_recovers_celltype_structure(tmp_path):
    """Within-celltype replicates should correlate higher than between."""
    import epione as epi

    pairs, sizes, bc2ct, _ = _write_synthetic_droplet_pairs(tmp_path)

    # Replicate split: cell1 vs cell2 of each celltype get separate cools
    # so the resulting 6×6 correlation matrix should show A1-A2, B1-B2,
    # C1-C2 as the strongest off-diagonal pairs.
    bc2rep = {cell: f"{ct}rep{cell[-1]}" for cell, ct in bc2ct.items()}
    # bc2rep maps each barcode to an own group label (one cell each)
    # so the 6 groups are A_cell1, A_cell2, B_cell1, ...

    out_dir = tmp_path / "rep"
    cools = epi.single.hic.pseudobulk_by_celltype(
        pairs, bc2rep, sizes, out_dir, binsize=100_000, balance=True,
    )
    # Drop empty bin combos: ensure every cool has > 0 contacts
    assert len(cools) == 6

    corr, names = epi.single.hic.cluster_correlation(
        cools, chromosomes=["chr1"], balance=True, log=True,
    )
    assert corr.shape == (6, 6)
    assert list(corr.index) == names
    # Diagonal should be ~1
    assert np.allclose(np.diag(corr.to_numpy()), 1.0, atol=1e-6)

    M = corr.to_numpy()
    # Group label format is ``"{celltype}rep{N}"`` (e.g. "Arep1");
    # the celltype letter is the leading character.
    cts = [n[0] for n in names]
    within: list[float] = []
    between: list[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            (within if cts[i] == cts[j] else between).append(M[i, j])
    assert np.mean(within) > np.mean(between), (
        f"within-celltype mean r ({np.mean(within):.3f}) should exceed "
        f"between-celltype mean r ({np.mean(between):.3f})"
    )


def test_plot_correlation_heatmap_renders(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import epione as epi

    # Build a tiny 4×4 correlation matrix manually
    M = np.array([
        [1.0, 0.85, 0.20, 0.18],
        [0.85, 1.0, 0.22, 0.21],
        [0.20, 0.22, 1.0, 0.78],
        [0.18, 0.21, 0.78, 1.0],
    ])
    df = pd.DataFrame(M, index=["A1", "A2", "B1", "B2"],
                      columns=["A1", "A2", "B1", "B2"])

    fig, ax, img = epi.pl.plot_correlation_heatmap(df, figsize=(3.5, 3.0))
    assert fig is not None
    plt.close(fig)
