"""Unit test extension for compartments + saddle."""
import numpy as np
import pandas as pd
import pytest


pytest.importorskip("cooler")
pytest.importorskip("cooltools")
pytest.importorskip("matplotlib")


def _make_cool_with_compartment_signal(tmp_path, seed=0):
    """Synthetic 4-Mb cool with 2 chromosomes and a clear A/B pattern.

    Within each chrom: bins 0..n/2 belong to "A", bins n/2..n belong to
    "B". Inject extra contacts within-A and within-B (compartment-like
    blocks), few between A and B. Should give a strong compartment
    eigenvector.
    """
    import pysam
    import epione as epi

    rng = np.random.default_rng(seed)
    chroms = [("chrA", 4_000_000), ("chrB", 4_000_000)]
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text("\n".join(f"{c}\t{s}" for c, s in chroms) + "\n")

    pairs = tmp_path / "synt.pairs"
    n_per = 60_000
    with pairs.open("w") as fh:
        fh.write("## pairs format v1.0\n")
        for c, s in chroms:
            fh.write(f"#chromsize: {c} {s}\n")
        fh.write("#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2\n")
        rid = 0
        for c, s in chroms:
            mid = s // 2
            # Within-A
            for _ in range(n_per // 3):
                p1 = rng.integers(0, mid); p2 = rng.integers(0, mid)
                if p1 > p2: p1, p2 = p2, p1
                fh.write(f"r{rid}\t{c}\t{p1}\t{c}\t{p2}\t+\t+\n"); rid+=1
            # Within-B
            for _ in range(n_per // 3):
                p1 = rng.integers(mid, s); p2 = rng.integers(mid, s)
                if p1 > p2: p1, p2 = p2, p1
                fh.write(f"r{rid}\t{c}\t{p1}\t{c}\t{p2}\t+\t+\n"); rid+=1
            # Between A and B (rare)
            for _ in range(n_per // 30):
                p1 = rng.integers(0, mid); p2 = rng.integers(mid, s)
                fh.write(f"r{rid}\t{c}\t{p1}\t{c}\t{p2}\t+\t+\n"); rid+=1
    pairs_gz = tmp_path / "synt.pairs.gz"
    pysam.tabix_compress(str(pairs), str(pairs_gz), force=True)
    cool = tmp_path / "synt.cool"
    epi.upstream.pairs_to_cool(pairs_gz, sizes, cool, binsize=100_000)
    epi.bulk.hic.balance_cool(cool, mad_max=10, min_nnz=1, ignore_diags=0)
    return cool


def test_compartments_recovers_AB_pattern(tmp_path):
    import epione as epi
    cool = _make_cool_with_compartment_signal(tmp_path)
    eig = epi.bulk.hic.compartments(
        cool, chromosomes=["chrA", "chrB"], n_eigs=2,
    )
    assert {"chrom", "start", "end", "E1", "E2"}.issubset(eig.columns)
    # E1 should change sign roughly halfway through each chromosome.
    for ch, sub in eig.groupby("chrom"):
        sub = sub.dropna(subset=["E1"]).reset_index(drop=True)
        first_half = sub.iloc[: len(sub)//2]["E1"].mean()
        second_half = sub.iloc[len(sub)//2:]["E1"].mean()
        assert (first_half * second_half) < 0, (
            f"{ch}: halves should have opposite-sign E1 means; "
            f"got {first_half:.2f} / {second_half:.2f}"
        )


def test_saddle_returns_strong_diagonal(tmp_path):
    import epione as epi
    cool = _make_cool_with_compartment_signal(tmp_path)
    eig = epi.bulk.hic.compartments(cool, chromosomes=["chrA","chrB"], n_eigs=1)
    sad, edges, count = epi.bulk.hic.saddle(
        cool, eig, n_bins=10, chromosomes=["chrA", "chrB"],
    )
    # cooltools.saddle pads with two tail bins (below-qrange,
    # above-qrange) so the matrix is ``(n_bins + 2) ** 2``.
    assert sad.shape == (12, 12)
    finite = sad[np.isfinite(sad)]
    assert finite.size > 0
    # Diagonal corners (AA + BB) should average above off-diagonal corners.
    # Skip the outer tail rows / cols (index 0 and -1).
    aa = sad[-4:-1, -4:-1].mean()
    bb = sad[1:4, 1:4].mean()
    ab = sad[-4:-1, 1:4].mean()
    ba = sad[1:4, -4:-1].mean()
    assert (aa + bb) > (ab + ba), (
        f"saddle should have stronger AA+BB than AB+BA: "
        f"AA={aa:.2f} BB={bb:.2f} AB={ab:.2f} BA={ba:.2f}"
    )


def test_plot_compartments_renders(tmp_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import epione as epi
    cool = _make_cool_with_compartment_signal(tmp_path)
    eig = epi.bulk.hic.compartments(cool, chromosomes=["chrA"], n_eigs=1)
    fig, ax = epi.pl.plot_compartments(eig, chromosome="chrA", figsize=(5, 1.5))
    assert fig is not None
    plt.close(fig)


def test_plot_saddle_renders(tmp_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import epione as epi
    cool = _make_cool_with_compartment_signal(tmp_path)
    eig = epi.bulk.hic.compartments(cool, chromosomes=["chrA","chrB"], n_eigs=1)
    sad, edges, count = epi.bulk.hic.saddle(
        cool, eig, n_bins=8, chromosomes=["chrA","chrB"],
    )
    fig, ax, img = epi.pl.plot_saddle(sad, edges, figsize=(3.5, 3.2))
    assert fig is not None
    plt.close(fig)
