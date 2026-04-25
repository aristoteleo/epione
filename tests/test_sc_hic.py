"""Smoke tests for epione.sc_hic — load → impute → embed end-to-end on
a synthetic 8-cell collection.

Each cell is a tiny .cool built from random 4DN-format pairs, so no
real Hi-C aligner / dataset is needed. Cells are split into two
groups with subtly different contact-decay distributions so that PCA
on the imputed matrices recovers a 2-component separation — which
exercises the full pipeline (load + impute + flatten + PCA).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("cooler")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")


def _make_cell_cool(out_dir: Path, cell_id: str, *,
                    n_pairs: int = 4_000, decay_scale: int = 10_000,
                    seed: int = 0):
    """Bin synthetic 4DN pairs for one cell into a 50-kb-resolution cool.

    Two-chromosome genome, 1.5 Mb each — enough bins for the
    convolution + RWR steps to actually do something, small enough for
    tests to run in <1 s per cell.
    """
    import pysam
    import epione as epi

    rng = np.random.default_rng(seed)
    chroms = [("chr_a", 1_500_000), ("chr_b", 1_500_000)]
    sizes_path = out_dir / f"{cell_id}.sizes"
    sizes_path.write_text("\n".join(f"{c}\t{s}" for c, s in chroms) + "\n")

    pairs_path = out_dir / f"{cell_id}.pairs"
    with pairs_path.open("w") as fh:
        fh.write("## pairs format v1.0\n")
        for c, s in chroms:
            fh.write(f"#chromsize: {c} {s}\n")
        fh.write(
            "#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2\n"
        )
        n_per_chrom = n_pairs // 2
        rid = 0
        for c, s in chroms:
            d = rng.exponential(decay_scale, n_per_chrom).astype(int)
            p1 = rng.integers(0, s - d.max() - 1, n_per_chrom)
            p2 = p1 + d
            mask = p2 < s
            for x, y in zip(p1[mask], p2[mask]):
                fh.write(f"r{rid}\t{c}\t{x}\t{c}\t{y}\t+\t+\n")
                rid += 1

    pairs_gz = out_dir / f"{cell_id}.pairs.gz"
    pysam.tabix_compress(str(pairs_path), str(pairs_gz), force=True)

    cool_path = out_dir / f"{cell_id}.cool"
    epi.hic.pairs_to_cool(
        pairs_path=pairs_gz, chrom_sizes=sizes_path,
        out_cool=cool_path, binsize=50_000,
    )
    return cool_path


def _build_collection(tmp_path: Path, n_per_group: int = 4):
    """Build two groups of cells with different decay scales — group A
    (short-range) vs group B (long-range) — and return their cool paths
    + obs metadata. The decay difference shows up in the imputed matrix
    distribution and should drive a leading PCA component.
    """
    cool_paths = []
    cell_ids = []
    groups = []
    seed = 0
    for grp_label, decay_scale in (("A", 8_000), ("B", 60_000)):
        for k in range(n_per_group):
            cid = f"{grp_label}_{k:02d}"
            cool_paths.append(_make_cell_cool(
                tmp_path, cid, decay_scale=decay_scale, seed=seed,
            ))
            cell_ids.append(cid)
            groups.append(grp_label)
            seed += 1
    obs = pd.DataFrame({"group": groups})
    return cool_paths, cell_ids, obs


def test_impute_cell_chromosome_pure_function():
    """Smoke test: a sparse symmetric input becomes a smooth, mostly-sparse
    output with non-negative values and (approximately) symmetric.
    """
    import epione as epi

    rng = np.random.default_rng(42)
    n = 30
    C = np.zeros((n, n))
    # Sprinkle ~30 contacts.
    rows = rng.integers(0, n, 30)
    cols = rng.integers(0, n, 30)
    for i, j in zip(rows, cols):
        C[i, j] += 1
        C[j, i] += 1

    P = epi.sc_hic.impute_cell_chromosome(
        C, pad=1, rwr_alpha=0.05, top_pct=0.1,
    )
    assert P.shape == (n, n)
    assert np.allclose(P, P.T, atol=1e-6)  # symmetric
    assert (P >= 0).all()
    # top-k (10%) cuts ~90% to zero.
    nnz = np.count_nonzero(P)
    assert nnz <= n * n * 0.2  # generous upper bound after symmetrise


def test_load_cool_collection_indexes_cells(tmp_path):
    import epione as epi

    cool_paths, cell_ids, obs = _build_collection(tmp_path, n_per_group=2)
    adata = epi.sc_hic.load_cool_collection(
        cool_paths, cell_ids=cell_ids, obs=obs,
    )
    assert adata.n_obs == 4
    assert "cool_path" in adata.obs.columns
    assert "group" in adata.obs.columns
    assert adata.uns["hic"]["resolution"] == 50_000
    # Both chromosomes propagated.
    assert set(adata.uns["hic"]["chromosomes"]) == {"chr_a", "chr_b"}
    n_bins = adata.uns["hic"]["n_chrom_bins"]
    assert n_bins["chr_a"] == 30  # 1.5 Mb / 50 kb
    assert n_bins["chr_b"] == 30


def test_impute_cells_writes_per_cell_npz(tmp_path):
    import epione as epi

    cool_paths, cell_ids, obs = _build_collection(tmp_path, n_per_group=2)
    adata = epi.sc_hic.load_cool_collection(
        cool_paths, cell_ids=cell_ids, obs=obs,
    )
    out_dir = tmp_path / "imputed"
    epi.sc_hic.impute_cells(
        adata, out_dir=out_dir,
        pad=1, rwr_alpha=0.05, top_pct=0.1,
        progress=False,
    )
    assert adata.uns["hic"]["imputed_dir"] == str(out_dir)
    for cid in cell_ids:
        npz = out_dir / f"{cid}.npz"
        assert npz.exists()
        z = np.load(npz)
        assert set(z.files) == {"chr_a", "chr_b"}
        assert z["chr_a"].shape == (30, 30)


def test_impute_cells_skips_existing(tmp_path):
    import epione as epi

    cool_paths, cell_ids, obs = _build_collection(tmp_path, n_per_group=2)
    adata = epi.sc_hic.load_cool_collection(
        cool_paths, cell_ids=cell_ids, obs=obs,
    )
    out_dir = tmp_path / "imputed"
    epi.sc_hic.impute_cells(adata, out_dir=out_dir, progress=False)
    # Touch an .npz with junk content; if overwrite=False it must not
    # be touched.
    junk = out_dir / f"{cell_ids[0]}.npz"
    junk.write_bytes(b"")
    epi.sc_hic.impute_cells(
        adata, out_dir=out_dir, overwrite=False, progress=False,
    )
    assert junk.read_bytes() == b""


def test_embedding_separates_two_groups(tmp_path):
    """End-to-end: load → impute → embed and check that PC1 correlates
    with the group label. With group A (short-range) vs group B
    (long-range) decay this should be near-perfect even at 8 cells.
    """
    import epione as epi

    cool_paths, cell_ids, obs = _build_collection(tmp_path, n_per_group=4)
    adata = epi.sc_hic.load_cool_collection(
        cool_paths, cell_ids=cell_ids, obs=obs,
    )
    epi.sc_hic.impute_cells(
        adata, out_dir=tmp_path / "imputed",
        pad=1, rwr_alpha=0.05, top_pct=0.1, progress=False,
    )
    new = epi.sc_hic.embedding(
        adata, n_components=4, standardise=True,
    )
    assert "X_pca" in new.obsm
    assert new.obsm["X_pca"].shape[0] == 8
    # PC1 should split the two groups; check absolute Spearman > 0.7.
    from scipy.stats import spearmanr
    grp_num = (new.obs["group"].astype(str) == "B").astype(int).values
    rho, _ = spearmanr(new.obsm["X_pca"][:, 0], grp_num)
    assert abs(rho) > 0.7, f"PC1 doesn't separate groups (rho={rho:.2f})"


def test_plot_embedding_renders(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import epione as epi

    cool_paths, cell_ids, obs = _build_collection(tmp_path, n_per_group=2)
    adata = epi.sc_hic.load_cool_collection(
        cool_paths, cell_ids=cell_ids, obs=obs,
    )
    epi.sc_hic.impute_cells(
        adata, out_dir=tmp_path / "imputed",
        pad=1, rwr_alpha=0.05, top_pct=0.1, progress=False,
    )
    new = epi.sc_hic.embedding(adata, n_components=2)

    fig, ax = epi.sc_hic.plot_embedding(
        new, basis="X_pca", color="group", figsize=(4, 3),
    )
    assert fig is not None and ax is not None
    plt.close(fig)


def test_plot_cell_contacts_imputed_vs_raw(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import epione as epi

    cool_paths, cell_ids, obs = _build_collection(tmp_path, n_per_group=1)
    adata = epi.sc_hic.load_cool_collection(
        cool_paths, cell_ids=cell_ids, obs=obs,
    )
    cid = cell_ids[0]

    # Raw works without impute_cells having run.
    fig_raw, _ = epi.sc_hic.plot_cell_contacts(
        adata, cell_id=cid, chromosome="chr_a", use_imputed=False,
        figsize=(4, 3.5),
    )
    plt.close(fig_raw)

    epi.sc_hic.impute_cells(
        adata, out_dir=tmp_path / "imputed",
        pad=1, rwr_alpha=0.05, top_pct=0.1, progress=False,
    )
    fig_imp, _ = epi.sc_hic.plot_cell_contacts(
        adata, cell_id=cid, chromosome="chr_a", use_imputed=True,
        figsize=(4, 3.5),
    )
    plt.close(fig_imp)
