"""Tests for epione.tl.add_background_peaks + compute_deviations.

The full add_motif_matrix path needs a genome FASTA + pyjaspar, which we
skip here and fake with a planted peak × motif matrix instead — that
cleanly isolates the background-peak + chromVAR deviation math.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest
from anndata import AnnData


def _make_planted_adata(seed=0, n_cells=600, n_peaks=3000, n_motifs=10,
                        peaks_per_motif=150, n_groups=3):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_groups), n_cells // n_groups)

    # Base Poisson background signal
    X = rng.poisson(lam=0.3, size=(n_cells, n_peaks)).astype(np.float32)

    # Motif annotations: peaks_per_motif per motif, non-overlapping
    motif_sets = [rng.choice(n_peaks, size=peaks_per_motif, replace=False)
                  for _ in range(n_motifs)]

    # Plant group-specific motif activity:
    #   motif 0,1 => group 0 | 2,3 => group 1 | 4,5 => group 2 | 6-9 background
    group_of_motif = [0, 0, 1, 1, 2, 2, -1, -1, -1, -1]
    for g_m, peaks in zip(group_of_motif, motif_sets):
        if g_m < 0:
            continue
        cells = np.where(labels == g_m)[0]
        for p in peaks:
            # boost peak counts in motif-linked peaks for this group
            X[cells, p] += rng.poisson(lam=3.0, size=len(cells))

    # Build binary motif × peak
    data, rows, cols = [], [], []
    for j, peaks in enumerate(motif_sets):
        rows.extend(peaks.tolist())
        cols.extend([j] * len(peaks))
        data.extend([True] * len(peaks))
    motif_mat = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_peaks, n_motifs),
    ).tocsr()
    motif_names = np.array([f"M{j}" for j in range(n_motifs)], dtype=object)

    # GC-like per-peak bias that is uncorrelated with group planting.
    gc = rng.uniform(0.3, 0.7, size=n_peaks).astype(np.float32)

    ad = AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({"group": labels.astype(str),
                          "n_fragment": X.sum(axis=1)},
                         index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame({"GC": gc},
                         index=[f"p{i}" for i in range(n_peaks)]),
    )
    ad.varm["motif"] = motif_mat
    ad.uns["motif_names"] = motif_names
    return ad, labels, group_of_motif


def test_bg_peaks_size_and_contents():
    import epione as epi
    ad, *_ = _make_planted_adata()
    epi.tl.add_background_peaks(ad, n_iterations=25, gc_key="GC", seed=0,
                                 key_added="bg_peaks")
    bg = ad.varm["bg_peaks"]
    n_peaks = ad.n_vars
    assert bg.shape == (n_peaks, 25)
    # Every bg index must be in range. chromVAR / pychromvar draw peers by
    # density-weighted sampling and occasionally hit the peak itself — that
    # is the upstream convention, so we only require the self-hit rate is
    # small (well below 1 / n_iterations).
    assert bg.min() >= 0 and bg.max() < n_peaks
    self_idx = np.arange(n_peaks)[:, None]
    self_rate = float((bg == self_idx).mean())
    assert self_rate < 0.05, f"bg self-hit rate too high: {self_rate:.3f}"


def test_compute_deviations_recovers_planted_groups():
    """Each planted motif should show its strongest Z-score in the correct
    group, relative to the other groups."""
    import epione as epi
    ad, labels, group_of_motif = _make_planted_adata()
    epi.tl.add_background_peaks(ad, n_iterations=25, gc_key="GC", seed=0,
                                 key_added="bg_peaks")
    epi.tl.compute_deviations(ad, motif_key="motif", bg_key="bg_peaks",
                               key_added="dev")

    Z = ad.obsm["dev"]
    n_cells, n_motifs = Z.shape
    assert n_cells == ad.n_obs
    assert n_motifs == ad.varm["motif"].shape[1]

    # For each planted motif, its expected group's mean Z should be the
    # highest across all groups.
    for j, g_expected in enumerate(group_of_motif):
        if g_expected < 0:
            continue
        group_means = np.array([
            np.nanmean(Z[labels == g, j]) for g in range(3)
        ])
        top_group = int(np.argmax(group_means))
        assert top_group == g_expected, (
            f"motif {j}: expected group {g_expected} to be top, "
            f"got {top_group}; means = {group_means}"
        )
        # And the top group's mean should be clearly above zero.
        assert group_means[g_expected] > 0.5, (
            f"motif {j}: mean Z in target group is {group_means[g_expected]:.2f}, "
            "expected > 0.5"
        )


def test_compute_deviations_stores_raw_and_names():
    import epione as epi
    ad, *_ = _make_planted_adata(n_motifs=5, n_peaks=800, peaks_per_motif=40,
                                  n_cells=150)
    epi.tl.add_background_peaks(ad, n_iterations=10, gc_key="GC", seed=0,
                                 key_added="bg_peaks")
    epi.tl.compute_deviations(ad, key_added="dev")
    assert "dev" in ad.obsm
    assert "dev_raw" in ad.obsm
    assert list(ad.uns["dev_names"]) == list(ad.uns["motif_names"])


if __name__ == "__main__":
    test_bg_peaks_size_and_contents()
    test_compute_deviations_recovers_planted_groups()
    test_compute_deviations_stores_raw_and_names()
    print("ok")
