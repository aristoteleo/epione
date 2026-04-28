"""A/B compartment analysis for bulk Hi-C.

Wraps :func:`cooltools.eigs_cis` (eigendecomposition of the
distance-corrected, observed-over-expected matrix) to recover the
canonical PC1 eigenvector that separates A (active / open) and B
(repressive / closed) compartments along each chromosome. The result
is a per-bin track with one signed value per genomic bin: positive
in A, negative in B. The sign convention is fixed against an external
GC-content track (or a user-provided phasing track), so that A always
points to higher GC content (or phase signal) — without that the
eigenvector sign is arbitrary.

Companion :func:`saddle` quantifies compartmentalisation strength as a
2-D heatmap of mean observed-over-expected contact across pairs of
bins binned by their compartment score (saddle plot — the canonical
"AA / AB / BB / BA" strength matrix used in the Hi-C literature).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


def compartments(
    cool_path: Union[str, Path],
    *,
    resolution: Optional[int] = None,
    chromosomes: Optional[Sequence[str]] = None,
    n_eigs: int = 3,
    phasing_track: Optional[pd.DataFrame] = None,
    fasta_path: Optional[Union[str, Path]] = None,
    ignore_diags: int = 2,
) -> pd.DataFrame:
    """Compute per-bin A/B compartment eigenvectors from a balanced ``.cool``.

    Arguments:
        cool_path: path to a balanced ``.cool`` (or ``.mcool::resolutions/N``).
            For ``.mcool``, pass either the URI form or the bare path
            with ``resolution=`` to pick a layer.
        resolution: bin resolution in bp. Required for ``.mcool``;
            ignored for single-resolution ``.cool``.
        chromosomes: subset of chromosomes to score. Default = every
            chromosome present in the cool.
        n_eigs: number of eigenvectors to retain (cooltools returns the
            top ``n_eigs``; the first ``E1`` is the conventional A/B
            score, ``E2``/``E3`` capture finer structure). Default 3.
        phasing_track: optional external 1-D track to fix eigenvector
            sign. DataFrame with ``chrom``, ``start``, ``end``, plus a
            value column (e.g. GC content). The eigenvector at each
            chromosome is sign-flipped so that the within-chromosome
            correlation with the track is positive. If ``None`` and
            ``fasta_path`` is given, GC content is computed from the
            FASTA. If both ``None``, no phasing — sign is arbitrary.
        fasta_path: FASTA used to compute per-bin GC content as a
            phasing track (overridden by ``phasing_track`` if given).
        ignore_diags: diagonals to mask before eigendecomposition.
            Default 2 (skip main + first off-diagonal, dominated by
            self-ligation / read-pair artefacts).

    Returns:
        ``pandas.DataFrame`` with per-bin rows and columns
        ``chrom``, ``start``, ``end``, ``E1``, ``E2``, ..., ``E{n_eigs}``.
        ``E1 > 0`` marks A compartment, ``E1 < 0`` marks B (after sign
        phasing).
    """
    import cooler
    import cooltools

    clr = cooler.Cooler(_resolve_uri(cool_path, resolution))
    if chromosomes is None:
        chromosomes = list(clr.chromnames)
    # cooltools requires view_df to be ordered the same as the cool's
    # chromnames, so re-sort the user-supplied subset to match.
    chrom_order = {c: i for i, c in enumerate(clr.chromnames)}
    chromosomes = sorted(set(chromosomes), key=lambda c: chrom_order.get(c, 1 << 30))

    view_df = pd.DataFrame({
        "chrom": list(chromosomes),
        "start": [0] * len(chromosomes),
        "end": [int(clr.chromsizes[c]) for c in chromosomes],
        "name": list(chromosomes),
    })

    # Build phasing track if FASTA is given and no explicit track passed.
    if phasing_track is None and fasta_path is not None:
        phasing_track = _gc_track(clr, fasta_path, view_df)

    eigvals, eig_track = cooltools.eigs_cis(
        clr=clr,
        phasing_track=phasing_track,
        view_df=view_df,
        n_eigs=int(n_eigs),
        ignore_diags=int(ignore_diags),
    )
    return eig_track


def _resolve_uri(path: Union[str, Path], resolution: Optional[int]) -> str:
    """Build the right cooler URI for both ``.cool`` and ``.mcool``."""
    p = str(path)
    if "::" in p:
        return p
    if resolution is None:
        return p
    return f"{p}::resolutions/{int(resolution)}"


def _gc_track(clr, fasta_path: Union[str, Path], view_df: pd.DataFrame) -> pd.DataFrame:
    """Per-bin GC content from a FASTA — a robust phasing signal.

    A regions: GC ↑ (gene-rich). B: GC ↓ (gene-poor).
    """
    import bioframe
    bins = clr.bins()[:]
    bins = bins[bins["chrom"].isin(view_df["chrom"])].reset_index(drop=True)
    fa = bioframe.load_fasta(str(fasta_path))
    gc = bioframe.frac_gc(bins[["chrom", "start", "end"]], fa)
    return gc


def saddle(
    cool_path: Union[str, Path],
    eig_track: pd.DataFrame,
    *,
    resolution: Optional[int] = None,
    n_bins: int = 50,
    qrange: tuple = (0.025, 0.975),
    contact_type: str = "cis",
    track_column: str = "E1",
    chromosomes: Optional[Sequence[str]] = None,
):
    """Saddle plot — A/B compartmentalisation strength.

    Bins genomic bins by their compartment eigenvector quantile
    (default ``E1``) and computes the average observed-over-expected
    contact frequency for each pair of quantile bins. The resulting
    ``n_bins × n_bins`` matrix has the canonical "AA top-right strong,
    BB bottom-left strong, AB / BA off-diagonal weak" pattern in a
    healthy library. The strength ratio
    ``(AA + BB) / (AB + BA)`` is the standard scalar summary.

    Arguments:
        cool_path: balanced ``.cool`` / ``.mcool::resolutions/N``.
        eig_track: per-bin compartment track from :func:`compartments`,
            or any DataFrame with ``chrom``, ``start``, ``end``, plus
            ``track_column``.
        resolution: bp resolution for ``.mcool``.
        n_bins: number of quantile bins along each axis. ``50`` matches
            the cooltools default; raise for sharper but slower plots.
        qrange: lower / upper quantile clip for the eigenvector. Tails
            get squashed into the extreme bins to avoid empty slots.
        contact_type: ``'cis'`` (intra-chromosomal, recommended) or
            ``'trans'``.
        track_column: column in ``eig_track`` to bin on. Default
            ``'E1'`` (the A/B eigenvector).
        chromosomes: subset; default = all in ``eig_track``.

    Returns:
        ``(saddle_matrix, edges, sum_count)`` — the ``n_bins × n_bins``
        observed-over-expected matrix, the quantile edges in
        eigenvector units, and the ``n_bins × n_bins`` count of
        contributing bin-pairs (lets you mask sparsely-sampled cells).
    """
    import cooler
    import cooltools
    from cooltools.api.expected import expected_cis

    clr = cooler.Cooler(_resolve_uri(cool_path, resolution))
    if chromosomes is None:
        chromosomes = list(eig_track["chrom"].dropna().unique())
    chrom_order = {c: i for i, c in enumerate(clr.chromnames)}
    chromosomes = sorted(set(chromosomes), key=lambda c: chrom_order.get(c, 1 << 30))

    view_df = pd.DataFrame({
        "chrom": list(chromosomes),
        "start": [0] * len(chromosomes),
        "end": [int(clr.chromsizes[c]) for c in chromosomes],
        "name": list(chromosomes),
    })

    expected = expected_cis(
        clr=clr,
        view_df=view_df,
        ignore_diags=2,
        chunksize=1_000_000,
    )

    Q_LO, Q_HI = qrange
    # cooltools.saddle handles the digitization internally; pass the
    # numeric eigenvector track directly. The track DataFrame must
    # have exactly four columns: chrom, start, end, <numeric value>.
    track_in = eig_track[["chrom", "start", "end", track_column]].copy()
    sum_, count = cooltools.saddle(
        clr=clr,
        expected=expected,
        track=track_in,
        contact_type=contact_type,
        n_bins=n_bins,
        view_df=view_df,
        qrange=(Q_LO, Q_HI),
    )
    # Observed-over-expected average per quantile cell.
    with np.errstate(divide="ignore", invalid="ignore"):
        saddle_mat = np.divide(sum_, count, where=count > 0)
    edges = np.linspace(Q_LO, Q_HI, n_bins + 1)
    return saddle_mat, edges, count
