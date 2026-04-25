"""Insulation score and TAD-boundary calling for bulk Hi-C.

Wraps :func:`cooltools.insulation` (the diamond-insulation score
of Crane et al. 2015 / cooltools 0.7) to score every bin's
"insulation" — i.e. how much it interrupts surrounding contact
density. Sliding a diamond-shaped window of size ``window_bp``
along the diagonal, the score is

    log2 ( median upstream-of-bin / median downstream-of-bin )

normalised; deep dips (negative) mark putative TAD boundaries. The
companion :func:`tad_boundaries` thresholds the score (Li-method by
default) to emit a discrete BED of boundaries with confidence
estimates, matching panel h of Maziak 2026 / Chang 2024.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


def _resolve_uri(path: Union[str, Path], resolution: Optional[int]) -> str:
    p = str(path)
    if "::" in p:
        return p
    if resolution is None:
        return p
    return f"{p}::resolutions/{int(resolution)}"


def insulation(
    cool_path: Union[str, Path],
    *,
    window_bp: Union[int, Sequence[int]] = 100_000,
    resolution: Optional[int] = None,
    chromosomes: Optional[Sequence[str]] = None,
    ignore_diags: Optional[int] = None,
    threshold: Union[str, float] = "Li",
    nproc: int = 1,
) -> pd.DataFrame:
    """Diamond insulation score per bin from a balanced ``.cool``.

    Arguments:
        cool_path: ``.cool`` / ``.mcool::resolutions/N`` (must be
            ICE-balanced; pass through :func:`balance_cool` first).
        window_bp: diamond-window size in bp. ``100_000`` (100 kb)
            is the cooltools default for mammalian TADs at 10–25 kb
            bin resolution. Pass a list to score multiple windows in
            one shot — useful for picking the right window for an
            unfamiliar dataset.
        resolution: bp resolution for ``.mcool``.
        chromosomes: subset; default = every chromosome in the cool.
        ignore_diags: number of near-diagonal bins to mask. Default
            (``None``) lets cooltools pick a sensible value from the
            cool's metadata.
        threshold: boundary-strength cutoff method passed through to
            cooltools. ``'Li'`` (default) is the automated method
            from Crane et al. 2015; pass a float to set a hard
            cutoff.
        nproc: parallel processes for the per-chromosome scoring.

    Returns:
        ``pandas.DataFrame`` with one row per bin and columns:

        * ``chrom``, ``start``, ``end``: bin coordinates
        * ``log2_insulation_score_<W>``: insulation score for window
          ``W`` (one column per requested ``window_bp``)
        * ``boundary_strength_<W>``: per-bin minimum boundary strength
        * ``is_boundary_<W>``: bool, ``True`` for thresholded boundaries

        Bins masked by ICE balancing show ``NaN`` insulation scores.
    """
    import cooler
    import cooltools

    clr = cooler.Cooler(_resolve_uri(cool_path, resolution))
    if chromosomes is not None:
        view = pd.DataFrame({
            "chrom": list(chromosomes),
            "start": [0] * len(chromosomes),
            "end": [int(clr.chromsizes[c]) for c in chromosomes],
            "name": list(chromosomes),
        })
    else:
        view = None

    if isinstance(window_bp, (int, np.integer)):
        window_bp = [int(window_bp)]
    else:
        window_bp = [int(w) for w in window_bp]

    df = cooltools.insulation(
        clr=clr,
        window_bp=window_bp,
        view_df=view,
        ignore_diags=ignore_diags,
        threshold=threshold,
        nproc=int(nproc),
    )
    return df


def tad_boundaries(
    insulation_df: pd.DataFrame,
    *,
    window_bp: Optional[int] = None,
    min_strength: Optional[float] = None,
) -> pd.DataFrame:
    """Extract TAD boundaries from an insulation table.

    Arguments:
        insulation_df: output of :func:`insulation`.
        window_bp: which window to use when the insulation table holds
            multiple. Default = the smallest window present.
        min_strength: optional override for the boundary-strength
            threshold; if ``None``, use cooltools' ``is_boundary_<W>``
            (Li-method auto-threshold).

    Returns:
        BED-like DataFrame with ``chrom``, ``start``, ``end``,
        ``window_bp``, ``insulation_score``, ``boundary_strength``,
        sorted by genomic coordinate.
    """
    cols = [c for c in insulation_df.columns
            if c.startswith("log2_insulation_score_")]
    if not cols:
        raise KeyError(
            "insulation_df has no log2_insulation_score_* column — "
            "did the cooler have weight column for ICE balancing?"
        )
    avail = [int(c.rsplit("_", 1)[-1]) for c in cols]
    if window_bp is None:
        window_bp = min(avail)
    if window_bp not in avail:
        raise KeyError(
            f"window_bp={window_bp} not in insulation_df; available: "
            f"{sorted(avail)}"
        )

    score_col = f"log2_insulation_score_{window_bp}"
    strength_col = f"boundary_strength_{window_bp}"
    is_bnd_col = f"is_boundary_{window_bp}"

    if min_strength is not None:
        mask = insulation_df[strength_col] >= float(min_strength)
    else:
        mask = insulation_df[is_bnd_col].astype(bool)

    out = insulation_df.loc[mask, ["chrom", "start", "end",
                                   score_col, strength_col]].copy()
    out = out.rename(columns={
        score_col: "insulation_score",
        strength_col: "boundary_strength",
    })
    out["window_bp"] = int(window_bp)
    out = out[["chrom", "start", "end", "window_bp",
               "insulation_score", "boundary_strength"]]
    return out.sort_values(["chrom", "start"]).reset_index(drop=True)
