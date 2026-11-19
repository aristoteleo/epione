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
    method: str = "cooltools",
    min_depth: Optional[int] = None,
    max_depth: Optional[int] = None,
    step: Optional[int] = None,
) -> pd.DataFrame:
    """Diamond insulation score per bin from a balanced ``.cool``.

    ``method="cooltools"`` (default) is :func:`cooltools.insulation`.
    ``method="hicexplorer"`` computes HiCExplorer's TAD-separation score
    (``hicFindTADs``, 3.7.6): the mean z-score of the diamond between each
    bin's left and right flanks, at several window sizes. Pass its output to
    :func:`tad_boundaries` for HiCExplorer's boundary calling.

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
        nproc: parallel processes for the per-chromosome scoring
            (``method="cooltools"`` only).
        method: ``"cooltools"`` or ``"hicexplorer"``.
        min_depth, max_depth, step: ``method="hicexplorer"`` window sizes in
            bp (``--minDepth`` / ``--maxDepth`` / ``--step``); windows are
            ``min_depth + int(step * k ** 1.5)`` up to ``max_depth``. Defaults
            follow HiCExplorer (for 1–20 kb bins: 10, 40 and 2 bins).
            ``window_bp``, ``ignore_diags``, ``threshold`` and ``nproc`` do
            not apply.

    Returns:
        ``pandas.DataFrame`` with one row per bin and columns:

        * ``chrom``, ``start``, ``end``: bin coordinates
        * ``log2_insulation_score_<W>``: insulation score for window
          ``W`` (one column per requested ``window_bp``)
        * ``boundary_strength_<W>``: per-bin minimum boundary strength
        * ``is_boundary_<W>``: bool, ``True`` for thresholded boundaries

        Bins masked by ICE balancing show ``NaN`` insulation scores.

        With ``method="hicexplorer"``: ``chrom``, ``start``, ``end`` (bins
        after HiCExplorer drops masked bins and closes the gaps), one
        ``tad_separation_score_<W>`` column per window and their mean
        ``tad_separation_score``. Bins whose diamond is undefined at any
        window are omitted, as in HiCExplorer. The table's ``attrs`` remember
        the input so :func:`tad_boundaries` can compute boundary p-values.
    """
    if method == "hicexplorer":
        return _insulation_hicexplorer(cool_path, resolution=resolution, chromosomes=chromosomes,
                                       min_depth=min_depth, max_depth=max_depth, step=step)
    if method != "cooltools":
        raise ValueError("method must be 'cooltools' or 'hicexplorer'")
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

    if ignore_diags is None:
        # Read it from the balancing weights' attributes ourselves: cooltools
        # 0.7.1 does this through ``Cooler._load_attrs``, which cooler 0.12
        # removed, and then reports the value as missing.
        with clr.open("r") as grp:
            w = grp["bins"].get("weight")
            if w is not None and "ignore_diags" in w.attrs:
                ignore_diags = int(w.attrs["ignore_diags"])

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
    delta: float = 0.01,
    threshold_comparisons: float = 0.01,
    correct_for_multiple_testing: Optional[str] = "fdr",
    min_boundary_distance: Optional[int] = None,
) -> pd.DataFrame:
    """Extract TAD boundaries from an insulation table.

    The method follows the table: a cooltools insulation table is thresholded
    on its ``is_boundary_*`` / ``boundary_strength_*`` columns; a
    ``method="hicexplorer"`` table goes through ``hicFindTADs``' boundary
    calling (consensus minima of the mean TAD-separation score, filtered by
    ``delta`` and a Wilcoxon rank-sum p-value against the flanking diamonds).

    Arguments:
        insulation_df: output of :func:`insulation`.
        window_bp: which window to use when the insulation table holds
            multiple. Default = the smallest window present.
        min_strength: optional override for the boundary-strength
            threshold; if ``None``, use cooltools' ``is_boundary_<W>``
            (Li-method auto-threshold).
        delta, threshold_comparisons, correct_for_multiple_testing,
        min_boundary_distance: HiCExplorer boundary filters (``--delta``,
            ``--thresholdComparisons``, ``--correctForMultipleTesting`` =
            ``"fdr"`` / ``"bonferroni"`` / ``None``, ``--minBoundaryDistance``
            in bp, default 4 bins); used only for a ``method="hicexplorer"``
            table.

    Returns:
        BED-like DataFrame with ``chrom``, ``start``, ``end``,
        ``window_bp``, ``insulation_score``, ``boundary_strength``,
        sorted by genomic coordinate.

        For a ``method="hicexplorer"`` table, HiCExplorer's boundary records:
        ``chrom``, ``start``, ``end`` (between the centres of the two bins
        that meet at the boundary), ``name``, ``tad_separation_score``,
        ``delta``, ``pvalue``, plus ``bin_start`` / ``next_boundary_start``
        used by :func:`tad_domains`.
    """
    if insulation_df.attrs.get("epione_method") == "hicexplorer":
        from ._tads_hicexplorer import find_boundaries
        hm, params = _hicexplorer_state(insulation_df)
        return find_boundaries(insulation_df, hm, params, delta=delta,
                               threshold_comparisons=threshold_comparisons,
                               correct_for_multiple_testing=correct_for_multiple_testing,
                               min_boundary_distance=min_boundary_distance)
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


# ------------------------------------------------------------------ hicexplorer
_HICEXPLORER_CACHE: "dict" = {}


def _insulation_hicexplorer(cool_path, *, resolution, chromosomes, min_depth, max_depth, step):
    from ._hicmatrix import HiCMat
    from ._tads_hicexplorer import tad_separation_score

    uri = _resolve_uri(cool_path, resolution)
    hm = HiCMat.from_cool(uri)
    table, hm, params = tad_separation_score(hm, min_depth=min_depth, max_depth=max_depth,
                                             step=step, chromosomes=chromosomes)
    key = f"{uri}|{chromosomes}|{params['minDepth']}|{params['maxDepth']}|{params['step']}"
    _HICEXPLORER_CACHE.clear()  # keep only the latest z-score matrix in memory
    _HICEXPLORER_CACHE[key] = (hm, params)
    table.attrs["epione_method"] = "hicexplorer"
    table.attrs["hicexplorer"] = {"cool": uri, "chromosomes": None if chromosomes is None else list(chromosomes),
                                  "min_depth": params["minDepth"], "max_depth": params["maxDepth"],
                                  "step": params["step"], "key": key}
    return table


def _hicexplorer_state(table):
    """The z-score matrix behind a hicexplorer insulation table (recomputed if evicted)."""
    from ._hicmatrix import HiCMat
    from ._tads_hicexplorer import tad_separation_score

    info = table.attrs["hicexplorer"]
    if info["key"] in _HICEXPLORER_CACHE:
        return _HICEXPLORER_CACHE[info["key"]]
    hm = HiCMat.from_cool(info["cool"])
    _, hm, params = tad_separation_score(hm, min_depth=info["min_depth"], max_depth=info["max_depth"],
                                         step=info["step"], chromosomes=info["chromosomes"])
    return hm, params


def tad_domains(boundaries: pd.DataFrame) -> pd.DataFrame:
    """TAD domains between consecutive boundaries.

    Arguments:
        boundaries: output of :func:`tad_boundaries`.

    Returns:
        BED-like DataFrame ``chrom``, ``start``, ``end``, ``name`` (plus the
        boundary score). For HiCExplorer boundaries this is ``hicFindTADs``'
        ``_domains.bed``: each domain runs from a boundary bin's start to the
        next boundary bin's start on the same chromosome, so the region before
        a chromosome's first boundary and after its last is not a domain. For
        cooltools boundaries, domains run between consecutive boundary bins
        (from one boundary's ``end`` to the next one's ``start``).
    """
    if "next_boundary_start" in boundaries.columns:
        from ._tads_hicexplorer import domains_from_boundaries
        return domains_from_boundaries(boundaries)
    recs = []
    for chrom, g in boundaries.sort_values(["chrom", "start"]).groupby("chrom", sort=False):
        s = g["start"].to_numpy()
        e = g["end"].to_numpy()
        for k in range(len(g) - 1):
            recs.append((chrom, int(e[k]), int(s[k + 1])))
    out = pd.DataFrame(recs, columns=["chrom", "start", "end"])
    out["name"] = [f"TAD_{i + 1}" for i in range(len(out))]
    return out
