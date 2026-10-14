"""Loop calling and aggregate pile-up analysis for bulk Hi-C.

Wraps :func:`cooltools.dots` (HICCUPS-style dot finder) to call
chromatin loops, and :func:`cooltools.pileup` to aggregate-stack a
contact-matrix region around a set of features (loops, boundaries,
peaks). These are the core analyses of Maziak 2026 Fig 1 / Fig 3
(stage-specific loop emergence + APA pile-ups) and Chang 2024 Fig 1
(per-cell-type loop comparison).

Both functions take a balanced ``.cool`` and optional pre-computed
expected; if expected is missing, it's computed on-the-fly.
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


def _ensure_expected_cis(clr, view, ignore_diags: int = 2):
    """Compute (or return cached) expected_cis for a cooler + view."""
    from cooltools.api.expected import expected_cis
    return expected_cis(
        clr=clr, view_df=view, ignore_diags=ignore_diags,
        chunksize=1_000_000,
    )


def loops(
    cool_path: Union[str, Path],
    *,
    resolution: Optional[int] = None,
    chromosomes: Optional[Sequence[str]] = None,
    max_loci_separation: int = 10_000_000,
    fdr: float = 0.1,
    clustering_radius: int = 20_000,
    nproc: int = 1,
    method: str = "cooltools",
    window_size: int = 5,
    peak_width: int = 2,
    p_value_preselection: float = 0.1,
    peak_interactions_threshold: float = 10,
    obs_exp_threshold: float = 1.5,
    p_value: float = 0.025,
    max_loop_distance: int = 2_000_000,
    expected: str = "mean",
) -> pd.DataFrame:
    """Call chromatin loops (HICCUPS-style dots) on a balanced cool.

    ``method="cooltools"`` (default) wraps :func:`cooltools.dots`.
    ``method="hicexplorer"`` reproduces ``hicDetectLoops`` (HiCExplorer
    3.7.6, Wolff et al. 2020): a negative-binomial preselection of enriched
    pixels per distance, neighbourhood maxima, then Wilcoxon tests of the
    peak against a donut and the background.

    Wraps :func:`cooltools.dots` end-to-end: builds the ``expected_cis``
    on the fly, runs the dot-finder with the standard 4-kernel set,
    applies BH-FDR correction at ``fdr``, and clusters nearby calls
    within ``clustering_radius`` so each loop is a single anchor pair.

    Arguments:
        cool_path: balanced ``.cool`` / ``.mcool::resolutions/N`` (must
            have a ``weight`` column).
        resolution: bp resolution for ``.mcool``.
        chromosomes: subset; default = all in the cool.
        max_loci_separation: maximum loop anchor separation (bp).
            ``10_000_000`` (10 Mb) is the cooltools / HICCUPS default.
        fdr: BH-FDR threshold per ``lambda_bin``. ``0.1`` = standard.
        clustering_radius: anchor-pair clustering distance (bp);
            within this, multiple sub-significant pixels are merged
            into one loop call. ``20_000`` = cooltools default.
        nproc: parallel workers for tile scanning (cooltools).
        method: ``"cooltools"`` or ``"hicexplorer"``.
        window_size, peak_width, p_value_preselection,
        peak_interactions_threshold, obs_exp_threshold, p_value,
        max_loop_distance, expected: ``method="hicexplorer"`` only, with
            HiCExplorer's defaults (``--windowSize``, ``--peakWidth``,
            ``--pValuePreselection``, ``--peakInteractionsThreshold``,
            ``--obsExpThreshold``, ``--pValue``, ``--maxLoopDistance`` in bp,
            ``--expected`` = ``"mean"`` / ``"mean_nonzero"`` /
            ``"mean_nonzero_ligation"``). ``max_loci_separation``, ``fdr`` and
            ``clustering_radius`` apply to cooltools only.

    Returns:
        BEDPE-shaped DataFrame: one row per called loop. Columns:
        ``chrom1, start1, end1, chrom2, start2, end2`` (anchors),
        plus cooltools-internal columns (``count``, ``la_exp.*``,
        ``la_p.*``, etc.) — keep them for filtering/sorting; drop
        before exporting if you only want the BEDPE shape.

        With ``method="hicexplorer"``: ``chrom1, start1, end1, chrom2,
        start2, end2, pvalue`` (the peak-vs-background rank-sum p-value).
    """
    if method == "hicexplorer":
        from ._loops_hicexplorer import detect_loops
        return detect_loops(_resolve_uri(cool_path, resolution), chromosomes=chromosomes,
                            window_size=window_size, peak_width=peak_width,
                            p_value_preselection=p_value_preselection,
                            peak_interactions_threshold=peak_interactions_threshold,
                            obs_exp_threshold=obs_exp_threshold, p_value=p_value,
                            max_loop_distance=max_loop_distance, expected=expected)
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
        view = pd.DataFrame({
            "chrom": list(clr.chromnames),
            "start": [0] * len(clr.chromnames),
            "end": [int(clr.chromsizes[c]) for c in clr.chromnames],
            "name": list(clr.chromnames),
        })

    expected = _ensure_expected_cis(clr, view, ignore_diags=2)

    df = cooltools.dots(
        clr=clr,
        expected=expected,
        view_df=view,
        max_loci_separation=int(max_loci_separation),
        lambda_bin_fdr=float(fdr),
        clustering_radius=int(clustering_radius),
        nproc=int(nproc),
    )
    return df


def pileup(
    cool_path: Union[str, Path],
    features_df: pd.DataFrame,
    *,
    resolution: Optional[int] = None,
    flank: int = 100_000,
    chromosomes: Optional[Sequence[str]] = None,
    expected: Optional[pd.DataFrame] = None,
    nproc: int = 1,
) -> np.ndarray:
    """Aggregate-stack contact matrix around a set of genomic features.

    For each feature in ``features_df`` (loops, boundaries, ChIP peaks
    etc.), pull the contact-matrix sub-region of side ``2*flank+binsize``
    centred on the feature and average across the set. Returns the
    ``(2*flank+binsize) // binsize`` × ditto observed-over-expected
    average — the canonical APA / boundary-pile-up plot from Maziak /
    Chang papers.

    Arguments:
        cool_path: balanced ``.cool`` / ``.mcool::resolutions/N``.
        features_df: bed3 (boundary pile-up) or bedpe (loop APA)
            DataFrame. cooltools auto-detects shape from columns.
        resolution: bp resolution for ``.mcool``.
        flank: bp on each side of the feature centre. The output
            matrix is square with side ``2*flank/binsize + 1``.
        chromosomes: subset for the on-the-fly expected.
        expected: pre-computed expected_cis (skip recompute).
        nproc: parallel workers.

    Returns:
        2-D ``numpy.ndarray`` of mean observed-over-expected contact
        frequency. The centre cell is the feature itself; corners are
        ``flank`` bp away on both axes.
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
        view = pd.DataFrame({
            "chrom": list(clr.chromnames),
            "start": [0] * len(clr.chromnames),
            "end": [int(clr.chromsizes[c]) for c in clr.chromnames],
            "name": list(clr.chromnames),
        })

    if expected is None:
        expected = _ensure_expected_cis(clr, view, ignore_diags=2)

    stack = cooltools.pileup(
        clr=clr,
        features_df=features_df,
        view_df=view,
        expected_df=expected,
        flank=int(flank),
        nproc=int(nproc),
    )
    # cooltools returns a 3-D stack (n_features, side, side); average
    # along the feature axis to give the standard APA matrix.
    arr = np.asarray(stack)
    if arr.ndim == 3:
        arr = np.nanmean(arr, axis=0)
    return arr
