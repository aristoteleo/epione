"""Per-site TF binding: Epione's default entry point.

:func:`tf_binding` scores every motif site for occupancy by its factor and
dispatches on ``method``:

``'occupancy'`` (default)
    :func:`~epione.tl.tf_occupancy` -- the thermodynamic, accessibility-
    gated motif score: Tn5 accessibility around the site (sub-nucleosomal
    fragments up-weighted) times the Boltzmann weight of the PWM affinity,
    summed over the factor's sites within the ~150 bp a ChIP peak
    resolves. It is a motif-accessibility score, **not** a protection
    footprint: a footprint (protection) likelihood was tested during
    design and did not improve the ranking of ChIP-bound sites.
``'print'``
    :func:`~epione.tl.tf_binding_score` -- PRINT / scPrinter's footprint-
    based binding probability, bit-identical to scPrinter. Needs
    ``regions`` (the peaks the footprints are computed over), strand-aware
    sites, torch and the PRINT weights.

Why the default changed (Epione 2026-10)
----------------------------------------
On ENCODE snATAC with ENCODE ChIP-seq ground truth (bound = site centre in
the factor's conservative IDR peak; accessible motif matches only), median
per-TF AUROC:

==============  ============  ======  ========  =====  =====
dataset         occupancy     TOBIAS  Tn5±100   PRINT  HINT
==============  ============  ======  ========  =====  =====
K562 (design)   0.890         0.851   0.845     0.725  0.644
GM12878         0.911         0.895   0.905     0.774  0.651
HepG2           0.909         0.898   0.904     0.747  0.673
MCF-7           0.887         0.843   0.847     0.725  0.644
==============  ============  ======  ========  =====  =====

GM12878, HepG2 and MCF-7 were held out (constants frozen on K562 first).
The occupancy score beat PRINT by 0.14-0.16 AUROC everywhere, beat TOBIAS
and the plain insertion count on MCF-7 (and TOBIAS on HepG2) and tied them
elsewhere, and runs ~70x faster than ``tf_binding_score`` on the same
input and threads. PRINT remains available unchanged as ``method='print'``
and as :func:`~epione.tl.tf_binding_score`; aggregate and multi-scale
footprints (:func:`~epione.tl.get_footprints`,
:func:`~epione.tl.multi_scale_footprint_region`) are unaffected.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

__all__ = ["TF_BINDING_METHODS", "tf_binding"]

#: Per-site binding methods :func:`tf_binding` dispatches to; the first is the default.
TF_BINDING_METHODS = ("occupancy", "print")


def tf_binding(
    adata,
    sites: pd.DataFrame,
    *,
    method: str = "occupancy",
    groupby: Optional[str] = None,
    regions: Optional[pd.DataFrame] = None,
    **kwargs,
):
    """Per-site TF binding score (default: thermodynamic occupancy).

    Parameters
    ----------
    adata
        AnnData with ``uns['files']['fragments']``.
    sites
        Motif sites. For ``'occupancy'``: ``chrom``, ``start``, ``end``, a
        motif column (``motif_key``, default ``'motif'``) and a PWM log-odds
        column (``score_key``, default ``'score'``; log2 unless
        ``score_base`` says otherwise). For ``'print'``: ``chrom``,
        ``start``, ``end``, ``strand``.
    method
        ``'occupancy'`` (default, :func:`~epione.tl.tf_occupancy`) or
        ``'print'`` (:func:`~epione.tl.tf_binding_score`).
    groupby
        ``obs`` column for pseudobulks. ``'occupancy'`` accepts ``None``
        (all cells); ``'print'`` needs a column.
    regions
        Peaks for ``'print'`` (footprints are computed per region); ignored
        by ``'occupancy'``.
    **kwargs
        Passed to the chosen function (e.g. ``n_jobs``, ``motif_key``,
        ``score_key`` for occupancy; ``bias``, ``dispersion`` for PRINT).

    Returns
    -------
    :class:`~epione.tl.OccupancyResult` or :class:`~epione.tl.TFBindingResult`
        Both carry ``groups``, ``sites``, ``score`` (``[n_groups, n_sites]``)
        and ``to_frame()``.

    Examples
    --------
    >>> hits = epi.tl.hint_match_motifs(peaks, "JASPAR2024.jaspar", genome)
    >>> res = epi.tl.tf_binding(adata, hits, motif_key="name", n_jobs=8)
    >>> prt = epi.tl.tf_binding(adata, hits, method="print", regions=peaks,
    ...                         groupby="celltype")
    """
    method = str(method).lower()
    if method == "occupancy":
        from ._occupancy import tf_occupancy
        return tf_occupancy(adata, sites, groupby=groupby, **kwargs)
    if method == "print":
        from ._print import tf_binding_score
        if regions is None:
            raise ValueError("method='print' needs regions= (the peaks to compute footprints over)")
        if groupby is None:
            raise ValueError("method='print' needs groupby= (an obs column)")
        cols = [c for c in ("chrom", "start", "end", "strand") if c in sites.columns]
        return tf_binding_score(adata, regions, groupby=groupby, sites=sites[cols], **kwargs)
    raise ValueError(f"unknown method {method!r}; one of {TF_BINDING_METHODS}")
