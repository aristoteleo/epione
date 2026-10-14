"""Bulk Hi-C analysis — companion to :mod:`epione.single.hic`.

Operates on a single deeply-sequenced ``.cool`` / ``.mcool``.
Phase 1 covers ICE balancing; Phase 2A adds compartments + saddle
+ pyramid; Phase 2B adds insulation + boundaries; Phase 2C (this
commit) adds loops + APA pile-up — the full Maziak 2026 / Chang
2024 reproduction stack.

Public API (flat — call as ``epi.bulk.hic.X``):
    * :func:`balance_cool`            — ICE-balance a ``.cool``
    * :func:`compartments`            — per-bin A/B eigenvectors
    * :func:`saddle`                  — compartment-strength matrix
    * :func:`insulation`              — diamond-insulation score
    * :func:`tad_boundaries`          — boundary BED from insulation
    * :func:`loops`                   — HICCUPS-style dot finder
    * :func:`pileup`                  — APA / aggregate stack
    * :func:`tad_domains`             — TAD intervals between boundaries
    * :func:`compare_matrices`        — diff / ratio / log2ratio of two matrices
    * :func:`correlate_matrices`      — pairwise Pearson / Spearman of matrices
    * :func:`differential_tads`       — Wilcoxon test of TADs between conditions
    * :func:`h5_to_cool`              — HiCExplorer ``.h5`` → ``.cool``

``compartments``, ``insulation`` / ``tad_boundaries`` and ``loops`` take
``method="cooltools"`` (default) or ``method="hicexplorer"`` (ports of
hicPCA, hicFindTADs and hicDetectLoops, HiCExplorer 3.7.6).
    * :func:`plot_contact_matrix`     — region heatmap
    * :func:`plot_contact_triangle`   — pyramid view of region
    * :func:`plot_decay_curve`        — P(s) QC plot
    * :func:`plot_coverage`           — coverage + ICE weight panels
    * :func:`plot_compartments`       — chromosome E1 track
    * :func:`plot_saddle`             — A/B saddle heatmap
    * :func:`plot_insulation`         — insulation score line + boundaries
    * :func:`plot_loops`              — heatmap with dot overlay
    * :func:`plot_apa`                — APA aggregate heatmap
"""
from __future__ import annotations

from ._balance import balance_cool
from ._compartments import compartments, saddle
from ._insulation import insulation, tad_boundaries, tad_domains
from ._loops import loops, pileup
from ._compare import compare_matrices, correlate_matrices
from ._differential import differential_tads
from ._io import h5_to_cool
# Plotting helpers live in :mod:`epione.pl` since v0.4 (PR 3); import
# them from there so ``epi.bulk.hic.plot_*`` still resolves.
from epione.pl._contact import (
    plot_contact_matrix,
    plot_contact_triangle,
    plot_decay_curve,
    plot_coverage,
    plot_compartments,
    plot_saddle,
    plot_insulation,
    plot_loops,
    plot_apa,
)

__all__ = [
    "balance_cool",
    "compartments",
    "saddle",
    "insulation",
    "tad_boundaries",
    "tad_domains",
    "loops",
    "pileup",
    "compare_matrices",
    "correlate_matrices",
    "differential_tads",
    "h5_to_cool",
    "plot_contact_matrix",
    "plot_contact_triangle",
    "plot_decay_curve",
    "plot_coverage",
    "plot_compartments",
    "plot_saddle",
    "plot_insulation",
    "plot_loops",
    "plot_apa",
]
