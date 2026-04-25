"""Bulk Hi-C analysis — companion to :mod:`epione.single.hic`.

Operates on a single deeply-sequenced ``.cool`` / ``.mcool``. Phase
1 covers ICE balancing; Phase 2 adds compartments + saddle + the
pyramid contact view; Phase 2B (this commit) adds insulation score
+ TAD-boundary calling.

Public API (flat — call as ``epi.bulk.hic.X``):
    * :func:`balance_cool`            — ICE-balance a ``.cool``
    * :func:`compartments`            — per-bin A/B eigenvectors
    * :func:`saddle`                  — compartment-strength matrix
    * :func:`insulation`              — diamond-insulation score
    * :func:`tad_boundaries`          — boundary BED from insulation
    * :func:`plot_contact_matrix`     — region heatmap
    * :func:`plot_contact_triangle`   — pyramid view of region
    * :func:`plot_decay_curve`        — P(s) QC plot
    * :func:`plot_coverage`           — coverage + ICE weight panels
    * :func:`plot_compartments`       — chromosome E1 track
    * :func:`plot_saddle`             — A/B saddle heatmap
    * :func:`plot_insulation`         — insulation score line + boundaries
"""
from __future__ import annotations

from ._balance import balance_cool
from ._compartments import compartments, saddle
from ._insulation import insulation, tad_boundaries
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
)

__all__ = [
    "balance_cool",
    "compartments",
    "saddle",
    "insulation",
    "tad_boundaries",
    "plot_contact_matrix",
    "plot_contact_triangle",
    "plot_decay_curve",
    "plot_coverage",
    "plot_compartments",
    "plot_saddle",
    "plot_insulation",
]
