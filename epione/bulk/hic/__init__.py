"""Bulk Hi-C analysis — companion to :mod:`epione.single.hic`.

Operates on a single deeply-sequenced ``.cool`` / ``.mcool``
(typically built by :mod:`epione.upstream` from a paired FASTQ via
the bwa + pairtools + cooler chain). Phase 1 covers ICE balancing;
Phase 2 (this commit) adds A/B compartments and saddle plots.

Public API (flat — call as ``epi.bulk.hic.X``):
    * :func:`balance_cool` — ICE-balance a ``.cool`` in place
    * :func:`compartments` — per-bin A/B eigenvectors via cooltools
    * :func:`saddle`       — A/B compartmentalisation strength matrix
    * :func:`plot_contact_matrix`  — log-scale region heatmap
    * :func:`plot_decay_curve`     — P(s) curve, the canonical QC plot
    * :func:`plot_coverage`        — per-bin coverage + ICE weight panels
    * :func:`plot_compartments`    — per-bin E1 track for one chromosome
    * :func:`plot_saddle`          — compartmentalisation strength heatmap
"""
from __future__ import annotations

from ._balance import balance_cool
from ._compartments import compartments, saddle
# Plotting helpers live in :mod:`epione.pl` since v0.4 (PR 3); import
# them from there so ``epi.bulk.hic.plot_*`` still resolves.
from epione.pl._contact import (
    plot_contact_matrix,
    plot_decay_curve,
    plot_coverage,
    plot_compartments,
    plot_saddle,
)

__all__ = [
    "balance_cool",
    "compartments",
    "saddle",
    "plot_contact_matrix",
    "plot_decay_curve",
    "plot_coverage",
    "plot_compartments",
    "plot_saddle",
]
