"""Bulk Hi-C analysis — companion to :mod:`epione.single.hic`.

Operates on a single deeply-sequenced ``.cool`` (typically built by
:mod:`epione.upstream` from a paired FASTQ via the bwa+pairtools+cooler
chain). Phase 1 covers ICE matrix balancing; Phase 2 will add
A/B compartments, TADs, and loops.

Public API (flat — call as ``epi.bulk.hic.X``):
    * :func:`balance_cool` — ICE-balance a ``.cool`` in place
    * :func:`plot_contact_matrix` — log-scale heatmap for a region
    * :func:`plot_decay_curve`    — P(s) curve, the canonical QC plot
    * :func:`plot_coverage`       — per-bin coverage + ICE weight panels
"""
from __future__ import annotations

from ._balance import balance_cool
# Plotting helpers live in :mod:`epione.pl` since v0.4 (PR 3); import
# them from there so ``epi.bulk.hic.plot_contact_matrix`` still resolves.
from epione.pl._contact import (
    plot_contact_matrix,
    plot_decay_curve,
    plot_coverage,
)

__all__ = [
    "balance_cool",
    "plot_contact_matrix",
    "plot_decay_curve",
    "plot_coverage",
]
