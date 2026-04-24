"""Hi-C analysis module — native Python wrappers over cooler + cooltools
+ pairtools for contact matrix construction, balancing, and visualisation.

Design mirrors :mod:`epione.align` (upstream) and
:mod:`epione.bulk.footprint_archr` (analysis wrapper): every entry
point takes canonical paths (BAM, chrom.sizes, .cool) and returns the
next artefact in the pipeline, so a tutorial can thread a FASTQ ->
BAM -> pairs -> .cool -> balanced .cool -> compartments / TADs / loops
flow with one epione call per stage.

Phase 1 (this module) covers:
    * :func:`pairs_from_bam`   — BAM → deduped, filtered .pairs.gz
    * :func:`pairs_to_cool`    — pairs.gz → .cool at a given binsize
    * :func:`balance_cool`     — ICE-balance a .cool in place
    * :func:`plot_contact_matrix` — quick log-scale heatmap for a region
"""
from __future__ import annotations

from .build import pairs_from_bam, pairs_to_cool, HIC_TOOLS
from .correct import balance_cool
from .plot import plot_contact_matrix

__all__ = [
    "pairs_from_bam",
    "pairs_to_cool",
    "balance_cool",
    "plot_contact_matrix",
    "HIC_TOOLS",
]
