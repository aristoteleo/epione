"""Bulk ATAC-seq analysis (v0.4 layout, populated in PR 2b).

Flat API — call as ``epi.bulk.atac.X``:

  - BigWig matrix tooling: :class:`bigwig`, :func:`plot_matrix`,
    :func:`plot_matrix_line`, :class:`plotloc`,
    :func:`gene_expression_from_bigwigs`, plus ``getScorePerBigWigBin``
    helpers.
  - ArchR-style bulk footprint: :func:`footprint_archr`,
    :func:`bam_to_fragments_bulk`.
  - HOMER motif enrichment: :func:`run_homer_motifs` (Perl wrapper) and
    :func:`find_motifs_genome` (pure-Python reimplementation).
  - Env / tool resolution shared with :mod:`epione.upstream`:
    :func:`tool_path`, :func:`check_tools`, :data:`ATAC_TOOLS`,
    :data:`RNA_TOOLS`, :data:`MOTIF_TOOLS`.

Backward-compatible aliases live at :mod:`epione.bulk` (re-exports
this module's surface).
"""
from __future__ import annotations

from ._bigwig import (
    bigwig,
    plot_matrix,
    plot_matrix_line,
    plotloc,
    gene_expression_from_bigwigs,
)
from ._getScorePerBigWigBin import *  # noqa: F401,F403

from ._footprint_archr import footprint_archr, bam_to_fragments_bulk
from ._env import tool_path, check_tools, ATAC_TOOLS, RNA_TOOLS, MOTIF_TOOLS
from ._motif import run_homer_motifs, find_motifs_genome

__all__ = [
    "bigwig",
    "plot_matrix",
    "plot_matrix_line",
    "plotloc",
    "gene_expression_from_bigwigs",
    "footprint_archr",
    "bam_to_fragments_bulk",
    "tool_path",
    "check_tools",
    "ATAC_TOOLS",
    "RNA_TOOLS",
    "MOTIF_TOOLS",
    "run_homer_motifs",
    "find_motifs_genome",
]
