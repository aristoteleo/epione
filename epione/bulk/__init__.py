"""Bulk-experiment analysis namespace (v0.4 layout).

Modality subpackages (call as ``epi.bulk.<modality>.X``):

  - :mod:`epione.bulk.atac` — bulk ATAC / ChIP / CUT&RUN tooling
    (footprint, BigWig matrix, motif enrichment).
  - :mod:`epione.bulk.hic`  — bulk Hi-C analysis (matrix balance,
    contact-matrix plot, P(s) curve, coverage diagnostics).

For backward compatibility (old code calling ``epi.bulk.X`` directly),
the ATAC surface is also re-exported at this level. New code should
prefer the explicit ``epi.bulk.atac.X`` form.
"""
from __future__ import annotations

# Modality subpackages.
from . import atac, hic

# Backward-compat re-exports of the ATAC surface (so legacy
# ``epi.bulk.bigwig`` etc. keep working). PR 5 may remove this.
from .atac import (  # noqa: F401
    bigwig,
    plot_matrix,
    plot_matrix_line,
    plotloc,
    gene_expression_from_bigwigs,
    footprint_archr,
    bam_to_fragments_bulk,
    tool_path,
    check_tools,
    ATAC_TOOLS,
    RNA_TOOLS,
    MOTIF_TOOLS,
    run_homer_motifs,
    find_motifs_genome,
)
from .atac._getScorePerBigWigBin import *  # noqa: F401,F403

__all__ = [
    "atac",
    "hic",
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
