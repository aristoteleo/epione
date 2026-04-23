# Lightweight imports (no heavy deps)
from ._fastq2frags import (
    bulk_fastq_to_frag,
    align_fastq_to_bam,
    filter_bam,
    bam_to_frags,
    ensure_aligner_index,
    fetch_genome_fasta,
    fetch_dataset_fastq_pairs,
    list_dataset_fastqs_tar,
    extract_fastqs_from_tar,
)


# Optional: bigwig visualization utilities (require numpy/pyBigWig/etc.)
from ._bigwig import bigwig, plot_matrix, plot_matrix_line, plotloc, gene_expression_from_bigwigs
from ._getScorePerBigWigBin import *  # noqa: F401,F403

# ArchR-style bulk footprint (alternative to the TOBIAS backend).
from ._footprint_archr import footprint_archr, bam_to_fragments_bulk

# Env / tool resolution for the bulk upstream pipeline.
from ._env import tool_path, check_tools, ATAC_TOOLS, RNA_TOOLS, MOTIF_TOOLS

# HOMER motif-enrichment: Perl wrapper + pure-Python reimplementation.
from ._motif import run_homer_motifs, find_motifs_genome

