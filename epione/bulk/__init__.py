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
try:
    from ._bigwig import bigwig, plot_matrix, plot_matrix_line, plotloc
    from ._getScorePerBigWigBin import *  # noqa: F401,F403
except Exception:  # ImportError and others
    # Defer heavy deps; users focusing on alignment/fragments can still import this module
    bigwig = None
    plot_matrix = None
    plot_matrix_line = None
    plotloc = None
