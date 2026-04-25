"""DEPRECATED: ``epione.align`` was renamed to ``epione.upstream`` in v0.4.

The old import path still works but emits a :class:`DeprecationWarning`.
This alias will be removed in v0.5; please update tutorials / scripts
to ``epione.upstream`` (same flat API; submodules ``bowtie2``,
``bwa_mem2``, ``samtools``, ``reference``, ``bigwig``, ``macs2``,
``atac``, ``pipeline``, ``fastq``, and ``_env`` all live there now).
"""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "epione.align was renamed to epione.upstream in v0.4 — please "
    "update imports. This alias will be removed in v0.5.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export submodules so ``epione.align.bowtie2`` etc. still resolve.
from epione.upstream import (  # noqa: F401
    bowtie2, bwa_mem2, samtools, reference, bigwig, macs2, atac, pipeline,
    fastq,
)
from epione.upstream import _env  # noqa: F401  (test references it)

# Re-export the entire flat function surface.
from epione.upstream import (  # noqa: F401
    ensure_fasta_unzipped,
    ensure_fasta_index,
    ensure_chrom_sizes,
    ensure_aligner_index,
    prepare_reference,
    fetch_genome_fasta,
    fetch_genome_annotation,
    trim_fastq_pair,
    sort_bam, index_bam, merge_bams, filter_bam,
    bam_to_bigwig,
    call_peaks_macs2,
    shift_atac_bam,
    bam_to_frags,
    tool_path, check_tools, build_env, run_cmd, ensure_dir,
    resolve_executable,
    ATAC_TOOLS, RNA_TOOLS, MOTIF_TOOLS, HIC_TOOLS,
)

__all__ = [
    "bowtie2", "bwa_mem2", "samtools", "reference", "bigwig", "macs2",
    "atac", "pipeline", "fastq",
    "ensure_fasta_unzipped", "ensure_fasta_index", "ensure_chrom_sizes",
    "ensure_aligner_index", "prepare_reference", "fetch_genome_fasta",
    "fetch_genome_annotation", "trim_fastq_pair",
    "sort_bam", "index_bam", "merge_bams", "filter_bam",
    "bam_to_bigwig", "call_peaks_macs2", "shift_atac_bam", "bam_to_frags",
    "tool_path", "check_tools", "build_env", "run_cmd", "ensure_dir",
    "resolve_executable",
    "ATAC_TOOLS", "RNA_TOOLS", "MOTIF_TOOLS", "HIC_TOOLS",
]
