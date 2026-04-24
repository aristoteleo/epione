"""Alignment and upstream preprocessing utilities."""

from . import bowtie2, bwa_mem2, samtools, reference, bigwig, macs2, atac, pipeline, fastq

from .reference import (
    ensure_fasta_unzipped,
    ensure_fasta_index,
    ensure_chrom_sizes,
    ensure_aligner_index,
    prepare_reference,
    fetch_genome_fasta,
    fetch_genome_annotation,
)
from .fastq import trim_fastq_pair
from .samtools import sort_bam, index_bam, merge_bams, filter_bam
from .bigwig import bam_to_bigwig
from .macs2 import call_peaks_macs2
from .atac import shift_atac_bam
from .pipeline import bam_to_frags
from ._env import (
    tool_path, check_tools, build_env, run_cmd, ensure_dir,
    resolve_executable,
    ATAC_TOOLS, RNA_TOOLS, MOTIF_TOOLS, HIC_TOOLS,
)

__all__ = [
    "bowtie2",
    "bwa_mem2",
    "samtools",
    "reference",
    "bigwig",
    "macs2",
    "atac",
    "pipeline",
    "fastq",
    "ensure_fasta_unzipped",
    "ensure_fasta_index",
    "ensure_chrom_sizes",
    "ensure_aligner_index",
    "prepare_reference",
    "fetch_genome_fasta",
    "fetch_genome_annotation",
    "trim_fastq_pair",
    "sort_bam",
    "index_bam",
    "merge_bams",
    "filter_bam",
    "bam_to_bigwig",
    "call_peaks_macs2",
    "shift_atac_bam",
    "bam_to_frags",
    "tool_path",
    "check_tools",
    "build_env",
    "run_cmd",
    "ensure_dir",
    "resolve_executable",
    "ATAC_TOOLS",
    "RNA_TOOLS",
    "MOTIF_TOOLS",
    "HIC_TOOLS",
]
