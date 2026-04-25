"""Pipeline-orchestration layer — FASTQ → BAM → bigwig / pairs → cool.

This subpackage holds anything that runs an external aligner /
format-converter / peak-caller and produces a derived file. Pure
in-memory I/O readers live in :mod:`epione.io`; analysis on the
resulting matrices lives in the modality packages
(:mod:`epione.bulk.hic`, :mod:`epione.single.atac`, ...).

Public API:

ATAC / ChIP / generic upstream:
    :func:`trim_fastq_pair`            — fastp wrapper
    :func:`bam_to_bigwig`              — BAM → coverage bigwig
    :func:`call_peaks_macs2`           — MACS2 narrow / broad peak calling
    :func:`shift_atac_bam`             — Tn5 shift for ATAC BAMs
    :func:`bam_to_frags`               — BAM → fragment TSV / pseudobulk
    :func:`sort_bam` / :func:`index_bam` / :func:`merge_bams` / :func:`filter_bam`
    :mod:`bowtie2` / :mod:`bwa_mem2`   — aligner wrappers (call sub-functions)
    :func:`prepare_reference`, :func:`fetch_genome_fasta`, ...

Hi-C upstream:
    :func:`pairs_from_bam`             — BAM → pairs via pairtools
    :func:`pairs_to_cool`              — pairs → ``.cool`` via cooler cload
    :data:`HIC_TOOLS`                  — required CLIs for the chain

Env / tool resolution:
    :func:`tool_path`, :func:`check_tools`, :func:`build_env`,
    :func:`run_cmd`, :func:`ensure_dir`, :func:`resolve_executable`,
    :data:`ATAC_TOOLS`, :data:`RNA_TOOLS`, :data:`MOTIF_TOOLS`
"""
from __future__ import annotations

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
from ._pairs import pairs_from_bam, pairs_to_cool

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
    "pairs_from_bam",
    "pairs_to_cool",
]
