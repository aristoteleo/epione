"""Pipeline-orchestration layer — FASTQ → BAM → bigwig / pairs → cool.

This subpackage contains the *upstream* of every analysis: anything
that runs an external aligner / format-converter / peak-caller and
produces a derived file. Pure I/O readers live in :mod:`epione.io`;
in-memory analysis lives in the modality packages
(:mod:`epione.bulk.hic`, :mod:`epione.single.atac`, ...).

Phase 1 entry points (Hi-C-specific; ATAC bowtie2/bwa/MACS2 land in
PR 2 when :mod:`epione.align` migrates here):

    * :func:`pairs_from_bam` — Hi-C BAM → pairs.gz via pairtools
    * :func:`pairs_to_cool`  — pairs.gz → ``.cool`` via cooler cload
    * :data:`HIC_TOOLS`      — required CLIs for the Hi-C upstream chain
"""
from __future__ import annotations

from ._pairs import pairs_from_bam, pairs_to_cool, HIC_TOOLS

__all__ = [
    "pairs_from_bam",
    "pairs_to_cool",
    "HIC_TOOLS",
]
