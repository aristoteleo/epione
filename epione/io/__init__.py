"""Pure-format readers / writers — no subprocess, no analysis.

Public API:

    :func:`read_ATAC_10x`         CellRanger / 10x ATAC matrix → AnnData
    :func:`read_gtf`              GTF / GFF3 → DataFrame
    :func:`read_features`         feature/peak file readers
    :func:`get_gene_annotation`   coordinate ↔ gene annotation
    :func:`convert_gff_to_gtf`    format conversion
    :func:`save`, :func:`load`    pickle / h5ad cache helpers
    :func:`cached`                memoising decorator (writes to disk)
    :func:`read_10x_multiome`     cellranger-arc Multiome output → (rna, atac)
"""
from __future__ import annotations

from ._read import (
    read_ATAC_10x,
    read_gtf,
    read_features,
    get_gene_annotation,
    convert_gff_to_gtf,
)
from ._helpers import save, load, cached
from ._multiome import read_10x_multiome

__all__ = [
    "read_ATAC_10x",
    "read_gtf",
    "read_features",
    "get_gene_annotation",
    "convert_gff_to_gtf",
    "save",
    "load",
    "cached",
    "read_10x_multiome",
]
