"""Single-cell ATAC-seq analysis (v0.4 layout, populated in PR 2b).

Flat API — call as ``epi.single.atac.X``:

  - Peak calling: :func:`macs3` (per-cluster MACS3), :func:`merge_peaks`.
  - Pseudobulk: :func:`pseudobulk`, :func:`pseudobulk_with_fragments`,
    fragment-file readers, performance-backend helpers.
  - Motif annotation: :func:`add_dna_sequence`, :func:`match_motif`.
  - BAM split (per-cluster): :func:`split_bam_clusters` via internal
    plumbing.
  - pySCENIC TF-network bridges: :mod:`._pyscenic`.

Backward-compatible aliases live at :mod:`epione.single` (re-exports
this module's surface).
"""
from __future__ import annotations

from ._call_peaks import macs3, merge_peaks

from ._pseudobulk import (
    pseudobulk,
    pseudobulk_with_fragments,
    read_fragments_from_file,
    read_fragments_with_dask_parallel,
    check_performance_backends,
    get_performance_recommendations,
    install_performance_backend,
    quick_install_pandarallel,
)

from ._motif import add_dna_sequence, match_motif

__all__ = [
    "macs3",
    "merge_peaks",
    "pseudobulk",
    "pseudobulk_with_fragments",
    "read_fragments_from_file",
    "read_fragments_with_dask_parallel",
    "check_performance_backends",
    "get_performance_recommendations",
    "install_performance_backend",
    "quick_install_pandarallel",
    "add_dna_sequence",
    "match_motif",
]
