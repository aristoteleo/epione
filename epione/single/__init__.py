"""Single-cell experiment analysis namespace (v0.4 layout).

Modality subpackages (call as ``epi.single.<modality>.X``):

  - :mod:`epione.single.atac` — scATAC tooling (peak calling, gene
    activity, chromVAR, coaccessibility, pseudobulk).
  - :mod:`epione.single.hic`  — sc-Hi-C tooling (scHiCluster impute,
    cell embedding, per-cell contact plots).

For backward compatibility (old code calling ``epi.single.X`` directly),
the ATAC surface is also re-exported at this level. New code should
prefer the explicit ``epi.single.atac.X`` form.
"""
from __future__ import annotations

# Modality subpackages.
from . import atac, hic

# Backward-compat re-exports of the ATAC surface (so legacy
# ``epi.single.macs3`` etc. keep working). PR 5 may remove this.
from .atac import (  # noqa: F401
    macs3,
    merge_peaks,
    pseudobulk,
    pseudobulk_with_fragments,
    read_fragments_from_file,
    read_fragments_with_dask_parallel,
    check_performance_backends,
    get_performance_recommendations,
    install_performance_backend,
    quick_install_pandarallel,
    add_dna_sequence,
    match_motif,
)

__all__ = [
    "atac",
    "hic",
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
