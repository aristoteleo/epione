from ._call_peaks import macs3, merge_peaks

from ._pseudobulk import (
    pseudobulk,
    pseudobulk_with_fragments,
    read_fragments_from_file,
    read_fragments_with_dask_parallel,
    check_performance_backends,
    get_performance_recommendations,
    install_performance_backend,
    quick_install_pandarallel
)

from ._motif import (add_dna_sequence,match_motif)

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