from ._data import (
    import_fragments, make_peak_matrix, make_gene_matrix, add_tile_matrix,
    concat_samples,
)
from ._fragment_store import release_fragment_store
from ._metric import (
    frag_size_distr, tsse, ensure_tabix_index,
    tss_enrichment,nucleosome_signal,select_features,
 )
from ._qc import (
    qc,scrublet
)

from ._neighbors import (
    neighbors,
)
from ._doublets import (
    archr_doublets, filter_doublets, scdblfinder, amulet, clamulet,
    combine_doublet_scores,
)



__all__ = [
    "import_fragments",
    "release_fragment_store",
    "concat_samples",
    "frag_size_distr",
    "tsse",
    "make_peak_matrix",
    "make_gene_matrix",
    "add_tile_matrix",
    "ensure_tabix_index",
    "tss_enrichment",
    "nucleosome_signal",
    "select_features",
    "qc",
    "scrublet",
    "neighbors",
    # doublets
    "archr_doublets",
    "filter_doublets",
    "scdblfinder",
    "amulet",
    "clamulet",
    "combine_doublet_scores",
]