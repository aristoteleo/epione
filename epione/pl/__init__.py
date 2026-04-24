from ._metric import frag_size_distr, tss_enrichment,fragment_histogram,plot_joint
from ._distance import cumulative_distance
from ._motif import homer_motif_table
from ._differential import volcano, ma_plot
from ._base import *

from ._embedding import (
    embedding, umap, tsne, pca, diffmap, draw_graph
)

from ._peak2gene import plot_peak2gene

# Footprint plot re-exported from tl (same object — plot is the
# natural consumer of the Footprint dataclass that get_footprints
# returns). Lazy-import because _footprint pulls pyfaidx/pysam.
def plot_footprints(*args, **kwargs):
    from ..tl._footprint import plot_footprints as _fp
    return _fp(*args, **kwargs)


def plot_multi_scale_footprint(*args, **kwargs):
    from ..tl._multi_scale_footprint_plot import plot_multi_scale_footprint as _fp
    return _fp(*args, **kwargs)


def plot_multi_scale_footprint_region(*args, **kwargs):
    from ..tl._multi_scale_region import plot_multi_scale_footprint_region as _fp
    return _fp(*args, **kwargs)

__all__ = [
    "frag_size_distr", "tss_enrichment", "fragment_histogram", "plot_joint",
    "cumulative_distance",
    "homer_motif_table",
    "volcano",
    "ma_plot",
    "embedding", "umap", "tsne", "pca", "diffmap", "draw_graph",
    "plot_peak2gene",
    "plot_footprints",
    "plot_multi_scale_footprint",
    "plot_multi_scale_footprint_region",
]