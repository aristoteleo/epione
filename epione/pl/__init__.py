from ._metric import frag_size_distr, tss_enrichment,fragment_histogram,plot_joint
from ._distance import cumulative_distance
from ._motif import homer_motif_table
from ._differential import volcano, ma_plot
from ._base import *

from ._embedding import (
    embedding, umap, tsne, pca, diffmap, draw_graph,
    plot_embedding,  # sc-Hi-C cell-embedding scatter (v0.4 PR3)
)

from ._contact import (
    plot_contact_matrix,
    plot_contact_triangle,
    plot_decay_curve,
    plot_coverage,
    plot_cell_contacts,
    plot_compartments,
    plot_saddle,
    plot_insulation,
    plot_loops,
    plot_apa,
    plot_correlation_heatmap,
)

from ._peak2gene import plot_peak2gene
from ._trajectory import plot_trajectory, plot_trajectory_heatmap
from ._browser_track import plot_browser_track, browser_track_coverage
from ._igv import igv_browser, IGVBrowser
from ._groups import plot_groups, plot_confusion_matrix
from ._var_dev import var_dev, tf_regulators
from ._marker_heatmap import (
    marker_heatmap, marker_heatmap_matrix, peak2gene_heatmap, prepare_peak2gene_heatmap,
)
from ._enrich_heatmap import enrich_heatmap
from ._spatial import spatial, spatial_qc

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


def plot_tf_binding_score(*args, **kwargs):
    from ..tl._print_plot import plot_tf_binding_score as _fp
    return _fp(*args, **kwargs)


def plot_hint_differential(*args, **kwargs):
    from ..tl._hint_plot import plot_hint_differential as _fp
    return _fp(*args, **kwargs)


def plot_hint_profile(*args, **kwargs):
    from ..tl._hint_plot import plot_hint_profile as _fp
    return _fp(*args, **kwargs)

__all__ = [
    "spatial", "spatial_qc",
    "plot_tf_binding_score",
    "plot_hint_differential", "plot_hint_profile",
    "frag_size_distr", "tss_enrichment", "fragment_histogram", "plot_joint",
    "cumulative_distance",
    "homer_motif_table",
    "volcano",
    "ma_plot",
    "embedding", "umap", "tsne", "pca", "diffmap", "draw_graph",
    "plot_embedding",
    "plot_contact_matrix", "plot_contact_triangle",
    "plot_decay_curve", "plot_coverage",
    "plot_cell_contacts",
    "plot_compartments", "plot_saddle",
    "plot_insulation",
    "plot_loops", "plot_apa",
    "plot_correlation_heatmap",
    "plot_peak2gene",
    "plot_browser_track", "browser_track_coverage",
    "igv_browser", "IGVBrowser",
    "enrich_heatmap",
    "plot_footprints",
    "plot_multi_scale_footprint",
    "plot_multi_scale_footprint_region",
    "plot_trajectory",
    "plot_trajectory_heatmap",
    "plot_groups",
    "plot_confusion_matrix",
    "var_dev",
    "tf_regulators",
    "marker_heatmap", "marker_heatmap_matrix",
    "peak2gene_heatmap", "prepare_peak2gene_heatmap",
]