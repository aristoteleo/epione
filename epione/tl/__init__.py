
# ``_bindetect`` / ``_atacorrect`` depend on a Cython ``signals.pyx``
# extension built against an older numpy ABI; they're lazy-imported so
# a numpy 2.x environment can still use the core preprocess / chromVAR
# / gene-score pipeline. Call ``epi.tl.bindetect(...)`` etc. and the
# extension will be loaded (and fail informatively) only at that point.
import importlib

def __getattr__(name):
    if name == "bindetect":
        return importlib.import_module("._bindetect", __name__).bindetect
    if name in ("AtacBias", "atacorrect_core", "atacorrect",
                "save_atacorrect_results_to_bigwig"):
        mod = importlib.import_module("._atacorrect", __name__)
        return getattr(mod, name)
    if name in _FOOTPRINT_ATTRS:
        mod = importlib.import_module("._footprint", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from ._score_bigwig import (
    FootprintScorer, score_bigwig_core, score_bigwig,
    calculate_aggregate_scores, compare_methods
)

from ._plotting_tobias import (
    plot_aggregate, plot_aggregate_tobias
)

from ._motif_tools import (
    MotifMatrix, FormatMotifs, ClusterMotifs,
    format_motifs, cluster_motifs
)
from ._network_tools import (
    FragmentFilter, NetworkBuilder,
    filter_fragments, create_network
)

from ._plotting import (
    FootprintPlotter, PlotTracks,
    plot_heatmap, plot_tracks
)

# ``_footprint`` also pulls in ``_bindetect_functions`` (which needs the
# Cython ``signals.pyx``). Lazy-load it for the same reason.
_FOOTPRINT_ATTRS = {
    "get_footprints", "plot_footprints", "FootprintResult",
    "getFootprints", "plotFootprints",
}

from ._reducedimension import lsi

from ._iterative_lsi import iterative_lsi

from ._peak_to_gene import peak_to_gene

from ._coaccessibility import coaccessibility

from ._marker_features import find_marker_features

from ._motif_matrix import add_motif_matrix
from ._motif_database import build_motif_database, query_motif_database
from ._background_peaks import add_background_peaks
from ._chromvar import compute_deviations
from ._gene_score import add_gene_score_matrix

from ._umap import umap

from ._clusters import clusters


__all__ = [
    "bindetect",
    "AtacBias", "atacorrect_core", "atacorrect", 
    "save_atacorrect_results_to_bigwig",
    "FootprintScorer", "score_bigwig_core", "score_bigwig",
    "calculate_aggregate_scores", "compare_methods",
    "plot_aggregate", "plot_aggregate_tobias",
    "MotifMatrix", "FormatMotifs", "ClusterMotifs",
    "format_motifs", "cluster_motifs",
    "FragmentFilter", "NetworkBuilder",
    "filter_fragments", "create_network",
    "FootprintPlotter", "PlotTracks",
    "plot_heatmap", "plot_tracks",
    "get_footprints", "plot_footprints", "FootprintResult",
    "getFootprints", "plotFootprints",
    "lsi",
    "iterative_lsi",
    "peak_to_gene",
    "coaccessibility",
    "find_marker_features",
    "add_motif_matrix",
    "build_motif_database",
    "query_motif_database",
    "add_background_peaks",
    "compute_deviations",
    "umap",
    "clusters",
]
