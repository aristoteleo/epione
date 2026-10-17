
# ``_bindetect`` / ``_atacorrect`` / ``_score_bigwig`` depend (directly
# or transitively) on the Cython ``signals.pyx`` / ``_footprint_cython``
# extension built against an older numpy ABI. Touching any of them at
# package-import time prints a numpy 1.x/2.x ABI banner to stderr on
# modern numpy, which makes the first notebook cell look scary. All of
# them are therefore lazy-imported so a numpy 2.x environment can still
# use the core preprocess / chromVAR / gene-score pipeline silently;
# the extension only loads (and fails informatively) when you actually
# call into footprint / score_bigwig functionality.
import importlib

# ``_footprint`` pulls in ``_bindetect_functions`` (which also needs
# the Cython ``signals.pyx``).
_FOOTPRINT_ATTRS = {
    "get_footprints", "plot_footprints",
    "getFootprints", "plotFootprints",
    "Footprint", "compute_tn5_bias_table",
}
_MULTI_SCALE_FP_ATTRS = {
    "multi_scale_footprint", "MultiScaleFootprint", "footprint_score",
}
_MULTI_SCALE_FP_PLOT_ATTRS = {
    "plot_multi_scale_footprint",
}
_MULTI_SCALE_REGION_ATTRS = {
    "multi_scale_footprint_region", "plot_multi_scale_footprint_region",
    "rank_sites_by_cut_density", "MultiScaleRegion",
}
_SCORE_BIGWIG_ATTRS = {
    "FootprintScorer", "score_bigwig_core", "score_bigwig",
    "calculate_aggregate_scores", "compare_methods",
}
_PLOTTING_TOBIAS_ATTRS = {"plot_aggregate", "plot_aggregate_tobias"}
# PRINT / scPrinter: pluggable Tn5 bias, learned null, TF binding score.
_TN5_BIAS_ATTRS = {
    "tn5_bias", "Tn5Bias", "PrintBias", "PrecomputedBias", "KmerBias", "ChromBPNetBias",
}
_PRINT_ATTRS = {
    "tf_binding_score", "TFBindingResult", "DispersionModel",
    "print_footprint_score", "print_footprint_pvalues",
    "fetch_print_model", "PRINT_SCALES",
}
_PRINT_PLOT_ATTRS = {"plot_tf_binding_score"}
# Thermodynamic occupancy: accessibility-gated Boltzmann binding, per-site score.
_OCCUPANCY_ATTRS = {
    "tf_occupancy", "OccupancyResult", "occupancy_from_counts",
    "neighbourhood_logsumexp", "site_insertion_counts", "OCCUPANCY_DEFAULTS",
}
# Default per-site TF binding entry point (method='occupancy' | 'print').
_TF_BINDING_ATTRS = {"tf_binding", "TF_BINDING_METHODS"}
# HINT-ATAC (RGT rgt-hint): bias-corrected signal, HMM footprints, differential.
_HINT_ATTRS = {
    "HintBias", "HintHMM", "HintDifferentialResult", "fetch_hint_file",
    "hint_differential", "hint_footprints", "hint_match_motifs", "hint_signal",
    "load_hint_bias_table", "load_hint_hmm",
}
_HINT_PLOT_ATTRS = {"plot_hint_differential", "plot_hint_profile"}
# SpatialGlue (AGPL-3.0 port; needs torch) -- spatial multi-omics integration.
_SPATIALGLUE_ATTRS = {
    "spatialglue", "spatialglue_preprocess", "spatialglue_graphs",
    "lsi_spatialglue", "clr_normalize_each_cell",
}


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
    if name in _MULTI_SCALE_FP_ATTRS:
        mod = importlib.import_module("._multi_scale_footprint", __name__)
        return getattr(mod, name)
    if name in _MULTI_SCALE_FP_PLOT_ATTRS:
        mod = importlib.import_module("._multi_scale_footprint_plot", __name__)
        return getattr(mod, name)
    if name in _MULTI_SCALE_REGION_ATTRS:
        mod = importlib.import_module("._multi_scale_region", __name__)
        return getattr(mod, name)
    if name in _TN5_BIAS_ATTRS:
        mod = importlib.import_module("._tn5_bias", __name__)
        return getattr(mod, name)
    if name in _PRINT_ATTRS:
        mod = importlib.import_module("._print", __name__)
        return getattr(mod, name)
    if name in _PRINT_PLOT_ATTRS:
        mod = importlib.import_module("._print_plot", __name__)
        return getattr(mod, name)
    if name in _OCCUPANCY_ATTRS:
        mod = importlib.import_module("._occupancy", __name__)
        return getattr(mod, name)
    if name in _TF_BINDING_ATTRS:
        mod = importlib.import_module("._tf_binding", __name__)
        return getattr(mod, name)
    if name in _HINT_ATTRS:
        mod = importlib.import_module("._hint", __name__)
        return getattr(mod, name)
    if name in _HINT_PLOT_ATTRS:
        mod = importlib.import_module("._hint_plot", __name__)
        return getattr(mod, name)
    if name in _SPATIALGLUE_ATTRS:
        mod = importlib.import_module("._spatialglue", __name__)
        return getattr(mod, name)
    if name in _SCORE_BIGWIG_ATTRS:
        mod = importlib.import_module("._score_bigwig", __name__)
        return getattr(mod, name)
    if name in _PLOTTING_TOBIAS_ATTRS:
        mod = importlib.import_module("._plotting_tobias", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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

from ._reducedimension import lsi

from ._iterative_lsi import iterative_lsi
from ._birmt import birmt

from ._integrate import integrate
from ._transfer_labels import transfer_labels
from ._joint_embedding import joint_embedding

from ._peak_to_gene import peak_to_gene

from ._coaccessibility import coaccessibility
from ._e2g import e2g_links, e2g_features, count_insertions
from ._contact import (
    contact_links, contact_gene_score, fit_contact_gamma, powerlaw_contact,
)

from ._marker_features import find_marker_features

from ._motif_matrix import add_motif_matrix
from ._motif_database import build_motif_database, query_motif_database
from ._background_peaks import add_background_peaks
from ._chromvar import compute_deviations
from ._tf_activity import tf_activity, add_motif_occupancy
from ._gene_score import add_gene_score_matrix
from ._peak_anno_enrichment import (
    peak_anno_enrichment, custom_enrichment,
    add_peak_annotations, add_archr_annotations,
)

from ._umap import umap

from ._clusters import clusters

from ._trajectory import (
    add_trajectory,
    get_trajectory,
    trajectory_heatmap_matrix,
    correlate_trajectories,
)
from ._impute import add_impute_weights, impute_matrix
from ._tf_regulators import (
    var_deviations, correlate_matrices, positive_tf_regulators, group_means,
)

from ._differential import (
    differential_peaks,
    count_reads_in_peaks,
    peak_signal_matrix,
)

from ._harmony import harmony, run_harmony
from ._ot_batch import ot_batch, batch_correct
from ._module_score import add_module_score, confusion_matrix
from ._bulk_projection import LSIModel, project_bulk_atac, deconvolve_bulk
from ._multimodal import combined_dims, wnn, multivi
from ._tsne import tsne
from ._spatial import spatial_neighbors, spatial_smooth, spatial_domains, spatial_moran
from ._cnv import (
    cnv, cnv_make_windows, cnv_count_matrix, cnv_subclones,
    cnv_karyotype_measures, cnv_burden,
)
from ._atac_models import scbasset, peakvi


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
    "get_footprints", "plot_footprints",
    "getFootprints", "plotFootprints",
    "multi_scale_footprint", "MultiScaleFootprint",
    "footprint_score", "plot_multi_scale_footprint",
    "multi_scale_footprint_region", "plot_multi_scale_footprint_region",
    "tn5_bias", "Tn5Bias", "PrintBias", "PrecomputedBias", "KmerBias", "ChromBPNetBias",
    "tf_binding_score", "TFBindingResult", "DispersionModel",
    "print_footprint_score", "print_footprint_pvalues",
    "fetch_print_model", "PRINT_SCALES", "plot_tf_binding_score",
    "tf_occupancy", "OccupancyResult", "occupancy_from_counts",
    "neighbourhood_logsumexp", "site_insertion_counts", "OCCUPANCY_DEFAULTS",
    "rank_sites_by_cut_density", "MultiScaleRegion",
    "HintBias", "HintHMM", "HintDifferentialResult", "fetch_hint_file",
    "hint_differential", "hint_footprints", "hint_match_motifs", "hint_signal",
    "load_hint_bias_table", "load_hint_hmm",
    "plot_hint_differential", "plot_hint_profile",
    "lsi",
    "iterative_lsi",
    "birmt",
    "integrate",
    "transfer_labels",
    "joint_embedding",
    "peak_to_gene",
    "coaccessibility",
    "contact_links",
    "e2g_links",
    "e2g_features",
    "count_insertions",
    "contact_gene_score",
    "fit_contact_gamma",
    "powerlaw_contact",
    "find_marker_features",
    "add_motif_matrix",
    "build_motif_database",
    "query_motif_database",
    "add_background_peaks",
    "compute_deviations",
    "tf_activity",
    "add_motif_occupancy",
    "add_impute_weights",
    "impute_matrix",
    "peak_anno_enrichment", "custom_enrichment",
    "add_peak_annotations", "add_archr_annotations",
    "umap",
    "clusters",
    "differential_peaks",
    "count_reads_in_peaks",
    "peak_signal_matrix",
    "harmony",
    "run_harmony",
    "ot_batch",
    "batch_correct",
    "add_trajectory",
    "get_trajectory",
    "trajectory_heatmap_matrix",
    "correlate_trajectories",
    "add_module_score",
    "confusion_matrix",
    "var_deviations",
    "correlate_matrices",
    "positive_tf_regulators",
    "group_means",
    "LSIModel",
    "project_bulk_atac",
    "deconvolve_bulk",
    "combined_dims",
    "wnn",
    "multivi",
    "tsne",
    "spatial_neighbors", "spatial_smooth", "spatial_domains", "spatial_moran",
    "spatialglue", "spatialglue_preprocess", "spatialglue_graphs",
    "lsi_spatialglue", "clr_normalize_each_cell",
    "cnv", "cnv_make_windows", "cnv_count_matrix", "cnv_subclones",
    "cnv_karyotype_measures", "cnv_burden",
    "scbasset",
    "peakvi",
]
