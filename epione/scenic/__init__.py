"""SCENIC+ in Epione: enhancer-driven gene regulatory networks from single-cell multiome data.

Ports of the SCENIC+ 1.0 pipeline (pycisTopic → pycistarget → scenicplus),
reimplemented without Ray/Dask and verified bit for bit against the upstream
packages (see ``tests/parity/scenic_*``).
"""
from ._topics_matrix import cistopic_matrix
from ._topics_dar import (
    ImputedFeatures,
    binarize_topics,
    find_diff_features,
    find_highly_variable_features,
    impute_accessibility,
    normalize_scores,
    write_region_sets,
)
from ._topics import (
    TopicModel,
    add_topic_model,
    evaluate_topic_models,
    run_topic_models,
)


from ._ranking_db import CisTargetDB, RegionIndex, read_bed_regions, target_to_query
from ._motif_annotation import load_motif_annotations, cistromes_per_region_set
from ._cistarget import MotifEnrichmentResult, run_cistarget, recovery_aucs
from ._dem import run_dem, dem_test, foreground_background_regions, p_adjust_bh
from ._menr import merge_cistromes, write_hdf5, read_hdf5
from ._motif_db import motif_score_db, scores_to_rankings
from ._data import fetch_cistarget_database, fetch_motif_annotation, jaspar_annotation

__all__ = [
    "ImputedFeatures",
    "TopicModel",
    "binarize_topics",
    "cistopic_matrix",
    "find_diff_features",
    "find_highly_variable_features",
    "impute_accessibility",
    "normalize_scores",
    "add_topic_model",
    "evaluate_topic_models",
    "run_topic_models",
    "write_region_sets",
    # cisTarget / DEM
    "CisTargetDB", "RegionIndex", "read_bed_regions", "target_to_query",
    "load_motif_annotations", "cistromes_per_region_set",
    "MotifEnrichmentResult", "run_cistarget", "recovery_aucs",
    "run_dem", "dem_test", "foreground_background_regions", "p_adjust_bh",
    "merge_cistromes", "write_hdf5", "read_hdf5",
    "motif_score_db", "scores_to_rankings",
    "fetch_cistarget_database", "fetch_motif_annotation", "jaspar_annotation",
]
