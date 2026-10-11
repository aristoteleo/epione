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
]
