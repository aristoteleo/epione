"""Pure-Python infrastructure — no I/O, no subprocess, no per-modality logic.

Subpackage map:

    :class:`Genome`              core.genome     reference-genome class
    :func:`register_datasets`    core.genome     pooch fetcher registry
    :mod:`epione.core.console`   console / Rich-style print helpers
    :mod:`epione.core.logger`    standard-lib logger configuration
    :mod:`epione.core.motifs`    PWM I/O + scanning (find_motifs_genome, ...)
    :mod:`epione.core.regions`   BED interval ops (OneRegion, RegionList)
    :mod:`epione.core.utilities` numpy / scipy / pandas helpers
    :mod:`epione.core._sampling` peak / region sampling helpers
    :mod:`epione.core._compat`   numpy-2.x / scanpy-1.10 compat shims
    :mod:`epione.core._findgenes`gene / coordinate annotation lookup

Cython kernels live here too:
    :mod:`epione.core._footprint_cython` ATAC bias-correct kernel
    :mod:`epione.core.signals`           rolling-window signal ops
    :mod:`epione.core.sequences`         k-mer / DNA helpers

Pre-built reference *instances* live in :mod:`epione.data`;
example datasets live in :mod:`epione.datasets`.
"""
from __future__ import annotations

# Public surface (kept narrow — submodules expose the rest).
from .genome import Genome, register_datasets
from ._findgenes import find_genes, Annotation
from ._sampling import (
    expression_matched_sample,
    distance_to_nearest_peak,
    filter_distal_peaks,
    classify_peaks_by_overlap,
)
from ._compat import obs_to_pandas, var_to_pandas

from . import console, logger, motifs, regions, utilities

__all__ = [
    "Genome",
    "register_datasets",
    "find_genes",
    "Annotation",
    "expression_matched_sample",
    "distance_to_nearest_peak",
    "filter_distal_peaks",
    "classify_peaks_by_overlap",
    "obs_to_pandas",
    "var_to_pandas",
    "console",
    "logger",
    "motifs",
    "regions",
    "utilities",
]
