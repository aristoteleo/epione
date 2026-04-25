"""DEPRECATED: ``epione.utils`` was carved up in v0.4 PR4.

The contents moved as follows:

  =====================================  ====================================
  v0.3 location                          v0.4 canonical location
  =====================================  ====================================
  ``epione.utils._genome``               :mod:`epione.core.genome` (renamed
                                         from ``_genome``; the class is now
                                         a public symbol)
  ``epione.utils.genome`` (instances)    :mod:`epione.data.genomes`
                                         (re-exported via ``epione.data``)
  ``epione.utils.console``               :mod:`epione.core.console`
  ``epione.utils.logger``                :mod:`epione.core.logger`
  ``epione.utils.motifs``                :mod:`epione.core.motifs`
  ``epione.utils.regions``               :mod:`epione.core.regions`
  ``epione.utils.utilities``             :mod:`epione.core.utilities`
  ``epione.utils._compat``               :mod:`epione.core._compat`
  ``epione.utils._findgenes``            :mod:`epione.core._findgenes`
  ``epione.utils._sampling``             :mod:`epione.core._sampling`
  ``epione.utils._read``                 :mod:`epione.io._read`
                                         (public via :func:`epione.io.read_*`)
  ``epione.utils._io`` (save/load)       :mod:`epione.io._helpers`
                                         (public via :func:`epione.io.save`)
  Cython ``.pyx`` (signals, sequences,   :mod:`epione.core` (kept as a unit)
  _footprint_cython)
  =====================================  ====================================

The old import paths still resolve (this module re-exports the public
symbols from their new homes) but emit a :class:`DeprecationWarning`.
The shim will be removed in v0.5; please migrate to the canonical
paths.
"""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "epione.utils was carved up in v0.4 — see module docstring for the "
    "old → new mapping. This shim will be removed in v0.5.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the symbols that were previously imported into utils.__init__.
# This keeps ``from epione.utils import X`` working for the duration of v0.4.
from epione.core.genome import Genome, register_datasets  # noqa: F401
from epione.data.genomes import GRCh37, GRCh38, hg19, hg38  # noqa: F401
from epione.io._read import (  # noqa: F401
    read_ATAC_10x,
    read_gtf,
    read_features,
    get_gene_annotation,
    convert_gff_to_gtf,
)
from epione.io._helpers import save, load, cached  # noqa: F401
from epione.core._findgenes import find_genes, Annotation  # noqa: F401
from epione.core._sampling import (  # noqa: F401
    expression_matched_sample,
    distance_to_nearest_peak,
    filter_distal_peaks,
    classify_peaks_by_overlap,
)
from epione.core._compat import obs_to_pandas, var_to_pandas  # noqa: F401

# Module-level shim attributes so ``from epione.utils import console``
# / ``from epione.utils import motifs`` keep resolving.
from epione.core import console, logger, motifs, regions, utilities  # noqa: F401
# ``epi.utils.genome.hg19`` style access — instances live at
# :mod:`epione.data.genomes` since PR 4.
from epione.data import genomes as genome  # noqa: F401

# The thin merge_peaks wrapper specific to utils stays here (it's a
# pandas-tolerant shim around single.atac._call_peaks.merge_peaks).
from ._call_peaks import merge_peaks  # noqa: F401

__all__ = [
    "Genome",
    "register_datasets",
    "GRCh37",
    "GRCh38",
    "hg19",
    "hg38",
    "read_ATAC_10x",
    "read_gtf",
    "read_features",
    "get_gene_annotation",
    "convert_gff_to_gtf",
    "save",
    "load",
    "cached",
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
    "merge_peaks",
]
