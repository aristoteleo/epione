"""epione top-level package — v0.4 architecture.

Subpackage map (call-chain shown alongside):

    epione.bulk.{atac, hic}      bulk analyses, modality-specific
    epione.single.{atac, hic}    single-cell analyses, modality-specific
    epione.pp / .tl / .pl        cross-modality preprocessing / analysis / plotting
    epione.io                    pure format read/write (BED, cool, pairs, h5ad, ...)
    epione.upstream              FASTQ → BAM → bigwig / pairs → cool pipelines
    epione.core                  pure-Python infrastructure (Genome, motifs, regions, console)
    epione.data                  reference registries (GRCh38, JASPAR, ...)
    epione.datasets              fetchable example datasets for tutorials

Subpackages are imported lazily-friendly (heavy optional deps deferred
to first use). Deprecated aliases (``epione.hic``, ``epione.sc_hic``)
still resolve in v0.4 with a :class:`DeprecationWarning`; they will be
removed in v0.5.
"""

# Suppress noisy third-party warnings that fire on import. anndata's
# own ``__init__`` triggers seven FutureWarnings from its deprecated
# top-level re-exports; Biopython warns when imported from inside a
# source tree; the Cython fallback in ``epione.tl._score_bigwig`` also
# emits a UserWarning on numpy-2.x systems. None of these are
# actionable by a tutorial user, so silence them by default.
import warnings as _warnings
_warnings.filterwarnings("ignore", category=FutureWarning, module=r"anndata\..*")
_warnings.filterwarnings("ignore", category=FutureWarning, module=r"anndata$")
_warnings.filterwarnings(
    "ignore",
    message=r"You may be importing Biopython from inside the source tree.*",
)
_warnings.filterwarnings("ignore", module=r"Bio\..*")

# Canonical v0.4 subpackages.
from . import (
    align,        # → migrates into ``upstream`` in PR 2
    bulk,         # bulk.atac, bulk.hic
    core,         # populated incrementally
    data,         # populated incrementally
    datasets,     # populated incrementally
    io,           # populated incrementally
    pp,
    pl,
    single,       # single.atac, single.hic
    tl,
    upstream,
    utils,        # → migrates into core/io in PR 4
)

# Deprecated aliases (kept for v0.4; removed in v0.5).
# Importing them is what triggers the DeprecationWarning, so we don't
# eagerly resolve them here; users hit the warning the first time they
# write ``import epione.hic`` / ``import epione.sc_hic``.

__all__ = [
    "align",
    "bulk",
    "core",
    "data",
    "datasets",
    "io",
    "pp",
    "pl",
    "single",
    "tl",
    "upstream",
    "utils",
]
