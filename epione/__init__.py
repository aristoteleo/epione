"""epione top-level package — v0.4 architecture.

Subpackage map (call-chain shown alongside):

    epione.bulk.{atac, hic}      bulk analyses, modality-specific
    epione.single.{atac, hic}    single-cell analyses, modality-specific
    epione.pp / .tl / .pl        cross-modality preprocessing / analysis / plotting
    epione.scenic                SCENIC+ enhancer-driven GRNs (topics → motifs → eGRNs)
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

# Canonical v0.4 subpackages, resolved on first attribute access.
#
# These used to be imported eagerly, which made ``import epione`` fail
# whenever any one subpackage's optional dependency was absent -- a
# missing pyranges or MOODS took down the entire namespace, including
# parts that did not need it. That is also why a torch-only environment
# could not reach ``epione.models``. Deferring the import keeps
# ``epione.pp``, ``from epione import tl`` and ``import epione.tl`` all
# working, while an unrelated missing dependency now surfaces only when
# the subpackage that needs it is actually touched.
_SUBPACKAGES = (
    "align",        # → migrates into ``upstream`` in PR 2
    "bulk",         # bulk.atac, bulk.hic
    "core",         # populated incrementally
    "data",         # populated incrementally
    "datasets",     # populated incrementally
    "io",           # populated incrementally
    "models",       # bridges to external sequence→function models
    "pp",
    "pl",
    "scenic",       # SCENIC+ eGRN inference (pycisTopic / pycistarget / scenicplus ports)
    "single",       # single.atac, single.hic
    "tl",
    "upstream",
    "utils",        # → migrates into core/io in PR 4
)


def __getattr__(name):
    if name in _SUBPACKAGES:
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod  # cache, so this runs once per subpackage
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBPACKAGES))

# Deprecated aliases (kept for v0.4; removed in v0.5).
# Importing them is what triggers the DeprecationWarning, so we don't
# eagerly resolve them here; users hit the warning the first time they
# write ``import epione.hic`` / ``import epione.sc_hic``.

__all__ = list(_SUBPACKAGES)
