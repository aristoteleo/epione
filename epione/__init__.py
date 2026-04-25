"""epione top-level package.

Keep __init__ lightweight to avoid importing heavy optional dependencies at import time.
Subpackages (bulk, pp, single, pl, utils) can be imported on demand:
    import epione.bulk
    import epione.pp
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
# Filter by message so we don't need to import Biopython (which itself
# prints a BiopythonWarning when imported from inside a source tree).
_warnings.filterwarnings(
    "ignore",
    message=r"You may be importing Biopython from inside the source tree.*",
)
_warnings.filterwarnings("ignore", module=r"Bio\..*")

from . import (
    align,
    bulk,
    hic,
    pp,
    sc_hic,
    single,
    pl,
    tl,
    utils,
)

__all__ = [
    'bulk',
    'align',
    'hic',
    'pp',
    'sc_hic',
    'single',
    'pl',
    'tl',
    'utils',
]
