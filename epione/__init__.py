"""epione top-level package.

Keep __init__ lightweight to avoid importing heavy optional dependencies at import time.
Subpackages (bulk, pp, single, pl, utils) can be imported on demand:
    import epione.bulk
    import epione.pp
"""

from . import (
    bulk,
    pp,
    single,
    pl,
    tl,
    utils,
)

__all__ = [
    'bulk',
    'pp',
    'single',
    'pl',
    'tl',
    'utils',
]
