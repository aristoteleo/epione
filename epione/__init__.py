"""epione top-level package.

Keep __init__ lightweight to avoid importing heavy optional dependencies at import time.
Subpackages (bulk, pp, single, pl, utils) can be imported on demand:
    import epione.bulk
    import epione.pp
"""

__all__ = [
    'bulk',
    'pp',
    'single',
    'pl',
    'utils',
]
