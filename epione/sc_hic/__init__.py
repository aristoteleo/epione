"""DEPRECATED: ``epione.sc_hic`` was renamed to ``epione.single.hic`` in v0.4.

The old import path still works but emits a :class:`DeprecationWarning`.
This alias will be removed in v0.5; please update tutorials / scripts to
``epione.single.hic`` (same flat API, just a different namespace).
"""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "epione.sc_hic was renamed to epione.single.hic in v0.4 — please "
    "update imports. This alias will be removed in v0.5.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the entire flat API from the new location.
from epione.single.hic import (  # noqa: F401
    load_cool_collection,
    load_scool_cells,
    impute_cell_chromosome,
    impute_cells,
    embedding,
    plot_embedding,
    plot_cell_contacts,
)

__all__ = [
    "load_cool_collection",
    "load_scool_cells",
    "impute_cell_chromosome",
    "impute_cells",
    "embedding",
    "plot_embedding",
    "plot_cell_contacts",
]
