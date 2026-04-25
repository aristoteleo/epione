"""DEPRECATED: ``epione.hic`` was split in v0.4 between

* :mod:`epione.upstream` — pipeline ops (``pairs_from_bam``, ``pairs_to_cool``)
* :mod:`epione.bulk.hic` — bulk Hi-C analysis (``balance_cool``, plotting)

The old import path still works but emits a :class:`DeprecationWarning`.
This alias will be removed in v0.5.
"""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "epione.hic was split in v0.4: use epione.upstream for "
    "pairs_from_bam / pairs_to_cool and epione.bulk.hic for balance_cool "
    "+ plotting. This alias will be removed in v0.5.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the canonical surface from the new locations so user code
# still works.
from epione.upstream import pairs_from_bam, pairs_to_cool, HIC_TOOLS  # noqa: F401
from epione.bulk.hic import (  # noqa: F401
    balance_cool,
    plot_contact_matrix,
    plot_decay_curve,
    plot_coverage,
)

__all__ = [
    "pairs_from_bam",
    "pairs_to_cool",
    "HIC_TOOLS",
    "balance_cool",
    "plot_contact_matrix",
    "plot_decay_curve",
    "plot_coverage",
]
