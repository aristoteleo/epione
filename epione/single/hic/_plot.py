"""DEPRECATED: sc-Hi-C plotting helpers moved to :mod:`epione.pl` in v0.4 PR3.

* :func:`plot_embedding`     → :mod:`epione.pl._embedding`
* :func:`plot_cell_contacts` → :mod:`epione.pl._contact`

Re-exported here so :mod:`epione.single.hic` users keep working.
"""
from __future__ import annotations

from epione.pl._embedding import plot_embedding  # noqa: F401
from epione.pl._contact import plot_cell_contacts  # noqa: F401
