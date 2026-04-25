"""Pure-Python infrastructure — no I/O, no subprocess, no per-modality logic.

Future home of:

    * :class:`Genome` class (currently :mod:`epione.utils._genome`)
    * PWM operations (currently :mod:`epione.utils.motifs`)
    * BED interval ops (currently :mod:`epione.utils.regions`)
    * Console / logger / numpy-scipy compat shims
      (currently :mod:`epione.utils.console`, ``logger``, ``_compat``)

Pre-built reference *instances* (e.g. ``GRCh38``, ``GRCm39``) live in
:mod:`epione.data`; downloadable example datasets live in
:mod:`epione.datasets`.

Populated incrementally during the v0.4 refactor.
"""
from __future__ import annotations

__all__: list[str] = []
