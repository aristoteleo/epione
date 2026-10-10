"""Bridges to external sequence→function models.

Distinct from :mod:`epione.external`, which vendors self-contained
algorithms into the tree. Subpackages here adapt Epione's data model to
a *third-party trained model* that lives in its own package and usually
its own environment, so each one is imported only on demand and none is
imported by ``import epione``.

Currently:

``seq2print``
    scPrinter's sequence -> multi-scale footprint model, with cell-state
    LoRA. Model blocks and training targets match scPrinter exactly.

``scooby``
    Single-cell sequence→profile prediction (Borzoi backbone, LoRA,
    cell-state-conditioned decoder). The target and fold-preparation
    layers work without torch; the model layers need a torch
    environment.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_SUBPACKAGES = ("scooby", "seq2print")

__all__ = list(_SUBPACKAGES)


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__


if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from . import scooby  # noqa: F401
