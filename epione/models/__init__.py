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

``chrombpnet``
    ChromBPNet's bias-factorised base-resolution accessibility model,
    rewritten in PyTorch from the official TensorFlow code; official
    ``.h5`` weights and ENCODE releases load without TensorFlow.

``grelu``
    A bridge to gReLU (Genentech): Epione models become gReLU models, so
    variant-effect prediction, ISM and attributions run on them.

``scbasset``
    scBasset's sequence -> per-cell accessibility CNN, rewritten in
    PyTorch from the official TensorFlow code; its last layer is a cell
    embedding (``epi.tl.scbasset``).

``scooby``
    Single-cell sequence→profile prediction (Borzoi backbone, LoRA,
    cell-state-conditioned decoder). The target and fold-preparation
    layers work without torch; the model layers need a torch
    environment.

Pretrained weights of every family are listed and loaded through one
registry, :mod:`epione.models.registry` (no torch needed to list):

>>> epi.models.list_pretrained(organism="human")
>>> net = epi.models.load_pretrained("chrombpnet_encode_k562", kind="bias")
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_SUBPACKAGES = ("chrombpnet", "grelu", "registry", "scbasset", "scooby", "seq2print")
_REGISTRY_API = ("ChecksumError", "PretrainedEntry", "fetch_pretrained", "get_pretrained",
                 "list_pretrained", "load_pretrained", "pretrained_cache_dir", "register",
                 "register_family", "unregister")

__all__ = list(_SUBPACKAGES) + list(_REGISTRY_API)


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        return importlib.import_module(f".{name}", __name__)
    if name in _REGISTRY_API:
        return getattr(importlib.import_module(".registry", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__


if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from . import scooby  # noqa: F401
