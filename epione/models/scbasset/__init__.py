"""scBasset -- sequence-based modelling of single-cell ATAC (Yuan & Kelley,
Nat Methods 2022; calico/scBasset, Apache-2.0).

A convolutional network reads the 1,344 bp under each peak and predicts its
accessibility in every cell; the last layer's kernel is the cell embedding.
This is a PyTorch rewrite of the official TensorFlow code -- no TensorFlow
is imported, and official ``best_model.h5`` weights load with h5py:

* :mod:`._data` -- :func:`peak_sequences`, :func:`binarize`,
  :func:`split_train_test_val` (the official preprocessing);
* :mod:`._model` -- :class:`ScBassetNet` and the official training recipe;
* :mod:`._api` -- :class:`ScBasset`: fit, cell embedding, intercept,
  denoised accessibility, motif-injection TF activity.

Most users call :func:`epione.tl.scbasset`. Needs ``torch``.
"""
from __future__ import annotations

import importlib

_ATTRS = {
    "_data": {"SEQ_LEN", "peak_table", "peak_sequences", "encode_sequences", "binarize",
              "split_train_test_val", "read_fasta"},
    "_model": {"ScBassetNet", "KerasMultiLabelAUC", "train", "predict_logits", "load_keras_h5",
               "tower_filters"},
    "_api": {"ScBasset", "motif_library", "MOTIF_URL", "MOTIF_MD5"},
}
_WHERE = {a: m for m, names in _ATTRS.items() for a in names}
__all__ = sorted(_WHERE)


def __getattr__(name):
    if name in _WHERE:
        return getattr(importlib.import_module(f".{_WHERE[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
