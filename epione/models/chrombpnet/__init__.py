"""ChromBPNet -- bias-factorised base-resolution accessibility models.

ChromBPNet (Pampari *et al.*, 2023; kundajelab/chrombpnet, MIT) predicts
base-resolution ATAC/DNase profiles and counts from 2,114 bp of sequence,
with a separate convolutional model for the Tn5/DNase sequence bias so the
accessibility model learns bias-free TF syntax.

This is a PyTorch implementation of the official TensorFlow code -- no
TensorFlow or Keras is imported anywhere:

* :mod:`._model` -- :class:`BPNet`, :class:`ChromBPNet`, matching the
  official Keras layers one to one;
* :mod:`._keras` -- :func:`load_keras_h5`, official ``.h5`` weights read
  with h5py;
* :mod:`._pretrained` -- :func:`load_pretrained`, ENCODE's released
  models (GM12878, K562, HepG2, IMR-90, T-cell), md5-pinned;
* :mod:`._data` -- the official data recipe: GC-matched non-peaks,
  hyper-parameter steps, per-epoch jitter/revcomp batches; signal from a
  bigWig or straight from a fragments file;
* :mod:`._train` -- :func:`train_bias`, :func:`train_chrombpnet`, the
  official training recipe;
* :mod:`._predict` -- prediction, official metrics, DeepLIFT-SHAP
  contributions, ISM, marginal footprints.

Folds are ENCODE's, identical to :data:`epione.models.seq2print.FOLDS`.
The bias model also plugs into footprinting as a Tn5 bias backend:
``epi.tl.tn5_bias(method="chrombpnet", ...)``.
"""
from __future__ import annotations

import importlib

_ATTRS = {
    "_model": {"BPNet", "ChromBPNet", "build_bpnet", "glorot_init_", "one_hot", "predict"},
    "_keras": {"load_keras_h5"},
    "_pretrained": {"PRETRAINED", "load_pretrained", "pretrained_path"},
    "_data": {"one_hot_encode", "read_narrowpeak", "open_signal", "BigWigSignal",
              "InsertionSignal", "bias_hyperparams", "chrombpnet_hyperparams",
              "gc_matched_nonpeaks", "genomewide_gc", "load_regions", "ChromBPNetBatches",
              "filter_edge_regions", "get_seq_onehot"},
    "_train": {"train_bias", "train_chrombpnet", "fit_model", "adjust_bias_logcounts",
               "multinomial_nll", "TrainResult"},
    "_predict": {"predict_regions", "evaluate", "contribution_scores", "ism",
                 "marginal_footprint", "dinuc_shuffle", "softmax"},
}
__all__ = sorted(set().union(*_ATTRS.values()) | {"chromosome_folds", "FOLDS"})


def __getattr__(name):
    if name in ("chromosome_folds", "FOLDS"):
        return getattr(importlib.import_module("epione.models.seq2print._splits"), name)
    for mod, names in _ATTRS.items():
        if name in names:
            return getattr(importlib.import_module(f".{mod}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
