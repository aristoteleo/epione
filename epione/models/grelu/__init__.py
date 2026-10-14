"""gReLU bridge -- variant effects, ISM and attributions for Epione models.

gReLU (Genentech, MIT) is a toolkit around sequence models: a Lightning
wrapper with one prediction API, variant-effect prediction, in-silico
mutagenesis, attributions (captum, DeepLIFT-SHAP) and motif analyses, plus
a model zoo (Borzoi, Enformer). This bridge makes Epione's own sequence
models first-class gReLU models and accepts native gReLU models too:

* :func:`wrap` -- an Epione :class:`~epione.models.chrombpnet.BPNet` /
  ``ChromBPNet`` / :class:`~epione.models.seq2print.Seq2PRINT` (or any
  one-hot ``nn.Module``) as a gReLU ``LightningModel``;
* :func:`predict_variant_effects`, :func:`ism`, :func:`attributions` --
  one API across model kinds, exact pass-throughs to gReLU, local FASTA
  genomes;
* :func:`pseudobulk_anndata` -- groups x peaks training data in gReLU's
  ``AnnDataSeqDataset`` layout.

Requires ``pip install epione[grelu]``; imported only on use.
"""
from __future__ import annotations

import importlib

_NAMES = {"HEADS", "wrap", "predict_variant_effects", "ism", "attributions",
          "pseudobulk_anndata"}
__all__ = sorted(_NAMES)


def __getattr__(name):
    if name in _NAMES:
        return getattr(importlib.import_module("._bridge", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
