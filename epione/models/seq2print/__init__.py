"""seq2PRINT -- predict PRINT footprints from DNA, per cell state.

scPrinter's seq2PRINT is a dilated CNN that reads 1,840 bp of sequence
and predicts PRINT's multi-scale footprint z-scores over the central
800 bp (99 scales) plus the window's log coverage. Fine-tuned with
cell-state LoRA it becomes a single-cell sequence-to-footprint model, and
its attributions give base-resolution TF binding.

Layout:

* :mod:`._modules`, :mod:`._footprint` -- scPrinter's network blocks and
  training targets, vendored verbatim (MIT) so models and targets match
  scPrinter exactly and checkpoints load either way.
* :mod:`._model` -- :func:`build_seq2print`, :class:`Seq2PRINT`,
  :meth:`Seq2PRINT.add_lora`.
* :mod:`._dataset` -- :class:`Seq2PRINTDataset`, windows read straight
  from fragments (no per-group bigWig export).
* :mod:`._train` -- :func:`fit`, :func:`evaluate` (scPrinter's loss,
  metrics and optimiser recipe, without the CLI/config/wandb layer).
* :mod:`._splits` -- :func:`chromosome_folds`, scPrinter's folds.
* :mod:`._interpret` -- :func:`delta_effects` (planted-motif effects on
  predicted footprints at every scale).

Requires torch. Models are trained per dataset; the tutorial's are on
Hugging Face (:func:`load_pretrained`).
"""
from __future__ import annotations

import importlib

_ATTRS = {
    "_model": {"Seq2PRINT", "build_seq2print", "LORA_TARGETS"},
    "_footprint": {"footprint_target_model", "multiscaleFoot", "dispModel"},
    "_dataset": {"Seq2PRINTDataset", "summits_from_peaks"},
    "_train": {"fit", "evaluate", "seq2print_loss", "TrainResult"},
    "_splits": {"chromosome_folds", "FOLDS"},
    "_pretrained": {"load_pretrained", "save_pretrained", "PRETRAINED"},
    "_interpret": {"delta_effects", "plant_motif", "zscore2pval"},
}
__all__ = sorted(set().union(*_ATTRS.values()))


def __getattr__(name):
    for mod, names in _ATTRS.items():
        if name in names:
            return getattr(importlib.import_module(f".{mod}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
