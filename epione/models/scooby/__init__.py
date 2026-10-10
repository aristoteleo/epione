"""scooby bridge — single-cell sequence→profile modelling on Epione data.

scooby (Hingerl, Martens et al., *Nat Methods* 2026) predicts scATAC
insertion and scRNA coverage profiles from DNA sequence at single-cell
resolution, by fine-tuning a Borzoi backbone with LoRA and decoding it
through a hypernetwork conditioned on a cell's position in a precomputed
embedding. It is the closest published baseline for cell-state-resolved
sequence→function modelling, so Epione carries a bridge to it rather
than a reimplementation: the point is to be able to *benchmark* against
upstream numbers.

What this bridge changes, and why
---------------------------------
Upstream drives training from a genome-wide ``cells × base-pair`` sparse
insertion matrix in ``adata.obsm['insertion']``, built by a forked
snapATAC2 (``snapatac2-scooby``) whose numpy pin conflicts with scooby's
own, so upstream recommends two separate environments. Epione already
keeps fragments on disk and reads windows on demand, and that design
transfers directly to model targets:

* :mod:`._targets` reads a training window straight out of a
  tabix-indexed fragments file. This removes the forked snapATAC2 and
  the genome-wide matrix, and it is cheap because insertion density is
  low — see that module for the arithmetic.
* :mod:`._dataset` extracts sequence with ``pysam.FastaFile``, removing
  the ``enformer_pytorch`` data path as well.
* :mod:`._prepare` assigns cross-validation folds to contiguous
  genomic blocks with a flank buffer, so a training window's 524 kb
  input cannot reach into a held-out window's 197 kb target.

Only the model and loss remain genuinely upstream, imported from the
``scooby`` package when present.

Resolution warning
------------------
Per-cell targets are viable only when a cell contributes enough
fragments to a 197 kb window. Call
:func:`expected_fragments_per_window` with your own numbers first: for
spatial ATAC at ~1.5e4 fragments per spot the answer is ~1 fragment, so
``groups=`` (pseudobulk over cell types, timepoints or spatial domains)
is the usable mode and ``neighbors=`` needs a very large k.

Layering
--------
:mod:`._targets` and :mod:`._prepare` import no torch, so they can be
used and tested in Epione's own environment. Anything touching the
model requires a torch environment; see ``docs`` in the repository for
the environment recipe.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_TARGETS = {
    "ATAC_SCALE",
    "BIN_SIZE",
    "FragmentInsertionReader",
    "bin_sum",
    "ensure_tabix_index",
    "groups_to_matrix",
    "insertions_to_coverage",
}
_PREPARE = {
    "FLANK",
    "INPUT_LENGTH",
    "TARGET_LENGTH",
    "expected_fragments_per_window",
    "export_embedding",
    "export_neighbors",
    "groups_from_obs",
    "make_fold_bed",
}
_DATASET = {
    "ScoobyATACDataset",
    "one_hot_dna",
    "read_fold_bed",
    "worker_init",
}
_MODEL = {
    "load_scooby",
    "materialize_meta_buffers",
}

__all__ = sorted(_TARGETS | _PREPARE | _DATASET | _MODEL)


def __getattr__(name: str):
    if name in _TARGETS:
        mod = importlib.import_module("._targets", __name__)
    elif name in _PREPARE:
        mod = importlib.import_module("._prepare", __name__)
    elif name in _DATASET:
        mod = importlib.import_module("._dataset", __name__)
    elif name in _MODEL:
        mod = importlib.import_module("._model", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(mod, name)


def __dir__():
    return __all__


if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from ._dataset import (  # noqa: F401
        ScoobyATACDataset,
        one_hot_dna,
        read_fold_bed,
        worker_init,
    )
    from ._prepare import (  # noqa: F401
        FLANK,
        INPUT_LENGTH,
        TARGET_LENGTH,
        expected_fragments_per_window,
        export_embedding,
        export_neighbors,
        groups_from_obs,
        make_fold_bed,
    )
    from ._model import load_scooby, materialize_meta_buffers  # noqa: F401
    from ._targets import (  # noqa: F401
        ATAC_SCALE,
        BIN_SIZE,
        FragmentInsertionReader,
        bin_sum,
        ensure_tabix_index,
        groups_to_matrix,
        insertions_to_coverage,
    )
