"""ACESO — ACcessibility Estimation from Spatial transcriptOmics.

Infers genome-wide, single-cell chromatin accessibility at 32 bp resolution for cells measured
only by a targeted spatial RNA panel (MERFISH, Xenium, CosMx, ...).

Architecture
------------
State space (stage 0)
    The chromatin state of a cell is its EpiAgent embedding z (ATAC foundation model,
    1,355,445-cCRE vocabulary). For a non-human genome the cCREs are lifted over and keep their
    token ids (:mod:`._ccre`). EpiAgent is fine-tuned on the reference with its official UFE
    recipe (:mod:`._state`).
State encoder (stage 1)
    scGPT reads any gene panel of a cell and is distilled onto z (:mod:`._encoder`).
    **Panel-agnostic training**: every batch draws a random gene panel and depth
    (:mod:`._panel`), so one encoder serves any targeted platform.
Decoder (stage 2)
    scooby (Borzoi + LoRA + cell-state hypernetwork, via :mod:`epione.models.scooby`) predicts
    per-cell 32 bp ATAC profiles from 524 kb of DNA. It is conditioned on the *predicted* z, the
    same input it receives at inference (:mod:`._decoder`).
Joint fine-tuning (stage 3)
    **End-to-end**: the decoder's sequence-level loss is back-propagated into the state encoder,
    with an alignment term that keeps z in the EpiAgent space (:mod:`._joint`).
Inference
    panel RNA -> z -> per-cell profiles, averaged over forward and reverse-complement strands;
    aggregation to peaks and groups (:mod:`._predict`). Paper-style metrics are in :mod:`._eval`.

Environments
------------
Stage 0 needs ``epiagent`` (flash-attn). Stages 1–3 and inference need ``torch``, the official
scGPT code (``scgpt``) and upstream ``scooby``. Nothing here is imported by ``import epione``.
"""
from __future__ import annotations

import importlib

_MODULES = {
    "_ccre": ["lift_ccre_bed", "tokenize_fragments", "sentences_from_tfidf"],
    "_state": ["load_epiagent", "embed_cells", "UFEConfig", "finetune_ufe", "knn_gate"],
    "_panel": ["mgi_orthologs", "hvg_rank", "PanelSpace", "PanelSampler", "thin_counts"],
    "_encoder": ["StateEncoder", "EncoderConfig", "train_encoder", "predict_states", "alignment_loss"],
    "_decoder": ["DecoderConfig", "MultiFragmentReader", "build_decoder", "train_decoder", "profile_pearson"],
    "_joint": ["AcesoModel", "JointConfig", "train_joint", "cell_index_embedding"],
    "_predict": ["predict_profiles", "aggregate_regions"],
    "_eval": ["knn_neighbors", "knn_smoothed_pearson", "pseudobulk_pearson", "knn_transfer_accuracy"],
}
_NAME_TO_MODULE = {n: m for m, names in _MODULES.items() for n in names}
__all__ = sorted(_NAME_TO_MODULE)


def __getattr__(name: str):
    if name in _NAME_TO_MODULE:
        return getattr(importlib.import_module(f".{_NAME_TO_MODULE[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
