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
    scGPT reads any gene panel of a cell and is distilled onto z. **This is what ACESO is**: any
    panel, at any depth, maps to one fixed chromatin-state space, so a new dataset needs no refit.
    Panel-agnostic training draws a different gene subset and depth every batch (:mod:`._panel`).
Decoder (stage 2)
    DNA plus z -> the cell's profile. The read-out on z is linear on a frozen sequence trunk; the
    state-conditioned alternatives (mixture of LoRA experts :mod:`._molora`, FiLM :mod:`._filmhead`,
    attention) were trained and lost on the held-out benchmark -- the MoLoRA gate collapsed to a
    per-cell entropy of 1.351 of a possible ln 4 = 1.386, which makes the expert mixture one
    unconditional adapter. They stay here as the ablations they are.
Joint fine-tuning (stage 3)
    **End-to-end**: the decoder's sequence-level loss is back-propagated into the state encoder,
    with an alignment term that keeps z in the EpiAgent space (:mod:`._joint`).
Inference
    panel RNA -> z -> per-cell profiles, averaged over forward and reverse-complement strands;
    aggregation to peaks and groups (:mod:`._predict`). Paper-style metrics are in :mod:`._eval`.

The two published trainings
---------------------------
v1 (:mod:`._v1_encoder`, :mod:`._v1_decoder`)
    One reference multiome: a mouse embryo time course. The encoder is distilled onto the
    fine-tuned EpiAgent z with mixed panels; the decoder is an AlphaGenome 32 bp trunk with a linear
    per-cell read-out, finished with a within-type contrast over developmental stage, cell-cycle
    score and the leading within-type transcriptome components.
v2 (:mod:`._store`, :mod:`._v2_encoder`, :mod:`._v2_decoder`)
    A corpus of two million human nuclei across organs. The encoder reads two views of each cell --
    an arbitrary gene subset and a wide random view used as a stop-gradient teacher -- so it learns
    which genes carry the state rather than only invariance to missing ones. The decoder adds a
    per-gene RNA head beside the 32 bp ATAC head, pooling the trunk over each gene's exons
    (:mod:`._exon`), and takes named within-type RNA axes as explicit conditioning (:mod:`._axes`),
    because a within-type shift arrives in a 512-dimensional z as a 0.0013 cosine wiggle.

Figures are drawn with :mod:`._plot`, in the paper's typography and palette.

Environments
------------
Stage 0 needs ``epiagent`` (flash-attn). Stages 1–3 and inference need ``torch``, the official
scGPT code (``scgpt``) and upstream ``scooby``; the AlphaGenome trunk needs ``alphagenome_pytorch``.
Where each of those lives is resolved by :mod:`._sites`, which fails at start-up rather than per
item. Nothing here is imported by ``import epione``.
"""
from __future__ import annotations

import importlib

_MODULES = {
    "_ccre": ["lift_ccre_bed", "tokenize_fragments", "sentences_from_tfidf"],
    "_state": ["load_epiagent", "embed_cells", "UFEConfig", "finetune_ufe", "knn_gate"],
    "_panel": ["mgi_orthologs", "hvg_rank", "PanelSpace", "PanelSampler", "thin_counts"],
    "_encoder": ["StateEncoder", "EncoderConfig", "train_encoder", "predict_states", "alignment_loss"],
    "_decoder": ["DecoderConfig", "MultiFragmentReader", "build_decoder", "train_decoder", "profile_pearson", "trainable_state", "recalibrate_bn"],
    "_molora": ["AcesoDecoder", "MoLoRAConfig", "ExpertLoRA", "PrototypeGate", "build_aceso_decoder"],
    "_backbone": ["BACKBONES", "SlicedLoRA", "add_attention_lora"],
    "_filmhead": ["FiLMHeadDecoder", "FiLMConfig", "build_film_decoder"],
    "_joint": ["AcesoModel", "JointConfig", "train_joint", "cell_index_embedding"],
    "_predict": ["predict_profiles", "aggregate_regions"],
    "_eval": ["knn_neighbors", "knn_smoothed_pearson", "pseudobulk_pearson", "knn_transfer_accuracy"],
    "_alphagenome": ["AlphaGenomeScooby"],
    # where the resources live on this machine, checked before a GPU is taken
    "_sites": ["site_path", "require_sites", "register_site"],
    # the published trainings: v1 on one reference multiome, v2 on the two-million-nucleus corpus
    "_v1_encoder": ["V1EncoderConfig", "train_v1_encoder"],
    "_v1_decoder": ["V1DecoderConfig", "train_v1_decoder"],
    "_store": ["StoreConfig", "build_training_store", "TrainingStore"],
    "_axes": ["AXES", "build_axes"],
    "_exon": ["ExonMasks", "build_exon_masks"],
    "_v2_encoder": ["V2EncoderConfig", "train_v2_encoder"],
    "_v2_decoder": ["V2DecoderConfig", "train_v2_decoder"],
    # the paper's figures
    "_plot": ["PALETTE", "CELLTYPE_COLORS", "celltype_colors", "paper_style", "save_figure",
              "read_training_log", "plot_training_curve", "plot_tracks", "plot_benchmark",
              "plot_panel_size", "plot_state_embedding"],
}
_NAME_TO_MODULE = {n: m for m, names in _MODULES.items() for n in names}
__all__ = sorted(_NAME_TO_MODULE)


def __getattr__(name: str):
    if name in _NAME_TO_MODULE:
        return getattr(importlib.import_module(f".{_NAME_TO_MODULE[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
