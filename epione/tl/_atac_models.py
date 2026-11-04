"""Learned scATAC cell embeddings: scBasset and PeakVI.

``scbasset`` trains Epione's PyTorch rewrite of scBasset (Yuan & Kelley,
Nat Methods 2022; calico/scBasset, Apache-2.0) -- see
:mod:`epione.models.scbasset`. ``peakvi`` is a thin wrapper around
scvi-tools ``PEAKVI`` (Ashuach *et al.*, Nat Methods 2022; BSD-3-Clause),
placed like :func:`epione.tl.multivi`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _base(key_added: str) -> str:
    return key_added[2:] if key_added.startswith("X_") else key_added


def scbasset(
    adata,
    genome=None,
    *,
    codes: Optional[np.ndarray] = None,
    layer: Optional[str] = None,
    chrom_key: Optional[str] = None,
    start_key: Optional[str] = None,
    end_key: Optional[str] = None,
    bottleneck: int = 32,
    max_epochs: int = 1000,
    batch_size: int = 128,
    lr: float = 0.01,
    patience: int = 50,
    seed: int = 0,
    split_seed: int = 10,
    device: str = "auto",
    key_added: str = "X_scbasset",
    verbose: bool = True,
    log_every: int = 10,
):
    """Cell embedding of a cell x peak matrix with scBasset.

    A CNN reads the 1,344 bp centred on each peak and predicts the peak's
    accessibility in every cell; each cell's weight vector in the last layer
    is its embedding and the bias its depth-like intercept. Training follows
    the official ``scbasset_train.py`` (defaults are the official ones;
    peaks split 90/5/5 into train/test/validation with seed 10). Uses a GPU
    when one is available; needs ``torch``.

    scBasset's own recipe keeps peaks accessible in more than 5 % of cells.

    Arguments:
        adata: cells x peaks AnnData with counts (binarised internally).
        genome: FASTA path, ``pyfaidx.Fasta`` or an Epione genome
            (``epi.data.hg38``); not needed when ``codes`` is given.
        codes: precomputed ``(n_peaks, 1344)`` int8 codes
            (:func:`epione.models.scbasset.peak_sequences`).
        layer: use ``adata.layers[layer]`` instead of ``X``.
        chrom_key, start_key, end_key: ``var`` coordinate columns; by default
            ``chr``/``chrom``, ``start``/``chromStart``, ``end``/``chromEnd`` or
            parsed from ``var_names`` (``chr1:100-600``).
        bottleneck: embedding size (32).
        max_epochs, batch_size, lr, patience: training (1000, 128, 0.01, 50;
            early stopping on the training AUC, best epoch kept).
        seed: initialisation, augmentation and batch order.
        split_seed: peak split seed (10, as the official preprocessing).
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
        key_added: ``obsm`` key of the embedding.

    Returns:
        The trained :class:`epione.models.scbasset.ScBasset`. Writes
        ``adata.obsm[key_added]``, ``adata.obs["<key>_intercept"]`` and
        ``adata.uns["<key>"]`` (parameters, training history, mean test-peak
        AUROC). Use ``model.impute()`` for denoised accessibility and
        ``model.motif_score(tf)`` for motif-injection TF activity.
    """
    from ..models.scbasset import ScBasset, peak_sequences, peak_table

    X = adata.layers[layer] if layer is not None else adata.X
    if hasattr(X, "to_memory"):
        X = X.to_memory()
    if codes is None:
        if genome is None:
            raise ValueError("pass genome= (FASTA / epi.data.hg38) or precomputed codes=")
        peaks = peak_table(adata.var, chrom_key, start_key, end_key)
        codes = peak_sequences(peaks, genome, seed=seed)
    model = ScBasset(codes, X, bottleneck=bottleneck, split_seed=split_seed)
    model.fit(epochs=max_epochs, batch_size=batch_size, lr=lr, patience=patience, seed=seed,
              device=device, verbose=verbose, log_every=log_every)
    base = _base(key_added)
    adata.obsm[key_added] = model.cell_embedding()
    adata.obs[f"{base}_intercept"] = model.intercept()
    auroc = model.test_auroc(device=device)
    adata.uns[base] = {
        "params": {"bottleneck": bottleneck, "max_epochs": max_epochs, "batch_size": batch_size,
                   "lr": lr, "patience": patience, "seed": seed, "split_seed": split_seed},
        "epochs_run": int(model.history.get("epochs_run", 0)),
        "best_epoch": int(model.history.get("best_epoch", -1)) + 1,
        "train_auc": np.asarray(model.history.get("auc", []), dtype=float),
        "test_auroc": float(np.nanmean(auroc)),
    }
    return model


def peakvi(
    adata,
    *,
    layer: Optional[str] = None,
    batch_key: Optional[str] = None,
    n_latent: Optional[int] = None,
    max_epochs: int = 500,
    accelerator: str = "auto",
    seed: int = 0,
    key_added: str = "X_peakvi",
    model_kwargs: Optional[dict] = None,
    train_kwargs: Optional[dict] = None,
):
    """Cell embedding of a cell x peak matrix with PeakVI (Ashuach *et al.*, Nat Methods 2022).

    A thin wrapper around scvi-tools ``PEAKVI`` (the official
    implementation). Needs ``pip install epione[multimodal]``. Uses a GPU
    when one is available. scvi-tools' tutorial keeps peaks detected in more
    than 5 % of cells.

    Arguments:
        adata: cells x peaks AnnData with counts (PeakVI binarises them).
        layer: use ``adata.layers[layer]`` instead of ``X``.
        batch_key: optional ``obs`` column with batch labels.
        n_latent: latent dimensions (scvi-tools default when ``None``).
        max_epochs: training epochs (scvi-tools' early stopping on by default).
        accelerator: lightning accelerator, ``"auto"``, ``"gpu"`` or ``"cpu"``.
        seed: ``scvi.settings.seed``.
        key_added: ``obsm`` key of the latent space.
        model_kwargs, train_kwargs: passed to ``PEAKVI(...)`` and ``.train(...)``.

    Returns:
        The trained ``scvi.model.PEAKVI`` (set up on a copy of ``adata``).
        ``model.get_normalized_accessibility()`` gives imputed accessibility and
        ``model.differential_accessibility(groupby=...)`` differential
        accessibility; both use the copy, whose ``obs`` matches ``adata.obs``.
    """
    try:
        import scvi
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("epi.tl.peakvi needs scvi-tools: pip install 'epione[multimodal]'") from e

    scvi.settings.seed = seed
    a = adata.copy()
    if hasattr(a.X, "to_memory"):
        a.X = a.X.to_memory()
    if layer is not None:
        a.X = a.layers[layer]
    scvi.model.PEAKVI.setup_anndata(a, batch_key=batch_key)
    kw = dict(model_kwargs or {})
    if n_latent is not None:
        kw["n_latent"] = n_latent
    model = scvi.model.PEAKVI(a, **kw)
    model.train(max_epochs=max_epochs, accelerator=accelerator, **(train_kwargs or {}))
    adata.obsm[key_added] = model.get_latent_representation()
    return model


__all__ = ["scbasset", "peakvi"]
