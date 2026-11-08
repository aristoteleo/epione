"""Motif effects on a trained seq2PRINT model (scPrinter's delta effects).

:func:`delta_effects` plants a motif's consensus at the centre of background
sequences and averages how the predicted footprints (-log10 p, every scale and
position) and log coverage change -- scPrinter's ``seq2print_delta``.
:func:`plant_motif` does the planting; :func:`zscore2pval` is scPrinter's
footprint z-score to -log10 p transform (``zscore2pval_torch``, copied
unchanged from ``scprinter/utils.py`` at commit ``6edde74``, MIT licence,
Copyright (c) 2023 Broad Institute).

scPrinter runs delta effects as a CLI job over peak files and genome
objects; here it takes one-hot arrays and returns arrays.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch

__all__ = ["delta_effects", "plant_motif", "zscore2pval"]


# --- scPrinter, verbatim ------------------------------------------------------
# fmt: off

def zscore2pval_torch(footprint):
    """
    Convert z-scores to p-values from z-test (pytorch version)
    Parameters
    ----------
    footprint

    Returns
    -------
    torch.tensor of -log 10 p-values

    """
    # Calculate the CDF of the normal distribution for the given footprint
    pval = torch.distributions.Normal(0, 1).cdf(footprint)

    # Clamp pval to prevent log(0) which leads to -inf. Use a very small value as the lower bound.
    eps = torch.finfo(pval.dtype).eps
    pval_clamped = torch.clamp(pval, min=eps)

    # Compute the negative log10, using the clamped values to avoid -inf
    pval_log = -torch.log10(pval_clamped)

    # Optionally, to handle values very close to 1 (which would result in a negative pval_log),
    # you can clamp the output to be non-negative. This is a design choice depending on your requirements.
    pval_log = torch.clamp(pval_log, min=0, max=10)

    return pval_log


# fmt: on
# --- epione -------------------------------------------------------------------


def zscore2pval(z):
    """Footprint z-score -> -log10 p (one-sided, clipped to [0, 10]), as
    scPrinter reports footprint strength. Accepts arrays or tensors."""
    if isinstance(z, torch.Tensor):
        return zscore2pval_torch(z)
    return zscore2pval_torch(torch.as_tensor(np.asarray(z, dtype=np.float32))).numpy()


def _plain_model(model, state):
    """The model to explain: LoRA models are collapsed to one state."""
    if getattr(model, "embeddings", None) is not None:
        if state is None:
            raise ValueError("this is a LoRA model: pass state= (a cell-state index)")
        return model.collapse(int(state))
    if state is not None:
        raise ValueError("state= is only for LoRA models")
    return model


_ACGT = np.array(list("ACGT"))


def _consensus(pfm, background, mode, rng):
    """scPrinter's ``add_motif_to_seq``: argmax of the motif (ties broken
    by the background base) or a sample from it."""
    pfm = np.asarray(pfm, dtype=np.float64)
    m = pfm + background * 1e-3
    if mode == "argmax":
        return m.argmax(0)
    p = m / m.sum(0, keepdims=True)
    u = rng.random((1, p.shape[1]))
    return np.argmax(np.cumsum(p, 0) > u, axis=0)


def plant_motif(backgrounds, pfm, *, mode: str = "argmax", seed: int = 0) -> np.ndarray:
    """Write a motif into the centre of each background sequence.

    ``pfm`` is ``[4, w]`` (rows A, C, G, T; counts or probabilities).
    ``mode='argmax'`` writes the consensus; ``'multinomial'`` samples a
    site from the PFM for each background. Returns a copy.
    """
    B = np.array(backgrounds.cpu() if isinstance(backgrounds, torch.Tensor) else backgrounds,
                 dtype=np.float32, copy=True)
    pfm = np.asarray(pfm, dtype=np.float64)
    if pfm.ndim != 2 or pfm.shape[0] != 4:
        raise ValueError("pfm must be [4, width] (rows A, C, G, T)")
    w = pfm.shape[1]
    pad = (B.shape[-1] - w) // 2
    rng = np.random.default_rng(seed)
    for i in range(len(B)):
        idx = _consensus(pfm, B[i, :, pad:pad + w], mode, rng)
        B[i, :, pad:pad + w] = 0.0
        B[i, idx, pad + np.arange(w)] = 1.0
    return B


@torch.no_grad()
def _mean_outputs(model, X, batch_size, dev, amp):
    """Mean over sequences of -log10 p footprints and of log coverage."""
    foot, cov = 0.0, 0.0
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size]).float().to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                            enabled=bool(amp) and dev.type == "cuda"):
            a, b = model(xb)
        foot = foot + zscore2pval_torch(a.float()).sum(0).cpu()
        cov = cov + b.float().sum(0).cpu()
    return (foot / len(X)).numpy(), float(cov / len(X))


def delta_effects(
    model,
    motifs: Mapping[str, np.ndarray],
    backgrounds,
    *,
    state: Optional[int] = None,
    mode: str = "argmax",
    batch_size: int = 64,
    amp: bool = True,
    seed: int = 0,
    device: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """How planting each motif changes predicted footprints and coverage.

    scPrinter's ``seq2print_delta``: each motif is written into the centre
    of every background sequence (:func:`plant_motif`), and the model's
    predictions with the motif are compared with the predictions without
    it, averaged over the backgrounds. For a LoRA model the comparison is
    for one cell ``state``; scPrinter equalises coverage covariates first,
    which collapsing to a state makes unnecessary.

    Parameters
    ----------
    model
        A trained :class:`Seq2PRINT` (LoRA models need ``state``).
    motifs
        ``{name: pfm}``, each ``[4, width]`` with rows A, C, G, T.
    backgrounds
        ``[n, 4, model.dna_len]`` one-hot sequences: real peaks, or their
        dinucleotide shuffles for a sequence-context-free estimate.
    mode, seed
        ``'argmax'`` plants the consensus (scPrinter's default);
        ``'multinomial'`` samples a site per background.

    Returns
    -------
    ``(footprint, coverage)``: ``footprint[name]`` is the mean change in
    footprint strength, ``[99 scales, output_len]`` in -log10 p;
    ``coverage[name]`` is the mean change in predicted log coverage.
    """
    plain = _plain_model(model, state).eval()
    dev = torch.device(device) if device is not None else next(plain.parameters()).device
    plain = plain.to(dev)
    B = np.asarray(backgrounds.cpu() if isinstance(backgrounds, torch.Tensor) else backgrounds,
                   dtype=np.float32)
    base_f, base_c = _mean_outputs(plain, B, batch_size, dev, amp)
    foot, cov = {}, {}
    for name, pfm in motifs.items():
        f, c = _mean_outputs(plain, plant_motif(B, pfm, mode=mode, seed=seed),
                             batch_size, dev, amp)
        foot[name], cov[name] = f - base_f, c - base_c
    return foot, cov
