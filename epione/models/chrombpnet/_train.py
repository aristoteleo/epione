"""Train ChromBPNet bias and accessibility models, the official way.

Reproduces ``chrombpnet/training/train.py`` (Keras ``model.fit``) and the
two hyper-parameter steps of the official pipelines (kundajelab/chrombpnet,
MIT, commit ``09938fd``) in PyTorch:

* loss = multinomial NLL of the profile (``losses.multinomial_nll``, the
  full TensorFlow-Probability log-likelihood, factorial terms included) +
  ``counts_loss_weight`` x MSE of ``log(1 + total counts)``;
* Adam, learning rate 1e-3, Keras defaults (beta 0.9/0.999, epsilon 1e-7);
* Keras Glorot-uniform initialisation;
* per-epoch regeneration of jitter / revcomp / negatives
  (:class:`~._data.ChromBPNetBatches`); validation without jitter or
  revcomp, with a fixed non-peak subsample (``DataFrame.sample(...,
  random_state=seed)``);
* ``val_loss`` as Keras reports it (batch-size-weighted mean of per-batch
  losses); early stopping with patience 5 that restores the best weights;
* the bias model's count head is rescaled before ChromBPNet training
  (``adjust_bias_model_logcounts``).

No TensorFlow is used.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._data import (
    ChromBPNetBatches,
    _split,
    bias_hyperparams,
    chrombpnet_hyperparams,
    load_regions,
    read_narrowpeak,
)

__all__ = [
    "multinomial_nll",
    "TrainResult",
    "fit_model",
    "adjust_bias_logcounts",
    "train_bias",
    "train_chrombpnet",
]


def multinomial_nll(logits, counts):
    """TFP ``Multinomial(total_count=n, logits).log_prob`` summed over the
    batch and divided by the batch size, negated -- upstream's loss."""
    import torch

    counts = counts.to(logits.dtype)
    n = counts.sum(-1)
    logp = torch.log_softmax(logits, dim=-1)
    ll = torch.lgamma(n + 1) - torch.lgamma(counts + 1).sum(-1) + (counts * logp).sum(-1)
    return -ll.sum() / logits.shape[0]


@dataclass
class TrainResult:
    """What :func:`fit_model` returns."""

    model: object
    history: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    params: Dict = field(default_factory=dict)
    seconds: float = 0.0


def _seed_all(seed: int):
    import random

    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batch_losses(model, X, y, y_log, clw, dev):
    import torch
    x = torch.as_tensor(np.ascontiguousarray(np.transpose(X, (0, 2, 1))),
                        dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    yl = torch.as_tensor(y_log, dtype=torch.float32, device=dev)
    prof, cnt = model(x)
    p_loss = multinomial_nll(prof, yt)
    c_loss = torch.mean((cnt - yl) ** 2)
    return p_loss, c_loss, p_loss + clw * c_loss


def fit_model(model, train: ChromBPNetBatches, valid: ChromBPNetBatches, *,
              counts_loss_weight: float, epochs: int = 50, early_stop: int = 5,
              learning_rate: float = 1e-3, device: Optional[str] = None,
              verbose: bool = True, trainable=None) -> TrainResult:
    """Keras-``fit`` equivalent for one model.

    ``trainable`` limits the optimised parameters (default: those with
    ``requires_grad``; the frozen bias inside :class:`ChromBPNet` is
    excluded automatically).
    """
    import torch

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    params = [p for p in (trainable or model.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-7)
    clw = float(counts_loss_weight)

    best, best_state, best_epoch, wait = float("inf"), None, -1, 0
    history: List[Dict[str, float]] = []
    t0 = time.time()
    for epoch in range(epochs):
        te = time.time()
        model.train()
        tot = {"loss": 0.0, "profile": 0.0, "counts": 0.0, "n": 0}
        for X, y, yl in train:
            opt.zero_grad(set_to_none=True)
            p, c, loss = _batch_losses(model, X, y, yl, clw, dev)
            loss.backward()
            opt.step()
            b = X.shape[0]
            tot["loss"] += loss.item() * b
            tot["profile"] += p.item() * b
            tot["counts"] += c.item() * b
            tot["n"] += b
        model.eval()
        vt = {"loss": 0.0, "profile": 0.0, "counts": 0.0, "n": 0}
        with torch.no_grad():
            for X, y, yl in valid:
                p, c, loss = _batch_losses(model, X, y, yl, clw, dev)
                b = X.shape[0]
                vt["loss"] += loss.item() * b
                vt["profile"] += p.item() * b
                vt["counts"] += c.item() * b
                vt["n"] += b
        rec = {
            "epoch": epoch,
            "loss": tot["loss"] / tot["n"],
            "logits_profile_predictions_loss": tot["profile"] / tot["n"],
            "logcount_predictions_loss": tot["counts"] / tot["n"],
            "val_loss": vt["loss"] / vt["n"],
            "val_logits_profile_predictions_loss": vt["profile"] / vt["n"],
            "val_logcount_predictions_loss": vt["counts"] / vt["n"],
            "seconds": time.time() - te,
        }
        history.append(rec)
        if verbose:
            print(f"epoch {epoch}: loss {rec['loss']:.4f} val_loss {rec['val_loss']:.4f} "
                  f"({rec['seconds']:.0f}s)", flush=True)
        if rec["val_loss"] < best:
            best, best_epoch, wait = rec["val_loss"], epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            wait += 1
            if wait >= early_stop:
                break
        train.on_epoch_end()
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model=model, history=history, best_epoch=best_epoch,
                       best_val_loss=best, seconds=time.time() - t0)


def adjust_bias_logcounts(bias_model, seqs: np.ndarray, cts: np.ndarray, *,
                          batch_size: int = 64, device: Optional[str] = None) -> float:
    """``adjust_bias_model_logcounts``: add the mean of
    ``log(1 + observed) - predicted`` to the count head's bias.

    ``seqs`` are one-hot ``(N, L, 4)`` non-peak sequences and ``cts`` their
    total counts. Returns the delta applied.
    """
    import torch

    from ._model import predict
    x = np.transpose(seqs, (0, 2, 1))
    _, pred = predict(bias_model, x, batch_size=batch_size, device=device)
    delta = float(np.mean(np.log(1 + cts) - pred.ravel()))
    with torch.no_grad():
        bias_model.count_dense.bias += delta
    return delta


def _fold_regions(df, fold, mode):
    return _split(df, fold, mode) if df is not None and len(df) else None


def train_bias(peaks, nonpeaks, genome, signal, fold, *, seed: int = 1234,
               inputlen: int = 2114, outputlen: int = 1000, filters: int = 128,
               n_dil_layers: int = 4, bias_threshold_factor: float = 0.5,
               batch_size: int = 64, epochs: int = 50, early_stop: int = 5,
               learning_rate: float = 1e-3, device: Optional[str] = None,
               verbose: bool = True) -> TrainResult:
    """``chrombpnet bias pipeline`` minus I/O: hyper-parameters, then train
    a BPNet on the filtered non-peaks only (no jitter, revcomp on)."""
    from ._model import build_bpnet, glorot_init_

    hp = bias_hyperparams(peaks, nonpeaks, signal, fold, inputlen=inputlen,
                          outputlen=outputlen, bias_threshold_factor=bias_threshold_factor,
                          filters=filters, n_dil_layers=n_dil_layers)
    _seed_all(seed)
    rng = np.random.RandomState(seed)
    non = hp["nonpeaks"]
    tr_s, tr_c = load_regions(_fold_regions(non, fold, "train"), genome, signal, inputlen, outputlen, 0)
    va_s, va_c = load_regions(_fold_regions(non, fold, "valid"), genome, signal, inputlen, outputlen, 0)
    train = ChromBPNetBatches(None, None, tr_s, tr_c, inputlen=inputlen, outputlen=outputlen,
                              batch_size=batch_size, negative_sampling_ratio=1.0,
                              add_revcomp=True, shuffle=True, rng=rng)
    valid = ChromBPNetBatches(None, None, va_s, va_c, inputlen=inputlen, outputlen=outputlen,
                              batch_size=batch_size, add_revcomp=False, shuffle=False, rng=rng)
    model = glorot_init_(build_bpnet("bias", filters=filters, n_dil_layers=n_dil_layers,
                                     input_len=inputlen, output_len=outputlen))
    res = fit_model(model, train, valid, counts_loss_weight=hp["params"]["counts_loss_weight"],
                    epochs=epochs, early_stop=early_stop, learning_rate=learning_rate,
                    device=device, verbose=verbose)
    res.params = {**hp["params"], "seed": seed,
                  "counts_sum_min_thresh": hp["counts_sum_min_thresh"],
                  "counts_sum_max_thresh": hp["counts_sum_max_thresh"]}
    return res


def train_chrombpnet(peaks, nonpeaks, genome, signal, fold, bias_model, *, seed: int = 1234,
                     inputlen: int = 2114, outputlen: int = 1000, filters: int = 512,
                     n_dil_layers: int = 8, max_jitter: int = 500,
                     negative_sampling_ratio: float = 0.1, batch_size: int = 64,
                     epochs: int = 50, early_stop: int = 5, learning_rate: float = 1e-3,
                     device: Optional[str] = None, verbose: bool = True,
                     hyperparam_rng: Optional[np.random.RandomState] = None) -> TrainResult:
    """``chrombpnet pipeline`` minus I/O.

    ``bias_model`` is a trained :class:`~._model.BPNet` (e.g. from
    :func:`train_bias` or an official ``bias_scaled`` file). It is copied,
    rescaled on the training non-peaks, frozen, and combined with a fresh
    512 x 8 accessibility model that is trained on peaks (jitter 500) plus
    a ``negative_sampling_ratio`` share of non-peaks per epoch.
    """
    from ._model import ChromBPNet, build_bpnet, glorot_init_

    hp = chrombpnet_hyperparams(peaks, nonpeaks, signal, fold, inputlen=inputlen,
                                outputlen=outputlen, max_jitter=max_jitter,
                                negative_sampling_ratio=negative_sampling_ratio,
                                filters=filters, n_dil_layers=n_dil_layers,
                                rng=hyperparam_rng)
    bias = copy.deepcopy(bias_model).eval()
    adj_seqs, _ = load_regions(hp["bias_adjust_regions"], genome, signal, inputlen, outputlen, 0)
    delta = adjust_bias_logcounts(bias, adj_seqs, hp["bias_adjust_counts"], device=device)

    _seed_all(seed)
    rng = np.random.RandomState(seed)
    pk, non = hp["peaks"], hp["nonpeaks"]
    trp_s, trp_c = load_regions(_fold_regions(pk, fold, "train"), genome, signal,
                                inputlen, outputlen, max_jitter)
    trn_s, trn_c = load_regions(_fold_regions(non, fold, "train"), genome, signal,
                                inputlen, outputlen, 0)
    vap = _fold_regions(pk, fold, "valid")
    van = _fold_regions(non, fold, "valid")
    if van is not None and vap is not None:
        van = van.sample(n=int(float(negative_sampling_ratio) * vap.shape[0]),
                         replace=False, random_state=seed)
    vap_s, vap_c = load_regions(vap, genome, signal, inputlen, outputlen, 0)
    van_s, van_c = load_regions(van, genome, signal, inputlen, outputlen, 0)
    train = ChromBPNetBatches(trp_s, trp_c, trn_s, trn_c, inputlen=inputlen, outputlen=outputlen,
                              batch_size=batch_size, negative_sampling_ratio=negative_sampling_ratio,
                              add_revcomp=True, shuffle=True, rng=rng)
    valid = ChromBPNetBatches(vap_s, vap_c, van_s, van_c, inputlen=inputlen, outputlen=outputlen,
                              batch_size=batch_size, negative_sampling_ratio=1.0,
                              add_revcomp=False, shuffle=False, rng=rng)
    acc = glorot_init_(build_bpnet("chrombpnet", filters=filters, n_dil_layers=n_dil_layers,
                                   input_len=inputlen, output_len=outputlen))
    model = ChromBPNet(bias, acc)
    res = fit_model(model, train, valid, counts_loss_weight=hp["params"]["counts_loss_weight"],
                    epochs=epochs, early_stop=early_stop, learning_rate=learning_rate,
                    device=device, verbose=verbose, trainable=acc.parameters())
    res.params = {**hp["params"], "seed": seed, "bias_logcount_delta": delta,
                  "counts_sum_min_thresh": hp["counts_sum_min_thresh"],
                  "counts_sum_max_thresh": hp["counts_sum_max_thresh"]}
    return res
