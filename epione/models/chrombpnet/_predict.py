"""Prediction, evaluation and interpretation for ChromBPNet models.

* :func:`predict_regions` -- profiles and counts at summit-centred regions;
* :func:`evaluate` -- the official count / profile metrics
  (``training/metrics.py``: count Pearson and Spearman on
  ``log(1 + total)``, per-region Jensen-Shannon distance of the observed
  profile, normalised with a 0.001 pseudocount, against the predicted
  probabilities);
* :func:`contribution_scores` -- DeepLIFT-SHAP with dinucleotide-shuffled
  references, as ``chrombpnet contribs_bw`` computes it (counts head: the
  log-count output; profile head: the weighted mean-normalised logits),
  projected onto the input one-hot. Needs ``captum``;
* :func:`ism` -- in-silico saturation mutagenesis, no extra dependency;
* :func:`marginal_footprint` -- predicted profile with a motif inserted in
  shuffled backgrounds, and its depth relative to the flanks.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ._data import get_cts, get_seq

__all__ = [
    "predict_regions",
    "evaluate",
    "dinuc_shuffle",
    "contribution_scores",
    "ism",
    "marginal_footprint",
    "softmax",
]


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def predict_regions(model, regions, genome, *, batch_size: int = 64,
                    device: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Predict at summit-centred regions.

    Returns ``profile_logits (N, out)``, ``log_counts (N,)``, ``probs`` and
    ``expected_counts = probs * exp(log_counts)``. For a
    :class:`~._model.ChromBPNet` pass ``model.accessibility`` to get the
    bias-corrected predictions.
    """
    from ._model import predict

    seqs = get_seq(regions, genome, model.input_len)
    prof, cnt = predict(model, np.transpose(seqs, (0, 2, 1)), batch_size=batch_size,
                        device=device)
    probs = softmax(prof)
    return {"profile_logits": prof, "log_counts": cnt[:, 0], "probs": probs,
            "expected_counts": probs * np.exp(cnt[:, 0])[:, None]}


def evaluate(model, regions, genome, signal, *, batch_size: int = 64,
             device: Optional[str] = None, pseudocount: float = 0.001) -> Dict[str, float]:
    """Official metrics on ``regions`` (no jitter, summit-centred)."""
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import pearsonr, spearmanr

    pred = predict_regions(model, regions, genome, batch_size=batch_size, device=device)
    obs = get_cts(regions, signal, model.output_len)
    true_log = np.log(1 + obs.sum(-1))
    import warnings
    with warnings.catch_warnings():
        # all-zero observed windows give NaN, skipped by nanmedian as upstream
        warnings.simplefilter("ignore", RuntimeWarning)
        jsd = np.array([jensenshannon(o / (pseudocount + np.nansum(o)), p)
                        for o, p in zip(obs, pred["probs"])])
    return {
        "count_pearson": float(pearsonr(true_log, pred["log_counts"])[0]),
        "count_spearman": float(spearmanr(true_log, pred["log_counts"])[0]),
        "count_mse": float(np.mean((true_log - pred["log_counts"]) ** 2)),
        "profile_median_jsd": float(np.nanmedian(jsd)),
        "n_regions": int(len(obs)),
    }


# ---------------------------------------------------------------------------
# Shuffles and attributions
# ---------------------------------------------------------------------------

def dinuc_shuffle(onehot: np.ndarray, n: int = 1,
                  rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Dinucleotide-preserving shuffles of one ``(L, 4)`` one-hot sequence
    (Altschul-Erickson via a random Eulerian walk, the algorithm of
    ``deeplift.dinuc_shuffle`` used by upstream). Returns ``(n, L, 4)``."""
    rng = rng if rng is not None else np.random.RandomState()
    tokens = onehot.argmax(axis=1)
    tokens[onehot.sum(axis=1) == 0] = 4  # N
    L = len(tokens)
    chars = np.unique(tokens)
    next_inds = []
    for c in chars:
        idx = np.where(tokens[:-1] == c)[0]
        next_inds.append(idx + 1)
    out = np.zeros((n, L, 4), dtype=onehot.dtype)
    for i in range(n):
        last = []
        for t in range(len(chars)):
            inds = np.arange(len(next_inds[t]))
            if len(inds) > 1:
                inds[:-1] = rng.permutation(len(inds) - 1)
            last.append(inds)
        counters = np.zeros(len(chars), dtype=int)
        res = np.empty(L, dtype=tokens.dtype)
        res[0] = tokens[0]
        cmap = {c: k for k, c in enumerate(chars)}
        for j in range(1, L):
            t = cmap[res[j - 1]]
            res[j] = tokens[next_inds[t][last[t][counters[t]]]]
            counters[t] += 1
        valid = res < 4
        out[i, np.arange(L)[valid], res[valid]] = 1
    return out


def _scalar_model(model, head: str):
    """Scalar target per sequence. ``'counts'``: the log-count output.
    ``'profile'``: upstream's ``sum(softmax(stop_grad(l)) * (l - mean(l)))``
    over positions (weighted mean-normalised logits)."""
    import torch
    from torch import nn

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            prof, cnt = self.m(x)
            if head == "counts":
                return cnt
            mn = prof - prof.mean(dim=-1, keepdim=True)
            w = torch.softmax(prof.detach(), dim=-1)
            return (w * mn).sum(dim=-1, keepdim=True)

    return Wrap(model)


def contribution_scores(model, seqs: np.ndarray, *, head: str = "counts",
                        n_shuffles: int = 20, seed: int = 0,
                        device: Optional[str] = None, batch_size: int = 8) -> np.ndarray:
    """DeepLIFT-SHAP contributions, projected onto the input.

    ``seqs``: one-hot ``(N, L, 4)``. ``model``: a BPNet, usually the
    bias-corrected ``ChromBPNet.accessibility``. Each sequence gets
    ``n_shuffles`` dinucleotide-shuffled references, as upstream. Returns
    ``(N, L, 4)`` hypothetical-times-one-hot scores.
    """
    try:
        from captum.attr import DeepLiftShap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("contribution_scores needs captum (pip install captum); "
                          "ism() needs nothing extra") from exc
    import torch

    dev = torch.device(device or next(model.parameters()).device)
    wrap = _scalar_model(model, head).to(dev).eval()
    expl = DeepLiftShap(wrap)
    rng = np.random.RandomState(seed)
    out = np.zeros(seqs.shape, dtype=np.float32)
    for i in range(len(seqs)):
        refs = dinuc_shuffle(seqs[i], n_shuffles, rng)
        x = torch.as_tensor(np.transpose(seqs[i:i + 1], (0, 2, 1)), dtype=torch.float32,
                            device=dev).requires_grad_(True)
        base = torch.as_tensor(np.transpose(refs, (0, 2, 1)), dtype=torch.float32, device=dev)
        attr = expl.attribute(x, baselines=base, target=0)
        out[i] = np.transpose(attr.detach().cpu().numpy()[0], (1, 0))
    return out * seqs


def ism(model, seq: np.ndarray, *, positions: Optional[Sequence[int]] = None,
        head: str = "counts", device: Optional[str] = None, batch_size: int = 64) -> np.ndarray:
    """In-silico mutagenesis of one ``(L, 4)`` sequence: effect of every
    substitution on the log-counts (``head='counts'``) or on the total
    predicted profile log-probability at the centre (``'profile'``).
    Returns ``(len(positions), 4)``; the reference base is 0."""
    from ._model import predict

    pos = np.arange(seq.shape[0]) if positions is None else np.asarray(positions)
    variants = []
    for p in pos:
        for b in range(4):
            s = seq.copy()
            s[p] = 0
            s[p, b] = 1
            variants.append(s)
    X = np.transpose(np.stack([seq] + variants), (0, 2, 1))
    prof, cnt = predict(model, X, batch_size=batch_size, device=device)
    if head == "counts":
        score = cnt[:, 0]
    else:
        score = np.log(softmax(prof)[:, prof.shape[1] // 2 - 5: prof.shape[1] // 2 + 5].sum(-1))
    return (score[1:] - score[0]).reshape(len(pos), 4)


def marginal_footprint(model, motif: str, *, backgrounds: Optional[np.ndarray] = None,
                       genome=None, regions=None, n: int = 500, seed: int = 0,
                       center_halfwidth: int = 5, flank=(150, 250),
                       device: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Average predicted profile with ``motif`` inserted at the centre of
    dinucleotide-shuffled backgrounds, minus nothing (raw probabilities).

    Backgrounds: ``backgrounds`` (``(M, L, 4)``), or ``n`` shuffles of
    sequences taken at ``regions`` from ``genome``. Returns the mean
    profile with and without the motif and ``depth`` = mean probability
    within ±``center_halfwidth`` bp of the centre divided by the mean in
    the ±``flank`` bp windows (a value < 1 is a footprint).
    """
    from ._data import one_hot_encode
    from ._model import predict

    rng = np.random.RandomState(seed)
    L = model.input_len
    if backgrounds is None:
        if genome is None or regions is None:
            raise ValueError("give backgrounds= or genome= and regions=")
        seqs = get_seq(regions.iloc[:n], genome, L)
        backgrounds = np.stack([dinuc_shuffle(s, 1, rng)[0] for s in seqs])
    mot = one_hot_encode([motif])[0]
    with_m = backgrounds.copy()
    a = L // 2 - len(motif) // 2
    with_m[:, a:a + len(motif)] = mot
    p0, _ = predict(model, np.transpose(backgrounds, (0, 2, 1)), device=device)
    p1, _ = predict(model, np.transpose(with_m, (0, 2, 1)), device=device)
    prof0, prof1 = softmax(p0).mean(0), softmax(p1).mean(0)
    c = len(prof1) // 2
    centre = prof1[c - center_halfwidth: c + center_halfwidth + 1].mean()
    lo, hi = flank
    fl = np.concatenate([prof1[c - hi: c - lo], prof1[c + lo: c + hi]]).mean()
    return {"profile_with_motif": prof1, "profile_background": prof0,
            "depth": float(centre / fl)}
