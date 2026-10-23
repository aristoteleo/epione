"""Sequence-based TF binding scores from a seq2PRINT model (scPrinter's
``seq_tfbs_seq2print``).

For every window and cell state, :func:`tf_binding`:

1. computes hypothetical DeepLIFT/SHAP attributions (20 dinucleotide
   shuffles) for two outputs, through gReLU (:func:`epione.models.grelu.attributions`):
   the log coverage, and the TF-sized footprints (``head='footprint_tf'``,
   scPrinter's ``JustSumWrapper(nth_output=0-29, threshold=0.301)``);
2. projects them onto the sequence (one value per base) and puts each head
   on a common scale: ``(x - median) / (q95 - q05)`` over the central 800 bp;
3. runs scPrinter's two published TF-binding networks (one per head,
   Zenodo 15085406) on the central 1,200 bp, forward and reverse-complement
   averaged, and averages the two networks.

The result is a binding probability for every base of the central 800 bp.
Motif sites are then scored by reading the track over the site.

scPrinter computes the attributions with its own DeepLIFT/SHAP code and
its own shuffles; tangermeme (used by gReLU) applies the same rules, so the
scores agree up to the random references. scPrinter also writes the
attributions to a bigWig first, which averages windows where peaks
overlap; here each window is scored on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = ["tf_binding", "TFBindingTracks", "one_hot_regions", "SEQ_TFBS_HEADS"]

#: attribution head -> registry name of its TF-binding network
SEQ_TFBS_HEADS = {"coverage": "seq2print_tfbs_count", "footprint_tf": "seq2print_tfbs_footprint"}

_TFBS_IN = 1200            # the networks read 1,200 bp ...
_TFBS_CROP = 100           # ... return 1,000 and the outer 100 on each side are dropped


@dataclass
class TFBindingTracks:
    """Per-base TF binding probability for windows x cell states.

    ``score[s, i, j]``: state ``states[s]``, window ``i``, base ``j`` of the
    central 800 bp (window centre at ``j = 400``). ``attributions[head]``
    holds the normalised projected attributions ``[states, windows, L]``
    when requested; ``norm[head]`` is the ``(q05, median, q95)`` used.
    """

    score: np.ndarray
    states: List
    regions: Optional[pd.DataFrame] = None
    norm: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    attributions: Dict[str, np.ndarray] = field(default_factory=dict)

    def site_scores(self, sites: pd.DataFrame, *, how: str = "max") -> pd.DataFrame:
        """Score motif sites: the ``max`` (or ``mean``) of the track over each
        site. ``sites`` needs ``window`` (row in ``regions``) and ``start`` /
        ``end`` in genome coordinates; returns sites x states."""
        if self.regions is None:
            raise ValueError("site_scores needs the regions the tracks were computed on")
        centre = _centres(self.regions).to_numpy()
        w = sites["window"].to_numpy()
        lo = sites["start"].to_numpy() - (centre[w] - 400)
        hi = sites["end"].to_numpy() - (centre[w] - 400)
        if (lo < 0).any() or (hi > 800).any():
            raise ValueError("some sites fall outside the central 800 bp of their window")
        f = np.max if how == "max" else np.mean
        out = np.array([[f(self.score[s, wi, a:b]) for s in range(len(self.states))]
                        for wi, a, b in zip(w, lo, hi)])
        return pd.DataFrame(out, index=sites.index, columns=[str(s) for s in self.states])


def _centres(regions: pd.DataFrame) -> pd.Series:
    if "summit" in regions:
        return regions["summit"].astype(int)
    return ((regions["start"] + regions["end"]) // 2).astype(int)


def one_hot_regions(fasta, regions: pd.DataFrame, length: int = 1840) -> np.ndarray:
    """One-hot ``[n, 4, length]`` (A, C, G, T; N = all zero) centred on each
    region's ``summit`` column, or its midpoint."""
    import pyfaidx

    fa = fasta if isinstance(fasta, pyfaidx.Fasta) else pyfaidx.Fasta(str(fasta))
    lut = np.full(256, 4, dtype=np.int64)
    for i, b in enumerate("ACGT"):
        lut[ord(b)] = lut[ord(b.lower())] = i
    out = np.zeros((len(regions), 4, length), dtype=np.float32)
    for k, (chrom, c) in enumerate(zip(regions["chrom"], _centres(regions))):
        s = int(c) - length // 2
        seq = str(fa[chrom][max(s, 0):s + length])
        if s < 0:
            seq = "N" * (-s) + seq
        seq = seq.ljust(length, "N")
        idx = lut[np.frombuffer(seq.encode(), dtype=np.uint8)]
        ok = idx < 4
        out[k, idx[ok], np.nonzero(ok)[0]] = 1.0
    return out


def _load_net(head: str, device):
    import torch

    from epione.models.registry import get_pretrained

    net = torch.jit.load(str(get_pretrained(SEQ_TFBS_HEADS[head]).fetch()), map_location="cpu")
    net.with_motif = False
    return net.to(device).eval()


def _run_net(net, feats: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    """scPrinter's ``forward_pass_model``: sigmoid, forward and reversed
    input averaged, then the outer 100 bp dropped."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            x = torch.as_tensor(feats[i:i + batch_size]).float().to(device)[:, None, :]
            fwd = torch.sigmoid(net(x))[:, 0, :]
            rev = torch.flip(torch.sigmoid(net(torch.flip(x, [2])))[:, 0, :], [1])
            out.append(((fwd + rev) / 2).cpu().numpy())
    return np.concatenate(out)[:, _TFBS_CROP:-_TFBS_CROP]


def tf_binding(
    model,
    X,
    *,
    states: Optional[Sequence[int]] = None,
    regions: Optional[pd.DataFrame] = None,
    n_shuffles: int = 20,
    seed: int = 0,
    batch_size: int = 64,
    norm: Optional[Dict[str, Tuple[float, float, float]]] = None,
    keep_attributions: bool = False,
    device: Optional[str] = None,
) -> TFBindingTracks:
    """Base-resolution TF binding probabilities from a seq2PRINT model.

    Parameters
    ----------
    model
        A trained :class:`Seq2PRINT`. With cell-state LoRA, ``states`` picks
        the states to score (each is collapsed to a plain model first).
    X
        One-hot windows ``[n, 4, model.dna_len]`` centred on peak summits
        (:func:`one_hot_regions`).
    states
        LoRA state indices; ``None`` for a model without LoRA.
    regions
        The windows' coordinates (``chrom`` + ``summit`` or ``start``/``end``),
        kept for :meth:`TFBindingTracks.site_scores`.
    n_shuffles, seed, batch_size
        DeepLIFT/SHAP references and batching (gReLU / tangermeme).
    norm
        ``{head: (q05, median, q95)}`` to reuse a normalisation (e.g. from
        a larger peak set); by default it is computed from these windows,
        pooled over states, as scPrinter does for a LoRA model.

    Returns
    -------
    :class:`TFBindingTracks`
    """
    import torch

    from epione.models.grelu import attributions as grelu_attributions

    from ._interpret import _plain_model

    X = np.asarray(X.cpu() if isinstance(X, torch.Tensor) else X, dtype=np.float32)
    L = X.shape[-1]
    if L < _TFBS_IN or L != getattr(model, "dna_len", L):
        raise ValueError(f"X must be [n, 4, model.dna_len] windows (got length {L})")
    lora = getattr(model, "embeddings", None) is not None
    if lora and states is None:
        raise ValueError("this is a LoRA model: pass states=[...]")
    state_list = list(states) if lora else [None]
    dev = torch.device(device) if device is not None else next(model.parameters()).device

    proj = {h: [] for h in SEQ_TFBS_HEADS}
    for s in state_list:
        m = _plain_model(model, s).float().eval()
        for h in SEQ_TFBS_HEADS:
            a = grelu_attributions(m, X, head=h, method="deepshap", hypothetical=True,
                                   n_shuffles=n_shuffles, seed=seed, device=str(dev),
                                   batch_size=batch_size)
            proj[h].append((np.asarray(a, dtype=np.float32) * X).sum(1))   # [n, L]
        del m

    norm_used, scores, kept = {}, [], {}
    lo_in = (L - _TFBS_IN) // 2
    for h in SEQ_TFBS_HEADS:
        P = np.stack(proj[h])                                              # [S, n, L]
        if norm is not None and h in norm:
            q05, med, q95 = norm[h]
        else:
            flank = (L - 800) // 2
            v = P[..., flank:L - flank]
            q05, med, q95 = (float(np.quantile(v, q)) for q in (0.05, 0.5, 0.95))
        norm_used[h] = (q05, med, q95)
        P = (P - med) / (q95 - q05)
        if keep_attributions:
            kept[h] = P
        net = _load_net(h, dev)
        feats = P[..., lo_in:lo_in + _TFBS_IN].reshape(-1, _TFBS_IN)
        scores.append(_run_net(net, feats, dev).reshape(P.shape[0], P.shape[1], -1))
    score = np.mean(scores, axis=0)
    return TFBindingTracks(score=score, states=[s for s in state_list], regions=regions,
                           norm=norm_used, attributions=kept)
