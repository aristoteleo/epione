"""Inference: states -> per-cell 32 bp profiles for target windows, and aggregation.

A target window is 196,608 bp (6,144 x 32 bp bins); the model reads the window +- 163,840 bp of
sequence (524,288 bp), N-padded at chromosome ends, as in :mod:`epione.models.scooby`. As in the
scooby paper, predictions are averaged over the forward and reverse-complement strands (the RC
prediction is flipped back along bins). States are processed in chunks of cells, because the
hypernetwork decoder is per cell; the Borzoi trunk runs once per window and strand.

Genome-wide single-cell output is large (226k cells x 6,144 bins per window), so
:func:`predict_profiles` yields one window at a time and accepts a ``reduce`` callable, e.g.
:func:`aggregate_regions` (cells x peaks) or a group mean.
"""
from __future__ import annotations

from typing import Callable, Iterable, Iterator, Optional

import numpy as np
import pandas as pd


def _window_onehot(fasta, chrom: str, start: int, end: int, flank: int) -> np.ndarray:
    from epione.models.scooby._dataset import one_hot_dna
    i0, i1 = start - flank, end + flank
    L = fasta.get_reference_length(chrom)
    x = np.zeros((i1 - i0, 4), np.float32)
    s0, s1 = max(0, i0), min(L, i1)
    if s1 > s0:
        x[s0 - i0:s0 - i0 + (s1 - s0)] = one_hot_dna(fasta.fetch(chrom, s0, s1))
    return x


def predict_profiles(decoder, fasta_path: str, windows: pd.DataFrame, states: np.ndarray, *, cell_chunk: int = 256,
                     rc_average: bool = True, reduce: Optional[Callable] = None, device: str = "cuda") -> Iterator:
    """Yield ``(window_row, out)`` per window; ``out`` = (bins x cells) profile, or ``reduce(window_row, profile)``.

    ``windows``: chrom/start/end target windows (196,608 bp). ``states``: (cells x d) predicted z
    (rows with NaN are returned as NaN). Uses ``decoder.forward_seq_to_emb`` once per strand and
    ``forward_convs_on_emb`` per cell chunk when available; otherwise the full forward per chunk.
    """
    import pysam
    import torch
    from epione.models.scooby._prepare import FLANK, TARGET_LENGTH
    fa = pysam.FastaFile(fasta_path)
    S = torch.from_numpy(np.nan_to_num(states).astype(np.float32))
    bad = np.isnan(states).any(1)
    decoder.eval()
    fast = hasattr(decoder, "forward_seq_to_emb") and hasattr(decoder, "forward_cell_embs_only") \
        and hasattr(decoder, "forward_convs_on_emb")
    for _, w in windows.iterrows():
        if int(w["end"]) - int(w["start"]) != TARGET_LENGTH:
            raise ValueError("target windows must be 196,608 bp")
        x = _window_onehot(fa, w["chrom"], int(w["start"]), int(w["end"]), FLANK)
        strands = [x] + ([x[::-1, ::-1].copy()] if rc_average else [])
        prof = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for si, xs in enumerate(strands):
                xt = torch.from_numpy(xs)[None].permute(0, 2, 1).to(device)
                emb = decoder.forward_seq_to_emb(xt) if fast else None
                parts = []
                for c0 in range(0, len(S), cell_chunk):
                    z = S[c0:c0 + cell_chunk][None].to(device)
                    if fast:
                        w_, b_ = decoder.forward_cell_embs_only(z)
                        o = decoder.forward_convs_on_emb(emb, w_, b_)
                    else:
                        o = decoder(xt, z)
                    parts.append(o.float().squeeze(0).cpu().numpy())
                p = np.concatenate(parts, axis=1)                 # bins x cells
                if si == 1:
                    p = p[::-1]
                prof = p if prof is None else (prof + p) / 2
        prof[:, bad] = np.nan
        yield w, (reduce(w, prof) if reduce is not None else prof)


def aggregate_regions(regions: pd.DataFrame, bin_size: int = 32) -> Callable:
    """``reduce`` for :func:`predict_profiles`: sum predicted signal over each region's bins that fall
    in the window -> (cells x regions-in-window) plus the region index."""
    reg = regions.reset_index(drop=True)

    def reduce(w, prof):
        sel = reg[(reg["chrom"] == w["chrom"]) & (reg["end"] > w["start"]) & (reg["start"] < w["end"])]
        out = np.zeros((prof.shape[1], len(sel)), np.float32)
        for j, (_, r) in enumerate(sel.iterrows()):
            b0 = max(0, (int(r["start"]) - int(w["start"])) // bin_size)
            b1 = min(prof.shape[0], -(-(int(r["end"]) - int(w["start"])) // bin_size))
            out[:, j] = prof[b0:b1].sum(0)
        return sel.index.to_numpy(), out
    return reduce
