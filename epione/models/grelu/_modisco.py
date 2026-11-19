"""Motifs from attributions: TF-MoDISco patterns named by TOMTOM.

:func:`modisco` takes hypothetical attributions that are already computed
(:func:`epione.models.grelu.attributions` with ``hypothetical=True``, from
any model and output head), so one set of attributions can be clustered,
plotted and compared without re-running the model. It runs
``modiscolite.tfmodisco.TFMoDISco`` on the central window, trims each
pattern to the positions that carry its contribution, and matches the
trimmed motif against a reference set with ``memelite.tomtom`` (both are
gReLU dependencies). gReLU's own ``run_modisco`` does the same in one call
from a model, writing an HTML report instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd

__all__ = ["modisco", "ModiscoResult", "read_meme"]


@dataclass
class ModiscoResult:
    """TF-MoDISco patterns and their reference matches.

    ``table``: one row per pattern (``pattern``, ``sign``, ``n_seqlets``,
    ``consensus``, ``match``, ``pval``, then up to ``n_matches`` matches as
    ``match_2``/``pval_2``, ...), sorted as TF-MoDISco returns them.
    ``contrib[pattern]``: trimmed contribution logo ``[w, 4]`` (A, C, G, T),
    ready for :func:`epione.pl.plot_seqlogo`; ``ppm[pattern]``: the trimmed
    position probability matrix ``[w, 4]``.
    """

    table: pd.DataFrame
    contrib: Dict[str, np.ndarray] = field(default_factory=dict)
    ppm: Dict[str, np.ndarray] = field(default_factory=dict)


def read_meme(path) -> Dict[str, np.ndarray]:
    """Motifs of a MEME file as ``{"ID name": [4, w] probabilities}``."""
    from memelite.io import read_meme as _read

    return {k: np.asarray(v, dtype=np.float64) for k, v in _read(str(path)).items()}


def _import_tfmodisco():
    """modiscolite's package import runs ``matplotlib.use('pdf')`` (in its
    report module), which silently stops inline figures in a notebook;
    put the caller's backend back."""
    import matplotlib

    backend = matplotlib.get_backend()
    from modiscolite.tfmodisco import TFMoDISco

    if matplotlib.get_backend() != backend:
        matplotlib.use(backend, force=True)
    return TFMoDISco


def _trim(contrib: np.ndarray, threshold: float):
    s = np.abs(contrib).sum(1)
    keep = np.flatnonzero(s >= threshold * s.max())
    return slice(int(keep.min()), int(keep.max()) + 1)


def modisco(
    hypothetical,
    X,
    *,
    window: Optional[int] = 400,
    motifs: Union[None, str, Mapping[str, np.ndarray]] = None,
    n_matches: int = 3,
    trim_threshold: float = 0.3,
    max_seqlets: int = 20000,
    sliding_window_size: int = 20,
    flank_size: int = 5,
    target_seqlet_fdr: float = 0.05,
    n_leiden_runs: int = 2,
    **kwargs,
) -> ModiscoResult:
    """Cluster recurring high-attribution sequences into motifs.

    Parameters
    ----------
    hypothetical, X
        Hypothetical attributions and the one-hot sequences, both
        ``[n, 4, L]``.
    window
        Use the central ``window`` bp (``None``: all of it). Peaks' motifs
        sit near the summit; the flanks add mostly noise.
    motifs
        Reference motifs for naming: a MEME file or ``{name: [4, w]}``.
        ``None`` skips matching.
    trim_threshold
        Keep positions whose total |contribution| is at least this
        fraction of the pattern's maximum (modiscolite's report uses 0.3).
    max_seqlets, sliding_window_size, flank_size, target_seqlet_fdr, n_leiden_runs, **kwargs
        Passed to ``TFMoDISco`` (as ``max_seqlets_per_metacluster`` for the
        first). The defaults are chrombpnet's tutorial settings, lighter
        than modiscolite's (50 Leiden runs).
    """
    TFMoDISco = _import_tfmodisco()

    H = np.asarray(hypothetical, dtype=np.float32)
    S = np.asarray(X, dtype=np.float32)
    if H.shape != S.shape or H.ndim != 3 or H.shape[1] != 4:
        raise ValueError("hypothetical and X must both be [n, 4, L]")
    if window is not None and window < H.shape[-1]:
        lo = (H.shape[-1] - window) // 2
        H, S = H[..., lo:lo + window], S[..., lo:lo + window]
    pos, neg = TFMoDISco(
        hypothetical_contribs=H.transpose(0, 2, 1), one_hot=S.transpose(0, 2, 1),
        max_seqlets_per_metacluster=max_seqlets, sliding_window_size=sliding_window_size,
        flank_size=flank_size, target_seqlet_fdr=target_seqlet_fdr,
        n_leiden_runs=n_leiden_runs, verbose=False, **kwargs)

    ref = read_meme(motifs) if isinstance(motifs, (str, bytes)) or hasattr(motifs, "__fspath__") \
        else (dict(motifs) if motifs is not None else None)
    rows, contrib, ppm = [], {}, {}
    for sign, patterns in (("pos", pos or []), ("neg", neg or [])):
        for i, p in enumerate(patterns):
            name = f"{sign}_{i}"
            sl = _trim(np.asarray(p.contrib_scores), trim_threshold)
            contrib[name] = np.asarray(p.contrib_scores)[sl]
            ppm[name] = np.asarray(p.sequence)[sl]
            rows.append({"pattern": name, "sign": sign, "n_seqlets": len(p.seqlets),
                         "consensus": "".join("ACGT"[j] for j in ppm[name].argmax(1))})
    table = pd.DataFrame(rows, columns=["pattern", "sign", "n_seqlets", "consensus"])
    if ref and len(table):
        from memelite import tomtom

        names = list(ref)
        pvals = tomtom([ppm[n].T for n in table.pattern], [ref[k] for k in names])[0]
        for k in range(n_matches):
            col, pcol = ("match", "pval") if k == 0 else (f"match_{k + 1}", f"pval_{k + 1}")
            order = np.argsort(pvals, axis=1)[:, k]
            table[col] = [names[j] for j in order]
            table[pcol] = pvals[np.arange(len(order)), order]
    return ModiscoResult(table=table, contrib=contrib, ppm=ppm)
