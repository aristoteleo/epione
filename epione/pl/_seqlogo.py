"""Sequence logos without extra dependencies.

``plot_seqlogo`` draws letters scaled by a value per base and position:
attribution scores (signed, ``[L, 4]`` or ``[4, L]``) or a motif's
information content (:func:`pfm_information`).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = ["plot_seqlogo", "pfm_information"]

#: Classic nucleotide colours (A green, C blue, G orange, T red).
BASE_COLORS = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "T": "#D62839"}

_GLYPHS = {}


def _glyph(letter: str):
    """Letter outline normalised to the unit square."""
    if letter not in _GLYPHS:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath

        path = TextPath((0, 0), letter, size=1,
                        prop=FontProperties(family="DejaVu Sans", weight="bold"))
        v = path.vertices
        lo, hi = v.min(0), v.max(0)
        _GLYPHS[letter] = (path, lo, hi - lo)
    return _GLYPHS[letter]


def _draw(ax, letter, x, bottom, height, color, width=0.95):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    from matplotlib.transforms import Affine2D

    path, lo, size = _glyph(letter)
    t = (Affine2D().translate(-lo[0], -lo[1]).scale(width / size[0], height / size[1])
         .translate(x - width / 2, bottom))
    ax.add_patch(PathPatch(Path(t.transform(path.vertices), path.codes),
                           facecolor=color, edgecolor="none", lw=0))


def plot_seqlogo(
    values,
    ax=None,
    *,
    start: int = 0,
    alphabet: str = "ACGT",
    colors: Optional[dict] = None,
    min_height: float = 0.0,
    ylabel: Optional[str] = None,
):
    """Draw a sequence logo.

    Parameters
    ----------
    values
        ``[L, 4]`` or ``[4, L]`` (columns / rows in ``alphabet`` order).
        Positive values stack upwards from 0, negative ones downwards, the
        largest magnitude outermost. Typical inputs: projected attributions
        (one non-zero base per position) or :func:`pfm_information`.
    ax
        Axes to draw on; default a new figure.
    start
        x coordinate of the first position (e.g. a genomic offset).
    min_height
        Skip letters smaller than this (keeps very long logos fast).

    Returns
    -------
    The axes.
    """
    import matplotlib.pyplot as plt

    v = np.asarray(values, dtype=float)
    if v.ndim != 2 or 4 not in v.shape:
        raise ValueError("values must be [L, 4] or [4, L]")
    if v.shape[1] != len(alphabet):
        v = v.T
    colors = {**BASE_COLORS, **(colors or {})}
    if ax is None:
        _, ax = plt.subplots(figsize=(max(3, 0.12 * len(v)), 1.6))
    for i, row in enumerate(v):
        x = start + i
        pos = [(h, b) for h, b in zip(row, alphabet) if h > min_height]
        neg = [(h, b) for h, b in zip(row, alphabet) if h < -min_height]
        y = 0.0
        for h, b in sorted(pos):
            _draw(ax, b, x, y, h, colors[b])
            y += h
        y = 0.0
        for h, b in sorted(neg, reverse=True):
            _draw(ax, b, x, y + h, -h, colors[b])
            y += h
    lo = min(0.0, float(np.minimum(v, 0).sum(1).min()))
    hi = max(0.0, float(np.maximum(v, 0).sum(1).max()))
    pad = 0.05 * (hi - lo or 1.0)
    ax.set_xlim(start - 0.5, start + len(v) - 0.5)
    ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)
    ax.axhline(0, color="0.5", lw=0.5, zorder=0)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def pfm_information(pfm, *, pseudocount: float = 0.0, background: Sequence[float] = (0.25,) * 4):
    """Letter heights for a motif logo: ``p * IC`` per position.

    ``pfm`` is ``[4, w]`` or ``[w, 4]`` counts or probabilities (A, C, G, T).
    Returns ``[w, 4]``.
    """
    p = np.asarray(pfm, dtype=float)
    if p.shape[0] == 4 and p.shape[1] != 4:
        p = p.T
    p = p + pseudocount
    p = p / p.sum(1, keepdims=True)
    bg = np.asarray(background, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = np.nansum(np.where(p > 0, p * np.log2(p / bg), 0.0), axis=1)
    return p * ic[:, None]
