"""Cumulative distance plot for TSS-to-peak (or feature-to-feature) distances.

Paired with :func:`epione.utils.distance_to_nearest_peak`, this is the
standard "how close is each gene class to a TF peak?" curve that appears
throughout ChIP/CUT&RUN papers.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def cumulative_distance(
    distances: Mapping[str, np.ndarray],
    *,
    max_kb: float = 20,
    colors: Optional[Mapping[str, str]] = None,
    linewidths: Optional[Mapping[str, float]] = None,
    ks_between: Optional[Tuple[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (3.4, 3.2),
    title: str = "",
    xlabel: str = "Distance from TSS (kb)",
    ylabel: str = "Cumulative distribution",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a step CDF curve for each group in ``distances``.

    Each entry of ``distances`` is an array of non-negative distances (bp or
    kb; must all share one unit - if any distance is larger than ``max_kb``
    the function treats the input as bp and converts to kb). The plotted y
    axis is the fraction of items in that group whose nearest peak is within
    x kb; curves plateau below 1 whenever some items fall beyond ``max_kb``.

    Arguments:
        distances: mapping ``{group_label: 1-D distance array}``.
        max_kb: right edge of the plotted window (kb). Values past this are
            excluded from the numerator but kept in the denominator so the
            plateau encodes "fraction within max_kb".
        colors: optional per-group colour mapping.
        linewidths: optional per-group line width mapping.
        ks_between: if given as ``(label1, label2)``, overlay the
            two-sided Kolmogorov-Smirnov p-value in the upper-left corner.
        ax: axis to draw on; if None, a fresh figure is created.
        figsize: used only when ``ax`` is None.
        title, xlabel, ylabel: passed through to matplotlib.

    Returns:
        ``(fig, ax)`` — the matplotlib figure + axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Distances are expected in the same unit as ``max_kb`` (kb by default).
    # If the caller passed bp, the median will be far larger than max_kb *
    # 100 — we warn but do not silently rescale.
    colors = dict(colors or {})
    linewidths = dict(linewidths or {})
    for label, arr in distances.items():
        arr = np.asarray(arr, dtype=float)
        total = len(arr)
        xs = np.sort(arr[arr <= max_kb])
        ys = np.arange(1, len(xs) + 1) / total if total else np.array([])
        ax.step(
            xs,
            ys,
            where="post",
            label=label,
            color=colors.get(label),
            lw=linewidths.get(label, 1.5),
        )

    if ks_between is not None:
        from scipy.stats import ks_2samp

        a, b = ks_between
        ks = ks_2samp(np.asarray(distances[a]), np.asarray(distances[b]))
        ax.text(
            0.03,
            0.97,
            f"P = {ks.pvalue:.1g}",
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="top",
        )

    ax.axhline(0, ls="--", color="k", lw=0.7, alpha=0.7)
    ax.set_xlim(0, max_kb)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax
