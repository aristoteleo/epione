"""Volcano and MA plots for differential-analysis results.

Input schema matches :func:`epione.tl.differential_peaks` output —
columns ``baseMean, log2FoldChange, lfcSE, stat, pvalue, padj`` — but
the column names are also parameterised so the plots work on any
DESeq2 / edgeR-style table.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def volcano(
    res: pd.DataFrame,
    *,
    lfc_col: str = "log2FoldChange",
    pval_col: str = "padj",
    lfc_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    top_n_labels: int = 10,
    label_col: Optional[str] = None,
    up_color: str = "#C73E3E",
    down_color: str = "#3A6EB3",
    ns_color: str = "#bbbbbb",
    point_size: float = 8.0,
    figsize: Tuple[float, float] = (6.0, 6.0),
    ax=None,
    title: str = "",
    max_neglog_p: Optional[float] = None,
):
    """Volcano plot: log₂FC on x, -log₁₀(padj) on y, thresholded colouring.

    Arguments:
        res: results DataFrame with at least ``lfc_col`` + ``pval_col``.
        lfc_col: column holding log2 fold-changes (default ``log2FoldChange``).
        pval_col: column holding adjusted p-values (default ``padj``). Pass
            ``pvalue`` to plot raw p's instead.
        lfc_thresh: absolute log2FC above which a feature is called significant.
        pval_thresh: adjusted-p cutoff for significance (both tails).
        top_n_labels: text-label the ``top_n_labels`` most significant
            features (ranked by ``pval_col``, ties broken by |``lfc_col``|).
            Uses the DataFrame index unless ``label_col`` is given.
        label_col: column to use as labels instead of the index.
        up_color, down_color, ns_color: hex colours for the three states.
        point_size: scatter marker size (points²).
        figsize: figure size when ``ax`` is None.
        ax: optional existing matplotlib axis to draw into.
        title: axes title.
        max_neglog_p: clip y-axis at this -log10(p) value. Points above are
            drawn at the cap with an open-triangle marker. Leave ``None`` to
            auto-clip at the 99th percentile when any p's fall below 1e-300.

    Returns:
        ``(fig, ax)`` where ``fig`` is the matplotlib figure containing ``ax``.
    """
    import matplotlib.pyplot as plt
    df = res[[lfc_col, pval_col]].copy()
    # Preserve labels before dropping NaNs.
    if label_col is not None:
        df["__label"] = res[label_col].astype(str)
    else:
        df["__label"] = df.index.astype(str)
    df = df.dropna(subset=[lfc_col, pval_col])

    # -log10(padj). Clamp zeros so log is finite.
    p = np.maximum(df[pval_col].to_numpy(), 1e-300)
    neglog = -np.log10(p)

    # Auto cap at 99th percentile if any extreme p's slipped in.
    if max_neglog_p is None and neglog.max() > 50:
        max_neglog_p = float(np.quantile(neglog, 0.99))
    if max_neglog_p is not None:
        clipped = neglog > max_neglog_p
        neglog = np.where(clipped, max_neglog_p, neglog)
    else:
        clipped = np.zeros_like(neglog, dtype=bool)

    lfc = df[lfc_col].to_numpy()
    sig = df[pval_col].to_numpy() < pval_thresh
    up = sig & (lfc > lfc_thresh)
    down = sig & (lfc < -lfc_thresh)
    ns = ~(up | down)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(lfc[ns], neglog[ns], s=point_size,
               c=ns_color, alpha=0.5, lw=0, rasterized=True)
    ax.scatter(lfc[down], neglog[down], s=point_size,
               c=down_color, alpha=0.85, lw=0, rasterized=True)
    ax.scatter(lfc[up], neglog[up], s=point_size,
               c=up_color, alpha=0.85, lw=0, rasterized=True)
    # Clipped points: open triangles at the cap.
    if clipped.any():
        ax.scatter(lfc[clipped], neglog[clipped], s=point_size * 2,
                   facecolors="none",
                   edgecolors=np.where(lfc[clipped] > 0, up_color, down_color),
                   marker="^", lw=0.8)

    # Reference lines.
    ax.axhline(-np.log10(pval_thresh), color="#666666", lw=0.6, ls="--")
    ax.axvline(+lfc_thresh, color="#666666", lw=0.6, ls="--")
    ax.axvline(-lfc_thresh, color="#666666", lw=0.6, ls="--")

    # Labels for top_n_labels most-significant features.
    if top_n_labels and top_n_labels > 0:
        order = np.argsort(df[pval_col].to_numpy())
        top_idx = order[:top_n_labels]
        for i in top_idx:
            ax.annotate(
                df["__label"].iloc[i],
                (lfc[i], neglog[i]),
                fontsize=7, color="#222", alpha=0.9,
                xytext=(3, 3), textcoords="offset points",
            )

    ax.set_xlabel("log$_2$ fold change")
    ax.set_ylabel(f"-log$_{{10}}$({pval_col})")
    ax.set_title(title)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Legend summary (counts).
    import matplotlib.patches as mpatches
    legend_items = [
        mpatches.Patch(color=up_color, label=f"up (n={int(up.sum())})"),
        mpatches.Patch(color=down_color, label=f"down (n={int(down.sum())})"),
        mpatches.Patch(color=ns_color, label=f"ns (n={int(ns.sum())})"),
    ]
    ax.legend(handles=legend_items, loc="upper right",
              frameon=False, fontsize=8)
    return fig, ax


def ma_plot(
    res: pd.DataFrame,
    *,
    mean_col: str = "baseMean",
    lfc_col: str = "log2FoldChange",
    pval_col: str = "padj",
    lfc_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    up_color: str = "#C73E3E",
    down_color: str = "#3A6EB3",
    ns_color: str = "#bbbbbb",
    point_size: float = 6.0,
    figsize: Tuple[float, float] = (6.0, 4.0),
    ax=None,
    title: str = "",
):
    """MA plot: mean expression on x (log-scaled), log₂FC on y.

    Arguments match :func:`volcano` where overlapping.

    Returns:
        ``(fig, ax)`` where ``fig`` contains the rendered axis.
    """
    import matplotlib.pyplot as plt
    df = res[[mean_col, lfc_col, pval_col]].copy().dropna()
    mean = np.maximum(df[mean_col].to_numpy(), 1e-3)
    lfc = df[lfc_col].to_numpy()
    sig = df[pval_col].to_numpy() < pval_thresh
    up = sig & (lfc > lfc_thresh)
    down = sig & (lfc < -lfc_thresh)
    ns = ~(up | down)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(mean[ns], lfc[ns], s=point_size,
               c=ns_color, alpha=0.45, lw=0, rasterized=True)
    ax.scatter(mean[down], lfc[down], s=point_size,
               c=down_color, alpha=0.85, lw=0, rasterized=True)
    ax.scatter(mean[up], lfc[up], s=point_size,
               c=up_color, alpha=0.85, lw=0, rasterized=True)

    ax.axhline(0, color="#666666", lw=0.6)
    ax.axhline(+lfc_thresh, color="#666666", lw=0.5, ls="--")
    ax.axhline(-lfc_thresh, color="#666666", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(f"{mean_col} (log scale)")
    ax.set_ylabel("log$_2$ fold change")
    ax.set_title(title)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


__all__ = ["volcano", "ma_plot"]
