"""ArchR-style plots for motif variability and positive TF regulators
(ports of ArchR ``getVarDeviations(plot=TRUE)`` and the ArchR-book
TF-regulator scatter)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ARCHR_COMET = LinearSegmentedColormap.from_list(
    "archr_comet", ["#E6E7E8", "#3A97FF", "#8816A7", "black"])


def _table(data, key: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    return data.uns[key]


def var_dev(
    data,
    key: str = "motif_deviations_var",
    *,
    n: int = 25,
    ax: Optional[plt.Axes] = None,
    figsize=(5, 5),
    label_size: float = 6,
    show: bool = True,
):
    """Rank plot of motif variability (ArchR ``plotVarDev``).

    Arguments:
        data: the DataFrame returned by :func:`epione.tl.var_deviations` or an
            AnnData holding it in ``uns[key]``.
        key: ``uns`` key when ``data`` is an AnnData.
        n: number of top-ranked motifs to label.
        ax: optional matplotlib axes.
        figsize: figure size when ``ax`` is None.
        label_size: font size of the labels.
        show: call ``plt.show()``.

    Returns:
        ``(fig, ax)``.
    """
    df = _table(data, key).sort_values("rank")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    sc = ax.scatter(df["rank"], df["combinedVars"], c=df["combinedVars"], cmap=ARCHR_COMET,
                    s=10, linewidths=0)
    top = df.head(n)
    y = df["combinedVars"].to_numpy()
    span = y.max() - min(0.0, y.min())
    xr = df["rank"].max()
    # Two staggered label columns; within a column labels keep a minimum gap.
    gap = span / (np.ceil(max(n, 1) / 2) + 1)
    last = [np.inf, np.inf]
    for i, (_, r) in enumerate(top.iterrows()):
        col = i % 2
        yt = min(r["combinedVars"], last[col] - gap)
        last[col] = yt
        ax.annotate(
            str(r["name"]), (r["rank"], r["combinedVars"]),
            xytext=(0.08 * xr + col * 0.22 * xr, yt),
            fontsize=label_size, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.4", lw=0.4),
            arrowprops=dict(arrowstyle="-", lw=0.3, color="0.3"),
        )
    ax.set_ylim(min(ax.get_ylim()[0], min(last) - gap), None)
    ax.set_xlabel("Rank Sorted Annotations")
    ax.set_ylabel("Variability")
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="combinedVars")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if show:
        plt.show()
    return fig, ax


def tf_regulators(
    data,
    key: str = "positive_tf_regulators",
    *,
    label_col: Optional[str] = None,
    n_label: int = 20,
    ax: Optional[plt.Axes] = None,
    figsize=(5, 5),
    show: bool = True,
):
    """Correlation-to-gene vs. max motif delta scatter, positive regulators
    highlighted (ArchR book figure).

    Arguments:
        data: the DataFrame from :func:`epione.tl.positive_tf_regulators` or
            an AnnData holding it in ``uns[key]``.
        key: ``uns`` key when ``data`` is an AnnData.
        label_col: column with labels (default: first ``*_name`` column).
        n_label: label at most this many ``YES`` regulators.
        ax, figsize, show: matplotlib controls.

    Returns:
        ``(fig, ax)``.
    """
    df = _table(data, key)
    label_col = label_col or [c for c in df.columns if c.endswith("_name")][0]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    yes = df["TFRegulator"] == "YES"
    ax.scatter(df.loc[~yes, "cor"], df.loc[~yes, "maxDelta"], s=8, c="darkgrey", lw=0, label="NO")
    ax.scatter(df.loc[yes, "cor"], df.loc[yes, "maxDelta"], s=10, c="firebrick", lw=0, label="YES")
    for _, r in df[yes].head(n_label).iterrows():
        ax.annotate(str(r[label_col]), (r["cor"], r["maxDelta"]), fontsize=6,
                    xytext=(3, 2), textcoords="offset points")
    ax.axvline(0, ls="--", lw=0.8, c="0.3")
    ax.set_ylim(0, df["maxDelta"].max() * 1.05)
    ax.set_xlabel("Correlation To Gene Score")
    ax.set_ylabel("Max TF Motif Delta")
    ax.legend(title="TFRegulator", frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if show:
        plt.show()
    return fig, ax
