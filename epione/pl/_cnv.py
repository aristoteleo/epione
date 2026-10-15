"""Plots for :func:`epione.tl.cnv` results (epiAneufinder karyogram and profiles)."""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["karyogram", "cnv_profile"]

# epiAneufinder colours: 0-somy darkorchid3, 1-somy springgreen2, 2-somy red3
CNV_COLORS = {0: "#9A32CD", 1: "#00EE76", 2: "#CD0000"}
CNV_LABELS = {0: "loss", 1: "normal", 2: "gain"}


def _dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def _draw_dendrogram(ax, merge, height, order):
    """Draw an R-style dendrogram (``plot.hclust`` geometry), leaves on y."""
    n = merge.shape[0] + 1
    pos = np.empty(n)
    pos[np.asarray(order) - 1] = np.arange(n)
    node_x = {}
    node_h = {}
    for s in range(n - 1):
        xy = []
        for m in merge[s]:
            if m < 0:
                xy.append((pos[-m - 1], 0.0))
            else:
                xy.append((node_x[m], node_h[m]))
        h = float(height[s])
        (y1, h1), (y2, h2) = xy
        ax.plot([h1, h, h, h2], [y1, y1, y2, y2], color="black", lw=0.5)
        node_x[s + 1] = (y1 + y2) / 2.0
        node_h[s + 1] = h
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(float(np.max(height)) * 1.02 if len(height) else 1, 0)
    ax.axis("off")


def karyogram(
    cnv_adata,
    groupby: Optional[Union[str, Sequence[str]]] = None,
    *,
    order: str = "hclust",
    dendrogram: bool = True,
    title: Optional[str] = None,
    palette: Optional[dict] = None,
    figsize=(16, 8),
    legend: bool = True,
    rasterized: bool = True,
    show: Optional[bool] = None,
    save: Optional[str] = None,
):
    """Genome-wide CNV heatmap of all cells (epiAneufinder ``Karyogram.png``).

    Rows are cells and columns are windows, faceted by chromosome with widths
    proportional to the number of windows (as ``facet_grid(space = "free")``).
    With ``order="hclust"`` the cells are ordered by Ward (``ward.D``)
    clustering of their profiles and the dendrogram is drawn on the left, as in
    ``plot_karyo_gainloss``; ``groupby`` adds annotation bars as in
    ``plot_karyo_annotated``.

    Parameters
    ----------
    cnv_adata
        Output of :func:`epione.tl.cnv`.
    groupby
        ``obs`` column(s) drawn as annotation bars (e.g. ``"cnv_subclone"``).
    order
        ``"hclust"`` (dendrogram order), ``"groupby"`` (sort by the first
        ``groupby`` column, hclust order within groups) or ``"none"``.
    dendrogram
        Draw the dendrogram (only for ``order="hclust"``).
    palette
        Colours for annotation categories ``{category: colour}``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from ..tl._cnv import _get_hclust

    X = _dense(cnv_adata.X).astype(np.int8)
    n = X.shape[0]
    groups = [groupby] if isinstance(groupby, str) else list(groupby or [])
    merge = height = None
    if order in ("hclust", "groupby") and n > 1:
        merge, height, hc_order = _get_hclust(cnv_adata)
        idx = np.asarray(hc_order) - 1
    else:
        idx = np.arange(n)
    if order == "groupby" and groups:
        g = cnv_adata.obs[groups[0]].iloc[idx]
        codes = pd.Categorical(g).codes
        idx = idx[np.argsort(codes, kind="stable")]
    show_dend = dendrogram and order == "hclust" and merge is not None

    chrom = cnv_adata.var["chrom"].astype(str).to_numpy()
    chroms = list(dict.fromkeys(chrom))
    widths = [int((chrom == c).sum()) for c in chroms]

    fig = plt.figure(figsize=figsize)
    left_cols = []
    if show_dend:
        left_cols.append(0.08 * sum(widths))
    left_cols += [0.012 * sum(widths)] * len(groups)
    gs = fig.add_gridspec(1, len(left_cols) + len(chroms),
                          width_ratios=left_cols + widths, wspace=0.0)
    col = 0
    if show_dend:
        _draw_dendrogram(fig.add_subplot(gs[0, col]), merge, height, hc_order)
        col += 1
    cmap = ListedColormap([CNV_COLORS[0], CNV_COLORS[1], CNV_COLORS[2]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    ann_handles = []
    for g in groups:
        ax = fig.add_subplot(gs[0, col]); col += 1
        cat = pd.Categorical(cnv_adata.obs[g].iloc[idx])
        cats = list(cat.categories)
        if palette and all(c in palette for c in cats):
            cols = [palette[c] for c in cats]
        else:
            base = plt.get_cmap("tab20" if len(cats) > 10 else "tab10")
            cols = [base(i % base.N) for i in range(len(cats))]
        ax.imshow(cat.codes[:, None], aspect="auto", interpolation="nearest",
                  cmap=ListedColormap(cols), vmin=-0.5, vmax=len(cats) - 0.5,
                  rasterized=rasterized)
        ax.set_xticks([0]); ax.set_xticklabels([g], rotation=90, fontsize=8)
        ax.set_yticks([])
        ann_handles += [Patch(color=c, label=f"{g}: {k}") for k, c in zip(cats, cols)]
    Xo = X[idx]
    start = 0
    for ci, (c, w) in enumerate(zip(chroms, widths)):
        ax = fig.add_subplot(gs[0, col + ci])
        ax.imshow(Xo[:, start:start + w], aspect="auto", interpolation="nearest",
                  cmap=cmap, norm=norm, rasterized=rasterized)
        start += w
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(c.replace("chr", ""), fontsize=9)
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    if title is not None:
        fig.suptitle(title, fontsize=14)
    if legend:
        h = [Patch(color=CNV_COLORS[k], label=CNV_LABELS[k]) for k in (0, 1, 2)]
        fig.legend(handles=h + ann_handles, loc="center left", bbox_to_anchor=(0.91, 0.5),
                   frameon=False, fontsize=9)
        fig.subplots_adjust(right=0.9)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def cnv_profile(
    cnv_adata,
    cell: str,
    *,
    layer: str = "gc_corrected",
    figsize=(16, 3.2),
    ax=None,
    show: Optional[bool] = None,
):
    """Per-window counts of one cell coloured by CNV state
    (the ``somyplot`` panel of epiAneufinder ``plot_single_cell_profile``).

    Windows outside the cell's 1 %-99.9 % count quantiles are hidden, as in R.
    """
    import matplotlib.pyplot as plt

    i = list(cnv_adata.obs_names).index(cell)
    y = _dense(cnv_adata.layers[layer][i]).ravel().astype(float)
    s = _dense(cnv_adata.X[i]).ravel()
    keep = (y < np.quantile(y, 0.999)) & (y > np.quantile(y, 0.01))
    x = np.arange(y.size)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for k in (1, 0, 2):
        m = keep & (s == k)
        ax.scatter(x[m], y[m], s=4, alpha=0.5, color=CNV_COLORS[k], label=CNV_LABELS[k],
                   rasterized=True, linewidths=0)
    chrom = cnv_adata.var["chrom"].astype(str).to_numpy()
    edges = np.flatnonzero(np.r_[True, chrom[1:] != chrom[:-1], True])
    for e in edges[1:-1]:
        ax.axvline(e - 0.5, color="grey", lw=0.4)
    mids = (edges[:-1] + edges[1:]) / 2
    ax.set_xticks(mids)
    ax.set_xticklabels([c.replace("chr", "") for c in chrom[edges[:-1]]], fontsize=8)
    ax.set_xlim(0, y.size)
    ax.set_ylabel("counts")
    ax.set_title(f"{cell}  (library size {int(y[keep].sum())})", fontsize=10)
    ax.legend(markerscale=4, frameon=False, loc="upper right", fontsize=8)
    if show:
        plt.show()
    return ax
