"""Figures for PRINT footprints and TF binding scores.

Both plots answer the same question PRINT asks -- *where are fewer cuts
than the sequence bias predicts?* -- so both put the raw evidence (the
cut track, with the bias-expected cuts drawn over it) directly above the
derived score, sharing one genomic x-axis. scPrinter's own figures show
the score alone, which makes a footprint impossible to sanity-check by
eye.

Conventions follow the rest of ``epione.pl``: return ``(fig, axes)``,
draw into a fresh figure, ``show=True`` displays and closes it in a
notebook.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["plot_tf_binding_score", "GROUP_COLORS"]

#: Categorical palette (validated for colour-vision deficiency), in the
#: fixed order groups are assigned. A 9th+ group reuses the order -- in
#: practice, subset ``groups=`` instead.
GROUP_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def _style_axis(ax, *, left=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def draw_cut_track(ax, x, cuts, bias, color, label=None, *, smooth=0):
    """Observed cuts (filled) with bias-expected cuts (dashed) on top.

    The bias is rescaled so its total matches the observed total over
    the window: what matters is where the two shapes part, not their
    absolute scale.
    """
    cuts = np.asarray(cuts, dtype=float)
    bias = np.asarray(bias, dtype=float)
    if smooth and smooth > 1:
        k = np.ones(int(smooth)) / int(smooth)
        cuts = np.convolve(cuts, k, mode="same")
        bias = np.convolve(bias, k, mode="same")
    expected = bias * (cuts.sum() / bias.sum()) if bias.sum() > 0 else bias
    ax.fill_between(x, 0, cuts, color=color, alpha=0.75, lw=0, step="mid")
    ax.plot(x, expected, color=INK2, lw=0.9, ls=(0, (3, 2)))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, max(cuts.max(), expected.max(), 1e-9) * 1.15)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _style_axis(ax)
    if label:
        ax.text(0.005, 0.95, label, transform=ax.transAxes, va="top",
                ha="left", fontsize=8.5, fontweight="bold", color=INK)


def plot_tf_binding_score(
    result,
    *,
    region: Optional[str] = None,
    groups: Optional[Sequence[str]] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "Blues",
    smooth: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show: bool = True,
):
    """Cut tracks and PRINT binding scores along one region.

    One cut track per group (observed cuts filled; bias-expected cuts
    dashed), then a heatmap of binding probability -- groups x sites --
    on the same genomic axis, so a high score can be traced back to the
    dip in cuts that produced it.

    Parameters
    ----------
    result
        :class:`~epione.tl.TFBindingResult` from
        :func:`~epione.tl.tf_binding_score` (with ``keep_tracks=True``).
    region
        ``'chrom:start-end'`` to show. Defaults to the region with the
        highest maximum score.
    groups
        Subset / order of groups.
    vmin, vmax
        Colour range of the score heatmap; it is a probability, so 0-1
        by default.
    smooth
        Running-mean width (bp) applied to the cut tracks for display
        only. 0 shows raw counts.

    Returns
    -------
    ``(fig, axes)`` with one axis per track plus the heatmap last.
    """
    import matplotlib.pyplot as plt

    if not result.tracks:
        raise ValueError("result has no tracks; rerun tf_binding_score(keep_tracks=True)")
    sites = result.sites
    if region is None:
        best = (sites.assign(_m=result.score.max(0))
                .groupby("region")["_m"].max().idxmax())
        region = best
    if region not in result.tracks:
        raise KeyError(f"{region!r} not among scored regions")
    tr = result.tracks[region]

    names = list(result.groups)
    show_groups = [g for g in (groups or names) if g in names]
    if not show_groups:
        raise ValueError(f"no groups to plot; available {names!r}")
    gi = [names.index(g) for g in show_groups]

    mask = (sites["region"] == region).to_numpy()
    pos = sites.loc[mask, "center"].to_numpy()
    sc = result.score[np.ix_(gi, np.flatnonzero(mask))]
    order = np.argsort(pos)
    pos, sc = pos[order], sc[:, order]

    s, e = tr["start"], tr["end"]
    x = np.arange(s, e)
    n = len(show_groups)
    if figsize is None:
        figsize = (9.5, 0.9 * n + 0.32 * n + 1.6)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n + 1, 2, width_ratios=[60, 1],
                          height_ratios=[1.0] * n + [max(0.35 * n, 0.8)],
                          hspace=0.12, wspace=0.02, top=0.93, bottom=0.09)
    axes = []
    for k, g in enumerate(show_groups):
        ax = fig.add_subplot(gs[k, 0], sharex=axes[0] if axes else None)
        draw_cut_track(ax, x, tr["insertions"][names.index(g)], tr["bias"],
                       GROUP_COLORS[k % len(GROUP_COLORS)], g, smooth=smooth)
        ax.tick_params(labelbottom=False)
        axes.append(ax)
    axes[0].text(0.995, 0.95, "filled: observed cuts   dashed: bias-expected",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=7.5, color=INK2)

    hax = fig.add_subplot(gs[n, 0], sharex=axes[0])
    # Each site is drawn as a cell centred on it, as wide as the spacing.
    if len(pos) > 1:
        mids = (pos[1:] + pos[:-1]) / 2
        edges = np.concatenate([[pos[0] - (mids[0] - pos[0])], mids,
                                [pos[-1] + (pos[-1] - mids[-1])]])
    else:
        edges = np.array([pos[0] - 5, pos[0] + 5])
    mesh = hax.pcolormesh(edges, np.arange(n + 1), sc, cmap=cmap,
                          vmin=vmin, vmax=vmax, shading="flat")
    hax.set_yticks(np.arange(n) + 0.5)
    hax.set_yticklabels(show_groups, fontsize=8, color=INK)
    hax.invert_yaxis()
    hax.set_xlim(x[0], x[-1])
    _style_axis(hax)
    hax.set_xlabel(f"{region.split(':')[0]} position (bp)", color=INK2, fontsize=9)
    hax.ticklabel_format(axis="x", style="plain", useOffset=False)
    axes.append(hax)

    cax = fig.add_subplot(gs[n, 1])
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.set_label("binding p", fontsize=8, color=INK2)

    if title is None:
        title = (f"PRINT TF binding score · {region} · "
                 f"{result.bias_model} bias")
    fig.suptitle(title, fontsize=11, fontweight="bold", color=INK, y=0.995)
    if show:
        try:
            from IPython.display import display
            display(fig)
            plt.close(fig)
        except Exception:
            pass
    return fig, axes
