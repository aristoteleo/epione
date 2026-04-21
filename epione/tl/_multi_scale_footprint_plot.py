"""scPrinter-style 2D footprint heatmap (scale × position)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ._multi_scale_footprint import MultiScaleFootprint


def plot_multi_scale_footprint(
    msfp: MultiScaleFootprint,
    *,
    groups: Optional[Sequence[str]] = None,
    order: Optional[Sequence[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r",
    trim: int = 0,
    zoom: Optional[int] = None,
    show_score: bool = True,
    show_tf_zone: bool = True,
    tf_zone_scales: Tuple[int, int] = (20, 40),
    tf_zone_radius: int = 5,
    figsize: Optional[Tuple[float, float]] = None,
    ncols: int = 4,
    show_positions: bool = True,
    show: bool = True,
):
    """Render the ``MultiScaleFootprint.dispersion`` tensor as a 2D
    heatmap, one panel per group (matches scPrinter's comparison plot).

    Parameters
    ----------
    msfp
        Output of :func:`multi_scale_footprint`.
    groups, order
        Subset / reorder the group panels.
    vmin, vmax
        Colour-scale limits. Default: symmetric around 0, auto from
        the 99th percentile of ``|dispersion|`` across plotted groups.
    cmap
        Colour map. ``'RdBu_r'`` (default) gives red = footprint,
        blue = anti-footprint. Use ``'Blues'`` for the scPrinter look.
    trim
        Number of positions to drop from each edge of the position
        axis (useful to hide the outer flanks that the null model
        defines as baseline). Ignored if ``zoom`` is set.
    zoom
        If given, restrict the position axis to ``±zoom`` bp around
        the motif centre. Overrides ``trim``. Recommended for
        small-scale TF footprint inspection (e.g. ``zoom=50``).
    show_score
        Annotate each panel title with its :func:`footprint_score`
        value (mean dispersion in the TF-protection zone), so you can
        rank cell types by footprint strength at a glance.
    show_tf_zone
        Overlay a translucent band at ``tf_zone_scales`` — the scale
        range where single-TF footprints typically dominate — to guide
        the reader's eye to the biologically meaningful region.
    tf_zone_scales
        ``(min_scale, max_scale)`` in bp for the overlay band and the
        footprint-score calculation.
    tf_zone_radius
        Half-width (bp) of the position zone used by the footprint
        score.
    ncols
        Max columns in the subplot grid.
    show_positions
        Label the x-axis in bp offsets.

    Returns
    -------
    ``(fig, axes)`` — the matplotlib figure and a flat list of axes.
    """
    import matplotlib.pyplot as plt

    disp = msfp.dispersion
    scales = msfp.scales
    positions = msfp.positions

    # Pick groups
    available = list(msfp.groups)
    if groups is not None:
        usable = [g for g in groups if g in available]
    else:
        usable = list(available)
    if order is not None:
        usable = [g for g in order if g in usable]
    if not usable:
        raise ValueError(f"no groups to plot; available = {available!r}")

    gidx = [available.index(g) for g in usable]
    panels = disp[gidx]

    if zoom is not None:
        mask = np.abs(positions) <= int(zoom)
        panels = panels[..., mask]
        positions = positions[mask]
    elif trim > 0:
        panels = panels[..., trim:-trim]
        positions = positions[trim:-trim]

    scores = None
    if show_score:
        from ._multi_scale_footprint import footprint_score
        scores = footprint_score(msfp, position_radius=tf_zone_radius,
                                  scale_range=tf_zone_scales)

    # Auto colour scale.
    if vmin is None or vmax is None:
        lim = float(np.nanpercentile(np.abs(panels), 99))
        if vmax is None: vmax = lim
        if vmin is None: vmin = -lim if cmap.lower() in ("rdbu_r", "rdbu",
            "coolwarm", "bwr", "seismic") else 0.0

    n = len(usable)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (3.6 * ncols, 3.0 * nrows)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize,
        squeeze=False, sharex=True, sharey=True,
    )
    axes_flat = axes.ravel().tolist()

    # Extent so imshow labels are accurate.
    x_lo, x_hi = float(positions[0]), float(positions[-1])
    y_lo, y_hi = float(scales[0]), float(scales[-1])

    for i, g in enumerate(usable):
        ax = axes_flat[i]
        im = ax.imshow(
            panels[i],
            aspect="auto",
            origin="lower",
            extent=[x_lo, x_hi, y_lo, y_hi],
            cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        title = g
        if scores is not None:
            title += f"  (score={scores[g]:+.3f})"
        ax.set_title(title, fontsize=9)
        if show_positions:
            ax.axvline(0, color="black", lw=0.4, ls=":")
        if show_tf_zone:
            zmin, zmax = tf_zone_scales
            if y_lo <= zmax and y_hi >= zmin:
                ax.axhspan(max(zmin, y_lo), min(zmax, y_hi),
                            color="orange", alpha=0.08, zorder=0)
        if i % ncols == 0:
            ax.set_ylabel("scale (bp)")
        if i // ncols == nrows - 1:
            ax.set_xlabel("position (bp)")

    # Hide unused axes.
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    # Shared colour bar.
    cbar = fig.colorbar(
        im, ax=axes, shrink=0.6, pad=0.02, aspect=30,
        label=(f"dispersion ({msfp.null} null)"
                if msfp.null != "none" else "-log10 P(X ≤ obs)"),
    )

    fig.suptitle(
        f"{msfp.motif} · multi-scale footprint"
        f"  (scales {int(scales[0])}–{int(scales[-1])} bp)",
        fontsize=11, y=0.995,
    )
    if show:
        from IPython.display import display
        display(fig)
        plt.close(fig)
    return fig, axes_flat


__all__ = ["plot_multi_scale_footprint"]
