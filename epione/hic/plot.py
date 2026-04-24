"""Hi-C contact-matrix visualisation.

Minimal, publication-shaped log-scale heatmap for a genomic region,
reading directly from a ``.cool`` file via the cooler API. A richer
composite "Hi-C + tracks + TADs" panel will land in Phase 4 alongside
:mod:`epione.pl._peak2gene` / :func:`epione.bulk.bigwig.plot_track_multi`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def plot_contact_matrix(
    cool_path: Union[str, Path],
    region: str,
    *,
    balance: bool = True,
    cmap: str = "YlOrRd",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    log: bool = True,
    figsize: Tuple[float, float] = (6.0, 5.5),
    title: Optional[str] = None,
    ax=None,
    colorbar: bool = True,
):
    """Heatmap of a single genomic region at the ``.cool``'s resolution.

    Arguments:
        cool_path: path to a ``.cool`` file (possibly already balanced).
        region: UCSC-style interval — ``'chr2L'`` for a whole chromosome,
            ``'chr2L:10_000_000-15_000_000'`` for a slice.
        balance: if True and the cool has a ``weight`` column (from
            :func:`balance_cool`), plot balanced contacts; else raw counts.
        cmap: matplotlib colour-map. ``'YlOrRd'`` matches the HiCExplorer /
            cooler defaults; try ``'magma'`` / ``'viridis'`` for dark
            backgrounds.
        vmin, vmax: colour-scale limits in the *pre-log* space when
            ``log=True``. Defaults auto-detect: 1st/98th percentile of
            finite values.
        log: ``log(1 + contacts)`` colour mapping. Almost always what you
            want for Hi-C.
        figsize: figure size when ``ax`` is None.
        title: optional axes title. Defaults to ``<cool basename> — <region>``.
        ax: existing matplotlib axis to draw into.
        colorbar: append a colour-bar on the right.

    Returns:
        ``(fig, ax, img)`` — the matplotlib figure, axis, and the image
        returned by ``ax.imshow`` so the caller can customise downstream.
    """
    import cooler
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    clr = cooler.Cooler(str(cool_path))
    mat = clr.matrix(balance=balance).fetch(region)
    finite = mat[np.isfinite(mat)]
    if vmin is None:
        vmin = float(np.quantile(finite, 0.01)) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    if log:
        # clip to strictly positive for LogNorm; masked pixels become white.
        vmin = max(vmin, 1e-6)
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Resolve region extents for axis ticks.
    if ":" in region:
        chrom, span = region.split(":", 1)
        lo_str, hi_str = span.replace(",", "_").replace("_", "").split("-")
        lo, hi = int(lo_str), int(hi_str)
    else:
        chrom = region
        lo = 0
        hi = clr.chromsizes[chrom]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    img = ax.imshow(
        mat, origin="upper",
        extent=[lo, hi, hi, lo],
        cmap=cmap, norm=norm, interpolation="none",
    )
    ax.set_xlabel(f"{chrom} position (bp)")
    ax.set_ylabel(f"{chrom} position (bp)")
    ax.set_title(title or f"{Path(cool_path).name} — {region}")
    if colorbar:
        fig.colorbar(img, ax=ax, shrink=0.8,
                     label="log contact" if log else "contact")
    return fig, ax, img
