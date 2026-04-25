"""Hi-C contact-matrix visualisation (bulk + single-cell).

Single home for every Hi-C heatmap / decay / coverage helper, regardless
of whether the source is a bulk ``.cool`` or a per-cell imputed ``.npz``
inside an sc-Hi-C AnnData (output of :mod:`epione.single.hic`).

Functions:

    * :func:`plot_contact_matrix`  bulk-cool log-scale region heatmap
    * :func:`plot_decay_curve`     P(s) genomic-distance vs mean-contact
    * :func:`plot_coverage`        per-bin coverage + ICE weight panels
    * :func:`plot_cell_contacts`   per-cell heatmap from sc-Hi-C imputed
                                    matrices (raw vs imputed view)

A richer composite "Hi-C + tracks + TADs" panel will land in
:func:`plot_track_multi` (PR alongside the v0.4 datasets module).
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
    # cooler's region parser rejects Python-style underscores + commas
    # inside coordinate literals (chr2L:10_000_000-13_000_000), but
    # underscores are legal in chrom names (chr_synt). Strip only on
    # the span side of the colon.
    if ":" in region:
        _chrom_part, _span_part = region.split(":", 1)
        region_clean = (
            _chrom_part + ":" + _span_part.replace(",", "").replace("_", "")
        )
    else:
        region_clean = region
    mat = clr.matrix(balance=balance).fetch(region_clean)
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


def plot_decay_curve(
    cool_path: Union[str, Path],
    *,
    chromosomes: Optional[list] = None,
    balance: bool = True,
    max_offset: Optional[int] = None,
    figsize: Tuple[float, float] = (5.5, 4.0),
    title: Optional[str] = None,
    color: str = "#3a6eb3",
    ax=None,
):
    """Contact-decay curve P(s) — mean contact frequency vs genomic
    separation, the canonical Hi-C QC plot.

    For each diagonal offset ``s`` (in bin units), we average all
    intra-chromosome cells at that offset across the requested
    chromosomes. Plotted on log-log axes the curve should be roughly
    linear with slope around -1 for a healthy library; a flat
    plateau or non-monotone shape signals contamination, undersampled
    libraries, or aggressive over-balancing.

    Arguments:
        cool_path: path to the ``.cool`` (balanced or raw).
        chromosomes: subset to compute on. Default: all.
        balance: use balanced contacts when True (requires
            ``balance_cool`` to have been run).
        max_offset: cap the x-axis at this many bins. ``None`` = full
            chromosome span.
        figsize, title, color: cosmetic. Default colour matches the
            ``Tcell`` blue used in the footprint tutorial.
        ax: optional existing axis.

    Returns:
        ``(fig, ax, df)`` — the figure / axis and a long DataFrame of
        ``(offset_bin, distance_bp, mean_contact, n_pairs)`` per row,
        useful for further plotting / fitting.
    """
    import cooler
    import pandas as pd
    import matplotlib.pyplot as plt

    clr = cooler.Cooler(str(cool_path))
    binsize = int(clr.binsize)
    chromosomes = chromosomes or list(clr.chromnames)

    rows = []
    for chrom in chromosomes:
        mat = clr.matrix(balance=balance, sparse=False).fetch(chrom)
        n = mat.shape[0]
        upper = max_offset if max_offset is not None else n - 1
        for k in range(min(upper + 1, n)):
            diag = np.diag(mat, k=k)
            finite = diag[np.isfinite(diag)]
            if finite.size == 0:
                continue
            rows.append({
                "chrom": chrom,
                "offset_bin": k,
                "distance_bp": k * binsize,
                "mean_contact": float(finite.mean()),
                "n_pairs": int(finite.size),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "no contacts to plot — check that the cool has data for the "
            "requested chromosomes."
        )

    # Aggregate across chromosomes by offset.
    agg = (df.groupby("offset_bin")
           .agg(distance_bp=("distance_bp", "first"),
                mean_contact=("mean_contact", "mean"),
                n_pairs=("n_pairs", "sum"))
           .reset_index())
    # Drop offset 0 — self-contacts are dominated by diagonal artefacts.
    agg = agg[agg["offset_bin"] > 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.loglog(agg["distance_bp"], agg["mean_contact"],
              color=color, lw=1.4)
    ax.set_xlabel("genomic distance (bp)")
    ax.set_ylabel("mean contact" + (" (balanced)" if balance else " (raw)"))
    ax.set_title(title or f"P(s) — {Path(cool_path).name}")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return fig, ax, df


def plot_coverage(
    cool_path: Union[str, Path],
    *,
    balance: bool = False,
    bins: int = 50,
    figsize: Tuple[float, float] = (8.0, 3.5),
    ax=None,
):
    """Per-bin total contact-coverage distribution — diagnostic for
    `balance_cool`'s `mad_max` / `min_nnz` filtering.

    Renders side-by-side: (left) the histogram of per-bin coverage
    on a log y-axis, (right) the per-bin balance ``weight`` if
    available, also on log y. Bins with zero/NaN weight are masked
    by ICE and excluded from balanced contacts.

    Arguments:
        cool_path: path to the ``.cool``.
        balance: if True, take coverage from the balanced matrix; else
            raw counts. Most informative as raw + showing where the
            ICE mask landed.
        bins: histogram bin count.
        figsize, ax: cosmetic.

    Returns:
        ``(fig, axes)`` with two axes (coverage hist, weight hist).
    """
    import cooler
    import matplotlib.pyplot as plt

    clr = cooler.Cooler(str(cool_path))
    bin_tab = clr.bins()[:]
    # Per-bin sum from raw or balanced matrix.
    cov_per_bin = np.zeros(len(bin_tab), dtype=np.float64)
    for chrom in clr.chromnames:
        mat = clr.matrix(balance=balance, sparse=False).fetch(chrom)
        # Bins for this chrom.
        idx = bin_tab.query("chrom == @chrom").index.to_numpy()
        cov_per_bin[idx] = np.nansum(mat, axis=1)
    has_weight = "weight" in bin_tab.columns

    if ax is None:
        fig, axes = plt.subplots(1, 2 if has_weight else 1, figsize=figsize)
        if not has_weight:
            axes = [axes]
    else:
        fig = ax.figure
        axes = [ax]

    finite = cov_per_bin[cov_per_bin > 0]
    axes[0].hist(np.log10(finite + 1e-9), bins=bins,
                 color="#3a6eb3", alpha=0.7, edgecolor="white", lw=0.4)
    axes[0].set_xlabel(r"log$_{10}$ per-bin coverage")
    axes[0].set_ylabel("bins")
    axes[0].set_title("Per-bin contact coverage")
    n_zero = int((cov_per_bin == 0).sum())
    axes[0].text(0.02, 0.95, f"{n_zero} zero-coverage bins",
                 transform=axes[0].transAxes,
                 va="top", ha="left", fontsize=8, color="#666")

    if has_weight:
        w = bin_tab["weight"].to_numpy()
        finite_w = w[np.isfinite(w) & (w > 0)]
        axes[1].hist(np.log10(finite_w), bins=bins,
                     color="#c13e3e", alpha=0.7, edgecolor="white", lw=0.4)
        axes[1].set_xlabel(r"log$_{10}$ ICE weight")
        axes[1].set_ylabel("bins")
        axes[1].set_title("ICE balance weights")
        n_masked = int(np.isnan(w).sum())
        axes[1].text(0.02, 0.95, f"{n_masked} masked bins",
                     transform=axes[1].transAxes,
                     va="top", ha="left", fontsize=8, color="#666")

    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig, axes


def plot_cell_contacts(
    adata,
    cell_id: str,
    *,
    chromosome: str,
    use_imputed: bool = True,
    log: bool = True,
    cmap: str = "YlOrRd",
    figsize: Tuple[float, float] = (5.0, 4.5),
    ax=None,
):
    """Per-cell contact heatmap for a single chromosome.

    Useful for sanity-checking imputation: the raw matrix should look
    speckled (sparse contacts), the imputed one should show a clear
    diagonal + off-diagonal density.

    Arguments:
        adata: AnnData from :func:`epione.single.hic.load_cool_collection`
            (or :func:`load_scool_cells`).
        cell_id: row in ``adata.obs_names``.
        chromosome: e.g. ``'chr1'``.
        use_imputed: read the imputed ``.npz`` (requires
            :func:`epione.single.hic.impute_cells` to have run).
            ``False`` reads raw counts from the original ``.cool``
            for comparison.
        log, cmap, figsize, ax: cosmetic.

    Returns:
        ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    if cell_id not in adata.obs_names:
        raise KeyError(f"cell_id {cell_id!r} not in adata.obs_names")

    if use_imputed:
        info = adata.uns.get("hic", {})
        imputed_dir = info.get("imputed_dir")
        if imputed_dir is None:
            raise ValueError(
                "adata.uns['hic']['imputed_dir'] not set — call with "
                "use_imputed=False or run epione.single.hic.impute_cells "
                "first"
            )
        z = np.load(Path(imputed_dir) / f"{cell_id}.npz")
        if chromosome not in z.files:
            raise KeyError(
                f"chromosome {chromosome!r} not in imputed file for "
                f"{cell_id}; available: {sorted(z.files)}"
            )
        mat = z[chromosome]
    else:
        import cooler
        cool_path = adata.obs.loc[cell_id, "cool_path"]
        clr = cooler.Cooler(str(cool_path))
        mat = np.asarray(clr.matrix(balance=False).fetch(chromosome),
                         dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    finite = mat[np.isfinite(mat) & (mat > 0)]
    vmin = float(np.quantile(finite, 0.01)) if finite.size else 0.0
    vmax = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    if log and vmax > 0:
        vmin = max(vmin, 1e-6)
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))
    else:
        norm = Normalize(vmin=vmin, vmax=vmax if vmax > 0 else 1.0)

    img = ax.imshow(mat, cmap=cmap, norm=norm, interpolation="none")
    ax.set_title(
        f"{cell_id} - {chromosome}"
        + (" (imputed)" if use_imputed else " (raw)")
    )
    ax.set_xlabel("bin")
    ax.set_ylabel("bin")
    fig.colorbar(img, ax=ax, shrink=0.8,
                 label="log contact" if log else "contact")
    return fig, ax
