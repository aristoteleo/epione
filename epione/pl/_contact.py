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


def plot_saddle(
    saddle_mat,
    edges=None,
    *,
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: Tuple[float, float] = (4.5, 4.0),
    title: Optional[str] = None,
    label: str = "log2(O/E)",
    ax=None,
):
    """A/B compartment saddle plot.

    Heatmap of mean log2(observed / expected) contact frequency across
    bin-pairs binned by their compartment eigenvector quantile. Strong
    AA / BB on the diagonal corners + weak AB / BA on the
    anti-diagonal corners is the signature of a well-compartmentalised
    genome; flatness signals weak compartmentalisation (e.g. early
    embryos, mitotic).

    Arguments:
        saddle_mat: ``(n_bins, n_bins)`` matrix from
            :func:`epione.bulk.hic.saddle`. log2-transformed
            internally if the input is on linear scale (we detect
            ``≥0`` everywhere → assume linear O/E and log2 it).
        edges: per-bin quantile edges (only used to label the colour
            bar / axes; pass the second return of
            :func:`epione.bulk.hic.saddle`).
        cmap, vmin, vmax: colour scale for log2(O/E). Default
            ``coolwarm`` with ±1 limits is the cooltools convention.
        figsize, title, ax: cosmetic.
        label: colour-bar label.

    Returns:
        ``(fig, ax, img)``.
    """
    import matplotlib.pyplot as plt

    M = np.asarray(saddle_mat, dtype=np.float64)
    if (M >= 0).all():
        with np.errstate(divide="ignore"):
            M = np.log2(M)
        M = np.where(np.isfinite(M), M, np.nan)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    img = ax.imshow(M, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                    interpolation="none")
    n = M.shape[0]
    # Tick at corners + middle for orientation.
    ax.set_xticks([0, n / 2 - 0.5, n - 1])
    ax.set_yticks([0, n / 2 - 0.5, n - 1])
    ax.set_xticklabels(["B", " ", "A"])
    ax.set_yticklabels(["B", " ", "A"])
    ax.set_xlabel("compartment quantile")
    ax.set_ylabel("compartment quantile")
    ax.set_title(title or "A/B saddle")
    fig.colorbar(img, ax=ax, shrink=0.8, label=label)
    return fig, ax, img


def plot_compartments(
    eig_track,
    *,
    chromosome: str,
    track_column: str = "E1",
    figsize: Tuple[float, float] = (8.0, 1.8),
    color_pos: str = "#c13e3e",
    color_neg: str = "#3a6eb3",
    title: Optional[str] = None,
    ax=None,
):
    """Compartment eigenvector track for one chromosome.

    Bar plot in two colours — red where ``E1 > 0`` (A compartment),
    blue where ``E1 < 0`` (B). The horizontal axis is genomic position
    in megabases.

    Arguments:
        eig_track: per-bin DataFrame from
            :func:`epione.bulk.hic.compartments`.
        chromosome: which chrom to plot.
        track_column: column to plot. Default ``E1``.
        figsize, color_pos, color_neg, title, ax: cosmetic.

    Returns:
        ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    sub = eig_track.loc[eig_track["chrom"] == chromosome].copy()
    if sub.empty:
        raise KeyError(
            f"chromosome {chromosome!r} not in eig_track; available: "
            f"{sorted(eig_track['chrom'].unique())}"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    starts = sub["start"].to_numpy() / 1e6
    widths = (sub["end"] - sub["start"]).to_numpy() / 1e6
    vals = sub[track_column].to_numpy()
    colors = [color_pos if v > 0 else color_neg for v in np.nan_to_num(vals)]
    ax.bar(starts, vals, width=widths, color=colors, align="edge",
           edgecolor="none")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel(f"{chromosome} (Mb)")
    ax.set_ylabel(track_column)
    ax.set_title(title or f"compartments — {chromosome}")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return fig, ax


def plot_contact_triangle(
    cool_path: Union[str, Path],
    region: str,
    *,
    balance: bool = True,
    cmap: str = "Reds",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    max_distance: Optional[int] = None,
    log: bool = False,
    figsize: Tuple[float, float] = (8.0, 1.6),
    title: Optional[str] = None,
    ax=None,
    colorbar: bool = True,
):
    """45-degree rotated 'pyramid' contact map for a genomic region.

    Renders the upper triangle of the contact matrix as a flat
    horizontal band — the canonical Hi-C pyramid view used in
    Maziak et al. 2026 Fig 1c (and in HiGlass / cooltools tutorials).
    Genomic position runs along the x-axis; contact distance from
    the diagonal runs up the y-axis. Useful for comparing many
    stages stacked above one another.

    Arguments:
        cool_path: ``.cool`` / ``.mcool::resolutions/N`` path.
        region: UCSC-style interval. Underscores / commas in
            coordinates are stripped before passing to cooler.
        balance: use balanced contacts (requires ``balance_cool`` to
            have been run, or an mcool layer with ``weight``).
        cmap, vmin, vmax: matplotlib colour mapping. ``vmax=None``
            auto-detects the 99-th percentile of finite values.
        max_distance: cap the displayed contact distance in bp; the
            band above that is cropped. Useful for fine-resolution
            cool where short-range contacts dominate. ``None`` =
            full upper triangle (height = window width / 2).
        log: log-scale colour mapping (``log10(1 + contacts)``).
            Default linear, matching the paper's Fig 1c.
        figsize, title, ax, colorbar: cosmetic.

    Returns:
        ``(fig, ax, mesh)`` — the figure, axis, and the
        ``QuadMesh`` returned by ``ax.pcolormesh``.
    """
    import cooler
    import matplotlib.pyplot as plt

    clr = cooler.Cooler(str(cool_path))

    if ":" in region:
        chrom, span = region.split(":", 1)
        lo_str, hi_str = span.replace(",", "").replace("_", "").split("-")
        lo, hi = int(lo_str), int(hi_str)
    else:
        chrom = region
        lo = 0
        hi = int(clr.chromsizes[chrom])

    binsize = int(clr.binsize)

    # Pad the fetched window by max_distance/2 on each side so that
    # cells whose midpoint sits inside [lo, hi] but whose endpoints are
    # outside still get rendered. Without this, the rendered band has
    # triangular cut-offs at both edges instead of rectangular ends.
    pad = int(max_distance) // 2 if max_distance is not None else 0
    chrom_size = int(clr.chromsizes[chrom])
    fetch_lo = max(0, lo - pad)
    fetch_hi = min(chrom_size, hi + pad)
    region_clean = f"{chrom}:{fetch_lo}-{fetch_hi}"
    mat = clr.matrix(balance=balance).fetch(region_clean)
    n = mat.shape[0]

    # Build a (max_d+1, n) array where row d is the k=d diagonal of the
    # contact matrix — i.e. M_diag[d, p] is the contact between bin p and
    # bin p+d in the *padded* window.
    if max_distance is not None:
        max_d = min(int(max_distance) // binsize, n - 1)
    else:
        max_d = n - 1

    M_diag = np.full((max_d + 1, n), np.nan)
    for d in range(max_d + 1):
        diag = np.diag(mat, k=d)
        M_diag[d, : len(diag)] = diag

    # Per-row x corners: shift row d by d/2 bins so cell midpoints align
    # with the bin-pair midpoint. ``fetch_lo`` is the start of the
    # padded fetched window (used as the absolute genomic origin of
    # the pcolormesh; ``set_xlim`` below crops to the requested
    # [lo, hi] range so the band ends are flat / rectangular).
    p_arr = np.arange(n + 1, dtype=float)
    d_arr = np.arange(max_d + 2, dtype=float)
    Xc = p_arr[None, :] * binsize + fetch_lo + d_arr[:, None] * binsize * 0.5
    Yc = (d_arr[:, None] * binsize) * np.ones_like(Xc)

    M = M_diag

    if log:
        with np.errstate(invalid="ignore"):
            M = np.log10(M + 1.0)
    finite = M[np.isfinite(M)]
    if vmax is None:
        vmax = float(np.quantile(finite, 0.99)) if finite.size else 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(
        Xc, Yc, M, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, max_distance if max_distance is not None else (hi - lo))
    # Aspect = 'auto' so the band is wide-and-thin (matching the
    # canonical Maziak Fig 1c stacked layout); the caller controls the
    # actual visual aspect via ``figsize``.
    ax.set_xlabel(f"{chrom} position (bp)")
    ax.set_ylabel("contact distance (bp)")
    ax.set_title(title or f"{Path(cool_path).name} — {region}")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if colorbar:
        fig.colorbar(mesh, ax=ax, shrink=0.7,
                     label="log10(1+contact)" if log else "contact")
    return fig, ax, mesh


def plot_insulation(
    insulation_df,
    *,
    chromosome: str,
    window_bp: Optional[int] = None,
    region_start: Optional[int] = None,
    region_end: Optional[int] = None,
    boundaries: bool = True,
    boundaries_df=None,
    figsize: Tuple[float, float] = (8.0, 1.6),
    color: str = "#3a6eb3",
    title: Optional[str] = None,
    ax=None,
):
    """Per-bin insulation-score line plot for one chromosome.

    Renders the diamond-insulation score from :func:`epione.bulk.hic.insulation`
    as a line; optionally drops vertical ticks at the called TAD
    boundaries (red by default). Designed to stack on top of a
    contact-pyramid panel in figures of the Maziak 2026 / Chang 2024
    style.

    Arguments:
        insulation_df: output of :func:`epione.bulk.hic.insulation`.
        chromosome: which chrom to plot.
        window_bp: which insulation window (when the table holds
            several). Default = smallest available.
        region_start, region_end: optional zoom in bp.
        boundaries: draw vertical ticks at boundaries from
            ``boundaries_df`` (or auto-extract via
            :func:`epione.bulk.hic.tad_boundaries`).
        boundaries_df: pre-computed boundary DataFrame; if ``None``
            and ``boundaries=True``, derive on the fly.
        figsize, color, title, ax: cosmetic.

    Returns:
        ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    cols = [c for c in insulation_df.columns
            if c.startswith("log2_insulation_score_")]
    if not cols:
        raise KeyError(
            "insulation_df missing log2_insulation_score_* columns"
        )
    avail = [int(c.rsplit("_", 1)[-1]) for c in cols]
    if window_bp is None:
        window_bp = min(avail)
    score_col = f"log2_insulation_score_{int(window_bp)}"

    sub = insulation_df.loc[insulation_df["chrom"] == chromosome].copy()
    if sub.empty:
        raise KeyError(
            f"chromosome {chromosome!r} not in insulation_df"
        )
    if region_start is not None:
        sub = sub[sub["end"] > region_start]
    if region_end is not None:
        sub = sub[sub["start"] < region_end]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    centers = (sub["start"] + sub["end"]) / 2 / 1e6
    ax.plot(centers, sub[score_col], color=color, lw=1.0)
    ax.axhline(0, color="black", lw=0.4, alpha=0.6)

    if boundaries:
        if boundaries_df is None:
            from ..bulk.hic._insulation import tad_boundaries
            boundaries_df = tad_boundaries(insulation_df, window_bp=window_bp)
        bsub = boundaries_df[boundaries_df["chrom"] == chromosome]
        if region_start is not None:
            bsub = bsub[bsub["end"] > region_start]
        if region_end is not None:
            bsub = bsub[bsub["start"] < region_end]
        bcenters = (bsub["start"] + bsub["end"]) / 2 / 1e6
        ymin, ymax = ax.get_ylim()
        ax.vlines(bcenters, ymin, ymin + 0.1 * (ymax - ymin),
                  color="#c13e3e", lw=0.6, alpha=0.85)

    ax.set_xlabel(f"{chromosome} (Mb)")
    ax.set_ylabel(f"log2 insulation\n(window {int(window_bp)/1000:.0f} kb)")
    ax.set_title(title or f"insulation — {chromosome}")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return fig, ax


def plot_apa(
    apa_matrix,
    *,
    flank: Optional[int] = None,
    binsize: Optional[int] = None,
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    log: bool = True,
    figsize: Tuple[float, float] = (4.0, 3.6),
    title: Optional[str] = None,
    ax=None,
    colorbar: bool = True,
):
    """Aggregate peak-analysis (APA) heatmap from
    :func:`epione.bulk.hic.pileup`.

    Arguments:
        apa_matrix: 2-D mean observed-over-expected matrix.
        flank, binsize: tick-label hints; if both supplied, axis ticks
            show distance from feature centre in kb.
        cmap, vmin, vmax: ``coolwarm ±1`` log2(O/E) is the cooltools
            APA convention. ``log`` log2-transforms a linear input.
        figsize, title, ax, colorbar: cosmetic.

    Returns:
        ``(fig, ax, img)``.
    """
    import matplotlib.pyplot as plt

    M = np.asarray(apa_matrix, dtype=np.float64)
    if log and (M >= 0).all():
        with np.errstate(divide="ignore"):
            M = np.log2(M)
        M = np.where(np.isfinite(M), M, np.nan)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    img = ax.imshow(M, origin="lower", cmap=cmap,
                    vmin=vmin, vmax=vmax, interpolation="none")
    n = M.shape[0]
    if flank is not None and binsize is not None:
        # tick at ±flank corners + 0 centre
        ticks = [0, n // 2, n - 1]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        labels = [f"-{flank/1000:.0f} kb", "0", f"+{flank/1000:.0f} kb"]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    ax.set_title(title or "APA")
    if colorbar:
        fig.colorbar(img, ax=ax, shrink=0.8,
                     label="log2(O/E)" if log else "O/E")
    return fig, ax, img


def plot_loops(
    cool_path,
    region: str,
    loops_df,
    *,
    balance: bool = True,
    cmap: str = "YlOrRd",
    log: bool = True,
    figsize: Tuple[float, float] = (6.0, 5.5),
    loop_color: str = "#1f4e79",
    loop_marker: str = "o",
    loop_size: float = 30.0,
    title: Optional[str] = None,
    ax=None,
):
    """Contact-matrix heatmap with called loop dots overlaid.

    Renders :func:`plot_contact_matrix` for ``region`` and adds
    scatter markers at every loop in ``loops_df`` whose anchors fall
    inside the window — the canonical figure 1 / 3 panel from
    Maziak 2026 / Chang 2024 showing dot-finder calls in context.

    Arguments:
        cool_path: ``.cool`` / ``.mcool::resolutions/N``.
        region: UCSC interval (underscores / commas allowed).
        loops_df: BEDPE-shaped DataFrame from
            :func:`epione.bulk.hic.loops` (must have ``chrom1, start1,
            end1, chrom2, start2, end2``).
        balance, cmap, log: forwarded to ``plot_contact_matrix``.
        figsize, ax, title: cosmetic.
        loop_color, loop_marker, loop_size: scatter style for the
            overlay.

    Returns:
        ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    fig, ax_, _ = plot_contact_matrix(
        cool_path, region=region,
        balance=balance, cmap=cmap, log=log,
        figsize=figsize, title=title, ax=ax, colorbar=True,
    )

    chrom = region.split(":", 1)[0] if ":" in region else region
    if ":" in region:
        span = region.split(":", 1)[1]
        lo_str, hi_str = span.replace(",", "").replace("_", "").split("-")
        lo, hi = int(lo_str), int(hi_str)
    else:
        import cooler
        c = cooler.Cooler(str(cool_path))
        lo = 0
        hi = int(c.chromsizes[chrom])

    sub = loops_df[
        (loops_df["chrom1"] == chrom) & (loops_df["chrom2"] == chrom)
        & (loops_df["start1"] >= lo) & (loops_df["end1"] <= hi)
        & (loops_df["start2"] >= lo) & (loops_df["end2"] <= hi)
    ]
    x = (sub["start1"] + sub["end1"]) / 2
    y = (sub["start2"] + sub["end2"]) / 2
    # plot_contact_matrix uses extent=[lo, hi, hi, lo] — so y-axis is
    # flipped. Plot loops at both (x, y) and (y, x) so they show up on
    # both upper / lower triangles, matching the symmetric heatmap.
    ax_.scatter(np.r_[x, y], np.r_[y, x],
                s=loop_size, marker=loop_marker,
                facecolors="none", edgecolors=loop_color, linewidths=1.2)
    return fig, ax_
