"""
Peak-to-Gene link visualization (ArchR-style, no BigWig required).

`epione.pl.plot_peak2gene` renders an IGV-style track plot modelled after
ArchR's `plotBrowserTrack(loops=getPeak2GeneLinks(...))`:

    - **Coverage** rows: per-group filled curves, coloured with the ArchR
      "stallion" palette (20 distinct colours). Computed on-the-fly from the
      peak matrix – no BigWig files needed. If BigWigs are supplied through
      ``bigwig_files={label: path}`` they are read instead.
    - **Peaks** row: thin red tick marks.
    - **Peak2GeneLinks** row: arcs between peak centre and gene TSS, coloured
      by correlation magnitude on a purple→blue gradient. A colour bar is
      shown on the right.
    - **Genes** row: gene bodies coloured by strand (red = minus, blue =
      plus) with the target gene highlighted in green; text labels for every
      gene overlapping the window.

The idea: users in a Python/AnnData workflow shouldn't have to bake group
coverage into BigWig just to inspect a P2G link. A two-line numpy groupby on
the peak matrix gives an equally good visual that matches the ArchR look.
"""
from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence, Tuple, Union, Dict, List
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from anndata import AnnData


# ArchR's default stallion categorical palette (20 colours, cycles thereafter)
ARCHR_STALLION = [
    "#D51F26", "#272E6A", "#208A42", "#89288F", "#F47D2B",
    "#FEE500", "#8A9FD1", "#C06CAB", "#E6C2DC", "#90D5E4",
    "#89C75F", "#F37B7D", "#9983BD", "#D24B27", "#3BBCA8",
    "#6E4B9E", "#0C727C", "#7E1416", "#D8A767", "#3D3D3D",
]

# ArchR's default loop colormap ("solarExtra" palette distilled to a sequential
# colormap that runs from light grey through blue to dark purple).
ARCHR_LOOP_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "archr_loop",
    ["#E6E6E6", "#CFD6E8", "#8FA3D3", "#4765B5", "#2E3781", "#1B154B"],
)

# Genes: same as ArchR's default — red for minus strand, blue for plus
_GENE_STRAND_COL = {"-": "#D51F26", "+": "#272E6A", "*": "#3D3D3D"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_region(region: Union[str, Tuple[str, int, int]]) -> Tuple[str, int, int]:
    if isinstance(region, (tuple, list)):
        chrom, start, end = region
        return str(chrom), int(start), int(end)
    if isinstance(region, str):
        if ":" in region:
            chrom, rest = region.split(":", 1)
            start, end = rest.replace(",", "").split("-")
            return chrom, int(start), int(end)
    raise ValueError(f"Could not parse region {region!r}")


def _pick_region_for_gene(links: pd.DataFrame, gene: str, pad: int = 50_000):
    sub = links[links["gene"] == gene]
    if sub.empty:
        raise KeyError(f"gene {gene!r} not in peak_to_gene table")
    chrom = sub["chrom"].iloc[0]
    start = int(min(sub["peak_start"].min(), sub["gene_start"].min())) - pad
    end = int(max(sub["peak_end"].max(), sub["gene_end"].max())) + pad
    return chrom, max(start, 0), end


def _coverage_from_peaks(
    adata, group_by, chrom, start, end, var_df, n_bins=1000,
):
    """Interpolate per-peak mean accessibility into a smooth 1-D signal per
    group, for ArchR-style filled-curve display.

    Returns dict[group] → 1D array of length ``n_bins`` along ``[start, end)``.
    """
    groups = adata.obs[group_by].astype("category")
    mask = (var_df["chrom"] == chrom) & (var_df["end"] >= start) & (var_df["start"] <= end)
    if not mask.any():
        return None, None
    pk = np.where(mask)[0]
    centers = ((var_df.iloc[pk]["start"] + var_df.iloc[pk]["end"]) // 2).to_numpy()
    widths = (var_df.iloc[pk]["end"] - var_df.iloc[pk]["start"]).to_numpy()
    sub = adata.X[:, pk]
    out = {}
    x_grid = np.linspace(start, end, n_bins)
    bin_w = (end - start) / n_bins
    for g in groups.cat.categories:
        cells = np.where(groups.values == g)[0]
        if len(cells) == 0:
            out[g] = np.zeros(n_bins, dtype=np.float32)
            continue
        if sp.issparse(sub):
            v = np.asarray(sub[cells].mean(axis=0)).ravel()
        else:
            v = sub[cells].mean(axis=0)
        # "Spread" each peak's intensity over its own width by depositing into
        # the right-covering bins.
        signal = np.zeros(n_bins, dtype=np.float32)
        for c, w, val in zip(centers, widths, v):
            if val <= 0:
                continue
            lo = int(max(0, (c - w / 2 - start) // bin_w))
            hi = int(min(n_bins, (c + w / 2 - start) // bin_w + 1))
            if lo >= hi:
                continue
            signal[lo:hi] = np.maximum(signal[lo:hi], float(val))
        out[g] = signal
    return out, x_grid


def _half_ellipse(x0, x1, y_radius, direction="down", n=100):
    # Port of ArchR ArchRBrowser.R::getArchDF.
    #   angles ∈ [π, 2π] → lower half-circle (sin<0). For "up" arcs we flip
    #   y by rotating to [0, π]. rx = half-span, so the curve passes through
    #   both endpoints exactly — shape is a parametric half-ellipse, not a
    #   quadratic Bezier.
    cx = 0.5 * (x0 + x1)
    rx = 0.5 * abs(x1 - x0)
    if direction == "down":
        angles = np.linspace(np.pi, 2 * np.pi, n)
    else:
        angles = np.linspace(0.0, np.pi, n)
    xs = rx * np.cos(angles) + cx
    ys = y_radius * np.sin(angles)
    verts = np.column_stack([xs, ys])
    codes = np.full(n, MplPath.LINETO, dtype=np.uint8)
    codes[0] = MplPath.MOVETO
    return MplPath(verts, codes)


def _read_bigwig(path, chrom, start, end, bins=1000):
    import pyBigWig
    with pyBigWig.open(path) as bw:
        v = bw.values(chrom, int(start), int(end), numpy=True)
    v = np.nan_to_num(v, nan=0.0)
    if len(v) > bins:
        step = len(v) // bins
        v = v[: step * bins].reshape(bins, step).mean(axis=1)
    return v


def _format_bp(x, _pos=None):
    if x >= 1e9:
        return f"{x/1e9:.2f}G"
    if x >= 1e6:
        return f"{x/1e6:.2f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}k"
    return f"{int(x):,}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_peak2gene(
    adata: AnnData,
    gene: Optional[str] = None,
    region: Optional[Union[str, Tuple[str, int, int]]] = None,
    *,
    links_key: str = "peak_to_gene",
    fdr_thresh: float = 0.05,
    min_abs_r: float = 0.0,
    group_by: Optional[str] = None,
    group_order: Optional[Sequence[str]] = None,
    palette: Optional[Sequence[str]] = None,
    bigwig_files: Optional[Dict[str, str]] = None,
    gene_annotation: Optional[pd.DataFrame] = None,
    pad_bp: int = 50_000,
    figsize: Tuple[float, float] = (8, 9),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    arc_cmap: Optional[mpl.colors.Colormap] = None,
    coverage_ylim: Optional[Tuple[float, float]] = None,
    arc_direction: Literal["up", "down"] = "down",
    show: bool = True,
):
    """ArchR-style arc plot of peak-to-gene links on a genomic window.

    Parameters
    ----------
    adata
        AnnData with ``adata.uns[links_key]`` (produced by
        :func:`epione.tl.peak_to_gene`) and peak coords in ``adata.var_names``.
    gene, region
        Either a gene name (auto-picks the window spanning all its links) or an
        explicit ``(chrom, start, end)`` / ``'chrX:1000-20000'`` region. Both
        can be passed; in that case ``region`` wins but ``gene`` is still
        highlighted.
    links_key
        Key under ``adata.uns`` where the P2G table lives.
    fdr_thresh, min_abs_r
        Only links with ``fdr ≤ fdr_thresh`` and ``|correlation| ≥ min_abs_r``
        are drawn.
    group_by
        ``adata.obs`` column name. If given, one filled-curve coverage row is
        drawn per category using the ArchR stallion palette.
    group_order
        Explicit category ordering; if ``None`` the existing ``.cat.categories``
        order is used.
    palette
        Override the ArchR stallion palette for the coverage rows.
    bigwig_files
        Optional dict ``{label: path}``. Takes precedence over ``group_by``.
    gene_annotation
        DataFrame with ``gene_name, chrom, start, end, strand``. If omitted,
        only genes present in the links table are drawn (no strand colour).
    arc_cmap
        Override the default ArchR loop colormap (solarExtra-like gradient).

    Returns
    -------
    ``(fig, axes)`` tuple.
    """
    if links_key not in adata.uns:
        raise KeyError(
            f"adata.uns[{links_key!r}] not found. Run epione.tl.peak_to_gene first."
        )
    links: pd.DataFrame = adata.uns[links_key]
    if not isinstance(links, pd.DataFrame):
        links = pd.DataFrame(links)

    palette = list(palette) if palette is not None else ARCHR_STALLION
    cmap = arc_cmap if arc_cmap is not None else ARCHR_LOOP_CMAP

    # ---- Region resolution -------------------------------------------------
    if region is not None:
        chrom, start, end = _parse_region(region)
        region_gene = gene
    elif gene is not None:
        chrom, start, end = _pick_region_for_gene(links, gene, pad=pad_bp)
        region_gene = gene
    else:
        raise ValueError("Provide either `gene=` or `region=`")

    # ---- Parse peak coords from adata.var_names ----------------------------
    var_df = pd.DataFrame(index=adata.var_names)
    parts = pd.Series(list(adata.var_names)).str.split(":", n=1, expand=True)
    coords = parts[1].str.split("-", expand=True)
    var_df["chrom"] = parts[0].values
    var_df["start"] = coords[0].astype(int).values
    var_df["end"] = coords[1].astype(int).values

    # ---- Filter links --------------------------------------------------------
    sub = links[
        (links["chrom"] == chrom)
        & (links["peak_end"] >= start) & (links["peak_start"] <= end)
        & (links["fdr"] <= fdr_thresh) & (links["correlation"].abs() >= min_abs_r)
    ].copy()
    sub["peak_center"] = (sub["peak_start"] + sub["peak_end"]) // 2
    sub = sub[(sub["peak_center"] >= start) & (sub["peak_center"] <= end)
              & (sub["tss"] >= start) & (sub["tss"] <= end)]

    # ---- Layout ------------------------------------------------------------
    if ax is None:
        n_extra = 0
        if group_by is not None:
            raw = list(adata.obs[group_by].astype("category").cat.categories)
            if group_order is not None:
                cats = list(group_order)
            else:
                # "Natural" sort so C1..C11 is in numeric order, not C1,C10..
                import re
                def _nk(s):
                    return [int(p) if p.isdigit() else p.lower()
                            for p in re.split(r"(\d+)", str(s))]
                cats = sorted(raw, key=_nk)
            n_extra = len(cats)
        elif bigwig_files is not None:
            cats = list(bigwig_files.keys())
            n_extra = len(cats)
        else:
            cats = []
        n_rows = n_extra + 3   # coverage + peaks + arcs + genes
        if n_extra > 0 and figsize[1] < 4.5 + 0.28 * n_extra:
            figsize = (figsize[0], 4.5 + 0.28 * n_extra)
        height_ratios = [0.45] * n_extra + [0.22, 1.6, 0.9]
        fig, axes = plt.subplots(
            n_rows, 1, figsize=figsize, sharex=True,
            gridspec_kw=dict(height_ratios=height_ratios, hspace=0.0),
        )
    else:
        fig = ax.figure
        axes = [ax]
        cats = []
        n_extra = 0

    cov_axes = axes[:n_extra]
    peak_ax = axes[n_extra]
    arc_ax = axes[n_extra + 1]
    gene_ax = axes[n_extra + 2]

    # ---- Coverage rows -----------------------------------------------------
    if group_by is not None and n_extra:
        signal, x_grid = _coverage_from_peaks(adata, group_by, chrom, start, end, var_df)
        # Global y-limit (Archr-style "shared scale" — take max across groups)
        y_max = 0.0
        for v in signal.values():
            if len(v):
                y_max = max(y_max, float(v.max()))
        y_lim = coverage_ylim or (0.0, y_max * 1.05 if y_max > 0 else 1.0)
        for i, (cax, g) in enumerate(zip(cov_axes, cats)):
            col = palette[i % len(palette)]
            sig = signal.get(g, np.zeros_like(x_grid))
            cax.fill_between(x_grid, 0, sig, color=col, linewidth=0)
            cax.plot(x_grid, sig, color=col, linewidth=0.6)
            cax.set_ylim(*y_lim)
            cax.set_yticks([])
            cax.spines[["top", "right", "left"]].set_visible(False)
            # Right-side cluster label (ArchR style)
            cax.text(1.01, 0.5, g, transform=cax.transAxes, fontsize=8,
                     va="center", ha="left", color=col, fontweight="bold")
        # Left-side axis label on the topmost coverage row, rotated
        cov_axes[0].set_ylabel("Coverage\n(Norm. ATAC Signal)", fontsize=8, rotation=90)
    elif bigwig_files is not None and n_extra:
        y_max = 0.0
        sigs = {}
        for g, path in bigwig_files.items():
            s = _read_bigwig(path, chrom, start, end, bins=1000)
            sigs[g] = s
            y_max = max(y_max, float(s.max()))
        y_lim = coverage_ylim or (0.0, y_max * 1.05 if y_max > 0 else 1.0)
        x_grid = np.linspace(start, end, 1000)
        for i, (cax, g) in enumerate(zip(cov_axes, cats)):
            col = palette[i % len(palette)]
            cax.fill_between(x_grid, 0, sigs[g], color=col, linewidth=0)
            cax.set_ylim(*y_lim)
            cax.set_yticks([])
            cax.spines[["top", "right", "left"]].set_visible(False)
            cax.text(1.01, 0.5, g, transform=cax.transAxes, fontsize=8,
                     va="center", ha="left", color=col, fontweight="bold")

    # ---- Peaks row ---------------------------------------------------------
    peaks_in_window = var_df[
        (var_df["chrom"] == chrom)
        & (var_df["end"] >= start) & (var_df["start"] <= end)
    ]
    for _, row in peaks_in_window.iterrows():
        peak_ax.vlines(
            (row["start"] + row["end"]) / 2, 0.1, 0.9,
            colors="#D51F26", linewidths=0.7,
        )
    peak_ax.set_ylim(0, 1)
    peak_ax.set_yticks([])
    peak_ax.spines[["top", "right", "left"]].set_visible(False)
    peak_ax.text(1.01, 0.5, "Peaks", transform=peak_ax.transAxes, fontsize=9,
                 va="center", ha="left", fontweight="bold")

    # ---- Arc row -----------------------------------------------------------
    # Exact port of ArchR ArchRBrowser.R::getArchDF: each link is a half
    # ellipse with x-radius = |peak-tss|/2 and y-radius scaled linearly so
    # the longest link reaches R_MAX and shorter ones are proportionally
    # shallower — so a long-range dome visually contains the short-range
    # ones. Sorted largest→smallest so short arcs render on top.
    rx_series = (sub["peak_center"] - sub["tss"]).abs().astype(float) / 2.0
    max_rx = float(rx_series.max()) if len(rx_series) else 1.0
    sub_sorted = sub.assign(_rx=rx_series).sort_values(
        ["_rx", "correlation"],
        key=lambda s: s.abs() if s.name == "correlation" else s,
        ascending=[False, True],
    ).drop(columns="_rx")

    R_MAX = 100.0  # arbitrary plotting units (matches ArchR's r=100)
    for _, row in sub_sorted.iterrows():
        r_abs = abs(float(row["correlation"]))
        col = cmap(np.clip(r_abs, 0.0, 1.0))
        lw = 0.35 + 1.3 * r_abs
        rx = 0.5 * abs(float(row["peak_center"] - row["tss"]))
        ry = R_MAX * (rx / max_rx) if max_rx > 0 else 0.0
        path = _half_ellipse(row["peak_center"], row["tss"], ry,
                             direction=arc_direction)
        arc_ax.add_patch(PathPatch(path, facecolor="none",
                                   edgecolor=col, linewidth=lw))
    if arc_direction == "down":
        arc_ax.set_ylim(-R_MAX * 1.05, R_MAX * 0.02)
        arc_ax.axhline(0, color="black", linewidth=0.5, zorder=0)
    else:
        arc_ax.set_ylim(-R_MAX * 0.02, R_MAX * 1.05)
    arc_ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    arc_ax.set_xlim(start, end)
    arc_ax.set_yticks([])
    arc_ax.text(1.01, 0.5, "Peak2GeneLinks",
                transform=arc_ax.transAxes, fontsize=9,
                va="center", ha="left", fontweight="bold")

    # Colour bar (small, inside the arc track's right margin)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax_bar = inset_axes(arc_ax, width="1.5%", height="60%",
                         loc="center right",
                         bbox_to_anchor=(1.18, 0, 1, 1),
                         bbox_transform=arc_ax.transAxes, borderpad=0)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 1))
    cb = plt.colorbar(sm, cax=cax_bar)
    cb.set_label("|r|", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # ---- Gene row ----------------------------------------------------------
    gene_ax.set_ylim(0, 1.4)
    gene_ax.set_yticks([])
    gene_ax.spines[["top", "right", "left"]].set_visible(False)
    gene_ax.text(1.01, 0.5, "Genes", transform=gene_ax.transAxes, fontsize=9,
                 va="center", ha="left", fontweight="bold")

    # Prefer supplied gene_annotation (has strand); fall back to the links table
    if gene_annotation is not None:
        ga = gene_annotation.copy()
        ga.columns = [c.lower() for c in ga.columns]
        win = ga[(ga["chrom"].astype(str) == chrom)
                 & (ga["end"] >= start) & (ga["start"] <= end)]
    else:
        gmap = links[links["chrom"] == chrom][
            ["gene", "gene_start", "gene_end"]
        ].drop_duplicates("gene")
        gmap = gmap[(gmap["gene_end"] >= start) & (gmap["gene_start"] <= end)]
        win = gmap.rename(columns={"gene": "gene_name",
                                    "gene_start": "start",
                                    "gene_end": "end"})
        win["strand"] = "+"
    if len(win) > 0:
        # Split genes into two rows to avoid label collisions
        win = win.sort_values("start").reset_index(drop=True)
        rows = []
        last_ends = [-np.inf, -np.inf]
        min_gap = (end - start) * 0.05
        for i, r in win.iterrows():
            chosen = None
            for ridx in range(len(last_ends)):
                if r["start"] > last_ends[ridx] + min_gap:
                    chosen = ridx
                    break
            if chosen is None:
                chosen = np.argmin(last_ends)
            last_ends[chosen] = float(r["end"])
            rows.append(chosen)
        win["_row"] = rows
        # y-coords for 2 rows
        row_y = {0: 0.55, 1: 0.15}
        for _, r in win.iterrows():
            hl = (r["gene_name"] == region_gene)
            col = _GENE_STRAND_COL.get(str(r.get("strand", "+")), "#3D3D3D")
            if hl:
                col = "#208A42"  # ArchR stallion green for the highlighted gene
            y = row_y[int(r["_row"])]
            gene_ax.add_patch(Rectangle(
                (r["start"], y), r["end"] - r["start"], 0.22,
                facecolor=col, edgecolor="black" if hl else "none",
                linewidth=0.8 if hl else 0, alpha=0.9,
            ))
            # Strand arrow: small wedge at centre
            cx = (r["start"] + r["end"]) / 2
            gene_ax.plot([cx], [y + 0.11], marker=(3, 0, 0 if str(r.get("strand","+")) == "+" else 180),
                         markersize=4, color="white", markeredgecolor=col, markeredgewidth=0.6)
            # Label — ArchR style: small text below gene body
            gene_ax.text(cx, y - 0.02, r["gene_name"],
                         ha="center", va="top",
                         fontsize=6.5, color=col,
                         fontweight="bold" if hl else "normal")

    # X-axis: only on the bottom, formatted genomic style
    gene_ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_format_bp))
    gene_ax.set_xlabel(f"{chrom}  ({end - start:,} bp)", fontsize=9)

    # Title – placed at the figure level (suptitle) so it doesn't clobber the
    # topmost coverage row's ylabel.
    if title is None:
        title = (f"{region_gene}" if region_gene else chrom)
        title += (f"  |  {chrom}:{start:,}-{end:,}"
                  f"  |  {len(sub):,} links (FDR ≤ {fdr_thresh}, |r| ≥ {min_abs_r})")
    fig.suptitle(title, fontsize=10, y=0.995)

    # Shared x range
    for a in axes:
        a.set_xlim(start, end)

    if show:
        plt.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, axes
