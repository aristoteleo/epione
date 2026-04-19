"""
Peak-to-Gene link visualization (no BigWig required).

`epione.pl.plot_peak2gene` renders an arc-style track plot:

    - arcs: each significant peak-gene pair, coloured by sign(r), alpha ∝ |r|,
            width ∝ -log10(FDR)
    - peak track: peak boxes; peaks with any significant link are highlighted
    - gene track: gene bodies + TSS arrow; target gene highlighted
    - optional cluster coverage: if ``adata`` carries a categorical ``group_by``
      column, we compute pseudobulk coverage per group *directly from the peak
      matrix* – no BigWig files or on-disk coverage needed. If the user happens
      to have BigWigs already, pass them via ``bigwig_files`` and we'll overlay
      those signals instead.

The idea: users in a Python/AnnData workflow shouldn't have to bake group
coverage into BigWig just to inspect a P2G link. A two-line numpy groupby on
the peak matrix gives an equally good visual.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union, Dict, List
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from anndata import AnnData


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


def _pseudobulk_by_group(adata, group_by, peak_idx, chrom, start, end, var_df):
    """Per-group mean accessibility across peaks that overlap a window."""
    groups = adata.obs[group_by].astype("category")
    # Peaks overlapping the window
    mask = (var_df["chrom"] == chrom) & (var_df["end"] >= start) & (var_df["start"] <= end)
    if not mask.any():
        return None, None, None
    pk = np.where(mask)[0]
    sub = adata.X[:, pk]
    # Per-group mean
    out = {}
    for g in groups.cat.categories:
        cells = np.where(groups.values == g)[0]
        if len(cells) == 0:
            continue
        if sp.issparse(sub):
            v = np.asarray(sub[cells].mean(axis=0)).ravel()
        else:
            v = sub[cells].mean(axis=0)
        out[g] = v
    centers = ((var_df.iloc[pk]["start"] + var_df.iloc[pk]["end"]) // 2).to_numpy()
    return out, centers, pk


def _bezier_arc(x0, x1, height, n=40):
    """Quadratic Bezier arc from (x0, 0) up to (midpoint, height) down to
    (x1, 0). Returns the matplotlib Path verts."""
    xm = 0.5 * (x0 + x1)
    verts = [(x0, 0), (xm, height), (x1, 0)]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    return MplPath(verts, codes)


def _read_bigwig(path, chrom, start, end, bins=400):
    import pyBigWig
    with pyBigWig.open(path) as bw:
        v = bw.values(chrom, int(start), int(end), numpy=True)
    v = np.nan_to_num(v, nan=0.0)
    # down-sample to `bins`
    if len(v) > bins:
        step = len(v) // bins
        v = v[: step * bins].reshape(bins, step).mean(axis=1)
    return v


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
    bigwig_files: Optional[Dict[str, str]] = None,
    gene_annotation: Optional[pd.DataFrame] = None,
    pad_bp: int = 50_000,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    arc_height: float = 1.0,
    show: bool = True,
):
    """Arc plot of peak-to-gene links on a genomic window.

    Parameters
    ----------
    adata
        AnnData with ``adata.uns[links_key]`` (produced by
        :func:`epione.tl.peak_to_gene`) and peak coords in ``adata.var_names``.
    gene, region
        Either a gene name (auto-picks the window spanning all its links) or an
        explicit ``(chrom, start, end)`` / ``'chrX:1000-20000'`` region.
    links_key
        Key under ``adata.uns`` where the P2G table lives.
    fdr_thresh, min_abs_r
        Only arcs with ``fdr <= fdr_thresh`` and ``|correlation| >= min_abs_r``
        are drawn.
    group_by
        ``adata.obs`` column name. If given, the figure includes a pseudobulk
        coverage row per category, computed **on the fly** from
        ``adata.X`` – no BigWig files needed.
    bigwig_files
        Optional dict ``{label: path}``. If given (and ``group_by`` is not),
        each BigWig is drawn as a coverage row instead.
    gene_annotation
        Optional DataFrame with columns ``gene_name, chrom, start, end,
        strand`` used to draw extra genes overlapping the window. If omitted,
        only the target gene (from the links table) is drawn.
    pad_bp
        Extra padding around the inferred gene window when ``gene=`` is used.
    arc_height
        Visual scale for the arcs (1.0 = flush with the peak row top).
    """
    if links_key not in adata.uns:
        raise KeyError(
            f"adata.uns[{links_key!r}] not found. Run epione.tl.peak_to_gene first."
        )
    links: pd.DataFrame = adata.uns[links_key]
    if not isinstance(links, pd.DataFrame):
        links = pd.DataFrame(links)

    # ---- Region resolution -------------------------------------------------
    if region is not None:
        chrom, start, end = _parse_region(region)
        # Keep `gene` as highlight target even when `region` is explicit.
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

    # ---- Filter the links table to this window -----------------------------
    sub = links[
        (links["chrom"] == chrom)
        & (links["peak_end"] >= start) & (links["peak_start"] <= end)
        & (links["fdr"] <= fdr_thresh) & (links["correlation"].abs() >= min_abs_r)
    ].copy()
    if region_gene is not None:
        # Always include all links of the highlighted gene
        pass
    sub["peak_center"] = (sub["peak_start"] + sub["peak_end"]) // 2
    sub = sub[(sub["peak_center"] >= start) & (sub["peak_center"] <= end)
              & (sub["tss"] >= start) & (sub["tss"] <= end)]

    # ---- Layout ------------------------------------------------------------
    if ax is None:
        n_extra = 0
        if group_by is not None:
            n_extra = int(adata.obs[group_by].astype("category").cat.categories.size)
        elif bigwig_files is not None:
            n_extra = len(bigwig_files)
        n_rows = n_extra + 3     # arcs, peaks, genes
        # Auto-scale height: 0.3"/group for coverage rows, plus ~3.5" for the
        # arc/peak/gene trio. Overrides the user's figsize[1] when n_extra>0
        # so labels don't collide.
        if n_extra > 0 and figsize[1] < 3.5 + 0.3 * n_extra:
            figsize = (figsize[0], 3.5 + 0.3 * n_extra)
        # Coverage rows get short bars; the arc + peak + gene rows take up
        # enough space for labels and curves to breathe.
        height_ratios = [0.8] * n_extra + [2.5, 0.55, 0.9]
        fig, axes = plt.subplots(
            n_rows, 1, figsize=figsize, sharex=True,
            gridspec_kw=dict(height_ratios=height_ratios, hspace=0.18),
        )
        if n_rows == 1:
            axes = [axes]
    else:
        fig = ax.figure
        axes = [ax]
        n_extra = 0

    cov_axes = axes[:n_extra] if n_extra else []
    arc_ax, peak_ax, gene_ax = axes[-3], axes[-2], axes[-1]

    # ---- Pseudobulk coverage (on-the-fly from adata.X) --------------------
    if group_by is not None and n_extra:
        pb, centers, pk_idx = _pseudobulk_by_group(
            adata, group_by, None, chrom, start, end, var_df
        )
        if pb is None:
            for cax in cov_axes:
                cax.set_axis_off()
        else:
            groups = list(pb.keys())
            # Align cov_axes order to groups order
            for cax, g in zip(cov_axes, groups):
                sig = pb[g]
                cax.bar(centers, sig, width=(end - start) / 300,
                        color="#444", edgecolor="none")
                cax.set_ylabel(g, rotation=0, ha="right", va="center", fontsize=8)
                cax.set_yticks([])
                cax.spines[["top", "right"]].set_visible(False)
    elif bigwig_files is not None and n_extra:
        for cax, (lbl, path) in zip(cov_axes, bigwig_files.items()):
            sig = _read_bigwig(path, chrom, start, end, bins=400)
            x = np.linspace(start, end, len(sig))
            cax.fill_between(x, 0, sig, color="#444", linewidth=0)
            cax.set_ylabel(lbl, rotation=0, ha="right", va="center", fontsize=8)
            cax.set_yticks([])
            cax.spines[["top", "right"]].set_visible(False)

    # ---- Arc track --------------------------------------------------------
    max_fdr = -np.log10(np.clip(sub["fdr"].to_numpy(), 1e-300, 1))
    mfdr = max_fdr.max() if len(sub) else 1.0
    for _, row in sub.iterrows():
        r = row["correlation"]
        col = "#d62728" if r > 0 else "#1f77b4"   # red for positive, blue for negative
        alpha = float(np.clip(abs(r), 0.1, 1.0))
        lw = 0.5 + 3.0 * (-np.log10(max(row["fdr"], 1e-300)) / mfdr)
        path = _bezier_arc(row["peak_center"], row["tss"], arc_height)
        arc_ax.add_patch(PathPatch(path, facecolor="none", edgecolor=col,
                                   alpha=alpha, linewidth=lw))
    arc_ax.set_ylim(0, arc_height * 1.05)
    arc_ax.set_yticks([])
    arc_ax.set_ylabel("links", fontsize=9)
    arc_ax.spines[["top", "right", "left"]].set_visible(False)

    # ---- Peak track -------------------------------------------------------
    peaks_in_window = var_df[
        (var_df["chrom"] == chrom)
        & (var_df["end"] >= start) & (var_df["start"] <= end)
    ]
    sig_peak_names = set(sub["peak"].unique())
    for name, row in peaks_in_window.iterrows():
        is_sig = name in sig_peak_names
        peak_ax.add_patch(Rectangle(
            (row["start"], 0.2), row["end"] - row["start"], 0.6,
            facecolor="#d62728" if is_sig else "#bbbbbb",
            edgecolor="none", alpha=0.9 if is_sig else 0.5,
        ))
    peak_ax.set_ylim(0, 1)
    peak_ax.set_yticks([])
    peak_ax.set_ylabel("peaks", fontsize=9)
    peak_ax.spines[["top", "right", "left"]].set_visible(False)

    # ---- Gene track -------------------------------------------------------
    genes_in_window = pd.DataFrame()
    if gene_annotation is not None:
        ga = gene_annotation.copy()
        ga.columns = [c.lower() for c in ga.columns]
        genes_in_window = ga[
            (ga["chrom"].astype(str) == chrom)
            & (ga["end"] >= start) & (ga["start"] <= end)
        ]
    # Also include whatever's in the links table for completeness
    gmap = links[links["chrom"] == chrom][["gene", "gene_start", "gene_end", "tss"]].drop_duplicates("gene")
    gmap = gmap[(gmap["gene_end"] >= start) & (gmap["gene_start"] <= end)]
    for _, row in gmap.iterrows():
        g = row["gene"]
        hl = (g == region_gene)
        gene_ax.add_patch(Rectangle(
            (row["gene_start"], 0.4), row["gene_end"] - row["gene_start"], 0.2,
            facecolor="#2ca02c" if hl else "#888888",
            edgecolor="black" if hl else "none",
            linewidth=0.8 if hl else 0, alpha=0.9,
        ))
        # TSS marker
        gene_ax.plot([row["tss"]], [0.8], marker="v",
                     color="#2ca02c" if hl else "#555555",
                     markersize=6 if hl else 4)
        # Label (target gene only, to reduce clutter)
        if hl:
            gene_ax.text(row["tss"], 1.0, g, ha="center", va="bottom",
                         fontsize=9, fontweight="bold", color="#2ca02c")
    gene_ax.set_ylim(0, 1.3)
    gene_ax.set_yticks([])
    gene_ax.set_ylabel("genes", fontsize=9)
    gene_ax.spines[["top", "right", "left"]].set_visible(False)

    # Tick + title
    for a in axes:
        a.set_xlim(start, end)
    gene_ax.set_xlabel(f"{chrom}:{start:,}-{end:,}  ({(end - start):,} bp)")
    if title is None:
        title = f"peak-to-gene: {region_gene or chrom}" \
                f"  |  {len(sub):,} significant links  (FDR ≤ {fdr_thresh})"
    axes[0].set_title(title, fontsize=10)

    # Legend for arcs
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#d62728", lw=2, label="r > 0 (activating)"),
        Line2D([0], [0], color="#1f77b4", lw=2, label="r < 0 (repressing)"),
    ]
    arc_ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False)

    if show:
        plt.tight_layout()
    return fig, axes
