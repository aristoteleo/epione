"""Figures for ACESO, drawn the way the paper draws them.

The paper's panels are made of five shapes: a training curve, stacked accessibility tracks over a locus,
a benchmark comparison, a panel-size sweep, and an embedding coloured by cell type. Those are here, with
the same typography (12 pt text, 13 pt titles), the same palette, and the same rule that a panel carries
no explanatory prose -- the caption does. :func:`save_figure` writes an SVG whose text is text and whose
font is named plainly, which is what Illustrator needs to open it.

    from epione.models.aceso import paper_style, plot_tracks, save_figure
    paper_style()
    fig, axes = plot_tracks({"Cardiomyocyte, measured": obs, "Cardiomyocyte, predicted": pred},
                            chrom="chr11", start=47_000_000, regions=peaks, genes=gtf)
    save_figure(fig, "fig_myh6")

Fonts
-----
The SVG names one font plainly (default ``"Arial"``), because matplotlib's default writes a CSS fallback
list that Illustrator refuses. If Arial itself is not installed, pass ``font_dir=`` to
:func:`paper_style` pointing at metric-compatible TrueType files (Liberation Sans is the usual choice);
the figure is then laid out with those metrics and still named Arial in the SVG.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

FS, FS_TITLE = 12, 13

#: Colours that mean the same thing in every ACESO figure.
PALETTE = {
    "aceso": "#4a148c",          # the model
    "scooby": "#e0a030",         # the sequence-decoder baseline with its own cell embedding
    "reference": "#616161",      # references that read measured ATAC (type mean, nearest neighbours)
    "measured": "#424242",       # a measured track
    "predicted": "#2E7D32",      # a predicted track
    "opening": "#E07B39",        # elements that open along a state axis
    "closing": "#3B6FB6",        # elements that close along it
    "other": "#E3E3E3",
}

#: 22 distinct cell-type colours, saturated first so the largest populations are the legible ones.
CELLTYPE_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
                   "#bcbd22", "#393b79", "#7f7f7f", "#8c6d31", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
                   "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#c7c7c7"]


def celltype_colors(labels: Sequence[str], other: str = PALETTE["other"]) -> Dict[str, str]:
    """Map cell-type names to colours, in the order given; names beyond the palette get the grey."""
    out = {}
    for i, g in enumerate(dict.fromkeys(map(str, labels))):
        out[g] = CELLTYPE_COLORS[i] if i < len(CELLTYPE_COLORS) else other
    return out


def paper_style(fontsize: int = FS, title_fontsize: int = FS_TITLE, font_dir: Optional[str] = None) -> None:
    """Set the paper's typography and axes defaults on the current matplotlib session."""
    import matplotlib
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    families = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
    if font_dir:
        for f in glob.glob(str(Path(font_dir) / "*.ttf")):
            font_manager.fontManager.addfont(f)
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": families,
                                "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    plt.rcParams.update({"font.size": fontsize, "axes.titlesize": title_fontsize, "axes.labelsize": fontsize,
                         "xtick.labelsize": fontsize, "ytick.labelsize": fontsize, "legend.fontsize": fontsize,
                         "legend.title_fontsize": fontsize, "axes.linewidth": 0.9, "xtick.major.width": 0.9,
                         "ytick.major.width": 0.9, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
                         "axes.spines.top": False, "axes.spines.right": False})


def _name_one_font(path: str, font: str) -> None:
    s = open(path).read()
    s = re.sub(r"font-family:[^;\"]*", f"font-family: {font}", s)
    s = re.sub(r'font-family="[^"]*"', f'font-family="{font}"', s)
    open(path, "w").write(s)


def _check_svg(path: str, limit: float = 1e4) -> None:
    """Refuse to leave behind an SVG with infinite or absurd coordinates: Illustrator cannot open one."""
    s = open(path).read()
    geo = " ".join(re.findall(r' d="([^"]+)"', s)
                   + re.findall(r' (?:x|y|x1|x2|y1|y2|width|height)="([^"]+)"', s))
    bad = re.findall(r"\b(?:inf|nan)\b", geo, re.I)
    big = [v for v in (float(x) for x in re.findall(r"-?\d+\.?\d*(?:e[+-]?\d+)?", geo)) if abs(v) > limit]
    if bad or big:
        raise ValueError(f"{path}: {len(bad)} inf/nan and {len(big)} coordinates beyond +-{limit:g} pt "
                         f"(e.g. {big[:3]}); on a log axis start bars at the axis floor, not 0")


def save_figure(fig, stem, *, formats: Sequence[str] = ("svg", "pdf", "png"), dpi: int = 300,
                tight: bool = False, font: str = "Arial") -> None:
    """Write ``stem.svg`` (text as text, one font name), ``stem.pdf`` (TrueType) and ``stem.png``.

    ``tight=False`` keeps the exact figure size, which is what panel assembly needs; set the margins with
    ``fig.subplots_adjust`` instead of relying on a tight bounding box.
    """
    for ext in formats:
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight" if tight else None)
        if ext == "svg":
            _name_one_font(f"{stem}.svg", font)
            _check_svg(f"{stem}.svg")


def read_training_log(run) -> "object":
    """Read a run's ``train.log`` (one JSON object per line) into a DataFrame, missing fields as NaN."""
    import pandas as pd

    p = Path(run)
    p = p / "train.log" if p.is_dir() else p
    rows = []
    for line in open(p):
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        raise ValueError(f"{p}: no JSON records; is this an ACESO training log?")
    return pd.DataFrame(rows)


def plot_training_curve(run, *, keys: Sequence[str] = ("loss", "val_pearson_log"), ax=None,
                        figsize=(5.2, 3.2), smooth: int = 0, show: bool = True):
    """Training loss and validation metric against step, from a run directory or a ``train.log``."""
    import matplotlib.pyplot as plt

    df = read_training_log(run)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    axes = [ax]
    for i, k in enumerate([k for k in keys if k in df.columns]):
        d = df.dropna(subset=[k])
        y = d[k].to_numpy(float)
        if smooth > 1 and len(y) > smooth:
            y = np.convolve(y, np.ones(smooth) / smooth, mode="same")
        a = ax if i == 0 else ax.twinx()
        if i:
            axes.append(a)
            a.spines["right"].set_visible(True)
        a.plot(d["step"], y, lw=1.4, color=PALETTE["aceso"] if i == 0 else PALETTE["predicted"])
        a.set_ylabel(k, fontsize=FS, color=PALETTE["aceso"] if i == 0 else PALETTE["predicted"])
        a.tick_params(labelsize=FS)
    ax.set_xlabel("step", fontsize=FS)
    if show:
        plt.show()
    return fig, (axes[0] if len(axes) == 1 else axes)


def plot_tracks(tracks: Dict[str, Sequence[float]], *, chrom: str = "", start: int = 0, bin_size: int = 32,
                colors: Optional[Dict[str, str]] = None, regions=None, genes=None, figsize=(6.4, None),
                track_height: float = 0.42, ylabel: Optional[str] = None, share_y: bool = True,
                show: bool = True):
    """Stacked accessibility profiles over one window, with the regions and the gene model underneath.

    ``tracks`` maps a row label to a 1-D profile over consecutive ``bin_size`` bp bins starting at
    ``start``. ``regions`` is a DataFrame of called elements with ``start`` / ``end`` (and optionally
    ``name``), drawn as a bar under the tracks and labelled with their coordinates. ``genes`` is a
    DataFrame with ``name`` / ``start`` / ``end`` / ``strand`` (and optionally ``exon_starts`` /
    ``exon_ends``), drawn as a gene model with the direction of transcription marked at the promoter.
    ``ylabel`` is written down the right-hand side, where it cannot collide with the row names.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    labels = list(tracks)
    # One inch grid for the whole panel, so a row cannot land on the one below it: tracks, then the
    # element bar with its coordinates, then the gene model, then the scale bar.
    REGION_IN, GENE_IN, SCALE_IN, PAD_IN = 0.46, 0.60, 0.40, 0.14
    rows_in = (track_height * len(labels) + (REGION_IN if regions is not None and len(regions) else 0)
               + (GENE_IN if genes is not None and len(genes) else 0) + SCALE_IN)
    h = figsize[1] or (rows_in + 2 * PAD_IN)
    fig = plt.figure(figsize=(figsize[0], h))
    colors = colors or {}
    n_bins = max(len(np.asarray(v)) for v in tracks.values())
    x0, x1 = start, start + n_bins * bin_size
    label_in = 0.088 * max(len(lb) for lb in labels) + 0.12                   # room for the row names
    left = min(0.52, label_in / figsize[0])
    top, width = 1 - PAD_IN / h, min(0.96 - left, 0.98 - left)
    ymax = max(float(np.max(np.asarray(v))) for v in tracks.values()) if share_y else None
    axes = []
    for i, lb in enumerate(labels):
        y = np.asarray(tracks[lb], float)
        ax = fig.add_axes([left, top - (i + 1) * track_height / h, width, track_height / h * 0.86])
        c = colors.get(lb, PALETTE["measured"] if "measur" in lb.lower() else PALETTE["predicted"])
        ax.fill_between(np.arange(len(y)) * bin_size + start, y, step="mid", color=c, lw=0)
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, (ymax if share_y else float(y.max())) * 1.05 or 1)
        ax.set_yticks([])
        ax.set_xticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
        ax.text(-0.01, 0.5, lb, transform=ax.transAxes, ha="right", va="center", fontsize=FS)
        axes.append(ax)
    y_cursor = top - len(labels) * track_height / h

    if regions is not None and len(regions):
        y_cursor -= REGION_IN / h
        ax = fig.add_axes([left, y_cursor, width, REGION_IN / h])
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        for _, r in regions.iterrows():
            ax.add_patch(Rectangle((r["start"], 0.66), max(r["end"] - r["start"], (x1 - x0) / 400), 0.30,
                                   color=PALETTE["reference"], lw=0))
            ax.text((r["start"] + r["end"]) / 2, 0.56, f"{chrom}:{int(r['start']):,}-{int(r['end']):,}",
                    ha="center", va="top", fontsize=FS, clip_on=False)
        axes.append(ax)

    if genes is not None and len(genes):
        y_cursor -= GENE_IN / h
        ax = fig.add_axes([left, y_cursor, width, GENE_IN / h])
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        for _, g in genes.iterrows():
            s, e = float(g["start"]), float(g["end"])
            ax.plot([s, e], [0.62, 0.62], color="#333333", lw=1.0, solid_capstyle="butt")
            es = g.get("exon_starts"), g.get("exon_ends")
            if es[0] is not None and es[1] is not None and len(np.atleast_1d(es[0])):
                for a, b in zip(np.atleast_1d(es[0]), np.atleast_1d(es[1])):
                    ax.add_patch(Rectangle((a, 0.50), max(b - a, bin_size), 0.24, color="#333333", lw=0))
            tss, d = (s, 1) if str(g.get("strand", "+")) == "+" else (e, -1)
            ax.plot([tss, tss], [0.62, 0.98], color="#333333", lw=1.0)
            ax.annotate("", xy=(tss + d * (x1 - x0) * 0.03, 0.98), xytext=(tss, 0.98),
                        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.0, shrinkA=0, shrinkB=0))
            ax.text(tss, 0.42, str(g["name"]), ha="center", va="top", fontsize=FS, style="italic",
                    clip_on=False)
        axes.append(ax)

    y_cursor -= SCALE_IN / h
    scale = fig.add_axes([left, max(y_cursor, 0.01), width, SCALE_IN / h])
    scale.set_xlim(x0, x1)
    scale.set_ylim(0, 1)
    scale.axis("off")
    kb = 10 ** int(np.floor(np.log10((x1 - x0) / 5)))
    scale.plot([x0, x0 + kb], [0.80, 0.80], color="k", lw=1.2, solid_capstyle="butt")
    scale.text(x0 + kb / 2, 0.62, f"{kb / 1000:g} kb", ha="center", va="top", fontsize=FS)
    scale.text(x1, 0.62, f"{chrom}:{x0:,}-{x1:,}" if chrom else f"{x0:,}-{x1:,}",
               ha="right", va="top", fontsize=FS)
    axes.append(scale)
    if ylabel:
        fig.text(left + width + 0.012, (top + y_cursor) / 2, ylabel, rotation=270, va="center",
                 ha="left", fontsize=FS)
    if show:
        plt.show()
    return fig, axes


def plot_benchmark(table, *, metric: str = "value", method: str = "method", group: Optional[str] = None,
                   colors: Optional[Dict[str, str]] = None, ax=None, figsize=(5.0, 3.0),
                   xlabel: Optional[str] = None, show: bool = True):
    """One metric per method as horizontal bars, optionally grouped (for example by test stratum)."""
    import matplotlib.pyplot as plt

    df = table
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    colors = colors or {}
    methods = list(dict.fromkeys(df[method].astype(str)))
    groups = list(dict.fromkeys(df[group].astype(str))) if group else [None]
    hgt = 0.8 / len(groups)
    for gi, g in enumerate(groups):
        d = df if g is None else df[df[group].astype(str) == g]
        d = d.set_index(d[method].astype(str)).reindex(methods)
        y = np.arange(len(methods))[::-1] + (gi - (len(groups) - 1) / 2) * hgt
        ax.barh(y, d[metric].to_numpy(float), hgt * 0.92, lw=0,
                color=[colors.get(m, PALETTE["aceso"] if m.lower().startswith("aceso")
                                  else PALETTE["scooby"]) for m in methods],
                alpha=1.0 if gi == len(groups) - 1 else 0.55,
                label=None if g is None else str(g))
    ax.set_yticks(np.arange(len(methods))[::-1])
    ax.set_yticklabels(methods, fontsize=FS)
    ax.set_xlabel(xlabel or metric, fontsize=FS)
    ax.tick_params(labelsize=FS)
    if group:
        ax.legend(frameon=False, fontsize=FS, loc="lower right")
    if show:
        plt.show()
    return fig, ax


def plot_panel_size(table, *, size: str = "n_genes", metric: str = "value", method: str = "method",
                    colors: Optional[Dict[str, str]] = None, ax=None, figsize=(4.2, 3.0),
                    ylabel: Optional[str] = None, show: bool = True):
    """A metric against the number of genes in the input panel, one line per method, log x."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    colors = colors or {}
    for m, d in table.groupby(table[method].astype(str)):
        d = d.sort_values(size)
        c = colors.get(m, PALETTE["aceso"] if m.lower().startswith("aceso") else PALETTE["scooby"])
        ax.plot(d[size], d[metric], "-o", ms=4.5, lw=1.6, color=c, mec="white", mew=0.7, label=m)
    ax.set_xscale("log")
    ax.set_xlabel("genes in the panel", fontsize=FS)
    ax.set_ylabel(ylabel or metric, fontsize=FS)
    ax.tick_params(labelsize=FS)
    ax.legend(frameon=False, fontsize=FS)
    if show:
        plt.show()
    return fig, ax


def plot_state_embedding(xy, labels: Sequence[str], *, colors: Optional[Dict[str, str]] = None,
                         ax=None, figsize=(3.6, 3.6), size: float = 2.0, label_points: bool = True,
                         title: str = "", show: bool = True):
    """A two-dimensional embedding of cell states, coloured by label, each label named on the plot."""
    import matplotlib.pyplot as plt

    xy = np.asarray(xy, float)
    labels = np.asarray(labels).astype(str)
    colors = colors or celltype_colors(labels)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    for g in dict.fromkeys(labels):
        m = labels == g
        ax.scatter(xy[m, 0], xy[m, 1], s=size, c=colors.get(g, PALETTE["other"]), lw=0, rasterized=True)
        if label_points:
            ax.text(np.median(xy[m, 0]), np.median(xy[m, 1]), g.replace("_", " "), fontsize=FS,
                    ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, loc="left")
    if show:
        plt.show()
    return fig, ax
