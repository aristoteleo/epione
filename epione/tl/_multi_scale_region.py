"""scPrinter-style multi-scale footprint on a raw Tn5 insertion track.

``epi.tl.multi_scale_footprint_region`` takes a set of motif positions
(one or many), extracts per-bp Tn5 cuts and hexamer bias in ±``flank``
bp, sums over positions (strand-corrected), and runs an edge-vs-centre
binomial test at each (scale, position). The output is a 2D heatmap
(scale × position) per celltype — the classic scPrinter visualisation
that shows **TF footprint at small scales + nucleosomes at large
scales** (when applied to a single region; aggregating across many
motif sites averages out nucleosome phasing but sharpens the TF
signal).

Algorithmic correspondence with scPrinter (Hu *et al.* 2024):

* Window geometry: **centre width = 2 R** (footprint radius),
  **left/right flank width = R / 2** each, adjacent to centre — a port
  of ``scprinter.footprint.footprintWindowSum``.
* Statistic: **-log10 max(p_left, p_right)** where each p-value is a
  one-sided Normal CDF on the centre-vs-total ratio — a port of
  ``scprinter.footprint.footprintScoring`` with ``return_pval=True``.
* Null: the **binomial normal approximation** ``(μ = bias_centre / bias_total,
  σ = √(μ(1-μ)/n_total))``. This is the closed-form version of
  scPrinter's pretrained MLP null calibrator — loses the learned
  overdispersion correction but has zero dependencies.
* Smoothing: ``maximum_filter`` + rolling mean along the position axis —
  port of ``scprinter.footprint.regionFootprintScore``.

PRINT mode
----------
``bias`` and ``null`` make the two approximations above optional. With
``bias=epi.tl.tn5_bias(genome)`` (the PRINT CNN) and ``null='auto'`` the
learned dispersion model replaces the binomial null and the result is
scPrinter's score exactly (see :mod:`epione.tl._print`). Insertions are
then read with the cut convention the bias model was trained under, which
for PRINT differs from the hexamer table's by one base at fragment ends.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..utils import console
from epione.core.genome import Genome
from ._print import (
    DispersionModel, _pvalue_to_score, _rz_conv, _window_sums,
    print_footprint_pvalues,
)
from ._tn5_bias import KmerBias, Tn5Bias


@dataclass
class MultiScaleRegion:
    """scPrinter-style 2D footprint result."""
    groups: List[str]
    scales: np.ndarray              # footprint radii (bp)
    positions: np.ndarray           # -flank..flank
    score: np.ndarray               # (n_groups, n_scales, n_positions) -log10 p
    insertions: np.ndarray          # (n_groups, n_positions) aggregated cuts
    bias: np.ndarray                # (n_positions,) aggregated hexamer bias
    n_sites: int
    flank: int
    kmer_length: Optional[int]
    smooth_radius: Union[int, str]
    #: Bias model that produced :attr:`bias` (``'kmer'``, ``'print'``, ...).
    bias_model: str = "kmer"
    #: Null used for the p-values: ``'binomial'`` or ``'dispersion'``.
    null: str = "binomial"

    @property
    def footprint_size(self) -> np.ndarray:
        """Centre window width (=2 · scale), which is what scPrinter
        labels on the y-axis of its heatmaps."""
        return 2 * self.scales


# ---------------------------------------------------------------------------
# scPrinter port: windowing + scoring
# ---------------------------------------------------------------------------

# ``_rz_conv`` and ``_window_sums`` are shared with :mod:`._print` so
# both nulls see identical (float64, scPrinter-exact) window sums.


def _footprint_pvals(insertions: np.ndarray, bias: np.ndarray,
                     foot_R: int, flank_R: int) -> np.ndarray:
    """Binomial-null version of ``footprintScoring``.

    Tests *centre depletion* (one-sided) on both sides and returns
    ``max(p_left, p_right)``; positions with no cuts on either side get
    p = 1. Scoring (-log10, smoothing) happens in
    :func:`._print._pvalue_to_score`, shared with the dispersion null.
    """
    from scipy.stats import norm
    b_left, b_centre, b_right = _window_sums(bias, foot_R, flank_R)
    i_left, i_centre, i_right = _window_sums(insertions, foot_R, flank_R)
    total_left = i_centre + i_left
    total_right = i_centre + i_right

    mu_l = b_centre / np.maximum(b_centre + b_left, 1e-12)
    mu_r = b_centre / np.maximum(b_centre + b_right, 1e-12)
    sd_l = np.sqrt(np.maximum(mu_l * (1 - mu_l), 1e-12)
                    / np.maximum(total_left, 1))
    sd_r = np.sqrt(np.maximum(mu_r * (1 - mu_r), 1e-12)
                    / np.maximum(total_right, 1))

    fg_l = i_centre / np.maximum(total_left, 1e-9)
    fg_r = i_centre / np.maximum(total_right, 1e-9)
    p_left = norm.cdf(fg_l, mu_l, sd_l)
    p_right = norm.cdf(fg_r, mu_r, sd_r)
    p = np.maximum(p_left, p_right)
    # Mask positions with no coverage on either side.
    mask = (total_left < 1) | (total_right < 1)
    return np.where(mask, 1.0, p)


def _region_footprint_score(insertions: np.ndarray, bias: np.ndarray,
                             foot_R: int, flank_R: int,
                             smooth_R: Optional[int] = None,
                             *, null: str = "binomial",
                             dispersion: Optional[DispersionModel] = None,
                             ) -> np.ndarray:
    """One radius of ``scprinter.footprint.regionFootprintScore``:
    p-values under the chosen null, then -log10 + max-filter + mean.

    ``smooth_R=None`` means scPrinter's ``int(R / 2)``; 0 disables
    smoothing.
    """
    if smooth_R is None:
        smooth_R = int(foot_R / 2)
    if null == "dispersion":
        p = print_footprint_pvalues(insertions, bias, foot_R, dispersion,
                                    flank_radius=flank_R)
    else:
        p = _footprint_pvals(insertions, bias, foot_R, flank_R)
    return _pvalue_to_score(p, int(smooth_R))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _normalise_regions(regions) -> pd.DataFrame:
    if isinstance(regions, pd.DataFrame):
        df = regions.copy()
    else:
        df = pd.DataFrame(list(regions))
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for src in ("chrom", "chromosome", "seqname", "seqnames"):
        if src in lower: rename[lower[src]] = "chrom"; break
    for src in ("strand",):
        if src in lower: rename[lower[src]] = "strand"; break
    df = df.rename(columns=rename)
    if "center" not in df.columns:
        if "start" in lower and "end" in lower:
            s = lower["start"]; e = lower["end"]
            df["center"] = ((df[s].astype(int) + df[e].astype(int)) // 2).astype(int)
        else:
            raise ValueError("regions needs ``center`` or ``start`` + ``end``")
    if "strand" not in df.columns:
        df["strand"] = "+"
    df["center"] = df["center"].astype(int)
    df["chrom"]  = df["chrom"].astype(str)
    df["strand"] = df["strand"].astype(str)
    return df[["chrom", "center", "strand"]].reset_index(drop=True)


def rank_sites_by_cut_density(
    adata,
    regions,
    *,
    groupby: str,
    target_group: str,
    flank: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """Rank motif positions by Tn5 cut density in one celltype.

    Returns ``regions`` with an extra ``{target_group}_cuts`` column,
    sorted in descending order. Useful for picking the "most active"
    sites of a TF in a specific lineage before aggregate footprinting.

    This reads fragments once per site, so for thousands of sites it
    takes minutes — cache the result.
    """
    import pysam
    frag_file = str(adata.uns["files"]["fragments"])
    bc_to_g = {bc: g for bc, g in zip(adata.obs_names,
                                       adata.obs[groupby].astype(str))
                if g == target_group}
    df = _normalise_regions(regions)
    tbx = pysam.TabixFile(frag_file)
    cuts = np.zeros(len(df), dtype=np.int32)
    for ri, row in df.iterrows():
        c = row["chrom"]; cen = int(row["center"])
        lo, hi = cen - flank, cen + flank + 1
        if c not in tbx.contigs: continue
        for line in tbx.fetch(c, lo, hi):
            parts = line.split("\t")
            if len(parts) < 4: continue
            if bc_to_g.get(parts[3]) is None: continue
            try:
                s = int(parts[1]); e = int(parts[2])
            except ValueError:
                continue
            for cut in (s, e - 1):
                if lo <= cut < hi:
                    cuts[ri] += 1
    tbx.close()
    out = df.copy()
    out[f"{target_group}_cuts"] = cuts
    if verbose:
        console.level2(
            f"ranked {len(df):,} sites by {target_group} cuts — "
            f"range [{cuts.min()}, {cuts.max()}]"
        )
    return out.sort_values(f"{target_group}_cuts", ascending=False).reset_index(drop=True)


def multi_scale_footprint_region(
    adata,
    regions,
    *,
    groupby: str,
    genome: Union[Genome, str, Path, None] = None,
    bias: Union[str, Tn5Bias, None] = None,
    null: str = "auto",
    bias_table: Optional[np.ndarray] = None,
    flank: int = 200,
    scales: Optional[Sequence[int]] = None,
    kmer_length: int = 6,
    min_cells_per_group: int = 10,
    smooth_radius: Union[int, str] = "auto",
    dispersion: Optional[DispersionModel] = None,
    verbose: bool = True,
) -> MultiScaleRegion:
    """Multi-scale footprint for one or many sites (scPrinter / PRINT).

    For each site in ``regions``, the per-base Tn5 cut track and the bias
    track in ±``flank`` bp around its centre are extracted, negative-strand
    sites reversed, and both summed across sites. Every (scale, position)
    is then tested for a cut-depleted centre against its flanks.

    Parameters
    ----------
    adata
        AnnData with ``uns['files']['fragments']`` and a ``groupby``
        column in ``obs``.
    regions
        DataFrame or iterable of dicts with ``chrom`` / ``center``
        (or ``start`` + ``end``) / optional ``strand``. One site gives
        scPrinter's single-region heatmap (TF dots at small scales,
        nucleosome triangles at large ones); hundreds of motif sites give
        an aggregate in which the TF signal sharpens and nucleosome
        phasing averages out.
    groupby
        ``obs`` column -- one panel per category.
    genome
        :class:`~epione.core.genome.Genome` or FASTA. Needed unless
        ``bias`` is a ready :class:`~epione.tl.Tn5Bias`.
    bias
        Expected-cut model. ``None`` / ``'kmer'``: Epione's hexamer table
        (estimated from the fragments unless ``bias_table`` is given) --
        the original behaviour. ``'print'``: the PRINT CNN, the bias
        scPrinter uses. Or any :class:`~epione.tl.Tn5Bias`, e.g. from
        :func:`epi.tl.tn5_bias`.
    null
        ``'dispersion'``: PRINT's learned null (requires PRINT bias, which
        it was fitted to). ``'binomial'``: closed-form normal
        approximation, no learned parameters. ``'auto'``: dispersion with
        a PRINT bias, binomial otherwise.
    bias_table
        Precomputed hexamer table for the kmer bias.
    flank
        Half-width (bp) of the extraction window. Scores are computed
        inside it, so within about ``2 * R`` of either edge the flanks
        are truncated and large-scale scores there are unreliable (this
        matches scPrinter, which computes within the region too). Read
        ~200 bp more than you want to show and pass ``zoom=`` to the plot.
    scales
        Footprint radii. Default ``range(2, 101, 2)`` (centre widths
        4..200 bp: small = TF, large = nucleosome).
    min_cells_per_group
        Drop groups with fewer cells than this.
    smooth_radius
        ``'auto'``: ``int(R / 2)`` per radius (scPrinter); an int fixes
        it; 0 disables smoothing.
    dispersion
        A loaded :class:`~epione.tl.DispersionModel`; loaded on demand.

    Returns
    -------
    :class:`MultiScaleRegion` with ``.score`` of shape
    ``(n_groups, n_scales, n_positions)``.

    Examples
    --------
    PRINT footprint of one peak summit, per cell type::

        bias = epi.tl.tn5_bias(epi.data.mm10)
        fp = epi.tl.multi_scale_footprint_region(
            adata, [{"chrom": "chr19", "center": 41_829_870}],
            groupby="celltype", bias=bias)
        epi.pl.plot_multi_scale_footprint_region(fp)
    """
    from ._print import _groups_and_reader

    # -- bias model and null -------------------------------------------------
    if bias is None or (isinstance(bias, str) and bias.lower() in ("kmer", "hexamer")):
        if genome is None:
            raise ValueError("genome is required for the hexamer bias")
        if bias_table is None:
            if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
                raise ValueError(
                    "adata.uns['files']['fragments'] missing — run "
                    "epi.pp.import_fragments first"
                )
        bias_model = KmerBias(
            genome, bias_table,
            fragment_file=adata.uns.get("files", {}).get("fragments"),
            kmer_length=kmer_length,
        )
    elif isinstance(bias, str):
        from ._tn5_bias import tn5_bias
        bias_model = tn5_bias(genome, method=bias)
    elif isinstance(bias, Tn5Bias):
        bias_model = bias
    else:
        raise TypeError("bias must be None, a method name, or a Tn5Bias")

    is_print = bias_model.name.startswith("print")
    if null == "auto":
        null = "dispersion" if is_print else "binomial"
    if null not in ("dispersion", "binomial"):
        raise ValueError("null must be 'auto', 'dispersion' or 'binomial'")
    if null == "dispersion":
        if not is_print:
            raise ValueError(
                "the dispersion null was fitted to PRINT bias sums; with a "
                f"{bias_model.name!r} bias its p-values are uncalibrated. Use "
                "bias='print' or null='binomial'."
            )
        if dispersion is None:
            dispersion = DispersionModel.load()

    # -- groups, sites --------------------------------------------------------
    G, keep, reader = _groups_and_reader(
        adata, groupby, min_cells_per_group, shift=bias_model.shift)
    regions_df = _normalise_regions(regions)
    if len(regions_df) == 0:
        raise ValueError("regions is empty")

    N = 2 * flank + 1
    agg_ins = np.zeros((len(keep), N), dtype=np.float32)
    agg_bias = np.zeros(N, dtype=np.float64)
    n_sites = 0
    for _, row in regions_df.iterrows():
        c, cen, strand = row["chrom"], int(row["center"]), row["strand"]
        lo, hi = cen - flank, cen + flank + 1
        try:
            b = bias_model.track(c, lo, hi)
        except KeyError:
            continue                      # contig absent from the genome
        ins = reader.group_insertions(c, lo, hi, G)
        if strand == "-":
            ins, b = ins[:, ::-1], b[::-1]
        agg_ins += ins
        agg_bias += b
        n_sites += 1
    reader.close()
    if n_sites == 0:
        raise RuntimeError("no valid regions after filtering")

    if verbose:
        console.level1(
            f"multi_scale_footprint_region: {n_sites:,} site(s) × "
            f"{len(keep)} groups · {bias_model.name} bias · {null} null"
        )

    # -- score ----------------------------------------------------------------
    if scales is None:
        scales = list(range(2, 101, 2))
    scales_arr = np.asarray(list(scales), dtype=np.int32)
    stack = np.zeros((len(keep), len(scales_arr), N), dtype=np.float32)
    for si, R in enumerate(scales_arr):
        sR = None if smooth_radius in ("auto", None) else int(smooth_radius)
        stack[:, si, :] = _region_footprint_score(
            agg_ins, agg_bias, int(R), int(R), smooth_R=sR,
            null=null, dispersion=dispersion,
        )

    return MultiScaleRegion(
        groups=list(keep),
        scales=scales_arr,
        positions=np.arange(-flank, flank + 1),
        score=stack,
        insertions=agg_ins,
        bias=agg_bias.astype(np.float32),
        n_sites=n_sites,
        flank=flank,
        kmer_length=getattr(bias_model, "k", None),
        smooth_radius=smooth_radius,
        bias_model=bias_model.name,
        null=null,
    )


def plot_multi_scale_footprint_region(
    result: MultiScaleRegion,
    *,
    groups: Optional[Sequence[str]] = None,
    order: Optional[Sequence[str]] = None,
    ncols: int = 4,
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = None,
    vpercentile: float = 100.0,
    cmap: str = "Blues",
    zoom: Optional[int] = None,
    tracks: bool = True,
    smooth: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show: bool = True,
):
    """scPrinter-style heatmap (footprint size x position) per group.

    With ``tracks=True`` (default) each heatmap sits under that group's
    cut track -- observed cuts filled, bias-expected cuts dashed -- on the
    same x-axis, so a footprint can be checked against the dip in cuts
    that produced it.

    Parameters
    ----------
    result
        Output of :func:`multi_scale_footprint_region`.
    groups, order
        Subset / reorder the panels.
    ncols
        Panels per row.
    vmin, vmax
        Colour range, shared by all panels so groups are comparable.
        ``vmax=None`` uses the ``vpercentile``-th percentile of the
        plotted data (100 = global max; 99 clips outliers).
    zoom
        Restrict the position axis to ``±zoom`` bp.
    tracks
        Draw the cut track above each heatmap.
    smooth
        Running-mean width (bp) for the displayed cut track only.

    Returns
    -------
    ``(fig, axes)``: a list of heatmap axes (and, with tracks, a parallel
    list in ``fig.track_axes``).
    """
    import matplotlib.pyplot as plt
    from ._print_plot import GROUP_COLORS, INK, INK2, MUTED, _style_axis, draw_cut_track

    available = list(result.groups)
    usable = [g for g in (groups or available) if g in available]
    if order is not None:
        usable = [g for g in order if g in usable]
    if not usable:
        raise ValueError(f"no groups to plot; available = {available!r}")
    idx = [available.index(g) for g in usable]

    pos = result.positions
    panels = result.score[idx]
    ins = result.insertions[idx]
    bias = result.bias
    if zoom is not None:
        m = np.abs(pos) <= int(zoom)
        panels, ins, bias, pos = panels[..., m], ins[:, m], bias[m], pos[m]

    if vmax is None:
        vmax = max(float(np.nanpercentile(panels, vpercentile)), 1e-6)
    if vmin is None:
        vmin = 0.0

    n = len(usable)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (3.3 * ncols + 0.6, (3.4 if tracks else 2.8) * nrows)
    fig = plt.figure(figsize=figsize)
    per = 2 if tracks else 1
    hr = ([0.9, 2.4] if tracks else [1.0]) * nrows
    gs = fig.add_gridspec(nrows * per, ncols + 1,
                          width_ratios=[1.0] * ncols + [0.05],
                          height_ratios=hr, hspace=0.18, wspace=0.18)

    sizes = 2 * result.scales
    heat_axes, track_axes = [], []
    im = None
    for i, g in enumerate(usable):
        r, c = divmod(i, ncols)
        hax = fig.add_subplot(gs[r * per + (per - 1), c])
        im = hax.imshow(
            panels[i], aspect="auto", origin="lower",
            extent=[pos[0], pos[-1], float(sizes[0]), float(sizes[-1])],
            cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        hax.axvline(0, color=INK, lw=0.4, ls=":")
        _style_axis(hax)
        if c == 0:
            hax.set_ylabel("footprint size (bp)", color=INK2, fontsize=9)
        if r == nrows - 1:
            hax.set_xlabel("position (bp)", color=INK2, fontsize=9)
        heat_axes.append(hax)
        label = f"{g}  ·  {int(ins[i].sum()):,} cuts"
        if tracks:
            tax = fig.add_subplot(gs[r * per, c], sharex=hax)
            draw_cut_track(tax, pos, ins[i], bias,
                           GROUP_COLORS[i % len(GROUP_COLORS)], label,
                           smooth=smooth)
            tax.tick_params(labelbottom=False)
            track_axes.append(tax)
        else:
            hax.set_title(label, fontsize=9, color=INK)

    cax = fig.add_subplot(gs[(nrows * per) - 1, ncols])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.set_label("-log10 p", fontsize=8, color=INK2)

    if title is None:
        title = (f"Multi-scale footprint · {result.n_sites:,} site"
                 f"{'s' if result.n_sites > 1 else ''} · "
                 f"{result.bias_model} bias · {result.null} null")
    fig.suptitle(title, fontsize=11, fontweight="bold", color=INK, y=0.995)
    fig.track_axes = track_axes
    if show:
        try:
            from IPython.display import display
            display(fig)
            plt.close(fig)
        except Exception:
            pass
    return fig, heat_axes


__all__ = [
    "multi_scale_footprint_region",
    "plot_multi_scale_footprint_region",
    "rank_sites_by_cut_density",
    "MultiScaleRegion",
]
