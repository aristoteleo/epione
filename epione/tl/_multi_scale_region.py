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
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..utils import console
from ..utils._genome import Genome
from ._footprint import (
    _seq_to_codes, _codes_to_kmers, compute_tn5_bias_table,
)


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
    kmer_length: int
    smooth_radius: Union[int, str]

    @property
    def footprint_size(self) -> np.ndarray:
        """Centre window width (=2 · scale), which is what scPrinter
        labels on the y-axis of its heatmaps."""
        return 2 * self.scales


# ---------------------------------------------------------------------------
# scPrinter port: windowing + scoring
# ---------------------------------------------------------------------------

def _rz_conv(a: np.ndarray, n: int) -> np.ndarray:
    """Centred rolling sum of width ``2n``. Pure port of
    ``scprinter.utils.rz_conv``."""
    if n == 0:
        return a
    pad_shape = a.shape[:-1] + (n,)
    padded = np.concatenate(
        [np.zeros(pad_shape, dtype=a.dtype), a,
         np.zeros(pad_shape, dtype=a.dtype)], axis=-1,
    )
    cs = np.cumsum(padded, axis=-1)
    return cs[..., 2 * n :] - cs[..., : -2 * n]


def _window_sums(x: np.ndarray, foot_R: int, flank_R: int):
    """Per-position (left_flank, centre, right_flank) sums.

    Geometry matches ``scprinter.footprint.footprintWindowSum``:
    centre has width ``2 · foot_R``; each flank has width
    ``flank_R // 2``; flanks are adjacent to centre on both sides.
    """
    half_flank = flank_R // 2
    shift = half_flank + foot_R
    pad_shape = x.shape[:-1] + (shift,)
    zero = np.zeros(pad_shape, dtype=x.dtype)
    left_shifted = np.concatenate([zero, x], axis=-1)
    left_flank = _rz_conv(left_shifted, half_flank)[..., : x.shape[-1]]
    right_shifted = np.concatenate([x, zero], axis=-1)
    right_flank = _rz_conv(right_shifted, half_flank)[..., shift:]
    centre = _rz_conv(x, foot_R)
    return left_flank, centre, right_flank


def _footprint_pvals(insertions: np.ndarray, bias: np.ndarray,
                     foot_R: int, flank_R: int) -> np.ndarray:
    """Pure-binomial-null version of ``footprintScoring``.

    Tests *centre depletion* (one-sided) on both sides; returns
    ``-log10 max(p_left, p_right)``.
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
    p = np.where(mask, 1.0, p)
    return -np.log10(np.clip(p, 1e-20, 1.0))


def _region_footprint_score(insertions: np.ndarray, bias: np.ndarray,
                             foot_R: int, flank_R: int,
                             smooth_R: Optional[int] = None) -> np.ndarray:
    """Port of ``scprinter.footprint.regionFootprintScore`` (return_pval
    branch): -log10 p + maximum_filter + rolling mean along position."""
    from scipy.ndimage import maximum_filter
    if smooth_R is None:
        smooth_R = max(foot_R // 2, 1)
    F = _footprint_pvals(insertions, bias, foot_R, flank_R)
    F[np.isnan(F)] = 0; F[np.isinf(F)] = 20
    mf_shape = [0] * F.ndim; mf_shape[-1] = 2 * smooth_R
    F = maximum_filter(F, tuple(mf_shape), origin=-1)
    F = _rz_conv(F, smooth_R) / (2 * smooth_R)
    F[np.isnan(F)] = 0; F[np.isinf(F)] = 20
    return F


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
    genome: Union[Genome, str, Path],
    bias_table: Optional[np.ndarray] = None,
    flank: int = 200,
    scales: Optional[Sequence[int]] = None,
    kmer_length: int = 6,
    min_cells_per_group: int = 10,
    smooth_radius: Union[int, str] = "auto",
    verbose: bool = True,
) -> MultiScaleRegion:
    """scPrinter-style multi-scale footprint for one or many motif sites.

    For each motif position in ``regions``, extract the per-bp Tn5 cut
    track and the hexamer bias track in ±``flank`` bp around the motif
    centre, strand-correct (negative-strand sites are reversed), sum
    across sites. Then run scPrinter's edge-vs-centre binomial test at
    each (scale, position).

    Parameters
    ----------
    adata
        AnnData with ``uns['files']['fragments']`` and a ``groupby``
        column in ``obs``.
    regions
        DataFrame or iterable of dicts with ``chrom`` / ``center``
        (or ``start`` + ``end``) / optional ``strand``. Pass ``N=1``
        for single-site scPrinter-Fig-1h-style heatmap (TF dot +
        nucleosome triangles); ``N > 100`` for aggregate (clean TF
        signal, nucleosome washes out).
    groupby
        ``obs`` column — one panel per category.
    genome
        ``Genome`` or FASTA path for the hexamer bias track.
    bias_table
        Pre-computed hexamer bias (``compute_tn5_bias_table`` output).
        If ``None`` it's computed on the fly.
    flank
        Half-width (bp) of the extraction / plot window.
    scales
        Footprint radii to scan. Default: ``range(2, 101, 2)`` →
        centre widths 4..200 bp (small = TF, large = nucleosome).
    min_cells_per_group
        Drop groups with fewer cells than this.
    smooth_radius
        Rolling-mean radius for the final score. ``'auto'`` means
        ``max(R // 2, 1)`` per scale (scPrinter default). ``0``
        disables smoothing.

    Returns
    -------
    :class:`MultiScaleRegion` with ``.score`` of shape
    ``(n_groups, n_scales, n_positions)``.
    """
    import pyfaidx
    import pysam

    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise ValueError(
            "adata.uns['files']['fragments'] missing — run "
            "epi.pp.import_fragments first"
        )
    frag_file = str(adata.uns["files"]["fragments"])

    regions_df = _normalise_regions(regions)
    if len(regions_df) == 0:
        raise ValueError("regions is empty")

    # Hexamer bias
    if bias_table is None:
        fasta = genome.fasta if isinstance(genome, Genome) else str(genome)
        bias_table = compute_tn5_bias_table(
            frag_file, fasta, kmer_length=kmer_length
        )
    fasta = genome.fasta if isinstance(genome, Genome) else str(genome)
    fa = pyfaidx.Fasta(str(fasta))

    # Groups / barcodes
    obs_col = adata.obs[groupby].astype(str)
    counts = obs_col.value_counts()
    keep = sorted(g for g, n in counts.items() if n >= min_cells_per_group)
    if not keep:
        raise ValueError("no groups with >= min_cells_per_group cells")
    g_idx = {g: i for i, g in enumerate(keep)}
    bc_to_g = {bc: g for bc, g in zip(adata.obs_names, obs_col)
                if g in g_idx}

    k = int(kmer_length); half_k = k // 2
    N = 2 * flank + 1
    agg_ins = np.zeros((len(keep), N), dtype=np.float32)
    agg_bias = np.zeros(N, dtype=np.float64)
    n_ins_sites = 0
    n_bias_sites = 0

    tbx = pysam.TabixFile(frag_file)
    for _, row in regions_df.iterrows():
        c, cen, strand = row["chrom"], int(row["center"]), row["strand"]
        lo, hi = cen - flank, cen + flank + 1
        if c not in tbx.contigs: continue

        # Per-site insertion track
        ins = np.zeros((len(keep), N), dtype=np.float32)
        for line in tbx.fetch(c, lo, hi):
            parts = line.split("\t")
            if len(parts) < 4: continue
            bc = parts[3]; g = bc_to_g.get(bc)
            if g is None: continue
            try:
                s = int(parts[1]); e = int(parts[2])
            except ValueError:
                continue
            gi = g_idx[g]
            for cut in (s, e - 1):
                if lo <= cut < hi:
                    ins[gi, cut - lo] += 1
        if strand == "-":
            ins = ins[:, ::-1]
        agg_ins += ins
        n_ins_sites += 1

        # Per-site bias track
        seq_start = lo - half_k
        seq_len = N + k - 1
        try:
            seq = str(fa[c][seq_start : seq_start + seq_len]).upper()
        except Exception:
            continue
        codes = _seq_to_codes(seq)
        km = _codes_to_kmers(codes, k)
        if len(km) != N: continue
        bp = np.where(km >= 0, bias_table[np.clip(km, 0, None)], 1.0)
        if strand == "-":
            bp = bp[::-1]
        agg_bias += bp
        n_bias_sites += 1

    tbx.close()
    if n_bias_sites == 0:
        raise RuntimeError("no valid regions after filtering")

    if verbose:
        console.level1(
            f"multi_scale_footprint_region: aggregated "
            f"{n_ins_sites:,} sites × {len(keep)} groups"
        )

    # Score at each scale
    if scales is None:
        scales = list(range(2, 101, 2))
    scales_arr = np.asarray(list(scales), dtype=np.int32)
    stack = np.zeros((len(keep), len(scales_arr), N), dtype=np.float32)
    for si, R in enumerate(scales_arr):
        sR = None if smooth_radius == "auto" else (
            None if smooth_radius is None else int(smooth_radius)
        )
        stack[:, si, :] = _region_footprint_score(
            agg_ins, agg_bias, int(R), int(R), smooth_R=sR,
        )
    positions = np.arange(-flank, flank + 1)

    return MultiScaleRegion(
        groups=list(keep),
        scales=scales_arr,
        positions=positions,
        score=stack,
        insertions=agg_ins,
        bias=agg_bias.astype(np.float32),
        n_sites=n_bias_sites,
        flank=flank,
        kmer_length=k,
        smooth_radius=smooth_radius,
    )


def plot_multi_scale_footprint_region(
    result: MultiScaleRegion,
    *,
    groups: Optional[Sequence[str]] = None,
    order: Optional[Sequence[str]] = None,
    ncols: int = 4,
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = None,
    vpercentile: float = 99.0,
    cmap: str = "Blues",
    zoom: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show: bool = True,
):
    """Render the scPrinter-style 2D heatmap (footprint size × position)
    per celltype group — one panel each.

    Parameters
    ----------
    result
        Output of :func:`multi_scale_footprint_region`.
    groups, order
        Subset / reorder panel layout.
    ncols
        Number of columns in the subplot grid.
    vmin, vmax
        Colour range. ``vmax=None`` (default) sets it to the
        ``vpercentile``-th percentile of the plotted data — so the
        colour bar adapts to each aggregate's actual signal range.
        Values above ``vmax`` saturate to the darkest colour, which
        is desired when one group has a much stronger footprint than
        the others (it stays saturated while the weaker groups still
        render on the same shared colour bar).
        Pass an explicit value (e.g. scPrinter's own ``vmax=2.0``) to
        override.
    vpercentile
        Percentile used when ``vmax`` is not given. Default 99.
    zoom
        Restrict position axis to ``±zoom`` bp.
    """
    import matplotlib.pyplot as plt
    disp = result.score
    scales = result.scales
    positions = result.positions
    insertions = result.insertions

    available = list(result.groups)
    usable = [g for g in (groups or available) if g in available]
    if order is not None:
        usable = [g for g in order if g in usable]
    if not usable:
        raise ValueError(f"no groups to plot; available = {available!r}")

    idx = [available.index(g) for g in usable]
    panels = disp[idx]
    pos = positions
    if zoom is not None:
        mask = np.abs(pos) <= int(zoom)
        panels = panels[..., mask]
        pos = pos[mask]

    # Auto-scale vmax to the 99-pctl of the plotted data so we don't
    # bake in scPrinter's vmax=2.5 (which was tuned for their MLP-
    # calibrated null; raw pvalue_log10 values here can exceed 10
    # and would otherwise all saturate to the same colour).
    if vmax is None:
        vmax = float(np.nanpercentile(panels, vpercentile))
        vmax = max(vmax, 1e-6)
    if vmin is None:
        vmin = 0.0

    n = len(usable)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (3.3 * ncols, 2.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                              squeeze=False, sharex=True, sharey=True)
    axes_flat = axes.ravel().tolist()

    y_lo, y_hi = float(2 * scales[0]), float(2 * scales[-1])
    for i, g in enumerate(usable):
        ax = axes_flat[i]
        im = ax.imshow(
            panels[i], aspect="auto", origin="lower",
            extent=[pos[0], pos[-1], y_lo, y_hi],
            cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        n_cuts = int(insertions[available.index(g)].sum())
        ax.set_title(f"{g}  (n_cuts={n_cuts:,})", fontsize=9)
        ax.axvline(0, color="black", lw=0.4, ls=":")
        if i % ncols == 0:
            ax.set_ylabel("footprint size (bp)")
        if i // ncols == nrows - 1:
            ax.set_xlabel("position (bp)")
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.colorbar(im, ax=axes, shrink=0.6, label="-log10 p")
    if title is None:
        title = (f"Multi-scale footprint · {result.n_sites:,} region"
                  f"{'s' if result.n_sites > 1 else ''}"
                  f" · scPrinter-style binomial null")
    fig.suptitle(title, fontsize=11, y=0.99)
    if show:
        try:
            from IPython.display import display
            display(fig); plt.close(fig)
        except Exception:
            pass
    return fig, axes_flat


__all__ = [
    "multi_scale_footprint_region",
    "plot_multi_scale_footprint_region",
    "rank_sites_by_cut_density",
    "MultiScaleRegion",
]
