"""Grouped single-cell genome browser track (ArchR ``plotBrowserTrack``).

Port of ArchR 1.0.3 ``ArchRBrowser.R`` (MIT): ``.groupRegionSumArrows``,
``.bulkTracks``, ``.featureTracks``, ``.loopTracks`` and ``.geneTracks``.
Coverage is computed directly from tabix-indexed fragment files, so the
numbers match ArchR's Arrow-file reads tile for tile.
"""
from __future__ import annotations

import re
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from anndata import AnnData



# ArchR "stallion" in paletteDiscrete order (names mixed-sorted 1..20).
ARCHR_STALLION_ORDERED = [
    "#D51F26", "#272E6A", "#208A42", "#89288F", "#F47D2B",
    "#FEE500", "#8A9FD1", "#C06CAB", "#D8A767", "#90D5E4",
    "#89C75F", "#F37B7D", "#9983BD", "#D24B27", "#3BBCA8",
    "#6E4B9E", "#0C727C", "#7E1416", "#E6C2DC", "#3D3D3D",
]
# Raw vector order, used by colorRampPalette when interpolating > 20 colours.
_ARCHR_STALLION_RAW = [
    "#D51F26", "#272E6A", "#208A42", "#89288F", "#F47D2B",
    "#FEE500", "#8A9FD1", "#C06CAB", "#E6C2DC", "#90D5E4",
    "#89C75F", "#F37B7D", "#9983BD", "#D24B27", "#3BBCA8",
    "#6E4B9E", "#0C727C", "#7E1416", "#D8A767", "#3D3D3D",
]
_LOOP_COLORS = ["#E6E7E8", "#3A97FF", "#8816A7", "#000000"]
_GENE_STRAND_COLORS = {"-": "#1C86EE", "+": "#FF0000", "*": "#FF0000"}
_MM = 72.27 / 25.4  # ggplot2 linewidth unit (mm) in points


# ---------------------------------------------------------------------------
# Small R-compatible helpers
# ---------------------------------------------------------------------------

def _mixedsort(values) -> list:
    """Natural sort, matching ``gtools::mixedsort`` for typical labels."""
    def key(v):
        parts = re.split(r"(\d+)", str(v))
        return [(0, int(p)) if p.isdigit() else (1, p) for p in parts if p != ""]
    return sorted(pd.unique(pd.Series(list(values), dtype=object)), key=key)


def _color_ramp(colors: Sequence[str], n: int) -> list:
    """``grDevices::colorRampPalette(colors)(n)`` (linear RGB interpolation)."""
    rgb = np.array([mpl.colors.to_rgb(c) for c in colors]) * 255.0
    pos = np.linspace(0.0, 1.0, len(colors))
    t = np.linspace(0.0, 1.0, n)
    out = np.column_stack([np.interp(t, pos, rgb[:, k]) for k in range(3)])
    out = np.floor(out + 0.5).astype(int)
    return ["#%02X%02X%02X" % tuple(row) for row in out]


def archr_palette_discrete(values) -> Dict[str, str]:
    """ArchR ``paletteDiscrete(values, set="stallion")``."""
    vals = _mixedsort(values)
    if len(vals) > len(ARCHR_STALLION_ORDERED):
        cols = _color_ramp(_ARCHR_STALLION_RAW, len(vals))
    else:
        cols = ARCHR_STALLION_ORDERED[: len(vals)]
    return dict(zip(vals, cols))


def _first_col(df: pd.DataFrame, names: Sequence[str]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of the columns {list(names)} found in {list(df.columns)}")


def _standardise_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``chrom/start/end`` (+ strand/symbol when present)."""
    out = pd.DataFrame({
        "chrom": df[_first_col(df, ["chrom", "seqnames", "Chromosome", "chr"])].astype(str).to_numpy(),
        "start": df[_first_col(df, ["start", "Start"])].astype(np.int64).to_numpy(),
        "end": df[_first_col(df, ["end", "End"])].astype(np.int64).to_numpy(),
    })
    for src, dst in [(["strand", "Strand"], "strand"),
                     (["symbol", "gene_name", "gene", "name"], "symbol"),
                     (["value", "correlation", "score"], "value")]:
        try:
            out[dst] = df[_first_col(df, src)].to_numpy()
        except KeyError:
            pass
    return out


def _parse_region(region) -> Tuple[str, int, int]:
    if isinstance(region, (tuple, list)):
        chrom, start, end = region
        return str(chrom), int(start), int(end)
    if isinstance(region, str) and ":" in region:
        chrom, rest = region.split(":", 1)
        start, end = rest.replace(",", "").split("-")
        return chrom, int(start), int(end)
    raise ValueError(f"Could not parse region {region!r}")


def gene_region(
    gene_annotation: pd.DataFrame,
    gene: str,
    upstream: int = 50_000,
    downstream: int = 50_000,
) -> Tuple[str, int, int]:
    """TSS-centred window, as ``plotBrowserTrack(geneSymbol=...)`` builds it.

    The TSS is the gene start on ``+``/``*`` and the gene end on ``-``; the
    window is ``[TSS - upstream, TSS + downstream]`` (1-based, closed).
    """
    g = _standardise_intervals(gene_annotation)
    hit = g[g["symbol"].astype(str).str.lower() == str(gene).lower()]
    if hit.empty:
        raise KeyError(f"gene {gene!r} not found in gene_annotation")
    row = hit.iloc[0]
    tss = int(row["end"]) if str(row.get("strand", "+")) == "-" else int(row["start"])
    return str(row["chrom"]), tss - int(upstream), tss + int(downstream)


_LEC_M1 = 4294967087
_LEC_M2 = 4294944443


class RRandom:
    """R's RNG: ``set.seed(seed)`` then ``runif()`` / ``sample()``.

    Reproduces R >= 3.6 (``sample.kind = "Rejection"``) draws bit for bit
    for ``kind="Mersenne-Twister"`` (R's default) and
    ``kind="L'Ecuyer-CMRG"`` (set by ArchR's ``addArchRThreads(threads > 1)``).
    """

    def __init__(self, seed: int, kind: str = "Mersenne-Twister"):
        self.kind = kind
        lcg = lambda v: (69069 * v + 1) & 0xFFFFFFFF  # noqa: E731
        s = seed & 0xFFFFFFFF
        for _ in range(50):
            s = lcg(s)
        if kind == "Mersenne-Twister":
            key = np.empty(625, dtype=np.uint32)
            for j in range(625):
                s = lcg(s)
                key[j] = s
            self._bitgen = np.random.MT19937()
            self._bitgen.state = {"bit_generator": "MT19937",
                                  "state": {"key": key[1:], "pos": 624}}
        elif kind == "L'Ecuyer-CMRG":
            st = []
            for _ in range(6):
                s = lcg(s)
                while s >= _LEC_M2:
                    s = lcg(s)
                st.append(s)
            self._lec = st
        else:
            raise ValueError(f"unsupported RNG kind {kind!r}")

    def _raw(self) -> float:
        if self.kind == "Mersenne-Twister":
            return float(self._bitgen.random_raw()) * 2.3283064365386963e-10
        s = self._lec
        p1 = (1403580 * s[1] - 810728 * s[0]) % _LEC_M1
        s[0], s[1], s[2] = s[1], s[2], p1
        p2 = (527612 * s[5] - 1370589 * s[3]) % _LEC_M2
        s[3], s[4], s[5] = s[4], s[5], p2
        return ((p1 - p2) if p1 > p2 else (p1 - p2 + _LEC_M1)) * 2.328306549295727688e-10

    def unif_rand(self) -> float:
        x = self._raw()
        if x <= 0.0:
            return 0.5 * 2.328306437080797e-10
        if 1.0 - x <= 0.0:
            return 1.0 - 0.5 * 2.328306437080797e-10
        return x

    def _rbits(self, bits: int) -> int:
        v = 0
        for _ in range(0, bits + 1, 16):
            v = 65536 * v + int(np.floor(self.unif_rand() * 65536))
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: int) -> int:
        if dn <= 0:
            return 0
        bits = int(np.ceil(np.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample(self, n: int, k: int) -> np.ndarray:
        """0-based indices of ``sample.int(n, k)`` (without replacement)."""
        x = list(range(n))
        out = np.empty(k, dtype=np.int64)
        for i in range(k):
            j = self.unif_index(n)
            out[i] = x[j]
            n -= 1
            x[j] = x[n]
        return out


# ---------------------------------------------------------------------------
# Coverage (.groupRegionSumArrows)
# ---------------------------------------------------------------------------

def _select_cells(
    groups: pd.Series,
    min_cells: Optional[int],
    max_cells: int,
    use_groups: Optional[Sequence[str]],
    cells: Optional[Sequence[str]],
    seed: Optional[int],
    rng: str = "archr",
) -> pd.Series:
    groups = groups.dropna().astype(str)
    if min_cells is not None:
        tab = groups.value_counts()
        groups = groups[groups.isin(tab.index[tab >= min_cells])]
    if use_groups is not None:
        groups = groups[groups.isin([str(g) for g in use_groups])]
    if cells is not None:
        cells = pd.Index([str(c) for c in cells])
        missing = cells.difference(groups.index)
        if len(missing):
            raise KeyError(f"{len(missing)} injected cells are not in the filtered groups, e.g. {missing[0]!r}")
        return groups.loc[cells]
    tab = groups.value_counts()
    if (tab > max_cells).any():
        if rng in ("archr", "R"):
            kind = "L'Ecuyer-CMRG" if rng == "archr" else "Mersenne-Twister"
            rr = RRandom(1 if seed is None else int(seed), kind)
            draw = lambda n: rr.sample(n, max_cells)  # noqa: E731
        elif rng == "numpy":
            gen = np.random.default_rng(seed)
            draw = lambda n: gen.choice(n, max_cells, replace=False)  # noqa: E731
        else:
            raise ValueError("rng must be 'archr', 'R' or 'numpy'")
        keep = []
        for g in sorted(tab.index):
            idx = groups.index[groups.to_numpy() == g]
            if len(idx) > max_cells:
                idx = idx[draw(len(idx))]
            keep.append(idx)
        groups = groups.loc[np.concatenate(keep)]
    return groups


def _fragment_tiles(
    path: str,
    chrom: str,
    start: int,
    end: int,
    barcode_to_col: Mapping[str, int],
    tile_size: int,
    count_ends: bool,
):
    import pysam

    tiles, cols = [], []
    verbosity = pysam.set_verbosity(0)  # silence htslib "index older than data" noise
    try:
        tb = pysam.TabixFile(str(path))
    finally:
        pysam.set_verbosity(verbosity)
    with tb:
        if chrom not in tb.contigs:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        for line in tb.fetch(chrom, max(start - 1, 0), end):
            f = line.split("\t", 4)
            j = barcode_to_col.get(f[3])
            if j is None:
                continue
            s1 = int(f[1]) + 1
            e1 = int(f[2])
            tiles.append((s1 // tile_size) * tile_size)
            cols.append(j)
            if count_ends:
                tiles.append((e1 // tile_size) * tile_size)
                cols.append(j)
    return np.asarray(tiles, np.int64), np.asarray(cols, np.int64)


def browser_track_coverage(
    adata: AnnData,
    region,
    *,
    fragments: Union[str, Mapping[str, str]],
    group_by: str = "Clusters",
    sample_key: Optional[str] = "Sample",
    barcode_sep: Optional[str] = "#",
    use_groups: Optional[Sequence[str]] = None,
    tile_size: int = 250,
    norm_method: str = "ReadsInTSS",
    min_cells: Optional[int] = 25,
    max_cells: int = 500,
    cells: Optional[Sequence[str]] = None,
    seed: Optional[int] = 1,
    rng: str = "archr",
    count_ends: bool = False,
) -> pd.DataFrame:
    """Normalised per-group coverage of a region (ArchR ``.groupRegionSumArrows``).

    For every group, each tile of ``tile_size`` bp gets the number of cells
    with at least one fragment starting in that tile, scaled by
    ``1e4 / sum(norm_method)`` over the group's cells.

    Arguments:
        adata: cells in ``obs_names``; ``obs`` must hold ``group_by``,
            ``sample_key`` and the ``norm_method`` column.
        region: ``"chr:start-end"`` or ``(chrom, start, end)``, 1-based closed.
        fragments: one fragment file, or ``{sample: path}``. Files must be
            tabix-indexed (``epi.pp.ensure_tabix_index``).
        group_by: ``obs`` column defining the tracks.
        sample_key: ``obs`` column mapping cells to fragment files.
        barcode_sep: cell names are ``f"{sample}{barcode_sep}{barcode}"``
            (ArchR convention); ``None`` when ``obs_names`` are raw barcodes.
        use_groups: restrict to these groups.
        tile_size: tile width in bp.
        norm_method: ``"ReadsInTSS"``, ``"ReadsInPromoter"``, ``"nFrags"``
            (sum of that ``obs`` column per group), ``"nCells"`` or ``"None"``.
        min_cells: drop groups with fewer cells.
        max_cells: subsample larger groups to this many cells.
        cells: explicit cell list to use instead of random subsampling
            (e.g. the cells ArchR drew, for exact parity).
        seed: seed for the subsampling.
        rng: ``"archr"`` (default) reproduces ``set.seed(seed); sample()``
            under R's L'Ecuyer-CMRG generator, which ArchR switches to in
            ``addArchRThreads(threads > 1)`` -- i.e. the exact cells ArchR
            draws; ``"R"`` uses R's default Mersenne-Twister (ArchR with
            ``threads = 1``); ``"numpy"`` uses ``numpy.random.default_rng``.
        count_ends: ArchR 1.0.3 computes both tile indices from the fragment
            start, so only starts are counted (``False``, default). ``True``
            also counts the fragment end tile.

    Returns:
        DataFrame with columns ``x`` (tile start, NaN for ArchR's padding row
        past the last tile), ``y`` (normalised signal) and ``group``, in
        ArchR's row order.
    """
    chrom, start, end = _parse_region(region)
    if group_by not in adata.obs:
        raise KeyError(f"adata.obs[{group_by!r}] not found")
    groups = pd.Series(adata.obs[group_by].astype(object).to_numpy(),
                       index=adata.obs_names.astype(str))
    groups = _select_cells(groups, min_cells, max_cells, use_groups, cells, seed, rng)
    unique_groups = _mixedsort(groups.unique())
    region_tiles = np.arange(start // tile_size, end // tile_size + 2, dtype=np.int64) * tile_size
    tile_index = {int(t): i for i, t in enumerate(region_tiles)}
    group_idx = pd.Index(unique_groups).get_indexer(groups.to_numpy())

    if isinstance(fragments, (str, bytes)) or hasattr(fragments, "__fspath__"):
        fragments = {None: str(fragments)}
    cell_names = groups.index.to_numpy()
    if sample_key is not None and None not in fragments:
        samples = adata.obs.loc[cell_names, sample_key].astype(str).to_numpy()
    else:
        samples = np.full(len(cell_names), None, dtype=object)

    pairs = []
    for sample, path in fragments.items():
        mask = samples == sample if sample is not None else np.ones(len(cell_names), bool)
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        if barcode_sep is not None and sample is not None:
            prefix = f"{sample}{barcode_sep}"
            bcs = [c[len(prefix):] if c.startswith(prefix) else c for c in cell_names[idx]]
        else:
            bcs = list(cell_names[idx])
        t, j = _fragment_tiles(path, chrom, start, end, dict(zip(bcs, idx)),
                               tile_size, count_ends)
        rows = np.array([tile_index.get(int(v), -1) for v in t], dtype=np.int64)
        ok = rows >= 0
        pairs.append(rows[ok] * len(cell_names) + j[ok])

    group_mat = np.zeros((len(region_tiles), len(unique_groups)), dtype=np.float64)
    if pairs:
        code = np.unique(np.concatenate(pairs))
        np.add.at(group_mat, (code // len(cell_names), group_idx[code % len(cell_names)]), 1.0)

    # ArchR's long-format layout: non-zero tiles (column-major), then the
    # flanking zero tiles, de-duplicated, row 0 dropped.
    col, row = np.nonzero(group_mat.T)
    row = row + 1
    col = col + 1
    y = group_mat[row - 1, col - 1]
    r_all = np.concatenate([row, row - 1, row + 1])
    c_all = np.concatenate([col, col, col])
    y_all = np.concatenate([y, np.zeros_like(y), np.zeros_like(y)])
    df = pd.DataFrame({"row": r_all, "col": c_all, "y": y_all})
    df = df[~df.duplicated(["row", "col"])]
    df = df[df["row"] > 0]
    x = np.full(len(df), np.nan)
    inside = df["row"].to_numpy() <= len(region_tiles)
    x[inside] = region_tiles[df["row"].to_numpy()[inside] - 1]
    out = pd.DataFrame({
        "x": x,
        "y": df["y"].to_numpy(),
        "group": np.asarray(unique_groups, dtype=object)[df["col"].to_numpy() - 1],
    })

    nm = norm_method.lower()
    if nm in ("readsintss", "readsinpromoter", "nfrags"):
        v = adata.obs.loc[groups.index, norm_method].astype(np.float64)
        factors = v.groupby(groups.to_numpy()).sum()
    elif nm == "ncells":
        factors = groups.value_counts().astype(np.float64)
    elif nm == "none":
        factors = pd.Series(1e4, index=unique_groups)
    else:
        raise ValueError(f"Norm Method Not Recognized : {norm_method}")
    scale = 1e4 / factors
    out["y"] = out["y"].to_numpy() * scale.reindex(out["group"]).to_numpy()
    out.attrs.update(region=(chrom, start, end), cells=list(groups.index),
                     groups=list(unique_groups))
    return out


def clip_coverage(df: pd.DataFrame, ylim: Optional[Tuple[float, float]] = None):
    """``.bulkTracks`` y-clipping: ``[0, q(0.999)]`` or quantiles ``ylim``."""
    y = df["y"].to_numpy(dtype=np.float64)
    if ylim is not None:
        lim = np.quantile(y, list(ylim))
    else:
        lim = np.array([0.0, np.quantile(y, 0.999)])
    out = df.copy()
    out["y"] = np.clip(y, lim[0], lim[1])
    return out, (float(lim[0]), float(lim[1]))


# ---------------------------------------------------------------------------
# Feature / loop / gene tracks
# ---------------------------------------------------------------------------

def _overlaps(df: pd.DataFrame, chrom: str, start: int, end: int, within: bool = False):
    on_chrom = df["chrom"].astype(str).to_numpy() == chrom
    if within:
        return df[on_chrom & (df["start"].to_numpy() >= start) & (df["end"].to_numpy() <= end)]
    return df[on_chrom & (df["start"].to_numpy() <= end) & (df["end"].to_numpy() >= start)]


def feature_track_data(features, region) -> pd.DataFrame:
    """Intervals of each feature set overlapping the region (``.featureTracks``)."""
    chrom, start, end = _parse_region(region)
    if isinstance(features, pd.DataFrame):
        features = {"FeatureTrack": features}
    frames = []
    for name, df in features.items():
        sub = _overlaps(_standardise_intervals(df), chrom, start, end)
        frames.append(sub[["chrom", "start", "end"]].assign(name=name))
    return pd.concat(frames, ignore_index=True)


def loops_from_coaccessibility(co: pd.DataFrame, min_correlation: float = 0.5,
                               resolution: Optional[int] = 1000) -> pd.DataFrame:
    """Convert ``epi.tl.coaccessibility`` output to ``chrom/start/end/value`` loops."""
    co = co[co["correlation"] >= min_correlation]
    a = co["peak1_center"].to_numpy()
    b = co["peak2_center"].to_numpy()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    if resolution:
        lo = np.floor(lo / resolution) * resolution + resolution / 2
        hi = np.floor(hi / resolution) * resolution + resolution / 2
    out = pd.DataFrame({"chrom": co["chrom"].astype(str).to_numpy(),
                        "start": lo.astype(np.int64), "end": hi.astype(np.int64),
                        "value": co["correlation"].to_numpy()})
    out = out.sort_values("value", ascending=False).drop_duplicates(["chrom", "start", "end"])
    return out.reset_index(drop=True)


def loop_track_data(loops, region, n: int = 100, r: float = 100.0) -> pd.DataFrame:
    """Half-ellipse arcs for loops fully inside the region (``getArchDF``)."""
    chrom, start, end = _parse_region(region)
    if isinstance(loops, pd.DataFrame):
        loops = {"Loops": loops}
    frames = []
    for name, df in loops.items():
        sub = _overlaps(_standardise_intervals(df), chrom, start, end, within=True)
        if sub.empty:
            continue
        s = sub["start"].to_numpy(np.float64)
        e = sub["end"].to_numpy(np.float64)
        val = sub["value"].to_numpy(np.float64) if "value" in sub else np.ones(len(sub))
        rx = (e - s) / 2.0
        rscale = r * (rx / rx.max())
        cx = s + rx
        ang = np.linspace(np.pi, 2 * np.pi, n)
        x = (rx[:, None] * np.cos(ang)[None, :] + cx[:, None]).ravel()
        y = (rscale[:, None] * np.sin(ang)[None, :]).ravel()
        frames.append(pd.DataFrame({
            "x": x, "y": y,
            "id": np.repeat([f"l{i + 1}" for i in range(len(sub))], n),
            "value": np.repeat(val, n), "name": name,
        }))
    if not frames:
        return pd.DataFrame(columns=["x", "y", "id", "value", "name"])
    return pd.concat(frames, ignore_index=True)


def gene_track_data(gene_annotation: pd.DataFrame, region) -> pd.DataFrame:
    """Genes overlapping the region with ArchR's row packing (``.geneTracks``)."""
    chrom, start, end = _parse_region(region)
    g = _standardise_intervals(gene_annotation)
    if "strand" not in g:
        g["strand"] = "*"
    g = g[g["chrom"] == chrom].copy()
    g = g.sort_values(["start", "end"], kind="mergesort")
    g = _overlaps(g, chrom, start, end).copy()
    if g.empty:
        return pd.DataFrame(columns=["chrom", "start", "end", "strand", "symbol", "cluster"])
    width = (g["end"] - g["start"] + 1).to_numpy()
    g["start"] = np.maximum(g["start"].to_numpy(), start)
    g["end"] = np.minimum(g["end"].to_numpy(), end)
    st = g["start"].to_numpy()
    en = g["end"].to_numpy()
    med = float(np.median(width))
    cluster = np.zeros(len(g), dtype=np.int64)
    for i in range(len(g)):
        if i == 0:
            cluster[i] = 1
            continue
        maxc = int(cluster.max())
        for j in range(1, maxc + 1):
            cid = maxc - j + 1
            last = np.where(cluster == cid)[0][-1]
            if st[i] > en[last] + med:
                cluster[i] = cid
        if cluster[i] == 0:
            cluster[i] = cluster[i - 1] + 1
    g["cluster"] = cluster
    return g[["chrom", "start", "end", "strand", "symbol", "cluster"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _draw_bulk(fig, spec, cov, groups, palette, region, lim, norm_method, title, highlight):
    chrom, start, end = region
    sub = spec.subgridspec(len(groups), 1, hspace=0.0)
    axes = []
    for k, grp in enumerate(groups):
        ax = fig.add_subplot(sub[k])
        d = cov[(cov["group"] == grp) & cov["x"].between(start, end)].sort_values("x", kind="mergesort")
        col = palette.get(grp, "#3D3D3D")
        if len(d):
            ax.fill_between(d["x"].to_numpy(), 0, d["y"].to_numpy(), color=col, lw=0.5 * _MM * 0.5, edgecolor=col)
        ax.set_xlim(start, end)
        ax.set_ylim(lim)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=2, labelsize=6)
        if k < len(groups) - 1:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        ax.text(1.01, 0.5, str(grp), transform=ax.transAxes, va="center", ha="left",
                fontsize=7, bbox=dict(boxstyle="square,pad=0.2", fc="#D9D9D9", ec="black", lw=0.5))
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        _draw_highlight(ax, highlight, region)
        axes.append(ax)
    axes[-1].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _p: f"{int(round(v))}"))
    axes[0].set_title(f"{chrom}:{start - 1}-{end} {title}".rstrip(), fontsize=8, loc="left")
    mid = axes[len(axes) // 2]
    mid.set_ylabel(f"Coverage\n(Norm. ATAC Signal Range ({round(min(lim), 2):g}-"
                   f"{round(max(lim), 2):g}) by {norm_method})", fontsize=6)
    return axes


def _draw_highlight(ax, highlight, region):
    if highlight is None:
        return
    chrom, start, end = region
    h = _overlaps(_standardise_intervals(highlight), chrom, start, end)
    for s, e in zip(h["start"], h["end"]):
        ax.axvspan(max(s, start), min(e, end), color="firebrick", alpha=0.2, lw=0)


def _strip(ax, label):
    ax.text(1.01, 0.5, label, transform=ax.transAxes, va="center", ha="left", fontsize=7)


def _draw_features(ax, fdata, region, highlight):
    chrom, start, end = region
    names = list(dict.fromkeys(fdata["name"]))
    pal = archr_palette_discrete(list(reversed(names)))
    for k, nm in enumerate(names):
        d = fdata[fdata["name"] == nm]
        segs = [[(s, k), (e, k)] for s, e in zip(d["start"], d["end"])]
        ax.add_collection(LineCollection(segs, colors=pal[nm], linewidths=2 * _MM))
    ax.set_xlim(start, end)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_yticks([] if len(names) == 1 else range(len(names)))
    if len(names) > 1:
        ax.set_yticklabels(names, fontsize=6)
    ax.set_xticks([])
    _draw_highlight(ax, highlight, region)
    _strip(ax, "Peaks")


def _draw_loops(ax, ldata, vmin, vmax, region, highlight):
    chrom, start, end = region
    cmap = mpl.colors.LinearSegmentedColormap.from_list("archr_loops", _color_ramp(_LOOP_COLORS, 100))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    if len(ldata):
        for _, d in ldata.groupby("id", sort=False):
            ax.plot(d["x"].to_numpy(), d["y"].to_numpy(), color=cmap(norm(d["value"].iloc[0])), lw=0.5 * _MM)
    ax.set_xlim(start, end)
    ax.set_ylim(-100, 0)
    ax.set_xticks([]); ax.set_yticks([])
    _draw_highlight(ax, highlight, region)
    names = list(dict.fromkeys(ldata["name"])) if len(ldata) else []
    _strip(ax, names[0] if len(names) == 1 else "Loops")
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


def _draw_genes(ax, gdata, exons, region, highlight):
    chrom, start, end = region
    span = end - start
    if len(gdata):
        for _, r in gdata.iterrows():
            col = _GENE_STRAND_COLORS.get(str(r["strand"]), "#FF0000")
            ax.plot([r["start"], r["end"]], [r["cluster"], r["cluster"]], color=col, lw=2 * _MM,
                    solid_capstyle="butt")
            if exons is not None and len(exons):
                ex = exons[exons["symbol"] == r["symbol"]]
                for s, e in zip(ex["start"], ex["end"]):
                    ax.plot([s, e], [r["cluster"], r["cluster"]], color=col, lw=4 * _MM,
                            solid_capstyle="butt")
            minus = str(r["strand"]) == "-"
            xl = r["end"] + 0.01 * span if minus else r["start"] - 0.01 * span
            yl = r["cluster"] + (0.25 if minus else -0.25)
            ax.text(min(max(xl, start), end), yl, str(r["symbol"]), fontsize=5, color=col,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=0.4))
        ax.set_ylim(0.5, gdata["cluster"].max() + 0.5)
    ax.set_xlim(start, end)
    ax.set_xticks([]); ax.set_yticks([])
    _draw_highlight(ax, highlight, region)
    _strip(ax, "Genes")


def plot_browser_track(
    adata: AnnData,
    region=None,
    *,
    gene: Optional[str] = None,
    fragments: Union[str, Mapping[str, str]],
    group_by: str = "Clusters",
    sample_key: Optional[str] = "Sample",
    barcode_sep: Optional[str] = "#",
    use_groups: Optional[Sequence[str]] = None,
    plot_summary: Sequence[str] = ("bulkTrack", "featureTrack", "loopTrack", "geneTrack"),
    sizes: Sequence[float] = (10, 1.5, 3, 4),
    features=None,
    loops=None,
    gene_annotation: Optional[pd.DataFrame] = None,
    exon_annotation: Optional[pd.DataFrame] = None,
    upstream: int = 50_000,
    downstream: int = 50_000,
    tile_size: int = 250,
    max_cells: int = 500,
    min_cells: int = 25,
    norm_method: str = "ReadsInTSS",
    highlight: Optional[pd.DataFrame] = None,
    ylim: Optional[Tuple[float, float]] = None,
    palette: Optional[Mapping[str, str]] = None,
    cells: Optional[Sequence[str]] = None,
    seed: Optional[int] = 1,
    rng: str = "archr",
    count_ends: bool = False,
    title: str = "",
    figsize: Tuple[float, float] = (6.0, 7.0),
    show: bool = True,
    return_data: bool = False,
):
    """ArchR-style grouped genome browser track (``plotBrowserTrack``).

    Draws, top to bottom: one normalised coverage track per group, a peak
    track, a loop (co-accessibility / peak-to-gene) track and a gene track,
    with ArchR's default panel heights ``sizes=(10, 1.5, 3, 4)``.

    Arguments:
        adata: cells in ``obs_names`` with ``group_by``, ``sample_key`` and
            ``norm_method`` columns in ``obs`` (e.g. the peak-matrix AnnData).
        region: ``"chr:start-end"`` or ``(chrom, start, end)``, 1-based closed.
        gene: alternatively, a gene symbol; the window is the TSS
            ``- upstream`` / ``+ downstream`` (needs ``gene_annotation``).
        fragments: tabix-indexed fragment file or ``{sample: path}``.
        group_by: ``obs`` column defining the coverage tracks.
        sample_key: ``obs`` column mapping cells to fragment files.
        barcode_sep: separator in ``obs_names`` between sample and barcode;
            ``None`` if ``obs_names`` are raw fragment barcodes.
        use_groups: subset (and order) of groups to draw.
        plot_summary: any of ``"bulkTrack"``, ``"featureTrack"``,
            ``"loopTrack"``, ``"geneTrack"``.
        sizes: relative heights of the tracks, in ``plot_summary`` order.
        features: peaks as a DataFrame (``chrom/start/end``) or
            ``{name: DataFrame}`` for several feature rows.
        loops: DataFrame with ``chrom/start/end/value`` (or
            :func:`loops_from_coaccessibility` output), or ``{name: DataFrame}``.
        gene_annotation: genes with ``chrom/start/end/strand/symbol``.
        exon_annotation: exons with ``chrom/start/end/symbol``.
        upstream, downstream: window around the TSS when ``gene`` is given.
        tile_size: coverage tile width in bp.
        max_cells, min_cells: per-group subsampling cap / minimum group size.
        norm_method: ``"ReadsInTSS"`` (default), ``"ReadsInPromoter"``,
            ``"nFrags"``, ``"nCells"`` or ``"None"``.
        highlight: intervals to shade.
        ylim: quantiles of the signal used as y-limits (ArchR semantics);
            default ``[0, q(0.999)]``.
        palette: ``{group: colour}``; default ArchR ``paletteDiscrete``.
        cells: explicit cells to use instead of random subsampling.
        seed: seed for the subsampling.
        rng: ``"archr"`` reproduces ArchR's draw after ``set.seed(seed)``
            (L'Ecuyer-CMRG); ``"R"`` for Mersenne-Twister; ``"numpy"``.
        count_ends: also count fragment end tiles (ArchR 1.0.3 counts starts only).
        title: appended to the region in the title.
        figsize: figure size.
        show: call ``plt.show()``.
        return_data: also return the plotted tables.

    Returns:
        The figure, or ``(fig, data)`` with ``data`` holding ``region``,
        ``coverage`` (raw), ``bulk`` (clipped), ``ylim``, ``features``,
        ``loops`` and ``genes``.
    """
    if region is None:
        if gene is None or gene_annotation is None:
            raise ValueError("pass `region`, or `gene` together with `gene_annotation`")
        region = gene_region(gene_annotation, gene, upstream, downstream)
    region = _parse_region(region)
    chrom, start, end = region

    summary = [s.lower() for s in plot_summary]
    size_map = dict(zip(summary, sizes))
    data = {"region": region}
    panels = []

    if "bulktrack" in summary:
        cov = browser_track_coverage(
            adata, region, fragments=fragments, group_by=group_by, sample_key=sample_key,
            barcode_sep=barcode_sep, use_groups=use_groups, tile_size=tile_size,
            norm_method=norm_method, min_cells=min_cells, max_cells=max_cells,
            cells=cells, seed=seed, rng=rng, count_ends=count_ends,
        )
        bulk, lim = clip_coverage(cov, ylim)
        data.update(coverage=cov, bulk=bulk, ylim=lim)
        groups = list(use_groups) if use_groups is not None else cov.attrs["groups"]
        if palette is None:
            palette = archr_palette_discrete(adata.obs[group_by].dropna().astype(str).unique())
        panels.append("bulktrack")
    if "featuretrack" in summary and features is not None:
        data["features"] = feature_track_data(features, region)
        panels.append("featuretrack")
    if "looptrack" in summary and loops is not None:
        ldict = {"Loops": loops} if isinstance(loops, pd.DataFrame) else dict(loops)
        allv = np.concatenate([_standardise_intervals(d)["value"].to_numpy(np.float64) for d in ldict.values()])
        data["loops"] = loop_track_data(ldict, region)
        data["loop_limits"] = (float(allv.min()), float(allv.max()))
        panels.append("looptrack")
    if "genetrack" in summary and gene_annotation is not None:
        data["genes"] = gene_track_data(gene_annotation, region)
        panels.append("genetrack")

    exons = None
    if exon_annotation is not None:
        exons = _overlaps(_standardise_intervals(exon_annotation), chrom, start, end)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(len(panels), 1, height_ratios=[size_map[p] for p in panels],
                          hspace=0.16, left=0.1, right=0.8, top=0.95, bottom=0.04)
    mappable = None
    for k, p in enumerate(panels):
        if p == "bulktrack":
            _draw_bulk(fig, gs[k], data["bulk"], groups, palette, region, data["ylim"],
                       norm_method, title, highlight)
        elif p == "featuretrack":
            _draw_features(fig.add_subplot(gs[k]), data["features"], region, highlight)
        elif p == "looptrack":
            ax = fig.add_subplot(gs[k])
            mappable = _draw_loops(ax, data["loops"], *data["loop_limits"], region, highlight)
            cax = ax.inset_axes([1.24, 0.1, 0.03, 0.8])
            cb = fig.colorbar(mappable, cax=cax)
            cb.ax.tick_params(labelsize=5)
            cb.ax.set_title("value", fontsize=6)
        elif p == "genetrack":
            _draw_genes(fig.add_subplot(gs[k]), data["genes"], exons, region, highlight)
    if show:
        plt.show()
    return (fig, data) if return_data else fig
