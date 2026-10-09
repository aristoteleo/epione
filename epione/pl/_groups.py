"""Per-group distribution plots (port of ArchR ``plotGroups`` / ``ggGroup``;
ArchR is MIT-licensed, Granja et al. 2021).

The statistics behind the figure (sampled cells, quantile clipping, group
order, ggridges / ggplot2 densities and box statistics) follow the R code
paths of ArchR 1.0.3, ggridges 0.5.7, ggplot2 4.0 and ``stats::density`` in
R >= 4.4 so they can be compared number-for-number with ArchR.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

from ..core._r_random import RRandom

__all__ = ["plot_groups", "group_plot_data", "plot_confusion_matrix", "archr_palette_discrete"]

# ArchR "stallion" palette keyed by its names; paletteDiscrete() orders by mixedsort(names).
_STALLION = {
    "1": "#D51F26", "2": "#272E6A", "3": "#208A42", "4": "#89288F", "5": "#F47D2B",
    "6": "#FEE500", "7": "#8A9FD1", "8": "#C06CAB", "19": "#E6C2DC", "10": "#90D5E4",
    "11": "#89C75F", "12": "#F37B7D", "13": "#9983BD", "14": "#D24B27", "15": "#3BBCA8",
    "16": "#6E4B9E", "17": "#0C727C", "18": "#7E1416", "9": "#D8A767", "20": "#3D3D3D",
}
_WHITE_BLUE = ["#fff7fb", "#ece7f2", "#d0d1e6", "#a6bddb", "#74a9cf", "#3690c0", "#0570b0", "#045a8d", "#023858"]

_NUM = re.compile(r"(\d+(?:\.\d+)?)")


def _mixed_key(s: str):
    parts = _NUM.split(str(s))
    key = []
    for k, p in enumerate(parts):
        if p == "":
            continue
        key.append((0, float(p), "") if k % 2 else (1, 0.0, p))
    return key


def _mixedsort(values) -> list:
    """gtools::mixedsort for alphanumeric labels (C1, C2, ..., C10)."""
    return sorted(values, key=_mixed_key)


def _color_ramp(cols: Sequence[str], n: int) -> List[str]:
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in cols], dtype=float)
    pos = np.linspace(0, 1, len(cols))
    t = np.linspace(0, 1, n)
    out = np.column_stack([np.interp(t, pos, rgb[:, k]) for k in range(3)])
    return ["#%02X%02X%02X" % tuple(int(v + 0.5) for v in row) for row in out]


def archr_palette_discrete(values, set: str = "stallion") -> Dict[str, str]:
    """ArchR ``paletteDiscrete``: colours assigned to ``mixedsort(unique(values))``."""
    if set != "stallion":
        raise ValueError("only the 'stallion' set is bundled")
    vals = _mixedsort(pd.unique(pd.Series(values).astype(str)))
    names = _mixedsort(list(_STALLION))
    ordered = [_STALLION[k] for k in names]
    cols = ordered[: len(vals)] if len(vals) <= len(ordered) else _color_ramp(list(_STALLION.values()), len(vals))
    return dict(zip(vals, cols))


# ----------------------------------------------------------------------------- R numerics

def _r_quantile7(x: np.ndarray, probs) -> np.ndarray:
    x = np.sort(np.asarray(x, dtype=np.float64))
    probs = np.atleast_1d(np.asarray(probs, dtype=np.float64))
    n = len(x)
    index = 1 + max(n - 1, 0) * probs
    lo = np.floor(index).astype(int)
    hi = np.ceil(index).astype(int)
    qs = x[lo - 1].copy()
    i = np.where((index > lo) & (x[hi - 1] != qs))[0]
    h = (index - lo)[i]
    qs[i] = (1 - h) * qs[i] + h * x[hi[i] - 1]
    return qs


def _bw_nrd0(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        raise ValueError("need at least 2 data points")
    hi = float(np.std(x, ddof=1))
    q = _r_quantile7(x, [0.25, 0.75])
    lo = min(hi, (q[1] - q[0]) / 1.34)
    if not lo:
        lo = hi or abs(x[0]) or 1.0
    return 0.9 * lo * len(x) ** (-0.2)


def _r_seq(start: float, stop: float, n: int) -> np.ndarray:
    out = np.empty(n)
    if n > 0:
        out[0] = start
    if n > 1:
        out[-1] = stop
    if n > 2:
        by = (stop - start) / (n - 1)
        i = np.arange(1, n - 1)
        out[1:-1] = np.where(i < n // 2, start + i * by, stop - (n - 1 - i) * by)
    return out


def _r_density(x: np.ndarray, bw: float, lo_x: float, hi_x: float, n: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """``stats::density(x, bw=bw, from=lo_x, to=hi_x, n=n)``, gaussian kernel, R >= 4.4."""
    x = np.asarray(x, dtype=np.float64)
    nx = len(x)
    n_user = n
    n = max(n, 512)
    if n > 512:
        n = 2 ** int(math.ceil(math.log2(n)))
    lo = lo_x - 4 * bw
    up = hi_x + 4 * bw
    y = np.zeros(2 * n)
    xdelta = (up - lo) / (n - 1)
    w = 1.0 / nx
    xpos = (x - lo) / xdelta
    ix = np.floor(xpos).astype(np.int64)
    fx = xpos - ix
    inside = (ix >= 0) & (ix <= n - 2)
    np.add.at(y, ix[inside], w * (1 - fx[inside]))
    np.add.at(y, ix[inside] + 1, w * fx[inside])
    left = ix == -1
    np.add.at(y, np.zeros(left.sum(), dtype=np.int64), w * fx[left])
    right = ix == n - 1
    np.add.at(y, ix[right], w * (1 - fx[right]))
    kords = _r_seq(0.0, (2 * n - 1) / (n - 1) * (up - lo), 2 * n)
    kords[n + 1:] = -kords[n - 1:0:-1]
    kords = np.exp(-0.5 * (kords / bw) ** 2) / (bw * math.sqrt(2 * math.pi))
    conv = np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(kords))).real[:n]
    conv = np.maximum(0.0, conv)
    xords = _r_seq(lo, up, n)
    xs = _r_seq(lo_x, hi_x, n_user)
    return xs, np.interp(xs, xords, conv)


def _quantile_cut(x: np.ndarray, lo: float, hi: float, max_if0: bool = True) -> np.ndarray:
    q = _r_quantile7(x, [lo, hi])
    if q[1] == 0 and max_if0:
        q[1] = np.max(x)
    return np.clip(x, q[0], q[1]) if q[0] <= q[1] else x


# ----------------------------------------------------------------------------- data layer

def _feature_values(adata, name: str, layer: Optional[str]) -> np.ndarray:
    lower = pd.Index(pd.Index(adata.var_names).astype(str).str.lower())
    hit = np.where(lower == name.lower())[0]
    if len(hit) == 0:
        raise KeyError(f"FeatureName ({name}) does not exist!")
    X = adata.layers[layer] if layer is not None else adata.X
    col = X[:, hit[0]]
    col = col.toarray() if sparse.issparse(col) else np.asarray(col)
    return col.ravel().astype(np.float64)


def _values(adata, name: str, color_by: Optional[str], layer: Optional[str], impute_weights,
            log2_norm: Optional[bool]) -> np.ndarray:
    use_obs = color_by == "obs" or (color_by is None and name in adata.obs.columns)
    if use_obs:
        v = np.asarray(adata.obs[name], dtype=np.float64)
    else:
        v = _feature_values(adata, name, layer)
        if impute_weights is not None:
            ws = impute_weights if isinstance(impute_weights, (list, tuple)) else [impute_weights]
            v = sum(np.asarray(w @ v).ravel() for w in ws) / len(ws)
    if log2_norm:
        v = np.log2(v + 1)
    return v


def group_plot_data(
    adata,
    group_by: str = "Sample",
    name: str = "TSSEnrichment",
    *,
    color_by: Optional[str] = None,
    layer: Optional[str] = None,
    impute_weights=None,
    max_cells: Optional[int] = 1000,
    quant_cut: Optional[Tuple[float, float]] = (0.002, 0.998),
    log2_norm: Optional[bool] = None,
    plot_as: str = "ridges",
    ridge_scale: float = 2.0,
    seed: Optional[int] = 1,
    rng: Optional[RRandom] = None,
) -> dict:
    """Numbers behind :func:`plot_groups` (sampled cells, plotted values and the
    density / box statistics ggplot2 would draw). See :func:`plot_groups`."""
    groups = pd.Series(np.asarray(adata.obs[group_by]).astype(str), index=adata.obs_names)
    values = _values(adata, name, color_by, layer, impute_weights, log2_norm)
    if rng is None:
        rng = RRandom(seed if seed is not None else np.random.SeedSequence().entropy % (2**31))
    if max_cells is not None:
        idx = []
        for g in sorted(pd.unique(groups.values)):
            members = np.where(groups.values == g)[0]
            if len(members) > max_cells:
                members = rng.sample(members, max_cells)
            idx.extend(members.tolist())
        idx = np.asarray(idx, dtype=np.int64)
    else:
        idx = np.arange(len(groups))
    y = values[idx]
    if quant_cut is not None:
        y = _quantile_cut(y, min(quant_cut), max(quant_cut))
    x = groups.values[idx]
    uniq = _mixedsort(pd.unique(x))
    ridges = plot_as.lower() in ("ridges", "ggridges")
    levels = uniq[::-1] if ridges else uniq
    pos = {g: k + 1 for k, g in enumerate(levels)}
    out = dict(cells=adata.obs_names.values[idx], x=x, y=y, levels=levels,
               palette=archr_palette_discrete(groups.values), name=name, group_by=group_by,
               plot_as="ridges" if ridges else "violin")
    if ridges:
        out["ridges"] = _ridge_stats(x, y, pos, ridge_scale)
    else:
        out["violin"], out["box"] = _violin_stats(x, y, pos)
    return out


def _ridge_stats(x, y, pos, scale) -> pd.DataFrame:
    by_group = {g: y[x == g] for g in sorted(pos, key=pos.get)}
    bws = [_bw_nrd0(v) for v in by_group.values() if len(v) > 1]
    bw = float(np.mean(bws))
    lo, hi = np.nanmin(y) - 3 * bw, np.nanmax(y) + 3 * bw
    frames = []
    for g, v in by_group.items():
        if len(v) < 3:
            continue
        xs, d = _r_density(v, bw, lo, hi)
        frames.append(pd.DataFrame(dict(group=g, y=pos[g], x=xs, density=d, ndensity=d / d.max())))
    df = pd.concat(frames, ignore_index=True)
    uy = np.unique(df["y"])
    hmax = df["density"].max()
    iscale = 1.0 if len(uy) < 2 else (uy.max() - uy.min()) / ((len(uy) - 1) * hmax)
    df["height"] = df["density"]
    df["ymin"] = df["y"].astype(float)
    df["ymax"] = df["y"] + iscale * scale * df["height"]
    df.attrs["bandwidth"] = bw
    return df


def _violin_stats(x, y, pos) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames, boxes = [], []
    for g in sorted(pos, key=pos.get):
        v = y[x == g]
        if len(v) < 2:
            continue
        bw = _bw_nrd0(v)
        grid, d = _r_density(v, bw, v.min(), v.max())
        dens = pd.DataFrame(dict(y=grid, density=d, scaled=d / d.max(), count=d * len(v)))
        q = _r_quantile7(v, [0.25, 0.5, 0.75])
        quant = pd.DataFrame(dict(y=q, quantile=[0.25, 0.5, 0.75]))
        for col in ("density", "scaled", "count"):
            quant[col] = np.interp(q, dens["y"].values, dens[col].values)
        dens = dens[~dens["y"].isin(q)]
        dens["quantile"] = np.nan
        dg = pd.concat([dens, quant], ignore_index=True)
        dg["group"] = g
        dg["x"] = pos[g]
        frames.append(dg)
        st = _r_quantile7(v, [0, 0.25, 0.5, 0.75, 1])
        iqr = st[3] - st[1]
        outl = (v < st[1] - 1.5 * iqr) | (v > st[3] + 1.5 * iqr)
        if outl.any():
            rest = np.concatenate([st[1:4], v[~outl]])
            st[0], st[4] = rest.min(), rest.max()
        boxes.append(dict(group=g, x=pos[g], ymin=st[0], lower=st[1], middle=st[2], upper=st[3],
                          ymax=st[4], n=len(v)))
    df = pd.concat(frames, ignore_index=True)
    df["violinwidth"] = df["density"] / df["density"].max()
    return df, pd.DataFrame(boxes)


# ----------------------------------------------------------------------------- drawing

def _draw(d: dict, ax, base_size: float, size: float, alpha: float, palette: Optional[Dict[str, str]]):
    pal = dict(d["palette"])
    if palette is not None:
        pal.update(palette)
    levels = d["levels"]
    if d["plot_as"] == "ridges":
        r = d["ridges"]
        for g in sorted(r["group"].unique(), key=lambda g: -levels.index(g) - 1):
            sub = r[r["group"] == g]
            ax.fill_between(sub["x"], sub["ymin"], sub["ymax"], color=pal[g], alpha=alpha,
                            linewidth=0, zorder=len(levels) - levels.index(g))
            ax.plot(sub["x"], sub["ymax"], color="black", linewidth=0.5,
                    zorder=len(levels) - levels.index(g) + 0.5)
        ax.set_yticks(range(1, len(levels) + 1))
        ax.set_yticklabels(levels)
        top = float(r["ymax"].max())
        ax.set_ylim(1 - 0.01 * (top - 1), top + 0.03 * (top - 1))
        ax.set_xlabel(d["name"], fontsize=base_size + 2)
        ax.set_ylabel(d["group_by"], fontsize=base_size + 2)
    else:
        v, b = d["violin"], d["box"]
        for g in levels:
            sub = v[v["group"] == g].sort_values("y")
            if sub.empty:
                continue
            xc = sub["x"].iloc[0]
            half = sub["violinwidth"].values * 0.9 / 2
            ax.fill_betweenx(sub["y"], xc - half, xc + half, color=pal[g], alpha=alpha, linewidth=0.5,
                             edgecolor=pal[g])
        for _, row in b.iterrows():
            c = pal[row["group"]]
            x0, x1 = row["x"] - 0.375, row["x"] + 0.375
            ax.add_patch(__import__("matplotlib").patches.Rectangle(
                (x0, row["lower"]), x1 - x0, row["upper"] - row["lower"], fill=False, edgecolor=c,
                linewidth=size))
            ax.plot([x0, x1], [row["middle"]] * 2, color=c, linewidth=size * 2)
            ax.plot([row["x"]] * 2, [row["ymin"], row["lower"]], color=c, linewidth=size)
            ax.plot([row["x"]] * 2, [row["upper"], row["ymax"]], color=c, linewidth=size)
        ax.set_xticks(range(1, len(levels) + 1))
        ax.set_xticklabels(levels, rotation=90)
        ax.set_xlim(0.4, len(levels) + 0.6)
        ax.set_xlabel(d["group_by"], fontsize=base_size + 2)
        ax.set_ylabel(d["name"], fontsize=base_size + 2)
    ax.tick_params(labelsize=base_size + 1)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_color("black")
    ax.grid(False)


def plot_groups(
    adata,
    group_by: str = "Sample",
    name: Union[str, Sequence[str]] = "TSSEnrichment",
    *,
    color_by: Optional[str] = None,
    layer: Optional[str] = None,
    impute_weights=None,
    max_cells: Optional[int] = 1000,
    quant_cut: Optional[Tuple[float, float]] = (0.002, 0.998),
    log2_norm: Optional[bool] = None,
    palette: Optional[Dict[str, str]] = None,
    plot_as: str = "ridges",
    ridge_scale: float = 2.0,
    base_size: float = 6,
    size: float = 0.5,
    alpha: float = 1.0,
    seed: Optional[int] = 1,
    figsize: Optional[Tuple[float, float]] = None,
    ax=None,
    return_data: bool = False,
    show: Optional[bool] = None,
):
    r"""Distribution of a per-cell value across groups (ArchR ``plotGroups``).

    Draws ridge plots (default) or violin + box plots of an ``obs`` column
    (e.g. ``TSSEnrichment`` or a module score) or of a feature from ``X`` /
    ``layer`` (e.g. a gene score), one distribution per ``group_by`` level.

    As in ArchR, each group is down-sampled to ``max_cells`` cells (sampling
    emulates R, so ``seed=s`` equals ``set.seed(s)`` before ``plotGroups``),
    values are clipped to the ``quant_cut`` quantiles of the sampled cells, ridge
    groups are ordered by reverse mixed sort (violins by mixed sort), and colours
    follow ArchR's ``paletteDiscrete(set="stallion")``.

    Arguments:
        adata: AnnData.
        group_by: ``obs`` column with the groups.
        name: ``obs`` column or feature name (case-insensitive), or a list of them.
        color_by: ``"obs"`` or ``"matrix"`` to force where ``name`` is looked up.
        layer: matrix layer for feature values.
        impute_weights: optional cells x cells weight matrix (or list of them,
            averaged) applied to feature values, like ArchR ``imputeMatrix``.
        max_cells: maximum cells per group (``None`` keeps all).
        quant_cut: lower / upper quantile used to clip values.
        log2_norm: apply ``log2(x + 1)``. ArchR 1.0.3 resolves but never applies
            this argument in ``plotGroups``; ``None`` (default) matches it.
        palette: ``{group: colour}`` overrides.
        plot_as: ``"ridges"`` or ``"violin"``.
        ridge_scale: ridge height scaling (ggridges ``scale``).
        base_size, size, alpha: text size, box line width, fill alpha.
        seed: R-compatible seed for cell sampling.
        figsize, ax: matplotlib figure size or an existing axes (single ``name``).
        return_data: also return the data dict(s) from :func:`group_plot_data`.
        show: call ``plt.show()``.

    Returns:
        A figure (or ``{name: figure}`` for several names); with
        ``return_data=True`` a ``(figure(s), data)`` tuple.
    """
    import matplotlib.pyplot as plt

    names = [name] if isinstance(name, str) else list(name)
    rng = RRandom(seed if seed is not None else np.random.SeedSequence().entropy % (2**31))
    # ArchR samples cells once and reuses them for every name.
    state = rng._bitgen.state
    figs, datas = {}, {}
    for nm in names:
        rng._bitgen.state = state
        d = group_plot_data(adata, group_by, nm, color_by=color_by, layer=layer,
                            impute_weights=impute_weights, max_cells=max_cells, quant_cut=quant_cut,
                            log2_norm=log2_norm, plot_as=plot_as, ridge_scale=ridge_scale, rng=rng)
        if ax is not None and len(names) == 1:
            axx, fig = ax, ax.figure
        else:
            fs = figsize or ((3.2, 0.35 * len(d["levels"]) + 1.2) if d["plot_as"] == "ridges"
                             else (0.35 * len(d["levels"]) + 1.4, 3.0))
            fig, axx = plt.subplots(figsize=fs)
        _draw(d, axx, base_size, size, alpha, palette)
        fig.tight_layout()
        figs[nm], datas[nm] = fig, d
    if show:
        plt.show()
    res = figs[names[0]] if isinstance(name, str) else figs
    if return_data:
        return res, (datas[names[0]] if isinstance(name, str) else datas)
    return res


def plot_confusion_matrix(
    cm: pd.DataFrame,
    *,
    normalize: Optional[str] = "row",
    cmap=None,
    annotate: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    ax=None,
):
    """Heatmap of a :func:`epione.tl.confusion_matrix` result.

    ``normalize="row"`` divides by row sums (ArchR tutorial ``cM / rowSums(cM)``),
    ``"col"`` by column sums, ``None`` shows raw counts. The default colormap is
    ArchR's ``whiteBlue``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    m = cm.astype(float)
    if normalize == "row":
        m = m.div(m.sum(axis=1).replace(0, np.nan), axis=0)
    elif normalize in ("col", "column"):
        m = m.div(m.sum(axis=0).replace(0, np.nan), axis=1)
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("whiteBlue", _WHITE_BLUE)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (0.45 * m.shape[1] + 2.0, 0.3 * m.shape[0] + 1.2))
    else:
        fig = ax.figure
    im = ax.imshow(m.values, aspect="auto", cmap=cmap, vmin=0)
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels([str(c) for c in m.columns], rotation=90)
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels([str(r) for r in m.index])
    if annotate:
        for (r, c), val in np.ndenumerate(cm.values):
            ax.text(c, r, f"{val}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.05, label="fraction" if normalize else "cells")
    return fig
