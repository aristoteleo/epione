"""
ArchR-style trajectory plots: :func:`plot_trajectory` (``plotTrajectory``)
and :func:`plot_trajectory_heatmap` (``plotTrajectoryHeatmap``).
Ported from ArchR 1.0.3 (MIT licence).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from anndata import AnnData

from ._peak2gene import ARCHR_STALLION
from ..tl._trajectory import (
    _center_roll_mean,
    _get_quantiles,
    _smooth_spline,
    _traj_matrix,
    trajectory_heatmap_matrix,
)

ARCHR_PALETTES = {
    "horizonExtra": ["#000436", "#021EA9", "#1632FB", "#6E34FC", "#C732D5",
                     "#FD619D", "#FF9965", "#FFD32B", "#FFFC5A"],
    "solarExtra": ["#3361A5", "#248AF3", "#14B3FF", "#88CEEF", "#C1D5DC",
                   "#EAD397", "#FDB31A", "#E42A2A", "#A31D1D"],
    "blueYellow": ["#352A86", "#343DAE", "#0262E0", "#1389D2", "#2DB7A3",
                   "#A5BE6A", "#F8BA43", "#F6DA23", "#F8FA0D"],
}


def _archr_cmap(name: str) -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list(name, ARCHR_PALETTES[name])


def _quantile_cut(x: np.ndarray, lo: float, hi: float, max_if0: bool = True) -> np.ndarray:
    """ArchR ``.quantileCut``."""
    x = np.asarray(x, dtype=float).copy()
    q = np.quantile(x, [lo, hi])
    if q[1] == 0 and max_if0:
        q[1] = np.max(x)
    x[x < q[0]] = q[0]
    x[x > q[1]] = q[1]
    return x


def _impute(values: np.ndarray, impute_weights) -> np.ndarray:
    """ArchR ``imputeMatrix`` for one feature: mean over reps of ``v %*% W``."""
    ws = impute_weights if isinstance(impute_weights, (list, tuple)) else [impute_weights]
    acc = np.zeros_like(values, dtype=float)
    for w in ws:
        acc += np.asarray(w.T @ values if sp.issparse(w) else np.asarray(w).T @ values).ravel()
    return acc / len(ws)


def _trajectory_frames(
    adata: AnnData,
    trajectory_key: str = "Trajectory",
    color_by: Optional[str] = None,
    basis: str = "X_umap",
    layer: Optional[str] = None,
    log2_norm: Optional[bool] = None,
    impute_weights=None,
    quant_cut: Optional[Tuple[float, float]] = (0.01, 0.99),
    smooth_window: int = 5,
):
    """Data behind ArchR ``plotTrajectory``.

    Returns ``(emb, values, df_points, df_arrow)`` where ``df_points`` is the
    pseudo-time × value frame of panel 2 (sorted by pseudo-time) and
    ``df_arrow`` the smoothed arrow path drawn on the embedding.
    """
    name = trajectory_key if color_by is None else color_by
    emb = np.asarray(adata.obsm[basis])[:, :2]
    pt = adata.obs[trajectory_key].values.astype(float)
    if quant_cut is None:
        quant_cut = (0.0, 1.0)

    if name in adata.obs.columns:
        values = adata.obs[name].values
        discrete = not pd.api.types.is_numeric_dtype(adata.obs[name])
        log2_norm_eff = log2_norm
    else:
        if name not in adata.var_names:
            raise KeyError(f"{name!r} is neither an obs column nor a feature")
        X = adata.layers[layer] if layer is not None else adata.X
        col = X[:, adata.var_names.get_loc(name)]
        values = np.asarray(col.toarray() if sp.issparse(col) else col, dtype=float).ravel()
        discrete = False
        log2_norm_eff = True if log2_norm is None else log2_norm

    if discrete:
        values = values.astype(str)
    else:
        values = np.asarray(values, dtype=float)
        if impute_weights is not None:
            values = _impute(values, impute_weights)
        if name != trajectory_key:
            values = _quantile_cut(values, min(quant_cut), max(quant_cut))
        if log2_norm_eff:
            values = np.log2(values + 1)
        values = values.copy()
        values[np.isnan(pt)] = np.nan

    df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "PseudoTime": pt,
                       "value": values if not discrete else np.nan})
    df = df.iloc[np.argsort(df["PseudoTime"].values, kind="stable")]
    df = df[~df["PseudoTime"].isna()]

    key = np.floor(df["PseudoTime"].values / 1.01)
    arrow = df.groupby(key, sort=True)[["x", "y", "PseudoTime", "value"]].mean()
    arrow = arrow.reset_index(drop=True)
    arrow["x"] = _center_roll_mean(arrow["x"].values, smooth_window)
    arrow["y"] = _center_roll_mean(arrow["y"].values, smooth_window)
    arrow = pd.concat([arrow, df.iloc[[-1]][["x", "y", "PseudoTime", "value"]]], ignore_index=True)
    points = pd.DataFrame({"x": df["PseudoTime"].values, "y": df["value"].values})
    return emb, values, points, arrow


def plot_trajectory(
    adata: AnnData,
    trajectory_key: str = "Trajectory",
    color_by: Optional[str] = None,
    basis: str = "X_umap",
    layer: Optional[str] = None,
    log2_norm: Optional[bool] = None,
    impute_weights=None,
    quant_cut: Optional[Tuple[float, float]] = (0.01, 0.99),
    smooth_window: int = 5,
    add_arrow: bool = True,
    size: float = 3,
    cmap: Optional[Union[str, mpl.colors.Colormap]] = None,
    figsize: Tuple[float, float] = (9, 4),
    show: bool = True,
):
    """Embedding + pseudo-time scatter for a trajectory (ArchR ``plotTrajectory``).

    Left panel: embedding coloured by pseudo-time (default) or by a feature
    (``color_by``), with the inferred trajectory arrow. Right panel: value
    vs pseudo-time with a smoothing-spline trend.

    Arguments:
        adata: AnnData with ``obs[trajectory_key]`` and ``obsm[basis]``.
        trajectory_key: pseudo-time column (from :func:`epione.tl.add_trajectory`).
        color_by: ``obs`` column or feature name; default the pseudo-time.
        basis: 2-D embedding in ``adata.obsm``.
        layer: matrix layer used for feature values.
        log2_norm: ``log2(x + 1)`` feature values. Default ``True`` for
            features and ``False`` for ``obs`` columns (as ArchR does for
            gene scores / cell metadata).
        impute_weights: optional cells × cells weight matrix (or list of
            them, averaged) used to smooth feature values (ArchR MAGIC).
        quant_cut: quantile clipping of feature values.
        smooth_window: rolling-mean window applied to the arrow path.
        add_arrow: draw the arrow path.
        size: point size.
        cmap: colormap (default ArchR ``horizonExtra`` for metadata,
            ``solarExtra`` for features).
        figsize: figure size.
        show: call ``plt.show()``.

    Returns:
        ``(fig, axes)``.
    """
    emb, values, points, arrow = _trajectory_frames(
        adata, trajectory_key, color_by, basis, layer, log2_norm,
        impute_weights, quant_cut, smooth_window,
    )
    name = trajectory_key if color_by is None else color_by
    if cmap is None:
        cmap = _archr_cmap("horizonExtra" if name in adata.obs.columns else "solarExtra")
    elif isinstance(cmap, str) and cmap in ARCHR_PALETTES:
        cmap = _archr_cmap(cmap)

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    vals = np.asarray(values, dtype=float)
    na = np.isnan(vals)
    ax.scatter(emb[na, 0], emb[na, 1], s=size, c="lightgrey", linewidths=0, rasterized=True)
    order = np.where(~na)[0]
    sc = ax.scatter(emb[order, 0], emb[order, 1], s=size, c=vals[order], cmap=cmap,
                    linewidths=0, rasterized=True)
    if add_arrow:
        ax.plot(arrow["x"], arrow["y"], color="black", lw=1.5)
        ax.annotate("", xy=(arrow["x"].iloc[-1], arrow["y"].iloc[-1]),
                    xytext=(arrow["x"].iloc[-2], arrow["y"].iloc[-2]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{basis.replace('X_', '').upper()}1"); ax.set_ylabel(f"{basis.replace('X_', '').upper()}2")
    ax.set_title(f"{basis.replace('X_', '').upper()} colored by {name}", fontsize=9)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

    ax = axes[1]
    ok = ~points["y"].isna().values
    ax.scatter(points["x"][ok], points["y"][ok], c=points["x"][ok], cmap=cmap, s=size,
               linewidths=0, rasterized=True)
    if ok.sum() > 10:
        ux, fit = _smooth_spline(points["x"][ok].values, points["y"][ok].values, spar=0.8)
        ax.plot(ux, fit, color="black", lw=1.2)
    ax.set_xlabel("PseudoTime"); ax.set_ylabel(name)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_trajectory_heatmap(
    traj: AnnData,
    var_cutoff: Optional[float] = 0.9,
    max_features: Optional[int] = 25000,
    scale_rows: bool = True,
    limits: Sequence[float] = (-1.5, 1.5),
    grep_exclude: Optional[str] = None,
    cmap: Optional[Union[str, mpl.colors.Colormap]] = None,
    color_columns: bool = False,
    label_markers: Optional[Sequence[str]] = None,
    label_top: Optional[int] = 50,
    label_rows: bool = False,
    row_order: Optional[Sequence] = None,
    figsize: Tuple[float, float] = (5, 7),
    title: Optional[str] = None,
    show: bool = True,
):
    """Heatmap of features along a trajectory (ArchR ``plotTrajectoryHeatmap``).

    The plotted matrix is :func:`epione.tl.trajectory_heatmap_matrix`.

    Arguments:
        traj: output of :func:`epione.tl.get_trajectory`.
        var_cutoff, max_features, scale_rows, limits, grep_exclude, row_order:
            see :func:`epione.tl.trajectory_heatmap_matrix`.
        cmap: colormap; default ArchR ``blueYellow`` when the trajectory was
            built with ``matrix_name='GeneScoreMatrix'`` and ``solarExtra``
            otherwise.
        color_columns: add a top bar coloured by ``traj.obs['label']``.
        label_markers: feature names to label (case-insensitive).
        label_top: label the top-N most variable features.
        label_rows: label every row.
        figsize: figure size.
        title: colour-bar title (default the trajectory source).
        show: call ``plt.show()``.

    Returns:
        ``(fig, mat)`` — the figure and the plotted matrix.
    """
    mat = trajectory_heatmap_matrix(traj, var_cutoff, max_features, scale_rows, limits,
                                    grep_exclude, row_order)
    params = traj.uns.get("trajectory_params", {})
    src = str(params.get("matrix_name") or params.get("source", ""))
    if cmap is None:
        cmap = _archr_cmap("blueYellow" if src.lower() == "genescorematrix" else "solarExtra")
    elif isinstance(cmap, str) and cmap in ARCHR_PALETTES:
        cmap = _archr_cmap(cmap)

    full = _traj_matrix(traj)
    if grep_exclude is not None:
        full = full.loc[~full.index.str.contains(grep_exclude, regex=True)]
    full = full.loc[~full.isna().any(axis=1)]
    var_q = _get_quantiles(full.var(axis=1, ddof=1).values)
    by_var = full.index[np.argsort(-var_q, kind="stable")]
    labels = list(by_var[:label_top]) if label_top else []
    if label_markers:
        low = {n.lower(): n for n in mat.index}
        labels += [low[m.lower()] for m in label_markers if m.lower() in low]
    labels = [l for l in dict.fromkeys(labels) if l in mat.index]

    fig = plt.figure(figsize=figsize)
    top = 0.04 if color_columns and "label" in traj.obs else 0.0
    ax = fig.add_axes([0.05, 0.08, 0.62, 0.86 - top])
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, interpolation="nearest",
                   vmin=np.nanmin(mat.values), vmax=np.nanmax(mat.values))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Pseudo-time")
    if label_rows:
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(mat.index, fontsize=4)
        ax.yaxis.tick_right()
    elif labels:
        pos = [mat.index.get_loc(l) for l in labels]
        ys = np.linspace(0, mat.shape[0] - 1, len(pos)) if len(pos) > 1 else np.array(pos, float)
        o = np.argsort(pos)
        for y_lab, p in zip(ys, np.asarray(pos)[o]):
            ax.annotate(mat.index[p].split(":")[-1], xy=(mat.shape[1] - 0.5, p),
                        xytext=(mat.shape[1] + 6, y_lab), fontsize=5, va="center",
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", lw=0.3, color="0.4"))
    if top:
        lab = traj.obs["label"].astype(str).values
        cats = list(dict.fromkeys(lab))
        pal = {c: ARCHR_STALLION[i % len(ARCHR_STALLION)] for i, c in enumerate(cats)}
        cax = fig.add_axes([0.05, 0.94 - top + 0.005, 0.62, top - 0.01])
        cax.imshow([[mpl.colors.to_rgb(pal[c]) for c in lab]], aspect="auto")
        cax.set_xticks([]); cax.set_yticks([])
        handles = [mpl.patches.Patch(color=pal[c], label=c) for c in cats]
        cax.legend(handles=handles, ncol=len(cats), fontsize=6, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, 1.0))
    cb = fig.add_axes([0.9, 0.3, 0.025, 0.3])
    fig.colorbar(im, cax=cb)
    cb.set_title(title or ("Z-score" if scale_rows else ""), fontsize=7)
    if show:
        plt.show()
    return fig, mat
