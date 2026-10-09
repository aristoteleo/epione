"""ArchR ``plotEnrichHeatmap`` for :func:`epione.tl.peak_anno_enrichment` results.

Port of ArchR (MIT) ``plotEnrichHeatmap`` + ``.binarySort`` + ``.rowScale``.
Column clustering reproduces R's ``hclust(dist(x))`` (complete linkage,
Murtagh's nearest-neighbour Fortran routine + ``hcass2`` leaf ordering) so the
returned matrix matches ArchR's ``returnMatrix = TRUE`` exactly.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ARCHR_COMET = ["#E6E7E8", "#3A97FF", "#8816A7", "black"]


def _r_dist(x: np.ndarray) -> np.ndarray:
    """R ``dist(x)`` (euclidean), sequential accumulation like ``stats:::C_Cdist``."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = 0.0
            for d in (x[i] - x[j]).tolist():
                s += d * d
            out[i, j] = out[j, i] = math.sqrt(s)
    return out


def r_hclust_complete(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``hclust(dist(x), method = "complete")`` -> (merge, height, order).

    ``merge`` uses R's convention (negative = singleton, 1-based); ``order``
    is 0-based.
    """
    D = _r_dist(x)
    n = D.shape[0]
    if n < 2:
        return np.zeros((0, 2), int), np.zeros(0), np.arange(n)
    diss = D.copy()
    inf = 1e300
    flag = np.ones(n, dtype=bool)
    nn = np.zeros(n, dtype=int)
    disnn = np.full(n, inf)
    ia = np.zeros(n - 1, dtype=int)
    ib = np.zeros(n - 1, dtype=int)
    crit = np.zeros(n - 1)
    for i in range(n - 1):
        dmin, jm = inf, 0
        for j in range(i + 1, n):
            if dmin > diss[i, j]:
                dmin, jm = diss[i, j], j
        nn[i], disnn[i] = jm, dmin
    jj = 0
    for step in range(n - 1):
        dmin, im, jm = inf, 0, 0
        for i in range(n - 1):
            if flag[i] and disnn[i] < dmin:
                dmin, im, jm = disnn[i], i, nn[i]
        i2, j2 = min(im, jm), max(im, jm)
        ia[step], ib[step], crit[step] = i2, j2, dmin
        flag[j2] = False
        dmin = inf
        for k in range(n):
            if flag[k] and k != i2:
                v = max(diss[i2, k], diss[j2, k])
                diss[i2, k] = diss[k, i2] = v
                if i2 < k:
                    if v < dmin:
                        dmin, jj = v, k
                elif v < disnn[k]:
                    disnn[k], nn[k] = v, i2
        disnn[i2], nn[i2] = dmin, jj
        for i in range(n - 1):
            if flag[i] and (nn[i] == i2 or nn[i] == j2):
                dmin = inf
                for j in range(i + 1, n):
                    if flag[j] and diss[i, j] < dmin:
                        dmin, jj = diss[i, j], j
                nn[i], disnn[i] = jj, dmin
    # hcass2: convert to R merge + order (1-based observation ids)
    IA, IB = ia + 1, ib + 1
    iia, iib = IA.copy(), IB.copy()
    for i in range(n - 2):
        k = min(IA[i], IB[i])
        for j in range(i + 1, n - 1):
            if IA[j] == k:
                iia[j] = -(i + 1)
            if IB[j] == k:
                iib[j] = -(i + 1)
    iia, iib = -iia, -iib
    for i in range(n - 1):
        if iia[i] > 0 and iib[i] < 0:
            iia[i], iib[i] = iib[i], iia[i]
        if iia[i] > 0 and iib[i] > 0:
            k1, k2 = min(iia[i], iib[i]), max(iia[i], iib[i])
            iia[i], iib[i] = k1, k2
    order = [iia[n - 2], iib[n - 2]]
    for i in range(n - 2, 0, -1):
        for j in range(len(order)):
            if order[j] == i:
                order[j] = iia[i - 1]
                order.insert(j + 1, iib[i - 1])
                break
    order = -np.asarray(order) - 1
    merge = np.column_stack([iia, iib])
    return merge, crit, order


def _binary_sort(m: pd.DataFrame, lmat: pd.DataFrame, cluster_cols: bool):
    """ArchR ``.binarySort`` with a supplied logical matrix."""
    vals, lv = m.to_numpy(), lmat.to_numpy(dtype=np.float64)
    hc = None
    col_idx = np.arange(m.shape[1])
    if cluster_cols:
        merge, height, order = r_hclust_complete(vals.T)
        hc = dict(merge=merge, height=height, order=order, labels=list(m.columns))
        col_idx = order
    lv = lv[:, col_idx]
    keys = [-lv[:, j] for j in range(lv.shape[1] - 1, -1, -1)]
    row_idx = np.lexsort(keys) if keys else np.arange(m.shape[0])
    out = m.iloc[row_idx, col_idx]
    return out, hc


def _r_rint(x: float) -> float:
    return float(np.rint(x))


def enrich_heatmap(
    enrich: Dict[str, pd.DataFrame],
    *,
    n: int = 10,
    cutoff: float = 20,
    p_max: float = math.inf,
    cluster_cols: bool = True,
    binary_cluster_rows: bool = True,
    label_rows: bool = True,
    transpose: bool = False,
    return_matrix: bool = False,
    cmap=None,
    figsize: Optional[Tuple[float, float]] = None,
    ax=None,
    show: bool = True,
):
    """Heatmap of peak-annotation enrichment (ArchR ``plotEnrichHeatmap``).

    Arguments:
        enrich: result of :func:`epione.tl.peak_anno_enrichment`.
        n: top annotations per group (by ``mlog10Padj``) to keep.
        cutoff: minimum ``mlog10Padj`` for an annotation to be kept.
        p_max: values above are clipped before row scaling. Unless
            ``p_max == 100`` (ArchR's rule), row names get the row maximum
            appended, e.g. ``"GATA1 (85)"``.
        cluster_cols: hierarchically cluster groups (complete linkage).
        binary_cluster_rows: order rows by ArchR's binary sort on
            ``mlog10Padj >= 0.9 * row max``.
        label_rows: show row labels.
        transpose: groups as rows (ArchR tutorial style).
        return_matrix: return the plotted matrix (row-scaled ``mlog10Padj``
            x 100) instead of plotting, identical to ArchR ``returnMatrix=TRUE``.
        cmap: colormap; default ArchR ``comet``.

    Returns:
        ``pd.DataFrame`` if ``return_matrix`` else ``(fig, ax)``.
    """
    mat = enrich["mlog10Padj"] if isinstance(enrich, dict) else enrich
    mat = mat.astype(np.float64)
    keep = []
    for j in range(mat.shape[1]):
        col = mat.iloc[:, j].to_numpy()
        key = np.where(np.isnan(col), np.inf, -col)
        idx = np.argsort(key, kind="stable")[:n]
        for i in idx:
            if col[i] > cutoff:
                keep.append(mat.index[i])
    keep = list(dict.fromkeys(keep))
    if not keep:
        raise ValueError("No enrichments found for your cutoff!")
    mat = mat.loc[keep]
    vals = mat.to_numpy()
    pass_mat = pd.DataFrame((vals >= 0.9 * vals.max(axis=1, keepdims=True)).astype(float),
                            index=mat.index, columns=mat.columns)
    vals = np.where(vals > p_max, p_max, vals)
    rmax = vals.max(axis=1)
    scaled = vals / rmax[:, None]
    rows = list(mat.index)
    if p_max != 100:
        rows = [f"{r} ({int(_r_rint(mx))})" for r, mx in zip(rows, rmax)]
    mat2 = pd.DataFrame(scaled * 100, index=rows, columns=mat.columns)
    pass_mat.index = rows

    hc = None
    clustered = False
    if mat2.shape[0] > 1 and mat2.shape[1] > 1:
        if binary_cluster_rows:
            bs, hc = _binary_sort(mat2, pass_mat.loc[mat2.index, mat2.columns], cluster_cols)
            mat2 = bs[list(mat2.columns)]
        else:
            clustered = True
    if transpose:
        if hc is not None:
            mat2 = mat2.iloc[:, hc["order"]]
        mat2 = mat2.T
    if return_matrix:
        return mat2
    return _draw(mat2, hc, transpose, clustered, cluster_cols, label_rows, cmap, figsize, ax, show)


def _draw(mat2, hc, transpose, cluster_rows, cluster_cols, label_rows, cmap, figsize, ax, show):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("archr_comet", ARCHR_COMET)
    data = mat2
    if cluster_rows and not transpose:
        from scipy.cluster.hierarchy import leaves_list, linkage
        data = data.iloc[leaves_list(linkage(data.to_numpy(), "complete"))]
        if cluster_cols:
            data = data.iloc[:, leaves_list(linkage(data.to_numpy().T, "complete"))]
    if not transpose and hc is not None:
        data = data.iloc[:, hc["order"]]
    nr, nc = data.shape
    if figsize is None:
        figsize = (max(4, 0.32 * nc + 3), max(3, 0.16 * nr + 1.8))
    if ax is None:
        fig = plt.figure(figsize=figsize)
        if hc is not None:
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 8] if not transpose else [0.01, 1], hspace=0.02)
            dax = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[1])
            if not transpose:
                _draw_dendrogram(dax, hc)
            dax.axis("off")
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    im = ax.imshow(data.to_numpy(), aspect="auto", cmap=cmap, vmin=0,
                   vmax=float(np.nanmax(data.to_numpy())), interpolation="nearest")
    ax.set_xticks(range(nc))
    ax.set_xticklabels(data.columns, rotation=90, fontsize=7)
    if label_rows:
        ax.set_yticks(range(nr))
        ax.set_yticklabels(data.index, fontsize=6)
    else:
        ax.set_yticks([])
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Norm. Enrichment -log10(P-adj) [0-Max]", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    if show:
        plt.show()
    return fig, ax


def _draw_dendrogram(ax, hc):
    merge, height, order = hc["merge"], hc["height"], hc["order"]
    n = len(order)
    pos = np.empty(n)
    pos[order] = np.arange(n)
    node_x, node_y = {}, {}

    def coord(k):
        if k < 0:
            return pos[-k - 1], 0.0
        return node_x[k], node_y[k]

    for i, (a, b) in enumerate(merge, start=1):
        xa, ya = coord(a)
        xb, yb = coord(b)
        h = height[i - 1]
        ax.plot([xa, xa, xb, xb], [ya, h, h, yb], color="black", lw=0.7)
        node_x[i], node_y[i] = (xa + xb) / 2, h
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, max(height) * 1.02 if len(height) else 1)
