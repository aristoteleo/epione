"""
Supervised pseudo-time trajectories (port of ArchR ``Trajectory.R``).

ArchR 1.0.3 (MIT licence) functions ported here:

- :func:`add_trajectory` — ``addTrajectory``: outlier filtering, initial
  alignment, per-dimension cubic smoothing spline (``stats::smooth.spline``
  with fixed ``spar``), kNN projection of cells onto the spline, quantile
  pseudo-time in ``[0, 100]``.
- :func:`get_trajectory` — ``getTrajectory``: sum cells into pseudo-time
  bins, depth-normalise, log2, centred rolling mean.
- :func:`trajectory_heatmap_matrix` — ``plotTrajectoryHeatmap(returnMatrix=TRUE)``.
- :func:`correlate_trajectories` — ``correlateTrajectories``.

``stats::smooth.spline`` is re-implemented exactly (same unique-x
reduction, knot placement, penalty ratio and the ``.3330`` constant used by
R's ``sgram.f``) so the fitted splines agree with R to machine precision.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.interpolate import BSpline
from scipy.linalg import cho_solve_banded, cholesky_banded
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from scipy.stats import t as t_dist
from anndata import AnnData


# ---------------------------------------------------------------------------
# ArchR helpers
# ---------------------------------------------------------------------------

def _get_quantiles(v: np.ndarray) -> np.ndarray:
    """ArchR ``.getQuantiles``: ``trunc(rank(v)) / length(v)``."""
    v = np.asarray(v, dtype=float)
    return np.trunc(rankdata(v, method="average")) / len(v)


def _center_roll_mean(v: np.ndarray, k: int) -> np.ndarray:
    """ArchR ``.centerRollMean`` (right-aligned ``frollmean`` padded at both ends)."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if k > n:
        return np.full(n, np.nan)
    o1 = np.lib.stride_tricks.sliding_window_view(v, k).mean(axis=1)  # o1[i] = R o1[k + i]
    half = k // 2
    left = half - 1 if k % 2 == 0 else half
    return np.concatenate([np.repeat(o1[0], left), o1, np.repeat(o1[-1], half)])


def _r_quantile(x: np.ndarray, q: float) -> float:
    """R ``quantile(type = 7)``."""
    return float(np.quantile(np.asarray(x, dtype=float), q))


# ---------------------------------------------------------------------------
# stats::smooth.spline (fixed spar) — exact port
# ---------------------------------------------------------------------------

def _nknots_smspl(n: int) -> int:
    if n < 50:
        return n
    a1, a2, a3, a4 = np.log2(50), np.log2(100), np.log2(140), np.log2(200)
    if n < 200:
        v = 2 ** (a1 + (a2 - a1) * (n - 50) / 150)
    elif n < 800:
        v = 2 ** (a2 + (a3 - a2) * (n - 200) / 600)
    elif n < 3200:
        v = 2 ** (a3 + (a4 - a3) * (n - 800) / 2400)
    else:
        v = 200 + (n - 3200) ** 0.2
    return int(np.trunc(v))


def _r_seq_length_out(start: float, stop: float, n: int) -> np.ndarray:
    """R ``seq.int(from, to, length.out = n)`` (C implementation)."""
    out = np.empty(n)
    out[0] = start
    if n > 1:
        out[-1] = stop
    if n > 2:
        by = (stop - start) / (n - 1)
        for i in range(1, n - 1):
            out[i] = start + i * by if i < n // 2 else stop - (n - 1 - i) * by
    return out


def _smooth_spline(x: np.ndarray, y: np.ndarray, spar: float = 1.0):
    """``stats::smooth.spline(x, y, spar = spar)`` with unit weights.

    Returns ``(ux, fitted)`` — the sorted unique x values and the fitted
    values at those points (``fit$x`` and ``fit$y`` in R).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    q75, q25 = np.quantile(x, [0.75, 0.25])
    tol = 1e-6 * (q75 - q25)
    xx = np.round((x - x.mean()) / tol)
    order = np.argsort(x, kind="stable")
    xxs = xx[order]
    nd = np.concatenate([[True], xxs[:-1] < xxs[1:]])
    ux = x[order][nd]
    nx = len(ux)
    if nx <= 3:
        raise ValueError("need at least four unique 'x' values")
    grp = np.cumsum(nd) - 1                     # group id of each sorted point
    wbar = np.bincount(grp, minlength=nx).astype(float)
    ybar = np.bincount(grp, weights=y[order], minlength=nx) / wbar

    xbar = (ux - ux[0]) / (ux[-1] - ux[0])
    nknots = _nknots_smspl(nx)
    sel = np.trunc(_r_seq_length_out(1.0, float(nx), nknots)).astype(int) - 1
    knot = np.concatenate([np.repeat(xbar[0], 3), xbar[sel], np.repeat(xbar[-1], 3)])
    nk = nknots + 2

    # X'WX (banded) and X'Wy — stxwx
    B = BSpline.design_matrix(xbar, knot, 3).toarray()      # nx × nk
    XtWX = B.T @ (B * wbar[:, None])
    XtWy = B.T @ (wbar * ybar)

    # SIGMA — sgram.f (linear B'' per interval, R's literal .3330 constant)
    basis = BSpline(knot, np.eye(nk), 3, extrapolate=True)
    d2 = basis.derivative(2)
    sigma = np.zeros((nk, nk))
    for i in range(3, nk):                                 # interval [knot[i], knot[i+1]]
        a, b = knot[i], knot[i + 1]
        wpt = b - a
        if wpt <= 0:
            continue
        cols = np.arange(i - 3, i + 1)
        yw1 = d2(a)[cols]
        yw2 = d2(b)[cols] - yw1
        term = (np.outer(yw1, yw1) + (np.outer(yw2, yw1) + np.outer(yw1, yw2)) * 0.5
                + np.outer(yw2, yw2) * 0.3330)
        sigma[np.ix_(cols, cols)] += wpt * term

    t1 = np.trace(XtWX[2:nk - 3, 2:nk - 3])
    t2 = np.trace(sigma[2:nk - 3, 2:nk - 3])
    lam = (t1 / t2) * 16.0 ** (spar * 6.0 - 2.0)

    A = XtWX + lam * sigma
    ab = np.zeros((4, nk))                                  # upper banded storage
    for d in range(4):
        ab[3 - d, d:] = np.diagonal(A, offset=d)
    coef = cho_solve_banded((cholesky_banded(ab), False), XtWy)
    return ux, B @ coef


# ---------------------------------------------------------------------------
# addTrajectory
# ---------------------------------------------------------------------------

def _get_rep(adata: AnnData, use_rep: str) -> np.ndarray:
    if use_rep in adata.obsm:
        return np.asarray(adata.obsm[use_rep], dtype=float)
    raise KeyError(f"use_rep={use_rep!r} not found in adata.obsm")


def _project_on_spline(spline: np.ndarray, query: np.ndarray):
    dist, idx = cKDTree(spline).query(query, k=3)
    knn_diff = np.where(idx[:, 1] > idx[:, 2], 1.0, -1.0)
    # ArchR uses 1-based nabor indices; keep that so DistanceIdx matches R.
    return dist[:, 0], (idx[:, 0] + 1) + knn_diff * _get_quantiles(dist[:, 0])


def add_trajectory(
    adata: AnnData,
    trajectory: Sequence[str],
    groupby: str = "Clusters",
    use_rep: str = "X_birmt",
    pre_filter_quantile: float = 0.9,
    post_filter_quantile: float = 0.9,
    use_all: bool = False,
    spar: float = 1.0,
    key_added: str = "Trajectory",
    return_df: bool = False,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """Fit a supervised pseudo-time trajectory (ArchR ``addTrajectory``).

    Cells of the groups listed in ``trajectory`` (in order) are used to fit
    a smoothing spline through ``adata.obsm[use_rep]``; every cell is then
    placed along the spline by nearest-neighbour projection and assigned a
    pseudo-time in ``(0, 100]``.

    Arguments:
        adata: AnnData with ``obs[groupby]`` and ``obsm[use_rep]``.
        trajectory: ordered group labels, e.g. ``["C11", "C10", "C5"]``.
        groupby: ``adata.obs`` column holding the groups.
        use_rep: ``adata.obsm`` key of the space used for fitting (a reduced
            dimension such as ``'X_iterative_lsi'`` or an embedding such as
            ``'X_umap'``).
        pre_filter_quantile: drop cells further than this distance quantile
            from their group centroid before fitting.
        post_filter_quantile: drop cells whose distance to the spline exceeds
            this quantile.
        use_all: also place cells outside the trajectory groups (kept when
            closer to the spline than the furthest retained trajectory cell).
        spar: smoothing parameter passed to the ``smooth.spline`` port.
            ArchR also passes ``df = 250``, which R ignores whenever ``spar``
            is given, so it is not exposed here.
        key_added: ``adata.obs`` column receiving the pseudo-time (NaN for
            cells not on the trajectory).
        return_df: return the per-cell ``Distance`` / ``DistanceIdx`` /
            ``Trajectory`` frame (ArchR ``saveDF``).
        verbose: print progress.

    Returns:
        ``pd.DataFrame`` when ``return_df`` else ``None``. The spline is
        stored in ``adata.uns[key_added + '_spline']``.
    """
    trajectory = [str(t) for t in trajectory]
    groups = adata.obs[groupby].astype(str).values
    rep = _get_rep(adata, use_rep)
    df, diag = _trajectory_core(rep, groups, np.asarray(adata.obs_names), trajectory,
                                pre_filter_quantile, post_filter_quantile, use_all, spar)
    adata.obs[key_added] = pd.Series(df["Trajectory"], index=df.index).reindex(adata.obs_names).values
    adata.uns[f"{key_added}_spline"] = diag["spline"]
    adata.uns[f"{key_added}_params"] = dict(
        trajectory=trajectory, groupby=groupby, use_rep=use_rep,
        pre_filter_quantile=pre_filter_quantile, post_filter_quantile=post_filter_quantile,
        use_all=use_all, spar=spar,
    )
    if verbose:
        print(f"[add_trajectory] {len(df)} cells placed on {' -> '.join(trajectory)} "
              f"-> adata.obs['{key_added}']")
    return df if return_df else None


def _trajectory_core(rep, groups, names, trajectory, pre_filter_quantile=0.9,
                     post_filter_quantile=0.9, use_all=False, spar=1.0):
    """Body of ArchR ``addTrajectory``; returns the trajectory frame and intermediates."""
    in_traj = np.isin(groups, trajectory)
    n_groups = len(set(groups[in_traj]))
    if n_groups == 0:
        raise ValueError("trajectory does not span any groups in groupby!")
    if n_groups < 3:
        raise ValueError("trajectory must span at least 3 groups in groupby!")
    mat = rep[in_traj]
    g = groups[in_traj]
    cell_ids = names[in_traj]

    kept = []
    for grp in trajectory:
        ix = np.where(g == grp)[0]
        matx = mat[ix]
        diffx = np.sqrt(((matx - matx.mean(axis=0)) ** 2).sum(axis=1))
        kept.append(ix[diffx <= _r_quantile(diffx, pre_filter_quantile)])

    times, rows = [], []
    for x, ix in enumerate(kept, start=1):
        matx = mat[ix]
        if x != len(trajectory):
            meanx = mat[kept[x]].mean(axis=0)
            diffx = np.sqrt(((matx - meanx) ** 2).sum(axis=1))
            times.append((1 - _get_quantiles(diffx)) + x)
        else:
            meanx = mat[kept[x - 2]].mean(axis=0)
            diffx = np.sqrt(((matx - meanx) ** 2).sum(axis=1))
            times.append(_get_quantiles(diffx) + x)
        rows.append(ix)
    initial_time = np.concatenate(times)
    fit_rows = np.concatenate(rows)

    spline = np.column_stack([
        _smooth_spline(initial_time, mat[fit_rows, j], spar=spar)[1]
        for j in range(mat.shape[1])
    ])

    knn_dist, knn_idx = cKDTree(spline).query(mat, k=3)
    dist, didx = _project_on_spline(spline, mat)
    keep = dist <= _r_quantile(dist, post_filter_quantile)
    df = pd.DataFrame({"Distance": dist[keep], "DistanceIdx": didx[keep]}, index=cell_ids[keep])

    if use_all:
        out_ix = ~in_traj
        dist2, didx2 = _project_on_spline(spline, rep[out_ix])
        keep2 = dist2 < df["Distance"].max()
        df2 = pd.DataFrame({"Distance": dist2[keep2], "DistanceIdx": didx2[keep2]},
                           index=names[out_ix][keep2])
        df = pd.concat([df, df2])

    df["Trajectory"] = 100 * _get_quantiles(df["DistanceIdx"].values)
    diag = dict(
        initial_time=pd.Series(initial_time, index=cell_ids[fit_rows]),
        spline=spline, knn_idx=knn_idx + 1, knn_dist=knn_dist, query_cells=cell_ids,
    )
    return df, diag


# ---------------------------------------------------------------------------
# getTrajectory
# ---------------------------------------------------------------------------

def _feature_matrix(adata: AnnData, layer: Optional[str], use_rep: Optional[str],
                    feature_names: Optional[Sequence[str]]):
    if use_rep is not None:
        X = adata.obsm[use_rep]
        if isinstance(X, pd.DataFrame):
            names = list(X.columns) if feature_names is None else list(feature_names)
            X = X.values
        else:
            if feature_names is None:
                key = f"{use_rep}_names"
                feature_names = adata.uns.get(key, [f"{use_rep}_{i}" for i in range(X.shape[1])])
            names = list(feature_names)
        var = pd.DataFrame(index=pd.Index(names, dtype=str))
    else:
        X = adata.layers[layer] if layer is not None else adata.X
        var = adata.var.copy()
    return X, var


def get_trajectory(
    adata: AnnData,
    trajectory_key: str = "Trajectory",
    layer: Optional[str] = None,
    use_rep: Optional[str] = None,
    feature_names: Optional[Sequence[str]] = None,
    group_every: float = 1,
    log2_norm: bool = True,
    scale_to: Optional[float] = 10000,
    smooth_window: Optional[int] = 11,
    trajectory_label: Optional[str] = None,
    matrix_name: Optional[str] = None,
) -> AnnData:
    """Summarise a feature matrix along a pseudo-time trajectory (ArchR ``getTrajectory``).

    Cells are binned into non-overlapping pseudo-time windows of width
    ``group_every`` (``(0,1], (1,2], ...``), values are **summed** per bin,
    optionally depth-normalised to ``scale_to`` and log2-transformed, then
    smoothed with a centred rolling mean of ``smooth_window`` bins.

    Arguments:
        adata: cells × features AnnData carrying ``obs[trajectory_key]``.
        trajectory_key: pseudo-time column written by :func:`add_trajectory`.
        layer: use ``adata.layers[layer]`` instead of ``adata.X``.
        use_rep: use ``adata.obsm[use_rep]`` (e.g. chromVAR z-scores) instead
            of the main matrix.
        feature_names: feature names for ``use_rep`` matrices.
        group_every: bin width in pseudo-time percentiles.
        log2_norm: ``log2(x + 1)`` after scaling. Skipped (as in ArchR) when
            any value is negative — set ``False`` for deviation matrices.
        scale_to: column-sum normalisation target; ``None`` disables it.
        smooth_window: rolling-mean window (in bins); ``None`` disables it.
        trajectory_label: optional ``adata.obs`` column; the majority label of
            each bin is stored in ``obs['label']``.
        matrix_name: name of the source matrix (ArchR ``useMatrix``, e.g.
            ``'GeneScoreMatrix'``); only used to pick the default heatmap
            palette (``blueYellow`` for gene scores, ``solarExtra`` otherwise).

    Returns:
        AnnData with bins as observations (``T.0_1``, ...) and the input
        features as variables. ``layers['mat']`` holds the binned matrix,
        ``layers['smoothMat']`` the smoothed one (also in ``X``).
    """
    pt = adata.obs[trajectory_key].values.astype(float)
    ok = ~np.isnan(pt)
    if not np.all((pt[ok] >= 0) & (pt[ok] <= 100)):
        raise ValueError("Trajectory values must be between 0 and 100.")
    breaks = np.arange(0, 100 + group_every / 2, group_every, dtype=float)
    breaks = breaks[breaks <= 100]
    nb = len(breaks) - 1

    def _fmt(v):
        return f"{v:g}"

    bin_names = [f"T.{_fmt(breaks[i])}_{_fmt(breaks[i + 1])}" for i in range(nb)]
    member = np.full(adata.n_obs, -1)
    for b in range(nb):
        member[ok & (pt > breaks[b]) & (pt <= breaks[b + 1])] = b

    X, var = _feature_matrix(adata, layer, use_rep, feature_names)
    sel = np.where(member >= 0)[0]
    G = sp.csr_matrix((np.ones(len(sel)), (member[sel], sel)), shape=(nb, adata.n_obs))
    mat = G @ X
    mat = np.asarray(mat.toarray() if sp.issparse(mat) else mat, dtype=float).T   # features × bins

    with np.errstate(divide="ignore", invalid="ignore"):
        if scale_to is not None:
            if np.any(mat < 0):
                print("Some values are below 0; continuing without depth normalization!")
            else:
                mat = mat / mat.sum(axis=0) * scale_to
        if log2_norm:
            if np.any(mat < 0):
                print("Some values are below 0; continuing without log2 normalization!")
            else:
                mat = np.log2(mat + 1)

    out = AnnData(X=mat.T.copy(), obs=pd.DataFrame(index=bin_names), var=var)
    out.layers["mat"] = mat.T.copy()
    if smooth_window is not None:
        smooth = np.vstack([_center_roll_mean(r, smooth_window) for r in mat])
        out.layers["smoothMat"] = smooth.T
        out.X = smooth.T.copy()
    if trajectory_label is not None:
        lab = adata.obs[trajectory_label].astype(str).values
        labels = []
        for b in range(nb):
            vals = lab[member == b]
            if len(vals) == 0:
                labels.append(np.nan)
                continue
            u, c = np.unique(vals, return_counts=True)
            # R: names(sort(table(x), decreasing = TRUE))[1] — ties keep table (sorted) order
            labels.append(u[np.argsort(-c, kind="stable")[0]])
        out.obs["label"] = labels
    out.uns["trajectory_params"] = dict(
        trajectory_key=trajectory_key, group_every=group_every, log2_norm=log2_norm,
        scale_to=scale_to, smooth_window=smooth_window,
        source=use_rep if use_rep is not None else (layer or "X"),
        matrix_name=matrix_name,
    )
    return out


# ---------------------------------------------------------------------------
# plotTrajectoryHeatmap(returnMatrix = TRUE)
# ---------------------------------------------------------------------------

def _traj_matrix(traj: AnnData) -> pd.DataFrame:
    """Features × bins frame of the first assay (smoothMat if present, as in ArchR)."""
    M = traj.layers["smoothMat"] if "smoothMat" in traj.layers else traj.layers.get("mat", traj.X)
    return pd.DataFrame(np.asarray(M, dtype=float).T, index=traj.var_names, columns=traj.obs_names)


def trajectory_heatmap_matrix(
    traj: AnnData,
    var_cutoff: Optional[float] = 0.9,
    max_features: Optional[int] = 25000,
    scale_rows: bool = True,
    limits: Sequence[float] = (-1.5, 1.5),
    grep_exclude: Optional[str] = None,
    row_order: Optional[Sequence] = None,
) -> pd.DataFrame:
    """Matrix drawn by ArchR ``plotTrajectoryHeatmap`` (``returnMatrix = TRUE``).

    Keeps the most variable features (top ``1 - var_cutoff`` fraction, at most
    ``max_features``), row z-scores and clips them to ``limits`` and orders
    rows by the bin of their maximum.

    Arguments:
        traj: output of :func:`get_trajectory`.
        var_cutoff: variance-quantile cutoff.
        max_features: maximum number of features.
        scale_rows: row-wise z-score.
        limits: clipping limits for the z-scores.
        grep_exclude: regex; matching feature names are dropped.
        row_order: explicit row order (indices or names) — disables the
            variance selection, as in ArchR.

    Returns:
        ``pd.DataFrame`` (features × bins) in plotting order.
    """
    mat = _traj_matrix(traj)
    if grep_exclude is not None:
        mat = mat.loc[~mat.index.str.contains(grep_exclude, regex=True)]
    mat = mat.loc[~mat.isna().any(axis=1)]
    var_q = _get_quantiles(mat.var(axis=1, ddof=1).values)
    if row_order is None:
        mat = mat.iloc[np.argsort(-var_q, kind="stable")]
        if var_cutoff is None and max_features is None:
            n = mat.shape[0]
        elif var_cutoff is None:
            n = max_features
        elif max_features is None:
            n = (1 - var_cutoff) * mat.shape[0]
        else:
            n = min((1 - var_cutoff) * mat.shape[0], max_features)
        n = int(np.trunc(min(n, mat.shape[0])))
        mat = mat.iloc[:n]
    if scale_rows:
        v = mat.values
        z = (v - v.mean(axis=1, keepdims=True)) / v.std(axis=1, ddof=1, keepdims=True)
        z = np.clip(z, min(limits), max(limits))
        mat = pd.DataFrame(z, index=mat.index, columns=mat.columns)
    if mat.shape[0] == 0:
        raise ValueError("No Features Remaining!")
    if row_order is not None:
        ro = list(row_order)
        return mat.loc[ro] if isinstance(ro[0], str) else mat.iloc[ro]
    idx = np.argsort(np.nanargmax(mat.values, axis=1), kind="stable")
    return mat.iloc[idx]


# ---------------------------------------------------------------------------
# correlateTrajectories
# ---------------------------------------------------------------------------

def _match_names(names: Sequence[str], remove: Optional[Sequence[str]]) -> np.ndarray:
    s = pd.Series(list(names), dtype=str)
    rem = [r.lower() for r in (remove or [])]
    if "underscore" in rem:
        s = s.str.replace(r"_.*", "", regex=True)
    if "dash" in rem:
        s = s.str.replace(r"-.*", "", regex=True)
    if "numeric" in rem:
        s = s.str.replace(r"[0-9]+", "", regex=True)
    if "dot" in rem:
        s = s.str.replace(r"\..*", "", regex=True)
    return s.values


def _bh(p: np.ndarray) -> np.ndarray:
    """R ``p.adjust(method = 'fdr')`` (NAs kept, not counted)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    o = np.argsort(-pv, kind="stable")
    ranks = np.arange(n, 0, -1)
    adj = np.minimum.accumulate(n / ranks * pv[o])
    res = np.empty(n)
    res[o] = np.minimum(adj, 1)
    out[ok] = res
    return out


def _row_cor(X: np.ndarray, Y: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    a = X[ix]
    b = Y[iy]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


def correlate_trajectories(
    traj1: AnnData,
    traj2: AnnData,
    cor_cutoff: float = 0.5,
    var_cutoff1: float = 0.8,
    var_cutoff2: float = 0.8,
    remove_from_name1: Optional[Sequence[str]] = ("underscore", "dash"),
    remove_from_name2: Optional[Sequence[str]] = ("underscore", "dash"),
    name_key1: Optional[str] = None,
    name_key2: Optional[str] = None,
    force: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Correlate two trajectories feature-by-feature (ArchR ``correlateTrajectories``).

    Features are paired by name after stripping suffixes
    (``remove_from_name*``: any of ``'underscore'``, ``'dash'``,
    ``'numeric'``, ``'dot'``), then the Pearson correlation of the
    **unsmoothed** binned matrices (``layers['mat']``) is computed with a
    Student-t p-value and BH FDR.

    Arguments:
        traj1, traj2: outputs of :func:`get_trajectory` with the same bins.
        cor_cutoff: minimum correlation of reported pairs.
        var_cutoff1, var_cutoff2: minimum variance quantile in each trajectory.
        remove_from_name1, remove_from_name2: suffix-stripping rules.
        name_key1, name_key2: ``var`` column holding feature names
            (default: ``var_names``).
        force: continue when fewer than 5% of features can be matched.

    Returns:
        dict with ``'correlatedMappings'`` (filtered) and ``'allMappings'``.
    """
    n1 = traj1.var[name_key1].astype(str).values if name_key1 else np.asarray(traj1.var_names)
    n2 = traj2.var[name_key2].astype(str).values if name_key2 else np.asarray(traj2.var_names)
    m1 = _match_names(n1, remove_from_name1)
    m2 = _match_names(n2, remove_from_name2)
    if not np.isin(m1, m2).any():
        raise ValueError("Matching of traj1 and traj2 resulted in no mappings!")
    match_p = max(np.isin(m1, m2).mean(), np.isin(m2, m1).mean())
    if match_p < 0.05 and not force:
        raise ValueError("Matching resulted in less than 5% mappings! Set force=True to continue!")

    pos2 = pd.Series(np.arange(len(m2))).groupby(m2).apply(list).to_dict()
    i1, i2 = [], []
    for x, nm in enumerate(m1):
        for y in pos2.get(nm, []):
            i1.append(x)
            i2.append(y)
    i1 = np.asarray(i1, dtype=int)
    i2 = np.asarray(i2, dtype=int)

    X = np.asarray(traj1.layers["mat"], dtype=float).T
    Y = np.asarray(traj2.layers["mat"], dtype=float).T
    r = _row_cor(X, Y, i1, i2)
    df_ = traj1.n_obs - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = r / np.sqrt((1 - r ** 2) / df_)
    pval = 2 * t_dist.cdf(-np.abs(tstat), df_)
    out = pd.DataFrame({
        "idx1": i1 + 1, "idx2": i2 + 1,
        "matchname1": m1[i1], "matchname2": m2[i2],
        "name1": np.asarray(traj1.var_names)[i1], "name2": np.asarray(traj2.var_names)[i2],
        "Correlation": r,
        "VarAssay1": _get_quantiles(X.var(axis=1, ddof=1))[i1],
        "VarAssay2": _get_quantiles(Y.var(axis=1, ddof=1))[i2],
        "TStat": tstat, "Pval": pval, "FDR": _bh(pval),
    })
    pf = out[(out["Correlation"] > cor_cutoff) & (out["VarAssay1"] > var_cutoff1)
             & (out["VarAssay2"] > var_cutoff2)]
    return {"correlatedMappings": pf.reset_index(drop=True), "allMappings": out}
