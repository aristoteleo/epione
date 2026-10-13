"""Metrics for cross-modality translation.

Four families, as used by the scTranslation benchmark (Yang et al., KDD 2026)
and implemented here from their definitions:

* clustering -- Leiden clusters of the *predicted* profiles against the
  cell-type labels (ARI, AMI, NMI, homogeneity);
* regression -- per-cell and per-feature Pearson / Spearman correlation,
  MSE and MAE between predicted and measured profiles;
* distribution -- maximum mean discrepancy with a multi-bandwidth Gaussian
  kernel and scib's graph integration LISI of measured vs predicted cells in
  a joint PCA space;
* classification -- AUROC / AUPRC of the predictions against the binarised
  measured matrix.

Every function takes ``pred`` and ``truth`` as cells x features arrays (dense
or sparse) or AnnData, with rows of the same cells in the same order.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp

__all__ = [
    "pearson_per_cell",
    "pearson_per_feature",
    "spearman_per_cell",
    "mse_per_cell",
    "mae_per_cell",
    "regression_scores",
    "cluster_scores",
    "joint_pca",
    "mmd",
    "mmd_rbf",
    "lisi",
    "ilisi",
    "ilisi_graph",
    "binarize_truth",
    "auroc",
    "auprc",
]


def _array(x) -> np.ndarray:
    if hasattr(x, "X") and hasattr(x, "obs"):
        x = x.X
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def _pair(pred, truth) -> tuple[np.ndarray, np.ndarray]:
    p, t = _array(pred), _array(truth)
    if p.shape != t.shape:
        raise ValueError(f"pred {p.shape} and truth {t.shape} differ in shape")
    return p, t


# ---------------------------------------------------------------- regression
def _rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = np.einsum("ij,ij->i", a, b)
    den = np.sqrt(np.einsum("ij,ij->i", a, a) * np.einsum("ij,ij->i", b, b))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[den == 0] = np.nan
    return r


def pearson_per_cell(pred, truth) -> np.ndarray:
    """Pearson correlation of each cell's predicted and measured profile.

    Cells whose profile is constant in either matrix get ``nan``.
    """
    p, t = _pair(pred, truth)
    return _rowwise_pearson(p, t)


def pearson_per_feature(pred, truth) -> np.ndarray:
    """Pearson correlation of each feature across cells (``nan`` if constant)."""
    p, t = _pair(pred, truth)
    return _rowwise_pearson(p.T, t.T)


def _rank_rows(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(a, axis=1, method="average")


def spearman_per_cell(pred, truth) -> np.ndarray:
    """Spearman correlation per cell (Pearson of average ranks)."""
    p, t = _pair(pred, truth)
    return _rowwise_pearson(_rank_rows(p), _rank_rows(t))


def mse_per_cell(pred, truth) -> np.ndarray:
    p, t = _pair(pred, truth)
    return np.mean((p - t) ** 2, axis=1)


def mae_per_cell(pred, truth) -> np.ndarray:
    p, t = _pair(pred, truth)
    return np.mean(np.abs(p - t), axis=1)


def regression_scores(pred, truth, *, convention: str = "standard") -> dict:
    """Per-cell PCC, SPCC, MSE, MAE and the per-feature correlation CORR.

    * ``PCC`` -- mean over cells of the Pearson correlation of the predicted and
      measured profile, skipping cells where it is undefined;
    * ``SPCC`` -- the same with Spearman correlation, divided by the same
      number of cells as PCC (the scTranslation convention);
    * ``MSE`` / ``MAE`` -- mean over cells of the per-cell mean squared /
      absolute error;
    * ``CORR`` -- mean over features of the Pearson correlation across cells,
      over features where it is finite.

    With ``convention="sctranslation"``, CORR reproduces the benchmark's code.
    That code averages over features whose *prediction* is not constant, and
    scores a feature with constant measured values as 0 (sklearn's
    ``r_regression(force_finite=True)``).
    """
    p, t = _pair(pred, truth)
    pcc = _rowwise_pearson(p, t)
    spcc = _rowwise_pearson(_rank_rows(p), _rank_rows(t))
    n_ok = int(np.isfinite(pcc).sum())
    corr = _rowwise_pearson(p.T, t.T)
    if convention == "sctranslation":
        varying_pred = (p.max(axis=0) > p.min(axis=0))
        corr = np.where(np.isfinite(corr), corr, 0.0)[varying_pred]
    elif convention != "standard":
        raise ValueError(f"convention must be 'standard' or 'sctranslation', not {convention!r}")
    return {
        "PCC": float(np.nansum(pcc) / n_ok) if n_ok else float("nan"),
        "SPCC": float(np.nansum(spcc) / n_ok) if n_ok else float("nan"),
        "MSE": float(np.mean((p - t) ** 2)),
        "MAE": float(np.mean(np.abs(p - t))),
        "CORR": float(np.mean(corr[np.isfinite(corr)])) if np.isfinite(corr).any() else float("nan"),
    }


# ---------------------------------------------------------------- clustering
def _pca(X: np.ndarray, n_comps: Optional[int], seed: int = 0) -> np.ndarray:
    """scanpy PCA (zero-centred, ARPACK) with ``min(50, n_features - 1)`` components."""
    import scanpy as sc

    n_comps = min(50 if n_comps is None else n_comps, X.shape[0] - 1, X.shape[1] - 1)
    return sc.pp.pca(np.asarray(X, dtype=np.float32), n_comps=n_comps,
                     svd_solver="arpack", random_state=seed).astype(np.float64)


def cluster_scores(
    pred,
    labels: Sequence,
    *,
    n_pcs: Optional[int] = 50,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    flavor: str = "leidenalg",
    seed: int = 0,
) -> dict:
    """Leiden-cluster the predicted profiles and compare to ``labels``.

    PCA (``n_pcs``), a ``n_neighbors`` kNN graph and Leiden at ``resolution``
    are computed with scanpy defaults, as in scTranslation's
    ``calculate_cluster_index``. Returns ARI, AMI, NMI and homogeneity.
    """
    import scanpy as sc
    from anndata import AnnData
    from sklearn import metrics as skm

    X = _array(pred).astype(np.float32)
    ad = AnnData(X=X)
    ad.obsm["X_pca"] = _pca(X, n_pcs, seed).astype(np.float32)
    sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep="X_pca", random_state=seed)
    if flavor == "igraph":
        sc.tl.leiden(ad, resolution=resolution, random_state=seed, flavor="igraph",
                     n_iterations=2, directed=False)
    else:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            sc.tl.leiden(ad, resolution=resolution, random_state=seed, flavor="leidenalg",
                         n_iterations=-1)
    lab = np.asarray(labels).astype(str)
    clu = ad.obs["leiden"].to_numpy()
    return {
        "ARI": float(skm.adjusted_rand_score(lab, clu)),
        "AMI": float(skm.adjusted_mutual_info_score(lab, clu)),
        "NMI": float(skm.normalized_mutual_info_score(lab, clu)),
        "HOM": float(skm.homogeneity_score(lab, clu)),
    }


# -------------------------------------------------------------- distribution
def joint_pca(pred, truth, *, n_pcs: Optional[int] = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """PCA of measured and predicted cells stacked together (measured first)."""
    p, t = _pair(pred, truth)
    Z = _pca(np.vstack([t, p]), n_pcs, seed)
    return Z[len(t):], Z[: len(t)]


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def mmd_rbf(
    x: np.ndarray,
    y: np.ndarray,
    *,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    bandwidth: Optional[float] = None,
) -> float:
    """Biased MMD^2 with a sum of ``kernel_num`` Gaussian kernels.

    With ``Z = [x; y]`` and ``D`` the squared distances, the base bandwidth is
    the mean off-diagonal entry of ``D`` (unless given); the kernels use
    ``bandwidth * kernel_mul**(i - kernel_num // 2)``. Returns
    ``mean(K_xx) + mean(K_yy) - 2 mean(K_xy)`` (Gretton et al., 2012). For
    ``len(x) == len(y)`` this equals the ``mean(XX + YY - XY - YX)`` form.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Z = np.vstack([x, y])
    n = len(Z)
    D = _sq_dists(Z, Z)
    if bandwidth is None:
        bandwidth = D.sum() / (n * n - n)
    base = bandwidth / kernel_mul ** (kernel_num // 2)
    K = sum(np.exp(-D / (base * kernel_mul**i)) for i in range(kernel_num))
    nx = len(x)
    return float(K[:nx, :nx].mean() + K[nx:, nx:].mean() - 2.0 * K[:nx, nx:].mean())


def mmd(pred, truth, *, n_pcs: Optional[int] = 50, max_cells: Optional[int] = 5000, seed: int = 0, **kwargs) -> float:
    """MMD between measured and predicted cells in their joint PCA space.

    ``max_cells`` subsamples the same cells from both sets (memory is
    quadratic in the number of cells).
    """
    p, t = _pair(pred, truth)
    if max_cells is not None and len(p) > max_cells:
        idx = np.random.default_rng(seed).choice(len(p), max_cells, replace=False)
        p, t = p[idx], t[idx]
    zp, zt = joint_pca(p, t, n_pcs=n_pcs, seed=seed)
    return mmd_rbf(zt, zp, **kwargs)


def _simpson_from_dists(dists: np.ndarray, nbr_labels: np.ndarray, n_labels: int,
                        perplexity: float, tol: float = 1e-5, n_iter: int = 50) -> np.ndarray:
    """Inverse Simpson index per cell with Gaussian weights calibrated to ``perplexity``.

    Same calibration as Korsunsky et al. (2019, harmony's ``compute_lisi``):
    a per-cell bisection on the kernel precision until the entropy of the
    neighbour weights equals ``log(perplexity)``.
    """
    n = dists.shape[0]
    logU = np.log(perplexity)
    out = np.empty(n)
    for i in range(n):
        d = dists[i]
        beta, betamin, betamax = 1.0, -np.inf, np.inf
        for _ in range(n_iter):
            P = np.exp(-d * beta)
            sP = P.sum()
            if sP == 0:
                H, P = 0.0, np.full_like(d, 1.0 / len(d))
            else:
                H = np.log(sP) + beta * np.sum(d * P) / sP
                P = P / sP
            diff = H - logU
            if abs(diff) < tol:
                break
            if diff > 0:
                betamin = beta
                beta = beta * 2 if betamax == np.inf else (beta + betamax) / 2
            else:
                betamax = beta
                beta = beta / 2 if betamin == -np.inf else (beta + betamin) / 2
        w = np.bincount(nbr_labels[i], weights=P, minlength=n_labels)
        out[i] = 1.0 / np.sum(w * w)
    return out


def lisi(embedding: np.ndarray, labels: Sequence, *, perplexity: float = 30.0, n_neighbors: Optional[int] = None) -> np.ndarray:
    """Per-cell LISI of ``labels`` in ``embedding`` (Korsunsky et al., 2019).

    Uses the ``3 * perplexity`` nearest neighbours (excluding the cell itself)
    with Euclidean distances, as harmony's ``compute_lisi``.
    """
    from sklearn.neighbors import NearestNeighbors

    emb = np.asarray(embedding, dtype=np.float64)
    codes = np.unique(np.asarray(labels).astype(str), return_inverse=True)[1]
    k = int(n_neighbors or 3 * perplexity)
    k = min(k, len(emb) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    dist, idx = nn.kneighbors(emb)
    dist, idx = dist[:, 1:], idx[:, 1:]
    return _simpson_from_dists(dist, codes[idx], int(codes.max()) + 1, perplexity)


def _graph_knn(conn: sp.csr_matrix, k: int, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` nearest cells of every cell by shortest-path length on ``conn``.

    Edge lengths are the graph's weights (scib uses the kNN connectivities
    as lengths). Rows with fewer than ``k`` reachable cells are padded with
    ``-1`` / ``nan``.
    """
    from scipy.sparse.csgraph import dijkstra

    n = conn.shape[0]
    idx = np.full((n, k), -1, dtype=np.int64)
    dist = np.full((n, k), np.nan)
    for s0 in range(0, n, chunk):
        rows = np.arange(s0, min(n, s0 + chunk))
        D = dijkstra(conn, directed=True, indices=rows)
        D[np.arange(len(rows)), rows] = np.inf  # exclude the cell itself
        kk = min(k, n - 1)
        part = np.argpartition(D, kk - 1, axis=1)[:, :kk]
        pd_ = np.take_along_axis(D, part, axis=1)
        order = np.argsort(pd_, axis=1, kind="stable")
        part = np.take_along_axis(part, order, axis=1)
        pd_ = np.take_along_axis(pd_, order, axis=1)
        ok = np.isfinite(pd_)
        idx[rows[:, None].repeat(kk, 1)[ok], np.nonzero(ok)[1]] = part[ok]
        dist[rows[:, None].repeat(kk, 1)[ok], np.nonzero(ok)[1]] = pd_[ok]
    return idx, dist


def _simpson_graph(dist, idx, labels, n_labels, perplexity, tol=1e-5, n_iter=50):
    """scib's ``compute_simpson_index_graph``: Simpson index per cell from path lengths.

    A cell without ``k`` reachable neighbours gets 1; a degenerate kernel
    (entropy 0) gets -1, as in scib.
    """
    logU = np.log(perplexity)
    out = np.empty(len(dist))
    for i in range(len(dist)):
        if (idx[i] < 0).any():
            out[i] = 1.0
            continue
        D = np.nan_to_num(dist[i].astype(np.float64), nan=0.0)

        def hbeta(beta):
            P = np.exp(-D * beta)
            sP = P.sum()
            if sP == 0:
                return 0.0, np.zeros_like(D)
            return np.log(sP) + beta * np.sum(D * P) / sP, P / sP

        beta, bmin, bmax = 1.0, -np.inf, np.inf
        H, P = hbeta(beta)
        diff, tries = H - logU, 0
        while abs(diff) > tol and tries < n_iter:
            if diff > 0:
                bmin = beta
                beta = beta * 2 if bmax == np.inf else (beta + bmax) / 2
            else:
                bmax = beta
                beta = beta / 2 if bmin == -np.inf else (beta + bmin) / 2
            H, P = hbeta(beta)
            diff, tries = H - logU, tries + 1
        if H == 0:
            out[i] = -1.0
            continue
        w = np.bincount(labels[idx[i]], weights=P, minlength=n_labels)
        out[i] = float(np.dot(w, w))
    return out


def ilisi_graph(embedding: np.ndarray, labels: Sequence, *, k0: int = 90, n_neighbors: int = 15,
                scale: bool = True, seed: int = 0) -> float:
    """scib's ``ilisi_graph(type_="knn")`` on a scanpy kNN graph of ``embedding``.

    A ``n_neighbors`` graph is built with ``sc.pp.neighbors`` (UMAP
    connectivities); each cell's ``k0`` nearest cells by shortest-path length
    over the connectivities define its neighbourhood; the Simpson index of
    ``labels`` is computed with perplexity ``floor(k0 / 3)``. Returns the
    median iLISI, scaled to ``(lisi - 1) / (n_labels - 1)`` if ``scale``.
    """
    import scanpy as sc
    from anndata import AnnData

    emb = np.asarray(embedding, dtype=np.float32)
    codes = np.unique(np.asarray(labels).astype(str), return_inverse=True)[1]
    n_lab = int(codes.max()) + 1
    ad = AnnData(X=np.zeros((len(emb), 1), dtype=np.float32))
    ad.obsm["X_pca"] = emb
    sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep="X_pca", random_state=seed)
    conn = sp.csr_matrix(ad.obsp["connectivities"], dtype=np.float64)
    conn.data = np.maximum(conn.data, 3e-308)
    idx, dist = _graph_knn(conn, k0)
    simpson = _simpson_graph(dist, idx, codes, n_lab, float(np.floor(k0 / 3)))
    v = float(np.nanmedian(1.0 / simpson))
    return (v - 1.0) / (n_lab - 1) if scale else v


def ilisi(pred, truth, *, n_pcs: Optional[int] = 50, method: str = "graph", perplexity: float = 30.0,
          scale: bool = True, seed: int = 0) -> float:
    """Median integration LISI of measured vs predicted cells in their joint PCA space.

    Higher means predicted cells mix with measured cells. ``method="graph"``
    (default) is scib's kNN-graph iLISI as used by scTranslation (see
    :func:`ilisi_graph`); ``"embedding"`` is harmony's LISI on Euclidean
    neighbours (:func:`lisi`). With ``scale`` the value is mapped to [0, 1].
    """
    zp, zt = joint_pca(pred, truth, n_pcs=n_pcs, seed=seed)
    lab = np.r_[np.zeros(len(zt), int), np.ones(len(zp), int)]
    Z = np.vstack([zt, zp])
    if method == "graph":
        return ilisi_graph(Z, lab, scale=scale, seed=seed)
    v = float(np.median(lisi(Z, lab, perplexity=perplexity)))
    return v - 1.0 if scale else v


# ------------------------------------------------------------ classification
def binarize_truth(truth, how="median") -> np.ndarray:
    """Binary target for AUROC/AUPRC.

    ``"median"`` -- above the median of the whole measured matrix (the
    scTranslation convention); ``"nonzero"`` -- any positive value; a number
    -- above that threshold.
    """
    t = _array(truth)
    if how == "median":
        thr = np.median(t)
    elif how == "nonzero":
        thr = 0.0
    else:
        thr = float(how)
    return (t > thr).astype(np.int8)


def _score_binary(fn, pred, truth, binarize, average, convention="standard"):
    p, t = _pair(pred, truth)
    if convention == "sctranslation":
        # the benchmark passes (truth, pred) to a function expecting (pred, truth):
        # the prediction is binarised and the measured values are the scores
        p, y = t, binarize_truth(p, binarize)
    elif convention == "standard":
        y = binarize_truth(t, binarize)
    else:
        raise ValueError(f"convention must be 'standard' or 'sctranslation', not {convention!r}")
    if average == "global":
        if y.min() == y.max():
            return float("nan")
        return float(fn(y.ravel(), p.ravel()))
    axis_vals = []
    P, Y = (p, y) if average == "cell" else (p.T, y.T)
    for pi, yi in zip(P, Y):
        if yi.min() != yi.max():
            axis_vals.append(fn(yi, pi))
    return float(np.mean(axis_vals)) if axis_vals else float("nan")


def auroc(pred, truth, *, binarize="median", average: str = "global", convention: str = "standard") -> float:
    """Area under the ROC curve of ``pred`` for the binarised ``truth``.

    ``average="global"`` pools every entry (scTranslation); ``"feature"`` /
    ``"cell"`` average the per-feature / per-cell AUROC over those with both
    classes present.

    ``convention="sctranslation"`` reproduces the benchmark's code, which swaps
    the two roles: the *prediction* is binarised at its median and the measured
    values are the scores. That number is much lower (about 0.57 vs 0.89 for
    scButterfly on PBMC 10k) and is provided only for comparison with
    published tables.
    """
    from sklearn.metrics import roc_auc_score

    return _score_binary(roc_auc_score, pred, truth, binarize, average, convention)


def auprc(pred, truth, *, binarize="median", average: str = "global", convention: str = "standard") -> float:
    """Average precision (area under the precision-recall curve); see :func:`auroc`."""
    from sklearn.metrics import average_precision_score

    return _score_binary(average_precision_score, pred, truth, binarize, average, convention)
