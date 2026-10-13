"""Doublet detection for scATAC-seq.

Four entry points share this module:

* :func:`archr_doublets` ports ArchR's ``addDoubletScores``: simulated doublets
  projected into an iterative-LSI embedding, scored by how often they land next
  to each cell.
* :func:`filter_doublets` removes cells, by ArchR's ``filterDoublets`` rule or by
  a threshold on any score column written here or by :func:`epione.pp.scrublet`.
* :func:`scdblfinder`, :func:`amulet` and :func:`clamulet` are thin wrappers over
  `pyscdblfinder <https://github.com/omicverse/py-scDblFinder>`_ (>= 0.3.0), the
  omicverse port of R scDblFinder, which owns their numerical parity.
* :func:`combine_doublet_scores` is the Fisher p-value combination of AMULET and
  scDblFinder that the scDblFinder vignette recommends for scATAC-seq.

The ArchR port follows ArchR 1.0.3 (MIT License, Copyright (c) 2020 Jeffrey
Granja and Ryan Corces): ``addDoubletScores``, ``.simulateProjectDoublets``,
``addIterativeLSI``, ``.computeLSI``, ``.projectLSI``, ``addClusters`` and
``filterDoublets``. Parity is metric-level (see
``tests/parity/doublets/MANIFEST.md``): the R and Python random number
generators, irlba, Seurat's Louvain and uwot have no bit-exact Python
counterparts.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ..utils import console


# ---------------------------------------------------------------------------
# ArchR LSI (``.computeLSI`` / ``.projectLSI``), features x cells as in ArchR
# ---------------------------------------------------------------------------

def _binarize_csc(mat: sp.csc_matrix) -> sp.csc_matrix:
    mat = mat.tocsc(copy=True).astype(np.float64)
    mat.data[mat.data > 0] = 1.0
    mat.eliminate_zeros()
    return mat


def _tfidf(mat: sp.csc_matrix, n_col: float, row_sums: np.ndarray, lsi_method: int,
           scale_to: float) -> sp.csc_matrix:
    """ArchR's term frequency / inverse document frequency, features x cells."""
    col_sums = np.asarray(mat.sum(axis=0)).ravel()
    mat = mat @ sp.diags(1.0 / col_sums)
    if lsi_method == 1:
        idf = np.log(1.0 + n_col / row_sums)
        mat = sp.diags(idf) @ mat
    elif lsi_method == 2:
        mat = sp.diags(n_col / row_sums) @ mat
        mat = mat.tocsc()
        mat.data = np.log(mat.data * scale_to + 1.0)
    elif lsi_method == 3:
        mat = mat.tocsc()
        mat.data = np.log(mat.data + 1.0)
        mat = sp.diags(np.log(1.0 + n_col / row_sums)) @ mat
    else:
        raise ValueError("lsi_method must be 1, 2 or 3")
    return sp.csc_matrix(mat)


def _truncated_svd(mat, n_dims: int, seed: int):
    """Top singular triplets (irlba in ArchR). Signs are made deterministic."""
    from sklearn.utils.extmath import randomized_svd
    u, d, vt = randomized_svd(mat, n_components=n_dims, n_oversamples=20, n_iter=10,
                              random_state=seed)
    # Flip so the largest-|u| entry of each component is positive.
    sign = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])])
    sign[sign == 0] = 1
    return u * sign, d, (vt.T * sign)


def _compute_lsi(mat: sp.csc_matrix, n_dims: int, lsi_method: int = 1,
                 scale_to: float = 1e4, seed: int = 1) -> dict:
    """``.computeLSI`` with ``binarize=TRUE`` and ``outlierQuantiles=NULL``.

    ``mat`` is features x cells. Cells with no counts in the features are
    excluded, as in ArchR.
    """
    mat = _binarize_csc(mat)
    col_sums = np.asarray(mat.sum(axis=0)).ravel()
    keep_cells = np.where(col_sums > 0)[0]
    mat = mat[:, keep_cells]
    row_sums = np.asarray(mat.sum(axis=1)).ravel()
    idx = np.where(row_sums > 0)[0]
    mat = mat[idx, :]
    row_sums = row_sums[idx]
    n_col = mat.shape[1]
    tfidf = _tfidf(mat, n_col, row_sums, lsi_method, scale_to)
    n_dims = int(min(n_dims, min(tfidf.shape) - 1))
    u, d, v = _truncated_svd(tfidf, n_dims, seed)
    return dict(mat_svd=v * d, cells=keep_cells, idx=idx, row_sums=row_sums,
                n_col=n_col, u=u, d=d, lsi_method=lsi_method, scale_to=scale_to)


def _project_lsi(mat: sp.csc_matrix, lsi: dict):
    """``.projectLSI``: returns (cells kept, cells x dims projection)."""
    mat = _binarize_csc(sp.csc_matrix(mat)[lsi["idx"], :])
    col_sums = np.asarray(mat.sum(axis=0)).ravel()
    keep = np.where(col_sums > 0)[0]
    mat = mat[:, keep]
    tfidf = _tfidf(mat, lsi["n_col"], lsi["row_sums"], lsi["lsi_method"], lsi["scale_to"])
    # V = t(mat) %*% u %*% diag(1/d); matSVD = V %*% diag(d) = t(mat) %*% u
    return keep, np.asarray(tfidf.T @ lsi["u"])


def _abs_cor(mat_svd: np.ndarray, depth: np.ndarray) -> np.ndarray:
    x = depth - depth.mean()
    y = mat_svd - mat_svd.mean(axis=0)
    num = x @ y
    den = np.sqrt((x ** 2).sum() * (y ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return np.abs(np.nan_to_num(r))


# ---------------------------------------------------------------------------
# ArchR clustering (``addClusters``, Seurat method) on a cells x dims matrix
# ---------------------------------------------------------------------------

def _knn_index(data: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Exact Euclidean kNN of ``query`` rows among ``data`` rows (``nabor::knn``)."""
    from sklearn.neighbors import NearestNeighbors
    k = int(min(k, data.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(data)
    return nn.kneighbors(query, return_distance=False)


def _seurat_louvain(mat: np.ndarray, resolution: float, n_start: int, seed: int,
                    k_param: int = 20, prune_snn: float = 1 / 15) -> np.ndarray:
    """Seurat ``FindNeighbors`` (SNN, Jaccard, prune 1/15) + ``FindClusters``.

    Louvain is run ``n_start`` times and the partition with the highest
    modularity is kept, as Seurat does. igraph's multilevel algorithm replaces
    Seurat's Java-derived implementation.
    """
    import igraph as ig
    n = mat.shape[0]
    k = int(min(k_param, n))
    nn = _knn_index(mat, mat, k)
    rows = np.repeat(np.arange(n), k)
    adj = sp.csr_matrix((np.ones(n * k), (rows, nn.ravel())), shape=(n, n))
    snn = (adj @ adj.T).tocoo()
    w = snn.data / (k + (k - snn.data))
    keep = (w >= prune_snn) & (snn.row < snn.col)
    g = ig.Graph(n=n, edges=list(zip(snn.row[keep], snn.col[keep])), directed=False)
    g.es["weight"] = w[keep]
    best, best_q = None, -np.inf
    rng = np.random.RandomState(seed)
    # igraph draws from Python's ``random`` module; seed it so the result is reproducible.
    import random
    state = random.getstate()
    random.seed(seed)
    try:
        for _ in range(int(n_start)):
            perm = rng.permutation(n)
            gp = g.permute_vertices(perm.tolist())
            part = gp.community_multilevel(weights="weight", resolution=resolution)
            memb = np.asarray(part.membership)[perm]
            q = g.modularity(memb.tolist(), weights="weight", resolution=resolution)
            if q > best_q:
                best, best_q = memb, q
    finally:
        random.setstate(state)
    return best


def _archr_clusters(mat: np.ndarray, *, resolution: float = 2.0, sample_cells: Optional[int] = 10000,
                    max_clusters: Optional[int] = 6, n_outlier: int = 5, knn_assign: int = 10,
                    n_start: int = 10, seed: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n = mat.shape[0]
    if sample_cells is not None and sample_cells < n:
        idx = np.sort(rng.choice(n, sample_cells, replace=False))
        sub = _seurat_louvain(mat[idx], resolution, n_start, seed)
        rest = np.setdiff1d(np.arange(n), idx)
        knn = _knn_index(mat[idx], mat[rest], knn_assign)
        labels = np.empty(n, dtype=np.int64)
        labels[idx] = sub
        uniq = np.unique(sub)
        votes = np.stack([(sub[knn] == u).sum(axis=1) for u in uniq], axis=1)
        labels[rest] = uniq[np.argmax(votes, axis=1)]
    else:
        labels = _seurat_louvain(mat, resolution, n_start, seed)
    # Clusters smaller than nOutlier are reassigned by kNN majority vote.
    uniq, counts = np.unique(labels, return_counts=True)
    for c in uniq[counts < n_outlier]:
        idxi = np.where(labels == c)[0]
        others = np.where(labels != c)[0]
        if len(others) == 0:
            continue
        knn = _knn_index(mat[others], mat[idxi], knn_assign)
        for row, nbrs in zip(idxi, knn):
            vals, cnt = np.unique(labels[others][nbrs], return_counts=True)
            labels[row] = vals[np.argmax(cnt)]
    # More than maxClusters: hclust (complete linkage) of cluster means, cut.
    uniq = np.unique(labels)
    if max_clusters is not None and len(uniq) > max_clusters:
        from scipy.cluster.hierarchy import fcluster, linkage
        means = np.stack([mat[labels == u].mean(axis=0) for u in uniq])
        cut = fcluster(linkage(means, method="complete", metric="euclidean"),
                       t=max_clusters, criterion="maxclust")
        labels = cut[np.searchsorted(uniq, labels)]
    return labels


# ---------------------------------------------------------------------------
# ArchR iterative LSI, as configured by addDoubletScores
# ---------------------------------------------------------------------------

def _archr_iterative_lsi(X: sp.csr_matrix, depth: np.ndarray, *, dims: int = 30,
                         lsi_method: int = 1, cor_cut_off: float = 0.75,
                         iterations: int = 2, var_features: int = 25000,
                         total_features: int = 500000, filter_quantile: float = 0.995,
                         sample_cells_pre: Optional[int] = 10000,
                         cluster_resolution: float = 2.0, max_clusters: int = 6,
                         scale_to: float = 1e4, seed: int = 1) -> dict:
    """ArchR ``addIterativeLSI`` with addDoubletScores' ``LSIParams``.

    ``X`` is cells x features. Returns the final LSI model with ``features``
    (column indices into ``X``), ``dims_kept`` and the kept-dims embedding.
    """
    rng = np.random.RandomState(seed)
    n_cells = X.shape[0]
    Xb = X.tocsc(copy=True).astype(np.float64)
    Xb.data[:] = 1.0
    total_acc = np.asarray(Xb.sum(axis=0)).ravel()
    order = np.argsort(-total_acc, kind="stable")
    rm_top = int(np.floor((1 - filter_quantile) * total_features))
    if (total_acc > 0).sum() > 2.25 * var_features:
        top = order[rm_top: rm_top + var_features]
    else:
        top = order[:var_features]
    features = np.sort(top)
    features = features[total_acc[features] > 0]
    if len(features) < var_features:
        raise ValueError(f"Not enough features ({len(features)}) for {var_features} varFeatures")
    log_depth = np.log10(depth + 1.0)
    if sample_cells_pre is not None and n_cells < sample_cells_pre:
        sample_cells_pre = None

    lsi = None
    for j in range(1, iterations + 1):
        final = j == iterations
        if not final and sample_cells_pre is not None:
            cells = np.sort(rng.choice(n_cells, sample_cells_pre, replace=False))
        else:
            cells = np.arange(n_cells)
        lsi = _compute_lsi(Xb[cells][:, features].T.tocsc(), dims, lsi_method, scale_to, seed)
        lsi_cells = cells[lsi["cells"]]
        cor = _abs_cor(lsi["mat_svd"], log_depth[lsi_cells])
        lsi.update(features=features, lsi_cells=lsi_cells, cor_to_depth=cor)
        if final:
            break
        dims_pf = np.where(cor <= cor_cut_off)[0]
        if len(dims_pf) < 2:
            raise ValueError("Fewer than 2 LSI dims left after the depth-correlation filter")
        clusters = _archr_clusters(lsi["mat_svd"][:, dims_pf], resolution=cluster_resolution,
                                   max_clusters=max_clusters, seed=seed)
        if len(np.unique(clusters)) == 1:
            continue
        group_features = np.sort(order[:total_features])
        Xg = Xb[lsi_cells][:, group_features]
        onehot = sp.csr_matrix((np.ones(len(clusters)), (clusters, np.arange(len(clusters)))))
        onehot = onehot[np.unique(clusters)]
        gmat = np.asarray((onehot @ Xg).todense()).T  # features x groups
        gmat = np.log2(gmat / gmat.sum(axis=0, keepdims=True) * scale_to + 1.0)
        var = gmat.var(axis=1, ddof=1)
        top = np.argsort(-var, kind="stable")[:var_features]
        features = group_features[np.sort(top)]
    dims_kept = np.where(lsi["cor_to_depth"] <= cor_cut_off)[0]
    if len(dims_kept) < 2:
        raise ValueError("Reduced LSI dims below 2; increase dims or cor_cut_off")
    lsi["dims_kept"] = dims_kept
    return lsi


def _binom_scores(count_knn: np.ndarray, n_cells: int):
    """ArchR's binomial doublet score and enrichment from neighbour counts."""
    from scipy.stats import binom
    scale_to = 10000.0
    scale_by = scale_to / n_cells
    x = np.round(count_knn * scale_by)
    size = np.round(count_knn.sum() * scale_by)
    p = binom.sf(x - 1, size, 1.0 / scale_to)
    padj = np.minimum(p * len(p), 1.0)
    score = -np.log10(np.maximum(padj, 4.940656e-324))
    total = count_knn.sum()
    enrich = 10000.0 * count_knn / total if total > 0 else np.zeros_like(count_knn, dtype=float)
    return score, enrich


def _archr_doublets_one(X: sp.csr_matrix, depth: np.ndarray, *, k: int, n_trials: int,
                        n_sample: int, dims: int, lsi_method: int, cor_cut_off: float,
                        knn_method: str, umap_kwargs: dict, force: bool, seed: int,
                        lsi_kwargs: dict) -> dict:
    import umap
    n_cells = X.shape[0]
    lsi = _archr_iterative_lsi(X, depth, dims=dims, lsi_method=lsi_method,
                               cor_cut_off=cor_cut_off, seed=seed, **lsi_kwargs)
    kept = lsi["dims_kept"]
    mat = X[:, lsi["features"]].T.tocsc()  # features x cells, ArchR orientation
    emb_cells = lsi["lsi_cells"]
    emb = lsi["mat_svd"][:, kept]
    params = dict(n_neighbors=40, min_dist=0.4, metric="euclidean")
    params.update(umap_kwargs or {})
    model = umap.UMAP(n_components=2, random_state=seed, **params).fit(emb)
    umap_orig = model.embedding_

    rng = np.random.RandomState(seed)
    total_trials = n_trials * max(n_cells // n_sample, 1)
    sim = []
    for _ in range(total_trials):
        idx1 = rng.randint(0, n_cells, n_sample)
        idx2 = rng.randint(0, n_cells, n_sample)
        parts = []
        for idx in (idx1, idx2):
            sub = mat[:, idx].tocsc(copy=True)
            nnz = sub.nnz
            drop = rng.choice(nnz, int(np.floor(nnz * 0.5)), replace=False)
            sub.data[drop] = 0
            sub.eliminate_zeros()
            parts.append(sub)
        _, proj = _project_lsi(parts[0] + parts[1], lsi)
        sim.append(proj)
    sim = np.vstack(sim)[:, kept]
    og_keep, og = _project_lsi(mat, lsi)
    og = og[:, kept]
    all_lsi = np.vstack([sim, og])
    proj_umap = model.transform(all_lsi)
    n_sim = sim.shape[0]
    og_umap = proj_umap[n_sim:]
    # Correlation of the re-projected original cells with the fitted UMAP.
    pos = pd.Index(og_keep).get_indexer(emb_cells)
    ok = pos >= 0
    cor_umap = np.array([np.corrcoef(umap_orig[ok, d], og_umap[pos[ok], d])[0, 1] for d in (0, 1)])
    out = dict(cor_umap=cor_umap, n_sim=n_sim, dims_kept=kept, umap=umap_orig,
               umap_cells=emb_cells, doublet_umap=proj_umap[:n_sim])
    score = np.full(n_cells, -1.0)
    enrich = np.full(n_cells, -1.0)
    if cor_umap.mean() < 0.9 and not force:
        console.warn("[archr_doublets] UMAP projection correlation "
                     f"{cor_umap.mean():.3f} < 0.9: little heterogeneity, returning -1 "
                     "(set force=True to score anyway, as in ArchR)")
        out.update(score=score, enrichment=enrich)
        return out
    if knn_method.lower() == "lsi":
        data, query = og, sim
    else:
        data, query = og_umap, proj_umap[:n_sim]
    knn = _knn_index(data, query, k)
    count = np.bincount(knn.ravel(), minlength=data.shape[0]).astype(float)
    # ArchR scores every LSI cell; cells dropped by the projection get 0 counts.
    count_all = np.zeros(len(emb_cells))
    count_all[ok] = count[pos[ok]]
    s, e = _binom_scores(count_all, len(emb_cells))
    score[emb_cells] = s
    enrich[emb_cells] = e
    out.update(score=score, enrichment=enrich)
    return out


def _get_matrix(adata: AnnData, layer: Optional[str]):
    X = adata.layers[layer] if layer is not None else adata.X
    if type(X).__name__ in ("BackedArray", "_SubsetBackedArray"):
        X = sp.vstack([c for _, _, c in X.chunked()])
    return sp.csr_matrix(X)


def archr_doublets(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
    sample_key: Optional[str] = None,
    depth_key: Optional[str] = "n_fragment",
    k: int = 10,
    n_trials: int = 5,
    n_sample: int = 1000,
    dims: int = 30,
    lsi_method: int = 1,
    cor_cut_off: float = 0.75,
    knn_method: str = "umap",
    umap_kwargs: Optional[dict] = None,
    iterations: int = 2,
    var_features: int = 25000,
    total_features: int = 500000,
    filter_quantile: float = 0.995,
    sample_cells_pre: Optional[int] = 10000,
    cluster_resolution: float = 2.0,
    max_clusters: int = 6,
    force: bool = False,
    seed: int = 1,
    key_added: str = "archr_doublet",
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    """ArchR doublet scores (port of ``addDoubletScores``, ArchR 1.0.3).

    For each sample separately: iterative LSI on the binarised tile (or peak)
    matrix, then ``n_trials * max(n_cells // 1000, 1)`` rounds of 1000 synthetic
    doublets. Each is the sum of two random cells after keeping half of each
    cell's non-zero entries. The doublets are projected into the LSI and into a
    UMAP fitted on the real cells. Each cell is scored by how many simulated
    doublets have it among their ``k`` nearest real cells.

    Parameters
    ----------
    adata
        Cells x 500-bp tiles (ArchR's default input) or peaks. Counts are
        binarised.
    layer
        Use ``adata.layers[layer]`` instead of ``adata.X``.
    sample_key
        ``adata.obs`` column with the capture/sample. Doublets only form
        within a capture, so scores are computed per sample, as ArchR does per
        Arrow file. ``None`` treats all cells as one sample.
    depth_key
        ``adata.obs`` column with fragments per cell (ArchR's ``nFrags``),
        used to drop LSI dims correlated with depth. Falls back to the
        matrix row sums if absent.
    k, n_trials, n_sample, dims, lsi_method, cor_cut_off
        ArchR's ``k``, ``nTrials``, ``nSample``, ``max(dimsToUse)``,
        ``LSIMethod`` and ``corCutOff``.
    knn_method
        ``"umap"`` (ArchR default) or ``"lsi"``: the space in which neighbours
        of simulated doublets are counted.
    umap_kwargs
        Overrides for umap-learn (defaults: 40 neighbours, ``min_dist=0.4``,
        euclidean, as in ArchR).
    iterations, var_features, total_features, filter_quantile, sample_cells_pre,
    cluster_resolution, max_clusters
        Iterative-LSI settings, defaulting to ArchR's.
    force
        If the re-projected UMAP correlates below 0.9 with the fitted one,
        ArchR returns -1 (too little heterogeneity). ``force=True`` scores
        anyway.
    seed
        Random seed.
    key_added
        Writes ``obs[f"{key_added}_score"]`` (-log10 Bonferroni p, ArchR's
        ``DoubletScore``) and ``obs[f"{key_added}_enrichment"]``
        (``DoubletEnrichment``).
    inplace
        If False, return the scores as a DataFrame instead.

    Returns
    -------
    ``None`` (``inplace=True``) or a DataFrame indexed like ``adata.obs``.

    Notes
    -----
    Parity with ArchR is metric-level, not bit-exact: R's RNG, irlba,
    Seurat's Louvain and uwot are replaced by NumPy, a randomized SVD, igraph
    and umap-learn. The gates, and how they compare with ArchR's own
    seed-to-seed variation, are in ``tests/parity/doublets/MANIFEST.md``.

    Examples
    --------
    >>> epi.pp.add_tile_matrix(adata, bin_size=500)
    >>> epi.pp.archr_doublets(adata, sample_key="sample")
    >>> epi.pp.filter_doublets(adata, method="archr", sample_key="sample")
    """
    X = _get_matrix(adata, layer)
    if depth_key is not None and depth_key in adata.obs:
        depth = adata.obs[depth_key].to_numpy(dtype=float)
    else:
        depth = np.asarray(X.sum(axis=1)).ravel().astype(float)
    groups = (adata.obs[sample_key].astype(str).to_numpy() if sample_key is not None
              else np.zeros(adata.n_obs, dtype=object))
    score = np.full(adata.n_obs, -1.0)
    enrich = np.full(adata.n_obs, -1.0)
    info = {}
    lsi_kwargs = dict(iterations=iterations, var_features=var_features,
                      total_features=total_features, filter_quantile=filter_quantile,
                      sample_cells_pre=sample_cells_pre,
                      cluster_resolution=cluster_resolution, max_clusters=max_clusters)
    for g in pd.unique(groups):
        idx = np.where(groups == g)[0]
        console.level1(f"[archr_doublets] sample {g}: {len(idx):,} cells")
        res = _archr_doublets_one(X[idx], depth[idx], k=k, n_trials=n_trials,
                                  n_sample=n_sample, dims=dims, lsi_method=lsi_method,
                                  cor_cut_off=cor_cut_off, knn_method=knn_method,
                                  umap_kwargs=umap_kwargs, force=force, seed=seed,
                                  lsi_kwargs=lsi_kwargs)
        score[idx] = res["score"]
        enrich[idx] = res["enrichment"]
        info[str(g)] = dict(umap_projection_r2=float(res["cor_umap"].mean() ** 2),
                            n_simulated=int(res["n_sim"]),
                            dims_kept=(res["dims_kept"] + 1).tolist())
    df = pd.DataFrame({f"{key_added}_score": score, f"{key_added}_enrichment": enrich},
                      index=adata.obs_names)
    if not inplace:
        return df
    for c in df.columns:
        adata.obs[c] = df[c].to_numpy()
    adata.uns[key_added] = info
    console.level2(f"Added obs['{key_added}_score'] and obs['{key_added}_enrichment']")
    return None


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

def _archr_filter_mask(enrich: np.ndarray, score: np.ndarray, groups: np.ndarray,
                       cut_enrich: Optional[float], cut_score: Optional[float],
                       filter_ratio: float) -> np.ndarray:
    """Cells ArchR's ``filterDoublets`` removes (True = remove)."""
    remove = np.zeros(len(enrich), dtype=bool)
    for g in pd.unique(groups):
        idx = np.where(groups == g)[0]
        n = len(idx)
        e = enrich[idx]
        # order(decreasing=TRUE): stable, NA last
        o = idx[np.argsort(np.where(np.isnan(e), np.inf, -e), kind="stable")]
        if cut_enrich is not None:
            o = o[enrich[o] >= cut_enrich]
        if cut_score is not None:
            o = o[score[o] >= cut_score]
        n_keep = int(filter_ratio * n * (n / 100000))  # R's head() truncates
        remove[o[:max(n_keep, 0)]] = True
    return remove


def filter_doublets(
    adata: AnnData,
    method: str = "archr",
    *,
    score_key: Optional[str] = None,
    threshold: Optional[float] = None,
    sample_key: Optional[str] = None,
    key_added: Optional[str] = "doublet_removed",
    cut_enrich: Optional[float] = 1.0,
    cut_score: Optional[float] = -np.inf,
    filter_ratio: float = 1.0,
    archr_key: str = "archr_doublet",
    inplace: bool = True,
) -> Optional[AnnData]:
    """Remove doublets flagged by any Epione doublet method.

    Parameters
    ----------
    adata
        AnnData carrying doublet scores in ``.obs``.
    method
        How to decide which cells to remove:

        - ``"archr"``: ArchR's ``filterDoublets``. Within each sample, rank
          cells by ``{archr_key}_enrichment``, keep those with enrichment
          ``>= cut_enrich`` and score ``>= cut_score``, and remove the top
          ``filter_ratio * n² / 100000`` of them. This expects about
          ``n/100000`` doublets per cell loaded, the 10x rule of thumb.
        - ``"threshold"``: remove cells with ``obs[score_key] >= threshold``.
          For p- or q-value columns (``amulet_q``, ``doublet_combined_p``),
          cells with values ``<= threshold`` are removed instead.
        - ``"class"``: remove cells where ``obs[score_key]`` is ``"doublet"``
          or ``True``, e.g. ``scDblFinder_class``, scrublet's ``is_doublet``
          or ``amulet_class``.
    score_key
        Column for ``"threshold"`` and ``"class"``.
    threshold
        Cut-off for ``"threshold"``.
    sample_key
        Sample column for ``"archr"``. ``None`` treats all cells as one
        sample.
    key_added
        If set, the boolean removal mask is stored in ``adata.obs[key_added]``
        of the input before subsetting.
    cut_enrich, cut_score, filter_ratio, archr_key
        ArchR ``filterDoublets`` settings and the prefix written by
        :func:`archr_doublets`.
    inplace
        Subset ``adata`` in place (default). If False, return a filtered
        copy.

    Returns
    -------
    ``None`` or the filtered copy.
    """
    obs = adata.obs
    if method == "archr":
        e_key, s_key = f"{archr_key}_enrichment", f"{archr_key}_score"
        if e_key not in obs:
            raise KeyError(f"obs['{e_key}'] not found; run epi.pp.archr_doublets first")
        groups = (obs[sample_key].astype(str).to_numpy() if sample_key is not None
                  else np.zeros(adata.n_obs, dtype=object))
        score = obs[s_key].to_numpy(dtype=float) if s_key in obs else np.zeros(adata.n_obs)
        remove = _archr_filter_mask(obs[e_key].to_numpy(dtype=float), score, groups,
                                    cut_enrich, cut_score, filter_ratio)
    elif method == "threshold":
        if score_key is None or threshold is None:
            raise ValueError("method='threshold' needs score_key and threshold")
        v = obs[score_key].to_numpy(dtype=float)
        is_p = score_key.endswith(("_p", "_q", "_pvalue", "_qvalue", "p.value", "q.value"))
        remove = (v <= threshold) if is_p else (v >= threshold)
    elif method == "class":
        if score_key is None:
            raise ValueError("method='class' needs score_key")
        v = obs[score_key]
        remove = (v.astype(str).str.lower() == "doublet").to_numpy() | (v.astype(str) == "True").to_numpy()
    else:
        raise ValueError("method must be 'archr', 'threshold' or 'class'")
    remove = np.asarray(remove, dtype=bool)
    console.level1(f"[filter_doublets] removing {remove.sum():,} of {adata.n_obs:,} cells "
                   f"({100 * remove.mean():.1f}%)")
    if key_added:
        adata.obs[key_added] = remove
    if inplace:
        adata._inplace_subset_obs(~remove)
        return None
    return adata[~remove].copy()


# ---------------------------------------------------------------------------
# pyscdblfinder wrappers (scDblFinder, AMULET, clamulet)
# ---------------------------------------------------------------------------

_PYSCDBL_MIN = "0.3.0"


def _pyscdblfinder(feature: str):
    try:
        mod = importlib.import_module("pyscdblfinder")
    except ImportError as exc:  # pragma: no cover - message checked in tests
        raise ImportError(
            f"epi.pp.{feature} needs pyscdblfinder>={_PYSCDBL_MIN} "
            "(pip install 'epione[doublets]' or pip install pyscdblfinder)"
        ) from exc
    ver = getattr(mod, "__version__", "0")
    from packaging.version import Version
    if Version(ver) < Version(_PYSCDBL_MIN):
        raise ImportError(f"epi.pp.{feature} needs pyscdblfinder>={_PYSCDBL_MIN}, found {ver}; "
                          "its scATAC support (aggregateFeatures, amulet, clamulet) starts at 0.3.0")
    return mod


def _call(fn: Callable, *args, **kwargs):
    """Call ``fn`` passing only the keyword arguments it accepts."""
    sig = inspect.signature(fn)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*args, **kwargs)
    unknown = [k for k in kwargs if k not in sig.parameters]
    if unknown:
        raise TypeError(f"{fn.__module__}.{fn.__name__} does not accept {unknown}")
    return fn(*args, **kwargs)


def _barcodes(adata: AnnData, barcode_key: Optional[str]) -> np.ndarray:
    """Fragment-file barcodes for each cell (``sample#barcode`` → ``barcode``)."""
    if barcode_key is not None:
        return adata.obs[barcode_key].astype(str).to_numpy()
    names = adata.obs_names.astype(str)
    return np.array([n.split("#", 1)[1] if "#" in n else n for n in names])


def _fragments(adata: AnnData, fragments) -> str:
    if fragments is not None:
        return str(fragments)
    files = adata.uns.get("files", {}) if hasattr(adata, "uns") else {}
    if "fragments" in files:
        return str(files["fragments"])
    raise ValueError("Pass fragments=... (no adata.uns['files']['fragments'])")


def _col(df: pd.DataFrame, *names: str):
    for n in names:
        if n in df.columns:
            return df[n].to_numpy()
    raise KeyError(f"none of {names} in pyscdblfinder output columns {list(df.columns)}")


def scdblfinder(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
    sample_key: Optional[str] = None,
    aggregate_features: bool = True,
    n_features: int = 25,
    processing: str = "normFeatures",
    artificial_doublets: Optional[float] = None,
    dbr: Optional[float] = None,
    random_state: int = 0,
    key_added: str = "scDblFinder",
    **kwargs,
) -> None:
    """scDblFinder for scATAC-seq (wrapper over ``pyscdblfinder``).

    The defaults are the scDblFinder vignette's scATAC-seq recipe:
    ``aggregateFeatures=TRUE, nfeatures=25, processing="normFeatures"``.
    Correlated tiles/peaks are aggregated into ``n_features`` meta-features,
    artificial doublets are simulated, and a gradient-boosted classifier is
    trained on kNN features.

    Parameters
    ----------
    adata
        Cells x tiles or peaks, raw counts.
    layer
        Count layer to use instead of ``adata.X``.
    sample_key
        Capture/sample column. Each sample is run separately, as R's default
        ``multiSampleMode="split"`` does.
    aggregate_features, n_features, processing, artificial_doublets, dbr, random_state
        Passed to :func:`pyscdblfinder.sc_dbl_finder`.
    key_added
        Writes ``obs[f"{key_added}_score"]`` and ``obs[f"{key_added}_class"]``
        (``"singlet"``/``"doublet"``).
    **kwargs
        Any other ``sc_dbl_finder`` argument.
    """
    mod = _pyscdblfinder("scdblfinder")
    X = _get_matrix(adata, layer)
    groups = (adata.obs[sample_key].astype(str).to_numpy() if sample_key is not None
              else np.zeros(adata.n_obs, dtype=object))
    score = np.full(adata.n_obs, np.nan)
    klass = np.empty(adata.n_obs, dtype=object)
    thresholds = {}
    for g in pd.unique(groups):
        idx = np.where(groups == g)[0]
        args = dict(aggregate_features=aggregate_features, n_features=n_features,
                    processing=processing, dbr=dbr, random_state=random_state, **kwargs)
        if artificial_doublets is not None:
            args["artificial_doublets"] = artificial_doublets
        res = _call(mod.sc_dbl_finder, X[idx].T.tocsr(), **args)
        real = res.real_cells() if hasattr(res, "real_cells") else res.table.iloc[: len(idx)]
        score[idx] = _col(real, "score")
        klass[idx] = _col(real, "class")
        thresholds[str(g)] = float(getattr(res, "score_threshold", np.nan))
    adata.obs[f"{key_added}_score"] = score
    adata.obs[f"{key_added}_class"] = pd.Categorical(klass, categories=["singlet", "doublet"])
    adata.uns[key_added] = dict(threshold=thresholds, backend="pyscdblfinder",
                                version=getattr(mod, "__version__", None))
    console.level2(f"Added obs['{key_added}_score'] and obs['{key_added}_class']")


def amulet(
    adata: AnnData,
    fragments=None,
    *,
    barcode_key: Optional[str] = None,
    regions_to_exclude=None,
    q_threshold: float = 0.01,
    key_added: str = "amulet",
    **kwargs,
) -> None:
    """AMULET (Thibodeau et al. 2021), wrapper over ``pyscdblfinder.amulet``.

    A diploid nucleus yields at most two fragments over any base. Cells with
    unexpectedly many loci covered by more than two fragments (Poisson test,
    BH-adjusted) are called multiplets. This catches homotypic doublets that
    simulation-based methods miss, but needs deep cells.

    Parameters
    ----------
    adata
        Cells to score. Only these barcodes are read from the fragments.
    fragments
        Fragment file (tabix-indexed is faster). Defaults to
        ``adata.uns['files']['fragments']``.
    barcode_key
        ``obs`` column holding the fragment-file barcode. By default it is the
        obs name, minus any ``sample#`` prefix.
    regions_to_exclude
        BED path or GRanges-like table of repeats/blacklist to exclude. The
        backend default excludes chrM, chrX and chrY. As recommended by the
        AMULET authors, pass a repeat filter for your genome.
    q_threshold
        Cells with ``q < q_threshold`` get ``amulet_class = "doublet"``.
    key_added
        Writes ``obs[f"{key_added}_nAbove2"]``, ``_p``, ``_q`` and ``_class``.
    **kwargs
        Any other ``pyscdblfinder.amulet`` argument.
    """
    mod = _pyscdblfinder("amulet")
    bcs = _barcodes(adata, barcode_key)
    args = dict(barcodes=list(pd.unique(bcs)), **kwargs)
    if regions_to_exclude is not None:
        args["regions_to_exclude"] = regions_to_exclude
    res = _call(mod.amulet, _fragments(adata, fragments), **args)
    res = res.reindex(bcs)
    adata.obs[f"{key_added}_nAbove2"] = _col(res, "nAbove2", "n_above2", "n_above_2")
    p = _col(res, "p.value", "p_value", "pvalue")
    q = _col(res, "q.value", "q_value", "qvalue")
    adata.obs[f"{key_added}_p"] = p
    adata.obs[f"{key_added}_q"] = q
    adata.obs[f"{key_added}_class"] = pd.Categorical(
        np.where(q < q_threshold, "doublet", "singlet"), categories=["singlet", "doublet"])
    console.level2(f"Added obs['{key_added}_nAbove2'/'_p'/'_q'/'_class']")


def clamulet(
    adata: AnnData,
    fragments=None,
    *,
    barcode_key: Optional[str] = None,
    key_added: str = "clamulet",
    **kwargs,
) -> None:
    """clamulet: scDblFinder-style classifier using AMULET's overlap counts.

    Wrapper over ``pyscdblfinder.clamulet``. Artificial doublets are made by
    summing fragment coverages, so the count of loci covered by more than two
    reads can be computed for them and used as a classifier feature. It is
    slower than :func:`amulet` + :func:`scdblfinder` combined with
    :func:`combine_doublet_scores`, which the scDblFinder authors found more
    robust.

    Writes ``obs[f"{key_added}_score"]``.
    """
    mod = _pyscdblfinder("clamulet")
    bcs = _barcodes(adata, barcode_key)
    res = _call(mod.clamulet, _fragments(adata, fragments), barcodes=list(pd.unique(bcs)), **kwargs)
    res = res.reindex(bcs)
    adata.obs[f"{key_added}_score"] = _col(res, "score")
    console.level2(f"Added obs['{key_added}_score']")


def combine_doublet_scores(
    adata: AnnData,
    *,
    p_key: str = "amulet_p",
    score_key: str = "scDblFinder_score",
    clip: float = 0.001,
    key_added: str = "doublet_combined_p",
) -> None:
    """Fisher combination of an AMULET p-value and a scDblFinder score.

    This follows the scDblFinder scATAC-seq vignette: treat ``1 - score`` as a
    p-value, clip both at ``clip`` and combine them with Fisher's method
    (``pchisq(-2 Σ log p, 2k, lower.tail=FALSE)``). The scDblFinder authors
    found this more robust across datasets than either method alone. Low
    values indicate doublets.
    """
    from scipy.stats import chi2
    p = np.column_stack([adata.obs[p_key].to_numpy(dtype=float),
                         1.0 - adata.obs[score_key].to_numpy(dtype=float)])
    p = np.maximum(p, clip)
    stat = -2.0 * np.log(p).sum(axis=1)
    adata.obs[key_added] = chi2.sf(stat, 2 * p.shape[1])
    console.level2(f"Added obs['{key_added}']")
