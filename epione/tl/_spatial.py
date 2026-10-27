"""Spatial-aware analysis for spot-level spatial ATAC (and RNA).

* :func:`spatial_neighbors` -- spot graph from ``obsm["spatial"]`` or from the
  DBiT / Visium array grid, stored under the squidpy keys
  (``obsp["spatial_connectivities"]``, ``obsp["spatial_distances"]``,
  ``uns["spatial_neighbors"]``) so squidpy's ``gr.*`` functions accept it.
* :func:`spatial_smooth` -- neighbourhood averaging of an embedding or matrix
  over that graph (``(1 - lam) * x_i + lam * mean_{j in N(i)} x_j``).
  This is a plain, documented smoother, not a published method; it is
  useful on sparse spatial ATAC where a 50 um spot holds a few thousand
  fragments.
* :func:`spatial_domains` -- clustering of a (joint or smoothed) embedding
  into a fixed number of domains with the routine SpatialGlue uses: PCA
  then mclust ``EEE`` (pymclustR, pure Python), or a Leiden resolution
  search, or k-means.
* :func:`spatial_moran` -- Moran's I of features over the spot graph.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["spatial_neighbors", "spatial_smooth", "spatial_domains", "spatial_moran"]


def _coords(adata, spatial_key):
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm[{spatial_key!r}] not found; read the data with epi.io.read_spatial_atac or add_spatial")
    return np.asarray(adata.obsm[spatial_key], dtype=float)


def spatial_neighbors(
    adata,
    *,
    coord_type: Literal["grid", "generic"] = "grid",
    n_neighs: int = 4,
    n_rings: int = 1,
    radius: Optional[float] = None,
    spatial_key: str = "spatial",
    key_added: str = "spatial",
    include_self: bool = False,
):
    """Build the spot neighbourhood graph.

    Arguments:
        adata: AnnData (or MuData) with ``obsm[spatial_key]``; for
            ``coord_type="grid"`` also ``obs["array_row"]`` / ``obs["array_col"]``.
        coord_type: ``"grid"`` -- neighbours on the barcode array: with
            ``n_neighs=4`` the 4-connected (DBiT square grid), ``8`` the
            8-connected, ``6`` the Visium hexagonal ring; ``n_rings`` rings.
            ``"generic"`` -- k nearest neighbours (``n_neighs``) on the
            coordinates, or all spots within ``radius`` when given.
        key_added: graph keys are ``obsp[f"{key_added}_connectivities"]`` and
            ``obsp[f"{key_added}_distances"]`` (squidpy layout).
        include_self: add self loops to the connectivities.

    Returns:
        ``None``; writes ``obsp`` and ``uns[f"{key_added}_neighbors"]``.
    """
    from sklearn.neighbors import NearestNeighbors

    xy = _coords(adata, spatial_key)
    n = xy.shape[0]
    if coord_type == "grid":
        if not {"array_row", "array_col"} <= set(adata.obs.columns):
            raise KeyError("coord_type='grid' needs obs['array_row'] and obs['array_col']")
        r = adata.obs["array_row"].to_numpy(int)
        c = adata.obs["array_col"].to_numpy(int)
        lut = {(a, b): i for i, (a, b) in enumerate(zip(r, c))}
        if n_neighs == 4:
            def ring(k):
                return [(dr, dc) for dr in range(-k, k + 1) for dc in range(-k, k + 1)
                        if abs(dr) + abs(dc) <= k and (dr, dc) != (0, 0)]
        elif n_neighs == 8:
            def ring(k):
                return [(dr, dc) for dr in range(-k, k + 1) for dc in range(-k, k + 1) if (dr, dc) != (0, 0)]
        elif n_neighs == 6:
            # Visium: array_col steps by 2 within a row, rows are offset by 1
            def ring(k):
                return [(dr, dc) for dr in range(-k, k + 1) for dc in range(-2 * k, 2 * k + 1)
                        if (dr, dc) != (0, 0) and (abs(dr) + abs(dc)) % 2 == 0
                        and abs(dc) + abs(dr) <= 2 * k]
        else:
            raise ValueError("grid n_neighs must be 4, 6 or 8")
        offs = ring(n_rings)
        rows, cols = [], []
        for i, (a, b) in enumerate(zip(r, c)):
            for dr, dc in offs:
                j = lut.get((a + dr, b + dc))
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        rows = np.asarray(rows, int)
        cols = np.asarray(cols, int)
    else:
        nn = NearestNeighbors().fit(xy)
        if radius is not None:
            G = nn.radius_neighbors_graph(xy, radius=radius, mode="connectivity").tocoo()
            m = G.row != G.col
            rows, cols = G.row[m], G.col[m]
        else:
            _, idx = nn.kneighbors(xy, n_neighbors=n_neighs + 1)
            rows = np.repeat(np.arange(n), n_neighs)
            cols = idx[:, 1:].ravel()
    d = np.linalg.norm(xy[rows] - xy[cols], axis=1)
    A = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32).tocsr()
    D = sp.csr_matrix((d, (rows, cols)), shape=(n, n))
    D = D.maximum(D.T).tocsr()
    if include_self:
        A = (A + sp.eye(n, dtype=np.float32, format="csr")).tocsr()
    target = adata
    target.obsp[f"{key_added}_connectivities"] = A
    target.obsp[f"{key_added}_distances"] = D
    target.uns[f"{key_added}_neighbors"] = {
        "connectivities_key": f"{key_added}_connectivities",
        "distances_key": f"{key_added}_distances",
        "params": {"n_neighbors": n_neighs, "coord_type": coord_type, "radius": radius,
                   "n_rings": n_rings, "transform": None},
    }


def _graph(adata, key):
    k = f"{key}_connectivities"
    if k not in adata.obsp:
        raise KeyError(f"obsp[{k!r}] not found; run epi.tl.spatial_neighbors first")
    return adata.obsp[k]


def spatial_smooth(
    adata,
    *,
    use_rep: Optional[str] = "X_lsi",
    layer: Optional[str] = None,
    lam: float = 0.5,
    n_iter: int = 1,
    graph_key: str = "spatial",
    key_added: Optional[str] = None,
):
    """Average features over spatial neighbours.

    ``x_i <- (1 - lam) * x_i + lam * mean_{j in N(i)} x_j``, applied
    ``n_iter`` times over the spot graph from :func:`spatial_neighbors`.
    ``lam=0`` returns the input, ``lam=1`` the pure neighbour mean.

    Arguments:
        use_rep: ``obsm`` key to smooth (default ``"X_lsi"``); ``None`` means
            ``X`` (or ``layer``).
        layer: smooth ``layers[layer]`` instead of ``X`` when ``use_rep`` is None.
        key_added: output key; default ``f"{use_rep}_smooth"`` in ``obsm``
            or ``"smoothed"`` in ``layers``.

    Returns:
        The smoothed array (also stored).
    """
    A = _graph(adata, graph_key).astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    W = sp.diags(np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0)) @ A
    if use_rep is not None:
        X = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    else:
        X = adata.layers[layer] if layer is not None else adata.X
    has_nb = (deg > 0)[:, None]
    for _ in range(int(n_iter)):
        nb = W @ X
        if sp.issparse(X):
            X = ((1 - lam) * X + lam * nb).tocsr()
        else:
            X = np.where(has_nb, (1 - lam) * X + lam * np.asarray(nb), X)
    if use_rep is not None:
        adata.obsm[key_added or f"{use_rep}_smooth"] = X
    else:
        adata.layers[key_added or "smoothed"] = X
    return X


def _pca(X, n_comps):
    from sklearn.decomposition import PCA

    return PCA(n_components=n_comps).fit_transform(np.asarray(X))


def spatial_domains(
    adata,
    n_domains: int,
    *,
    use_rep: Optional[str] = None,
    method: Literal["mclust", "leiden", "kmeans", "stagate", "graphst", "spaceflow",
                    "cellcharter", "cast", "binary", "aegle"] = "mclust",
    use_pca: bool = True,
    n_comps: int = 20,
    model_name: str = "EEE",
    random_seed: int = 2020,
    start: float = 0.1,
    end: float = 3.0,
    increment: float = 0.01,
    key_added: str = "domain",
    **method_kwargs,
):
    """Cluster an embedding into ``n_domains`` spatial domains.

    Mirrors ``SpatialGlue.utils.clustering``: optional PCA (``n_comps``) of
    ``obsm[use_rep]``, then

    * ``"mclust"``: Gaussian mixture with covariance model ``model_name``
      (``EEE`` upstream) and ``G = n_domains``, via pymclustR (pure-Python
      port of CRAN mclust; ``pip install pymclustR``);
    * ``"leiden"``: scanpy neighbours (k = 50) and a descending resolution
      search from ``end`` to ``start`` until ``n_domains`` clusters appear;
    * ``"kmeans"``: scikit-learn k-means (n_init = 10).

    * ``"aegle"``: Epione's native spatial-ATAC method (:func:`epione.tl.aegle`):
      graph-Tikhonov smoothing of the biwhitened peak matrix with a generalised
      Marchenko-Pastur rank decision, then a Potts-prior Gaussian mixture. It works on the
      spots x peaks matrix ``adata.X`` (``use_rep``, ``use_pca`` and ``n_comps`` are
      ignored); ``method_kwargs`` go to :func:`~epione.tl.aegle` (e.g. ``rna=`` for
      ATAC + RNA, ``select=True`` for count-splitting hyper-parameter selection).

    Labels go to ``obs[key_added]`` (categorical, 1-based for mclust as in R).
    Works on AnnData and MuData (``obsm`` of the container). ``use_rep``
    defaults to ``"SpatialGlue"`` for these three clustering methods.

    Spatial-domain models wrapped from omicverse (imported only when
    called; ``pip install omicverse``): ``"stagate"``, ``"graphst"``,
    ``"spaceflow"``, ``"cellcharter"``, ``"cast"`` and ``"binary"`` train
    the model on ``obsm[use_rep]`` (default ``"X_birmt"``; ``None`` is not
    possible here -- call :func:`epione.tl.stagate` etc. with
    ``use_rep=None`` for the peak / tile matrix; ``"binary"`` always uses
    the matrix) and cluster its embedding. ``random_seed`` is the
    clustering seed; ``method_kwargs`` go to the method's function
    (:func:`epione.tl.stagate`, :func:`~epione.tl.graphst`,
    :func:`~epione.tl.spaceflow`, :func:`~epione.tl.cellcharter`,
    :func:`~epione.tl.cast`, :func:`~epione.tl.binary`), e.g. ``seed``,
    ``device``, ``cluster_method``.
    """
    if method == "aegle":
        from ._aegle import aegle

        return aegle(adata, int(n_domains), seed=random_seed, key_added=key_added, **method_kwargs)
    from ._spatial_ov import OV_DOMAIN_METHODS

    if method in OV_DOMAIN_METHODS:
        from ._spatial_ov import _run

        if method == "binary":
            use_rep = None
        elif use_rep is None:
            use_rep = "X_birmt"
        return _run(method, adata, n_domains, use_rep=use_rep, key_added=key_added,
                    random_seed=random_seed, **method_kwargs)
    if method_kwargs:
        raise TypeError(f"unexpected arguments for method={method!r}: {sorted(method_kwargs)}")
    if use_rep is None:
        use_rep = "SpatialGlue"
    X = np.asarray(adata.obsm[use_rep], dtype=float)
    if use_pca:
        np.random.seed(random_seed)
        X = _pca(X, n_comps)
        adata.obsm[f"{use_rep}_pca"] = X
    if method == "mclust":
        try:
            from mclust_py import Mclust
        except ImportError as e:  # pragma: no cover
            raise ImportError("method='mclust' needs pymclustR (pip install pymclustR)") from e
        np.random.seed(random_seed)
        fit = Mclust(X, G=int(n_domains), model_names=[model_name])
        lab = np.asarray(fit.classification).astype(int)
    elif method == "kmeans":
        from sklearn.cluster import KMeans

        lab = KMeans(n_clusters=int(n_domains), n_init=10, random_state=random_seed).fit_predict(X) + 1
    elif method == "leiden":
        import anndata as ad
        import scanpy as sc

        tmp = ad.AnnData(np.zeros((X.shape[0], 1), dtype=np.float32))
        tmp.obsm["rep"] = X
        sc.pp.neighbors(tmp, n_neighbors=50, use_rep="rep")
        lab = None
        for res in sorted(np.arange(start, end, increment), reverse=True):
            sc.tl.leiden(tmp, random_state=0, resolution=res)  # as upstream search_res
            if tmp.obs["leiden"].nunique() == n_domains:
                lab = tmp.obs["leiden"].astype(int).to_numpy() + 1
                break
        if lab is None:
            raise RuntimeError("no Leiden resolution gave n_domains clusters; widen start/end or shrink increment")
    else:
        raise ValueError(f"unknown method {method!r}")
    cats = sorted(set(lab))
    adata.obs[key_added] = pd.Categorical(lab.astype(int), categories=cats)
    return adata.obs[key_added]


def spatial_moran(
    adata,
    *,
    use_rep: Optional[str] = None,
    features: Optional[Sequence[str]] = None,
    layer: Optional[str] = None,
    graph_key: str = "spatial",
) -> pd.Series:
    """Moran's I of each feature over the spot graph (row-standardised weights).

    Arguments:
        use_rep: ``obsm`` key whose columns are scored; else ``X`` / ``layer``
            restricted to ``features``.

    Returns:
        Series of Moran's I indexed by feature (column index for ``use_rep``).
    """
    A = _graph(adata, graph_key).astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    W = sp.diags(np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0)) @ A
    if use_rep is not None:
        X = np.asarray(adata.obsm[use_rep], dtype=float)
        names = getattr(adata.obsm[use_rep], "columns", None)
        names = list(names) if names is not None else list(range(X.shape[1]))
    else:
        sub = adata[:, list(features)] if features is not None else adata
        X = sub.layers[layer] if layer is not None else sub.X
        X = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float)
        names = list(sub.var_names)
    Z = X - X.mean(0)
    num = (Z * (W @ Z)).sum(0)
    den = (Z ** 2).sum(0)
    n, s0 = X.shape[0], W.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        I = (n / s0) * num / den
    return pd.Series(I, index=names, name="moran_I")
