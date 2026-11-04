"""Thin wrappers around the general spatial-domain methods shipped by omicverse.

STAGATE, GraphST, SpaceFlow, CellCharter, CAST and BINARY were written for
spatial transcriptomics and are already maintained in omicverse
(``omicverse.space``). Epione does not port or vendor them: each function
below imports omicverse *inside the call*, so omicverse is neither a
dependency nor an extra of Epione (``omicverse.epi`` wraps Epione, and a
hard dependency would be circular). Install it separately with
``pip install omicverse``; the GNN methods also need ``torch_geometric``,
SpaceFlow needs ``gudhi`` and CAST needs ``dgl`` (omicverse's own optional
dependencies).

What the wrappers add is the ATAC side:

* **input** -- an Epione embedding (``use_rep``, default ``"X_birmt"``) or the
  spot x peak / tile matrix (``use_rep=None``). A matrix is reduced to the
  ``n_top_features`` most accessible columns (detection rate, as
  :func:`epione.pp.select_features`), then log TF-IDF
  (``log1p(tf * idf * 1e4)``) for the continuous-input methods, or kept
  binary for BINARY. The methods run on a private copy, so the caller's
  ``obsp`` / ``uns`` are never modified;
* **output** -- the embedding in ``obsm[key_added]`` and, when ``n_domains``
  is given, domains in ``obs[domain_key]`` as 1-based categorical integers,
  the same layout as :func:`epione.tl.spatial_domains`. The embedding is
  clustered with the routine of :func:`~epione.tl.spatial_domains`
  (``cluster_method="mclust"``: mclust ``EEE`` via pymclustR, as the
  STAGATE / GraphST / BINARY tutorials do; ``"leiden"`` or ``"kmeans"``
  also work). CellCharter clusters with its own Gaussian mixture.

All six are also reachable as ``epi.tl.spatial_domains(adata, n_domains,
method="stagate" | "graphst" | "spaceflow" | "cellcharter" | "cast" |
"binary", ...)``.
"""
from __future__ import annotations

import os
import random
import tempfile
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["stagate", "graphst", "spaceflow", "cellcharter", "cast", "binary", "OV_DOMAIN_METHODS"]

#: ``spatial_domains(method=...)`` names served by this module.
OV_DOMAIN_METHODS = ("stagate", "graphst", "spaceflow", "cellcharter", "cast", "binary")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _omicverse(method: str):
    try:
        import omicverse as ov
    except ImportError as e:
        raise ImportError(
            f"epi.tl.{method} wraps the implementation in omicverse, which Epione does not install. "
            "Install it with `pip install omicverse` (plus its optional torch_geometric / gudhi / dgl "
            "dependencies for this method)."
        ) from e
    return ov


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - every wrapped method needs torch anyway
        pass


def _coords(adata, spatial_key):
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm[{spatial_key!r}] not found; read the data with epi.io.read_spatial_atac or add_spatial")
    xy = adata.obsm[spatial_key]
    xy = xy.toarray() if sp.issparse(xy) else np.asarray(xy)
    return np.asarray(xy[:, :2], dtype=np.float64)


def _top_features(X, n_top: int) -> np.ndarray:
    # Same score as epione.pp.select_features: detection rate, ties by total count.
    total = np.asarray(X.sum(axis=0)).ravel()
    n_pos = np.asarray((X != 0).sum(axis=0)).ravel()
    score = n_pos.astype(np.float64) + total / (total.max() + 1e-9) * 1e-6
    n = min(int(n_top), X.shape[1])
    return np.sort(np.argsort(-score, kind="stable")[:n])


def _features(adata, use_rep, layer, n_top_features, transform):
    """Dense (or binary sparse) spot x feature input and its feature names."""
    if use_rep is not None:
        if use_rep not in adata.obsm:
            raise KeyError(
                f"obsm[{use_rep!r}] not found; compute an embedding first (e.g. epi.tl.birmt(adata)) "
                "or pass use_rep=None to use the peak / tile matrix"
            )
        if transform == "binary":
            raise ValueError("BINARY binarises counts: pass use_rep=None (peak / tile matrix), not an embedding")
        X = adata.obsm[use_rep]
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        return np.ascontiguousarray(X, dtype=np.float32), [f"{use_rep}_{i}" for i in range(X.shape[1])]
    if not hasattr(adata, "var_names") or getattr(adata, "mod", None) is not None:
        raise ValueError("use_rep=None needs an AnnData with the peak / tile matrix in X (or `layer`)")
    X = adata.layers[layer] if layer is not None else adata.X
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    cols = _top_features(X, n_top_features)
    X = X[:, cols].astype(np.float32)
    names = list(np.asarray(adata.var_names)[cols])
    B = (X > 0).astype(np.float32)
    if transform == "binary":
        return B.tocsr(), names
    # log TF-IDF on the binarised matrix (Signac method 1 / ArchR-style).
    rs = np.asarray(B.sum(1)).ravel()
    cs = np.asarray(B.sum(0)).ravel()
    tf = sp.diags(1.0 / np.maximum(rs, 1.0)) @ B
    idf = B.shape[0] / np.maximum(cs, 1.0)
    T = (tf @ sp.diags(idf)).toarray() * 1e4
    return np.log1p(T).astype(np.float32), names


def _work(adata, X, names, spatial_key):
    """A private AnnData holding only what the wrapped method reads."""
    import anndata as ad

    w = ad.AnnData(X)
    w.obs_names = np.asarray(adata.obs_names).astype(str)
    w.var_names = pd.Index(names).astype(str)
    w.obsm["spatial"] = _coords(adata, spatial_key)
    return w


class _no_new_figures:
    """Close figures the wrapped code opens (STAGATE batch stats etc.), keep the caller's."""

    def __enter__(self):
        try:
            import matplotlib.pyplot as plt

            self._plt, self._before = plt, set(plt.get_fignums())
        except ImportError:  # pragma: no cover
            self._plt = None
        return self

    def __exit__(self, *exc):
        if self._plt is not None:
            for n in set(self._plt.get_fignums()) - self._before:
                self._plt.close(n)
        return False


def _cluster(adata, n_domains, key, *, cluster_method, use_pca, n_comps, cluster_seed, domain_key, **kw):
    from ._spatial import spatial_domains

    if n_domains is None:
        return None
    n_comps = min(int(n_comps), adata.obsm[key].shape[1])
    if n_comps >= adata.obsm[key].shape[1]:
        use_pca = False
    return spatial_domains(adata, int(n_domains), use_rep=key, method=cluster_method, use_pca=use_pca,
                           n_comps=n_comps, random_seed=cluster_seed, key_added=domain_key, **kw)


def _median_spacing(xy):
    from sklearn.neighbors import NearestNeighbors

    d, _ = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)
    return float(np.median(d[:, 1]))


_CLUSTER_DOC = """
        n_domains: number of domains; ``None`` only computes the embedding.
        cluster_method: ``"mclust"`` (EEE, pymclustR), ``"leiden"`` or ``"kmeans"``
            -- see :func:`epione.tl.spatial_domains`.
        cluster_seed: seed of the clustering step (``spatial_domains``' ``random_seed``).
        domain_key: ``obs`` column for the domains."""


# ---------------------------------------------------------------------------
# STAGATE
# ---------------------------------------------------------------------------

def stagate(
    adata,
    n_domains: Optional[int] = None,
    *,
    use_rep: Optional[str] = "X_birmt",
    layer: Optional[str] = None,
    n_top_features: int = 3000,
    spatial_key: str = "spatial",
    rad_cutoff: Optional[float] = None,
    num_batch_x: int = 1,
    num_batch_y: int = 1,
    hidden_dims: Sequence[int] = (512, 30),
    num_epoch: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
    key_added: str = "STAGATE",
    cluster_method: str = "mclust",
    cluster_seed: int = 2020,
    domain_key: str = "domain",
):
    """STAGATE graph-attention autoencoder (via ``omicverse.space.pySTAGATE``).

    Dong & Zhang, "Deciphering spatial domains from spatially resolved
    transcriptomics with an adaptive graph attention auto-encoder",
    *Nat Commun* 13, 1739 (2022), doi:10.1038/s41467-022-29439-6.
    Upstream licence MIT (zhanglabtools/STAGATE, QIFEIDKN/STAGATE_pyG);
    omicverse redistributes the PyG version under GPL-3.0.

    Arguments:
        use_rep: ``obsm`` embedding fed to the encoder; ``None`` uses the top
            ``n_top_features`` columns of ``X`` / ``layer`` after log TF-IDF.
        rad_cutoff: radius of the spot graph in coordinate units; default
            1.5 x the median nearest-spot distance (8 neighbours on the DBiT
            square grid, 6 on the Visium hexagonal grid).
        num_batch_x, num_batch_y: STAGATE's spatial tiling for mini-batches;
            ``1, 1`` (default) trains on the whole graph, as the original
            STAGATE does for slices of this size (omicverse's default is 3 x 2).
        hidden_dims, num_epoch, lr, weight_decay: STAGATE defaults.
        device: torch device (falls back to CPU without CUDA).
        seed: seeds python / numpy / torch before the model is built
            (pySTAGATE has no seed argument).
        key_added: ``obsm`` key of the 30-d embedding (default ``"STAGATE"``).
    {CLUSTER_ARGS}

    Returns:
        The domain labels (``obs[domain_key]``) or, with ``n_domains=None``,
        the embedding.
    """
    ov = _omicverse("stagate")
    X, names = _features(adata, use_rep, layer, n_top_features, "tfidf")
    w = _work(adata, X, names, spatial_key)
    w.obs["X"] = w.obsm["spatial"][:, 0]
    w.obs["Y"] = w.obsm["spatial"][:, 1]
    if rad_cutoff is None:
        rad_cutoff = 1.5 * _median_spacing(w.obsm["spatial"])
    _seed(seed)
    with _no_new_figures():
        model = ov.space.pySTAGATE(
            w, num_batch_x=int(num_batch_x), num_batch_y=int(num_batch_y), spatial_key=["X", "Y"],
            rad_cutoff=float(rad_cutoff), num_epoch=int(num_epoch), lr=lr, weight_decay=weight_decay,
            hidden_dims=list(hidden_dims), device=device,
        )
        model.train()
        model.predicted()
    adata.obsm[key_added] = np.asarray(w.obsm["STAGATE"])
    out = _cluster(adata, n_domains, key_added, cluster_method=cluster_method, use_pca=False, n_comps=30,
                   cluster_seed=cluster_seed, domain_key=domain_key)
    return adata.obsm[key_added] if out is None else out


# ---------------------------------------------------------------------------
# GraphST
# ---------------------------------------------------------------------------

def graphst(
    adata,
    n_domains: Optional[int] = None,
    *,
    use_rep: Optional[str] = "X_birmt",
    layer: Optional[str] = None,
    n_top_features: int = 3000,
    spatial_key: str = "spatial",
    n_pcs: int = 20,
    epochs: int = 600,
    learning_rate: float = 1e-3,
    dim_output: int = 64,
    alpha: float = 10,
    beta: float = 1,
    datatype: str = "10X",
    device: str = "cpu",
    seed: int = 41,
    key_added: str = "GraphST",
    cluster_method: str = "mclust",
    cluster_seed: int = 2020,
    domain_key: str = "domain",
):
    """GraphST contrastive graph autoencoder (via ``omicverse.space.clusters(methods=["GraphST"])``).

    Long et al., "Spatially informed clustering, integration, and
    deconvolution of spatial transcriptomics with GraphST", *Nat Commun* 14,
    1155 (2023), doi:10.1038/s41467-023-36796-3. Upstream licence:
    AGPL-3.0 (JinmiaoChenLab/GraphST LICENSE file); omicverse vendors it.

    The encoder input is the embedding ``use_rep`` (or the log TF-IDF of the
    top features). GraphST's output is a reconstruction of that input;
    as in the GraphST tutorial it is reduced to ``n_pcs`` principal
    components before clustering.

    Arguments:
        n_pcs: PCs of the GraphST reconstruction (``obsm[f"{key_added}_pca"]``,
            capped at the input dimension - 1); the domains are called on them.
        epochs, learning_rate, dim_output, alpha, beta: GraphST defaults.
        datatype: ``"10X"`` (dense spot graph, 3 nearest spots) or
            ``"Stereo"`` / ``"Slide"`` (sparse graph; use it above ~20k spots,
            the dense variant needs several n x n float64 arrays).
        seed: GraphST's ``random_seed`` (upstream default 41).
        key_added: ``obsm`` key of the reconstruction; PCs go to
            ``f"{key_added}_pca"``.
    {CLUSTER_ARGS}

    Returns:
        The domain labels or, with ``n_domains=None``, the PCs.
    """
    ov = _omicverse("graphst")
    X, names = _features(adata, use_rep, layer, n_top_features, "tfidf")
    w = _work(adata, X, names, spatial_key)
    w.layers["counts"] = w.X.copy()  # omicverse otherwise tries to recover counts from log data
    w.var["highly_variable"] = True
    n_pcs = int(max(1, min(n_pcs, X.shape[1] - 1, X.shape[0] - 1)))
    import torch

    kw = dict(device=torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu"),
              n_pcs=n_pcs, epochs=int(epochs), learning_rate=learning_rate, dim_output=int(dim_output),
              alpha=alpha, beta=beta, datatype=datatype, random_seed=int(seed))
    with _no_new_figures():
        w = ov.space.clusters(w, methods=["GraphST"], methods_kwargs={"GraphST": kw}, spatial_key="spatial")
    adata.obsm[key_added] = np.asarray(w.obsm["GraphST_embedding"])
    adata.obsm[f"{key_added}_pca"] = np.asarray(w.obsm["graphst|original|X_pca"])
    out = _cluster(adata, n_domains, f"{key_added}_pca", cluster_method=cluster_method, use_pca=False,
                   n_comps=n_pcs, cluster_seed=cluster_seed, domain_key=domain_key)
    return adata.obsm[f"{key_added}_pca"] if out is None else out


# ---------------------------------------------------------------------------
# SpaceFlow
# ---------------------------------------------------------------------------

def spaceflow(
    adata,
    n_domains: Optional[int] = None,
    *,
    use_rep: Optional[str] = "X_birmt",
    layer: Optional[str] = None,
    n_top_features: int = 3000,
    spatial_key: str = "spatial",
    spatial_regularization_strength: float = 0.1,
    z_dim: int = 50,
    lr: float = 1e-3,
    epochs: int = 1000,
    max_patience: int = 50,
    min_stop: int = 100,
    gpu: int = 0,
    regularization_acceleration: bool = True,
    edge_subset_sz: int = 1_000_000,
    seed: int = 42,
    key_added: str = "SpaceFlow",
    cluster_method: str = "mclust",
    use_pca: bool = True,
    n_comps: int = 20,
    cluster_seed: int = 2020,
    domain_key: str = "domain",
):
    """SpaceFlow deep graph infomax with a spatial regulariser (via ``omicverse.space.pySpaceFlow``).

    Ren et al., "Identifying multicellular spatiotemporal organization of
    cells with SpaceFlow", *Nat Commun* 13, 4076 (2022),
    doi:10.1038/s41467-022-31739-w. Upstream licence MIT (hongleir/SpaceFlow).
    SpaceFlow on ATAC alone is the second-best published result on the P22
    mouse brain ATAC-RNA labels of COSMOS (ARI 0.58; see the Epione spatial
    survey).

    The features (``use_rep`` or log TF-IDF of the top features) go to the
    GCN encoder as they are; the spot graph is SpaceFlow's alpha-complex
    graph (10-NN distance cut; needs ``gudhi``). Note the upstream code
    always uses ``cuda:{gpu}`` when CUDA is visible, and, with
    ``regularization_acceleration=True`` (default), samples both spatial
    coordinates of a pair from the same index -- omicverse keeps this
    upstream behaviour, so the spatial term is effectively
    ``mean(1 - d_z / max d_z)`` over random pairs.

    Arguments:
        spatial_regularization_strength, z_dim, lr, epochs, max_patience,
        min_stop, regularization_acceleration, edge_subset_sz: SpaceFlow
            ``train`` arguments, upstream defaults.
        seed: SpaceFlow's ``random_seed``.
        use_pca, n_comps: PCA of the 50-d embedding before clustering
            (``spatial_domains`` default). Upstream SpaceFlow segments with
            Leiden (resolution 1, 50 neighbours); pass
            ``cluster_method="leiden"`` to search a resolution that gives
            ``n_domains``.
        key_added: ``obsm`` key of the embedding.
    {CLUSTER_ARGS}

    Returns:
        The domain labels or, with ``n_domains=None``, the embedding.
    """
    ov = _omicverse("spaceflow")
    X, names = _features(adata, use_rep, layer, n_top_features, "tfidf")
    w = _work(adata, X, names, spatial_key)
    with _no_new_figures():
        model = ov.space.pySpaceFlow(w)
        emb = model.train(spatial_regularization_strength=spatial_regularization_strength, z_dim=int(z_dim),
                          lr=lr, epochs=int(epochs), max_patience=int(max_patience), min_stop=int(min_stop),
                          random_seed=int(seed), gpu=int(gpu),
                          regularization_acceleration=regularization_acceleration,
                          edge_subset_sz=int(edge_subset_sz))
    adata.obsm[key_added] = np.asarray(emb)
    out = _cluster(adata, n_domains, key_added, cluster_method=cluster_method, use_pca=use_pca, n_comps=n_comps,
                   cluster_seed=cluster_seed, domain_key=domain_key)
    return adata.obsm[key_added] if out is None else out


# ---------------------------------------------------------------------------
# CellCharter
# ---------------------------------------------------------------------------

def cellcharter(
    adata,
    n_domains: int,
    *,
    use_rep: Optional[str] = "X_birmt",
    spatial_key: str = "spatial",
    n_layers: int = 3,
    aggregations="mean",
    graph_key: Optional[str] = None,
    delaunay: bool = True,
    n_neighs: int = 6,
    trim_long_links: bool = True,
    distance_percentile: float = 99.0,
    covariance_type: str = "full",
    backend: str = "auto",
    trainer_params: Optional[dict] = None,
    seed: int = 1024,
    key_added: str = "X_cellcharter",
    domain_key: str = "domain",
):
    """CellCharter neighbourhood aggregation + Gaussian mixture (via ``omicverse.space.cellcharter``).

    Varrone et al., "CellCharter reveals spatial cell niches associated with
    tissue remodeling and cell plasticity", *Nat Genet* 56, 74-84 (2024),
    doi:10.1038/s41588-023-01588-4. Upstream licence BSD-3-Clause
    (CSOgroup/cellcharter); omicverse carries a minimal re-implementation
    that uses torchgmm when installed and scikit-learn's GaussianMixture
    otherwise (``backend``).

    The embedding ``use_rep`` (CellCharter's scVI latent space is replaced
    by the ATAC embedding, as CellCharter allows any dense representation)
    is concatenated with its mean over the 1..``n_layers``-hop spatial
    neighbours, and the result is clustered into ``n_domains`` components.

    Arguments:
        use_rep: dense ``obsm`` embedding (required; a peak matrix is far
            too wide for a full-covariance mixture -- run ``epi.tl.birmt`` first).
        n_layers, aggregations: CellCharter's aggregation (default 3 hops, mean).
        graph_key: reuse Epione's spot graph ``obsp[f"{graph_key}_connectivities"]``
            (and ``_distances``) from :func:`epione.tl.spatial_neighbors`;
            default ``None`` builds CellCharter's Delaunay graph
            (``delaunay``, ``n_neighs``) and trims links above the
            ``distance_percentile``-th percentile when ``trim_long_links``.
        covariance_type, backend, trainer_params: mixture settings.
        seed: ``random_state`` of the mixture.
        key_added: ``obsm`` key of the aggregated features.
        domain_key: ``obs`` column for the domains (1-based integers).

    Returns:
        The domain labels.
    """
    ov = _omicverse("cellcharter")
    if use_rep is None:
        raise ValueError("cellcharter needs a dense embedding (use_rep), e.g. epi.tl.birmt(adata) -> 'X_birmt'")
    X, names = _features(adata, use_rep, None, 0, "tfidf")
    w = _work(adata, X, names, spatial_key)
    w.obsm["rep"] = X
    if graph_key is not None:
        for s in ("connectivities", "distances"):
            k = f"{graph_key}_{s}"
            if k not in adata.obsp:
                raise KeyError(f"obsp[{k!r}] not found; run epi.tl.spatial_neighbors(key_added={graph_key!r})")
            w.obsp[f"spatial_{s}"] = sp.csr_matrix(adata.obsp[k], copy=True)
        w.uns["spatial_neighbors"] = {"connectivities_key": "spatial_connectivities",
                                      "distances_key": "spatial_distances", "params": {"radius": None}}
    _seed(seed)
    with _no_new_figures():
        ov.space.cellcharter(
            w, n_clusters=int(n_domains), use_rep="rep", spatial_key="spatial", n_layers=n_layers,
            aggregations=aggregations, out_key="X_cellcharter", cluster_key="cellcharter",
            build_spatial_graph=graph_key is None, delaunay=delaunay, n_neighs=n_neighs,
            trim_long_links=trim_long_links, distance_percentile=distance_percentile, random_state=seed,
            covariance_type=covariance_type, backend=backend, trainer_params=trainer_params,
        )
    agg = w.obsm["X_cellcharter"]
    adata.obsm[key_added] = agg.toarray() if sp.issparse(agg) else np.asarray(agg)
    lab = pd.Series(np.asarray(w.obs["cellcharter"]).astype(int) + 1, index=adata.obs_names)
    adata.obs[domain_key] = pd.Categorical(lab.to_numpy(), categories=sorted(set(lab)))
    return adata.obs[domain_key]


# ---------------------------------------------------------------------------
# CAST
# ---------------------------------------------------------------------------

def cast(
    adata,
    n_domains: Optional[int] = None,
    *,
    use_rep: Optional[str] = "X_birmt",
    layer: Optional[str] = None,
    n_top_features: int = 3000,
    spatial_key: str = "spatial",
    sample_key: Optional[str] = None,
    epochs: int = 400,
    graph_strategy: str = "convex",
    device: str = "cpu",
    output_path: Optional[str] = None,
    seed: int = 0,
    key_added: str = "X_cast",
    cluster_method: str = "kmeans",
    use_pca: bool = True,
    n_comps: int = 20,
    cluster_seed: int = 2020,
    domain_key: str = "domain",
):
    """CAST Mark graph self-supervised embedding (via ``omicverse.space.CAST``).

    Tang et al., "Search and match across spatial omics samples at
    single-cell resolution", *Nat Methods* 21, 1818-1829 (2024),
    doi:10.1038/s41592-024-02410-7. Upstream licence MIT
    (wanglab-broad/CAST). Needs ``dgl``.

    CAST Mark trains a GCNII encoder with the CCA-SSG objective on a
    Delaunay spot graph per sample and returns a 512-d embedding; the CAST
    tutorial calls domains with k-means on it (default here).

    Arguments:
        sample_key: ``obs`` column with sample labels (one graph per sample);
            ``None`` treats the slice as one sample.
        epochs, graph_strategy: CAST Mark settings (upstream defaults).
        device: ``"cpu"`` or ``"cuda:<i>"``.
        output_path: where CAST writes its model / loss files; default a
            temporary directory removed afterwards.
        seed: seeds python / numpy / torch before training.
        use_pca, n_comps: PCA of the 512-d embedding before clustering.
        key_added: ``obsm`` key of the embedding.
    {CLUSTER_ARGS}

    Returns:
        The domain labels or, with ``n_domains=None``, the embedding.
    """
    ov = _omicverse("cast")
    X, names = _features(adata, use_rep, layer, n_top_features, "tfidf")
    w = _work(adata, X, names, spatial_key)
    w.layers["features"] = w.X
    w.obs["_cast_sample"] = (adata.obs[sample_key].astype(str).to_numpy() if sample_key is not None
                             else "sample1")
    gpu = -1 if device == "cpu" else int(str(device).split(":")[-1] or 0)
    _seed(seed)
    tmp = tempfile.TemporaryDirectory(prefix="epione_cast_") if output_path is None else None
    try:
        path = tmp.name if tmp is not None else output_path
        os.makedirs(path, exist_ok=True)
        with _no_new_figures():
            w = ov.space.CAST(w, sample_key="_cast_sample", basis="spatial", layer="features", output_path=path,
                              gpu_t=gpu, device=device, epoch_t=int(epochs), if_plot=False,
                              graph_strategy=graph_strategy)
    finally:
        if tmp is not None:
            tmp.cleanup()
    adata.obsm[key_added] = np.asarray(w.obsm["X_cast"])
    out = _cluster(adata, n_domains, key_added, cluster_method=cluster_method, use_pca=use_pca, n_comps=n_comps,
                   cluster_seed=cluster_seed, domain_key=domain_key)
    return adata.obsm[key_added] if out is None else out


# ---------------------------------------------------------------------------
# BINARY
# ---------------------------------------------------------------------------

def binary(
    adata,
    n_domains: Optional[int] = None,
    *,
    use_rep: Optional[str] = None,
    layer: Optional[str] = None,
    n_top_features: int = 3000,
    spatial_key: str = "spatial",
    use_method: str = "KNN",
    cutoff: float = 6,
    pos_weight: float = 10,
    hidden_dims: Sequence[int] = (512, 30),
    n_epochs: int = 1000,
    lr: float = 1e-3,
    gradient_clipping: float = 5.0,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
    key_added: str = "BINARY",
    cluster_method: str = "mclust",
    cluster_seed: int = 2020,
    domain_key: str = "domain",
):
    """BINARY graph attention on binarised profiles (via ``omicverse.external.BINARY``).

    Lin et al., "Complete spatially resolved gene expression is not necessary
    for identifying spatial domains", *Cell Genomics* 4, 100565 (2024),
    doi:10.1016/j.xgen.2024.100565. omicverse redistributes it under GPL-3.0.

    BINARY reads only presence / absence, which is what a spot x peak (or
    tile) ATAC matrix is, so its input is the matrix (``use_rep=None``):
    the top ``n_top_features`` most accessible columns of ``X`` / ``layer``,
    binarised. The omicverse functions ``Count2Binary``,
    ``Mutil_Construct_Spatial_Graph`` and ``train_BINARY`` are called
    directly because ``ov.space.clusters(methods=["BINARY"])`` always
    re-derives counts from log-normalised data first.

    Arguments:
        use_method, cutoff: spot graph (``"KNN"`` with ``cutoff`` neighbours,
            or ``"Radius"`` with radius ``cutoff``); upstream default KNN 6.
        pos_weight, hidden_dims, n_epochs, lr, gradient_clipping,
        weight_decay: ``train_BINARY`` defaults.
        seed: ``train_BINARY``'s ``random_seed``.
        key_added: ``obsm`` key of the 30-d embedding.
    {CLUSTER_ARGS}

    Returns:
        The domain labels or, with ``n_domains=None``, the embedding.
    """
    _omicverse("binary")
    try:
        from omicverse.external import BINARY as B
    except ImportError as e:  # pragma: no cover
        raise ImportError("omicverse.external.BINARY could not be imported (needs torch_geometric)") from e
    X, names = _features(adata, use_rep, layer, n_top_features, "binary")
    w = _work(adata, X, names, spatial_key)
    w.obs["_binary_sample"] = "sample1"
    w = B.Count2Binary(w)
    with _no_new_figures():
        B.Mutil_Construct_Spatial_Graph(w, use_method=use_method, cutoff=cutoff, obs_key="_binary_sample",
                                        use_list=None)
        w = B.train_BINARY(w, pos_weight=pos_weight, device=device, hidden_dims=list(hidden_dims),
                           n_epochs=int(n_epochs), lr=lr, key_added="BINARY", gradient_clipping=gradient_clipping,
                           weight_decay=weight_decay, verbose=verbose, random_seed=int(seed))
    adata.obsm[key_added] = np.asarray(w.obsm["BINARY"])
    out = _cluster(adata, n_domains, key_added, cluster_method=cluster_method, use_pca=False, n_comps=30,
                   cluster_seed=cluster_seed, domain_key=domain_key)
    return adata.obsm[key_added] if out is None else out


for _f in (stagate, graphst, spaceflow, cast, binary):
    _f.__doc__ = _f.__doc__.replace("    {CLUSTER_ARGS}", _CLUSTER_DOC.lstrip("\n"))
del _f

_FUNCS = {"stagate": stagate, "graphst": graphst, "spaceflow": spaceflow, "cellcharter": cellcharter,
          "cast": cast, "binary": binary}


def _run(method, adata, n_domains, *, use_rep, key_added, random_seed, **kwargs):
    """Entry used by ``spatial_domains(method=...)``."""
    f = _FUNCS[method]
    if use_rep is not None:
        kwargs["use_rep"] = use_rep
    kwargs.setdefault("domain_key", key_added)
    if method != "cellcharter":
        kwargs.setdefault("cluster_seed", random_seed)
    return f(adata, n_domains, **kwargs)
