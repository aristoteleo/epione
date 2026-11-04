import numpy as np
from anndata import AnnData

from ._leiden import leiden,louvain
from ..external.rph_kmeans import RPHKMeans

def clusters(
    adata:AnnData,
    method:str='leiden',
    leiden_kwargs=None,
    louvain_kwargs=None,
    rph_kmeans_kwargs=None,
    kmeans_kwargs=None,
    hierarchical_kwargs=None,
    use_rep:str='X_pca',
    key_added:str='clusters',
    backend:str='igraph',
    **kwargs
):
    r"""
    Cluster cells into subgroups using different methods.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    method : str
        Method to use for clustering.
    leiden_kwargs : dict
        Keyword arguments for leiden.
    louvain_kwargs : dict
        Keyword arguments for louvain.
    rph_kmeans_kwargs : dict
        Keyword arguments for rph_kmeans.
    kmeans_kwargs : dict
        Keyword arguments for kmeans.
    hierarchical_kwargs : dict
        Keyword arguments for hierarchical.
    use_rep : str
        Use the representation of the data.
    key_added : str
        Key to store the cluster labels.
    backend : {'igraph', 'leidenalg', 'rapids'}
        Leiden implementation (``method='leiden'``; ignored otherwise).
        ``'igraph'`` (default): igraph's C Leiden with the objective of the
        previous default -- modularity (≡ leidenalg's RBConfiguration) at
        ``resolution`` (default 1), edge weights, iterated to convergence
        (``n_iterations=-1``), undirected graph. On 50k / 100k-cell graphs
        its partitions agree with leidenalg's as closely as leidenalg's own
        seeds agree with each other (ARI ~0.6) at equal or higher modularity,
        8-10x faster (~25 s vs ~4 min at 100k cells; leidenalg needs hours at
        1M). ``'leidenalg'`` (alias ``'epione'``): the previous default,
        :func:`epione.tl.leiden` with leidenalg (``leiden_kwargs`` passed
        through). ``'rapids'``: cuGraph Leiden through
        :func:`rapids_singlecell.tl.leiden` on the same neighbour graph;
        needs ``rapids_singlecell`` (see the GPU acceleration page);
        ``leiden_kwargs`` are passed through except ``flavor`` /
        ``directed`` / ``partition_type``.
    **kwargs
        Additional arguments for clustering algorithm.
        
    Returns
    -------
    AnnData
        Annotated data matrix with cluster labels in adata.obs[key_added].

    Notes
    -----
    leiden_kwargs:
        resolution: float
            Parameter controlling clustering coarseness. Higher values = more clusters.
            Set to None if using partition_type that doesn't accept resolution.
        random_state: int
            Random seed for reproducibility.
        n_iterations: int
            Number of iterations. -1 runs until optimal clustering.
        partition_type: type
            Partition type for leidenalg. Defaults to RBConfigurationVertexPartition.
        neighbors_key: str
            Key for neighbor connectivities.
        obsp: str
            Key in obsp for adjacency.
        copy: bool
            Whether to copy adata.
        flavor: str
    louvain_kwargs:
        resolution: float
            Parameter controlling clustering coarseness. Higher values = more clusters.
            Set to None if using partition_type that doesn't accept resolution.
        random_state: int
            Random seed for reproducibility.
        partition_type: type
            Partition type for vtraag. Defaults to RBConfigurationVertexPartition.
        neighbors_key: str
            Key for neighbor connectivities.
        obsp: str
            Key in obsp for adjacency.
        copy: bool
            Whether to copy adata.
        flavor: str
    rph_kmeans_kwargs:
        n_init: int
            Number of initializations.
        n_clusters: int
            Number of clusters.
        min_cluster_size: int
            Minimum cluster size.
        max_point: int
            Maximum number of points.
        allow_fewer_clusters: bool
            Whether to allow fewer clusters.
        random_state: int
            Random seed for reproducibility.
        proj_num: int
            Number of vector for random projection process.
        max_iter: int
            Maximum number of iterations.
        sample_dist_num: int
            Number of paired samples chosen to decide default w.
        bkt_improve: str
            Methods of improving bucket quality.
        radius_divide: float
            Radius for 'radius' bucket-improving method.
        bkt_size_keepr: float
            Keep ratio for 'min_bkt_size' bucket-improving method.
        center_dist_keepr: float
            Keep ratio for 'min_center_dist' bucket-improving method.
        min_point: int
            Optional lower bound on the number of reduced points (skeleton size).
        reduced_kmeans_kwargs: dict
            kwargs of kmeans to find centers of reduced point.
        final_kmeans_kwargs: dict
            kwargs of kmeans after center initialization.
        verbose: int
            Controls the verbosity.
        random_state: int
            Global random seed to make projection, sampling and KMeans reproducible.
    kmeans_kwargs:
        n_clusters: int
            Number of clusters.
        random_state: int
            Random seed for reproducibility.
    hierarchical_kwargs:
        n_clusters: int
            Number of clusters.
        random_state: int
            Random seed for reproducibility.
    """
    if backend not in ('igraph', 'leidenalg', 'epione', 'rapids'):
        raise ValueError(
            f"backend must be 'igraph', 'leidenalg' or 'rapids', got {backend!r}")
    if backend == 'rapids' and method != 'leiden':
        raise ValueError("backend='rapids' is available for method='leiden' only")
    if method == 'leiden':
        if leiden_kwargs is None:
            leiden_kwargs = {
                'key_added':key_added
            }
        else:
            leiden_kwargs['key_added'] = key_added
        if backend == 'igraph':
            # igraph's C Leiden on the objective of the leidenalg call below:
            # modularity == RBConfiguration at the same resolution, weighted,
            # iterated to convergence, undirected
            kw = dict(leiden_kwargs)
            if kw.get('flavor', 'igraph') != 'igraph':
                raise ValueError("backend='igraph' runs flavor='igraph'; use "
                                 "backend='leidenalg' for leidenalg")
            kw['flavor'] = 'igraph'
            kw.setdefault('n_iterations', -1)
            kw.setdefault('directed', False)
            kw.pop('partition_type', None)
            return leiden(adata, **kw)
        if backend == 'rapids':
            from ..core._rapids import rapids_leiden
            kw = {k: v for k, v in leiden_kwargs.items()
                  if k not in ('flavor', 'directed', 'partition_type')}
            rapids_leiden(adata, **kw)
            return adata
        return leiden(adata, **leiden_kwargs)
    elif method == 'louvain':
        if louvain_kwargs is None:
            louvain_kwargs = {
                'key_added':key_added
            }
        else:
            louvain_kwargs['key_added'] = key_added
        return louvain(adata, **louvain_kwargs)
    elif method == 'rph_kmeans':
        if rph_kmeans_kwargs is None:
            rph_kmeans_kwargs = {
                'n_init':100, 'n_clusters':20,
                'min_cluster_size':50,'max_point':200,'allow_fewer_clusters':True,
                'random_state':42
            }

        clt = RPHKMeans(**rph_kmeans_kwargs)
        if use_rep not in adata.obsm:
            raise ValueError(f"use_rep {use_rep} not found in adata.obsm")
        clt_labels = clt.fit_predict(adata.obsm[use_rep])
        adata.obs[key_added] = [str(i) for i in clt_labels]
        return adata
    elif method == 'kmeans++':
        if kmeans_kwargs is None:
            kmeans_kwargs = {
                'n_clusters':20,
                'random_state':42
            }
        from sklearn.cluster import KMeans
        clt = KMeans(init='k-means++',**kmeans_kwargs)
        clt_labels = clt.fit_predict(adata.obsm[use_rep])
        adata.obs[key_added] = [str(i) for i in clt_labels]
        return adata
    elif method == 'hierarchical':
        if hierarchical_kwargs is None:
            hierarchical_kwargs = {
                'n_clusters':20,
                'random_state':42
            }
        from sklearn.cluster import AgglomerativeClustering
        clt = AgglomerativeClustering(**hierarchical_kwargs)
        clt_labels = clt.fit_predict(adata.obsm[use_rep])
        adata.obs[key_added] = [str(i) for i in clt_labels]
        return adata
    else:
        raise ValueError(f"Method {method} not supported")