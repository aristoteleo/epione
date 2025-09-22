"""Standalone neighbors function with removed relative dependencies."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, Mapping, NamedTuple, TypedDict
from collections.abc import Callable
from types import MappingProxyType

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix, issparse
from sklearn.utils import check_random_state

if TYPE_CHECKING:
    from typing import NotRequired
    from anndata import AnnData


# Type definitions
_Method = Literal["umap", "gauss"]
_Metric = Literal[
    "euclidean", "manhattan", "cosine", "correlation", "hamming", 
    "jaccard", "chebyshev", "minkowski", "braycurtis", "canberra"
]
_MetricFn = Callable[[np.ndarray, np.ndarray], float]
KnnTransformerLike = Any  # Would be sklearn-compatible transformer
_KnownTransformer = Literal["pynndescent", "sklearn", "rapids"]
_LegacyRandom = int | np.random.RandomState | None


class NeighborsParams(TypedDict):
    """Parameters for neighbors computation."""
    n_neighbors: int
    method: _Method
    random_state: _LegacyRandom
    metric: _Metric | _MetricFn
    metric_kwds: NotRequired[Mapping[str, Any]]
    use_rep: NotRequired[str]
    n_pcs: NotRequired[int]


class KwdsForTransformer(TypedDict):
    """Keyword arguments passed to a transformer."""
    n_neighbors: int
    metric: _Metric | _MetricFn
    metric_params: Mapping[str, Any]
    random_state: _LegacyRandom


def _choose_representation(
    adata: AnnData,
    use_rep: str | None = None,
    n_pcs: int | None = None
) -> np.ndarray:
    """Choose the representation for computing neighbors."""
    if use_rep is not None and use_rep not in adata.obsm:
        raise ValueError(f"Representation {use_rep} not found in adata.obsm")
    
    if use_rep is not None:
        X = adata.obsm[use_rep]
    elif "X_pca" in adata.obsm:
        X = adata.obsm["X_pca"]
    else:
        X = adata.X
    
    # Handle sparse matrices
    if issparse(X):
        X = X.toarray()
    
    # Limit to n_pcs if specified
    if n_pcs is not None and X.shape[1] > n_pcs:
        X = X[:, :n_pcs]
    
    return X


def _get_indices_distances_from_sparse_matrix(
    D: csr_matrix,
    n_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract indices and distances from sparse distance matrix."""
    indices = np.zeros((D.shape[0], n_neighbors), dtype=np.int32)
    distances = np.zeros((D.shape[0], n_neighbors), dtype=np.float32)
    
    for i in range(D.shape[0]):
        row_data = D.getrow(i).toarray().ravel()
        row_indices = np.argsort(row_data)[:n_neighbors]
        indices[i] = row_indices
        distances[i] = row_data[row_indices]
    
    return indices, distances


def _get_sparse_matrix_from_indices_distances(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    shape: tuple[int, int] | None = None,
    keep_self: bool = False
) -> csr_matrix:
    """Construct sparse distance matrix from indices and distances."""
    n_obs = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    
    if shape is None:
        shape = (n_obs, n_obs)
    
    row_indices = np.repeat(np.arange(n_obs), n_neighbors)
    col_indices = knn_indices.ravel()
    data = knn_distances.ravel()
    
    if not keep_self:
        # Remove self-connections
        mask = row_indices != col_indices
        row_indices = row_indices[mask]
        col_indices = col_indices[mask]
        data = data[mask]
    
    return csr_matrix((data, (row_indices, col_indices)), shape=shape)


def compute_connectivities_umap(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    n_obs: int,
    n_neighbors: int,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: float = 1.0,
) -> csr_matrix:
    """Compute connectivities using UMAP method."""
    from umap.umap_ import fuzzy_simplicial_set
    
    X = csr_matrix(([], ([], [])), shape=(n_obs, 1))
    
    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors=n_neighbors,
        random_state=None,
        metric="precomputed",
        knn_indices=knn_indices,
        knn_dists=knn_distances,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )
    
    return connectivities[0]


def compute_connectivities_gauss(
    distances: np.ndarray | csr_matrix,
    n_neighbors: int,
    knn: bool = True
) -> csr_matrix:
    """Compute connectivities using Gaussian kernel."""
    if issparse(distances):
        distances = distances.toarray()
    
    n_obs = distances.shape[0]
    
    # Compute adaptive kernel width (distance to n_neighbors-th neighbor)
    sigma = np.zeros(n_obs)
    for i in range(n_obs):
        sorted_dists = np.sort(distances[i])
        # Use n_neighbors-th distance as kernel width
        sigma[i] = sorted_dists[min(n_neighbors, len(sorted_dists) - 1)]
    
    # Compute Gaussian kernel
    connectivities = np.zeros_like(distances)
    for i in range(n_obs):
        if sigma[i] > 0:
            connectivities[i] = np.exp(-distances[i]**2 / (2 * sigma[i]**2))
        
        if knn:
            # Keep only n_neighbors closest
            sorted_indices = np.argsort(distances[i])
            connectivities[i, sorted_indices[n_neighbors:]] = 0
    
    # Make symmetric
    connectivities = (connectivities + connectivities.T) / 2
    
    return csr_matrix(connectivities)


def neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    *,
    use_rep: str | None = None,
    knn: bool = True,
    method: _Method = "umap",
    transformer: KnnTransformerLike | _KnownTransformer | None = None,
    metric: _Metric | _MetricFn = "euclidean",
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    random_state: _LegacyRandom = 0,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """
    Compute the nearest neighbors distance matrix and neighborhood graph.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    n_neighbors
        The size of local neighborhood (in terms of number of neighboring data
        points) used for manifold approximation. Larger values result in more
        global views of the manifold, while smaller values result in more local
        data being preserved. In general values should be in the range 2 to 100.
    n_pcs
        Use n_pcs principal components for computing neighborhoods.
    use_rep
        Use the indicated representation. 'X' or any key for .obsm is valid.
        If None, use 'X_pca' if present, otherwise use .X
    knn
        If True, use a hard threshold to restrict the number of neighbors to
        n_neighbors, that is, consider a knn graph. Otherwise, use a Gaussian
        Kernel to assign low weights to neighbors more distant than the
        n_neighbors nearest neighbor.
    method
        Use 'umap' or 'gauss' for computing connectivities.
    transformer
        KNN search implementation. If None, uses sklearn for small datasets,
        pynndescent for large ones.
    metric
        Distance metric to use.
    metric_kwds
        Options for the metric.
    random_state
        Random seed.
    key_added
        If not specified, the neighbors data is stored in .uns['neighbors'],
        distances and connectivities are stored in .obsp['distances'] and
        .obsp['connectivities'] respectively.
    copy
        Return a copy instead of writing to adata.
    
    Returns
    -------
    Returns None if copy=False, else returns an AnnData object.
    """
    print(f"Computing neighbors with n_neighbors={n_neighbors}")
    
    adata = adata.copy() if copy else adata
    
    # Adjust n_neighbors for very small datasets
    if n_neighbors > adata.shape[0]:
        n_neighbors = 1 + int(0.5 * adata.shape[0])
        warnings.warn(f"n_obs too small: adjusting to n_neighbors = {n_neighbors}")
    
    # Choose representation
    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
    
    # Setup transformer
    if transformer is None or transformer == "sklearn":
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(
            n_neighbors=min(n_neighbors, X.shape[0] - 1),
            algorithm="auto" if X.shape[0] > 100 else "brute",
            metric=metric,
            metric_params=dict(metric_kwds),
            n_jobs=-1
        )
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        
    elif transformer == "pynndescent":
        try:
            import pynndescent
            
            index = pynndescent.NNDescent(
                X,
                n_neighbors=n_neighbors,
                metric=metric,
                metric_kwds=dict(metric_kwds),
                random_state=random_state,
                n_jobs=-1,
            )
            indices, distances = index.neighbor_graph
            
        except ImportError:
            warnings.warn("pynndescent not installed, falling back to sklearn")
            return neighbors(
                adata, n_neighbors, n_pcs,
                use_rep=use_rep, knn=knn, method=method,
                transformer="sklearn", metric=metric,
                metric_kwds=metric_kwds, random_state=random_state,
                key_added=key_added, copy=False
            )
    else:
        # Use custom transformer
        if hasattr(transformer, 'fit_transform'):
            distances_matrix = transformer.fit_transform(X)
            indices, distances = _get_indices_distances_from_sparse_matrix(
                distances_matrix, n_neighbors
            )
        else:
            raise ValueError(f"Unknown transformer: {transformer}")
    
    # Create sparse distance matrix
    distances_matrix = _get_sparse_matrix_from_indices_distances(
        indices, distances, shape=(adata.n_obs, adata.n_obs)
    )
    
    # Compute connectivities
    if method == "umap":
        try:
            connectivities = compute_connectivities_umap(
                indices, distances, adata.n_obs, n_neighbors
            )
        except ImportError:
            warnings.warn("UMAP not installed, falling back to Gaussian kernel")
            connectivities = compute_connectivities_gauss(
                distances_matrix, n_neighbors, knn=knn
            )
    elif method == "gauss":
        connectivities = compute_connectivities_gauss(
            distances_matrix, n_neighbors, knn=knn
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Store results
    if key_added is None:
        key_added = "neighbors"
        conns_key = "connectivities"
        dists_key = "distances"
    else:
        conns_key = f"{key_added}_connectivities"
        dists_key = f"{key_added}_distances"
    
    adata.uns[key_added] = {}
    neighbors_dict = adata.uns[key_added]
    
    neighbors_dict["connectivities_key"] = conns_key
    neighbors_dict["distances_key"] = dists_key
    neighbors_dict["params"] = NeighborsParams(
        n_neighbors=n_neighbors,
        method=method,
        random_state=random_state,
        metric=metric,
    )
    
    if metric_kwds:
        neighbors_dict["params"]["metric_kwds"] = metric_kwds
    if use_rep is not None:
        neighbors_dict["params"]["use_rep"] = use_rep
    if n_pcs is not None:
        neighbors_dict["params"]["n_pcs"] = n_pcs
    
    adata.obsp[dists_key] = distances_matrix
    adata.obsp[conns_key] = connectivities
    
    print(f"Finished computing neighbors")
    print(f"    Added to .uns['{key_added}']")
    print(f"    .obsp['{dists_key}'], distances for each pair of neighbors")
    print(f"    .obsp['{conns_key}'], weighted adjacency matrix")
    
    return adata if copy else None