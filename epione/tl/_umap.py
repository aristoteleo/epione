"""Standalone UMAP function with removed relative dependencies."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, Mapping
import numpy as np
from sklearn.utils import check_array, check_random_state
from scipy.sparse import issparse, coo_matrix

if TYPE_CHECKING:
    from anndata import AnnData

# Type definitions
_InitPos = Literal["paga", "spectral", "random"]
_LegacyRandom = int | np.random.RandomState | None


class NeighborsView:
    """Simple neighbors view to access neighbor data."""
    
    def __init__(self, adata: AnnData, key: str = "neighbors"):
        self.adata = adata
        self.key = key
        self._neighbors = adata.uns.get(key, {})
    
    def __getitem__(self, item):
        if item == "connectivities":
            conn_key = self._neighbors.get("connectivities_key", "connectivities")
            return self.adata.obsp[conn_key]
        elif item == "distances":
            dist_key = self._neighbors.get("distances_key", "distances")
            return self.adata.obsp[dist_key]
        elif item == "params":
            return self._neighbors.get("params", {})
        else:
            return self._neighbors.get(item)
    
    def __contains__(self, item):
        if item in ["connectivities", "distances"]:
            return True
        return item in self._neighbors


def _choose_representation(
    adata: AnnData,
    use_rep: str | None = None,
    n_pcs: int | None = None,
    silent: bool = False
) -> np.ndarray:
    """Choose the representation for computation."""
    if use_rep is not None and use_rep not in adata.obsm:
        if not silent:
            warnings.warn(f"Representation {use_rep} not found in adata.obsm")
        use_rep = None
    
    if use_rep is not None:
        X = adata.obsm[use_rep]
    elif "X_pca" in adata.obsm:
        X = adata.obsm["X_pca"]
    else:
        X = adata.X if adata.X is not None else adata.raw.X
    
    # Handle sparse matrices
    if issparse(X):
        X = X.toarray()
    
    # Limit to n_pcs if specified
    if n_pcs is not None and X.shape[1] > n_pcs:
        X = X[:, :n_pcs]
    
    return np.asarray(X)


def get_init_pos_from_paga(
    adata: AnnData,
    random_state: _LegacyRandom = 0,
    neighbors_key: str = "neighbors"
) -> np.ndarray:
    """Initialize positions from PAGA graph."""
    if "paga" not in adata.uns:
        raise ValueError(
            "No PAGA layout found. Run `sc.tl.paga` first, "
            "or choose another initialization method."
        )
    
    paga_data = adata.uns["paga"]
    
    # Get group assignments
    if "groups" in paga_data:
        groups_key = paga_data.get("groups")
        if groups_key in adata.obs:
            groups = adata.obs[groups_key].cat.codes.values
        else:
            raise ValueError(f"Groups key {groups_key} not found in adata.obs")
    else:
        raise ValueError("No groups information in PAGA data")
    
    # Get PAGA positions
    if "pos" in paga_data:
        paga_pos = np.asarray(paga_data["pos"])
    else:
        raise ValueError("No PAGA positions found. Run `sc.pl.paga` first.")
    
    # Map PAGA positions to cells
    n_groups = paga_pos.shape[0]
    init_pos = np.zeros((adata.n_obs, 2))
    
    # Add small random noise to avoid overlapping points
    random_state = check_random_state(random_state)
    noise_level = 0.1
    
    for i in range(adata.n_obs):
        group = groups[i]
        if 0 <= group < n_groups:
            init_pos[i] = paga_pos[group] + random_state.randn(2) * noise_level
        else:
            # Random initialization for cells without group
            init_pos[i] = random_state.randn(2)
    
    return init_pos


def umap(
    adata: AnnData,
    *,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: _InitPos | np.ndarray | None = "spectral",
    random_state: _LegacyRandom = 0,
    a: float | None = None,
    b: float | None = None,
    method: Literal["umap", "rapids"] = "umap",
    key_added: str | None = None,
    neighbors_key: str = "neighbors",
    copy: bool = False,
) -> AnnData | None:
    """
    Embed the neighborhood graph using UMAP (Uniform Manifold Approximation and Projection).
    
    UMAP is a manifold learning technique suitable for visualizing high-dimensional data.
    It optimizes the embedding such that it best reflects the topology of the data,
    which is represented using a neighborhood graph.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    min_dist
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points on
        the manifold are drawn closer together, while larger values will result
        on a more even dispersal of points. Default is 0.5.
    spread
        The effective scale of embedded points. In combination with min_dist
        this determines how clustered/clumped the embedded points are.
    n_components
        The number of dimensions of the embedding.
    maxiter
        The number of iterations (epochs) of the optimization. Called n_epochs
        in the original UMAP. If None, uses 500 for <=10k cells, 200 for >10k cells.
    alpha
        The initial learning rate for the embedding optimization.
    gamma
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.
    negative_sample_rate
        The number of negative edge/1-simplex samples to use per positive
        edge/1-simplex sample in optimizing the low dimensional embedding.
    init_pos
        How to initialize the low dimensional embedding. Options are:
        - Any key in adata.obsm
        - 'paga': positions from PAGA graph
        - 'spectral': use a spectral embedding of the graph
        - 'random': assign initial embedding positions at random
        - A numpy array of initial embedding positions
    random_state
        Random seed. If int, seed for random number generator;
        If RandomState, the random number generator;
        If None, uses np.random.
    a
        More specific parameter controlling the embedding. If None,
        set automatically as determined by min_dist and spread.
    b
        More specific parameter controlling the embedding. If None,
        set automatically as determined by min_dist and spread.
    method
        Implementation to use:
        - 'umap': Standard UMAP implementation
        - 'rapids': GPU accelerated (requires cuML)
    key_added
        If not specified, the embedding is stored in obsm['X_umap'] and
        parameters in uns['umap']. If specified, stored in obsm[key_added]
        and uns[key_added].
    neighbors_key
        Key in uns to find neighbors settings and connectivity matrix.
    copy
        Return a copy instead of writing to adata.
    
    Returns
    -------
    Returns None if copy=False, else returns an AnnData object.
    Sets the following fields:
    - adata.obsm['X_umap' | key_added]: UMAP coordinates
    - adata.uns['umap' | key_added]: UMAP parameters
    """
    adata = adata.copy() if copy else adata
    
    # Determine storage keys
    key_obsm = "X_umap" if key_added is None else key_added
    key_uns = "umap" if key_added is None else key_added
    
    # Check for neighbors
    if neighbors_key not in adata.uns:
        raise ValueError(
            f"Neighbors not found at .uns['{neighbors_key}']. "
            "Run neighbors() function first."
        )
    
    print(f"Computing UMAP embedding...")
    
    # Access neighbors data
    neighbors = NeighborsView(adata, neighbors_key)
    
    # Check if neighbors were computed with UMAP method
    if "params" in neighbors:
        neigh_method = neighbors["params"].get("method", "unknown")
        if neigh_method != "umap":
            warnings.warn(
                f"Neighbors were computed with method '{neigh_method}', not 'umap'. "
                "This may affect UMAP results."
            )
    
    # Import UMAP with warning suppression
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
        try:
            import umap
            from umap.umap_ import find_ab_params, simplicial_set_embedding
        except ImportError:
            raise ImportError(
                "umap-learn package not installed. "
                "Install with: pip install umap-learn"
            )
    
    # Calculate a and b parameters if not provided
    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)
    
    # Store parameters
    adata.uns[key_uns] = {
        "params": {
            "a": a,
            "b": b,
            "min_dist": min_dist,
            "spread": spread,
            "n_components": n_components,
            "random_state": random_state if random_state != 0 else 0
        }
    }
    
    # Handle initialization
    if isinstance(init_pos, str) and init_pos in adata.obsm:
        # Use existing embedding
        init_coords = adata.obsm[init_pos]
    elif isinstance(init_pos, str) and init_pos == "paga":
        # Initialize from PAGA
        init_coords = get_init_pos_from_paga(
            adata, random_state=random_state, neighbors_key=neighbors_key
        )
    else:
        # Let UMAP handle initialization (spectral, random, or custom array)
        init_coords = init_pos
    
    # Ensure correct dtype for initialization coordinates
    if hasattr(init_coords, "dtype"):
        init_coords = check_array(init_coords, dtype=np.float32, accept_sparse=False)
    
    # Setup random state
    random_state = check_random_state(random_state)
    
    # Get data representation
    neigh_params = neighbors["params"] if "params" in neighbors else {}
    X = _choose_representation(
        adata,
        use_rep=neigh_params.get("use_rep", None),
        n_pcs=neigh_params.get("n_pcs", None),
        silent=True,
    )
    
    if method == "umap":
        # Standard UMAP implementation
        default_epochs = 500 if neighbors["connectivities"].shape[0] <= 10000 else 200
        n_epochs = default_epochs if maxiter is None else maxiter
        
        # Get connectivity matrix in COO format
        connectivities = neighbors["connectivities"]
        if not isinstance(connectivities, coo_matrix):
            connectivities = connectivities.tocoo()
        
        # Run UMAP embedding
        X_umap, _ = simplicial_set_embedding(
            data=X,
            graph=connectivities,
            n_components=n_components,
            initial_alpha=alpha,
            a=a,
            b=b,
            gamma=gamma,
            negative_sample_rate=negative_sample_rate,
            n_epochs=n_epochs,
            init=init_coords,
            random_state=random_state,
            metric=neigh_params.get("metric", "euclidean"),
            metric_kwds=neigh_params.get("metric_kwds", {}),
            densmap=False,
            densmap_kwds={},
            output_dens=False,
            verbose=False,
        )
        
    elif method == "rapids":
        # GPU-accelerated UMAP
        warnings.warn(
            "method='rapids' requires cuML library. "
            "Consider using standard 'umap' method if cuML is not available.",
            UserWarning
        )
        
        metric = neigh_params.get("metric", "euclidean")
        if metric != "euclidean":
            raise ValueError(
                f"Rapids UMAP only supports 'euclidean' metric, got '{metric}'"
            )
        
        try:
            from cuml import UMAP
        except ImportError:
            raise ImportError(
                "cuML not installed. Install with appropriate CUDA version "
                "or use method='umap' instead."
            )
        
        n_neighbors = neigh_params.get("n_neighbors", 15)
        n_epochs = 500 if maxiter is None else maxiter
        
        # Ensure contiguous array for RAPIDS
        X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
        
        # Create and run RAPIDS UMAP
        umap_rapids = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            n_epochs=n_epochs,
            learning_rate=alpha,
            init=init_pos if init_pos != "paga" else "spectral",
            min_dist=min_dist,
            spread=spread,
            negative_sample_rate=negative_sample_rate,
            a=a,
            b=b,
            verbose=False,
            random_state=random_state,
        )
        X_umap = umap_rapids.fit_transform(X_contiguous)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Store results
    adata.obsm[key_obsm] = X_umap
    
    print(f"    Finished computing UMAP")
    print(f"    Added:")
    print(f"        '{key_obsm}', UMAP coordinates (adata.obsm)")
    print(f"        '{key_uns}', UMAP parameters (adata.uns)")
    
    return adata if copy else None

