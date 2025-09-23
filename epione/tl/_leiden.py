"""Standalone Leiden clustering function with removed relative dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, Literal
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from natsort import natsorted

if TYPE_CHECKING:
    from anndata import AnnData

# Type definitions
CSBase = Union[sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix]
_LegacyRandom = Union[int, np.random.RandomState, None]


def leiden(
    adata: AnnData,
    resolution: float = 1,
    *,
    restrict_to: Optional[Tuple[str, Sequence[str]]] = None,
    random_state: _LegacyRandom = 0,
    key_added: str = "leiden",
    adjacency: Optional[CSBase] = None,
    directed: Optional[bool] = None,
    use_weights: bool = True,
    n_iterations: int = -1,
    partition_type: Optional[type] = None,
    neighbors_key: Optional[str] = None,
    obsp: Optional[str] = None,
    copy: bool = False,
    flavor: Optional[Literal["leidenalg", "igraph"]] = None,
    **clustering_args,
) -> Optional[AnnData]:
    """
    Cluster cells into subgroups using the Leiden algorithm.
    
    The Leiden algorithm is an improved version of the Louvain algorithm.
    This requires having run neighbors() or similar first.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    resolution : float
        Parameter controlling clustering coarseness. Higher values = more clusters.
        Set to None if using partition_type that doesn't accept resolution.
    random_state : int or RandomState
        Random seed for reproducibility.
    restrict_to : tuple, optional
        Restrict clustering to specific categories: (obs_key, list_of_categories).
    key_added : str
        Key in adata.obs to store cluster labels.
    adjacency : sparse matrix, optional
        Adjacency matrix. If None, uses neighbor connectivities.
    directed : bool, optional
        Whether to treat graph as directed. Default True for leidenalg, False for igraph.
    use_weights : bool
        Whether to use edge weights in computation.
    n_iterations : int
        Number of iterations. -1 runs until optimal clustering.
    partition_type : type, optional
        Partition type for leidenalg. Defaults to RBConfigurationVertexPartition.
    neighbors_key : str, optional
        Key for neighbor connectivities.
    obsp : str, optional
        Use .obsp[obsp] as adjacency.
    copy : bool
        Whether to copy adata or modify in place.
    flavor : {'leidenalg', 'igraph'}, optional
        Which implementation to use. Default 'leidenalg'.
    **clustering_args
        Additional arguments for clustering algorithm.
        
    Returns
    -------
    AnnData or None
        Returns AnnData if copy=True, else None.
        Sets adata.obs[key_added] with cluster labels.
    """
    # Handle flavor selection
    if flavor is None:
        flavor = "leidenalg"
        warnings.warn(
            "In the future, the default backend for leiden will be igraph. "
            "To use igraph, pass: flavor='igraph' and n_iterations=2.",
            FutureWarning,
            stacklevel=2
        )
    
    if flavor not in {"igraph", "leidenalg"}:
        raise ValueError(f"flavor must be 'igraph' or 'leidenalg', got {flavor!r}")
    
    # Check for required packages
    try:
        import igraph
    except ImportError:
        raise ImportError("Please install igraph: pip install python-igraph")
    
    if flavor == "leidenalg":
        try:
            import leidenalg
        except ImportError:
            raise ImportError(
                "Please install leidenalg: "
                "conda install -c conda-forge leidenalg or pip install leidenalg"
            )
        if directed is None:
            directed = True
    else:  # igraph
        if directed:
            raise ValueError("Cannot use igraph's leiden with directed graph")
        if partition_type is not None:
            raise ValueError("Do not pass partition_type when using igraph")
        directed = False
    
    print(f"Running Leiden clustering with {flavor}...")
    
    # Copy if requested
    adata = adata.copy() if copy else adata
    
    # Get adjacency matrix
    if adjacency is None:
        adjacency = _choose_graph(adata, obsp, neighbors_key)
    
    # Handle restriction to specific categories
    if restrict_to is not None:
        restrict_key, restrict_categories = restrict_to
        adjacency, restrict_indices = restrict_adjacency(
            adata,
            restrict_key,
            restrict_categories=restrict_categories,
            adjacency=adjacency,
        )
    
    # Convert to igraph
    g = get_igraph_from_adjacency(adjacency, directed=directed)
    
    # Setup clustering arguments
    clustering_args = dict(clustering_args)
    clustering_args["n_iterations"] = n_iterations
    
    # Run clustering
    if flavor == "leidenalg":
        if resolution is not None:
            clustering_args["resolution_parameter"] = resolution
        if partition_type is None:
            partition_type = leidenalg.RBConfigurationVertexPartition
        if use_weights:
            clustering_args["weights"] = np.array(g.es["weight"]).astype(np.float64)
        clustering_args["seed"] = random_state
        part = leidenalg.find_partition(g, partition_type, **clustering_args)
    else:  # igraph
        if use_weights:
            clustering_args["weights"] = "weight"
        if resolution is not None:
            clustering_args["resolution"] = resolution
        clustering_args.setdefault("objective_function", "modularity")
        # Set random seed for igraph (handle different versions)
        if random_state is not None:
            if isinstance(random_state, int):
                _set_igraph_random_state(random_state)
        part = g.community_leiden(**clustering_args)
    
    # Get cluster assignments
    groups = np.array(part.membership)
    
    # Handle restricted clustering
    if restrict_to is not None:
        if key_added == "leiden":
            key_added += "_R"
        groups = rename_groups(
            adata,
            key_added=key_added,
            restrict_key=restrict_key,
            restrict_categories=restrict_categories,
            restrict_indices=restrict_indices,
            groups=groups,
        )
    
    # Store results
    adata.obs[key_added] = pd.Categorical(
        values=groups.astype("U"),
        categories=natsorted(map(str, np.unique(groups))),
    )
    
    # Store parameters
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = dict(
        resolution=resolution,
        random_state=random_state,
        n_iterations=n_iterations,
    )
    
    n_clusters = len(np.unique(groups))
    print(f"    Finished: found {n_clusters} clusters")
    print(f"    Added '{key_added}' to adata.obs (categorical)")
    
    return adata if copy else None


# Helper functions

def _set_igraph_random_state(seed):
    """Set random state for igraph, handling different versions."""
    import igraph
    
    try:
        # Try newer igraph versions (0.10+)
        import random
        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass
    
    try:
        # Try igraph 0.9.x method
        if hasattr(igraph, 'set_random_number_generator'):
            if hasattr(igraph, 'RNGType'):
                igraph.set_random_number_generator(igraph.RNGType.MT19937)
            igraph.random.seed(seed)
    except Exception:
        pass
    
    try:
        # Try older igraph versions
        if hasattr(igraph, 'seed'):
            igraph.seed(seed)
    except Exception:
        pass


def _choose_graph(adata, obsp=None, neighbors_key=None):
    """Choose connectivity graph from adata."""
    if obsp is not None and neighbors_key is not None:
        raise ValueError("Cannot specify both obsp and neighbors_key")
    
    if obsp is not None:
        return adata.obsp[obsp]
    
    if neighbors_key is None:
        neighbors_key = "neighbors"
    
    # Try to get connectivities
    if neighbors_key in adata.uns:
        neighbors = adata.uns[neighbors_key]
        if "connectivities_key" in neighbors:
            conn_key = neighbors["connectivities_key"]
        else:
            conn_key = "connectivities"
    else:
        # Fallback to default
        conn_key = "connectivities"
    
    if conn_key in adata.obsp:
        return adata.obsp[conn_key]
    else:
        raise ValueError(
            f"No connectivity matrix found. "
            f"Please run neighbors() first or specify adjacency/obsp."
        )


def get_igraph_from_adjacency(adjacency, directed=False):
    """Convert adjacency matrix to igraph Graph."""
    import igraph as ig
    
    # Convert to COO format for easier extraction
    if not sparse.issparse(adjacency):
        adjacency = sparse.csr_matrix(adjacency)
    
    adjacency = adjacency.tocoo()
    
    # Create edge list
    edges = list(zip(adjacency.row, adjacency.col))
    
    # Create graph
    g = ig.Graph(edges=edges, directed=directed)
    
    # Add weights if present
    if adjacency.data is not None:
        g.es["weight"] = adjacency.data
    
    # Ensure we have the right number of vertices
    n_vertices = adjacency.shape[0]
    if g.vcount() < n_vertices:
        g.add_vertices(n_vertices - g.vcount())
    
    return g


def restrict_adjacency(
    adata,
    restrict_key,
    restrict_categories,
    adjacency,
):
    """Restrict adjacency matrix to specific categories."""
    # Get mask for selected categories
    restrict_indices = []
    
    if restrict_key not in adata.obs:
        raise ValueError(f"'{restrict_key}' not found in adata.obs")
    
    obs_values = adata.obs[restrict_key]
    for cat in restrict_categories:
        mask = obs_values == cat
        restrict_indices.extend(np.where(mask)[0].tolist())
    
    restrict_indices = np.array(restrict_indices)
    
    # Subset adjacency matrix
    adjacency = adjacency[restrict_indices, :][:, restrict_indices]
    
    return adjacency, restrict_indices


def rename_groups(
    adata,
    key_added,
    restrict_key,
    restrict_categories,
    restrict_indices,
    groups,
):
    """Rename groups after restricted clustering."""
    # Create full group array
    groups_full = np.array(adata.obs[restrict_key].values, dtype=object)
    
    # Get existing groups if they exist
    if key_added in adata.obs:
        groups_full = adata.obs[key_added].values.copy()
    
    # Update restricted indices with new groups
    # Offset group numbers by max existing group
    if key_added in adata.obs:
        existing_groups = pd.Categorical(groups_full).categories
        if len(existing_groups) > 0:
            max_group = max([int(g) for g in existing_groups if g.isdigit()])
            groups = groups + max_group + 1
    
    groups_full[restrict_indices] = groups.astype(str)
    
    return groups_full


# Convenience functions for different clustering algorithms
def louvain(
    adata: AnnData,
    resolution: Optional[float] = None,
    random_state: _LegacyRandom = 0,
    restrict_to: Optional[Tuple[str, Sequence[str]]] = None,
    key_added: str = "louvain",
    adjacency: Optional[CSBase] = None,
    flavor: Literal["vtraag", "igraph"] = "igraph",  # Changed default to igraph
    directed: bool = False,  # Changed default to False
    use_weights: bool = False,
    partition_type: Optional[type] = None,
    partition_kwargs: Optional[dict] = None,
    neighbors_key: Optional[str] = None,
    obsp: Optional[str] = None,
    copy: bool = False,
    **kwargs  # Added to catch any extra arguments
) -> Optional[AnnData]:
    """
    Cluster cells using the Louvain algorithm.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    resolution : float, optional
        Resolution parameter for clustering (only for vtraag flavor).
    random_state : int
        Random seed.
    restrict_to : tuple, optional
        Restrict clustering to specific categories: (obs_key, list_of_categories).
    key_added : str
        Key for storing results.
    adjacency : sparse matrix, optional
        Adjacency matrix to use.
    flavor : {'vtraag', 'igraph'}
        Which implementation to use. 'vtraag' uses the louvain package,
        'igraph' uses igraph's built-in method.
    directed : bool
        Whether to treat graph as directed.
    use_weights : bool
        Whether to use edge weights.
    partition_type : type, optional
        Partition type for vtraag flavor.
    partition_kwargs : dict, optional
        Additional kwargs for vtraag partition.
    neighbors_key : str, optional
        Key for neighbors.
    obsp : str, optional
        Key in obsp for adjacency.
    copy : bool
        Whether to copy adata.
        
    Returns
    -------
    AnnData or None
        Returns AnnData if copy=True.
    """
    try:
        import igraph
    except ImportError:
        raise ImportError("Please install igraph: pip install python-igraph")
    
    if partition_kwargs is None:
        partition_kwargs = {}
    
    print(f"Running Louvain clustering with flavor='{flavor}'...")
    
    # Handle flavor-specific requirements
    if flavor == "vtraag":
        try:
            import louvain
        except ImportError:
            # Fall back to igraph if louvain package not available
            print("Warning: louvain package not found, falling back to igraph flavor")
            flavor = "igraph"
    
    if flavor == "igraph" and resolution is not None:
        print("Warning: resolution parameter has no effect for flavor='igraph'")
    
    if (flavor != "vtraag") and (partition_type is not None):
        raise ValueError("partition_type is only valid when flavor='vtraag'")
    
    # Copy if requested
    adata = adata.copy() if copy else adata
    
    # Get adjacency matrix
    if adjacency is None:
        adjacency = _choose_graph(adata, obsp, neighbors_key)
    
    # Handle restriction to specific categories
    if restrict_to is not None:
        restrict_key, restrict_categories = restrict_to
        adjacency, restrict_indices = restrict_adjacency(
            adata,
            restrict_key,
            restrict_categories=restrict_categories,
            adjacency=adjacency,
        )
    
    # Convert to igraph
    if directed and flavor == "igraph":
        directed = False
    
    g = get_igraph_from_adjacency(adjacency, directed=directed)
    weights = np.array(g.es["weight"]).astype(np.float64) if use_weights else None
    
    # Run clustering
    if flavor == "vtraag":
        import louvain
        
        if partition_type is None:
            partition_type = louvain.RBConfigurationVertexPartition
        
        if resolution is not None:
            partition_kwargs["resolution_parameter"] = resolution
        
        if use_weights:
            partition_kwargs["weights"] = weights
        
        # Handle random state for different louvain versions
        try:
            from packaging.version import Version
            if Version(louvain.__version__) < Version("0.7.0"):
                louvain.set_rng_seed(random_state)
            else:
                partition_kwargs["seed"] = random_state
        except:
            # Try both methods if version check fails
            try:
                partition_kwargs["seed"] = random_state
            except:
                try:
                    louvain.set_rng_seed(random_state)
                except:
                    pass
        
        part = louvain.find_partition(g, partition_type, **partition_kwargs)
    
    else:  # igraph flavor
        # Set random seed for igraph
        if random_state is not None and isinstance(random_state, int):
            _set_igraph_random_state(random_state)
        
        # Use igraph's built-in Louvain (multilevel community detection)
        part = g.community_multilevel(weights=weights)
    
    groups = np.array(part.membership)
    
    # Handle restricted clustering
    if restrict_to is not None:
        if key_added == "louvain":
            key_added += "_R"
        groups = rename_groups(
            adata,
            key_added=key_added,
            restrict_key=restrict_key,
            restrict_categories=restrict_categories,
            restrict_indices=restrict_indices,
            groups=groups,
        )
    
    # Store results
    adata.obs[key_added] = pd.Categorical(
        values=groups.astype("U"),
        categories=natsorted(map(str, np.unique(groups))),
    )
    
    # Store parameters
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = dict(
        resolution=resolution,
        random_state=random_state,
    )
    
    n_clusters = len(np.unique(groups))
    print(f"    Finished: found {n_clusters} clusters")
    print(f"    Added '{key_added}' to adata.obs (categorical)")
    
    return adata if copy else None


