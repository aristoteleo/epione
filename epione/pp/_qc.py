from anndata import AnnData
import snapatac2._snapatac2 as internal
import numpy as np 
import snapatac2
import scipy.sparse as ss

from snapatac2.tools._embedding import spectral

from ..utils import console


def qc(
    adata:AnnData,
    tresh=None,
    gtf_file=None,
    n_features=500000,
              ):
    r"""
    Filter cells based on QC metrics.

    Parameters
    ----------
    adata: An AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to peaks.
    tresh: A dictionary of QC thresholds. The keys should be 'fragment_counts_min', 'fragment_counts_max',
          'TSS_score_min', 'TSS_score_max', 'Nucleosome_singal_max'.
            Only used if mode is 'seurat'. Default is None.

    Returns
    -------
    adata：An AnnData object containing cells that passed QC filters.

    """
    console.level1("Performing QC...")
    if tresh is None :
        tresh={'fragment_counts_min': 2000,
                'fragment_counts_max': 100000,
                'TSS_score_min': 0.1,
                'TSS_score_max': 50,
                'Nucleosome_singal_max': 4,
                }
    
    adata.obs['selected']=True

    console.level2("Filtering cells based on fragment counts...")
    adata.obs.loc[adata.obs.n_fragment<=tresh['fragment_counts_min'],'selected']=False
    adata.obs.loc[adata.obs.n_fragment>=tresh['fragment_counts_max'],'selected']=False 
    if 'tss_score' not in adata.obs.columns:
        if gtf_file is not None:
            from ..utils import read_gtf
            from ._metric import tss_enrichment
            gtf=read_gtf(gtf_file)
            features=gtf.loc[(gtf['feature']=='transcript')&(gtf['source']=='HAVANA')]
            features.loc[:, 'Chromosome'] = features['seqname']
            features.loc[:, 'Start'] = features['start']
            features.loc[:, 'End'] = features['end']
            tss=tss_enrichment(
                adata,
                features,
                n_tss=100000,
            )
            console.level2("Added TSS score to adata.obs['tss_score']")
        else:
            console.level2("TSS score not found in adata.obs['tss_score'], gtf_file is required to compute TSS score")
            raise ValueError("gtf_file is required to compute TSS score")

    adata.obs.loc[adata.obs.tss_score<=tresh['TSS_score_min'],'selected']=False 
    adata.obs.loc[adata.obs.tss_score>=tresh['TSS_score_max'],'selected']=False  


    if 'nucleosome_signal' not in adata.obs.columns:
        from ._metric import nucleosome_signal
        console.level2("Computing nucleosome signal...")
        nucleosome_signal(
            adata,
            n=10e4 * adata.n_obs,
        )
        console.level2("Added nucleosome signal to adata.obs['nucleosome_signal']")
    
    console.level2(f"Filtering cells based on nucleosome signal...")
    #filter cell's number of nucleosome signal
    
    adata.obs.loc[adata.obs.nucleosome_signal>=tresh['Nucleosome_singal_max'],'selected']=False 
    selected_cells=adata.obs.loc[adata.obs.selected==True].index
    console.level2(f"Filtered {adata.obs.loc[adata.obs.selected==False].shape[0]} cells")
    adata._inplace_subset_obs(selected_cells)
    


    return adata

def scrublet(
    adata: internal.AnnData | list[internal.AnnData],
    features: str | np.ndarray | None = "selected",
    n_comps: int = 15,
    sim_doublet_ratio: float = 2.0,
    expected_doublet_rate: float = 0.1,
    n_neighbors: int | None = None,
    use_approx_neighbors=False,
    random_state: int = 0,
    inplace: bool = True,
    n_jobs: int = 8,
    verbose: bool = True,
) -> None:
    """
    Compute probability of being a doublet using the scrublet algorithm.

    This function identifies doublets by generating simulated doublets using
    randomly pairing chromatin accessibility profiles of individual cells.
    The simulated doublets are then embedded alongside the original cells using
    the spectral embedding algorithm in this package.
    A k-nearest-neighbor classifier is trained to distinguish between the simulated
    doublets and the authentic cells.
    This trained classifier produces a "doublet score" for each cell.
    The doublet scores are then converted into probabilities using a Gaussian mixture model.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `adata` can also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    features
        Boolean index mask, where `True` means that the feature is kept, and
        `False` means the feature is removed.
    n_comps
        Number of components. 15 is usually sufficient. The algorithm is not sensitive
        to this parameter.
    sim_doublet_ratio
        Number of doublets to simulate relative to the number of observed cells.
    expected_doublet_rate
        Expected doublet rate.
    n_neighbors
        Number of neighbors used to construct the KNN graph of observed
        cells and simulated doublets. If `None`, this is 
        set to round(0.5 * sqrt(n_cells))
    use_approx_neighbors
        Whether to use approximate search.
    random_state
        Random state.
    inplace
        Whether update the AnnData object inplace
    n_jobs
        Number of jobs to run in parallel.
    verbose
        Whether to print progress messages.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None:
        if ``inplace = True``, it updates adata with the following fields:
            - ``adata.obs["doublet_probability"]``: probability of being a doublet
            - ``adata.obs["doublet_score"]``: doublet score
    """
    if isinstance(adata, list):
        result = snapatac2._utils.anndata_par(
            adata,
            lambda x: scrublet(x, features, n_comps, sim_doublet_ratio,
                               expected_doublet_rate, n_neighbors,
                               use_approx_neighbors, random_state,
                               inplace, n_jobs, verbose=False),
            n_jobs=n_jobs,
        )
        if inplace:
            return None
        else:
            return result

    if isinstance(features, str):
        if features in adata.var:
            features = adata.var[features].to_numpy()
        else:
            raise NameError("Please call `select_features` first or explicitly set `features = None`")

    if features is None:
        count_matrix = adata.X[:]
    else:
        count_matrix = adata.X[:, features]

    if min(count_matrix.shape) == 0: raise NameError("Matrix is empty")

    if n_neighbors is None:
        n_neighbors = int(round(0.5 * np.sqrt(count_matrix.shape[0])))
    console.level1("Performing scrublet...")
    doublet_scores_obs, doublet_scores_sim, _, _ = scrub_doublets_core(
        count_matrix, n_neighbors, sim_doublet_ratio, expected_doublet_rate,
        n_comps=n_comps,
        use_approx_neighbors = use_approx_neighbors,
        random_state=random_state,
        verbose=verbose,
    )
    probs = get_doublet_probability(
        doublet_scores_sim, doublet_scores_obs, random_state,
    )

    console.level1("Scrublet completed")
    console.level2("Added doublet probability to adata.obs['doublet_probability']")
    console.level2("Added doublet score to adata.obs['doublet_score']")
    console.level2("Added scrublet simulation doublet score to adata.uns['scrublet_sim_doublet_score']")

    #ratio of doublets
    if inplace:
        adata.obs["doublet_probability"] = probs
        adata.obs["doublet_score"] = doublet_scores_obs
        adata.uns["scrublet_sim_doublet_score"] = doublet_scores_sim
        adata.obs["is_doublet"] = probs > 0.5
        doublet_rate = adata.obs["is_doublet"].sum() / adata.obs.shape[0]
        console.level2("Ratio of doublets: {:.2f}%".format(doublet_rate*100))
    else:
        return probs, doublet_scores_obs


def scrub_doublets_core(
    count_matrix: ss.spmatrix,
    n_neighbors: int,
    sim_doublet_ratio: float,
    expected_doublet_rate: float,
    synthetic_doublet_umi_subsampling: float =1.0,
    n_comps: int = 30,
    use_approx_neighbors: bool = False,
    random_state: int = 0,
    verbose: bool = False,
) -> None:
    """
    Modified scrublet pipeline for single-cell ATAC-seq data.

    Automatically sets a threshold for calling doublets, but it's best to check 
    this by running plot_histogram() afterwards and adjusting threshold 
    with call_doublets(threshold=new_threshold) if necessary.

    Arguments
    ---------
    synthetic_doublet_umi_subsampling
        Rate for sampling UMIs when creating synthetic doublets. If 1.0, 
        each doublet is created by simply adding the UMIs from two randomly 
        sampled observed transcriptomes. For values less than 1, the 
        UMI counts are added and then randomly sampled at the specified
        rate.
    n_comps
        Number of principal components used to embed the transcriptomes prior
        to k-nearest-neighbor graph construction.

    Returns
    -------
    doublet_scores_obs_, doublet_errors_obs_,
    doublet_scores_sim_, doublet_errors_sim_,
    predicted_doublets_, z_scores_ 
    threshold_, detected_doublet_rate_,
    detectable_doublet_fraction_, overall_doublet_rate_,
    doublet_parents_, doublet_neighbor_parents_ 
    """
    import gc

    total_counts_obs = count_matrix.sum(1).A.squeeze()

    if verbose: console.level2('Simulating doublets...')
    (count_matrix_sim, total_counts_sim, _) = simulate_doublets(
        count_matrix, total_counts_obs, sim_doublet_ratio,
        synthetic_doublet_umi_subsampling, random_state
    )

    if verbose: console.level2('Spectral embedding ...')
    n = count_matrix.shape[0]
    merged_matrix = ss.vstack([count_matrix, count_matrix_sim])
    del count_matrix_sim
    gc.collect()
    _, evecs = spectral(
        AnnData(X=merged_matrix),
        features=None,
        n_comps=n_comps,
        inplace=False,
    )
    manifold = np.asanyarray(evecs)
    manifold_obs = manifold[0:n, ]
    manifold_sim = manifold[n:, ]

    if verbose: console.level2('Calculating doublet scores...')
    doublet_scores_obs, doublet_scores_sim = calculate_doublet_scores(
        manifold_obs, manifold_sim, k = n_neighbors,
        exp_doub_rate = expected_doublet_rate,
        use_approx_neighbors = use_approx_neighbors,
        random_state = random_state,
    )

    return (doublet_scores_obs, doublet_scores_sim, manifold_obs, manifold_sim)


def simulate_doublets(
    count_matrix: ss.spmatrix,
    total_counts: np.ndarray,
    sim_doublet_ratio: int = 2,
    synthetic_doublet_umi_subsampling: float = 1.0,
    random_state: int = 0,
) -> tuple[ss.spmatrix, np.ndarray, np.ndarray]:
    """
    Simulate doublets by adding the counts of random cell pairs.

    Parameters
    ----------
    count_matrix
    total_counts
        Total insertion counts in each cell
    sim_doublet_ratio
        Number of doublets to simulate relative to the number of cells.
    synthetic_doublet_umi_subsampling
        Rate for sampling UMIs when creating synthetic doublets. If 1.0, 
        each doublet is created by simply adding the UMIs from two randomly 
        sampled observed transcriptomes. For values less than 1, the 
        UMI counts are added and then randomly sampled at the specified rate.

    Returns
    -------
    count_matrix_sim

    """
    n_obs = count_matrix.shape[0]
    n_sim = int(n_obs * sim_doublet_ratio)

    np.random.seed(random_state)
    pair_ix = np.random.randint(0, n_obs, size=(n_sim, 2))

    count_matrix_sim = count_matrix[pair_ix[:,0],:] + count_matrix[pair_ix[:,1],:]
    total_counts_sim = total_counts[pair_ix[:,0]] + total_counts[pair_ix[:,1]]

    if synthetic_doublet_umi_subsampling < 1:
        pass
        #count_matrix_sim, total_counts_sim = subsample_counts(
        #    count_matrix_sim, synthetic_doublet_umi_subsampling, total_counts_sim,
        #    random_seed=random_state
        #)
    return (count_matrix_sim, total_counts_sim, pair_ix)

def calculate_doublet_scores(
    manifold_obs: np.ndarray,
    manifold_sim: np.ndarray,
    k: int = 40,
    exp_doub_rate: float = 0.1,
    stdev_doub_rate: float = 0.03,
    use_approx_neighbors=False,
    random_state: int = 0,
) -> None:
    """
    Parameters
    ----------
    manifold_obs
        Manifold of observations
    manifold_sim
        Manifold of simulated doublets
    k
        Number of nearest neighbors
    exp_doub_rate
    stdev_doub_rate
    random_state
    """
    from sklearn.neighbors import NearestNeighbors

    n_obs = manifold_obs.shape[0]
    n_sim = manifold_sim.shape[0]

    manifold = np.vstack((manifold_obs, manifold_sim))
    doub_labels = np.concatenate(
        (np.zeros(n_obs, dtype=int), np.ones(n_sim, dtype=int))
    )

    # Adjust k (number of nearest neighbors) based on the ratio of simulated to observed cells
    k_adj = int(round(k * (1 + n_sim / float(n_obs))))
    
    # Find k_adj nearest neighbors
    if use_approx_neighbors:
        knn = internal.approximate_nearest_neighbour_graph(
            manifold.astype(np.float32), k_adj)
    else:
        knn = internal.nearest_neighbour_graph(manifold, k_adj)
    indices = knn.indices
    indptr = knn.indptr
    neighbors = np.vstack(
        [indices[indptr[i]:indptr[i+1]] for i in range(len(indptr) - 1)]
    )
    
    # Calculate doublet score based on ratio of simulated cell neighbors vs. observed cell neighbors
    doub_neigh_mask = doub_labels[neighbors] == 1
    n_sim_neigh = doub_neigh_mask.sum(1)
    n_obs_neigh = doub_neigh_mask.shape[1] - n_sim_neigh
    
    rho = exp_doub_rate
    r = n_sim / float(n_obs)
    nd = n_sim_neigh.astype(float)
    ns = n_obs_neigh.astype(float)
    N = float(k_adj)
    
    # Bayesian
    q=(nd+1)/(N+2)
    Ld = q*rho/r/(1-rho-q*(1-rho-rho/r))

    se_q = np.sqrt(q*(1-q)/(N+3))
    se_rho = stdev_doub_rate

    se_Ld = q*rho/r / (1-rho-q*(1-rho-rho/r))**2 * np.sqrt((se_q/q*(1-rho))**2 + (se_rho/rho*(1-q))**2)

    doublet_scores_obs = Ld[doub_labels == 0]
    doublet_scores_sim = Ld[doub_labels == 1]
    doublet_errors_obs = se_Ld[doub_labels==0]
    doublet_errors_sim = se_Ld[doub_labels==1]

    return (doublet_scores_obs, doublet_scores_sim)

def get_doublet_probability(
    doublet_scores_sim: np.ndarray,
    doublet_scores: np.ndarray,
    random_state: int = 0,
    verbose: bool = False,
):
    from sklearn.mixture import BayesianGaussianMixture

    X = doublet_scores_sim.reshape((-1, 1))
    gmm = BayesianGaussianMixture(
        n_components=2, n_init=10, max_iter=1000, random_state=random_state
    ).fit(X)

    if verbose:
        console.level2("GMM means: {}".format(gmm.means_))

    i = np.argmax(gmm.means_)
    return gmm.predict_proba(doublet_scores.reshape((-1, 1)))[:,i]