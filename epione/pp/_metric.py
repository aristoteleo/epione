from ..utils._genome import Genome
import snapatac2._snapatac2 as internal
import numpy as np
import snapatac2
from ..utils import console

def frag_size_distr(
    adata: internal.AnnData | list[internal.AnnData],
    *,
    max_recorded_size: int = 1000,
    add_key: str = "frag_size_distr",
    inplace: bool = True,
    n_jobs: int = 8,
) -> np.ndarray | list[np.ndarray] | None:
    """ Compute the fragment size distribution of the dataset. 

    This function computes the fragment size distribution of the dataset.
    Note that it does not operate at the single-cell level.
    The result is stored in a vector where each element represents the number of fragments
    and the index represents the fragment length. The first posision of the vector is
    reserved for fragments with size larger than the `max_recorded_size` parameter.
    :func:`~snapatac2.pp.import_fragments` must be ran first in order to use this function.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `adata` could also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    max_recorded_size
        The maximum fragment size to record in the result.
        Fragments with length larger than `max_recorded_size` will be recorded in the first
        position of the result vector.
    add_key
        Key used to store the result in `adata.uns`.
    inplace
        Whether to add the results to `adata.uns` or return it.
    n_jobs
        Number of jobs to run in parallel when `adata` is a list.
        If `n_jobs=-1`, all CPUs will be used.

    Returns
    -------
    np.ndarray | list[np.ndarray] | None
        If `inplace = True`, directly adds the results to `adata.uns['`add_key`']`.
        Otherwise return the results.
    """
    if isinstance(adata, list):
        
        return snapatac2._utils.anndata_par(
            adata,
            lambda x: frag_size_distr(x, add_key=add_key, max_recorded_size=max_recorded_size, inplace=inplace),
            n_jobs=n_jobs,
        )
    else:
        console.level1("Computing fragment size distribution for adata...")
        result = np.array(internal.fragment_size_distribution(adata, max_recorded_size))
        if inplace:
            adata.uns[add_key] = result
            console.level2("Added fragment size distribution to adata.uns['{}']".format(add_key))
        else:
            console.level2("Returned fragment size distribution")
            return result
