from ..utils._genome import Genome
import snapatac2._snapatac2 as internal
import numpy as np
import snapatac2
from pathlib import Path
from ..utils import console
import os


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

def tsse(
    adata: internal.AnnData | list[internal.AnnData],
    gene_anno: Genome | Path,
    *,
    exclude_chroms: list[str] | str | None = ["chrM", "M"],
    inplace: bool = True,
    n_jobs: int = 8,
) -> np.ndarray | list[np.ndarray] | None:
    """ Compute the TSS enrichment score (TSSe) for each cell.

    :func:`~snapatac2.pp.import_fragments` must be ran first in order to use this function.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `adata` could also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    gene_anno
        A :class:`~snapatac2.Genome` object or a GTF/GFF file containing the gene annotation.
    exclude_chroms
        A list of chromosomes to exclude.
    inplace
        Whether to add the results to `adata.obs` or return it as a dictionary.
    n_jobs
        Number of jobs to run in parallel when `adata` is a list.
        If `n_jobs=-1`, all CPUs will be used.

    Returns
    -------
    tuple[np.ndarray, tuple[float, float]] | list[tuple[np.ndarray, tuple[float, float]]] | None
        If `inplace = True`, cell-level TSSe scores are computed and stored in `adata.obs['tsse']`.
        Library-level TSSe scores are stored in `adata.uns['library_tsse']`.
        Fraction of fragments overlapping TSS are stored in `adata.uns['frac_overlap_TSS']`.
        If `inplace = False`, return a tuple containing all these values.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_fragments(snap.datasets.pbmc500(downsample=True), chrom_sizes=snap.genome.hg38, sorted_by_barcode=False)
    >>> snap.metrics.tsse(data, snap.genome.hg38)
    >>> print(data.obs['tsse'].head())
    AAACTGCAGACTCGGA-1    32.129514
    AAAGATGCACCTATTT-1    22.052786
    AAAGATGCAGATACAA-1    27.109808
    AAAGGGCTCGCTCTAC-1    24.990329
    AAATGAGAGTCCCGCA-1    33.264463
    Name: tsse, dtype: float64
    """
    gene_anno = gene_anno.annotation if isinstance(gene_anno, Genome) else gene_anno
 
    if isinstance(adata, list):
        result = snapatac2._utils.anndata_par(
            adata,
            lambda x: tsse(x, gene_anno, exclude_chroms=exclude_chroms, inplace=inplace),
            n_jobs=n_jobs,
        )
    else:
        console.level1("Computing TSS enrichment score for adata...")
        result = internal.tss_enrichment(adata, gene_anno, exclude_chroms)
        result['tsse'] = np.array(result['tsse'])
        result['TSS_profile'] = np.array(result['TSS_profile'])
        if inplace:
            adata.obs["tsse"] = result['tsse']
            adata.uns['library_tsse'] = result['library_tsse']
            adata.uns['frac_overlap_TSS'] = result['frac_overlap_TSS']
            adata.uns['TSS_profile'] = result['TSS_profile']
            console.level2("Added TSS enrichment score to adata.obs['tsse']")
            console.level2("Added library TSS enrichment score to adata.uns['library_tsse']")
            console.level2("Added fraction of fragments overlapping TSS to adata.uns['frac_overlap_TSS']")
            console.level2("Added TSS profile to adata.uns['TSS_profile']")
        else:
            console.level2("Returned TSS enrichment score")
            return result
    if inplace:
        return None
    else:
        return result


def ensure_tabix_index(path, preset="bed"):
    """
    Ensure that the given .tsv.gz file has a corresponding .tbi index file.
    If not, create one by calling pysam.tabix_index.
    
    Parameters
    ----------
    path : str
        The path of the bgzip compressed file (.tsv.gz)
    preset : str
        The preset of the index, default "bed". Can be "vcf", "gff" etc.
    """
    import pysam
    index_path = path + ".tbi"
    if os.path.exists(index_path):
        print(f"Index already exists: {index_path}")
    else:
        print(f"Index not found, creating: {index_path}")
        # pysam will automatically check if the file is bgzip compressed
        pysam.tabix_index(path, preset=preset, force=True)
        print("Index created!")



