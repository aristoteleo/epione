from ..utils._genome import Genome
import snapatac2._snapatac2 as internal
import numpy as np
import snapatac2
from pathlib import Path
from ..utils import console
import os
from typing import Union, Optional, Literal
import pandas as pd
from anndata import AnnData
from ..utils.genome import Genome


from tqdm import tqdm



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


def tss_enrichment(
    adata: AnnData,
    features: Optional[pd.DataFrame] = None,
    extend_upstream: int = 2000,
    extend_downstream: int = 2000,
    n_tss: int = 10000,
    return_tss: bool = True,
    random_state=None,
    barcodes: Optional[str] = None,
) -> np.ndarray | list[np.ndarray] | None:
    """
    Calculate TSS enrichment according to ENCODE guidelines. Adds a column `tss_score` to the `.obs` DataFrame and
    optionally returns a tss score object.

    Parameters
    ----------
    data
        AnnData object with peak counts or multimodal MuData object with 'atac' modality.
    features
        A DataFrame with feature annotation, e.g. genes.
        Annotation has to contain columns: Chromosome, Start, End.
    extend_upsteam
        Number of nucleotides to extend every gene upstream (2000 by default to extend gene coordinates to promoter regions)
    extend_downstream
        Number of nucleotides to extend every gene downstream (0 by default)
    n_tss
        How many randomly chosen TSS sites to pile up. The fewer the faster. Default: 2000.
    return_tss
        Whether to return the TSS pileup matrix. Needed for enrichment plots.
    random_state : int, array-like, BitGenerator, np.random.RandomState, optional
        Argument passed to pandas.DataFrame.sample() for sampling features.
    barcodes
        Column name in the .obs of the AnnData
        with barcodes corresponding to the ones in the fragments file.

    Returns
    ----------
    AnnData
        AnnData object with a 'tss_score' column in the .obs slot.

    """
    import pysam
    from tqdm import tqdm
    console.level1("Computing TSS enrichment score for adata...")
    if features is None:
        # Try to gene gene annotation in the data.mod['rna']
        raise ValueError(
                "Argument `features` is required. It should be a BED-like DataFrame with gene coordinates and names."
            )

    if features.shape[0] > n_tss:
        # Only use n_tss randomly chosen sites to make function faster
        features = features.sample(n=n_tss, random_state=random_state)

    n = adata.n_obs
    n_features = extend_downstream + extend_upstream + 1

    # Dictionary with matrix positions
    if barcodes and barcodes in adata.obs.columns:
        d = {k: v for k, v in zip(adata.obs.loc[:, barcodes], range(n))}
    else:
        d = {k: v for k, v in zip(adata.obs.index, range(n))}

    # Not sparse since we expect most positions to be filled
    mx = np.zeros((n, n_features), dtype=int)

    if "files" not in adata.uns:
        raise ValueError("'files' not found in adata.uns")
    if "fragments" not in adata.uns["files"]:
        raise ValueError("'fragments' not found in adata.uns['files']")

    #dertermine the fragments tbi file
    fragments_tbi = adata.uns["files"]["fragments"] + ".tbi"
    if not os.path.exists(fragments_tbi):
        console.level2("Index not found, creating: {}".format(fragments_tbi))
        ensure_tabix_index(adata.uns["files"]["fragments"])

    fragments = pysam.TabixFile(adata.uns["files"]["fragments"], parser=pysam.asBed())

    # Subset the features to the chromosomes present in the fragments file
    chromosomes = fragments.contigs
    features = features[features.Chromosome.isin(chromosomes)]

    # logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Counting fragments in {n} cells for {features.shape[0]} features...")

    for i in tqdm(
        range(features.shape[0]), desc="Fetching Regions..."
    ):  # iterate over features (e.g. genes)
        f = features.iloc[i]
        tss_start = f.Start - extend_upstream  # First position of the TSS region
        for fr in fragments.fetch(
            f.Chromosome, f.Start - extend_upstream, f.Start + extend_downstream
        ):
            try:
                rowind = d[fr.name]  # cell barcode (e.g. GTCAGTCAGTCAGTCA-1)
                score = int(fr.score)  # number of cuts per fragment (e.g. 2)
                colind_start = max(fr.start - tss_start, 0)
                colind_end = min(fr.end - tss_start, n_features)  # ends are non-inclusive in bed
                mx[rowind, colind_start:colind_end] += score
            except:
                pass

    fragments.close()

    anno = pd.DataFrame(
        {"TSS_position": range(-extend_upstream, extend_downstream + 1)},
    )
    anno.index = anno.index.astype(str)
    tss_pileup=AnnData(X=mx, obs=adata.obs, var=anno, dtype=int)

    flank_means, center_means = _calculate_tss_score(data=tss_pileup)

    tss_pileup.X = tss_pileup.X / flank_means[:, None]

    tss_scores = center_means / flank_means

    adata.obs["tss_score"] = tss_scores
    console.level2("Added TSS enrichment score to adata.obs['tss_score']")
    tss_pileup.obs["tss_score"] = tss_scores
    console.level1("Created TSS enrichment score")
    console.level2("Added TSS enrichment score to tss_pileup.obs['tss_score']")
    if return_tss:
        console.level2("Returned TSS enrichment score")
        return tss_pileup
    else:
        return None



def _calculate_tss_score(data, flank_size: int = 100, center_size: int = 1001):
    """
    Calculate TSS enrichment scores (defined by ENCODE) for each cell.

    Parameters
    ----------
    data
        AnnData object with TSS positons as generated by `tss_pileup`.
    flank_size
        Number of nucleotides in the flank on either side of the region (ENCODE standard: 100bp).
    center_size
        Number of nucleotides in the center on either side of the region (ENCODE standard: 1001bp).
    """
    region_size = data.X.shape[1]

    if center_size > region_size:
        raise ValueError(
            f"`center_size` ({center_size}) must smaller than the piled up region ({region_size})."
        )

    if center_size % 2 == 0:
        raise ValueError(f"`center_size` must be an uneven number, but is {center_size}.")

    # Calculate flank means
    flanks = np.hstack((data.X[:, :flank_size], data.X[:, -flank_size:]))
    flank_means = flanks.mean(axis=1)

    # Replace 0 means with population average (to not have 0 division after)
    flank_means[flank_means == 0] = flank_means.mean()

    # Calculate center means
    center_dist = (region_size - center_size) // 2  # distance from the edge of data region
    centers = data.X[:, center_dist:-center_dist]
    center_means = centers.mean(axis=1)

    return flank_means, center_means



    


def nucleosome_signal(
    adata: AnnData,
    n: Union[int, float] = None,
    nucleosome_free_upper_bound: int = 147,
    mononuleosomal_upper_bound: int = 294,
    barcodes: Optional[str] = None,
):
    """
    Computes the ratio of nucleosomal cut fragments to nucleosome-free fragments per cell.
    Nucleosome-free fragments are shorter than 147 bp while mono-mucleosomal fragments are between
    147 bp and 294 bp long.

    Parameters
    ----------
    data
        AnnData object with peak counts or multimodal MuData object with 'atac' modality.
    n
        Number of fragments to count. If `None`, 1e4 fragments * number of cells.
    nucleosome_free_upper_bound
        Number of bases up to which a fragment counts as nucleosome free. Default: 147
    mononuleosomal_upper_bound
        Number of bases up to which a fragment counts as mononuleosomal. Default: 294
    barcodes
        Column name in the .obs of the AnnData
        with barcodes corresponding to the ones in the fragments file.
    """
    console.level1("Computing nucleosome signal for adata...")

    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise KeyError(
            "There is no fragments file located yet. Run muon.atac.tl.locate_fragments first."
        )

    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is not available. It is required to work with the fragments file. Install pysam from PyPI (`pip install pysam`) or from GitHub (`pip install git+https://github.com/pysam-developers/pysam`)"
        )

    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise KeyError(
            "There is no fragments file located yet. "
        )

    fragments_tbi = adata.uns["files"]["fragments"] + ".tbi"
    if not os.path.exists(fragments_tbi):
        console.level2("Index not found, creating: {}".format(fragments_tbi))
        ensure_tabix_index(adata.uns["files"]["fragments"])

    fragments = pysam.TabixFile(adata.uns["files"]["fragments"], parser=pysam.asBed())

    # Dictionary with matrix row indices
    if barcodes and barcodes in adata.obs.columns:
        d = {k: v for k, v in zip(adata.obs.loc[:, barcodes], range(adata.n_obs))}
    else:
        d = {k: v for k, v in zip(adata.obs.index, range(adata.n_obs))}
    mat = np.zeros(shape=(adata.n_obs, 2), dtype=int)

    fr = fragments.fetch()

    if n is None:
        n = int(adata.n_obs * 1e4)
    else:
        n = int(n)  # Cast n to int

    for i in tqdm(range(n), desc="Reading Fragments"):
        try:
            f = next(fr)
            length = f.end - f.start
            row_ind = d[f.name]
            if length < nucleosome_free_upper_bound:
                mat[row_ind, 0] += 1
            elif length < mononuleosomal_upper_bound:
                mat[row_ind, 1] += 1
        except StopIteration:
            break
        except KeyError:
            pass
        # if i % 1000000 == 0:
        #     print(f"Read {i/1000000} Mio. fragments.", end='\r')

    # Prevent division by 0
    mat[mat[:, 0] == 0, :] += 1

    # Calculate nucleosome signal
    nucleosome_enrichment = mat[:, 1] / mat[:, 0]
    # nucleosome_enrichment[mat[:,0] == 0] = 0

    adata.obs["nucleosome_signal"] = nucleosome_enrichment

    # Message for the user
    console.level2('Added a "nucleosome_signal" column to the .obs slot of the AnnData object')
    console.level1("Created nucleosome signal")
    console.level2("Added nucleosome signal to adata.obs['nucleosome_signal']")
    return None


def _find_most_accessible_features(
    feature_count,
    filter_lower_quantile,
    filter_upper_quantile,
    total_features,
) -> np.ndarray:
    idx = np.argsort(feature_count)
    for i in range(idx.size):
        if feature_count[idx[i]] > 0:
            break
    idx = idx[i:]
    n = idx.size
    n_lower = int(filter_lower_quantile * n)
    n_upper = int(filter_upper_quantile * n)
    idx = idx[n_lower:n-n_upper]
    return idx[::-1][:total_features]
 
def select_features(
    adata: internal.AnnData | internal.AnnDataSet | list[internal.AnnData],
    n_features: int = 500000,
    filter_lower_quantile: float = 0.005,
    filter_upper_quantile: float = 0.005,
    whitelist: Path | None = None,
    blacklist: Path | None = None,
    max_iter: int = 1,
    inplace: bool = True,
    n_jobs: int = 8,
    verbose: bool = True,
) -> np.ndarray | list[np.ndarray] | None:
    """
    Perform feature selection by selecting the most accessibile features across
    all cells unless `max_iter` > 1.

    Note
    ----
    This function does not perform the actual subsetting. The feature mask is used by
    various functions to generate submatrices on the fly.
    Features that are zero in all cells will be always removed regardless of the
    filtering criteria.
    For more discussion about feature selection, see: https://github.com/scverse/SnapATAC2/discussions/116.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `adata` can also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    n_features
        Number of features to keep. Note that the final number of features
        may be smaller than this number if there is not enough features that pass
        the filtering criteria.
    filter_lower_quantile
        Lower quantile of the feature count distribution to filter out.
        For example, 0.005 means the bottom 0.5% features with the lowest counts will be removed.
    filter_upper_quantile
        Upper quantile of the feature count distribution to filter out.
        For example, 0.005 means the top 0.5% features with the highest counts will be removed.
        Be aware that when the number of feature is very large, the default value of 0.005 may
        risk removing too many features.
    whitelist
        A user provided bed file containing genome-wide whitelist regions.
        None-zero features listed here will be kept regardless of the other
        filtering criteria.
        If a feature is present in both whitelist and blacklist, it will be kept.
    blacklist 
        A user provided bed file containing genome-wide blacklist regions.
        Features that are overlapped with these regions will be removed.
    max_iter
        If greater than 1, this function will perform iterative clustering and feature selection
        based on variable features found using previous clustering results.
        This is similar to the procedure implemented in ArchR, but we do not recommend it,
        see https://github.com/scverse/SnapATAC2/issues/111.
        Default value is 1, which means no iterative clustering is performed.
    inplace
        Perform computation inplace or return result.
    n_jobs
        Number of parallel jobs to use when `adata` is a list.
    verbose
        Whether to print progress messages.
    
    Returns
    -------
    np.ndarray | None:
        If `inplace = False`, return a boolean index mask that does filtering,
        where `True` means that the feature is kept, `False` means the feature is removed.
        Otherwise, store this index mask directly to `.var['selected']`.
    """
    if isinstance(adata, list):
        result = snapatac2._utils.anndata_par(
            adata,
            lambda x: select_features(x, n_features, filter_lower_quantile,
                                      filter_upper_quantile, whitelist,
                                      blacklist, max_iter, inplace, verbose=False),
            n_jobs=n_jobs,
        )
        if inplace:
            return None
        else:
            return result

    count = np.zeros(adata.shape[1])
    for batch, _, _ in adata.chunked_X(2000):
        count += np.ravel(batch.sum(axis = 0))
    if inplace:
        adata.var['count'] = count

    selected_features = _find_most_accessible_features(
        count, filter_lower_quantile, filter_upper_quantile, n_features)

    if blacklist is not None:
        blacklist = np.array(internal.intersect_bed(adata.var_names, str(blacklist)))
        selected_features = selected_features[np.logical_not(blacklist[selected_features])]

    # Iteratively select features
    iter = 1
    while iter < max_iter:
        embedding = snapatac2.tl.spectral(adata, features=selected_features, inplace=False)[1]
        clusters = snapatac2.tl.leiden(snapatac2.pp.knn(embedding, inplace=False))
        rpm = snapatac2.tl.aggregate_X(adata, groupby=clusters).X
        var = np.var(np.log(rpm + 1), axis=0)
        selected_features = np.argsort(var)[::-1][:n_features]

        # Apply blacklist to the result
        if blacklist is not None:
            selected_features = selected_features[np.logical_not(blacklist[selected_features])]
        iter += 1

    result = np.zeros(adata.shape[1], dtype=bool)
    result[selected_features] = True

    # Finally, apply whitelist to the result
    if whitelist is not None:
        whitelist = np.array(internal.intersect_bed(adata.var_names, str(whitelist)))
        whitelist &= count != 0
        result |= whitelist
    
    if verbose:
        console.level1(f"Selected {result.sum()} features.")

    if inplace:
        adata.var["selected"] = result
    else:
        return result

def make_peak_matrix(
    adata: internal.AnnData | internal.AnnDataSet,
    *,
    use_rep: str | list[str] | None = None,
    inplace: bool = False,
    file: Path | None = None,
    backend: Literal['hdf5'] = 'hdf5',
    peak_file: Path | None = None,
    chunk_size: int = 500,
    use_x: bool = False,
    min_frag_size: int | None = None,
    max_frag_size: int | None = None,
    counting_strategy: Literal['fragment', 'insertion', 'paired-insertion'] = 'paired-insertion',
    value_type: Literal['target', 'total', 'fraction'] = 'target',
    summary_type: Literal['sum', 'mean'] = 'sum',
) -> internal.AnnData:
    """Generate cell by peak count matrix.

    This function will generate a cell by peak count matrix and store it in a 
    new .h5ad file.

    :func:`~snapatac2.pp.import_fragments` must be ran first in order to use this function.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    use_rep
        This is used to read peak information from `.uns[use_rep]`.
        The peaks can also be provided by a list of strings:
        ["chr1:1-100", "chr2:2-200"].
    inplace
        Whether to add the tile matrix to the AnnData object or return a new AnnData object.
    file
        File name of the output h5ad file used to store the result. If provided,
        result will be saved to a backed AnnData, otherwise an in-memory AnnData
        is used. This has no effect when `inplace=True`.
    backend
        The backend to use for storing the result. If `None`, the default backend will be used.
    peak_file
        Bed file containing the peaks. If provided, peak information will be read
        from this file.
    chunk_size
        Chunk size
    use_x
        If True, use the matrix stored in `.X` as raw counts.
        Otherwise the `.obsm['insertion']` is used.
    min_frag_size
        Minimum fragment size to include.
    max_frag_size
        Maximum fragment size to include.
    counting_strategy
        The strategy to compute feature counts. It must be one of the following:
        "fragment", "insertion", or "paired-insertion". "fragment" means the
        feature counts are assigned based on the number of fragments that overlap
        with a region of interest. "insertion" means the feature counts are assigned
        based on the number of insertions that overlap with a region of interest.
        "paired-insertion" is similar to "insertion", but it only counts the insertions
        once if the pair of insertions of a fragment are both within the same region
        of interest [Miao24]_.
        Note that this parameter has no effect if input are single-end reads.
    value_type
        The type of value to use from `.obsm['_values']`, only available when 
        data is imported using :func:`~snapatac2.pp.import_values`. It must be one of the following:
        "target", "total", or "fraction". "target" means the value is the number
        of recrods that are with postive measurements, e.g., number of methylated bases.
        "total" means the value is the total number of measurements, e.g., methylated bases plus
        unmethylated bases. "fraction" means the value is the fraction of the
        records that are positive, e.g., the fraction of methylated bases.
    summary_type
        The type of summary to use when multiple values are found in a bin. This parameter
        is only used when `.obsm['_values']` exists, which is created by :func:`~snapatac2.pp.import_values`. 
        It must be one of the following: "sum" or "mean".

    Returns
    -------
    AnnData | ad.AnnData | None
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to peaks. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.

    See Also
    --------
    add_tile_matrix
    make_gene_matrix

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_fragments(snap.datasets.pbmc500(downsample=True), chrom_sizes=snap.genome.hg38, sorted_by_barcode=False)
    >>> peak_mat = snap.pp.make_peak_matrix(data, peak_file=snap.datasets.cre_HEA())
    >>> print(peak_mat)
    AnnData object with n_obs × n_vars = 585 × 1154611
        obs: 'n_fragment', 'frac_dup', 'frac_mito'
    """
    import gzip

    if peak_file is not None and use_rep is not None:
        raise RuntimeError("'peak_file' and 'use_rep' cannot be both set") 

    if use_rep is None and peak_file is None:
        use_rep = "peaks"

    if isinstance(use_rep, str):
        df = adata.uns[use_rep]
        peaks = df[df.columns[0]]
    else:
        peaks = use_rep

    if peak_file is not None:
        if Path(peak_file).suffix == ".gz":
            with gzip.open(peak_file, 'rt') as f:
                peaks = [line.strip() for line in f]
        else:
            with open(peak_file, 'r') as f:
                peaks = [line.strip() for line in f]

    if inplace:
        out = None
    elif file is None:
        if adata.isbacked:
            out = AnnData(obs=adata.obs[:].to_pandas())
        else:
            out = AnnData(obs=adata.obs[:])
    else:
        out = internal.AnnData(filename=file, backend=backend, obs=adata.obs[:])
    internal.mk_peak_matrix(adata, peaks, chunk_size, use_x, counting_strategy, value_type, summary_type, min_frag_size, max_frag_size, out)
    return out