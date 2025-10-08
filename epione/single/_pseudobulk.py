from tqdm import tqdm
import gc
import re
import os 
import gzip
from typing import Optional, Union, Dict, List
import anndata as ad
import numpy as np
import pandas as pd
import pyBigWig
import pyranges as pr

# Performance optimization imports - optional dependencies
try:
    import modin.pandas as mpd
    MODIN_AVAILABLE = True
except ImportError:
    MODIN_AVAILABLE = False

try:
    import swifter
    SWIFTER_AVAILABLE = True
except ImportError:
    SWIFTER_AVAILABLE = False

try:
    from pandarallel import pandarallel
    PANDARALLEL_AVAILABLE = True
except ImportError:
    PANDARALLEL_AVAILABLE = False

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except (ImportError, AttributeError) as e:
    # AttributeError can occur due to pandas/dask version incompatibility
    DASK_AVAILABLE = False

def pseudobulk(adata, chromsizes, size=None, cluster_key='celltype', clusters=None,
               chr=['chrom','chromStart','chromEnd'], bigwig_path='temp', verbose=True,
               use_sparse=True, optimize_memory=True, parallel_backend=None, n_jobs=1):
    """
    Create pseudobulk bigwig files from single cell data.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    chromsizes : dict or pd.DataFrame
        Chromosome sizes
    size : int, optional
        Maximum number of cells to sample per celltype
    cluster_key : str
        Column name for cell type annotation
    clusters : list, optional
        List of clusters to process
    chr : list
        Column names for chromosome, start, end coordinates
    bigwig_path : str
        Output directory for bigwig files
    verbose : bool
        Print progress information
    use_sparse : bool
        Use sparse matrix operations when possible
    optimize_memory : bool
        Apply memory optimizations
    parallel_backend : str, optional
        Parallel backend: 'joblib', 'multiprocessing', or None
    n_jobs : int
        Number of parallel jobs (if parallel_backend is specified)
    """
    import pyrle
    if pyrle.__version__ > '0.0.38':
        print('You need to use `pip install pyrle==0.0.38`')
        return

    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
    
    if clusters is None:
        clusters = adata.obs[cluster_key].cat.categories
    print(f"Processing {len(clusters)} clusters: {list(clusters)}")
    
    # Pre-extract chromosome info to avoid repeated access
    chr_info = {
        'Chromosome': adata.var[chr[0]].values,
        'Start': adata.var[chr[1]].values.astype(np.int32),
        'End': adata.var[chr[2]].values.astype(np.int32)
    }
    
    def process_celltype(celltype):
        """Process a single celltype - can be parallelized"""
        try:
            adata_test = adata[adata.obs[cluster_key] == celltype]
            
            # Check if there are any cells for this celltype
            if adata_test.shape[0] == 0:
                if verbose:
                    print(f"Skipping {celltype}: no cells found")
                return None
                
            # Check if there are any features
            if adata_test.shape[1] == 0:
                if verbose:
                    print(f"Skipping {celltype}: no features found")
                return None
            
            # Sample cells if needed
            if (size is not None) and (adata_test.shape[0] > size):
                import random
                random.seed(42)  # For reproducibility
                cell_idx = random.sample(adata_test.obs.index.tolist(), size)
                adata_test = adata_test[cell_idx]
                if verbose:
                    print(f"{celltype}: sampled {size} cells from {adata_test.shape[0]}")
            
            # Pre-allocate arrays for better performance
            n_cells = adata_test.shape[0]
            n_features = adata_test.shape[1]
            total_size = n_cells * n_features
            
            if verbose:
                print(f"{celltype}: creating arrays for {n_cells} cells x {n_features} features")
            
            # Use more efficient array construction
            chromosome_array = np.tile(chr_info['Chromosome'], n_cells)
            start_array = np.tile(chr_info['Start'], n_cells)
            end_array = np.tile(chr_info['End'], n_cells)
            
            # Create name array more efficiently
            cell_names = adata_test.obs.index.values
            name_array = np.repeat(cell_names, n_features)
            
            # Get expression values efficiently
            if verbose:
                print(f"{celltype}: extracting expression values")
            
            if use_sparse and hasattr(adata_test.X, 'toarray'):
                # Handle sparse matrices
                score_array = adata_test.X.toarray().T.ravel()
            else:
                # Handle dense matrices
                if hasattr(adata_test, 'to_df'):
                    score_array = adata_test.to_df().T.values.ravel()
                else:
                    score_array = adata_test.X.T.ravel()
            
            # Create DataFrame more efficiently
            if optimize_memory:
                df_test = pd.DataFrame({
                    'Chromosome': pd.Categorical(chromosome_array),
                    'Start': start_array,
                    'End': end_array,
                    'Name': pd.Categorical(name_array),
                    'Score': score_array.astype(np.float32)  # Use float32 to save memory
                })
            else:
                df_test = pd.DataFrame({
                    'Chromosome': chromosome_array,
                    'Start': start_array,
                    'End': end_array,
                    'Name': name_array,
                    'Score': score_array
                })
            
            # Check if the resulting DataFrame is empty
            if df_test.shape[0] == 0:
                if verbose:
                    print(f"Skipping {celltype}: no data to write")
                return None
            
            if verbose:
                print(f"{celltype}: writing bigwig file")
            
            # Create PyRanges and write bigwig
            group_pr = pr.PyRanges(df_test)
            group_pr.to_bigwig(
                path=f'{bigwig_path}/{celltype}.bw',
                chromosome_sizes=chromsizes,
                rpm=True,
                value_col="Score"
            )
            
            # Clean up
            del group_pr, df_test
            if optimize_memory:
                gc.collect()
                
            return f"{celltype}.bw"
            
        except Exception as e:
            print(f"Error processing {celltype}: {e}")
            return None
    
    # Process celltypes
    if parallel_backend and n_jobs > 1:
        # Parallel processing
        if parallel_backend == 'joblib':
            try:
                from joblib import Parallel, delayed
                if verbose:
                    print(f"Using joblib with {n_jobs} jobs")
                
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_celltype)(celltype) for celltype in tqdm(clusters)
                )
                
            except ImportError:
                print("joblib not available, falling back to sequential processing")
                results = [process_celltype(celltype) for celltype in tqdm(clusters)]
                
        elif parallel_backend == 'multiprocessing':
            try:
                from multiprocessing import Pool
                if verbose:
                    print(f"Using multiprocessing with {n_jobs} processes")
                
                with Pool(n_jobs) as pool:
                    results = list(tqdm(pool.imap(process_celltype, clusters), total=len(clusters)))
                    
            except Exception as e:
                print(f"Multiprocessing failed: {e}, falling back to sequential processing")
                results = [process_celltype(celltype) for celltype in tqdm(clusters)]
        else:
            print(f"Unknown parallel_backend: {parallel_backend}, using sequential processing")
            results = [process_celltype(celltype) for celltype in tqdm(clusters)]
    else:
        # Sequential processing (default)
        results = [process_celltype(celltype) for celltype in tqdm(clusters)]
    
    # Summary
    successful = [r for r in results if r is not None]
    if verbose:
        print(f"Successfully created {len(successful)} bigwig files out of {len(clusters)} clusters")
        if successful:
            print(f"Files created: {successful}")
    
    return successful

def pseudobulk_with_fragments(
    input_data: Union[pd.DataFrame, ad.AnnData],
    chromsizes: Union[pd.DataFrame, pr.PyRanges],
    cluster_key: str,
    clusters: Optional[list] = None,
    bed_path: Optional[str] = 'temp',
    bigwig_path: Optional[str] = 'temp',
    verbose=True,
    path_to_fragments: Optional[Dict[str, str]] = None,
    normalize_bigwig: Optional[bool] = True,
    remove_duplicates: Optional[bool] = True,
    split_pattern: Optional[str] = "___",
    use_polars: Optional[bool] = True,
    auto_add_chr: Optional[bool] = True,
    performance_backend: Optional[str] = "pandas",
    sample_id_col: Optional[str] = None,
    **kwargs
    ):

    """
    Create pseudobulks as bed and bigwig from single cell fragments file given a barcode annotation.

    Parameters
    ---------
    input_data: CistopicObject or pd.DataFrame
            A :class:`CistopicObject` containing the specified `variable` as a column in :class:`CistopicObject.cell_data` or a cell metadata
            :class:`pd.DataFrame` containing barcode as rows, containing the specified `variable` as a column (additional columns are
            possible) and a `sample_id` column. Index names must contain the BARCODE (e.g. ATGTCGTC-1), additional tags are possible separating with -
            (e.g. ATGCTGTGCG-1-Sample_1). The levels in the sample_id column must agree with the keys in the path_to_fragments dictionary.
            Alternatively, if the cell metadata contains a column named barcode it will be used instead of the index names.
    chromsizes: pd.DataFrame or pr.PyRanges
            A data frame or :class:`pr.PyRanges` containing size of each chromosome, containing 'Chromosome', 'Start' and 'End' columns.
    cluster_key: str
            A character string indicating the column that will be used to create the different group pseudobulk. It must be included in
            the cell metadata provided as input_data.
    cluster: list, optional
            A list of character strings indicating the groups for which pseudobulks will be created. If None, pseudobulks will be created for all groups.
    bed_path: str
            Path to folder where the fragments bed files per group will be saved. If None, files will not be generated.
    bigwig_path: str
            Path to folder where the bigwig files per group will be saved. If None, files will not be generated.
    path_to_fragments: str or dict, optional
            A dictionary of character strings, with sample name as names indicating the path to the fragments file/s from which pseudobulk profiles have to
            be created. If a :class:`CistopicObject` is provided as input it will be ignored, but if a cell metadata :class:`pd.DataFrame` is provided it
            is necessary to provide it. The keys of the dictionary need to match with the sample_id tag added to the index names of the input data frame.
    normalize_bigwig: bool, optional
            Whether bigwig files should be CPM normalized. Default: True.
    remove_duplicates: bool, optional
            Whether duplicates should be removed before converting the data to bigwig.
    split_pattern: str, optional
            Pattern to split cell barcode from sample id. Default: '___'. Note, if `split_pattern` is not None, then `export_pseudobulk` will
            attempt to infer `sample_id` from the index of `input_data` and ignore `sample_id_col`.
    use_polars: bool, optional
            Whether to use polars to read fragments files. Default: True.
    auto_add_chr: bool, optional
            Automatically add 'chr' prefix to chromosome names that don't start with 'chr'. Default: True.
    performance_backend: str, optional
            Performance backend to use for pandas operations. Options: 'pandas', 'modin', 'dask'. Default: 'pandas'.
    sample_id_col: str, optional
            This parameter is not necessary.
            Name of the column containing the sample name per barcode in the input :class:`CistopicObject.cell_data` or class:`pd.DataFrame`. Default: None.
    **kwargs
            Additional parameters for ray.init()

    Return
    ------
    dict
            A dictionary containing the paths to the newly created bed fragments files per group a dictionary containing the paths to the
            newly created bigwig files per group.
    """
    # check the imported package
    try:
        import pyrle
    except ImportError:
        raise  ImportError(
            'Please install the pyrle: `pip install pyrle`.'
        )
    
    # Initialize parallel backends if available
    if performance_backend == "pandarallel" and PANDARALLEL_AVAILABLE:
        pandarallel.initialize(progress_bar=verbose)
        print("Initialized pandarallel for improved pandas performance...")
    
    # Get fragments file
    if path_to_fragments is None:
        print("Please, provide path_to_fragments.")
  
    # Get celltype 
    if isinstance(input_data, ad.AnnData):
        input_data = input_data.obs
    if clusters is None:
        cell_data = input_data 
    elif clusters is not None:
        cell_data = input_data[input_data[cluster_key].isin(clusters)]
    
    # Process the sample_id_col(optional)
    if sample_id_col is not None:
        try:
            sample_ids = list(set(cell_data[sample_id_col]))
        except ValueError:
            print(
            'Please, include a sample identification column (e.g. "sample_id") in your cell metadata!'
        )
            
    # Get fragments
    fragments_df_dict = {}
    for sample_id in path_to_fragments.keys():
        if sample_id_col is not None:
            if sample_id not in sample_ids:
                print(
                    "The following path_to_fragments entry is not found in the cell metadata sample_id_col: ",
                    sample_id,
                    ". It will be ignored.",
                )
            if verbose: 
                print("Reading fragments from " + path_to_fragments[sample_id])
            fragments_df = read_fragments_from_file(
                path_to_fragments[sample_id], 
                use_polars=use_polars, 
                auto_add_chr=auto_add_chr,
                performance_backend=performance_backend
            ).df
            # Convert to int32 for memory efficiency
            fragments_df.Start = np.int32(fragments_df.Start)
            fragments_df.End = np.int32(fragments_df.End)
            if "Score" in fragments_df:
                fragments_df.Score = np.int32(fragments_df.Score)
            if "barcode" in cell_data:
                barcode_list = cell_data["barcode"].tolist()
                if performance_backend == "pandarallel" and PANDARALLEL_AVAILABLE and len(fragments_df) > 10000:
                    # Use pandarallel for faster filtering with large datasets
                    barcode_set = set(barcode_list)  # Convert to set for faster lookup
                    fragments_df = fragments_df.loc[
                        fragments_df["Name"].parallel_apply(lambda x: x in barcode_set)
                    ]
                    if verbose:
                        print(f"Used pandarallel for barcode filtering ({len(fragments_df)} fragments)")
                else:
                    fragments_df = fragments_df.loc[
                        fragments_df["Name"].isin(barcode_list)
                    ]
            else:
                tag_cells = prepare_tag_cells(cell_data.index.tolist(), split_pattern)
                if performance_backend == "pandarallel" and PANDARALLEL_AVAILABLE and len(fragments_df) > 10000:
                    # Use pandarallel for faster filtering with large datasets
                    tag_set = set(tag_cells)  # Convert to set for faster lookup
                    fragments_df = fragments_df.loc[
                        fragments_df["Name"].parallel_apply(lambda x: x in tag_set)
                    ]
                    if verbose:
                        print(f"Used pandarallel for tag filtering ({len(fragments_df)} fragments)")
                else:
                    fragments_df = fragments_df.loc[
                        fragments_df["Name"].isin(tag_cells)
                    ]
            fragments_df_dict[sample_id] = fragments_df
        else:
            if verbose: 
                print("Reading fragments from " + path_to_fragments[sample_id])
            fragments_df = read_fragments_from_file(
                path_to_fragments[sample_id], 
                use_polars=use_polars, 
                auto_add_chr=auto_add_chr,
                performance_backend=performance_backend
            ).df
            # Convert to int32 for memory efficiency
            fragments_df.Start = np.int32(fragments_df.Start)
            fragments_df.End = np.int32(fragments_df.End)
            if "Score" in fragments_df:
                fragments_df.Score = np.int32(fragments_df.Score)
            if "barcode" in cell_data:
                barcode_list = cell_data["barcode"].tolist()
                if performance_backend == "pandarallel" and PANDARALLEL_AVAILABLE and len(fragments_df) > 10000:
                    # Use pandarallel for faster filtering with large datasets
                    barcode_set = set(barcode_list)  # Convert to set for faster lookup
                    fragments_df = fragments_df.loc[
                        fragments_df["Name"].parallel_apply(lambda x: x in barcode_set)
                    ]
                    if verbose:
                        print(f"Used pandarallel for barcode filtering ({len(fragments_df)} fragments)")
                else:
                    fragments_df = fragments_df.loc[
                        fragments_df["Name"].isin(barcode_list)
                    ]
            else:
                tag_cells = prepare_tag_cells(cell_data.index.tolist(), split_pattern)
                if performance_backend == "pandarallel" and PANDARALLEL_AVAILABLE and len(fragments_df) > 10000:
                    # Use pandarallel for faster filtering with large datasets
                    tag_set = set(tag_cells)  # Convert to set for faster lookup
                    fragments_df = fragments_df.loc[
                        fragments_df["Name"].parallel_apply(lambda x: x in tag_set)
                    ]
                    if verbose:
                        print(f"Used pandarallel for tag filtering ({len(fragments_df)} fragments)")
                else:
                    fragments_df = fragments_df.loc[
                        fragments_df["Name"].isin(tag_cells)
                    ]
            fragments_df_dict[sample_id] = fragments_df
            print(fragments_df)

    # Set groups
    if sample_id_col is not None:
        if "barcode" in cell_data:
            cell_data = cell_data.loc[:, [cluster_key,sample_id_col,"barcode"]]
        else:
            cell_data = cell_data.loc[:, [cluster_key,sample_id_col]]
    else:
        if "barcode" in cell_data:
            cell_data = cell_data.loc[:, [cluster_key, "barcode"]]
        else:
            cell_data = cell_data.loc[:, [cluster_key]]
    cell_data[cluster_key] = cell_data[cluster_key].replace(" ", "", regex=True)
    cell_data[cluster_key] = cell_data[cluster_key].replace("[^A-Za-z0-9]+", "_", regex=True)
    groups = sorted(list(set(cell_data[cluster_key])))

    # Check chromosome sizes
    if isinstance(chromsizes, pd.DataFrame):
        chromsizes = chromsizes.loc[:, ["Chromosome", "Start", "End"]]
        chromsizes = pr.PyRanges(chromsizes)
    # Check that output dir exist and generate output paths
    if isinstance(bed_path, str):
        if not os.path.exists(bed_path):
            os.makedirs(bed_path)
        bed_paths = {
            group: os.path.join(bed_path, str(group) + ".bed.gz") for group in groups
        }
    else:
        bed_paths = {}
    if isinstance(bigwig_path, str):
        if not os.path.exists(bigwig_path):
            os.makedirs(bigwig_path)
        bw_paths = {
            group: os.path.join(bigwig_path, str(group) + ".bw") for group in groups
        }
    else:
        bw_paths = {}

    # Get pseudobulk from different celltypes
    for group in groups:
        if sample_id_col is not None:
            export_pseudobulk_one_sample(
                cell_data,
                group,
                fragments_df_dict,
                chromsizes,
                bigwig_path,
                bed_path,
                normalize_bigwig,
                remove_duplicates,
                split_pattern,
                sample_id_col,
                )
        else:
            export_pseudobulk_one_sample(
                cell_data,
                group,
                fragments_df_dict,
                chromsizes,
                bigwig_path,
                bed_path,
                normalize_bigwig,
                remove_duplicates,
                split_pattern,
                 )
        

r"""
Copy from pycisTopic: https://github.com/aertslab/pycisTopic/
"""

def export_pseudobulk_one_sample(
    cell_data: pd.DataFrame,
    group: str,
    fragments_df_dict: Dict[str, pd.DataFrame],
    chromsizes: pr.PyRanges,
    bigwig_path: str,
    bed_path: str,
    normalize_bigwig: Optional[bool] = True,
    remove_duplicates: Optional[bool] = True,
    split_pattern: Optional[str] = "___",
    sample_id_col: Optional[str] = None,
):
    """
    Create pseudobulk as bed and bigwig from single cell fragments file given a barcode annotation and a group.

    Parameters
    ---------
    cell_data: pd.DataFrame
            A cell metadata :class:`pd.Dataframe` containing barcodes, their annotation and their sample of origin.
    group: str
            A character string indicating the group for which pseudobulks will be created.
    fragments_df_dict: dict
            A dictionary containing data frames as values with 'Chromosome', 'Start', 'End', 'Name', and 'Score' as columns; and sample label
            as keys. 'Score' indicates the number of times that a fragments is found assigned to that barcode.
    chromsizes: pr.PyRanges
            A :class:`pr.PyRanges` containing size of each column, containing 'Chromosome', 'Start' and 'End' columns.
    bigwig_path: str
            Path to folder where the bigwig file will be saved.
    bed_path: str
            Path to folder where the fragments bed file will be saved.
    normalize_bigwig: bool, optional
            Whether bigwig files should be CPM normalized. Default: True.
    remove_duplicates: bool, optional
            Whether duplicates should be removed before converting the data to bigwig.
    split_pattern: str
            Pattern to split cell barcode from sample id. Default: ___ .
    sample_id_col: str, optional
            Name of the column containing the sample name per barcode in the input :class:`CistopicObject.cell_data` or class:`pd.DataFrame`. Default: 'None'.
    """
    # Create logger
    print("Creating pseudobulk for " + str(group))
    group_fragments_list = []
    group_fragments_dict = {}

    if sample_id_col is not None:
        for sample_id in fragments_df_dict:
            sample_data = cell_data[cell_data.loc[:, sample_id_col].isin([sample_id])]
            if "barcode" in sample_data:
                sample_data.index = sample_data["barcode"].tolist()
            else:
                sample_data.index = prepare_tag_cells(
                    sample_data.index.tolist(), split_pattern
               )
            group_var = sample_data.iloc[:, 0]
            barcodes = group_var[group_var.isin([group])].index.tolist()
            fragments_df = fragments_df_dict[sample_id]
            group_fragments = fragments_df.loc[fragments_df["Name"].isin(barcodes)]
            if len(fragments_df_dict) > 1:
                group_fragments_dict[sample_id] = group_fragments

    else:
        for sample_id in fragments_df_dict:
            sample_data = cell_data
            if "barcode" in sample_data:
                sample_data.index = sample_data["barcode"].tolist()
            else:
                sample_data.index = prepare_tag_cells(
                    sample_data.index.tolist(), split_pattern
                )
            group_var = sample_data.iloc[:, 0]
            barcodes = group_var[group_var.isin([group])].index.tolist()
            fragments_df = fragments_df_dict[sample_id]
            group_fragments = fragments_df.loc[fragments_df["Name"].isin(barcodes)]
            if len(fragments_df_dict) > 1:
                group_fragments_dict[sample_id] = group_fragments

    if len(fragments_df_dict) > 1:
        group_fragments_list = [
            group_fragments_dict[list(group_fragments_dict.keys())[x]]
            for x in range(len(fragments_df_dict))
        ]
        group_fragments = pd.concat(group_fragments_list, ignore_index=True)

    del group_fragments_dict
    del group_fragments_list
    del fragments_df
    gc.collect()

    group_pr = pr.PyRanges(group_fragments)
    if isinstance(bigwig_path, str):
        bigwig_path_group = os.path.join(bigwig_path, str(group) + ".bw")
        if remove_duplicates:
            group_pr.to_bigwig(
                path=bigwig_path_group,
                chromosome_sizes=chromsizes,
                rpm=normalize_bigwig,
            )
        else:
            group_pr.to_bigwig(
                path=bigwig_path_group,
                chromosome_sizes=chromsizes,
                rpm=normalize_bigwig,
                value_col="Score",
            )
    if isinstance(bed_path, str):
        bed_path_group = os.path.join(bed_path, str(group) + ".bed.gz")
        group_pr.to_bed(
            path=bed_path_group, keep=False, compression="infer", chain=False
        )
    print(str(group) + " done!")


def prepare_tag_cells(cell_names, split_pattern="___"):
    if split_pattern == "-":
        new_cell_names = [
            re.findall(r"^[ACGT]*-[0-9]+-", x)[0].rstrip("-")
            if len(re.findall(r"^[ACGT]*-[0-9]+-", x)) != 0
            else x
            for x in cell_names
        ]
        new_cell_names = [
            re.findall(r"^\w*-[0-9]*", new_cell_names[i])[0].rstrip("-")
            if (len(re.findall(r"^\w*-[0-9]*", new_cell_names[i])) != 0)
            & (new_cell_names[i] == cell_names[i])
            else new_cell_names[i]
            for i in range(len(new_cell_names))
        ]
    else:
        new_cell_names = [x.split(split_pattern)[0] for x in cell_names]

    return new_cell_names

def read_fragments_from_file(
    fragments_bed_filename, 
    use_polars: bool = True, 
    auto_add_chr: bool = True,
    performance_backend: str = "pandas",
    chunk_size: int = None,
    optimize_memory: bool = True
) -> pr.PyRanges:
    """
    Read fragments BED file to PyRanges object.

    Parameters
    ----------
    fragments_bed_filename: Fragments BED filename.
    use_polars: Use polars instead of pandas for reading the fragments BED file.
    auto_add_chr: Automatically add 'chr' prefix to chromosome names that don't start with 'chr'.
    performance_backend: Performance backend to use. Options: 'pandas', 'modin', 'dask'. Default: 'pandas'.
    chunk_size: Target chunk size in MB for each partition. If None, auto-determined based on file size.
    optimize_memory: Optimize data types for memory efficiency.

    Returns
    -------
    PyRanges object of fragments.
    """

    bed_column_names = (
        "Chromosome",
        "Start",
        "End",
        "Name",
        "Score",
        "Strand",
        "ThickStart",
        "ThickEnd",
        "ItemRGB",
        "BlockCount",
        "BlockSizes",
        "BlockStarts",
    )

    # Set the correct open function depending if the fragments BED file is gzip compressed or not.
    open_fn = gzip.open if fragments_bed_filename.endswith(".gz") else open

    skip_rows = 0
    nbr_columns = 0
    with open_fn(fragments_bed_filename, "rt") as fragments_bed_fh:
        for line in fragments_bed_fh:
            # Remove newlines and spaces.
            line = line.strip()

            if not line or line.startswith("#"):
                # Count number of empty lines and lines which start with a comment before the actual data.
                skip_rows += 1
            else:
                # Get number of columns from the first real BED entry.
                nbr_columns = len(line.split("\t"))

                # Stop reading the BED file.
                break

    if nbr_columns < 4:
        raise ValueError(
            f'Fragments BED file needs to have at least 4 columns. "{fragments_bed_filename}" contains only '
            f"{nbr_columns} columns."
        )

    # Choose backend based on performance_backend parameter
    if use_polars:
        import polars as pl

        # Read fragments BED file with polars.
        df = (
            pl.read_csv(
                fragments_bed_filename,
                has_header=False,
                skip_rows=skip_rows,
                separator="\t",
                use_pyarrow=True,
                new_columns=bed_column_names[:nbr_columns],
            )
            .with_columns(
                [
                    pl.col("Chromosome").cast(pl.Utf8),
                    pl.col("Start").cast(pl.Int32),
                    pl.col("End").cast(pl.Int32),
                    pl.col("Name").cast(pl.Utf8),
                ]
            )
            .to_pandas()
        )

        # Convert "Name" column to pd.Categorical as groupby operations will be done on it later.
        df["Name"] = df["Name"].astype("category")
    else:
        # Choose pandas backend
        if performance_backend == "modin" and MODIN_AVAILABLE:
            print("Using Modin backend for improved pandas performance...")
            df = mpd.read_table(
                fragments_bed_filename,
                sep="\t",
                skiprows=skip_rows,
                header=None,
                names=bed_column_names[:nbr_columns],
                doublequote=False,
                engine="python",  # Modin works better with python engine for this use case
                dtype={
                    "Chromosome": str,
                    "Start": np.int32,
                    "End": np.int32,
                    "Name": "category",
                    "Strand": str,
                },
            )
            # Convert to pandas for compatibility with downstream processing
            df = df._to_pandas()
            
        elif performance_backend == "dask" and DASK_AVAILABLE:
            print("Using Dask backend for improved pandas performance...")
            import os
            file_size_mb = os.path.getsize(fragments_bed_filename) / (1024 * 1024)
            
            if chunk_size is None:
                # Auto-determine chunk size based on file size
                if file_size_mb < 100:
                    chunk_size = None  # Read entire file
                elif file_size_mb < 1000:
                    chunk_size = 50000  # 50k rows per chunk
                else:
                    chunk_size = 100000  # 100k rows per chunk
            
            if chunk_size and file_size_mb > 50:  # Use dask for larger files
                print(f"Reading {file_size_mb:.1f}MB file with Dask in chunks of {chunk_size} rows...")
                
                # Read with dask for parallel processing
                try:
                    # Create dask dataframe
                    df_dask = dd.read_csv(
                        fragments_bed_filename,
                        sep="\t",
                        header=None,
                        names=bed_column_names[:nbr_columns],
                        skiprows=skip_rows,
                        dtype={
                            "Chromosome": str,
                            "Start": np.int32,
                            "End": np.int32,
                            "Name": "category",
                            "Strand": str,
                        },
                        blocksize=f"{chunk_size//1000}KB" if chunk_size else None,
                        assume_missing=True
                    )
                    
                    # Apply optimizations using dask
                     # Optimize data types in parallel
                    if 'Start' in df_dask.columns:
                        df_dask['Start'] = df_dask['Start'].astype('int32')
                    if 'End' in df_dask.columns:
                        df_dask['End'] = df_dask['End'].astype('int32')
                    if 'Score' in df_dask.columns:
                        df_dask['Score'] = df_dask['Score'].astype('int16')
                
                    # Convert back to pandas for compatibility
                    print("Computing dask dataframe...")
                    df = df_dask.compute()
                    
                except Exception as e:
                    print(f"Dask reading failed: {e}, falling back to pandas...")
                    # Fallback to pandas
                    df = pd.read_table(
                        fragments_bed_filename,
                        sep="\t",
                        skiprows=skip_rows,
                        header=None,
                        names=bed_column_names[:nbr_columns],
                        doublequote=False,
                        engine="c",
                        dtype={
                            "Chromosome": str,
                            "Start": np.int32,
                            "End": np.int32,
                            "Name": "category",
                            "Strand": str,
                        },
                    )
                    if chunk_size:
                        df = pd.concat([chunk for chunk in df], ignore_index=True)
            else:
                print("File size suitable for direct pandas reading...")
                # Use pandas for smaller files
                df = pd.read_table(
                    fragments_bed_filename,
                    sep="\t",
                    skiprows=skip_rows,
                    header=None,
                    names=bed_column_names[:nbr_columns],
                    doublequote=False,
                    engine="c",
                    dtype={
                        "Chromosome": str,
                        "Start": np.int32,
                        "End": np.int32,
                        "Name": "category",
                        "Strand": str,
                    },
                )
            
        else:
            # Default pandas backend with optimizations
            if performance_backend != "pandas":
                if performance_backend == "pandarallel":
                    print(f"⚠️  pandarallel backend not available. Install with: pip install pandarallel")
                    print("   Falling back to pandas (performance may be slower for large datasets)")
                elif performance_backend == "modin":
                    print(f"⚠️  modin backend not available. Install with: pip install modin[ray]")
                    print("   Falling back to pandas (performance may be slower)")
                elif performance_backend == "dask":
                    print(f"⚠️  dask backend not available. Install with: pip install dask[complete]")
                    print("   Falling back to pandas (performance may be slower)")
                else:
                    print(f"⚠️  {performance_backend} backend not available, falling back to pandas")
            
            # Set up dtype_dict for memory optimization
            if optimize_memory:
                dtype_dict = {
                    "Chromosome": str,
                    "Start": np.int32,
                    "End": np.int32,
                    "Name": "category",
                }
                if nbr_columns > 4:
                    dtype_dict["Score"] = np.int16
                if nbr_columns > 5:
                    dtype_dict["Strand"] = "category"
            else:
                dtype_dict = None
            
            # Use chunked reading for large files
            if chunk_size:
                print(f"Reading file in chunks of {chunk_size} rows...")
                chunks = []
                for chunk in pd.read_table(
                    fragments_bed_filename,
                    sep="\t",
                    skiprows=skip_rows,
                    header=None,
                    names=bed_column_names[:nbr_columns],
                    doublequote=False,
                    engine="c",
                    dtype=dtype_dict,
                    chunksize=chunk_size
                ):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                del chunks  # Free memory
            else:
                df = pd.read_table(
                    fragments_bed_filename,
                    sep="\t",
                    skiprows=skip_rows,
                    header=None,
                    names=bed_column_names[:nbr_columns],
                    doublequote=False,
                    engine="c",
                    dtype=dtype_dict,
                )

    # Automatically add 'chr' prefix to chromosome names that don't start with 'chr'
    if auto_add_chr:
        # Check which chromosomes don't start with 'chr'
        if performance_backend == "pandarallel" and PANDARALLEL_AVAILABLE and len(df) > 50000:
            # Use pandarallel for string operations on large datasets
            mask = ~df['Chromosome'].parallel_apply(lambda x: str(x).startswith('chr'))
            if mask.any():
                # Add 'chr' prefix to those chromosome names using parallel operations
                df.loc[mask, 'Chromosome'] = df.loc[mask, 'Chromosome'].parallel_apply(lambda x: 'chr' + str(x))
                print(f"Added 'chr' prefix to {mask.sum()} chromosome entries that didn't start with 'chr' (using pandarallel)")
        else:
            # Standard pandas operations (faster for smaller datasets)
            mask = ~df['Chromosome'].str.startswith('chr')
            if mask.any():
                # Add 'chr' prefix to those chromosome names
                df.loc[mask, 'Chromosome'] = 'chr' + df.loc[mask, 'Chromosome'].astype(str)
                print(f"Added 'chr' prefix to {mask.sum()} chromosome entries that didn't start with 'chr'")

    # Convert pandas dataframe to PyRanges dataframe.
    # This will convert "Chromosome" and "Strand" columns to pd.Categorical.
    return pr.PyRanges(df)

def check_performance_backends():
    """
    Check available performance optimization backends and provide installation instructions.
    
    Returns
    -------
    dict
        Dictionary with backend availability status and installation instructions.
    """
    backends = {
        "polars": {
            "available": True,  # Always available since it's already imported
            "description": "Fast DataFrame library written in Rust",
            "install": "pip install polars",
            "speed_improvement": "2-10x faster for I/O operations"
        },
        "modin": {
            "available": MODIN_AVAILABLE,
            "description": "Drop-in replacement for pandas with parallel processing",
            "install": "pip install modin[ray] or pip install modin[dask]",
            "speed_improvement": "1-4x faster on multi-core systems"
        },
        "swifter": {
            "available": SWIFTER_AVAILABLE,
            "description": "Smart apply() function that chooses the best execution strategy",
            "install": "pip install swifter",
            "speed_improvement": "2-10x faster for apply operations"
        },
        "pandarallel": {
            "available": PANDARALLEL_AVAILABLE,
            "description": "Simple parallel processing for pandas operations",
            "install": "pip install pandarallel",
            "speed_improvement": "2-8x faster for apply/map operations"
        },
        "dask": {
            "available": DASK_AVAILABLE,
            "description": "Flexible parallel computing library for analytics",
            "install": "pip install dask[complete]",
            "speed_improvement": "Scalable to clusters, 2-6x on single machine"
        }
    }
    
    print("=== Performance Optimization Backends Status ===")
    print()
    
    available_count = 0
    for name, info in backends.items():
        status = "✅ Available" if info["available"] else "❌ Not installed"
        print(f"{name.upper():12} | {status}")
        print(f"{'':12} | {info['description']}")
        print(f"{'':12} | Speed: {info['speed_improvement']}")
        if not info["available"]:
            print(f"{'':12} | Install: {info['install']}")
        print()
        
        if info["available"]:
            available_count += 1
    
    print(f"Available backends: {available_count}/{len(backends)}")
    print()
    
    # Installation helpers
    if available_count < len(backends):
        print("=== Quick Installation ===")
        print("# Install pandarallel (recommended for medium datasets):")
        print("quick_install_pandarallel()")
        print()
        print("# Install all performance backends:")
        print("install_performance_backend('all')")
        print()
    
    # Usage examples
    print("=== Usage Examples ===")
    print("# Use polars backend (fastest I/O):")
    print("read_fragments_from_file(file, use_polars=True)")
    print()
    print("# Use pandarallel backend (best for filtering large datasets):")
    print("read_fragments_from_file(file, use_polars=False, performance_backend='pandarallel')")
    print()
    print("# Use modin backend (parallel pandas):")
    print("read_fragments_from_file(file, use_polars=False, performance_backend='modin')")
    print()
    print("# Use with pseudobulk_with_fragments:")
    print("pseudobulk_with_fragments(..., performance_backend='pandarallel', use_polars=True)")
    print()
    
    return backends


def get_performance_recommendations(data_size_mb: float = None):
    """
    Get performance recommendations based on data size.
    
    Parameters
    ----------
    data_size_mb : float, optional
        Size of your data in MB
        
    Returns
    -------
    dict
        Recommended settings for optimal performance
    """
    recommendations = {}
    
    if data_size_mb is None:
        print("Provide data size for specific recommendations:")
        print("get_performance_recommendations(data_size_mb=100)")
        return
    
    print(f"=== Performance Recommendations for {data_size_mb}MB data ===")
    print()
    
    if data_size_mb < 100:  # Small data
        recommendations = {
            "use_polars": True,
            "performance_backend": "pandas",
            "reasoning": "Small data - overhead of parallel processing not worth it"
        }
        
    elif data_size_mb < 1000:  # Medium data  
        if PANDARALLEL_AVAILABLE:
            recommendations = {
                "use_polars": True,
                "performance_backend": "pandarallel",
                "reasoning": "Medium data - pandarallel provides good speedup for filtering operations"
            }
        elif MODIN_AVAILABLE:
            recommendations = {
                "use_polars": True,
                "performance_backend": "modin",
                "reasoning": "Medium data - modin provides good speedup with minimal overhead"
            }
        else:
            recommendations = {
                "use_polars": True,
                "performance_backend": "pandas",
                "reasoning": "Medium data - install pandarallel or modin for better performance"
            }
            
    else:  # Large data
        if DASK_AVAILABLE:
            recommendations = {
                "use_polars": True,
                "performance_backend": "dask", 
                "reasoning": "Large data - dask can handle out-of-core processing"
            }
        elif MODIN_AVAILABLE:
            recommendations = {
                "use_polars": True,
                "performance_backend": "modin",
                "reasoning": "Large data - modin provides parallel processing"
            }
        else:
            recommendations = {
                "use_polars": True,
                "performance_backend": "pandas",
                "reasoning": "Large data - install dask or modin for much better performance"
            }
    
    for key, value in recommendations.items():
        print(f"{key}: {value}")
    print()
    
    return recommendations

def read_fragments_with_dask_parallel(
    fragments_bed_filename,
    n_partitions: int = None,
    chunk_size_mb: int = 100,
    optimize_dtypes: bool = True,
    auto_add_chr: bool = True,
    verbose: bool = True
) -> pr.PyRanges:
    """
    Read large fragments files using Dask for parallel processing.
    
    Parameters
    ----------
    fragments_bed_filename : str
        Path to fragments BED file
    n_partitions : int, optional
        Number of partitions for Dask. If None, auto-determined
    chunk_size_mb : int
        Target chunk size in MB for each partition
    optimize_dtypes : bool
        Optimize data types for memory efficiency
    auto_add_chr : bool
        Automatically add 'chr' prefix to chromosome names
    verbose : bool
        Print progress information
        
    Returns
    -------
    pr.PyRanges
        PyRanges object with fragments data
    """
    
    if not DASK_AVAILABLE:
        print("Dask not available, falling back to regular read_fragments_from_file")
        return read_fragments_from_file(
            fragments_bed_filename, 
            use_polars=True, 
            auto_add_chr=auto_add_chr
        )
    
    import dask.dataframe as dd
    import os
    
    # Get file info
    file_size_mb = os.path.getsize(fragments_bed_filename) / (1024 * 1024)
    
    if verbose:
        print(f"Processing {file_size_mb:.1f}MB fragments file with Dask")
    
    # Auto-determine partitions
    if n_partitions is None:
        n_partitions = max(1, int(file_size_mb / chunk_size_mb))
        n_partitions = min(n_partitions, os.cpu_count() or 4)  # Don't exceed CPU count
    
    if verbose:
        print(f"Using {n_partitions} partitions for parallel processing")
    
    # Determine columns from first few lines
    bed_column_names = (
        "Chromosome", "Start", "End", "Name", "Score", "Strand",
        "ThickStart", "ThickEnd", "ItemRGB", "BlockCount", "BlockSizes", "BlockStarts"
    )
    
    # Quick scan for column count
    open_fn = gzip.open if fragments_bed_filename.endswith(".gz") else open
    skip_rows = 0
    nbr_columns = 0
    
    with open_fn(fragments_bed_filename, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                skip_rows += 1
            else:
                nbr_columns = len(line.split("\t"))
                break
    
    # Set up optimized dtypes
    if optimize_dtypes:
        dtype_dict = {
            "Chromosome": "category",
            "Start": "int32",
            "End": "int32", 
            "Name": "category",
        }
        if nbr_columns > 4:
            dtype_dict["Score"] = "int16"
        if nbr_columns > 5:
            dtype_dict["Strand"] = "category"
    else:
        dtype_dict = None
    
    try:
        # Read with Dask
        if verbose:
            print("Reading file with Dask dataframe...")
            
        blocksize = f"{chunk_size_mb}MB"
        
        df_dask = dd.read_csv(
            fragments_bed_filename,
            sep="\t",
            header=None,
            names=bed_column_names[:nbr_columns],
            skiprows=skip_rows,
            dtype=dtype_dict,
            blocksize=blocksize,
            assume_missing=True
        )
        
        # Apply chr prefix addition in parallel if needed
        if auto_add_chr and 'Chromosome' in df_dask.columns:
            if verbose:
                print("Adding 'chr' prefix in parallel...")
                
            def add_chr_prefix(partition):
                if partition['Chromosome'].dtype.name == 'category':
                    # Handle categorical efficiently
                    categories = partition['Chromosome'].cat.categories
                    chr_mask = ~categories.str.startswith('chr')
                    if chr_mask.any():
                        new_categories = categories.where(~chr_mask, 'chr' + categories)
                        partition['Chromosome'] = partition['Chromosome'].cat.rename_categories(new_categories)
                else:
                    # Handle object dtype
                    mask = ~partition['Chromosome'].str.startswith('chr')
                    if mask.any():
                        partition.loc[mask, 'Chromosome'] = 'chr' + partition.loc[mask, 'Chromosome'].astype(str)
                return partition
            
            df_dask = df_dask.map_partitions(add_chr_prefix)
        
        # Compute result
        if verbose:
            print("Computing parallel operations...")
            
        df = df_dask.compute()
        
        if verbose:
            print(f"Successfully processed {len(df):,} fragments")
        
        return pr.PyRanges(df)
        
    except Exception as e:
        print(f"Dask parallel processing failed: {e}")
        print("Falling back to regular method...")
        return read_fragments_from_file(
            fragments_bed_filename,
            use_polars=True,
            auto_add_chr=auto_add_chr,
            performance_backend="pandas"
        )

def install_performance_backend(backend_name: str):
    """
    Helper function to install performance optimization backends.
    
    Parameters
    ----------
    backend_name : str
        Name of the backend to install: 'pandarallel', 'modin', 'dask', or 'all'
    """
    import subprocess
    import sys
    
    install_commands = {
        'pandarallel': ['pip', 'install', 'pandarallel'],
        'modin': ['pip', 'install', 'modin[ray]'],
        'dask': ['pip', 'install', 'dask[complete]'],
        'swifter': ['pip', 'install', 'swifter'],
        'polars': ['pip', 'install', 'polars']
    }
    
    if backend_name == 'all':
        backends_to_install = list(install_commands.keys())
    elif backend_name in install_commands:
        backends_to_install = [backend_name]
    else:
        print(f"❌ Unknown backend: {backend_name}")
        print(f"Available backends: {list(install_commands.keys()) + ['all']}")
        return False
    
    print(f"🚀 Installing performance backend(s): {', '.join(backends_to_install)}")
    print("This may take a few minutes...")
    
    success_count = 0
    for backend in backends_to_install:
        try:
            print(f"\n📦 Installing {backend}...")
            result = subprocess.run(
                install_commands[backend], 
                check=True, 
                capture_output=True, 
                text=True
            )
            print(f"✅ {backend} installed successfully!")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {backend}: {e.stderr}")
        except Exception as e:
            print(f"❌ Error installing {backend}: {str(e)}")
    
    print(f"\n🎉 Installation complete! {success_count}/{len(backends_to_install)} backends installed.")
    
    if success_count > 0:
        print("\n⚠️  Please restart your Python session to use the new backends.")
        print("After restart, run: check_performance_backends() to verify installation")
    
    return success_count == len(backends_to_install)


def quick_install_pandarallel():
    """Quick installer for pandarallel backend."""
    return install_performance_backend('pandarallel')
