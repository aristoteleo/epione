import h5py
from scipy.sparse import csr_matrix
import pandas as pd
import anndata as ad
import numpy as np
from scipy.io import mmread
import re

def read_ATAC_10x(matrix, cell_names='', var_names='', path_file=''):
    """
    Copy from Episcanpy
    Load sparse matrix (including matrices corresponding to 10x data) as AnnData objects.
    read the mtx file, tsv file coresponding to cell_names and the bed file containing the variable names

    Parameters
    ----------
    matrix: sparse count matrix

    cell_names: optional, tsv file containing cell names

    var_names: optional, bed file containing the feature names

    Return
    ------
    AnnData object

    """

    
    mat = mmread(''.join([path_file, matrix])).tocsr().transpose()
    
    with open(path_file+cell_names) as f:
        barcodes = f.readlines() 
        barcodes = [x[:-1] for x in barcodes]
        
    with open(path_file+var_names) as f:
        var_names = f.readlines()
        var_names = ["_".join(x[:-1].split('\t')) for x in var_names]
        
    adata = ad.AnnData(mat, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=var_names))
    adata.uns['omic'] = 'ATAC'
    
    return(adata)

def read_gtf(gtf_path):
    """
    Read a GTF file into a pandas DataFrame and split the attribute column.

    This removes the dependency on omicverse by implementing the minimal
    functionality needed: loading columns and expanding attributes.

    Parameters
    ----------
    gtf_path : str or path-like
        Path to the GTF file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with standard GTF columns plus extracted attributes.
    """
    # Load raw GTF columns (ignore comment lines starting with '#')
    columns = [
        "seqname", "source", "feature", "start", "end",
        "score", "strand", "frame", "attribute"
    ]
    df = pd.read_csv(gtf_path, sep="\t", header=None, comment="#", dtype=str)
    # Assign standard column names up to the number of loaded columns
    df.columns = columns[: df.shape[1]]

    # Enforce dtypes: start/end as int, others as str (already str via dtype)
    if "start" in df.columns:
        df["start"] = df["start"].astype(int)
    if "end" in df.columns:
        df["end"] = df["end"].astype(int)

    # Split attribute key-value pairs like: key "value";
    if "attribute" in df.columns:
        pattern = re.compile(r'([^\s]+) "([^"]+)";')
        splitted = pd.DataFrame.from_records(
            np.vectorize(lambda x: {k: v for k, v in pattern.findall(x or "")})(df["attribute"]),
            index=df.index,
        )
        # Merge attributes into main DataFrame (attributes may add gene_id, gene_name, etc.)
        df = df.assign(**splitted)

    return pd.DataFrame(df)
