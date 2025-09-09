import h5py
from scipy.sparse import csr_matrix
import pandas as pd
import anndata as ad
import numpy as np
from scipy.io import mmread

import re
import gzip

from typing import Union
from tqdm import tqdm

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

def read_gtf(
    gtf_path,
    required_attrs=("gene_id", "gene_name", "transcript_id"),
    feature_whitelist=None,
    chr_prefix=None,
    keep_attribute=True,
):
    """
    Fast GTF reader with inline attribute parsing (no pandas CSV parser).

    Notes:
    - Streams the file line-by-line (supports .gz) to reduce overhead.
    - Parses only a small set of attributes by default for speed.
    - Keeps the original "attribute" string column for compatibility.

    Parameters
    ----------
    gtf_path : str or path-like
        Path to the GTF file (supports .gz).
    required_attrs : tuple of str
        Attribute keys to extract (e.g., "gene_id", "gene_name").

    Returns
    -------
    pandas.DataFrame
        DataFrame with standard GTF columns and selected attributes.
    """

    req_set = set(required_attrs or ())
    cols = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame"]
    if keep_attribute:
        cols.append("attribute")

    data = {c: [] for c in cols}
    for key in req_set:
        data[key] = []

    def parse_attr_fast(attr_text: str, keys_set: set) -> dict:
        # Expect tokens like: key "value"; key2 "value2";
        out = {}
        if not attr_text:
            return out
        for field in attr_text.split(';'):
            field = field.strip()
            if not field:
                continue
            # split only on the first space to preserve values
            sp = field.split(' ', 1)
            if len(sp) != 2:
                continue
            k, v = sp[0], sp[1]
            if k in keys_set:
                v = v.strip().strip('"')
                out[k] = v
        return out

    feature_set = set(feature_whitelist) if feature_whitelist else None
    prefix = str(chr_prefix) if chr_prefix else None
    opener = gzip.open if str(gtf_path).endswith('.gz') else open
    with opener(gtf_path, 'rt') as f:
        for line in f:
            if not line or line[0] == '#':
                continue
            parts = line.rstrip('\n').split('\t', 8)
            if len(parts) < 8:
                continue
            # Unpack with graceful fallback for optional attribute column
            seqname = parts[0]
            source = parts[1]
            feature = parts[2]
            if prefix and not seqname.startswith(prefix):
                continue
            if feature_set and feature not in feature_set:
                continue
            try:
                start = int(parts[3])
            except Exception:
                # Sometimes GTF can contain malformed entries; skip them
                continue
            try:
                end = int(parts[4])
            except Exception:
                continue
            score = parts[5] if len(parts) > 5 else '.'
            strand = parts[6] if len(parts) > 6 else '.'
            frame = parts[7] if len(parts) > 7 else '.'
            attribute = parts[8] if (keep_attribute and len(parts) > 8) else ''

            data["seqname"].append(seqname)
            data["source"].append(source)
            data["feature"].append(feature)
            data["start"].append(start)
            data["end"].append(end)
            data["score"].append(score)
            data["strand"].append(strand)
            data["frame"].append(frame)
            if keep_attribute:
                data["attribute"].append(attribute)

            if req_set:
                d = parse_attr_fast(attribute, req_set)
                for k in req_set:
                    data[k].append(d.get(k, None))

    df = pd.DataFrame(data)
    return df


def fetch_regions_to_df(
    fragment_path: str,
    features: Union[pd.DataFrame, str],
    extend_upstream: int = 0,
    extend_downstream: int = 0,
    relative_coordinates=False,
) -> pd.DataFrame:
    """
    Parse peak annotation file and return it as DataFrame.

    Parameters
    ----------
    fragment_path
        Location of the fragments file (must be tabix indexed).
    features
        A DataFrame with feature annotation, e.g. genes or a string of format `chr1:1-2000000` or`chr1-1-2000000`.
        Annotation has to contain columns: Chromosome, Start, End.
    extend_upsteam
        Number of nucleotides to extend every gene upstream (2000 by default to extend gene coordinates to promoter regions)
    extend_downstream
        Number of nucleotides to extend every gene downstream (0 by default)
    relative_coordinates
        Return the coordinates with their relative position to the middle of the features.
    """

    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is not available. It is required to work with the fragments file. Install pysam from PyPI (`pip install pysam`) or from GitHub (`pip install git+https://github.com/pysam-developers/pysam`)"
        )

    if isinstance(features, str):
        features = parse_region_string(features)

    fragments = pysam.TabixFile(fragment_path, parser=pysam.asBed())
    n_features = features.shape[0]

    dfs = []
    for i in tqdm(
        range(n_features), desc="Fetching Regions..."
    ):  # iterate over features (e.g. genes)
        f = features.iloc[i]
        fr = fragments.fetch(f.Chromosome, f.Start - extend_upstream, f.End + extend_downstream)
        df = pd.DataFrame(
            [(x.contig, x.start, x.end, x.name, x.score) for x in fr],
            columns=["Chromosome", "Start", "End", "Cell", "Score"],
        )
        if df.shape[0] != 0:
            df["Feature"] = f.Chromosome + "_" + str(f.Start) + "_" + str(f.End)

            if relative_coordinates:
                middle = int(f.Start + (f.End - f.Start) / 2)
                df.Start = df.Start - middle
                df.End = df.End - middle

            dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df




def parse_region_string(region: str) -> pd.DataFrame:
    feat_list = re.split("-|:", region)
    feature_df = pd.DataFrame(columns=["Chromosome", "Start", "End"])
    feature_df.loc[0] = feat_list
    feature_df = feature_df.astype({"Start": int, "End": int})

    return feature_df
