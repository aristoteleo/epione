from anndata import AnnData
from ..utils import console
import numpy as np
from scipy.sparse.linalg import svds
from typing import Union, Literal, Optional
import scipy as sp
import sklearn.preprocessing
import sklearn.utils.extmath
import scipy.sparse
import math

import gc

# The ``spectral`` / ``multi_spectral`` wrappers around snapATAC2's Rust
# embedding routines have been removed — for LSI / iterative-LSI use
# ``epi.tl.lsi`` / ``epi.tl.iterative_lsi`` (both pure Python) instead.


def lsi_muon(adata: AnnData, scale_embeddings=True, n_comps=50, remove_first_component=True):
    """
    Run Latent Semantic Indexing

    PARAMETERS
    ----------
    adata: AnnData
            AnnData object or MuData object with 'atac' modality
    scale_embeddings: bool (default: True)
            Scale embeddings to zero mean and unit variance
    n_comps: int (default: 50)
            Number of components to calculate with SVD
    remove_first_component: bool (default: True)
            Whether to remove the first component which is typically 
            associated with number of peaks or counts per cell
    """

    # In an unlikely scnenario when there are less 50 features, set n_comps to that value
    n_comps = min(n_comps, adata.X.shape[1])

    console.info("Performing SVD")
    cell_embeddings, svalues, peaks_loadings = svds(adata.X, k=n_comps)

    # Re-order components in the descending order
    cell_embeddings = cell_embeddings[:, ::-1]
    svalues = svalues[::-1]
    peaks_loadings = peaks_loadings[::-1, :]

    if scale_embeddings:
        cell_embeddings = (cell_embeddings - cell_embeddings.mean(axis=0)) / cell_embeddings.std(
            axis=0
        )

    stdev = svalues / np.sqrt(adata.X.shape[0] - 1)

    # Remove first component if requested (typically associated with peak count per cell)
    if remove_first_component and n_comps > 1:
        console.info("Removing first component (typically associated with peak count per cell)")
        cell_embeddings = cell_embeddings[:, 1:]
        peaks_loadings = peaks_loadings[1:, :]
        stdev = stdev[1:]

    adata.obsm["X_lsi"] = cell_embeddings
    adata.uns["lsi"] = {"stdev": stdev}
    adata.varm["LSI"] = peaks_loadings.T

    return None

def lsi_glue(
        adata: AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, 
        remove_first_component: bool = True,
        **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    
    # Remove first component if requested (typically associated with peak count per cell)
    if remove_first_component and n_components > 1:
        X_lsi = X_lsi[:, 1:]
    
    adata.obsm["X_lsi"] = X_lsi


def lsi(
        adata: AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, 
        use_fast_tfidf: bool = True,
        chunk_size: int = 10000,
        remove_first_component: bool = True,
        **kwargs
) -> None:
    r"""
    Optimized LSI analysis (following the Seurat v3 approach)
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    n_components : int, default 20
        Number of components to compute
    use_highly_variable : bool, optional
        Whether to use highly variable features
    use_fast_tfidf : bool, default True
        Whether to use optimized TF-IDF computation
    chunk_size : int, default 10000
        Chunk size for processing large matrices
    remove_first_component : bool, default True
        Whether to remove the first component which is typically 
        associated with number of peaks or counts per cell
    **kwargs
        Additional arguments passed to randomized_svd
    """
    console.info("Starting optimized LSI analysis...")

    # Feature selection
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata

    # Materialise X if it's an anndataoom BackedArray — TF-IDF uses
    # ``X / X.sum(...)`` and similar arithmetic ops that the lazy
    # BackedArray doesn't implement. For LSI we need the full matrix
    # in RAM anyway (randomized SVD scans it).
    X_mat = adata_use.X
    if type(X_mat).__name__ in ("BackedArray", "_SubsetBackedArray",
                                 "TransformedBackedArray", "ScaledBackedArray"):
        # Assemble by concatenating chunks.
        parts = []
        for _, _, chunk in X_mat.chunked():
            parts.append(chunk)
        if scipy.sparse.issparse(parts[0]):
            X_mat = scipy.sparse.vstack(parts).tocsr()
        else:
            X_mat = np.vstack(parts)

    console.info("Computing TF-IDF normalization...")
    if use_fast_tfidf:
        X = tfidf_fast(X_mat)
    else:
        X = tfidf(X_mat)
    
    console.info("Applying L1 normalization and log transformation...")
    # Optimized L1 normalization
    if scipy.sparse.issparse(X):
        # For sparse matrices, use more efficient normalization
        row_sums = X.sum(axis=1).A1
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        X_norm = X.multiply(1 / row_sums.reshape(-1, 1))
    else:
        # For dense matrices, use sklearn
        X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    
    # Optimized log transformation
    if scipy.sparse.issparse(X_norm):
        # For sparse matrices, avoid creating dense intermediate results
        X_norm.data = np.log1p(X_norm.data * 1e4)
    else:
        X_norm = np.log1p(X_norm * 1e4)
    
    console.info("Performing randomized SVD...")
    # Use randomized SVD with optimized parameters
    X_lsi = sklearn.utils.extmath.randomized_svd(
        X_norm, 
        n_components, 
        n_iter='auto',  # Let sklearn choose optimal iterations
        **kwargs
    )[0]
    
    console.info("Standardizing embeddings...")
    # Optimized standardization
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    
    # Remove first component if requested (typically associated with peak count per cell)
    if remove_first_component and n_components > 1:
        console.info("Removing first component (typically associated with peak count per cell)")
        X_lsi = X_lsi[:, 1:]
    
    adata.obsm["X_lsi"] = X_lsi
    console.info("LSI analysis completed!")


def tfidf(X: Union[np.ndarray, sp.sparse.spmatrix]) -> Union[np.ndarray, sp.sparse.spmatrix]:
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def tfidf_fast(X: Union[np.ndarray, sp.sparse.spmatrix]) -> Union[np.ndarray, sp.sparse.spmatrix]:
    r"""
    Optimized TF-IDF normalization for sparse matrices
    """
    if scipy.sparse.issparse(X):
        # For sparse matrices, use more efficient operations
        row_sums = X.sum(axis=1).A1  # Convert to 1D array
        col_sums = X.sum(axis=0).A1
        
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        col_sums[col_sums == 0] = 1
        
        # TF component
        tf = X.multiply(1 / row_sums.reshape(-1, 1))
        
        # IDF component
        idf = X.shape[0] / col_sums
        
        return tf.multiply(idf)
    else:
        # Fallback to original implementation for dense matrices
        return tfidf(X)



def idf(data, features=None):
    n, m = data.shape
    count = np.zeros(m)
    for batch, _, _ in data.chunked_X(2000):
        batch.data = np.ones(batch.indices.shape, dtype=np.float64)
        count += np.ravel(batch.sum(axis = 0))
    if features is not None:
        count = count[features]
    return np.log(n / (1 + count))


