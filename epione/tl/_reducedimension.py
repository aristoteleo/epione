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
        device: str = "cpu",
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
    device : str, default "cpu"
        ``"cpu"`` runs the NumPy / SciPy implementation below. ``"cuda"``
        (or ``"cuda:N"``, or ``"auto"`` when a GPU is visible) runs TF-IDF and
        the randomized SVD with torch on the GPU: the same arithmetic in the
        same precision, and a torch port of sklearn's ``randomized_svd`` that
        draws the same random test matrix, so for a fixed ``random_state``
        the embedding agrees with the CPU one up to floating-point rounding.
        The sparse matrix is streamed in row blocks when it does not fit the
        GPU memory budget (``EPIONE_GPU_MEMORY_GB``). Needs
        ``pip install epione[gpu]``.
    **kwargs
        Additional arguments passed to randomized_svd
    """
    from ..core._gpu import resolve_device, is_torch_device
    device = resolve_device(device)
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

    if is_torch_device(device):
        console.info(f"TF-IDF + randomized SVD on {device} (torch)...")
        X_lsi = _lsi_tfidf_svd_torch(X_mat, n_components, device, **kwargs)
        return _lsi_finish(adata, X_lsi, n_components, remove_first_component)

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


def _lsi_finish(adata, X_lsi, n_components, remove_first_component):
    """Post-SVD steps of :func:`lsi`, shared by the CPU and torch paths."""
    console.info("Standardizing embeddings...")
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    if remove_first_component and n_components > 1:
        console.info("Removing first component (typically associated with peak count per cell)")
        X_lsi = X_lsi[:, 1:]
    adata.obsm["X_lsi"] = X_lsi
    console.info("LSI analysis completed!")


def _lsi_operand_torch(X, dev, budget=None):
    """TF-IDF -> L1 -> ``log1p(x * 1e4)`` of ``X`` as a device
    :class:`~epione.core._gpu.SparseOperand` (the torch twin of the CPU
    ``tfidf_fast`` + ``X.multiply(1 / row_sums)`` + ``log1p`` steps).

    Reproduces the CPU path's dtype promotion: integer or float64 counts are
    normalised in float64, float32 counts in float32, and every stored value
    goes through the same sequence of roundings.
    """
    from ..core._gpu import (
        SparseOperand, csr_row_blocks, csr_row_ids, get_torch, memory_budget,
        to_torch_csr, torch_dtype, _csr_rows_view,
    )
    torch = get_torch()
    X = X.tocsr() if scipy.sparse.issparse(X) else scipy.sparse.csr_matrix(X)
    dt = np.float32 if X.dtype == np.float32 else np.float64
    tdt = torch_dtype(dt)
    n_obs, n_vars = X.shape
    # Row / column sums in the CPU path's dtype (scipy sums ints exactly).
    row = np.asarray(X.sum(axis=1)).ravel()
    col = np.asarray(X.sum(axis=0)).ravel()
    row = np.where(row == 0, 1, row)
    col = np.where(col == 0, 1, col)
    inv_row = torch.as_tensor((1 / row).astype(dt, copy=False), device=dev)
    idf = torch.as_tensor((n_obs / col).astype(dt, copy=False), device=dev)

    def _tfidf(v, r, c):
        return (v.to(tdt) * inv_row[r]) * idf[c]

    if budget is None:
        budget = memory_budget(dev)
    # pass 1: L1 row sums of the TF-IDF matrix
    rs = torch.zeros(n_obs, dtype=tdt, device=dev)
    for blk in csr_row_blocks(X.indptr, max(budget // 64, 1)):
        T = to_torch_csr(_csr_rows_view(X, blk), dev)
        rr = csr_row_ids(T) + blk.start
        cc = T.col_indices().to(torch.int64)
        rs.index_add_(0, rr, _tfidf(T.values(), rr, cc))
        del T, rr, cc
    rs = torch.where(rs == 0, torch.ones_like(rs), rs)
    inv_rs = 1 / rs

    def _value_fn(v, r, c):
        return torch.log1p((_tfidf(v, r, c) * inv_rs[r]) * 1e4)

    return SparseOperand(X, dev, tdt, budget=budget, value_fn=_value_fn)


def _lsi_tfidf_svd_torch(X, n_components, device, **kwargs):
    """torch twin of the CPU TF-IDF + randomized SVD of :func:`lsi`."""
    from ..core._gpu import torch_device
    from ..core._gpu_ops import randomized_svd
    dev = torch_device(device)
    op = _lsi_operand_torch(X, dev)
    U = randomized_svd(op, n_components, dev=dev, n_iter=kwargs.pop("n_iter", "auto"),
                       **kwargs)[0]
    return U


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


