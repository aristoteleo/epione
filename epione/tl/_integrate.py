"""Unified cross-modality integration entry point.

``epi.tl.integrate(atac, rna, method='cca', ...)`` — wraps ``cca_py``
(ArchR-style CCA for scRNA ↔ scATAC label transfer) and exposes a
single signature usable from tutorials. Future methods ('harmony',
'mnn', 'scanorama') plug in under the same dispatcher.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np


def integrate(
    adata1,
    adata2,
    *,
    method: Literal["cca"] = "cca",
    features: Optional[Sequence[str]] = None,
    layer: Optional[str] = None,
    num_cc: int = 30,
    standardize_inputs: bool = True,
    l2_normalize: bool = True,
    key_added: str = "X_cca",
    seed: Optional[int] = 42,
    **kwargs,
):
    """Cross-modality (or cross-dataset) integration into a shared embedding.

    After running, both ``adata1`` and ``adata2`` gain
    ``obsm[key_added]`` — a shared ``(n_obs, num_cc)`` embedding. Use
    :func:`epi.tl.transfer_labels` (or a plain kNN) on top of that
    to transfer celltype annotations across modalities.

    Parameters
    ----------
    adata1, adata2
        Two AnnData objects (cells × genes / gene-scores). CCA
        aligns them on a shared feature subset.
    method
        Only ``'cca'`` right now — wraps :mod:`cca_py`. Future values
        plug into the same dispatcher.
    features
        Explicit shared-feature set. Default: intersection of the two
        objects' ``var['highly_variable']`` flags (when present) or
        all shared ``var_names``.
    num_cc
        Number of canonical components (ArchR default 30).
    standardize_inputs
        Per-feature z-score before CCA.
    l2_normalize
        After CCA, L2-normalise each cell's embedding. Matches Seurat's
        ``FindTransferAnchors(reduction='cca')`` behaviour and gives
        cleaner cosine-kNN transfers downstream.
    key_added
        obsm key to write on both objects.
    seed
        Reproducibility seed for the SVD.

    Returns
    -------
    The underlying CCA result object (``cca_py.RunCCAResult``) — mostly
    useful for diagnostics. The main side-effect is the embedding
    written back to ``obsm[key_added]`` on both ``adata1`` and
    ``adata2``.

    Examples
    --------
    >>> epi.tl.integrate(atac, rna, method='cca',
    ...                  features=shared_hvg, num_cc=30)
    >>> # ArchR-style kNN label transfer in the shared space
    >>> from sklearn.neighbors import NearestNeighbors
    >>> nn = NearestNeighbors(n_neighbors=30, metric='cosine')
    >>> nn.fit(rna.obsm['X_cca'])
    >>> _, idx = nn.kneighbors(atac.obsm['X_cca'])
    >>> # majority-vote transfer of rna.obs['celltype'] etc.
    """
    method = method.lower()
    if method != "cca":
        raise NotImplementedError(
            f"integrate: method={method!r} not implemented; supported: 'cca'"
        )

    from cca_py import run_cca_anndata

    res = run_cca_anndata(
        adata1, adata2,
        features=features,
        layer=layer,
        num_cc=num_cc,
        standardize_inputs=standardize_inputs,
        key_added=key_added,
        seed=seed,
        **kwargs,
    )

    if l2_normalize:
        for a in (adata1, adata2):
            Z = a.obsm[key_added].astype("float64")
            Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
            a.obsm[key_added] = Z.astype("float32")

    return res


__all__ = ["integrate"]
