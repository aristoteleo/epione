"""Joint UMAP of two AnnData objects in a shared embedding.

``epi.tl.joint_embedding(adata1, adata2, use_rep='X_cca')`` concatenates
two AnnData objects, stitches their shared embedding together, and runs
``scanpy.pp.neighbors`` + ``scanpy.tl.umap`` on top — the standard
diagnostic plot for cross-modality integrations (Seurat /
addGeneIntegrationMatrix / chromium multiome).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def joint_embedding(
    adata1,
    adata2,
    *,
    use_rep: str = "X_cca",
    labels: Sequence[str] = ("query", "reference"),
    label_key: str = "modality",
    label_columns: Optional[Sequence[Optional[str]]] = None,
    merged_label_key: str = "celltype_joint",
    strip_prefix: Optional[str] = None,
    n_neighbors: int = 30,
    metric: str = "cosine",
    random_state: int = 0,
    compute_umap: bool = True,
):
    """Concat + shared-embedding + neighbours + UMAP in one call.

    Parameters
    ----------
    adata1, adata2
        Two AnnData objects with matching ``obsm[use_rep]``.
    use_rep
        Shared embedding to transfer onto the joint object.
    labels
        Two-element sequence naming each modality (used to fill the
        ``label_key`` obs column and as concat keys).
    label_key
        Name of the obs column identifying which modality each cell is from.
    label_columns
        Pair of obs columns (one per object) to merge into a single
        joint celltype label at ``merged_label_key``. ``None`` skips
        the merge for an object (cells get empty-string labels).
        Example: ``('celltype_coarse', 'BioClassification')``.
    merged_label_key
        Obs column name for the merged label.
    strip_prefix
        Optional regex to strip from the ``adata2`` (reference) side's
        label column when merging — e.g. ``r'^\\d+_'`` for Granja 2019.
    n_neighbors
        scanpy neighbors parameter.
    metric
        Distance metric for neighbours.
    random_state
        UMAP seed.
    compute_umap
        If ``False``, only compute the neighbour graph (skip UMAP).

    Returns
    -------
    A new ``AnnData`` containing both objects' cells in the shared
    embedding, with ``obsm[use_rep]``, neighbours, and (unless
    ``compute_umap=False``) ``obsm['X_umap']`` computed.
    """
    import anndata as ad
    import scanpy as sc
    import pandas as pd

    name1, name2 = labels
    joint = ad.concat(
        {name1: adata1, name2: adata2},
        label=label_key, merge="unique", pairwise=False,
    )
    joint.obsm[use_rep] = np.vstack(
        [adata1.obsm[use_rep], adata2.obsm[use_rep]]
    )

    if label_columns is not None:
        col1, col2 = label_columns
        lab1 = (adata1.obs[col1].astype(str).to_numpy() if col1 is not None
                else np.array([""] * adata1.n_obs))
        lab2_series = (adata2.obs[col2].astype(str)
                       if col2 is not None else pd.Series([""] * adata2.n_obs))
        if strip_prefix is not None and col2 is not None:
            lab2_series = lab2_series.str.replace(strip_prefix, "", regex=True)
        joint.obs[merged_label_key] = np.concatenate([lab1, lab2_series.to_numpy()])

    sc.pp.neighbors(joint, use_rep=use_rep,
                    n_neighbors=n_neighbors, metric=metric)
    if compute_umap:
        sc.tl.umap(joint, random_state=random_state)

    return joint


__all__ = ["joint_embedding"]
