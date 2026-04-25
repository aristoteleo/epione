"""Cell embedding from imputed sc-Hi-C matrices.

Concatenates the upper triangle of every ``<cell>.npz`` written by
:func:`impute_cells`, runs PCA on the resulting cell × feature matrix,
and stores the result on the input ``AnnData``. Standard scanpy
``sc.pp.neighbors`` / ``sc.tl.umap`` / ``sc.tl.leiden`` work directly on
the result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _upper_triangle_indices(n: int, k: int = 1):
    """Indices of strictly upper triangle (k=1 excludes the diagonal).

    Excluding the diagonal drops self-contacts which dominate raw counts
    and would otherwise flatten the PCA onto coverage differences.
    """
    return np.triu_indices(n, k=k)


def _features_from_npz(npz_path: Path, chromosomes: Sequence[str],
                       layout) -> np.ndarray:
    """Pull upper triangles for the requested chromosomes from one cell's
    ``.npz``, concatenate in ``chromosomes`` order, and return a 1-D
    feature vector. ``layout`` is the cached upper-tri indices per chrom.
    """
    z = np.load(npz_path)
    chunks = []
    for ch in chromosomes:
        if ch not in z.files:
            n = layout[ch][2]
            # cell missing this chrom — emit zeros so feature dimension
            # stays uniform across cells.
            chunks.append(np.zeros(layout[ch][3], dtype=np.float32))
            continue
        P = z[ch]
        iu, ju = layout[ch][0], layout[ch][1]
        chunks.append(P[iu, ju].astype(np.float32))
    return np.concatenate(chunks)


def embedding(
    adata,
    *,
    n_components: int = 20,
    chromosomes: Optional[Sequence[str]] = None,
    standardise: bool = True,
    random_state: int = 0,
):
    """PCA embedding of imputed sc-Hi-C contacts.

    Arguments:
        adata: AnnData from :func:`load_cool_collection` after
            :func:`impute_cells` has been run.
        n_components: PCA components. Capped at ``min(n_cells - 1,
            n_features)``.
        chromosomes: subset to use as features. Default = the same set
            imputation ran on (``adata.uns['hic']['chromosomes']``).
        standardise: per-feature z-score before PCA (recommended; the
            scHiCluster paper does this implicitly via SVD on the
            mean-centred matrix).
        random_state: forwarded to ``sklearn.decomposition.PCA``.

    Returns:
        Mutates ``adata`` in place and returns it. Sets
        ``adata.X`` (the cell × feature matrix, ``float32``),
        ``adata.obsm['X_pca']``, ``adata.varm['PCs']``,
        ``adata.uns['pca']`` (variance / explained-ratio).
    """
    from sklearn.decomposition import PCA

    info = adata.uns.get("hic", {})
    imputed_dir = info.get("imputed_dir")
    if imputed_dir is None:
        raise ValueError(
            "adata.uns['hic']['imputed_dir'] not set — run "
            "epione.sc_hic.impute_cells() first"
        )
    imputed_dir = Path(imputed_dir)

    chromosomes = list(chromosomes) if chromosomes is not None else list(
        info.get("impute_params", {}).get("chromosomes",
                                          info.get("chromosomes", []))
    )
    if not chromosomes:
        raise ValueError("no chromosomes specified for embedding")

    # Build a layout cache: per chrom, the upper-tri indices and
    # length, so feature concatenation is consistent across cells.
    n_bins = info.get("n_chrom_bins", {})
    layout = {}
    for ch in chromosomes:
        n = int(n_bins.get(ch, 0))
        if n < 2:
            # need at least 2 bins for any upper-tri entries
            iu, ju = np.array([], dtype=int), np.array([], dtype=int)
        else:
            iu, ju = _upper_triangle_indices(n, k=1)
        layout[ch] = (iu, ju, n, len(iu))

    cell_ids = list(adata.obs_names)
    feats = []
    for cid in cell_ids:
        npz_path = imputed_dir / f"{cid}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"missing imputed file for cell {cid}: {npz_path}. "
                "Run impute_cells with overwrite=False to fill gaps."
            )
        feats.append(_features_from_npz(npz_path, chromosomes, layout))
    X = np.vstack(feats).astype(np.float32)

    # Drop all-zero features so PCA isn't dominated by structurally-empty
    # bin pairs (e.g. inside masked centromere bands).
    var_mask = X.std(axis=0) > 0
    X = X[:, var_mask]

    if standardise and X.size:
        X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
        X = np.nan_to_num(X, nan=0.0)

    n_pc = max(1, min(n_components, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=n_pc, random_state=random_state)
    X_pca = pca.fit_transform(X)

    # Anndata accepts X with any dtype; we stay float32 to keep memory low.
    adata._inplace_subset_var(np.array([], dtype=int))  # clear var
    import anndata as ad
    new = ad.AnnData(
        X=X.astype(np.float32),
        obs=adata.obs.copy(),
        uns=dict(adata.uns),
    )
    new.obsm["X_pca"] = X_pca.astype(np.float32)
    new.uns["pca"] = {
        "variance": pca.explained_variance_.astype(np.float32),
        "variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
    }
    new.uns["hic"] = dict(info)
    new.uns["hic"]["embedding_chromosomes"] = list(chromosomes)
    new.uns["hic"]["n_features"] = int(X.shape[1])
    return new
