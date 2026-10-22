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
    from ._imputed_io import ImputedCell
    z = ImputedCell(npz_path)
    chunks = []
    for ch in chromosomes:
        if ch not in z.files:
            n = layout[ch][2]
            # cell missing this chrom — emit zeros so feature dimension
            # stays uniform across cells.
            chunks.append(np.zeros(layout[ch][3], dtype=np.float32))
            continue
        iu, ju = layout[ch][0], layout[ch][1]
        chunks.append(z.values(ch, iu, ju))
    return np.concatenate(chunks)


def embedding(
    adata,
    *,
    n_components: int = 20,
    chromosomes: Optional[Sequence[str]] = None,
    standardise: bool = True,
    random_state: int = 0,
    method: str = "pca",
    device=None,
    higashi_params: Optional[dict] = None,
    work_dir=None,
    dim: int = 50,
    dist: int = 1_000_000,
    scale_factor: float = 100_000,
    norm_sig: bool = False,
):
    """Cell embedding: PCA on imputed contacts (default) or Higashi.

    ``method="pca"`` (default) is described below. ``method="higashi"``
    returns Higashi's stage-1 hypergraph cell embedding
    (``obsm['X_higashi']``, ``higashi_params['dimensions']`` wide, also in
    ``X``). It needs no prior :func:`impute_cells`; when
    :func:`impute_cells` (``method="higashi"``) already trained a model
    with the same settings that model is reused. ``n_components`` and
    ``standardise`` are ignored for Higashi; ``random_state``, ``device``
    and ``higashi_params`` are as in :func:`impute_cells`, and ``work_dir``
    (optional) is where ``higashi_model.pt`` is saved.

    ``method="schicluster"`` is scHiCluster's ``hicluster embedding``, the
    usual choice at 100 kb, where whole upper triangles are too large: per
    chromosome, the imputed contacts within ``dist`` of the diagonal are
    scaled by ``scale_factor`` and reduced by a ``dim``-component truncated
    SVD; the reductions are concatenated and reduced again
    (``obsm['X_schicluster']``; ``norm_sig`` divides by the singular
    values). The defaults are the CLI's.

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
    if method == "higashi":
        from ._higashi_api import higashi_embedding
        return higashi_embedding(adata, chromosomes=chromosomes, random_state=random_state, device=device,
                                 higashi_params=higashi_params, work_dir=work_dir)
    if method == "schicluster":
        return _schicluster_embedding(adata, chromosomes=chromosomes, dim=dim, dist=dist,
                                      scale_factor=scale_factor, norm_sig=norm_sig,
                                      random_state=random_state)
    if method != "pca":
        raise ValueError(f"method must be 'pca', 'schicluster' or 'higashi', got {method!r}")
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


def _band_indices(n: int, dist: int, resolution: int):
    """Upper-triangle pixels within ``dist`` of the diagonal (scHiCluster ``make_idx``),
    in row-major order, without enumerating the whole upper triangle."""
    k = min(int(np.ceil(dist / resolution + 1)) - 1, n - 1)   # offsets d with 1 <= d < dist/res + 1
    if k < 1:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    lengths = np.minimum(k, n - 1 - np.arange(n))
    iu = np.repeat(np.arange(n, dtype=np.int64), lengths)
    starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    ju = iu + 1 + (np.arange(iu.size, dtype=np.int64) - starts)
    return iu, ju


def _truncated_svd(M: np.ndarray, dim: int, norm_sig: bool, random_state):
    """scHiCluster ``svd``: ARPACK truncated SVD, optionally divided by the singular values."""
    from sklearn.decomposition import TruncatedSVD

    dim = min(dim, M.shape[0] - 1, M.shape[1] - 1)
    model = TruncatedSVD(n_components=dim, algorithm="arpack", random_state=random_state)
    decomp = model.fit_transform(M)
    if norm_sig:
        keep = model.singular_values_ > 0
        decomp = decomp[:, keep] / model.singular_values_[keep][None, :]
    return decomp


def _schicluster_embedding(adata, *, chromosomes, dim, dist, scale_factor, norm_sig,
                           random_state):
    """scHiCluster ``hicluster embedding`` on the imputed ``.npz`` of every cell.

    Per chromosome: the imputed contacts within ``dist`` of the diagonal
    (cells × pixels), multiplied by ``scale_factor``, reduced by a
    ``dim``-component truncated SVD; the chromosome reductions are
    concatenated and reduced again to ``dim`` components.
    """
    import anndata as ad
    from ._imputed_io import ImputedCell

    info = adata.uns.get("hic", {})
    imputed_dir = info.get("imputed_dir")
    if imputed_dir is None:
        raise ValueError("adata.uns['hic']['imputed_dir'] not set; run impute_cells() first")
    imputed_dir = Path(imputed_dir)
    resolution = int(info["resolution"])
    chromosomes = list(chromosomes) if chromosomes is not None else list(
        info.get("impute_params", {}).get("chromosomes", info.get("chromosomes", [])))
    n_bins = info.get("n_chrom_bins", {})
    chromosomes = [c for c in chromosomes if int(n_bins.get(c, 0)) > 2]
    idx = {c: _band_indices(int(n_bins[c]), int(dist), resolution) for c in chromosomes}
    cells = list(adata.obs_names)
    mats = {c: np.zeros((len(cells), idx[c][0].size), dtype=np.float32) for c in chromosomes}
    for i, cid in enumerate(cells):
        npz_path = imputed_dir / f"{cid}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"missing imputed file for cell {cid}: {npz_path}")
        z = ImputedCell(npz_path)
        for c in chromosomes:
            if c in z.files:
                mats[c][i] = z.values(c, *idx[c])
    parts = []
    for c in chromosomes:
        M = mats.pop(c)
        M *= np.float32(scale_factor)
        parts.append(_truncated_svd(M, dim, norm_sig, random_state))
    final = _truncated_svd(np.concatenate(parts, axis=1), dim, norm_sig, random_state)
    new = ad.AnnData(X=final.astype(np.float32), obs=adata.obs.copy(), uns=dict(adata.uns))
    new.obsm["X_schicluster"] = final.astype(np.float32)
    new.uns["hic"] = dict(info)
    new.uns["hic"]["embedding"] = {"method": "schicluster", "dim": int(dim), "dist": int(dist),
                                   "scale_factor": float(scale_factor), "norm_sig": bool(norm_sig),
                                   "chromosomes": chromosomes}
    return new
