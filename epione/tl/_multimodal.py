"""Joint representations of paired RNA + ATAC cells.

``combined_dims`` ports ArchR ``addCombinedDims`` (ArchR is MIT licensed,
Copyright (c) 2020 Jeffrey Granja and Ryan Corces). ``wnn`` is a
reimplementation of weighted nearest neighbours from Hao *et al.*, Cell 2021
(STAR Methods), written from the paper; no Seurat code is used. ``multivi``
is a thin wrapper around scvi-tools ``MULTIVI`` (BSD-3-Clause).
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------
# combined_dims (ArchR addCombinedDims)
# --------------------------------------------------------------------------

def _row_zscores(m: np.ndarray) -> np.ndarray:
    return (m - m.mean(axis=1, keepdims=True)) / m.std(axis=1, ddof=1, keepdims=True)


def _as_list(x, n, name):
    if x is None or np.isscalar(x) or isinstance(x, (bool, np.bool_)):
        return [x] * n
    x = list(x)
    if len(x) != n:
        raise ValueError(f"{name} must have one entry per representation ({n}), got {len(x)}")
    return x


def combined_dims(
    adata,
    reps: Sequence[str],
    *,
    dim_weights: Optional[Sequence[float]] = None,
    dims_to_use: Optional[Sequence[int]] = None,
    scale_dims: Union[bool, Sequence[bool]] = False,
    cor_cutoff: Optional[float] = 0.75,
    cor_to_depth: Optional[Mapping[str, Optional[np.ndarray]]] = None,
    depth_key: Optional[str] = None,
    key_added: str = "X_combined",
) -> np.ndarray:
    """Concatenate reduced dimensions into one joint embedding (ArchR ``addCombinedDims``).

    Each representation is optionally row-z-scored (``scale_dims``), stripped
    of dimensions whose absolute correlation with sequencing depth exceeds
    ``cor_cutoff``, divided by the square root of the sum of its column
    variances, multiplied by its weight, and the blocks are concatenated.
    With equal weights every modality contributes the same total variance,
    whatever its number of dimensions.

    Arguments:
        adata: AnnData holding the representations in ``obsm``.
        reps: ``obsm`` keys to combine, e.g. ``["X_lsi", "X_pca"]``.
        dim_weights: one weight per representation (default all 1).
        dims_to_use: 1-based dimensions to consider in every representation
            (ArchR ``dimsToUse``); default all.
        scale_dims: row-z-score before combining (ArchR ``scaleDims``). A
            single bool or one per representation.
        cor_cutoff: drop dimensions with ``|cor(dim, log10(depth + 1))|``
            above this value (ArchR ``corCutOff``). ``None`` keeps all.
        cor_to_depth: precomputed depth correlations per representation, as
            ArchR stores them (``corToDepth$scaled`` or ``$none`` matching
            ``scale_dims``). Representations missing from the mapping, or
            mapped to ``None``, use ``depth_key`` if given, otherwise zeros.
        depth_key: ``obs`` column with per-cell fragment counts, used to
            compute depth correlations that ``cor_to_depth`` does not give.
        key_added: ``obsm`` key for the result.

    Returns:
        The combined ``(n_obs, n_dims)`` matrix, also written to
        ``adata.obsm[key_added]``. ``adata.uns[key_added]`` records the kept
        dimensions of each representation.

    Examples:
        >>> epi.tl.combined_dims(adata, ["X_lsi", "X_pca_rna"], key_added="X_combined")
        >>> sc.pp.neighbors(adata, use_rep="X_combined")
    """
    reps = list(reps)
    n = len(reps)
    weights = [1.0] * n if dim_weights is None else [float(w) for w in _as_list(dim_weights, n, "dim_weights")]
    scales = [bool(s) for s in _as_list(scale_dims, n, "scale_dims")]
    cor_to_depth = dict(cor_to_depth or {})
    depth = None
    if depth_key is not None:
        depth = np.log10(np.asarray(adata.obs[depth_key], dtype=np.float64) + 1.0)

    blocks, kept = [], {}
    for rep, w, scale in zip(reps, weights, scales):
        mat = np.asarray(adata.obsm[rep], dtype=np.float64)
        if scale:
            mat = _row_zscores(mat)
        ctd = cor_to_depth.get(rep)
        if ctd is None:
            if depth is not None:
                ctd = np.array([np.corrcoef(mat[:, j], depth)[0, 1] for j in range(mat.shape[1])])
            else:
                ctd = np.zeros(mat.shape[1])
        ctd = np.asarray(ctd, dtype=np.float64)
        if dims_to_use is None:
            cand = np.arange(len(ctd))
        else:
            # R's intersect(dimsToUse, seq_along(corToDepth)): keeps dimsToUse order, drops duplicates.
            cand = np.array(list(dict.fromkeys(int(d) - 1 for d in dims_to_use if 1 <= int(d) <= len(ctd))), dtype=int)
        if cor_cutoff is not None:
            cand = cand[np.abs(ctd[cand]) <= cor_cutoff]
        rd = mat[:, cand]
        col_var = rd.var(axis=0, ddof=1)
        blocks.append(rd * (1.0 / np.sqrt(col_var.sum())) * w)
        kept[rep] = (cand + 1).tolist()

    out = np.hstack(blocks)
    adata.obsm[key_added] = out
    adata.uns[key_added] = {"reps": reps, "dims_kept": kept, "dim_weights": weights, "scale_dims": scales}
    return out


# --------------------------------------------------------------------------
# WNN (Hao et al., Cell 2021)
# --------------------------------------------------------------------------

def _l2norm(x: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm


def _knn(x: np.ndarray, k: int, n_jobs: int):
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs).fit(x)
    dist, idx = nn.kneighbors(x)
    # Put the cell itself first even under exact duplicates.
    rows = np.arange(x.shape[0])
    self_pos = np.argmax(idx == rows[:, None], axis=1)
    has_self = (idx == rows[:, None]).any(axis=1)
    for i in np.where(~has_self | (self_pos != 0))[0]:
        others = idx[i][idx[i] != i][: k - 1]
        idx[i] = np.concatenate([[i], others])
        dist[i] = np.linalg.norm(x[idx[i]] - x[i], axis=1)
    return dist, idx


def _kernel(d: np.ndarray, d1: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """exp(-max(d - d1, 0) / (sigma - d1)) with per-cell offset d1 and bandwidth sigma.

    ``d`` is ``(n,)`` or ``(n, c)``; ``d1`` and ``sigma`` are ``(n,)``.
    """
    width = np.maximum(sigma - d1, 1e-12)
    if d.ndim == 2:
        d1, width = d1[:, None], width[:, None]
    return np.exp(-np.maximum(d - d1, 0.0) / width)


def _modality_stats(x: np.ndarray, k: int, knn_range: int, n_jobs: int):
    n = x.shape[0]
    dist_r, idx_r = _knn(x, min(knn_range, n), n_jobs)
    idx_k = idx_r[:, :k]
    d1 = dist_r[:, 1] if dist_r.shape[1] > 1 else np.zeros(n)
    # Jaccard similarity of k-NN sets between each cell and its knn_range candidates.
    A = sp.csr_matrix((np.ones(n * k), idx_k.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))
    S = (A @ A.T).tocsr()
    inter = np.asarray(S[np.repeat(np.arange(n), idx_r.shape[1]), idx_r.ravel()]).reshape(n, -1)
    jac = inter / (2 * k - inter)
    top = np.argsort(-jac, axis=1, kind="stable")[:, :k]
    sigma = np.take_along_axis(dist_r, top, axis=1).mean(axis=1)
    return dist_r, idx_r, idx_k, d1, sigma


def wnn(
    adata,
    reps: Sequence[str] = ("X_pca", "X_lsi"),
    *,
    dims: Optional[Sequence[Optional[Sequence[int]]]] = None,
    modality_names: Optional[Sequence[str]] = None,
    n_neighbors: int = 20,
    knn_range: int = 200,
    epsilon: float = 1e-4,
    l2_norm: bool = True,
    key_added: str = "wnn",
    n_jobs: int = -1,
):
    """Weighted nearest neighbours of paired multimodal cells (Hao *et al.*, Cell 2021).

    For every cell and modality, WNN asks how well the cell's own profile is
    predicted by its neighbours in that modality versus by its neighbours in
    the other modalities. The ratio of the two affinities, softmaxed across
    modalities, gives per-cell modality weights; the final graph links each
    cell to the neighbours with the highest weighted sum of per-modality
    affinities. Reimplemented from the paper's STAR Methods.

    Arguments:
        adata: AnnData with one embedding per modality in ``obsm`` (same cells).
        reps: ``obsm`` keys, one per modality, e.g. RNA PCA and ATAC LSI.
        dims: 1-based dimensions to use per representation (``None`` = all);
            for LSI the first, depth-correlated component is usually dropped,
            e.g. ``dims=[None, range(2, 51)]``.
        modality_names: names for the weight columns (default ``reps``).
        n_neighbors: neighbours per cell in the modality and WNN graphs.
        knn_range: candidate neighbours considered per modality.
        epsilon: constant added to the cross-modality affinity.
        l2_norm: L2-normalise each embedding per cell first.
        key_added: graph prefix. Writes ``obsp[f"{key_added}_connectivities"]``,
            ``obsp[f"{key_added}_distances"]``, ``uns[key_added]`` (scanpy
            neighbors format, so ``sc.tl.umap(adata, neighbors_key=key_added)``
            and ``sc.tl.leiden(adata, neighbors_key=key_added)`` work) and
            ``obs[f"{key_added}_weight_<modality>"]``.
        n_jobs: threads for the kNN searches.

    Returns:
        ``(weights, indices)``: the ``(n_obs, n_modalities)`` modality weights
        and the ``(n_obs, n_neighbors)`` WNN neighbour indices (the cell itself first).

    Examples:
        >>> epi.tl.wnn(adata, reps=["X_pca", "X_lsi"], dims=[None, range(2, 51)])
        >>> sc.tl.umap(adata, neighbors_key="wnn")
        >>> sc.tl.leiden(adata, neighbors_key="wnn")
    """
    reps = list(reps)
    m = len(reps)
    if m < 2:
        raise ValueError("wnn needs at least two representations")
    names = list(modality_names) if modality_names is not None else reps
    dims = _as_list(dims, m, "dims") if dims is not None else [None] * m
    k = int(n_neighbors)

    xs = []
    for rep, d in zip(reps, dims):
        x = np.asarray(adata.obsm[rep], dtype=np.float64)
        if d is not None:
            x = x[:, np.asarray(list(d), dtype=int) - 1]
        xs.append(_l2norm(x) if l2_norm else x)
    n = xs[0].shape[0]
    stats = [_modality_stats(x, k, knn_range, n_jobs) for x in xs]

    # Within- and cross-modality predictions and their affinities.
    scores = np.zeros((n, m))
    for a in range(m):
        x = xs[a]
        _, _, idx_k_a, d1, sigma = stats[a]
        within = x[idx_k_a].mean(axis=1)
        th_within = _kernel(np.linalg.norm(x - within, axis=1), d1, sigma)
        cross = np.zeros(n)
        for b in range(m):
            if b == a:
                continue
            pred = x[stats[b][2]].mean(axis=1)
            cross += _kernel(np.linalg.norm(x - pred, axis=1), d1, sigma)
        cross /= (m - 1)
        scores[:, a] = th_within / (cross + epsilon)
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)

    # Candidate neighbours: union of each modality's knn_range nearest cells.
    cand = np.concatenate([s[1] for s in stats], axis=1)
    cand = np.sort(cand, axis=1)
    dup = np.zeros_like(cand, dtype=bool)
    dup[:, 1:] = cand[:, 1:] == cand[:, :-1]
    rows = np.repeat(np.arange(n), cand.shape[1])
    flat = cand.ravel()
    affinity = np.zeros(flat.shape[0])
    for a in range(m):
        x = xs[a]
        _, _, _, d1, sigma = stats[a]
        d = np.linalg.norm(x[rows] - x[flat], axis=1).reshape(n, -1)
        affinity += (weights[:, a][:, None] * _kernel(d, d1, sigma)).ravel()
    affinity = affinity.reshape(n, -1)
    affinity[dup] = -np.inf
    affinity[cand == np.arange(n)[:, None]] = np.inf  # keep the cell itself first
    order = np.argsort(-affinity, axis=1, kind="stable")[:, :k]
    nn_idx = np.take_along_axis(cand, order, axis=1)
    nn_aff = np.take_along_axis(affinity, order, axis=1)
    nn_aff[:, 0] = 1.0

    # scanpy-style graph: distances = 1 - affinity (self excluded), connectivities symmetric.
    r = np.repeat(np.arange(n), k - 1)
    c = nn_idx[:, 1:].ravel()
    a_ = np.clip(nn_aff[:, 1:].ravel(), 0.0, 1.0)
    dist = sp.csr_matrix((np.maximum(1.0 - a_, 1e-12), (r, c)), shape=(n, n))
    conn = sp.csr_matrix((a_, (r, c)), shape=(n, n))
    conn = conn.maximum(conn.T).tocsr()

    adata.obsp[f"{key_added}_distances"] = dist
    adata.obsp[f"{key_added}_connectivities"] = conn
    adata.uns[key_added] = {
        "connectivities_key": f"{key_added}_connectivities",
        "distances_key": f"{key_added}_distances",
        "params": {"method": "wnn", "n_neighbors": k, "knn_range": int(knn_range), "reps": reps,
                   "epsilon": float(epsilon), "l2_norm": bool(l2_norm)},
    }
    for j, name in enumerate(names):
        adata.obs[f"{key_added}_weight_{name}"] = weights[:, j]
    return weights, nn_idx


# --------------------------------------------------------------------------
# MultiVI (scvi-tools)
# --------------------------------------------------------------------------

def multivi(
    rna,
    atac,
    *,
    batch_key: Optional[str] = None,
    n_latent: Optional[int] = None,
    max_epochs: int = 500,
    accelerator: str = "auto",
    seed: int = 0,
    key_added: str = "X_multivi",
    model_kwargs: Optional[dict] = None,
    train_kwargs: Optional[dict] = None,
):
    """Joint latent space of paired RNA + ATAC cells with MultiVI (Ashuach *et al.*, Nat Methods 2023).

    A thin wrapper around scvi-tools ``MULTIVI``. Needs the optional extra
    ``pip install epione[multimodal]``. Uses a GPU when one is available.

    Arguments:
        rna: AnnData of raw RNA counts.
        atac: AnnData of raw ATAC peak counts, the same cells in the same order.
        batch_key: optional ``obs`` column with batch labels.
        n_latent: latent dimensions (scvi-tools default when ``None``).
        max_epochs: training epochs (early stopping on by default).
        accelerator: lightning accelerator, ``"auto"``, ``"gpu"`` or ``"cpu"``.
        seed: ``scvi.settings.seed``.
        key_added: ``obsm`` key written on both ``rna`` and ``atac``.
        model_kwargs, train_kwargs: passed to ``MULTIVI(...)`` and ``.train(...)``.

    Returns:
        The trained ``scvi.model.MULTIVI``; the latent space is written to
        ``rna.obsm[key_added]`` and ``atac.obsm[key_added]``.
    """
    try:
        import scvi
        import mudata as md
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("epi.tl.multivi needs scvi-tools: pip install 'epione[multimodal]'") from e

    if rna.n_obs != atac.n_obs or not np.array_equal(np.asarray(rna.obs_names), np.asarray(atac.obs_names)):
        raise ValueError("rna and atac must contain the same cells in the same order")
    scvi.settings.seed = seed
    r = rna.copy()
    a = atac.copy()
    r.layers["counts"] = r.X.copy()
    a.layers["counts"] = a.X.copy()
    mdata = md.MuData({"rna": r, "atac": a})
    if batch_key is not None:
        mdata.obs[batch_key] = np.asarray(rna.obs[batch_key])
    scvi.model.MULTIVI.setup_mudata(
        mdata, rna_layer="counts", atac_layer="counts", batch_key=batch_key,
        modalities={"rna_layer": "rna", "atac_layer": "atac", **({"batch_key": "rna"} if batch_key else {})},
    )
    kw = dict(model_kwargs or {})
    if n_latent is not None:
        kw["n_latent"] = n_latent
    model = scvi.model.MULTIVI(mdata, n_genes=r.n_vars, n_regions=a.n_vars, **kw)
    model.train(max_epochs=max_epochs, accelerator=accelerator, **(train_kwargs or {}))
    z = model.get_latent_representation()
    rna.obsm[key_added] = z
    atac.obsm[key_added] = z
    return model


__all__ = ["combined_dims", "wnn", "multivi"]
