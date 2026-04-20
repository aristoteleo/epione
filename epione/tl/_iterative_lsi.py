"""
ArchR-style Iterative LSI for scATAC-seq data.

Reference:
    Granja, J.M., Corces, M.R., Pierce, S.E. et al. ArchR is a scalable software
    package for integrative single-cell chromatin accessibility analysis.
    Nat Genet 53, 403-411 (2021).

Algorithm (LSIMethod = 2, "Seurat" style, default in ArchR):
    1. Initial feature selection: keep the top `total_features` most accessible
       features (binarized), then clip the top `filter_quantile` tail.
    2. Optionally sub-sample `sample_cells_pre` cells for the first round to
       keep runtime bounded on very large datasets.
    3. TF-logIDF normalise, randomized SVD, drop components that correlate with
       sequencing depth (|r| > cor_cut_off).
    4. Cluster the embedding with Leiden (resolution can vary per iteration).
    5. Pick the top `var_features` features whose log-normalised pseudobulk
       accessibility varies most across clusters.
    6. Repeat steps 3-5 `iterations - 1` more times; the final LSI is run on
       all cells using the last feature set.

Results stored on the AnnData:
    adata.obsm[key_added]              : (n_obs, n_dims) final embedding
    adata.varm[key_added + "_loadings"]: (n_vars, n_dims) feature loadings
                                         (zero for features not selected in the
                                         final round)
    adata.uns[key_added]               : dict with keys
        "stdev"          - singular values of the final SVD
        "features_final" - boolean mask (n_vars,) of features used last round
        "depth_cor"      - per-component correlation with log10(depth)
        "kept_dims"      - indices of components kept after cor_cut_off filter
        "iterations"     - list of per-iteration info (features, clusters)
        "params"         - the parameters the call was made with
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence, Union
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
import sklearn.utils.extmath

from ..utils import console


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _binarize(X):
    """Return a binary (0/1) copy of X with dtype float32."""
    if sp.issparse(X):
        out = X.copy().astype(np.float32)
        out.data = (out.data > 0).astype(np.float32)
        out.eliminate_zeros()
        return out
    return (X > 0).astype(np.float32)


def _row_sums(X):
    if sp.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return X.sum(axis=1)


def _col_sums(X):
    if sp.issparse(X):
        return np.asarray(X.sum(axis=0)).ravel()
    return X.sum(axis=0)


def _tf_logidf(X, scale_to: float = 10000.0):
    """Seurat v3 / ArchR LSIMethod=2 normalisation.

    Computes log1p( (X / rowSums(X)) * idf * scale_to )
    where idf = log(1 + ncells / colSums(X)).

    Returns sparse CSR if X is sparse, else dense float32.
    """
    n_cells = X.shape[0]
    col = _col_sums(X)
    col_safe = np.where(col > 0, col, 1.0)
    idf = np.log(1.0 + n_cells / col_safe)

    row = _row_sums(X)
    row_safe = np.where(row > 0, row, 1.0)

    if sp.issparse(X):
        X = X.tocsr()
        # tf = X / row
        inv_row = (1.0 / row_safe).astype(np.float32)
        tf = X.multiply(inv_row[:, None])
        # * idf
        tf = tf.multiply(idf.astype(np.float32))
        # * scale_to
        tf = tf.tocsr()
        tf.data = np.log1p(tf.data * scale_to).astype(np.float32)
        return tf
    else:
        tf = (X / row_safe[:, None]) * idf[None, :] * scale_to
        return np.log1p(tf).astype(np.float32)


def _randomized_svd(X, n_components: int, random_state: int = 0):
    """Randomized SVD returning U, s, Vt with largest singular values first."""
    n_components = int(min(n_components, min(X.shape) - 1))
    U, s, Vt = sklearn.utils.extmath.randomized_svd(
        X, n_components=n_components,
        random_state=random_state, n_iter="auto",
    )
    return U, s, Vt


def _project(X_new, V, s):
    """Project new cells into an existing SVD space: X_new @ V @ diag(1/s)."""
    # V is (n_vars, k); s is (k,)
    if sp.issparse(X_new):
        proj = X_new.dot(V)
    else:
        proj = X_new @ V
    s_safe = np.where(s > 0, s, 1.0)
    return proj / s_safe[None, :]


def _select_top_features(X_bin, n_keep: int, filter_quantile: float):
    """Rank features by total accessibility and keep the top n_keep, after
    trimming features above the upper `filter_quantile`."""
    totals = _col_sums(X_bin)
    n_feat = totals.shape[0]

    # Trim the top quantile (ArchR: filterQuantile = 0.995 -> drop top 0.5%)
    upper = np.quantile(totals[totals > 0], filter_quantile) if (totals > 0).any() else np.inf
    mask = (totals > 0) & (totals <= upper)

    idx = np.where(mask)[0]
    ranked = idx[np.argsort(-totals[idx])]
    keep = ranked[:n_keep]

    out = np.zeros(n_feat, dtype=bool)
    out[keep] = True
    return out


def _scale_dims(U: np.ndarray) -> np.ndarray:
    """Z-score each SVD component (column-wise)."""
    out = U - U.mean(axis=0, keepdims=True)
    sd = out.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return out / sd


def _correlate_with_depth(U: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Pearson correlation between each SVD component and log10(depth)."""
    x = np.log10(np.asarray(depth, dtype=np.float64) + 1.0)
    x = x - x.mean()
    x_sd = x.std()
    if x_sd == 0:
        return np.zeros(U.shape[1])
    cors = np.zeros(U.shape[1])
    for i in range(U.shape[1]):
        y = U[:, i] - U[:, i].mean()
        y_sd = y.std()
        if y_sd == 0:
            cors[i] = 0.0
        else:
            cors[i] = float(np.dot(x, y) / (x_sd * y_sd * len(x)))
    return cors


def _variable_features_from_clusters(
    X_bin, clusters: np.ndarray, var_features: int, scale_to: float = 10000.0,
) -> np.ndarray:
    """Top-N variable features across cluster pseudobulks.

    Mirrors ArchR's ``.getTopFeatures`` when ``selection == "var"``: take the
    **mean** binary accessibility per cluster (= fraction of cells with the
    feature open), log2-transform with a pseudocount of 1, then pick the
    ``var_features`` rows with the highest across-cluster variance.
    ``scale_to`` is kept only for API parity – it is not applied to the mean.

    Falls back to "most accessible features" if every column is constant
    across clusters (degenerate case on tiny data).
    """
    labels = np.asarray(clusters)
    uniq, idx, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n_feat = X_bin.shape[1]
    pb = np.zeros((len(uniq), n_feat), dtype=np.float32)

    X_csr = X_bin.tocsr() if sp.issparse(X_bin) else X_bin

    for i, c in enumerate(uniq):
        sel = np.where(labels == c)[0]
        if sp.issparse(X_csr):
            sub = X_csr[sel]
            sums = np.asarray(sub.sum(axis=0)).ravel()
        else:
            sub = X_csr[sel]
            sums = sub.sum(axis=0)
        # ArchR-style: per-cluster MEAN (fraction of cells with peak open).
        n_i = max(int(counts[i]), 1)
        pb[i] = np.log2(sums / n_i + 1.0)

    var = pb.var(axis=0, ddof=1)
    n_keep = int(min(var_features, (var > 0).sum()))
    if n_keep == 0:
        totals = _col_sums(X_bin)
        keep = np.argsort(-totals)[: min(var_features, n_feat)]
    else:
        keep = np.argsort(-var)[:n_keep]
    out = np.zeros(n_feat, dtype=bool)
    out[keep] = True
    return out


# ---------------------------------------------------------------------------
# Internal clustering helper
# ---------------------------------------------------------------------------

def _knn_leiden(
    emb: np.ndarray, n_neighbors: int, resolution: float, seed: int,
) -> np.ndarray:
    """Build a cosine kNN graph on `emb` and run Leiden."""
    import scanpy as sc
    from anndata import AnnData as _A

    # Tiny adata with emb as .obsm["X_lsi"]
    tmp = _A(np.zeros((emb.shape[0], 1), dtype=np.float32))
    tmp.obsm["X_emb"] = emb.astype(np.float32)
    sc.pp.neighbors(
        tmp, use_rep="X_emb",
        n_neighbors=int(n_neighbors), metric="cosine", random_state=seed,
    )
    sc.tl.leiden(
        tmp, resolution=float(resolution),
        random_state=seed, key_added="leiden",
        flavor="igraph", directed=False, n_iterations=2,
    )
    return tmp.obs["leiden"].to_numpy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def iterative_lsi(
    adata: AnnData,
    n_components: int = 30,
    iterations: int = 2,
    var_features: int = 25000,
    total_features: int = 500000,
    filter_quantile: float = 0.995,
    resolution: Union[float, Sequence[float]] = 0.2,
    n_neighbors: int = 20,
    sample_cells_pre: Optional[int] = 10000,
    sample_cells_final: Optional[int] = None,
    binarize: bool = True,
    scale_to: float = 10000.0,
    scale_dims: bool = True,
    cor_cut_off: float = 0.75,
    depth_col: Optional[str] = None,
    seed: int = 0,
    key_added: str = "X_iterative_lsi",
    layer: Optional[str] = None,
    copy: bool = False,
    verbose: bool = True,
) -> Optional[AnnData]:
    """Perform ArchR-style iterative LSI on a cell-by-feature matrix.

    Parameters
    ----------
    adata
        Cell-by-peak or cell-by-tile AnnData.
    n_components
        Number of singular vectors to retain *before* depth filtering.
    iterations
        Number of LSI rounds. Must be >= 1. Typical value is 2 or 3.
    var_features
        Number of top-variable features to keep after the first round.
    total_features
        Number of features retained by the initial top-accessibility filter.
    filter_quantile
        Upper accessibility quantile to clip (mirrors ArchR filterQuantile).
    resolution
        Leiden resolution. Scalar, or a sequence of length ``iterations - 1``
        giving the resolution for each intermediate clustering step.
    n_neighbors
        k for the kNN graph used during intermediate clustering.
    sample_cells_pre
        If set and the dataset has more cells than this, run the first LSI on
        a random subsample; remaining cells are projected into that SVD space
        for clustering purposes only. Pass ``None`` to use all cells.
    sample_cells_final
        Same idea for the final SVD (rarely needed).
    binarize
        Binarise the matrix before LSI (ATAC default).
    scale_to
        Multiplier inside log1p (ArchR uses 10000).
    scale_dims
        Z-score each SVD component after fitting.
    cor_cut_off
        Drop components whose absolute correlation with log10(depth) exceeds
        this value. Set to 1.0 to keep all components.
    depth_col
        Column in ``adata.obs`` giving per-cell depth. If ``None``, row sums
        of the input matrix are used.
    seed
        Seed for SVD and Leiden.
    key_added
        Base key for ``.obsm``, ``.varm`` (``key_added + '_loadings'``) and
        ``.uns`` outputs.
    layer
        Optional ``adata.layers[layer]`` to use instead of ``adata.X``.
    copy
        If True, work on and return a copy.
    verbose
        Print progress.

    Examples
    --------
    >>> import epione as ep
    >>> # adata: cells x tiles (or peaks); binary or counts
    >>> ep.tl.iterative_lsi(
    ...     adata,
    ...     iterations=2,
    ...     var_features=25000,
    ...     depth_col="n_fragment",
    ...     seed=1,
    ... )
    >>> adata.obsm["X_iterative_lsi"].shape
    (n_obs, n_components_kept)

    See also
    --------
    epione.tl.lsi : a single-round LSI (no iterative feature refinement).
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    if copy:
        adata = adata.copy()

    # Resolve source matrix
    X_full = adata.layers[layer] if layer is not None else adata.X
    n_obs, n_vars = X_full.shape

    # 1. Binarise if requested
    X_bin = _binarize(X_full) if binarize else X_full
    if sp.issparse(X_bin):
        X_bin = X_bin.tocsr()

    # 2. Initial top-accessibility feature set
    features = _select_top_features(X_bin, n_keep=total_features, filter_quantile=filter_quantile)
    if verbose:
        console.info(f"[iterative_lsi] Initial feature set: {int(features.sum()):,} / {n_vars:,}")

    # Per-cell depth for correlation filtering
    if depth_col is not None and depth_col in adata.obs.columns:
        depth_all = adata.obs[depth_col].to_numpy()
    else:
        depth_all = _row_sums(X_bin)

    # Normalise resolution into a list of length (iterations - 1)
    if iterations == 1:
        resolutions = []
    elif isinstance(resolution, (int, float, np.integer, np.floating)):
        resolutions = [float(resolution)] * (iterations - 1)
    else:
        resolutions = [float(r) for r in resolution]
        if len(resolutions) < iterations - 1:
            resolutions = resolutions + [resolutions[-1]] * (iterations - 1 - len(resolutions))
        resolutions = resolutions[: iterations - 1]

    rng = np.random.default_rng(seed)
    iter_info = []

    # Iterative rounds -----------------------------------------------------
    for it in range(iterations):
        is_final = it == iterations - 1

        # Choose cells for this round's SVD
        if is_final and sample_cells_final and n_obs > sample_cells_final:
            fit_idx = rng.choice(n_obs, size=sample_cells_final, replace=False)
        elif (not is_final) and sample_cells_pre and n_obs > sample_cells_pre:
            fit_idx = rng.choice(n_obs, size=sample_cells_pre, replace=False)
        else:
            fit_idx = np.arange(n_obs)

        # Subset to current feature mask
        feat_idx = np.where(features)[0]
        if sp.issparse(X_bin):
            X_sub = X_bin[fit_idx][:, feat_idx]
        else:
            X_sub = X_bin[np.ix_(fit_idx, feat_idx)]

        if verbose:
            console.info(
                f"[iterative_lsi] Iter {it+1}/{iterations} | "
                f"fit on {len(fit_idx):,} cells x {len(feat_idx):,} features"
            )

        # TF-logIDF + SVD
        X_norm = _tf_logidf(X_sub, scale_to=scale_to)
        U, s, Vt = _randomized_svd(X_norm, n_components=n_components, random_state=seed + it)

        # Depth-correlation filter
        depth_fit = depth_all[fit_idx]
        depth_cor = _correlate_with_depth(U, depth_fit)
        kept_dims = np.where(np.abs(depth_cor) <= cor_cut_off)[0]
        if kept_dims.size == 0:
            # Keep everything as a safety net
            kept_dims = np.arange(U.shape[1])

        # Project any cells not used for fitting into the same SVD space for
        # clustering / final embedding purposes.
        if len(fit_idx) == n_obs:
            U_full = U
        else:
            if sp.issparse(X_bin):
                X_rest = X_bin[:, feat_idx]
            else:
                X_rest = X_bin[:, feat_idx]
            X_rest_norm = _tf_logidf(X_rest, scale_to=scale_to)
            U_full = _project(X_rest_norm, Vt.T, s)
            # Overwrite the rows we actually fitted with the exact values
            U_full[fit_idx] = U

        emb = U_full[:, kept_dims]
        if scale_dims:
            emb = _scale_dims(emb)

        iter_record = {
            "iteration": it + 1,
            "n_cells_fit": int(len(fit_idx)),
            "n_features": int(features.sum()),
            "kept_dims": kept_dims.tolist(),
            "depth_cor": depth_cor.tolist(),
        }

        if not is_final:
            res = resolutions[it]
            clusters = _knn_leiden(emb, n_neighbors=n_neighbors, resolution=res, seed=seed + it)
            iter_record["resolution"] = res
            iter_record["n_clusters"] = int(len(np.unique(clusters)))

            # Variable-feature selection for the next round
            features = _variable_features_from_clusters(
                X_bin, clusters, var_features=var_features, scale_to=scale_to,
            )
            if verbose:
                console.info(
                    f"[iterative_lsi]   -> {iter_record['n_clusters']} clusters; "
                    f"selected {int(features.sum()):,} variable features for next round"
                )

        iter_info.append(iter_record)

        if is_final:
            # Store the final embedding and loadings
            adata.obsm[key_added] = emb.astype(np.float32)

            loadings = np.zeros((n_vars, emb.shape[1]), dtype=np.float32)
            loadings[feat_idx] = Vt[kept_dims].T.astype(np.float32)
            adata.varm[key_added + "_loadings"] = loadings

            # Flatten per-iteration history into array-of-arrays for h5ad
            # friendliness (nested dicts with variable-length lists don't
            # serialise cleanly).
            iter_summary = {
                "iteration": np.asarray(
                    [r["iteration"] for r in iter_info], dtype=np.int32),
                "n_cells_fit": np.asarray(
                    [r["n_cells_fit"] for r in iter_info], dtype=np.int64),
                "n_features": np.asarray(
                    [r["n_features"] for r in iter_info], dtype=np.int64),
                "n_clusters": np.asarray(
                    [r.get("n_clusters", 0) for r in iter_info], dtype=np.int32),
                "resolution": np.asarray(
                    [r.get("resolution", np.nan) for r in iter_info],
                    dtype=np.float32),
            }

            adata.uns[key_added] = {
                "stdev": s.astype(np.float32),
                "features_final": features.astype(np.int8),
                "depth_cor": depth_cor.astype(np.float32),
                "kept_dims": kept_dims.astype(np.int32),
                "iterations": iter_summary,
                "params": {
                    k: (list(v) if isinstance(v, (list, tuple))
                        else ("" if v is None else v))
                    for k, v in dict(
                        n_components=n_components,
                        iterations=iterations,
                        var_features=var_features,
                        total_features=total_features,
                        filter_quantile=filter_quantile,
                        resolution=list(resolutions),
                        n_neighbors=n_neighbors,
                        sample_cells_pre=sample_cells_pre,
                        sample_cells_final=sample_cells_final,
                        binarize=binarize,
                        scale_to=scale_to,
                        scale_dims=scale_dims,
                        cor_cut_off=cor_cut_off,
                        depth_col=depth_col,
                        seed=seed,
                    ).items()
                },
            }
            if verbose:
                console.info(
                    f"[iterative_lsi] Done. Stored embedding "
                    f"({emb.shape[0]:,} x {emb.shape[1]}) in adata.obsm['{key_added}']"
                )

    return adata if copy else None
