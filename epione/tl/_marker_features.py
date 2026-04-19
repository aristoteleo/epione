"""
Group-specific feature discovery with bias-matched background.

Ports ArchR's :func:`getMarkerFeatures`. Given a cells × features AnnData
(peak matrix, gene-score matrix, gene-expression matrix, ...), for every
value in ``adata.obs[group_by]`` compute per-feature

- Mean in the group (``mean_fg``)
- Mean in a *matched* background (``mean_bg``)
- ``log2_fc = log2((mean_fg + eps) / (mean_bg + eps))``
- Wilcoxon rank-sum p-value (two-sided), BH-corrected across features
- Also reports "Pct.1" / "Pct.2" (fraction of cells with value > 0)

Bias-matched background
-----------------------
ArchR's key trick for ATAC: each foreground cell is paired with a similar
"biased" background cell (same depth, similar TSS / GC). We reproduce this:

- Standardise the ``bias_vars`` columns of ``adata.obs`` across *all* cells.
- For each foreground cell, pick the nearest k background cells (Euclidean
  in standardised space, without replacement if ``replace=False``).
- Concatenate the matched bg cells into a pooled background.

If ``bias_vars`` is None, the full complement is used as background.

Speed
-----
Wilcoxon test is vectorised per group: we rank each column of the combined
``(n_fg + n_bg) × n_features`` matrix once and derive U / z / p in closed
form. Sparse input matrices are processed column-chunked so peak count can
be large (500k+).
"""
from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as ss
from anndata import AnnData

from ..utils import console


# ---------------------------------------------------------------------------
# Bias matching
# ---------------------------------------------------------------------------

def _standardise(arr: np.ndarray) -> np.ndarray:
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (arr - mu) / sd


def _match_background(
    fg_idx: np.ndarray,
    bg_idx: np.ndarray,
    bias: np.ndarray,          # (n_cells, n_bias) float
    k: int = 1,
    replace: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Return indices (subset of ``bg_idx``) matched to ``fg_idx`` in bias space."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    if bias is None or len(bg_idx) == 0:
        return bg_idx

    bg_bias = bias[bg_idx]
    fg_bias = bias[fg_idx]
    k_use = max(1, int(k))
    nn = NearestNeighbors(n_neighbors=min(k_use, len(bg_idx))).fit(bg_bias)
    _, knn = nn.kneighbors(fg_bias, return_distance=True)

    if replace:
        picked = knn.ravel()
    else:
        # Greedy match: iterate fg cells, pick first unused neighbour.
        used = set()
        picked = []
        for row in knn:
            for j in row:
                if j not in used:
                    used.add(j)
                    picked.append(j)
                    break
        picked = np.array(picked, dtype=np.int64)
    matched_rows = bg_idx[picked]
    if len(matched_rows) == 0:
        matched_rows = bg_idx
    return matched_rows


# ---------------------------------------------------------------------------
# Vectorised Wilcoxon on a sparse/dense cell-by-feature matrix
# ---------------------------------------------------------------------------

def _wilcoxon_group_vs_bg(
    X_fg,                      # (n_fg, n_features)
    X_bg,                      # (n_bg, n_features)
    eps: float = 1e-6,
    chunk: int = 5000,
):
    """Vectorised two-sided Wilcoxon rank-sum per feature with tie correction.

    Returns a dict of arrays of length ``n_features``: mean_fg, mean_bg,
    pct_fg, pct_bg, log2fc, U, z, p_value.
    """
    n_fg = X_fg.shape[0]
    n_bg = X_bg.shape[0]
    n_feat = X_fg.shape[1]
    if n_feat != X_bg.shape[1]:
        raise ValueError("fg and bg have different n_features")

    out = {
        "mean_fg": np.zeros(n_feat, dtype=np.float64),
        "mean_bg": np.zeros(n_feat, dtype=np.float64),
        "pct_fg":  np.zeros(n_feat, dtype=np.float64),
        "pct_bg":  np.zeros(n_feat, dtype=np.float64),
        "U":       np.zeros(n_feat, dtype=np.float64),
        "z":       np.zeros(n_feat, dtype=np.float64),
        "p_value": np.ones(n_feat,  dtype=np.float64),
        "log2_fc": np.zeros(n_feat, dtype=np.float64),
    }

    N = n_fg + n_bg
    # Precompute constants
    mean_U = n_fg * n_bg / 2.0
    base_var = n_fg * n_bg * (N + 1) / 12.0

    for s in range(0, n_feat, chunk):
        e = min(s + chunk, n_feat)
        cols = slice(s, e)
        A = X_fg[:, cols]
        B = X_bg[:, cols]
        if sp.issparse(A):
            A_d = A.toarray()
        else:
            A_d = np.asarray(A, dtype=np.float32)
        if sp.issparse(B):
            B_d = B.toarray()
        else:
            B_d = np.asarray(B, dtype=np.float32)

        # Stack vertically (fg on top)
        combined = np.vstack([A_d, B_d])

        # Per-column rank (ties averaged). scipy.stats.rankdata is column-wise
        # with axis=0 and handles ties.
        ranks = ss.rankdata(combined, axis=0, method="average")

        r_fg = ranks[:n_fg].sum(axis=0)
        U = r_fg - n_fg * (n_fg + 1) / 2.0

        # Tie correction per column: T = sum(t^3 - t) for tied groups
        # Use approximate via (N^3 - sum of tied terms)/12
        # Fast implementation: compute ties per column
        T = np.zeros(e - s, dtype=np.float64)
        # Column-wise tie groups — identify runs of equal values after sort
        sorted_c = np.sort(combined, axis=0)
        # Count consecutive duplicates per column
        diffs = np.diff(sorted_c, axis=0)
        eq = (diffs == 0)
        for j in range(sorted_c.shape[1]):
            # Walk column j; count tie-group sizes
            lengths = []
            run = 1
            col = eq[:, j]
            for v in col:
                if v:
                    run += 1
                else:
                    if run > 1:
                        lengths.append(run)
                    run = 1
            if run > 1:
                lengths.append(run)
            if lengths:
                L = np.asarray(lengths, dtype=np.float64)
                T[j] = ((L ** 3 - L).sum())

        var_U = base_var - (n_fg * n_bg * T) / (12.0 * N * (N - 1))
        var_U = np.clip(var_U, 1e-12, None)
        z = (U - mean_U) / np.sqrt(var_U)
        p = 2.0 * ss.norm.sf(np.abs(z))

        mean_fg = A_d.mean(axis=0)
        mean_bg = B_d.mean(axis=0)
        pct_fg = (A_d > 0).mean(axis=0)
        pct_bg = (B_d > 0).mean(axis=0)
        log2fc = np.log2((mean_fg + eps) / (mean_bg + eps))

        out["mean_fg"][cols] = mean_fg
        out["mean_bg"][cols] = mean_bg
        out["pct_fg"][cols]  = pct_fg
        out["pct_bg"][cols]  = pct_bg
        out["U"][cols]       = U
        out["z"][cols]       = z
        out["p_value"][cols] = p
        out["log2_fc"][cols] = log2fc

    return out


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.clip(q, 0, 1)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_marker_features(
    adata: AnnData,
    *,
    group_by: str,
    groups: Optional[Sequence[str]] = None,
    bias_vars: Optional[Sequence[str]] = None,
    k_match: int = 1,
    replace: bool = False,
    max_cells_per_group: Optional[int] = 2000,
    layer: Optional[str] = None,
    test: Literal["wilcoxon"] = "wilcoxon",
    key_added: str = "markers",
    eps: float = 1e-6,
    seed: int = 0,
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per-group marker features with bias-matched background (ArchR style).

    Parameters
    ----------
    adata
        Cells × features AnnData. Typically a peak matrix, gene-score matrix,
        or gene-expression matrix.
    group_by
        Column in ``adata.obs`` giving group labels (e.g. Leiden cluster).
    groups
        Subset of group labels to process. If None, all groups are tested.
    bias_vars
        Column names in ``adata.obs`` used to match background cells to
        foreground cells (e.g. ``['n_fragment', 'TSSEnrichment']``). If None,
        all non-group cells are pooled as background without matching.
    k_match
        Number of background neighbours per foreground cell. ``k_match=1,
        replace=False`` is "1-to-1 greedy matching" (ArchR default behaviour
        in spirit).
    replace
        If True, background cells can be re-used across foreground cells.
    max_cells_per_group
        If > 0, subsample the foreground group to this many cells before
        matching. Keeps runtime bounded on huge groups. Background is capped
        to ``k_match × len(foreground)`` cells after matching anyway.
    layer
        Use ``adata.layers[layer]`` instead of ``adata.X``.
    test
        Only ``'wilcoxon'`` is implemented (rank-sum with tie correction,
        two-sided).
    key_added
        Result table is stored in ``adata.uns[key_added]``.
    eps
        Small pseudocount for log2 fold-change.
    seed
        RNG seed for subsampling and tie-breaking.

    Returns
    -------
    pandas.DataFrame with columns
        ``group, feature, mean_fg, mean_bg, pct_fg, pct_bg, log2_fc,
        U, z, p_value, fdr, n_fg, n_bg``.
    """
    if test != "wilcoxon":
        raise NotImplementedError("only test='wilcoxon' is implemented")

    if group_by not in adata.obs.columns:
        raise KeyError(f"adata.obs[{group_by!r}] not found")

    rng = np.random.default_rng(seed)
    labels = adata.obs[group_by].astype("category")
    if groups is None:
        groups = list(labels.cat.categories)
    groups = [str(g) for g in groups]

    # Resolve feature matrix
    X = adata.layers[layer] if layer is not None else adata.X
    features = np.asarray(adata.var_names)
    n_feat = X.shape[1]

    # Resolve bias matrix
    bias = None
    if bias_vars is not None:
        missing = [v for v in bias_vars if v not in adata.obs.columns]
        if missing:
            raise KeyError(f"bias_vars not in adata.obs: {missing}")
        bias_raw = adata.obs[list(bias_vars)].to_numpy(dtype=np.float64)
        bias = _standardise(bias_raw)

    all_results = []
    for g in groups:
        fg_idx = np.where(labels.values == g)[0]
        bg_idx = np.where(labels.values != g)[0]
        if fg_idx.size == 0 or bg_idx.size == 0:
            console.warn(f"[find_marker_features] skipping group {g!r} "
                         f"(fg={fg_idx.size}, bg={bg_idx.size})")
            continue
        # Subsample fg if too large
        if max_cells_per_group and fg_idx.size > max_cells_per_group:
            fg_idx = rng.choice(fg_idx, size=max_cells_per_group, replace=False)

        # Bias-match background
        if bias is not None:
            bg_pick = _match_background(
                fg_idx, bg_idx, bias, k=k_match, replace=replace, seed=seed,
            )
        else:
            bg_pick = bg_idx
        # Cap to k_match × |fg| for consistency (when no bias matching was done)
        target_bg = k_match * len(fg_idx)
        if len(bg_pick) > target_bg:
            bg_pick = rng.choice(bg_pick, size=target_bg, replace=False)

        if verbose:
            console.info(
                f"[find_marker_features] group={g!r} | fg={len(fg_idx):,} cells | "
                f"bg={len(bg_pick):,} cells (matched={'yes' if bias is not None else 'no'})"
            )

        X_fg = X[fg_idx]
        X_bg = X[bg_pick]
        stats = _wilcoxon_group_vs_bg(X_fg, X_bg, eps=eps)
        df = pd.DataFrame({
            "group":   g,
            "feature": features,
            "mean_fg": stats["mean_fg"].astype(np.float32),
            "mean_bg": stats["mean_bg"].astype(np.float32),
            "pct_fg":  stats["pct_fg"].astype(np.float32),
            "pct_bg":  stats["pct_bg"].astype(np.float32),
            "log2_fc": stats["log2_fc"].astype(np.float32),
            "U":       stats["U"].astype(np.float32),
            "z":       stats["z"].astype(np.float32),
            "p_value": stats["p_value"].astype(np.float32),
        })
        df["fdr"] = _bh_fdr(stats["p_value"]).astype(np.float32)
        df["n_fg"] = len(fg_idx)
        df["n_bg"] = len(bg_pick)
        all_results.append(df)

    out = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    if inplace:
        adata.uns[key_added] = out
        adata.uns[key_added + "_params"] = {
            "group_by": group_by,
            "bias_vars": list(bias_vars) if bias_vars else [],
            "k_match": int(k_match),
            "replace": bool(replace),
            "max_cells_per_group": int(max_cells_per_group) if max_cells_per_group else 0,
            "test": test,
            "eps": float(eps),
            "seed": int(seed),
        }

    return out
