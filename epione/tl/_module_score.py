"""Module scores and confusion matrices (ports of ArchR ``addModuleScore`` and
``confusionMatrix``; ArchR is MIT-licensed, Granja et al. 2021)."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import sparse

from ..core._r_random import RRandom

__all__ = ["add_module_score", "confusion_matrix"]


def confusion_matrix(i: Sequence, j: Sequence) -> pd.DataFrame:
    """Cross-tabulate two label vectors (ArchR ``confusionMatrix``).

    Rows are the unique values of ``i`` and columns the unique values of ``j``,
    both in order of first appearance (R ``unique``), so the result is
    cell-for-cell identical to ArchR's sparse matrix.

    Arguments:
        i: labels for the rows (e.g. ``adata.obs['Clusters']``).
        j: labels for the columns, same length as ``i``.

    Returns:
        ``pandas.DataFrame`` of integer counts.
    """
    i = np.asarray(i, dtype=object)
    j = np.asarray(j, dtype=object)
    if i.shape != j.shape:
        raise ValueError("i and j must have the same length")
    ci, ui = pd.factorize(i, sort=False, use_na_sentinel=False)
    cj, uj = pd.factorize(j, sort=False, use_na_sentinel=False)
    m = sparse.coo_matrix(
        (np.ones(len(i), dtype=np.int64), (ci, cj)), shape=(len(ui), len(uj))
    ).toarray()
    return pd.DataFrame(m, index=pd.Index(ui), columns=pd.Index(uj))


def _r_level_sort(values):
    # R factor() levels: sort(unique()); approximate en_US collation for ties in case.
    return sorted(values, key=lambda s: (s.casefold(), s.swapcase()))


def _sequential_feature_sums(X) -> np.ndarray:
    """Per-feature (column) sums accumulated cell by cell, like ArchR's arrow rowSums."""
    if sparse.issparse(X):
        Xt = sparse.csr_matrix(X.T)
        Xt.sort_indices()
        return np.asarray(Xt @ np.ones(Xt.shape[1], dtype=np.float64)).ravel()
    X = np.asarray(X, dtype=np.float64)
    acc = np.zeros(X.shape[1], dtype=np.float64)
    for row in X:
        acc += row
    return acc


def _feature_sums(adata, X, sample_key: Optional[str]) -> np.ndarray:
    if sample_key is None:
        return _sequential_feature_sums(X)
    samples = pd.Series(np.asarray(adata.obs[sample_key])).astype(str)
    total = None
    for s in pd.unique(samples):
        part = _sequential_feature_sums(X[np.where(samples.values == s)[0]])
        total = part if total is None else total + part
    return total


def _resolve_features(var_names: pd.Index, feats: Sequence[str]) -> list:
    lower = pd.Index(var_names.astype(str).str.lower())
    first = pd.Series(np.arange(len(lower)), index=lower)
    first = first[~first.index.duplicated(keep="first")]
    out = []
    for f in feats:
        key = str(f).lower()
        if key not in first.index:
            raise KeyError(f"FeatureName ({f}) does not exist in adata.var_names!")
        out.append(int(first[key]))
    return out


def add_module_score(
    adata,
    features: Union[Mapping[str, Sequence[str]], Sequence[Sequence[str]], Sequence[str]],
    *,
    name: str = "Module",
    n_bin: int = 25,
    n_bgd: int = 100,
    seed: Optional[int] = 1,
    layer: Optional[str] = None,
    sample_key: Optional[str] = None,
    feature_sums: Optional[np.ndarray] = None,
    background: Optional[Mapping[str, Sequence]] = None,
    archr_bin_indexing: bool = True,
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    r"""Signal-matched module scores (ArchR ``addModuleScore``).

    For every module, all features are ranked by their total signal and split
    into ``n_bin`` equal-count bins (zero-signal features form bin 0). For each
    module feature, ``n_bgd`` background features are drawn from its bin; the
    score of a cell is ``mean(module features) - mean(background features)``.

    Sampling uses an exact emulation of R's Mersenne-Twister so that
    ``seed=s`` draws the same background features as ArchR after
    ``set.seed(s)``. ArchR quirks are reproduced: modules are processed in R
    ``factor`` order of their names; a feature that appears more than once in
    ``module + background`` (e.g. a module gene drawn as its own background)
    enters the means as an all-zero row after its first occurrence; and (see
    ``archr_bin_indexing``) the background pool is looked up by position.

    ArchR 1.0.3 itself mislabels genes when several multi-gene modules are given
    in one call and fails on one-gene modules; this function implements the
    intended per-module behaviour (identical to ArchR whenever ArchR is correct).

    Arguments:
        adata: AnnData with cells x features in ``X`` (or ``layer``), e.g. a
            gene score matrix.
        features: ``{"B": ["MS4A1", ...], ...}`` (columns ``f"{name}.{key}"``)
            or a list of feature lists (columns ``f"{name}{i}"``). Names are
            matched case-insensitively against ``adata.var_names``.
        name: prefix for the output columns.
        n_bin: number of signal bins.
        n_bgd: background features drawn per module feature.
        seed: R-compatible seed (``set.seed(seed)``); ``None`` draws a random one.
        layer: use ``adata.layers[layer]`` instead of ``X``.
        sample_key: ``obs`` column with the sample of each cell. Feature totals are
            then accumulated per sample (in order of appearance) and added, which
            reproduces ArchR's per-Arrow ``rowSums`` bit for bit; without it the
            totals can differ in the last bits and re-order near-tied features.
        feature_sums: precomputed per-feature totals (overrides ``sample_key``).
        background: ``{column_name: [feature names or indices]}`` to use instead
            of random draws (one list per module, concatenated over its features).
        archr_bin_indexing: ArchR 1.0.3 selects each feature's background pool
            with ``binList[bin]`` (positional). When zero-signal features exist
            (bin 0) this draws from the next-lower signal bin. ``True`` (default)
            reproduces ArchR exactly; ``False`` draws from the feature's own bin.
        inplace: write the scores to ``adata.obs`` and parameters to
            ``adata.uns['module_score']``.
        verbose: print progress.

    Returns:
        DataFrame of scores (cells x modules).
    """
    X = adata.layers[layer] if layer is not None else adata.X
    if sparse.issparse(X):
        X = sparse.csr_matrix(X)
    var_names = pd.Index(adata.var_names)
    n_feat = len(var_names)

    if isinstance(features, Mapping):
        modules = {f"{name}.{k}": list(v) for k, v in features.items()}
    else:
        feats = list(features)
        if feats and isinstance(feats[0], str):
            feats = [feats]
        modules = {f"{name}{i + 1}": list(v) for i, v in enumerate(feats)}
    mod_idx = {m: _resolve_features(var_names, v) for m, v in modules.items()}

    if feature_sums is not None:
        sums = np.asarray(feature_sums, dtype=np.float64)
        if sums.shape != (n_feat,):
            raise ValueError("feature_sums must have one value per feature")
    else:
        sums = _feature_sums(adata, X, sample_key)

    order = np.argsort(sums, kind="stable")
    sorted_sums = sums[order]
    bins_sorted = np.zeros(n_feat, dtype=np.int64)
    nz = np.where(sorted_sums > 0)[0]
    if len(nz):
        width = -(-len(nz) // n_bin)
        bins_sorted[nz] = -(-np.arange(1, len(nz) + 1) // width)
    run_lengths = np.diff(np.flatnonzero(np.r_[True, bins_sorted[1:] != bins_sorted[:-1], True]))
    if n_bgd > run_lengths.min():
        raise ValueError(f"nBgd must be lower than {run_lengths.min()}!")
    feature_bin = np.empty(n_feat, dtype=np.int64)
    feature_bin[order] = bins_sorted
    bin_keys = [int(b) for b in np.unique(bins_sorted)]
    bin_members = {b: order[bins_sorted == b] for b in bin_keys}
    empty = np.array([], dtype=np.int64)

    def draw_pool(b: int) -> np.ndarray:
        if not archr_bin_indexing:
            return bin_members[b]
        # ArchR indexes split(Match, Bins) by position with the numeric bin, so
        # with a zero-signal bin present the pool is the next-lower bin.
        return bin_members[bin_keys[b - 1]] if 1 <= b <= len(bin_keys) else empty

    rng = RRandom(seed if seed is not None else np.random.SeedSequence().entropy % (2**31))
    scores = {}
    used_bgd = {}
    for m in _r_level_sort(list(mod_idx)):
        fgd = mod_idx[m]
        if background is not None:
            bgd = _resolve_background(var_names, background[m])
        else:
            bgd = []
            for f in fgd:
                pool = draw_pool(int(feature_bin[f]))
                if len(pool):
                    bgd.extend(rng.sample(pool, n_bgd).tolist())
        rows = np.asarray(fgd + bgd, dtype=np.int64)
        _, first_pos = np.unique(rows, return_index=True)
        keep = np.zeros(len(rows), dtype=bool)
        keep[first_pos] = True
        n_f = len(fgd)
        fg_cols = rows[:n_f][keep[:n_f]]
        bg_cols = rows[n_f:][keep[n_f:]]
        fg_sum = np.asarray(X[:, fg_cols].sum(axis=1)).ravel()
        bg_sum = np.asarray(X[:, bg_cols].sum(axis=1)).ravel()
        bg_mean = bg_sum / len(bgd) if len(bgd) else np.full(len(fg_sum), np.nan)
        scores[m] = fg_sum / n_f - bg_mean
        used_bgd[m] = [str(var_names[k]) for k in bgd]
        if verbose:
            print(f"{m}: {n_f} features, {len(bgd)} background features")

    out = pd.DataFrame({m: scores[m] for m in modules}, index=adata.obs_names)
    if inplace:
        for m in out.columns:
            adata.obs[m] = out[m].values
        adata.uns["module_score"] = {
            "params": dict(n_bin=n_bin, n_bgd=n_bgd, seed=seed, layer=layer, sample_key=sample_key,
                           archr_bin_indexing=archr_bin_indexing),
            "features": {m: [str(var_names[k]) for k in mod_idx[m]] for m in modules},
            "background": used_bgd,
        }
    return out


def _resolve_background(var_names: pd.Index, items: Sequence) -> list:
    items = list(items)
    if items and isinstance(items[0], (int, np.integer)):
        return [int(k) for k in items]
    pos = var_names.get_indexer([str(k) for k in items])
    if (pos < 0).any():
        raise KeyError("background features not found in adata.var_names")
    return pos.tolist()
