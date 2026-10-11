"""SCENIC+ TF-to-gene and region-to-gene links.

Ports of ``scenicplus.TF_to_gene.calculate_TFs_to_genes_relationships`` (GRNBoost2 /
arboreto) and ``scenicplus.enhancer_to_gene.calculate_regions_to_genes_relationships``.
The tree models run through :mod:`epione.scenic._sktree` (scikit-learn 1.3.2 semantics
in numba, no dask / arboreto / joblib process pool), genes are fitted in a thread pool,
and the tables are assembled with the same pandas operations as upstream so that rows,
row order and values match scenicplus 1.0.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

from ._sktree import gbm_importances, rf_importances

__all__ = ["tf_to_gene", "region_to_gene", "hashseed0_set_order"]

# arboreto / scenicplus regressor settings
SGBM_KWARGS = dict(learning_rate=0.01, n_estimators=5000, max_features=0.1, subsample=0.9)
GBM_KWARGS = dict(learning_rate=0.01, n_estimators=500, max_features=0.1)
RF_KWARGS = dict(n_estimators=1000, max_features="sqrt")
EARLY_STOP_WINDOW_LENGTH = 25
RHO_THRESHOLD = 0.03


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _matrix(data, var_names=None, obs_names=None, dtype=np.float32):
    """(dense cells x features array, obs names, var names) from AnnData / DataFrame / array."""
    if hasattr(data, "X") and hasattr(data, "var_names"):
        X, obs, var = data.X, list(data.obs_names), list(data.var_names)
    elif isinstance(data, pd.DataFrame):
        X, obs, var = data.to_numpy(), list(data.index), list(data.columns)
    else:
        X = data
        if var_names is None:
            raise ValueError("var_names are required when passing a matrix")
        obs = list(obs_names) if obs_names is not None else list(range(X.shape[0]))
        var = list(var_names)
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    if dtype is not None and X.dtype != dtype:
        X = X.astype(dtype)
    return X, obs, var


def _nargsort(values, ascending=True):
    """``pandas.core.sorting.nargsort`` (``DataFrame.sort_values`` on one column, no NaN)."""
    items = np.asarray(values)
    idx = np.arange(len(items))
    if not ascending:
        items = items[::-1]
        idx = idx[::-1]
    indexer = idx[items.argsort(kind="quicksort")]
    if not ascending:
        indexer = indexer[::-1]
    return indexer


def hashseed0_set_order(items: Sequence[str], intersect_with: Sequence[str] | None = None) -> list:
    """Iteration order of ``list(set(items))`` (or ``list(set(items) & set(intersect_with))``)
    in a CPython process started with ``PYTHONHASHSEED=0``.

    scenicplus derives the multiome cell order and some TF lists from Python set iteration;
    its reference runs use ``PYTHONHASHSEED=0``.  This reproduces that order exactly by
    evaluating the same expression in a child interpreter.
    """
    code = ("import json,sys\n"
            "d=json.load(sys.stdin)\n"
            "r=list(set(d['a'])&set(d['b'])) if d['b'] is not None else list(set(d['a']))\n"
            "json.dump(r,sys.stdout)\n")
    env = dict(os.environ, PYTHONHASHSEED="0")
    payload = json.dumps({"a": list(items), "b": None if intersect_with is None else list(intersect_with)})
    out = subprocess.run([sys.executable, "-c", code], input=payload, capture_output=True,
                         text=True, env=env, check=True)
    return json.loads(out.stdout)


def _pearson_pairs(X, i_idx, j_idx, block=256):
    """``np.corrcoef`` entries for column pairs (i, j) of the float matrix ``X`` (cells x genes)."""
    cols = np.unique(np.concatenate([i_idx, j_idx]))
    pos = np.full(X.shape[1], -1, dtype=np.int64)
    pos[cols] = np.arange(len(cols))
    Z = np.array(X[:, cols], dtype=np.float64, order="F")
    n = Z.shape[0]
    Z -= Z.mean(axis=0)
    var = np.einsum("ij,ij->j", Z, Z) / (n - 1)
    sd = np.sqrt(var)
    a, b = pos[i_idx], pos[j_idx]
    rho = np.empty(len(a), dtype=np.float64)
    order = np.argsort(a, kind="stable")
    a_s = a[order]
    starts = np.flatnonzero(np.r_[True, a_s[1:] != a_s[:-1]])
    ends = np.r_[starts[1:], len(a_s)]
    with np.errstate(divide="ignore", invalid="ignore"):
        for s, e in zip(starts, ends):
            k = a_s[s]
            sel = order[s:e]
            c = (Z[:, b[sel]].T @ Z[:, k]) / (n - 1)
            r = c / sd[k] / sd[b[sel]]
            rho[sel] = r
    return np.clip(rho, -1.0, 1.0)


def _run_parallel(fn, items, n_jobs):
    if n_jobs is None or n_jobs <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(fn, items))


# ---------------------------------------------------------------------------
# TF -> gene
# ---------------------------------------------------------------------------
def _add_correlation(adj: pd.DataFrame, X, gene_pos, rho_threshold):
    tf_i = np.fromiter((gene_pos[t] for t in adj["TF"]), dtype=np.int64, count=len(adj))
    tg_i = np.fromiter((gene_pos[t] for t in adj["target"]), dtype=np.int64, count=len(adj))
    rhos = _pearson_pairs(X, tf_i, tg_i) if len(adj) else np.empty(0)
    regulations = (rhos > rho_threshold).astype(int) - (rhos < -rho_threshold).astype(int)
    return pd.DataFrame({
        "TF": adj["TF"].to_numpy(),
        "target": adj["target"].to_numpy(),
        "importance": adj["importance"].to_numpy(),
        "regulation": regulations,
        "rho": rhos,
    })


def tf_to_gene(
    expression,
    tf_names: Iterable[str],
    *,
    method: Literal["GBM", "RF"] = "GBM",
    seed: int = 666,
    n_jobs: int = 1,
    rho_threshold: float = RHO_THRESHOLD,
    targets: Iterable[str] | None = None,
    self_link_order: Literal["sorted", "upstream"] = "sorted",
    var_names=None,
) -> pd.DataFrame:
    """TF-to-gene adjacencies as ``scenicplus`` computes them (GRNBoost2 + correlation).

    For every target gene a gradient-boosting model (``method="GBM"``: arboreto's SGBM,
    5000 trees max, ``subsample=0.9``, OOB early stopping with window 25; importances
    multiplied by the number of trees) or a random forest (``method="RF"``) predicts its
    expression from the TFs' expression.  Links with importance > 0 are kept, Pearson
    ``rho`` and ``regulation`` (sign at ``|rho| > rho_threshold``) are added, and every TF
    is injected as its own target with importance ``max + 1e-5``.

    Parameters
    ----------
    expression
        Cells x genes expression (raw counts in SCENIC+): AnnData, DataFrame or matrix with
        ``var_names``.  Values are used as float32, like the scenicplus MuData.
    tf_names
        Candidate regulators; the predictor columns are the genes that are TFs, in
        expression-matrix column order (as in arboreto).
    targets
        Restrict the regressions to these genes (default: all genes).
    self_link_order
        Order of the injected TF -> TF rows.  scenicplus emits them in Python set order,
        which depends on the hash seed; ``"upstream"`` reproduces the order of a
        ``PYTHONHASHSEED=0`` run, ``"sorted"`` sorts them by TF.
    n_jobs
        Threads (genes are independent; the result does not depend on ``n_jobs``).

    Returns
    -------
    DataFrame ``TF, target, importance, regulation, rho``.
    """
    X, _, genes = _matrix(expression, var_names=var_names)
    X = np.asfortranarray(X)
    tf_set = set(tf_names)
    tf_cols = np.array([i for i, g in enumerate(genes) if g in tf_set], dtype=np.int64)
    if len(tf_cols) == 0:
        raise ValueError("Intersection of gene names and tf_names is empty.")
    gene_pos = {g: i for i, g in enumerate(genes)}
    target_idx = list(range(len(genes))) if targets is None else [gene_pos[g] for g in targets if g in gene_pos]
    tf_name_arr = np.asarray(genes, dtype=object)[tf_cols]
    is_tf = np.zeros(len(genes), dtype=bool)
    is_tf[tf_cols] = True

    def fit(gi):
        cols = tf_cols[tf_cols != gi] if is_tf[gi] else tf_cols
        if len(cols) == 0:
            return None
        names = tf_name_arr[tf_cols != gi] if is_tf[gi] else tf_name_arr
        y = X[:, gi]
        if method.upper() == "GBM":
            imp, n_est = gbm_importances(X, y, cols=cols, seed=seed,
                                         early_stop_window=EARLY_STOP_WINDOW_LENGTH, **SGBM_KWARGS)
            imp = imp * n_est
        elif method.upper() == "RF":
            imp = rf_importances(X, y, cols=cols, seed=seed, **RF_KWARGS)
        else:
            raise ValueError(f"Unsupported method: {method}")
        keep = imp > 0
        if not keep.any():
            return None
        names, imp = names[keep], imp[keep]
        o = _nargsort(imp, ascending=False)
        return names[o], imp[o], genes[gi]

    res = _run_parallel(fit, target_idx, n_jobs)
    res = [r for r in res if r is not None]
    tf_col = np.concatenate([r[0] for r in res]) if res else np.empty(0, dtype=object)
    imp_col = np.concatenate([r[1] for r in res]) if res else np.empty(0)
    tg_col = np.concatenate([np.full(len(r[0]), r[2], dtype=object) for r in res]) if res else np.empty(0, dtype=object)
    o = _nargsort(imp_col, ascending=False)
    adj = pd.DataFrame({"TF": tf_col[o], "target": tg_col[o], "importance": imp_col[o]})
    adj = _add_correlation(adj, X, gene_pos, rho_threshold)

    # _inject_TF_as_its_own_target
    origin = adj.iloc[_nargsort(adj["TF"].to_numpy(dtype=object))]
    max_imp = origin.groupby("TF")["importance"].max()
    tfs = list(set(origin["TF"].to_list()))
    if self_link_order == "upstream":
        tfs = hashseed0_set_order(origin["TF"].to_list())
    else:
        tfs = sorted(tfs)
    self_adj = pd.DataFrame({"TF": tfs, "target": tfs,
                             "importance": max_imp.loc[tfs].to_numpy() + 0.00001})
    self_adj = _add_correlation(self_adj, X, gene_pos, rho_threshold)
    return pd.concat([origin, self_adj]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# region -> gene
# ---------------------------------------------------------------------------
def _spearman_columns(A, y):
    """``scipy.stats.spearmanr(A[:, j], y)[0]`` for every column (NaN for constant input)."""
    n = A.shape[0]
    out = np.full(A.shape[1], np.nan)
    if (y == y[0]).all():
        return out
    ry = rankdata(y)
    ry = ry - ry.mean()
    R = rankdata(A, axis=0)
    R -= R.mean(axis=0)
    const = (A == A[:1]).all(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = (R.T @ ry) / (n - 1)
        r = c / np.sqrt(np.einsum("ij,ij->j", R, R) / (n - 1)) / np.sqrt(ry @ ry / (n - 1))
    out[~const] = np.clip(r[~const], -1.0, 1.0)
    return out


def region_to_gene(
    expression,
    accessibility,
    search_space: pd.DataFrame,
    *,
    importance: Literal["GBM", "RF"] = "GBM",
    correlation: Literal["SR", "PR"] = "SR",
    mask_expr_dropout: bool = False,
    seed: int = 666,
    n_jobs: int = 1,
    add_distance: bool = True,
    targets: Iterable[str] | None = None,
    expression_var_names=None,
    accessibility_var_names=None,
) -> pd.DataFrame:
    """Region-to-gene links as ``scenicplus`` computes them.

    For each gene, a GBM (500 trees, ``max_features=0.1``) or random forest (1000 trees,
    ``max_features="sqrt"``) predicts expression from the accessibility of the regions in
    its search space (:func:`get_search_space`); the feature importances are the link
    strengths and the Spearman (``"SR"``) or Pearson (``"PR"``) correlation gives the sign.

    Parameters
    ----------
    expression
        Cells x genes expression (AnnData / DataFrame / matrix), float32 like scenicplus.
    accessibility
        Cells x regions accessibility with the same cells in the same order (imputed
        accessibility in SCENIC+).
    search_space
        ``Name``, ``Gene`` (and ``Distance``) table from :func:`get_search_space`.
    targets
        Only fit these genes.  The region (feature) order of each gene is still derived
        from the whole search space, exactly as scenicplus does, so the result for a gene
        does not depend on which other genes are fitted.

    Returns
    -------
    DataFrame ``target, region, importance, rho, importance_x_rho, importance_x_abs_rho``
    (+ ``Distance`` when ``add_distance``).
    """
    EXPm, _, gnames = _matrix(expression, var_names=expression_var_names)
    ACC, _, rnames = _matrix(accessibility, var_names=accessibility_var_names, dtype=None)
    if EXPm.shape[0] != ACC.shape[0]:
        raise ValueError("expression and accessibility must have the same cells")
    if len(set(gnames)) != len(gnames):
        raise ValueError("Expression matrix contains duplicate gene names")
    if len(set(rnames)) != len(rnames):
        raise ValueError("Chromatin accessibility matrix contains duplicate gene names")
    rnames_arr = np.asarray(rnames, dtype=object)
    region_pos = pd.Series(np.arange(len(rnames)), index=pd.Index(rnames_arr))
    ss = search_space[search_space["Name"].isin(rnames_arr)]
    ss = ss[ss["Gene"].isin(set(gnames))]
    # _get_acc_idx_per_gene
    rn = ss["Name"].to_numpy()
    gn = ss["Gene"].to_numpy()
    s = np.argsort(gn)
    rn, gn = rn[s], gn[s]
    uniq, first = np.unique(gn, return_index=True)
    bounds = np.r_[first, len(gn)]
    acc_idx = [region_pos.loc[rn[bounds[i]:bounds[i + 1]]].to_numpy() for i in range(len(uniq))]
    if targets is not None:
        tset = set(targets)
        sel = [i for i, g in enumerate(uniq) if g in tset]
        uniq = uniq[sel]
        acc_idx = [acc_idx[i] for i in sel]
    gpos = {g: i for i, g in enumerate(gnames)}
    exp_cols = np.array([gpos[g] for g in uniq], dtype=np.int64)

    imp_kind = importance.upper()
    if imp_kind not in ("GBM", "RF"):
        raise ValueError("Unsuported regression model")

    def fit(i):
        cols = acc_idx[i]
        y = EXPm[:, exp_cols[i]]
        A = ACC[:, cols]
        if mask_expr_dropout:
            nz = y != 0
            y, A = y[nz], A[nz]
        Af = np.asfortranarray(A, dtype=np.float32)
        if imp_kind == "GBM":
            imp, _ = gbm_importances(Af, y, seed=seed, **GBM_KWARGS)
        else:
            imp = rf_importances(Af, y, seed=seed, **RF_KWARGS)
        if correlation.upper() == "SR":
            rho = _spearman_columns(A, y)
        elif correlation.upper() == "PR":
            Ad = A.astype(np.float64)
            rho = np.array([np.corrcoef(Ad[:, j], y.astype(np.float64))[0, 1] for j in range(A.shape[1])])
        else:
            raise ValueError("Unsuported regression model")
        return imp, rho

    res = _run_parallel(fit, range(len(uniq)), n_jobs)
    target = np.concatenate([np.full(len(c), g, dtype=object) for g, c in zip(uniq, acc_idx)]) if len(uniq) else np.empty(0, dtype=object)
    region = np.concatenate([rnames_arr[c] for c in acc_idx]) if len(uniq) else np.empty(0, dtype=object)
    imp = np.concatenate([r[0] for r in res]) if res else np.empty(0)
    rho = np.concatenate([r[1] for r in res]) if res else np.empty(0)
    out = pd.DataFrame({"target": target, "region": region, "importance": imp, "rho": rho})
    out["importance_x_rho"] = out["rho"] * out["importance"]
    out["importance_x_abs_rho"] = abs(out["rho"]) * out["importance"]
    if add_distance:
        ss_rn = search_space.rename({"Name": "region", "Gene": "target"}, axis=1).copy()
        out = out.merge(ss_rn, on=["region", "target"])
    return out
