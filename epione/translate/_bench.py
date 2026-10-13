"""scib-style benchmarking of translation methods.

:func:`evaluate` scores one prediction; :func:`benchmark` scores many methods
and tasks into a tidy table; :func:`summarize` turns that into per-category
and overall scores (each metric min-max scaled across methods, with
lower-is-better metrics flipped, then averaged within its category);
:func:`plot_benchmark` draws the scib-like results table.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import _metrics as M

# metric -> (category, higher_is_better)
METRICS = {
    "ARI": ("clustering", True),
    "AMI": ("clustering", True),
    "NMI": ("clustering", True),
    "HOM": ("clustering", True),
    "PCC": ("regression", True),
    "SPCC": ("regression", True),
    "CORR": ("regression", True),
    "MSE": ("regression", False),
    "MAE": ("regression", False),
    "MMD": ("distribution", False),
    "iLISI": ("distribution", True),
    "AUROC": ("classification", True),
    "AUPRC": ("classification", True),
}
CATEGORIES = ("clustering", "regression", "distribution", "classification")


def _labels(truth, labels):
    if labels is None:
        return None
    if isinstance(labels, str):
        return np.asarray(truth.obs[labels])
    return np.asarray(labels)


def _quiet_sparse(fn):
    """Silence scipy's SparseEfficiencyWarning raised inside the metric stack."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import warnings
        from scipy.sparse import SparseEfficiencyWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseEfficiencyWarning)
            return fn(*args, **kwargs)

    return wrapper


@_quiet_sparse
def evaluate(
    pred,
    truth,
    *,
    labels: Union[str, Sequence, None] = None,
    task: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
    n_pcs: int = 50,
    binarize="median",
    auc_average: str = "global",
    mmd_max_cells: Optional[int] = 5000,
    convention: str = "standard",
    exclude: Sequence[str] = (),
    seed: int = 0,
) -> dict:
    """All metrics for one predicted matrix.

    Parameters
    ----------
    pred, truth
        Predicted and measured profiles of the same cells (AnnData or arrays).
        ``pred`` is reordered to ``truth``'s cells and features when both are
        AnnData.
    labels
        Cell-type labels (an ``obs`` column of ``truth`` or an array); needed
        for the clustering metrics.
    task
        ``"rna2atac"`` or ``"atac2rna"`` (recorded only).
    categories
        Subset of ``clustering``, ``regression``, ``distribution``,
        ``classification``.
    exclude
        Metric names to leave out (e.g. ``("MSE", "MAE")`` when methods are
        scored in different units).
    convention
        ``"standard"`` (default) or ``"sctranslation"``. The latter reproduces
        the scTranslation code for AUROC/AUPRC and CORR, which differs from
        the stated definitions (see :func:`epione.translate.metrics.auroc`).
        Every other metric is identical under both.
    """
    if hasattr(pred, "obs") and hasattr(truth, "obs"):
        pred = pred[truth.obs_names, truth.var_names]
    if categories is None:
        categories = CATEGORIES
    lab = _labels(truth, labels)
    # drop cells with a non-finite value in either matrix (scTranslation does the same)
    p_arr, t_arr = M._pair(pred, truth)
    ok = np.isfinite(p_arr).all(1) & np.isfinite(t_arr).all(1)
    if not ok.all():
        p_arr, t_arr = p_arr[ok], t_arr[ok]
        lab = None if lab is None else lab[ok]
    pred, truth = p_arr, t_arr
    out = {}
    if "clustering" in categories and lab is not None:
        out.update(M.cluster_scores(pred, lab, n_pcs=n_pcs, seed=seed))
    if "regression" in categories:
        out.update(M.regression_scores(pred, truth, convention=convention))
    if "distribution" in categories:
        out["MMD"] = M.mmd(pred, truth, n_pcs=n_pcs, max_cells=mmd_max_cells, seed=seed)
        out["iLISI"] = M.ilisi(pred, truth, n_pcs=n_pcs, seed=seed)
    if "classification" in categories:
        out["AUROC"] = M.auroc(pred, truth, binarize=binarize, average=auc_average, convention=convention)
        out["AUPRC"] = M.auprc(pred, truth, binarize=binarize, average=auc_average, convention=convention)
    return {k: v for k, v in out.items() if k not in set(exclude)}


def benchmark(
    predictions: Mapping,
    truth: Mapping,
    *,
    labels: Union[str, Sequence, None] = None,
    **kwargs,
) -> pd.DataFrame:
    """Score several methods on one or more tasks.

    Parameters
    ----------
    predictions
        ``{task: {method: pred}}``, e.g.
        ``{"rna2atac": {"scButterfly": atac_pred}, "atac2rna": {...}}``.
    truth
        ``{task: measured AnnData}`` with the same cells as the predictions,
        or ``{task: {method: measured AnnData}}`` when methods predict in
        different spaces (e.g. TF-IDF vs binary ATAC). Each method is then
        scored against its own target; pass ``exclude=("MSE", "MAE")``, since
        errors in different units can't be ranked against each other.
    labels
        Cell-type labels (an ``obs`` column name or an array).
    **kwargs
        Passed to :func:`evaluate`.

    Returns
    -------
    Tidy DataFrame with columns ``task, method, metric, category,
    higher_is_better, value``.
    """
    rows = []
    for task, by_method in predictions.items():
        if task not in truth:
            raise KeyError(f"no truth for task {task!r}")
        for method, pred in by_method.items():
            t = truth[task][method] if isinstance(truth[task], Mapping) else truth[task]
            scores = evaluate(pred, t, labels=labels, task=task, **kwargs)
            for metric, value in scores.items():
                cat, hib = METRICS[metric]
                rows.append((task, method, metric, cat, hib, value))
    return pd.DataFrame(rows, columns=["task", "method", "metric", "category", "higher_is_better", "value"])


def summarize(results: pd.DataFrame, *, weights: Optional[Mapping[str, float]] = None) -> pd.DataFrame:
    """Per-category and overall scores, scib style.

    Each metric is min-max scaled across methods within a task (lower-is-better
    metrics are flipped so 1 is best; a metric that is equal for every method
    scores 1). Category scores are the mean of their scaled metrics; the
    overall score is the weighted mean of categories (equal weights by
    default). Rows are ``(task, method)``, sorted by overall score.
    """
    df = results.copy()

    def scale(g):
        v = g["value"].astype(float)
        lo, hi = v.min(), v.max()
        if not hi > lo:
            return pd.Series(1.0, index=v.index)
        s = (v - lo) / (hi - lo)
        return s.where(g["higher_is_better"], 1 - s)

    df["scaled"] = df.groupby(["task", "metric"], group_keys=False)[["value", "higher_is_better"]].apply(scale)
    cat = df.pivot_table(index=["task", "method"], columns="category", values="scaled", aggfunc="mean")
    cats = [c for c in CATEGORIES if c in cat.columns]
    w = pd.Series({c: (weights or {}).get(c, 1.0) for c in cats})
    cat = cat[cats]
    cat["overall"] = (cat[cats] * w).sum(axis=1, min_count=1) / cat[cats].notna().mul(w).sum(axis=1)
    raw = df.pivot_table(index=["task", "method"], columns="metric", values="value")
    raw = raw[[m for m in METRICS if m in raw.columns]]
    out = pd.concat({"metric": raw, "score": cat}, axis=1)
    return out.sort_values([("score", "overall")], ascending=False).sort_index(level=0, sort_remaining=False)


def plot_benchmark(results: pd.DataFrame, task: Optional[str] = None, *, ax=None, cmap: str = "viridis"):
    """scib-like table: a circle per metric (size and colour = scaled score),
    bars for category and overall scores, one row per method.

    Returns the matplotlib Figure.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import colors

    summ = summarize(results)
    tasks = summ.index.get_level_values(0).unique()
    task = task or tasks[0]
    s = summ.loc[task].sort_values(("score", "overall"), ascending=False)
    sub = results[results["task"] == task]
    metrics = [m for m in METRICS if m in set(sub["metric"])]
    cats = [c for c in CATEGORIES if ("score", c) in s.columns]
    scaled = {}
    for m in metrics:
        v = s[("metric", m)].astype(float)
        lo, hi = v.min(), v.max()
        if not hi > lo:
            scaled[m] = pd.Series(1.0, index=v.index)
        else:
            z = (v - lo) / (hi - lo)
            scaled[m] = z if METRICS[m][1] else 1 - z

    ncol = len(metrics) + len(cats) + 1
    nrow = len(s)
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.0 + 0.75 * ncol, 0.9 + 0.55 * nrow))
    else:
        fig = ax.figure
    norm = colors.Normalize(0, 1)
    cmap_ = matplotlib.colormaps[cmap]
    for i, method in enumerate(s.index):
        y = nrow - 1 - i
        for j, m in enumerate(metrics):
            z = scaled[m].loc[method]
            val = s.loc[method, ("metric", m)]
            ax.scatter(j, y, s=60 + 340 * z, color=cmap_(norm(z)), edgecolor="none")
            ax.text(j, y - 0.38, f"{val:.3g}", ha="center", va="center", fontsize=7)
        for k, c in enumerate(cats + ["overall"]):
            x = len(metrics) + k
            z = s.loc[method, ("score", c)]
            if np.isfinite(z):
                ax.barh(y, 0.8 * z, left=x - 0.4, height=0.5, color=cmap_(norm(z)))
                ax.text(x, y - 0.38, f"{z:.2f}", ha="center", va="center", fontsize=7)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(list(s.index)[::-1])
    ax.set_xticks(range(ncol))
    ax.set_xticklabels(metrics + [c.capitalize() for c in cats] + ["Overall"], rotation=45, ha="right")
    ax.axvline(len(metrics) - 0.5, color="0.7", lw=0.8)
    ax.set_xlim(-0.6, ncol - 0.4)
    ax.set_ylim(-0.8, nrow - 0.4)
    for sp_ in ("top", "right", "left"):
        ax.spines[sp_].set_visible(False)
    ax.set_title(task)
    fig.tight_layout()
    return fig
