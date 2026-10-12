"""AUCell enrichment of eRegulon gene / region signatures in single cells.

Reproduces SCENIC+'s ``score_eRegulons``: every cell's features are ranked by decreasing
value with ties broken by a random permutation drawn from ``np.random.default_rng(seed)``
(one permutation per cell, in cell order), and the AUC of each signature's recovery curve is
computed over the top ``round(auc_threshold * n_features) - 1`` ranks (ctxcore ``aucs``).

Differences from upstream, all exact rewrites:

* the ranking of a matrix is computed once (:func:`rank_cells`) and reused for any number of
  signature sets (SCENIC+ re-ranks for direct and for extended eRegulons);
* only the top of each ranking is kept, as a sparse cells x features matrix;
* with unit weights the AUC of a cell is ``(k * cutoff - sum(ranks below cutoff)) / max_auc``
  (k = signature features below the cutoff), evaluated in integer arithmetic for all
  signatures at once with one sparse product. The value equals ctxcore's
  ``sum(diff(x) * cumsum(w)) / max_auc`` exactly, because every intermediate is an integer.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from ._np126 import argsort_np126

__all__ = ["CellRanking", "rank_cells", "aucell", "eregulon_signatures", "score_eregulons"]


@dataclass
class CellRanking:
    """Top-of-ranking matrix: ``top[i, f] = rank + 1`` for features ranked below ``cutoff``."""

    top: sparse.csr_matrix
    cells: np.ndarray
    features: np.ndarray
    cutoff: int


def _rank_cutoff(auc_threshold: float, n_features: int) -> int:
    cutoff = int(round(auc_threshold * n_features))
    if not (0 < cutoff <= n_features - 1):
        raise ValueError(f"auc_threshold {auc_threshold} gives an invalid rank cutoff for {n_features} features")
    return cutoff - 1


def rank_cells(X, cells: Sequence[str] | None = None, features: Sequence[str] | None = None, *,
               auc_threshold: float = 0.05, seed: int = 123, n_jobs: int = 8,
               dtype=None) -> CellRanking:
    """Rank features within each cell (descending, random tie-breaking) and keep the top.

    Parameters
    ----------
    X
        cells x features matrix (dense, scipy sparse, AnnData or DataFrame). The values are
        ranked in their own dtype; SCENIC+ ranks gene expression as float32 and imputed
        accessibility as int32, and ties are broken differently if the dtype differs.
    auc_threshold
        Fraction of the ranking used for the AUC (SCENIC+ default 0.05).
    seed
        Seed of the tie-breaking generator (SCENIC+ uses 123, fresh for every matrix).
    n_jobs
        Threads used for sorting (permutations are always drawn in cell order).
    """
    if hasattr(X, "obs_names") and hasattr(X, "X"):
        cells = np.asarray(X.obs_names) if cells is None else cells
        features = np.asarray(X.var_names) if features is None else features
        X = X.X
    elif isinstance(X, pd.DataFrame):
        cells = X.index.to_numpy() if cells is None else cells
        features = X.columns.to_numpy() if features is None else features
        X = X.to_numpy()
    n_cells, n_feat = X.shape
    if dtype is not None:
        X = X.astype(dtype)
    cutoff = _rank_cutoff(auc_threshold, n_feat)
    is_sparse = sparse.issparse(X)
    if is_sparse:
        X = sparse.csr_matrix(X)
    rng = np.random.default_rng(seed=seed)

    def row(i):
        if is_sparse:
            x = np.zeros(n_feat, dtype=X.dtype)
            sl = slice(X.indptr[i], X.indptr[i + 1])
            x[X.indices[sl]] = X.data[sl]
        else:
            x = np.asarray(X[i])
        return x

    def top_of(args):
        i, perm = args
        order = perm[argsort_np126((-row(i))[perm])]
        return order[:cutoff]

    tops = np.empty((n_cells, cutoff), dtype=np.int64)
    batch = 4 * max(1, n_jobs)
    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as ex:
        for s in range(0, n_cells, batch):
            ids = range(s, min(s + batch, n_cells))
            # permutations are drawn sequentially in cell order; sorting runs in parallel
            jobs = [(i, rng.permutation(n_feat)) for i in ids]
            for i, t in zip(ids, ex.map(top_of, jobs)):
                tops[i] = t
    data = np.tile(np.arange(1, cutoff + 1, dtype=np.int64), n_cells)
    top = sparse.csr_matrix((data, tops.ravel(), np.arange(0, n_cells * cutoff + 1, cutoff)),
                            shape=(n_cells, n_feat))
    return CellRanking(top=top, cells=np.asarray(cells), features=np.asarray(features), cutoff=cutoff)


def aucell(ranking: CellRanking, signatures: Mapping[str, Sequence[str]], min_fraction: float = 0.8) -> pd.DataFrame:
    """AUC of each signature in each cell (cells x signatures).

    A signature with fewer than ``min_fraction`` of its features present in the ranking gets
    AUC 0, as in ctxcore.
    """
    names = list(signatures.keys())
    fpos = pd.Index(ranking.features)
    rows, cols, ok = [], [], np.zeros(len(names), dtype=bool)
    n_present = np.zeros(len(names), dtype=np.int64)
    for j, name in enumerate(names):
        genes = list(dict.fromkeys(signatures[name]))
        idx = fpos.get_indexer(genes)
        idx = np.unique(idx[idx >= 0])
        n_present[j] = len(idx)
        ok[j] = len(idx) > 0 and float(len(idx)) / float(len(genes)) >= min_fraction
        rows.append(idx)
        cols.append(np.full(len(idx), j))
    S = sparse.csr_matrix((np.ones(sum(len(r) for r in rows), dtype=np.int64),
                           (np.concatenate(rows) if rows else np.empty(0, int),
                            np.concatenate(cols) if cols else np.empty(0, int))),
                          shape=(len(ranking.features), len(names)))
    top = ranking.top
    ind = top.copy()
    ind.data = np.ones_like(ind.data)
    k = (ind @ S).toarray()                       # signature features ranked below the cutoff
    rank_sum = (top @ S).toarray() - k            # sum of their 0-based ranks
    num = k * ranking.cutoff - rank_sum           # integer: sum(diff(x) * cumsum(1))
    max_auc = ((ranking.cutoff + 1) * n_present.astype(np.float64)).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = num.astype(np.float64) / max_auc[None, :]
    auc[:, ~ok] = 0.0
    return pd.DataFrame(auc, index=pd.Index(ranking.cells, name="Cell"), columns=names)


def eregulon_signatures(eregulons: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Gene- and region-based signatures keyed by ``Gene_signature_name`` / ``Region_signature_name``."""
    genes = {k: list(dict.fromkeys(v)) for k, v in eregulons.groupby("Gene_signature_name")["Gene"]}
    regions = {k: list(dict.fromkeys(v)) for k, v in eregulons.groupby("Region_signature_name")["Region"]}
    return {"Gene_based": genes, "Region_based": regions}


def score_eregulons(eregulons: pd.DataFrame, gex_ranking: CellRanking, acc_ranking: CellRanking) -> dict[str, pd.DataFrame]:
    """Gene-based (on expression) and region-based (on accessibility) eRegulon AUCs."""
    sig = eregulon_signatures(eregulons)
    return {"Gene_based": aucell(gex_ranking, sig["Gene_based"]),
            "Region_based": aucell(acc_ranking, sig["Region_based"])}
