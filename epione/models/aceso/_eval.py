"""Evaluation metrics.

* :func:`knn_smoothed_pearson` — the scooby paper's single-cell metric: per cell, Pearson between
  log1p(predicted profile) and log1p(mean observed profile of its k nearest neighbours, k = 100,
  the cell itself included), over the bins of held-out windows. Single-cell ATAC is too sparse to
  compare against a cell's own counts.
* :func:`pseudobulk_pearson` — per group (cell type, sample, spatial domain): Pearson of log1p
  summed prediction vs log1p summed observation, and the across-group (differential) version on
  group-centred values.
* :func:`knn_transfer_accuracy` — state quality: kNN cell-type transfer from reference states to
  query states (e.g. true EpiAgent z of training cells -> predicted z of test cells).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _rowwise_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson per column of two (bins x cells) arrays."""
    A = A - A.mean(0); B = B - B.mean(0)
    den = np.sqrt((A * A).sum(0) * (B * B).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (A * B).sum(0) / den


def knn_neighbors(states: np.ndarray, k: int = 100) -> np.ndarray:
    """(cells x k) neighbour indices (cosine, self included)."""
    from sklearn.neighbors import NearestNeighbors
    S = states / np.linalg.norm(states, axis=1, keepdims=True)
    return NearestNeighbors(n_neighbors=k, metric="cosine").fit(S).kneighbors(S, return_distance=False)


def knn_smoothed_pearson(pred: np.ndarray, obs: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Per-cell log Pearson vs neighbour-averaged observations.

    ``pred``: bins x cells (predictions for the evaluated cells); ``obs``: bins x N observed targets
    of the neighbour universe; ``neighbors``: (cells x k) indices into ``obs`` columns.
    """
    smooth = np.stack([obs[:, nb].mean(1) for nb in neighbors], axis=1)
    return _rowwise_pearson(np.log1p(pred), np.log1p(smooth))


def pseudobulk_pearson(pred: np.ndarray, obs: np.ndarray, groups: Sequence, min_cells: int = 20) -> dict:
    """Per-group and differential (group-centred) log Pearson of summed profiles."""
    g = np.asarray(groups).astype(str)
    names = [x for x in np.unique(g) if (g == x).sum() >= min_cells]
    P = np.stack([np.log1p(pred[:, g == x].sum(1)) for x in names], 1)
    O = np.stack([np.log1p(obs[:, g == x].sum(1)) for x in names], 1)
    per_group = dict(zip(names, _rowwise_pearson(P, O)))
    diff = float(np.corrcoef((P - P.mean(1, keepdims=True)).ravel(), (O - O.mean(1, keepdims=True)).ravel())[0, 1])
    return {"per_group": per_group, "differential": diff}


def knn_transfer_accuracy(ref_states: np.ndarray, ref_labels: Sequence, query_states: np.ndarray,
                          query_labels: Sequence, k: int = 15, min_class: int = 20) -> float:
    """kNN(k) label transfer accuracy (L2-normalised states; rare reference classes dropped)."""
    from sklearn.neighbors import KNeighborsClassifier
    yr, yq = np.asarray(ref_labels).astype(str), np.asarray(query_labels).astype(str)
    u, n = np.unique(yr, return_counts=True); ok = set(u[n >= min_class])
    mr, mq = np.isin(yr, list(ok)) & ~np.isnan(ref_states).any(1), np.isin(yq, list(ok)) & ~np.isnan(query_states).any(1)
    nrm = lambda S: S / np.linalg.norm(S, axis=1, keepdims=True)
    clf = KNeighborsClassifier(k).fit(nrm(ref_states[mr]), yr[mr])
    return float(np.mean(clf.predict(nrm(query_states[mq])) == yq[mq]))
