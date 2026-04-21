"""kNN-based label transfer in a shared embedding.

``epi.tl.transfer_labels(query, reference, reference_label=...)`` looks
up the ``k`` nearest reference cells for each query cell in a shared
embedding (typically produced by :func:`epi.tl.integrate`) and writes
the weighted-vote label + confidence back onto the query AnnData.

This is the Python analogue of Seurat's ``TransferData`` / ArchR's
``addGeneIntegrationMatrix`` label-transfer step.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def transfer_labels(
    query,
    reference,
    *,
    reference_label: str,
    use_rep: str = "X_cca",
    k: int = 30,
    metric: str = "cosine",
    weighted: bool = True,
    strip_prefix: Optional[str] = None,
    key_added: str = "celltype",
    score_key: str = "transfer_score",
    raw_key: Optional[str] = None,
    neighbors_key: Optional[str] = None,
):
    """Transfer a label column from ``reference`` onto ``query`` via kNN.

    Parameters
    ----------
    query, reference
        Two AnnData objects with matching ``obsm[use_rep]`` embeddings
        (run :func:`epi.tl.integrate` first).
    reference_label
        Column of ``reference.obs`` to transfer.
    use_rep
        Embedding to use for neighbour search.
    k
        Neighbourhood size (ArchR default 30).
    metric
        Distance metric passed to :class:`sklearn.neighbors.NearestNeighbors`.
    weighted
        If ``True`` (default), weight each neighbour's vote by
        ``exp(-d/mean(d))`` — closer neighbours count more, matching
        Seurat's TransferData. If ``False``, uniform majority vote.
    strip_prefix
        Optional regex, applied to the transferred labels. Granja 2019
        BioClassification uses ``"12_CD14.Mono.2"`` style prefixes;
        pass ``r'^\\d+_'`` to drop them.
    key_added
        Column name written to ``query.obs``.
    score_key
        Confidence column name (vote-weight fraction, in [0, 1]).
    raw_key
        If given, also store the raw (unstripped) categorical under
        this column name — handy to keep a record of the original
        reference categories.
    neighbors_key
        If given, store ``(idx, dist)`` kNN arrays under
        ``query.obsm[f'{neighbors_key}_idx']`` and
        ``query.obsm[f'{neighbors_key}_dist']`` so downstream analyses
        (e.g. neighbour-label composition plots) can reuse them
        without recomputing the kNN.

    Returns
    -------
    ``None``. Side-effects are written back onto ``query``.
    """
    from sklearn.neighbors import NearestNeighbors

    if use_rep not in query.obsm or use_rep not in reference.obsm:
        raise KeyError(
            f"Both query and reference need obsm['{use_rep}']. "
            f"Run epi.tl.integrate() first."
        )
    if reference_label not in reference.obs.columns:
        raise KeyError(f"reference.obs has no column {reference_label!r}")

    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(reference.obsm[use_rep])
    dist, idx = nn.kneighbors(query.obsm[use_rep])

    ref_labels = reference.obs[reference_label].astype(str).to_numpy()

    if weighted:
        # exp(-d / mean(d)) — softmax-ish weighting per cell
        w = np.exp(-dist / (dist.mean(axis=1, keepdims=True) + 1e-9))
    else:
        w = np.ones_like(dist)

    transferred = np.empty(query.n_obs, dtype=object)
    confidence = np.empty(query.n_obs, dtype=np.float32)

    for i in range(query.n_obs):
        neigh = ref_labels[idx[i]]
        wi = w[i]
        vote = {}
        for lab, wk in zip(neigh, wi):
            vote[lab] = vote.get(lab, 0.0) + wk
        top_lab, top_w = max(vote.items(), key=lambda kv: kv[1])
        transferred[i] = top_lab
        confidence[i] = top_w / wi.sum()

    raw = pd.Categorical(transferred)
    if raw_key is not None:
        query.obs[raw_key] = raw

    if strip_prefix is not None:
        stripped = pd.Series(transferred).astype(str).str.replace(
            strip_prefix, "", regex=True
        )
        query.obs[key_added] = pd.Categorical(stripped.to_numpy())
    else:
        query.obs[key_added] = raw

    query.obs[score_key] = confidence

    if neighbors_key is not None:
        query.obsm[f"{neighbors_key}_idx"] = idx.astype(np.int32)
        query.obsm[f"{neighbors_key}_dist"] = dist.astype(np.float32)


__all__ = ["transfer_labels"]
