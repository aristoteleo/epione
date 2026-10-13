"""t-SNE embedding with openTSNE (Poličar *et al.*, J Stat Softw 2024; BSD-3-Clause).

openTSNE implements FIt-SNE (Linderman *et al.*, Nat Methods 2019): FFT-
accelerated interpolation for the repulsive forces and approximate nearest
neighbours for the attractive ones, so it scales to hundreds of thousands of
cells.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def tsne(
    adata,
    use_rep: str = "X_lsi",
    *,
    dims: Optional[Sequence[int]] = None,
    perplexity: float = 30.0,
    metric: str = "cosine",
    initialization: str = "pca",
    n_iter: int = 750,
    learning_rate: Optional[float] = None,
    exaggeration: Optional[float] = None,
    random_state: int = 0,
    n_jobs: int = -1,
    key_added: str = "X_tsne",
) -> np.ndarray:
    """Compute a t-SNE embedding of a reduced representation.

    For scATAC the default runs on LSI with cosine distance, as ArchR's
    ``addTSNE`` does; for RNA pass ``use_rep="X_pca"``. Plot it with
    :func:`epione.pl.tsne`.

    Arguments:
        adata: AnnData with ``obsm[use_rep]``.
        use_rep: representation to embed.
        dims: 1-based dimensions of ``use_rep`` to use (default all), e.g.
            ``range(2, 31)`` to drop the depth-correlated first LSI component.
        perplexity: effective number of neighbours.
        metric: distance on ``use_rep``.
        initialization: ``"pca"`` (default, preserves global structure),
            ``"spectral"`` or ``"random"``.
        n_iter: optimisation iterations after early exaggeration.
        learning_rate: default ``n_obs / 12`` (openTSNE's ``"auto"``).
        exaggeration: late exaggeration factor (``None`` = 1); values around
            2–4 tighten clusters on large datasets.
        random_state: seed.
        n_jobs: threads.
        key_added: ``obsm`` key for the 2-D result.

    Returns:
        The ``(n_obs, 2)`` embedding, also written to ``adata.obsm[key_added]``.

    Examples:
        >>> epi.tl.tsne(adata, use_rep="X_lsi", dims=range(2, 31))
        >>> epi.pl.tsne(adata, color="cell_type")
    """
    try:
        from openTSNE import TSNE
    except ImportError as e:  # pragma: no cover
        raise ImportError("epi.tl.tsne needs openTSNE: pip install openTSNE") from e

    x = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    if dims is not None:
        x = x[:, np.asarray(list(dims), dtype=int) - 1]
    model = TSNE(
        perplexity=perplexity,
        metric=metric,
        initialization=initialization,
        n_iter=n_iter,
        learning_rate="auto" if learning_rate is None else learning_rate,
        exaggeration=exaggeration,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=False,
    )
    emb = np.asarray(model.fit(x))
    adata.obsm[key_added] = emb
    adata.uns[key_added.removeprefix("X_")] = {
        "params": {"use_rep": use_rep, "perplexity": perplexity, "metric": metric,
                   "initialization": initialization, "n_iter": n_iter, "random_state": random_state}
    }
    return emb


__all__ = ["tsne"]
