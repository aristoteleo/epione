"""UMAP layout optimisation on a torch device (``epi.tl.umap(device=...)``).

The same objective, schedule and update rule as umap-learn's
``optimize_layout_euclidean`` (``move_other=True``): every edge of the fuzzy
graph is sampled once every ``epochs_per_sample`` epochs, applies the clipped
attractive gradient to both ends, and is followed by
``negative_sample_rate``-scaled random negative samples on its head, with a
learning rate decaying linearly from ``initial_alpha``. As in cuML's GPU
UMAP, all edges active in an epoch are updated in parallel (summed with
``index_add_`` instead of umap-learn's sequential Hogwild updates), so the
embedding is a different realisation of the same optimisation, not a
bit-copy of the CPU one. Device memory is ``O(n_edges)``: a few int64 /
float32 vectors over the graph's edges, processed in bounded edge chunks.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def optimize_layout_torch(
    embedding: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    epochs_per_sample: np.ndarray,
    n_epochs: int,
    a: float,
    b: float,
    *,
    dev,
    gamma: float = 1.0,
    initial_alpha: float = 1.0,
    negative_sample_rate: float = 5.0,
    seed: int = 0,
    max_edges_per_chunk: int = 1 << 23,
) -> np.ndarray:
    """umap-learn's SGD schedule with parallel per-epoch updates on ``dev``."""
    from ..core._gpu import get_torch
    torch = get_torch()
    n = embedding.shape[0]
    Y = torch.as_tensor(np.ascontiguousarray(embedding, dtype=np.float32), device=dev)
    h = torch.as_tensor(np.asarray(head, dtype=np.int64), device=dev)
    t = torch.as_tensor(np.asarray(tail, dtype=np.int64), device=dev)
    eps = torch.as_tensor(np.asarray(epochs_per_sample, dtype=np.float64), device=dev)
    eps_neg = eps / float(negative_sample_rate)
    next_s = eps.clone()
    next_neg = eps_neg.clone()
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    a = float(a)
    b = float(b)
    E = h.shape[0]
    for ep in range(int(n_epochs)):
        alpha = float(initial_alpha) * (1.0 - ep / float(n_epochs))
        active = torch.nonzero(next_s <= ep, as_tuple=False).squeeze(1)
        if active.numel() == 0:
            continue
        upd = torch.zeros_like(Y)
        for c0 in range(0, active.numel(), max_edges_per_chunk):
            idx = active[c0:c0 + max_edges_per_chunk]
            j, k = h[idx], t[idx]
            diff = Y[j] - Y[k]
            d2 = (diff * diff).sum(1)
            gc = torch.where(
                d2 > 0,
                -2.0 * a * b * d2.clamp_min(1e-30).pow(b - 1.0) / (a * d2.pow(b) + 1.0),
                torch.zeros_like(d2))
            g = (gc[:, None] * diff).clamp_(-4.0, 4.0) * alpha
            upd.index_add_(0, j, g)
            upd.index_add_(0, k, -g)
            # negative samples of these edges
            n_neg = ((ep - next_neg[idx]) / eps_neg[idx]).floor().clamp_min(0).to(torch.int64)
            next_neg[idx] += n_neg.to(torch.float64) * eps_neg[idx]
            tot = int(n_neg.sum())
            if tot:
                jn = torch.repeat_interleave(j, n_neg)
                kn = torch.randint(0, n, (tot,), device=dev, generator=gen)
                keep = jn != kn
                jn, kn = jn[keep], kn[keep]
                diff = Y[jn] - Y[kn]
                d2 = (diff * diff).sum(1)
                # umap-learn: coincident points (d2 == 0) get gradient 4.0
                gc = torch.where(d2 > 0, 2.0 * float(gamma) * b / ((0.001 + d2) * (a * d2.pow(b) + 1.0)),
                                 torch.zeros_like(d2))
                g = torch.where((gc > 0)[:, None], (gc[:, None] * diff).clamp(-4.0, 4.0),
                                torch.full_like(diff, 4.0))
                upd.index_add_(0, jn, g * alpha)
                del jn, kn, diff, d2, gc, g
        next_s[active] += eps[active]
        Y += upd
        del upd, active
    return Y.cpu().numpy()


def spectral_layout_torch(graph, dim: int, random_state, *, dev, tol: float = 1e-4,
                          niter: int = 2000) -> Optional[np.ndarray]:
    """umap-learn's spectral initialisation (eigenvectors 2..dim+1 of the
    normalised graph Laplacian ``I - D^-1/2 A D^-1/2``) with torch's LOBPCG on
    ``dev``: the largest eigenpairs of ``D^-1/2 A D^-1/2``, same tolerance
    (1e-4) and the same start block as umap-learn's LOBPCG branch.

    Returns ``None`` for a disconnected graph (the caller then uses
    umap-learn's multi-component layout) or when LOBPCG fails."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph
    from ..core._gpu import get_torch
    torch = get_torch()
    n_comp, _ = scipy.sparse.csgraph.connected_components(graph)
    if n_comp > 1:
        return None
    G = sp.csr_matrix(graph, dtype=np.float64)
    deg = np.asarray(G.sum(axis=0)).ravel()
    sqrt_deg = np.sqrt(deg)
    inv = 1.0 / sqrt_deg
    G = G.tocoo()
    vals = G.data * inv[G.row] * inv[G.col]
    idx = torch.as_tensor(np.vstack([G.row, G.col]).astype(np.int64), device=dev)
    M = torch.sparse_coo_tensor(idx, torch.as_tensor(vals, device=dev),
                                size=G.shape).coalesce().to_sparse_csr()
    del idx, vals, G
    k = dim + 1
    gen = (random_state if isinstance(random_state, (np.random.Generator, np.random.RandomState))
           else np.random.default_rng(seed=random_state))
    X = gen.normal(size=(M.shape[0], k))
    X[:, 0] = sqrt_deg / np.linalg.norm(sqrt_deg)
    X = torch.as_tensor(X, device=dev)
    try:
        ev, V = torch.lobpcg(M, k=k, X=X, tol=tol, niter=niter, largest=True)
    except Exception:
        return None
    if not torch.isfinite(V).all():
        return None
    order = torch.argsort(ev, descending=True)[1:k]
    return V[:, order].cpu().numpy()


def multi_component_layout_fast(data, graph, n_components, component_labels, dim, random_state,
                                metric="euclidean", metric_kwds={}, init="random", tol=0.0,
                                maxiter=0):
    """umap-learn's ``multi_component_layout`` with the per-component work
    done on precomputed row index lists instead of a boolean mask over all
    cells and a fresh ``graph.tocsr()`` per component (O(n_components x nnz),
    hours for tens of thousands of components). Same sub-matrices in the
    same order and the same random draws, hence the same layout."""
    from sklearn.metrics import pairwise_distances
    from umap import spectral as us
    labels = np.asarray(component_labels)
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(n_components + 1))
    groups = [order[bounds[c]:bounds[c + 1]] for c in range(n_components)]
    result = np.empty((graph.shape[0], dim), dtype=np.float32)
    if n_components > 2 * dim:
        meta_embedding = _component_layout_fast(data, n_components, groups, dim, random_state,
                                                metric=metric, metric_kwds=metric_kwds,
                                                labels=labels)
    else:
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]
    G = graph.tocsr()
    for label in range(n_components):
        idx = groups[label]
        component_graph = G[idx, :].tocsc()
        component_graph = component_graph[:, idx].tocoo()
        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0
        if component_graph.shape[0] < 2 * dim or component_graph.shape[0] <= dim + 1:
            result[idx] = (random_state.uniform(low=-data_range, high=data_range,
                                                size=(component_graph.shape[0], dim))
                           + meta_embedding[label])
        else:
            component_embedding = us._spectral_layout(
                data=None, graph=component_graph, dim=dim, random_state=random_state,
                metric=metric, metric_kwds=metric_kwds, init=init, tol=tol, maxiter=maxiter)
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[idx] = component_embedding + meta_embedding[label]
    return result


def _component_layout_fast(data, n_components, groups, dim, random_state, metric="euclidean",
                           metric_kwds={}, labels=None):
    """umap-learn's ``component_layout`` with the centroids taken over
    precomputed row index lists (same rows, same order, same means)."""
    from umap import spectral as us
    if data is None or metric == "precomputed" or callable(metric) or metric not in (
            "euclidean", "l2", "cosine", "correlation", "manhattan", "l1", "cityblock"):
        return us.component_layout(data, n_components, labels, dim, random_state,
                                   metric=metric, metric_kwds=metric_kwds)
    import scipy.sparse
    from sklearn.metrics import pairwise_distances
    from sklearn.manifold import SpectralEmbedding
    component_centroids = np.empty((n_components, data.shape[1]), dtype=np.float64)
    for label in range(n_components):
        component_centroids[label] = data[groups[label]].mean(axis=0)
    if scipy.sparse.isspmatrix(component_centroids):
        component_centroids = component_centroids.toarray()
    distance_matrix = pairwise_distances(component_centroids, metric=metric, **metric_kwds)
    affinity_matrix = np.exp(-(distance_matrix ** 2))
    component_embedding = SpectralEmbedding(
        n_components=dim, affinity="precomputed", random_state=random_state
    ).fit_transform(affinity_matrix)
    component_embedding /= component_embedding.max()
    return component_embedding
