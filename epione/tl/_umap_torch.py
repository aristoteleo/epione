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
