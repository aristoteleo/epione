"""GPU LDA sampler (``run_topic_models(method="gpu")``); requires PyTorch.

Blocked parallel collapsed Gibbs sampling: the tokens, in a fixed random
order, are cut into ``n_blocks`` blocks; a sweep resamples the blocks one after
another, every token of a block at once from the current counts with the
token's own assignment removed,

    p(z_i = k) ∝ (n_wk − δ + β) / (n_k − δ + Vβ) · (n_dk − δ + α),   δ = [z_i = k],

and the counts of the tokens that changed topic are updated with ``index_add``
before the next block. Each block
therefore sees the assignments of all earlier blocks, which keeps the chain close
to sequential Gibbs (one fully synchronous update of all tokens converges to a
visibly lower likelihood). Only the column of the token's own topic is corrected,
and running sums over topics are a matmul with a triangular matrix.
torch is imported lazily; it is not a dependency of Epione.
"""
from __future__ import annotations

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("method='gpu' needs PyTorch (pip install torch)") from e
    return torch


def fit_gpu(WS, DS, n_docs, n_words, n_topics, n_iter, alpha, eta, random_state,
            device=None, n_blocks: int = 16):
    """Fit one model on the GPU; returns ``(nzw, ndz, nz)`` as numpy int32 arrays."""
    torch = _torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    K = int(n_topics)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(random_state))
    w = torch.as_tensor(np.asarray(WS, dtype=np.int64), device=dev)
    d = torch.as_tensor(np.asarray(DS, dtype=np.int64), device=dev)
    N = w.shape[0]
    z = torch.randint(0, K, (N,), generator=gen, device=dev)
    wk = w * K
    dk = d * K
    beta = float(eta)
    vbeta = beta * n_words

    def counts(z):
        nwz = torch.bincount(wk + z, minlength=n_words * K).view(n_words, K)
        ndz = torch.bincount(dk + z, minlength=n_docs * K).view(n_docs, K)
        nz = torch.bincount(z, minlength=K)
        return nwz.to(torch.float32), ndz.to(torch.float32), nz.to(torch.float64)

    # running sums over topics as one matmul with an upper-triangular matrix
    # (torch.cumsum along a short last axis is two orders of magnitude slower)
    tri = torch.triu(torch.ones(K, K, device=dev, dtype=torch.float32))
    nwz, ndz, nz = counts(z)
    # tokens are visited in a fixed random order, cut into n_blocks contiguous
    # blocks; each block sees the counts left by the previous blocks, which keeps
    # the chain close to sequential Gibbs while every block is sampled in parallel
    n_blocks = max(1, min(int(n_blocks), N))
    perm = torch.randperm(N, generator=gen, device=dev)
    w, d, z, wk, dk = w[perm], d[perm], z[perm], wk[perm], dk[perm]
    bounds = [(N * b) // n_blocks for b in range(n_blocks + 1)]
    # inside a block the order is irrelevant; sorting by cell makes the
    # cell-topic gathers coalesced
    order = torch.cat([s0 + torch.argsort(d[s0:s1], stable=True)
                       for s0, s1 in zip(bounds[:-1], bounds[1:])])
    w, d, z, wk, dk = w[order], d[order], z[order], wk[order], dk[order]
    nwz_flat = nwz.view(-1)
    ndz_flat = ndz.view(-1)
    for _ in range(n_iter):
        for b in range(n_blocks):
            s0, s1 = bounds[b], bounds[b + 1]
            wc, dc, zc = w[s0:s1], d[s0:s1], z[s0:s1]
            nzf = nz.to(torch.float32)   # topic totals accumulate in float64 (> 2**24)
            p = (nwz[wc] + beta) / (nzf + vbeta) * (ndz[dc] + alpha)
            own = ((nwz_flat[wk[s0:s1] + zc] - 1 + beta) / (nzf[zc] - 1 + vbeta)
                   * (ndz_flat[dk[s0:s1] + zc] - 1 + alpha))
            p.scatter_(1, zc[:, None], own[:, None])
            cdf = p @ tri
            u = torch.rand(s1 - s0, 1, generator=gen, device=dev) * cdf[:, -1:]
            zn = torch.searchsorted(cdf, u).squeeze(1).clamp_(max=K - 1)
            ch = (zn != zc).nonzero().squeeze(1)
            if ch.numel():
                zo, znw = zc[ch], zn[ch]
                one = torch.ones(ch.numel(), device=dev, dtype=torch.float32)
                wkc, dkc = wk[s0:s1][ch], dk[s0:s1][ch]
                nwz_flat.index_add_(0, wkc + zo, -one)
                nwz_flat.index_add_(0, wkc + znw, one)
                ndz_flat.index_add_(0, dkc + zo, -one)
                ndz_flat.index_add_(0, dkc + znw, one)
                one64 = one.to(torch.float64)
                nz.index_add_(0, zo, -one64)
                nz.index_add_(0, znw, one64)
                z[s0:s1] = zn
    to_np = lambda t: t.to(torch.int32).cpu().numpy()
    return np.ascontiguousarray(to_np(nwz).T), to_np(ndz), to_np(nz)
