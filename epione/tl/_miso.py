# SPDX-License-Identifier: GPL-3.0-or-later
#
# Clean-room re-implementation of the MISO model (Coleman et al., "Resolving
# tissue complexity by multi-modal spatial omics modeling with MISO",
# Nature Methods 22, 2025, doi:10.1038/s41592-024-02574-2).
#
# Licence note: the upstream code (https://github.com/kpcoleman/miso, "Penn
# Software MISO") is licensed for non-profit research use only and forbids
# distribution to commercial third parties, which is incompatible with
# Epione's GPL-3.0. No upstream code is included here: this module was written
# from the paper's description of the model and from reading the upstream
# implementation to match its documented hyper-parameters (128 PCs, 32-d
# embeddings, Gaussian affinity with bandwidth 30, 1000 epochs, Adam 1e-3,
# rank-32 projection of pairwise interaction terms, k-means with seed 100).
# Image (histology) features are not implemented; omics modalities only.
"""MISO-style multi-modal embedding: per-modality spectral auto-encoders + interaction terms."""
from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["miso", "miso_features"]


def miso_features(adata, modality: str = "atac", min_cells: int = 10) -> np.ndarray:
    """Default MISO input of one omics layer: ``min_cells`` feature filter + log1p (dense).

    ``modality`` ``"rna"``/``"atac"``: filter + log1p; ``"protein"``: per-spot
    centred log-ratio; ``"metabolite"``: log1p. Works on a copy.
    """
    import scanpy as sc

    a = adata.copy()
    a.var_names_make_unique()
    if modality in ("rna", "atac"):
        sc.pp.filter_genes(a, min_cells=min_cells)
        sc.pp.log1p(a)
    elif modality == "protein":
        X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X, dtype=float)
        out = np.empty_like(X, dtype=float)
        for i, row in enumerate(X):
            g = np.exp(np.sum(np.log1p(row[row > 0])) / len(row))
            out[i] = np.log1p(row / g)
        return out
    elif modality == "metabolite":
        sc.pp.log1p(a)
    else:
        raise ValueError("modality must be 'rna', 'atac', 'protein' or 'metabolite'")
    return a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)


def _gaussian_affinity(Z, bandwidth, n_neighbors=None):
    """exp(-d^2 / (2 bandwidth^2)) over all pairs, or over a k-NN graph if ``n_neighbors``."""
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import kneighbors_graph

    if n_neighbors is None:
        d = pairwise_distances(Z)
        return np.exp(-(d ** 2) / (2.0 * bandwidth ** 2))
    g = kneighbors_graph(Z, n_neighbors=n_neighbors, mode="distance")
    g.data = np.exp(-(g.data ** 2) / (2.0 * bandwidth ** 2))
    g.eliminate_zeros()
    return g


def _train_view(X, A, *, latent, epochs, lr, device, sparse):
    """Linear encoder -> orthogonal map (embedding) -> linear decoder; MSE + affinity-weighted distances."""
    import torch
    import torch.nn as nn
    from torch.nn.utils.parametrizations import orthogonal

    d = X.shape[1]
    enc = nn.Linear(d, latent)
    rot = orthogonal(nn.Linear(latent, latent))
    dec = nn.Linear(latent, d)
    net = nn.ModuleDict({"enc": enc, "rot": rot, "dec": dec}).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    if sparse:
        A = A.tocoo()
        ii = torch.as_tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=device)
        w = torch.as_tensor(A.data, dtype=torch.float32, device=device)
    else:
        At = torch.triu(torch.as_tensor(A, dtype=torch.float32, device=device))
    for _ in range(int(epochs)):
        opt.zero_grad()
        Y = rot(enc(Xt))
        recon = dec(Y)
        if sparse:
            smooth = (torch.norm(Y[ii[0]] - Y[ii[1]], dim=1) * w).mean()
        else:
            smooth = (torch.triu(torch.cdist(Y, Y)) * At).mean()
        loss = nn.functional.mse_loss(recon, Xt) + smooth
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return rot(enc(Xt))


def miso(
    features: Sequence[np.ndarray],
    *,
    n_clusters: Optional[int] = None,
    pairs="all",
    keep_views="all",
    n_pcs: int = 128,
    latent: int = 32,
    bandwidth: float = 30.0,
    n_neighbors: Optional[int] = None,
    epochs: int = 1000,
    lr: float = 1e-3,
    seed: int = 100,
    kmeans_seed: int = 100,
    device: Optional[str] = None,
    adata=None,
    key_added: str = "X_miso",
    domain_key: str = "miso_domain",
):
    """MISO joint embedding (and k-means domains) of several spatial omics layers.

    Clean-room re-implementation (no upstream code; see the module header for
    the licence reason) of MISO (Coleman et al., *Nat Methods* 2025). For each
    layer: standardise, PCA to ``n_pcs`` when wider, train a linear encoder +
    orthogonal map (``latent``-d embedding) with a linear decoder on
    reconstruction MSE plus the affinity-weighted mean pairwise distance of the
    embeddings (Gaussian affinity of the standardised features, bandwidth
    ``bandwidth``; ``n_neighbors`` for a sparse k-NN affinity). Pairwise
    interaction terms are per-spot outer products of two layers' embeddings,
    projected on their top ``latent`` principal directions. The output is the
    concatenation of the standardised embeddings and interaction terms.

    Arguments:
        features: list of per-spot feature matrices (same spot order), e.g.
            ``[miso_features(rna, "rna"), miso_features(atac, "atac")]``.
        n_clusters: k-means domains (``random_state=kmeans_seed``).
        pairs: layer pairs for interaction terms (``"all"`` or list of tuples;
            ``None`` for none); keep_views: layers whose own embedding is kept.
        adata: optional AnnData receiving ``obsm[key_added]`` / ``obs[domain_key]``.

    Returns:
        ``(embedding, labels or None)``.
    """
    import random

    import torch
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    feats = [np.asarray(f.toarray() if sp.issparse(f) else f, dtype=float) for f in features]
    n = feats[0].shape[0]
    if any(f.shape[0] != n for f in feats):
        raise ValueError("all feature matrices need the same spots")
    std = [StandardScaler().fit_transform(f) for f in feats]
    aff = [_gaussian_affinity(s, bandwidth, n_neighbors) for s in std]
    pcs = [PCA(n_pcs).fit_transform(s) if s.shape[1] > n_pcs else s for s in std]
    Y = [_train_view(p, A, latent=latent, epochs=epochs, lr=lr, device=dev, sparse=n_neighbors is not None)
         for p, A in zip(pcs, aff)]
    views = list(range(len(Y))) if keep_views == "all" else list(keep_views)
    combos = list(combinations(range(len(Y)), 2)) if pairs == "all" else (list(pairs) if pairs else [])
    blocks = [StandardScaler().fit_transform(Y[i].cpu().numpy()) for i in views]
    for i, j in combos:
        inter = (Y[i][:, :, None] * Y[j][:, None, :]).reshape(n, -1)
        V = torch.pca_lowrank(inter, q=latent)[2]
        blocks.append(StandardScaler().fit_transform((inter @ V).cpu().numpy()))
    emb = np.concatenate(blocks, axis=1)
    labels = None
    if n_clusters is not None:
        from sklearn.cluster import KMeans

        labels = KMeans(int(n_clusters), random_state=kmeans_seed).fit_predict(emb)
    if adata is not None:
        adata.obsm[key_added] = emb
        if labels is not None:
            adata.obs[domain_key] = pd.Categorical(labels.astype(str))
    return emb, labels
