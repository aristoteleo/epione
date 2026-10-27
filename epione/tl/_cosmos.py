# SPDX-License-Identifier: MIT
#
# Vendored port of COSMOS (Zhou, Xiao, Dong, Tang, Xiao & Xu, "Cooperative
# integration of spatially resolved multi-omics data with COSMOS",
# Nature Communications 16, 2025, doi:10.1038/s41467-024-55204-y).
# Upstream: https://github.com/Lin-Xu-lab/COSMOS
#           commit 56ea355be51e64d9253e2871b8bd447fdfd0d230
# Copyright (c) 2024 The University of Texas Southwestern Medical Center, MIT.
# COSMOS itself contains code adapted from SpaceFlow (c) 2022 Honglei, MIT;
# pyWNN (c) 2021 Dylan Kotliar, MIT; and PyTorch Geometric's DeepGraphInfomax
# (c) 2023 PyG Team, MIT. The full licence texts are in
# epione/tl/_licenses/COSMOS.LICENSE and the repository NOTICE file.
#
# Changes from upstream (none alters the numbers, see the parity record
# tests/parity/spatial_cosmos/MANIFEST.md):
#   * torch_geometric.nn.GCNConv and torch_geometric.nn.inits are replaced by
#     the plain-torch re-implementations in epione.tl._gnn (same arithmetic,
#     same random-number consumption);
#   * the Cosmos class is wrapped by the functional entry point cosmos();
#     results are written to AnnData instead of being returned only;
#   * the device is an argument instead of a module-level global;
#   * gudhi / cmcrameri / networkx / matplotlib imports (unused by training)
#     are dropped; progress printing goes through ``verbose``.
# Upstream behaviour kept on purpose (documented in cosmos()):
#   * with regularization_acceleration=True the second coordinate subset
#     reuses the first index (``c2`` is drawn with ``cell_random_subset_1``),
#     so the spatial penalty is zero; set ``fix_regularization_index=True``
#     to use the second index (changes results);
#   * ``best_params = model.state_dict()`` aliases the live parameters, so the
#     returned model is the final one, not the best-loss one.
"""COSMOS: spatial multi-omics integration with a WNN-weighted graph infomax."""
from __future__ import annotations

import copy
import random
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["cosmos", "pywnn"]


# --------------------------------------------------------------------------
# pyWNN (COSMOS/pyWNN.py, after https://github.com/dylkot/pyWNN)
# --------------------------------------------------------------------------
def _get_nearestneighbor(knn, neighbor=1):
    indices, indptr, data = knn.indices, knn.indptr, knn.data
    nn_idx = []
    for i in range(knn.shape[0]):
        cols = indices[indptr[i]:indptr[i + 1]]
        rowvals = data[indptr[i]:indptr[i + 1]]
        idx = np.argsort(rowvals)
        nn_idx.append(cols[idx[neighbor - 1]])
    return np.array(nn_idx)


def _compute_bw(knn_adj, embedding, n_neighbors=20):
    intersect = knn_adj.dot(knn_adj.T)
    indices, indptr, data = intersect.indices, intersect.indptr, intersect.data
    data = data / ((n_neighbors * 2) - data)
    bandwidth = []
    for i in range(intersect.shape[0]):
        cols = indices[indptr[i]:indptr[i + 1]]
        rowvals = data[indptr[i]:indptr[i + 1]]
        idx = np.argsort(rowvals)
        valssort = rowvals[idx]
        numinset = len(cols)
        if numinset < n_neighbors:
            sys.exit("Fewer than 20 cells with Jacard sim > 0")
        else:
            curval = valssort[n_neighbors]
            for num in range(n_neighbors, numinset):
                if valssort[num] != curval:
                    break
                else:
                    num += 1
            minjacinset = cols[idx][:num]
            if num < n_neighbors:
                sys.exit(-1)
            else:
                euc_dist = ((embedding[minjacinset, :] - embedding[i, :]) ** 2).sum(axis=1) ** .5
                euc_dist_sorted = np.sort(euc_dist)[::-1]
                bandwidth.append(np.mean(euc_dist_sorted[:n_neighbors]))
    return np.array(bandwidth)


def _compute_affinity(dist_to_predict, dist_to_nn, bw):
    affinity = dist_to_predict - dist_to_nn
    affinity[affinity < 0] = 0
    affinity = affinity * -1
    affinity = np.exp(affinity / (bw - dist_to_nn))
    return affinity


def _dist_from_adj(adjacency, embed1, embed2, nndist1, nndist2):
    dist1 = sp.lil_matrix(adjacency.shape)
    dist2 = sp.lil_matrix(adjacency.shape)
    indices, indptr = adjacency.indices, adjacency.indptr
    for i in range(adjacency.shape[0]):
        for j in range(indptr[i], indptr[i + 1]):
            col = indices[j]
            a = (((embed1[i, :] - embed1[col, :]) ** 2).sum() ** .5) - nndist1[i]
            dist1[i, col] = np.nan if a == 0 else a
            b = (((embed2[i, :] - embed2[col, :]) ** 2).sum() ** .5) - nndist2[i]
            dist2[i, col] = np.nan if b == 0 else b
    return sp.csr_matrix(dist1), sp.csr_matrix(dist2)


def _select_topK(dist, n_neighbors=20):
    indices, indptr, data = dist.indices, dist.indptr, dist.data
    nrows = dist.shape[0]
    final_data, final_col_ind = [], []
    for i in range(nrows):
        cols = indices[indptr[i]:indptr[i + 1]]
        rowvals = data[indptr[i]:indptr[i + 1]]
        idx = np.argsort(rowvals)
        final_data.append(rowvals[idx[(-1 * n_neighbors):]])
        final_col_ind.append(cols[idx[(-1 * n_neighbors):]])
    final_data = np.concatenate(final_data)
    final_col_ind = np.concatenate(final_col_ind)
    final_row_ind = np.tile(np.arange(nrows), (n_neighbors, 1)).reshape(-1, order="F")
    return sp.csr_matrix((final_data, (final_row_ind, final_col_ind)), shape=(nrows, dist.shape[1]))


class pywnn:
    """Weighted nearest neighbours (Hao et al. 2021) as shipped in COSMOS."""

    def __init__(self, adata, reps=("X_pca", "X_apca"), n_neighbors=20, npcs=(20, 20), seed=14, distances=None):
        import scanpy as sc
        from sklearn import preprocessing

        self.seed = seed
        np.random.seed(seed)
        reps = list(reps)
        npcs = list(npcs)
        if len(reps) > 2:
            sys.exit("WNN currently only implemented for 2 modalities")
        self.adata = adata.copy()
        self.reps = [r + "_norm" for r in reps]
        self.npcs = npcs
        for (i, r) in enumerate(reps):
            self.adata.obsm[self.reps[i]] = preprocessing.normalize(adata.obsm[r][:, 0:npcs[i]])
        self.n_neighbors = n_neighbors
        if distances is None:
            sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=npcs[0], use_rep=self.reps[0], metric="euclidean", key_added="1")
            sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=npcs[1], use_rep=self.reps[1], metric="euclidean", key_added="2")
            sc.pp.neighbors(self.adata, n_neighbors=200, n_pcs=npcs[0], use_rep=self.reps[0], metric="euclidean", key_added="1_200")
            sc.pp.neighbors(self.adata, n_neighbors=200, n_pcs=npcs[1], use_rep=self.reps[1], metric="euclidean", key_added="2_200")
            self.distances = ["1_distances", "2_distances", "1_200_distances", "2_200_distances"]
        else:
            self.distances = distances
        for d in self.distances:
            if type(self.adata.obsp[d]) is not sp.csr_matrix:
                self.adata.obsp[d] = sp.csr_matrix(self.adata.obsp[d])
        self.NNdist, self.NNidx, self.NNadjacency, self.BWs = [], [], [], []
        for (i, r) in enumerate(self.reps):
            nn = _get_nearestneighbor(self.adata.obsp[self.distances[i]])
            dist_to_nn = ((self.adata.obsm[r] - self.adata.obsm[r][nn, :]) ** 2).sum(axis=1) ** .5
            nn_adj = (self.adata.obsp[self.distances[i]] > 0).astype(int)
            nn_adj_wdiag = nn_adj.copy()
            nn_adj_wdiag.setdiag(1)
            bw = _compute_bw(nn_adj_wdiag, self.adata.obsm[r], n_neighbors=self.n_neighbors)
            self.NNidx.append(nn)
            self.NNdist.append(dist_to_nn)
            self.NNadjacency.append(nn_adj)
            self.BWs.append(bw)
        self.weights = []
        self.WNN = None

    def compute_weights(self):
        cmap = {0: 1, 1: 0}
        affinity_ratios = []
        self.within, self.cross = [], []
        for (i, r) in enumerate(self.reps):
            within_predict = self.NNadjacency[i].dot(self.adata.obsm[r]) / (self.n_neighbors - 1)
            cross_predict = self.NNadjacency[cmap[i]].dot(self.adata.obsm[r]) / (self.n_neighbors - 1)
            within_predict_dist = ((self.adata.obsm[r] - within_predict) ** 2).sum(axis=1) ** .5
            cross_predict_dist = ((self.adata.obsm[r] - cross_predict) ** 2).sum(axis=1) ** .5
            within_affinity = _compute_affinity(within_predict_dist, self.NNdist[i], self.BWs[i])
            cross_affinity = _compute_affinity(cross_predict_dist, self.NNdist[i], self.BWs[i])
            affinity_ratios.append(within_affinity / (cross_affinity + 0.0001))
            self.within.append(within_predict_dist)
            self.cross.append(cross_predict_dist)
        self.weights.append(1 / (1 + np.exp(affinity_ratios[1] - affinity_ratios[0])))
        self.weights.append(1 - self.weights[0])

    def compute_wnn(self, adata):
        self.compute_weights()
        union_adj_mat = ((self.adata.obsp[self.distances[2]] + self.adata.obsp[self.distances[3]]) > 0).astype(int)
        full_dists = _dist_from_adj(union_adj_mat, self.adata.obsm[self.reps[0]], self.adata.obsm[self.reps[1]],
                                    self.NNdist[0], self.NNdist[1])
        weighted_dist = sp.csr_matrix(union_adj_mat.shape)
        for (i, dist) in enumerate(full_dists):
            dist = sp.diags(-1 / (self.BWs[i] - self.NNdist[i]), format="csr").dot(dist)
            dist.data = np.exp(dist.data)
            ind = np.isnan(dist.data)
            dist.data[ind] = 1
            dist = sp.diags(self.weights[i]).dot(dist)
            weighted_dist += dist
        self.WNN = _select_topK(weighted_dist, n_neighbors=self.n_neighbors)
        WNNdist = self.WNN.copy()
        x = (1 - WNNdist.data) / 2
        x[x < 0] = 0
        x[x > 1] = 1
        WNNdist.data = np.sqrt(x)
        self.WNNdist = WNNdist
        adata.obsp["WNN"] = self.WNN
        adata.obsp["WNN_distance"] = self.WNNdist
        adata.obsm["Weights"] = np.array(self.weights).T
        adata.obsm[self.reps[0]] = self.adata.obsm[self.reps[0]]
        adata.obsm[self.reps[1]] = self.adata.obsm[self.reps[1]]
        adata.uns["WNN"] = {"connectivities_key": "WNN", "distances_key": "WNN_distance",
                            "params": {"n_neighbors": self.n_neighbors, "method": "WNN",
                                       "random_state": self.seed, "metric": "euclidean",
                                       "use_rep": self.reps[0], "n_pcs": self.npcs[0]}}
        return adata


# --------------------------------------------------------------------------
# model (COSMOS/cosmos.py, COSMOS/modulesWNN.py)
# --------------------------------------------------------------------------
_EPS = 1e-15


def _build_model_classes():
    import torch
    import torch.nn as nn

    from ._gnn import GCNConv, reset_module, uniform_

    class GraphEncoderWNNit(nn.Module):
        def __init__(self, nsample, in_channels1, in_channels2, hidden_channels, device):
            super().__init__()
            self.conv1 = GCNConv(in_channels1, hidden_channels, cached=False)
            self.conv2 = GCNConv(in_channels2, hidden_channels, cached=False)
            self.prelu = nn.PReLU(hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels, cached=False)
            self.conv4 = GCNConv(hidden_channels, hidden_channels, cached=False)
            self.prelu2 = nn.PReLU(hidden_channels)
            self.prelu3 = nn.PReLU(hidden_channels)
            self.prelu4 = nn.PReLU(hidden_channels)
            self.mddim = hidden_channels
            self._device = device

        def forward(self, x1, x2, edge_index, adata, w, w1, w2):
            x1 = self.conv1(x1, edge_index)
            x1 = self.prelu(x1)
            x1 = self.conv3(x1, edge_index)
            x1 = self.prelu2(x1)
            x1 = nn.functional.normalize(x1, p=2.0, dim=1)
            x2 = self.conv2(x2, edge_index)
            x2 = self.prelu3(x2)
            x2 = self.conv4(x2, edge_index)
            x2 = self.prelu4(x2)
            x2 = nn.functional.normalize(x2, p=2.0, dim=1)
            if w == 1:
                adata.obsm["Omics1_PCA"] = x1.detach().cpu().numpy()
                adata.obsm["Omics2_PCA"] = x2.detach().cpu().numpy()
                wnn = pywnn(adata, reps=["Omics1_PCA", "Omics2_PCA"], npcs=[self.mddim, self.mddim], n_neighbors=20, seed=14)
                adata = wnn.compute_wnn(adata)
                ww = adata.obsm["Weights"].astype(np.float32)
                w1 = torch.reshape(torch.from_numpy(ww[:, 0]), (-1, 1)).to(self._device)
                w2 = torch.reshape(torch.from_numpy(ww[:, 1]), (-1, 1)).to(self._device)
            x1 = x1 * w1
            x2 = x2 * w2
            return x1 + x2, w1, w2

    class DeepGraphInfomaxWNN(nn.Module):
        def __init__(self, hidden_channels, encoder, summary, corruption):
            super().__init__()
            self.hidden_channels = hidden_channels
            self.encoder = encoder
            self.summary = summary
            self.corruption = corruption
            self.weight = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
            self.reset_parameters()

        def reset_parameters(self):
            reset_module(self.encoder)
            reset_module(self.summary)
            uniform_(self.hidden_channels, self.weight)

        def forward(self, *args, **kwargs):
            pos_z, w1, w2 = self.encoder(*args, **kwargs)
            cor = self.corruption(*args, **kwargs)
            cor = cor if isinstance(cor, tuple) else (cor,)
            cor_args = cor[:len(args)]
            cor_kwargs = copy.copy(kwargs)
            for key, value in zip(kwargs.keys(), cor[len(args):]):
                cor_kwargs[key] = value
            neg_z, _, _ = self.encoder(*cor_args, **cor_kwargs)
            summary = self.summary(pos_z, *args, **kwargs)
            return pos_z, neg_z, summary, w1, w2

        def discriminate(self, z, summary, sigmoid=True):
            summary = summary.t() if summary.dim() > 1 else summary
            value = torch.matmul(z, torch.matmul(self.weight, summary))
            return torch.sigmoid(value) if sigmoid else value

        def loss(self, pos_z, neg_z, summary):
            pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + _EPS).mean()
            neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + _EPS).mean()
            return pos_loss + neg_loss

    def corruption(x1, x2, edge_index, adata, w, w1, w2):
        return x1[torch.randperm(x1.size(0))], x2[torch.randperm(x2.size(0))], edge_index, adata, 0, w1, w2

    return GraphEncoderWNNit, DeepGraphInfomaxWNN, corruption


def _dense(X):
    if sp.issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def _resolve_device(device):
    import torch

    if device is None or device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return str(device)


def cosmos(
    adata_omics1,
    adata_omics2,
    *,
    use_rep1: Optional[str] = None,
    use_rep2: Optional[str] = None,
    spatial_key: str = "spatial",
    n_neighbors: int = 10,
    spatial_regularization_strength: float = 0.05,
    z_dim: int = 50,
    lr: float = 1e-3,
    wnn_epoch: int = 100,
    total_epoch: int = 1000,
    max_patience_bef: int = 10,
    max_patience_aft: int = 30,
    min_stop: int = 100,
    random_seed: int = 42,
    device: Optional[str] = None,
    regularization_acceleration: bool = True,
    edge_subset_sz: int = 1000000,
    fix_regularization_index: bool = False,
    key_added: str = "X_cosmos",
    verbose: bool = False,
):
    """COSMOS joint embedding of two spatial modalities (e.g. RNA + ATAC).

    Vendored PyTorch port (no torch_geometric) of COSMOS (Zhou et al.,
    *Nat Commun* 2025; MIT licence, UT Southwestern). Each modality is encoded
    by two GCN layers over a k-NN spatial graph; the two L2-normalised
    embeddings are combined with per-spot weights computed once by weighted
    nearest neighbours (WNN) at epoch ``wnn_epoch`` and the model is trained
    with a deep-graph-infomax loss plus a spatial-distance penalty.

    The keyword defaults are those of upstream ``Cosmos.train``. The upstream
    P22 mouse-brain ATAC-RNA tutorial uses ``spatial_regularization_strength=0.01,
    wnn_epoch=500, min_stop=200, random_seed=20`` with ``adata_omics1=RNA`` and
    ``adata_omics2=ATAC`` (features are ``X`` as shipped: 3,000 genes, 50 ATAC
    LSI components).

    Arguments:
        adata_omics1, adata_omics2: AnnData of the two modalities, same spots in
            the same order; ``obsm[spatial_key]`` is read from ``adata_omics1``.
        use_rep1, use_rep2: ``obsm`` keys used as features instead of ``X``.
        n_neighbors: k of the spatial k-NN graph (``preprocessing_data``).
        spatial_regularization_strength, z_dim, lr, wnn_epoch, total_epoch,
            max_patience_bef, max_patience_aft, min_stop, random_seed,
            regularization_acceleration, edge_subset_sz: as upstream ``train``.
        device: ``"cpu"``, ``"cuda:0"`` ...; default CUDA if available.
        fix_regularization_index: upstream draws both coordinate subsets of the
            accelerated spatial penalty with the same index, which makes that
            penalty zero. Default ``False`` keeps this behaviour; ``True`` uses
            the second index as evidently intended (results then differ from
            upstream).
        key_added: ``obsm`` key for the embedding in both AnnData objects;
            the per-spot modality weights go to ``obsm[key_added + "_weights"]``.

    Returns:
        The ``(n_spots, z_dim)`` embedding (also stored).

    Reference:
        Zhou Y, Xiao X, Dong L, Tang C, Xiao G, Xu L. Cooperative integration of
        spatially resolved multi-omics data with COSMOS. Nat Commun 16, 27 (2025).
    """
    import anndata as ad
    import torch
    from sklearn.neighbors import kneighbors_graph

    GraphEncoderWNNit, DeepGraphInfomaxWNN, corruption = _build_model_classes()
    dev = _resolve_device(device)

    X1 = _dense(adata_omics1.obsm[use_rep1] if use_rep1 else adata_omics1.X)
    X2 = _dense(adata_omics2.obsm[use_rep2] if use_rep2 else adata_omics2.X)
    if X1.shape[0] != X2.shape[0]:
        raise ValueError("adata_omics1 and adata_omics2 must have the same spots")
    spatial_locs = np.asarray(adata_omics1.obsm[spatial_key], dtype=float)
    # carrier AnnData for the WNN step (upstream passes adata1_preprocessed)
    carrier = ad.AnnData(np.zeros((X1.shape[0], 1), dtype=np.float32))
    carrier.obs_names = adata_omics1.obs_names.copy()

    # preprocessing_data(do_norm=False, do_log=False, n_top_genes=None, do_pca=False)
    spatial_graph = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode="distance")

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    model = DeepGraphInfomaxWNN(
        hidden_channels=z_dim,
        encoder=GraphEncoderWNNit(X1.shape[0], X1.shape[1], X2.shape[1], z_dim, dev),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(dev)

    expr1 = torch.tensor(X1.copy()).float().to(dev)
    expr2 = torch.tensor(X2.copy()).float().to(dev)
    g = spatial_graph.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(np.vstack((g.row, g.col)).astype(np.int64)).to(dev)

    model.train()
    min_loss = np.inf
    patience_aft = 0
    patience_bef = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_params = model.state_dict()
    w1 = 0.5
    w2 = 0.5
    params = {"n_neighbors": n_neighbors, "z_dim": z_dim, "lr": lr,
              "spatial_regularization_strength": spatial_regularization_strength,
              "wnn_epoch": wnn_epoch, "total_epoch": total_epoch, "random_seed": random_seed,
              "fix_regularization_index": fix_regularization_index}
    epoch = 0
    t0 = time.perf_counter()
    for epoch in range(1, total_epoch):
        train_loss = 0.0
        torch.set_grad_enabled(True)
        optimizer.zero_grad()
        if epoch == wnn_epoch or patience_bef > max_patience_bef:
            z, neg_z, summary, w1, w2 = model(expr1, expr2, edge_list, carrier, 1, 0, 0)
            wnn_epoch = 0
            min_loss = np.inf
            max_patience_bef = total_epoch
        else:
            z, neg_z, summary, w1, w2 = model(expr1, expr2, edge_list, carrier, 0, w1, w2)
        loss = model.loss(z, neg_z, summary)
        coords = torch.tensor(spatial_locs).float().to(dev)
        if spatial_regularization_strength > 0:
            if regularization_acceleration or X1.shape[0] > 5000:
                s1 = torch.randint(0, z.shape[0], (edge_subset_sz,)).to(dev)
                s2 = torch.randint(0, z.shape[0], (edge_subset_sz,)).to(dev)
                z1, z2 = torch.index_select(z, 0, s1), torch.index_select(z, 0, s2)
                c1 = torch.index_select(coords, 0, s1)
                c2 = torch.index_select(coords, 0, s2 if fix_regularization_index else s1)
                pdist = torch.nn.PairwiseDistance(p=2)
                z_dists = pdist(z1, z2)
                z_dists = z_dists / torch.max(z_dists)
                sp_dists = pdist(c1, c2)
                sp_dists = sp_dists / torch.max(sp_dists)
                n_items = z_dists.size(dim=0)
            else:
                z_dists = torch.cdist(z, z, p=2)
                z_dists = torch.div(z_dists, torch.max(z_dists)).to(dev)
                sp_dists = torch.cdist(coords, coords, p=2)
                sp_dists = torch.div(sp_dists, torch.max(sp_dists)).to(dev)
                n_items = z.size(dim=0) * z.size(dim=0)
            penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(dev)
        else:
            penalty_1 = 0
        loss = loss + spatial_regularization_strength * penalty_1
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if epoch > wnn_epoch:
            if train_loss > min_loss:
                patience_aft += 1
            else:
                patience_aft = 0
                min_loss = train_loss
                best_params = model.state_dict()
        else:
            if train_loss > min_loss:
                patience_bef += 1
            else:
                patience_bef = 0
                min_loss = train_loss
                best_params = model.state_dict()
        if verbose and epoch % 10 == 1:
            print(f"Epoch {epoch}/{total_epoch}, Loss: {train_loss}")
        if patience_aft > max_patience_aft and epoch > min_stop:
            break

    model.load_state_dict(best_params)
    z, _, _, w1, w2 = model(expr1, expr2, edge_list, carrier, 0, w1, w2)
    embedding = z.cpu().detach().numpy()
    ww = np.hstack((w1.cpu().detach().numpy().reshape(-1, 1), w2.cpu().detach().numpy().reshape(-1, 1)))
    elapsed = time.perf_counter() - t0
    for a in (adata_omics1, adata_omics2):
        a.obsm[key_added] = embedding
        a.obsm[f"{key_added}_weights"] = ww
        a.uns[key_added] = {"method": "COSMOS", "epochs_run": int(epoch), "train_seconds": elapsed,
                            "params": params}
    return embedding
