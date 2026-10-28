# SPDX-License-Identifier: MIT
#
# Vendored port of PRESENT (Li, Cui, Chen, Gao, Liu, Pan, Chen & Jiang,
# "Cross-modality representation and multi-sample integration of spatially
# resolved omics data", Briefings in Bioinformatics 27, bbag214, 2026).
# Upstream: https://github.com/lizhen18THU/PRESENT (PyPI: bio-present)
#           commit c88a609b34aae9b84c2c8a7ffb6824bae5c78f23
# Copyright (c) 2024 Zhen Li, MIT licence (full text in
# epione/tl/_licenses/PRESENT.LICENSE and the repository NOTICE file).
# PRESENT's GATConv is itself adapted from PyTorch Geometric (MIT, PyG Team).
#
# Changes from upstream (see tests/parity/spatial_present/MANIFEST.md):
#   * torch_geometric (GATConv / MessagePassing, ClusterData + ClusterLoader,
#     ToUndirected, dense_to_sparse, Data) is replaced by plain torch:
#     epione.tl._gnn.GATConv and the Cluster-GCN batching below, written to
#     reproduce PyG 2.6 edge order, partition permutation and loader RNG use;
#   * METIS partitioning uses ``torch.ops.torch_sparse.partition`` when the
#     optional torch-sparse package is importable (exactly what PyG calls);
#     otherwise a recursive coordinate bisection of the spot coordinates into
#     the same number of balanced parts (documented deviation, results differ);
#   * episcanpy.pp.filter_features / filter_cells are the scanpy calls they wrap;
#   * genomicranges-based peak_sets_alignment (reference-data path only) is
#     re-implemented with numpy;
#   * PRESENT_function works on copies of the inputs instead of filtering
#     them in place, takes ``random_seed`` (upstream does not seed), and writes
#     results back to the input AnnData; tqdm bars are optional.
"""PRESENT: Bayesian GAT + ZINB/ZIP decoders + contrastive alignment for spatial omics."""
from __future__ import annotations

import os
import random
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy
import scipy.sparse
from scipy.stats.mstats import gmean

__all__ = ["present", "present_preprocess_peaks"]


# --------------------------------------------------------------------------
# Utils.py
# --------------------------------------------------------------------------
def _setup_seed(seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _data2input(data):
    import torch

    if scipy.sparse.issparse(data):
        data = data.toarray()
    if not isinstance(data, torch.Tensor):
        data = torch.LongTensor(data) if str(data.dtype).startswith("int") else torch.FloatTensor(data)
    return data


def TFIDF(count_mat, type_=2):
    """TF-IDF of a peak x cell matrix (upstream ``Utils.TFIDF``; type 2 = Signac)."""
    def tfidf1(count_mat):
        if not scipy.sparse.issparse(count_mat):
            count_mat = scipy.sparse.coo_matrix(count_mat)
        nfreqs = count_mat.multiply(1.0 / count_mat.sum(axis=0))
        tfidf_mat = nfreqs.multiply(np.log(1 + 1.0 * count_mat.shape[1] / count_mat.sum(axis=1)).reshape(-1, 1)).tocoo()
        return scipy.sparse.csr_matrix(tfidf_mat)

    def tfidf2(count_mat):
        if not scipy.sparse.issparse(count_mat):
            count_mat = scipy.sparse.coo_matrix(count_mat)
        tf_mat = count_mat.multiply(1.0 / count_mat.sum(axis=0))
        signac_mat = (1e4 * tf_mat).multiply(1.0 * count_mat.shape[1] / count_mat.sum(axis=1).reshape(-1, 1))
        signac_mat = signac_mat.log1p()
        return scipy.sparse.csr_matrix(signac_mat)

    def tfidf3(count_mat):
        from sklearn.feature_extraction.text import TfidfTransformer

        model = TfidfTransformer(smooth_idf=False, norm="l2")
        model = model.fit(np.transpose(count_mat))
        model.idf_ -= 1
        tf_idf = np.transpose(model.transform(np.transpose(count_mat)))
        return scipy.sparse.csr_matrix(tf_idf)

    if type_ == 1:
        return tfidf1(count_mat)
    elif type_ == 2:
        return tfidf2(count_mat)
    return tfidf3(count_mat)


def _CLR_transform(ADT_mat):
    ADT_mat_processed = ADT_mat.todense() if scipy.sparse.issparse(ADT_mat) else ADT_mat.copy()
    gmean_list = []
    for i in range(ADT_mat_processed.shape[0]):
        temp = [ADT_mat_processed[i, j] for j in range(ADT_mat_processed.shape[1]) if not ADT_mat_processed[i, j] == 0]
        gmean_temp = gmean(temp)
        gmean_list.append(gmean_temp)
        for j in range(ADT_mat_processed.shape[1]):
            if not ADT_mat_processed[i, j] == 0:
                ADT_mat_processed[i, j] = np.log(ADT_mat_processed[i, j] / gmean_temp)
    return scipy.sparse.csr_matrix(ADT_mat_processed), gmean_list


def _construct_pseudo_bulk(adata_ref, key, min_samples=11):
    import scanpy as sc

    index_list = []
    key_index = adata_ref.obs[key].values
    bulks = np.unique(key_index)
    if min_samples is not None:
        min_samples += 1
    if min_samples is not None and min_samples > bulks.shape[0]:
        times = min_samples // bulks.shape[0]
        r = (1 / times + 1) / 2
        for item in bulks:
            cur_index = np.argwhere(key_index == item).reshape(-1)
            index_list.append(cur_index)
            length = cur_index.shape[0]
            for i in range(times):
                shuffle_index = cur_index.copy()
                np.random.shuffle(shuffle_index)
                index_list.append(shuffle_index[0:max(1, int(length * r))])
    else:
        for item in bulks:
            index_list.append(np.argwhere(key_index == item).reshape(-1))
    rows = [np.average(adata_ref[index, :].X.toarray(), axis=0).reshape(1, -1) for index in index_list]
    return sc.AnnData(np.concatenate(rows, axis=0), var=adata_ref.var.copy())


def _optim_parameters(net, included=None, excluded=None):
    def belongs_to(cur_layer, layers):
        return any(layer in cur_layer for layer in layers)

    params = []
    if included is not None:
        included = included if isinstance(included, list) else [included]
        for cur_layer, param in net.named_parameters():
            if belongs_to(cur_layer, included) and param.requires_grad:
                params.append(param)
    else:
        excluded = (excluded if isinstance(excluded, list) else [excluded]) if excluded is not None else []
        for cur_layer, param in net.named_parameters():
            if not belongs_to(cur_layer, excluded) and param.requires_grad:
                params.append(param)
    return iter(params)


class _EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience, self.delta = patience, delta
        self.counter, self.best_loss, self.early_stop = 0, None, False

    def __call__(self, train_loss):
        if self.best_loss is None:
            self.best_loss = train_loss
            self.counter = 0
        elif self.best_loss - train_loss < self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = train_loss
            self.counter = 0


def peak_sets_alignment(adata_list, sep=(":", "-"), min_width=20, max_width=10000, min_gap_width=1):
    """Upstream ``Utils.peak_sets_alignment`` (identical to INSTINCT's); numpy re-implementation."""
    from ._instinct import peak_sets_alignment as _psa

    return _psa(adata_list, sep=sep, min_width=min_width, max_width=max_width, min_gap_width=min_gap_width)


def _gene_sets_alignment(adata_list):
    for i in range(len(adata_list)):
        adata_list[i].var_names = [item.lower() for item in adata_list[i].var_names]
        adata_list[i].var_names_make_unique()
    common = np.intersect1d(adata_list[0].var_names, adata_list[1].var_names)
    for i in range(2, len(adata_list)):
        common = np.intersect1d(common, adata_list[i].var_names)
    return [a[:, common] for a in adata_list]


def _ref_feature_alignment(adata_ref, adata, omics=None):
    if omics is None or omics.upper() in ["RNA", "ADT"]:
        return _gene_sets_alignment([adata_ref, adata])
    return peak_sets_alignment([adata_ref, adata])


def _StrLabel2Idx(string_labels):
    from sklearn.preprocessing import LabelEncoder

    return np.array(LabelEncoder().fit_transform(string_labels))


def _Integrated_3D_graph(batch_label, spatial_mat, joint_mat=None, rna_mat=None, cas_mat=None, adt_mat=None,
                         intra_neighbors=6, intra_metric="euclidean", inter_neighbors=6, inter_metric="cosine"):
    from sklearn.neighbors import NearestNeighbors

    def find_intra_neighbor(matrix, k, metric="euclidean"):
        k = min(matrix.shape[0] - 1, k)
        if k > 0:
            nn_model = NearestNeighbors(n_neighbors=k, algorithm="auto", metric=metric).fit(matrix)
            _, indices = nn_model.kneighbors(matrix, n_neighbors=k + 1)
            return indices[:, 1:]
        return None

    def find_inter_neighbor(matrix1, matrix2, k, metric="cosine"):
        if matrix1.shape[1] != matrix2.shape[1]:
            raise ValueError("invalid feature dimension")
        k = min(matrix2.shape[0], k)
        if k > 0:
            nn_model = NearestNeighbors(n_neighbors=k, algorithm="auto", metric=metric).fit(matrix2)
            _, indices = nn_model.kneighbors(matrix1, n_neighbors=k)
            return indices
        return None

    row_index, col_index, data = [], [], []
    N = spatial_mat.shape[0]
    batch_label = np.asarray(batch_label)
    samples_indices = np.arange(N)
    for cur in np.unique(batch_label):
        cur_idx = batch_label == cur
        cur_original_idx = samples_indices[cur_idx]
        cur_spatial_mat = spatial_mat[cur_idx]
        indices = find_intra_neighbor(cur_spatial_mat, k=intra_neighbors, metric=intra_metric)
        intra_indices = cur_original_idx[indices] if indices is not None else None
        other_idx = batch_label != cur
        other_original_idx = samples_indices[other_idx]
        inter_indices = []
        for m in (joint_mat, rna_mat, cas_mat, adt_mat):
            if m is None:
                continue
            if scipy.sparse.issparse(m):
                m = m.toarray()
            idx = find_inter_neighbor(m[cur_idx], m[other_idx], k=inter_neighbors, metric=inter_metric)
            if idx is not None:
                inter_indices.append(other_original_idx[idx])
        inter_indices = np.concatenate(inter_indices, axis=1) if len(inter_indices) > 0 else None
        for i in range(cur_spatial_mat.shape[0]):
            if intra_indices is not None:
                cur_list = intra_indices[i].tolist()
                row_index += cur_list
                col_index += [cur_original_idx[i]] * len(cur_list)
                data += [1] * len(cur_list)
            if inter_indices is not None:
                cur_list = list(set(inter_indices[i].tolist()))
                row_index += cur_list
                col_index += [cur_original_idx[i]] * len(cur_list)
                data += [-1] * len(cur_list)
    return scipy.sparse.coo_matrix((data, (row_index, col_index)), shape=(N, N))


# --------------------------------------------------------------------------
# graph data + Cluster-GCN batching (replaces torch_geometric.loader)
# --------------------------------------------------------------------------
class _Graph:
    """Minimal stand-in for ``torch_geometric.data.Data`` (edge_index, edge_attr, node_index)."""

    def __init__(self, edge_index, edge_attr, node_index, num_nodes):
        self.edge_index, self.edge_attr, self.node_index, self.num_nodes = edge_index, edge_attr, node_index, num_nodes


def _geometric_data(graph, node_index):
    """``Utils.GeometricData`` + ``ToUndirected()``: dense -> edge list, symmetrised (sum)."""
    import torch

    g = _data2input(graph)
    edge_index = g.nonzero().t()
    edge_attr = g[edge_index[0], edge_index[1]]
    n = g.size(0)
    row = torch.cat([edge_index[0], edge_index[1]])
    col = torch.cat([edge_index[1], edge_index[0]])
    attr = torch.cat([edge_attr, edge_attr])
    idx = row * n + col
    idx, perm = idx.sort()
    row, col, attr = row[perm], col[perm], attr[perm]
    mask = torch.ones(idx.numel(), dtype=torch.bool)
    mask[1:] = idx[1:] > idx[:-1]
    seg = torch.cumsum(mask.long(), 0) - 1
    out_attr = attr.new_zeros(int(mask.sum())).scatter_add_(0, seg, attr)
    return _Graph(torch.stack([row[mask], col[mask]]), out_attr, _data2input(node_index), n)


def _index2ptr(index, size):
    import torch

    counts = torch.bincount(index, minlength=size)
    return torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])


def _coordinate_bisection(coords, num_parts):
    """Balanced recursive coordinate bisection (fallback when METIS is unavailable)."""
    labels = np.zeros(coords.shape[0], dtype=np.int64)

    def split(idx, parts, offset):
        if parts == 1 or idx.size == 0:
            labels[idx] = offset
            return
        left_parts = parts // 2
        span = coords[idx].max(0) - coords[idx].min(0)
        ax = int(np.argmax(span))
        order = idx[np.argsort(coords[idx, ax], kind="stable")]
        cut = int(round(idx.size * left_parts / parts))
        split(order[:cut], left_parts, offset)
        split(order[cut:], parts - left_parts, offset + left_parts)

    split(np.arange(coords.shape[0]), int(num_parts), 0)
    return labels


def _metis(edge_index, num_nodes, num_parts, coords=None):
    import torch

    order = (edge_index[0] * num_nodes + edge_index[1]).argsort()
    row, col = edge_index[0][order], edge_index[1][order]
    indptr = _index2ptr(row, num_nodes)
    try:
        import torch_sparse  # noqa: F401  (registers torch.ops.torch_sparse)

        return torch.ops.torch_sparse.partition(indptr.cpu(), col.cpu(), None, num_parts, False), "metis"
    except (ImportError, AttributeError, RuntimeError, OSError):
        pass
    if coords is None:
        raise ImportError("PRESENT mini-batching needs coordinates or torch-sparse (METIS)")
    return torch.from_numpy(_coordinate_bisection(np.asarray(coords, float), num_parts)), "coordinate_bisection"


class _ClusterBatches:
    """``ClusterData(recursive=False)`` + ``ClusterLoader(batch_size, shuffle=True)`` of PyG 2.6."""

    def __init__(self, graph: _Graph, num_parts: int, coords=None):
        import torch

        n = graph.num_nodes
        cluster, self.partitioner = _metis(graph.edge_index, n, num_parts, coords)
        cluster, node_perm = cluster.sort(stable=False)
        self.partptr = _index2ptr(cluster, num_parts)
        arange = torch.empty_like(node_perm)
        arange[node_perm] = torch.arange(cluster.numel())
        ei = arange[graph.edge_index]
        edge_perm = torch.arange(ei.size(1))
        key = ei[0] * n + ei[1]
        key, perm = key.sort()
        row, col = ei[0][perm], ei[1][perm]
        self.edge_attr = graph.edge_attr[edge_perm[perm]]
        self.indptr = _index2ptr(row, n)
        self.index = col
        self.node_index = graph.node_index[node_perm]
        self.num_parts = int(num_parts)
        self.n = n

    def collate(self, batch):
        import torch

        batch = torch.tensor(batch)
        node_start = self.partptr[batch]
        node_end = self.partptr[batch + 1]
        edge_start = self.indptr[node_start]
        edge_end = self.indptr[node_end]
        rows, cols, nodes, attrs, cumsum = [], [], [], [], 0
        for i in range(batch.numel()):
            nodes.append(torch.arange(node_start[i], node_end[i]))
            indptr = self.indptr[node_start[i]:node_end[i] + 1] - edge_start[i]
            counts = indptr[1:] - indptr[:-1]
            rows.append(torch.repeat_interleave(torch.arange(counts.numel()), counts) + cumsum)
            cols.append(self.index[edge_start[i]:edge_end[i]])
            attrs.append(self.edge_attr[edge_start[i]:edge_end[i]])
            cumsum += indptr.numel() - 1
        node = torch.cat(nodes)
        row = torch.cat(rows)
        col = torch.cat(cols)
        attr = torch.cat(attrs)
        lut = torch.full((self.n,), -1, dtype=torch.long)
        lut[node] = torch.arange(node.numel())
        mapped = lut[col]
        mask = mapped >= 0
        node_index = torch.cat([self.node_index[s:e] for s, e in zip(node_start, node_end)])
        return _Graph(torch.stack([row[mask], mapped[mask]]), attr[mask], node_index, cumsum)

    def loader(self, batch_size=10, shuffle=True):
        import torch

        return torch.utils.data.DataLoader(range(self.num_parts), batch_size=batch_size, shuffle=shuffle,
                                           collate_fn=self.collate)


# --------------------------------------------------------------------------
# Layers.py
# --------------------------------------------------------------------------
def _build_layers():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Poisson
    from torch.nn.parameter import Parameter

    from ._gnn import GATConv

    class BayesianLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.in_features, self.out_features = in_features, out_features
            self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
            self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))
            self.weight_eps = None

        def reset_parameters(self, prior_mu, prior_log_sigma):
            self.weight_mu.data = prior_mu
            self.weight_log_sigma.data = torch.ones_like(self.weight_log_sigma.data) * prior_log_sigma.to(self.weight_log_sigma.data.device)
            self.prior_mu = prior_mu
            self.prior_log_sigma = prior_log_sigma

        def freeze(self):
            self.weight_eps = torch.randn_like(self.weight_log_sigma)

        def unfreeze(self):
            self.weight_eps = None

        def _kld_loss(self, mu_0, log_sigma_0, mu_1, log_sigma_1):
            kl = log_sigma_1 - log_sigma_0 + (torch.exp(log_sigma_0) ** 2 + (mu_0 - mu_1) ** 2) / (2 * torch.exp(log_sigma_1) ** 2) - 0.5
            return kl.mean()

        def bayesian_kld_loss(self):
            device = self.weight_mu.data.device
            return self._kld_loss(self.weight_mu, self.weight_log_sigma, self.prior_mu.to(device), self.prior_log_sigma.to(device))

        def forward(self, x):
            device = self.weight_mu.data.device
            if self.weight_eps is None:
                weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma).to(device)
            else:
                weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps.to(device)
            return F.linear(x, weight, None)

    def _act(activ):
        if activ == "elu":
            return nn.ELU(inplace=True)
        if activ == "gelu":
            return nn.GELU(approximate="tanh")
        if activ == "leaky_relu":
            return nn.LeakyReLU(inplace=True)
        return nn.ReLU(inplace=True)

    def _norm(norm, d):
        return nn.LayerNorm(d, eps=1e-6) if norm == "LayerNorm" else nn.BatchNorm1d(d, eps=1e-6)

    class MLP_Module(nn.Module):
        def __init__(self, d_in, d_hid, d_out, norm="LayerNorm", activ="elu"):
            super().__init__()
            self.activation = _act(activ)
            self.module_list = nn.ModuleList([nn.Linear(d_in, d_hid[0])])
            self.module_list.append(_norm(norm, d_hid[0]))
            for i in range(1, len(d_hid)):
                self.module_list.append(nn.Linear(d_hid[i - 1], d_hid[i]))
                self.module_list.append(_norm(norm, d_hid[i]))
            self.output_layer = nn.Linear(d_hid[-1], d_out)

        def forward(self, x):
            for i in range(0, len(self.module_list), 2):
                x = self.module_list[i](x)
                x = self.module_list[i + 1](x)
                x = self.activation(x)
            return self.output_layer(x)

    class BayesianGATEncoder(nn.Module):
        def __init__(self, d_in, d_out, d_hid=(1024, 512), dropout=0.1, norm="LayerNorm", activ="elu"):
            super().__init__()
            self.d_prior = d_out // 3
            self.d_in, self.d_out = d_in, d_out
            self.activation = _act(activ)
            self.bayesian_fc = BayesianLinear(d_in, self.d_prior)
            self.enc_fc = nn.Linear(d_in, d_hid[0])
            self.enc_norm1 = _norm(norm, d_hid[0])
            self.enc_gatconv1 = GATConv(in_channels=d_hid[0], out_channels=d_hid[1], dropout=dropout)
            self.enc_norm2 = _norm(norm, d_hid[1])
            self.enc_gatconv2 = GATConv(in_channels=d_hid[1], out_channels=d_out - self.d_prior, dropout=dropout)
            self.enc_norm3 = _norm(norm, d_out)

        def prior_initialize(self, prior, tight_factor=10):
            if not isinstance(prior, torch.FloatTensor):
                prior = torch.FloatTensor(prior)
            assert prior.shape[0] == self.d_prior, "prior weight dimension not match"
            prior_log_sigma = torch.log(prior.std() / tight_factor)
            self.bayesian_fc.reset_parameters(prior, prior_log_sigma)

        def bnn_loss(self):
            return self.bayesian_fc.bayesian_kld_loss()

        def freeze(self):
            self.bayesian_fc.freeze()

        def unfreeze(self):
            self.bayesian_fc.unfreeze()

        def forward(self, x, edge_index):
            bayesian_out = self.bayesian_fc(x)
            x = self.enc_fc(x)
            x = self.enc_norm1(x)
            x = self.activation(x)
            x = self.enc_gatconv1(x, edge_index, return_attention_weights=None)
            x = self.enc_norm2(x)
            x = self.activation(x)
            x = self.enc_gatconv2(x, edge_index, return_attention_weights=None)
            x = torch.cat([x, bayesian_out], -1)
            return self.enc_norm3(x)

    class _CountDecoder(nn.Module):
        def __init__(self, d_in, d_hid, norm, activ, basic_module, dropout):
            super().__init__()
            self.basic_module = basic_module
            self.activation = _act(activ)
            first = nn.Linear(d_in, d_hid[0]) if basic_module == "Linear" else GATConv(in_channels=d_in, out_channels=d_hid[0], dropout=dropout)
            self.module_list = nn.ModuleList([first])
            self.module_list.append(_norm(norm, d_hid[0]))
            for i in range(1, len(d_hid)):
                if basic_module == "Linear":
                    self.module_list.append(nn.Linear(d_hid[i - 1], d_hid[i]))
                else:
                    self.module_list.append(GATConv(in_channels=d_hid[i - 1], out_channels=d_hid[i], dropout=dropout))
                self.module_list.append(_norm(norm, d_hid[i]))

        def _trunk(self, x, edge_index):
            for i in range(0, len(self.module_list), 2):
                x = self.module_list[i](x) if self.basic_module == "Linear" else self.module_list[i](x, edge_index)
                x = self.module_list[i + 1](x)
                x = self.activation(x)
            return x

    class ZINBDecoder(_CountDecoder):
        def __init__(self, d_in, d_hid, d_out, zero_inflaten=True, is_recons=False, norm="LayerNorm", activ="elu", basic_module="Linear", dropout=0.1):
            super().__init__(d_in, d_hid, norm, activ, basic_module, dropout)
            self.zero_inflaten, self.is_recons = zero_inflaten, is_recons
            self.pi_output = nn.Linear(d_hid[-1], d_out) if zero_inflaten else None
            self.disp_output = nn.Linear(d_hid[-1], d_out)
            self.mean_output = nn.Linear(d_hid[-1], d_out)
            self.expr_output = nn.Linear(d_hid[-1], d_out) if is_recons else None

        def forward(self, x, edge_index=None):
            x = self._trunk(x, edge_index)
            pi_out = F.sigmoid(self.pi_output(x)) if self.zero_inflaten else None
            disp_out = torch.clamp(F.softplus(self.disp_output(x)), 1e-4, 1e4)
            mean_out = torch.clamp(F.softplus(self.mean_output(x)), 1e-5, 1e6)
            recons_expr = self.expr_output(x) if self.is_recons else None
            return pi_out, disp_out, mean_out, recons_expr

    class ZIPDecoder(_CountDecoder):
        def __init__(self, d_in, d_hid, d_out, zero_inflaten=True, is_recons=False, norm="LayerNorm", activ="elu", basic_module="Linear", dropout=0.1):
            super().__init__(d_in, d_hid, norm, activ, basic_module, dropout)
            self.zero_inflaten, self.is_recons = zero_inflaten, is_recons
            self.pi_output = nn.Linear(d_hid[-1], d_out) if zero_inflaten else None
            self.rho_output = nn.Linear(d_hid[-1], d_out)
            self.expr_output = nn.Linear(d_hid[-1], d_out) if is_recons else None
            self.peak_bias = nn.Parameter(torch.randn(1, d_out))

        def forward(self, x, edge_index=None):
            x = self._trunk(x, edge_index)
            pi = F.sigmoid(self.pi_output(x)) if self.zero_inflaten else None
            rho = self.rho_output(x)
            omega = F.softmax(rho + self.peak_bias, dim=-1)
            recons_expr = self.expr_output(x) if self.is_recons else None
            return pi, None, omega, recons_expr

    def ZINBLoss(x, pi, disp, mean, scale_factor=1.0, ridge_lambda=0.5):
        eps = 1e-10
        mean = mean * scale_factor
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        if pi is not None:
            nb_final = t1 + t2
            nb_case = nb_final - torch.log(1.0 - pi + eps)
            zero_nb = torch.pow(disp / (disp + mean + eps), disp)
            zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
            result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
            if ridge_lambda > 0:
                result += ridge_lambda * torch.square(pi)
            return torch.mean(result)
        return torch.mean(t1 + t2)

    def ZIPLoss(x, pi, omega, scale_factor=1.0, ridge_lambda=0.5):
        eps = 1e-10
        lamb = omega * scale_factor
        if pi is not None:
            po_case = -Poisson(lamb).log_prob(x) - torch.log(1.0 - pi + eps)
            zero_po = torch.exp(-lamb)
            zero_case = -torch.log(pi + ((1.0 - pi) * zero_po) + eps)
            result = torch.where(torch.le(x, 1e-8), zero_case, po_case)
            if ridge_lambda > 0:
                result += ridge_lambda * torch.square(pi)
            return torch.mean(result)
        return torch.mean(-Poisson(lamb).log_prob(x))

    def NLL_loss(x, pi, param1, param2, scale_factor=1.0, ridge_lambda=0.5):
        if param1 is not None:
            return ZINBLoss(x, pi, param1, param2, scale_factor, ridge_lambda)
        return ZIPLoss(x, pi, param2, scale_factor, ridge_lambda)

    def IOA_loss(rna_enc_out, cas_enc_out, adt_enc_out, ori_fused_out, tau=0.1, times=10, sample_size=None):
        ioa_loss = 0
        size = ori_fused_out.shape[0]
        if sample_size is None:
            sample_size = 5 * size // times
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        num_omics = (rna_enc_out is not None) + (cas_enc_out is not None) + (adt_enc_out is not None)
        if num_omics < 1:
            return 0
        for i in range(times):
            if sample_size == "all" or sample_size > size:
                f, r, c, a = ori_fused_out, rna_enc_out, cas_enc_out, adt_enc_out
            else:
                index = random.sample(list(range(size)), sample_size)
                f = ori_fused_out[index, :]
                r = rna_enc_out[index, :] if rna_enc_out is not None else None
                c = cas_enc_out[index, :] if cas_enc_out is not None else None
                a = adt_enc_out[index, :] if adt_enc_out is not None else None
            log_fused = F.log_softmax(torch.matmul(f, f.t()) / tau, dim=1)
            for e in (r, c, a):
                if e is not None:
                    ioa_loss += kl_loss(log_fused, F.softmax(torch.matmul(e, e.t()) / tau, dim=1)) / num_omics
        return ioa_loss / times

    def IBA_loss(xlat_batch, positive_indices, negative_indices, tau=0.1):
        loss, count = 0, 0
        for i in range(xlat_batch.shape[0]):
            if positive_indices[i] is not None and negative_indices[i] is not None:
                above = torch.exp(F.cosine_similarity(xlat_batch[i:i + 1], xlat_batch[positive_indices[i]]) / tau).sum()
                below = torch.exp(F.cosine_similarity(xlat_batch[i:i + 1], xlat_batch[negative_indices[i]]) / tau).sum()
                loss += -torch.log(above / (above + below))
                count += 1
        return loss / count if count > 0 else 0

    def IBP_loss(xlat_batch, target_batch, batch_index, tau=0.1):
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loss, count = 0, 0
        for cur_batch in torch.unique(batch_index):
            idx = batch_index == cur_batch
            if idx.sum() > 1:
                cx, ct = xlat_batch[idx], target_batch[idx]
                loss += kl_loss(F.log_softmax(torch.matmul(cx, cx.t()) / tau, dim=1), F.softmax(torch.matmul(ct, ct.t()) / tau, dim=1))
                count += 1
        return loss / count if count > 0 else 0

    return dict(BayesianGATEncoder=BayesianGATEncoder, MLP_Module=MLP_Module, ZINBDecoder=ZINBDecoder,
                ZIPDecoder=ZIPDecoder, NLL_loss=NLL_loss, IOA_loss=IOA_loss, IBA_loss=IBA_loss, IBP_loss=IBP_loss)


# --------------------------------------------------------------------------
# Model.py
# --------------------------------------------------------------------------
def _progress(it, show, desc):
    if show:
        try:
            from tqdm import tqdm

            bar = tqdm(it)
            bar.set_description(desc)
            return bar
        except ImportError:  # pragma: no cover
            pass
    return it


def _prepare_omics(self, rna_counts, cas_counts, adt_counts, ref_rna_counts, ref_rna_anno,
                   ref_cas_counts, ref_cas_anno, ref_adt_counts, ref_adt_anno):
    """Normalisation + PCA priors shared by PRESENT_RP / PRESENT_BC.model_train (verbatim)."""
    import scanpy as sc
    import torch

    csr = scipy.sparse.csr_matrix
    if self.is_rna:
        assert rna_counts is not None and rna_counts.shape[1] == self.rna_dim, "Invalid shape of input rna_counts"
        rna_counts = csr(rna_counts) if not scipy.sparse.issparse(rna_counts) else rna_counts
        if ref_rna_counts is not None:
            ref_rna_counts = csr(ref_rna_counts) if not scipy.sparse.issparse(ref_rna_counts) else ref_rna_counts
    if self.is_cas:
        assert cas_counts is not None and cas_counts.shape[1] == self.cas_dim, "Invalid shape of input cas_counts"
        cas_counts = csr(cas_counts) if not scipy.sparse.issparse(cas_counts) else cas_counts
        if ref_cas_counts is not None:
            ref_cas_counts = csr(ref_cas_counts) if not scipy.sparse.issparse(ref_cas_counts) else ref_cas_counts
    if self.is_adt:
        assert adt_counts is not None and adt_counts.shape[1] == self.adt_dim, "Invalid shape of input adt_counts"
        adt_counts = csr(adt_counts) if not scipy.sparse.issparse(adt_counts) else adt_counts
        if ref_adt_counts is not None:
            ref_adt_counts = csr(ref_adt_counts) if not scipy.sparse.issparse(ref_adt_counts) else ref_adt_counts

    rna_prior = rna_norm = rna_libsize = None
    if self.is_rna:
        sdata = sc.AnnData(rna_counts.copy())
        sc.pp.normalize_total(sdata)
        sc.pp.log1p(sdata)
        if ref_rna_counts is not None:
            rdata = sc.AnnData(ref_rna_counts.copy())
            sc.pp.normalize_total(rdata)
            sc.pp.log1p(rdata)
            if ref_rna_anno is not None:
                rdata.obs["anno"] = np.array(ref_rna_anno)
                rdata = _construct_pseudo_bulk(rdata, key="anno", min_samples=self.d_prior)
            sc.tl.pca(rdata, n_comps=self.d_prior)
            rna_prior = torch.FloatTensor(rdata.varm["PCs"].T.copy())
        else:
            sc.tl.pca(sdata, n_comps=self.d_prior)
            rna_prior = torch.FloatTensor(sdata.varm["PCs"].T.copy())
        rna_norm = sdata.X.copy()
        dense = rna_counts.toarray()
        rna_libsize = (dense.sum(axis=1) / np.median(dense.sum(axis=1))).reshape(-1, 1)
    cas_prior = cas_norm = cas_libsize = None
    if self.is_cas:
        sdata = sc.AnnData(cas_counts.copy())
        sdata.X = TFIDF(sdata.X.T).T.copy()
        if ref_cas_counts is not None:
            rdata = sc.AnnData(ref_cas_counts.copy())
            rdata.X = TFIDF(rdata.X.T).T.copy()
            if ref_cas_anno is not None:
                rdata.obs["anno"] = np.array(ref_cas_anno)
                rdata = _construct_pseudo_bulk(rdata, key="anno", min_samples=self.d_prior)
            sc.tl.pca(rdata, n_comps=self.d_prior)
            cas_prior = torch.FloatTensor(rdata.varm["PCs"].T.copy())
        else:
            sc.tl.pca(sdata, n_comps=self.d_prior)
            cas_prior = torch.FloatTensor(sdata.varm["PCs"].T.copy())
        cas_norm = sdata.X.copy()
        cas_libsize = cas_counts.toarray().sum(axis=1).reshape(-1, 1)
    adt_prior = adt_norm = None
    if self.is_adt:
        sdata = sc.AnnData(adt_counts.copy())
        sdata.X, _ = _CLR_transform(sdata.X)
        sc.pp.scale(sdata)
        if ref_adt_counts is not None:
            rdata = sc.AnnData(ref_adt_counts.copy())
            rdata.X, _ = _CLR_transform(rdata.X)
            sc.pp.scale(rdata)
            if ref_adt_anno is not None:
                rdata.obs["anno"] = np.array(ref_adt_anno)
                rdata = _construct_pseudo_bulk(rdata, key="anno", min_samples=self.d_prior_adt)
            sc.tl.pca(rdata, n_comps=self.d_prior_adt)
            adt_prior = torch.FloatTensor(rdata.varm["PCs"].T.copy())
        else:
            sc.tl.pca(sdata, n_comps=self.d_prior_adt)
            adt_prior = torch.FloatTensor(sdata.varm["PCs"].T.copy())
        adt_norm = sdata.X.copy()
    return (rna_counts, cas_counts, adt_counts, rna_prior, rna_norm, rna_libsize,
            cas_prior, cas_norm, cas_libsize, adt_prior, adt_norm)


def _batch_inputs(self, np_index, device, rna_counts, rna_norm, rna_libsize, cas_counts, cas_norm, cas_libsize, adt_norm):
    d = _data2input
    r = (d(rna_counts[np_index]).to(device), d(rna_norm[np_index]).to(device), d(rna_libsize[np_index]).to(device)) if self.is_rna else (None, None, None)
    c = (d(cas_counts[np_index]).to(device), d(cas_norm[np_index]).to(device), d(cas_libsize[np_index]).to(device)) if self.is_cas else (None, None, None)
    a = d(adt_norm[np_index]).to(device) if self.is_adt else None
    return r, c, a


def _build_models():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.neighbors import kneighbors_graph

    L = _build_layers()
    BayesianGATEncoder, MLP_Module = L["BayesianGATEncoder"], L["MLP_Module"]
    ZINBDecoder, ZIPDecoder = L["ZINBDecoder"], L["ZIPDecoder"]
    NLL_loss, IOA_loss, IBA_loss, IBP_loss = L["NLL_loss"], L["IOA_loss"], L["IBA_loss"], L["IBP_loss"]

    class _Common(nn.Module):
        def prior_initialize(self, prior_rna, prior_cas, prior_adt, tight_factor=10):
            if self.is_rna:
                self.rna_encoder.prior_initialize(prior_rna, tight_factor)
            if self.is_cas:
                self.cas_encoder.prior_initialize(prior_cas, tight_factor)
            if self.is_adt:
                self.adt_encoder.prior_initialize(prior_adt, tight_factor)

        def bnn_loss(self):
            loss = 0
            if self.is_rna:
                loss += self.rna_encoder.bnn_loss()
            if self.is_cas:
                loss += self.cas_encoder.bnn_loss()
            if self.is_adt:
                loss += self.adt_encoder.bnn_loss()
            return loss

        def freeze(self):
            for e in (self.rna_encoder, self.cas_encoder, self.adt_encoder):
                if e is not None:
                    e.freeze()

        def unfreeze(self):
            for e in (self.rna_encoder, self.cas_encoder, self.adt_encoder):
                if e is not None:
                    e.unfreeze()

        def forward_encoder(self, rna, cas, adt, edge_index):
            out = []
            if self.is_rna:
                rna = self.rna_encoder(rna, edge_index)
                out.append(rna)
            if self.is_cas:
                cas = self.cas_encoder(cas, edge_index)
                out.append(cas)
            if self.is_adt:
                adt = self.adt_encoder(adt, edge_index)
                out.append(adt)
            if self.num_modalities > 1:
                out = self.fusion_layer(torch.cat(out, -1))
            else:
                out = out[0]
            return out, rna, cas, adt

    class PRESENT_RP(_Common):
        def __init__(self, rna_dim=None, cas_dim=None, adt_dim=None, d_lat=50, d_hid=(1024, 512),
                     rna_zero_inflaten=True, cas_zero_inflaten=True, is_recons=True, k_neighbors=6,
                     tau=0.1, dec_basic_module="Linear"):
            super().__init__()
            self.is_rna, self.is_cas, self.is_adt = rna_dim is not None, cas_dim is not None, adt_dim is not None
            self.num_modalities = self.is_adt + self.is_rna + self.is_cas
            self.rna_dim, self.cas_dim, self.adt_dim = rna_dim, cas_dim, adt_dim
            self.d_lat = d_lat
            self.d_prior = d_lat // 3
            self.d_lat_adt = d_lat_adt = min(d_lat, adt_dim) if adt_dim is not None else 0
            self.d_prior_adt = d_lat_adt // 3 if adt_dim is not None else 0
            self.tau, self.k_neighbors, self.is_recons, self.dec_basic_module = tau, k_neighbors, is_recons, dec_basic_module
            assert self.num_modalities > 0, "number of omics layer should be larger than 0"
            assert k_neighbors > 0, "k_neighbors should be larger than 0"
            self.rna_encoder = BayesianGATEncoder(rna_dim, d_lat, d_hid=d_hid) if self.is_rna else None
            self.cas_encoder = BayesianGATEncoder(cas_dim, d_lat, d_hid=d_hid) if self.is_cas else None
            self.adt_encoder = BayesianGATEncoder(adt_dim, d_lat_adt, d_hid=(2 * adt_dim // 3 + d_lat_adt // 3, adt_dim // 3 + 2 * d_lat_adt // 3)) if self.is_adt else None
            nin = int(self.is_rna + self.is_cas) * d_lat + d_lat_adt
            self.fusion_layer = MLP_Module(nin, (nin, d_lat), d_lat) if self.num_modalities > 1 else None
            if self.num_modalities == 1 and self.is_adt:
                d_lat = d_lat_adt
            self.rna_decoder = ZINBDecoder(d_lat, (d_hid[1], d_hid[0]), rna_dim, zero_inflaten=rna_zero_inflaten, is_recons=self.is_recons, basic_module=dec_basic_module) if self.is_rna else None
            self.cas_decoder = ZIPDecoder(d_lat, (d_hid[1], d_hid[0]), cas_dim, zero_inflaten=cas_zero_inflaten, is_recons=self.is_recons, basic_module=dec_basic_module) if self.is_cas else None
            self.adt_decoder = MLP_Module(d_lat, (adt_dim // 3 + 2 * d_lat_adt // 3, 2 * adt_dim // 3 + d_lat_adt // 3), adt_dim) if self.is_adt else None

        def forward_decoder(self, x_lat, edge_index=None):
            rna = cas = adt = None
            if self.is_rna:
                rna = self.rna_decoder(x_lat, edge_index)
            if self.is_cas:
                cas = self.cas_decoder(x_lat, edge_index)
            if self.is_adt:
                adt = self.adt_decoder(x_lat)
            return rna, cas, adt

        def forward(self, rna_norm, rna_counts, rna_libsize, cas_norm, cas_counts, cas_libsize, adt_norm, edge_index,
                    rna_ridge_lambda=0.5, cas_ridge_lambda=0.5):
            x_lat, rna_lat, cas_lat, adt_lat = self.forward_encoder(rna_norm, cas_norm, adt_norm, edge_index)
            ioa_loss = IOA_loss(rna_lat.data if self.is_rna else None, cas_lat.data if self.is_cas else None,
                                adt_lat.data if self.is_adt else None, x_lat, tau=self.tau) if self.num_modalities > 1 else None
            rna, cas, adt = self.forward_decoder(x_lat) if self.dec_basic_module == "Linear" else self.forward_decoder(x_lat, edge_index)
            nll_loss = 0
            mse_loss = 0
            if self.is_rna:
                nll_loss += NLL_loss(rna_counts, rna[0], rna[1], rna[2], scale_factor=rna_libsize, ridge_lambda=rna_ridge_lambda)
                if self.is_recons:
                    mse_loss += F.mse_loss(rna[3], rna_norm)
            if self.is_cas:
                nll_loss += NLL_loss(cas_counts, cas[0], cas[1], cas[2], scale_factor=cas_libsize, ridge_lambda=cas_ridge_lambda)
                if self.is_recons:
                    mse_loss += F.mse_loss(cas[3], cas_norm)
            if self.is_adt:
                mse_loss += F.mse_loss(adt, adt_norm)
            return ioa_loss, nll_loss, mse_loss

        def inner_inference(self, edge_index, rna_norm=None, cas_norm=None, adt_norm=None, impute=True, device=None):
            device = device or torch.device("cpu")
            self.to(device)
            self.eval()
            d = _data2input
            rna_norm = d(rna_norm).to(device) if rna_norm is not None else None
            cas_norm = d(cas_norm).to(device) if cas_norm is not None else None
            adt_norm = d(adt_norm).to(device) if adt_norm is not None else None
            edge_index = d(edge_index).to(device)
            ri = ci = ai = None
            with torch.no_grad():
                x_lat, rna_lat, cas_lat, adt_lat = self.forward_encoder(rna_norm, cas_norm, adt_norm, edge_index)
                if impute:
                    rna, cas, adt = self.forward_decoder(x_lat) if self.dec_basic_module == "Linear" else self.forward_decoder(x_lat, edge_index)
            x_lat = x_lat.cpu().numpy()
            if self.is_rna:
                rna_lat = rna_lat.cpu().numpy()
                ri = rna[2].cpu().numpy() if impute else None
            if self.is_cas:
                cas_lat = cas_lat.cpu().numpy()
                ci = cas[2].cpu().numpy() if impute else None
            if self.is_adt:
                adt_lat = adt_lat.cpu().numpy()
                ai = adt.cpu().numpy() if impute else None
            return x_lat, (rna_lat, cas_lat, adt_lat), (ri, ci, ai)

        def model_train(self, spa_mat, rna_counts=None, cas_counts=None, adt_counts=None, impute=False,
                        ref_rna_counts=None, ref_rna_anno=None, ref_cas_counts=None, ref_cas_anno=None,
                        ref_adt_counts=None, ref_adt_anno=None, epochs=100, lr=1e-3, weight_decay=1e-4,
                        batch_size=320, patience=20, delta=1e-4, device=None, rna_ridge_lambda=0.5,
                        cas_ridge_lambda=0.5, alpha1=1.0, alpha2=1.0, alpha3=1.0, alpha4=1.0, progress=False):
            device = device or torch.device("cuda:0")
            (rna_counts, cas_counts, adt_counts, rna_prior, rna_norm, rna_libsize, cas_prior, cas_norm,
             cas_libsize, adt_prior, adt_norm) = _prepare_omics(self, rna_counts, cas_counts, adt_counts,
                                                                ref_rna_counts, ref_rna_anno, ref_cas_counts,
                                                                ref_cas_anno, ref_adt_counts, ref_adt_anno)
            self.prior_initialize(rna_prior, cas_prior, adt_prior)
            self.unfreeze()
            self.to(device)
            knn_graph = kneighbors_graph(np.array(spa_mat), n_neighbors=self.k_neighbors, include_self=False).T.copy()
            geo = _geometric_data(knn_graph, torch.arange(spa_mat.shape[0]))
            batches = _ClusterBatches(geo, int(np.ceil(geo.num_nodes / batch_size)) * 10, coords=spa_mat)
            self.partitioner = batches.partitioner
            train_loader = batches.loader(batch_size=10, shuffle=True)
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            early_stopping = _EarlyStopping(patience=patience, delta=delta)
            self.epochs_run = 0
            for epoch in _progress(range(epochs), progress, "Model training"):
                self.train()
                losses = []
                for cur in train_loader:
                    np_index = cur.node_index.numpy()
                    edge_index = cur.edge_index.to(device)
                    (rc, rn, rl), (cc, cn, cl), an = _batch_inputs(self, np_index, device, rna_counts, rna_norm, rna_libsize,
                                                                   cas_counts, cas_norm, cas_libsize, adt_norm)
                    ioa_loss, nll_loss, mse_loss = self.forward(rn, rc, rl, cn, cc, cl, an, edge_index,
                                                                rna_ridge_lambda=rna_ridge_lambda, cas_ridge_lambda=cas_ridge_lambda)
                    bnn_loss = self.bnn_loss()
                    if self.num_modalities > 1 and self.is_recons:
                        loss = alpha1 * bnn_loss + alpha2 * ioa_loss + alpha3 * nll_loss + alpha4 * mse_loss
                    elif self.is_recons:
                        loss = alpha1 * bnn_loss + alpha3 * nll_loss + alpha4 * mse_loss
                    elif self.num_modalities > 1:
                        loss = alpha1 * bnn_loss + alpha2 * ioa_loss + alpha3 * nll_loss
                    else:
                        loss = alpha1 * bnn_loss + alpha3 * nll_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                self.epochs_run = epoch + 1
                early_stopping(np.mean(losses))
                if early_stopping.early_stop:
                    break
            self.freeze()
            return self.inner_inference(geo.edge_index, rna_norm, cas_norm, adt_norm, impute=impute)

    class PRESENT_BC(_Common):
        def __init__(self, rna_dim=None, cas_dim=None, adt_dim=None, n_batches=None, d_lat=50, d_hid=(1024, 512),
                     rna_zero_inflaten=True, cas_zero_inflaten=True, is_recons=True, intra_neighbors=6,
                     inter_neighbors=6, tau=0.1):
            super().__init__()
            if n_batches is None or n_batches == 1:
                n_batches = None
                self.n_batches = None
            else:
                self.n_batches = n_batches
            self.is_rna, self.is_cas, self.is_adt = rna_dim is not None, cas_dim is not None, adt_dim is not None
            self.is_recons = is_recons,  # upstream: a 1-tuple (always truthy); kept
            self.num_modalities = self.is_adt + self.is_rna + self.is_cas
            assert self.n_batches is not None and self.n_batches > 1, "Input batches should be larger than 1"
            assert self.num_modalities > 0, "Input modalities should be larger than 0"
            self.rna_dim, self.cas_dim, self.adt_dim = rna_dim, cas_dim, adt_dim
            self.d_lat = d_lat
            self.d_prior = d_lat // 3
            self.d_lat_adt = d_lat_adt = min(d_lat, adt_dim) if adt_dim is not None else 0
            self.d_prior_adt = d_lat_adt // 3 if adt_dim is not None else 0
            self.tau, self.intra_neighbors, self.inter_neighbors = tau, intra_neighbors, inter_neighbors
            self.rna_encoder = BayesianGATEncoder(rna_dim, d_lat, d_hid=d_hid) if self.is_rna else None
            self.cas_encoder = BayesianGATEncoder(cas_dim, d_lat, d_hid=d_hid) if self.is_cas else None
            self.adt_encoder = BayesianGATEncoder(adt_dim, d_lat_adt, d_hid=(2 * adt_dim // 3 + d_lat_adt // 3, adt_dim // 3 + 2 * d_lat_adt // 3)) if self.is_adt else None
            nin = int(self.is_rna + self.is_cas) * d_lat + d_lat_adt
            self.fusion_layer = MLP_Module(nin, (nin, d_lat), d_lat)
            self.batch_discriminator = MLP_Module(d_lat, (d_lat, d_lat), d_lat)
            if self.num_modalities == 1 and self.is_adt:
                d_lat = d_lat_adt
            self.register_parameter("batch_embeddings", nn.Parameter(torch.eye(n_batches), requires_grad=False))
            d_lat = d_lat + n_batches
            self.rna_decoder = ZINBDecoder(d_lat, (d_hid[1], d_hid[0]), rna_dim, zero_inflaten=rna_zero_inflaten, is_recons=is_recons) if self.is_rna else None
            self.cas_decoder = ZIPDecoder(d_lat, (d_hid[1], d_hid[0]), cas_dim, zero_inflaten=cas_zero_inflaten, is_recons=is_recons) if self.is_cas else None
            self.adt_decoder = MLP_Module(d_lat, (adt_dim // 3 + 2 * d_lat_adt // 3, 2 * adt_dim // 3 + d_lat_adt // 3), adt_dim) if self.is_adt else None

        def forward_decoder(self, x_lat, batch_indices):
            x_lat = torch.cat([x_lat, self.batch_embeddings[batch_indices]], -1)
            rna = self.rna_decoder(x_lat) if self.is_rna else None
            cas = self.cas_decoder(x_lat) if self.is_cas else None
            adt = self.adt_decoder(x_lat) if self.is_adt else None
            return rna, cas, adt

        def forward(self, rna_norm, rna_counts, rna_libsize, cas_norm, cas_counts, cas_libsize, adt_norm, edge_index,
                    batch_indices, positive_indices=None, negative_indices=None, target_lat=None,
                    rna_ridge_lambda=0.5, cas_ridge_lambda=0.5, stage=1):
            x_lat, rna_lat, cas_lat, adt_lat = self.forward_encoder(rna_norm, cas_norm, adt_norm, edge_index)
            if self.num_modalities == 1 and stage == 2:
                x_lat = self.fusion_layer(x_lat)
            iba_loss = IBA_loss(x_lat, positive_indices, negative_indices, tau=self.tau) if positive_indices is not None and negative_indices is not None else 0
            ibp_loss = IBP_loss(x_lat, target_lat, batch_indices, tau=self.tau) if target_lat is not None else 0
            ioa_loss = IOA_loss(rna_lat.data if self.is_rna else None, cas_lat.data if self.is_cas else None,
                                adt_lat.data if self.is_adt else None, x_lat, tau=self.tau) if self.num_modalities > 1 else 0
            rna, cas, adt = self.forward_decoder(x_lat, batch_indices)
            nll_loss = 0
            mse_loss = 0
            if self.is_rna:
                nll_loss += NLL_loss(rna_counts, rna[0], rna[1], rna[2], scale_factor=rna_libsize, ridge_lambda=rna_ridge_lambda)
                if self.is_recons:
                    mse_loss += F.mse_loss(rna[3], rna_norm)
            if self.is_cas:
                nll_loss += NLL_loss(cas_counts, cas[0], cas[1], cas[2], scale_factor=cas_libsize, ridge_lambda=cas_ridge_lambda)
                if self.is_recons:
                    mse_loss += F.mse_loss(cas[3], cas_norm)
            if self.is_adt:
                mse_loss += F.mse_loss(adt, adt_norm)
            return x_lat, nll_loss, mse_loss, ioa_loss, iba_loss, ibp_loss

        def inner_inference(self, edge_index, rna_norm=None, cas_norm=None, adt_norm=None, batch_indices=None,
                            impute=True, batch_fusion=True, device=None):
            device = device or torch.device("cpu")
            self.to(device)
            self.eval()
            d = _data2input
            rna_norm = d(rna_norm).to(device) if self.is_rna and rna_norm is not None else None
            cas_norm = d(cas_norm).to(device) if self.is_cas and cas_norm is not None else None
            adt_norm = d(adt_norm).to(device) if self.is_adt and adt_norm is not None else None
            batch_indices = d(batch_indices).to(device) if batch_indices is not None else None
            edge_index = edge_index.to(device)
            ri = ci = ai = None
            with torch.no_grad():
                x_lat, rna_lat, cas_lat, adt_lat = self.forward_encoder(rna_norm, cas_norm, adt_norm, edge_index)
                if self.num_modalities == 1 and batch_fusion:
                    x_lat = self.fusion_layer(x_lat)
                if impute:
                    rna, cas, adt = self.forward_decoder(x_lat, batch_indices)
            x_lat = x_lat.cpu().numpy()
            if self.is_rna:
                rna_lat = rna_lat.cpu().numpy()
                ri = rna[2].cpu().numpy() if impute else None
            if self.is_cas:
                cas_lat = cas_lat.cpu().numpy()
                ci = cas[2].cpu().numpy() if impute else None
            if self.is_adt:
                adt_lat = adt_lat.cpu().numpy()
                ai = adt.cpu().numpy() if impute else None
            return x_lat, (rna_lat, cas_lat, adt_lat), (ri, ci, ai)

        def model_train(self, spa_mat, rna_counts=None, cas_counts=None, adt_counts=None, batch_label=None, impute=False,
                        ref_rna_counts=None, ref_rna_anno=None, ref_cas_counts=None, ref_cas_anno=None,
                        ref_adt_counts=None, ref_adt_anno=None, negative_samples=0.25, rna_ridge_lambda=0.5,
                        cas_ridge_lambda=0.5, epochs=100, lr=1e-3, weight_decay=1e-4, batch_size=320, patience=20,
                        delta=1e-4, alpha1=1.0, alpha2=1.0, alpha3=1.0, alpha4=1.0, beta1=1.0, beta2=1.0, beta3=1.0,
                        device=None, progress=False):
            device = device or torch.device("cuda:0")
            (rna_counts, cas_counts, adt_counts, rna_prior, rna_norm, rna_libsize, cas_prior, cas_norm,
             cas_libsize, adt_prior, adt_norm) = _prepare_omics(self, rna_counts, cas_counts, adt_counts,
                                                                ref_rna_counts, ref_rna_anno, ref_cas_counts,
                                                                ref_cas_anno, ref_adt_counts, ref_adt_anno)
            assert batch_label is not None and np.unique(batch_label).shape[0] == self.n_batches, "Invalid input batch_label"
            self.prior_initialize(rna_prior, cas_prior, adt_prior)
            self.unfreeze()
            self.to(device)
            graph = _Integrated_3D_graph(batch_label=batch_label, spatial_mat=spa_mat, rna_mat=rna_norm, cas_mat=cas_norm,
                                         adt_mat=adt_norm, intra_neighbors=self.intra_neighbors,
                                         inter_neighbors=self.inter_neighbors, inter_metric="cosine")
            batch_num_label = _StrLabel2Idx(batch_label)
            geo = _geometric_data(graph.toarray(), torch.arange(spa_mat.shape[0]))
            nparts = int(np.ceil(geo.num_nodes / batch_size)) * 10
            batches = _ClusterBatches(geo, nparts, coords=spa_mat)
            self.partitioner = batches.partitioner
            train_loader = batches.loader(batch_size=10, shuffle=True)
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            early_stopping = _EarlyStopping(patience=patience, delta=delta)
            for epoch in _progress(range(epochs), progress, "First-stage trains basic model"):
                self.train()
                losses = []
                for cur in train_loader:
                    np_index = cur.node_index.numpy()
                    edge_index = cur.edge_index.to(device)
                    batch_indices = _data2input(batch_num_label[np_index]).to(device)
                    (rc, rn, rl), (cc, cn, cl), an = _batch_inputs(self, np_index, device, rna_counts, rna_norm, rna_libsize,
                                                                   cas_counts, cas_norm, cas_libsize, adt_norm)
                    x_lat, nll_loss, mse_loss, ioa_loss, iba_loss, ibp_loss = self.forward(
                        rn, rc, rl, cn, cc, cl, an, edge_index, batch_indices=batch_indices,
                        rna_ridge_lambda=rna_ridge_lambda, cas_ridge_lambda=cas_ridge_lambda, stage=1)
                    bnn_loss = self.bnn_loss()
                    if self.num_modalities > 1 and self.is_recons:
                        loss = alpha1 * bnn_loss + alpha2 * ioa_loss + alpha3 * nll_loss + alpha4 * mse_loss
                    elif self.is_recons:
                        loss = alpha1 * bnn_loss + alpha3 * nll_loss + alpha4 * mse_loss
                    elif self.num_modalities > 1:
                        loss = alpha1 * bnn_loss + alpha2 * ioa_loss + alpha3 * nll_loss
                    else:
                        loss = alpha1 * bnn_loss + alpha3 * nll_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                early_stopping(np.mean(losses))
                if early_stopping.early_stop:
                    break
            early_stopping = _EarlyStopping(patience=patience // 2, delta=delta)
            disc_optimizer = torch.optim.Adam(self.batch_discriminator.parameters(), lr=lr, weight_decay=weight_decay)
            gen_optimizer = torch.optim.Adam(_optim_parameters(self, excluded="batch_discriminator"), lr=lr, weight_decay=weight_decay)
            for epoch in _progress(range(epochs), progress, "Second-stage trains BC model"):
                joint_lat, _, _ = self.inner_inference(geo.edge_index, rna_norm, cas_norm, adt_norm, impute=False, batch_fusion=False)
                self.to(device)
                graph = _Integrated_3D_graph(batch_label=batch_label, spatial_mat=spa_mat, joint_mat=joint_lat,
                                             intra_neighbors=self.intra_neighbors, inter_neighbors=self.inter_neighbors)
                geo = _geometric_data(graph.toarray(), torch.arange(spa_mat.shape[0]))
                batches = _ClusterBatches(geo, nparts, coords=spa_mat)
                train_loader = batches.loader(batch_size=10, shuffle=True)
                self.train()
                losses = []
                for cur in train_loader:
                    np_index = cur.node_index.numpy()
                    edge_index = cur.edge_index.to(device)
                    edge_attr = cur.edge_attr.to(device)
                    batch_indices = _data2input(batch_num_label[np_index]).to(device)
                    batch_indices_np = batch_num_label[np_index]
                    joint_lat_batch = _data2input(joint_lat[np_index]).to(device)
                    positive_indices, negative_indices = [], []
                    for i in range(cur.num_nodes):
                        idx = (edge_index[1] == i) & (edge_attr == -1)
                        num_positive = idx.cpu().numpy().sum()
                        positive_indices.append(edge_index[0, idx] if num_positive > 0 else None)
                        batch_neighbors = edge_index[0, (edge_index[1] == i) & (edge_attr == 1)].cpu().numpy()
                        cand = np.setdiff1d(np.arange(cur.num_nodes)[batch_indices_np == batch_indices_np[i]], batch_neighbors)
                        if negative_samples is None:
                            num_negative = min(num_positive, cand.shape[0])
                        elif isinstance(negative_samples, int):
                            num_negative = min(num_positive * negative_samples, cand.shape[0])
                        else:
                            num_negative = int(np.ceil(cand.shape[0] * negative_samples))
                        sel = np.random.choice(cand, num_negative, replace=False)
                        negative_indices.append(_data2input(sel).to(device))
                    (rc, rn, rl), (cc, cn, cl), an = _batch_inputs(self, np_index, device, rna_counts, rna_norm, rna_libsize,
                                                                   cas_counts, cas_norm, cas_libsize, adt_norm)
                    x_lat, rna_lat, cas_lat, adt_lat = self.forward_encoder(rn, cn, an, edge_index)
                    if self.num_modalities == 1:
                        x_lat = self.fusion_layer(x_lat)
                    disc_loss = beta3 * F.cross_entropy(self.batch_discriminator(x_lat.detach()), batch_indices)
                    disc_optimizer.zero_grad()
                    disc_loss.backward()
                    disc_optimizer.step()
                    x_lat, nll_loss, mse_loss, ioa_loss, iba_loss, ibp_loss = self.forward(
                        rn, rc, rl, cn, cc, cl, an, edge_index, batch_indices, positive_indices, negative_indices,
                        joint_lat_batch, rna_ridge_lambda=rna_ridge_lambda, cas_ridge_lambda=cas_ridge_lambda, stage=2)
                    bnn_loss = self.bnn_loss()
                    gen_loss = -F.cross_entropy(self.batch_discriminator(x_lat), batch_indices)
                    if self.num_modalities > 1 and self.is_recons:
                        loss = alpha1 * bnn_loss + alpha2 * ioa_loss + alpha3 * nll_loss + alpha4 * mse_loss + beta1 * iba_loss + beta2 * ibp_loss + beta3 * gen_loss
                    elif self.is_recons:
                        loss = alpha1 * bnn_loss + alpha3 * nll_loss + alpha4 * mse_loss + beta1 * iba_loss + beta2 * ibp_loss + beta3 * gen_loss
                    elif self.num_modalities > 1:
                        loss = alpha1 * bnn_loss + alpha2 * ioa_loss + alpha3 * nll_loss + beta1 * iba_loss + beta2 * ibp_loss + beta3 * gen_loss
                    else:
                        loss = alpha1 * bnn_loss + alpha3 * nll_loss + beta1 * iba_loss + beta2 * ibp_loss + beta3 * gen_loss
                    gen_optimizer.zero_grad()
                    loss.backward()
                    gen_optimizer.step()
                    losses.append(loss.item() + disc_loss.item())
                early_stopping(np.mean(losses))
                if early_stopping.early_stop:
                    break
            self.freeze()
            return self.inner_inference(geo.edge_index, rna_norm, cas_norm, adt_norm, impute=impute, batch_fusion=True)

    return PRESENT_RP, PRESENT_BC


# --------------------------------------------------------------------------
# Main.py: PRESENT_function
# --------------------------------------------------------------------------
def _run_leiden(adata, n_cluster, use_rep="embeddings", key_added="Nleiden", range_min=0, range_max=3, max_steps=30, tolerance=0):
    """Upstream ``Utils.run_leiden``: bisection on the Leiden resolution to reach ``n_cluster``."""
    import scanpy as sc

    sc.pp.neighbors(adata, use_rep=use_rep)
    this_step = 0
    this_min, this_max = float(range_min), float(range_max)
    while this_step < max_steps:
        this_resolution = this_min + ((this_max - this_min) / 2)
        sc.tl.leiden(adata, resolution=this_resolution)
        this_clusters = adata.obs["leiden"].nunique()
        if this_clusters > n_cluster + tolerance:
            this_max = this_resolution
        elif this_clusters < n_cluster - tolerance:
            this_min = this_resolution
        else:
            adata.obs[key_added] = adata.obs["leiden"]
            return adata
        this_step += 1
    adata.obs[key_added] = adata.obs["leiden"]
    return adata


def present_preprocess_peaks(adata_atac, peak_min_cells_fraction=0.03):
    """Peak / spot filter of ``PRESENT_function`` (episcanpy filter_features / filter_cells)."""
    import scanpy as sc

    sc.pp.filter_genes(adata_atac, min_cells=int(adata_atac.shape[0] * peak_min_cells_fraction))
    sc.pp.filter_cells(adata_atac, min_genes=1)
    if "n_genes" in adata_atac.obs.columns:
        adata_atac.obs["nb_features"] = adata_atac.obs.pop("n_genes")
    return adata_atac


def present(
    adata_rna=None,
    adata_atac=None,
    adata_adt=None,
    *,
    spatial_key: str = "spatial",
    batch_key: Optional[str] = None,
    rdata_rna=None, rdata_rna_anno: Optional[str] = None,
    rdata_atac=None, rdata_atac_anno: Optional[str] = None,
    rdata_adt=None, rdata_adt_anno: Optional[str] = None,
    gene_min_cells: int = 1,
    peak_min_cells_fraction: float = 0.03,
    protein_min_cells: int = 1,
    num_hvg: int = 3000,
    nclusters: Optional[int] = 10,
    d_lat: int = 50,
    k_neighbors: int = 6,
    intra_neighbors: int = 6,
    inter_neighbors: int = 6,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 320,
    device: Optional[str] = None,
    random_seed: Optional[int] = 2024,
    key_added: str = "X_present",
    domain_key: str = "present_domain",
    progress: bool = False,
):
    """PRESENT joint embedding and spatial domains of spatial RNA / ATAC / ADT data.

    Vendored PyTorch port (no torch_geometric, no episcanpy) of PRESENT
    (Li et al., *Brief Bioinform* 2026; MIT licence, Zhen Li). Each omics layer
    is encoded by a Bayesian graph-attention encoder whose prior is the layer's
    PCA loadings; layers are fused by an MLP with an inter-omics alignment
    (contrastive KL) loss and decoded by ZINB (RNA) / zero-inflated Poisson
    (ATAC) / MSE (ADT) decoders; training uses Cluster-GCN mini-batches. With
    ``batch_key`` the multi-sample model (``PRESENT_BC``) adds a 3-D
    spatial + cross-sample graph and adversarial batch correction.

    Arguments mirror upstream ``PRESENT_function``; inputs are **raw counts**
    with spot coordinates in ``obsm[spatial_key]``. Inputs are not modified
    except for the result keys below. ATAC-only, RNA-only and RNA+ATAC(+ADT)
    are all supported.

    Arguments:
        adata_rna, adata_atac, adata_adt: AnnData per omics layer (same spots).
        nclusters: number of domains for the upstream Leiden resolution search;
            ``None`` skips clustering.
        device: ``"cuda"``/``"cuda:0"``/``"cpu"``; default CUDA if available.
        random_seed: seeds python, numpy and torch (upstream ``setup_seed``)
            before preprocessing; ``None`` leaves the RNG state alone.
        key_added: ``obsm`` key of the joint embedding (rows of spots removed by
            the QC filters are NaN) in every input AnnData.
        domain_key: ``obs`` key of the Leiden domains.

    Returns:
        AnnData of the kept spots with ``obsm["embeddings"]`` and
        ``obs["LeidenClusters"]`` (the upstream return value).

    Reference:
        Li Z, Cui X, Chen X, Gao Z, Liu Y, Pan Y, Chen S, Jiang R. Cross-modality
        representation and multi-sample integration of spatially resolved omics
        data. Brief Bioinform 27, bbag214 (2026).
    """
    import scanpy as sc
    import torch

    if random_seed is not None:
        _setup_seed(random_seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device("cuda:0" if device == "cuda" else device)
    originals = {"rna": adata_rna, "atac": adata_atac, "adt": adata_adt}
    adata_rna = adata_rna.copy() if adata_rna is not None else None
    adata_atac = adata_atac.copy() if adata_atac is not None else None
    adata_adt = adata_adt.copy() if adata_adt is not None else None
    rdata_rna = rdata_rna.copy() if rdata_rna is not None else None
    rdata_atac = rdata_atac.copy() if rdata_atac is not None else None
    rdata_adt = rdata_adt.copy() if rdata_adt is not None else None
    assert gene_min_cells >= 0 and protein_min_cells >= 0
    assert 0 <= peak_min_cells_fraction <= 1
    assert k_neighbors > 0 and intra_neighbors > 0 and inter_neighbors > 0

    index = None
    if adata_rna is not None:
        sc.pp.filter_genes(adata_rna, min_cells=gene_min_cells)
        sc.pp.filter_cells(adata_rna, min_genes=1)
        sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=num_hvg, subset=True)
        if rdata_rna is not None:
            rdata_rna, adata_rna = _ref_feature_alignment(rdata_rna, adata_rna, omics="RNA")
            if rdata_rna_anno is not None:
                rdata_rna_anno = rdata_rna.obs[rdata_rna_anno].values
        sc.pp.filter_genes(adata_rna, min_cells=1)
        sc.pp.filter_cells(adata_rna, min_genes=1)
        index = adata_rna.obs_names
    if adata_atac is not None:
        present_preprocess_peaks(adata_atac, peak_min_cells_fraction)
        if rdata_atac is not None:
            rdata_atac, adata_atac = _ref_feature_alignment(rdata_atac, adata_atac, omics="ATAC")
            if rdata_atac_anno is not None:
                rdata_atac_anno = rdata_atac.obs[rdata_atac_anno].values
        sc.pp.filter_genes(adata_atac, min_cells=1)
        sc.pp.filter_cells(adata_atac, min_genes=1)
        index = np.intersect1d(adata_atac.obs_names, index) if index is not None else adata_atac.obs_names
    if adata_adt is not None:
        sc.pp.filter_genes(adata_adt, min_cells=protein_min_cells)
        sc.pp.filter_cells(adata_adt, min_genes=1)
        if rdata_adt is not None:
            rdata_adt, adata_adt = _ref_feature_alignment(rdata_adt, adata_adt, omics="ADT")
            if rdata_adt_anno is not None:
                rdata_adt_anno = rdata_adt.obs[rdata_adt_anno].values
        sc.pp.filter_genes(adata_adt, min_cells=1)
        sc.pp.filter_cells(adata_adt, min_genes=1)
        index = np.intersect1d(adata_adt.obs_names, index) if index is not None else adata_adt.obs_names
    assert index is not None, "Please input at least one omics layer in anndata.AnnData format"

    spatial_mtx = batch_indices = None
    rna_dim = cas_dim = adt_dim = None
    for name, a in (("rna", adata_rna), ("atac", adata_atac), ("adt", adata_adt)):
        if a is None:
            continue
        a = a[index, :]
        if name == "rna":
            adata_rna, rna_dim = a, a.shape[1]
        elif name == "atac":
            adata_atac, cas_dim = a, a.shape[1]
        else:
            adata_adt, adt_dim = a, a.shape[1]
        if batch_key is not None and batch_key in a.obs.keys():
            batch_indices = a.obs[batch_key].values.astype(str)
        if spatial_key in a.obsm.keys():
            spatial_mtx = a.obsm[spatial_key]
    assert spatial_mtx is not None, "Please provide spatial mtx in adata.obsm under spatial_key"
    if batch_key is not None:
        assert batch_indices is not None, "Invalid batch_key: cannot find batch_key in adata.obs"

    PRESENT_RP, PRESENT_BC = _build_models()
    common = dict(spa_mat=spatial_mtx,
                  rna_counts=adata_rna.X if adata_rna is not None else None,
                  ref_rna_counts=rdata_rna.X if rdata_rna is not None else None, ref_rna_anno=rdata_rna_anno,
                  cas_counts=adata_atac.X if adata_atac is not None else None,
                  ref_cas_counts=rdata_atac.X if rdata_atac is not None else None, ref_cas_anno=rdata_atac_anno,
                  adt_counts=adata_adt.X if adata_adt is not None else None,
                  ref_adt_counts=rdata_adt.X if rdata_adt is not None else None, ref_adt_anno=rdata_adt_anno,
                  impute=False, epochs=epochs, lr=lr, batch_size=batch_size, device=dev, progress=progress)
    if batch_key is not None and batch_indices is not None:
        n_batches = np.unique(batch_indices).shape[0]
        model = PRESENT_BC(rna_dim=rna_dim, cas_dim=cas_dim, adt_dim=adt_dim, n_batches=n_batches, d_lat=d_lat,
                           intra_neighbors=intra_neighbors, inter_neighbors=inter_neighbors).to(dev)
        embeddings, _, _ = model.model_train(batch_label=batch_indices, **common)
    else:
        model = PRESENT_RP(rna_dim=rna_dim, cas_dim=cas_dim, adt_dim=adt_dim, d_lat=d_lat, k_neighbors=k_neighbors).to(dev)
        embeddings, _, _ = model.model_train(**common)

    base = adata_rna if adata_rna is not None else (adata_atac if adata_atac is not None else adata_adt)
    out = sc.AnnData(pd.DataFrame(embeddings, index=index), obs=base.obs.copy(), obsm=dict(base.obsm))
    out.obsm["embeddings"] = embeddings
    if nclusters is not None:
        out = _run_leiden(out, n_cluster=nclusters, use_rep="embeddings", key_added="LeidenClusters")
    out.uns["present"] = {"partitioner": getattr(model, "partitioner", None), "random_seed": random_seed,
                          "epochs": epochs, "d_lat": d_lat, "k_neighbors": k_neighbors, "nclusters": nclusters}
    for a in originals.values():
        if a is None:
            continue
        pos = pd.Index(a.obs_names).get_indexer(pd.Index(index))
        emb = np.full((a.n_obs, embeddings.shape[1]), np.nan, dtype=np.float32)
        emb[pos] = embeddings
        a.obsm[key_added] = emb
        if nclusters is not None:
            lab = pd.Series(pd.NA, index=a.obs_names, dtype="object")
            lab.iloc[pos] = out.obs["LeidenClusters"].astype(str).to_numpy()
            a.obs[domain_key] = pd.Categorical(lab)
        a.uns[key_added] = dict(out.uns["present"])
    return out
