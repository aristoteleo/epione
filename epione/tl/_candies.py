# SPDX-License-Identifier: MIT
#
# Vendored port of CANDIES (Liu et al., "Cross-modal denoising and integration
# of spatial multi-omics data with CANDIES", Advanced Science 2026,
# doi:10.1002/advs.202523754).
# Upstream: https://github.com/YeLiu-Lab/CANDIES
#           commit 85ce567446721de83c09a6feed0f20d9744b031b
# Copyright (c) 2025 zouwanpeng, MIT licence (full text in
# epione/tl/_licenses/CANDIES.LICENSE and the repository NOTICE file).
# CANDIES' DiT blocks follow facebookresearch/DiT; the timm ``Mlp`` used there
# (Apache-2.0, Ross Wightman) is re-implemented here as a 2-layer MLP.
#
# Changes from upstream (see tests/parity/spatial_candies/MANIFEST.md):
#   * torch_geometric.nn.GCNConv (ATAC graph auto-encoder, applied to a torch
#     sparse normalised adjacency) -> epione.tl._gnn.GCNConv, same arithmetic;
#   * timm.models.vision_transformer.Mlp -> _Mlp (same layers, same order);
#   * POT ``ot.dist(metric="euclidean")`` -> the same expansion in numpy;
#   * R mclust via rpy2 -> pymclustR (pure-Python port of CRAN mclust);
#   * import-time side effects removed: upstream DiTs.py seeds everything,
#     enables torch.use_deterministic_algorithms(True) and queries CUDA at
#     import; here this happens inside candies() (deterministic=True default);
#   * the notebook pipeline (tutorial/E18_mouse_brain.ipynb) is the function
#     candies(); the modality-selection cell (diagnostic print) is not run, the
#     modality to denoise is the ``denoise`` argument.
"""CANDIES: conditional-diffusion denoising + dual-graph contrastive integration."""
from __future__ import annotations

import math
import os
import random
from typing import Optional

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp

__all__ = ["candies", "candies_lsi"]


# --------------------------------------------------------------------------
# seeding / preprocessing (DiTs.seed_everything, preprocess1.py, get_graph.py)
# --------------------------------------------------------------------------
def _seed_everything(seed: int, dt: bool = True):
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(dt)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _tfidf(X):
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    tf = X / X.sum(axis=1, keepdims=True)
    return tf * idf


def candies_lsi(adata, n_components: int = 20, use_highly_variable: Optional[bool] = None, **kwargs) -> None:
    """Upstream ``preprocess1.lsi`` (Seurat-v3 LSI); drops the first component -> ``obsm["X_lsi"]``."""
    import sklearn.preprocessing
    import sklearn.utils.extmath

    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = _tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def _pca(adata, use_reps=None, n_comps=10):
    from sklearn.decomposition import PCA

    p = PCA(n_components=n_comps)
    if use_reps is not None:
        return p.fit_transform(adata.obsm[use_reps])
    X = adata.X
    return p.fit_transform(X.toarray() if sp.issparse(X) else X)


def _ot_euclidean(x1, x2):
    """``ot.dist(x1, x2, metric="euclidean")`` for numpy inputs (POT euclidean_distances)."""
    a2 = np.einsum("ij,ij->i", x1, x1)
    b2 = np.einsum("ij,ij->i", x2, x2)
    c = -2 * np.dot(x1, x2.T)
    c += a2[:, None]
    c += b2[None, :]
    c = np.maximum(c, 0)
    c = np.sqrt(c)
    if x1 is x2:
        c = c * (1 - np.eye(x1.shape[0], dtype=c.dtype))
    return c


def _construct_interaction(adata, n_neighbors=4):
    position = adata.obsm["spatial"]
    distance_matrix = _ot_euclidean(position, position)
    n_spot = distance_matrix.shape[0]
    adata.obsm["distance_matrix"] = distance_matrix
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        distance = distance_matrix[i, :].argsort()
        for t in range(1, n_neighbors + 1):
            interaction[i, distance[t]] = 1
    adata.obsm["graph_neigh"] = interaction
    adj = interaction + interaction.T
    adata.obsm["adj"] = np.where(adj > 1, 1, adj)


def _construct_interaction_knn(adata, n_neighbors=4):
    from sklearn.neighbors import NearestNeighbors

    position = adata.obsm["spatial"]
    n_spot = position.shape[0]
    _, indices = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position).kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    adata.obsm["graph_neigh"] = interaction
    adj = interaction + interaction.T
    adata.obsm["adj"] = np.where(adj > 1, 1, adj)


def _zinb_preprocess(adata, n_top_genes):
    import scanpy as sc

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def _get_feature(adata):
    X = adata[:, adata.var["highly_variable"]].X
    feat = X.toarray()[:, ] if sp.issparse(X) else X[:, ]
    ids = np.random.permutation(np.arange(feat.shape[0]))  # upstream 'feat_a' augmentation (RNG use kept)
    adata.obsm["feat"] = feat
    adata.obsm["feat_a"] = feat[ids]


def _normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d = sp.diags(d_inv_sqrt)
    return adj.dot(d).transpose().dot(d).toarray()


def _to_torch_sparse(m):
    import torch

    m = m.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((m.row, m.col)).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(m.data), torch.Size(m.shape))


def _preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    d = sp.diags(np.power(rowsum, -0.5).flatten())
    return _to_torch_sparse(adj_.dot(d).transpose().dot(d).tocoo())


def _graph_by_coordinate(cell_position, n_neighbors=10):
    from sklearn.neighbors import NearestNeighbors

    _, indices = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position).kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    return pd.DataFrame({"x": x, "y": y, "value": np.ones(x.size)})


def _construct_neighbor_graph(a1, a2, datatype="SPOTS", n_neighbors=3):
    from sklearn.neighbors import kneighbors_graph

    if datatype in ["Stereo-CITE-seq", "Spatial-epigenome-transcriptome"]:
        n_neighbors = 6
    a1.uns["adj_spatial"] = _graph_by_coordinate(a1.obsm["spatial"], n_neighbors=n_neighbors)
    a2.uns["adj_spatial"] = _graph_by_coordinate(a2.obsm["spatial"], n_neighbors=n_neighbors)
    a1.obsm["adj_feature"] = kneighbors_graph(a1.obsm["feat"], 20, mode="connectivity", metric="correlation", include_self=False)
    a2.obsm["adj_feature"] = kneighbors_graph(a2.obsm["feat"], 20, mode="connectivity", metric="correlation", include_self=False)
    return a1, a2


def _adjacent_matrix_preprocessing(a1, a2):
    import torch

    def dense_adj(df):
        n = df["x"].max() + 1
        A = sp.coo_matrix((df["value"], (df["x"], df["y"])), shape=(n, n)).toarray()
        A = A + A.T
        return np.where(A > 1, 1, A)

    def feat_adj(m):
        A = torch.FloatTensor(m.copy().toarray())
        A = A + A.T
        return np.where(A > 1, 1, A)

    return {"adj_spatial_omics1": _preprocess_graph(dense_adj(a1.uns["adj_spatial"])),
            "adj_spatial_omics2": _preprocess_graph(dense_adj(a2.uns["adj_spatial"])),
            "adj_feature_omics1": _preprocess_graph(feat_adj(a1.obsm["adj_feature"])),
            "adj_feature_omics2": _preprocess_graph(feat_adj(a2.obsm["adj_feature"]))}


# --------------------------------------------------------------------------
# networks (ZINB_encoder.py, AutoEncoder.py, DiTs.py, diff_scheduler.py,
# train_diff.py, sampler.py, integration.py)
# --------------------------------------------------------------------------
def _build():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parameter import Parameter

    from ._gnn import GCNConv

    class MeanAct(nn.Module):
        def forward(self, x):
            return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

    class DispAct(nn.Module):
        def forward(self, x):
            return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

    class ZINBLoss(nn.Module):
        def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0, device=None):
            eps = 1e-10
            scale_factor = torch.Tensor([1.0]).to(device)[:, None]
            mean = mean * scale_factor
            t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
            t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
            nb_case = t1 + t2 - torch.log(1.0 - pi + eps)
            zero_nb = torch.pow(disp / (disp + mean + eps), disp)
            zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
            result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
            if ridge_lambda > 0:
                result += ridge_lambda * torch.square(pi)
            return torch.mean(result)

    class NBLoss(nn.Module):
        def forward(self, x, mean, disp, scale_factor=1.0, ridge_lambda=0.0, device=None):
            eps = 1e-10
            scale_factor = torch.Tensor([1.0]).to(device)[:, None]
            mean = mean * scale_factor
            t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
            t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
            result = torch.mean(t1 + t2)
            if ridge_lambda > 0:
                result += ridge_lambda * torch.square(mean)
            return result

    class EnDecoder(nn.Module):
        def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
            super().__init__()
            self.in_features, self.out_features = in_features, out_features
            self.graph_neigh, self.dropout, self.act = graph_neigh, dropout, act
            self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
            self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
            torch.nn.init.xavier_uniform_(self.weight1)
            torch.nn.init.xavier_uniform_(self.weight2)
            self._dec_mean = nn.Sequential(nn.Linear(self.in_features, self.in_features), MeanAct())
            self._dec_disp = nn.Sequential(nn.Linear(self.in_features, self.in_features), DispAct())
            self._dec_pi = nn.Sequential(nn.Linear(self.in_features, self.in_features), nn.Sigmoid())

        def encode(self, feat, adj):
            z = F.dropout(feat, self.dropout, self.training)
            return torch.mm(adj, torch.mm(z, self.weight1))

        def decode(self, hiden_emb, adj):
            h = torch.mm(adj, torch.mm(hiden_emb, self.weight2))
            return h, self._dec_mean(h), self._dec_disp(h), self._dec_pi(h)

    def train_zinb(adata, device, learning_rate=0.0005, weight_decay=0.0, n_top_genes=3000, epochs=500,
                   random_seed=41, alpha=10, datatype="10X", dim_output=None, progress=False):
        """Upstream ``encoder_ZINB(...).train()``: returns (embedding, normalised adjacency)."""
        adata = adata.copy()
        zinb_loss = ZINBLoss().to(device)
        _seed_everything(random_seed)
        if "highly_variable" not in adata.var.keys():
            _zinb_preprocess(adata, n_top_genes)
        if "adj" not in adata.obsm.keys():
            (_construct_interaction_knn if datatype in ["Stereo", "Slide"] else _construct_interaction)(adata)
        if "feat" not in adata.obsm.keys():
            _get_feature(adata)
        features = torch.FloatTensor(adata.obsm["feat"].copy()).to(device)
        adj = adata.obsm["adj"]
        graph_neigh = torch.FloatTensor(adata.obsm["graph_neigh"].copy() + np.eye(adj.shape[0])).to(device)
        if datatype in ["Stereo", "Slide"]:
            a = sp.coo_matrix(adj)
            a_ = a + sp.eye(a.shape[0])
            d = sp.diags(np.power(np.array(a_.sum(1)), -0.5).flatten())
            adj = _to_torch_sparse(a_.dot(d).transpose().dot(d).tocoo()).to(device)
        else:
            adj = torch.FloatTensor(_normalize_adj(adj) + np.eye(adj.shape[0])).to(device)
        model = EnDecoder(features.shape[1], dim_output, graph_neigh).to(device)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate, weight_decay=weight_decay)
        model.train()
        copyemb = None
        it = range(epochs)
        if progress:
            from tqdm import tqdm
            it = tqdm(it)
        for epoch in it:
            hiden_feat = model.encode(features, adj)
            x_recon, meanbatch, dispbatch, pibatch = model.decode(hiden_feat, adj)
            if epoch == 0:
                copyemb = hiden_feat
            re_loss = F.mse_loss(features, x_recon)
            zl = zinb_loss(features, meanbatch, dispbatch, pibatch, device=device)
            loss = (alpha * re_loss + 0.5 * zl) * 0.2
            if torch.isnan(x_recon).sum().item() > 0:
                hiden_feat = copyemb
                break
            copyemb = hiden_feat
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            return hiden_feat.detach().cpu().numpy(), adj.detach().cpu().numpy()

    class GCNEncoder(nn.Module):  # ATACEncoder / RNAEncoder / ProteinEncoder
        def __init__(self, in_channels, hidden_channels, latent_dim):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, latent_dim)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.1, training=self.training)
            return self.conv2(x, edge_index)

    class ATACDecoder(nn.Module):
        def __init__(self, d_in, d_hid, d_out):
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_hid)
            self.fc2 = nn.Linear(d_hid, d_out)
            self._dec_mean = nn.Sequential(nn.Linear(d_out, d_out), MeanAct())
            self._dec_disp = nn.Sequential(nn.Linear(d_out, d_out), DispAct())

        def forward(self, x):
            recon = self.fc2(F.relu(self.fc1(x)))
            return recon, self._dec_mean(recon), self._dec_disp(recon)

    def train_atac(X_lsi, adj_spatial, device, hidden_channels=128, epochs=1000, atac_latent_dim=64):
        """Upstream ``AutoEncoder.train_atac`` (GCN auto-encoder on the LSI matrix)."""
        enc = GCNEncoder(X_lsi.shape[1], hidden_channels, atac_latent_dim).to(device)
        dec = ATACDecoder(atac_latent_dim, hidden_channels, X_lsi.shape[1]).to(device)
        feats = torch.tensor(X_lsi, dtype=torch.float).to(device)
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=0.001)
        mse, nb = nn.MSELoss(), NBLoss()
        for epoch in range(epochs):
            enc.train()
            dec.train()
            recons, _mean, _disp = dec(enc(feats, adj_spatial))
            loss = mse(recons, feats) + nb(feats, _mean, _disp, device=device)
            opt.zero_grad()
            loss.backward()
            opt.step()
        enc.eval()
        return enc(feats, adj_spatial).cpu().detach().numpy()

    # ---------------- DiT (DiTs.py) ----------------
    class _Mlp(nn.Module):  # timm.models.vision_transformer.Mlp(in, hidden, act_layer, drop=0)
        def __init__(self, in_features, hidden_features, act_layer, drop=0.0):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
            self.act = act_layer()
            self.drop1 = nn.Dropout(drop)
            self.norm = nn.Identity()
            self.fc2 = nn.Linear(hidden_features, in_features, bias=True)
            self.drop2 = nn.Dropout(drop)

        def forward(self, x):
            return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))

    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = x.mean(dim=1)
            return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

    class SelfAttention2(nn.Module):
        def __init__(self, feature_dim, hidden_size):
            super().__init__()
            self.key = nn.Linear(feature_dim, feature_dim)
            self.query = nn.Linear(feature_dim, feature_dim)
            self.value = nn.Linear(feature_dim, feature_dim)
            self.out = nn.Linear(feature_dim, hidden_size)

        def forward(self, x):
            K, Q, V = self.key(x), self.query(x), self.value(x)
            s = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float32))
            return self.out(torch.matmul(F.softmax(s, dim=-1), V).mean(dim=1))

    class Attention2(nn.Module):
        def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
            super().__init__()
            assert dim % num_heads == 0, "dim should be divisible by num_heads"
            self.num_heads = num_heads
            self.scale = (dim // num_heads) ** -0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(self, x):
            C, G = x.shape
            qkv = self.qkv(x).reshape(C, 3, self.num_heads, G // self.num_heads)
            qkv = qkv.permute(1, 2, 0, 3)  # einops 'c n h fph -> n h c fph'
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = (attn @ v).transpose(0, 1)  # c h fph
            x = x.reshape(C, G)  # einops 'c h fph -> c (h fph)'
            return self.proj_drop(self.proj(x))

    def modulate(x, shift, scale):
        return x * (1 + scale) + shift

    class TimestepEmbedder(nn.Module):
        def __init__(self, hidden_size, frequency_embedding_size=256):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(),
                                     nn.Linear(hidden_size, hidden_size, bias=True))
            self.frequency_embedding_size = frequency_embedding_size

        @staticmethod
        def timestep_embedding(t, dim, max_period=10000):
            half = dim // 2
            freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
            args = t[:, None].float() * freqs[None]
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
            return emb

        def forward(self, t):
            return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))

    class DiTblock(nn.Module):
        def __init__(self, feature_dim=2000, mlp_ratio=4.0, num_heads=10):
            super().__init__()
            self.norm1 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-4)
            self.attn = Attention2(feature_dim, num_heads=num_heads, qkv_bias=True)
            self.norm2 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-4)
            self.mlp = _Mlp(in_features=feature_dim, hidden_features=int(feature_dim * mlp_ratio), act_layer=lambda: nn.GELU(), drop=0)
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(feature_dim, 6 * feature_dim, bias=True))

        def forward(self, x, c):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

    class FinalLayer(nn.Module):
        def __init__(self, hidden_size, out_size):
            super().__init__()
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-4)
            self.linear = nn.Linear(hidden_size, out_size, bias=True)
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

        def forward(self, x, c):
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            return self.linear(modulate(self.norm_final(x), shift, scale))

    class DiT_diff(nn.Module):
        def __init__(self, st_input_size, condi_input_size, hidden_size, depth, classes, num_heads, mlp_ratio=4.0):
            super().__init__()
            self.st_input_size, self.condi_input_size, self.hidden_size = st_input_size, condi_input_size, hidden_size
            self.depth, self.classes, self.num_heads, self.mlp_ratio = depth, classes, num_heads, mlp_ratio
            self.in_layer = nn.Sequential(nn.Linear(self.st_input_size, self.hidden_size))
            self.x_in_layer = nn.Sequential(nn.Linear(self.condi_input_size, self.hidden_size))
            self.cond_layer = nn.Sequential(nn.Linear(self.condi_input_size, self.hidden_size))
            self.cond_layer_atten = SelfAttention2(self.condi_input_size, self.hidden_size * 2)
            self.cond_layer_mlp = SimpleMLP(self.condi_input_size, self.hidden_size, self.hidden_size * 2)
            self.condi_emb = nn.Embedding(self.classes, hidden_size)
            self.time_emb = TimestepEmbedder(hidden_size=self.hidden_size * 2)
            self.mlp = SimpleMLP(self.hidden_size * 2 * 2, self.hidden_size, self.hidden_size * 2)
            self.blks = nn.ModuleList([DiTblock(self.hidden_size * 2, mlp_ratio=self.mlp_ratio, num_heads=self.num_heads)
                                       for _ in range(self.depth)])
            self.out_layer = FinalLayer(self.hidden_size * 2, self.st_input_size)
            self.initialize_weights()

        def initialize_weights(self):
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

            self.apply(_basic_init)
            nn.init.normal_(self.condi_emb.weight, std=0.02)
            nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
            nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)
            for block in self.blks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.out_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.out_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.out_layer.linear.weight, 0)
            nn.init.constant_(self.out_layer.linear.bias, 0)

        def forward(self, x, x_hat, t, y, **kwargs):
            x = self.in_layer(x.float())
            x_hat = self.x_in_layer(x_hat.float())
            c = self.time_emb(t) + self.cond_layer_mlp(y)
            x = torch.cat([x, x_hat], dim=1)
            for blk in self.blks:
                x = blk(x, c)
            return self.out_layer(x, c)

    # ---------------- noise scheduler (diff_scheduler.py, cosine) ----------------
    class NoiseScheduler:
        def __init__(self, num_timesteps=1000, device=None):
            self.num_timesteps = num_timesteps
            ab = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
            betas = [min(1 - ab((i + 1) / num_timesteps) / ab(i / num_timesteps), 0.999) for i in range(num_timesteps)]
            self.betas = torch.from_numpy(np.array(betas).astype(np.float32))
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
            self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
            self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5
            self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
            self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)
            self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
            for k in ("betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev", "sqrt_alphas_cumprod",
                      "sqrt_one_minus_alphas_cumprod", "sqrt_inv_alphas_cumprod", "sqrt_inv_alphas_cumprod_minus_one",
                      "posterior_mean_coef1", "posterior_mean_coef2"):
                setattr(self, k, getattr(self, k).to(device))

        def reconstruct_x0(self, x_t, t, noise):
            s1 = self.sqrt_inv_alphas_cumprod[t].reshape(-1, 1).to(x_t.device)
            s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].reshape(-1, 1).to(x_t.device)
            return torch.clamp(s1 * x_t - s2 * noise, min=-1, max=1)

        def q_posterior(self, x_0, x_t, t):
            s1 = self.posterior_mean_coef1[t].reshape(-1, 1).to(x_t.device)
            s2 = self.posterior_mean_coef2[t].reshape(-1, 1).to(x_t.device)
            return s1 * x_0 + s2 * x_t

        def get_variance(self, t):
            if t == 0:
                return 0
            v = self.betas[t] * (1.0 - self.alphas_cumprod_prev[t]) / (1.0 - self.alphas_cumprod[t])
            return v.clip(1e-20).to(t.device)

        def step(self, model_output, timestep, sample):
            t = timestep
            pred_original_sample = self.reconstruct_x0(sample, t, model_output)
            pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
            variance = 0
            if t > 0:
                noise = torch.randn_like(model_output)
                variance = (self.get_variance(t) ** 0.5) * noise
            return pred_prev_sample + variance, pred_original_sample

        def add_noise(self, x_start, x_noise, timesteps):
            s1 = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1).to(x_start.device)
            s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1).to(x_start.device)
            return s1 * x_start + s2 * x_noise

    class ConditionalDiffusionDataset:
        def __init__(self, a1, a2):
            self.adata_omics1, self.adata_omics2 = a1, a2
            self.st_sample = torch.tensor(a1, dtype=torch.float32)
            self.con_sample = torch.tensor(a2, dtype=torch.float32)
            self.con_data = torch.tensor(a2, dtype=torch.float32)

        def __len__(self):
            return len(self.adata_omics1)

        def __getitem__(self, idx):
            return self.st_sample[idx], self.con_sample[idx], self.con_data

    def normal_train_diff(model, dataloader, lr=1e-4, num_epoch=1400, diffusion_step=1000, device=None, patience=20, progress=False):
        from torch.optim.lr_scheduler import StepLR

        noise_scheduler = NoiseScheduler(num_timesteps=diffusion_step, device=device)
        criterion = nn.MSELoss()
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        best_loss, patience_counter = float("inf"), 0
        model.train()
        epochs = range(num_epoch)
        if progress:
            from tqdm import tqdm
            epochs = tqdm(epochs, ncols=100)
        for epoch in epochs:
            epoch_loss = 0.0
            for i, (x, x_hat, x_cond) in enumerate(dataloader):
                x, x_hat, x_cond = x.float().to(device), x_hat.float().to(device), x_cond.float().to(device)
                x_noise = torch.randn(x.shape).to(device)
                timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).to(device)
                x_t = noise_scheduler.add_noise(x, x_noise, timesteps=timesteps)
                noise_pred = model(x_t, x_hat, t=timesteps.to(device), y=x_cond)
                loss = criterion(x_noise, noise_pred)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
            scheduler.step()
            epoch_loss = epoch_loss / (i + 1)
            if epoch_loss < best_loss:
                best_loss, patience_counter = epoch_loss, 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    def sample_diff(model, dataloader, noise_scheduler, device, num_step=1000, sample_shape=(7060, 2000),
                    sample_intermediate=200, bias=None):
        model.eval()
        x_t = torch.randn(sample_shape[0], sample_shape[1]).to(device)
        timesteps = list(range(num_step))[::-1]
        if sample_intermediate:
            timesteps = timesteps[:sample_intermediate]
        for time in timesteps:
            with torch.no_grad():
                noise, i = [], 0
                for _, x_hat, x_cond in dataloader:
                    x_hat, x_cond = x_hat.float().to(device), x_cond.float().to(device)
                    t = torch.from_numpy(np.repeat(time, x_cond.shape[0])).long().to(device)
                    noise.append(model(x_t[i:i + len(x_cond)], x_hat, t, x_cond, condi_flag=True))
                    i = i + len(x_cond)
                model_output = torch.cat(noise, dim=0)
            x_t, _ = noise_scheduler.step(model_output / bias, torch.from_numpy(np.array(time)).long().to(device), x_t)
        return x_t.detach().cpu().numpy()

    def run_diff(dataset, k=3, batch_size=32, learning_rate=1e-4, num_epoch=1000, diffusion_step=500, hidden_size=256,
                 depth=16, head=8, device="cuda:0", patience=20, bias=None, progress=False):
        from sklearn.model_selection import KFold
        from torch.utils.data import DataLoader, SubsetRandomSampler

        res_mtx = np.zeros((len(dataset), dataset.adata_omics1.shape[1]))
        for fold, (train_idx, val_idx) in enumerate(KFold(n_splits=k, shuffle=True).split(np.arange(len(dataset)))):
            train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
            test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_idx)
            model = DiT_diff(st_input_size=dataset.adata_omics1.shape[1], condi_input_size=dataset.adata_omics2.shape[1],
                             hidden_size=hidden_size, depth=depth, num_heads=head, mlp_ratio=4.0, classes=6)
            model.to(device)
            model.train()
            normal_train_diff(model, dataloader=train_dataloader, lr=learning_rate, num_epoch=num_epoch,
                              diffusion_step=diffusion_step, device=device, patience=patience, progress=progress)
            noise_scheduler = NoiseScheduler(num_timesteps=diffusion_step, device=device)
            model.eval()
            test_gt = torch.cat([b[0] for b in test_dataloader], dim=0)
            res_mtx[val_idx, :] = sample_diff(model, device=device, dataloader=test_dataloader, noise_scheduler=noise_scheduler,
                                              num_step=diffusion_step, sample_shape=(test_gt.shape[0], test_gt.shape[1]),
                                              sample_intermediate=diffusion_step, bias=bias)
        return res_mtx

    # ---------------- integration.py ----------------
    class Encoder(nn.Module):
        def __init__(self, in_feat, out_feat):
            super().__init__()
            self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
            torch.nn.init.xavier_uniform_(self.weight)

        def forward(self, feat, adj):
            return torch.spmm(adj, torch.mm(feat, self.weight))

    class AttentionLayer(nn.Module):
        def __init__(self, in_feat, out_feat):
            super().__init__()
            self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
            self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
            torch.nn.init.xavier_uniform_(self.w_omega)
            torch.nn.init.xavier_uniform_(self.u_omega)

        def forward(self, emb1, emb2):
            emb = torch.cat([torch.unsqueeze(torch.squeeze(emb1), dim=1), torch.unsqueeze(torch.squeeze(emb2), dim=1)], dim=1)
            v = F.tanh(torch.matmul(emb, self.w_omega))
            vu = torch.matmul(v, self.u_omega)
            alpha = F.softmax(torch.squeeze(vu) + 1e-6, dim=1)  # upstream: implicit dim (=1 for 2-D input)
            return torch.squeeze(torch.matmul(torch.transpose(emb, 1, 2), torch.unsqueeze(alpha, -1))), alpha

    class Encode_all(nn.Module):
        def __init__(self, d_in1, d_out1, d_in2, d_out2):
            super().__init__()
            self.encoder_omics1 = Encoder(d_in1, d_out1)
            self.encoder_omics2 = Encoder(d_in2, d_out2)
            self.decoder_omics1 = Encoder(d_out1, d_in1)
            self.decoder_omics2 = Encoder(d_out2, d_in2)
            self.atten_inte = AttentionLayer(d_out1, d_out2)

        def forward(self, f1, f2, as1, af1, as2, af2):
            s1, s2 = self.encoder_omics1(f1, as1), self.encoder_omics2(f2, as2)
            t1, t2 = self.encoder_omics1(f1, af1), self.encoder_omics2(f2, af2)
            sc_ = torch.mean(torch.stack([s1, s2]), dim=0)
            fc_ = torch.mean(torch.stack([t1, t2]), dim=0)
            comb, alpha = self.atten_inte(sc_, fc_)
            return {"emb_latent_spatial_omics1": s1, "emb_latent_spatial_omics2": s2,
                    "emb_latent_feature_omics1": t1, "emb_latent_feature_omics2": t2,
                    "emb_latent_combined": comb, "alpha_omics_1_2": alpha,
                    "emb_recon_spatial_omics1": self.decoder_omics1(comb, as1),
                    "emb_recon_spatial_omics2": self.decoder_omics2(comb, as2),
                    "emb_recon_feature_omics1": self.decoder_omics1(comb, af1),
                    "emb_recon_feature_omics2": self.decoder_omics2(comb, af2)}

    def nt_xent(z1, z2, temperature=0.07):
        n = z1.shape[0]
        labels = torch.cat([torch.arange(n)] * 2, dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z1.device)
        features = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
        sim = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z1.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim = sim[~mask].view(sim.shape[0], -1)
        positives = sim[labels.bool()].view(labels.shape[0], -1)
        negatives = sim[~labels.bool()].view(sim.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1) / temperature
        return nn.CrossEntropyLoss()(logits, torch.zeros(logits.shape[0], dtype=torch.long).to(z1.device))

    def train_and_infer(f1, f2, as1, af1, as2, af2, epochs, device, dim_out1=64, dim_out2=64, learning_rate=0.001,
                        weight_decay=0.0, patience=10, min_delta=0.001):
        model = Encode_all(f1.shape[1], dim_out1, f2.shape[1], dim_out2)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
        model = model.to(device)
        f1, f2, as1, af1, as2, af2 = (t.to(device) for t in (f1, f2, as1, af1, as2, af2))
        best_loss, patience_counter = float("inf"), 0
        model.train()
        for epoch in range(epochs):
            model.train()
            r = model(f1, f2, as1, af1, as2, af2)
            recon = (F.mse_loss(r["emb_recon_spatial_omics1"], f1) + F.mse_loss(r["emb_recon_spatial_omics2"], f2)
                     + F.mse_loss(r["emb_recon_feature_omics1"], f1) + F.mse_loss(r["emb_recon_feature_omics2"], f2))
            contra = nt_xent(r["emb_latent_spatial_omics1"], r["emb_latent_spatial_omics2"]) + \
                nt_xent(r["emb_latent_feature_omics1"], r["emb_latent_feature_omics2"])
            total = recon + contra
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            if total.item() + min_delta < best_loss:
                best_loss, patience_counter = total.item(), 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        model.eval()
        with torch.no_grad():
            return model(f1, f2, as1, af1, as2, af2)

    return dict(train_zinb=train_zinb, train_atac=train_atac, run_diff=run_diff, Dataset=ConditionalDiffusionDataset,
                train_and_infer=train_and_infer)


def candies(
    adata_rna,
    adata_atac,
    *,
    denoise: str = "rna",
    rna_min_cells: int = 10,
    n_top_genes: int = 3000,
    n_lsi: int = 51,
    zinb_epochs: int = 300,
    atac_epochs: int = 600,
    latent_dim: int = 64,
    diff_k: int = 3,
    diff_batch_size: int = 512,
    diff_hidden_size: int = 256,
    diff_lr: float = 1e-3,
    diff_epochs: int = 1000,
    diffusion_step: int = 800,
    diff_depth: int = 6,
    diff_heads: int = 16,
    diff_patience: int = 40,
    exposure_bias: float = 0.5,
    integration_epochs: int = 100,
    seed_offset: int = 0,
    deterministic: bool = True,
    n_domains: Optional[int] = None,
    device: Optional[str] = None,
    key_added: str = "X_candies",
    domain_key: str = "candies_domain",
    progress: bool = False,
):
    """CANDIES denoising + integration of spatial RNA + ATAC (MISAR-seq style data).

    Vendored PyTorch port (no torch_geometric, no timm, no POT, no R) of CANDIES
    (Liu et al., *Adv Sci* 2026; MIT licence). Steps, as in the upstream
    MISAR-seq tutorial: RNA HVG/normalise/scale + PCA, ATAC LSI; a ZINB graph
    auto-encoder (RNA) and a GCN auto-encoder (ATAC) give 64-d embeddings; a
    conditional diffusion transformer (k-fold, cosine schedule) denoises the
    ``denoise`` modality's embedding conditioned on the other one; a dual
    (spatial + feature graph) GCN with attention fusion and NT-Xent alignment
    gives the joint embedding.

    Arguments:
        adata_rna, adata_atac: raw counts, same spots, ``obsm["spatial"]``.
        denoise: ``"rna"`` (tutorial default for MISAR-seq: RNA denoised,
            conditioned on ATAC) or ``"atac"``.
        seed_offset: added to every upstream seed (ZINB 41, ATAC AE 2024,
            diffusion 2024, integration 2025).
        deterministic: ``torch.use_deterministic_algorithms`` during the run
            (upstream sets it globally at import).
        n_domains: if given, domains by mclust ``EEE`` on PCA(20) of the joint
            embedding (upstream ``clustering(method="mclust", use_pca=True)``,
            here via pymclustR).
        device: default ``"cuda:0"`` if available (upstream requires CUDA).

    Returns:
        The joint embedding (also ``obsm[key_added]`` of both inputs;
        ``obsm[key_added + "_denoised"]`` holds the denoised modality).

    Reference:
        Liu Y, et al. Cross-modal denoising and integration of spatial
        multi-omics data with CANDIES. Adv Sci (2026), doi:10.1002/advs.202523754.
    """
    import scanpy as sc
    import torch

    if denoise not in ("rna", "atac"):
        raise ValueError("denoise must be 'rna' or 'atac'")
    dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    prev_det = torch.are_deterministic_algorithms_enabled()
    _seed_everything(2024, deterministic)  # upstream: DiTs.py import side effect
    try:
        B = _build()
        a1 = adata_rna.copy()
        a2 = adata_atac.copy()
        a1.var_names_make_unique()
        a2.var_names_make_unique()
        sc.pp.filter_genes(a1, min_cells=rna_min_cells)
        sc.pp.highly_variable_genes(a1, flavor="seurat_v3", n_top_genes=n_top_genes)
        sc.pp.normalize_total(a1, target_sum=1e4)
        sc.pp.log1p(a1)
        sc.pp.scale(a1)
        a2 = a2[a1.obs_names].copy()
        if "X_lsi" not in a2.obsm.keys():
            sc.pp.highly_variable_genes(a2, flavor="seurat_v3", n_top_genes=n_top_genes)
            candies_lsi(a2, use_highly_variable=False, n_components=n_lsi)
        a2.obsm["feat"] = a2.obsm["X_lsi"].copy()
        a1.obsm["feat"] = _pca(a1[:, a1.var["highly_variable"]], n_comps=a2.obsm["X_lsi"].shape[1])
        a1, a2 = _construct_neighbor_graph(a1, a2)
        adj = _adjacent_matrix_preprocessing(a1, a2)
        adj_sp2 = adj["adj_spatial_omics2"].to(dev)
        emb1, _ = B["train_zinb"](a1, device=torch.device(dev), epochs=zinb_epochs, dim_output=latent_dim,
                                  random_seed=41 + seed_offset, progress=progress)
        _seed_everything(2024 + seed_offset, deterministic)
        emb2 = B["train_atac"](a2.obsm["X_lsi"], adj_sp2, device=dev, epochs=atac_epochs, atac_latent_dim=latent_dim)
        # align the two embeddings by spatial coordinate (tutorial cell)
        df1 = pd.DataFrame(emb1, index=[tuple(c) for c in a1.obsm["spatial"]])
        df2 = pd.DataFrame(emb2, index=[tuple(c) for c in a2.obsm["spatial"]]).reindex(df1.index)
        x1, x2 = df1.to_numpy(), df2.to_numpy()
        target, cond = (x1, x2) if denoise == "rna" else (x2, x1)
        ds = B["Dataset"](target, cond)
        _seed_everything(2024 + seed_offset, deterministic)
        den = B["run_diff"](ds, k=diff_k, batch_size=diff_batch_size, hidden_size=diff_hidden_size, learning_rate=diff_lr,
                            num_epoch=diff_epochs, diffusion_step=diffusion_step, depth=diff_depth, head=diff_heads,
                            device=dev, patience=diff_patience, bias=exposure_bias, progress=progress)
        if denoise == "rna":
            a1.obsm["feat"], a2.obsm["feat"] = den, emb2
        else:
            a1.obsm["feat"], a2.obsm["feat"] = emb1, den
        a1, a2 = _construct_neighbor_graph(a1, a2)
        adj = _adjacent_matrix_preprocessing(a1, a2)
        f1 = torch.FloatTensor(a1.obsm["feat"].copy()).to(dev)
        f2 = torch.FloatTensor(a2.obsm["feat"].copy()).to(dev)
        _seed_everything(2025 + seed_offset, deterministic)
        res = B["train_and_infer"](f1, f2, adj["adj_spatial_omics1"].to(dev), adj["adj_feature_omics1"].to(dev),
                                   adj["adj_spatial_omics2"].to(dev), adj["adj_feature_omics2"].to(dev),
                                   epochs=integration_epochs, device=dev)
        emb = res["emb_latent_combined"].detach().cpu().numpy().copy()
    finally:
        torch.use_deterministic_algorithms(prev_det)
    labels = None
    if n_domains is not None:
        import anndata as ad

        from ._spatial import spatial_domains

        tmp = ad.AnnData(np.zeros((emb.shape[0], 1), dtype=np.float32))
        tmp.obsm["E"] = emb
        labels = spatial_domains(tmp, int(n_domains), use_rep="E", method="mclust", use_pca=True, n_comps=20).to_numpy()
    for a in (adata_rna, adata_atac):
        pos = pd.Index(a.obs_names).get_indexer(a1.obs_names)
        out = np.full((a.n_obs, emb.shape[1]), np.nan, dtype=np.float32)
        out[pos] = emb
        a.obsm[key_added] = out
        d = np.full((a.n_obs, den.shape[1]), np.nan, dtype=np.float32)
        d[pos] = den
        a.obsm[key_added + "_denoised"] = d
        if labels is not None:
            lab = pd.Series(pd.NA, index=a.obs_names, dtype="object")
            lab.iloc[pos] = labels.astype(str)
            a.obs[domain_key] = pd.Categorical(lab)
        a.uns[key_added] = {"method": "CANDIES", "denoise": denoise, "seed_offset": seed_offset}
    return emb
