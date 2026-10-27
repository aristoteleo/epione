# SPDX-License-Identifier: MIT
#
# Vendored port of SpaMosaic (Yan et al., "SpaMosaic: mosaic integration of
# spatial multi-omics", Nature Genetics 2026, doi:10.1038/s41588-026-02573-3).
# Upstream: https://github.com/JinmiaoChenLab/SpaMosaic
#           commit cc1336755de8d1ddd1337b3cd67c9d41a61c2cb4
# Copyright (c) 2025 Jinmiao Lab, MIT licence (full text in
# epione/tl/_licenses/SPAMOSAIC.LICENSE and the repository NOTICE file).
# Parts of SpaMosaic follow STAGATE (spatial graph) and scGLUE (LSI); both MIT.
#
# Changes from upstream (see tests/parity/spatial_ports/spamosaic/MANIFEST.md):
#   * torch_geometric MessagePassing / add_remaining_self_loops and
#     torch_scatter.scatter_add (WLGCN propagation) -> plain torch, same order;
#   * only the default 'wlgcn' architecture (configs/wlgcn.yaml) is ported;
#     the HGT / GAT / hetero-LGCN alternatives (and their PyG HeteroData use)
#     are not;
#   * Annoy is used for the cross-batch MNN search when installed (as
#     upstream); without it an exact scikit-learn search on the same
#     normalised data and metric is used (documented deviation);
#   * Harmony batch correction uses harmony-pytorch when requested
#     (batch_corr=True), as upstream; it is an optional import;
#   * R mclust via rpy2 -> pymclustR; torch.use_deterministic_algorithms(True),
#     which upstream sets globally, is restored after the run.
"""SpaMosaic: modality-agnostic contrastive graph integration of spatial multi-omics."""
from __future__ import annotations

import itertools
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sps

__all__ = ["spamosaic", "spamosaic_preprocess"]

_WLGCN_CONFIG = dict(dropout=0.2, hid_dim=128, n_dec_l=1, n_layer=2, out_dim=32, slope=0.2)


# --------------------------------------------------------------------------
# preprocessing.py
# --------------------------------------------------------------------------
def _sparse_log1p_scale(X, scale=1e4):
    if sps.issparse(X):
        X = X.copy()
        X.data = np.log1p(X.data * scale)
        return X
    return np.log1p(X * scale)


class _TfIdf:
    def fit(self, X):
        self.idf = X.shape[0] / (1e-8 + X.sum(axis=0))
        return self

    def transform(self, X):
        if sps.issparse(X):
            return X.multiply(1 / (1e-8 + X.sum(axis=1))).multiply(self.idf)
        return X / (1e-8 + X.sum(axis=1, keepdims=True)) * self.idf


class _LSI:
    """Upstream ``lsiTransformer`` (TF-IDF, L1, log1p(1e4 x), TruncatedSVD arpack rs=777, z-score)."""

    def __init__(self, n_components=20, drop_first=True, log=True, norm=True, z_score=True, tfidf=True, svd=True):
        import sklearn.decomposition
        import sklearn.preprocessing

        self.drop_first = drop_first
        self.n_components = n_components + drop_first
        self.log, self.norm, self.z_score, self.tfidf, self.svd = log, norm, z_score, tfidf, svd
        self.tf = _TfIdf()
        self.normalizer = sklearn.preprocessing.Normalizer(norm="l1")
        self.pca = sklearn.decomposition.TruncatedSVD(n_components=self.n_components, random_state=777, algorithm="arpack")

    def fit_transform(self, X, obs_names):
        Xf = self.tf.fit(X).transform(X) if self.tfidf else X
        if self.norm:
            Xf = self.normalizer.fit_transform(Xf)
        if self.log:
            Xf = _sparse_log1p_scale(Xf, 1e4)
        self.pca.fit(Xf)
        Xp = self.tf.transform(X) if self.tfidf else X
        if self.norm:
            Xp = self.normalizer.transform(Xp)
        if self.log:
            Xp = _sparse_log1p_scale(Xp, 1e4)
        if self.svd:
            Xp = self.pca.transform(Xp)
        if self.z_score:
            Xp -= Xp.mean(axis=1, keepdims=True)
            Xp /= (1e-8 + Xp.std(axis=1, ddof=1, keepdims=True))
        return pd.DataFrame(Xp, index=obs_names).iloc[:, int(self.drop_first):]


def _harmony(latent, batch_labels, use_gpu=True):
    from harmony import harmonize  # harmony-pytorch (optional)

    df = pd.DataFrame(np.reshape(batch_labels, (-1, 1)), columns=["batch"])
    return harmonize(latent, df, batch_key="batch", use_gpu=use_gpu, verbose=False)


def _split_obsm(ads, ref, key):
    sizes = [a.n_obs for a in ads]
    for a, v in zip(ads, np.split(ref.obsm[key], np.cumsum(sizes[:-1]))):
        a.obsm[key] = v


def spamosaic_preprocess(ads: Sequence, modality: str, *, batch_corr: bool = False, n_features: Optional[int] = None,
                         n_comps: int = 50, batch_key: str = "src", key: str = "dimred_bc"):
    """Upstream ``RNA_preprocess`` (adapted) / ``Epigenome_preprocess`` / ``ADT_preprocess`` (clr), in place.

    ``ads`` is one modality's list of per-batch AnnData (``None`` for missing);
    the result goes to ``obsm[key]`` of each. ``n_features``: HVGs (RNA, default
    5000) or variable peaks (ATAC, default 100000).
    """
    import scanpy as sc

    measured = [a for a in ads if a is not None]
    cat = sc.concat(measured)
    if modality == "rna":
        n = n_features or 5000
        sc.pp.highly_variable_genes(cat, flavor="seurat_v3", n_top_genes=n, batch_key=batch_key)
        sub = cat[:, cat.var.query("highly_variable").index.to_numpy()]
        rep = _LSI(n_components=n_comps, drop_first=False, tfidf=False).fit_transform(sub.X, sub.obs_names).values
    elif modality in ("atac", "epigenome"):
        n = n_features or 100000
        sc.pp.highly_variable_genes(cat, flavor="seurat_v3", n_top_genes=n, batch_key=batch_key)
        sub = cat[:, cat.var.query("highly_variable").index.to_numpy()]
        rep = _LSI(n_components=n_comps, drop_first=True, tfidf=True).fit_transform(sub.X, sub.obs_names).values
    elif modality == "adt":
        X = cat.X.toarray() if sps.issparse(cat.X) else np.array(cat.X)

        def clr(x):
            return np.log1p(x / np.exp(np.sum(np.log1p(x[x > 0])) / len(x)))

        cat.X = np.apply_along_axis(clr, 1, X)
        sc.pp.pca(cat, n_comps=min(n_comps, cat.n_vars - 1))
        rep = cat.obsm["X_pca"]
    else:
        raise ValueError("modality must be 'rna', 'atac' or 'adt'")
    cat.obsm[key] = _harmony(rep, cat.obs[batch_key].to_list()) if (len(measured) > 1 and batch_corr) else rep
    _split_obsm(measured, cat, key)


# --------------------------------------------------------------------------
# build_graph.py, MNN.py
# --------------------------------------------------------------------------
def _spatial_net(adata, rad_cutoff=None, max_neigh=50):
    """STAGATE-style radius graph from up to ``max_neigh`` ball-tree neighbours, plus self loops."""
    import sklearn.neighbors

    coor = np.asarray(adata.obsm["spatial"])
    distances, indices = sklearn.neighbors.NearestNeighbors(n_neighbors=max_neigh + 1, algorithm="ball_tree").fit(coor).kneighbors(coor)
    indices, distances = indices[:, 1:], distances[:, 1:]
    c1 = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
    c2, d = indices.ravel(), distances.ravel()
    keep = d < rad_cutoff
    G = sps.coo_matrix((np.ones(int(keep.sum())), (c1[keep], c2[keep])), shape=(adata.n_obs, adata.n_obs))
    adata.uns["adj"] = G + sps.eye(G.shape[0])


def _nn_search(ds1, ds2, names1, names2, knn, metric="manhattan", norm=True, n_trees=10):
    from sklearn.preprocessing import normalize

    if norm:
        ds1, ds2 = normalize(ds1), normalize(ds2)
    try:
        from annoy import AnnoyIndex

        a = AnnoyIndex(ds2.shape[1], metric=metric)
        for i in range(ds2.shape[0]):
            a.add_item(i, ds2[i, :])
        a.build(n_trees)
        ind = np.array([a.get_nns_by_vector(ds1[i, :], knn, search_k=-1) for i in range(ds1.shape[0])])
    except ImportError:  # exact search, same metric (deviation: upstream is approximate)
        from sklearn.neighbors import NearestNeighbors

        ind = NearestNeighbors(n_neighbors=knn, metric="manhattan" if metric == "manhattan" else "euclidean").fit(ds2).kneighbors(ds1, return_distance=False)
    return {(names1[a], names2[b]) for a, row in enumerate(ind) for b in row}


def _mnn(ds1, ds2, names1, names2, knn1=20, knn2=20):
    m1 = _nn_search(ds1, ds2, names1, names2, knn1)
    m2 = _nn_search(ds2, ds1, names2, names1, knn2)
    return m1 & {(b, a) for a, b in m2}


def _remove_outlier(mnn_set, ad1, ad2, contamination="auto"):
    from sklearn.ensemble import IsolationForest

    pairs = list(mnn_set)
    x1 = ad1[[p[0] for p in pairs]].obsm["spatial"]
    x2 = ad2[[p[1] for p in pairs]].obsm["spatial"]
    y = IsolationForest(max_samples="auto", contamination=contamination, random_state=0).fit_predict(np.c_[x1, x2, x1 - x2])
    return {p for p, v in zip(pairs, y) if v == 1}


def _build_mnn_graph(bridge_ads, test_ads, use_rep, knn_base=10, auto_knn=False, auto_thr=0.8, rmv_outlier=False,
                     contamination="auto"):
    if len(bridge_ads) == 0:
        bridge_ads, test_ads = test_ads, []

    def ksize(ai, aj):
        if not auto_knn:
            return knn_base, knn_base
        r = min(ai.n_obs, aj.n_obs) / max(ai.n_obs, aj.n_obs)
        if r >= auto_thr:
            return knn_base, knn_base
        return (max(1, int(knn_base * r)), knn_base) if ai.n_obs > aj.n_obs else (knn_base, max(1, int(knn_base * r)))

    def compute(ai, aj):
        k1, k2 = ksize(ai, aj)
        pairs = _mnn(ai.obsm[use_rep], aj.obsm[use_rep], ai.obs_names, aj.obs_names, knn1=k1, knn2=k2)
        return _remove_outlier(pairs, ai, aj, contamination) if (rmv_outlier and pairs) else pairs

    out = set()
    for i in range(len(bridge_ads)):
        for j in range(i + 1, len(bridge_ads)):
            out |= compute(bridge_ads[i], bridge_ads[j])
    for i in range(len(bridge_ads)):
        for j in range(len(test_ads)):
            out |= compute(bridge_ads[i], test_ads[j])
    return out


# --------------------------------------------------------------------------
# architectures/wlgcn.py, loss.py, train_utils.py
# --------------------------------------------------------------------------
def _build():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from ._gnn import scatter_sum

    def add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes):
        """PyG ``add_remaining_self_loops`` with edge weights: existing loops keep their weight."""
        mask = edge_index[0] != edge_index[1]
        loop_index = torch.arange(0, num_nodes, device=edge_index.device).view(1, -1).repeat(2, 1)
        loop_attr = edge_weight.new_full((num_nodes,), fill_value)
        inv = ~mask
        loop_attr[edge_index[0][inv]] = edge_weight[inv]
        return torch.cat([edge_index[:, mask], loop_index], dim=1), torch.cat([edge_weight[mask], loop_attr], dim=0)

    def sym_norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 2 if improved else 1, num_nodes)
        row, col = edge_index
        deg = scatter_sum(edge_weight, row, num_nodes)
        dis = deg.pow(-0.5)
        dis[dis == float("inf")] = 0
        return edge_index, dis[row] * edge_weight * dis[col]

    class WLGCN_vanilla(nn.Module):
        def __init__(self, K=1):
            super().__init__()
            self.K = K

        def forward(self, x, edge_index, edge_weight=None):
            ei, norm = sym_norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)
            xs = [x]
            for _ in range(self.K):
                xs.append(scatter_sum(norm.view(-1, 1) * xs[-1].index_select(0, ei[0]), ei[1], x.size(0)))
            return torch.cat(xs, dim=1)

    class WLGCN(nn.Module):
        def __init__(self, input_size, output_size, K=8, dec_l=1, hidden_size=512, dropout=0.2, slope=0.2):
            super().__init__()
            self.conv1 = WLGCN_vanilla(K=K)
            self.fc1 = nn.Linear(input_size * (K + 1), hidden_size)
            self.bn = nn.BatchNorm1d(hidden_size)
            self.dropout1 = nn.Dropout(p=dropout)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.negative_slope = slope
            if dec_l == 1:
                self.decoder = nn.Linear(output_size, input_size)
            else:
                self.decoder = nn.Sequential(nn.Linear(output_size, output_size), nn.ReLU(), nn.Linear(output_size, input_size))

        def forward(self, feature, edge_index, edge_weight=None):
            x = self.conv1(feature, edge_index, edge_weight)
            x = F.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)
            x = self.dropout1(self.bn(x))
            x = self.fc2(x)
            r = self.decoder(x)
            return F.normalize(x, p=2, dim=1), r

    class CL_loss(nn.Module):
        def __init__(self, batch_size, rep=3, bias=0):
            super().__init__()
            self.register_buffer("negatives_mask", (~torch.eye(batch_size * rep, batch_size * rep, dtype=bool)).float())
            iids = torch.arange(batch_size).repeat(rep)
            pos = iids.view(-1, 1) == iids.view(1, -1)
            pos[torch.eye(pos.size(0), dtype=torch.bool)] = 0
            self.register_buffer("pos_mask", pos.float())
            self.bias = bias

        def forward(self, simi):
            simi_max, _ = torch.max(simi, dim=1, keepdim=True)
            simi = simi - simi_max.detach()
            positives = (simi * self.pos_mask).sum(dim=1) / self.pos_mask.sum(dim=1)
            negatives = (torch.exp(simi) * self.negatives_mask).sum(dim=1)
            return -(positives - torch.log(negatives + self.bias)).mean()

    def graph_decode(z, edge_index):
        return torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1))

    def negative_sampling(edge_index, num_nodes, num_neg):
        """Uniform node-pair negatives avoiding observed edges (stand-in for PyG negative_sampling)."""
        existing = set((edge_index[0] * num_nodes + edge_index[1]).tolist())
        out = []
        while len(out) < num_neg:
            cand = torch.randint(0, num_nodes * num_nodes, (2 * (num_neg - len(out)) + 16,)).tolist()
            out.extend(c for c in cand if c not in existing)
        idx = torch.tensor(out[:num_neg], device=edge_index.device)
        return torch.stack([idx // num_nodes, idx % num_nodes])

    def graph_recon_crit(z, pos):
        pos_loss = -torch.log(graph_decode(z, pos) + 1e-20).mean()
        neg = negative_sampling(pos, z.size(0), pos.size(1))
        return pos_loss - torch.log(1 - graph_decode(z, neg) + 1e-20).mean()

    def train_model(mod_model, mod_optims, crit1, crit2, loss_type, w_rec_g, feats, edges, edge_w, edge_c, meta,
                    split, T, n_epochs, device):
        bridge = [bi for bi, v in meta.items() if v[0]]
        test = [bi for bi, v in meta.items() if not v[0]]
        for epoch in range(1, n_epochs + 1):
            for k in mod_model:
                mod_model[k].train()
            for k in mod_optims:
                mod_optims[k].zero_grad()
            embs, recs = {}, {}
            for k in mod_model:
                z, r = mod_model[k](feats[k], edges[k], edge_w[k])
                embs[k] = list(torch.split(z, split[k]))
                recs[k] = list(torch.split(r, split[k]))
            split_in = {k: list(torch.split(v, split[k])) for k, v in feats.items()}
            l2, l2_step = torch.FloatTensor([0]).to(device), 0
            for bi in test:
                for k in meta[bi][1]:
                    f_rec = 0 if w_rec_g == 1 else crit2(recs[k][bi], split_in[k][bi])
                    if w_rec_g > 0:
                        sub = edges[k][:, edge_c[k] == bi]
                        g_rec = graph_recon_crit(embs[k][bi], sub - torch.unique(sub).min())
                    else:
                        g_rec = 0.0
                    l2 += w_rec_g * g_rec + (1 - w_rec_g) * f_rec
                    l2_step += 1
            l1, l1_step = 0, 0
            for bi in bridge:
                m = meta[bi]
                if m[3] > 1:
                    perm = torch.randperm(m[1]).to(device)
                    for k in m[5]:
                        embs[k][bi] = embs[k][bi][perm]
                for i in range(m[3]):
                    f = [embs[k][bi][i * m[2]:(i + 1) * m[2]] for k in m[5]]
                    if loss_type == "adapted":
                        f = torch.cat(f)
                        l1 += crit1[bi]((f @ f.t()) / T)
                    else:
                        logit = (f[0] @ f[1].T) / T
                        target = torch.arange(m[2]).to(device)
                        l1 += 0.5 * (crit1(logit, target) + crit1(logit.T, target))
                    l1_step += 1
            loss = l1 / max(1, l1_step) + l2 / max(1, l2_step)
            loss.backward()
            for k in mod_optims:
                mod_optims[k].step()
        return mod_model

    return WLGCN, CL_loss, train_model


def spamosaic(
    mod_batch: Dict[str, List],
    *,
    input_key: str = "dimred_bc",
    batch_key: str = "src",
    radius_cutoff: float = 2000,
    intra_knns=10,
    inter_knn_base: int = 10,
    inter_auto_knn: bool = False,
    inter_auto_thr: float = 0.8,
    rmv_outlier: bool = False,
    contamination="auto",
    w_g: float = 0.8,
    seed: int = 1234,
    lr: float = 0.01,
    n_epochs: int = 100,
    T: float = 0.01,
    bias: float = 0,
    loss_type: str = "adapted",
    use_mini_thr: int = 8000,
    mini_batch_size: int = 1024,
    w_rec_g: float = 0.0,
    n_clusters: Optional[int] = None,
    device: Optional[str] = None,
    key_added: str = "X_spamosaic",
    domain_key: str = "spamosaic_domain",
):
    """SpaMosaic joint embedding of spatial multi-omics batches (vertical / horizontal / mosaic).

    Vendored PyTorch port (no torch_geometric; WLGCN backbone only) of SpaMosaic
    (Yan et al., *Nat Genet* 2026; MIT, Jinmiao Lab). ``mod_batch`` maps each
    modality (``"rna"``, ``"atac"``, ``"adt"``) to a list with one AnnData (or
    ``None`` if that batch lacks the modality) per batch; every AnnData needs
    ``obsm[input_key]`` (see :func:`spamosaic_preprocess`), ``obsm["spatial"]``
    and ``obs[batch_key]``. Per modality, a spatial radius graph per batch plus
    mutual-nearest-neighbour edges across batches (weight ``w_g``) feed a
    2-hop light-GCN encoder; batches measuring >= 2 modalities ("bridges") align
    the modality embeddings with a multi-positive contrastive loss, single-modality
    batches are trained by feature reconstruction. The embedding of a spot is the
    mean of its measured modalities' embeddings.

    Arguments:
        n_clusters: if given, mclust ``EEE`` domains (pymclustR) on the concatenated
            embedding (upstream tutorial ``clustering(algo="mclust")``).
        Other arguments: upstream ``SpaMosaic(...)`` / ``.train(net="wlgcn", ...)``.

    Returns:
        List (one per batch) of AnnData with ``obsm[key_added]`` and ``obsm["spatial"]``
        (and ``obs[domain_key]``); embeddings are also written to the inputs.
    """
    import anndata as ad
    import torch
    import torch.nn as nn

    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    prev_det = torch.are_deterministic_algorithms_enabled()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)
    try:
        WLGCN, CL_loss, train_model = _build()
        mods = list(mod_batch.keys())
        n_batches = len(mod_batch[mods[0]])
        barc2batch = {}
        for i in range(n_batches):
            for m in mods:
                if mod_batch[m][i] is not None:
                    barc2batch.update({b: i for b in mod_batch[m][i].obs_names})
                    break
        knns = list(intra_knns) if isinstance(intra_knns, (list, tuple)) else [intra_knns] * n_batches
        for m in mods:
            for k, a in zip(knns, mod_batch[m]):
                if a is not None:
                    _spatial_net(a, rad_cutoff=radius_cutoff, max_neigh=k)
        mask = np.array([[mod_batch[m][b] is not None for b in range(n_batches)] for m in mods])
        bridge_ids = np.where(mask.sum(0) >= 2)[0]
        test_ids = np.where(mask.sum(0) < 2)[0]
        split = {m: [mod_batch[m][b].n_obs if mod_batch[m][b] is not None else 0 for b in range(n_batches)] for m in mods}
        bridge2mod = {b: [m for m in mods if mod_batch[m][b] is not None] for b in bridge_ids}
        test2mod = {b: [m for m in mods if mod_batch[m][b] is not None] for b in test_ids}
        mnn = {}
        for m in mods:
            if w_g != 0:
                mnn[m] = _build_mnn_graph([mod_batch[m][b] for b in bridge_ids if mod_batch[m][b] is not None],
                                          [mod_batch[m][b] for b in test_ids if mod_batch[m][b] is not None], input_key,
                                          inter_knn_base, inter_auto_knn, inter_auto_thr, rmv_outlier, contamination)
            else:
                mnn[m] = set()
        feats, edges, edge_w, edge_c, dims = {}, {}, {}, {}, {}
        for m in mods:
            ads = [a for a in mod_batch[m] if a is not None]
            obs = pd.concat([a.obs.copy() for a in ads])
            X = np.vstack([a.obsm[input_key] for a in ads])
            intra = sps.block_diag([a.uns["adj"] for a in ads]).tocoo()
            name2idx = dict(zip(obs.index, np.arange(len(obs))))
            idx2name = {v: k for k, v in name2idx.items()}
            if len(mnn[m]) > 0:
                mi = np.array([[name2idx[e[0]], name2idx[e[1]]] for e in mnn[m]])
                ms = np.vstack([mi, mi[:, ::-1]])
                inter = sps.coo_matrix((np.ones(len(ms)), (ms[:, 0], ms[:, 1])), shape=(len(obs), len(obs))) * w_g
            else:
                inter = sps.coo_matrix((len(obs), len(obs)))
            g = (intra + inter).tocoo() if w_g != 0 else intra
            edges[m] = torch.tensor(np.vstack([g.row, g.col]), dtype=torch.long).to(dev)
            edge_w[m] = torch.tensor(g.data, dtype=torch.float32).to(dev)
            feats[m] = torch.tensor(X, dtype=torch.float32).to(dev)
            dims[m] = X.shape[1]
            b1 = np.array([barc2batch[idx2name[i]] for i in g.row])
            b2 = np.array([barc2batch[idx2name[i]] for i in g.col])
            edge_c[m] = torch.tensor(np.where(b1 == b2, b1, -1), dtype=torch.long).to(dev)
        c = _WLGCN_CONFIG
        models = {m: WLGCN(dims[m], c["out_dim"], K=c["n_layer"], dec_l=c["n_dec_l"], hidden_size=c["hid_dim"],
                           dropout=c["dropout"], slope=c["slope"]).to(dev) for m in mods}
        optims = {m: torch.optim.Adam(models[m].parameters(), lr=lr, weight_decay=5e-4) for m in mods}
        meta = {}
        for b, ms in bridge2mod.items():
            n_cell = split[ms[0]][b]
            nlb = n_cell if n_cell <= use_mini_thr else mini_batch_size
            meta[b] = (True, n_cell, nlb, n_cell // nlb, len(ms), ms)
        for b, ms in test2mod.items():
            meta[b] = (False, ms)
        crit1 = ({b: CL_loss(meta[b][2], rep=meta[b][4], bias=bias).to(dev) for b in bridge_ids}
                 if loss_type == "adapted" else nn.CrossEntropyLoss().to(dev))
        models = train_model(models, optims, crit1, nn.MSELoss().to(dev), loss_type, w_rec_g, feats, edges, edge_w,
                             edge_c, meta, split, T, n_epochs, dev)
        for m in mods:
            models[m].eval()
            z, _ = models[m](feats[m], edges[m], edge_w[m])
            for b, zb in enumerate(torch.split(z, split[m])):
                if mod_batch[m][b] is not None:
                    mod_batch[m][b].obsm[key_added] = zb.detach().cpu().numpy()
    finally:
        torch.use_deterministic_algorithms(prev_det)
    finals = []
    for b in range(n_batches):
        embs = [mod_batch[m][b].obsm[key_added] for m in mods if mod_batch[m][b] is not None]
        tmp = [mod_batch[m][b] for m in mods if mod_batch[m][b] is not None][-1]
        a = ad.AnnData(np.zeros((embs[0].shape[0], 2)), obs=tmp.obs.copy(), obsm={"spatial": tmp.obsm["spatial"]})
        a.obsm[key_added] = np.mean(embs, axis=0)
        finals.append(a)
    if n_clusters is not None:
        from ._spatial_methods import _mclust

        lab = _mclust(np.vstack([a.obsm[key_added] for a in finals]), n_clusters, use_pca=False)
        for a, l in zip(finals, np.split(lab, np.cumsum([x.n_obs for x in finals])[:-1])):
            a.obs[domain_key] = pd.Categorical(l.astype(str))
    return finals
