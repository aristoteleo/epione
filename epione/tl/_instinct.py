# SPDX-License-Identifier: MIT
#
# Vendored port of INSTINCT (Liu, Li, Chen, Cui, Gao & Jiang, "Multi-sample
# integration of spatial chromatin accessibility sequencing data via
# stochastic domain translation", Nature Communications 16, 2025,
# doi:10.1038/s41467-025-56535-0).
# Upstream: https://github.com/yyLIU12138/INSTINCT
#           commit 5753b31fb1b9607511bb28d0904c89c69b70d3cf
# Copyright (c) 2024 Yuyao LIU, MIT licence (full text in
# epione/tl/_licenses/INSTINCT.LICENSE and the repository NOTICE file).
#
# INSTINCT is plain PyTorch upstream; the networks, losses and the training
# loop below are verbatim. Changes (see tests/parity/spatial_instinct/MANIFEST.md):
#   * episcanpy.pp.filter_features / filter_cells -> the scanpy calls they wrap;
#   * peak_sets_alignment (GenomicRanges / IRanges / biocutils) is re-implemented
#     with numpy + scipy.sparse (same merge, filter and summation rules);
#   * print statements are behind ``verbose``; the model's device is an argument;
#   * a functional entry point instinct() runs the upstream tutorial pipeline.
"""INSTINCT: multi-slice spatial ATAC integration by stochastic domain translation."""
from __future__ import annotations

import random
import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["instinct", "instinct_preprocess", "peak_sets_alignment"]


# --------------------------------------------------------------------------
# utils.py
# --------------------------------------------------------------------------
def TFIDF(count_mat, type_=2):
    from ._present import TFIDF as _tfidf

    return _tfidf(count_mat, type_=type_)


def _parse_peaks(names, sep=(":", "-")):
    chrom, start, end = [], [], []
    for region in names:
        chrom.append(region.split(sep[0])[0])
        if sep[0] == sep[1]:
            s, e = region.split(sep[0])[1:]
        else:
            s, e = region.split(sep[0])[1].split(sep[1])
        start.append(int(s))
        end.append(int(e))
    return np.asarray(chrom, dtype=object), np.asarray(start, dtype=np.int64), np.asarray(end, dtype=np.int64)


def peak_sets_alignment(adata_list, sep=(":", "-"), min_width=20, max_width=10000, min_gap_width=1,
                        peak_region: Optional[str] = None, verbose: bool = False):
    """Merge the peak sets of several samples into one common set.

    Re-implementation (numpy) of upstream ``peak_sets_alignment``, which uses
    BiocPy ``GenomicRanges``: every sample's peaks are sorted (chromosome name,
    start, end), all peaks are pooled and reduced (an interval whose start is
    less than ``min_gap_width`` past the running end is merged into it), merged
    regions that contain at least one peak of every sample and whose width is
    in ``(min_width, max_width)`` are kept, and each sample's counts are summed
    over its peaks inside each kept region. Peak names are ``chr:start-end``
    (end exclusive, as written by upstream).

    Returns:
        List of new AnnData objects (obs / obsm copied) with the common peaks.
    """
    import anndata as ad

    per = []
    for a in adata_list:
        regions = a.var_names if peak_region is None else a.obs[peak_region]
        c, s, e = _parse_peaks(list(regions), sep)
        order = np.lexsort((e, s, c.astype(str)))
        per.append((c[order].astype(str), s[order], e[order], order))
    allc = np.concatenate([p[0] for p in per]); alls = np.concatenate([p[1] for p in per])
    alle = np.concatenate([p[2] for p in per])
    o = np.lexsort((alle, alls, allc))
    allc, alls, alle = allc[o], alls[o], alle[o]
    mc, ms, me = [], [], []
    cur_c, cur_s, cur_e = None, None, None
    for c, s, e in zip(allc.tolist(), alls.tolist(), alle.tolist()):
        if cur_c == c and s - cur_e < min_gap_width:
            if e > cur_e:
                cur_e = e
        else:
            if cur_c is not None:
                mc.append(cur_c); ms.append(cur_s); me.append(cur_e)
            cur_c, cur_s, cur_e = c, s, e
    if cur_c is not None:
        mc.append(cur_c); ms.append(cur_s); me.append(cur_e)
    mc = np.asarray(mc); ms = np.asarray(ms, dtype=np.int64); me = np.asarray(me, dtype=np.int64)
    if verbose:
        print("Peak merged")
    # region of every sample peak (regions are disjoint and sorted)
    chrom_rows = {ch: np.where(mc == ch)[0] for ch in np.unique(mc)}
    region_of, present = [], np.ones(len(mc), dtype=bool)
    for c, s, e, _ in per:
        r = np.empty(len(c), dtype=np.int64)
        for ch in np.unique(c):
            m = c == ch
            rows = chrom_rows[ch]
            r[m] = rows[np.searchsorted(ms[rows], s[m], side="right") - 1]
        region_of.append(r)
        hit = np.zeros(len(mc), dtype=bool)
        hit[r] = True
        present &= hit
    width = me - ms
    keep = present & (width > min_width) & (width < max_width)
    kept = np.where(keep)[0]
    col_of = np.full(len(mc), -1, dtype=np.int64)
    col_of[kept] = np.arange(kept.size)
    names = [f"{c}:{s}-{e}" for c, s, e in zip(mc[kept], ms[kept], me[kept])]
    if verbose:
        print("Peak filtered")
    out = []
    for a, (c, s, e, order), r in zip(adata_list, per, region_of):
        X = sp.csc_matrix(a.X)[:, order]
        cols = col_of[r]
        ok = cols >= 0
        M = sp.csr_matrix((np.ones(int(ok.sum()), dtype=np.int64), (np.where(ok)[0], cols[ok])), shape=(len(c), kept.size))
        Xm = (X @ M).tocsr()
        out.append(ad.AnnData(Xm, obs=a.obs, var=pd.DataFrame(index=names), obsm=a.obsm))
    if verbose:
        print("Matrix merged")
    return out


def instinct_preprocess(adata_list, adata_concat, binarize=False, use_fragment_count=False, tfidf=True,
                        min_cells_rate=0.03, min_features=1, tfidf_type=2):
    """Upstream ``preprocess_CAS``: peak / spot filters, optional read->fragment, TF-IDF; in place."""
    import scanpy as sc

    sc.pp.filter_genes(adata_concat, min_cells=int(min_cells_rate * adata_concat.shape[0]))
    sc.pp.filter_cells(adata_concat, min_genes=min_features)
    if "n_genes" in adata_concat.obs.columns:
        adata_concat.obs["nb_features"] = adata_concat.obs.pop("n_genes")
    if binarize and use_fragment_count:
        raise ValueError("'binarize' and 'use_fragment_count' cannot be set to True at the same time !")
    elif binarize:
        X = adata_concat.X.copy()
        if sp.issparse(X):
            X.data = (X.data != 0).astype(X.dtype)
        else:
            X = (X != 0).astype(X.dtype)
        adata_concat.X = X
    elif use_fragment_count:
        adata_concat.X = sp.csr_matrix(np.ceil((adata_concat.X / 2).toarray()))
    if tfidf:
        adata_concat.X = TFIDF(adata_concat.X.T, type_=tfidf_type).T.copy()
    else:
        sc.pp.normalize_total(adata_concat, target_sum=10000)
        sc.pp.log1p(adata_concat)
    for i in range(len(adata_list)):
        obs_list = [item for item in adata_list[i].obs_names if item in adata_concat.obs_names]
        adata_list[i] = adata_list[i][obs_list, adata_concat.var_names]


def _create_neighbor_graph(adata_list, adata_concat, rad_cutoff=None, rad_coef=1.5, coor_key="spatial", verbose=False):
    from sklearn.metrics import pairwise_distances

    rad_cutoff = rad_cutoff or []
    G_list = []
    if len(rad_cutoff) == 0:
        for i in range(len(adata_list)):
            loc = np.array(adata_list[i].obsm[coor_key])
            d = pairwise_distances(loc)
            min_dist = np.sort(np.unique(d), axis=None)[1]
            G_list.append((d < rad_coef * min_dist).astype(float))
    elif len(rad_cutoff) != len(adata_list):
        raise ValueError("The length of 'rad_cutoff' should be the number of slices !")
    else:
        for i in range(len(adata_list)):
            loc = np.array(adata_list[i].obsm[coor_key])
            G_list.append((pairwise_distances(loc) < rad_cutoff[i]).astype(float))
    if verbose:
        for i, g in enumerate(G_list):
            print(f"slice {i}: {np.mean(np.sum(g, axis=1)):.4f} neighbors per spot on average including itself")
    adata_concat.uns["graph_list"] = G_list


# --------------------------------------------------------------------------
# networks.py, loss_functions.py, model.py (verbatim apart from printing)
# --------------------------------------------------------------------------
def _build():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    class GATlayer(nn.Module):
        def __init__(self, input_dim, output_dim, is_last=False):
            super().__init__()
            self.is_last = is_last
            self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
            self.v0 = nn.Parameter(torch.Tensor(output_dim, 1))
            self.v1 = nn.Parameter(torch.Tensor(output_dim, 1))
            nn.init.xavier_uniform_(self.W)
            nn.init.xavier_uniform_(self.v0)
            nn.init.xavier_uniform_(self.v1)
            self.alphas = None

        def forward(self, node_features, graph, tied_alphas=None):
            X = torch.matmul(node_features, self.W)
            if not self.is_last:
                if tied_alphas is not None:
                    alphas = tied_alphas
                else:
                    f0 = torch.matmul(X, self.v0)
                    f1 = torch.matmul(X, self.v1)
                    E = graph * (f0 + f1.T)
                    alphas = (torch.sigmoid(E) - 0.5).to_sparse()
                    alphas = torch.sparse.softmax(alphas, dim=1)
                    alphas = alphas.to_dense()
                    self.alphas = alphas
                X = torch.matmul(alphas, X)
                X = F.elu(X, alpha=1.0)
            return X

    class _Stack(nn.Module):
        def forward(self, *a):
            Z = self.layer_list[0](*a)
            if self.num_layer > 1:
                for i in range(1, len(self.layer_list)):
                    Z = self.layer_list[i](Z, *a[1:])
            return Z

    def _stack(self, layer_cls, input_dim, hidden_dims, out_dim):
        self.num_layer = len(hidden_dims) + 1
        self.layer_list = nn.ModuleList()
        if self.num_layer != 1:
            self.layer_list.append(layer_cls(input_dim, hidden_dims[0]))
            if self.num_layer >= 3:
                for i in range(1, len(hidden_dims)):
                    self.layer_list.append(layer_cls(hidden_dims[i - 1], hidden_dims[i]))
            self.layer_list.append(layer_cls(hidden_dims[-1], out_dim, is_last=True))
        else:
            self.layer_list.append(layer_cls(input_dim, out_dim, is_last=True))

    class GATEncoder(_Stack):
        def __init__(self, input_dim, hidden_dims, latent_dim):
            super().__init__()
            _stack(self, GATlayer, input_dim, hidden_dims, latent_dim)

    class MLPSingleLayer(nn.Module):
        def __init__(self, input_dim, output_dim, is_last=False):
            super().__init__()
            self.is_last = is_last
            self.linear = nn.Linear(input_dim, output_dim)
            nn.init.xavier_uniform_(self.linear.weight)

        def forward(self, node_features):
            X = self.linear(node_features)
            if not self.is_last:
                X = F.elu(X, alpha=1.0)
            return X

    class MLPEncoder(_Stack):
        def __init__(self, input_dim, hidden_dims, latent_dim):
            super().__init__()
            _stack(self, MLPSingleLayer, input_dim, hidden_dims, latent_dim)

    class MLPDecoder(_Stack):
        def __init__(self, latent_dim, hidden_dims, output_dim):
            super().__init__()
            _stack(self, MLPSingleLayer, latent_dim, hidden_dims, output_dim)

    class Discriminator(nn.Module):
        def __init__(self, input_dim, n_cls, hidden_dim_list):
            super().__init__()
            self.num_layer = len(hidden_dim_list) + 1
            if self.num_layer != 1:
                self.linear_list = nn.ModuleList()
                linear = nn.Linear(input_dim, hidden_dim_list[0])
                nn.init.xavier_uniform_(linear.weight)
                self.linear_list.append(linear)
                self.src_layer = nn.Linear(hidden_dim_list[-1], 1)
                nn.init.xavier_uniform_(self.src_layer.weight)
                self.cls_layer = nn.Linear(hidden_dim_list[-1], n_cls)
                nn.init.xavier_uniform_(self.cls_layer.weight)
            else:
                self.src_layer = nn.Linear(input_dim, 1)
                nn.init.xavier_uniform_(self.src_layer.weight)
                self.cls_layer = nn.Linear(input_dim, n_cls)
                nn.init.xavier_uniform_(self.cls_layer.weight)
            if self.num_layer >= 3:
                for i in range(1, len(hidden_dim_list)):
                    linear = nn.Linear(hidden_dim_list[i - 1], hidden_dim_list[i])
                    nn.init.xavier_uniform_(linear.weight)
                    self.linear_list.append(linear)

        def forward(self, X):
            if self.num_layer != 1:
                h = self.linear_list[0](X)
                h = torch.softmax(h, dim=1)
                if self.num_layer >= 3:
                    for i in range(1, self.num_layer - 1):
                        h = self.linear_list[i](h)
                        h = torch.softmax(h, dim=1)
                return self.src_layer(h), self.cls_layer(h)
            return self.src_layer(X), self.cls_layer(X)

    class NoiseGenerator(nn.Module):
        def __init__(self, latent_dim, n_cls):
            super().__init__()
            self.linear = nn.Linear(latent_dim + n_cls, latent_dim)
            nn.init.zeros_(self.linear.weight.data)
            nn.init.zeros_(self.linear.bias.data)

        def forward(self, latent_features, node_cls):
            return self.linear(torch.concat([latent_features, node_cls], dim=1))

    def classification_loss(cls_output, cls_target):
        return F.cross_entropy(cls_output, cls_target)

    def adversarial_loss_g(Y_src):
        return (torch.log1p(torch.exp(-Y_src))).mean()

    def adversarial_loss_d(X_src, Y_src):
        return (torch.log1p(torch.exp(-X_src)) + torch.log1p(torch.exp(Y_src))).mean()

    def latent_loss(D_concat, alpha, margin=None):
        if margin:
            return torch.mean(torch.relu(torch.clamp_max(D_concat, margin) + alpha))
        return torch.mean(torch.relu(D_concat + alpha))

    def mse_loss(X, X_rec):
        return torch.mean((X_rec - X) ** 2)

    class INSTINCT_Model:
        """Upstream ``INSTINCT_Model`` (GAT encoder) / ``INSTINCT_MLP_Model`` (``use_graph=False``)."""

        def __init__(self, adata_list, adata_concat, input_mat_key="X_pca", input_dim=100, hidden_dims_G=(50,),
                     latent_dim=30, hidden_dims_D=(50,), lambda_adv=1, lambda_cls=10, lambda_la=20, lambda_rec=10,
                     seed=1236, learn_rates=(1e-3, 5e-4), training_steps=(500, 500), use_cos=True, margin=10,
                     alpha=1, k=50, device=None, use_graph=True, verbose=False):
            hidden_dims_G, hidden_dims_D = list(hidden_dims_G), list(hidden_dims_D)
            learn_rates, training_steps = list(learn_rates), list(training_steps)
            self.lambda_adv, self.lambda_cls, self.lambda_la, self.lambda_rec = lambda_adv, lambda_cls, lambda_la, lambda_rec
            self.latent_dim, self.use_cos, self.margin, self.alpha, self.k = latent_dim, use_cos, margin, alpha, k
            self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            self.use_graph, self.verbose = use_graph, verbose
            self.n_cls = len(adata_list)
            self.n_spots = adata_concat.shape[0]
            self.training_steps = training_steps
            self.slice_index = np.zeros((self.n_spots,))
            self.indices = [0]
            idx = 0
            for i in range(self.n_cls):
                idx += adata_list[i].shape[0]
                self.indices.append(idx)
                self.slice_index[self.indices[-2]:self.indices[-1]] = i
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True
            Enc = GATEncoder if use_graph else MLPEncoder
            self.encoder = Enc(input_dim, hidden_dims_G, latent_dim).to(self.device)
            self.decoder = MLPDecoder(latent_dim, hidden_dims_G, input_dim).to(self.device)
            self.noise_generator = NoiseGenerator(latent_dim, self.n_cls).to(self.device)
            self.discriminator = Discriminator(input_dim, self.n_cls, hidden_dims_D).to(self.device)
            self.params_G = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.noise_generator.parameters())
            self.params_D = self.discriminator.parameters()
            self.optimizer_G = optim.Adam(self.params_G, lr=learn_rates[0])
            self.optimizer_D = optim.Adam(self.params_D, lr=learn_rates[1])
            self.X = torch.from_numpy(adata_concat.obsm[input_mat_key]).float().to(self.device)
            self.X_target = torch.tensor(self.slice_index, dtype=torch.int64).to(self.device)
            self.X_target_matrix = torch.eye(self.n_cls)[self.slice_index.astype(float)].to(self.device)
            if not use_graph:
                self.G = None
            elif "graph_list" in adata_concat.uns.keys():
                G = np.zeros((adata_concat.X.shape[0], adata_concat.X.shape[0]), dtype=float)
                for i in range(len(adata_list)):
                    G[self.indices[i]:self.indices[i + 1], self.indices[i]:self.indices[i + 1]] = adata_concat.uns["graph_list"][i].copy().astype(float)
                self.G = torch.from_numpy(G).float().to(self.device)
            elif "graph" in adata_concat.obsm.keys():
                self.G = torch.from_numpy(adata_concat.obsm["graph"].copy()).float().to(self.device)
            else:
                raise ValueError("No graph is provided !")

        def _enc(self):
            return self.encoder(self.X, self.G) if self.use_graph else self.encoder(self.X)

        def train(self, report_loss=False, report_interval=100):
            t0 = time.time()
            for m in (self.encoder, self.decoder, self.noise_generator, self.discriminator):
                m.train()
            for step in range(self.training_steps[0]):
                Z_X = self._enc()
                X_rec = self.decoder(Z_X)
                X_src, X_cls = self.discriminator(self.X)
                X_rec_src, X_rec_cls = self.discriminator(X_rec)
                loss_adv_g = adversarial_loss_g(X_rec_src)
                loss_adv_d = adversarial_loss_d(X_src, X_rec_src)
                loss_rec = mse_loss(self.X, X_rec)
                loss_cls_real = classification_loss(X_cls, self.X_target)
                loss_cls_fake = classification_loss(X_rec_cls, self.X_target)
                loss_d = self.lambda_adv * loss_adv_d + self.lambda_cls * loss_cls_real
                self.optimizer_D.zero_grad()
                loss_d.backward(retain_graph=True)
                loss_g = self.lambda_adv * loss_adv_g + self.lambda_cls * loss_cls_fake + self.lambda_rec * loss_rec
                self.optimizer_G.zero_grad()
                loss_g.backward()
                self.optimizer_D.step()
                self.optimizer_G.step()
                if report_loss and ((step + 1) % report_interval == 0 or step == 0):
                    print(f"Step: {step}  loss_G: {loss_g.item():.4f}, loss_D: {loss_d.item():.4f}")
                torch.cuda.empty_cache()
            for step in range(self.training_steps[1]):
                cls_for_slices = [random.choice(random.sample([i for i in range(self.n_cls) if i != j], self.n_cls - 1))
                                  for j in range(self.n_cls)]
                Y_target = np.zeros((self.n_spots,))
                for i in range(self.n_cls):
                    Y_target[self.indices[i]:self.indices[i + 1]] = cls_for_slices[i]
                Y_target = torch.tensor(Y_target, dtype=torch.int64).to(self.device)
                Z_X = self._enc()
                Noise_X = self.noise_generator(Z_X, self.X_target_matrix)
                Z_X_hat = Z_X + Noise_X
                X_rec = self.decoder(Z_X_hat)
                with torch.no_grad():
                    knn_indices_list, random_indices_list = self.find_knn(self.indices, cls_for_slices, Z_X, self.use_cos, self.k)
                Noise_Y = self.simulate_noise(self.indices, cls_for_slices, Noise_X, knn_indices_list)
                Z_Y_hat = Z_X + Noise_Y
                Y = self.decoder(Z_Y_hat)
                D_concat = self.calculate_distance(self.indices, cls_for_slices, Z_X, knn_indices_list, random_indices_list)
                X_src, X_cls = self.discriminator(self.X)
                Y_src, Y_cls = self.discriminator(Y)
                loss_adv_g = adversarial_loss_g(Y_src)
                loss_adv_d = adversarial_loss_d(X_src, Y_src)
                loss_la = latent_loss(D_concat, self.alpha, margin=self.margin)
                loss_rec = mse_loss(self.X, X_rec)
                loss_cls_real = classification_loss(X_cls, self.X_target)
                loss_cls_fake = classification_loss(Y_cls, Y_target)
                loss_d = self.lambda_adv * loss_adv_d + self.lambda_cls * loss_cls_real
                self.optimizer_D.zero_grad()
                loss_d.backward(retain_graph=True)
                loss_g = self.lambda_adv * loss_adv_g + self.lambda_cls * loss_cls_fake + self.lambda_la * loss_la + self.lambda_rec * loss_rec
                self.optimizer_G.zero_grad()
                loss_g.backward()
                self.optimizer_D.step()
                self.optimizer_G.step()
                if report_loss and ((step + 1) % report_interval == 0 or step == 0):
                    print(f"Step: {step}  loss_G: {loss_g.item():.4f}, loss_D: {loss_d.item():.4f}")
                torch.cuda.empty_cache()
            self.train_seconds = time.time() - t0

        def eval(self, adata_list):
            for m in (self.encoder, self.decoder, self.noise_generator, self.discriminator):
                m.eval()
            Z_X = self._enc()
            Noise_X = self.noise_generator(Z_X, self.X_target_matrix)
            Z_X = Z_X.cpu().detach().numpy()
            Noise_X = Noise_X.cpu().detach().numpy()
            for i in range(len(adata_list)):
                adata_list[i].obsm["INSTINCT_latent"] = Z_X[self.indices[i]:self.indices[i + 1]]
                adata_list[i].obsm["INSTINCT_latent_noise"] = Noise_X[self.indices[i]:self.indices[i + 1]]
            return Z_X, Noise_X

        def find_knn(self, spot_idx_list, target_slice_list, latent_features, use_cos=True, k=50):
            knn_indices_list, random_indices_list = [], []
            for i in range(len(target_slice_list)):
                start_idx, end_idx = spot_idx_list[i], spot_idx_list[i + 1]
                ts, te = spot_idx_list[target_slice_list[i]], spot_idx_list[target_slice_list[i] + 1]
                if use_cos:
                    _, knn_indices = F.cosine_similarity(latent_features[start_idx:end_idx].unsqueeze(1),
                                                         latent_features[ts:te].unsqueeze(0), dim=2).topk(k, dim=1, largest=True)
                else:
                    _, knn_indices = torch.norm(latent_features[start_idx:end_idx].unsqueeze(1) - latent_features[ts:te].unsqueeze(0),
                                                p=2, dim=2).topk(k, dim=1, largest=False)
                knn_indices_list.append(knn_indices)
                random_indices = torch.tensor(np.array([random.choices(range(0, end_idx - start_idx)) for _ in range(k)]).T[0]).long().to(self.device)
                random_indices_list.append(random_indices)
            return knn_indices_list, random_indices_list

        def simulate_noise(self, spot_idx_list, target_slice_list, ori_noise, knn_indices_list):
            out = []
            for i in range(len(target_slice_list)):
                ts, te = spot_idx_list[target_slice_list[i]], spot_idx_list[target_slice_list[i] + 1]
                out.append(ori_noise[ts:te][knn_indices_list[i]].mean(dim=1))
            return torch.concat(out, dim=0)

        def calculate_distance(self, spot_idx_list, target_slice_list, latent_features, knn_indices_list, random_indices_list):
            D_concat = []
            for i in range(len(target_slice_list)):
                start_idx, end_idx = spot_idx_list[i], spot_idx_list[i + 1]
                ts, te = spot_idx_list[target_slice_list[i]], spot_idx_list[target_slice_list[i] + 1]
                D_pos = torch.norm(latent_features[start_idx:end_idx].unsqueeze(1) - latent_features[ts:te][knn_indices_list[i]].unsqueeze(0),
                                   p=2, dim=3).squeeze(0)
                D_neg = torch.norm(latent_features[start_idx:end_idx].unsqueeze(1) - latent_features[start_idx:end_idx][random_indices_list[i]].unsqueeze(0),
                                   p=2, dim=2)
                D_concat.append(D_pos - D_neg)
            return torch.concat(D_concat, dim=0)

    return INSTINCT_Model


def instinct(
    adata_list: Sequence,
    slice_names: Optional[Sequence[str]] = None,
    *,
    align_peaks: Optional[bool] = None,
    use_fragment_count: bool = True,
    min_cells_rate: float = 0.03,
    n_pcs: int = 100,
    pca_random_state: int = 1234,
    rad_coef: float = 1.5,
    coor_key: str = "spatial",
    hidden_dims_G=(50,),
    latent_dim: int = 30,
    hidden_dims_D=(50,),
    lambda_adv: float = 1, lambda_cls: float = 10, lambda_la: float = 20, lambda_rec: float = 10,
    seed: int = 1236,
    learn_rates=(1e-3, 5e-4),
    training_steps=(500, 500),
    use_cos: bool = True, margin: float = 10, alpha: float = 1, k: int = 50,
    use_graph: bool = True,
    n_domains: Optional[int] = None,
    gmm_random_state: int = 1234,
    device: Optional[str] = None,
    key_added: str = "X_instinct",
    domain_key: str = "instinct_domain",
    verbose: bool = False,
):
    """INSTINCT multi-slice embedding (and domains) of spatial ATAC samples.

    Vendored PyTorch port of INSTINCT (Liu et al., *Nat Commun* 2025; MIT,
    Yuyao Liu). Runs the upstream tutorial pipeline: peak-set alignment across
    slices, concatenation, ``preprocess_CAS`` (3% peak filter, read->fragment
    counts, Signac TF-IDF), PCA to ``n_pcs`` on the dense matrix, radius graph
    per slice (``rad_coef`` x nearest-spot distance), then the GAT
    auto-encoder with a stochastic domain-translation GAN (500 + 500 steps).

    Arguments:
        adata_list: one AnnData per slice with raw peak counts (``var_names``
            ``chr:start-end``) and ``obsm[coor_key]``. Not modified except for
            ``obsm[key_added]`` / ``obs[domain_key]`` on spots that pass QC.
        slice_names: names for the slices (default ``slice0``, ...).
        align_peaks: run ``peak_sets_alignment``; default when the slices'
            peak sets differ.
        n_domains: if given, cluster the latent space with a tied-covariance
            Gaussian mixture (``random_state=gmm_random_state``), as the
            upstream evaluation scripts do.
        use_graph: ``False`` gives the upstream ``INSTINCT_MLP_Model``.
        Other keyword arguments are upstream ``INSTINCT_Model`` arguments.

    Returns:
        The concatenated, preprocessed AnnData with ``obsm[key_added]`` (and
        ``obs[domain_key]``); ``obs["slice_name"]`` names the slice.

    Reference:
        Liu Y, Li Z, Chen X, Cui X, Gao Z, Jiang R. Multi-sample integration of
        spatial chromatin accessibility sequencing data via stochastic domain
        translation. Nat Commun 16, 1234 (2025).
    """
    import anndata as ad
    import torch
    from sklearn.decomposition import PCA

    if len(adata_list) < 2:
        raise ValueError("INSTINCT integrates two or more slices; pass a list of AnnData")
    slice_names = list(slice_names) if slice_names is not None else [f"slice{i}" for i in range(len(adata_list))]
    originals = list(adata_list)
    cas_list = [a.copy() for a in adata_list]
    for a in cas_list:
        if "insertion" in a.obsm:
            del a.obsm["insertion"]
    if align_peaks is None:
        align_peaks = any(not cas_list[0].var_names.equals(a.var_names) for a in cas_list[1:])
    if align_peaks:
        cas_list = peak_sets_alignment(cas_list, verbose=verbose)
    for j in range(len(cas_list)):
        cas_list[j].obs_names = [x + "_" + slice_names[j] for x in cas_list[j].obs_names]
    adata_concat = ad.concat(cas_list, label="slice_name", keys=slice_names)
    instinct_preprocess(cas_list, adata_concat, use_fragment_count=use_fragment_count, min_cells_rate=min_cells_rate)
    pca = PCA(n_components=n_pcs, random_state=pca_random_state)
    adata_concat.obsm["X_pca"] = pca.fit_transform(adata_concat.X.toarray())
    _create_neighbor_graph(cas_list, adata_concat, rad_coef=rad_coef, coor_key=coor_key, verbose=verbose)
    dev = torch.device(device) if device is not None else None
    Model = _build()
    model = Model(cas_list, adata_concat, input_mat_key="X_pca", input_dim=n_pcs, hidden_dims_G=hidden_dims_G,
                  latent_dim=latent_dim, hidden_dims_D=hidden_dims_D, lambda_adv=lambda_adv, lambda_cls=lambda_cls,
                  lambda_la=lambda_la, lambda_rec=lambda_rec, seed=seed, learn_rates=learn_rates,
                  training_steps=training_steps, use_cos=use_cos, margin=margin, alpha=alpha, k=k, device=dev,
                  use_graph=use_graph, verbose=verbose)
    model.train(report_loss=verbose)
    Z, noise = model.eval(cas_list)
    adata_concat.obsm[key_added] = Z
    adata_concat.obsm[key_added + "_noise"] = noise
    if n_domains is not None:
        from sklearn.mixture import GaussianMixture

        gm = GaussianMixture(n_components=int(n_domains), covariance_type="tied", random_state=gmm_random_state)
        adata_concat.obs[domain_key] = pd.Categorical(gm.fit_predict(Z))
    adata_concat.uns[key_added] = {"method": "INSTINCT", "seed": seed, "train_seconds": model.train_seconds,
                                   "training_steps": list(training_steps), "aligned_peaks": bool(align_peaks)}
    # write back to the inputs (spots that passed QC)
    for j, a in enumerate(originals):
        sub = adata_concat[adata_concat.obs["slice_name"] == slice_names[j]]
        names = [x[: -(len(slice_names[j]) + 1)] for x in sub.obs_names]
        pos = pd.Index(a.obs_names).get_indexer(pd.Index(names))
        emb = np.full((a.n_obs, Z.shape[1]), np.nan, dtype=np.float32)
        emb[pos] = sub.obsm[key_added]
        a.obsm[key_added] = emb
        if n_domains is not None:
            lab = pd.Series(pd.NA, index=a.obs_names, dtype="object")
            lab.iloc[pos] = sub.obs[domain_key].astype(str).to_numpy()
            a.obs[domain_key] = pd.Categorical(lab)
    return adata_concat
