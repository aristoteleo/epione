# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Port of PRAGA (Huang et al., "PRAGA: Prototype-aware graph adaptive
# aggregation for spatial multi-modal omics analysis", AAAI 2025,
# doi:10.1609/aaai.v39i1.32010).
# Upstream: https://github.com/Xubin-s-Lab/PRAGA
#           commit 4adb11c96fc7ddad800fa1787eadcc8b91b42784
# Upstream LICENSE.md is the GNU Affero General Public License v3.0, so this
# file stays under AGPL-3.0-or-later (Epione is GPL-3.0; GPLv3 s13 / AGPLv3 s13
# allow combining them, each part keeping its licence - as for
# epione/tl/_spatialglue.py). Parts of PRAGA come from SpatialGlue (AGPL-3.0,
# Yahui Long) and from DeepDPM (MIT, Meitar Ronen, BGU-CS-VIL): the
# split/merge prototype update and the NIW prior below.
#
# Changes from upstream (see tests/parity/spatial_ports/praga/MANIFEST.md):
#   * the hard-coded device ``cuda:1`` (main.py, R5) and ``.cuda()`` are a
#     ``device`` argument;
#   * the command-line ``Arg`` namespace is replaced by keyword arguments with
#     the upstream defaults / run.sh values;
#   * mclust (rpy2 + R) -> pymclustR; print statements removed;
#   * the 'Spatial-epigenome-transcriptome' preprocessing branch of main.py is
#     praga_preprocess(); training (Train.train) is verbatim, including the
#     CUDA autocast region (a no-op on CPU).
"""PRAGA: learnable feature graphs + prototype contrast for spatial multi-omics."""
from __future__ import annotations

import copy
import os
import random
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["praga", "praga_preprocess"]

_SCHEDULES = {
    # datatype: (epochs, learning_rate, weight_decay, cl_weight, init_K)
    "SPOTS": (200, 0.01, 5e-3, 1, 10),
    "Stereo-CITE-seq": (300, 0.01, 5e-2, 1, 10),
    "10x": (30, 0.01, 5e-3, None, None),
    "Spatial-epigenome-transcriptome": (300, 0.01, 5e-2, 2, 14),
}


def _fix_seed(seed):
    import torch
    from torch.backends import cudnn

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# --------------------------------------------------------------------------
# preprocess.py (SpatialGlue-derived)
# --------------------------------------------------------------------------
def _pca(adata, use_reps=None, n_comps=10):
    from sklearn.decomposition import PCA

    p = PCA(n_components=n_comps)
    if use_reps is not None:
        return p.fit_transform(adata.obsm[use_reps])
    X = adata.X
    return p.fit_transform(X.toarray() if sp.issparse(X) else X)


def _lsi(adata, n_components=20, use_highly_variable=None, **kwargs):
    from ._candies import candies_lsi

    candies_lsi(adata, n_components=n_components, use_highly_variable=use_highly_variable, **kwargs)


def _graph_by_coordinate(cell_position, n_neighbors=3):
    from sklearn.neighbors import NearestNeighbors

    _, indices = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position).kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    return pd.DataFrame({"x": x, "y": y, "value": np.ones(x.size)})


def _construct_neighbor_graph(a1, a2, datatype="SPOTS", n_neighbors=3, knn_k=20):
    from sklearn.neighbors import kneighbors_graph

    if datatype in ["Stereo-CITE-seq", "Spatial-epigenome-transcriptome"]:
        n_neighbors = 6
    a1.uns["adj_spatial"] = _graph_by_coordinate(a1.obsm["spatial"], n_neighbors=n_neighbors)
    a2.uns["adj_spatial"] = _graph_by_coordinate(a2.obsm["spatial"], n_neighbors=n_neighbors)
    a1.obsm["adj_feature"] = kneighbors_graph(a1.obsm["feat"], knn_k, mode="distance", metric="correlation", include_self=False)
    a2.obsm["adj_feature"] = kneighbors_graph(a2.obsm["feat"], knn_k, mode="distance", metric="correlation", include_self=False)
    return {"adata_omics1": a1, "adata_omics2": a2}


def _adjacent_matrix_preprocessing(a1, a2):
    from ._candies import _adjacent_matrix_preprocessing as amp

    return amp(a1, a2)


def praga_preprocess(adata_rna, adata_atac, *, knn_k: int = 20):
    """``main.py`` 'Spatial-epigenome-transcriptome' branch: RNA PCA(50) + ATAC LSI(50) + graphs (copies)."""
    import scanpy as sc

    a1, a2 = adata_rna.copy(), adata_atac.copy()
    a1.var_names_make_unique()
    a2.var_names_make_unique()
    sc.pp.filter_genes(a1, min_cells=10)
    sc.pp.filter_cells(a1, min_genes=200)
    sc.pp.highly_variable_genes(a1, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(a1, target_sum=1e4)
    sc.pp.log1p(a1)
    sc.pp.scale(a1)
    a1.obsm["feat"] = _pca(a1[:, a1.var["highly_variable"]], n_comps=50)
    a2 = a2[a1.obs_names].copy()
    if "X_lsi" not in a2.obsm.keys():
        sc.pp.highly_variable_genes(a2, flavor="seurat_v3", n_top_genes=3000)
        _lsi(a2, use_highly_variable=False, n_components=51)
    a2.obsm["feat"] = a2.obsm["X_lsi"].copy()
    return _construct_neighbor_graph(a1, a2, datatype="Spatial-epigenome-transcriptome", knn_k=knn_k)


# --------------------------------------------------------------------------
# model.py, Train_model.py, optimal_clustering_HLN.py, clustering_utils.py
# --------------------------------------------------------------------------
def _build():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import lgamma, mvlgamma
    from torch.nn.parameter import Parameter

    class Encoder(nn.Module):
        def __init__(self, in_feat, out_feat):
            super().__init__()
            self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
            torch.nn.init.xavier_uniform_(self.weight)

        def forward(self, feat, adj):
            e = torch.mm(feat, self.weight)
            return e, torch.spmm(adj, e)

    class Decoder(nn.Module):
        def __init__(self, in_feat, out_feat):
            super().__init__()
            self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
            torch.nn.init.xavier_uniform_(self.weight)

        def forward(self, feat, adj):
            return torch.spmm(adj, torch.mm(feat, self.weight))

    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    class Encoder_overall(nn.Module):
        def __init__(self, d_in1, d_out1, d_in2, d_out2):
            super().__init__()
            self.conv1X1_omics1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.conv1X1_omics2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.MLP = MLP(d_out1 * 2, d_out1, d_out1)
            self.encoder_omics1 = Encoder(d_in1, d_out1)
            self.decoder_omics1 = Decoder(d_out1, d_in1)
            self.encoder_omics2 = Encoder(d_in2, d_out2)
            self.decoder_omics2 = Decoder(d_out2, d_in2)

        def forward(self, f1, f2, as1, af1, as2, af2):
            c1 = torch.cat((as1.unsqueeze(0), af1.unsqueeze(0)), dim=0)
            c2 = torch.cat((as2.unsqueeze(0), af2.unsqueeze(0)), dim=0)
            af1 = self.conv1X1_omics1(c1).squeeze(0)
            af2 = self.conv1X1_omics2(c2).squeeze(0)
            _, e1 = self.encoder_omics1(f1, af1)
            _, e2 = self.encoder_omics2(f2, af2)
            comb = self.MLP(torch.cat((e1, e2), dim=1))
            return {"emb_latent_omics1": e1, "emb_latent_omics2": e2, "emb_latent_combined": comb,
                    "emb_recon_omics1": self.decoder_omics1(comb, as1), "emb_recon_omics2": self.decoder_omics2(comb, as2)}

    class Parametered_Graph(nn.Module):
        def __init__(self, adj, device):
            super().__init__()
            self.adj, self.device = adj, device
            n = adj.shape[0]
            self.paramed_adj_omics = nn.Parameter(torch.FloatTensor(n, n))
            self.paramed_adj_omics.data.copy_(adj)

        def forward(self, A=None):
            adj = (self.paramed_adj_omics + self.paramed_adj_omics.t()) / 2 if A is None else (A + A.t()) / 2
            adj = nn.ReLU(inplace=True)(adj)
            return self._normalize(adj.to(self.device) + torch.eye(adj.shape[0]).to(self.device)).to(self.device)

        def _normalize(self, mx):
            r_inv = mx.sum(1).pow(-1 / 2).flatten()
            r_inv[torch.isinf(r_inv)] = 0.0
            r_mat_inv = torch.diag(r_inv)
            return (r_mat_inv @ mx) @ r_mat_inv

    # ---------------- DeepDPM-derived prototype split / merge (clustering_utils.py) ----------------
    def pairwise_distance(data1, data2):
        return ((data1.unsqueeze(dim=1) - data2.unsqueeze(dim=0)) ** 2).sum(dim=-1)

    class NIW_prior:
        def __init__(self, a):
            self.prior_mu_0_choice, self.prior_sigma_choice = a.prior_mu_0, a.prior_sigma_choice
            self.prior_sigma_scale, self.niw_kappa, self.niw_nu = a.prior_sigma_scale, a.prior_kappa, a.prior_nu

        def init_priors(self, codes):
            if self.prior_mu_0_choice == "data_mean":
                self.niw_m = codes.mean(axis=0)
            if self.prior_sigma_choice == "isotropic":
                self.niw_psi = (torch.eye(codes.shape[1]) * self.prior_sigma_scale).double()
            elif self.prior_sigma_choice == "data_std":
                self.niw_psi = (torch.diag(codes.std(axis=0)) * self.prior_sigma_scale).double()
            else:
                raise NotImplementedError()
            return self.niw_m, self.niw_psi

        def compute_params_post(self, codes_k, mu_k):
            N_k = len(codes_k)
            sum_k = codes_k.sum(axis=0)
            kappa_star = self.niw_kappa + N_k
            nu_star = self.niw_nu + N_k
            mu_0_star = (self.niw_m * self.niw_kappa + sum_k) / kappa_star
            cm = codes_k - mu_k
            psi_star = (self.niw_psi + cm.T @ cm + (self.niw_kappa * N_k / kappa_star)
                        * (mu_k - self.niw_m).unsqueeze(1) @ (mu_k - self.niw_m).unsqueeze(0))
            return kappa_star, nu_star, mu_0_star, psi_star

        def compute_post_mus(self, N_ks, data_mus):
            return ((N_ks.reshape(-1, 1) * data_mus) + (self.niw_kappa * self.niw_m)) / (N_ks.reshape(-1, 1) + self.niw_kappa)

        def compute_post_cov(self, N_k, mu_k, data_cov_k):
            D = len(mu_k)
            if N_k > 0:
                return (self.niw_psi + data_cov_k * N_k + (((self.niw_kappa * N_k) / (self.niw_kappa + N_k))
                        * ((mu_k - self.niw_m).unsqueeze(1) * (mu_k - self.niw_m).unsqueeze(0)))) / (self.niw_nu + N_k + D + 2)
            return self.niw_psi

        def log_marginal_likelihood(self, codes_k, mu_k):
            kappa_star, nu_star, mu_0_star, psi_star = self.compute_params_post(codes_k, mu_k)
            (N_k, D) = codes_k.shape
            return (-(N_k * D / 2.0) * np.log(np.pi) + mvlgamma(torch.tensor(nu_star / 2.0), D)
                    - mvlgamma(torch.tensor(self.niw_nu) / 2.0, D) + (self.niw_nu / 2.0) * torch.logdet(self.niw_psi)
                    - (nu_star / 2.0) * torch.logdet(psi_star) + (D / 2.0) * (np.log(self.niw_kappa) - np.log(kappa_star)))

    def compute_data_covs_hard_assignment(labels, codes, K, mus):
        covs = []
        for k in range(K):
            codes_k = codes[labels == k]
            N_k = float(len(codes_k))
            if N_k > 0:
                d = codes_k - mus[k].cpu().repeat(len(codes_k), 1)
                covs.append(torch.matmul(d.T, d) / N_k)
            else:
                covs.append(torch.eye(codes.shape[1]) * 0.0005)
        return torch.stack(covs)

    def get_sub_assign_with_one_cluster(feat, labels, k, prior):
        from sklearn.mixture import GaussianMixture

        counts = []
        class_sub_feat = feat[labels == k]
        if len(class_sub_feat) <= 2:
            c = torch.tensor([0, len(class_sub_feat)])
            class_sub_assign = torch.ones(len(class_sub_feat), dtype=torch.long)
            mu_subs = torch.mean(class_sub_feat, dim=0, keepdim=True)
            mu_subs = torch.cat([torch.zeros_like(mu_subs), mu_subs], dim=0)
        else:
            gmm = GaussianMixture(n_components=2, random_state=0)
            gmm.fit(class_sub_feat)
            mu_subs = torch.tensor(gmm.means_)
            class_sub_assign = torch.tensor(gmm.predict(class_sub_feat))
            _, c = torch.unique(class_sub_assign, return_counts=True)
        counts.extend(c.cpu().numpy().tolist())
        data_covs_sub = compute_data_covs_hard_assignment(class_sub_assign, class_sub_feat, 2, mu_subs.cpu())
        mu_subs = prior.compute_post_mus(torch.tensor(counts), mu_subs.cpu())
        covs_sub = torch.stack([prior.compute_post_cov(counts[j], class_sub_feat[class_sub_assign == j].mean(axis=0), data_covs_sub[j])
                                for j in range(2)])
        pi_sub = torch.tensor(counts) / float(len(class_sub_feat))
        return mu_subs, covs_sub, pi_sub, class_sub_assign

    def log_hastings_split(alpha, N1, N2, ll1, ll2, ll):
        N = N1 + N2
        if N2 > 0 and N1 > 0:
            H = (np.log(alpha) + lgamma(N1) + ll1 + lgamma(N2) + ll2) - (lgamma(N) + ll)
        else:
            H = torch.zeros(1)
        return bool(H > 0)

    def log_hastings_merge(alpha, N1, N2, ll1, ll2, ll):
        lg1 = 0 if N1 == 0 else lgamma(torch.tensor(N1))
        lg2 = 0 if N2 == 0 else lgamma(torch.tensor(N2))
        N = N1 + N2
        H = (lgamma(torch.tensor(N)) - (np.log(alpha) + lg1 + lg2)) + (ll - (ll1 + ll2)) if N > 0 else torch.ones(1)
        return bool(H > 0)

    def split_and_merge_op(u_feat, a, device):
        from sklearn.mixture import GaussianMixture
        from sklearn.neighbors import NearestNeighbors

        class_num = a.class_num
        u_feat = u_feat.cpu().detach().numpy()
        gmm = GaussianMixture(n_components=class_num, random_state=0)
        gmm.fit(u_feat)
        centroids = torch.tensor(gmm.means_)
        labels = torch.tensor(gmm.predict(u_feat))
        u_feat = torch.tensor(u_feat)
        prior = NIW_prior(a)
        prior.init_priors(u_feat)
        uniq, counts = torch.unique(labels, return_counts=True)
        counts = counts.cpu()
        all_classes = torch.arange(class_num)
        mask = torch.ones(all_classes.size(0), dtype=torch.bool)
        mask[uniq] = False
        missing = all_classes[mask]
        new_index = torch.cat((uniq, missing))
        new_value = torch.cat((counts, torch.zeros(missing.size(0), dtype=counts.dtype)))
        order = torch.argsort(new_index)
        counts = new_value[order]
        pi = counts / float(len(u_feat))
        data_covs = compute_data_covs_hard_assignment(labels, u_feat, class_num, centroids.cpu())
        mus = prior.compute_post_mus(counts, centroids.cpu())
        covs = torch.stack([prior.compute_post_cov(counts[k], u_feat[labels == k].mean(axis=0), data_covs[k]) for k in range(len(centroids))])
        # Upstream fits the 2-component sub-cluster GMM once per *spot* (for that spot's
        # label). The fit depends only on the label (GaussianMixture(random_state=0) on the
        # same rows), so it is computed once per label and reused: exact memoisation, same
        # per-spot lists, ~n_spots/K times fewer GMM fits.
        cache = {}
        mu_sub_list, class_sub_assign_list = [], []
        for class_label in labels.cpu().numpy().tolist():
            if class_label not in cache:
                cache[class_label] = get_sub_assign_with_one_cluster(u_feat, labels, class_label, prior)
            mu_sub, _, _, csa = cache[class_label]
            mu_sub_list.append(mu_sub)
            class_sub_assign_list.append(csa)
        labelled = labels.unique()
        split_decisions = []
        sub_center = [[] for _ in range(len(labelled))]
        for _label in labelled:
            class_indices = labels == _label
            if len(u_feat[class_indices]) <= 5:
                split_decisions.append([_label.item(), False])
                continue
            count = 0
            mu_sub_0 = torch.zeros_like(mu_sub_list[0][0])
            mu_sub_1 = torch.zeros_like(mu_sub_list[0][0])
            for i, b in enumerate(class_indices):
                if b:
                    mu_sub_0 += mu_sub_list[i][0]
                    mu_sub_1 += mu_sub_list[i][1]
                    _csa = class_sub_assign_list[i]
                    count += 1
            mu_sub_0 /= count
            mu_sub_1 /= count
            num_sub_1 = sum(_csa)
            num_sub_0 = len(_csa) - num_sub_1
            class_feat = u_feat[class_indices]
            f0, f1 = class_feat[_csa == 0], class_feat[_csa == 1]
            sub_center[_label].append(mu_sub_0)
            sub_center[_label].append(mu_sub_1)
            d = log_hastings_split(1.0, torch.tensor(num_sub_0), torch.tensor(num_sub_1),
                                   prior.log_marginal_likelihood(f0, mu_sub_0), prior.log_marginal_likelihood(f1, mu_sub_1),
                                   prior.log_marginal_likelihood(class_feat, mus[_label]))
            split_decisions.append([_label.item(), d])
        remain = np.array([c for c, s in split_decisions if not s])
        remain_mus = centroids[remain].cpu()
        merge_decisions = []
        mu_nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(remain_mus.cpu().detach().numpy())
        for ri, cl in enumerate(remain):
            nn_ = mu_nn.kneighbors(centroids[cl].reshape(1, -1).cpu().detach().numpy(), return_distance=False)[0][1:].item()
            feat1, feat2 = u_feat[labels == cl], u_feat[labels == remain[nn_]]
            N1, N2 = feat1.shape[0], feat2.shape[0]
            N = N1 + N2
            mu1, mu2 = remain_mus[ri], remain_mus[nn_]
            mmean = (N1 / N) * mu1 + (N2 / N) * mu2 if N > 0 else torch.mean(torch.stack([mu1, mu2]), axis=0)
            md = log_hastings_merge(1.0, N1, N2, prior.log_marginal_likelihood(feat1, mu1),
                                    prior.log_marginal_likelihood(feat2, mu2),
                                    prior.log_marginal_likelihood(torch.cat([feat1, feat2], dim=0), mmean))
            if not any(_c == nn_ and _n == cl for _c, _, _n in merge_decisions):
                merge_decisions.append([cl, md, nn_])
        not_updated = labelled.cpu().numpy().tolist()
        not_updated += [i for i, s in split_decisions if not s]
        not_updated += [i for i, m, _ in merge_decisions if not m]
        not_updated = list(set(not_updated))
        new_centroids = centroids[not_updated].cpu()
        for cl, s in split_decisions:
            if s:
                new_centroids[cl] = sub_center[cl][0]
                new_centroids = torch.cat((new_centroids, sub_center[cl][1].unsqueeze(0)), dim=0)
        to_delete = []
        for cl, m, nn_ in merge_decisions:
            if m:
                mean_mu = (centroids[cl] + centroids[remain[nn_]]) / 2
                new_centroids = torch.cat((new_centroids, mean_mu.reshape(1, -1).cpu()))
                to_delete.append(cl)
                to_delete.append(nn_)
        centroids = new_centroids
        keep = torch.ones(centroids.size(0), dtype=torch.bool)
        keep[list(set(to_delete))] = False
        centroids = centroids[keep]
        return F.normalize(centroids, p=2, dim=1).to(device)

    class _PriorArg:
        def __init__(self, init_K):
            self.pi_prior, self.prior_dir_counts = "uniform", 0.1
            self.prior_mu_0, self.prior_sigma_choice, self.prior_sigma_scale = "data_mean", "isotropic", 0.005
            self.prior_kappa, self.prior_nu, self.class_num, self.temperature = 0.0001, init_K * 3, init_K, 2

    class R5(nn.Module):
        def __init__(self, weight, init_K, tau, device):
            super().__init__()
            self.tau, self.weight, self.begin, self.device = tau, weight, False, device
            self.arg = _PriorArg(init_K)

        def forward(self, feat, epoch):
            if epoch > 99:
                if (not self.begin) or (epoch % 50 == 0):
                    self.begin = True
                    self.centroids = split_and_merge_op(feat, self.arg, self.device).to(self.device)
                    _, pred = torch.min(pairwise_distance(feat, self.centroids), dim=1)
                else:
                    _, pred = torch.min(pairwise_distance(feat, self.centroids), dim=1)
                cl_loss = self.CL(feat, pred, self.centroids)
            else:
                cl_loss = 0
            return cl_loss * self.weight

        def CL(self, feat, pred, centers):
            loss = 0
            for label, center in enumerate(centers):
                center, feat = center.float(), feat.float()
                if feat[pred == label].shape[0] > 0 and feat[pred != label].shape[0] > 0:
                    pos = torch.div(torch.matmul(feat[pred == label], center.unsqueeze(0).T), self.tau)
                    neg = torch.div(torch.matmul(feat, center.unsqueeze(0).T), self.tau)
                    loss = loss - torch.log(torch.mean(torch.exp(pos.squeeze())) / torch.mean(torch.exp(neg.squeeze())))
            return loss / centers.shape[0]

    def train(data, datatype, device, rna_weight, atac_weight, knn_alpha, cl_weight, init_k, tau, dim_output=64):
        import torch.nn.functional as F

        epochs, lr, wd, clw, K = _SCHEDULES[datatype]
        clw = cl_weight if clw is None else clw
        K = init_k if K is None else K
        a1, a2 = data["adata_omics1"], data["adata_omics2"]
        adj = _adjacent_matrix_preprocessing(a1, a2)
        as1 = adj["adj_spatial_omics1"].to_dense().to(device)
        as2 = adj["adj_spatial_omics2"].to_dense().to(device)
        af1 = adj["adj_feature_omics1"].to_dense().to(device)
        af2 = adj["adj_feature_omics2"].to_dense().to(device)
        pg1 = Parametered_Graph(af1, device).to(device)
        pg2 = Parametered_Graph(af2, device).to(device)
        af1_copy, af2_copy = copy.deepcopy(af1), copy.deepcopy(af2)
        ema = knn_alpha if datatype == "10x" else 0.9
        clustering = R5(clw, K, tau, device)
        f1 = torch.FloatTensor(a1.obsm["feat"].copy()).to(device)
        f2 = torch.FloatTensor(a2.obsm["feat"].copy()).to(device)
        model = Encoder_overall(f1.shape[1], dim_output, f2.shape[1], dim_output).to(device)
        optimizer = torch.optim.SGD(list(model.parameters()) + list(pg1.parameters()) + list(pg2.parameters()),
                                    lr=lr, momentum=0.9, weight_decay=wd)
        model.train()
        use_amp = torch.device(device).type == "cuda"
        for epoch in range(epochs):
            with torch.autocast(device_type="cuda", enabled=use_amp):
                model.train()
                r = model(f1, f2, as1, af1, as2, af2)
                loss = rna_weight * F.mse_loss(f1, r["emb_recon_omics1"]) + atac_weight * F.mse_loss(f2, r["emb_recon_omics2"])
                u1, u2 = pg1(), pg2()
                loss_fro = (torch.norm(u1 - af1_copy.detach(), p="fro") + torch.norm(u2 - af2_copy.detach(), p="fro")) / 2
                loss = loss + clustering(r["emb_latent_combined"], epoch) + loss_fro
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            af1, af2 = pg1(), pg2()
            af1_copy = ema * af1_copy + (1 - ema) * af1.detach().clone()
            af2_copy = ema * af2_copy + (1 - ema) * af2.detach().clone()
        with torch.no_grad():
            model.eval()
            r = model(f1, f2, as1, af1, as2, af2)
        return {"emb_latent_omics1": F.normalize(r["emb_latent_omics1"], p=2, eps=1e-12, dim=1).detach().cpu().numpy(),
                "emb_latent_omics2": F.normalize(r["emb_latent_omics2"], p=2, eps=1e-12, dim=1).detach().cpu().numpy(),
                "PRAGA": F.normalize(r["emb_latent_combined"], p=2, eps=1e-12, dim=1).detach().cpu().numpy()}

    return train


def praga(
    adata_rna,
    adata_atac,
    *,
    datatype: str = "Spatial-epigenome-transcriptome",
    n_clusters: Optional[int] = 14,
    knn_k: int = 20,
    rna_weight: float = 1.0,
    atac_weight: float = 10.0,
    alpha: float = 0.9,
    cl_weight: float = 1.0,
    init_k: int = 14,
    tau: float = 2.0,
    random_seed: int = 2024,
    device: Optional[str] = None,
    preprocess: bool = True,
    key_added: str = "X_praga",
    domain_key: str = "praga_domain",
):
    """PRAGA joint embedding (and domains) of spatial RNA + ATAC (or ADT).

    PyTorch port of PRAGA (AAAI 2025; AGPL-3.0, this file keeps that licence):
    SpatialGlue-style GCN encoders whose feature graphs are learnable dense
    adjacencies (EMA-anchored by a Frobenius penalty) fused with the spatial
    graph by a 1x1 convolution, an MLP fusion of the two modality embeddings,
    and from epoch 100 a prototype contrastive loss whose prototypes are
    refreshed every 50 epochs by a DeepDPM-style split/merge of a Gaussian
    mixture. Defaults are the upstream ``run.sh`` mouse-brain ATAC-RNA setting.

    Arguments:
        adata_rna, adata_atac: raw counts with ``obsm["spatial"]`` (or, with
            ``preprocess=False``, AnnData that already hold ``obsm["feat"]``).
        datatype: upstream schedule key (epochs / lr / weight decay).
        n_clusters: mclust ``EEE`` domains on PCA(20) of the embedding
            (upstream ``clustering(method="mclust", use_pca=True)``, via pymclustR);
            ``None`` skips clustering.
        device: default CUDA if available (upstream uses CUDA autocast).

    Returns:
        The L2-normalised joint embedding of the spots kept by the RNA QC.
    """
    import torch

    dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    _fix_seed(random_seed)
    data = praga_preprocess(adata_rna, adata_atac, knn_k=knn_k) if preprocess else _construct_neighbor_graph(
        adata_rna.copy(), adata_atac.copy(), datatype=datatype, knn_k=knn_k)
    out = _build()(data, datatype, dev, rna_weight, atac_weight, alpha, cl_weight, init_k, tau)
    emb = out["PRAGA"]
    kept = data["adata_omics1"].obs_names
    labels = None
    if n_clusters is not None:
        from ._spatial_methods import _mclust

        labels = _mclust(emb, n_clusters)
    for a in (adata_rna, adata_atac):
        pos = pd.Index(a.obs_names).get_indexer(kept)
        e = np.full((a.n_obs, emb.shape[1]), np.nan, dtype=np.float32)
        e[pos] = emb
        a.obsm[key_added] = e
        if labels is not None:
            lab = pd.Series(pd.NA, index=a.obs_names, dtype="object")
            lab.iloc[pos] = labels.astype(str)
            a.obs[domain_key] = pd.Categorical(lab)
        a.uns[key_added] = {"method": "PRAGA", "datatype": datatype, "random_seed": random_seed}
    return emb
