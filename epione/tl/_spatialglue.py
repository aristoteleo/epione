# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Port of SpatialGlue (Long et al., "Deciphering spatial domains from
# spatial multi-omics with SpatialGlue", Nature Methods 2024).
# Upstream: https://github.com/JinmiaoChenLab/SpatialGlue
#           commit 7c976d811d27ace51ce47ae0ad94a068a7d222fa (v1.1.5)
# Upstream LICENSE.md is the GNU Affero General Public License v3.0
# (its setup.py metadata says "MIT"; we follow the LICENSE file, the
# stricter of the two). This file therefore stays under AGPL-3.0-or-later.
# Epione as a whole is GPL-3.0; GPLv3 section 13 and AGPLv3 section 13
# explicitly allow the two to be combined, each part keeping its licence.
#
# Ported faithfully: model (Encoder / Decoder / AttentionLayer /
# Encoder_overall), graph construction, loss, optimiser, epochs and
# weight factors per data type, and the preprocessing helpers of the
# official tutorials. Changes, none of which alter the numbers:
#   * torch.sparse.FloatTensor -> torch.sparse_coo_tensor (deprecated API);
#   * F.softmax gets the explicit dim=1 that the implicit rule picked;
#   * the seed is applied inside spatialglue() (upstream: fix_seed() call
#     in the tutorial, Train_SpatialGlue.random_seed is unused);
#   * mclust runs through pymclustR (pure-Python CRAN mclust port)
#     instead of rpy2 + R.
"""SpatialGlue: dual-attention graph integration of spatial multi-omics."""
from __future__ import annotations

import os
import random
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "spatialglue",
    "spatialglue_preprocess",
    "spatialglue_graphs",
    "lsi_spatialglue",
    "clr_normalize_each_cell",
]

# Per-datatype schedules, verbatim from SpatialGlue_pyG.Train_SpatialGlue.
_SCHEDULES = {
    "SPOTS": (600, [1, 5, 1, 1]),
    "Stereo-CITE-seq": (1500, [1, 10, 1, 10]),
    "10x": (200, [1, 5, 1, 10]),
    "Spatial-epigenome-transcriptome": (1600, [1, 5, 1, 1]),
}


# --------------------------------------------------------------------------
# seed
# --------------------------------------------------------------------------
def _fix_seed(seed: int) -> None:
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
# model (SpatialGlue/model.py)
# --------------------------------------------------------------------------
def _build_model_classes():
    import torch
    import torch.nn.functional as F
    from torch.nn import Module
    from torch.nn.parameter import Parameter

    class Encoder(Module):
        def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
            super().__init__()
            self.in_feat, self.out_feat = in_feat, out_feat
            self.dropout, self.act = dropout, act
            self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
            torch.nn.init.xavier_uniform_(self.weight)

        def forward(self, feat, adj):
            x = torch.mm(feat, self.weight)
            return torch.spmm(adj, x)

    class Decoder(Encoder):
        pass

    class AttentionLayer(Module):
        def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
            super().__init__()
            self.in_feat, self.out_feat = in_feat, out_feat
            self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
            self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
            torch.nn.init.xavier_uniform_(self.w_omega)
            torch.nn.init.xavier_uniform_(self.u_omega)

        def forward(self, emb1, emb2):
            emb = torch.cat(
                [torch.unsqueeze(torch.squeeze(emb1), dim=1), torch.unsqueeze(torch.squeeze(emb2), dim=1)],
                dim=1,
            )
            v = torch.tanh(torch.matmul(emb, self.w_omega))
            vu = torch.matmul(v, self.u_omega)
            alpha = F.softmax(torch.squeeze(vu) + 1e-6, dim=1)
            emb_combined = torch.matmul(torch.transpose(emb, 1, 2), torch.unsqueeze(alpha, -1))
            return torch.squeeze(emb_combined), alpha

    class EncoderOverall(Module):
        def __init__(self, d_in1, d_out1, d_in2, d_out2):
            super().__init__()
            # creation order fixes the RNG stream; keep it as upstream
            self.encoder_omics1 = Encoder(d_in1, d_out1)
            self.decoder_omics1 = Decoder(d_out1, d_in1)
            self.encoder_omics2 = Encoder(d_in2, d_out2)
            self.decoder_omics2 = Decoder(d_out2, d_in2)
            self.atten_omics1 = AttentionLayer(d_out1, d_out1)
            self.atten_omics2 = AttentionLayer(d_out2, d_out2)
            self.atten_cross = AttentionLayer(d_out1, d_out2)

        def forward(self, f1, f2, a_sp1, a_ft1, a_sp2, a_ft2):
            s1 = self.encoder_omics1(f1, a_sp1)
            s2 = self.encoder_omics2(f2, a_sp2)
            t1 = self.encoder_omics1(f1, a_ft1)
            t2 = self.encoder_omics2(f2, a_ft2)
            e1, al1 = self.atten_omics1(s1, t1)
            e2, al2 = self.atten_omics2(s2, t2)
            ec, al = self.atten_cross(e1, e2)
            r1 = self.decoder_omics1(ec, a_sp1)
            r2 = self.decoder_omics2(ec, a_sp2)
            x1 = self.encoder_omics2(self.decoder_omics2(e1, a_sp2), a_sp2)
            x2 = self.encoder_omics1(self.decoder_omics1(e2, a_sp1), a_sp1)
            return {
                "emb_latent_omics1": e1, "emb_latent_omics2": e2, "emb_latent_combined": ec,
                "emb_recon_omics1": r1, "emb_recon_omics2": r2,
                "emb_latent_omics1_across_recon": x1, "emb_latent_omics2_across_recon": x2,
                "alpha_omics1": al1, "alpha_omics2": al2, "alpha": al,
            }

    return EncoderOverall


# --------------------------------------------------------------------------
# graphs (SpatialGlue/preprocess.py)
# --------------------------------------------------------------------------
def _graph_by_coordinate(pos: np.ndarray, n_neighbors: int) -> sp.coo_matrix:
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(pos)
    _, idx = nbrs.kneighbors(pos)
    x = idx[:, 0].repeat(n_neighbors)
    y = idx[:, 1:].flatten()
    n = x.max() + 1  # upstream transform_adjacent_matrix
    return sp.coo_matrix((np.ones(x.size), (x, y)), shape=(n, n))


def _graph_by_feature(feat: np.ndarray, k: int = 20):
    from sklearn.neighbors import kneighbors_graph

    return kneighbors_graph(feat, k, mode="connectivity", metric="correlation", include_self=False)


def _normalize_graph(adj: np.ndarray):
    import torch

    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    d = sp.diags(np.power(rowsum, -0.5).flatten())
    a = adj_.dot(d).transpose().dot(d).tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((a.row, a.col)).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(a.data), torch.Size(a.shape))


def _symmetrize(a: np.ndarray) -> np.ndarray:
    a = a + a.T
    return np.where(a > 1, 1, a)


def spatialglue_graphs(
    feat1: np.ndarray, feat2: np.ndarray, pos1: np.ndarray, pos2: np.ndarray,
    *, datatype: str = "SPOTS", n_neighbors: int = 3,
) -> dict:
    """Spatial and feature graphs of SpatialGlue, normalised, as torch sparse tensors.

    Spatial graph: kNN on coordinates (k = 6 for Stereo-CITE-seq and
    Spatial-epigenome-transcriptome, else ``n_neighbors``), symmetrised.
    Feature graph: kNN (k = 20, correlation) on the input features,
    symmetrised. Both get self loops and symmetric degree normalisation.
    """
    import torch

    if datatype in ("Stereo-CITE-seq", "Spatial-epigenome-transcriptome"):
        n_neighbors = 6
    out = {}
    for i, (pos, feat) in enumerate(((pos1, feat1), (pos2, feat2)), start=1):
        a_sp = _symmetrize(_graph_by_coordinate(np.asarray(pos), n_neighbors).toarray())
        # upstream builds the feature graph as a float32 torch tensor
        a_ft = _symmetrize(torch.FloatTensor(_graph_by_feature(np.asarray(feat)).toarray()).numpy())
        out[f"adj_spatial_omics{i}"] = _normalize_graph(a_sp)
        out[f"adj_feature_omics{i}"] = _normalize_graph(a_ft)
    return out


# --------------------------------------------------------------------------
# preprocessing helpers (SpatialGlue/preprocess.py + Tutorial 4)
# --------------------------------------------------------------------------
def _pca(X, n_comps):
    from sklearn.decomposition import PCA

    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return PCA(n_components=n_comps).fit_transform(X)


def _tfidf(X):
    idf = X.shape[0] / X.sum(axis=0)
    if sp.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    tf = X / X.sum(axis=1, keepdims=True)
    return tf * idf


def lsi_spatialglue(adata, n_components: int = 20, use_highly_variable: Optional[bool] = None, **kwargs):
    """Seurat-v3 style LSI used by SpatialGlue (first component dropped).

    Writes ``adata.obsm["X_lsi"]`` with ``n_components - 1`` columns.
    """
    import sklearn.preprocessing
    import sklearn.utils.extmath

    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    a = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = _tfidf(a.X)
    X = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X = np.log1p(X * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def clr_normalize_each_cell(adata, inplace: bool = True):
    """Seurat CLR per cell (rows of ``X``), as SpatialGlue uses for protein."""

    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        return np.log1p(x / np.exp(s / len(x)))

    if not inplace:
        adata = adata.copy()
    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    adata.X = np.apply_along_axis(seurat_clr, 1, X)
    return adata


def spatialglue_preprocess(
    rna, atac, *, n_top_genes: int = 3000, n_comps_rna: int = 50, n_comps_atac: int = 51,
    min_cells: int = 10, min_genes: int = 200, random_seed: Optional[int] = 2022,
):
    """Preprocessing of the official Spatial-epigenome-transcriptome tutorial.

    RNA: ``filter_genes(min_cells=10)``, ``filter_cells(min_genes=200)``,
    seurat_v3 HVG (3000), normalise to 1e4, log1p, scale, PCA(50) on the HVGs
    -> ``rna.obsm["feat"]``. ATAC: restricted to the RNA spots, then
    ``X_lsi`` (if not already present: LSI with 51 components on all peaks,
    first dropped) -> ``atac.obsm["feat"]``. Upstream also computes peak HVGs
    before LSI but does not use them (``use_highly_variable=False``); that
    no-op step is skipped. ``rna`` is modified in place (raw counts are kept
    in ``rna.layers["counts"]``). ``random_seed`` seeds numpy / torch /
    ``random`` first, as the tutorial's ``fix_seed`` call does (the PCA is
    randomised); ``None`` leaves the global state alone.

    Returns:
        ``(rna, atac)`` with ``obsm["feat"]`` set, same spots in the same order.
    """
    import scanpy as sc

    if random_seed is not None:
        _fix_seed(random_seed)
    rna.layers["counts"] = rna.X.copy()
    sc.pp.filter_genes(rna, min_cells=min_cells)
    sc.pp.filter_cells(rna, min_genes=min_genes)
    sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    rna.obsm["feat"] = _pca(rna[:, rna.var["highly_variable"]].X, n_comps_rna)
    atac = atac[rna.obs_names].copy()
    if "X_lsi" not in atac.obsm:
        lsi_spatialglue(atac, use_highly_variable=False, n_components=n_comps_atac)
    atac.obsm["feat"] = np.asarray(atac.obsm["X_lsi"]).copy()
    return rna, atac


# --------------------------------------------------------------------------
# training (SpatialGlue/SpatialGlue_pyG.py)
# --------------------------------------------------------------------------
def _get_pair(data, modalities):
    try:
        import mudata as md

        if isinstance(data, md.MuData):
            return data[modalities[0]], data[modalities[1]], data
    except ImportError:  # pragma: no cover
        pass
    if isinstance(data, (tuple, list)) and len(data) == 2:
        return data[0], data[1], None
    if isinstance(data, dict):
        return data["adata_omics1"], data["adata_omics2"], None
    raise TypeError("pass a MuData, an (adata_omics1, adata_omics2) pair or SpatialGlue's data dict")


def spatialglue(
    data,
    *,
    modalities: Sequence[str] = ("rna", "atac"),
    use_rep: str = "feat",
    spatial_key: str = "spatial",
    datatype: str = "Spatial-epigenome-transcriptome",
    random_seed: int = 2022,
    device=None,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    epochs: Optional[int] = None,
    dim_output: int = 64,
    weight_factors: Optional[Sequence[float]] = None,
    n_neighbors: int = 3,
    key_added: str = "SpatialGlue",
    verbose: bool = True,
) -> dict:
    """Integrate two spatial modalities with SpatialGlue (Long et al. 2024).

    Each modality is encoded by a one-layer GCN on a spatial graph and on a
    feature graph; a within-modality attention mixes the two, a
    between-modality attention mixes the modalities, and modality-specific
    decoders reconstruct the inputs (plus a cross-modality consistency
    loss). The joint embedding (L2-normalised) is the integrated spot
    representation; the attention weights say which modality drives each
    spot.

    Arguments:
        data: ``MuData`` holding ``modalities``, an ``(omics1, omics2)``
            AnnData pair with the same spots in the same order, or the dict
            returned by upstream ``construct_neighbor_graph``.
        modalities: MuData modality names (omics1, omics2). For
            spatial ATAC-RNA use ``("rna", "atac")`` as upstream does.
        use_rep: ``obsm`` key of the input features in both modalities
            (see :func:`spatialglue_preprocess`).
        spatial_key: ``obsm`` key of the coordinates.
        datatype: ``"Spatial-epigenome-transcriptome"`` (spatial ATAC-RNA,
            CUT&Tag-RNA), ``"SPOTS"``, ``"Stereo-CITE-seq"`` or ``"10x"``;
            sets epochs, loss weights and the spatial k as upstream.
        random_seed: seed applied before graph construction and training
            (upstream tutorial default 2022).
        device: torch device; default CUDA when available.
        epochs, weight_factors: override the per-datatype schedule.
        learning_rate, weight_decay, dim_output, n_neighbors: as upstream.
        key_added: ``obsm`` key of the joint embedding.

    Returns:
        dict with ``emb_latent_omics1``, ``emb_latent_omics2``,
        ``SpatialGlue`` (joint), ``alpha_omics1``, ``alpha_omics2``,
        ``alpha`` (numpy arrays). When ``data`` is a MuData they are also
        written to ``data.obsm`` (``key_added``, ``"<key_added>_alpha"`` ...).
    """
    import torch
    import torch.nn.functional as F

    a1, a2, mdata = _get_pair(data, modalities)
    if a1.n_obs != a2.n_obs:
        raise ValueError("the two modalities must have the same spots in the same order")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _fix_seed(random_seed)
    ep, wf = _SCHEDULES.get(datatype, (600, [1, 5, 1, 1]))
    epochs = ep if epochs is None else int(epochs)
    wf = list(wf if weight_factors is None else weight_factors)

    f1 = np.asarray(a1.obsm[use_rep])
    f2 = np.asarray(a2.obsm[use_rep])
    adj = spatialglue_graphs(
        f1, f2, np.asarray(a1.obsm[spatial_key]), np.asarray(a2.obsm[spatial_key]),
        datatype=datatype, n_neighbors=n_neighbors,
    )
    adj = {k: v.to(device) for k, v in adj.items()}
    x1 = torch.FloatTensor(f1.copy()).to(device)
    x2 = torch.FloatTensor(f2.copy()).to(device)

    Model = _build_model_classes()
    model = Model(x1.shape[1], dim_output, x2.shape[1], dim_output).to(device)
    opt = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    args = (x1, x2, adj["adj_spatial_omics1"], adj["adj_feature_omics1"],
            adj["adj_spatial_omics2"], adj["adj_feature_omics2"])
    it = range(epochs)
    if verbose:
        from tqdm.auto import tqdm

        it = tqdm(it, desc="SpatialGlue")
    model.train()
    for _ in it:
        model.train()
        r = model(*args)
        loss = (
            wf[0] * F.mse_loss(x1, r["emb_recon_omics1"])
            + wf[1] * F.mse_loss(x2, r["emb_recon_omics2"])
            + wf[2] * F.mse_loss(r["emb_latent_omics1"], r["emb_latent_omics1_across_recon"])
            + wf[3] * F.mse_loss(r["emb_latent_omics2"], r["emb_latent_omics2_across_recon"])
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        model.eval()
        r = model(*args)
    out = {
        "emb_latent_omics1": F.normalize(r["emb_latent_omics1"], p=2, eps=1e-12, dim=1),
        "emb_latent_omics2": F.normalize(r["emb_latent_omics2"], p=2, eps=1e-12, dim=1),
        "SpatialGlue": F.normalize(r["emb_latent_combined"], p=2, eps=1e-12, dim=1),
        "alpha_omics1": r["alpha_omics1"],
        "alpha_omics2": r["alpha_omics2"],
        "alpha": r["alpha"],
    }
    out = {k: v.detach().cpu().numpy() for k, v in out.items()}
    out["loss"] = float(loss.detach().cpu())
    if mdata is not None:
        mdata.obsm[key_added] = out["SpatialGlue"]
        mdata.obsm[f"{key_added}_alpha"] = out["alpha"]
        for i, m in enumerate(modalities, start=1):
            mdata[m].obsm[f"{key_added}_latent"] = out[f"emb_latent_omics{i}"]
            mdata[m].obsm[f"{key_added}_alpha"] = out[f"alpha_omics{i}"]
        mdata.uns[key_added] = {"datatype": datatype, "epochs": epochs, "weight_factors": wf,
                                "random_seed": random_seed, "modalities": list(modalities)}
    return out
