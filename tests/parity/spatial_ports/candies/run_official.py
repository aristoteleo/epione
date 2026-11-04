"""CANDIES official arm: upstream tutorial (E18_mouse_brain.ipynb) cells run on MISAR E15.5-S1.

Seeds: the tutorial's fixed seeds shifted by --seed (ZINB 41+s, ATAC AE 2024+s, diffusion
2024+s, integration 2025+s). n_clusters 12. Modality-selection diagnostic cell skipped
(prints only). Outputs the CANDIES joint embedding (clustered later, identically for both arms).
"""
import argparse, sys, time, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
SRC = os.path.join(os.environ["SPATIAL_PARITY_REF"], "candies/src/CANDIES")
sys.path.insert(0, SRC); sys.path.insert(0, SRC + "/codes")
os.chdir(SRC)
import scanpy as sc, torch, pandas as pd
D = os.path.join(os.environ["SPATIAL_PARITY_DATA"], "misar_multisp/")
t0 = time.perf_counter()
adata_omics1 = sc.read_h5ad(D + "E13_5-S1_expr_gt.h5ad"); adata_omics2 = sc.read_h5ad(D + "E13_5-S1_atac_gt.h5ad")
adata_omics1.var_names_make_unique(); adata_omics2.var_names_make_unique()
adata_omics1.obsm["spatial"] = np.array(adata_omics1.obs[["array_col", "array_row"]])
adata_omics2.obsm["spatial"] = np.array(adata_omics2.obs[["array_col", "array_row"]])
from codes.DiTs import *
from codes.sampler import *
from codes.train_diff import *
from codes.ZINB_encoder import *
from codes.preprocess1 import *
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4); sc.pp.log1p(adata_omics1); sc.pp.scale(adata_omics1)
adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
if "X_lsi" not in adata_omics2.obsm.keys():
    sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
    lsi(adata_omics2, use_highly_variable=False, n_components=51)
adata_omics2.obsm["feat"] = adata_omics2.obsm["X_lsi"].copy()
adata_omics1_high = adata_omics1[:, adata_omics1.var["highly_variable"]]
adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=adata_omics2.obsm["X_lsi"].shape[1])
from codes.get_graph import construct_neighbor_graph, adjacent_matrix_preprocessing
device = "cuda:0"
adata_omics1, adata_omics2 = construct_neighbor_graph(adata_omics1, adata_omics2)
adj = adjacent_matrix_preprocessing(adata_omics1, adata_omics2)
adj_spatial_omics1 = adj["adj_spatial_omics1"].to(device); adj_spatial_omics2 = adj["adj_spatial_omics2"].to(device)
ae_model = encoder_ZINB(adata=adata_omics1, device=torch.device("cuda:0"), epochs=300, dim_output=64, random_seed=41 + a.seed)
adata_omics1.obsm["emb_ZINB"], adj_mat = ae_model.train()
seed_everything(2024 + a.seed)
from codes.AutoEncoder import train_atac
train_atac(adata_omics2, adj_spatial_omics2, epochs=600)
emb1 = adata_omics1.obsm["emb_ZINB"]; emb2 = adata_omics2.obsm["emb_latent_omics2"]
df1 = pd.DataFrame(emb1, index=[tuple(c) for c in adata_omics1.obsm["spatial"]])
df2 = pd.DataFrame(emb2, index=[tuple(c) for c in adata_omics2.obsm["spatial"]]).reindex(df1.index)
dataset = ConditionalDiffusionDataset(df1.to_numpy(), df2.to_numpy())
seed_everything(2024 + a.seed)
com_mtx = run_diff(dataset, k=3, batch_size=512, hidden_size=256, learning_rate=1e-3, num_epoch=1000, diffusion_step=800,
                   depth=6, head=16, pca_dim=50, device="cuda:0", classes=6, patience=40, bias=0.5)
adata_omics1.obsm["denoise_emb"] = com_mtx
from codes.integration import *
adata1 = adata_omics1.copy(); adata2 = adata_omics2.copy()
adata1.obsm["feat"] = adata1.obsm["denoise_emb"]; adata2.obsm["feat"] = adata2.obsm["emb_latent_omics2"]
adata1, adata2 = construct_neighbor_graph(adata1, adata2)
adj = adjacent_matrix_preprocessing(adata1, adata2)
f1 = torch.FloatTensor(adata1.obsm["feat"].copy()).to(device); f2 = torch.FloatTensor(adata2.obsm["feat"].copy()).to(device)
seed_everything(2025 + a.seed)
result = train_and_infer(features_omics1=f1, features_omics2=f2, adj_spatial_omics1=adj["adj_spatial_omics1"].to(device),
                         adj_feature_omics1=adj["adj_feature_omics1"].to(device), adj_spatial_omics2=adj["adj_spatial_omics2"].to(device),
                         adj_feature_omics2=adj["adj_feature_omics2"].to(device), device=device, epochs=100)
emb = result["emb_latent_combined"].detach().cpu().numpy().copy()
el = time.perf_counter() - t0
np.savez(a.out, emb=emb, denoise=com_mtx, emb_zinb=emb1, emb_atac=emb2, obs_names=np.asarray(adata1.obs_names),
         truth=adata1.obs["Annotation_for_Combined"].astype(str).to_numpy(), seconds=el)
print(f"official seed {a.seed}: {el:.1f}s emb {emb.shape}", flush=True)
