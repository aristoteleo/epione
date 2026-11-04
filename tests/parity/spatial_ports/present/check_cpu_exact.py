"""PRESENT: CPU exactness probe (graph, batches, short training) official vs port, same seed."""
import os
import sys, time, numpy as np, torch, anndata as ad, scanpy as sc
REF = os.path.join(os.environ["SPATIAL_PARITY_REF"], "present/src")
sys.path.insert(0, REF)
D = os.path.join(os.environ["SPATIAL_PARITY_DATA"], "misar_multisp") + "/"
rna = ad.read_h5ad(D + "E15_5-S1_expr_gt.h5ad"); atac = ad.read_h5ad(D + "E15_5-S1_atac_gt.h5ad")
rna.var_names_make_unique(); atac.var_names_make_unique()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
ep = int(sys.argv[2]) if len(sys.argv) > 2 else 3
keep = np.sort(np.random.default_rng(0).choice(rna.n_obs, n, replace=False))
rna, atac = rna[keep].copy(), atac[keep].copy()
for a in (rna, atac):
    for c in list(a.obs.columns):
        if c not in ("Annotation_for_Combined",):
            del a.obs[c]
# --- graph + batches
from PRESENT.Utils import GeometricData, setup_seed
import torch_geometric
from torch_geometric.loader import ClusterData, ClusterLoader
from sklearn.neighbors import kneighbors_graph
from epione.tl import _present as P
spa = rna.obsm["spatial"]
kg = kneighbors_graph(np.array(spa), n_neighbors=6, include_self=False).T.copy()
g1 = torch_geometric.transforms.ToUndirected()(GeometricData(kg, node_index=torch.arange(n)))
g2 = P._geometric_data(kg, torch.arange(n))
print("edge_index equal:", torch.equal(g1.edge_index, g2.edge_index), "edge_attr equal:", torch.equal(g1.edge_attr, g2.edge_attr))
nparts = int(np.ceil(n / 320)) * 10
cd = ClusterData(g1, num_parts=nparts, recursive=False, log=False)
cb = P._ClusterBatches(g2, nparts, coords=spa)
print("partitioner:", cb.partitioner, "partptr equal:", torch.equal(cd.partition.partptr, cb.partptr))
torch.manual_seed(3); b1 = [(c.node_index.clone(), c.edge_index.clone()) for c in ClusterLoader(cd, batch_size=10, shuffle=True)]
torch.manual_seed(3); b2 = [(c.node_index.clone(), c.edge_index.clone()) for c in cb.loader(10, True)]
print("batches equal:", len(b1) == len(b2) and all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(b1, b2)))
# --- short training, CPU
from PRESENT.Main import PRESENT_function
kw = dict(gene_min_cells=1, num_hvg=3000, peak_min_cells_fraction=0.03, nclusters=12, d_lat=50, k_neighbors=6, epochs=ep, lr=1e-3, batch_size=320)
setup_seed(7); t = time.time(); o = PRESENT_function(spatial_key="spatial", adata_rna=rna.copy(), adata_atac=atac.copy(), device="cpu", **kw); t1 = time.time() - t
t = time.time(); p = P.present(rna.copy(), atac.copy(), random_seed=7, device="cpu", **kw); t2 = time.time() - t
e1, e2 = o.obsm["embeddings"], p.obsm["embeddings"]
from sklearn.metrics import adjusted_rand_score as ARI
print(f"n={n} epochs={ep}: max|d emb|={np.abs(e1 - e2).max():.3e}  same obs order={list(o.obs_names) == list(p.obs_names)}  ARI(Leiden)={ARI(o.obs['LeidenClusters'], p.obs['LeidenClusters']):.4f}  t_off={t1:.1f}s t_port={t2:.1f}s")
