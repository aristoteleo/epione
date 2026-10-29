"""COSMOS parity: run one arm (official clone or Epione port) for one seed.

python run_arm.py --arm official|port --seed 20 --h5 <ATAC_RNA_Seq_MouseBrain_RNA_ATAC.h5>
       --ref-src <COSMOS clone> --out <npz>
Tutorial settings (Tutorials/tutorial_ATAC_RNA_Seq_MouseBrain.ipynb), CPU.
"""
import os
import argparse, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["official", "port"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--h5", required=True)
ap.add_argument("--ref-src", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()

import h5py
import scanpy as sc
import torch

torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "4")))  # threads from env
f = h5py.File(a.h5, "r")
X_RNA = np.array(f["X_RNA"]).astype("float64")
X_ATAC = np.array(f["X_ATAC"]).astype("float64")
loc = np.array(f["Pos"]).astype("float64")
adata1 = sc.AnnData(X_RNA); adata1.obsm["spatial"] = loc
adata2 = sc.AnnData(X_ATAC); adata2.obsm["spatial"] = loc

kw = dict(spatial_regularization_strength=0.01, z_dim=50, lr=1e-3, wnn_epoch=500, total_epoch=1000,
          max_patience_bef=10, max_patience_aft=30, min_stop=200, random_seed=a.seed,
          regularization_acceleration=True, edge_subset_sz=1000000)
t0 = time.perf_counter()
if a.arm == "official":
    sys.path.insert(0, a.ref_src)
    from COSMOS import cosmos
    m = cosmos.Cosmos(adata1=adata1, adata2=adata2)
    m.preprocessing_data(n_neighbors=10)
    emb = m.train(gpu=0, **kw)
    w = m.weights
else:
    import epione as epi
    emb = epi.tl.cosmos(adata1, adata2, n_neighbors=10, device="cpu", **kw)
    w = adata1.obsm["X_cosmos_weights"]
el = time.perf_counter() - t0
np.savez(a.out, emb=emb, weights=w, seconds=el)
print(f"{a.arm} seed {a.seed}: {el:.1f}s emb {emb.shape}", flush=True)
