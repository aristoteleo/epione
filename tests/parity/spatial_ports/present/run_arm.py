"""PRESENT parity: one arm, one seed, full MISAR E15.5-S1 RNA+ATAC, GPU (MANIFEST settings)."""
import os
import argparse, sys, time
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["official", "port"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
import anndata as ad, torch
REF = os.path.join(os.environ["SPATIAL_PARITY_REF"], "present/src")
D = os.path.join(os.environ["SPATIAL_PARITY_DATA"], "misar_multisp/")
rna = ad.read_h5ad(D + "E15_5-S1_expr_gt.h5ad"); atac = ad.read_h5ad(D + "E15_5-S1_atac_gt.h5ad")
rna.var_names_make_unique(); atac.var_names_make_unique()
kw = dict(gene_min_cells=1, num_hvg=3000, peak_min_cells_fraction=0.03, nclusters=12, d_lat=50,
          k_neighbors=6, epochs=100, lr=1e-3, batch_size=320)
t0 = time.perf_counter()
if a.arm == "official":
    sys.path.insert(0, REF)
    from PRESENT.Utils import setup_seed
    from PRESENT.Main import PRESENT_function
    setup_seed(a.seed)
    out = PRESENT_function(spatial_key="spatial", adata_rna=rna, adata_atac=atac, device=a.device, **kw)
else:
    import epione as epi
    out = epi.tl.present(rna, atac, random_seed=a.seed, device=a.device, **kw)
el = time.perf_counter() - t0
np.savez(a.out, emb=out.obsm["embeddings"], labels=out.obs["LeidenClusters"].astype(str).to_numpy(),
         obs_names=np.asarray(out.obs_names), truth=out.obs["Annotation_for_Combined"].astype(str).to_numpy(), seconds=el)
print(f"{a.arm} seed {a.seed}: {el:.1f}s emb {out.obsm['embeddings'].shape}", flush=True)
