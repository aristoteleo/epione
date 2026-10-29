"""CANDIES port arm (epi.tl.candies) on MISAR E15.5-S1, same settings as run_official.py."""
import os
import argparse, time
import numpy as np, scanpy as sc
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
import epione as epi
D = os.path.join(os.environ["SPATIAL_PARITY_DATA"], "misar_multisp/")
t0 = time.perf_counter()
rna = sc.read_h5ad(D + "E13_5-S1_expr_gt.h5ad"); atac = sc.read_h5ad(D + "E13_5-S1_atac_gt.h5ad")
rna.var_names_make_unique(); atac.var_names_make_unique()
rna.obsm["spatial"] = np.array(rna.obs[["array_col", "array_row"]]); atac.obsm["spatial"] = np.array(atac.obs[["array_col", "array_row"]])
emb = epi.tl.candies(rna, atac, seed_offset=a.seed, device="cuda:0")
el = time.perf_counter() - t0
np.savez(a.out, emb=emb, denoise=rna.obsm["X_candies_denoised"], obs_names=np.asarray(rna.obs_names),
         truth=rna.obs["Annotation_for_Combined"].astype(str).to_numpy(), seconds=el)
print(f"port seed {a.seed}: {el:.1f}s emb {emb.shape}", flush=True)
