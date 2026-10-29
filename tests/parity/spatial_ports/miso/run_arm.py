"""MISO parity: one arm, one seed (GPU), MISAR E15.5 RNA+ATAC."""
import os
import argparse, sys, time, random
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
import scanpy as sc, torch
D = os.path.join(os.environ["SPATIAL_PARITY_DATA"], "misar_multisp/")
rna = sc.read_h5ad(D + "E13_5-S1_expr_gt.h5ad"); atac = sc.read_h5ad(D + "E13_5-S1_atac_gt.h5ad")
truth = rna.obs["Annotation_for_Combined"].astype(str).to_numpy(); obs = np.asarray(rna.obs_names)
t0 = time.perf_counter()
if a.arm == "official":
    sys.path.insert(0, os.path.join(os.environ["SPATIAL_PARITY_REF"], "miso/clone_pta"))
    from miso.utils import preprocess
    from miso import Miso
    np.random.seed(a.seed); torch.manual_seed(a.seed); random.seed(a.seed); torch.cuda.manual_seed(a.seed)
    r = preprocess(rna, modality="rna"); c = preprocess(atac, modality="atac")
    m = Miso([r, c], ind_views="all", combs="all", sparse=False, device="cuda")
    m.train(); lab = m.cluster(n_clusters=12); emb = m.emb
else:
    import epione as epi
    r = epi.tl.miso_features(rna, "rna"); c = epi.tl.miso_features(atac, "atac")
    torch.cuda.manual_seed(a.seed)
    emb, lab = epi.tl.miso([r, c], n_clusters=12, seed=a.seed, device="cuda")
el = time.perf_counter() - t0
np.savez(a.out, emb=emb, labels=np.asarray(lab).astype(str), obs_names=obs, truth=truth, seconds=el)
print(f"{a.arm} seed {a.seed}: {el:.1f}s emb {emb.shape}", flush=True)
