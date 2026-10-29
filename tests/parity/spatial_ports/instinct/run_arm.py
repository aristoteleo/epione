"""INSTINCT parity: one arm, one seed (GPU). official: cached upstream preprocessing + upstream model."""
import os
import argparse, sys, time, pickle
import numpy as np, anndata as ad, torch
from sklearn.mixture import GaussianMixture
ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
D = os.path.join(os.environ["SPATIAL_PARITY_DATA"], "misar_multisp/")
C = os.path.join(os.environ.get("SPATIAL_PARITY_WORK", "."), "cache") + "/"
names = ["E11_0-S1", "E13_5-S1"]  # MANIFEST A1: DEV sections only
dev = torch.device("cuda")
t0 = time.perf_counter()
if a.arm == "official":
    sys.path.insert(0, os.path.join(os.environ["SPATIAL_PARITY_REF"], "instinct/src"))
    import INSTINCT
    concat = ad.read_h5ad(C + "official_concat.h5ad"); g = pickle.load(open(C + "official_graph.pkl", "rb"))
    concat.uns["graph_list"] = g["graph_list"]
    cas = [concat[concat.obs["slice_name"] == s].copy() for s in names]
    t1 = time.perf_counter()
    m = INSTINCT.INSTINCT_Model(cas, concat, seed=a.seed, device=dev)
    m.train(report_loss=False); m.eval(cas)
    Z = ad.concat(cas, label="slice_name", keys=names).obsm["INSTINCT_latent"]
    train_s = time.perf_counter() - t1
    obs = np.asarray(concat.obs_names); truth = concat.obs["Annotation_for_Combined"].astype(str).to_numpy()
else:
    import epione as epi
    cas = [ad.read_h5ad(D + s + "_atac_gt.h5ad") for s in names]
    concat = epi.tl.instinct(cas, names, seed=a.seed, device="cuda")
    Z = concat.obsm["X_instinct"]; train_s = concat.uns["X_instinct"]["train_seconds"]
    obs = np.asarray(concat.obs_names); truth = concat.obs["Annotation_for_Combined"].astype(str).to_numpy()
el = time.perf_counter() - t0
lab = GaussianMixture(n_components=13, covariance_type="tied", random_state=1234).fit_predict(Z)
np.savez(a.out, emb=Z, labels=lab.astype(str), obs_names=obs, truth=truth, seconds=el, train_seconds=train_s)
print(f"{a.arm} seed {a.seed}: total {el:.1f}s train {train_s:.1f}s", flush=True)
