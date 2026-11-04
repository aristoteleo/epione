"""SpaMosaic parity: one arm, one seed (GPU); MISAR E11.0 + E13.5, RNA + ATAC (two bridge batches)."""
import argparse, os, sys, time
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
import scanpy as sc
D = os.path.join(os.environ["SPATIAL_PARITY_DATA"], "misar_multisp") + "/"
names = ["E11_0-S1", "E13_5-S1"]
def load(kind):
    out = []
    for n in names:
        x = sc.read_h5ad(D + f"{n}_{kind}_gt.h5ad"); x.var_names_make_unique()
        x.obs_names = [o + "_" + n for o in x.obs_names]; x.obs["src"] = n
        if "insertion" in x.obsm: del x.obsm["insertion"]
        out.append(x)
    return out
rna, atac = load("expr"), load("atac")
truth = np.concatenate([x.obs["Annotation_for_Combined"].astype(str).to_numpy() for x in rna])
obs = np.concatenate([np.asarray(x.obs_names) for x in rna])
t0 = time.perf_counter()
if a.arm == "official":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, os.path.join(os.environ["SPATIAL_PARITY_REF"], "spamosaic/src"))
    from spamosaic.framework import SpaMosaic
    from spamosaic.preprocessing import RNA_preprocess, Epigenome_preprocess
    d = {"rna": rna, "atac": atac}
    RNA_preprocess(d["rna"], batch_corr=False, n_hvg=5000, batch_key="src", key="dimred_bc")
    Epigenome_preprocess(d["atac"], batch_corr=False, n_peak=50000, batch_key="src", key="dimred_bc")
    m = SpaMosaic(modBatch_dict=d, input_key="dimred_bc", batch_key="src", intra_knns=10, seed=a.seed, device="cuda:0")
    m.train(net="wlgcn", lr=0.01, T=0.01, n_epochs=100)
    finals = m.infer_emb(d, emb_key="emb", final_latent_key="merged_emb")
    emb = np.vstack([f.obsm["merged_emb"] for f in finals])
else:
    import epione as epi
    d = {"rna": rna, "atac": atac}
    epi.tl.spamosaic_preprocess(d["rna"], "rna", n_features=5000, batch_key="src")
    epi.tl.spamosaic_preprocess(d["atac"], "atac", n_features=50000, batch_key="src")
    finals = epi.tl.spamosaic(d, batch_key="src", intra_knns=10, seed=a.seed, lr=0.01, T=0.01, n_epochs=100, device="cuda:0")
    emb = np.vstack([f.obsm["X_spamosaic"] for f in finals])
el = time.perf_counter() - t0
inp = {f"in_{k}_{i}": x.obsm["dimred_bc"] for k in ("rna", "atac") for i, x in enumerate(d[k])}
np.savez(a.out, emb=emb, obs_names=obs, truth=truth, seconds=el, **inp)
print(f"{a.arm} seed {a.seed}: {el:.1f}s emb {emb.shape}", flush=True)
