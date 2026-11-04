"""PeakVI P2: MODE in {epione, direct}. usage: MODE SEED OUT"""
import sys, os, json, time
import numpy as np, anndata as ad
mode, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, os.environ["EPIONE_SRC"])
os.makedirs(out, exist_ok=True)
a = ad.read_h5ad(os.path.join(os.environ["EPIONE_SCBASSET_PARITY_DIR"], "data", "atac_ad.h5ad"))
t0 = time.time()
if mode == "epione":
    import epione as epi
    m = epi.tl.peakvi(a, seed=seed)
    z = a.obsm["X_peakvi"]
else:
    import scvi
    scvi.settings.seed = seed
    scvi.model.PEAKVI.setup_anndata(a)
    m = scvi.model.PEAKVI(a)
    m.train()
    z = m.get_latent_representation()
el = time.time() - t0
np.savez_compressed(f"{out}/result.npz", embedding=z)
json.dump({"seconds": el, "epochs_run": int(len(m.history["elbo_train"]))}, open(f"{out}/history.json", "w"))
print("done", el)
