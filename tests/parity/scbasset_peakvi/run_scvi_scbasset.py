"""Information row: scvi-tools SCBASSET on the same sequences/targets. usage: SEED EPOCHS OUT"""
import sys, os, json, time
import numpy as np, anndata as ad, h5py, scipy.sparse as sp
import scvi, torch
seed, epochs, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.makedirs(out, exist_ok=True)
D = os.path.join(os.environ["EPIONE_SCBASSET_PARITY_DIR"], "data")
a = ad.read_h5ad(f"{D}/atac_ad.h5ad")
with h5py.File(f"{D}/processed/all_seqs.h5", "r") as f:
    codes = f["X"][:].astype(np.int64)
with h5py.File(f"{D}/processed/splits.h5", "r") as f:
    test_ids = f["test_ids"][:]
b = a.transpose()  # regions x cells, as in the scvi-tools tutorial
b.layers["binary"] = (sp.csr_matrix(b.X) > 0).astype(np.float32)
b.obsm["dna_code"] = codes
scvi.settings.seed = seed
scvi.external.SCBASSET.setup_anndata(b, layer="binary", dna_code_key="dna_code")
m = scvi.external.SCBASSET(b)
t0 = time.time()
m.train(max_epochs=epochs)
el = time.time() - t0
mod = m.module.eval()
with torch.no_grad():
    logit = mod._get_accessibility(torch.from_numpy(codes[test_ids]).to(mod.device), batch_size=256)
p = torch.sigmoid(logit).cpu().numpy()
np.savez_compressed(f"{out}/result.npz", embedding=m.get_latent_representation(), intercept=m.get_cell_bias(),
                    test_ids=test_ids, test_prob=p.astype(np.float16))
json.dump({"seconds": el, "epochs_run": int(len(m.history["auroc_train"]))}, open(f"{out}/history.json", "w"))
print("done", el)
