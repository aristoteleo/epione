"""Epione side: epi.tl.scbasset on the shared input. usage: run_epione.py SEED EPOCHS OUT [PATIENCE]"""
import sys, time, json, os
import numpy as np, anndata as ad, torch
sys.path.insert(0, os.environ["EPIONE_SRC"])
import epione as epi
seed, epochs, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
patience = int(sys.argv[4]) if len(sys.argv) > 4 else 50
os.makedirs(out, exist_ok=True)
a = ad.read_h5ad(os.path.join(os.environ["EPIONE_SCBASSET_PARITY_DIR"], "data", "atac_ad.h5ad"))
t0 = time.time()
m = epi.tl.scbasset(a, os.environ["HG38_FASTA"], max_epochs=epochs, patience=patience, seed=seed, log_every=1)
el = time.time() - t0
p = m.predict(idx=m.test_idx)
np.savez_compressed(f"{out}/result.npz", embedding=m.cell_embedding(), intercept=m.intercept(),
                    test_ids=m.test_idx, test_prob=p.astype(np.float16))
m.save(f"{out}/model.pt")
json.dump({k: v for k, v in m.history.items()} | {"seconds": el,
          "max_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0},
          open(f"{out}/history.json", "w"))
print("done", el, a.uns["scbasset"]["test_auroc"])
