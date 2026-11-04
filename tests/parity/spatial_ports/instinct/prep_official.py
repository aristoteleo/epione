"""INSTINCT official arm, deterministic part (upstream runINSTINCT.py up to the graph), cached."""
import os
import sys, time, pickle
import numpy as np, anndata as ad
sys.path.insert(0, os.path.join(os.environ["SPATIAL_PARITY_REF"], "instinct/src"))
import INSTINCT
from sklearn.decomposition import PCA
D = os.path.join(os.environ["SPATIAL_PARITY_DATA"], "misar_multisp/")
C = os.path.join(os.environ.get("SPATIAL_PARITY_WORK", "."), "cache") + "/"
names = ["E11_0-S1", "E13_5-S1"]  # MANIFEST A1: DEV sections only
cas = []
for s in names:
    a = ad.read_h5ad(D + s + "_atac_gt.h5ad")
    if "insertion" in a.obsm: del a.obsm["insertion"]
    cas.append(a)
t0 = time.time(); cas = INSTINCT.peak_sets_alignment(cas); t_align = time.time() - t0
for j, a in enumerate(cas): a.write_h5ad(C + f"official_merged_{names[j]}.h5ad")
for j in range(len(cas)): cas[j].obs_names = [x + "_" + names[j] for x in cas[j].obs_names]
concat = ad.concat(cas, label="slice_name", keys=names)
t0 = time.time(); INSTINCT.preprocess_CAS(cas, concat, use_fragment_count=True, min_cells_rate=0.03)
pca = PCA(n_components=100, random_state=1234); concat.obsm["X_pca"] = pca.fit_transform(concat.X.toarray())
INSTINCT.create_neighbor_graph(cas, concat); t_prep = time.time() - t0
g = concat.uns.pop("graph_list")
concat.write_h5ad(C + "official_concat.h5ad")
pickle.dump({"graph_list": g, "slice_sizes": [c.shape[0] for c in cas], "t_align": t_align, "t_prep": t_prep}, open(C + "official_graph.pkl", "wb"))
print(f"official prep: align {t_align:.1f}s prep {t_prep:.1f}s concat {concat.shape}", flush=True)
