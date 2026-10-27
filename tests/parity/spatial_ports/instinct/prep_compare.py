"""INSTINCT gates D1/D2: port deterministic pipeline vs cached official outputs."""
import os
import time, pickle, json
import numpy as np, anndata as ad, scipy.sparse as sp
from epione.tl import _instinct as I
from sklearn.decomposition import PCA
D = os.path.join(os.environ["SPATIAL_PARITY_DATA"], "misar_multisp/")
C = os.path.join(os.environ.get("SPATIAL_PARITY_WORK", "."), "cache") + "/"
names = ["E11_0-S1", "E13_5-S1"]  # MANIFEST A1: DEV sections only
cas = [ad.read_h5ad(D + s + "_atac_gt.h5ad") for s in names]
for a in cas:
    if "insertion" in a.obsm: del a.obsm["insertion"]
t0 = time.time(); cas = I.peak_sets_alignment(cas); t_align = time.time() - t0
res = {"t_align_port": t_align}
for j, a in enumerate(cas):
    o = ad.read_h5ad(C + f"official_merged_{names[j]}.h5ad")
    res[f"names_equal_{names[j]}"] = bool(list(o.var_names) == list(a.var_names))
    res[f"merged_maxabs_{names[j]}"] = float(abs(sp.csr_matrix(o.X, dtype=float) - sp.csr_matrix(a.X, dtype=float)).max()) if o.shape == a.shape else None
    res[f"shape_{names[j]}"] = [list(o.shape), list(a.shape)]
for j in range(len(cas)): cas[j].obs_names = [x + "_" + names[j] for x in cas[j].obs_names]
concat = ad.concat(cas, label="slice_name", keys=names)
t0 = time.time(); I.instinct_preprocess(cas, concat, use_fragment_count=True, min_cells_rate=0.03)
concat.obsm["X_pca"] = PCA(n_components=100, random_state=1234).fit_transform(concat.X.toarray())
I._create_neighbor_graph(cas, concat); res["t_prep_port"] = time.time() - t0
oc = ad.read_h5ad(C + "official_concat.h5ad"); og = pickle.load(open(C + "official_graph.pkl", "rb"))
res["concat_names_equal"] = bool(list(oc.var_names) == list(concat.var_names) and list(oc.obs_names) == list(concat.obs_names))
res["tfidf_maxabs"] = float(abs(sp.csr_matrix(oc.X) - sp.csr_matrix(concat.X)).max())
res["X_pca_maxabs"] = float(np.abs(oc.obsm["X_pca"] - concat.obsm["X_pca"]).max())
res["graphs_equal"] = bool(all(np.array_equal(a, b) for a, b in zip(og["graph_list"], concat.uns["graph_list"])))
res["t_align_official"] = og["t_align"]; res["t_prep_official"] = og["t_prep"]
res["D1_pass"] = bool(all(v for k, v in res.items() if k.startswith("names_equal")) and res["concat_names_equal"] and res["graphs_equal"]
                      and all((res[k] is not None and res[k] <= 1e-12) for k in res if k.startswith("merged_maxabs")) and res["tfidf_maxabs"] <= 1e-12)
res["D2_pass"] = bool(res["X_pca_maxabs"] <= 1e-8)
json.dump(res, open("gates_D.json", "w"), indent=1); print(json.dumps(res, indent=1))
