"""Parity of epione.tl.project_bulk_atac / deconvolve_bulk (see MANIFEST.md).

Usage:  python parity_check.py <parity_dir> [heme_peaks.h5ad]

``<parity_dir>`` holds ``ref/`` (ArchR outputs) and the Corces count table
``GSE74912_ATACseq_All_Counts.txt.gz``. Results go to
``<parity_dir>/results.json``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors

import epione as epi
from epione.tl._bulk_projection import LSIModel, bulk_to_features, _read_bulk, _row_zscores

P = Path(sys.argv[1])
R = P / "ref"
HEME = sys.argv[2] if len(sys.argv) > 2 else None
res: dict = {}


def load_model() -> LSIModel:
    prm = json.loads((R / "lsi_params.json").read_text())
    feats = pd.read_csv(R / "lsi_features.tsv", sep="\t")
    idx = np.loadtxt(R / "lsi_idx.txt", dtype=np.int64) - 1
    U = pd.read_csv(R / "lsi_svd_u.tsv", sep="\t").to_numpy()
    d = np.loadtxt(R / "lsi_svd_d.txt")
    rowsm = np.loadtxt(R / "lsi_rowSm.txt")
    assert np.array_equal(idx, np.arange(len(feats))), "LSI idx is not all features"
    return LSIModel(features=feats, loadings=U, singular_values=d, feature_sums=rowsm,
                    n_cells=int(prm["nCol"]), flavor="archr", lsi_method=int(prm["LSIMethod"]),
                    scale_to=float(prm["scaleTo"]), binarize=bool(prm["binarize"]),
                    scale_dims=bool(prm["scaleDims"]), dims=None, seed=int(prm["seed"]),
                    closed=True)


model = load_model()
samples = (R / "bulk_samples.txt").read_text().split()
tab = pd.read_csv(P / "GSE74912_ATACseq_All_Counts.txt.gz", sep="\t")
bulk = tab[["Chr", "Start", "End"] + samples]

# ---------------------------------------------------------------- G0
counts, regs, names = _read_bulk(bulk, None)
t0 = time.time()
M, frac = bulk_to_features(counts, regs, model.features, closed=True, return_overlap=True)
t_map = time.time() - t0
ref_bm = scipy.io.mmread(R / "bulkMat.mtx").tocsc()
ref_frac = float((R / "overlap_ratio.txt").read_text())
g0 = (M != ref_bm).nnz == 0 and frac == ref_frac
res["G0"] = {"pass": bool(g0), "n_diff": int((M != ref_bm).nnz), "overlap": frac,
             "overlap_ref": ref_frac, "seconds": t_map}
print("G0", res["G0"])

# ---------------------------------------------------------------- G1
max_raw, max_sc = 0.0, 0.0
for x in range(1, len(samples) + 1):
    sim = scipy.io.mmread(R / "g1" / f"sim_{x:02d}.mtx").T.tocsr()
    raw_ref = pd.read_csv(R / "g1" / f"raw_{x:02d}.tsv", sep="\t").to_numpy()
    sc_ref = pd.read_csv(R / "g1" / f"scaled_{x:02d}.tsv", sep="\t").to_numpy()
    raw = model._raw_coords(sim)
    max_raw = max(max_raw, float(np.abs(raw - raw_ref).max() / np.abs(raw_ref).max()))
    max_sc = max(max_sc, float(np.abs(_row_zscores(raw) - sc_ref).max()))
res["G1"] = {"pass": bool(max_raw <= 1e-8 and max_sc <= 1e-8),
             "max_rel_raw": max_raw, "max_abs_scaled": max_sc}
print("G1", res["G1"])

# ---------------------------------------------------------------- single-cell side
lsi = pd.read_csv(R / "sc_lsi_scaled.tsv", sep="\t").set_index("cell")
um = pd.read_csv(R / "sc_umap.tsv", sep="\t").set_index("cell").loc[lsi.index]
adata = AnnData(X=sp.csr_matrix((len(lsi), 0), dtype=np.float32),
                obs=pd.DataFrame({"Clusters": pd.Categorical(um["Clusters"])}, index=lsi.index))
adata.obsm["X_lsi_archr"] = lsi.to_numpy()
adata.obsm["X_umap"] = um[["UMAP1", "UMAP2"]].to_numpy()
adata.uns["neighbors"] = {"params": {"n_neighbors": 30, "metric": "cosine"}}
adata.uns["umap"] = {"params": {"min_dist": 0.5}}
clusters = adata.obs["Clusters"].astype(str).to_numpy()

a_lsi = pd.read_csv(R / "archr_sim_lsi.tsv", sep="\t").set_index("cell")
a_umap = pd.read_csv(R / "archr_sim_umap.tsv", sep="\t").set_index("cell").loc[a_lsi.index]
a_run = json.loads((R / "archr_run.json").read_text())


def nearest_cluster(ref_xy, query, k=10):
    nn = NearestNeighbors(n_neighbors=k).fit(ref_xy)
    _, idx = nn.kneighbors(query)
    lab = clusters[idx]
    out = []
    for row in lab:
        v, c = np.unique(row, return_counts=True)
        out.append(v[np.argmax(c)])
    return np.asarray(out)


def per_sample(labels, smp):
    df = pd.crosstab(pd.Series(smp, name="s"), pd.Series(labels, name="c"), normalize="index")
    return df.reindex(columns=sorted(set(clusters)), fill_value=0.0)


# ---------------------------------------------------------------- G2 + G4
t0 = time.time()
ex = epi.tl.project_bulk_atac(adata, bulk, model=model, lsi_key="X_lsi_archr", n_cells=250,
                              rng="archr", seed=1, embedding="X_umap", verbose=False)
t_ex = time.time() - t0
e = pd.DataFrame(ex.obsm["X_lsi"], index=ex.obs_names)
same_names = list(e.index) == list(a_lsi.index)
d2 = float(np.abs(e.loc[a_lsi.index].to_numpy() - a_lsi.to_numpy()).max())
res["G2"] = {"pass": bool(same_names and d2 <= 1e-8), "same_cell_names": same_names,
             "max_abs": d2, "n_pseudocells": int(len(e)), "seconds_epione": t_ex,
             "seconds_archr": a_run["seconds"]}
print("G2", res["G2"])

smp = ex.obs["sample"].astype(str).to_numpy()
lab_e = nearest_cluster(ex.uns["reference_umap"], ex.obsm["X_umap"])
lab_a = nearest_cluster(adata.obsm["X_umap"], a_umap[["UMAP1", "UMAP2"]].to_numpy())
agree = float((lab_e == lab_a).mean())
maj_e = per_sample(lab_e, smp).idxmax(axis=1)
maj_a = per_sample(lab_a, smp).idxmax(axis=1)
maj_agree = float((maj_e == maj_a.loc[maj_e.index]).mean())
rc = [float(v) for v in ex.uns["project_bulk_atac"]["reprojection_correlation"]]
res["G4"] = {"pass": bool(agree >= 0.85 and maj_agree >= 0.95 and min(rc) >= 0.8),
             "per_cell_agreement": agree, "per_sample_majority_agreement": maj_agree,
             "reprojection_cor_epione": rc, "reprojection_cor_archr": a_run["reprojection_cor"]}
print("G4", res["G4"])

# ---------------------------------------------------------------- G3
t0 = time.time()
nu = epi.tl.project_bulk_atac(adata, bulk, model=model, lsi_key="X_lsi_archr", n_cells=250,
                              rng="numpy", seed=1, embedding=None, verbose=False)
t_np = time.time() - t0
ref_lsi = adata.obsm["X_lsi_archr"]
lab_n = nearest_cluster(ref_lsi, nu.obsm["X_lsi"])
lab_al = nearest_cluster(ref_lsi, a_lsi.to_numpy())
smp_a = np.asarray([c.rsplit("#", 1)[0] for c in a_lsi.index])
fn = per_sample(lab_n, nu.obs["sample"].astype(str).to_numpy())
fa = per_sample(lab_al, smp_a).loc[fn.index]
g3a = float((fn.idxmax(axis=1) == fa.idxmax(axis=1)).mean())
tvd = 0.5 * np.abs(fn.to_numpy() - fa.to_numpy()).sum(axis=1)
ratios = []
for s in samples:
    xa = a_lsi.to_numpy()[smp_a == s]
    xn = nu.obsm["X_lsi"][nu.obs["sample"].astype(str).to_numpy() == s]
    ca, cn = xa.mean(axis=0), xn.mean(axis=0)
    spread = np.linalg.norm(xa - ca, axis=1).mean()
    ratios.append(np.linalg.norm(cn - ca) / spread)
ratios = np.asarray(ratios)
res["G3"] = {"pass": bool(g3a >= 0.95 and tvd.mean() <= 0.10 and np.median(ratios) <= 0.25
                          and ratios.max() <= 0.5),
             "majority_agreement": g3a, "mean_tvd": float(tvd.mean()),
             "median_centroid_ratio": float(np.median(ratios)), "max_centroid_ratio": float(ratios.max()),
             "seconds": t_np}
print("G3", res["G3"])

# sorted populations: dominant nearest cluster (LSI) per Corces cell type, ArchR run
CODES = {"1": "HSC", "2": "MPP", "3": "LMPP", "4": "CMP", "5": "GMP", "6": "MEP", "7": "Mono",
         "8": "pDC?", "9": "CD4T", "10": "CD8T", "11": "NK", "13": "B", "14": "CLP", "15": "Ery"}


def ctype(s):
    for tok in s.split("-")[1:]:
        if tok in ("Bcell",):
            return "B"
        if tok in ("CD4",):
            return "CD4T"
        if tok in ("CD8",):
            return "CD8T"
        if tok in ("NK", "Nkcell"):
            return "NK"
        if tok in ("CLP", "CMP", "GMP", "HSC", "MEP", "MPP"):
            return tok
        if tok == "Erythroblast":
            return "Ery"
        code = "".join(ch for ch in tok if ch.isdigit())
        if code and tok[: len(code)] == code:
            return CODES.get(code, code)
    return "?"


ct = pd.Series({s: ctype(s) for s in samples})
dom = fa.groupby(ct.loc[fa.index]).mean().idxmax(axis=1)
res["sorted_populations_dominant_cluster"] = dom.to_dict()
print("sorted populations:", dom.to_dict())

# ---------------------------------------------------------------- G5
if HEME:
    import anndata as ad
    pk = ad.read_h5ad(HEME)
    rng = np.random.default_rng(0)
    groups = pk.obs["Clusters"].astype(str).to_numpy()
    cats = sorted(set(groups), key=lambda c: int(c[1:]))
    ref_mask = np.zeros(pk.n_obs, bool)
    for g in cats:
        ii = np.where(groups == g)[0]
        pick = rng.permutation(ii)[: len(ii) // 2]
        ref_mask[pick] = True
    ref = pk[ref_mask].copy()
    test_idx = {g: np.where((~ref_mask) & (groups == g))[0] for g in cats}
    X = pk.X.tocsr()
    truth, cols = [], []
    for b in range(50):
        prop = rng.dirichlet(np.ones(len(cats)))
        n_per = rng.multinomial(2000, prop)
        rows = np.concatenate([rng.choice(test_idx[g], size=k, replace=True)
                               for g, k in zip(cats, n_per) if k > 0])
        cols.append(np.asarray(X[rows].sum(axis=0)).ravel())
        truth.append(n_per / n_per.sum())
    B = sp.csr_matrix(np.vstack(cols).T.astype(np.int64))
    regions = epi.tl._bulk_projection._var_regions(pk)
    t0 = time.time()
    est = epi.tl.deconvolve_bulk(ref, B, "Clusters", regions=regions, weight="cells")
    t_dc = time.time() - t0
    truth = pd.DataFrame(truth, columns=cats)
    E = est[cats].to_numpy()
    T = truth.to_numpy()
    r = float(np.corrcoef(E.ravel(), T.ravel())[0, 1])
    mae = float(np.abs(E - T).mean())
    res["G5"] = {"pass": bool(r >= 0.90 and mae <= 0.05), "pearson": r, "mae": mae,
                 "median_residual_r2": float(est["residual_r2"].median()), "seconds": t_dc}
    print("G5", res["G5"])

(P / "results.json").write_text(json.dumps(res, indent=2, default=float))
print("ALL PASS" if all(v.get("pass", True) for v in res.values() if isinstance(v, dict) and "pass" in v)
      else "SOME GATES FAIL")
