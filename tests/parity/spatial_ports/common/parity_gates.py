"""Generic parity gates for stochastic spatial-embedding ports (pre-registered layout).

python parity_gates.py --runs <dir with official_<s>.npz, port_<s>.npz> --seeds 20 21 ...
       --labels <npy/txt of ground-truth labels> --n-domains 9 --out gates.json
       [--extra-keys weights]  [--cluster mclust|leiden|given] [--emb-key emb]

Gates (see each method's MANIFEST.md):
 G0  max|emb_off_s - emb_port_s| (+ extra keys) <= tol0, every seed  (exactness, reported)
 G1  ARI(dom_port_s, dom_off_s) >= 0.95 every seed                    (exactness, reported)
 G2  median_{s!=t} ARI(port_s, off_t) >= median_{s!=t} ARI(off_s, off_t) - 0.05
 G3  median_{s!=t} Procrustes(port_s, off_t) >= median_{s!=t} Procrustes(off_s, off_t) - 0.02
 G4  |mean ARI(dom, truth)_port - _off| <= max(2 SD_off, 0.02), same for NMI
"""
import argparse, itertools, json
import numpy as np
from scipy.spatial import procrustes
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--labels", default=None, help="npy/tsv of truth labels; default: the runs' own 'truth' key")
ap.add_argument("--n-domains", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--emb-key", default="emb")
ap.add_argument("--extra-keys", nargs="*", default=[])
ap.add_argument("--tol0", type=float, default=1e-5)
ap.add_argument("--cluster", default="mclust", choices=["mclust", "given"])
ap.add_argument("--label-key", default="labels", help="npz key with the arm's own domains when --cluster given")
ap.add_argument("--use-pca", type=int, default=1)
ap.add_argument("--n-comps", type=int, default=20)
a = ap.parse_args()

truth = None
if a.labels:
    truth = np.load(a.labels, allow_pickle=True) if a.labels.endswith(".npy") else np.loadtxt(a.labels, dtype=str, delimiter="\t")
    truth = np.asarray(truth).astype(str)

def domains(emb):
    import anndata as ad
    import epione as epi
    t = ad.AnnData(np.zeros((emb.shape[0], 1), dtype=np.float32)); t.obsm["E"] = emb
    epi.tl.spatial_domains(t, a.n_domains, use_rep="E", method="mclust", use_pca=bool(a.use_pca), n_comps=a.n_comps)
    return t.obs["domain"].astype(int).to_numpy()

R = {}
for arm in ("official", "port"):
    for s in a.seeds:
        z = np.load(f"{a.runs}/{arm}_{s}.npz", allow_pickle=True)
        d = {k: z[k] for k in z.files}
        d["dom"] = domains(d[a.emb_key]) if a.cluster == "mclust" else np.asarray(d[a.label_key]).astype(str)
        R[(arm, s)] = d
ref_obs = R[("official", a.seeds[0])].get("obs_names")
for k, d in R.items():
    if ref_obs is not None and "obs_names" in d:
        assert np.array_equal(np.asarray(d["obs_names"]).astype(str), np.asarray(ref_obs).astype(str)), f"spot order differs: {k}"
if truth is None:
    truth = np.asarray(R[("official", a.seeds[0])]["truth"]).astype(str)

def psim(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return 1.0 - procrustes(x, y)[2]

out = {"seeds": a.seeds, "per_seed": {}}
g0 = []; g1 = []
for s in a.seeds:
    o, p = R[("official", s)], R[("port", s)]
    diffs = {k: float(np.max(np.abs(np.asarray(o[k], float) - np.asarray(p[k], float)))) for k in [a.emb_key] + a.extra_keys}
    ari_op = ARI(o["dom"], p["dom"])
    out["per_seed"][s] = {"maxabs": diffs, "ARI_port_vs_off": ari_op,
                          "ARI_truth_off": ARI(truth, o["dom"]), "ARI_truth_port": ARI(truth, p["dom"]),
                          "NMI_truth_off": NMI(truth, o["dom"]), "NMI_truth_port": NMI(truth, p["dom"]),
                          "sec_off": float(o.get("seconds", np.nan)), "sec_port": float(p.get("seconds", np.nan))}
    g0.append(max(diffs.values()) <= a.tol0); g1.append(ari_op >= 0.95)
pairs = [(s, t) for s, t in itertools.product(a.seeds, a.seeds) if s != t]
ari_oo = [ARI(R[("official", s)]["dom"], R[("official", t)]["dom"]) for s, t in pairs]
ari_po = [ARI(R[("port", s)]["dom"], R[("official", t)]["dom"]) for s, t in pairs]
ps_oo = [psim(R[("official", s)][a.emb_key], R[("official", t)][a.emb_key]) for s, t in pairs]
ps_po = [psim(R[("port", s)][a.emb_key], R[("official", t)][a.emb_key]) for s, t in pairs]
ps_same = [psim(R[("port", s)][a.emb_key], R[("official", s)][a.emb_key]) for s in a.seeds]
def stat(arm, k): return np.array([out["per_seed"][s][f"{k}_{arm}"] for s in a.seeds])
g4 = {}
for k in ("ARI_truth", "NMI_truth"):
    off, por = stat("off", k), stat("port", k)
    tol = max(2 * off.std(ddof=1) if len(off) > 1 else 0.0, 0.02)
    g4[k] = {"mean_off": off.mean(), "sd_off": off.std(ddof=1) if len(off) > 1 else 0.0, "mean_port": por.mean(),
             "tol": tol, "pass": bool(abs(por.mean() - off.mean()) <= tol)}
out["gates"] = {
    "G0": {"pass": bool(all(g0)), "tol": a.tol0},
    "G1": {"pass": bool(all(g1)), "min_ARI": min(out["per_seed"][s]["ARI_port_vs_off"] for s in a.seeds)},
    "G2": {"median_cross": float(np.median(ari_po)), "median_seed2seed": float(np.median(ari_oo)),
           "pass": bool(np.median(ari_po) >= np.median(ari_oo) - 0.05)},
    "G3": {"median_cross": float(np.median(ps_po)), "median_seed2seed": float(np.median(ps_oo)),
           "same_seed_median": float(np.median(ps_same)),
           "pass": bool(np.median(ps_po) >= np.median(ps_oo) - 0.02)},
    "G4": g4,
}
out["PASS_required"] = bool(out["gates"]["G2"]["pass"] and out["gates"]["G3"]["pass"] and all(v["pass"] for v in g4.values()))
json.dump(out, open(a.out, "w"), indent=1, default=float)
print(json.dumps(out["gates"], indent=1, default=float)); print("PASS_required:", out["PASS_required"])
