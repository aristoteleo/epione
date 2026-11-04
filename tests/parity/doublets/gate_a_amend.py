"""Amendment A-1: gate statistics averaged over the 9 (Epione seed, ArchR seed) pairs."""
import os
import json, sys, itertools
import h5py, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
import anndata as ad, epione as epi
from epione.pp._doublets import _archr_filter_mask
OUT = os.environ["EPIONE_DOUBLETS_PARITY_DIR"] + "/archr_fixture/out"
ANA = OUT + "/../../analysis"
refs = {j: pd.read_csv(f"{OUT}/seed{j}_scores.tsv", sep="\t", index_col=0) for j in (1, 2, 3)}
base = json.load(open(f"{ANA}/archr_baseline.json"))["baseline"]
FIXED = dict(A1=0.70, A2=0.50, A3=0.60)

def removed(e, s):
    return _archr_filter_mask(np.asarray(e, float), np.asarray(s, float), np.zeros(len(e)), 1.0, -np.inf, 1.0)

def stats(e, s, r):
    ra, rb = removed(e, s), removed(r.DoubletEnrichment, r.DoubletScore)
    return dict(A1=spearmanr(e, r.DoubletEnrichment)[0], A2=(ra & rb).sum() / max((ra | rb).sum(), 1),
                A3=spearmanr(s, r.DoubletScore)[0])

res = {}
for sample in sorted(refs[1].Sample.unique()):
    with h5py.File(f"{OUT}/tile_{sample}.h5") as f:
        nf, nc = f["dim"][:]
        m = sp.csc_matrix((f["x"][:], f["i"][:], f["p"][:]), shape=(nf, nc)); cells = [c.decode() for c in f["cells"][:]]
    a = ad.AnnData(sp.csr_matrix(m.T)); a.obs_names = cells
    a.obs["n_fragment"] = refs[1].loc[cells, "nFrags"].to_numpy()
    eps = {i: epi.pp.archr_doublets(a, seed=i, inplace=False) for i in (1, 2, 3)}
    for i, df in eps.items():
        df.to_csv(f"{ANA}/epione_seed{i}_{sample}.tsv", sep="\t")
    pairs = {f"e{i}a{j}": stats(eps[i].archr_doublet_enrichment.to_numpy(), eps[i].archr_doublet_score.to_numpy(),
                                refs[j].loc[cells]) for i, j in itertools.product((1, 2, 3), (1, 2, 3))}
    gates = {}
    for g in FIXED:
        v = float(np.mean([p[g] for p in pairs.values()]))
        th = float(min(FIXED[g], base[sample][g]))
        gates[g] = dict(value=v, threshold=th, passed=bool(v >= th),
                        pair_min=float(min(p[g] for p in pairs.values())),
                        pair_max=float(max(p[g] for p in pairs.values())))
    res[sample] = dict(gates=gates, pairs=pairs)
    print(sample, {g: (round(x["value"], 3), round(x["threshold"], 3), x["passed"]) for g, x in gates.items()}, flush=True)
res["all_pass"] = {g: all(res[s]["gates"][g]["passed"] for s in refs[1].Sample.unique()) for g in FIXED}
json.dump(res, open(f"{ANA}/gate_a_amendment.json", "w"), indent=1, default=float)
print(res["all_pass"])
