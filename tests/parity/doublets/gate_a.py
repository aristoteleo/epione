"""Gate A: epi.pp.archr_doublets vs ArchR 1.0.3 on identical tile matrices (per sample)."""
import os
import json, sys, time
import h5py, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
import anndata as ad
import epione as epi
from epione.pp._doublets import _archr_filter_mask
OUT = os.environ["EPIONE_DOUBLETS_PARITY_DIR"] + "/archr_fixture/out"
ANA = OUT + "/../../analysis"
ref = pd.read_csv(f"{OUT}/seed1_scores.tsv", sep="\t", index_col=0)
base = json.load(open(f"{ANA}/archr_baseline.json"))["baseline"]
FIXED = dict(A1=0.70, A2=0.50, A3=0.60)

def load(sample):
    with h5py.File(f"{OUT}/tile_{sample}.h5") as f:
        nf, nc = f["dim"][:]
        m = sp.csc_matrix((f["x"][:], f["i"][:], f["p"][:]), shape=(nf, nc))
        cells = [c.decode() for c in f["cells"][:]]
    a = ad.AnnData(sp.csr_matrix(m.T))
    a.obs_names = cells
    a.obs["n_fragment"] = ref.loc[cells, "nFrags"].to_numpy()
    return a

def removed(e, s):
    return _archr_filter_mask(e, s, np.zeros(len(e)), 1.0, -np.inf, 1.0)

res, epi_runs = {}, {}
for sample in sorted(ref.Sample.unique()):
    a = load(sample)
    r = ref.loc[a.obs_names]
    for seed in (1, 2, 3):
        t0 = time.time()
        df = epi.pp.archr_doublets(a, seed=seed, inplace=False)
        epi_runs[(sample, seed)] = (df, time.time() - t0)
    df, secs = epi_runs[(sample, 1)]
    e, s = df.archr_doublet_enrichment.to_numpy(), df.archr_doublet_score.to_numpy()
    re_, rs = r.DoubletEnrichment.to_numpy(float), r.DoubletScore.to_numpy(float)
    ra, rb = removed(e, s), removed(re_, rs)
    stats = dict(A1=spearmanr(e, re_)[0], A2=(ra & rb).sum() / max((ra | rb).sum(), 1),
                 A3=spearmanr(s, rs)[0])
    gates = {g: dict(value=float(v), threshold=float(min(FIXED[g], base[sample][g])),
                     passed=bool(v >= min(FIXED[g], base[sample][g]))) for g, v in stats.items()}
    # Epione's own seed spread (information only)
    own = {}
    for i, j in ((1, 2), (1, 3), (2, 3)):
        x, y = epi_runs[(sample, i)][0], epi_runs[(sample, j)][0]
        own[f"{i}v{j}"] = float(spearmanr(x.archr_doublet_enrichment, y.archr_doublet_enrichment)[0])
    res[sample] = dict(n_cells=a.n_obs, gates=gates, epione_seed_spearman_enrichment=own,
                       seconds_seed1=secs, n_removed=dict(epione=int(ra.sum()), archr=int(rb.sum())))
    print(sample, {g: (round(v["value"], 3), round(v["threshold"], 3), v["passed"]) for g, v in gates.items()},
          f"{secs:.0f}s", flush=True)
    df.to_csv(f"{ANA}/epione_seed1_{sample}.tsv", sep="\t")
res["all_pass"] = {g: all(res[s]["gates"][g]["passed"] for s in ref.Sample.unique()) for g in FIXED}
json.dump(res, open(f"{ANA}/gate_a.json", "w"), indent=1, default=float)
print(res["all_pass"])
