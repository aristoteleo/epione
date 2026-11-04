"""Gate A noise baseline: pairwise agreement among ArchR seeds 1-3, per sample."""
import os
import itertools, json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from epione.pp._doublets import _archr_filter_mask
OUT = os.environ["EPIONE_DOUBLETS_PARITY_DIR"] + "/archr_fixture/out"
runs = {s: pd.read_csv(f"{OUT}/seed{s}_scores.tsv", sep="\t", index_col=0) for s in (1, 2, 3)}

def removed(df):
    m = _archr_filter_mask(df.DoubletEnrichment.to_numpy(float), df.DoubletScore.to_numpy(float),
                           df.Sample.to_numpy(), 1.0, -np.inf, 1.0)
    return set(df.index[m])

def agree(a, b):
    out = {}
    for s in sorted(a.Sample.unique()):
        x, y = a[a.Sample == s], b.loc[a.index[a.Sample == s]]
        ra, rb = removed(x), removed(y)
        out[s] = dict(A1=spearmanr(x.DoubletEnrichment, y.DoubletEnrichment)[0],
                      A2=len(ra & rb) / max(len(ra | rb), 1),
                      A3=spearmanr(x.DoubletScore, y.DoubletScore)[0])
    return out

pairs = {f"{i}v{j}": agree(runs[i], runs[j]) for i, j in itertools.combinations((1, 2, 3), 2)}
samples = sorted(runs[1].Sample.unique())
base = {s: {g: min(pairs[p][s][g] for p in pairs) for g in ("A1", "A2", "A3")} for s in samples}
json.dump(dict(pairs=pairs, baseline=base), open(f"{OUT}/../../analysis/archr_baseline.json", "w"), indent=1)
for s in samples: print(s, {g: round(v, 3) for g, v in base[s].items()})
