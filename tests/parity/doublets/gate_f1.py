"""Gate F1: epi filter_doublets(method='archr') vs ArchR filterDoublets on ArchR's own scores."""
import os
import json, sys
import numpy as np, pandas as pd
import anndata as ad
import epione as epi
OUT = os.environ["EPIONE_DOUBLETS_PARITY_DIR"] + "/archr_fixture/out"
df = pd.read_csv(f"{OUT}/seed1_scores.tsv", sep="\t", index_col=0)
a = ad.AnnData(obs=df.rename(columns={"DoubletScore": "archr_doublet_score",
                                      "DoubletEnrichment": "archr_doublet_enrichment"}))
epi.pp.filter_doublets(a, method="archr", sample_key="Sample", inplace=True)
ours = set(df.index) - set(a.obs_names)
archr = set(open(f"{OUT}/seed1_filterDoublets_removed.txt").read().split())
res = {s: dict(archr=len([c for c in archr if df.Sample[c] == s]),
               epione=len([c for c in ours if df.Sample[c] == s])) for s in sorted(df.Sample.unique())}
res["identical"] = ours == archr
print(json.dumps(res, indent=1)); json.dump(res, open(f"{OUT}/../../analysis/gate_f1.json", "w"), indent=1)
