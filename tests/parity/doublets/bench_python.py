"""Benchmark (Epione side) on GSE162690: tile matrix, scrublet, ArchR port.

Usage: python bench_python.py <LowLoading|HighLoading>
Writes bench/<s>_epione_scores.tsv (+ timings). Metrics are computed by bench_metrics.py.
"""
import os
import json, sys, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
import epione as epi

R = Path(os.environ["EPIONE_DOUBLETS_PARITY_DIR"])
D = R / "data/GSE162690"
s = sys.argv[1]
cells = pd.read_csv(D / f"{s}_cells.tsv", sep="\t", index_col=0)
bcs = [c.split("#", 1)[1] for c in cells.index]
sizes_f = D / "hg38.chrom.sizes"
if not sizes_f.exists():
    urllib.request.urlretrieve("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes", sizes_f)
sizes = pd.read_csv(sizes_f, sep="\t", header=None, index_col=0)[1]
sizes = {c: int(v) for c, v in sizes.items() if c in [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]}

t = {}
h5 = R / f"bench/{s}_tiles.h5ad"
if h5.exists():
    import anndata as ad
    adata = ad.read_h5ad(h5)
else:
    t0 = time.time()
    adata = epi.pp.import_fragments(D / f"CellLine_{s}.fragments.tsv.gz", chrom_sizes=sizes,
                                    whitelist=bcs, min_num_fragments=0)
    epi.pp.add_tile_matrix(adata, bin_size=500)
    t["tile_matrix"] = time.time() - t0
    adata.write_h5ad(h5)
adata = adata[[b for b in bcs if b in set(adata.obs_names)]].copy()
print(s, adata.shape, flush=True)

t0 = time.time()
epi.pp.archr_doublets(adata, seed=1)
t["archr_port"] = time.time() - t0

sc = adata.copy()
t0 = time.time()
epi.pp.select_features(sc)
epi.pp.scrublet(sc)
t["scrublet"] = time.time() - t0
adata.obs["scrublet_probability"] = sc.obs["doublet_probability"].to_numpy()
adata.obs["scrublet_score"] = sc.obs["doublet_score"].to_numpy()
adata.obs["scrublet_is_doublet"] = sc.obs["is_doublet"].to_numpy()

rm = adata.copy()
epi.pp.filter_doublets(rm, method="archr", inplace=True)
adata.obs["archr_port_removed"] = ~adata.obs_names.isin(rm.obs_names)

out = adata.obs[["n_fragment", "archr_doublet_score", "archr_doublet_enrichment",
                 "archr_port_removed", "scrublet_probability", "scrublet_score", "scrublet_is_doublet"]]
out.index = ["CellLine_" + s.replace("Loading", "") + "#" + b for b in out.index]
out.to_csv(R / f"bench/{s}_epione_scores.tsv", sep="\t")
json.dump(dict(timings=t, info=adata.uns.get("archr_doublet")), open(R / f"bench/{s}_epione_timings.json", "w"),
          indent=1, default=str)
print(t)
