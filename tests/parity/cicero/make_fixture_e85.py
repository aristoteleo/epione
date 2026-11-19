"""Fixtures F2 (chr19) and F3 (whole genome) from the E8.5 heart multiome ATAC export.

Cells x peaks counts (atac_counts.h5ad) are binarised (as the Cicero vignette's
make_atac_cds(binarize = TRUE)), restricted to the requested chromosomes and
stripped of all-zero peaks. Peak ids become chr_start_end (CellOracle format).
Reduced coordinates: emb_umap.harmony.rna.csv (2-D), identical for R and Python.
Usage: python make_fixture_e85.py <export_dir> <chrom.sizes> <outdir> <chr19|all>
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

exp, sizes, out, which = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
out.mkdir(parents=True, exist_ok=True)
a = ad.read_h5ad(exp / "atac_counts.h5ad")
X = a.X.tocsr().astype(np.float64)
X.data[:] = 1.0
peaks = pd.Index(a.var_names).str.replace(r"^(.*)[-:_](\d+)[-:_](\d+)$", r"\1_\2_\3", regex=True)
chrom = peaks.str.split("_").str[0]
keep = np.ones(len(peaks), bool) if which == "all" else np.asarray(chrom == which)
X = X[:, keep]
peaks, chrom = peaks[keep], chrom[keep]
nz = np.asarray(X.sum(0)).ravel() > 0
X, peaks, chrom = X[:, nz], peaks[nz], chrom[nz]
M = X.T.tocsc()                                   # peaks x cells
M.sort_indices()
M.indices.astype("<i4").tofile(out / "x.i.bin")
M.indptr.astype("<f8").tofile(out / "x.p.bin")
M.data.astype("<f8").tofile(out / "x.x.bin")
(out / "shape.txt").write_text(f"{M.shape[0]}\n{M.shape[1]}\n{M.nnz}\n")
(out / "peaks.txt").write_text("\n".join(peaks) + "\n")
(out / "cells.txt").write_text("\n".join(a.obs_names) + "\n")
um = pd.read_csv(exp / "emb_umap.harmony.rna.csv", index_col=0).loc[a.obs_names]
with open(out / "coords.csv", "w") as f:
    f.write("cell,UMAP_1,UMAP_2\n")
    for c, (u1, u2) in zip(um.index, um.to_numpy()):
        f.write(f"{c},{float(u1)!r},{float(u2)!r}\n")
g = pd.read_csv(sizes, sep="\t", header=None)
g = g[g[0].isin(set(chrom))]
g.to_csv(out / "genome.tsv", sep="\t", header=False, index=False)
print(which, M.shape, "nnz", M.nnz, "chroms", list(g[0]))
