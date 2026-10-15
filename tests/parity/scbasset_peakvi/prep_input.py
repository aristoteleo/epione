"""Shared input for scBasset / PeakVI parity (run once, Epione env).

PBMC10k multiome ATAC (scGLUE release, the file epi.datasets.pbmc10k_multiome
fetches), filtered exactly as calico/scBasset examples/PBMC_multiome/make_anndata.ipynb:
peaks accessible in > int(0.05 * n_cells) cells, chr1-22/X/Y. Writes
data/atac_ad.h5ad with var chr/start/end (the format scbasset_preprocess.py reads)
and raw counts in X (CSR).
"""
import sys
import anndata as ad
import numpy as np
import scipy.sparse as sp

src, out = sys.argv[1], sys.argv[2]
a = ad.read_h5ad(src)
X = sp.csr_matrix(a.X)
a.X = X
ncell = np.diff(X.tocsc().indptr)
thres = int(a.n_obs * 0.05)
chrs = ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]
keep = (ncell > thres) & a.var["chrom"].astype(str).isin(chrs).to_numpy()
b = a[:, keep].copy()
b.var = b.var[["chrom", "chromStart", "chromEnd"]].copy()
b.var["chr"] = b.var["chrom"].astype(str)
b.var["start"] = b.var["chromStart"].astype(int)
b.var["end"] = b.var["chromEnd"].astype(int)
b.var = b.var[["chr", "start", "end"]]
b.obs = b.obs[["cell_type"]].copy()
b.obs["cell_type"] = b.obs["cell_type"].astype(str).astype("category")
b.X = sp.csr_matrix(b.X, dtype=np.float32)
b.write_h5ad(out)
print(b, "max", b.X.max())
