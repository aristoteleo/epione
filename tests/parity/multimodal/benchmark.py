"""Benchmark of joint RNA + ATAC representations (MANIFEST section B).

Usage::

    EPIONE_MULTIMODAL_PARITY_DIR=... python benchmark.py multivi   # on a GPU
    EPIONE_MULTIMODAL_PARITY_DIR=... python benchmark.py run       # CPU; reads the MultiVI latent

Settings fixed before the run: RNA PCA and ATAC LSI exactly as in the WNN
parity (section W); MultiVI on features detected in >= 1% of cells (as in the
scvi-tools MultiVI tutorial), default architecture, seed 0; Leiden at
resolution 1.0 on a 20-NN graph, scores averaged over Leiden seeds 0-2.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

PARITY = Path(os.environ.get("EPIONE_MULTIMODAL_PARITY_DIR", ""))
OUT = PARITY / "benchmark"


def load():
    import epione as epi

    rna, atac = epi.datasets.pbmc10k_multiome(PARITY / "data")
    return rna, atac


def embeddings(rna, atac):
    import scanpy as sc
    import epione as epi

    r = rna.copy()
    sc.pp.highly_variable_genes(r, n_top_genes=2000, flavor="seurat_v3")
    hvg = r.var_names[r.var.highly_variable]
    sc.pp.normalize_total(r, target_sum=1e4)
    sc.pp.log1p(r)
    rl = r.copy()
    r = r[:, hvg].copy()
    sc.pp.scale(r, max_value=10)
    sc.tl.pca(r, n_comps=50, random_state=0)
    a = atac.copy()
    epi.tl.lsi(a, n_components=50, remove_first_component=False, random_state=0)
    return r.obsm["X_pca"].astype(np.float64), a.obsm["X_lsi"].astype(np.float64), rl, hvg


def gene_activity(rna, atac, upstream=2000):
    """Peak counts summed over gene body + 2 kb upstream (strand aware)."""
    g = rna.var
    p = atac.var
    rows, cols = [], []
    peaks_by_chrom = {c: (np.asarray(df["chromStart"]), np.asarray(df["chromEnd"]), np.where(p["chrom"] == c)[0])
                      for c, df in p.groupby("chrom", observed=True)}
    for gi, (c, s, e, strand) in enumerate(zip(g["chrom"], g["chromStart"], g["chromEnd"], g["strand"])):
        if c not in peaks_by_chrom or pd.isna(s):
            continue
        s, e = (s - upstream, e) if strand == "+" else (s, e + upstream)
        ps, pe, pidx = peaks_by_chrom[c]
        hit = pidx[(ps < e) & (pe > s)]
        rows.extend([gi] * len(hit))
        cols.extend(hit.tolist())
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(rna.n_vars, atac.n_vars))
    return sp.csr_matrix(atac.X) @ M.T


def run_multivi():
    import torch
    import epione as epi

    rna, atac = load()
    keep_g = np.asarray((rna.X > 0).sum(axis=0)).ravel() >= 0.01 * rna.n_obs
    keep_p = np.asarray((atac.X > 0).sum(axis=0)).ravel() >= 0.01 * atac.n_obs
    r, a = rna[:, keep_g].copy(), atac[:, keep_p].copy()
    t0 = time.time()
    epi.tl.multivi(r, a, seed=0, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    secs = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "multivi_latent.npy", r.obsm["X_multivi"])
    json.dump({"seconds": secs, "n_genes": int(keep_g.sum()), "n_peaks": int(keep_p.sum()),
               "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None},
              open(OUT / "multivi_run.json", "w"), indent=1)
    print("multivi done", secs)


def _graph_from_rep(x, k=20):
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(np.zeros((x.shape[0], 1)))
    a.obsm["X"] = x
    sc.pp.neighbors(a, n_neighbors=k, use_rep="X", random_state=0)
    return a


def _scores(a, ct, neighbors_key=None, emb=None, k=20):
    import scanpy as sc
    from sklearn import metrics as skm

    aris, nmis = [], []
    for seed in range(3):
        sc.tl.leiden(a, resolution=1.0, random_state=seed, flavor="igraph", n_iterations=-1, directed=False,
                     neighbors_key=neighbors_key, key_added="leiden")
        cl = np.asarray(a.obs["leiden"])
        aris.append(skm.adjusted_rand_score(ct, cl))
        nmis.append(skm.normalized_mutual_info_score(ct, cl))
    if emb is None:
        sc.tl.umap(a, neighbors_key=neighbors_key, random_state=0)
        emb = a.obsm["X_umap"]
    sil = skm.silhouette_score(emb, ct, sample_size=min(5000, len(ct)), random_state=0)
    key = (a.uns[neighbors_key]["distances_key"] if neighbors_key else "distances")
    D = a.obsp[key].tocsr()
    pur = []
    for i in range(D.shape[0]):
        nb = D.indices[D.indptr[i]:D.indptr[i + 1]]
        if len(nb):
            pur.append(np.mean(ct[nb] == ct[i]))
    return {"ARI": float(np.mean(aris)), "ARI_sd": float(np.std(aris)), "NMI": float(np.mean(nmis)),
            "silhouette": float(sil), "kNN_purity": float(np.mean(pur))}


def run():
    import anndata as ad
    import scanpy as sc
    import epione as epi

    rna, atac = load()
    ct = np.asarray(rna.obs["cell_type"]).astype(str)
    timings = {}
    pca, lsi, rl, hvg = embeddings(rna, atac)
    res = {}
    res["RNA (PCA)"] = _scores(_graph_from_rep(pca), ct, emb=pca)
    res["ATAC (LSI 2:50)"] = _scores(_graph_from_rep(lsi[:, 1:]), ct, emb=lsi[:, 1:])

    # CCA (cca_py via epi.tl.integrate) between RNA and ATAC gene activity; per cell, mean of both sides.
    t0 = time.time()
    ga = ad.AnnData(gene_activity(rna, atac).astype(np.float32), obs=rna.obs[[]].copy(), var=rna.var[[]].copy())
    sc.pp.normalize_total(ga, target_sum=1e4)
    sc.pp.log1p(ga)
    r_h, g_h = rl[:, hvg].copy(), ga[:, hvg].copy()
    epi.tl.integrate(g_h, r_h, method="cca", features=list(hvg), num_cc=30)
    cca = (g_h.obsm["X_cca"].astype(np.float64) + r_h.obsm["X_cca"].astype(np.float64)) / 2
    timings["CCA"] = time.time() - t0
    res["CCA (RNA vs gene activity)"] = _scores(_graph_from_rep(cca), ct, emb=cca)

    holder = ad.AnnData(np.zeros((rna.n_obs, 1)), obs=pd.DataFrame(index=rna.obs_names))
    holder.obsm["X_pca"], holder.obsm["X_lsi"], holder.obsm["X_lsi_2_50"] = pca, lsi, lsi[:, 1:]
    t0 = time.time()
    comb = epi.tl.combined_dims(holder, ["X_pca", "X_lsi_2_50"])
    timings["combined_dims"] = time.time() - t0
    res["combined_dims"] = _scores(_graph_from_rep(comb), ct, emb=comb)

    t0 = time.time()
    epi.tl.wnn(holder, reps=["X_pca", "X_lsi"], dims=[None, range(2, 51)], modality_names=["RNA", "ATAC"])
    timings["WNN"] = time.time() - t0
    res["WNN"] = _scores(holder, ct, neighbors_key="wnn")

    mv = OUT / "multivi_latent.npy"
    if mv.exists():
        z = np.load(mv)
        res["MultiVI"] = _scores(_graph_from_rep(z), ct, emb=z)
        timings["MultiVI"] = json.load(open(OUT / "multivi_run.json"))["seconds"]

    df = pd.DataFrame(res).T
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "benchmark.tsv", sep="\t")
    json.dump(timings, open(OUT / "timings.json", "w"), indent=1)
    print(df.round(3).to_string())
    print(json.dumps(timings, indent=1))


if __name__ == "__main__":
    {"multivi": run_multivi, "run": run}[sys.argv[1]]()
