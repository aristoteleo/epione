"""SpatialGlue parity: official implementation vs ``epione.tl.spatialglue``.

Pre-registered in MANIFEST.md (same folder). Runs, for every seed, the
official Tutorial 4 pipeline (Spatial-epigenome-transcriptome, mouse brain
P22 spatial ATAC-RNA-seq, Zhang et al. 2023) with the upstream package, and
the Epione port on the same inputs, then clusters both joint embeddings
with the same routine (PCA 20 -> mclust EEE, G = 18) and scores them.

Usage::

    python parity_spatialglue.py --data <Dataset7_Mouse_Brain_ATAC> \
        --ref-src <SpatialGlue clone> --out results.json [--seeds 2022 2023 2024]

The official arm imports upstream from ``--ref-src``; the only thing
replaced is ``mclust_R`` (rpy2 + R), because both arms are clustered by
``epi.tl.spatial_domains(method="mclust")`` (pymclustR) -- R is not used.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def _load(data):
    import scanpy as sc

    rna = sc.read_h5ad(f"{data}/adata_RNA.h5ad")
    atac = sc.read_h5ad(f"{data}/adata_peaks_normalized.h5ad")
    rna.var_names_make_unique()
    atac.var_names_make_unique()
    return rna, atac


def run_official(data, seed, device):
    import scanpy as sc
    import torch
    from SpatialGlue.preprocess import construct_neighbor_graph, fix_seed, lsi, pca
    from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue

    data_type = "Spatial-epigenome-transcriptome"
    fix_seed(seed)
    adata_omics1, adata_omics2 = _load(data)
    # --- verbatim Tutorial 4 preprocessing ---
    sc.pp.filter_genes(adata_omics1, min_cells=10)
    sc.pp.filter_cells(adata_omics1, min_genes=200)
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)
    adata_omics1_high = adata_omics1[:, adata_omics1.var["highly_variable"]]
    adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=50)
    adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
    if "X_lsi" not in adata_omics2.obsm.keys():
        sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
        lsi(adata_omics2, use_highly_variable=False, n_components=51)
    adata_omics2.obsm["feat"] = adata_omics2.obsm["X_lsi"].copy()
    d = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)
    model = Train_SpatialGlue(d, datatype=data_type, device=torch.device(device))
    out = model.train()
    return adata_omics1, out


def run_port(data, seed, device):
    import mudata as md

    import epione as epi

    rna, atac = _load(data)
    rna, atac = epi.tl.spatialglue_preprocess(rna, atac, random_seed=seed)
    mdata = md.MuData({"rna": rna, "atac": atac})
    out = epi.tl.spatialglue(mdata, modalities=("rna", "atac"), random_seed=seed,
                             device=device, verbose=False)
    return rna, out


def score(adata, emb):
    import anndata as ad
    from sklearn.metrics import adjusted_rand_score as ari
    from sklearn.metrics import normalized_mutual_info_score as nmi

    import epione as epi

    a = ad.AnnData(np.zeros((adata.n_obs, 1), dtype=np.float32), obs=adata.obs.copy())
    a.obsm["SpatialGlue"] = emb
    a.obsm["spatial"] = np.asarray(adata.obsm["spatial"])
    lab = epi.tl.spatial_domains(a, 18, use_rep="SpatialGlue", method="mclust",
                                 use_pca=True, n_comps=20, key_added="SpatialGlue").to_numpy()
    res = {"labels": lab.astype(int).tolist()}
    for ref in ("ATAC_clusters", "RNA_clusters"):
        y = adata.obs[ref].astype(str).to_numpy()
        res[f"ARI_{ref}"] = float(ari(y, lab))
        res[f"NMI_{ref}"] = float(nmi(y, lab))
    epi.tl.spatial_neighbors(a, coord_type="generic", n_neighs=6)
    oh = np.eye(18)[np.searchsorted(np.unique(lab), lab)]
    res["moran_mean"] = float(np.nanmean(_moran_onehot(a, oh)))
    return res


def _moran_onehot(a, oh):
    import epione as epi

    a.obsm["_oh"] = oh
    return epi.tl.spatial_moran(a, use_rep="_oh").to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref-src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="summarise per-seed result files instead of running")
    args = ap.parse_args()
    sys.path.insert(0, args.ref_src)

    rows = []
    if args.merge:
        for f in args.merge:
            with open(f) as fh:
                rows.extend(json.load(fh)["rows"])
        args.seeds = []
    for seed in args.seeds:
        t0 = time.time()
        a_off, o_off = run_official(args.data, seed, args.device)
        t1 = time.time()
        a_port, o_port = run_port(args.data, seed, args.device)
        t2 = time.time()
        assert list(a_off.obs_names) == list(a_port.obs_names)
        s_off = score(a_off, o_off["SpatialGlue"])
        s_port = score(a_port, o_port["SpatialGlue"])
        from sklearn.metrics import adjusted_rand_score as ari

        row = {
            "seed": seed,
            "n_spots": int(a_off.n_obs),
            "max_abs_diff_emb": float(np.abs(o_off["SpatialGlue"] - o_port["SpatialGlue"]).max()),
            "max_abs_diff_alpha": float(np.abs(o_off["alpha"] - o_port["alpha"]).max()),
            "ARI_port_vs_official": float(ari(s_off["labels"], s_port["labels"])),
            "time_official_s": t1 - t0,
            "time_port_s": t2 - t1,
        }
        for k in s_off:
            if k != "labels":
                row[f"official_{k}"] = s_off[k]
                row[f"port_{k}"] = s_port[k]
        rows.append(row)
        print(json.dumps(row), flush=True)

    summary = {}
    for m in ("ARI_ATAC_clusters", "ARI_RNA_clusters", "NMI_ATAC_clusters", "NMI_RNA_clusters", "moran_mean"):
        o = np.array([r[f"official_{m}"] for r in rows])
        p = np.array([r[f"port_{m}"] for r in rows])
        floor = 0.02
        tol = max(2 * o.std(ddof=1) if len(o) > 1 else 0.0, floor)
        summary[m] = {"official_mean": o.mean(), "official_sd": o.std(ddof=1) if len(o) > 1 else 0.0,
                      "port_mean": p.mean(), "delta": abs(p.mean() - o.mean()), "tol": tol,
                      "pass": bool(abs(p.mean() - o.mean()) <= tol)}
    summary["G0_max_abs_diff_emb"] = max(r["max_abs_diff_emb"] for r in rows)
    summary["G0_pass"] = bool(summary["G0_max_abs_diff_emb"] <= 1e-5)
    summary["G2_min_ARI_port_vs_official"] = min(r["ARI_port_vs_official"] for r in rows)
    summary["G2_pass"] = bool(summary["G2_min_ARI_port_vs_official"] >= 0.95)
    with open(args.out, "w") as fh:
        json.dump({"rows": rows, "summary": summary}, fh, indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
