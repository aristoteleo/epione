"""Python side of the ArchR parity check for var_deviations /
correlate_matrices / positive_tf_regulators.

Usage: python py_parity.py <ref dir written by r_reference.R> [out.json]
Gates are pre-registered in MANIFEST.md.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io, sparse, stats

import epione as epi
from epione.tl._tf_regulators import _knn_groups, _row_zscore


def _nlog10(p):
    return -np.log10(np.clip(np.asarray(p, dtype=float), 1e-300, None))


def load(ref: Path):
    cells = pd.read_csv(ref / "cells.tsv", sep="\t")
    rd = pd.read_csv(ref / "rD.tsv", sep="\t").set_index("barcode").loc[cells.barcode]
    lsi_raw = pd.read_csv(ref / "lsi_raw.tsv", sep="\t").set_index("barcode").loc[cells.barcode]
    gs = io.mmread(ref / "genescore.mtx").tocsr().T.tocsr()
    gsf = pd.read_csv(ref / "genescore_features.tsv", sep="\t")
    z = pd.read_csv(ref / "cisbp_z.tsv.gz", sep="\t").set_index("motif")[cells.barcode.tolist()]

    adata = ad.AnnData(X=gs, obs=cells.set_index("barcode"))
    adata.var_names = gsf["name"].astype(str).to_numpy()
    adata.obsm["X_rD"] = rd.to_numpy()
    adata.obsm["X_iterative_lsi"] = lsi_raw.to_numpy()
    adata.obsm["cisbp"] = z.to_numpy().T
    adata.uns["cisbp_names"] = z.index.to_numpy()
    return adata, gsf


def run(ref: Path) -> dict:
    ref = Path(ref)
    adata, gsf = load(ref)
    m = {}

    # 1-4 getVarDeviations
    fx = Path(os.environ.get("EPIONE_ARCHR_FIXTURE",
              "/scratch/users/steorra/analysis/25_epione/epione_dev/fixture/tables"))
    for tag, (key, names) in {
        "MotifMatrix": ("jaspar", None),
        "cisbpMatrix": ("cisbp", None),
    }.items():
        if tag == "MotifMatrix":
            zj = pd.read_csv(fx / "MotifMatrix_z.tsv.gz", sep="\t").set_index("motif")[adata.obs_names.tolist()]
            adata.obsm["jaspar"] = zj.to_numpy().T
            adata.uns["jaspar_names"] = zj.index.to_numpy()
        r = pd.read_csv(ref / f"vardev_{tag}.tsv", sep="\t")
        py = epi.tl.var_deviations(adata, key=key, sample_key="Sample", verbose=False)
        r = r.set_index("name"); p = py.set_index("name").loc[r.index]
        m[f"vardev_{tag}_combinedVars_maxabs"] = float(np.max(np.abs(p.combinedVars - r.combinedVars)))
        m[f"vardev_{tag}_combinedMeans_maxabs"] = float(np.max(np.abs(p.combinedMeans - r.combinedMeans)))
        m[f"vardev_{tag}_rank_mismatches"] = int((p["rank"].to_numpy() != r["rank"].to_numpy()).sum())

    # scaleDims port: rowZ(raw LSI)[, kept dims] == ArchR getReducedDims
    c2d = pd.read_csv(ref / "corToDepth.tsv", sep="\t")
    kept = np.flatnonzero(np.abs(c2d.scaled.to_numpy()[:30]) <= 0.75)
    m["reducedDims_maxabs"] = float(np.max(np.abs(_row_zscore(adata.obsm["X_iterative_lsi"])[:, kept]
                                                    - adata.obsm["X_rD"])))

    # 5 kNN groups with R's seed cells
    idx = np.loadtxt(ref / "knn_seed_idx.txt", dtype=np.int64)
    py_seed = epi.tl._tf_regulators.RRandom(1, "L'Ecuyer-CMRG").sample_int(adata.n_obs, 500)
    m["seed_cells_archr_rng_mismatches"] = int((py_seed != idx).sum())
    knn_r = pd.read_csv(ref / "knn_all.tsv", sep="\t", header=None).to_numpy() - 1
    keep_r = np.loadtxt(ref / "knn_keep.txt", dtype=np.int64) == 0
    groups_py = _knn_groups(adata.obsm["X_rD"], idx - 1, 100, 0.8)
    groups_r = knn_r[keep_r]
    same_n = groups_py.shape[0] == groups_r.shape[0]
    m["knn_groups_n_py"] = int(groups_py.shape[0]); m["knn_groups_n_r"] = int(groups_r.shape[0])
    m["knn_group_set_mismatches"] = (
        int(sum(set(a) != set(b) for a, b in zip(groups_py, groups_r))) if same_n else -1)

    # 6-8 correlateMatrices (injected = R's seed cells; identical to rng='R')
    r = pd.read_csv(ref / "correlateMatrices_GSM_cisbp.tsv", sep="\t")
    common = dict(key2="cisbp", name1="GeneScoreMatrix", name2="cisbpMatrix", use_rep="X_rD",
                  cor_cutoff=None, k=100, knn_iteration=500, overlap_cutoff=0.8, verbose=False)
    py = epi.tl.correlate_matrices(adata, seed_cells=idx, **common)
    py_free = epi.tl.correlate_matrices(adata, rng="archr", seed=1, **common)
    m["archr_rng_cor_maxabs"] = float(np.nanmax(np.abs(py_free["cor"].to_numpy() - py["cor"].to_numpy())))
    key = ["GeneScoreMatrix_name", "cisbpMatrix_name"]
    mr = r.merge(py, left_on=["GeneScoreMatrix_name", "cisbpMatrix_name"],
                 right_on=key, suffixes=("_r", "_py"))
    m["cor_pairs_r"] = int(len(r)); m["cor_pairs_py"] = int(len(py)); m["cor_pairs_merged"] = int(len(mr))
    ok = mr["cor_r"].notna() & mr["cor_py"].notna()
    m["cor_nan_mismatch"] = int((mr["cor_r"].isna() != mr["cor_py"].isna()).sum())
    m["cor_maxabs"] = float(np.max(np.abs(mr.loc[ok, "cor_r"] - mr.loc[ok, "cor_py"])))
    m["nlog10_pval_maxabs"] = float(np.max(np.abs(_nlog10(mr.loc[ok, "pval_r"]) - _nlog10(mr.loc[ok, "pval_py"]))))
    m["nlog10_padj_maxabs"] = float(np.max(np.abs(_nlog10(mr.loc[ok, "padj_r"]) - _nlog10(mr.loc[ok, "padj_py"]))))

    # free-running (numpy RNG) vs R
    py_np = epi.tl.correlate_matrices(adata, rng="numpy", seed=1, **common)
    mf = r.merge(py_np, on=key, suffixes=("_r", "_py"))
    okf = mf["cor_r"].notna() & mf["cor_py"].notna()
    m["free_numpy_spearman_cor"] = float(stats.spearmanr(mf.loc[okf, "cor_r"], mf.loc[okf, "cor_py"])[0])
    m["free_numpy_spearman_nlog10_padj"] = float(stats.spearmanr(
        _nlog10(mf.loc[okf, "padj_r"]), _nlog10(mf.loc[okf, "padj_py"]))[0])

    # 9 maxDelta
    gse = pd.read_csv(ref / "groupSE_cisbp_z.tsv", sep="\t").set_index("name")
    gm = epi.tl.group_means(adata, "Clusters", key="cisbp")
    gm = gm.loc[gse.index, [c for c in gse.columns if c != "maxDelta"]]
    m["groupSE_maxabs"] = float(np.max(np.abs(gm.to_numpy() - gse.drop(columns="maxDelta").to_numpy())))

    # 10 positive TF regulators, rng='R' (bit-exact R seed stream) and free numpy
    rr = pd.read_csv(ref / "positive_tf_regulators.tsv", sep="\t")
    common_tf = {k: v for k, v in common.items() if k not in ("key2", "name1", "name2")}
    tf = epi.tl.positive_tf_regulators(adata, motif_key="cisbp", groupby="Clusters",
                                       gene_label="GeneScoreMatrix", motif_label="cisbpMatrix",
                                       rng="archr", seed=1, **common_tf)
    md = rr.merge(tf, on="cisbpMatrix_name", suffixes=("_r", "_py"))
    m["maxDelta_maxabs"] = float(np.max(np.abs(md["maxDelta_r"] - md["maxDelta_py"])))
    yes_r = set(rr.loc[rr.TFRegulator == "YES", "GeneScoreMatrix_name"])
    yes_py = set(tf.loc[tf.TFRegulator == "YES", "GeneScoreMatrix_name"])
    m["tf_yes_r"] = sorted(yes_r); m["tf_yes_py_archr_rng"] = sorted(yes_py)
    m["tf_yes_identical_archr_rng"] = yes_r == yes_py
    m["tf_table_order_identical_archr_rng"] = bool(
        list(rr["cisbpMatrix_name"]) == list(tf["cisbpMatrix_name"]))
    tf_np = epi.tl.positive_tf_regulators(adata, motif_key="cisbp", groupby="Clusters",
                                          gene_label="GeneScoreMatrix", motif_label="cisbpMatrix",
                                          rng="numpy", seed=1, key_added=None, **common_tf)
    yes_np = set(tf_np.loc[tf_np.TFRegulator == "YES", "GeneScoreMatrix_name"])
    m["tf_yes_py_numpy"] = sorted(yes_np)
    m["free_numpy_tf_jaccard"] = len(yes_r & yes_np) / max(1, len(yes_r | yes_np))
    return m


if __name__ == "__main__":
    res = run(Path(sys.argv[1]))
    out = json.dumps(res, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    print(out)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(out)
