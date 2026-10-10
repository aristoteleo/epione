"""Compare Epione peak-annotation enrichment against ArchR reference outputs.

Usage:
    python compare.py <fixture_dir> <ref_dir> <ArchR-Hg19-v1.Anno> [--e2e] [--json out.json]

``ref_dir`` holds the ArchR 1.0.3 reference outputs (generated outside this repository).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
from scipy import sparse

import epione as epi
from epione.pl._enrich_heatmap import r_hclust_complete

NUMERIC = ["mlog10Padj", "mlog10p", "Enrichment", "BackgroundProporition", "nBackground",
           "BackgroundFrequency", "CompareProportion", "nCompare", "CompareFrequency"]


def load_fixture(fx):
    t = os.path.join(fx, "tables")
    peaks = pd.read_csv(os.path.join(t, "peaks.tsv"), sep="\t")
    names = (peaks["seqnames"] + ":" + peaks["start"].astype(str) + "-" + peaks["end"].astype(str)).to_numpy()
    adata = ad.AnnData(var=pd.DataFrame(index=pd.Index(names)))
    adata.var["chrom"] = peaks["seqnames"].to_numpy()
    adata.var["start"] = peaks["start"].to_numpy()
    adata.var["end"] = peaks["end"].to_numpy()
    M = scipy.io.mmread(os.path.join(t, "motif_matches.mtx")).tocsr()
    adata.varm["motif"] = M.astype(bool)
    adata.uns["motif_names"] = np.loadtxt(os.path.join(t, "motif_matches_motifs.txt"), dtype=str)
    return adata


def load_markers(ref):
    rd = pd.read_csv(os.path.join(ref, "markers_rowdata.tsv"), sep="\t")
    idx = pd.Index(rd["seqnames"] + ":" + rd["start"].astype(str) + "-" + rd["end"].astype(str))
    out = {}
    for f in sorted(os.listdir(ref)):
        if f.startswith("markers_") and f.endswith(".tsv.gz"):
            df = pd.read_csv(os.path.join(ref, f), sep="\t")
            df.index = idx
            out[f[len("markers_"):-len(".tsv.gz")]] = df
    return out


def load_enrich(ref, prefix):
    out = {}
    for a in NUMERIC + ["feature"]:
        df = pd.read_csv(os.path.join(ref, f"{prefix}_{a}.tsv.gz"), sep="\t", keep_default_na=False,
                         na_values=["NA", "NaN", ""])
        out[a] = df.set_index("feature")
    return out


def cmp_enrich(py, r):
    res = {}
    for a in NUMERIC:
        x = py[a].loc[r[a].index, r[a].columns].to_numpy(np.float64)
        y = r[a].to_numpy(np.float64)
        both_nan = np.isnan(x) & np.isnan(y)
        diff = np.where(both_nan, 0.0, np.abs(x - y))
        res[a] = float(np.nanmax(np.where(np.isnan(diff), np.inf, diff)))
    feat_ok = (py["feature"].loc[r["feature"].index, r["feature"].columns].astype(str).to_numpy()
               == r["feature"].astype(str).to_numpy()).all()
    return max(res.values()), res, bool(feat_ok)


def cmp_heatmap(py, path):
    r = pd.read_csv(path, sep="\t").set_index("row")
    same_names = list(py.index) == list(r.index) and list(map(str, py.columns)) == list(map(str, r.columns))
    err = float(np.max(np.abs(py.to_numpy() - r.to_numpy()))) if py.shape == r.shape else np.inf
    return err, same_names


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture"); ap.add_argument("ref"); ap.add_argument("anno")
    ap.add_argument("--e2e", action="store_true"); ap.add_argument("--json")
    a = ap.parse_args(argv)
    report = {}

    adata = load_fixture(a.fixture)
    markers = load_markers(a.ref)
    bgd = pd.read_csv(os.path.join(a.ref, "bgd_peaks.tsv.gz"), sep="\t").to_numpy() - 1
    adata.varm["bg_peaks"] = bgd
    cut = "FDR <= 0.1 & Log2FC >= 0.5"

    for tag, kw in [("motif_all", dict(cutoff=cut)),
                    ("motif_bgd", dict(cutoff=cut, background="bgdPeaks")),
                    ("motif_or", dict(cutoff="FDR <= 0.5 | Log2FC >= 2"))]:
        e = epi.tl.peak_anno_enrichment(markers, adata, annotation="motif", verbose=False, **kw)
        err, per, fok = cmp_enrich(e, load_enrich(a.ref, f"enrich_{tag}"))
        report[f"enrich_{tag}"] = dict(max_abs_err=err, per_assay=per, feature_identical=fok)
        if tag == "motif_all":
            e_all = e
        if tag == "motif_bgd":
            e_bgd = e

    # unrounded -log10 p diagnostic (group 1, matches in peakSet order, as in the ArchR reference run)
    from epione.tl._peak_anno_enrichment import _phyper_log_upper
    raw = pd.read_csv(os.path.join(a.ref, "raw_mlog10p_group1.tsv"), sep="\t")
    g1 = markers["FDR"].columns[0]
    idx = np.flatnonzero(((markers["FDR"][g1] <= 0.1) & (markers["Log2FC"][g1] >= 0.5)).to_numpy())
    M = sparse.csr_matrix(adata.varm["motif"]).astype(np.float64)
    cf = np.asarray(M[idx].sum(0)).ravel(); bf = np.asarray(M.sum(0)).ravel(); N = M.shape[0]
    mine = np.array([-_phyper_log_upper(c - 1, b, N - b, idx.size) / np.log(10) for c, b in zip(cf, bf)])
    rr = raw["raw"].to_numpy(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(rr == 0, np.abs(mine - rr), np.abs(mine - rr) / np.abs(rr))
    report["raw_mlog10p_max_rel_err"] = float(np.max(rel))

    hm = {
        "default": dict(), "n7_t": dict(n=7, transpose=True), "n5_nocc": dict(n=5, cluster_cols=False),
        "nobin": dict(n=7, binary_cluster_rows=False), "pmax30": dict(n=10, p_max=30, cutoff=5),
        "pmax100_t": dict(n=10, p_max=100, cutoff=5, transpose=True),
    }
    for k, kw in hm.items():
        m = epi.pl.enrich_heatmap(e_all, return_matrix=True, **kw)
        err, same = cmp_heatmap(m, os.path.join(a.ref, f"heatmap_{k}.tsv"))
        report[f"heatmap_{k}"] = dict(max_abs_err=err, dimnames_identical=same)
    m = epi.pl.enrich_heatmap(e_bgd, n=7, cutoff=2, return_matrix=True)
    err, same = cmp_heatmap(m, os.path.join(a.ref, "heatmap_bgd_n7.tsv"))
    report["heatmap_bgd_n7"] = dict(max_abs_err=err, dimnames_identical=same)

    orders = [list(map(int, l.split(","))) for l in open(os.path.join(a.ref, "hclust_orders.txt"))]
    ok = 0
    for i, o in enumerate(orders, start=1):
        x = pd.read_csv(os.path.join(a.ref, f"hclust_in_{i:02d}.tsv"), sep="\t").to_numpy()
        ok += list(r_hclust_complete(x)[2] + 1) == o
    report["hclust_order_identical"] = f"{ok}/{len(orders)}"

    # addPeakAnnotations
    regs = {k: os.path.join(a.ref, f"regions_{k}.bed") for k in ["promoters", "random500", "random5k", "nohit"]}
    epi.tl.add_peak_annotations(adata, regs, key_added="Region", verbose=False)
    R = scipy.io.mmread(os.path.join(a.ref, "region_matches.mtx")).tocsr()
    rn = [l.strip() for l in open(os.path.join(a.ref, "region_matches_names.txt"))]
    P = sparse.csr_matrix(adata.varm["Region"]).astype(np.float64)
    report["addPeakAnnotations"] = dict(
        names_identical=list(adata.uns["Region_names"]) == rn,
        n_mismatch=int(abs(P - (R != 0).astype(np.float64)).sum()) if P.shape == R.shape else -1)
    e = epi.tl.peak_anno_enrichment(markers, adata, annotation="Region", cutoff=cut, verbose=False)
    err, per, fok = cmp_enrich(e, load_enrich(a.ref, "enrich_region_all"))
    report["enrich_region_all"] = dict(max_abs_err=err, feature_identical=fok)

    # addArchRAnnotations
    for col in ["EncodeTFBS", "ATAC"]:
        epi.tl.add_archr_annotations(adata, db=a.anno, collection=col, verbose=False)
        R = scipy.io.mmread(os.path.join(a.ref, f"archr_{col}_matches.mtx")).tocsr()
        rn = [l.strip() for l in open(os.path.join(a.ref, f"archr_{col}_names.txt"))]
        P = sparse.csr_matrix(adata.varm[col]).astype(np.float64)
        report[f"addArchRAnnotations_{col}"] = dict(
            names_identical=list(adata.uns[f"{col}_names"]) == rn, shape=list(P.shape),
            n_mismatch=int(abs(P - (R != 0).astype(np.float64)).sum()) if P.shape == R.shape else -1)
        e = epi.tl.peak_anno_enrichment(markers, adata, annotation=col, cutoff=cut, verbose=False)
        err, per, fok = cmp_enrich(e, load_enrich(a.ref, f"enrich_{col}_all"))
        report[f"enrich_{col}_all"] = dict(max_abs_err=err, feature_identical=fok)

    # customEnrichment
    rng_df = pd.read_csv(os.path.join(a.ref, "custom_ranges.tsv"), sep="\t")
    for tag, kw in [("all", {}), ("bgd", dict(bg_key="bg_peaks"))]:
        py = epi.tl.custom_enrichment(rng_df, adata, annotation="motif", **kw)
        r = pd.read_csv(os.path.join(a.ref, f"custom_{tag}.tsv"), sep="\t")
        same_order = list(py["feature"]) == list(r["feature"])
        err = max(float(np.nanmax(np.abs(py.set_index("feature").loc[r["feature"], c].to_numpy(float)
                                         - r[c].to_numpy(float)))) for c in NUMERIC)
        report[f"customEnrichment_{tag}"] = dict(max_abs_err=err, row_order_identical=same_order)

    if a.e2e:
        report["end_to_end"] = end_to_end(a.fixture, adata, e_all, cut)

    print(json.dumps(report, indent=1, default=str))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def end_to_end(fx, adata_peaks, e_archr, cut):
    """Epione marker test -> enrichment, compared with the ArchR pipeline."""
    from scipy.stats import spearmanr
    t = os.path.join(fx, "tables")
    X = scipy.io.mmread(os.path.join(t, "PeakMatrix.mtx")).T.tocsr()
    cells = pd.read_csv(os.path.join(t, "cells.tsv"), sep="\t").set_index("barcode")
    bc = [l.strip() for l in open(os.path.join(t, "PeakMatrix_cells.txt"))]
    obs = cells.loc[bc, ["Clusters", "TSSEnrichment", "nFrags"]].copy()
    obs["log10_nFrags"] = np.log10(obs["nFrags"])
    pk = ad.AnnData(X=X.astype(np.float32), obs=obs, var=adata_peaks.var.copy())
    pk.varm["motif"] = adata_peaks.varm["motif"]
    pk.uns["motif_names"] = adata_peaks.uns["motif_names"]
    mk = epi.tl.find_marker_features(pk, group_by="Clusters", bias_vars=["TSSEnrichment", "log10_nFrags"],
                                     seed=1, verbose=False)
    e = epi.tl.peak_anno_enrichment(mk, pk, annotation="motif", cutoff=cut, verbose=False)
    a = e["mlog10Padj"]
    b = e_archr["mlog10Padj"].loc[a.index, a.columns]
    rho = spearmanr(a.to_numpy().ravel(), b.to_numpy().ravel()).correlation
    jac = []
    for g in a.columns:
        ta = set(a[g].sort_values(ascending=False).index[:10])
        tb = set(b[g].sort_values(ascending=False).index[:10])
        jac.append(len(ta & tb) / len(ta | tb))
    return dict(spearman_mlog10Padj=float(rho), mean_top10_jaccard=float(np.mean(jac)),
                per_group_top10_jaccard=dict(zip(map(str, a.columns), map(float, jac))))


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
