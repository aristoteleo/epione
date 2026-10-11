"""Parity of ``epione.scenic`` cisTarget / DEM / cistrome merging against the
SCENIC+ reference dumps (see MANIFEST.md).

usage: python parity_cistarget.py {cistarget,dem,menr,all} [out.json]

Needs ``EPIONE_SCENIC_PARITY_DIR`` (reference root containing ``A/`` and
``CONTRACT.md``) and ``EPIONE_SCENIC_DB_DIR`` (directory with the hg38 screen
v10 rankings/scores feathers and the motif2TF table).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import epione.scenic as sc

REF = Path(os.environ["EPIONE_SCENIC_PARITY_DIR"]) / "A"
DBD = Path(os.environ["EPIONE_SCENIC_DB_DIR"])
RANKINGS = DBD / "hg38_screen_v10_clust.regions_vs_motifs.rankings.feather"
SCORES = DBD / "hg38_screen_v10_clust.regions_vs_motifs.scores.feather"
ANNOT = DBD / "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
ANN = ["Direct_annot", "Orthology_annot"]


def region_sets():
    rs = REF / "binarize_dar" / "region_sets"
    out = {}
    for sub in sorted(os.listdir(rs)):
        out[sub] = {f[:-4]: sc.read_bed_regions(rs / sub / f)
                    for f in sorted(os.listdir(rs / sub)) if f.endswith(".bed")}
    return out


def _tsv(p):
    # round_trip: pandas' default float parser is not correctly rounded
    return pd.read_csv(p, sep="\t", keep_default_na=False, na_values=[""],
                       float_precision="round_trip")


def _pairs(df, a, b):
    return set(zip(df[a].astype(str), df[b].astype(str)))


def _tfset(x):
    return frozenset() if not isinstance(x, str) or x == "" else frozenset(x.split(", "))


def check_enrichment(name, res, d, kind, report):
    r2d_ref = _tsv(d / "regions_to_db.tsv")
    report["gate1_regions_to_db"].append(_pairs(r2d_ref, "Target", "Query") == _pairs(res.regions_to_db, "Target", "Query"))
    me_ref = _tsv(d / "motif_enrichment.tsv").set_index(d.joinpath("motif_enrichment.tsv").open().readline().split("\t")[0] or "Unnamed: 0")
    me = res.motif_enrichment
    same_motifs = set(me_ref.index) == set(me.index)
    report["gate4_or_10_motif_set"].append(same_motifs)
    if not same_motifs:
        report["mismatch"].append((name, "motif set", sorted(set(me_ref.index) ^ set(me.index))[:10]))
        return
    me = me.loc[me_ref.index]
    for col in ANN:
        if col in me_ref:
            report["gate6_annotation"].append(all(_tfset(a) == _tfset(b) for a, b in zip(me_ref[col], me[col])))
    if kind == "cisTarget":
        if len(me_ref):
            report["gate3_nes_maxerr"].append(float(np.abs(me_ref["NES"].to_numpy() - me["NES"].to_numpy()).max()))
            report["gate4_rank_at_max"].append(bool((me_ref["Rank_at_max"].to_numpy() == me["Rank_at_max"].to_numpy()).all()))
            report["gate4_motif_hits"].append(bool((me_ref["Motif_hits"].to_numpy() == me["Motif_hits"].to_numpy()).all()))
        auc_ref = _tsv(d / "all_motif_auc.tsv")
        auc_ref = auc_ref.set_index(auc_ref.columns[0])["AUC"]
        aucs = res.extra["aucs"].loc[auc_ref.index]
        report["gate2_auc_maxerr"].append(float(np.abs(auc_ref.to_numpy() - aucs.to_numpy()).max()))
    else:
        if len(me_ref):
            for c in ("Log2FC", "Mean_fg", "Mean_bg", "Adjusted_pval"):
                report["gate10_values_maxerr"].append(float(np.abs(me_ref[c].to_numpy() - me[c].to_numpy()).max()))
        pv = _tsv(d / "all_motif_pvals.tsv").set_index("motif")
        mine = res.extra["pvals"].loc[pv.index]
        report["gate9_pval_maxerr"].append(float(np.abs(pv["pval"].to_numpy() - mine["pval"].to_numpy()).max()))
        report["gate9_padj_maxerr"].append(float(np.abs(pv["padj"].to_numpy() - mine["Adjusted_pval"].to_numpy()).max()))
        fg_ref = _tsv(d / "foreground_regions.tsv")
        bg_ref = _tsv(d / "background_regions.tsv")
        fg, bg = res.extra["foreground_regions"], res.extra["background_regions"]
        report["gate8_foreground"].append(set(sc._ranking_db.frame_to_region_names(fg_ref)) == set(sc._ranking_db.frame_to_region_names(fg)))
        # the reference dump is ``PyRanges.df``: regions grouped by chromosome, sampling
        # order kept within each chromosome -> compare per-chromosome sequences
        def by_chrom(df):
            names = sc._ranking_db.frame_to_region_names(df)
            out = {}
            for c, n in zip(df["Chromosome"].astype(str), names):
                out.setdefault(c, []).append(n)
            return out
        report["gate8_background_order"].append(by_chrom(bg_ref) == by_chrom(bg))
    for kind_h in ("Database", "Region_set"):
        h_ref = _tsv(d / f"motif_hits_{kind_h}.tsv")
        h = {(m, r) for m, regs in res.motif_hits[kind_h].items() for r in regs}
        report["gate5_motif_hits"].append(_pairs(h_ref, "motif", "region") == h)
        c_ref = _tsv(d / f"cistromes_{kind_h}.tsv")
        c = {(t, r) for t, regs in res.cistromes[kind_h].items() for r in regs}
        ok = _pairs(c_ref, "cistrome", "region") == c
        # empty cistromes have no rows in the dump; compare their names too
        report["gate7_cistromes"].append(ok)
        if not ok:
            report["mismatch"].append((name, f"cistromes {kind_h}", len(_pairs(c_ref, "cistrome", "region") ^ c)))


def summarise(report):
    out = {}
    for k, v in report.items():
        if k == "mismatch":
            out[k] = v[:20]
        elif not v:
            out[k] = None
        elif isinstance(v[0], bool):
            out[k] = f"{sum(v)}/{len(v)} identical"
        else:
            out[k] = max(v)
    return out


def run_cistarget_parity(res=None):
    rs = region_sets()
    flat = {f"{g}_{k}": v for g, d in rs.items() for k, v in d.items()}
    t0 = time.perf_counter()
    if res is None:
        db = sc.CisTargetDB(str(RANKINGS))
        cache = os.environ.get("EPIONE_SCENIC_DB_CACHE")
        if cache and Path(cache + ".rankings.npz").exists():
            db.load_cache(cache + ".rankings.npz")
        res = sc.run_cistarget(flat, db, motif_annotation=str(ANNOT))
        if cache and not Path(cache + ".rankings.npz").exists():
            db.save_cache(cache + ".rankings.npz")
    wall = time.perf_counter() - t0
    report = {k: [] for k in ["gate1_regions_to_db", "gate2_auc_maxerr", "gate3_nes_maxerr",
                              "gate4_or_10_motif_set", "gate4_rank_at_max", "gate4_motif_hits",
                              "gate5_motif_hits", "gate6_annotation", "gate7_cistromes", "mismatch"]}
    skipped = (REF / "cistarget" / "skipped_region_sets.txt").read_text().split()
    for name in flat:
        if name in skipped:
            continue
        check_enrichment(name, res[name], REF / "cistarget" / name, "cisTarget", report)
    out = summarise(report)
    out["n_region_sets"] = len(res)
    out["wall_s"] = wall
    return out, res


def run_dem_parity(res=None):
    rs = region_sets()
    ga = pd.read_csv(REF / "inputs" / "genome_annotation.tsv", sep="\t")
    t0 = time.perf_counter()
    if res is None:
        db = sc.CisTargetDB(str(SCORES))
        cache = os.environ.get("EPIONE_SCENIC_DB_CACHE")
        if cache and Path(cache + ".scores.npz").exists():
            db.load_cache(cache + ".scores.npz")
        res = sc.run_dem(rs, db, genome_annotation=ga, motif_annotation=str(ANNOT))
        if cache and not Path(cache + ".scores.npz").exists():
            db.save_cache(cache + ".scores.npz")
    wall = time.perf_counter() - t0
    report = {k: [] for k in ["gate1_regions_to_db", "gate4_or_10_motif_set", "gate5_motif_hits",
                              "gate6_annotation", "gate7_cistromes", "gate8_foreground",
                              "gate8_background_order", "gate9_pval_maxerr", "gate9_padj_maxerr",
                              "gate10_values_maxerr", "mismatch"]}
    ref_names = sorted(p.name for p in (REF / "dem").iterdir() if p.is_dir())
    report["comparisons_same"] = [sorted(res) == ref_names]
    for name in ref_names:
        if name in res:
            check_enrichment(name, res[name], REF / "dem" / name, "DEM", report)
    out = summarise(report)
    out["n_comparisons"] = len(res)
    out["wall_s"] = wall
    return out, res


def run_menr_parity(ctx_res, dem_res):
    sp_in = REF / "scplus_inputs"
    regions = (sp_in / "scATAC_X_cols.txt").read_text().split()
    t0 = time.perf_counter()
    direct, extended = sc.merge_cistromes([dem_res, ctx_res], scplus_regions=regions)
    wall = time.perf_counter() - t0
    out = {}
    import scipy.io as sio
    for kind, a in (("direct", direct), ("extended", extended)):
        M = sio.mmread(str(sp_in / f"cistromes_{kind}.mtx")).tocsc()
        rows = (sp_in / f"cistromes_{kind}_rows.txt").read_text().split()
        cols = (sp_in / f"cistromes_{kind}_cols.txt").read_text().split()
        ref = {cols[j]: {rows[i] for i in M[:, j].nonzero()[0]} for j in range(M.shape[1])}
        X = a.X.tocsc()
        mine = {t: {a.obs_names[i] for i in X[:, j].nonzero()[0]} for j, t in enumerate(a.var_names)}
        var = pd.read_csv(sp_in / f"cistromes_{kind}_var.tsv", sep="\t", index_col=0)
        mot_ok = all(set(var.loc[t, "motifs"].split(",")) == set(a.var.loc[t, "motifs"].split(","))
                     for t in var.index) if set(var.index) == set(a.var_names) else False
        out[f"gate11_{kind}"] = {"same_tfs": set(ref) == set(mine), "same_membership": ref == mine,
                                 "same_motifs": mot_ok, "n_tfs": len(ref)}
    out["wall_s"] = wall
    return out


if __name__ == "__main__":
    what = sys.argv[1]
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    results = {}
    import pickle
    cache = Path(os.environ.get("EPIONE_SCENIC_WORK", ".")) / "parity_results.pkl"
    store = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    recheck = os.environ.get("RECHECK") == "1"
    if what in ("cistarget", "all"):
        results["cistarget"], store["ctx"] = run_cistarget_parity(store.get("ctx") if recheck else None)
        print(json.dumps(results["cistarget"], indent=1, default=str), flush=True)
    if what in ("dem", "all"):
        results["dem"], store["dem"] = run_dem_parity(store.get("dem") if recheck else None)
        print(json.dumps(results["dem"], indent=1, default=str), flush=True)
    if what in ("cistarget", "dem", "all"):
        cache.write_bytes(pickle.dumps(store))
    if what in ("menr", "all"):
        results["menr"] = run_menr_parity(store["ctx"], store["dem"])
        print(json.dumps(results["menr"], indent=1, default=str), flush=True)
    if dest:
        old = json.loads(dest.read_text()) if dest.exists() else {}
        old.update(results)
        dest.write_text(json.dumps(old, indent=1, default=str))
