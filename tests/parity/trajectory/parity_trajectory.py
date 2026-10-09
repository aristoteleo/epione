"""Compare the epione trajectory port with ArchR 1.0.3 reference outputs.

Usage::

    python parity_trajectory.py <ref_dir> <fixture_tables_dir> [out.json]

``<ref_dir>`` is produced by ``r_reference.R``; ``<fixture_tables_dir>`` holds
the exported GeneScoreMatrix / MotifMatrix tables of the same ArchR project.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from anndata import AnnData

import epione as epi
from epione.tl._trajectory import _trajectory_core
from epione.pl._trajectory import _trajectory_frames

TRAJ = ["C11", "C10", "C5"]


def read_mat(ref, name):
    dim = [int(v) for v in open(os.path.join(ref, f"{name}.dim")).read().split()]
    a = np.fromfile(os.path.join(ref, f"{name}.f64"), dtype="<f8").reshape(dim[::-1]).T
    rows = cols = None
    if os.path.exists(os.path.join(ref, f"{name}.rows")):
        rows = open(os.path.join(ref, f"{name}.rows")).read().splitlines()
    if os.path.exists(os.path.join(ref, f"{name}.cols")):
        cols = open(os.path.join(ref, f"{name}.cols")).read().splitlines()
    return pd.DataFrame(a, index=rows, columns=cols)


def max_abs(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, (a.shape, b.shape)
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return float("inf")
    return float(np.max(np.abs(a[~na] - b[~na]))) if (~na).any() else 0.0


def load_inputs(ref, tables):
    cells = open(os.path.join(ref, "cells.txt")).read().splitlines()
    clusters = open(os.path.join(ref, "clusters.txt")).read().splitlines()
    lsi = read_mat(ref, "lsi").loc[cells].values
    umap = read_mat(ref, "umap").loc[cells].values
    obs = pd.DataFrame({"Clusters": clusters}, index=cells)

    gs = sio.mmread(os.path.join(tables, "GeneScoreMatrix.mtx")).tocsr()
    gfeat = pd.read_csv(os.path.join(tables, "GeneScoreMatrix_features.tsv"), sep="\t")
    gcells = open(os.path.join(tables, "GeneScoreMatrix_cells.txt")).read().splitlines()
    pos = pd.Series(np.arange(len(gcells)), index=gcells)
    X = gs[:, pos.loc[cells].values].T.tocsr()
    var = gfeat.copy()
    var.index = (gfeat["seqnames"].astype(str) + ":" + gfeat["name"].astype(str)).values
    gsm = AnnData(X=X, obs=obs.copy(), var=var)
    gsm.obsm["X_iterative_lsi"] = lsi
    gsm.obsm["X_umap"] = umap

    mm = {}
    for kind in ("deviations", "z"):
        df = pd.read_csv(os.path.join(tables, f"MotifMatrix_{kind}.tsv.gz"), sep="\t", index_col=0)
        mm[kind] = df[cells].T
    return gsm, mm


def run(ref, tables):
    res = {}
    gsm, mm = load_inputs(ref, tables)
    groups = gsm.obs["Clusters"].astype(str).values
    names = np.asarray(gsm.obs_names)

    # -- addTrajectory (3 cases) -------------------------------------------
    for tag, rep_key, use_all in [("lsi", "X_iterative_lsi", False),
                                  ("umap", "X_umap", False),
                                  ("lsi_useall", "X_iterative_lsi", True)]:
        df, diag = _trajectory_core(gsm.obsm[rep_key], groups, names, TRAJ, use_all=use_all)
        it = read_mat(ref, f"initialTime_{tag}")
        res[f"{tag}.initialTime"] = max_abs(diag["initial_time"].values, it.iloc[:, 0].values)
        res[f"{tag}.initialTime_names"] = float(list(diag["initial_time"].index) == list(it.index))
        res[f"{tag}.matSpline"] = max_abs(diag["spline"], read_mat(ref, f"matSpline_{tag}").values)
        ki = read_mat(ref, f"knnIdx_{tag}").values
        res[f"{tag}.knnIdx_identical"] = float(np.mean(diag["knn_idx"] == ki))
        res[f"{tag}.knnDist"] = max_abs(diag["knn_dist"], read_mat(ref, f"knnDist_{tag}").values)
        rt = pd.read_csv(os.path.join(ref, f"trajectory_{tag}.tsv"), sep="\t", index_col=0)
        res[f"{tag}.cells_identical"] = float(set(rt.index) == set(df.index) and len(rt) == len(df))
        d = df.loc[rt.index]
        for col in ("Distance", "DistanceIdx", "Trajectory"):
            res[f"{tag}.{col}"] = max_abs(d[col].values, rt[col].values)

    # -- public API on the default case --------------------------------------
    epi.tl.add_trajectory(gsm, TRAJ, groupby="Clusters", use_rep="X_iterative_lsi",
                          key_added="Traj_lsi", verbose=False)
    rt = pd.read_csv(os.path.join(ref, "trajectory_lsi.tsv"), sep="\t", index_col=0)
    res["api.Trajectory"] = max_abs(gsm.obs.loc[rt.index, "Traj_lsi"].values, rt["Trajectory"].values)

    # -- getTrajectory ------------------------------------------------------
    trajGSM = epi.tl.get_trajectory(gsm, "Traj_lsi", trajectory_label="Clusters")
    for assay, layer in (("mat", "mat"), ("smooth", "smoothMat")):
        r = read_mat(ref, f"trajGSM_{assay}")
        p = pd.DataFrame(trajGSM.layers[layer].T, index=trajGSM.var_names, columns=trajGSM.obs_names)
        res[f"getTrajectory.GSM.{assay}_names"] = float(list(p.index) == list(r.index)
                                                     and list(p.columns) == list(r.columns))
        res[f"getTrajectory.GSM.{assay}"] = max_abs(p.values, r.values)
    labs = open(os.path.join(ref, "trajGSM_labels.txt")).read().splitlines()
    res["getTrajectory.GSM.labels_identical"] = float(np.mean(np.asarray(trajGSM.obs["label"]) == np.asarray(labs)))

    traj2 = epi.tl.get_trajectory(gsm, "Traj_lsi", group_every=2, smooth_window=6)
    for assay, layer in (("mat", "mat"), ("smooth", "smoothMat")):
        r = read_mat(ref, f"trajGSM_ge2_{assay}")
        res[f"getTrajectory.GSM_ge2.{assay}"] = max_abs(traj2.layers[layer].T, r.values)
        res[f"getTrajectory.GSM_ge2.cols"] = float(list(traj2.obs_names) == list(r.columns))

    mm_all = pd.concat([mm["deviations"].add_prefix("deviations:"), mm["z"].add_prefix("z:")], axis=1)
    gsm.obsm["MotifMatrix"] = mm_all
    trajMM = epi.tl.get_trajectory(gsm, "Traj_lsi", use_rep="MotifMatrix", log2_norm=False, scale_to=None)
    for assay, layer in (("mat", "mat"), ("smooth", "smoothMat")):
        r = read_mat(ref, f"trajMM_{assay}")
        p = pd.DataFrame(trajMM.layers[layer].T, index=trajMM.var_names, columns=trajMM.obs_names).loc[r.index]
        res[f"getTrajectory.MM.{assay}"] = max_abs(p.values, r.values)

    # -- plotTrajectoryHeatmap(returnMatrix = TRUE) -------------------------
    def heat(name, traj, **kw):
        r = read_mat(ref, name)
        p = epi.tl.trajectory_heatmap_matrix(traj, **kw)
        res[f"heatmap.{name}.rows_identical"] = float(list(p.index) == list(r.index))
        res[f"heatmap.{name}"] = max_abs(p.values, r.values) if p.shape == r.shape else float("inf")

    heat("heatGSM", trajGSM, var_cutoff=0.9)
    trajMMz = trajMM[:, [v for v in trajMM.var_names if v.startswith("z:")]].copy()
    heat("heatMM", trajMMz, var_cutoff=0.9)
    heat("heatGSM_opts", trajGSM, var_cutoff=0.75, max_features=1000, limits=(-2, 2), grep_exclude="^chrX:")

    # -- correlateTrajectories ------------------------------------------------
    id2 = pd.read_csv(os.path.join(ref, "jaspar_id2name.tsv"), sep="\t").set_index("id")["name"]
    trajMMn = trajMM.copy()
    kind = [v.split(":", 1)[0] for v in trajMM.var_names]
    mid = [v.split(":", 1)[1] for v in trajMM.var_names]
    trajMMn.var_names = [f"{k}:{id2[m].upper()}_{m}" for k, m in zip(kind, mid)]
    trajMMn.var["name"] = [f"{id2[m].upper()}_{m}" for m in mid]
    trajGSMn = trajGSM.copy()
    cor = epi.tl.correlate_trajectories(trajGSMn, trajMMn, name_key1="name", name_key2="name")
    ra = pd.read_csv(os.path.join(ref, "correlate_all.tsv"), sep="\t")
    pa = cor["allMappings"]
    res["correlate.n_mappings_equal"] = float(len(ra) == len(pa))
    res["correlate.pairs_identical"] = float(np.array_equal(ra[["idx1", "idx2"]].values, pa[["idx1", "idx2"]].values))
    for col in ("Correlation", "VarAssay1", "VarAssay2", "TStat", "Pval", "FDR"):
        res[f"correlate.{col}"] = max_abs(pa[col].values, ra[col].values)
    rpf = pd.read_csv(os.path.join(ref, "correlate_pf.tsv"), sep="\t")
    ppf = cor["correlatedMappings"]
    res["correlate.pf_identical"] = float(
        set(zip(rpf.name1, rpf.name2)) == set(zip(ppf.name1, ppf.name2)))

    # -- plotTrajectory frames ----------------------------------------------
    gsm.obs["Traj_lsi"] = gsm.obs["Traj_lsi"].astype(float)
    cebpb = [v for v in gsm.var_names if v.endswith(":CEBPB")][0]
    for nm, color_by in (("Traj_lsi", None), ("CEBPB", cebpb)):
        _, _, pts, arrow = _trajectory_frames(gsm, "Traj_lsi", color_by=color_by, basis="X_umap")
        ra = pd.read_csv(os.path.join(ref, f"plotTraj_arrow_{nm}.tsv"), sep="\t")
        rp = pd.read_csv(os.path.join(ref, f"plotTraj_points_{nm}.tsv"), sep="\t")
        cols = ["x", "y", "PseudoTime"] + (["value"] if nm == "CEBPB" else [])
        res[f"plotTrajectory.{nm}.arrow"] = max_abs(arrow[cols].values, ra[cols].values) \
            if len(arrow) == len(ra) else float("inf")
        res[f"plotTrajectory.{nm}.points"] = max_abs(pts.values, rp.values) \
            if len(pts) == len(rp) else float("inf")
    return res


GATES = {  # pre-registered in MANIFEST.md
    "abs": 1e-8,
    "exact_suffixes": ("_identical", "_names", "_equal", ".cols"),
}


def passed(key, val):
    if key.endswith(GATES["exact_suffixes"]):
        return val == 1.0
    return val <= GATES["abs"]


if __name__ == "__main__":
    ref, tables = sys.argv[1], sys.argv[2]
    res = run(ref, tables)
    width = max(len(k) for k in res)
    for k, v in res.items():
        print(f"{k:<{width}}  {v:.3e}  {'PASS' if passed(k, v) else 'FAIL'}")
    print("ALL PASS" if all(passed(k, v) for k, v in res.items()) else "SOME FAIL")
    if len(sys.argv) > 3:
        json.dump(res, open(sys.argv[3], "w"), indent=1)
