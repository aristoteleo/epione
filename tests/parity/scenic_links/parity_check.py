"""Parity check of epione.scenic links against scenicplus 1.0 reference outputs.

usage: python parity_check.py [--stages search_space,tf2g,r2g_gbm,r2g_rf]
                              [--n-targets N] [--n-jobs J] [--out results.json]

Reads ``$EPIONE_SCENIC_PARITY_DIR/A/{inputs,scplus_inputs,tf2g,r2g}`` (generated
outside this repository with scenicplus 1.0a2, see MANIFEST.md) and prints / writes the
gate metrics.  ``--n-targets`` restricts tf2g / r2g to a random subset of genes (the
full run is the registered one).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

import epione.scenic as sc


def _lines(p):
    return [x for x in Path(p).read_text().split("\n") if x]


def load(root: Path):
    s = root / "A" / "scplus_inputs"
    X = scipy.io.mmread(s / "scRNA_X.mtx").tocsr().astype(np.float32).toarray()
    genes = _lines(s / "scRNA_X_cols.txt")
    acc = np.load(s / "scATAC_X.npz")["X"]
    regions = _lines(s / "scATAC_X_cols.txt")
    return pd.DataFrame(X, columns=genes), acc, regions


def rel_err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-300)


def check_search_space(root, regions, genes):
    ga = pd.read_table(root / "A" / "inputs" / "genome_annotation.tsv")
    t = time.time()
    out = sc.get_search_space(regions, genes, ga, upstream=(1000, 150000), downstream=(1000, 150000),
                              extend_tss=(10, 10))
    dt = time.time() - t
    ref = pd.read_table(root / "A" / "scplus_inputs" / "search_space.tsv")
    return {"rows": len(out), "ref_rows": len(ref), "identical": bool(out.equals(ref)), "seconds": round(dt, 1)}


def check_tf2g(root, expr, n_jobs, targets):
    tfs = _lines(root / "A" / "tf2g" / "tf_names_allTF.txt")
    t = time.time()
    adj = sc.tf_to_gene(expr, tfs, method="exact", targets=targets, n_jobs=n_jobs,
                        self_link_order="upstream")
    dt = time.time() - t
    ref = pd.read_table(root / "A" / "tf2g" / "tf_to_gene_adj_GBM.tsv", float_precision="round_trip")
    res = {"seconds": round(dt, 1), "rows": len(adj), "ref_rows": len(ref)}
    if targets is not None:
        ref = ref[ref.target.isin(targets) & (ref.TF != ref.target)].reset_index(drop=True)
        adj = adj[adj.TF != adj.target]
        m = adj.merge(ref, on=["TF", "target"], how="outer", indicator=True)
        b = m[m["_merge"] == "both"]
        res.update(same_link_set=bool((m["_merge"] == "both").all()),
                   importance_max_rel=float(rel_err(b.importance_x, b.importance_y).max()),
                   importance_bit_exact=float((b.importance_x == b.importance_y).mean()),
                   rho_max_abs=float(np.nanmax(np.abs(b.rho_x - b.rho_y))),
                   regulation_mismatch=int((b.regulation_x != b.regulation_y).sum()))
        return res
    same_order = len(adj) == len(ref) and bool((adj.TF.values == ref.TF.values).all()
                                               and (adj.target.values == ref.target.values).all())
    m = adj.merge(ref, on=["TF", "target"], how="outer", indicator=True)
    b = m[m["_merge"] == "both"]
    res.update(same_link_set=bool((m["_merge"] == "both").all()), same_row_order=same_order,
               importance_max_rel=float(rel_err(b.importance_x, b.importance_y).max()),
               importance_bit_exact=float((b.importance_x == b.importance_y).mean()),
               rho_max_abs=float(np.nanmax(np.abs(b.rho_x - b.rho_y))),
               regulation_mismatch=int((b.regulation_x != b.regulation_y).sum()))
    return res


def check_r2g(root, expr, acc, regions, kind, n_jobs, targets):
    ss = pd.read_table(root / "A" / "scplus_inputs" / "search_space.tsv")
    t = time.time()
    out = sc.region_to_gene(expr, acc, ss, method="exact", importance=kind, n_jobs=n_jobs,
                            accessibility_var_names=regions, targets=targets)
    dt = time.time() - t
    ref = pd.read_table(root / "A" / "r2g" / f"region_to_gene_adj_{kind}.tsv", float_precision="round_trip")
    if targets is not None:
        ref = ref[ref.target.isin(targets)].reset_index(drop=True)
    same = len(out) == len(ref) and bool((out.target.values == ref.target.values).all()
                                         and (out.region.values == ref.region.values).all())
    res = {"seconds": round(dt, 1), "rows": len(out), "ref_rows": len(ref), "same_rows_and_order": same}
    if same:
        r1, r2 = out.rho.to_numpy(), ref.rho.to_numpy()
        res.update(
            importance_max_rel=float(rel_err(out.importance, ref.importance).max()),
            importance_bit_exact=float((out.importance.values == ref.importance.values).mean()),
            rho_nan_identical=bool(np.array_equal(np.isnan(r1), np.isnan(r2))),
            rho_max_abs=float(np.nanmax(np.abs(r1 - r2))),
            importance_x_rho_max_rel=float(np.nanmax(rel_err(out.importance_x_rho, ref.importance_x_rho))),
            importance_x_abs_rho_max_rel=float(np.nanmax(rel_err(out.importance_x_abs_rho, ref.importance_x_abs_rho))),
            distance_identical=bool((out.Distance.values == ref.Distance.values).all()),
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="search_space,tf2g,r2g_gbm,r2g_rf")
    ap.add_argument("--n-targets", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(os.environ["EPIONE_SCENIC_PARITY_DIR"])
    expr, acc, regions = load(root)
    genes = list(expr.columns)
    targets = None
    if a.n_targets:
        expressed = np.flatnonzero(expr.to_numpy().sum(0) > 0)
        rs = np.random.RandomState(0)
        targets = [genes[i] for i in sorted(rs.choice(expressed, a.n_targets, replace=False))]
    results = {}
    for st in a.stages.split(","):
        if st == "search_space":
            results[st] = check_search_space(root, regions, genes)
        elif st == "tf2g":
            results[st] = check_tf2g(root, expr, a.n_jobs, targets)
        elif st == "r2g_gbm":
            results[st] = check_r2g(root, expr, acc, regions, "GBM", a.n_jobs, targets)
        elif st == "r2g_rf":
            results[st] = check_r2g(root, expr, acc, regions, "RF", a.n_jobs, targets)
        print(st, json.dumps(results[st]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
