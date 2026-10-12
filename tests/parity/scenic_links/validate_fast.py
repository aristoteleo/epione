"""Registered validation of the fast link modes (MANIFEST "Fast modes", gates F1-F7).

usage: PYTHONHASHSEED=0 python validate_fast.py <parity_dir> <exact_seed667_dir> <out.json>
       [--n-jobs N] [--f7-targets K]

<parity_dir> is $EPIONE_SCENIC_PARITY_DIR (A/<stage>/ reference dumps from scenicplus 1.0a2,
block4/ extra dumps).  <exact_seed667_dir> holds a second exact run with seed 667
(``tf_to_gene_adj_GBM_seed667.tsv``, ``region_to_gene_adj_GBM_seed667.tsv``, ``timing.json``):
the exact method's own seed-to-seed spread, against which the gates are set.  Both are
generated outside this repository.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import spearmanr

import epione.scenic as sc


def _lines(p):
    return [x for x in Path(p).read_text().split("\n") if x]


def per_target(a: pd.DataFrame, b: pd.DataFrame, key: str, k: int, targets=None):
    """Median per-target Spearman (union of links, missing = 0) and top-k Jaccard."""
    if targets is not None:
        a, b = a[a.target.isin(targets)], b[b.target.isin(targets)]
    bg = dict(tuple(b.groupby("target")))
    rho, jac = [], []
    for t, ga in a.groupby("target"):
        gb = bg.get(t)
        if gb is None:
            continue
        u = sorted(set(ga[key]) | set(gb[key]))
        if len(u) < 3:
            continue
        va = ga.set_index(key).importance.reindex(u).fillna(0).to_numpy()
        vb = gb.set_index(key).importance.reindex(u).fillna(0).to_numpy()
        rho.append(spearmanr(va, vb)[0])
        ta, tb = set(ga.nlargest(k, "importance")[key]), set(gb.nlargest(k, "importance")[key])
        jac.append(len(ta & tb) / len(ta | tb))
    return {"median_spearman": float(np.nanmedian(rho)), f"median_top{k}_jaccard": float(np.median(jac)),
            "n_targets": len(rho)}


def egrn_agreement(ours: pd.DataFrame, ref: pd.DataFrame):
    """TF-set Jaccard and median per-eRegulon target-gene Jaccard (eRegulons in both)."""
    ta, tb = set(ours.TF), set(ref.TF)
    ga = ours.groupby("eRegulon_name").Gene.apply(set)
    gb = ref.groupby("eRegulon_name").Gene.apply(set)
    both = sorted(set(ga.index) & set(gb.index))
    j = [len(ga[e] & gb[e]) / len(ga[e] | gb[e]) for e in both]
    return {"tf_set_jaccard": len(ta & tb) / len(ta | tb), "median_target_gene_jaccard":
            float(np.median(j)) if j else float("nan"), "eRegulons_ours": int(ours.eRegulon_name.nunique()),
            "eRegulons_ref": int(ref.eRegulon_name.nunique()), "eRegulons_in_both": len(both),
            "tfs_ours": len(ta), "tfs_ref": len(tb)}


def load_cistromes(A: Path, kind: str):
    import anndata as ad
    from scipy import sparse

    base = A / "scplus_inputs" / f"cistromes_{kind}"
    X = sparse.csr_matrix(scipy.io.mmread(str(base) + ".mtx")).astype(bool)
    var = pd.read_table(str(base) + "_var.tsv", index_col=0)
    return ad.AnnData(X=X, obs=pd.DataFrame(index=_lines(str(base) + "_rows.txt")),
                      var=var.loc[_lines(str(base) + "_cols.txt")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parity_dir")
    ap.add_argument("seed667_dir")
    ap.add_argument("out")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--f7-targets", type=int, default=500)
    a = ap.parse_args()
    P, S7 = Path(a.parity_dir), Path(a.seed667_dir)
    A = P / "A"
    s = A / "scplus_inputs"
    res = {"n_jobs": a.n_jobs}

    X = scipy.io.mmread(s / "scRNA_X.mtx").tocsr().astype(np.float32).toarray()
    genes = _lines(s / "scRNA_X_cols.txt")
    expr = pd.DataFrame(X, columns=genes)
    tfs = _lines(A / "tf2g" / "tf_names_allTF.txt")
    acc = np.load(s / "scATAC_X.npz")["X"]
    regions = _lines(s / "scATAC_X_cols.txt")
    ss = pd.read_table(s / "search_space.tsv")

    ref_tf = pd.read_table(A / "tf2g" / "tf_to_gene_adj_GBM.tsv")
    ref_r2g = pd.read_table(A / "r2g" / "region_to_gene_adj_GBM.tsv")
    s7_tf = pd.read_table(S7 / "tf_to_gene_adj_GBM_seed667.tsv")
    s7_r2g = pd.read_table(S7 / "region_to_gene_adj_GBM_seed667.tsv")
    res["exact_timing_seed667"] = json.loads((S7 / "timing.json").read_text())

    # --- fast runs (full data)
    t = time.time()
    fast_tf = sc.tf_to_gene(expr, tfs, method="fast", n_jobs=a.n_jobs)
    res["fast_tf2g_seconds"] = time.time() - t
    print("fast tf2g", res["fast_tf2g_seconds"], flush=True)
    t = time.time()
    fast_r2g = sc.region_to_gene(expr, acc, ss, method="fast", n_jobs=a.n_jobs, accessibility_var_names=regions)
    res["fast_r2g_gbm_seconds"] = time.time() - t
    print("fast r2g", res["fast_r2g_gbm_seconds"], flush=True)
    t = time.time()
    corr_r2g = sc.region_to_gene(expr, acc, ss, method="corr", n_jobs=a.n_jobs, accessibility_var_names=regions)
    res["corr_r2g_seconds"] = time.time() - t
    print("corr r2g", res["corr_r2g_seconds"], flush=True)

    # --- F1-F4
    nself = lambda d: d[d.TF != d.target]
    big = set(ref_r2g.groupby("target").size().loc[lambda x: x >= 10].index)
    res["tf2g_exact_seed_spread"] = per_target(nself(s7_tf), nself(ref_tf), "TF", 50)
    res["tf2g_fast"] = per_target(nself(fast_tf), nself(ref_tf), "TF", 50)
    res["r2g_exact_seed_spread"] = per_target(s7_r2g, ref_r2g, "region", 10, big)
    res["r2g_fast"] = per_target(fast_r2g, ref_r2g, "region", 10, big)
    res["r2g_corr"] = per_target(corr_r2g, ref_r2g, "region", 10, big)
    print({k: res[k] for k in ("tf2g_exact_seed_spread", "tf2g_fast", "r2g_exact_seed_spread",
                               "r2g_fast", "r2g_corr")}, flush=True)

    # --- F7: thread invariance of the fast path (subset of targets)
    rs = np.random.RandomState(0)
    expressed = np.flatnonzero(X.sum(0) > 0)
    sub = [genes[i] for i in sorted(rs.choice(expressed, a.f7_targets, replace=False))]
    t1 = sc.tf_to_gene(expr, tfs, method="fast", targets=sub, n_jobs=1)
    tn = sc.tf_to_gene(expr, tfs, method="fast", targets=sub, n_jobs=a.n_jobs)
    r1 = sc.region_to_gene(expr, acc, ss, method="fast", targets=sub, n_jobs=1, accessibility_var_names=regions)
    rn = sc.region_to_gene(expr, acc, ss, method="fast", targets=sub, n_jobs=a.n_jobs, accessibility_var_names=regions)
    res["f7_thread_invariance"] = {"targets": len(sub), "tf2g_identical": bool(t1.equals(tn)),
                                   "r2g_identical": bool(r1.equals(rn))}
    # and the subset result equals the corresponding rows of the full run
    full_sub = fast_tf[fast_tf.target.isin(sub) & (fast_tf.TF != fast_tf.target)]
    part = t1[t1.TF != t1.target]
    m = full_sub.merge(part, on=["TF", "target"], how="outer", indicator=True)
    res["f7_thread_invariance"]["tf2g_subset_equals_full"] = bool(
        (m["_merge"] == "both").all() and (m.importance_x == m.importance_y).all())

    del acc, X  # the eGRN step does not need the matrices

    # --- F5-F6: downstream eGRN
    for kind in ("direct", "extended"):
        cis = load_cistromes(A, kind)
        rank = pd.read_parquet(P / "block4" / f"maxrank_{kind}.parquet")
        ref = pd.read_table(A / "egrn" / f"eRegulons_{kind}.tsv")
        for label, tf2g, r2g in (("fast", fast_tf, fast_r2g), ("exact_seed667", s7_tf, s7_r2g),
                                 ("fast_tf2g_corr_r2g", fast_tf, corr_r2g)):
            t = time.time()
            eg = sc.build_egrn(tf2g, r2g, cis, is_extended=(kind == "extended"), tf_region_rank=rank)
            res[f"egrn_{kind}_{label}"] = {**egrn_agreement(eg, ref), "seconds": time.time() - t}
            print(kind, label, res[f"egrn_{kind}_{label}"], flush=True)

    # --- gates
    g = {}
    g["F1"] = res["tf2g_fast"]["median_spearman"] >= min(0.8, res["tf2g_exact_seed_spread"]["median_spearman"])
    g["F2"] = res["tf2g_fast"]["median_top50_jaccard"] >= min(0.6, res["tf2g_exact_seed_spread"]["median_top50_jaccard"])
    g["F3"] = res["r2g_fast"]["median_spearman"] >= min(0.8, res["r2g_exact_seed_spread"]["median_spearman"])
    g["F4"] = res["r2g_fast"]["median_top10_jaccard"] >= min(0.6, res["r2g_exact_seed_spread"]["median_top10_jaccard"])
    for kind in ("direct", "extended"):
        f, e = res[f"egrn_{kind}_fast"], res[f"egrn_{kind}_exact_seed667"]
        g[f"F5_{kind}"] = f["tf_set_jaccard"] >= min(0.8, e["tf_set_jaccard"])
        g[f"F6_{kind}"] = f["median_target_gene_jaccard"] >= min(0.6, e["median_target_gene_jaccard"])
    g["F7"] = all(res["f7_thread_invariance"][k] for k in ("tf2g_identical", "r2g_identical",
                                                          "tf2g_subset_equals_full"))
    res["gates"] = {k: bool(v) for k, v in g.items()}
    print(res["gates"], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
