"""SCENIC+ block 4 parity: eGRN, AUCell, RSS against the SCENIC+ 1.0a2 reference.

usage: PYTHONHASHSEED=0 python parity_scenic_egrn.py <parity_dir> [out.json] [--ranking-db FILE]

<parity_dir> is $EPIONE_SCENIC_PARITY_DIR (A/<stage>/ reference dumps, block4/ extra dumps).
Must run with PYTHONHASHSEED=0 for the row-order gates (MANIFEST rows 4-5).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy import sparse


def _lines(p):
    return Path(p).read_text().split("\n")[:-1]


def load_cistromes(A: Path, kind: str):
    import anndata as ad

    base = A / "scplus_inputs" / f"cistromes_{kind}"
    X = sparse.csr_matrix(scipy.io.mmread(str(base) + ".mtx")).astype(bool)
    var = pd.read_table(str(base) + "_var.tsv", index_col=0)
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=_lines(str(base) + "_rows.txt")),
                   var=var.loc[_lines(str(base) + "_cols.txt")])
    return a


def load_dense(stem: Path):
    return pd.DataFrame(np.load(str(stem) + ".npy"), index=_lines(str(stem) + "_rows.txt"),
                        columns=_lines(str(stem) + "_cols.txt"))


def compare_tables(ours: pd.DataFrame, ref: pd.DataFrame):
    """(set_equal, order_equal, triplet_equal, details) on tables read back from TSV."""
    cols = [c for c in ref.columns if c != "triplet_rank"]
    a, b = ours[cols].reset_index(drop=True), ref[cols].reset_index(drop=True)
    key = cols
    sa = a.sort_values(key, kind="stable").reset_index(drop=True)
    sb = b.sort_values(key, kind="stable").reset_index(drop=True)
    set_equal = len(a) == len(b) and sa.equals(sb)
    order_equal = len(a) == len(b) and a.equals(b)
    trip = None
    if "triplet_rank" in ref.columns and "triplet_rank" in ours.columns:
        trip = order_equal and np.array_equal(ours["triplet_rank"].to_numpy(), ref["triplet_rank"].to_numpy())
    return set_equal, order_equal, trip, {"rows_ours": len(a), "rows_ref": len(b),
                                          "eRegulons_ours": int(ours["eRegulon_name"].nunique()),
                                          "eRegulons_ref": int(ref["eRegulon_name"].nunique())}


def intermediates(tf2g, r2g, cis, X4: Path, kind: str):
    """MANIFEST rows 1-2: per-gene thresholds, eModules and per-eModule GSEA."""
    from epione.scenic import _egrn as E

    out = {}
    ref = np.load(X4 / f"thresholds_{kind}.npz")
    keys = sorted(ref.files)
    methods = [(f"{q} quantile", "quantile", q) for q in (0.85, 0.90, 0.95)]
    methods += [(f"Top {n}", "top", n) for n in (5, 10, 15)] + [("BASC", "basc", None)]
    rho = r2g["rho"].to_numpy()
    errs, i = [], 0
    for sel in (rho < -0.05, rho > 0.05):
        rows = np.flatnonzero(sel)
        _, g = np.unique(r2g["target"].to_numpy()[rows], return_inverse=True)
        for _, _, thr in E._thresholds(g, r2g["importance"].to_numpy()[rows], methods):
            errs.append(float(np.max(np.abs(ref[keys[i]] - thr))))
            i += 1
    out[f"thresholds_{kind}_max_abs_err"] = max(errs)
    diag = {}
    E.build_egrn(tf2g, r2g, cis, is_extended=(kind == "extended"), _diagnostics=diag)
    em = pd.read_table(X4 / f"emodules_{kind}.tsv")
    md5 = [hashlib.md5("\n".join(f"{r}\t{t}" for r, t in zip(regs, tgts)).encode()).hexdigest()
           for _, _, regs, tgts in diag["emodules"]]
    ours_em = pd.DataFrame({"TF": [m[0] for m in diag["emodules"]], "context": [m[1] for m in diag["emodules"]],
                            "n_rows": [len(m[2]) for m in diag["emodules"]], "md5": md5})
    out[f"emodules_{kind}_equal"] = bool(len(ours_em) == len(em) and
                                         (ours_em[["TF", "context", "n_rows", "md5"]].to_numpy()
                                          == em[["TF", "context", "n_rows", "md5"]].to_numpy()).all())
    gs = pd.read_table(X4 / f"gsea_{kind}.tsv")
    og = pd.DataFrame([(i, s, nes, p, len(le), hashlib.md5("\n".join(le).encode()).hexdigest())
                       for i, s, nes, p, le in diag["gsea"]], columns=["idx", "tf2g", "NES", "pval", "n_le", "le_md5"])
    same_shape = len(og) == len(gs) and (og[["idx", "tf2g"]].to_numpy() == gs[["idx", "tf2g"]].to_numpy()).all()
    exact = X4 / f"gsea_{kind}_nes_pval.npy"  # exact float bits (the TSV keeps 16 digits)
    ref_np = np.load(exact) if exact.exists() else gs[["NES", "pval"]].to_numpy()
    if same_shape:
        nes_err = float(np.nanmax(np.abs(og["NES"].to_numpy() - ref_np[:, 0]))) if len(og) else 0.0
        nan_same = bool((np.isnan(og["NES"].to_numpy()) == np.isnan(ref_np[:, 0])).all())
        p_err = float(np.nanmax(np.abs(og["pval"].to_numpy() - ref_np[:, 1]))) if len(og) else 0.0
        le_equal = bool((og[["n_le", "le_md5"]].to_numpy() == gs[["n_le", "le_md5"]].to_numpy()).all())
    else:
        nes_err = p_err = float("nan")
        nan_same = le_equal = False
    out[f"gsea_{kind}"] = {"calls_ours": len(og), "calls_ref": len(gs), "same_calls": bool(same_shape),
                           "NES_max_abs_err": nes_err, "NES_nan_pattern_equal": nan_same,
                           "pval_max_abs_err": p_err, "leading_edge_equal": le_equal}
    return out


def run(P: Path, ranking_db: str | None = None):
    import epione.scenic as sc

    A = P / "A"
    X4 = P / "block4"
    res, timing = {}, {}
    tf2g = pd.read_table(A / "tf2g" / "tf_to_gene_adj_GBM.tsv")
    r2g = pd.read_table(A / "r2g" / "region_to_gene_adj_GBM.tsv")
    tables = {}
    for kind in ("direct", "extended"):
        cis = load_cistromes(A, kind)
        rank = None
        if ranking_db:
            t0 = time.time()
            motifs = {tf: m.split(",") for tf, m in zip(cis.var_names, cis.var["motifs"])}
            rank = sc.tf_region_max_rank(list(cis.obs_names), motifs, ranking_db)
            timing[f"maxrank_{kind}"] = time.time() - t0
            if (X4 / f"maxrank_{kind}.parquet").exists():
                ref_rank = pd.read_parquet(X4 / f"maxrank_{kind}.parquet")
                r2 = rank.loc[ref_rank.index, ref_rank.columns]
                res[f"maxrank_{kind}_equal"] = bool(np.array_equal(r2.to_numpy(), ref_rank.to_numpy())
                                                    and len(rank) == len(ref_rank))
        elif (X4 / f"maxrank_{kind}.parquet").exists():
            rank = pd.read_parquet(X4 / f"maxrank_{kind}.parquet")
        t0 = time.time()
        ours = sc.build_egrn(tf2g, r2g, cis, is_extended=(kind == "extended"), tf_region_rank=rank)
        timing[f"egrn_{kind}"] = time.time() - t0
        if (X4 / f"thresholds_{kind}.npz").exists():
            res.update(intermediates(tf2g, r2g, cis, X4, kind))
        out = Path(tempfile.mkdtemp(prefix="scenic_egrn_"))
        ours.to_csv(out / f"eRegulons_{kind}.tsv", sep="\t", header=True, index=False)
        ours_rt = pd.read_table(out / f"eRegulons_{kind}.tsv")
        ref = pd.read_table(A / "egrn" / f"eRegulons_{kind}.tsv")
        s, o, t, d = compare_tables(ours_rt, ref)
        res[f"egrn_{kind}"] = {"set_equal": bool(s), "order_equal": bool(o), "triplet_equal": t, **d}
        tables[kind] = ref  # AUCell is gated on the reference eRegulons (isolates the stage)

    # AUCell on the reference eRegulons
    gex = sparse.csr_matrix(scipy.io.mmread(str(A / "scplus_inputs" / "scRNA_X.mtx"))).astype(np.float32)
    cells = _lines(A / "scplus_inputs" / "scRNA_X_rows.txt")
    genes = _lines(A / "scplus_inputs" / "scRNA_X_cols.txt")
    acc = np.load(A / "scplus_inputs" / "scATAC_X.npz")["X"]
    acc_cells = _lines(A / "scplus_inputs" / "scATAC_X_rows.txt")
    regions = _lines(A / "scplus_inputs" / "scATAC_X_cols.txt")
    assert cells == acc_cells
    t0 = time.time()
    gr = sc.rank_cells(gex, cells, genes, n_jobs=int(os.environ.get("N_JOBS", "8")))
    timing["rank_gex"] = time.time() - t0
    t0 = time.time()
    ar = sc.rank_cells(acc, cells, regions, n_jobs=int(os.environ.get("N_JOBS", "8")), dtype=np.int32)
    timing["rank_acc"] = time.time() - t0
    aucs = {}
    for kind in ("direct", "extended"):
        t0 = time.time()
        auc = sc.score_eregulons(tables[kind], gr, ar)
        timing[f"aucell_{kind}"] = time.time() - t0
        for mod in ("Gene_based", "Region_based"):
            ref = load_dense(A / "aucell_rss" / f"AUC_{kind}_{mod}")
            ours = auc[mod]
            labels = list(ours.index) == list(ref.index) and list(ours.columns) == list(ref.columns)
            err = float(np.max(np.abs(ours.reindex(index=ref.index, columns=ref.columns).to_numpy() - ref.to_numpy())))
            res[f"auc_{kind}_{mod}"] = {"labels_equal": labels, "max_abs_err": err}
            aucs[(kind, mod)] = ref
    # RSS on the reference AUCs
    ct = pd.read_table(A / "aucell_rss" / "cell_type.tsv", index_col=0).iloc[:, 0]
    t0 = time.time()
    rss = pd.concat([sc.regulon_specificity_scores(aucs[(k, "Gene_based")], ct) for k in ("direct", "extended")], axis=1)
    timing["rss"] = time.time() - t0
    ref = load_dense(A / "aucell_rss" / "rss_gene_based")
    res["rss"] = {"labels_equal": list(rss.index) == list(ref.index) and list(rss.columns) == list(ref.columns),
                  "max_abs_err": float(np.max(np.abs(rss.reindex(index=ref.index, columns=ref.columns).to_numpy()
                                                     - ref.to_numpy())))}
    res["timing_s"] = {k: round(v, 2) for k, v in timing.items()}
    return res


if __name__ == "__main__":
    assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    db = None
    if "--ranking-db" in sys.argv:
        db = sys.argv[sys.argv.index("--ranking-db") + 1]
        args = [a for a in args if a != db]
    r = run(Path(args[0]), db)
    s = json.dumps(r, indent=1)
    print(s)
    if len(args) > 1:
        Path(args[1]).write_text(s)
