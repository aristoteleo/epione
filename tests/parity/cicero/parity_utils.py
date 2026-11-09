"""Addendum gates 19-24: exported Cicero utilities vs r_utils_driver.R dumps (fixture F1).
Usage: python parity_utils.py <ref_F1_utils_dir> <out.json>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import epione as epi


def rmat(d: Path, pre: str):
    rows = (d / f"{pre}_rows.txt").read_text().split("\n")
    cols = (d / f"{pre}_cols.txt").read_text().split("\n")
    rows = [r for r in rows if r]
    cols = [c for c in cols if c]
    m = sp.csc_matrix((np.fromfile(d / f"{pre}_x.f64", "<f8"), np.fromfile(d / f"{pre}_i.i32", "<i4"),
                       np.fromfile(d / f"{pre}_p.f64", "<f8").astype(np.int64)), shape=(len(rows), len(cols)))
    return rows, cols, m


def same_matrix(py_cells_by_feat, r_feat_by_cells) -> bool:
    a = sp.csr_matrix(py_cells_by_feat).T.tocsr()
    b = sp.csr_matrix(r_feat_by_cells)
    return a.shape == b.shape and (abs(a - b)).max() == 0 if a.shape == b.shape else False


def main(argv=None):
    argv = argv or sys.argv[1:]
    d, out = Path(argv[0]), argv[1]
    G = {}
    cd = pd.read_csv(d / "cicero_data.tsv", sep="\t")
    # 19 make_atac_cds
    cds = epi.tl.make_atac_cds(cd, binarize=True)
    rows, cols, m = rmat(d, "atac")
    fd = pd.read_csv(d / "atac_fdata.tsv", sep="\t", dtype={"chr": str})
    pdd = pd.read_csv(d / "atac_pdata.tsv", sep="\t")
    cds_c = epi.tl.make_atac_cds(cd, binarize=False)
    rc, cc, mc = rmat(d, "atac_counts")
    g19 = {"peak_order": list(cds.var_names) == rows, "cell_order": list(cds.obs_names) == cols,
           "matrix_binary": same_matrix(cds.X, m), "matrix_counts": same_matrix(cds_c.X, mc),
           "var_coords": bool((cds.var["chr"].astype(str).to_numpy() == fd["chr"].to_numpy()).all()
                              and (cds.var["bp1"].to_numpy() == fd["bp1"].to_numpy()).all()
                              and (cds.var["bp2"].to_numpy() == fd["bp2"].to_numpy()).all()),
           "num_cells_expressed": bool((cds.var["num_cells_expressed"].to_numpy() == fd["num_cells_expressed"].to_numpy()).all()),
           "num_genes_expressed": bool((cds.obs["num_genes_expressed"].to_numpy() == pdd["num_genes_expressed"].to_numpy()).all())}
    g19["pass"] = all(g19.values())
    G["19"] = g19
    # 20 aggregate_nearby_peaks
    g20 = {}
    for dist in (1000, 10000):
        a = epi.tl.aggregate_nearby_peaks(cds, distance=dist)
        r, c, mm = rmat(d, f"anp_{dist}")
        g20[str(dist)] = {"names": list(a.var_names) == r, "cells": list(a.obs_names) == c, "matrix": same_matrix(a.X, mm),
                          "n_bins": int(a.n_vars)}
    g20["pass"] = all(v["names"] and v["cells"] and v["matrix"] for v in g20.values() if isinstance(v, dict))
    G["20"] = g20
    # 21 aggregate_by_cell_bin
    cell_data = pd.read_csv(d / "cell_data.tsv", sep="\t").set_index("cell")
    cds2 = cds.copy()
    cds2.obs["timepoint"] = cell_data.loc[cds2.obs_names, "timepoint"].to_numpy()
    ab = epi.tl.aggregate_by_cell_bin(cds2, "timepoint")
    rows = [x for x in (d / "abc_rows.txt").read_text().split("\n") if x]
    cols = [x for x in (d / "abc_cols.txt").read_text().split("\n") if x]
    M = np.fromfile(d / "abc.f64", "<f8").reshape(len(rows), len(cols), order="F")
    pa = pd.read_csv(d / "abc_pdata.tsv", sep="\t")
    err = float(np.abs(ab.X.T.toarray() - M).max()) if (list(ab.var_names) == rows and list(ab.obs_names) == cols) else float("inf")
    sf_err = float(np.abs(ab.obs["Size_Factor"].to_numpy() - pa["Size_Factor"].to_numpy()).max())
    G["21"] = {"names_rows": list(ab.var_names) == rows, "names_cols": list(ab.obs_names) == cols, "max_abs_err": err,
               "size_factor_max_abs_err": sf_err,
               "num_genes_expressed": bool((ab.obs["num_genes_expressed"].to_numpy() == pa["num_genes_expressed"].to_numpy()).all()),
               "n": bool((ab.obs["n"].to_numpy() == pa["n"].to_numpy()).all())}
    G["21"]["pass"] = bool(err <= 1e-8 and sf_err <= 1e-8 and G["21"]["num_genes_expressed"])
    # 22 compare_connections
    c1 = pd.read_csv(d / "cc_conns1.tsv", sep="\t")
    alt = pd.read_csv(d / "cc_alt.tsv", sep="\t")
    g22 = {}
    for gap in (0, 1000):
        r = np.array([x == "TRUE" for x in (d / f"cc_maxgap{gap}.txt").read_text().split()])
        p = epi.tl.compare_connections(c1, alt, maxgap=gap)
        g22[str(gap)] = {"n": int(len(r)), "n_true_r": int(r.sum()), "n_true_py": int(p.sum()), "identical": bool((r == p).all())}
    g22["pass"] = all(v["identical"] for v in g22.values() if isinstance(v, dict))
    G["22"] = g22
    # 23 find_overlapping_coordinates
    pk = list(cds.var_names)
    q1 = "chr18:10,100-1246234"
    qs = ["chr18:10,100-50000", "chr18:2,000,000-2,100,000", "chr18:99999999-100000000", "chr18_157883_158536"]
    g23 = {}
    for gap in (0, 500):
        rs = [x for x in (d / f"foc_single_{gap}.txt").read_text().split("\n") if x]
        rm = [x for x in (d / f"foc_multi_{gap}.txt").read_text().split("\n") if x]
        ps = epi.tl.find_overlapping_coordinates(pk, q1, maxgap=gap)
        pm = ["NA" if x is None else x for x in epi.tl.find_overlapping_coordinates(pk, qs, maxgap=gap)]
        g23[str(gap)] = {"single": ps == rs, "multi": pm == rm, "n_single": len(rs), "n_multi": len(rm)}
    g23["pass"] = all(v["single"] and v["multi"] for v in g23.values() if isinstance(v, dict))
    G["23"] = g23
    # 24 df_for_coords
    rdf = pd.read_csv(d / "dfc.tsv", sep="\t", dtype={"chr": str, "Peak": str})
    pdf_ = epi.tl.df_for_coords(pk + ["chr1:2,039-30,239", "chrX:28884:101293", "chr18-8,575,097-8,839,855"])
    G["24"] = {"pass": bool((pdf_[["chr", "Peak"]].to_numpy() == rdf[["chr", "Peak"]].to_numpy()).all()
                            and (pdf_[["bp1", "bp2"]].to_numpy() == rdf[["bp1", "bp2"]].to_numpy()).all())}
    res = {"gates": G, "all_pass": all(v["pass"] for v in G.values())}
    txt = json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    Path(out).write_text(txt)
    print(txt)
    return res


if __name__ == "__main__":
    main()
