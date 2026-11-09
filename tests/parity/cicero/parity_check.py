"""Parity harness: epione.tl Cicero port vs the R reference dumps of r_reference_driver.R.

Usage:
    python parity_check.py <fixture_dir> <ref_dir> <out.json> [--gates 1,2,...] [--threads N]

Gate numbers and thresholds are those pre-registered in manifest.yaml / MANIFEST.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from epione.tl import _cicero as C
from epione.tl._glasso import glasso
from epione.tl._rrng import RRandom

THRESH = {3: 1e-8, 4: 1e-8, 5: 1e-6, 6: 1e-6, 9: 1e-8, 10: 0.9999, 11: 0.999, 12: 1e-8, 14: 1e-8, 15: 1e-8,
          16: 0.95, 17: 0.02}


# ---------------------------------------------------------------- IO
def read_fixture(d: Path) -> tuple[AnnData, pd.DataFrame, pd.DataFrame]:
    sh = [int(x) for x in (d / "shape.txt").read_text().split()]
    i = np.fromfile(d / "x.i.bin", dtype="<i4", count=sh[2])
    p = np.fromfile(d / "x.p.bin", dtype="<f8", count=sh[1] + 1).astype(np.int64)
    x = np.fromfile(d / "x.x.bin", dtype="<f8", count=sh[2])
    m = sp.csc_matrix((x, i, p), shape=(sh[0], sh[1]))           # peaks x cells
    peaks = (d / "peaks.txt").read_text().split("\n")[: sh[0]]
    cells = (d / "cells.txt").read_text().split("\n")[: sh[1]]
    ad = AnnData(X=m.T.tocsr(), obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=peaks))
    co = pd.read_csv(d / "coords.csv", index_col=0, dtype={0: str})
    co.index = co.index.astype(str)
    g = pd.read_csv(d / "genome.tsv", sep="\t", header=None)
    return ad, co.loc[cells], g


def rd(ref: Path, name, dtype="<f8"):
    return np.fromfile(ref / name, dtype=dtype)


def read_ref_cicero_x(ref: Path):
    sh = [int(x) for x in (ref / "cicero_x.shape.txt").read_text().split()]
    m = sp.csc_matrix((rd(ref, "cicero_x.x.f64"), rd(ref, "cicero_x.i.i32", "<i4"),
                       rd(ref, "cicero_x.p.f64").astype(np.int64)), shape=(sh[0], sh[1]))
    return m.T.tocsr()                                            # bins x peaks


def read_ref_conns(ref: Path) -> pd.DataFrame:
    p1 = (ref / "conns_peak1.txt").read_text().split("\n")
    p2 = (ref / "conns_peak2.txt").read_text().split("\n")
    co = rd(ref, "conns_coaccess.f64")
    n = len(co)
    return pd.DataFrame({"Peak1": p1[:n], "Peak2": p2[:n], "coaccess": co})


# ---------------------------------------------------------------- comparisons
def compare_conns(py: pd.DataFrame, r: pd.DataFrame) -> dict:
    a = pd.DataFrame({"Peak1": py["Peak1"].astype(str).to_numpy(), "Peak2": py["Peak2"].astype(str).to_numpy(),
                      "py": py["coaccess"].to_numpy()})
    b = pd.DataFrame({"Peak1": r["Peak1"].to_numpy(), "Peak2": r["Peak2"].to_numpy(), "r": r["coaccess"].to_numpy()})
    m = a.merge(b, on=["Peak1", "Peak2"], how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    na_py, na_r = both["py"].isna().to_numpy(), both["r"].isna().to_numpy()
    ok = ~na_py & ~na_r
    d = np.abs(both["py"].to_numpy()[ok] - both["r"].to_numpy()[ok])
    from scipy.stats import pearsonr, spearmanr
    x, y = both["py"].to_numpy()[ok], both["r"].to_numpy()[ok]
    s_py = set(map(tuple, a.loc[a["py"] > 0.25, ["Peak1", "Peak2"]].to_numpy()))
    s_r = set(map(tuple, b.loc[b["r"] > 0.25, ["Peak1", "Peak2"]].to_numpy()))
    jac = len(s_py & s_r) / max(1, len(s_py | s_r))
    same_order = (len(a) == len(b)) and bool((a["Peak1"].to_numpy() == b["Peak1"].to_numpy()).all()
                                             and (a["Peak2"].to_numpy() == b["Peak2"].to_numpy()).all())
    return {
        "n_rows_py": int(len(a)), "n_rows_r": int(len(b)),
        "only_py": int((m["_merge"] == "left_only").sum()), "only_r": int((m["_merge"] == "right_only").sum()),
        "na_py": int(a["py"].isna().sum()), "na_r": int(b["r"].isna().sum()),
        "na_pattern_mismatch": int((na_py != na_r).sum()),
        "max_abs_err": float(d.max()) if d.size else 0.0,
        "n_exact_equal": int((d == 0).sum()), "n_matched": int(ok.sum()),
        "pearson": float(pearsonr(x, y)[0]) if len(x) > 2 else float("nan"),
        "spearman": float(spearmanr(x, y)[0]) if len(x) > 2 else float("nan"),
        "jaccard_gt_025": float(jac), "n_gt_025_py": len(s_py), "n_gt_025_r": len(s_r),
        "same_row_order": same_order,
    }


def celloracle_format(conns: pd.DataFrame) -> dict:
    co = C.cicero_connections_to_celloracle(conns)
    pat = r"^[^_]+_\d+_\d+$"
    ok_cols = list(co.columns) == ["Peak1", "Peak2", "coaccess"]
    ok_ids = bool(co["Peak1"].str.match(pat).all() and co["Peak2"].str.match(pat).all())
    pairs = set(zip(co["Peak1"], co["Peak2"]))
    sym = all((b, a) in pairs for a, b in list(pairs)[:200000])
    return {"columns_ok": ok_cols, "ids_chr_start_end": ok_ids, "both_orientations": sym,
            "pass": bool(ok_cols and ok_ids and sym)}


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture")
    ap.add_argument("ref")
    ap.add_argument("out")
    ap.add_argument("--gates", default="all")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-conns", default=None)
    a = ap.parse_args(argv)
    fx, ref = Path(a.fixture), Path(a.ref)
    gates = None if a.gates == "all" else {int(g) for g in a.gates.split(",")}
    want = lambda g: gates is None or g in gates
    res: dict = {"fixture": str(fx), "ref": str(ref), "seed": a.seed, "timings_py": {}, "gates": {}}
    T = res["timings_py"]
    G = res["gates"]

    ad, coords, genome = read_fixture(fx)
    res["shape"] = list(ad.shape)

    # gate 4: glasso unit (probe windows + synthetic edge cases)
    if want(4):
        out = {}
        for tag, meta in (("probe", "probes.tsv"), ("syn", "synthetic.tsv")):
            if not (ref / meta).exists():
                continue
            tab = pd.read_csv(ref / meta, sep="\t")
            S_all, R_all = rd(ref, f"{tag}_S.f64"), rd(ref, f"{tag}_rho.f64")
            W_all, WI_all = rd(ref, f"{tag}_w.f64"), rd(ref, f"{tag}_wi.f64")
            ew = ewi = 0.0
            zp = 0
            nit = 0
            for _, row in tab.iterrows():
                n, o = int(row["n"]), int(row["offset"])
                sl = slice(o, o + n * n)
                S = S_all[sl].reshape(n, n, order="F")
                R = R_all[sl].reshape(n, n, order="F")
                g = glasso(S, R)
                W = W_all[sl].reshape(n, n, order="F")
                WI = WI_all[sl].reshape(n, n, order="F")
                ew = max(ew, float(np.abs(g.w - W).max()))
                ewi = max(ewi, float(np.abs(g.wi - WI).max()))
                zp += int(((g.wi == 0) != (WI == 0)).sum())
                nit += int(g.niter != int(row["niter"]))
            out[tag] = {"n": int(len(tab)), "max_abs_err_w": ew, "max_abs_err_wi": ewi,
                        "zero_pattern_mismatch": zp, "niter_mismatch": nit}
        G["4"] = {**out, "pass": all(v["max_abs_err_w"] <= 1e-8 and v["max_abs_err_wi"] <= 1e-8
                                     and v["zero_pattern_mismatch"] == 0 for v in out.values())}

    # gate 2: kNN
    if want(2) and (ref / "knn_index.i32").exists():
        k1 = 49
        rk = rd(ref, "knn_index.i32", "<i4").reshape(-1, k1) - 1
        t0 = time.perf_counter()
        pk = C._knn_index(coords.to_numpy(dtype=np.float64), k1)
        T["knn_index"] = time.perf_counter() - t0
        mism = int(sum(set(rk[i]) != set(pk[i]) for i in range(len(rk))))
        dup = int(coords.duplicated().sum())
        G["2"] = {"cells_with_different_neighbour_set": mism, "duplicated_coordinates": dup,
                  "same_order": bool((rk == pk).all()), "pass": mism == 0}

    # gates 1, 3: make_cicero_cds
    r_agg = (ref / "agg_cells.txt").read_text().split()
    r_chosen = np.array([int(s.split("_")[1]) - 1 for s in r_agg])
    t0 = time.perf_counter()
    cds = C.make_cicero_cds(ad, reduced_coordinates=coords, k=50, random_state=RRandom(a.seed))
    T["make_cicero_cds"] = time.perf_counter() - t0
    py_chosen = np.asarray(cds.uns["cicero"]["chosen"])
    G["1a_bins"] = {"n_bins_py": int(len(py_chosen)), "n_bins_r": int(len(r_chosen)),
                    "identical": bool(len(py_chosen) == len(r_chosen) and (py_chosen == r_chosen).all())}
    cds_inj = C.make_cicero_cds(ad, reduced_coordinates=coords, k=50, chosen=r_chosen, silent=True)
    rX = read_ref_cicero_x(ref)
    pX = cds_inj.X
    diff = abs(pX - rX)
    G["3"] = {"max_abs_err_matrix": float(diff.max()) if diff.nnz else 0.0,
              "nnz_py": int(pX.nnz), "nnz_r": int(rX.nnz),
              "max_abs_err_size_factor_final": float(np.abs(cds_inj.obs["Size_Factor"].to_numpy()
                                                           - rd(ref, "cicero_size_factor_final.f64")).max()),
              "num_genes_expressed_equal": bool((cds_inj.obs["num_genes_expressed"].to_numpy()
                                                 == rd(ref, "cicero_num_genes_expressed.f64")).all())}
    G["3"]["pass"] = (G["3"]["max_abs_err_matrix"] <= 1e-8 and G["3"]["max_abs_err_size_factor_final"] <= 1e-8
                      and G["3"]["num_genes_expressed_equal"])

    # gate 5: per-window distance parameters with R's sampled windows injected
    if (ref / "sampled_windows.tsv").exists() and want(5):
        sw = pd.read_csv(ref / "sampled_windows.tsv", sep="\t")
        t0 = time.perf_counter()
        dps, det = C.estimate_distance_parameter(cds_inj, genomic_coords=genome, sample_windows=sw["win"].tolist(),
                                                 return_details=True)
        T["estimate_distance_parameter_injected"] = time.perf_counter() - t0
        r_ok = sw[sw["status"] == "ok"]["dp"].to_numpy(dtype=float)
        st_r = sw["status"].replace({"ok": "ok"}).to_numpy()
        st_p = det["status"].to_numpy()
        rel = np.abs(dps - r_ok) / np.abs(r_ok) if len(dps) == len(r_ok) else np.array([np.inf])
        G["5"] = {"n_py": int(len(dps)), "n_r": int(len(r_ok)), "status_identical": bool((st_r == st_p).all()) if len(st_r) == len(st_p) else False,
                  "max_rel_err": float(rel.max()), "n_exact": int((rel == 0).sum()), "pass": bool(rel.max() <= 1e-6)}

    # gates 1b, 6, 7, 8-11, 13: full pipeline in rrng mode
    t0 = time.perf_counter()
    dps_py, det_py = C.estimate_distance_parameter(cds, genomic_coords=genome, random_state=RRandom(a.seed),
                                                   return_details=True)
    T["estimate_distance_parameter"] = time.perf_counter() - t0
    mean_py = C._r_mean(dps_py)
    if (ref / "sampled_windows.tsv").exists():
        sw = pd.read_csv(ref / "sampled_windows.tsv", sep="\t")
        G["1b_windows"] = {"identical": bool(len(sw) == len(det_py) and (sw["win"].to_numpy() == det_py["win"].to_numpy()).all())}
    G["1"] = {"pass": bool(G["1a_bins"]["identical"] and G.get("1b_windows", {"identical": True})["identical"])}
    mean_r = float(rd(ref, "mean_distance_parameter.f64")[0])
    G["6"] = {"mean_py": mean_py, "mean_r": mean_r, "rel_err": abs(mean_py - mean_r) / abs(mean_r),
              "pass": abs(mean_py - mean_r) / abs(mean_r) <= 1e-6}
    t0 = time.perf_counter()
    models = C.generate_cicero_models(cds, distance_parameter=mean_py, genomic_coords=genome, n_jobs=a.threads,
                                      keep_wi=False)
    T["generate_cicero_models"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    conns = C.assemble_connections(models, silent=True)
    T["assemble_connections"] = time.perf_counter() - t0
    T["run_cicero_steps_total"] = T["estimate_distance_parameter"] + T["generate_cicero_models"] + T["assemble_connections"]
    mt = pd.read_csv(ref / "models.tsv", sep="\t")
    st_py = [("glasso" if isinstance(m, C.CiceroModel) else m) for m in models.values()]
    niter_py = [m.niter if isinstance(m, C.CiceroModel) else np.nan for m in models.values()]
    same_names = list(models.keys()) == mt["window"].tolist()
    G["7"] = {"same_window_names": same_names, "status_identical": bool(same_names and st_py == mt["status"].tolist()),
              "niter_mismatch": int(np.nansum(np.asarray(niter_py, float) != mt["niter"].to_numpy(float))
                                    - np.isnan(mt["niter"].to_numpy(float)).sum())}
    G["7"]["pass"] = G["7"]["status_identical"]
    rc = read_ref_conns(ref)
    cmp = compare_conns(conns, rc)
    G["8"] = {k: cmp[k] for k in ("n_rows_py", "n_rows_r", "only_py", "only_r", "na_py", "na_r", "na_pattern_mismatch",
                                   "same_row_order")}
    G["8"]["pass"] = cmp["only_py"] == 0 and cmp["only_r"] == 0 and cmp["na_pattern_mismatch"] == 0
    G["9"] = {"max_abs_err": cmp["max_abs_err"], "n_matched": cmp["n_matched"], "n_exact_equal": cmp["n_exact_equal"],
              "pass": cmp["max_abs_err"] <= 1e-8}
    G["10"] = {"pearson": cmp["pearson"], "spearman": cmp["spearman"],
               "pass": cmp["pearson"] >= 0.9999 - 1e-12 and cmp["spearman"] >= 0.9999 - 1e-12}
    G["11"] = {"jaccard": cmp["jaccard_gt_025"], "n_py": cmp["n_gt_025_py"], "n_r": cmp["n_gt_025_r"],
               "pass": cmp["jaccard_gt_025"] >= 0.999}
    G["13"] = celloracle_format(conns)
    if a.save_conns:
        conns.to_parquet(a.save_conns) if a.save_conns.endswith(".parquet") else conns.to_csv(a.save_conns)

    # gate 12: injected mode (R bins + R's mean distance parameter)
    if want(12):
        models_i = C.generate_cicero_models(cds_inj, distance_parameter=mean_r, genomic_coords=genome,
                                            n_jobs=a.threads, keep_wi=False)
        cmp_i = compare_conns(C.assemble_connections(models_i, silent=True), rc)
        G["12"] = {"max_abs_err": cmp_i["max_abs_err"], "only_py": cmp_i["only_py"], "only_r": cmp_i["only_r"],
                   "na_pattern_mismatch": cmp_i["na_pattern_mismatch"],
                   "pass": cmp_i["max_abs_err"] <= 1e-8 and cmp_i["only_py"] == 0 and cmp_i["only_r"] == 0}

    # gates 16, 17: CCANs on R's connections
    cut_tab = ref / "ccan_cutoffs.tsv"
    if cut_tab.exists() and (want(16) or want(17)):
        from sklearn.metrics import adjusted_rand_score
        cc = pd.read_csv(cut_tab, sep="\t")
        r_cut = float(cc["cutoff"].iloc[0])
        seeds = cc["seed"].tolist()

        def labels(df, universe):
            s = pd.Series(df["CCAN"].to_numpy(), index=df["Peak"].astype(str).to_numpy())
            return s.reindex(universe).fillna(0).astype(int).to_numpy()

        rfix = {s: pd.read_csv(ref / f"ccan_fixedcut_seed{s}.tsv", sep="\t") for s in seeds}
        t0 = time.perf_counter()
        pyc = C.generate_ccans(rc, coaccess_cutoff_override=r_cut, random_state=a.seed, silent=True)
        T["generate_ccans_fixed_cutoff"] = time.perf_counter() - t0
        uni = sorted(set(pyc["Peak"]).union(*[set(v["Peak"]) for v in rfix.values()]))
        ari_py = [adjusted_rand_score(labels(rfix[s], uni), labels(pyc, uni)) for s in seeds]
        ari_rr = [adjusted_rand_score(labels(rfix[s1], uni), labels(rfix[s2], uni))
                  for i, s1 in enumerate(seeds) for s2 in seeds[i + 1:]]
        med_rr = float(np.median(ari_rr)) if ari_rr else 1.0
        thr = 0.95 if med_rr >= 0.95 else float(min(ari_rr))
        G["16"] = {"ari_py_vs_r_seed": dict(zip(map(str, seeds), map(float, ari_py))),
                   "ari_py_vs_r_first_seed": float(ari_py[0]), "r_vs_r_median": med_rr, "r_vs_r_min": float(min(ari_rr)) if ari_rr else 1.0,
                   "threshold_applied": thr, "n_ccan_py": int(pyc["CCAN"].nunique()),
                   "n_ccan_r": int(rfix[seeds[0]]["CCAN"].nunique()), "pass": bool(ari_py[0] >= thr)}
        t0 = time.perf_counter()
        pyc2 = C.generate_ccans(rc, random_state=a.seed, silent=True)
        T["generate_ccans_with_cutoff_search"] = time.perf_counter() - t0
        py_cut = pyc2.attrs["coaccess_cutoff"]
        rmin, rmax = float(cc["cutoff"].min()), float(cc["cutoff"].max())
        G["17"] = {"cutoff_py": py_cut, "cutoff_r": r_cut, "r_range": [rmin, rmax], "abs_diff": abs(py_cut - r_cut),
                   "pass": bool(abs(py_cut - r_cut) <= 0.02 + 1e-12 or (rmin - 1e-12 <= py_cut <= rmax + 1e-12))}

    # gates 14, 15: gene activity
    if (ref / "unnorm_ga.f64").exists() and (want(14) or want(15)):
        prom = pd.read_csv(fx / "promoters.tsv", sep="\t")
        adx = ad.copy()
        t0 = time.perf_counter()
        C.annotate_cds_by_site(adx, prom)
        ann_r = pd.read_csv(ref / "annotated_sites.tsv", sep="\t", keep_default_na=False, na_values=["NA"])
        ann_r = ann_r.set_index("peak").loc[adx.var_names]
        gene_eq = bool((adx.var["gene"].fillna("<NA>").astype(str).to_numpy()
                        == ann_r["gene"].fillna("<NA>").astype(str).to_numpy()).all())
        ga = C.build_gene_activity_matrix(adx, rc)
        T["build_gene_activity_matrix"] = time.perf_counter() - t0

        def rmat(nm):
            g = (ref / f"{nm}_genes.txt").read_text().split("\n")
            c = (ref / f"{nm}_cells.txt").read_text().split("\n")
            v = rd(ref, f"{nm}.f64")
            g = g[: v.size // max(1, len([x for x in c if x]))]
            c = [x for x in c if x]
            g = [x for x in g if x]
            return pd.DataFrame(v.reshape(len(g), len(c), order="F"), index=g, columns=c)

        R1 = rmat("unnorm_ga")
        P1 = pd.DataFrame(ga.X.T.toarray(), index=ga.var_names, columns=ga.obs_names)
        e14 = float(np.abs(P1.loc[R1.index, R1.columns].to_numpy() - R1.to_numpy()).max())
        # R drops all-zero genes/cells before normalising (vignette); do the same
        keep_g = np.asarray(ga.X.sum(axis=0)).ravel() != 0
        keep_c = np.asarray(ga.X.sum(axis=1)).ravel() != 0
        ga2 = ga[keep_c][:, keep_g].copy()
        num_genes = pd.Series(np.asarray((ad.X > 0).sum(axis=1)).ravel(), index=ad.obs_names)
        t0 = time.perf_counter()
        gn = C.normalize_gene_activities(ga2, num_genes)
        T["normalize_gene_activities"] = time.perf_counter() - t0
        R2 = rmat("ga")
        P2 = pd.DataFrame(gn.X.T.toarray(), index=gn.var_names, columns=gn.obs_names)
        e15 = float(np.abs(P2.loc[R2.index, R2.columns].to_numpy() - R2.to_numpy()).max() / np.abs(R2.to_numpy()).max())
        G["14"] = {"annotation_gene_column_identical": gene_eq, "genes_py": int(P1.shape[0]), "genes_r": int(R1.shape[0]),
                   "max_abs_err": e14, "pass": bool(gene_eq and e14 <= 1e-8 and set(P1.index) >= set(R1.index))}
        G["15"] = {"max_rel_err": e15, "genes_py": int(P2.shape[0]), "genes_r": int(R2.shape[0]),
                   "same_genes": bool(set(P2.index) == set(R2.index)), "pass": bool(e15 <= 1e-8 and set(P2.index) == set(R2.index))}

    tr = pd.read_csv(ref / "timings.tsv", sep="\t")
    res["timings_r"] = dict(zip(tr["step"], tr["seconds"]))
    res["all_pass"] = all(v.get("pass", True) for k, v in G.items() if isinstance(v, dict) and "pass" in v)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v.get("pass") for k, v in G.items() if isinstance(v, dict)}, indent=0))
    print("ALL PASS" if res["all_pass"] else "SOME GATES FAIL")
    return res


if __name__ == "__main__":
    main()
