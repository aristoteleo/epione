"""Run the gates in MANIFEST.md: Epione vs cooltools 0.7.1 and HiCExplorer 3.7.6.

Needs an environment with HiCExplorer 3.7.6, HiCMatrix, cooltools 0.7.1 and
Epione importable. References are called through their Python APIs so that
nothing is rounded by text output.

    python parity_check.py --data DIR --hicexplorer-src DIR --out results.json [--gates A,B0,...]

``--data`` holds the Zenodo 16416373 files
(``corrected_contact_matrix_dm3_large.h5``, ``dm3_genes.bed``,
``GM12878_25kb_cooler_coarsen.cool``); ``--hicexplorer-src`` is a checkout of
HiCExplorer tag 3.7.6 (for its test data).
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import hicexplorer  # noqa: F401  (sets the multiprocessing start method; must come first)

RESULTS = {}


def record(gate, **kw):
    RESULTS.setdefault(gate, {}).update(kw)
    print(gate, json.dumps(kw, default=str)[:400], flush=True)


def maxabs(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return float("inf")
    if (~na).sum() == 0:
        return 0.0
    return float(np.max(np.abs(a[~na] - b[~na])))


def rel_err(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return float("inf")
    m = ~na
    if m.sum() == 0:
        return 0.0
    den = np.maximum(np.abs(a[m]), 1e-300)
    return float(np.max(np.abs(a[m] - b[m]) / den))


# ---------------------------------------------------------------- fixtures
def prepare(data, src, work):
    fx = {}
    h5 = data / "corrected_contact_matrix_dm3_large.h5"
    fx["D_h5"] = h5
    fx["D_ref"] = work / "D_hicconvert.cool"
    if not fx["D_ref"].exists():
        from hicexplorer import hicConvertFormat
        hicConvertFormat.main(["--matrices", str(h5), "--outFileName", str(fx["D_ref"]),
                               "--inputFormat", "h5", "--outputFormat", "cool"])
    fx["D_epi"] = work / "D_epione.cool"
    if not fx["D_epi"].exists():
        from epione.bulk.hic import h5_to_cool
        h5_to_cool(h5, fx["D_epi"])
    td = src / "hicexplorer" / "test" / "test_data"
    fx["R_t"] = td / "hicDifferentialTAD" / "GSM2644945_Untreated-R1.100000_chr1.cool"
    fx["R_c"] = td / "hicDifferentialTAD" / "GSM2644947_Auxin2days-R1.100000_chr1.cool"
    fx["G"] = td / "hicDetectLoops" / "GSE63525_GM12878_insitu_primary_2_5mb.cool"
    fx["P"] = td / "hicPCA" / "mm9_reduced_chr1.cool"
    fx["genes_dm3"] = data / "dm3_genes.bed"
    fx["genes_mm9"] = td / "hicPCA" / "mm9_genes_sorted.bed12"
    # Amendment 1: C has no weight column; balance a copy once
    fx["C"] = work / "C_balanced.cool"
    if not fx["C"].exists():
        import shutil
        from epione.bulk.hic import balance_cool
        shutil.copy(data / "GM12878_25kb_cooler_coarsen.cool", fx["C"])
        balance_cool(fx["C"])
    return fx


# ------------------------------------------------------------------- B0
def gate_B0(fx):
    import cooler
    a = cooler.Cooler(str(fx["D_ref"]))
    b = cooler.Cooler(str(fx["D_epi"]))
    ba, bb = a.bins()[:], b.bins()[:]
    pa, pb = a.pixels()[:], b.pixels()[:]
    same_bins = ba[["chrom", "start", "end"]].astype(str).equals(bb[["chrom", "start", "end"]].astype(str))
    w = maxabs(ba["weight"], bb["weight"]) if "weight" in ba and "weight" in bb else None
    same_ids = np.array_equal(pa["bin1_id"].to_numpy(), pb["bin1_id"].to_numpy()) and \
        np.array_equal(pa["bin2_id"].to_numpy(), pb["bin2_id"].to_numpy())
    cnt = maxabs(pa["count"], pb["count"]) if same_ids else float("inf")
    record("B0", bins_identical=bool(same_bins), weight_maxabs=w, pixel_ids_identical=bool(same_ids),
           count_maxabs=cnt, n_pixels=(len(pa), len(pb)), n_bins=len(ba),
           pass_=bool(same_bins and same_ids and cnt == 0 and (w in (None, 0.0))))


# ---------------------------------------------------------------- B1-B3
def _hicexplorer_tads(path, kw, prefix):
    from hicexplorer.hicFindTADs import HicFindTads
    t0 = time.time()
    ft = HicFindTads(str(path), num_processors=1, max_depth=kw.get("max_depth"), min_depth=kw.get("min_depth"),
                     step=kw.get("step"), delta=kw.get("delta", 0.01), min_boundary_distance=None,
                     use_zscore=True, p_correct_for_multiple_testing=kw.get("mtc", "fdr"),
                     p_threshold_comparisons=kw.get("thr", 0.01), pChromosomes=None)
    ft.compute_spectra_matrix()
    t_score = time.time() - t0
    ft.find_boundaries()
    ft.save_domains_and_boundaries(str(prefix))
    return ft, time.time() - t0, t_score


def gate_B123(fx, name, path, kw, work):
    import epione.bulk.hic as h
    prefix = work / f"tads_{name}"
    ft, t_ref, t_ref_score = _hicexplorer_tads(path, kw, prefix)
    t0 = time.time()
    tab = h.insulation(str(path), method="hicexplorer", min_depth=kw.get("min_depth"),
                       max_depth=kw.get("max_depth"), step=kw.get("step"))
    t_score = time.time() - t0
    bnd = h.tad_boundaries(tab, delta=kw.get("delta", 0.01), threshold_comparisons=kw.get("thr", 0.01),
                           correct_for_multiple_testing=kw.get("mtc", "fdr"))
    dom = h.tad_domains(bnd)
    t_epi = time.time() - t0
    bm = ft.bedgraph_matrix
    win_cols = [c for c in tab.columns if c.startswith("tad_separation_score_")]
    rows_ok = (list(map(str, bm["chrom"])) == tab["chrom"].astype(str).tolist()
               and np.array_equal(bm["chr_start"], tab["start"].to_numpy())
               and np.array_equal(bm["chr_end"], tab["end"].to_numpy()))
    score_err = maxabs(bm["matrix"], tab[win_cols].to_numpy()) if rows_ok else float("inf")
    record(f"B1_{name}", rows_identical=bool(rows_ok), n_rows=(len(bm["chrom"]), len(tab)),
           windows=len(win_cols), max_abs_err=score_err, pass_=bool(rows_ok and score_err <= 1e-8),
           seconds_hicexplorer=round(t_ref_score, 2), seconds_epione=round(t_score, 2))
    ref_b = pd.read_csv(f"{prefix}_boundaries.bed", sep="\t", header=None)
    ref_set = set(zip(ref_b[0].astype(str), ref_b[1], ref_b[2]))
    epi_set = set(zip(bnd["chrom"].astype(str), bnd["start"], bnd["end"]))
    pv_ref = np.array([ft.boundaries["pvalues"][int(n[1:])] for n in ref_b[3]], dtype=float)
    dl_ref = np.array([ft.boundaries["delta"][int(n[1:])] for n in ref_b[3]], dtype=float)
    same = ref_set == epi_set
    if same:
        e = bnd.set_index(["chrom", "start", "end"]).loc[list(zip(ref_b[0].astype(str), ref_b[1], ref_b[2]))]
        pv_err = rel_err(pv_ref, e["pvalue"].to_numpy())
        dl_err = maxabs(dl_ref, e["delta"].to_numpy())
    else:
        pv_err = dl_err = float("inf")
    record(f"B2_{name}", n_boundaries=(len(ref_set), len(epi_set)), identical=bool(same),
           only_ref=len(ref_set - epi_set), only_epi=len(epi_set - ref_set),
           pvalue_rel_err=pv_err, delta_max_abs_err=dl_err,
           pass_=bool(same and pv_err <= 1e-8 and dl_err <= 1e-8),
           seconds_hicexplorer_total=round(t_ref, 2), seconds_epione_total=round(t_epi, 2))
    ref_d = pd.read_csv(f"{prefix}_domains.bed", sep="\t", header=None)
    rd = set(zip(ref_d[0].astype(str), ref_d[1], ref_d[2]))
    ed = set(zip(dom["chrom"].astype(str), dom["start"], dom["end"]))
    record(f"B3_{name}", n_domains=(len(rd), len(ed)), identical=bool(rd == ed), pass_=bool(rd == ed))
    return dom


# ------------------------------------------------------------------- B4
def gate_B4(fx, name, path, kw):
    import epione.bulk.hic as h
    from hicexplorer import hicPCA
    with tempfile.TemporaryDirectory() as tmp:
        outs = [f"{tmp}/pca{i}.bedgraph" for i in (1, 2)]
        args = ["--matrix", str(path), "--outputFileName", *outs, "--whichEigenvectors", "1", "2",
                "--format", "bedgraph",
                "--method", kw["obs_exp"]]
        if kw.get("track"):
            args += ["--extraTrack", str(kw["track"])]
        t0 = time.time()
        hicPCA.main(args)
        t_ref = time.time() - t0
        ref = [pd.read_csv(o, sep="\t", header=None) for o in outs]
    t0 = time.time()
    epi = h.compartments(str(path), method="hicexplorer", n_eigs=2, obs_exp=kw["obs_exp"],
                         extra_track=kw.get("track"))
    t_epi = time.time() - t0
    # bedgraph values are written with %.12f; compare against the rounded values
    # with a 5e-13 rounding allowance on top of the 1e-6 gate (rounding << gate).
    errs = []
    same_rows = True
    for j, r in enumerate(ref):
        same_rows &= (r[0].astype(str).tolist() == epi["chrom"].astype(str).tolist()
                      and np.array_equal(r[1].to_numpy(), epi["start"].to_numpy()))
        errs.append(maxabs(r[3].to_numpy(), np.round(epi[f"E{j + 1}"].to_numpy(), 12)) if same_rows else float("inf"))
    record(f"B4_{name}", rows_identical=bool(same_rows), max_abs_err=errs,
           pass_=bool(same_rows and max(errs) <= 1e-6),
           seconds_hicexplorer=round(t_ref, 2), seconds_epione=round(t_epi, 2),
           note="reference values read from hicPCA bedgraph (%.12f)")


# ------------------------------------------------------------------- B5
def gate_B5(fx):
    import cooler
    import epione.bulk.hic as h
    from hicexplorer import hicCompareMatrices
    for op in ("diff", "ratio", "log2ratio"):
        with tempfile.TemporaryDirectory() as tmp:
            out = f"{tmp}/cmp.cool"
            t0 = time.time()
            hicCompareMatrices.main(["--matrices", str(fx["R_t"]), str(fx["R_c"]), "--outFileName", out,
                                     "--operation", op])
            t_ref = time.time() - t0
            ref = cooler.Cooler(out).matrix(balance=False, sparse=True)[:].toarray()
        t0 = time.time()
        m, bins = h.compare_matrices(str(fx["R_t"]), str(fx["R_c"]), operation=op)
        t_epi = time.time() - t0
        epi = m.toarray()
        # HiCExplorer writes the upper triangle; compare the symmetric full matrix
        ref_full = np.triu(ref) + np.triu(ref, 1).T
        err = maxabs(ref_full, epi)
        record(f"B5_{op}", max_abs_err=err, pass_=bool(err <= 1e-13),
               seconds_hicexplorer=round(t_ref, 2), seconds_epione=round(t_epi, 2))


# ------------------------------------------------------------------- B6
def gate_B6(fx):
    import epione.bulk.hic as h
    import hicexplorer.hicCorrelate as hc

    captured = {}
    orig = hc.plot_correlation

    def grab(results, *a, **k):
        captured["r"] = np.array(results)
        return None

    hc.plot_correlation = grab
    cases = [("pearson", False, None), ("spearman", False, None), ("pearson", True, None),
             ("pearson", False, (200000, 5000000)), ("spearman", True, (200000, 5000000))]
    try:
        for method, log1p, rng in cases:
            with tempfile.TemporaryDirectory() as tmp:
                args = ["--matrices", str(fx["R_t"]), str(fx["R_c"]), "--method", method,
                        "--outFileNameHeatmap", f"{tmp}/h.png", "--outFileNameScatter", f"{tmp}/s.png"]
                if log1p:
                    args.append("--log1p")
                if rng:
                    args += ["--range", f"{rng[0]}:{rng[1]}"]
                hc.main(args)
            ref = captured["r"]
            epi = h.correlate_matrices([str(fx["R_t"]), str(fx["R_c"])], method=method, log1p=log1p,
                                       distance_range=rng).to_numpy()
            err = maxabs(ref, epi)
            record(f"B6_{method}_log1p{int(log1p)}_range{int(bool(rng))}", r_ref=float(ref[0, 1]),
                   r_epi=float(epi[0, 1]), max_abs_err=err, pass_=bool(err <= 1e-12))
    finally:
        hc.plot_correlation = orig


# ------------------------------------------------------------------- B7
def gate_B7(fx, domains_bed):
    import epione.bulk.hic as h
    from hicexplorer import hicDifferentialTAD
    for mode, rej in (("all", "one"), ("all", "all"), ("intra-TAD", "one"), ("left-inter-TAD", "all")):
        with tempfile.TemporaryDirectory() as tmp:
            pre = f"{tmp}/diff"
            t0 = time.time()
            hicDifferentialTAD.main(["--targetMatrix", str(fx["R_t"]), "--controlMatrix", str(fx["R_c"]),
                                     "--tadDomains", str(domains_bed), "--outFileNamePrefix", pre,
                                     "--pValue", "0.05", "--mode", mode, "--modeReject", rej,
                                     "--threads", "1"])
            t_ref = time.time() - t0
            cols = ["chrom", "start", "end", "name", "score", "strand", "p_l", "p_r", "p_i", "w_l", "w_r", "w_i"]
            acc = pd.read_csv(f"{pre}_accepted.diff_tad", sep="\t", comment="#", header=None, names=cols)
            rjt = pd.read_csv(f"{pre}_rejected.diff_tad", sep="\t", comment="#", header=None, names=cols)
        acc["rejected"] = False
        rjt["rejected"] = True
        ref = pd.concat([acc, rjt]).sort_values(["chrom", "start"]).reset_index(drop=True)
        t0 = time.time()
        epi = h.differential_tads(str(fx["R_t"]), str(fx["R_c"]), str(domains_bed), p_value=0.05,
                                  mode=mode, mode_reject=rej).sort_values(["chrom", "start"]).reset_index(drop=True)
        t_epi = time.time() - t0
        same_rows = ref[["start", "end"]].to_numpy().tolist() == epi[["start", "end"]].to_numpy().tolist()
        flags = same_rows and np.array_equal(ref["rejected"].to_numpy(), epi["rejected"].to_numpy())
        if same_rows:
            pe = rel_err(ref[["p_l", "p_r", "p_i"]].to_numpy(dtype=float),
                         epi[["pvalue_left_inter", "pvalue_right_inter", "pvalue_intra"]].to_numpy(dtype=float))
        else:
            pe = float("inf")
        record(f"B7_{mode}_{rej}", n_tads=(len(ref), len(epi)), rows_identical=bool(same_rows),
               flags_identical=bool(flags), n_rejected=(int(ref["rejected"].sum()), int(epi["rejected"].sum())),
               pvalue_rel_err=pe, pass_=bool(flags and pe <= 1e-8),
               seconds_hicexplorer=round(t_ref, 2), seconds_epione=round(t_epi, 2),
               note="reference p-values parsed from text (str(float) = repr, full precision)")


# ------------------------------------------------------------------- B8
def gate_B8(fx, name, path, kw, chromosomes=None):
    import epione.bulk.hic as h
    from hicexplorer import hicDetectLoops
    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/loops.bedgraph"
        args = ["--matrix", str(path), "-o", out, "-pit", str(kw["pit"]), "-p", str(kw["p"]),
                "-pp", str(kw["pp"]), "-w", str(kw["w"]), "-pw", str(kw["pw"]),
                "--maxLoopDistance", str(kw["max"]), "-t", "1", "-tpc", "1"]
        if chromosomes:
            args += ["--chromosomes", *chromosomes]
        t0 = time.time()
        hicDetectLoops.main(args)
        t_ref = time.time() - t0
        if os.path.exists(out) and os.path.getsize(out):
            ref = pd.read_csv(out, sep="\t", header=None)
        else:  # hicDetectLoops writes nothing when it finds no loop
            ref = pd.DataFrame({k: pd.Series(dtype=object) for k in range(7)})
    t0 = time.time()
    epi = h.loops(str(path), method="hicexplorer", chromosomes=chromosomes, window_size=kw["w"],
                  peak_width=kw["pw"], p_value_preselection=kw["pp"], peak_interactions_threshold=kw["pit"],
                  p_value=kw["p"], max_loop_distance=kw["max"])
    t_epi = time.time() - t0
    rk = list(zip(ref[0].astype(str), ref[1], ref[2], ref[3].astype(str), ref[4], ref[5]))
    ek = list(zip(epi["chrom1"].astype(str), epi["start1"], epi["end1"], epi["chrom2"].astype(str),
                  epi["start2"], epi["end2"]))
    rs, es = set(rk), set(ek)
    jac = len(rs & es) / len(rs | es) if (rs | es) else 1.0
    from scipy.stats import spearmanr
    shared = sorted(rs & es)
    rp = dict(zip(rk, ref[6].astype(float)))
    ep = dict(zip(ek, epi["pvalue"].astype(float)))
    if len(shared) >= 3:
        rho = float(spearmanr([rp[k] for k in shared], [ep[k] for k in shared])[0])
        prel = rel_err(np.array([rp[k] for k in shared]), np.array([ep[k] for k in shared]))
    else:
        rho, prel = (1.0 if shared else float("nan")), (rel_err(np.array([rp[k] for k in shared]),
                                                       np.array([ep[k] for k in shared])) if shared else 0.0)
    record(f"B8_{name}", n_loops=(len(rs), len(es)), jaccard=jac, pvalue_spearman_shared=rho,
           pvalue_rel_err_shared=prel, pass_=bool(jac >= 0.95 and (len(shared) < 3 or rho >= 0.99)),
           exact=bool(rs == es), seconds_hicexplorer=round(t_ref, 2), seconds_epione=round(t_epi, 2),
           note="reference p-values from text (%s formatting of floats = repr)")


# -------------------------------------------------------------------- A
def gate_A(fx, work):
    import cooler
    import cooltools
    from cooltools.api.expected import expected_cis
    import epione.bulk.hic as h

    uri = str(fx["C"])
    clr = cooler.Cooler(uri)
    chroms = [c for c in ["17", "18", "19", "20", "21", "22"] if c in clr.chromnames]
    record("A_fixture", chromosomes=chroms, binsize=clr.binsize,
           has_weight="weight" in clr.bins().columns)
    view = pd.DataFrame({"chrom": chroms, "start": 0, "end": [int(clr.chromsizes[c]) for c in chroms],
                         "name": chroms})
    # A1
    e_epi = h.compartments(uri, chromosomes=chroms, n_eigs=3)
    _, e_ref = cooltools.eigs_cis(clr=clr, phasing_track=None, view_df=view, n_eigs=3, ignore_diags=2)
    err = max(maxabs(e_ref[f"E{k}"], e_epi[f"E{k}"]) for k in (1, 2, 3))
    record("A1", max_abs_err=err, pass_=bool(err <= 1e-13))
    # A2
    sad, edges, cnt = h.saddle(uri, e_epi, chromosomes=chroms)
    exp = expected_cis(clr=clr, view_df=view, ignore_diags=2, chunksize=1_000_000)
    s_ref, c_ref = cooltools.saddle(clr=clr, expected=exp, track=e_epi[["chrom", "start", "end", "E1"]],
                                    contact_type="cis", n_bins=50, view_df=view, qrange=(0.025, 0.975))
    pos = c_ref > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ref_mat = s_ref / c_ref
    err = maxabs(ref_mat[pos], sad[pos])
    empty_nan = bool(np.all(np.isnan(sad[~pos]))) if (~pos).any() else True
    record("A2", max_abs_err_count_pos=err, count_zero_cells=int((~pos).sum()),
           count_zero_cells_are_nan=empty_nan,
           count_zero_cell_values=[float(x) for x in np.asarray(sad[~pos]).ravel()[:5]],
           pass_=bool(err <= 1e-13 and empty_nan))
    # A3 / A4
    ins_epi = h.insulation(uri, chromosomes=chroms, window_bp=[100_000, 250_000])
    ins_ref = cooltools.insulation(clr=clr, window_bp=[100_000, 250_000], view_df=view, ignore_diags=None,
                                   threshold="Li", nproc=1)
    sc = [c for c in ins_ref.columns if c.startswith("log2_insulation_score_")]
    err = max(maxabs(ins_ref[c], ins_epi[c]) for c in sc)
    bc = [c for c in ins_ref.columns if c.startswith("is_boundary_")]
    flags = all(np.array_equal(ins_ref[c].to_numpy(), ins_epi[c].to_numpy()) for c in bc)
    record("A3", max_abs_err=err, boundary_flags_identical=bool(flags), pass_=bool(err <= 1e-13 and flags))
    b_epi = h.tad_boundaries(ins_epi, window_bp=100_000)
    r = ins_ref[ins_ref["is_boundary_100000"].astype(bool)]
    same = set(zip(r["chrom"], r["start"], r["end"])) == set(zip(b_epi["chrom"], b_epi["start"], b_epi["end"]))
    record("A4", n=(len(r), len(b_epi)), identical=bool(same), pass_=bool(same))
    # A5 (dots) on two chromosomes to bound runtime
    sub = chroms[-2:]
    vsub = view[view["chrom"].isin(sub)].reset_index(drop=True)
    t0 = time.time()
    d_epi = h.loops(uri, chromosomes=sub)
    t_e = time.time() - t0
    exp2 = expected_cis(clr=clr, view_df=vsub, ignore_diags=2, chunksize=1_000_000)
    d_ref = cooltools.dots(clr=clr, expected=exp2, view_df=vsub, max_loci_separation=10_000_000,
                           lambda_bin_fdr=0.1, clustering_radius=20_000, nproc=1)
    key = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    a = d_ref.sort_values(key).reset_index(drop=True)
    b = d_epi.sort_values(key).reset_index(drop=True)
    same = a[key].astype(str).equals(b[key].astype(str))
    num = [c for c in a.columns if c not in key and pd.api.types.is_numeric_dtype(a[c])]
    err = max(maxabs(a[c], b[c]) for c in num) if same and num else (0.0 if same else float("inf"))
    record("A5", n=(len(a), len(b)), identical_set=bool(same), max_abs_err=err,
           pass_=bool(same and err <= 1e-13), seconds_epione=round(t_e, 2))
    # A6 pile-up on the A5 dots
    feats = a[key].head(200)
    p_epi = h.pileup(uri, feats, chromosomes=sub, flank=100_000)
    stack = np.asarray(cooltools.pileup(clr=clr, features_df=feats, view_df=vsub, expected_df=exp2,
                                        flank=100_000, nproc=1))
    record("A6_shape", cooltools_stack_shape=list(stack.shape))
    # cooltools 0.7 stacks features on the first axis
    ref_mean = np.nanmean(stack, axis=0)
    err = maxabs(ref_mean, p_epi)
    record("A6", max_abs_err=err, pass_=bool(err <= 1e-13))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--hicexplorer-src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--gates", default="B0,B1,B4,B5,B6,B7,A")
    a = ap.parse_args()
    work = a.work or Path(tempfile.mkdtemp())
    work.mkdir(parents=True, exist_ok=True)
    gates = set(a.gates.split(","))
    fx = prepare(a.data, a.hicexplorer_src, work)
    from importlib.metadata import version
    RESULTS["versions"] = {p: version(p) for p in ("HiCExplorer", "HiCMatrix", "cooltools", "cooler", "numpy",
                                                    "scipy", "pandas", "fit_nbinom")}
    if "B0" in gates:
        gate_B0(fx)
    dom_R = None
    if "B1" in gates:
        gate_B123(fx, "D", fx["D_ref"], dict(min_depth=30000, max_depth=100000, step=10000, thr=0.05,
                                            delta=0.01, mtc="fdr"), work)
        dom_R = gate_B123(fx, "R", fx["R_t"], dict(), work)
        gate_B123(fx, "R_bonferroni", fx["R_t"], dict(mtc="bonferroni", thr=0.05), work)
    if "B4" in gates:
        gate_B4(fx, "P_dist_norm", fx["P"], dict(obs_exp="dist_norm"))
        gate_B4(fx, "P_lieberman", fx["P"], dict(obs_exp="lieberman"))
        gate_B4(fx, "P_dist_norm_genes", fx["P"], dict(obs_exp="dist_norm", track=fx["genes_mm9"]))
        gate_B4(fx, "D_dist_norm_genes", fx["D_ref"], dict(obs_exp="dist_norm", track=fx["genes_dm3"]))
    if "B5" in gates:
        gate_B5(fx)
    if "B6" in gates:
        gate_B6(fx)
    if "B7" in gates:
        bed = work / "R_domains.bed"
        if dom_R is None:
            import epione.bulk.hic as h
            tab = h.insulation(str(fx["R_t"]), method="hicexplorer")
            dom_R = h.tad_domains(h.tad_boundaries(tab))
        out = dom_R[["chrom", "start", "end", "name", "tad_separation_score"]].copy()
        out["strand"] = "."
        out.to_csv(bed, sep="\t", header=False, index=False)
        gate_B7(fx, bed)
    if "B8" in gates:
        gate_B8(fx, "G_testsuite", fx["G"], dict(pit=1, p=0.5, pp=0.55, w=5, pw=2, max=3000000),
                chromosomes=["1", "2"])
        gate_B8(fx, "C_defaults_chr21_22", fx["C"], dict(pit=10, p=0.025, pp=0.1, w=5, pw=2, max=2000000),
                chromosomes=["21", "22"])
    if "A" in gates:
        gate_A(fx, work)
    a.out.write_text(json.dumps(RESULTS, indent=1, default=str))


if __name__ == "__main__":
    main()
