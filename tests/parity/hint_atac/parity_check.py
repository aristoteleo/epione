"""HINT-ATAC parity: Epione vs RGT 1.0.2 on the TOBIAS chr4 fixture.

Evaluates the gates pre-registered in MANIFEST.md. Reference outputs are
produced outside this repository (see MANIFEST.md, "Reference") and read
from ``$EPIONE_HINT_PARITY_DIR``::

    hmm_ref/atac_{single,paired}.npz      parameters read from the pickles
    tracks/{Bcell,Tcell}_{single,paired}.npz   per-peak intermediate tracks
    fp_{single,paired}/{Bcell,Tcell}.bed  rgt-hint footprinting
    match/{Bcell,Tcell}_mpbs.bed          rgt-motifanalysis matching
    diff_{bc,raw}/                        rgt-hint differential (--output-profiles)

The fixture (BAMs, ``genome.fa``, ``merged_peaks.bed``, ``motifs.jaspar``)
is read from ``$EPIONE_HINT_DATA``.

Usage::

    python parity_check.py [--gates H1,H2,...] [--n-jobs 8] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import epione.tl._hint as H

REF = Path(os.environ.get("EPIONE_HINT_PARITY_DIR", ""))
DATA = Path(os.environ.get("EPIONE_HINT_DATA", ""))
SAMPLES = ("Bcell", "Tcell")
MODELS = ("single", "paired")
WORK = Path(os.environ.get("EPIONE_HINT_WORK", REF / "epione"))


def _split(z, key):
    """Per-peak arrays from a ragged dump."""
    flat, lens = z[key], z[key + "__len"]
    out, o = [], 0
    for n in lens:
        out.append(flat[o:o + n])
        o += n
    return out


def _bias():
    return H.HintBias(DATA / "genome.fa")


def _peaks():
    return H._read_bed(DATA / "merged_peaks.bed")


# --------------------------------------------------------------------------
# H1-H5: per-peak tracks
# --------------------------------------------------------------------------

def _epione_tracks(sample, model, bias, hmm):
    import pysam
    bam = pysam.AlignmentFile(str(DATA / f"{sample}.bam"), "rb")
    fa = pysam.FastaFile(bias.fasta)
    sg = H.savitzky_golay_coefficients(9, 2, 1)
    fn = H._paired_tracks if model == "paired" else H._single_tracks
    res = []
    for chrom, s, e in _peaks()[["chrom", "start", "end"]].itertuples(index=False):
        tr = fn(bam, fa, bias, chrom, int(s), int(e), 5, -5, sg)
        if tr is not None:
            try:
                tr["states"] = hmm.decode(tr["X"])
            except ValueError:
                tr["states"] = np.zeros(0, dtype=int)
        res.append(tr)
    return res


def gate_h1_to_h5(results):
    out = {}
    hmm_ok = True
    for model in MODELS:
        ref = np.load(REF / "hmm_ref" / f"atac_{model}.npz")
        mine = H.load_hint_hmm(model)
        for k, v in (("startprob", mine.startprob), ("transmat", mine.transmat),
                     ("means", mine.means), ("covars", mine.covars)):
            hmm_ok &= bool(np.array_equal(ref[k], v))
    out["H1_hmm_params_identical"] = hmm_ok
    bias = _bias()
    h1 = h2 = True
    h3_max, h3_bits, h3_n = 0.0, 0, 0
    h4_max = 0.0
    agree = total = 0
    skipped = {}
    for sample in SAMPLES:
        for model in MODELS:
            t0 = time.time()
            z = np.load(REF / "tracks" / f"{sample}_{model}.npz")
            hmm = H.load_hint_hmm(model)
            mine = _epione_tracks(sample, model, bias, hmm)
            ref_bias_f, ref_bias_r = _split(z, "bias_f"), _split(z, "bias_r")
            ref_X = _split(z, "X")
            ref_states = _split(z, "states")
            nfeat = 8 if model == "paired" else 4
            if model == "single":
                raw_keys = [("raw_f", "raw_f"), ("raw_r", "raw_r")]
                bc_keys = [("bc_f", "bc_f"), ("bc_r", "bc_r")]
            else:
                raw_keys = [(f"raw_{s}_{t}", f"raw_{s}_{t}") for t in ("max145", "min145") for s in "fr"]
                bc_keys = [(f"bc_{s}_{t}", f"bc_{s}_{t}") for t in ("max145", "min145") for s in "fr"]
            ref_raw = {k: _split(z, k) for k, _ in raw_keys}
            ref_bc = {k: _split(z, k) for k, _ in bc_keys}
            n_skip_ref = sum(1 for x in ref_states if len(x) == 0)
            n_skip_me = sum(1 for tr in mine if tr is None or len(tr["states"]) == 0)
            skipped[f"{sample}_{model}"] = [n_skip_ref, n_skip_me]
            for p, tr in enumerate(mine):
                if tr is None:
                    if len(ref_states[p]):
                        h1 = h2 = False
                        total += len(ref_states[p])
                    continue
                if "bias_f" in tr:
                    h1 &= np.array_equal(tr["bias_f"], ref_bias_f[p]) and np.array_equal(tr["bias_r"], ref_bias_r[p])
                    for k, _ in raw_keys:
                        h2 &= np.array_equal(tr[k], ref_raw[k][p])
                for k, _ in bc_keys:
                    a, b = tr[k], ref_bc[k][p]
                    if len(a) != len(b):
                        h3_max = np.inf
                        continue
                    if len(a):
                        h3_max = max(h3_max, float(np.max(np.abs(a - b))))
                        h3_bits += int(np.sum(a == b))
                        h3_n += len(a)
                X = tr["X"].ravel()
                rX = ref_X[p]
                if len(X) != len(rX):
                    h4_max = np.inf
                elif len(X):
                    h4_max = max(h4_max, float(np.max(np.abs(X - rX))))
                st, rs = tr["states"], ref_states[p]
                if len(st) == len(rs):
                    agree += int(np.sum(st == rs))
                    total += len(rs)
                else:
                    total += max(len(rs), len(st))
            print(f"  tracks {sample}/{model}: {time.time() - t0:.0f}s", flush=True)
    out.update({
        "H1_kmer_bias_identical": bool(h1),
        "H2_raw_counts_identical": bool(h2),
        "H3_bc_max_abs_err": h3_max,
        "H3_bc_fraction_bit_identical": h3_bits / max(h3_n, 1),
        "H4_hmm_input_max_abs_err": h4_max,
        "H5_state_agreement": agree / max(total, 1),
        "skipped_peaks_ref_vs_epione": skipped,
    })
    out["H1_pass"] = bool(hmm_ok and h1)
    out["H2_pass"] = bool(h2)
    out["H3_pass"] = bool(h3_max <= 1e-13)
    out["H4_pass"] = bool(h4_max <= 1e-8)
    out["H5_pass"] = bool(out["H5_state_agreement"] >= 0.999)
    results.update(out)


# --------------------------------------------------------------------------
# H6-H7: footprints
# --------------------------------------------------------------------------

def _read_fp(path):
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 4],
                     names=["chrom", "start", "end", "tag_count"])
    return df


def _bp_jaccard(a, b):
    def cover(df):
        s = {}
        for c, x, y in df[["chrom", "start", "end"]].itertuples(index=False):
            s.setdefault(c, []).append((x, y))
        return s
    ca, cb = cover(a), cover(b)
    inter = union = 0
    for c in set(ca) | set(cb):
        n = max([y for _, y in ca.get(c, []) + cb.get(c, [])]) + 1
        ma = np.zeros(n, bool)
        mb = np.zeros(n, bool)
        for x, y in ca.get(c, []):
            ma[x:y] = True
        for x, y in cb.get(c, []):
            mb[x:y] = True
        inter += int(np.sum(ma & mb))
        union += int(np.sum(ma | mb))
    return inter / max(union, 1)


def gate_h6_h7(results, n_jobs):
    WORK.mkdir(parents=True, exist_ok=True)
    bias = _bias()
    out = {}
    ok6 = ok7 = True
    for model in MODELS:
        for sample in SAMPLES:
            t0 = time.time()
            path = WORK / f"fp_{model}_{sample}.bed"
            H.hint_footprints(DATA / f"{sample}.bam", DATA / "merged_peaks.bed", bias=bias,
                              paired_end=(model == "paired"), n_jobs=n_jobs, output=path,
                              verbose=False)
            mine = _read_fp(path)
            ref = _read_fp(REF / f"fp_{model}" / f"{sample}.bed")
            dt = time.time() - t0
            key = ["chrom", "start", "end"]
            m = ref.merge(mine, on=key, how="inner", suffixes=("_ref", "_ep"))
            jac = _bp_jaccard(ref, mine)
            f_ref = len(m) / max(len(ref), 1)
            f_ep = len(m) / max(len(mine), 1)
            tc_ok = bool((m["tag_count_ref"] == m["tag_count_ep"]).all())
            out[f"H6_{sample}_{model}"] = {"n_ref": len(ref), "n_epione": len(mine), "bp_jaccard": jac,
                                           "frac_ref_matched": f_ref, "frac_epione_matched": f_ep,
                                           "seconds": round(dt, 1)}
            out[f"H7_{sample}_{model}_tag_count_identical"] = tc_ok
            ok6 &= jac >= 0.99 and f_ref >= 0.99 and f_ep >= 0.99
            ok7 &= tc_ok
            print(f"  footprints {sample}/{model}: {out[f'H6_{sample}_{model}']}", flush=True)
    out["H6_pass"] = bool(ok6)
    out["H7_pass"] = bool(ok7)
    results.update(out)


# --------------------------------------------------------------------------
# H8: motif matching
# --------------------------------------------------------------------------

def _read_mpbs(path):
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3, 4, 5],
                       names=["chrom", "start", "end", "name", "score", "strand"],
                       float_precision="round_trip")


def gate_h8(results):
    WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    ok = True
    for sample in SAMPLES:
        path = WORK / f"{sample}_mpbs.bed"
        H.hint_match_motifs(REF / "fp_paired" / f"{sample}.bed", DATA / "motifs.jaspar",
                            DATA / "genome.fa", output=path)
        mine, ref = _read_mpbs(path), _read_mpbs(REF / "match" / f"{sample}_mpbs.bed")
        key = ["chrom", "start", "end", "name", "strand"]
        a = set(map(tuple, ref[key].values.tolist()))
        b = set(map(tuple, mine[key].values.tolist()))
        jac = len(a & b) / max(len(a | b), 1)
        m = ref.merge(mine, on=key, suffixes=("_ref", "_ep"))
        err = float(np.max(np.abs(m["score_ref"] - m["score_ep"]))) if len(m) else 0.0
        out[f"H8_{sample}"] = {"n_ref": len(ref), "n_epione": len(mine), "jaccard": jac, "score_max_abs_err": err}
        ok &= jac >= 0.999 and err <= 1e-6
        print(f"  matching {sample}: {out[f'H8_{sample}']}", flush=True)
    out["H8_pass"] = bool(ok)
    results.update(out)


# --------------------------------------------------------------------------
# H9: differential on the reference MPBS
# --------------------------------------------------------------------------

def _read_stats(path):
    return pd.read_csv(path, sep="\t", float_precision="round_trip").set_index("Motif")


def _compare_diff(res, refdir, tag):
    from scipy.stats import pearsonr, spearmanr
    ref = _read_stats(refdir / "differential_statistics.txt")
    mine = res.stats
    common = ref.index.intersection(mine.index)
    r, m = ref.loc[common], mine.loc[common]
    o = {"n_motifs_ref": len(ref), "n_motifs_epione": len(mine), "n_common": len(common)}
    o["num_identical"] = bool(len(common) == len(ref) == len(mine) and (r["Num"] == m["Num"]).all())
    fac = pd.read_csv(refdir / "differential_factor.txt", sep="\t", float_precision="round_trip")
    rf = fac.iloc[0][res.conditions].astype(float).values
    o["factor_max_rel_err"] = float(np.max(np.abs(res.factors.values.astype(np.float64) - rf) / np.abs(rf)))
    cols_ps = [c for c in ref.columns if c.startswith("Protection_Score_")]
    cols_tc = [c for c in ref.columns if c.startswith("TC_")]
    o["ps_tc_min_pearson"] = float(min(pearsonr(r[c], m[c])[0] for c in cols_ps + cols_tc))
    o["ps_sign_agreement"] = float(min((np.sign(r[c]) == np.sign(m[c])).mean() for c in cols_ps))
    o["ps_tc_max_abs_err"] = float(max(np.max(np.abs(r[c] - m[c])) for c in cols_ps + cols_tc))
    # Upstream writes float32 values with str(); compared as float32 they
    # are either the same number or not (the float64 error above is mostly
    # the 8-digit text representation).
    o["ps_tc_float32_identical"] = bool(all(
        (r[c].astype(np.float32).values == m[c].astype(np.float32).values).all() for c in cols_ps + cols_tc))
    o["factors_float32_identical"] = bool((rf.astype(np.float32) == res.factors.values).all())
    o["activity_pearson"] = float(pearsonr(r["TF_Activity"], m["TF_Activity"])[0])
    o["activity_sign_agreement"] = float((np.sign(r["TF_Activity"]) == np.sign(m["TF_Activity"])).mean())
    o["activity_max_abs_err"] = float(np.max(np.abs(r["TF_Activity"] - m["TF_Activity"])))
    o["neglog10p_spearman"] = float(spearmanr(-np.log10(r["P_values"]), -np.log10(m["P_values"]))[0])
    o["sig_set_identical"] = bool(set(r.index[r["P_values"] < 0.05]) == set(m.index[m["P_values"] < 0.05]))
    o["n_sig_ref"] = int((r["P_values"] < 0.05).sum())
    # profiles (--output-profiles writes <condition>_<motif>.txt)
    pmin, pmax_err, p32 = 1.0, 0.0, True
    for name in common:
        for i, c in enumerate(res.conditions):
            f = refdir / f"{c}_{name}.txt"
            rp = np.array([float(x) for x in open(f).read().split()])
            mp = res.profiles[name][i].astype(np.float64)
            if np.std(rp) > 0 and np.std(mp) > 0:
                pmin = min(pmin, float(pearsonr(rp, mp)[0]))
            pmax_err = max(pmax_err, float(np.max(np.abs(rp - mp))))
            p32 &= bool(np.array_equal(rp.astype(np.float32), res.profiles[name][i]))
    o["profile_min_pearson"] = pmin
    o["profile_max_abs_err"] = pmax_err
    o["profiles_float32_identical"] = p32
    o["activity_z_p_identical"] = bool(all((r[c] == m[c]).all() for c in ("TF_Activity", "Z_score", "P_values")))
    o["pass"] = {
        "H9a": o["num_identical"],
        "H9b": o["factor_max_rel_err"] <= 1e-6,
        "H9c": o["ps_tc_min_pearson"] >= 0.9999 - 1e-12 and o["ps_sign_agreement"] == 1.0,
        "H9d": o["activity_pearson"] >= 0.9999 - 1e-12 and o["activity_sign_agreement"] == 1.0,
        "H9e": o["neglog10p_spearman"] >= 0.999 and o["sig_set_identical"],
        "H9f": pmin >= 0.9999 - 1e-12,
    }
    print(f"  differential {tag}: {o}", flush=True)
    return o


def gate_h9(results, n_jobs):
    out = {}
    allpass = True
    mpbs = [REF / "match" / f"{s}_mpbs.bed" for s in SAMPLES]
    bams = [DATA / f"{s}.bam" for s in SAMPLES]
    bias = _bias()
    for tag, bc in (("bc", True), ("raw", False)):
        t0 = time.time()
        res = H.hint_differential(mpbs, bams, DATA / "genome.fa", conditions=list(SAMPLES),
                                  bias_correction=bc, bias=bias if bc else None, n_jobs=n_jobs,
                                  verbose=False)
        o = _compare_diff(res, REF / f"diff_{tag}", tag)
        o["seconds"] = round(time.time() - t0, 1)
        out[f"H9_{tag}"] = o
        allpass &= all(o["pass"].values())
    for g in ("H9a", "H9b", "H9c", "H9d", "H9e", "H9f"):
        out[f"{g}_pass"] = bool(all(out[f"H9_{t}"]["pass"][g] for t in ("bc", "raw")))
    results.update(out)


# --------------------------------------------------------------------------
# H10: end to end
# --------------------------------------------------------------------------

def gate_h10(results, n_jobs):
    from scipy.stats import pearsonr
    WORK.mkdir(parents=True, exist_ok=True)
    bias = _bias()
    mpbs = []
    for s in SAMPLES:
        fp = H.hint_footprints(DATA / f"{s}.bam", DATA / "merged_peaks.bed", bias=bias,
                               paired_end=True, n_jobs=n_jobs, verbose=False)
        mpbs.append(H.hint_match_motifs(fp, DATA / "motifs.jaspar", DATA / "genome.fa"))
    res = H.hint_differential(mpbs, [DATA / f"{s}.bam" for s in SAMPLES], DATA / "genome.fa",
                              conditions=list(SAMPLES), bias=bias, n_jobs=n_jobs, verbose=False)
    ref = _read_stats(REF / "diff_bc" / "differential_statistics.txt")
    common = ref.index.intersection(res.stats.index)
    r, m = ref.loc[common, "TF_Activity"], res.stats.loc[common, "TF_Activity"]
    o = {"n_common": len(common), "pearson": float(pearsonr(r, m)[0]),
         "sign_agreement": float((np.sign(r) == np.sign(m)).mean())}
    o["pass"] = o["pearson"] >= 0.99 and o["sign_agreement"] >= 0.95
    results["H10"] = o
    results["H10_pass"] = bool(o["pass"])
    print(f"  end to end: {o}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="H1,H6,H8,H9,H10")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if not REF.is_dir() or not DATA.is_dir():
        sys.exit("set EPIONE_HINT_PARITY_DIR and EPIONE_HINT_DATA")
    gates = a.gates.split(",")
    results = {}
    if "H1" in gates:
        gate_h1_to_h5(results)
    if "H6" in gates:
        gate_h6_h7(results, a.n_jobs)
    if "H8" in gates:
        gate_h8(results)
    if "H9" in gates:
        gate_h9(results, a.n_jobs)
    if "H10" in gates:
        gate_h10(results, a.n_jobs)
    summary = {k: v for k, v in results.items() if k.endswith("_pass")}
    print(json.dumps(summary, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1, default=float))
    return results


if __name__ == "__main__":
    main()
