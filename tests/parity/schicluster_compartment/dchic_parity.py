"""Gates D1-D5 of MANIFEST.md: the dcHiC ``analyze`` port against R.

Fixture D: the 34 + 1 HCEA major-type compartment tracks of
``analysis/diff_comp/all`` (copied to ``REF``), with a fresh
``Rscript dchicf.r --file input.txt --pcatype analyze --dirovwt T --diffdir all``
already run there (R 4.1.3, conda env ``dchic``). The Python driver runs in
``WORK`` (symlinks to the same ``*_pca`` inputs).

Usage::

    python dchic_parity.py [--ref DIR] [--work DIR] [--tmp DIR]

Writes ``results/dchic_D.json``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import filecmp
import json
import re
import shutil
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from epione.bulk.hic import _dchic as dc

HERE = Path(__file__).resolve().parent
DEV = Path("/large_storage/zhoulab/shengmao/epione_hic_dev")
R_ENV = Path("/home/shengmao/.local/share/mamba/envs/dchic")


def _rscript(*args):
    env = dict(os.environ, PATH=f"{R_ENV}/bin:" + os.environ["PATH"], CONDA_PREFIX=str(R_ENV))
    subprocess.run([str(R_ENV / "bin" / "Rscript"), *map(str, args)], check=True, env=env)


def _write_bin(path: Path, x: np.ndarray):
    x = np.asarray(x, dtype="<f8")
    x.ravel(order="F").tofile(path)
    path.with_suffix(".dim").write_text(f"{x.shape[0]} {x.shape[1]}\n")


def _read_bin(path: Path, shape):
    return np.fromfile(path, dtype="<f8").reshape(shape, order="F")


def _read_table(path: Path) -> pd.DataFrame:
    """A dcHiC table as R's read.table(h=T) sees it (numbers via R's parser)."""
    raw = pd.read_csv(path, sep="\t", dtype=str)
    out = raw[["chr"]].copy()
    for c in raw.columns[1:]:
        out[c] = dc.r_strtod(raw[c].tolist())
    return out


def gate_d1_d2(ref_out: Path, tmp: Path) -> dict:
    d1 = tmp / "d1_in"
    d2 = tmp / "d2_in"
    for d in (d1, d2, tmp / "d1_ref", tmp / "d2_ref"):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    rng = np.random.default_rng(20261020)
    mats1 = {}
    for f in sorted((ref_out / "pcQnm").glob("intra_sample_*_combined.pcQnm.bedGraph")):
        chrom = re.search(r"intra_sample_(.+)_combined", f.name).group(1)
        t = _read_table(f)
        mats1[f"hcea_{chrom}"] = t.iloc[:, 3:].to_numpy(np.float64)
    mats1["rand_t3_300x20"] = rng.standard_t(3, size=(300, 20))
    mats1["rand_t3_1500x25"] = rng.standard_t(3, size=(1500, 25)) @ rng.normal(size=(25, 25))
    mats1["rand_norm_120x40"] = rng.normal(size=(120, 40)) + rng.normal(size=(120, 1))
    for k, m in mats1.items():
        _write_bin(d1 / f"{k}.bin", m)
    mats2 = {}
    for f in sorted((ref_out / "pcOri").glob("intra_sample_*_combined.pcOri.bedGraph")):
        chrom = re.search(r"intra_sample_(.+)_combined", f.name).group(1)
        mats2[f"hcea_{chrom}"] = _read_table(f).iloc[:, 3:].to_numpy(np.float64)
    mats2["ties_int_300x6"] = rng.integers(0, 10, size=(300, 6)).astype(np.float64)
    mats2["ties_example_5x4"] = np.array([[5, 2, 3, 4], [4, 1, 4, 2], [3, 4, 6, 8], [4, 2, 8, 4], [1, 2, 4, 3]], float)
    mats2["norm_1000x8"] = rng.normal(size=(1000, 8))
    mats2["rounded_2000x35"] = np.round(rng.normal(size=(2000, 35)), 2)
    for k, m in mats2.items():
        _write_bin(d2 / f"{k}.bin", m)
    _rscript(HERE / "dchic_make_ref.R", tmp)

    d1res = {}
    for k, m in mats1.items():
        p = m.shape[1]
        R = _read_bin(tmp / "d1_ref" / f"{k}.bin", (p, p))
        estim, rsec = (tmp / "d1_ref" / f"{k}.estim").read_text().split()
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = dc.cov_rob(m)
        sec = time.perf_counter() - t0
        rel = float(np.abs(res["cov"] - R).max() / np.abs(R).max())
        d1res[k] = {"shape": list(m.shape), "R_estim": estim, "epione_estim": res["estim"],
                    "max_rel_err": rel, "degenerate_warning": len(w) > 0,
                    "epione_s": round(sec, 3), "R_s": float(rsec)}
    d1_worst = max(v["max_rel_err"] for v in d1res.values())
    d2res = {}
    for k, m in mats2.items():
        R = _read_bin(tmp / "d2_ref" / f"{k}.bin", m.shape)
        d2res[k] = {"shape": list(m.shape), "max_abs_err": float(np.abs(dc.normalize_quantiles(m) - R).max())}
    d2_worst = max(v["max_abs_err"] for v in d2res.values())
    return {
        "D1": {"pass": d1_worst <= 1e-10, "max_rel_err": d1_worst, "matrices": d1res},
        "D2": {"pass": d2_worst <= 1e-13, "max_abs_err": d2_worst, "matrices": d2res},
    }


def _all_files(root: Path):
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def gates_d3_d5(ref_out: Path, epi_out: Path) -> dict:
    fr, fe = _all_files(ref_out), _all_files(epi_out)
    common = sorted(fr & fe)
    identical = [f for f in common if filecmp.cmp(ref_out / f, epi_out / f, shallow=False)]

    # D3: per-chromosome pcOri / pcQnm tables (global and per condition): bins and the
    # written values must be identical.
    d3_files = [f for f in common if f.startswith(("pcOri/", "pcQnm/")) or
                re.search(r"_data/intra_.+_combined\.pc(Ori|Qnm)\.bedGraph$", f)]
    d3_bad = []
    for f in d3_files:
        a = pd.read_csv(ref_out / f, sep="\t", dtype=str)
        b = pd.read_csv(epi_out / f, sep="\t", dtype=str)
        if not a.equals(b):
            d3_bad.append(f)
    # D4: statistics of the combined table
    a = pd.read_csv(ref_out / "fdr_result/differential.intra_sample_combined.pcQnm.bedGraph", sep="\t")
    b = pd.read_csv(epi_out / "fdr_result/differential.intra_sample_combined.pcQnm.bedGraph", sep="\t")
    same_bins = (a[["chr", "start", "end"]].to_numpy() == b[["chr", "start", "end"]].to_numpy()).all()
    d4 = {"same_bins": bool(same_bins)}
    ok = bool(same_bins)
    for c in ["sample_maha", "pval", "padj"]:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        if c == "sample_maha":
            m = np.ones_like(x, dtype=bool)
            both_tiny_ok = True
        else:
            m = x >= 1e-300
            both_tiny_ok = bool(np.all(y[~m] <= 1e-300))
        rel = float(np.max(np.abs(x[m] - y[m]) / np.abs(x[m]))) if m.any() else 0.0
        d4[c] = {"max_rel_err": rel, "n_below_1e-300": int((~m).sum()), "tiny_agree": both_tiny_ok}
        ok &= rel <= 1e-8 and both_tiny_ok
    sig_r, sig_e = set(np.where(a["padj"] < 0.1)[0]), set(np.where(b["padj"] < 0.1)[0])
    d4["padj_lt_0.1"] = {"R": len(sig_r), "epione": len(sig_e), "identical_set": sig_r == sig_e}
    ok &= sig_r == sig_e
    pc_cols = [c for c in a.columns if c not in ("chr", "start", "end", "sample_maha", "pval", "padj", "dist_clust")]
    d4["score_columns_identical"] = bool((a[pc_cols].to_numpy() == b[pc_cols].to_numpy()).all())
    # per-chromosome tables before adjustment
    per = {}
    for f in common:
        mm = re.match(r"fdr_result/differential\.intra_sample_(chr[^_]+)_combined\.pcQnm\.bedGraph$", f)
        if mm:
            ra = pd.read_csv(ref_out / f, sep="\t")
            rb = pd.read_csv(epi_out / f, sep="\t")
            mk = ra["pval"] >= 1e-300
            per[mm.group(1)] = {
                "sample_maha": float(np.max(np.abs(ra.sample_maha - rb.sample_maha) / ra.sample_maha.abs())),
                "pval": float(np.max(np.abs(ra.pval[mk] - rb.pval[mk]) / ra.pval[mk])) if mk.any() else 0.0,
            }
    d4["per_chromosome_max_rel_err"] = per
    d4["pass"] = bool(ok)
    # D5: file names, columns, row order
    d5_bad = []
    for f in common:
        ha = (ref_out / f).read_text().split("\n", 1)[0]
        hb = (epi_out / f).read_text().split("\n", 1)[0]
        ka = pd.read_csv(ref_out / f, sep="\t", header=None, usecols=[0, 1, 2], dtype=str)
        kb = pd.read_csv(epi_out / f, sep="\t", header=None, usecols=[0, 1, 2], dtype=str)
        if ha.split("\t") != hb.split("\t") or not ka.equals(kb):
            d5_bad.append(f)
    return {
        "D3": {"pass": not d3_bad and bool(d3_files), "n_files": len(d3_files), "mismatched": d3_bad},
        "D4": d4,
        "D5": {"pass": fr == fe and not d5_bad, "n_files_R": len(fr), "n_files_epione": len(fe),
               "only_R": sorted(fr - fe), "only_epione": sorted(fe - fr), "layout_mismatched": d5_bad},
        "byte_identical_files": {"identical": len(identical), "of": len(common),
                                 "differing": sorted(set(common) - set(identical))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, default=DEV / "dchic_ref_hcea")
    ap.add_argument("--work", type=Path, default=DEV / "dchic_epi_hcea")
    ap.add_argument("--tmp", type=Path, default=DEV / "dchic_parity_work")
    a = ap.parse_args()

    ref_out = a.ref / "DifferentialResult" / "all"
    # Python driver, fresh output
    a.work.mkdir(parents=True, exist_ok=True)
    for d in a.ref.glob("*_pca"):
        link = a.work / d.name
        if not link.exists():
            link.symlink_to(d.resolve())
    shutil.copyfile(a.ref / "input.txt", a.work / "input.txt")
    if (a.work / "DifferentialResult").exists():
        shutil.rmtree(a.work / "DifferentialResult")
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dc.dchic_analyze(a.work / "input.txt", a.work, diffdir="all", dirovwt=True)
    epi_s = time.perf_counter() - t0
    n_warn = sum(issubclass(x.category, RuntimeWarning) for x in w)
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", (a.ref / "run.log").read_text())
    parts = [float(p) for p in m.group(1).split(":")] if m else []
    r_s = sum(v * 60 ** i for i, v in enumerate(reversed(parts))) if parts else None

    res = {"date": "2026-10-20", "fixture": str(a.ref), "speed": {"epione_s": round(epi_s, 1), "R_s": r_s,
           "note": "whole analyze step, wall clock; Epione with OPENBLAS_NUM_THREADS=1"},
           "degenerate_ogk_warnings": n_warn}
    res.update(gate_d1_d2(ref_out, a.tmp))
    res.update(gates_d3_d5(ref_out, a.work / "DifferentialResult" / "all"))
    out = HERE / "results" / "dchic_D.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float) + "\n")
    for g in ("D1", "D2", "D3", "D4", "D5"):
        print(g, "PASS" if res[g]["pass"] else "FAIL")
    print("byte-identical files:", res["byte_identical_files"]["identical"], "/", res["byte_identical_files"]["of"])
    print("speed:", res["speed"])


if __name__ == "__main__":
    main()
