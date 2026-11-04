"""Gates D8 and D9 (MANIFEST.md, Amendment D-3), and a rerun of D7.

D8: ``cov_mcd`` vs ``robust::covRob(x)`` in its ``mcd`` branch after
``set.seed(s)``: raw centre and covariance, and R's ``.Random.seed`` afterwards.
D9: ``dchic_analyze`` vs ``dchicf.r --pcatype analyze`` on HCEA's published raw
compartment scores of the six Hema major types (covRob mixes ``donostah`` and
``mcd`` across chromosomes, so the R random stream is threaded end to end).
D7 (three types, all ``donostah``) is rerun with the same comparison.

Usage::

    python covmcd_parity.py [--comp-hdf PATH] [--dev DIR]

The R side runs in the ``dchic`` env. Matrices are exchanged as raw float64.
Writes ``results/covmcd_D8.json`` and ``results/dchic_D9.json``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dchic_parity import DCHICF, DEV, R_ENV, _read_table, _rscript, _write_bin, gates_d3_d5  # noqa: E402

from epione.bulk.hic import _dchic as dc  # noqa: E402
from epione.bulk.hic._covmcd import cov_mcd  # noqa: E402
from epione.bulk.hic._r_rng import RRandom  # noqa: E402

HEMA = ["c0", "c1", "c7", "c15", "c24", "c27"]
RANDOM = [(1000, 5), (1500, 6), (2500, 8), (3000, 12), (4000, 15), (5000, 19)]


def write_dchic_inputs(comp: pd.DataFrame, types, wd: Path):
    """dcHiC inputs as gate D7 wrote them: one pc.bedGraph per type and chromosome."""
    wd.mkdir(parents=True, exist_ok=True)
    rows = []
    for ct in types:
        for c, g in comp.groupby("chrom", sort=False):
            p = wd / f"{ct}_100Kb_pca" / "intra_pca" / f"{ct}_100Kb_mat"
            p.mkdir(parents=True, exist_ok=True)
            g[["chrom", "start", "end", ct]].to_csv(p / f"{c}.pc.bedGraph", sep="\t", header=False, index=False)
        rows.append([".", ".", f"{ct}_100Kb", ct])
    pd.DataFrame(rows).to_csv(wd / "input.txt", sep="\t", header=False, index=False)


def run_r_analyze(wd: Path) -> float:
    if (wd / "DifferentialResult").exists():
        shutil.rmtree(wd / "DifferentialResult")
    env = dict(os.environ, PATH=f"{R_ENV}/bin:" + os.environ["PATH"], CONDA_PREFIX=str(R_ENV))
    t = time.perf_counter()
    with open(wd / "run.log", "w") as log:
        subprocess.run([str(R_ENV / "bin" / "Rscript"), DCHICF,
                        "--file", "input.txt", "--pcatype", "analyze", "--dirovwt", "T", "--diffdir", "all"],
                       cwd=wd, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    return time.perf_counter() - t


def run_epi_analyze(wd: Path) -> float:
    if (wd / "DifferentialResult").exists():
        shutil.rmtree(wd / "DifferentialResult")
    t = time.perf_counter()
    dc.dchic_analyze(wd / "input.txt", wd, diffdir="all", dirovwt=True)
    return time.perf_counter() - t


def end_to_end(ref: Path, epi: Path, r_seconds=None) -> dict:
    e_s = run_epi_analyze(epi)
    res = gates_d3_d5(ref / "DifferentialResult" / "all", epi / "DifferentialResult" / "all")
    res["speed"] = {"epione_s": round(e_s, 1), "R_s": None if r_seconds is None else round(r_seconds, 1)}
    res["pass"] = bool(res["D3"]["pass"] and res["D4"]["pass"] and res["D5"]["pass"])
    return res


def gate_d8(work: Path, ref6: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    names = []
    g = np.random.default_rng(2026)
    for n, p in RANDOM:
        x = g.standard_normal((n, p)) @ g.standard_normal((p, p))
        k = n // 10
        x[:k] += 4 * g.standard_normal(p)          # 10% shifted cluster
        nm = f"rand_{n}x{p}"
        _write_bin(work / f"{nm}.bin", x)
        names.append(nm)
    # the group matrices dcHiC passes to covRob for chr1 and chr2 (six Hema types):
    # condition means of the quantile-normalised scores, as R reads them back
    for c in ["chr1", "chr2"]:
        t = _read_table(ref6 / "DifferentialResult" / "all" / "pcQnm" / f"intra_sample_{c}_combined.pcQnm.bedGraph")
        x = t[[f"{ct}_100Kb" for ct in HEMA]].to_numpy(np.float64)
        nm = f"hema6_{c}"
        _write_bin(work / f"{nm}.bin", x)
        names.append(nm)
    cases = [(nm, s) for nm in names for s in (123, 21)]
    pd.DataFrame(cases).to_csv(work / "cases.tsv", sep="\t", header=False, index=False)
    _rscript(HERE / "covmcd_make_ref.R", work, work / "cases.tsv")
    tim = pd.read_csv(work / "timings_R.tsv", sep="\t")
    out, worst = {}, 0.0
    for (nm, s), (_, trow) in zip(cases, tim.iterrows()):
        dim = [int(v) for v in (work / f"{nm}.dim").read_text().split()]
        x = np.fromfile(work / f"{nm}.bin", dtype="<f8").reshape(dim, order="F")
        pre = work / f"{nm}_s{s}"
        rc = np.fromfile(f"{pre}.center.bin", dtype="<f8")
        rv = np.fromfile(f"{pre}.cov.bin", dtype="<f8").reshape(dim[1], dim[1], order="F")
        rs = np.fromfile(f"{pre}.seed.bin", dtype="<i4")
        rng = RRandom(s)
        t0 = time.perf_counter()
        c, v, det = cov_mcd(x, rng=rng, return_details=True)
        e_s = time.perf_counter() - t0
        ec = float(np.max(np.abs(c - rc) / np.maximum(np.abs(rc), 1e-300)))
        ev = float(np.max(np.abs(v - rv) / np.maximum(np.abs(rv), 1e-300)))
        same_seed = bool(np.array_equal(rng.random_seed(), rs))
        worst = max(worst, ec, ev)
        out[f"{nm}_s{s}"] = {"n": dim[0], "p": dim[1], "R_estim": trow["estim"], "center_max_rel_err": ec,
                             "cov_max_rel_err": ev, "center_identical": bool(np.array_equal(c, rc)),
                             "rng_state_identical": same_seed, "draws": det["draws"],
                             "seconds_epione": round(e_s, 3), "seconds_R": float(trow["seconds"])}
    ok = worst <= 1e-10 and all(r["rng_state_identical"] and r["R_estim"] == "mcd" for r in out.values())
    return {"D8": {"pass": bool(ok), "max_rel_err": worst, "cases": out}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp-hdf", type=Path,
                    default=Path(os.environ.get("EPIONE_HCEA_COMP_RAW", "comp.raw.hdf")),
                    help="HCEA analysis/compartment/L1/comp.raw.hdf (or set EPIONE_HCEA_COMP_RAW)")
    ap.add_argument("--dev", type=Path, default=DEV)
    a = ap.parse_args()

    comp = pd.read_hdf(a.comp_hdf, key="data")
    comp = comp.loc[comp["raw_binfilter"], ["chrom", "start", "end", *HEMA]].reset_index(drop=True)
    d9 = a.dev / "dchic_6hema"
    write_dchic_inputs(comp, HEMA, d9 / "ref")
    write_dchic_inputs(comp, HEMA, d9 / "epi")
    r_s = run_r_analyze(d9 / "ref")

    res8 = gate_d8(a.dev / "covmcd", d9 / "ref")
    res8["date"] = "2026-10-20"
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "covmcd_D8.json").write_text(json.dumps(res8, indent=2, default=float) + "\n")
    print("D8", "PASS" if res8["D8"]["pass"] else "FAIL", "max rel err", res8["D8"]["max_rel_err"])

    res9 = end_to_end(d9 / "ref", d9 / "epi", r_s)
    # which estimator R used, chromosome by chromosome (from the group sizes)
    est = {}
    for f in sorted((d9 / "ref" / "DifferentialResult" / "all" / "pcQnm").glob("intra_sample_*_combined.pcQnm.bedGraph")):
        n = sum(1 for _ in open(f)) - 1
        est[re.search(r"intra_sample_(chr[^_]+)_", f.name).group(1)] = "donostah" if n < 1000 else "mcd"
    res9["covRob_first_call_estimator"] = est
    d7 = a.dev / "dchic_3type"
    res7 = end_to_end(d7 / "ref", d7 / "epi")
    out = {"date": "2026-10-20", "D9": res9, "D7_rerun": res7}
    (HERE / "results" / "dchic_D9.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    for k, r in (("D9", res9), ("D7 rerun", res7)):
        print(k, "PASS" if r["pass"] else "FAIL", {g: r[g]["pass"] for g in ("D3", "D4", "D5")},
              "padj<0.1", r["D4"]["padj_lt_0.1"], "byte-identical", r["byte_identical_files"]["identical"],
              "/", r["byte_identical_files"]["of"], "speed", r["speed"])
        for c in ("sample_maha", "pval", "padj"):
            print("   ", c, r["D4"][c])


if __name__ == "__main__":
    main()
