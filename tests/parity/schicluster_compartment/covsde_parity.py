"""Gate D6 (MANIFEST.md, Amendment D-1): ``cov_sde`` vs ``robust::covRob`` donostah.

Usage:

    python covsde_parity.py <work_dir> <results.json> [--rerun-r]

``work_dir`` receives ``real_chr1`` / ``real_chr2`` (columns c0, c1, c7 of
``L1/comp.raw.hdf`` on that chromosome, rows with ``raw_binfilter``; pass
``--comp-hdf <path>`` to (re)write them). With ``--rerun-r`` the R side
(``covsde_make_ref.R``: random matrices + covRob references + timings) is run
first in the ``dchic`` env, which must be on PATH. Matrices are exchanged as raw
float64 (column-major) so both languages see identical bits.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FILES = ["rand_300x3", "rand_900x4", "rand_2500x3", "rand_4000x4", "rand_800x9",
         "real_chr1", "real_chr2"]


def _read_bin(path, shape=None):
    v = np.fromfile(path, dtype="<f8")
    return v if shape is None else v.reshape(shape, order="F")


def _write_real(work, comp_hdf):
    comp = pd.read_hdf(comp_hdf, key="data")
    comp = comp[comp["raw_binfilter"]]
    for ch in ("chr1", "chr2"):
        x = comp.loc[comp["chrom"] == ch, ["c0", "c1", "c7"]].to_numpy(np.float64)
        x.reshape(-1, order="F").astype("<f8").tofile(work / f"real_{ch}.bin")
        (work / f"real_{ch}.dim").write_text(f"{x.shape[0]} {x.shape[1]}\n")


def main(work, results_path, rerun_r=False, comp_hdf=None):
    from epione.bulk.hic._covsde import cov_sde

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if comp_hdf is not None:
        _write_real(work, comp_hdf)
    if rerun_r:
        script = Path(__file__).with_name("covsde_make_ref.R")
        out = subprocess.run(["Rscript", str(script)], cwd=work, check=True,
                             capture_output=True, text=True).stdout
        (work / "ref_timing.txt").write_text(out)
    r_time = {}
    for line in (work / "ref_timing.txt").read_text().splitlines():
        f, n, p, estim, sec = line.split()
        r_time[f] = {"estim": estim, "seconds": float(sec)}

    rows, worst = {}, 0.0
    for name in FILES:
        n, p = (int(v) for v in (work / f"{name}.dim").read_text().split())
        x = _read_bin(work / f"{name}.bin", (n, p))
        c_ref = _read_bin(work / f"{name}.center.bin")
        s_ref = _read_bin(work / f"{name}.cov.bin", (p, p))
        t = time.time()
        c, s, det = cov_sde(x, return_details=True)
        sec = time.time() - t
        rel_c = float(np.max(np.abs(c - c_ref)) / np.max(np.abs(c_ref)))
        rel_s = float(np.max(np.abs(s - s_ref)) / np.max(np.abs(s_ref)))
        elem_s = float(np.max(np.abs(s - s_ref) / np.abs(s_ref)))
        worst = max(worst, rel_c, rel_s)
        rows[name] = {"shape": list(x.shape), "r_estim": r_time[name]["estim"],
                      "max_rel_err_center": rel_c, "max_rel_err_cov": rel_s,
                      "max_elementwise_rel_err_cov": elem_s,
                      "identical_center": bool(np.array_equal(c, c_ref)),
                      "identical_cov": bool(np.array_equal(s, s_ref)),
                      "nresper": det["nresper"],
                      "seconds_epi": sec, "seconds_r": r_time[name]["seconds"]}
        print(name, rows[name])
    res = {"D6": {"per_matrix": rows, "max_rel_err": worst, "threshold": 1e-10,
                  "pass": bool(worst <= 1e-10)}}
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_path).write_text(json.dumps(res, indent=2))
    print("D6 pass:", res["D6"]["pass"], "max rel err:", worst)


if __name__ == "__main__":
    a = sys.argv[3:]
    main(sys.argv[1], sys.argv[2], "--rerun-r" in a,
         a[a.index("--comp-hdf") + 1] if "--comp-hdf" in a else None)
