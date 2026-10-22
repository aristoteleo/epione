"""Gates L1-L5 (see MANIFEST.md): epione loops vs HCEA diff_loop/all.

python l_gates.py --work DIR --nproc N
"""
import argparse
import json
import time
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

from epione.single import hic

ENTEX = Path("/large_storage/zhoulab/zhoujt/project/ENTEx/loop/majortype")
REF = Path("/large_storage/zhoulab/shengmao/Hic_data/HCEA/analysis/diff_loop/all")
BLACKLIST = "/large_storage/zhoulab/ref/blacklist/hg38_bismark_loop_blacklist.bed"
# HCEA's order: rows of L1color.tsv without c35, c36 (Amendment L-1)
TYPES = [c for c in pd.read_csv("/large_storage/zhoulab/shengmao/Hic_data/HCEA/metadata/L1color.tsv",
                                sep="\t", index_col=0).index if c not in ("c35", "c36")]
CHROMS = [f"chr{i}" for i in range(1, 23)]


def compare_cools(a, b):
    out = {}
    A, B = cooler.Cooler(str(a)), cooler.Cooler(str(b))
    worst, same_px, bit = 0.0, True, True
    for c in CHROMS:
        pa = A.pixels().fetch(c)
        pb = B.pixels().fetch(c)
        pa = pa.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
        pb = pb.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
        if len(pa) != len(pb) or not (pa[["bin1_id", "bin2_id"]].values == pb[["bin1_id", "bin2_id"]].values).all():
            same_px = False
            out[c] = f"pixels differ: {len(pa)} vs {len(pb)}"
            continue
        x, y = pa["count"].to_numpy(np.float64), pb["count"].to_numpy(np.float64)
        nan = np.isnan(x) | np.isnan(y)
        if not (np.isnan(x) == np.isnan(y)).all():
            same_px = False
        fin = ~nan & np.isfinite(x) & np.isfinite(y)
        bit &= bool(np.array_equal(x[~nan], y[~nan]))
        if fin.any():
            rel = np.abs(x[fin] - y[fin]) / np.maximum(np.abs(y[fin]), 1e-30)
            worst = max(worst, float(rel.max()))
    out.update({"same_pixels": same_px, "bit_identical": bit, "max_rel": worst})
    return out


def main(work, nproc):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    res = {}
    prefixes = {ct: ENTEX / ct / ct for ct in TYPES}

    t = time.time()
    anova = hic.loop_anova(prefixes, work / "anova", chromosomes=CHROMS, black_list=BLACKLIST,
                           black_list_pad=7, min_dist=5, max_dist=500, nproc=nproc)
    res["seconds_anova"] = round(time.time() - t, 1)
    res["L1"] = {m: compare_cools(anova[m], REF / f"majortype_{m}pv.cool") for m in "QET"}
    print("L1", json.dumps(res["L1"]), flush=True)

    loops = hic.merge_loops({ct: ENTEX / ct / f"{ct}.loop.bedpe" for ct in TYPES}, chromosomes=CHROMS)
    ref_loops = pd.read_csv(REF / "merged_loop.bedpe", sep="\t", header=None)
    mine = loops[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].to_numpy()
    res["L2"] = {"rows": [len(mine), len(ref_loops)],
                 "identical": bool(len(mine) == len(ref_loops) and (mine == ref_loops.to_numpy()).all())}
    print("L2", res["L2"], flush=True)

    res["L3"] = {}
    for m, f in (("Q", "loop_Q.hdf"), ("T", "loop_T.hdf")):
        vals = hic.loop_values(loops, {ct: f"{prefixes[ct]}.{m}.cool" for ct in TYPES}, nproc=nproc)
        ref = pd.read_hdf(REF / f, key="data")
        res["L3"][m] = {"identical": bool(np.array_equal(vals[TYPES].to_numpy(), ref[TYPES].to_numpy(),
                                                         equal_nan=True))}
    print("L3", res["L3"], flush=True)

    n_cells = int(sum(cooler.Cooler(f"{prefixes[ct]}.Q.cool").info["group_n_cells"] for ct in TYPES))
    diff = hic.differential_loops(loops, anova, n_groups=35, n_cells=n_cells, nproc=nproc)
    ref = pd.read_hdf(REF / "merged_loop.hdf", key="data")
    res["L4"], res["L5"] = {}, {}
    for m in "QET":
        a, b = diff[f"{m}anova"].to_numpy(), ref[f"{m}anova"].to_numpy()
        fin = np.isfinite(a) & np.isfinite(b)
        res["L4"][m] = {"bit_identical": bool(np.array_equal(a, b, equal_nan=True)),
                        "max_rel": float((np.abs(a[fin] - b[fin]) / np.maximum(np.abs(b[fin]), 1e-30)).max())}
        a, b = diff[f"{m}fdr"].to_numpy(), ref[f"{m}fdr"].to_numpy()
        fin = np.isfinite(a) & np.isfinite(b) & (b >= 1e-270)          # Amendment L-1
        res["L5"][m] = {"same_nan": bool((np.isnan(a) == np.isnan(b)).all()),
                        "n_compared": int(fin.sum()), "n_below_1e-270": int((b < 1e-270).sum()),
                        "max_rel": float((np.abs(a[fin] - b[fin]) / b[fin]).max()) if fin.any() else 0.0}
    res["n_cells"] = n_cells
    res["n_differential"] = int(diff["differential"].sum())
    print("L4", res["L4"], "L5", res["L5"], flush=True)
    (Path(__file__).parent / "results").mkdir(exist_ok=True)
    (Path(__file__).parent / "results" / "loops_L.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()
    main(a.work, a.nproc)
