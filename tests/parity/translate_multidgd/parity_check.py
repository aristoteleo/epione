"""multiDGD metric-level parity: Epione runs and the gates of MANIFEST.md.

    python parity_check.py train --out RUNS --modes noshuffle [--seeds 0,1,2] [--device cuda]
    python parity_check.py gates --out RUNS [--json results.json]

``EPIONE_TRANSLATE_PARITY_DIR`` points at the reference directory (contract
inputs, split and ``ref/multiDGD``); ``EPIONE_PBMC10K_DIR`` at the raw scGLUE
h5ads (fetched with ``epi.datasets.pbmc10k_multiome`` when unset). Each run
writes source-only translations (``rna2atac.h5ad``, ``atac2rna.h5ad``) and
joint reconstructions (``rna2atac_joint.h5ad``, ``atac2rna_joint.h5ad``).
Nothing here is written into the repo.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "py_parity_translate_scbutterfly", _here.parent / "translate_scbutterfly" / "parity_check.py")
_sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sb)

DIRS = _sb.DIRS
METRIC_MAP, FLOORS, G1_TOL = _sb.METRIC_MAP, _sb.FLOORS, _sb.G1_TOL
_tol, _evaluate_file = _sb._tol, _sb._evaluate_file
MODES = {"default": {}, "noshuffle": {"shuffle": False}, "fast": {"fast": True}}


def _parity_dir() -> Path:
    d = os.environ.get("EPIONE_TRANSLATE_PARITY_DIR")
    if not d:
        raise SystemExit("set EPIONE_TRANSLATE_PARITY_DIR")
    return Path(d)


def load_inputs(P: Path):
    import epione as epi
    import epione.translate as T

    rna, atac = epi.datasets.pbmc10k_multiome(os.environ.get("EPIONE_PBMC10K_DIR"))
    rna, atac = T.prepare_counts(rna, atac)
    pos = {n: i for i, n in enumerate(rna.obs_names)}
    split = {}
    for s in ("train", "valid", "test"):
        names = [l.strip() for l in open(P / "split" / "fold0" / f"{s}.txt") if l.strip()]
        split[s] = np.array([pos[n] for n in names])
    return rna, atac, split


def cmd_train(args):
    import torch
    import epione.translate as T

    P = _parity_dir()
    rna, atac, split = load_inputs(P)
    out = Path(args.out)
    if args.threads:
        torch.set_num_threads(args.threads)
    for seed in [int(s) for s in args.seeds.split(",")]:
        for mode in args.modes.split(","):  # interleaved per seed, so speed comparisons share conditions
            d = out / mode / f"seed{seed}"
            if (d / "run.json").exists():
                print("skip", d)
                continue
            d.mkdir(parents=True, exist_ok=True)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            model = T.fit(rna, atac, method="multidgd", train_idx=split["train"], val_idx=split["valid"],
                          device=args.device, seed=seed, **MODES[mode])
            train_s = time.time() - t0
            te = split["test"]
            t0 = time.time()
            model.predict(rna[te], "rna2atac", device=args.device).write_h5ad(d / "rna2atac.h5ad")
            model.predict(atac[te], "atac2rna", device=args.device).write_h5ad(d / "atac2rna.h5ad")
            jr, ja = model.reconstruct(rna[te], atac[te], device=args.device)
            ja.write_h5ad(d / "rna2atac_joint.h5ad")
            jr.write_h5ad(d / "atac2rna_joint.h5ad")
            test_s = time.time() - t0
            h = model.history
            run = {"mode": mode, "seed": seed, "train_seconds": train_s, "test_seconds": test_s,
                   "device": args.device, "torch": torch.__version__, "threads": args.threads,
                   "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
                   "epochs": h["epochs"], "stopped_at_epoch": h["stopped_at_epoch"]}
            (d / "run.json").write_text(json.dumps(run, indent=1))
            print(json.dumps(run), flush=True)


def cmd_gates(args):
    import scanpy as sc

    P = _parity_dir()
    R = next((P / "ref" / n for n in ("multiDGD", "multidgd") if (P / "ref" / n).exists()), P / "ref" / "multiDGD")
    out = Path(args.out)
    rna, atac, split = load_inputs(P)
    res = {"G0": {}, "G1": [], "speed": {}}
    ref_seeds = sorted(int(p.name[4:]) for p in (R / "rna2atac").glob("seed*") if (p / "metrics.json").exists())

    # ---- G0: prepare_counts equals the reference truth exactly
    truths = {}
    for d in DIRS:
        t = sc.read_h5ad(R / d / f"seed{ref_seeds[0]}" / "truth.h5ad")
        mine_ad = atac if d == "rna2atac" else rna
        truths[d] = t
        same = list(mine_ad.var_names) == list(t.var_names)
        mine = mine_ad[t.obs_names, t.var_names].X
        mine = mine.toarray() if hasattr(mine, "toarray") else np.asarray(mine)
        theirs = t.X.toarray() if hasattr(t.X, "toarray") else np.asarray(t.X)
        res["G0"][d] = {"pass": bool(same and np.array_equal(mine, theirs)), "same_features_in_order": same,
                        "max_abs_diff": float(np.max(np.abs(mine - theirs))), "shape": list(theirs.shape)}
    labels = {d: np.asarray(rna.obs["cell_type"].loc[truths[d].obs_names]) for d in DIRS}

    # ---- G1 + reference values
    ref_vals = {d: {m: [] for m in METRIC_MAP} for d in DIRS}
    for d in DIRS:
        for s in ref_seeds:
            sd = R / d / f"seed{s}"
            ref_m = json.loads((sd / "metrics.json").read_text())
            mine, std = _evaluate_file(sd / "pred.h5ad", truths[d], labels[d],
                                       out / "ref_metrics_cache" / d / f"seed{s}.json")
            res.setdefault("reference_standard_convention", []).append({"direction": d, "seed": s, **std})
            for m, em in METRIC_MAP.items():
                ref_vals[d][m].append(ref_m[m])
                tol = _tol(G1_TOL[m], ref_m[m])
                diff = abs(mine[em] - ref_m[m])
                res["G1"].append({"direction": d, "seed": s, "metric": m, "reference": ref_m[m],
                                  "epione_metric": mine[em], "abs_diff": diff, "tol": tol, "pass": bool(diff <= tol)})

    # ---- Epione runs
    def run_vals(mode, suffix=""):
        vals = {d: {m: [] for m in METRIC_MAP} for d in DIRS}
        secs = {}
        for sdir in sorted((out / mode).glob("seed*")):
            if not (sdir / "run.json").exists():
                continue
            secs[sdir.name] = json.loads((sdir / "run.json").read_text())["train_seconds"]
            for d in DIRS:
                mine, std = _evaluate_file(sdir / f"{d}{suffix}.h5ad", truths[d], labels[d],
                                           sdir / f"{d}{suffix}.metrics.json")
                res.setdefault("standard_convention", []).append(
                    {"mode": mode + suffix, "run": sdir.name, "direction": d, **std})
                for m, em in METRIC_MAP.items():
                    vals[d][m].append(mine[em])
        return vals, secs

    def gate(base, other, name):
        rows = []
        for d in DIRS:
            for m in METRIC_MAP:
                b, o = np.array(base[d][m], float), np.array(other[d][m], float)
                if len(b) == 0 or len(o) == 0:
                    continue
                sd = float(np.std(b, ddof=1)) if len(b) > 1 else 0.0
                tol = max(2 * sd, _tol(FLOORS[m], b.mean()))
                diff = float(abs(o.mean() - b.mean()))
                gated = not (m == "AUROC" and d == "atac2rna")
                rows.append({"direction": d, "metric": m, f"{name}_baseline_mean": float(b.mean()), "baseline_sd": sd,
                             "mean": float(o.mean()), "values": o.tolist(), "baseline_values": b.tolist(),
                             "abs_diff": diff, "tol": tol, "gated": gated, "pass": bool(diff <= tol)})
        return rows

    have = lambda mode: (out / mode).exists() and any((out / mode).glob("seed*/run.json"))
    res["G2"] = gate(ref_vals, run_vals("noshuffle", "_joint")[0], "reference") if have("noshuffle") else []
    res["G2_source"] = gate(ref_vals, run_vals("noshuffle")[0], "reference") if have("noshuffle") else []
    res["G2_shuffled"] = gate(ref_vals, run_vals("default", "_joint")[0], "reference") if have("default") else []
    res["G3"] = []
    if have("default") and have("fast"):
        ep_d, secs_d = run_vals("default")
        ep_f, secs_f = run_vals("fast")
        res["G3"] = gate(ep_d, ep_f, "default")
        ratios = [secs_d[k] / secs_f[k] for k in secs_d if k in secs_f]
        res["speed"] = {"default_seconds": secs_d, "fast_seconds": secs_f, "ratios": ratios,
                        "median_ratio": float(np.median(ratios)) if ratios else None,
                        "required": 1.5, "pass": bool(ratios and np.median(ratios) >= 1.5)}
    res["speed"]["reference_seconds"] = [
        json.loads((R / "rna2atac" / f"seed{s}" / "run.json").read_text()).get("train_seconds") for s in ref_seeds]

    gated = lambda rows: all(r["pass"] for r in rows if r["gated"]) if rows else None
    summary = {
        "G0": all(v["pass"] for v in res["G0"].values()),
        "G1_non_clustering": all(r["pass"] for r in res["G1"] if r["metric"] not in ("ARI", "AMI", "NMI", "HOM")),
        "G1_clustering": all(r["pass"] for r in res["G1"] if r["metric"] in ("ARI", "AMI", "NMI", "HOM")),
        "G2": gated(res["G2"]), "G2_source (not gated)": gated(res["G2_source"]),
        "G2_shuffled (not gated)": gated(res["G2_shuffled"]),
        "G3_metrics": gated(res["G3"]), "G3_speed": res["speed"].get("pass"),
    }
    res["summary"] = summary
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(summary, indent=1))
    for key in ("G2", "G2_source", "G2_shuffled", "G3"):
        for r in res[key]:
            flag = "PASS" if r["pass"] else ("fail" if r["gated"] else "(not gated)")
            base = r.get("reference_baseline_mean", r.get("default_baseline_mean"))
            print(f"{key:11s} {r['direction']:9s} {r['metric']:6s} base={base:.4f}±{r['baseline_sd']:.4f} "
                  f"epione={r['mean']:.4f} |d|={r['abs_diff']:.4f} tol={r['tol']:.4f} {flag}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--modes", default="noshuffle")
    t.add_argument("--seeds", default="0,1,2")
    t.add_argument("--device", default="cuda")
    t.add_argument("--threads", type=int, default=None)
    g = sub.add_parser("gates")
    g.add_argument("--out", required=True)
    g.add_argument("--json", default=None)
    a = ap.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_gates(a)


if __name__ == "__main__":
    main()
