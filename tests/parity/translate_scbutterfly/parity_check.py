"""Metric-level parity of epione.translate methods: Epione runs and the gates of MANIFEST.md.

    python parity_check.py [--method babel] train --out RUNS [--modes default,fast] [--seeds 0,1,2]
    python parity_check.py [--method babel] gates --out RUNS [--json results.json]

Used by tests/parity/translate_scbutterfly (default) and translate_babel.

``EPIONE_TRANSLATE_PARITY_DIR`` points at the reference directory (contract
inputs, split and ``ref/scbutterfly``). Nothing here is written into the repo.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

DIRS = ("rna2atac", "atac2rna")
METRIC_MAP = {"PCC": "PCC", "SPCC": "SPCC", "MSE": "MSE", "MAE": "MAE", "CORR": "CORR", "MMD": "MMD",
              "LISI": "iLISI", "AUROC": "AUROC", "ARI": "ARI", "AMI": "AMI", "NMI": "NMI", "HOM": "HOM"}
# G2/G3 floors: (kind, value); kind "abs" or "rel" (fraction of the baseline mean)
FLOORS = {"PCC": ("abs", 0.01), "SPCC": ("abs", 0.01), "CORR": ("abs", 0.01), "AUROC": ("abs", 0.005),
          "MSE": ("rel", 0.05), "MAE": ("rel", 0.05), "MMD": ("rel", 0.10), "LISI": ("abs", 0.02),
          "ARI": ("abs", 0.03), "AMI": ("abs", 0.03), "NMI": ("abs", 0.03), "HOM": ("abs", 0.03)}
# fit() arguments per run mode; "fast_a1" is MANIFEST amendment A1
MODES = {"default": {}, "fast": {"fast": True}, "fast_a1": {"fast": True, "batch_size": 128, "lr": 1e-3},
         # amendment A3: scTranslation's runner trains without shuffling
         "noshuffle": {"shuffle": False}}
G1_TOL = {"PCC": ("abs", 1e-4), "SPCC": ("abs", 1e-4), "CORR": ("abs", 1e-4), "AUROC": ("abs", 1e-4),
          "MSE": ("rel", 1e-4), "MAE": ("rel", 1e-4), "MMD": ("rel", 1e-3), "LISI": ("abs", 1e-3),
          "ARI": ("abs", 0.02), "AMI": ("abs", 0.02), "NMI": ("abs", 0.02), "HOM": ("abs", 0.02)}


def _parity_dir() -> Path:
    d = os.environ.get("EPIONE_TRANSLATE_PARITY_DIR")
    if not d:
        raise SystemExit("set EPIONE_TRANSLATE_PARITY_DIR")
    return Path(d)


METHOD = "scbutterfly"
REF_DIRS = {"scbutterfly": ("scbutterfly", "scButterfly"), "babel": ("babel", "BABEL"), "scpair": ("scpair", "scPair")}


def load_inputs(P: Path):
    """Inputs as each method's reference runner gets them (contract section 2)."""
    import scanpy as sc
    import epione.translate as T

    counts = sc.read_h5ad(P / "inputs" / "atac_counts.h5ad")
    if METHOD == "babel":
        rna = sc.read_h5ad(P / "inputs" / "rna_counts_hvg.h5ad")
        atac = T.prepare_atac(counts, use_tfidf=False, scale=False)
    elif METHOD == "scpair":
        # scPair binarises peaks internally; its truth files are the raw counts
        rna = sc.read_h5ad(P / "inputs" / "rna_counts_hvg.h5ad")
        atac = T.prepare_atac(counts, binarize=False, use_tfidf=False, scale=False)
    else:
        rna = sc.read_h5ad(P / "inputs" / "rna_lognorm_hvg.h5ad")
        atac = T.prepare_atac(counts)
    assert np.array_equal(rna.obs_names, atac.obs_names)
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
    modes = args.modes.split(",")
    for seed in [int(s) for s in args.seeds.split(",")]:
        for mode in modes:  # interleaved per seed, so speed comparisons share conditions
            d = out / mode / f"seed{seed}"
            if (d / "run.json").exists():
                print("skip", d)
                continue
            d.mkdir(parents=True, exist_ok=True)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            model = T.fit(rna, atac, method=METHOD, train_idx=split["train"], val_idx=split["valid"],
                          device=args.device, seed=seed, threads=args.threads, **MODES[mode])
            train_s = time.time() - t0
            t0 = time.time()
            te = split["test"]
            model.predict(rna[te], "rna2atac").write_h5ad(d / "rna2atac.h5ad")
            model.predict(atac[te], "atac2rna").write_h5ad(d / "atac2rna.h5ad")
            test_s = time.time() - t0
            h = model.history
            run = {"mode": mode, "seed": seed, "train_seconds": train_s, "test_seconds": test_s,
                   "device": args.device, "torch": torch.__version__,
                   "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
                   "epochs": ({k: len(v) for k, v in h.items() if isinstance(v, list) and k not in ("val_parts", "lr")}
                              or {k: len(v["val"]) for k, v in h.items() if isinstance(v, dict) and "val" in v})}
            (d / "run.json").write_text(json.dumps(run, indent=1))
            print(json.dumps(run), flush=True)


def _tol(kind_val, base):
    kind, v = kind_val
    return v * abs(base) if kind == "rel" else v


def _evaluate(pred, truth, labels, convention="sctranslation"):
    """Gates use the scTranslation convention for AUROC/CORR (see MANIFEST, G1)."""
    import epione.translate as T

    return T.evaluate(pred, truth, labels=labels, convention=convention)


def _positional(names) -> bool:
    return len(names) > 0 and all(str(n) == str(i) for i, n in enumerate(names[:50])) and str(names[-1]) == str(len(names) - 1)


def _align(ad, features, cells=None):
    """Reference files may name features (scButterfly) or cells (BABEL) 0..n-1; name them by position.

    Positional cells are the test cells in split order; G0 checks the values, which confirms the order.
    """
    if _positional(ad.var_names) and ad.n_vars == len(features):
        ad = ad.copy()
        ad.var_names = list(features)
    if cells is not None and _positional(ad.obs_names) and ad.n_obs == len(cells):
        ad = ad.copy()
        ad.obs_names = list(cells)
    return ad


def _evaluate_file(path: Path, truth, labels, cache: Path):
    """Metrics of a saved prediction under both conventions, cached in ``cache`` (keyed on mtime)."""
    import scanpy as sc

    key = path.stat().st_mtime
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("mtime") == key and "standard" in c:
            return c["sctranslation"], c["standard"]
    pred = _align(sc.read_h5ad(path), truth.var_names, truth.obs_names)
    m = _evaluate(pred, truth, labels, "sctranslation")
    from epione.translate import metrics as M

    pa, ta = M._pair(pred[truth.obs_names, truth.var_names], truth)
    std = {"AUROC": M.auroc(pa, ta), "AUPRC": M.auprc(pa, ta),
           "CORR": M.regression_scores(pa, ta)["CORR"]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"mtime": key, "sctranslation": m, "standard": std}))
    return m, std


def cmd_gates(args):
    import scanpy as sc

    P = _parity_dir()
    R = next((P / "ref" / n for n in REF_DIRS[METHOD] if (P / "ref" / n).exists()), P / "ref" / REF_DIRS[METHOD][0])
    out = Path(args.out)
    rna, atac, split = load_inputs(P)
    te = split["test"]
    res = {"G0": {}, "G1": [], "G2": [], "G3": [], "speed": {}}

    truths = {}
    ref_seeds = sorted(int(p.name[4:]) for p in (R / "rna2atac").glob("seed*") if (p / "metrics.json").exists())
    # ---- G0
    for d in DIRS:
        src = atac if d == "rna2atac" else rna
        if METHOD == "babel" and d == "atac2rna":
            import epione.translate as T
            # BABEL scores ATAC->RNA against the test cells' counts, size-normalised to their own median
            src = T.size_normalize(rna[te])
        t = sc.read_h5ad(R / d / f"seed{ref_seeds[0]}" / "truth.h5ad")
        positional = _positional(t.var_names)
        t = _align(t, src.var_names, rna.obs_names[te])
        truths[d] = t
        mine = src[t.obs_names, t.var_names].X
        mine = mine.toarray() if hasattr(mine, "toarray") else np.asarray(mine)
        theirs = t.X.toarray() if hasattr(t.X, "toarray") else np.asarray(t.X)
        exact = d == "atac2rna" and METHOD == "scbutterfly"
        ok = np.array_equal(mine, theirs) if exact else np.allclose(mine, theirs, rtol=1e-5, atol=1e-7)
        res["G0"][d] = {"pass": bool(ok), "max_abs_diff": float(np.max(np.abs(mine - theirs))),
                        "reference_features_positional": positional,
                        "shape": list(theirs.shape), "same_features": bool(t.n_vars == src.n_vars)}
    labels = {d: np.asarray(truths[d].obs["cell_type"]) if "cell_type" in truths[d].obs
              else np.asarray(rna.obs["cell_type"].loc[truths[d].obs_names]) for d in DIRS}

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
    def run_vals(mode):
        vals = {d: {m: [] for m in METRIC_MAP} for d in DIRS}
        secs = {}
        for sdir in sorted((out / mode).glob("seed*")):
            if not (sdir / "run.json").exists():
                continue
            secs[sdir.name] = json.loads((sdir / "run.json").read_text())["train_seconds"]
            for d in DIRS:
                mine, std = _evaluate_file(sdir / f"{d}.h5ad", truths[d], labels[d],
                                           sdir / f"{d}.metrics.json")
                res.setdefault("standard_convention", []).append({"mode": mode, "run": sdir.name, "direction": d, **std})
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

    ep_default, secs_d = run_vals("default")
    res["G2_shuffled"] = gate(ref_vals, ep_default, "reference")
    if (out / "noshuffle").exists():
        ep_ns, _ = run_vals("noshuffle")
        res["G2"] = gate(ref_vals, ep_ns, "reference")
    else:
        res["G2"] = res["G2_shuffled"]
    if (out / args.fast_mode).exists():
        ep_fast, secs_f = run_vals(args.fast_mode)
        res["G3"] = gate(ep_default, ep_fast, "default")
        ratios = [secs_d[k] / secs_f[k] for k in secs_d if k in secs_f]
        need = 1.5 if args.fast_mode == "fast" else 1.3
        res["speed"] = {"fast_mode": args.fast_mode, "default_seconds": secs_d, "fast_seconds": secs_f,
                        "ratios": ratios, "median_ratio": float(np.median(ratios)) if ratios else None,
                        "required": need, "pass": bool(ratios and np.median(ratios) >= need)}
    ref_secs = []
    for s in ref_seeds:
        rj = R / "rna2atac" / f"seed{s}" / "run.json"
        if rj.exists():
            ref_secs.append(json.loads(rj.read_text()).get("train_seconds"))
    res["speed"]["reference_seconds"] = ref_secs

    summary = {
        "G0": all(v["pass"] for v in res["G0"].values()),
        "G1_non_clustering": all(r["pass"] for r in res["G1"] if r["metric"] not in ("ARI", "AMI", "NMI", "HOM")),
        "G1_clustering": all(r["pass"] for r in res["G1"] if r["metric"] in ("ARI", "AMI", "NMI", "HOM")),
        "G2": all(r["pass"] for r in res["G2"] if r["gated"]) if res["G2"] else None,
        "G2_shuffled": all(r["pass"] for r in res["G2_shuffled"] if r["gated"]) if res["G2_shuffled"] else None,
        "G3_metrics": all(r["pass"] for r in res["G3"] if r["gated"]) if res["G3"] else None,
        "G3_speed": res["speed"].get("pass"),
    }
    res["summary"] = summary
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(summary, indent=1))
    for key in ("G2", "G2_shuffled", "G3"):
        for r in res[key]:
            flag = "PASS" if r["pass"] else ("fail" if r["gated"] else "(not gated)")
            base = r.get("reference_baseline_mean", r.get("default_baseline_mean"))
            print(f"{key} {r['direction']:9s} {r['metric']:6s} base={base:.4f}±{r['baseline_sd']:.4f} "
                  f"epione={r['mean']:.4f} |d|={r['abs_diff']:.4f} tol={r['tol']:.4f} {flag}")


def main():
    global METHOD, MODES
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="scbutterfly", choices=sorted(REF_DIRS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--modes", default="default,fast")
    t.add_argument("--seeds", default="0,1,2")
    t.add_argument("--device", default="cuda")
    t.add_argument("--threads", type=int, default=4)
    g = sub.add_parser("gates")
    g.add_argument("--out", required=True)
    g.add_argument("--json", default=None)
    g.add_argument("--fast-mode", default="fast", choices=["fast", "fast_a1"])
    a = ap.parse_args()
    METHOD = a.method
    if METHOD in ("babel", "scpair"):
        MODES = {"default": {}, "fast": {"fast": True}}
    cmd_train(a) if a.cmd == "train" else cmd_gates(a)


if __name__ == "__main__":
    main()
