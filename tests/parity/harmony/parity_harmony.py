"""Compare epi.tl.harmony with ArchR addHarmony reference outputs.

The reference outputs are generated outside this repository with ArchR 1.0.3.
Usage: python parity_harmony.py [<ref_dir> [<ref_dir_seed2> <seed2>]] [--json out.json]
<ref_dir> defaults to $EPIONE_ARCHR_PARITY_DIR/harmony.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from epione.tl._harmony import _Harmony, _l2_normalise_cols, r_kmeans, run_harmony
from epione.tl._rrng import RRandom


def _tsv(p):
    return pd.read_csv(p, sep="\t")


def load_inputs(ref: Path):
    lsi = _tsv(ref / "input_lsi.tsv.gz").set_index("barcode")
    meta = _tsv(ref / "input_meta.tsv").set_index("barcode").loc[lsi.index]
    return lsi, meta


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def check_rng(ref: Path, seed: int):
    r = RRandom(seed)
    out = {"seed_state": bool(np.array_equal(r.state(), np.loadtxt(ref / "rng_seed_state.txt", dtype=np.int64)))}
    out["runif"] = bool(np.array_equal(r.unif_rand(7), np.loadtxt(ref / "rng_runif.tsv", skiprows=1)))
    n = len(pd.read_csv(ref / "input_meta.tsv", sep="\t"))
    out["sample_int"] = bool(np.array_equal(r.sample_int(n, 100), np.loadtxt(ref / "rng_sample.txt", dtype=np.int64)))
    big = r.runif(5, 0, 2147483647).astype(np.int64)
    out["runif_rand_max"] = bool(np.array_equal(big, np.loadtxt(ref / "rng_runif_big.txt", dtype=np.int64)))
    return out


def check_kmeans(ref: Path, X: np.ndarray, seed: int):
    zcos = _l2_normalise_cols(X.T.copy())
    t0 = time.time()
    c = r_kmeans(zcos.T, 100, RRandom(seed), iter_max=25, nstart=10)
    t = time.time() - t0
    ref_c = _tsv(ref / "kmeans_centers.tsv.gz").to_numpy()
    return {"kmeans_centers_max_abs": max_abs(c, ref_c), "kmeans_seconds": t}


def check_injected(ref: Path, X: np.ndarray, meta: pd.DataFrame):
    centers = _tsv(ref / "kmeans_centers.tsv.gz").to_numpy()
    state = np.loadtxt(ref / "rng_state_after_init.txt", dtype=np.int64)
    ho = run_harmony(X, meta, "Sample", init_centers=centers, seed=None, rng_state=state,
                     return_object=True, verbose=False)
    res = {"R_init_max_abs": None, "iters": []}
    # replay the init on its own to compare R_init
    ho0 = run_harmony(X, meta, "Sample", init_centers=centers, seed=None, rng_state=state,
                      max_iter=0, return_object=True, verbose=False)
    res["R_init_max_abs"] = max_abs(ho0.R.T, _tsv(ref / "R_init.tsv.gz").to_numpy())
    r_rounds = np.loadtxt(ref / "kmeans_rounds.txt", dtype=int, ndmin=1)
    res["kmeans_rounds_py"] = [int(v) for v in ho.kmeans_rounds]
    res["kmeans_rounds_r"] = [int(v) for v in r_rounds]
    r_obj = np.loadtxt(ref / "objective_harmony.txt", ndmin=1)
    py_obj = np.asarray(ho.objective_harmony, dtype=np.float64)
    res["objective_max_rel"] = float(np.max(np.abs(py_obj - r_obj) / np.abs(r_obj))) if len(py_obj) == len(r_obj) else float("inf")
    # step-by-step: rerun keeping intermediate states
    rng = RRandom.from_state(state)
    sigma = np.repeat(0.1, 100)
    from epione.tl._harmony import _design
    Phi, B_vec, _ = _design(meta, ["Sample"])
    h = _Harmony(X.T.copy(), Phi, sigma, np.repeat(2.0, B_vec.sum()),
                 np.concatenate([[0.0], np.ones(B_vec.sum())]), 0.2, 20, 1e-3, 1e-2, 100, 0.05, rng)
    h.init_cluster(centers.T)
    for it in range(1, len(r_rounds) + 1):
        h.cluster()
        h.moe_correct_ridge()
        res["iters"].append({
            "iter": it,
            "R_max_abs": max_abs(h.R.T, _tsv(ref / f"R_iter{it:02d}.tsv.gz").to_numpy()),
            "Zcorr_max_abs": max_abs(h.Z_corr.T, _tsv(ref / f"Zcorr_iter{it:02d}.tsv.gz").to_numpy()),
        })
    archr = _tsv(ref / "archr_harmony.tsv.gz").set_index("barcode").to_numpy()
    res["final_max_abs"] = max_abs(ho.Z_corr.T, archr)
    res["n_iter_py"] = int(ho.n_iter)
    return res


def check_free(ref: Path, X: np.ndarray, meta: pd.DataFrame, seed: int, extended: bool):
    t0 = time.time()
    H = run_harmony(X, meta, "Sample", seed=seed, verbose=False)
    t = time.time() - t0
    archr = _tsv(ref / "archr_harmony.tsv.gz").set_index("barcode").to_numpy()
    out = {"seed": seed, "final_max_abs": max_abs(H, archr), "python_seconds": t,
           "pearson": float(np.corrcoef(H.ravel(), archr.ravel())[0, 1])}
    summ = dict(l.strip().split("\t") for l in open(ref / "summary.tsv"))
    out["archr_seconds"] = float(summ["archr_seconds"])
    if extended:
        from scipy.spatial import procrustes
        from sklearn.metrics import adjusted_rand_score
        from sklearn.neighbors import NearestNeighbors
        import anndata as ad
        import scanpy as sc
        out["procrustes_similarity"] = float(1.0 - procrustes(archr, H)[2])
        k = 30
        nn_r = NearestNeighbors(n_neighbors=k + 1).fit(archr).kneighbors(archr, return_distance=False)[:, 1:]
        nn_p = NearestNeighbors(n_neighbors=k + 1).fit(H).kneighbors(H, return_distance=False)[:, 1:]
        jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(nn_r, nn_p)]
        out["knn30_jaccard_mean"] = float(np.mean(jac))
        labs = []
        for emb in (archr, H):
            a = ad.AnnData(np.zeros((emb.shape[0], 1), dtype=np.float32), obsm={"X_h": emb})
            sc.pp.neighbors(a, use_rep="X_h", n_neighbors=30, random_state=0)
            sc.tl.leiden(a, resolution=0.8, random_state=0, flavor="igraph", n_iterations=2, directed=False)
            labs.append(a.obs["leiden"].to_numpy())
        out["leiden_ari"] = float(adjusted_rand_score(labs[0], labs[1]))
    return out


def main(argv):
    out_json = None
    if "--json" in argv:
        i = argv.index("--json"); out_json = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if argv:
        ref = Path(argv[0])
    else:
        import os
        root = os.environ.get("EPIONE_ARCHR_PARITY_DIR")
        if not root:
            sys.exit("pass <ref_dir> or set EPIONE_ARCHR_PARITY_DIR")
        ref = Path(root) / "harmony"
        argv = [str(ref), str(ref / "seed42"), "42"]
    lsi, meta = load_inputs(ref)
    X = lsi.to_numpy()
    report = {"rng": check_rng(ref, 1)}
    report.update(check_kmeans(ref, X, 1))
    report["injected"] = check_injected(ref, X, meta)
    report["free_seed1"] = check_free(ref, X, meta, 1, extended=True)
    if len(argv) >= 3:
        ref2, seed2 = Path(argv[1]), int(argv[2])
        report[f"free_seed{seed2}"] = check_free(ref2, X, meta, seed2, extended=False)
    print(json.dumps(report, indent=2))
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
