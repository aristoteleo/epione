"""rapids-singlecell side of the GPU parity protocol (gates G10-G12, MANIFEST.md).

Runs Epione's ``backend="rapids"`` dispatch on D1. Needs rapids-singlecell
(cuML / cuGraph / cuVS), i.e. glibc >= 2.28 -- on older systems run it inside
the RAPIDS container. Reads ``$EPIONE_ARCHR_HEME`` and the CPU graph written by
``run_side.py --side cpu --tasks graph``; writes to
``$EPIONE_GPU_PARITY_DIR/rapids``.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2, 3, 4]


def main():
    import anndata as ad
    import cupy
    import rapids_singlecell as rsc
    import epione as epi

    data = Path(os.environ["EPIONE_ARCHR_HEME"])
    root = Path(os.environ["EPIONE_GPU_PARITY_DIR"])
    out = root / "rapids"
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    def timeit(key, fn):
        t0 = time.perf_counter()
        fn()
        cupy.cuda.Device().synchronize()
        timings[key] = time.perf_counter() - t0
        print(f"[{key}] {timings[key]:.2f} s", flush=True)

    (out / "meta.json").write_text(json.dumps({
        "rapids_singlecell": rsc.__version__, "cupy": cupy.__version__,
        "cpus": os.environ.get("OMP_NUM_THREADS")}, indent=1))

    a = ad.read_h5ad(data / "heme_peaks.h5ad", backed="r")
    b = ad.AnnData(obs=a.obs[["Clusters"]].copy())
    b.obsm["X_iterative_lsi"] = np.asarray(a.obsm["X_iterative_lsi"])

    # G10: neighbours
    for metric in ("euclidean", "cosine"):
        c = b.copy()
        timeit(f"neighbors_{metric}", lambda: epi.pp.neighbors(
            c, n_neighbors=30, use_rep="X_iterative_lsi", metric=metric, backend="rapids"))
        sp.save_npz(out / f"neighbors_{metric}_dist.npz", sp.csr_matrix(c.obsp["distances"]))

    # G11 / G12 on the CPU-built graph
    g = b.copy()
    g.obsp["connectivities"] = sp.load_npz(root / "cpu" / "graph_conn.npz")
    g.obsp["distances"] = sp.load_npz(root / "cpu" / "graph_dist.npz")
    g.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                          "params": {"n_neighbors": 30, "method": "umap", "metric": "cosine",
                                     "use_rep": "X_iterative_lsi"}}
    for s in SEEDS:
        timeit(f"leiden_s{s}", lambda: epi.tl.clusters(
            g, method="leiden", key_added="cl", backend="rapids",
            leiden_kwargs=dict(resolution=1.2, random_state=s, n_iterations=-1)))
        np.save(out / f"leiden_s{s}.npy", g.obs["cl"].astype(str).to_numpy())
        timeit(f"umap_s{s}", lambda: epi.tl.umap(g, random_state=s, backend="rapids"))
        np.save(out / f"umap_s{s}.npy", np.asarray(g.obsm["X_umap"]))

    (out / "timings.json").write_text(json.dumps(timings, indent=1))


if __name__ == "__main__":
    main()
