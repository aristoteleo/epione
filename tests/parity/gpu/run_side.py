"""Run one side (CPU or GPU) of the GPU parity protocol in MANIFEST.md.

    python run_side.py --side cpu  [--tasks lsi,chromvar,...]
    python run_side.py --side cuda [--tasks ...]

Inputs come from ``$EPIONE_ARCHR_HEME`` (the ArchR hematopoiesis data
directory, see ``epione_heme_data/DATA_CONTRACT.md``) and ``$EPIONE_GPU_D3``
(the D3 h5ad). Outputs (``.npz`` / ``.parquet`` per task and a
``timings.json``) go to ``$EPIONE_GPU_PARITY_DIR/<label>``, where ``label``
defaults to the side. ``--label main`` together with a ``PYTHONPATH`` that
points at an export of ``origin/main`` produces the G0 reference.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

TASKS = [
    "lsi", "lsi_values", "ilsi1", "ilsi2", "neighbors", "chromvar_analytical",
    "chromvar_sample", "motif_scan", "peak_to_gene", "graph", "leiden", "umap",
    "d3_lsi", "d3_neighbors",
]
SEEDS = [0, 1, 2, 3, 4]


def _timeit(timings, key, fn):
    t0 = time.perf_counter()
    out = fn()
    timings[key] = time.perf_counter() - t0
    print(f"[{key}] {timings[key]:.2f} s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", required=True, choices=["cpu", "cuda"])
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    import anndata as ad
    import epione as epi

    dev = args.side
    data = Path(os.environ["EPIONE_ARCHR_HEME"])
    out = Path(os.environ["EPIONE_GPU_PARITY_DIR"]) / (args.label or args.side)
    out.mkdir(parents=True, exist_ok=True)
    tfile = out / "timings.json"
    timings = json.loads(tfile.read_text()) if tfile.exists() else {}
    tasks = args.tasks.split(",")
    meta = {"side": dev, "cpus": os.environ.get("OMP_NUM_THREADS"),
            "epione": epi.__file__}
    if dev == "cuda":
        import torch
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["torch"] = torch.__version__
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    kw = {} if dev == "cpu" else {"device": dev}

    def peaks():
        return ad.read_h5ad(data / "heme_peaks.h5ad")

    if "lsi" in tasks:
        a = peaks()
        _timeit(timings, "lsi", lambda: epi.tl.lsi(a, n_components=30, random_state=0, **kw))
        np.save(out / "lsi.npy", a.obsm["X_lsi"])

    if "lsi_values" in tasks:
        X = peaks().X
        if dev == "cpu":
            from epione.tl._reducedimension import tfidf_fast
            T = tfidf_fast(X)
            rs = T.sum(axis=1).A1
            rs[rs == 0] = 1
            T = T.multiply(1 / rs.reshape(-1, 1)).tocsr()
            T.data = np.log1p(T.data * 1e4)
        else:
            import torch
            from epione.tl._reducedimension import _lsi_operand_torch
            op = _lsi_operand_torch(X, torch.device(dev))
            assert op.resident
            T = sp.csr_matrix((op._Xd.values().cpu().numpy(),
                               op._Xd.col_indices().cpu().numpy(),
                               op._Xd.crow_indices().cpu().numpy()), shape=X.shape)
        T.sort_indices()
        sp.save_npz(out / "lsi_values.npz", T, compressed=False)

    if "ilsi1" in tasks or "ilsi2" in tasks:
        tl = ad.read_h5ad(data / "cache" / "tiles_lsi.h5ad")
        if "ilsi1" in tasks:
            a = tl.copy()
            _timeit(timings, "ilsi1", lambda: epi.tl.iterative_lsi(
                a, iterations=1, var_features=25000, n_components=30, seed=1,
                verbose=False, **kw))
            np.savez(out / "ilsi1.npz", emb=a.obsm["X_iterative_lsi"],
                     kept=a.uns["X_iterative_lsi"]["kept_dims"])
        if "ilsi2" in tasks:
            for s in SEEDS:
                a = tl.copy()
                _timeit(timings, f"ilsi2_s{s}", lambda: epi.tl.iterative_lsi(
                    a, iterations=2, var_features=25000, n_components=30,
                    resolution=2.0, seed=s, verbose=False, **kw))
                np.savez(out / f"ilsi2_s{s}.npz", emb=a.obsm["X_iterative_lsi"],
                         feat=a.uns["X_iterative_lsi"]["features_final"])
        del tl

    if "neighbors" in tasks:
        a = peaks()
        for metric in ("euclidean", "cosine"):
            _timeit(timings, f"neighbors_{metric}", lambda: epi.pp.neighbors(
                a, n_neighbors=30, use_rep="X_iterative_lsi", metric=metric, **kw))
            sp.save_npz(out / f"neighbors_{metric}_dist.npz", a.obsp["distances"].tocsr())
            sp.save_npz(out / f"neighbors_{metric}_conn.npz", a.obsp["connectivities"].tocsr())

    for meth in ("analytical", "sample"):
        if f"chromvar_{meth}" in tasks:
            a = peaks()
            _timeit(timings, f"chromvar_{meth}", lambda: epi.tl.compute_deviations(
                a, method=meth, verbose=False, **kw))
            np.savez(out / f"chromvar_{meth}.npz", z=a.obsm["motif_deviations"],
                     dev=a.obsm["motif_deviations_raw"])

    if "motif_scan" in tasks:
        from pyjaspar import jaspardb
        a = peaks()
        motifs = jaspardb(release="JASPAR2020").fetch_motifs(
            collection="CORE", tax_group=["vertebrates"])
        del a.varm["motif"]
        _timeit(timings, "motif_scan", lambda: epi.tl.add_motif_matrix(
            a, genome_fasta="unused", pwms=motifs, key_added="m", verbose=False, **kw))
        sp.save_npz(out / "motif_scan.npz", sp.csr_matrix(a.varm["m"]))

    if "peak_to_gene" in tasks:
        a = peaks()
        gs = ad.read_h5ad(data / "heme_gene_scores.h5ad")
        ann = gs.var[["chrom", "start", "end", "strand"]].copy()
        ann.insert(0, "gene_name", gs.var_names)
        res = _timeit(timings, "peak_to_gene", lambda: epi.tl.peak_to_gene(
            a, rna=gs, gene_annotation=ann.reset_index(drop=True),
            use_rep="X_iterative_lsi", verbose=False, inplace=False, **kw))
        res.to_parquet(out / "peak_to_gene.parquet")

    # Graph steps (CPU graph shared by both backends): G11 / G12 reference.
    if "graph" in tasks and dev == "cpu":
        a = peaks()
        epi.pp.neighbors(a, n_neighbors=30, use_rep="X_iterative_lsi", metric="cosine")
        sp.save_npz(out / "graph_conn.npz", a.obsp["connectivities"].tocsr())
        sp.save_npz(out / "graph_dist.npz", a.obsp["distances"].tocsr())
    if ("leiden" in tasks or "umap" in tasks) and dev == "cpu":
        a = peaks()
        a.obsp["connectivities"] = sp.load_npz(out / "graph_conn.npz")
        a.obsp["distances"] = sp.load_npz(out / "graph_dist.npz")
        a.uns["neighbors"] = {"connectivities_key": "connectivities",
                              "distances_key": "distances",
                              "params": {"n_neighbors": 30, "method": "umap",
                                         "metric": "cosine", "use_rep": "X_iterative_lsi"}}
        for s in SEEDS:
            if "leiden" in tasks:
                _timeit(timings, f"leiden_s{s}", lambda: epi.tl.clusters(
                    a, method="leiden", key_added="cl", leiden_kwargs=dict(
                        resolution=1.2, random_state=s, flavor="igraph",
                        n_iterations=2, directed=False)))
                np.save(out / f"leiden_s{s}.npy", a.obs["cl"].astype(str).to_numpy())
            if "umap" in tasks:
                _timeit(timings, f"umap_s{s}", lambda: epi.tl.umap(a, random_state=s))
                np.save(out / f"umap_s{s}.npy", a.obsm["X_umap"])

    if "d3_lsi" in tasks or "d3_neighbors" in tasks:
        d3 = ad.read_h5ad(os.environ["EPIONE_GPU_D3"])
        _timeit(timings, "d3_lsi", lambda: epi.tl.lsi(d3, n_components=30, random_state=0, **kw))
        np.save(out / "d3_lsi.npy", d3.obsm["X_lsi"])
        if "d3_neighbors" in tasks:
            # same input on both sides: the CPU run's LSI
            d3.obsm["X_lsi"] = np.load(out.parent / "cpu" / "d3_lsi.npy")
            _timeit(timings, "d3_neighbors", lambda: epi.pp.neighbors(
                d3, n_neighbors=30, use_rep="X_lsi", metric="euclidean", **kw))
            sp.save_npz(out / "d3_neighbors_dist.npz", d3.obsp["distances"].tocsr())

    tfile.write_text(json.dumps(timings, indent=1))


if __name__ == "__main__":
    main()
