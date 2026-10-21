"""Gates E1-E2 of MANIFEST.md: epione's scHiCluster embedding vs scHiCluster's.

Usage (dev env with scHiCluster installed):

    python e_gates.py <h200.scool> <work_dir> <results.json> [nproc]

``h200.scool`` is the H200 scool written by the compartment gates (s_gates.py).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AUTOS = [f"chr{i}" for i in range(1, 23)]
RES = 100_000


def main(scool, work, results_path, nproc=8):
    import cooler
    import epione as epi
    epi.pl  # noqa: B018
    from epione.single import hic
    from epione.single.hic._embed import _band_indices
    from schicluster.embedding.calc_embedding import embedding as ref_embedding, make_chrom_matrix
    from schicluster.impute.impute_chromosome import impute_chromosome
    from scipy.sparse import load_npz

    nproc = int(nproc)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    names = [c.split("/cells/")[-1] for c in cooler.fileops.list_scool_cells(scool)]
    ad = hic.load_scool_cells(scool, cell_names=names, chromosomes=AUTOS)
    results = {"cells": len(names)}

    t = time.time()
    hic.impute_cells(ad, out_dir=work / "imputed", output_dist=10_000_000, compress=True,
                     nproc=nproc, progress=False)
    results["seconds_impute"] = round(time.time() - t, 1)

    # imputation check: one cell, every chromosome, vs impute_chromosome directly
    z = np.load(work / "imputed" / f"{names[0]}.npz")
    imp_same = True
    for c in AUTOS:
        out = work / f"ref_{c}.hdf"
        impute_chromosome(scool_url=f"{scool}::/cells/{names[0]}", chrom=c, resolution=RES,
                          output_path=str(out), output_dist=10_000_000)
        E = load_npz(f"{out}.npz").toarray()
        ref = (E + E.T - np.diag(np.diag(E))).astype(np.float32)
        imp_same &= bool(np.array_equal(ref, z[c]))
    results["imputation_identical"] = imp_same

    # per-cell imputed cools for the reference (upper triangle of the same matrices)
    bins = cooler.Cooler(f"{scool}::/cells/{names[0]}").bins()[:][["chrom", "start", "end"]]
    bins = bins[bins["chrom"].isin(AUTOS)].reset_index(drop=True)
    offs = bins.groupby("chrom", sort=False, observed=True).apply(lambda g: g.index[0]).to_dict()
    cool_dir = work / "imputed_cools"
    cool_dir.mkdir(exist_ok=True)
    table = {}
    for cid in names:
        path = cool_dir / f"{cid}.cool"
        table[cid] = str(path)
        if path.exists():
            continue
        z = np.load(work / "imputed" / f"{cid}.npz")
        parts = []
        for c in AUTOS:
            P = z[c]
            iu, ju = np.nonzero(np.triu(P))
            parts.append(pd.DataFrame({"bin1_id": iu + offs[c], "bin2_id": ju + offs[c],
                                       "count": P[iu, ju]}))
        cooler.create_cooler(str(path), bins=bins, pixels=pd.concat(parts, ignore_index=True),
                             ordered=True, dtypes={"count": np.float32})
    tbl = work / "cell_table.tsv"
    pd.Series(table).to_csv(tbl, sep="\t", header=False)

    # E1: band matrices
    e1 = True
    cell_table = pd.Series(table)
    for c in AUTOS:
        n = int(ad.uns["hic"]["n_chrom_bins"][c])
        make_chrom_matrix(cell_table, c, n, work / f"band_{c}.npz", 100_000, 1_000_000, RES)
        ref = np.load(work / f"band_{c}.npz")["arr_0"]
        iu, ju = _band_indices(n, 1_000_000, RES)
        ours = np.vstack([np.load(work / "imputed" / f"{cid}.npz")[c][iu, ju] for cid in names])
        ours = ours.astype(np.float32) * np.float32(100_000)
        e1 &= bool(ref.shape == ours.shape and np.array_equal(ref, ours))
    results["E1"] = {"identical": e1, "pass": e1}
    print("E1", results["E1"], flush=True)

    # E2: final embedding, sign-aligned per component
    t = time.time()
    ref_embedding(str(tbl), str(work / "ref_embed"), dim=50, dist=1_000_000, resolution=RES,
                  scale_factor=100_000, norm_sig=False, cpu=nproc, save_model=False, save_raw=False)
    t_ref = time.time() - t
    ref = np.load(work / "ref_embed" / "decomp" / "total_decomp.npz")["arr_0"]
    t = time.time()
    ours = hic.embedding(ad, method="schicluster").obsm["X_schicluster"].astype(np.float64)
    t_epi = time.time() - t
    def aligned_err(a, b):
        sign = np.sign((a * b).sum(axis=0))
        sign[sign == 0] = 1
        return float(np.abs(a * sign - b).max())

    err = aligned_err(ours, ref)
    scale = float(np.abs(ref).max())
    corr = [abs(np.corrcoef(ours[:, k], ref[:, k])[0, 1]) for k in range(ref.shape[1])]

    # Amendment E-1: the reference's own run-to-run variation (global seeds 1, 2, 3)
    from sklearn.decomposition import TruncatedSVD

    bands = {c: np.load(work / f"band_{c}.npz")["arr_0"] for c in AUTOS}

    def reference_run(seed):
        np.random.seed(seed)
        parts = []
        for c in AUTOS:
            M = bands[c]
            d = min(50, M.shape[0] - 1, M.shape[1] - 1)
            parts.append(TruncatedSVD(n_components=d, algorithm="arpack").fit_transform(M))
        return TruncatedSVD(n_components=50, algorithm="arpack").fit_transform(np.concatenate(parts, 1))

    runs = [reference_run(s) for s in (1, 2, 3)]
    self_err = max(aligned_err(runs[i], runs[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    results["E2"] = {"shape": list(ours.shape), "ref_shape": list(ref.shape),
                     "max_abs_err": err, "max_abs_value": scale, "rel": err / scale,
                     "reference_self_max_abs_err": self_err, "reference_self_rel": self_err / scale,
                     "min_component_abs_corr": float(min(corr)),
                     "seconds_ref": round(t_ref, 1), "seconds_epi": round(t_epi, 1),
                     "pass": bool(ours.shape == ref.shape and err <= 2 * self_err
                                  and min(corr) >= 0.9999)}
    print("E2", results["E2"], flush=True)

    # E3 (Amendment E-2): the same with norm_sig=True, the lab's `--norm_sig`
    ref_ns = work / "ref_embed_normsig"
    ref_embedding(str(tbl), str(ref_ns), dim=50, dist=1_000_000, resolution=RES,
                  scale_factor=100_000, norm_sig=True, cpu=nproc, save_model=False, save_raw=False)
    ref3 = np.load(ref_ns / "decomp" / "total_decomp.npz")["arr_0"]
    ours3 = hic.embedding(ad, method="schicluster", norm_sig=True).obsm["X_schicluster"].astype(np.float64)

    def reference_run_ns(seed):
        np.random.seed(seed)
        parts = []
        for c in AUTOS:
            M = bands[c]
            d = min(50, M.shape[0] - 1, M.shape[1] - 1)
            m = TruncatedSVD(n_components=d, algorithm="arpack")
            x = m.fit_transform(M)
            k = m.singular_values_ > 0
            parts.append(x[:, k] / m.singular_values_[k][None, :])
        m = TruncatedSVD(n_components=50, algorithm="arpack")
        x = m.fit_transform(np.concatenate(parts, 1))
        k = m.singular_values_ > 0
        return x[:, k] / m.singular_values_[k][None, :]

    runs3 = [reference_run_ns(s) for s in (1, 2, 3)]
    self3 = max(aligned_err(runs3[i], runs3[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    err3 = aligned_err(ours3, ref3)
    scale3 = float(np.abs(ref3).max())
    corr3 = [abs(np.corrcoef(ours3[:, k], ref3[:, k])[0, 1]) for k in range(ref3.shape[1])]
    results["E3"] = {"shape": list(ours3.shape), "ref_shape": list(ref3.shape),
                     "max_abs_err": err3, "max_abs_value": scale3, "rel": err3 / scale3,
                     "reference_self_max_abs_err": self3, "reference_self_rel": self3 / scale3,
                     "min_component_abs_corr": float(min(corr3)),
                     "pass": bool(ours3.shape == ref3.shape and err3 <= 2 * self3
                                  and min(corr3) >= 0.9999)}
    print("E3", results["E3"], flush=True)
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main(*sys.argv[1:])
