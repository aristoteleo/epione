"""Gates T1-T5 of MANIFEST.md: Epione vs scHiCluster / HCEA contact decay.

Usage (dev env ``epione-hic``, scHiCluster installed):

    python t_gates.py <hcea_dir> <entex_analysis_dir> <chrom_sizes> <h200_cells.tsv> <results.json>

``hcea_dir`` holds ``sc_contact/`` and ``metadata/``; ``entex_analysis_dir`` holds
``decay/cell_*_decay.hdf5`` and ``compartment/L1/comp.impute{,.qnorm}.hdf``;
``h200_cells.tsv`` is the H200 cell table written by the compartment gates.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


# ---- HCEA fig4/02.decay_compartment.ipynb cell 11, verbatim except that the inputs are
# arguments (the notebook reads them from module globals) -------------------------------
chrom1, chrom2, pos1, pos2 = 1, 5, 2, 6
distbin = np.power(10, np.arange(5, 8.01, 3 / 100))
diffbin = np.arange(0, 30.01, 30 / 50)
sumbin = np.arange(-15, 15.01, 30 / 50)


def notebook_contact_hist(contact_path, ct, compall, chrom_sizes, res):
    data = pd.read_csv(contact_path, sep='\t', header=None, index_col=None)
    data = data.loc[(data[chrom1] == data[chrom2]) & data[chrom1].isin(chrom_sizes.index)]
    data['dist'] = np.abs(data[pos1] - data[pos2])
    l = data[chrom1] + '-' + (data[pos1] // res).astype(str)
    r = data[chrom2] + '-' + (data[pos2] // res).astype(str)
    datafilter = (compall.loc[l, 'raw_binfilter'].values) & (compall.loc[r, 'raw_binfilter'].values) & (data['dist'] > res)
    l, r, data = l[datafilter], r[datafilter], data.loc[datafilter]
    data['compdiff'] = np.abs(compall.loc[l, ct].values - compall.loc[r, ct].values)
    data['compsum'] = (compall.loc[l, ct].values + compall.loc[r, ct].values) / 2
    histdiff = np.histogram2d(data['dist'], data['compdiff'], bins=[distbin, diffbin])[0]
    histsum = np.histogram2d(data['dist'], data['compsum'], bins=[distbin, sumbin])[0]
    return histdiff, histsum


def _nb_worker(args):
    return notebook_contact_hist(*args)


def main(hcea_dir, entex, chrom_sizes_path, h200, results_path):
    import epione as epi
    epi.pl  # noqa: B018
    from epione.single import hic
    from schicluster.cool.contact_distance import compute_decay
    import qnorm

    results = {}
    sizes = pd.read_csv(chrom_sizes_path, sep="\t", header=None, index_col=0)[1]
    files = sorted(glob.glob(f"{hcea_dir}/sc_contact/*/*.3C.contact.rmbkl.tsv.gz"))
    cells = pd.Series(files, index=[os.path.basename(f).split(".3C.")[0] for f in files])
    h200_cells = pd.read_csv(h200, sep="\t", header=None, index_col=0)[1]

    # ---- T1 vs scHiCluster compute_decay (H200) ------------------------------------
    ad200 = hic.load_contact_tables(h200_cells.tolist(), cell_ids=list(h200_cells.index),
                                    chrom_sizes=sizes)
    t = time.time()
    dec, spar = hic.contact_decay(ad200, nproc=8, progress=False)
    t_epi = time.time() - t
    bins = hic.decay_bins(sizes)
    t = time.time()
    ref_dec, ref_spar = [], []
    for cid, path in h200_cells.items():
        s, d = compute_decay(cid, path, bins, sizes.to_frame(), 10_000)
        ref_spar.append(s); ref_dec.append(d)
    t_ref = time.time() - t
    ref_dec = pd.concat(ref_dec, axis=1).T
    ref_spar = pd.concat(ref_spar, axis=1).T.reindex(columns=spar.columns)
    same_dec = bool((ref_dec.loc[dec.index].to_numpy() == dec.to_numpy()).all())
    same_spar = bool(ref_spar.loc[spar.index].fillna(-1).to_numpy().astype(int).tolist()
                     == spar.fillna(-1).to_numpy().astype(int).tolist())
    results["T1"] = {"identical_decay": same_dec, "identical_sparsity": same_spar,
                     "n_bins": int(dec.shape[1]), "seconds_epi_8proc": t_epi,
                     "seconds_ref_1proc": t_ref, "pass": same_dec and same_spar}
    print("T1", results["T1"], flush=True)

    # ---- T2 / T3 vs published decay and Short/Long (8,040 cells) --------------------
    # Amendment T-1: HCEA's published histograms use chr1-22 + chrX (no chrY)
    sizes_x = sizes.loc[[f"chr{i}" for i in range(1, 23)] + ["chrX"]]
    ad = hic.load_contact_tables(cells.tolist(), cell_ids=list(cells.index), chrom_sizes=sizes_x)
    dec_all, _ = hic.contact_decay(ad, nproc=30, progress=False)
    pub = pd.concat([pd.read_hdf(x, key="data") for x in glob.glob(f"{entex}/decay/cell_*_decay.hdf5")])
    pub = pub.loc[dec_all.index]
    same = bool(pub.shape == dec_all.shape and (pub.to_numpy() == dec_all.to_numpy()).all())
    n_diff = int((pub.to_numpy() != dec_all.to_numpy()).any(axis=1).sum()) if pub.shape == dec_all.shape else -1
    results["T2"] = {"cells": int(len(dec_all)), "identical": same, "cells_differing": n_diff,
                     "pass": same}
    print("T2", results["T2"], flush=True)
    meta = pd.read_csv(f"{hcea_dir}/metadata/5kCG100k3C_summary.csv.gz", index_col=0)
    ratio = hic.decay_ratio(dec_all)
    err = float(np.abs(ratio.to_numpy() - meta.loc[ratio.index, "Short/Long"].to_numpy()).max())
    ratio_pub = hic.decay_ratio(pub)
    err_pub = float(np.abs(ratio_pub.to_numpy() - meta.loc[ratio.index, "Short/Long"].to_numpy()).max())
    results["T3"] = {"max_abs_err": err, "max_abs_err_from_published_decay": err_pub,
                     "pass": err <= 1e-12}
    print("T3", results["T3"], flush=True)

    # ---- T4 quantile normalisation ----------------------------------------------------
    imp = pd.read_hdf(f"{entex}/compartment/L1/comp.impute.hdf", key="data")
    qn_pub = pd.read_hdf(f"{entex}/compartment/L1/comp.impute.qnorm.hdf", key="data")
    keep = imp["raw_binfilter"].to_numpy()
    tcols = imp.columns[4:]
    x = imp.loc[keep, tcols]
    ours = hic.quantile_normalize(x)
    e4 = float(np.abs(ours.to_numpy() - qn_pub.loc[keep, tcols].to_numpy()).max())
    results["T4"] = {"types": len(tcols), "bins": int(keep.sum()), "max_abs_err": e4,
                     "pass": e4 <= 1e-10}
    print("T4", results["T4"], flush=True)
    rng = np.random.default_rng(0)
    ties = pd.DataFrame(rng.integers(0, 20, size=(500, 5)).astype(float))
    e4b = max(float(np.abs(hic.quantile_normalize(m).to_numpy()
                           - qnorm.quantile_normalize(m).to_numpy()).max()) for m in (x, ties))
    results["T4b"] = {"qnorm_version": qnorm.__version__, "max_abs_err": e4b, "pass": e4b <= 1e-13}
    print("T4b", results["T4b"], flush=True)

    # ---- T5 compartment_contact_hist vs notebook contact_hist (H200) ------------------
    res = 100_000
    autos = sizes.iloc[:22]
    compall = qn_pub.copy()
    compall.index = compall["chrom"].astype(str) + "-" + (compall["start"] // res).astype(str)
    meta200 = meta.loc[h200_cells.index]
    t = time.time()
    with ProcessPoolExecutor(8) as ex:
        ref = list(ex.map(_nb_worker, [(h200_cells[c], meta200.loc[c, "majortype"], compall, autos, res)
                                       for c in h200_cells.index]))
    t_ref = time.time() - t
    ref_by = {}
    for c, (hd, hs) in zip(h200_cells.index, ref):
        g = meta200.loc[c, "majortype"]
        a = ref_by.setdefault(g, [0, 0])
        a[0] = a[0] + hd
        a[1] = a[1] + hs
    scores = qn_pub.loc[keep, ["chrom", "start", "end", *tcols]].reset_index(drop=True)
    ad200.obs["majortype"] = meta200["majortype"].to_numpy()
    t = time.time()
    ours = hic.compartment_contact_hist(ad200, scores, "majortype", nproc=8, progress=False)
    t_epi = time.time() - t
    same5 = {g: bool(np.array_equal(ours[g]["diff"], ref_by[g][0]) and np.array_equal(ours[g]["sum"], ref_by[g][1]))
             for g in ref_by}
    results["T5"] = {"groups_identical": same5, "contacts_counted": {g: int(ref_by[g][0].sum()) for g in ref_by},
                     "seconds_epi_8proc": t_epi, "seconds_ref_8proc": t_ref, "pass": all(same5.values())}
    print("T5", results["T5"], flush=True)

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main(*sys.argv[1:6])
