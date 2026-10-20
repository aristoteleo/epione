"""Gates S1-S6 of MANIFEST.md: Epione vs scHiCluster single-cell compartments.

Usage (dev env ``epione-hic``, scHiCluster installed, bedtools on PATH):

    python s_gates.py <hcea_dir> <fasta> <chrom_sizes> <work_dir> <results.json>

``hcea_dir`` holds ``sc_contact/*/*.3C.contact.rmbkl.tsv.gz`` and
``metadata/5kCG100k3C_summary.csv.gz``.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main(hcea_dir, fasta, chrom_sizes, work, results_path):
    import cooler
    import xarray as xr
    import epione as epi
    epi.pl  # noqa: B018  (import order: utils before single)
    from epione.single import hic
    import schicluster.compartment.call_compartment as cc
    from schicluster.compartment.call_compartment import (
        get_cpg_profile, multiple_cell_compartment, single_chrom_compartment)

    # Upstream writes the cool's categorical ``chrom`` column into NetCDF, which current
    # xarray rejects. Cast it to str in the aggregation (file writing) step only; the
    # per-cell computation is untouched.
    _aggregate = cc.aggregate_compartment

    def _aggregate_str(cell_table, temp_dir, bins, output_path, calc_strength):
        bins = bins.copy()
        bins["chrom"] = bins["chrom"].astype(str)
        return _aggregate(cell_table, temp_dir, bins, output_path, calc_strength)

    cc.aggregate_compartment = _aggregate_str
    from schicluster.cool.scool import generate_scool_batch_data
    from schicluster.cool.merge import merge_cell_raw

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    res = 100_000
    sizes = pd.read_csv(chrom_sizes, sep="\t", index_col=0, header=None).squeeze(axis=1)
    autos = [f"chr{i}" for i in range(1, 23)]
    sizes = sizes.loc[autos]
    cs_path = work / "chrom_sizes.autosomes.tsv"
    sizes.to_csv(cs_path, sep="\t", header=False)
    results = {}

    # ---- fixture H200 ----------------------------------------------------
    files = sorted(glob.glob(f"{hcea_dir}/sc_contact/*/*.3C.contact.rmbkl.tsv.gz"))
    cells = pd.Series(files, index=[os.path.basename(f).split(".3C.")[0] for f in files])
    meta = pd.read_csv(f"{hcea_dir}/metadata/5kCG100k3C_summary.csv.gz", index_col=0)
    meta = meta.loc[cells.index]
    pick = (meta.groupby("majortype", group_keys=False)
            .apply(lambda g: g.sample(len(g) * 200 // len(meta) + 1, random_state=0)))
    pick = pick.sample(200, random_state=0) if len(pick) > 200 else pick
    cells = cells.loc[pick.index]
    cells.to_csv(work / "H200_cells.tsv", sep="\t", header=False)
    results["fixture"] = {"n_cells": len(cells),
                          "majortype": pick["majortype"].value_counts().to_dict()}

    # ---- S1 cpg_ratio ----------------------------------------------------
    t = time.time()
    ref_h5 = work / "cpg_ref.hdf"
    if not ref_h5.exists():
        get_cpg_profile(fasta, str(ref_h5), chrom_size_path=str(cs_path), resolution=res)
    t_ref = time.time() - t
    ref = pd.read_hdf(ref_h5)
    t = time.time()
    epi_cpg = hic.cpg_ratio(fasta, chrom_sizes=cs_path, resolution=res)
    t_epi = time.time() - t
    same_rows = (ref[["chrom", "start", "end"]].astype(str).to_numpy()
                 == epi_cpg[["chrom", "start", "end"]].astype(str).to_numpy()).all()
    same_vals = np.array_equal(ref["cpg_ratio"].to_numpy(), epi_cpg["cpg_ratio"].to_numpy())
    results["S1"] = {"rows": int(len(ref)), "identical_rows": bool(same_rows),
                     "identical_values": bool(same_vals),
                     "max_abs_err": float(np.abs(ref["cpg_ratio"].to_numpy()
                                                 - epi_cpg["cpg_ratio"].to_numpy()).max()),
                     "seconds_ref": t_ref, "seconds_epi": t_epi,
                     "pass": bool(same_rows and same_vals)}
    print("S1", results["S1"])
    cpg = ref

    # ---- S2 scool pixels -------------------------------------------------
    scool = work / "H200.scool"
    t = time.time()
    if scool.exists():
        scool.unlink()
    hic.contacts_to_scool(cells.tolist(), scool, chrom_sizes=cs_path, resolution=res,
                          cell_ids=list(cells.index), nproc=8)
    t_epi = time.time() - t
    bins = cooler.binnify(sizes, res)
    offsets = np.r_[0, np.cumsum(bins.groupby("chrom", sort=False).size().to_numpy())][:-1]
    chrom_offset = dict(zip(sizes.index, offsets))
    ref_pix_h5 = work / "pixels_ref.h5"
    t = time.time()
    if not ref_pix_h5.exists():
        generate_scool_batch_data(cells.to_dict(), resolution=res, chrom_offset=chrom_offset,
                                  chrom_size_path=str(cs_path), blacklist_1d_path=None,
                                  blacklist_2d_path=None, remove_duplicates=True,
                                  blacklist_resolution=10000, output_path=str(ref_pix_h5))
    t_ref = time.time() - t
    n_bad = 0
    with pd.HDFStore(ref_pix_h5, "r") as h:
        for cid in cells.index:
            a = h[cid].astype(np.int64).reset_index(drop=True)
            b = cooler.Cooler(f"{scool}::/cells/{cid}").pixels()[:][["bin1_id", "bin2_id", "count"]]
            b = b.astype(np.int64).reset_index(drop=True)
            if not (a.shape == b.shape and (a.to_numpy() == b.to_numpy()).all()):
                n_bad += 1
    results["S2"] = {"cells_differing": n_bad, "seconds_ref": t_ref, "seconds_epi": t_epi,
                     "pass": n_bad == 0}
    print("S2", results["S2"])

    # ---- S3 / S4 compartment_scores ---------------------------------------
    cpg_path = work / "cpg_ref.hdf"
    for gate, mode in (("S3", "cool"), ("S4", "tsv")):
        if mode == "cool":
            table = pd.Series([f"{scool}::/cells/{c}" for c in cells.index], index=cells.index)
        else:
            table = cells
        table_path = work / f"cell_table_{mode}.tsv"
        table.to_csv(table_path, sep="\t", header=False)
        prefix = work / f"ref_{mode}"
        t = time.time()
        if not Path(f"{prefix}.compartment.nc").exists():
            multiple_cell_compartment(str(table_path), str(prefix), str(cpg_path), cpu=8,
                                      calc_strength=False, mode=mode,
                                      chrom_size_path=str(cs_path), resolution=res)
        t_ref = time.time() - t
        ref_nc = xr.open_dataset(f"{prefix}.compartment.nc")
        ref_x = ref_nc["compartment"].sel(cell=list(cells.index)).to_numpy()
        if mode == "cool":
            ad = hic.load_scool_cells(scool, cell_names=list(cells.index))
        else:
            ad = hic.load_contact_tables(cells.tolist(), cell_ids=list(cells.index),
                                         chrom_sizes=cs_path, resolution=res)
        t = time.time()
        out = hic.compartment_scores(ad, cpg, nproc=8, dtype=np.float64, progress=False)
        t_epi = time.time() - t
        err = float(np.nanmax(np.abs(out.X - ref_x)))
        nan_same = bool((np.isnan(out.X) == np.isnan(ref_x)).all())
        results[gate] = {"shape": list(out.shape), "ref_shape": list(ref_x.shape),
                         "max_abs_err": err, "identical_nan_mask": nan_same,
                         "seconds_ref": t_ref, "seconds_epi": t_epi,
                         "pass": bool(out.shape == ref_x.shape and err <= 1e-12 and nan_same)}
        print(gate, results[gate])

    # ---- S5 per-chromosome strength ---------------------------------------
    ad = hic.load_scool_cells(scool, cell_names=list(cells.index))
    out = hic.compartment_scores(ad, cpg, calc_strength=True, nproc=8, dtype=np.float64,
                                 progress=False)
    per = out.uns["compartment_strength_per_chrom"]["values"]
    worst = 0.0
    for i, cid in enumerate(cells.index[:40]):
        clr = cooler.Cooler(f"{scool}::/cells/{cid}")
        for j, chrom in enumerate(out.uns["compartment_strength_per_chrom"]["chromosomes"]):
            m = clr.matrix(balance=False, sparse=True).fetch(chrom).tocsr()
            _, sc = single_chrom_compartment(m, cpg.loc[cpg["chrom"] == chrom, "cpg_ratio"],
                                             calc_strength=True)
            rel = np.abs(per[i, j] - sc) / np.maximum(np.abs(sc), 1e-300)
            worst = max(worst, float(np.nanmax(rel)))
    results["S5"] = {"cells_checked": 40, "max_rel_err": worst, "pass": worst <= 1e-12}
    print("S5", results["S5"])

    # ---- S6 merge_cells ----------------------------------------------------
    ad = hic.load_contact_tables(cells.tolist(), cell_ids=list(cells.index),
                                 chrom_sizes=cs_path, resolution=res,
                                 obs=pick[["majortype"]].reset_index(drop=True))
    t = time.time()
    paths = hic.merge_cells(ad, "majortype", work / "merge_epi", double_diagonal=True,
                            remove_duplicates=False, overwrite=True, nproc=3)
    t_epi = time.time() - t
    s6 = {}
    t_ref = 0.0
    for g, p in paths.items():
        tbl = work / f"merge_ref_{g}.tsv"
        cells.loc[pick.index[pick["majortype"] == g]].to_csv(tbl, sep="\t", header=False)
        refp = work / f"merge_ref_{g}.cool"
        t = time.time()
        if not refp.exists():
            merge_cell_raw(str(tbl), str(cs_path), str(refp), resolution=res)
        t_ref += time.time() - t
        a = cooler.Cooler(str(refp)).pixels()[:].astype(np.int64)
        b = cooler.Cooler(p).pixels()[:].astype(np.int64)
        s6[g] = bool(a.shape == b.shape and (a.to_numpy() == b.to_numpy()).all())
    results["S6"] = {"groups_identical": s6, "seconds_ref": t_ref, "seconds_epi": t_epi,
                     "pass": all(s6.values())}
    print("S6", results["S6"])

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main(*sys.argv[1:6])
