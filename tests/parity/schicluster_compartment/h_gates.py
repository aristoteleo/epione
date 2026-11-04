"""Gates H1-H4 of MANIFEST.md: Epione vs the HCEA compartment notebook.

Usage (dev env ``epione-hic``):

    python h_gates.py <entex_root> <cpg_txt> <results.json> [nproc]

``entex_root`` is the HCEA project root (``merged_cool_raw/`` and
``analysis/compartment/``); ``cpg_txt`` is ``hg38.100kbin.CpG.txt`` (bedtools nuc
output with a name column), the CpG track the notebook used.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RES = 100_000
AUTOS = [f"chr{i}" for i in range(1, 23)]


def notebook_compsaddle(cool, binall, modelall, chroms):
    """``compsaddle`` from HCEA fig4/04.compartment.ipynb (cell 57), verbatim
    except that the chromosome list, bin filter and loadings are arguments."""
    sad = np.zeros((50, 50))
    count = np.zeros((50, 50))
    comptmp = []
    for k, chrom in enumerate(chroms):
        Q = cool.matrix(balance=False, sparse=True).fetch(chrom).toarray()
        Q = Q - np.diag(np.diag(Q))
        binfilter = binall[k]
        Q = Q[binfilter][:, binfilter]
        decay = np.array([np.mean(np.diag(Q, i)) for i in range(Q.shape[0])])
        E = np.zeros(Q.shape)
        row, col = np.diag_indices(E.shape[0])
        E[row, col] = 1
        for i in range(1, E.shape[0]):
            E[row[:-i], col[i:]] = (Q[row[:-i], col[i:]] + 1e-5) / (decay[i] + 1e-5)
        E = E + E.T
        C = np.corrcoef(np.log2(E + 0.001))
        pc = (C - np.mean(C, axis=0)).dot(modelall[k])
        comptmp.append(pc)
        labels, groups = pd.qcut(pc, 50, labels=False, retbins=True)
        sad += np.array([[E[np.ix_(labels == i, labels == j)].sum() for i in range(50)]
                         for j in range(50)])
        count += np.array([[(labels == i).sum() * (labels == j).sum() for i in range(50)]
                           for j in range(50)])
    return np.concatenate(comptmp), sad, count


def main(entex, cpg_txt, results_path, nproc=8):
    import cooler
    import epione as epi
    epi.pl  # noqa: B018
    from epione.single import hic

    nproc = int(nproc)
    raw = Path(entex) / "merged_cool_raw"
    comp_dir = Path(entex) / "analysis" / "compartment"
    results = {}

    cpg = pd.read_csv(cpg_txt, sep="\t")
    cpg = pd.DataFrame({
        "chrom": cpg["#1_usercol"].astype(str), "start": cpg["2_usercol"],
        "end": cpg["3_usercol"],
        "cpg_ratio": cpg["14_user_patt_count"] / (cpg["13_seq_len"] - cpg["11_num_N"])})

    ref = pd.read_hdf(comp_dir / "merged_comp.hdf", key="data")
    ref = ref[ref["chrom"].isin(AUTOS)].reset_index(drop=True)

    # ---- H1 / H2 model ----------------------------------------------------
    t = time.time()
    model = hic.compartment_model(raw / "merged.mcool", cpg, resolution=RES, chromosomes=AUTOS,
                                  clip=None)  # Amendment H-1
    t_fit = time.time() - t
    key = lambda d: d["chrom"].astype(str) + ":" + d["start"].astype(str)
    m = model.set_index(key(model))
    r = ref.set_index(key(ref))
    m = m.loc[r.index]
    same_mask = bool((m["keep"].to_numpy() == r["raw_binfilter"].to_numpy()).all())
    results["H1"] = {"bins": int(len(r)), "kept_epi": int(m["keep"].sum()),
                     "kept_ref": int(r["raw_binfilter"].sum()), "identical_mask": same_mask,
                     "pass": same_mask}
    err = float(np.abs(m["loading"].to_numpy() - r["raw_pcloading"].to_numpy()).max())
    per_chrom_pc = model.groupby("chrom", sort=False)["pc"].first().to_dict()
    results["H2"] = {"max_abs_err": err, "pc_chosen": per_chrom_pc, "seconds_fit": t_fit,
                     "pass": bool(err <= 1e-8)}
    print("H1", results["H1"]); print("H2", results["H2"])

    # the reference model, to isolate projection from fitting
    ref_model = ref[["chrom", "start", "end"]].copy()
    ref_model["keep"] = ref["raw_binfilter"].to_numpy()
    ref_model["loading"] = ref["raw_pcloading"].to_numpy()

    # ---- H3 projection on every L1 type -----------------------------------
    comp_ref = pd.read_hdf(comp_dir / "L1" / "comp.raw.hdf", key="data")
    comp_ref = comp_ref[comp_ref["raw_binfilter"]].reset_index(drop=True)
    types = [c for c in comp_ref.columns if c.startswith("c") and c != "chrom"]
    cools = {ct: raw / "L1" / f"{ct}.mcool" for ct in types}
    t = time.time()
    proj = hic.project_compartments(cools, ref_model, resolution=RES, nproc=nproc)
    t_proj = time.time() - t
    proj_fit = hic.project_compartments(cools, model, resolution=RES, nproc=nproc)
    assert (proj[["chrom", "start"]].to_numpy() == comp_ref[["chrom", "start"]].to_numpy()).all()
    errs = {ct: float(np.abs(proj[ct].to_numpy() - comp_ref[ct].to_numpy()).max()) for ct in types}
    errs_fit = {ct: float(np.abs(proj_fit[ct].to_numpy() - comp_ref[ct].to_numpy()).max())
                for ct in types}
    results["H3"] = {"types": len(types), "max_abs_err": max(errs.values()),
                     "per_type": errs, "end_to_end_max_abs_err": max(errs_fit.values()),
                     "seconds": t_proj, "pass": bool(max(errs.values()) <= 1e-8)}
    print("H3", {k: v for k, v in results["H3"].items() if k != "per_type"})

    # ---- H4 saddle ---------------------------------------------------------
    chroms = AUTOS[:-1]  # the notebook's compsaddle loops over chrom_sizes.index[:-1]
    binall = [ref_model.loc[ref_model["chrom"] == c, "keep"].to_numpy() for c in chroms]
    modelall = [ref_model.loc[(ref_model["chrom"] == c) & ref_model["keep"], "loading"].to_numpy()
                for c in chroms]
    h4 = {}
    for ct in ["c0", "c1", "c7"]:
        uri = f"{raw}/L1/{ct}.mcool::resolutions/{RES}"
        t = time.time()
        sc_ref, sad_ref, cnt_ref = notebook_compsaddle(cooler.Cooler(uri), binall, modelall, chroms)
        t_ref = time.time() - t
        t = time.time()
        out = hic.compartment_saddle(uri, ref_model, chromosomes=chroms, clip=None)
        t_epi = time.time() - t
        rel_sum = float(np.max(np.abs(out["sum"] - sad_ref) / np.abs(sad_ref)))
        rel_cnt = float(np.max(np.abs(out["count"] - cnt_ref) / np.abs(cnt_ref)))
        sc_err = float(np.abs(out["scores"].to_numpy() - sc_ref).max())
        h4[ct] = {"max_rel_err_sum": rel_sum, "max_rel_err_count": rel_cnt,
                  "scores_max_abs_err": sc_err, "strength": out["strength"].to_dict(),
                  "seconds_ref": t_ref, "seconds_epi": t_epi}
    worst = max(max(v["max_rel_err_sum"], v["max_rel_err_count"]) for v in h4.values())
    results["H4"] = {"per_type": h4, "max_rel_err": worst, "pass": bool(worst <= 1e-12)}
    print("H4", {k: v for k, v in results["H4"].items() if k != "per_type"})

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main(*sys.argv[1:])
