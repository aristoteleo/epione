"""Gate evaluation for ``epi.tl.cnv`` vs R epiAneufinder 1.1.5 (see MANIFEST.md).

Usage::

    python parity_epianeufinder.py <R run dir> <python result.pkl> [--json out.json]

``<R run dir>`` holds ``epiAneufinder_results/`` and the ``export/`` folder
written by the reference driver (kept outside this repository); the pickle is
the AnnData returned by ``epi.tl.cnv`` with ``cnv_subclones`` k = 2..6 stored as
``obs['subclone_k{k}']`` and ``uns['cnv']['karyotype']``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from sklearn.metrics import adjusted_rand_score

from epione.tl._cnv import _getbp, _prune, _assign_gainloss, _r_hclust_ward_d, _r_cutree


def _bp_sets(df):
    return {k: sorted(zip(g["Chr"], g["bp"])) for k, g in df.groupby("cell")}


def _jaccard_tol(a, b, tol=1):
    """Jaccard of two breakpoint sets with a +-tol bin positional tolerance."""
    if not a and not b:
        return 1.0
    b_left = list(b)
    match = 0
    for (c, p) in a:
        for j, (c2, p2) in enumerate(b_left):
            if c2 == c and abs(p2 - p) <= tol:
                match += 1
                b_left.pop(j)
                break
    return match / (len(a) + len(b) - match)


def main(rdir, pkl, test="AD"):
    rdir = Path(rdir)
    ex = rdir / "export"
    with open(pkl, "rb") as fh:
        py = pickle.load(fh)
    out = {}

    # ---- G0 -------------------------------------------------------------
    rw = pd.read_csv(ex / "windows.tsv", sep="\t")
    pv = py.var
    rkb = pd.read_csv(ex / "kept_bins.tsv", sep="\t")
    key_r = list(zip(rkb.seqnames.astype(str), rkb.start - 1, rkb.end))
    key_p = list(zip(pv.chrom.astype(str), pv.start, pv.end))
    rcells = [l.strip() for l in open(ex / "cells.txt")]
    out["G0a_bins_equal"] = key_r == key_p
    out["G0a_cells_equal_set"] = set(rcells) == set(py.obs_names)
    out["G0a_cells_equal_order"] = rcells == list(py.obs_names)
    out["G0a_n_bins_r_py"] = [len(key_r), len(key_p)]
    out["G0a_n_cells_r_py"] = [len(rcells), py.n_obs]

    # GC on all count windows (before the zero filters) — compare via kept bins
    rwi = rw.set_index([rw.seqnames.astype(str), rw.start - 1, rw.end])
    sub = rwi.loc[key_p]
    out["G0b_gc_max_abs"] = float(np.max(np.abs(sub.GC.to_numpy() - pv.GC.to_numpy())))
    out["G0b_at_max_abs"] = float(np.max(np.abs(sub.AT.to_numpy() - pv.AT.to_numpy())))
    out["G0b_n_max_abs"] = float(np.max(np.abs(sub.N.to_numpy() - pv.N.to_numpy())))

    raw = scipy.io.mmread(ex / "raw_counts.mtx").tocsc()
    rbc = [l.strip() for l in open(ex / "raw_barcodes.txt")]
    ridx = pd.Index(rbc).get_indexer(py.obs_names)
    rows = pd.MultiIndex.from_tuples(list(zip(rw.seqnames.astype(str), rw.start - 1, rw.end))).get_indexer(
        pd.MultiIndex.from_tuples(key_p))
    Rraw = raw[:, ridx].toarray()[rows].T
    Praw = py.layers["counts"].toarray()
    out["G0c_raw_equal"] = bool(np.array_equal(Rraw, Praw))
    out["G0c_raw_max_abs"] = float(np.abs(Rraw - Praw).max())

    gcm = scipy.io.mmread(ex / "gc_counts.mtx").toarray().T  # cells x kept bins (R order)
    ci = pd.Index(rcells).get_indexer(py.obs_names)
    Rgc = gcm[ci]
    Pgc = py.layers["gc_corrected"].toarray()
    out["G0d_gc_equal_frac"] = float((Rgc == Pgc).mean())
    out["G0d_gc_max_abs"] = float(np.abs(Rgc - Pgc).max())

    # ---- G1a/G1b/G2b: Python segmentation on R's GC-corrected counts ------
    chrom = pv.chrom.astype(str).to_numpy()
    chroms = list(dict.fromkeys(chrom))
    bounds = [(int(np.flatnonzero(chrom == c)[0]), int(np.flatnonzero(chrom == c)[-1]) + 1) for c in chroms]
    rgb = pd.read_csv(ex / "getbp.tsv", sep="\t", dtype={"cell": str})
    rgb_g = {k: g for k, g in rgb.groupby(["cell", "Chr"])}
    n_pairs = 0; n_eq = 0; max_rel = 0.0
    st_agree = 0; st_total = 0
    t_seg = time.perf_counter()
    Rstates = pd.read_csv(rdir / "epiAneufinder_results" / "results_table.tsv", sep=" ")
    rs_cols = [c for c in Rstates.columns if c.startswith("cell-")]
    Rst = Rstates[rs_cols].to_numpy().T
    Rst_cells = [c[5:] for c in rs_cols]
    Rst = Rst[pd.Index(Rst_cells).get_indexer(py.obs_names)]
    for ri, cell in enumerate(rcells):
        y = gcm[ri]
        bps = []
        for c, (a, b) in zip(chroms, bounds):
            p, d = _getbp(y[a:b], k=4, minsize=1, test=test, minsize_cnv=0)
            g = rgb_g.get((cell, c))
            rp = [] if g is None else g.bp.tolist()
            rd = [] if g is None else g.dist.astype(float).tolist()
            n_pairs += 1
            if rp == p:
                n_eq += 1
                if rp:
                    rel = np.max(np.abs(np.array(rd) - np.array(d)) / np.maximum(np.abs(rd), 1e-300))
                    max_rel = max(max_rel, float(rel))
            bps.extend((chroms.index(c), pp, dd) for pp, dd in zip(p, d))
        order = sorted(range(len(bps)), key=lambda i: -bps[i][2])
        pr = _prune([bps[i] for i in order])
        cluster = np.empty(y.size, dtype=np.int64); off = 0
        for ci_, (a, b) in enumerate(bounds):
            cuts = sorted(pp for (cc, pp, _) in pr if cc == ci_)
            edges = [1] + cuts + [b - a + 1]
            cluster[a:b] = np.repeat(np.arange(1, len(edges)), np.diff(edges)) + off
            off += len(edges) - 1
        s = _assign_gainloss(y.astype(np.int64), cluster, is_int=True)
        rrow = Rstates[f"cell-{cell}"].to_numpy()
        st_agree += int((s == rrow).sum()); st_total += s.size
    out["G1a_getbp_identical_frac"] = n_eq / n_pairs
    out["G1a_pairs"] = n_pairs
    out["G1b_dist_max_rel"] = max_rel
    out["G2b_state_agree_given_R_gc"] = st_agree / st_total
    out["seg_on_R_input_seconds"] = time.perf_counter() - t_seg

    # ---- G1c: pruned breakpoints end-to-end ------------------------------
    rpb = pd.read_csv(ex / "pruned_bp.tsv", sep="\t", dtype={"cell": str})
    rsets = _bp_sets(rpb)
    pb = py.uns["cnv"]["breakpoints"].rename(columns={"chrom": "Chr"})
    psets = _bp_sets(pb)
    js = [_jaccard_tol(rsets.get(c, []), psets.get(c, []), 1) for c in py.obs_names]
    js0 = [_jaccard_tol(rsets.get(c, []), psets.get(c, []), 0) for c in py.obs_names]
    out["G1c_bp_jaccard_tol1_mean"] = float(np.mean(js))
    out["G1c_bp_jaccard_exact_mean"] = float(np.mean(js0))

    # ---- G2a -----------------------------------------------------------------
    Pst = np.asarray(py.X)
    out["G2a_state_agree"] = float((Rst == Pst).mean())
    out["G2a_cells_identical_frac"] = float((Rst == Pst).all(1).mean())

    # ---- G3a: subclones ----------------------------------------------------
    rsub = pd.read_csv(ex / "subclones.tsv", sep="\t", dtype={"cell": str})
    aris = {}
    for k in range(2, 7):
        r = rsub[rsub.k == k].set_index("cell").subclone.reindex(py.obs_names)
        if f"subclone_k{k}" in py.obs:
            aris[k] = float(adjusted_rand_score(r.to_numpy(), py.obs[f"subclone_k{k}"].astype(int).to_numpy()))
    out["G3a_subclone_ARI"] = aris
    # same clustering applied to R's state matrix (isolates the hclust port)
    merge, h, o = _r_hclust_ward_d(Rst.astype(float))
    aris_r = {}
    for k in range(2, 7):
        r = rsub[rsub.k == k].set_index("cell").subclone.reindex(py.obs_names).to_numpy()
        aris_r[k] = float(adjusted_rand_score(r, _r_cutree(merge, k)))
    out["G3a_hclust_port_on_R_states_ARI"] = aris_r

    # ---- G3b: karyotype measures -----------------------------------------
    kg = pd.read_csv(ex / "karyo_genomewide.tsv", sep="\t")
    kc = pd.read_csv(ex / "karyo_chrom.tsv", sep="\t").set_index("chrom")
    km = py.uns["cnv"]["karyotype"]
    d1 = np.abs(kg.to_numpy().ravel() - km["genomewide"].to_numpy().ravel()).max()
    d2 = np.abs(kc.loc[km["per_chromosome"].index].to_numpy() - km["per_chromosome"].to_numpy()).max()
    out["G3b_karyotype_max_abs"] = float(max(d1, d2))

    # ---- runtime -------------------------------------------------------------
    rt = ex / "runtime_seconds.txt"
    out["runtime_R_seconds"] = float(rt.read_text()) if rt.exists() else None
    out["runtime_py_seconds"] = py.uns["cnv"]["timings"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rdir"); ap.add_argument("pkl")
    ap.add_argument("--test", default="AD"); ap.add_argument("--json")
    a = ap.parse_args()
    res = main(a.rdir, a.pkl, a.test)
    s = json.dumps(res, indent=1, default=str)
    print(s)
    if a.json:
        Path(a.json).write_text(s)
