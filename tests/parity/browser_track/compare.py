"""Compare epione.pl.plot_browser_track against ArchR reference outputs.

Usage: python compare.py <ref_dir> <fixture_dir> <fragments_dir> [out.json]
``ref_dir`` is the output directory of r_reference.R.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from epione.pl._browser_track import (
    archr_palette_discrete, browser_track_coverage, clip_coverage,
    feature_track_data, gene_region, gene_track_data, loop_track_data,
)

GENES = ["CD34", "GATA1", "MS4A1", "LYZ", "CD3D"]
SAMPLES = ["scATAC_BMMC_R1", "scATAC_CD34_BMMC_R1", "scATAC_PBMC_R1"]


def load_inputs(fixture: Path, frag_dir: Path):
    import anndata as ad
    cells = pd.read_csv(fixture / "tables" / "cells.tsv", sep="\t", index_col=0)
    adata = ad.AnnData(obs=cells)
    genes = pd.read_csv(fixture / "tables" / "genes_hg19.tsv", sep="\t")
    peaks = pd.read_csv(fixture / "tables" / "peaks.tsv", sep="\t")
    frags = {s: str(frag_dir / f"{s}.fragments.tsv.gz") for s in SAMPLES}
    return adata, genes, peaks, frags


def _max_abs(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.abs(np.where(both_nan, 0.0, a - b))
    if np.isnan(d).any():
        return float("inf")
    return float(d.max()) if d.size else 0.0


def compare(ref: Path, fixture: Path, frag_dir: Path) -> dict:
    adata, genes, peaks, frags = load_inputs(fixture, frag_dir)
    loops = pd.read_csv(ref / "loops.tsv", sep="\t")
    res = {}
    pal_r = pd.read_csv(ref / "palette.tsv", sep="\t")
    pal_py = archr_palette_discrete(adata.obs["Clusters"].astype(str))
    res["palette_identical"] = bool(list(pal_py.values()) == list(pal_r["color"]) and
                                    list(pal_py.keys()) == list(pal_r["group"]))
    for g in GENES:
        out = {}
        reg_r = pd.read_csv(ref / f"{g}_region.tsv", sep="\t").iloc[0]
        reg = gene_region(genes, g)
        out["region_equal"] = reg == (reg_r["chrom"], int(reg_r["start"]), int(reg_r["end"]))
        cells_r = [l.strip() for l in open(ref / f"{g}_cells.txt") if l.strip()]
        cov = browser_track_coverage(adata, reg, fragments=frags, cells=cells_r)
        raw = pd.read_csv(ref / f"{g}_raw.tsv", sep="\t")
        out["n_rows_py"], out["n_rows_r"] = len(cov), len(raw)
        out["groups_equal"] = bool((cov["group"].astype(str).to_numpy() == raw["group"].astype(str).to_numpy()).all()) \
            if len(cov) == len(raw) else False
        out["x_max_abs"] = _max_abs(cov["x"], raw["x"])
        out["coverage_max_abs"] = _max_abs(cov["y"], raw["y"])
        bulk_r = pd.read_csv(ref / f"{g}_bulk.tsv", sep="\t")
        bulk, _ = clip_coverage(cov)
        out["bulk_clipped_max_abs"] = _max_abs(bulk["y"], bulk_r["y"])

        free_r = browser_track_coverage(adata, reg, fragments=frags, seed=1, rng="archr")
        out["free_archr_rng_same_cells"] = sorted(free_r.attrs["cells"]) == sorted(cells_r)
        out["free_archr_rng_max_abs"] = _max_abs(free_r["y"], raw["y"]) if len(free_r) == len(raw) else float("inf")

        key = ["x", "group"]
        rs = []
        for s in range(5):
            free = browser_track_coverage(adata, reg, fragments=frags, seed=s, rng="numpy")
            m = raw.dropna(subset=["x"]).merge(free.dropna(subset=["x"]), on=key, how="outer",
                                               suffixes=("_r", "_py")).fillna(0.0)
            rs.append(float(np.corrcoef(m["y_r"], m["y_py"])[0, 1]))
        # ArchR-vs-ArchR noise floor: R's draw under set.seed(2) vs set.seed(1).
        r2 = browser_track_coverage(adata, reg, fragments=frags, seed=2, rng="archr")
        m = raw.dropna(subset=["x"]).merge(r2.dropna(subset=["x"]), on=key, how="outer",
                                           suffixes=("_r", "_r2")).fillna(0.0)
        out["archr_seed1_vs_seed2_pearson"] = float(np.corrcoef(m["y_r"], m["y_r2"])[0, 1])
        out["free_numpy_pearson"] = rs[0]
        out["free_numpy_pearson_seeds_min_mean"] = [min(rs), float(np.mean(rs))]

        f_r = pd.read_csv(ref / f"{g}_features.tsv", sep="\t")
        f_py = feature_track_data(peaks, reg)
        out["features_identical"] = bool(
            sorted(zip(f_py["start"], f_py["end"])) == sorted(zip(f_r["start"], f_r["end"])))

        gpath = ref / f"{g}_genes.tsv"
        if gpath.exists():
            g_r = pd.read_csv(gpath, sep="\t")
            g_py = gene_track_data(genes, reg)
            cols = ["symbol", "start", "end", "strand", "cluster"]
            out["genes_identical"] = bool(
                len(g_py) == len(g_r) and
                (g_py[cols].astype(str).to_numpy() == g_r[cols].astype(str).to_numpy()).all())

        lpath = ref / f"{g}_loops.tsv"
        if lpath.exists():
            l_r = pd.read_csv(lpath, sep="\t")
            l_py = loop_track_data(loops, reg)
            out["n_loops"] = int(l_r["id"].nunique())
            out["loops_max_abs"] = max(_max_abs(l_py["x"], l_r["x"]), _max_abs(l_py["y"], l_r["y"]))
        res[g] = out
    return res


if __name__ == "__main__":
    ref, fixture, frag_dir = (Path(p) for p in sys.argv[1:4])
    res = compare(ref, fixture, frag_dir)
    text = json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(text)
    if len(sys.argv) > 4:
        Path(sys.argv[4]).write_text(text)
