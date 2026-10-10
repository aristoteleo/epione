"""Epione vs ArchR 1.0.3 parity for add_module_score / plot_groups / confusion_matrix.

Usage: python py_parity.py [<parity dir>] [out.json]

``<parity dir>`` defaults to ``$EPIONE_ARCHR_PARITY_DIR/module_score_groups`` and
holds ArchR 1.0.3 reference outputs generated outside this repository:
``tables/`` (GeneScoreMatrix + cell metadata exported from the ArchR project),
``ref/`` (ArchR results) and ``arrow_genescore_rowsums.npy``. Evaluates every
gate pre-registered in MANIFEST.md.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import anndata as ad

import epione as epi
from epione.pl._groups import group_plot_data

MODULES = {
    "B": ["MS4A1", "CD79A", "PAX5"],
    "T": ["CD3D", "CD3E", "CD2"],
    "Mono": ["CD14", "LYZ", "CEBPB"],
    "Erythroid": ["GATA1", "KLF1", "HBB"],
    "Progenitor": ["CD34"],
}
CONFIGS = [(25, 100, 1), (25, 25, 7)]


def load_gene_scores(tables: Path) -> ad.AnnData:
    m = sio.mmread(tables / "GeneScoreMatrix.mtx").T.tocsr()
    cells = (tables / "GeneScoreMatrix_cells.txt").read_text().split()
    feats = pd.read_csv(tables / "GeneScoreMatrix_features.tsv", sep="\t")
    obs = pd.read_csv(tables / "cells.tsv", sep="\t", index_col=0).loc[cells]
    a = ad.AnnData(m, obs=obs, var=feats.set_index(feats["name"].astype(str).values))
    a.var_names_make_unique()
    return a


def _scores(ref: Path, tag: str) -> pd.DataFrame:
    return pd.read_csv(ref / f"module_scores_{tag}.tsv", sep="\t", index_col=0)


def _bgd(ref: Path, tag: str) -> dict:
    b = pd.read_csv(ref / f"module_bgd_{tag}.tsv", sep="\t")
    return {m: (g["match"].values - 1).tolist() for m, g in b.groupby("module", sort=False)}


def module_score_parity(adata, ref: Path) -> list:
    rows = []
    for nbin, nbgd, seed in CONFIGS:
        sfx = f"nbin{nbin}_nbgd{nbgd}_seed{seed}"
        jobs = [(f"official_{m}", {m: g}) for m, g in MODULES.items() if len(g) >= 2]
        jobs.append(("patched_all", MODULES))
        for tag, feats in jobs:
            t = f"{tag}_{sfx}"
            r = _scores(ref, t).loc[adata.obs_names]
            rb = _bgd(ref, t)
            inj = epi.tl.add_module_score(adata, feats, n_bin=nbin, n_bgd=nbgd, background=rb,
                                          sample_key="Sample", inplace=False, verbose=False)
            free = epi.tl.add_module_score(adata, feats, n_bin=nbin, n_bgd=nbgd, seed=seed,
                                           sample_key="Sample", inplace=False, verbose=False)
            free_glob = epi.tl.add_module_score(adata, feats, n_bin=nbin, n_bgd=nbgd, seed=seed,
                                                inplace=False, verbose=False)
            adata2 = adata.copy()
            epi.tl.add_module_score(adata2, feats, n_bin=nbin, n_bgd=nbgd, seed=seed,
                                    sample_key="Sample", verbose=False)
            pb = {m: adata.var_names.get_indexer(v).tolist()
                  for m, v in adata2.uns["module_score"]["background"].items()}
            bg_equal = all(pb[m] == rb[m] for m in rb)
            cols = list(r.columns)
            rows.append(dict(
                config=t,
                bgd_exact=bool(bg_equal),
                max_abs_injected=float(np.abs(inj[cols].values - r.values).max()),
                max_abs_free=float(np.abs(free[cols].values - r.values).max()),
                max_abs_free_global_sums=float(np.abs(free_glob[cols].values - r.values).max()),
                pearson_free_global_sums=float(min(np.corrcoef(free_glob[c], r[c])[0, 1] for c in cols)),
            ))
    return rows


def binning_parity(adata, ref: Path, arrow_sums: np.ndarray | None) -> dict:
    rs = pd.read_csv(ref / "gs_rowsums.tsv", sep="\t")
    from epione.tl._module_score import _feature_sums
    s_sample = _feature_sums(adata, adata.X, "Sample")
    s_global = _feature_sums(adata, adata.X, None)
    out = dict(
        rowsums_vs_tsv15_max_rel=float(np.max(np.abs(s_sample - rs.rowSums.values) / np.maximum(1, np.abs(s_sample)))),
        order_sample_vs_global_npos=int((np.argsort(s_sample, kind="stable") != np.argsort(s_global, kind="stable")).sum()),
    )
    if arrow_sums is not None:
        out["sample_sums_equal_arrow_bitwise"] = bool(np.array_equal(s_sample, arrow_sums))
        out["global_sums_equal_arrow_bitwise"] = bool(np.array_equal(s_global, arrow_sums))
        out["global_sums_n_bitdiff"] = int((s_global != arrow_sums).sum())
    return out


def confusion_parity(adata, ref: Path) -> dict:
    r = pd.read_csv(ref / "cm_clusters_sample.tsv", sep="\t", index_col=0)
    p = epi.tl.confusion_matrix(adata.obs["Clusters"], adata.obs["Sample"])
    same_order = list(p.index) == list(r.index) and list(p.columns) == list(r.columns)
    return dict(same_order=bool(same_order), max_abs=float(np.abs(p.values - r.values).max()))


def arrow_rowsums(arrow_dir: Path, seqnames) -> np.ndarray | None:
    try:
        import h5py
    except ImportError:
        return None
    files = sorted(arrow_dir.glob("*.arrow"))
    if not files:
        return None
    tot = None
    for f in files:
        with h5py.File(f, "r") as h:
            v = np.concatenate([h[f"GeneScoreMatrix/{c}/rowSums"][:].ravel() for c in seqnames])
        tot = v if tot is None else tot + v
    return tot


def plot_groups_parity(adata, ref: Path) -> list:
    res = []
    cases = [("gs_{g}_{p}", g, "Clusters", 1000, 1, "pg_cells_clusters_seed1.txt") for g in ["CD34", "GATA1", "MS4A1", "LYZ"]]
    cases += [("tss_sample_{p}", "TSSEnrichment", "Sample", 500, 2, "pg_cells_sample_seed2.txt")]
    for tmpl, name, group_by, max_cells, seed, cell_file in cases:
        for plot_as in ["ridges", "violin"]:
            tag = tmpl.format(g=name, p=plot_as)
            d = group_plot_data(adata, group_by=group_by, name=name, plot_as=plot_as,
                                max_cells=max_cells, seed=seed)
            r_cells = (ref / cell_file).read_text().split()
            r_data = pd.read_csv(ref / f"pg_{tag}_data.tsv", sep="\t")
            r_levels = (ref / f"pg_{tag}_levels.txt").read_text().split("\n")
            r_levels = [x for x in r_levels if x]
            row = dict(case=tag,
                       cells_exact=list(d["cells"]) == r_cells,
                       values_max_abs=float(np.abs(d["y"] - r_data["y"].values).max()),
                       groups_exact=list(d["x"]) == list(r_data["x"].astype(str)),
                       levels_exact=list(d["levels"]) == r_levels)
            l1 = pd.read_csv(ref / f"pg_{tag}_layer1.tsv", sep="\t")
            if plot_as == "ridges":
                l1 = l1[l1["datatype"] == "ridgeline"]
                dens = d["ridges"]
                errs = []
                for gi, g in dens.groupby("y"):
                    rg = l1[l1["y"] == gi]
                    errs += [np.abs(g["x"].values - rg["x"].values).max(),
                             np.abs(g["density"].values - rg["density"].values).max(),
                             np.abs(g["ymax"].values - rg["ymax"].values).max()]
                row["density_max_abs"] = float(max(errs))
            else:
                v = d["violin"]
                errs = []
                for gi, g in v.groupby("x"):
                    rg = l1[l1["x"] == gi].sort_values(["y"], kind="stable")
                    g = g.sort_values(["y"], kind="stable")
                    errs += [np.abs(g["y"].values - rg["y"].values).max(),
                             np.abs(g["density"].values - rg["density"].values).max(),
                             np.abs(g["violinwidth"].values - rg["violinwidth"].values).max()]
                row["density_max_abs"] = float(max(errs))
                l2 = pd.read_csv(ref / f"pg_{tag}_layer2.tsv", sep="\t")
                b = d["box"]
                cols = ["ymin", "lower", "middle", "upper", "ymax"]
                row["box_max_abs"] = float(np.abs(b[cols].values - l2[cols].values).max())
            res.append(row)
    return res


def parity_dir() -> Path:
    return Path(os.environ.get("EPIONE_ARCHR_PARITY_DIR", "")) / "module_score_groups"


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else parity_dir()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    tables, ref = root / "tables", root / "ref"
    adata = load_gene_scores(tables)
    sums = root / "arrow_genescore_rowsums.npy"
    arrows = np.load(sums) if sums.exists() else None
    report = dict(
        confusion=confusion_parity(adata, ref),
        binning=binning_parity(adata, ref, arrows),
        module_score=module_score_parity(adata, ref),
        plot_groups=plot_groups_parity(adata, ref),
    )
    txt = json.dumps(report, indent=1)
    print(txt)
    if out is not None:
        out.write_text(txt)


if __name__ == "__main__":
    main()
