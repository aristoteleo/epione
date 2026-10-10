"""Compare Epione marker / peak-to-gene heatmaps with ArchR reference dumps.

Usage: python parity_check.py [<dir containing ref/>] [out.json]
(default dir: $EPIONE_ARCHR_PARITY_DIR/marker_heatmap)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import epione as epi
from epione.pl._marker_heatmap import MarkerAssays, peak2gene_heatmap_matrices

MARKER_GENES = ["CD34", "GATA1", "PAX5", "MS4A1", "EBF1", "MME", "CD14", "CEBPB", "MPO",
                "IRF8", "CD3D", "CD8A", "TBX21", "IL7R"]
CONFIGS = {
    "gs_default": ("GS", dict(cut_off="FDR <= 0.01 & Log2FC >= 1.25", label_markers=MARKER_GENES, transpose=True)),
    "gs_rows": ("GS", dict(cut_off="FDR <= 0.01 & Log2FC >= 1.25")),
    "gs_log2fc": ("GS", dict(cut_off="FDR <= 0.05 & abs(Log2FC) >= 1", plot_log2fc=True, limits=(-3, 3))),
    "gs_nobinary": ("GS", dict(cut_off="FDR <= 0.01 & Log2FC >= 1.5", binary_cluster_rows=False, n_label=5)),
    "gs_subset": ("GS", dict(subset_markers=[5, 40, 400, 900, 1500, 2600, 5000, 7000, 9000, 12000, 15000, 20000])),
    "pk_default": ("PK", dict(cut_off="FDR <= 0.1 & Log2FC >= 0.5", transpose=True)),
    "pk_rows": ("PK", dict(cut_off="FDR <= 0.1 & Log2FC >= 0.5")),
    "pk_invert": ("PK", dict(cut_off="FDR <= 0.1 & Log2FC <= -0.5", invert=True)),
    "pk_nocluster": ("PK", dict(cut_off="FDR <= 0.05 & Log2FC >= 1 & !(MeanBGD > 0.2)", cluster_cols=False,
                                scale_rows=False, log2_norm=False, limits=(0, 0.5))),
}
P2G_CONFIGS = {"p2g_default": dict(k=25, n_plot=25000, seed=1), "p2g_small": dict(k=12, n_plot=4000, seed=7)}


def read_bin(ref: Path, tag: str):
    dim = [int(x) for x in (ref / f"{tag}.dim").read_text().split()]
    a = np.fromfile(ref / f"{tag}.f64", dtype="<f8").reshape(dim[1], dim[0]).T
    rows = (ref / f"{tag}.rows").read_text().splitlines() if (ref / f"{tag}.rows").exists() else None
    cols = (ref / f"{tag}.cols").read_text().splitlines() if (ref / f"{tag}.cols").exists() else None
    return a, rows, cols


def read_vec(ref: Path, tag: str):
    return (ref / f"{tag}.txt").read_text().splitlines()


def load_markers(ref: Path, tag: str) -> MarkerAssays:
    groups = read_vec(ref, f"{tag}__groups")
    rd = pd.read_csv(ref / f"{tag}__rowData.tsv", sep="\t")
    assays = {}
    for f in sorted(ref.glob(f"{tag}__*.f64")):
        an = f.stem.split("__", 1)[1]
        a, _, _ = read_bin(ref, f.stem)
        assays[an] = pd.DataFrame(a, columns=groups)
    return MarkerAssays(assays, rd, read_vec(ref, f"{tag}__useMatrix")[0])


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def marker_parity(ref: Path):
    se = {"GS": load_markers(ref, "GS"), "PK": load_markers(ref, "PK")}
    out = {}
    for name, (tag, kw) in CONFIGS.items():
        res = epi.pl.marker_heatmap_matrix(se[tag], **kw)
        rmat, rrows, rcols = read_bin(ref, f"hm_{name}__mat")
        py = res.matrix
        rec = {
            "shape_equal": list(py.shape) == list(rmat.shape),
            "rows_equal": rrows == [str(x) for x in py.index],
            "cols_equal": rcols == [str(x) for x in py.columns],
        }
        rec["max_abs_err"] = float(np.nanmax(np.abs(py.to_numpy() - rmat))) if rec["shape_equal"] else None
        rec["nan_pattern_equal"] = bool(rec["shape_equal"] and np.array_equal(np.isnan(py.to_numpy()), np.isnan(rmat)))
        ro = np.array(read_vec(ref, f"hm_{name}__row_order"), dtype=int) - 1
        co = np.array(read_vec(ref, f"hm_{name}__col_order"), dtype=int) - 1
        rec["row_order_equal"] = bool(np.array_equal(ro, res.row_order))
        rec["col_order_equal"] = bool(np.array_equal(co, res.col_order))
        lab_file = ref / f"hm_{name}__labels.txt"
        if lab_file.exists():
            rlab = lab_file.read_text().splitlines()
            rat = np.array(read_vec(ref, f"hm_{name}__label_at"), dtype=int) - 1
            names = np.asarray(res.matrix.columns if res.transpose else res.matrix.index)
            # anno_mark stores `at` sorted, so compare the (position, label) pairs.
            r_pairs = sorted(zip(rat.tolist(), rlab))
            p_pairs = sorted(zip(res.label_index.tolist(), [str(x) for x in names[res.label_index]]))
            rec["labels_equal"] = [l for _, l in r_pairs] == [l for _, l in p_pairs]
            rec["label_positions_equal"] = r_pairs == p_pairs
        rec["n_features"] = int(py.shape[1] if res.transpose else py.shape[0])
        out[name] = rec
    return out


def load_p2g(ref: Path):
    a, _, _ = read_bin(ref, "p2g_all")
    links = pd.DataFrame(a, columns=["idxATAC", "idxRNA", "Correlation", "FDR", "VarQATAC", "VarQRNA"])
    atac, _, _ = read_bin(ref, "p2g_seATAC")
    rna, _, _ = read_bin(ref, "p2g_seRNA")
    peaks = read_vec(ref, "p2g_seATAC_peaks")
    genes = read_vec(ref, "p2g_seRNA_genes")
    knn = pd.read_csv(ref / "p2g_knn.tsv", sep="\t")
    cells = pd.read_csv(ref / "cells_clusters.tsv", sep="\t")
    pos = pd.Series(np.arange(len(cells)), index=cells["cell"])
    knn_index = [pos[g["cell"].to_numpy()].to_numpy() for _, g in knn.groupby("group", sort=True)]
    return dict(links=links, atac_knn=atac, rna_knn=rna, peak_names=peaks, gene_names=genes,
                knn_index=knn_index, cell_groups=cells["Clusters"].to_numpy())


def p2g_parity(ref: Path):
    inputs = load_p2g(ref)
    out = {}
    for tag, kw in P2G_CONFIGS.items():
        ratac, rrows, rcols = read_bin(ref, f"{tag}__ATAC")
        rrna, _, _ = read_bin(ref, f"{tag}__RNA")
        rk = np.array(read_vec(ref, f"{tag}__kmeansId"), dtype=int)
        rgroups = read_vec(ref, f"{tag}__colGroups")
        rcolorder = read_vec(ref, f"{tag}__colOrder")
        rpeak = read_vec(ref, f"{tag}__peak")
        k1 = np.array(read_vec(ref, f"{tag}__k1cluster"), dtype=int)
        kdf = pd.read_csv(ref / f"{tag}__kDF.tsv", sep="\t")

        def score(res):
            same_shape = res.atac.shape == ratac.shape
            return {
                "atac_max_abs_err": float(np.max(np.abs(res.atac.to_numpy() - ratac))) if same_shape else None,
                "rna_max_abs_err": float(np.max(np.abs(res.rna.to_numpy() - rrna))) if same_shape else None,
                "rows_equal": list(res.atac.index) == rrows,
                "col_order_equal": list(res.atac.columns) == rcolorder == rcols,
                "kmeans_id_equal": bool(np.array_equal(res.kmeans_id, rk)),
                "col_groups_equal": list(res.col_groups) == rgroups,
                "links_equal": list(res.links["peak"]) == rpeak,
                "kdf_equal": bool(np.array_equal(res.kdf["idx"].to_numpy(), kdf["idx"].to_numpy())),
                "n_links": int(res.atac.shape[0]),
            }

        inj = peak2gene_heatmap_matrices(**inputs, kmeans_cluster=k1, **kw)
        free = peak2gene_heatmap_matrices(**inputs, **kw)
        # The reference session ran addArchRThreads(8) -> RNGkind("L'Ecuyer-CMRG").
        out[tag] = {"injected": score(inj),
                    "free": dict(score(free), kmeans_ari=ari(free.kmeans_cluster, k1),
                                 kmeans_identical=bool(np.array_equal(free.kmeans_cluster, k1)))}
    return out


def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def e2e_metrics(work: Path):
    """Informational: Epione's own upstream tools vs ArchR (not gated)."""
    ref, cache = work / "ref", work / "cache"
    out = {}
    gs = load_markers(ref, "GS")
    names = gs.row_data["name"].astype(str).to_numpy()
    em = pd.read_parquet(cache / "epi_markers_gs.parquet")
    a_lfc = gs.assays["Log2FC"].set_axis(names)
    e_lfc = em.pivot(index="feature", columns="group", values="log2_fc").reindex(index=names, columns=gs.groups)
    ok = np.isfinite(a_lfc.to_numpy()) & np.isfinite(e_lfc.to_numpy())
    out["gs_log2fc_spearman"] = _spearman(a_lfc.to_numpy()[ok], e_lfc.to_numpy()[ok])
    a_fdr = gs.assays["FDR"].set_axis(names)
    e_fdr = em.pivot(index="feature", columns="group", values="fdr").reindex(index=names, columns=gs.groups)
    jac = {}
    for g in gs.groups:
        ta = a_lfc[g][a_fdr[g] <= 0.01].sort_values(ascending=False).index[:50]
        te = e_lfc[g][e_fdr[g] <= 0.01].sort_values(ascending=False).index[:50]
        jac[g] = len(set(ta) & set(te)) / max(len(set(ta) | set(te)), 1)
    out["gs_top50_jaccard_per_cluster"] = jac
    out["gs_top50_jaccard_median"] = float(np.median(list(jac.values())))
    cut = "FDR <= 0.01 & Log2FC >= 1.25"
    ha = set(epi.pl.marker_heatmap_matrix(gs, cut_off=cut).matrix.index)
    he = set(epi.pl.marker_heatmap_matrix(em, cut_off=cut, use_matrix="GeneScoreMatrix").matrix.index)
    out["gs_heatmap_marker_jaccard"] = len(ha & he) / len(ha | he)
    out["gs_heatmap_n_archr"], out["gs_heatmap_n_epione"] = len(ha), len(he)

    p2g = load_p2g(ref)
    L = p2g["links"]
    peaks = np.asarray(p2g["peak_names"]); genes = np.asarray(p2g["gene_names"])
    key_a = pd.Series(L["Correlation"].to_numpy(),
                      index=peaks[L["idxATAC"].astype(int) - 1] + "|" + genes[L["idxRNA"].astype(int) - 1])
    el = pd.read_parquet(cache / "epi_p2g.parquet")
    peak_fmt = el["chrom"].astype(str) + ":" + el["peak_start"].astype(str) + "-" + el["peak_end"].astype(str)
    key_e = pd.Series(el["correlation"].to_numpy(dtype=float), index=peak_fmt + "|" + el["gene"].astype(str))
    key_a = key_a[~key_a.index.duplicated()]; key_e = key_e[~key_e.index.duplicated()]
    key_a, key_e = key_a.dropna(), key_e.dropna()
    common = key_a.index.intersection(key_e.index)
    out["p2g_shared_pairs"] = int(len(common))
    out["p2g_pairs_archr"], out["p2g_pairs_epione"] = int(len(key_a)), int(len(key_e))
    out["p2g_correlation_spearman"] = _spearman(key_a[common], key_e[common])
    return out


def main():
    if len(sys.argv) > 1:
        work = Path(sys.argv[1])
    else:
        work = Path(os.environ["EPIONE_ARCHR_PARITY_DIR"]) / "marker_heatmap"
    ref = work / "ref"
    res = {"marker_heatmap": marker_parity(ref), "peak2gene_heatmap": p2g_parity(ref)}
    if (work / "cache" / "epi_p2g.parquet").exists():
        res["end_to_end_informational"] = e2e_metrics(work)
    txt = json.dumps(res, indent=2)
    print(txt)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(txt)


if __name__ == "__main__":
    main()
