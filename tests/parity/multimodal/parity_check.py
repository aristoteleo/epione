"""Parity checks for epione's multimodal functions (see MANIFEST.md).

Usage::

    EPIONE_MULTIMODAL_PARITY_DIR=/path/to/multimodal_parity \
        python tests/parity/multimodal/parity_check.py combined_dims
    python tests/parity/multimodal/parity_check.py wnn

References live outside the repository: ``combined_dims/`` holds ArchR's
inputs and ``addCombinedDims`` outputs, ``wnn/`` holds the shared embeddings
and Seurat's / muon's WNN results.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PARITY = Path(os.environ.get("EPIONE_MULTIMODAL_PARITY_DIR", ""))

C_CONFIGS = {
    # id: (dim_weights, dims_to_use, scaleDims as given to ArchR, cor_cutoff)
    "C1": (None, None, None, 0.75),
    "C2": ([1.0, 0.5], None, None, 0.75),
    "C3": (None, list(range(1, 21)), True, 0.75),
    "C4": (None, None, False, 0.5),
}


def _read_tsv(path):
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.to_numpy(dtype=np.float64), list(df.index)


def check_combined_dims() -> dict:
    import anndata as ad
    import epione as epi

    d = PARITY / "combined_dims"
    reps, info = {}, {}
    for rd in ("IterativeLSI", "Harmony"):
        reps[rd], cells = _read_tsv(d / f"in_{rd}.tsv")
        info[rd] = json.loads((d / f"in_{rd}.json").read_text())
    adata = ad.AnnData(np.zeros((len(cells), 1)), obs=pd.DataFrame(index=cells))
    for rd, m in reps.items():
        adata.obsm[rd] = m

    results = {}
    for cid, (w, dims, scale_arg, cut) in C_CONFIGS.items():
        # ArchR getReducedDims: a stored scaleDims of NA forces FALSE; NULL uses the stored value.
        scales, ctd = [], {}
        for rd in ("IterativeLSI", "Harmony"):
            stored = info[rd]["scaleDims"]
            if stored is None:
                s = False
            elif scale_arg is None:
                s = bool(stored)
            else:
                s = bool(scale_arg)
            scales.append(s)
            key = "corToDepth_scaled" if s else "corToDepth_none"
            ctd[rd] = info[rd][key] if info[rd][key] is not None else None
        out = epi.tl.combined_dims(adata, ["IterativeLSI", "Harmony"], dim_weights=w, dims_to_use=dims,
                                   scale_dims=scales, cor_cutoff=cut, cor_to_depth=ctd, key_added=cid)
        ref, ref_cells = _read_tsv(d / f"out_{cid}.tsv")
        assert ref_cells == cells
        same_shape = out.shape == ref.shape
        maxdiff = float(np.max(np.abs(out - ref))) if same_shape else float("inf")
        results[cid] = {"shape_epione": list(out.shape), "shape_archr": list(ref.shape),
                        "max_abs_diff": maxdiff, "pass": bool(same_shape and maxdiff <= 1e-10)}
    return results


def main(argv):
    what = argv[1] if len(argv) > 1 else "combined_dims"
    res = {"combined_dims": check_combined_dims, "wnn": check_wnn, "reader": check_reader}[what]()
    print(json.dumps(res, indent=2))
    return 0 if all(v.get("pass", True) for k, v in res.items() if isinstance(v, dict) and k != "muon (information)") else 1


def _knn_graph(idx_noself: np.ndarray):
    """Symmetric binary kNN graph from neighbour sets (the cell itself excluded)."""
    import scipy.sparse as sp

    n, k = idx_noself.shape
    rows = np.repeat(np.arange(n), k)
    cols = idx_noself.ravel()
    keep = cols >= 0
    g = sp.csr_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(n, n))
    return g.maximum(g.T).tocsr()


def _leiden(graph, resolution=1.0, seed=0):
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(np.zeros((graph.shape[0], 1)))
    a.obsp["connectivities"] = graph
    a.obsp["distances"] = graph
    a.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                          "params": {"method": "umap"}}
    sc.tl.leiden(a, resolution=resolution, random_state=seed, flavor="igraph", n_iterations=-1, directed=False)
    return np.asarray(a.obs["leiden"])


def _jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(a.shape[0])
    for i in range(a.shape[0]):
        sa, sb = set(a[i][a[i] >= 0]), set(b[i][b[i] >= 0])
        out[i] = len(sa & sb) / max(len(sa | sb), 1)
    return out


def check_wnn() -> dict:
    import time
    import anndata as ad
    import epione as epi
    from sklearn.metrics import adjusted_rand_score as ari

    d = PARITY / "wnn"
    pca, cells = _read_tsv(d / "emb_pca.tsv")
    lsi, _ = _read_tsv(d / "emb_lsi.tsv")
    ct = pd.read_csv(d / "cell_type.tsv", sep="\t", index_col=0)["cell_type"].reindex(cells).to_numpy()
    adata = ad.AnnData(np.zeros((len(cells), 1)), obs=pd.DataFrame(index=cells))
    adata.obsm["X_pca"], adata.obsm["X_lsi"] = pca, lsi
    t0 = time.time()
    w, idx = epi.tl.wnn(adata, reps=["X_pca", "X_lsi"], dims=[None, range(2, 51)], modality_names=["RNA", "ATAC"])
    secs = time.time() - t0
    ep_noself = idx[:, 1:]

    sw = pd.read_csv(d / "seurat_weights.tsv", sep="\t", index_col=0).reindex(cells)
    s_idx = pd.read_csv(d / "seurat_nn_idx.tsv", sep="\t").to_numpy() - 1
    self_first = float(np.mean(s_idx[:, 0] == np.arange(len(cells))))
    s_noself = np.array([[j for j in row if j != i][: s_idx.shape[1] - 1] for i, row in enumerate(s_idx)])

    mw = pd.read_csv(d / "muon_weights.tsv", sep="\t", index_col=0).reindex(cells)
    m_noself = np.load(d / "muon_nn_idx_noself.npy")

    def compare(ref_w, ref_noself):
        r = float(np.corrcoef(w[:, 0], ref_w)[0, 1])
        mad = float(np.median(np.abs(w[:, 0] - ref_w)))
        jac = _jaccard(ep_noself, ref_noself)
        cl_e = _leiden(_knn_graph(ep_noself))
        cl_r = _leiden(_knn_graph(ref_noself))
        a_er = float(ari(cl_e, cl_r))
        a_e, a_r = float(ari(ct, cl_e)), float(ari(ct, cl_r))
        return {
            "W1_weight_pearson": r, "W1_weight_median_absdiff": mad,
            "W1_pass": bool(r >= 0.95 and mad <= 0.05),
            "W2_median_jaccard": float(np.median(jac)), "W2_pass": bool(np.median(jac) >= 0.70),
            "W3_ari_partitions": a_er, "W3_pass": bool(a_er >= 0.80),
            "W4_celltype_ari_epione": a_e, "W4_celltype_ari_ref": a_r,
            "W4_pass": bool(abs(a_e - a_r) <= 0.05),
            "n_clusters_epione": int(len(set(cl_e))), "n_clusters_ref": int(len(set(cl_r))),
        }

    res = {
        "seurat": compare(sw["RNA.weight"].to_numpy(), s_noself),
        "muon (information)": compare(mw["rna:mod_weight"].to_numpy(), m_noself),
        "info": {"epione_seconds": secs, "seurat_self_first_fraction": self_first,
                 "epione_rna_weight_mean": float(w[:, 0].mean()),
                 "seurat_rna_weight_mean": float(sw["RNA.weight"].mean()),
                 "muon_rna_weight_mean": float(mw["rna:mod_weight"].mean())},
    }
    res["seurat"]["pass"] = all(res["seurat"][g] for g in ("W1_pass", "W2_pass", "W3_pass", "W4_pass"))
    return res


def check_reader() -> dict:
    """Gate T (with amendment T-A1): read_10x_multiome on the public cellranger-arc h5."""
    import h5py
    import scipy.sparse as sp
    import epione as epi

    d = PARITY / "reader"
    rna, atac = epi.io.read_10x_multiome(d)
    g_rna, _ = epi.datasets.pbmc10k_multiome(PARITY / "data")
    cells = g_rna.obs_names.intersection(rna.obs_names)
    out = {"n_cells_10x": int(rna.n_obs), "n_cells_glue": int(g_rna.n_obs), "n_shared": int(len(cells))}

    # T: RNA identical to scGLUE on shared cells, matched by Ensembl id.
    gid = pd.Index(g_rna.var["gene_ids"].astype(str))
    tid = pd.Index(rna.var["gene_ids"].astype(str))
    genes = gid.intersection(tid)
    a = sp.csr_matrix(g_rna[cells, gid.get_indexer(genes)].X)
    b = sp.csr_matrix(rna[cells, tid.get_indexer(genes)].X)
    out["T_rna_genes_shared"] = int(len(genes))
    out["T_rna_identical_to_glue"] = bool((a != b).nnz == 0)

    # T-A1a: independent h5py decode of the same file.
    h5 = next(d.glob("*filtered_feature_bc_matrix.h5"))
    with h5py.File(h5, "r") as f:
        g = f["matrix"]
        shape = tuple(g["shape"][()])  # (features, barcodes)
        X = sp.csc_matrix((g["data"][()], g["indices"][()], g["indptr"][()]), shape=shape).T.tocsr()
        ftype = np.array([x.decode() for x in g["features"]["feature_type"][()]])
        fid = np.array([x.decode() for x in g["features"]["id"][()]])
        bcs = np.array([x.decode() for x in g["barcodes"][()]])
    pk, ge = ftype == "Peaks", ftype == "Gene Expression"
    out["TA1a_barcodes_equal"] = bool(np.array_equal(bcs, np.asarray(atac.obs_names)))
    out["TA1a_peak_names_equal"] = bool(np.array_equal(fid[pk], np.asarray(atac.var_names)))
    out["TA1a_atac_identical"] = bool((X[:, pk] != sp.csr_matrix(atac.X)).nnz == 0)
    out["TA1a_rna_identical"] = bool((X[:, ge] != sp.csr_matrix(rna.X)).nnz == 0)

    # T-A1b (information): row sums vs per_barcode_metrics.
    out["TA1b_frac_atac_rowsum_eq_peak_region_cutsites"] = float(np.mean(
        np.asarray(atac.X.sum(axis=1)).ravel() == atac.obs["atac_peak_region_cutsites"].to_numpy()))
    out["TA1b_frac_rna_rowsum_eq_gex_umis_count"] = float(np.mean(
        np.asarray(rna.X.sum(axis=1)).ravel() == rna.obs["gex_umis_count"].to_numpy()))
    out["fragments_recorded_when_absent"] = "files" in atac.uns
    out["pass"] = bool(out["T_rna_identical_to_glue"] and out["TA1a_barcodes_equal"] and out["TA1a_peak_names_equal"]
                       and out["TA1a_atac_identical"] and out["TA1a_rna_identical"])
    return {"T": out}


if __name__ == "__main__":
    sys.exit(main(sys.argv))
