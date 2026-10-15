"""Fragment-level parity gates G2-G6 / G8 of MANIFEST.md.

usage: python parity_fragments.py REF_FRAGMENTS REF_CELLS TEST_FRAGMENTS TSS_BED OUT_JSON
           [--g4 0.90] [--skip-cluster]

REF_CELLS is a text file with one barcode per line (the reference's called
cells). All comparisons use chr1-22, X, Y only. Definitions are fixed in
MANIFEST.md; this script only implements them.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

NUCLEAR = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
CHROM_ID = {c: i for i, c in enumerate(NUCLEAR)}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, comment="#", usecols=[0, 1, 2, 3],
                     names=["chrom", "start", "end", "barcode"],
                     dtype={"chrom": "category", "start": np.int64, "end": np.int64, "barcode": "category"})
    df = df[df["chrom"].isin(NUCLEAR)].copy()
    # GEM-well suffix ("-1") is not part of the barcode
    cats = df["barcode"].cat.categories.astype(str).str.replace(r"-\d+$", "", regex=True)
    if cats.is_unique:
        df["barcode"] = df["barcode"].cat.rename_categories(cats)
    else:
        df["barcode"] = df["barcode"].astype(str).str.replace(r"-\d+$", "", regex=True).astype("category")
    df["cid"] = df["chrom"].map(CHROM_ID).astype(np.int64)
    # a fragment file may hold one barcode twice at a position only if
    # it was not de-duplicated; collapse to unique fragments
    df = df.drop_duplicates(["cid", "start", "end", "barcode"])
    return df[["cid", "start", "end", "barcode"]].reset_index(drop=True)


def per_barcode_counts(df: pd.DataFrame) -> pd.Series:
    return df["barcode"].astype(str).value_counts()


def frag_keys(df: pd.DataFrame) -> np.ndarray:
    return (df["cid"].to_numpy() << 50) | (df["start"].to_numpy() << 21) | (df["end"].to_numpy() - df["start"].to_numpy())


def per_cell_jaccard(ref: pd.DataFrame, test: pd.DataFrame, cells) -> np.ndarray:
    rk, tk = frag_keys(ref), frag_keys(test)
    rg = pd.Series(np.arange(len(ref))).groupby(ref["barcode"].astype(str).to_numpy()).indices
    tg = pd.Series(np.arange(len(test))).groupby(test["barcode"].astype(str).to_numpy()).indices
    out = []
    for c in cells:
        a = np.unique(rk[rg[c]]) if c in rg else np.zeros(0, np.int64)
        b = np.unique(tk[tg[c]]) if c in tg else np.zeros(0, np.int64)
        u = np.union1d(a, b).size
        out.append(np.intersect1d(a, b, assume_unique=True).size / u if u else np.nan)
    return np.asarray(out)


def load_tss(path: str):
    t = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1], names=["chrom", "pos"])
    t = t[t["chrom"].isin(NUCLEAR)]
    return {c: np.unique(g["pos"].to_numpy()) for c, g in t.groupby("chrom")}


def tss_scores(df: pd.DataFrame, cells, tss) -> np.ndarray:
    cells = list(cells)
    cidx = {c: i for i, c in enumerate(cells)}
    bc = df["barcode"].astype(str).map(cidx)
    keep = bc.notna().to_numpy()
    sub = df[keep]
    bi = bc[keep].to_numpy().astype(np.int64)
    center = np.zeros(len(cells)); flank = np.zeros(len(cells))
    for cid, chrom in enumerate(NUCLEAR):
        m = sub["cid"].to_numpy() == cid
        if not m.any() or chrom not in tss:
            continue
        T = tss[chrom]
        for pos in (sub["start"].to_numpy()[m], sub["end"].to_numpy()[m] - 1):
            j = np.searchsorted(T, pos)
            dl = np.abs(pos - T[np.clip(j - 1, 0, T.size - 1)])
            dr = np.abs(T[np.clip(j, 0, T.size - 1)] - pos)
            d = np.minimum(dl, dr)
            b = bi[m]
            np.add.at(center, b[d <= 50], 1)
            np.add.at(flank, b[(d >= 1901) & (d <= 2000)], 1)
    return (center / 101.0) / ((flank + 1.0) / 200.0)


def tile_matrix(df: pd.DataFrame, cells, tile=5000):
    import scipy.sparse as sp

    cells = list(cells)
    cidx = {c: i for i, c in enumerate(cells)}
    bc = df["barcode"].astype(str).map(cidx)
    keep = bc.notna().to_numpy()
    sub = df[keep]
    bi = bc[keep].to_numpy().astype(np.int64)
    off = np.arange(len(NUCLEAR), dtype=np.int64) * (300_000_000 // tile)
    rows, cols = [], []
    for pos in (sub["start"].to_numpy(), sub["end"].to_numpy() - 1):
        rows.append(bi)
        cols.append(off[sub["cid"].to_numpy()] + pos // tile)
    r, c = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(r.size, np.float32), (r, c)), shape=(len(cells), int(off[-1] + 300_000_000 // tile)))


def cluster(X, features):
    import anndata as ad
    import scanpy as sc
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    X = X[:, features].tocsr().astype(np.float64)
    rs = np.asarray(X.sum(1)).ravel(); rs[rs == 0] = 1
    cs = np.asarray(X.sum(0)).ravel()
    idf = X.shape[0] / (1.0 + cs)
    tf = X.multiply(1.0 / rs[:, None])
    T = tf.multiply(idf[None, :]).tocsr()
    T.data = np.log1p(T.data * 1e4)
    Z = TruncatedSVD(30, random_state=0).fit_transform(T)[:, 1:]
    Z = normalize(Z)
    a = ad.AnnData(np.zeros((Z.shape[0], 1), np.float32))
    a.obsm["X_lsi"] = Z
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X_lsi", random_state=0)
    sc.tl.leiden(a, resolution=1.0, random_state=0, flavor="igraph", n_iterations=2, directed=False)
    return a.obs["leiden"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("cells"); ap.add_argument("test"); ap.add_argument("tss"); ap.add_argument("out")
    ap.add_argument("--skip-cluster", action="store_true")
    ap.add_argument("--k-cells", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    cells = [l.strip().split("-")[0] for l in open(a.cells) if l.strip()]
    ref, test = load(a.ref), load(a.test)
    rc, tc = per_barcode_counts(ref), per_barcode_counts(test)
    K = a.k_cells or len(cells)
    top = set(tc.sort_values(ascending=False, kind="stable").index[:K])
    res = {"n_cells": len(cells), "n_ref_fragments": int(len(ref)), "n_test_fragments": int(len(test))}
    res["G2_jaccard_called_cells"] = len(top & set(cells)) / len(top | set(cells))
    # control (not a gate): the same top-K rule applied to the reference's own file
    rtop = set(rc.sort_values(ascending=False, kind="stable").index[:K])
    res["G2_control_ref_topK_vs_ref_cells"] = len(rtop & set(cells)) / len(rtop | set(cells))
    res["G2_topK_test_vs_topK_ref"] = len(top & rtop) / len(top | rtop)
    x = rc.reindex(cells).fillna(0).to_numpy(); y = tc.reindex(cells).fillna(0).to_numpy()
    res["G3_pearson_fragments_per_cell"] = float(np.corrcoef(x, y)[0, 1])
    res["median_ratio_test_over_ref"] = float(np.median(y / np.maximum(x, 1)))
    jac = per_cell_jaccard(ref, test, cells)
    res["G4_median_cell_jaccard"] = float(np.nanmedian(jac))
    res["G4_q10_cell_jaccard"] = float(np.nanquantile(jac, 0.1))
    tss = load_tss(a.tss)
    s_ref, s_test = tss_scores(ref, cells, tss), tss_scores(test, cells, tss)
    res["G5_pearson_tsse"] = float(np.corrcoef(s_ref, s_test)[0, 1])
    res["median_tsse_ref"], res["median_tsse_test"] = float(np.median(s_ref)), float(np.median(s_test))
    if not a.skip_cluster:
        from sklearn.metrics import adjusted_rand_score

        Xr, Xt = tile_matrix(ref, cells), tile_matrix(test, cells)
        tot = np.asarray(Xr.sum(0)).ravel()
        feats = np.sort(np.argsort(-tot, kind="stable")[:50_000])
        lr, lt = cluster(Xr, feats), cluster(Xt, feats)
        res["G6_ari_leiden"] = float(adjusted_rand_score(lr, lt))
        res["n_clusters_ref"], res["n_clusters_test"] = int(len(set(lr))), int(len(set(lt)))
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
