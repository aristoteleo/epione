"""Evaluate the gates of MANIFEST.md from the outputs of run_side.py / run_rapids.py.

    EPIONE_GPU_PARITY_DIR=... python parity_check.py [--json gates.json]

Reads ``<dir>/cpu``, ``<dir>/cuda``, ``<dir>/rapids`` and ``<dir>/main`` (G0);
gates whose inputs are missing are reported as "not run". Thresholds are the
pre-registered ones and are not parameters.
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.sparse as sp

SEEDS = [0, 1, 2, 3, 4]


def abs_cos(A, B):
    return np.abs((A * B).sum(0) / (np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0)))


def principal_cos_min(A, B):
    """Cosine of the largest principal angle between col(A) and col(B)."""
    Qa, _ = np.linalg.qr(A.astype(np.float64))
    Qb, _ = np.linalg.qr(B.astype(np.float64))
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(s.min())


def knn_jaccard(A: sp.csr_matrix, B: sp.csr_matrix):
    A, B = A.tocsr(), B.tocsr()
    jac = np.empty(A.shape[0])
    dmax = 0.0
    for i in range(A.shape[0]):
        a = dict(zip(A.indices[A.indptr[i]:A.indptr[i + 1]], A.data[A.indptr[i]:A.indptr[i + 1]]))
        b = dict(zip(B.indices[B.indptr[i]:B.indptr[i + 1]], B.data[B.indptr[i]:B.indptr[i + 1]]))
        inter = a.keys() & b.keys()
        jac[i] = len(inter) / max(len(a.keys() | b.keys()), 1)
        if inter:
            dmax = max(dmax, max(abs(float(a[k]) - float(b[k])) for k in inter))
    return float(jac.mean()), float(jac.min()), dmax


def knn_overlap(E1, E2, k=30):
    """Mean per-cell overlap of the k-NN (cosine) sets computed in two embeddings."""
    from sklearn.neighbors import NearestNeighbors
    i1 = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(E1).kneighbors(E1)[1][:, 1:]
    i2 = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(E2).kneighbors(E2)[1][:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(i1, i2)]))


def per_motif_pearson(A, B):
    """Pearson r across cells per column, ignoring NaN cells (same mask)."""
    rs = []
    for j in range(A.shape[1]):
        m = np.isfinite(A[:, j]) & np.isfinite(B[:, j])
        if m.sum() < 3 or np.std(A[m, j]) == 0:
            continue
        rs.append(np.corrcoef(A[m, j], B[m, j])[0, 1])
    return float(np.min(rs)), len(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    root = Path(os.environ["EPIONE_GPU_PARITY_DIR"])
    C, G, R, M = root / "cpu", root / "cuda", root / "rapids", root / "main"
    rows = []

    def gate(gid, what, value, thr, ok, note=""):
        rows.append(dict(gate=gid, output=what, value=value, threshold=thr,
                         status=("PASS" if ok else "FAIL") if ok is not None else "not run",
                         note=note))
        print(f"{gid:5s} {('PASS' if ok else 'FAIL') if ok is not None else 'not run':8s} {what}: {value}  [{thr}] {note}")

    def have(*paths):
        return all(p.exists() for p in paths)

    # ---- G0: CPU unchanged ----
    if M.exists():
        diffs = []
        for f in sorted(M.glob("*")):
            if f.suffix not in (".npy", ".npz", ".parquet") or not (C / f.name).exists():
                continue
            if f.suffix == ".npy":
                a, b = np.load(f, allow_pickle=True), np.load(C / f.name, allow_pickle=True)
                same = np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
            elif f.suffix == ".parquet":
                import pandas as pd
                same = pd.read_parquet(f).equals(pd.read_parquet(C / f.name))
            elif f.name.endswith(("_dist.npz", "_conn.npz", "lsi_values.npz", "motif_scan.npz")):
                a, b = sp.load_npz(f), sp.load_npz(C / f.name)
                same = a.shape == b.shape and (a != b).nnz == 0
            else:
                a, b = np.load(f, allow_pickle=True), np.load(C / f.name, allow_pickle=True)
                same = all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in a.files)
            if not same:
                diffs.append(f.name)
        n = len([f for f in M.glob("*") if f.suffix in (".npy", ".npz", ".parquet") and (C / f.name).exists()])
        gate("G0", "device='cpu' outputs main vs branch", f"{n - len(diffs)}/{n} files bit-identical",
             "bit-identical", len(diffs) == 0, ("differ: " + ", ".join(diffs)) if diffs else "")
    else:
        gate("G0", "device='cpu' outputs main vs branch", "-", "bit-identical", None)

    # ---- G1 ----
    if have(C / "lsi_values.npz", G / "lsi_values.npz"):
        A, B = sp.load_npz(C / "lsi_values.npz"), sp.load_npz(G / "lsi_values.npz")
        same_pat = np.array_equal(A.indptr, B.indptr) and np.array_equal(A.indices, B.indices)
        rel = float(np.max(np.abs(A.data - B.data) / np.abs(A.data))) if same_pat else np.inf
        gate("G1", "lsi TF-IDF/L1/log1p values", f"max rel err {rel:.2e}", "<= 1e-5", same_pat and rel <= 1e-5)
    else:
        gate("G1", "lsi TF-IDF/L1/log1p values", "-", "<= 1e-5", None)

    # ---- G2 ----
    for name, label in (("lsi.npy", "D1"), ("d3_lsi.npy", "D3")):
        if have(C / name, G / name):
            A, B = np.load(C / name), np.load(G / name)
            c, p = float(abs_cos(A, B).min()), principal_cos_min(A, B)
            gate("G2", f"lsi X_lsi ({label})", f"min |cos| {c:.6f}; principal cos {p:.6f}",
                 ">= 0.999 both", c >= 0.999 and p >= 0.999)
        else:
            gate("G2", f"lsi X_lsi ({label})", "-", ">= 0.999 both", None)

    # ---- G3 ----
    if have(C / "ilsi1.npz", G / "ilsi1.npz"):
        a, b = np.load(C / "ilsi1.npz"), np.load(G / "ilsi1.npz")
        same_k = np.array_equal(a["kept"], b["kept"])
        c = float(abs_cos(a["emb"], b["emb"]).min()) if same_k else 0.0
        p = principal_cos_min(a["emb"], b["emb"]) if same_k else 0.0
        gate("G3", "iterative_lsi(iterations=1) (D2)",
             f"min |cos| {c:.6f}; principal cos {p:.6f}; kept dims identical={same_k}",
             ">= 0.999; identical", same_k and c >= 0.999 and p >= 0.999)
    else:
        gate("G3", "iterative_lsi(iterations=1) (D2)", "-", ">= 0.999", None)

    # ---- G4 ----
    if all(have(C / f"ilsi2_s{s}.npz", G / f"ilsi2_s{s}.npz") for s in SEEDS):
        ca = {s: np.load(C / f"ilsi2_s{s}.npz") for s in SEEDS}
        ga = {s: np.load(G / f"ilsi2_s{s}.npz") for s in SEEDS}

        def jac(f1, f2):
            f1, f2 = f1.astype(bool), f2.astype(bool)
            return (f1 & f2).sum() / (f1 | f2).sum()
        jc = np.median([jac(ca[i]["feat"], ca[j]["feat"]) for i, j in combinations(SEEDS, 2)])
        jg = np.median([jac(ga[s]["feat"], ca[s]["feat"]) for s in SEEDS])
        kc = np.median([knn_overlap(ca[i]["emb"], ca[j]["emb"]) for i, j in combinations(SEEDS, 2)])
        kg = np.median([knn_overlap(ga[s]["emb"], ca[s]["emb"]) for s in SEEDS])
        gate("G4a", "iterative_lsi(2) variable-feature Jaccard (D2)",
             f"GPU-vs-CPU {jg:.4f}; CPU-vs-CPU {jc:.4f}", ">= CPU-vs-CPU - 0.02", jg >= jc - 0.02)
        gate("G4b", "iterative_lsi(2) 30-NN overlap (D2)",
             f"GPU-vs-CPU {kg:.4f}; CPU-vs-CPU {kc:.4f}", ">= CPU-vs-CPU - 0.02", kg >= kc - 0.02)
    else:
        gate("G4", "iterative_lsi(2) (D2)", "-", "vs CPU seeds", None)

    # ---- G5 ----
    for metric in ("euclidean", "cosine"):
        f = f"neighbors_{metric}_dist.npz"
        if have(C / f, G / f):
            mj, mn, dmax = knn_jaccard(sp.load_npz(C / f), sp.load_npz(G / f))
            gate("G5", f"pp.neighbors {metric} (D1)",
                 f"mean Jaccard {mj:.5f} (min {mn:.3f}); max |d diff| {dmax:.2e}",
                 ">= 0.99; <= 1e-4", mj >= 0.99 and dmax <= 1e-4)
            fc = f"neighbors_{metric}_conn.npz"
            if have(C / fc, G / fc):
                d = abs(sp.load_npz(C / fc) - sp.load_npz(G / fc))
                rows_diff = int((np.asarray(d.max(axis=1).todense()).ravel() > 1e-3).sum())
                print(f"      (info) connectivities: max |diff| {d.max():.3g}; rows with diff > 1e-3: {rows_diff}")
        else:
            gate("G5", f"pp.neighbors {metric} (D1)", "-", ">= 0.99", None)
    if have(C / "d3_neighbors_dist.npz", G / "d3_neighbors_dist.npz"):
        mj, mn, dmax = knn_jaccard(sp.load_npz(C / "d3_neighbors_dist.npz"),
                                   sp.load_npz(G / "d3_neighbors_dist.npz"))
        gate("G5", "pp.neighbors euclidean (D3)",
             f"mean Jaccard {mj:.5f} (min {mn:.3f}); max |d diff| {dmax:.2e}",
             ">= 0.99; <= 1e-4", mj >= 0.99 and dmax <= 1e-4)

    # ---- G6 / G7 ----
    for gid, meth in (("G6", "analytical"), ("G7", "sample")):
        f = f"chromvar_{meth}.npz"
        if have(C / f, G / f):
            a, b = np.load(C / f), np.load(G / f)
            rz, n = per_motif_pearson(a["z"], b["z"])
            rd, _ = per_motif_pearson(a["dev"], b["dev"])
            nan_same = np.array_equal(np.isnan(a["z"]), np.isnan(b["z"]))
            mz = float(np.nanmax(np.abs(a["z"] - b["z"])))
            gate(gid, f"chromVAR {meth} (D1, {n} motifs)",
                 f"min r(Z) {rz:.7f}; min r(dev) {rd:.7f}; max |dZ| {mz:.2e}; NaN mask same={nan_same}",
                 "r >= 0.9999", rz >= 0.9999 and rd >= 0.9999)
        else:
            gate(gid, f"chromVAR {meth} (D1)", "-", "r >= 0.9999", None)

    # ---- G8 ----
    if have(C / "motif_scan.npz", G / "motif_scan.npz"):
        A = sp.load_npz(C / "motif_scan.npz").astype(bool)
        B = sp.load_npz(G / "motif_scan.npz").astype(bool)
        n_diff = int((A != B).nnz)
        gate("G8", "add_motif_matrix hits (D1, 232,272 x 746)",
             f"{A.nnz:,} CPU hits, {B.nnz:,} GPU hits, {n_diff} differing cells",
             "identical", n_diff == 0 and A.shape == B.shape)
    else:
        gate("G8", "add_motif_matrix hits (D1)", "-", "identical", None)

    # ---- G9 ----
    if have(C / "peak_to_gene.parquet", G / "peak_to_gene.parquet"):
        import pandas as pd
        a, b = pd.read_parquet(C / "peak_to_gene.parquet"), pd.read_parquet(G / "peak_to_gene.parquet")
        same = (len(a) == len(b) and (a["peak"].values == b["peak"].values).all()
                and (a["gene"].values == b["gene"].values).all())
        dr = float(np.max(np.abs(a["correlation"].values - b["correlation"].values))) if same else np.inf
        sa = set(map(tuple, a.loc[a.fdr < 0.05, ["peak", "gene"]].values))
        sb = set(map(tuple, b.loc[b.fdr < 0.05, ["peak", "gene"]].values))
        j = len(sa & sb) / max(len(sa | sb), 1)
        gate("G9", f"peak_to_gene ({len(a):,} pairs, {len(sa):,} at FDR<0.05)",
             f"pairs identical={same}; max |dr| {dr:.2e}; FDR Jaccard {j:.5f}",
             "identical; <= 1e-4; >= 0.999", same and dr <= 1e-4 and j >= 0.999)
    else:
        gate("G9", "peak_to_gene", "-", "", None)

    # ---- G10-G12 (rapids) ----
    if have(R / "neighbors_euclidean_dist.npz", C / "neighbors_euclidean_dist.npz"):
        for metric in ("euclidean", "cosine"):
            f = f"neighbors_{metric}_dist.npz"
            if have(R / f):
                mj, mn, dmax = knn_jaccard(sp.load_npz(C / f), sp.load_npz(R / f))
                gate("G10", f"rapids neighbors {metric} (D1)",
                     f"mean Jaccard {mj:.5f} (min {mn:.3f}); max |d diff| {dmax:.2e}",
                     ">= 0.99", mj >= 0.99)
    else:
        gate("G10", "rapids neighbors (D1)", "-", ">= 0.99", None)
    if all(have(C / f"leiden_s{s}.npy", R / f"leiden_s{s}.npy") for s in SEEDS):
        from sklearn.metrics import adjusted_rand_score as ari
        lc = {s: np.load(C / f"leiden_s{s}.npy", allow_pickle=True) for s in SEEDS}
        lr = {s: np.load(R / f"leiden_s{s}.npy", allow_pickle=True) for s in SEEDS}
        cc = np.median([ari(lc[i], lc[j]) for i, j in combinations(SEEDS, 2)])
        gc = np.median([ari(lr[s], lc[s]) for s in SEEDS])
        nk = (np.median([len(set(lc[s])) for s in SEEDS]), np.median([len(set(lr[s])) for s in SEEDS]))
        gate("G11", "rapids Leiden ARI (D1)", f"GPU-vs-CPU {gc:.4f}; CPU-vs-CPU {cc:.4f}; clusters CPU {nk[0]:.0f} / rapids {nk[1]:.0f}",
             ">= CPU-vs-CPU - 0.05", gc >= cc - 0.05)
    else:
        gate("G11", "rapids Leiden (D1)", "-", "", None)
    if all(have(C / f"umap_s{s}.npy", R / f"umap_s{s}.npy") for s in SEEDS):
        import anndata as ad
        from sklearn.manifold import trustworthiness
        X = ad.read_h5ad(Path(os.environ["EPIONE_ARCHR_HEME"]) / "heme_peaks.h5ad", backed="r").obsm["X_iterative_lsi"][:]
        tc = [trustworthiness(X, np.load(C / f"umap_s{s}.npy"), n_neighbors=15) for s in SEEDS]
        tr = [trustworthiness(X, np.load(R / f"umap_s{s}.npy"), n_neighbors=15) for s in SEEDS]
        d = float(np.median(tr) - np.median(tc))
        gate("G12", "rapids UMAP trustworthiness (D1)",
             f"rapids {np.median(tr):.4f} - CPU {np.median(tc):.4f} = {d:+.4f}", ">= -0.02", d >= -0.02)
    else:
        gate("G12", "rapids UMAP (D1)", "-", "", None)

    # ---- G13 timings ----
    tim = {}
    for side, d in (("cpu", C), ("cuda", G), ("rapids", R)):
        if (d / "timings.json").exists():
            tim[side] = json.loads((d / "timings.json").read_text())
    if tim:
        keys = sorted(set().union(*[set(v) for v in tim.values()]))
        print("\nG13 timings (s):")
        for k in keys:
            c = tim.get("cpu", {}).get(k)
            g = tim.get("cuda", tim.get("rapids", {})).get(k) if k in tim.get("cuda", {}) else tim.get("rapids", {}).get(k)
            sp_ = f"{c / g:.1f}x" if c and g else ""
            print(f"  {k:24s} cpu {c if c is None else round(c, 2)!s:>8}  gpu {g if g is None else round(g, 2)!s:>8}  {sp_}")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(gates=rows, timings=tim), indent=1, default=float))


if __name__ == "__main__":
    main()
