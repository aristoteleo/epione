"""scBasset / PeakVI parity gates (see MANIFEST.md). Epione env (no TensorFlow).

Layout under $EPIONE_SCBASSET_PARITY_DIR:
    data/atac_ad.h5ad             shared input (reference/prep_input.py)
    data/processed/               official scbasset_preprocess.py output
    runs/official_s{0,1,2}/       official training + reference/export_official.py
    runs/epione_s{0,1,2}/         run_epione.py
    runs/keras_auc.npz            reference/keras_auc.py
$HG38_FASTA is the GRCh38 FASTA (epi.data.hg38).

    python parity_check.py g0 | g1 | g2 | p1 | p2 | info   [--json OUT]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import FLOORS, embedding_metrics, gate, knn_accuracy, per_cell_auroc  # noqa: E402

ROOT = Path(os.environ.get("EPIONE_SCBASSET_PARITY_DIR", "."))
DATA, RUNS = ROOT / "data", ROOT / "runs"


def _adata():
    import anndata as ad

    return ad.read_h5ad(DATA / "atac_ad.h5ad")


def g0():
    import h5py
    import pandas as pd
    import scipy.sparse as sp
    from epione.models.scbasset import binarize, peak_sequences, peak_table, split_train_test_val

    a = _adata()
    proc = DATA / "processed"
    res = {}
    pt = peak_table(a.var)
    bed = pd.read_csv(proc / "peaks.bed", sep="\t", header=None, names=["chr", "start", "end"])
    res["G0a_peaks_identical"] = bool(pt.reset_index(drop=True).equals(bed.astype({"chr": str})))
    codes, nmask = peak_sequences(pt, os.environ["HG38_FASTA"], return_n_mask=True)
    with h5py.File(proc / "all_seqs.h5", "r") as f:
        ref = f["X"][:]
    acgt = ~nmask
    res["G0b_n_peaks"] = int(codes.shape[0])
    res["G0b_mismatch_at_ACGT"] = int((codes[acgt] != ref[acgt]).sum())
    res["G0b_n_nonACGT_positions"] = int(nmask.sum())
    res["G0b_nonACGT_in_0_3"] = bool(((ref[nmask] >= 0) & (ref[nmask] <= 3)).all())
    res["G0b_pass"] = res["G0b_mismatch_at_ACGT"] == 0 and res["G0b_nonACGT_in_0_3"] and ref.shape == codes.shape
    Y = binarize(a.X)
    tr, te, va = split_train_test_val(a.n_vars)
    with h5py.File(proc / "splits.h5", "r") as f:
        s = {k: f[k][:] for k in ("train_ids", "test_ids", "val_ids")}
    res["G0d_split_identical"] = bool(np.array_equal(tr, s["train_ids"]) and np.array_equal(te, s["test_ids"])
                                      and np.array_equal(va, s["val_ids"]))
    ok = True
    for name, ids in (("train", tr), ("val", va), ("test", te)):
        m = sp.load_npz(proc / f"m_{name}.npz").tocsr()
        m.sort_indices()
        e = Y[ids]
        # The official m_*.npz keep the counts; its data generator sets the target to 1
        # at every stored index, so its binary targets are the sparsity pattern of m.
        ok &= bool(np.array_equal(m.indptr, e.indptr) and np.array_equal(m.indices, e.indices)
                   and np.all(m.data != 0) and np.all(e.data == 1))
    res["G0c_targets_identical"] = ok
    res["pass"] = bool(res["G0a_peaks_identical"] and res["G0b_pass"] and ok and res["G0d_split_identical"])
    return res


def g1():
    import torch
    from sklearn.metrics import roc_auc_score
    from epione.models.scbasset import KerasMultiLabelAUC, ScBasset
    from epione.models.scbasset._data import binarize  # noqa: F401
    from scipy.stats import rankdata

    res = {}
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(300, 8))
    lab = rng.integers(0, 4, 300)
    # brute-force kNN accuracy
    D = ((emb[:, None] - emb[None]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    nb = np.argsort(D, 1, kind="stable")[:, :15]
    pred = np.array([np.bincount(lab[r], minlength=4).argmax() for r in nb])
    res["G1a_knn_delta"] = abs(knn_accuracy(emb, lab.astype(str)) - float((pred == lab).mean()))
    y = (rng.random((200, 30)) < 0.2).astype(int)
    p = rng.random((200, 30)) + 0.3 * y

    def mw(yj, pj):
        r = rankdata(pj)
        n1 = yj.sum()
        n0 = len(yj) - n1
        return (r[yj == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    ref = np.mean([mw(y[:, j], p[:, j]) for j in range(30) if 0 < y[:, j].sum() < 200])
    res["G1a_auroc_delta"] = abs(per_cell_auroc(y, p) - ref)
    res["G1a_pass"] = res["G1a_knn_delta"] <= 1e-12 and res["G1a_auroc_delta"] <= 1e-12
    # G1b
    k = np.load(RUNS / "keras_auc.npz")
    m = KerasMultiLabelAUC(k["p"].shape[2])
    for pb, yb in zip(k["p"], k["y"]):
        m.update(torch.from_numpy(pb), torch.from_numpy(yb))
    res["G1b_epione"] = m.result()
    res["G1b_keras"] = float(k["auc"])
    res["G1b_delta"] = abs(res["G1b_epione"] - res["G1b_keras"])
    res["G1b_pass"] = res["G1b_delta"] <= 1e-5
    # G1c
    import anndata as ad
    import h5py

    a = _adata()
    with h5py.File(DATA / "processed" / "all_seqs.h5", "r") as f:
        codes = f["X"][:]
    sb = ScBasset(codes, a.X)
    sb.load_keras_weights(RUNS / "official_s0" / "best_model.h5")
    off = np.load(RUNS / "official_s0" / "result.npz")
    pe = sb.predict(idx=off["test_ids"], device="cpu")
    res["G1c_max_abs_dprob"] = float(np.abs(pe - off["test_prob"]).max())
    res["G1c_embedding_identical"] = bool(np.array_equal(sb.cell_embedding(), off["embedding"]))
    res["G1c_intercept_identical"] = bool(np.array_equal(sb.intercept(), off["intercept"]))
    res["G1c_pass"] = (res["G1c_max_abs_dprob"] <= 1e-4 and res["G1c_embedding_identical"]
                       and res["G1c_intercept_identical"])
    res["pass"] = bool(res["G1a_pass"] and res["G1b_pass"] and res["G1c_pass"])
    return res


def _side_metrics(prefix, seeds=(0, 1, 2), auroc=True):
    a = _adata()
    lab = a.obs["cell_type"].astype(str).to_numpy()
    Y = None
    rows = []
    for s in seeds:
        f = RUNS / f"{prefix}_s{s}" / "result.npz"
        if not f.exists():
            continue
        r = np.load(f)
        m = embedding_metrics(r["embedding"], lab)
        if auroc and "test_prob" in r:
            if Y is None:
                from epione.models.scbasset import binarize
                Y = binarize(a.X)
            m["test_AUROC"] = per_cell_auroc(Y[r["test_ids"]].toarray(), r["test_prob"].astype(np.float32))
        m["seed"] = s
        rows.append(m)
    return rows


def _compare(rows_a, rows_b, metrics):
    out = {}
    for k in metrics:
        out[k] = gate([r[k] for r in rows_a], [r[k] for r in rows_b], FLOORS[k])
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def g2():
    ep, of = _side_metrics("epione"), _side_metrics("official")
    res = {"epione": ep, "official": of}
    res.update(_compare(ep, of, ["ARI", "AMI", "kNN_acc", "silhouette", "test_AUROC"]))
    return res


def p1():
    import anndata as ad
    import scvi
    import epione as epi

    a = _adata()
    m = epi.tl.peakvi(a, seed=0, max_epochs=10, accelerator="cpu")
    z_e = a.obsm["X_peakvi"]
    b = _adata()
    scvi.settings.seed = 0
    scvi.model.PEAKVI.setup_anndata(b)
    d = scvi.model.PEAKVI(b)
    d.train(max_epochs=10, accelerator="cpu")
    z_d = d.get_latent_representation()
    dmax = float(np.abs(z_e - z_d).max())
    return {"P1_max_abs_dlatent": dmax, "pass": dmax == 0.0}


def p2():
    ep, di = _side_metrics("peakvi_epione", auroc=False), _side_metrics("peakvi_direct", auroc=False)
    res = {"epione": ep, "direct": di}
    dl = []
    for s in (0, 1, 2):
        fe, fd = RUNS / f"peakvi_epione_s{s}" / "result.npz", RUNS / f"peakvi_direct_s{s}" / "result.npz"
        if fe.exists() and fd.exists():
            dl.append(float(np.abs(np.load(fe)["embedding"] - np.load(fd)["embedding"]).max()))
    res["max_abs_dlatent_per_seed"] = dl
    res.update(_compare(ep, di, ["ARI", "AMI", "kNN_acc", "silhouette"]))
    return res


def info():
    sv, of = _side_metrics("scvi"), _side_metrics("official")
    res = {"scvi": sv, "official": of}
    if sv:
        res.update(_compare(sv, of, ["ARI", "AMI", "kNN_acc", "silhouette", "test_AUROC"]))
    return res


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["g0", "g1", "g2", "p1", "p2", "info"])
    ap.add_argument("--json")
    args = ap.parse_args()
    out = globals()[args.gate]()
    txt = json.dumps(out, indent=1, default=_default)
    print(txt)
    if args.json:
        Path(args.json).write_text(txt)
