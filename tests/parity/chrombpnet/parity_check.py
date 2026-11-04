"""ChromBPNet parity driver (Epione side). See MANIFEST.md.

The reference (official TensorFlow ChromBPNet) runs in a separate env outside
the repository and writes its outputs under ``$EPIONE_CHROMBPNET_PARITY_DIR``;
this script only reads those files. It never imports TensorFlow.

    python parity_check.py g1-inputs   # write the 512 fixed G1 sequences
    python parity_check.py g1-compare  # Epione vs reference predictions
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

PAR = Path(os.environ.get("EPIONE_CHROMBPNET_PARITY_DIR", "."))
G1_MODELS = ["bias_scaled", "chrombpnet_nobias", "chrombpnet"]


def g1_inputs(fasta: str):
    import pyfaidx

    from epione.models.chrombpnet import one_hot_encode
    fa = pyfaidx.Fasta(fasta)
    rng = np.random.RandomState(0)
    L = 2114
    seqs = []
    chroms = ["chr1", "chr3", "chr6"]
    lens = np.array([len(fa[c]) for c in chroms], dtype=float)
    while len(seqs) < 256:
        c = rng.choice(len(chroms), p=lens / lens.sum())
        mid = rng.randint(L, int(lens[c]) - L)
        s = str(fa[chroms[c]][mid - L // 2: mid + L // 2])
        seqs.append(s)
    rng1 = np.random.RandomState(1)
    seqs += ["".join(rng1.choice(list("ACGT"), L)) for _ in range(256)]
    X = one_hot_encode(seqs).astype(np.float32)
    (PAR / "g1").mkdir(parents=True, exist_ok=True)
    np.save(PAR / "g1" / "inputs.npy", X)
    print("wrote", X.shape)


def _jsd(p, q):
    """Jensen-Shannon distance in float64. For near-identical distributions
    the divergence can round to a tiny negative number, whose square root
    scipy returns as NaN; it is clipped at 0, as the definition requires."""
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = lambda a, b: np.where(a > 0, a * np.log(a / b), 0.0).sum(-1)
        js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return np.sqrt(np.clip(js, 0, None))


def g1_compare(model_dir: str, experiment: str, fold: int = 0):
    import torch

    from epione.models.chrombpnet import load_keras_h5, softmax
    from epione.models.chrombpnet._model import predict
    torch.set_num_threads(8)
    X = np.load(PAR / "g1" / "inputs.npy")
    res = {}
    for kind in G1_MODELS:
        h5 = Path(model_dir) / f"fold_{fold}" / f"model.{kind}.fold_{fold}.{experiment}.h5"
        ref = np.load(PAR / "g1" / f"ref_{kind}.npz")
        m = load_keras_h5(h5)
        prof, cnt = predict(m, np.transpose(X, (0, 2, 1)), device="cpu")
        dp = np.abs(prof - ref["profile"]).max()
        dc = np.abs(cnt - ref["counts"]).max()
        jsd = float(_jsd(softmax(prof.astype(np.float64)), softmax(ref["profile"].astype(np.float64))).mean())
        ok = dp <= 1e-3 and dc <= 1e-3 and jsd <= 1e-5
        res[kind] = {"max_abs_profile_logits": float(dp), "max_abs_logcounts": float(dc),
                     "mean_jsd": jsd, "pass": bool(ok)}
        print(kind, res[kind])
    (PAR / "g1" / "results.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
# G2: data recipe
# ---------------------------------------------------------------------------

def _g3_paths():
    d = PAR.parent / "g3" / "data"
    return {"data": d, "fasta": os.environ.get("EPIONE_HG38_FASTA", ""),
            "fragments": os.environ.get("EPIONE_PBMC3K_FRAGMENTS", ""),
            "bigwig": d / "data_unstranded.bw", "peaks": d / "peaks.narrowPeak",
            "nonpeaks": d / "nonpeaks_negatives.bed", "fold": d / "fold.json",
            "sizes": d / "hg38.chrom.sizes"}


def _sizes(path):
    import pandas as pd
    t = pd.read_csv(path, sep="\t", header=None)
    return dict(zip(t[0], t[1]))


def g2_insertions(n: int = 2000):
    from epione.models.chrombpnet import BigWigSignal, InsertionSignal
    p = _g3_paths()
    sizes = _sizes(p["sizes"])
    bw = BigWigSignal(p["bigwig"])
    ins = InsertionSignal(p["fragments"], sizes, shift=(0, 0))
    rng = np.random.RandomState(0)
    chroms = list(sizes)
    w = np.array([sizes[c] for c in chroms], float)
    bad, tot = 0, 0
    for _ in range(n):
        c = chroms[rng.choice(len(chroms), p=w / w.sum())]
        a = rng.randint(0, sizes[c] - 1000)
        x, y = ins.values(c, a, a + 1000), bw.values(c, a, a + 1000)
        tot += y.sum()
        bad += int(not np.array_equal(x, y))
    res = {"windows": n, "mismatching_windows": bad, "insertions_compared": float(tot),
           "pass": bad == 0}
    print("G2 insertions", res)
    return res


def g2_bias_hp(seed: int = 1234):
    import json as _json

    import pandas as pd

    from epione.models.chrombpnet import BigWigSignal, bias_hyperparams, read_narrowpeak
    p = _g3_paths()
    ref = PAR.parent / "g3" / f"ref_seed{seed}"
    fold = _json.load(open(p["fold"]))
    hp = bias_hyperparams(p["peaks"], p["nonpeaks"], BigWigSignal(p["bigwig"]), fold)
    ref_non = read_narrowpeak(ref / "filtered.bias_nonpeaks.bed")
    ref_peaks = read_narrowpeak(ref / "filtered.bias_peaks.bed")
    params = dict(l.split("\t") for l in open(ref / "bias_model_params.tsv").read().strip().split("\n"))
    data = dict(l.split("\t") for l in open(ref / "bias_data_params.tsv").read().strip().split("\n"))
    cols = ["chr", "start", "end", "summit"]
    same_non = hp["nonpeaks"][cols].reset_index(drop=True).equals(ref_non[cols].reset_index(drop=True))
    same_peaks = hp["peaks"][cols].reset_index(drop=True).equals(ref_peaks[cols].reset_index(drop=True))
    res = {
        "counts_loss_weight": [hp["params"]["counts_loss_weight"], float(params["counts_loss_weight"])],
        "counts_sum_min_thresh": [round(hp["counts_sum_min_thresh"], 2), float(data["counts_sum_min_thresh"])],
        "counts_sum_max_thresh": [round(hp["counts_sum_max_thresh"], 2), float(data["counts_sum_max_thresh"])],
        "nonpeaks_identical": bool(same_non), "peaks_identical": bool(same_peaks),
        "n_nonpeaks": [len(hp["nonpeaks"]), len(ref_non)],
    }
    res["pass"] = bool(same_non and same_peaks and all(a == b for a, b in
                       (res["counts_loss_weight"], res["counts_sum_min_thresh"], res["counts_sum_max_thresh"])))
    print("G2 bias hyperparameters", res)
    return res


def g2_chrombpnet_hp(seed: int = 1234):
    import json as _json

    import h5py

    from epione.models.chrombpnet import (BigWigSignal, adjust_bias_logcounts,
                                          chrombpnet_hyperparams, load_keras_h5,
                                          load_regions, read_narrowpeak)
    p = _g3_paths()
    ref = PAR.parent / "g3" / f"ref_seed{seed}"
    fold = _json.load(open(p["fold"]))
    sig = BigWigSignal(p["bigwig"])
    hp = chrombpnet_hyperparams(p["peaks"], p["nonpeaks"], sig, fold,
                                rng=np.random.RandomState(seed))
    cols = ["chr", "start", "end", "summit"]
    rp = read_narrowpeak(ref / "filtered.peaks.bed")
    rn = read_narrowpeak(ref / "filtered.nonpeaks.bed")
    params = dict(l.split("\t") for l in open(ref / "chrombpnet_model_params.tsv").read().strip().split("\n"))
    same_p = hp["peaks"][cols].reset_index(drop=True).equals(rp[cols].reset_index(drop=True))
    same_n = hp["nonpeaks"][cols].reset_index(drop=True).equals(rn[cols].reset_index(drop=True))
    # bias rescaling on the official (unscaled) bias weights
    bias = load_keras_h5(ref / "bias.h5")
    seqs, _ = load_regions(hp["bias_adjust_regions"], p["fasta"], sig, 2114, 1000, 0)
    delta = adjust_bias_logcounts(bias, seqs, hp["bias_adjust_counts"], device="cpu")
    scaled = load_keras_h5(ref / "bias_model_scaled.h5")
    db = float(abs(bias.count_dense.bias.item() - scaled.count_dense.bias.item()))
    res = {"peaks_identical": bool(same_p), "nonpeaks_identical": bool(same_n),
           "counts_loss_weight": [hp["params"]["counts_loss_weight"], float(params["counts_loss_weight"])],
           "bias_count_head_bias_abs_diff": db, "delta": delta}
    res["pass"] = bool(same_p and same_n and res["counts_loss_weight"][0] == res["counts_loss_weight"][1]
                       and db <= 1e-5)
    print("G2 chrombpnet hyperparameters", res)
    return res


# ---------------------------------------------------------------------------
# G3: training from scratch
# ---------------------------------------------------------------------------

CTCF_MA0139_1 = "TGGCCACCAGGGGGCGCTA"  # JASPAR MA0139.1 consensus


def g3_train(seed: int, device: str = "cuda"):
    import json as _json

    import torch

    from epione.models.chrombpnet import BigWigSignal, train_bias, train_chrombpnet
    p = _g3_paths()
    out = PAR.parent / "g3" / f"epione_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    fold = _json.load(open(p["fold"]))
    sig = BigWigSignal(p["bigwig"])
    b = train_bias(p["peaks"], p["nonpeaks"], p["fasta"], sig, fold, seed=seed, device=device)
    torch.save(b.model.state_dict(), out / "bias.pt")
    c = train_chrombpnet(p["peaks"], p["nonpeaks"], p["fasta"], sig, fold, b.model, seed=seed,
                         device=device, hyperparam_rng=np.random.RandomState(seed))
    torch.save(c.model.state_dict(), out / "chrombpnet.pt")
    for name, r in (("bias", b), ("chrombpnet", c)):
        (out / f"{name}.json").write_text(_json.dumps(
            {"params": r.params, "history": r.history, "best_epoch": r.best_epoch,
             "seconds": r.seconds}, indent=1, default=float))
    print("DONE", seed, b.seconds, c.seconds)


def _test_regions(path, fold, sizes, width=2114):
    from epione.models.chrombpnet import filter_edge_regions, read_narrowpeak
    df = read_narrowpeak(path)
    return filter_edge_regions(df[df["chr"].isin(fold["test"])], sizes, width)


def _load_side(side: str, seed: int):
    import torch

    from epione.models.chrombpnet import BPNet, ChromBPNet, build_bpnet, load_keras_h5
    g3 = PAR.parent / "g3"
    if side == "ref":
        d = g3 / f"ref_seed{seed}"
        full = load_keras_h5(d / "chrombpnet.h5")
        return full, load_keras_h5(d / "chrombpnet_nobias.h5"), load_keras_h5(d / "bias_model_scaled.h5")
    d = g3 / f"epione_seed{seed}"
    full = ChromBPNet(build_bpnet("bias"), build_bpnet("chrombpnet"))
    full.load_state_dict(torch.load(d / "chrombpnet.pt", map_location="cpu"))
    return full.eval(), full.accessibility.eval(), full.bias.eval()


def g3_eval(seeds=(1234, 1235, 1236), device: str = "cuda"):
    import json as _json

    from epione.models.chrombpnet import (BigWigSignal, dinuc_shuffle, evaluate, get_seq_onehot,
                                          marginal_footprint)
    p = _g3_paths()
    fold = _json.load(open(p["fold"]))
    sizes = _sizes(p["sizes"])
    sig = BigWigSignal(p["bigwig"])
    tp = _test_regions(p["peaks"], fold, sizes)
    tn = _test_regions(p["nonpeaks"], fold, sizes)
    rng = np.random.RandomState(0)
    bg_src = get_seq_onehot(tn.sample(n=500, random_state=0), p["fasta"], 2114)
    bgs = np.stack([dinuc_shuffle(x, 1, rng)[0] for x in bg_src])
    rows = []
    for side in ("ref", "epione"):
        for seed in seeds:
            try:
                full, nobias, bias = _load_side(side, seed)
            except FileNotFoundError as e:
                print("missing", side, seed, e)
                continue
            for m in (full, nobias, bias):
                m.to(device)
            e_full = evaluate(full, tp, p["fasta"], sig, device=device)
            e_bias = evaluate(bias, tn, p["fasta"], sig, device=device)
            fp = marginal_footprint(nobias, CTCF_MA0139_1, backgrounds=bgs, device=device)
            rows.append({"side": side, "seed": seed,
                         "a_count_pearson": e_full["count_pearson"],
                         "b_profile_median_jsd": e_full["profile_median_jsd"],
                         "c_bias_count_pearson_nonpeaks": e_bias["count_pearson"],
                         "d_ctcf_footprint_depth": fp["depth"],
                         "n_test_peaks": e_full["n_regions"], "n_test_nonpeaks": e_bias["n_regions"]})
            print(rows[-1], flush=True)
    floors = {"a_count_pearson": 0.02, "b_profile_median_jsd": 0.01,
              "c_bias_count_pearson_nonpeaks": 0.03, "d_ctcf_footprint_depth": 0.05}
    gates = {}
    for k, fl in floors.items():
        r = np.array([x[k] for x in rows if x["side"] == "ref"])
        e = np.array([x[k] for x in rows if x["side"] == "epione"])
        if len(r) < 2 or len(e) < 1:
            continue
        tol = max(2 * r.std(ddof=1), fl)
        d = abs(e.mean() - r.mean())
        gates[k] = {"ref_mean": r.mean(), "ref_sd": r.std(ddof=1), "epione_mean": e.mean(),
                    "epione_sd": e.std(ddof=1) if len(e) > 1 else float("nan"),
                    "abs_diff": d, "tol": tol, "pass": bool(d <= tol)}
    res = {"rows": rows, "gates": gates, "pass": bool(gates) and all(g["pass"] for g in gates.values())}
    (PAR / "g3_results.json").write_text(_json.dumps(res, indent=2, default=float))
    print(_json.dumps(gates, indent=1, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--fasta")
    ap.add_argument("--model-dir")
    ap.add_argument("--experiment", default="ENCSR637XSC")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    if a.cmd == "g1-inputs":
        g1_inputs(a.fasta)
    elif a.cmd == "g1-compare":
        g1_compare(a.model_dir, a.experiment)
    elif a.cmd == "g2":
        out = {"insertions": g2_insertions(), "bias_hp": g2_bias_hp(),
               "chrombpnet_hp": g2_chrombpnet_hp()}
        (PAR / "g2_results.json").write_text(json.dumps(out, indent=2, default=float))
    elif a.cmd == "g3-train":
        g3_train(a.seed)
    elif a.cmd == "g3-eval":
        g3_eval()
    elif a.cmd == "g2-insertions":
        g2_insertions()
    elif a.cmd == "g2-bias-hp":
        g2_bias_hp()
