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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--fasta")
    ap.add_argument("--model-dir")
    ap.add_argument("--experiment", default="ENCSR637XSC")
    a = ap.parse_args()
    if a.cmd == "g1-inputs":
        g1_inputs(a.fasta)
    elif a.cmd == "g1-compare":
        g1_compare(a.model_dir, a.experiment)
