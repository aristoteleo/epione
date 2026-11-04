"""gReLU bridge validation (see MANIFEST.md). Needs grelu installed.

    python parity_check.py v1      # adapter exactness
    python parity_check.py v2      # bridge == direct gReLU calls
    python parity_check.py v3      # CTCF sign test (ENCODE GM12878 ChromBPNet)
    python parity_check.py v4      # pseudobulk builder
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("EPIONE_GRELU_PARITY_DIR", "."))
FASTA = os.environ.get("EPIONE_HG38_FASTA", "")


def _models():
    import torch

    from epione.models.chrombpnet import BPNet, ChromBPNet, glorot_init_
    from epione.models.seq2print import build_seq2print
    torch.manual_seed(0)
    small = dict(input_len=256, output_len=100)
    b = glorot_init_(BPNet(filters=8, n_dil_layers=2, **small))
    a = glorot_init_(BPNet(filters=16, n_dil_layers=2, **small))
    full = ChromBPNet(b, a).eval()
    s2p = build_seq2print(n_filters=16, n_layers=2, output_len=100).eval()
    return {"bpnet": a.eval(), "chrombpnet": full, "seq2print": s2p}


def v1():
    import torch

    from epione.models.grelu import HEADS, wrap
    res = {}
    for kind, m in _models().items():
        L = m.dna_len if kind == "seq2print" else m.input_len
        x = torch.randint(0, 2, (8, 4, L)).float()
        for head in HEADS[kind]:
            lm = wrap(m, head=head)
            with torch.no_grad():
                y = lm(x)
                raw = m(x)
                if kind == "seq2print":
                    fp, cov = raw
                    ref = cov.reshape(-1, 1, 1) if head == "coverage" else fp
                else:
                    prof, cnt = raw
                    probs = torch.softmax(prof, -1)
                    ref = {"counts": cnt[:, :, None], "profile_logits": prof[:, None],
                           "profile": probs[:, None],
                           "expected_counts": (probs * torch.exp(cnt))[:, None]}[head]
            res[f"{kind}/{head}"] = bool(torch.equal(y, ref))
    res["pass"] = all(res.values())
    print("V1", res)
    return res


def _variants(n=20, seed=0):
    import pyfaidx
    fa = pyfaidx.Fasta(FASTA)
    rng = np.random.RandomState(seed)
    rows = []
    while len(rows) < n:
        pos = int(rng.randint(1_000_000, 200_000_000))  # 1-based
        ref = str(fa["chr1"][pos - 1:pos]).upper()
        if ref not in "ACGT":
            continue
        alt = rng.choice([b for b in "ACGT" if b != ref])
        rows.append(("chr1", pos, ref, alt))
    return pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt"])


def v2():
    import grelu.variant
    from grelu.interpret.score import ISM_predict, get_attributions

    from epione.models import grelu as eg
    m = _models()["chrombpnet"]
    lm = eg.wrap(m)
    var = _variants()
    a = eg.predict_variant_effects(m, var, FASTA, return_ad=False)
    from epione.models.grelu._bridge import _genome_arg, _no_lightning_logs
    with _no_lightning_logs():
        b = grelu.variant.predict_variant_effects(var, lm, genome=_genome_arg(FASTA),
                                                  compare_func="subtract", return_ad=False)
    rng = np.random.RandomState(1)
    seq = "".join(rng.choice(list("ACGT"), m.input_len))
    i1 = eg.ism(m, seq, return_df=False)
    with _no_lightning_logs():
        i2 = ISM_predict(seq, lm, compare_func="subtract", return_df=False)
    seqs = ["".join(rng.choice(list("ACGT"), m.input_len)) for _ in range(3)]
    s1 = eg.attributions(m, seqs, method="saliency")
    s2 = get_attributions(lm, seqs, method="saliency", seed=0)
    d1 = eg.attributions(m, seqs, method="deepshap", seed=0)
    d2 = get_attributions(lm, seqs, method="deepshap", seed=0)
    res = {"variant_effects": bool(np.array_equal(a, b)), "ism": bool(np.array_equal(i1, i2)),
           "saliency": bool(np.array_equal(s1, s2)), "deepshap": bool(np.array_equal(d1, d2))}
    res["pass"] = all(res.values())
    print("V2", res)
    return res


CTCF_CORE = re.compile(r"(?=(CCA[CG][CT]AG[AG][GT]GG[CA]))")
CTCF_CORE_RC = re.compile(r"(?=([TG]CC[AC][CT]CT[AG][GC]TGG))")
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def v3(peaks_path: str, n: int = 300):
    import pyfaidx

    from epione.models import grelu as eg
    from epione.models.chrombpnet import load_pretrained, read_narrowpeak
    fa = pyfaidx.Fasta(FASTA)
    pk = read_narrowpeak(peaks_path)
    pk = pk[pk["chr"].isin(["chr1", "chr3", "chr6"])].sort_values(["chr", "start"])
    motif, ctrl = [], []
    rng = np.random.RandomState(0)
    for c, s, smt in zip(pk["chr"], pk["start"], pk["summit"]):
        mid = s + smt
        a = mid - 200
        seq = str(fa[c][a:mid + 200]).upper()
        hits = []
        for m in CTCF_CORE.finditer(seq):
            hits.append((m.start(), "+"))
        for m in CTCF_CORE_RC.finditer(seq):
            hits.append((m.start(), "-"))
        if not hits:
            continue
        off, strand = sorted(hits)[0]
        # 10th core base: G of AG[AG]GG on +; on - the regex is the reverse
        # complement, so the same base sits at index 12 - 9 = 3 (a C).
        k = off + (9 if strand == "+" else 3)
        ref = seq[k]
        motif.append((c, a + k + 1, ref, COMP[ref]))
        far = [i for i in range(len(seq)) if all(abs(i - (h + 6)) >= 50 for h, _ in hits)
               and seq[i] in "ACGT"]
        if far:
            i = int(rng.choice(far))
            r = seq[i]
            ctrl.append((c, a + i + 1, r, rng.choice([b for b in "ACGT" if b != r])))
        if len(motif) >= n:
            break
    cols = ["chrom", "pos", "ref", "alt"]
    mv, cv = pd.DataFrame(motif, columns=cols), pd.DataFrame(ctrl, columns=cols)
    model = load_pretrained("encode_gm12878", "nobias", 0)
    em = eg.predict_variant_effects(model, mv, FASTA, return_ad=False).ravel()
    ec = eg.predict_variant_effects(model, cv, FASTA, return_ad=False).ravel()
    res = {"n_motif": len(em), "n_control": len(ec),
           "motif_median": float(np.median(em)), "motif_frac_negative": float((em < 0).mean()),
           "control_median": float(np.median(ec)), "control_frac_negative": float((ec < 0).mean())}
    res["pass"] = bool(res["motif_median"] < 0 and res["motif_frac_negative"] >= 0.7
                       and abs(res["control_median"]) < 0.01
                       and 0.4 <= res["control_frac_negative"] <= 0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"effect": np.r_[em, ec], "kind": ["motif"] * len(em) + ["control"] * len(ec)}
                 ).to_csv(OUT / "v3_effects.csv", index=False)
    print("V3", res)
    return res


def v4():
    import anndata as ad
    import scipy.sparse as sp

    from epione.models.grelu import pseudobulk_anndata
    rng = np.random.RandomState(0)
    X = sp.random(60, 10, density=0.3, random_state=0, format="csr", data_rvs=lambda k: rng.randint(1, 5, k))
    obs = pd.DataFrame({"ct": rng.choice(list("abc"), 60)}, index=[f"c{i}" for i in range(60)])
    var = pd.DataFrame(index=[f"chr1:{100*i}-{100*i+50}" for i in range(10)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    pb = pseudobulk_anndata(a, "ct")
    ref = pd.DataFrame(X.toarray(), index=obs.index).groupby(obs["ct"]).sum()
    res = {"x_equal": bool(np.array_equal(pb.X, ref.loc[pb.obs_names].to_numpy().astype(np.float32))),
           "var_cols": list(pb.var.columns), "start0": int(pb.var["start"].iloc[1])}
    res["pass"] = res["x_equal"] and res["var_cols"] == ["chrom", "start", "end"] and res["start0"] == 100
    print("V4", res)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--peaks")
    a = ap.parse_args()
    out = {"v1": v1, "v2": v2, "v4": v4}.get(a.cmd)
    if a.cmd == "v3":
        r = v3(a.peaks)
    elif a.cmd == "all":
        r = {"v1": v1(), "v2": v2(), "v3": v3(a.peaks), "v4": v4()}
    else:
        r = out()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.cmd}_results.json").write_text(json.dumps(r, indent=2, default=float))
