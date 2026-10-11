"""Gates F1-F5 of MANIFEST.md: fast LDA samplers vs the exact sampler.

usage::

    python fast_vs_exact.py <parity_dir> --exact e1.pkl e2.pkl e3.pkl \\
        --fast parallel=p1.pkl,p2.pkl,p3.pkl --fast gpu=g1.pkl,g2.pkl,g3.pkl [--out res.json]

Each pickle holds one ``TopicModel`` (K = 20, 500 sweeps, run A data).
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln
from sklearn.metrics import adjusted_rand_score


def complete_loglik(model) -> float:
    """log p(w, z) as in ``lda.LDA.loglikelihood`` (alpha = 50/K, eta = 0.1)."""
    nzw = model.extra["nzw"].astype(np.float64)
    ndz = model.extra["ndz"].astype(np.float64)
    K, V = nzw.shape
    a = float(model.parameters.loc["alpha", "Parameter"]) / K
    e = float(model.parameters.loc["eta", "Parameter"])
    nz = nzw.sum(axis=1)
    nd = ndz.sum(axis=1)
    ll = K * gammaln(e * V) - gammaln(e * V + nz).sum()
    ll += (gammaln(e + nzw[nzw > 0]) - gammaln(e)).sum()
    ll += (gammaln(a * K) - gammaln(a * K + nd)).sum()
    ll += (gammaln(a + ndz[ndz > 0]) - gammaln(a)).sum()
    return float(ll)


def match_topics(m1, m2):
    """Hungarian matching of topics by topic-region Pearson correlation."""
    a = m1.topic_region.to_numpy()
    b = m2.topic_region.to_numpy()
    a = (a - a.mean(0)) / a.std(0)
    b = (b - b.mean(0)) / b.std(0)
    corr = a.T @ b / a.shape[0]
    r, c = linear_sum_assignment(-corr)
    return r, c, corr[r, c]


def downstream(model, parity: Path):
    from epione import scenic

    otsu = scenic.binarize_topics(model, method="otsu")
    sets = [set(otsu[f"Topic{i + 1}"].index) for i in range(model.n_topic)]
    imp = scenic.impute_accessibility(model, scale_factor=10 ** 6)
    norm = scenic.normalize_scores(imp, 10 ** 4, compat="native")
    hvf = scenic.find_highly_variable_features(norm, min_disp=0.05, min_mean=0.0125, max_mean=3,
                                               n_bins=20)
    del norm
    cd = pd.read_csv(parity / "A" / "inputs" / "cistopic_cell_data.tsv", sep="\t", index_col=0)
    dars = scenic.find_diff_features(imp, cd["Seurat_cell_type"], var_features=hvf)
    return sets, {ct: set(df.index) for ct, df in dars.items()}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def pair_metrics(m1, d1, m2, d2):
    r, c, rr = match_topics(m1, m2)
    ari = adjusted_rand_score(m1.cell_topic.to_numpy().argmax(0), m2.cell_topic.to_numpy().argmax(0))
    sets1, dar1 = d1
    sets2, dar2 = d2
    topic_j = float(np.mean([jaccard(sets1[i], sets2[j]) for i, j in zip(r, c)]))
    dar_j = float(np.mean([jaccard(dar1[ct], dar2.get(ct, set())) for ct in dar1]))
    return {"F1_topic_r": float(rr.mean()), "F2_ari": float(ari), "F4_topic_jaccard": topic_j,
            "F5_dar_jaccard": dar_j}


def run_values(m):
    v = {"loglik": complete_loglik(m)}
    v.update({k: float(m.metrics[k].iloc[0]) for k in ("Arun_2010", "Cao_Juan_2009", "Mimno_2011")})
    return v


def evaluate(parity: Path, exact_paths, fast: dict):
    load = lambda p: pickle.load(open(p, "rb"))
    exact = [load(p) for p in exact_paths]
    ex_down = [downstream(m, parity) for m in exact]
    spread = [pair_metrics(exact[i], ex_down[i], exact[j], ex_down[j])
              for i, j in itertools.combinations(range(len(exact)), 2)]
    ex_vals = [run_values(m) for m in exact]
    out = {"exact_pairs": spread, "exact_values": ex_vals, "modes": {}}
    for mode, paths in fast.items():
        models = [load(p) for p in paths]
        downs = [downstream(m, parity) for m in models]
        agree = [pair_metrics(f, fd, e, ed) for f, fd in zip(models, downs)
                 for e, ed in zip(exact, ex_down)]
        vals = [run_values(m) for m in models]
        gates = {}
        for key in ("F1_topic_r", "F2_ari", "F4_topic_jaccard", "F5_dar_jaccard"):
            med = float(np.median([a[key] for a in agree]))
            lo = float(min(s[key] for s in spread))
            gates[key] = {"median_fast_vs_exact": med, "min_exact_vs_exact": lo, "pass": med >= lo}
        f3 = {}
        for key in ("loglik", "Arun_2010", "Cao_Juan_2009", "Mimno_2011"):
            ev = [v[key] for v in ex_vals]
            R = max(ev) - min(ev)
            lo, hi = min(ev) - R, max(ev) + R
            fv = [v[key] for v in vals]
            f3[key] = {"fast": fv, "exact_min": min(ev), "exact_max": max(ev),
                       "pass": all(lo <= x <= hi for x in fv)}
        gates["F3"] = f3
        out["modes"][mode] = {"pairs": agree, "values": vals, "gates": gates}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parity_dir", type=Path)
    ap.add_argument("--exact", nargs=3, required=True)
    ap.add_argument("--fast", action="append", default=[], help="mode=p1.pkl,p2.pkl,p3.pkl")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    fast = {s.split("=")[0]: s.split("=")[1].split(",") for s in args.fast}
    res = evaluate(args.parity_dir, args.exact, fast)
    for mode, r in res["modes"].items():
        print(mode, json.dumps(r["gates"], indent=1))
    if args.out:
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
