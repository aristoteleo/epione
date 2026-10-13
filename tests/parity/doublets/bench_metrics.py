"""Benchmark metrics on GSE162690 (truth: demuxlet DBL vs singlet; AMB excluded).

Usage: python bench_metrics.py   ->  analysis/benchmark.json, analysis/benchmark.tsv
Each method contributes a score (higher = more doublet-like) and, where it has
one, a default call.
"""
import os
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score

R = Path(os.environ["EPIONE_DOUBLETS_PARITY_DIR"])
D, B = R / "data/GSE162690", R / "bench"
rows = []
for s in ("LowLoading", "HighLoading"):
    gt = pd.read_csv(D / f"{s}_cells.tsv", sep="\t", index_col=0)
    gt = gt[gt.DemuxletClassify != "AMB"]
    y = (gt.DemuxletClassify == "DBL").to_numpy()
    scores, calls = {}, {}
    scores["ArchR (published)"] = gt.DoubletEnrichment
    f = B / f"{s}_epione_scores.tsv"
    if f.exists():
        e = pd.read_csv(f, sep="\t", index_col=0).reindex(gt.index)
        scores["ArchR port (enrichment)"] = e.archr_doublet_enrichment
        scores["ArchR port (score)"] = e.archr_doublet_score
        calls["ArchR port (enrichment)"] = e.archr_port_removed.astype(bool)
        scores["scrublet"] = e.scrublet_score
        calls["scrublet"] = e.scrublet_is_doublet.astype(bool)
    f = B / f"{s}_amulet_R.tsv"
    if f.exists():
        a = pd.read_csv(f, sep="\t", index_col=0).reindex(gt.index)
        scores["AMULET (R scDblFinder)"] = -np.log10(np.maximum(a["p.value"], 1e-300))
        calls["AMULET (R scDblFinder)"] = a["q.value"] < 0.01
    f = B / f"{s}_amulet_pyscdbl.tsv"
    if f.exists():
        ap_ = pd.read_csv(f, sep="\t", index_col=0).reindex(gt.index)
        scores["AMULET (pyscdblfinder branch)"] = -np.log10(np.maximum(ap_["amulet_p"], 1e-300))
        calls["AMULET (pyscdblfinder branch)"] = ap_["amulet_q"] < 0.01
        if "AMULET (R scDblFinder)" not in scores:
            a = ap_.rename(columns={"amulet_p": "p.value"})
            scores["_amulet_for_fisher"] = None
    f = B / f"{s}_scDblFinder_R.tsv"
    if f.exists():
        d = pd.read_csv(f, sep="\t", index_col=0).reindex(gt.index)
        scores["scDblFinder-ATAC (R)"] = d.score
        calls["scDblFinder-ATAC (R)"] = d["class"] == "doublet"
        if "AMULET (R scDblFinder)" in scores or "_amulet_for_fisher" in scores:
            p = np.maximum(np.c_[a["p.value"], 1 - d.score], 0.001)
            scores["Fisher(AMULET, scDblFinder)"] = -np.log10(chi2.sf(-2 * np.log(p).sum(1), 4))
    scores.pop("_amulet_for_fisher", None)
    f = B / f"{s}_clamulet_R.tsv"
    if f.exists():
        c = pd.read_csv(f, sep="\t", index_col=0).reindex(gt.index)
        scores["clamulet (R)"] = c.score
    for m, sc in scores.items():
        v = np.asarray(sc, dtype=float)
        ok = np.isfinite(v)
        r = dict(dataset=s, method=m, n=int(ok.sum()), n_doublets=int(y[ok].sum()),
                 AUROC=roc_auc_score(y[ok], v[ok]), AUPRC=average_precision_score(y[ok], v[ok]))
        if m in calls:
            cl = np.asarray(calls[m], dtype=bool)[ok]
            tp = (cl & y[ok]).sum()
            r.update(n_called=int(cl.sum()), precision=tp / max(cl.sum(), 1), recall=tp / max(y[ok].sum(), 1))
        rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(R / "analysis/benchmark.tsv", sep="\t", index=False)
b1 = {}
for s in df.dataset.unique():
    x = df[df.dataset == s].set_index("method")
    if "ArchR port (enrichment)" in x.index:
        d_ = abs(x.loc["ArchR port (enrichment)", "AUPRC"] - x.loc["ArchR (published)", "AUPRC"])
        b1[s] = dict(delta_AUPRC=float(d_), passed=bool(d_ <= 0.05))
json.dump(dict(table=rows, gate_B1=b1), open(R / "analysis/benchmark.json", "w"), indent=1, default=float)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(df.round(3).to_string(index=False)); print("B1:", b1)
