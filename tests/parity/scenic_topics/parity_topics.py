"""Parity of ``epione.scenic`` topic block against pycisTopic reference outputs.

The reference outputs are produced outside this repository with the SCENIC+
1.0 stack (see MANIFEST.md) and laid out as ``<parity_dir>/<run>/<stage>/``.

usage::

    python parity_topics.py <parity_dir> [--stages matrix lda_S lda_A binarize_dar]
        [--fragments F --blacklist B --cell-data C] [--models-pkl P] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp


def _read_tsv(path, **kw):
    # pandas' default float parser is not round-trip exact
    return pd.read_csv(path, sep="\t", float_precision="round_trip", **kw)


def _lines(path):
    return Path(path).read_text().split("\n")[:-1]


def _rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.where(b == 0, 1.0, np.abs(b))
    return float(np.max(np.abs(a - b) / denom)) if a.size else 0.0


def load_binary_adata(parity: Path, run: str = "A"):
    import anndata as ad

    d = parity / run / "inputs"
    m = sp.csr_matrix(sio.mmread(d / "binary_matrix.mtx")).astype(np.int32)
    a = ad.AnnData(sp.csr_matrix(m.T))
    a.obs_names = _lines(d / "binary_matrix_cols.txt")
    a.var_names = _lines(d / "binary_matrix_rows.txt")
    return a


def load_reference_model(parity: Path, run: str, k: int):
    from epione.scenic import TopicModel

    d = parity / run / "topics" / f"model_{k}"
    ct = pd.DataFrame(np.load(d / "cell_topic.npy"), index=_lines(d / "cell_topic_rows.txt"),
                      columns=_lines(d / "cell_topic_cols.txt"))
    tr = pd.DataFrame(np.load(d / "topic_region.npy"), index=_lines(d / "topic_region_rows.txt"),
                      columns=_lines(d / "topic_region_cols.txt"))
    return TopicModel(k, ct, tr, None, None, None, None, None)


# ---------------------------------------------------------------------------

def check_matrix(parity: Path, fragments, blacklist, cell_data):
    """Gates 1-2: fragment counts in consensus regions."""
    from epione.scenic import cistopic_matrix

    d = parity / "A" / "inputs"
    cd = pd.read_csv(cell_data, sep="\t", index_col=0)
    valid = [c.rsplit("-", 1)[0] for c in cd.index]
    cons = pd.read_csv(d / "consensus_regions.tsv", sep="\t", usecols=[0, 1, 2])
    t = time.time()
    a = cistopic_matrix(fragments, cons, barcodes=valid, blacklist=blacklist,
                        project="10x_multiome_brain", split_pattern="-", verbose=False)
    wall = time.time() - t
    rows = _lines(d / "fragment_matrix_rows.txt")
    cols = _lines(d / "fragment_matrix_cols.txt")
    out = {"wall_s": wall,
           "region_order_equal": list(a.var_names) == rows,
           "cell_order_equal": list(a.obs_names) == cols}
    if out["region_order_equal"] and out["cell_order_equal"]:
        F = sp.csr_matrix(sio.mmread(d / "fragment_matrix.mtx")).astype(np.int32)
        B = sp.csr_matrix(sio.mmread(d / "binary_matrix.mtx")).astype(np.int32)
        out["fragment_matrix_n_diff"] = int((sp.csr_matrix(a.X.T) != F).nnz)
        out["binary_matrix_n_diff"] = int((sp.csr_matrix(a.layers["binary"].T) != B).nnz)
    rc = _read_tsv(d / "cistopic_cell_data.tsv", index_col=0)
    rr = _read_tsv(d / "cistopic_region_data.tsv", index_col=0)
    o = a.obs.loc[rc.index]
    v = a.var.loc[rr.index]
    out["cell_counts_equal"] = bool(all(np.array_equal(o[c].to_numpy(), rc[c].to_numpy())
                                        for c in ("cisTopic_nr_frag", "cisTopic_nr_acc",
                                                  "Unique_nr_frag")))
    out["region_counts_equal"] = bool(all(np.array_equal(v[c].to_numpy(), rr[c].to_numpy())
                                          for c in ("Width", "cisTopic_nr_frag",
                                                    "cisTopic_nr_acc")))
    out["log10_columns_max_abs"] = float(max(
        np.abs(o[c].to_numpy() - rc[c].to_numpy()).max() for c in
        ("cisTopic_log_nr_frag", "cisTopic_log_nr_acc")))
    return out


def check_lda(parity: Path, run: str = "S", n_topics=(10, 20), n_iter: int = 5, models=None,
              n_jobs=None):
    """Gates 3-7: LDA models, metrics and model selection."""
    from epione.scenic import evaluate_topic_models, run_topic_models

    out = {"models": {}}
    if models is None:
        a = load_binary_adata(parity, run)
        t = time.time()
        models = run_topic_models(a, list(n_topics), n_iter=n_iter, n_jobs=n_jobs,
                                  method="exact", verbose=False)
        out["wall_s"] = time.time() - t
    for m in models:
        d = parity / run / "topics" / f"model_{m.n_topic}"
        ass = _read_tsv(d / "topic_ass.tsv")
        met = _read_tsv(d / "metrics.tsv")
        coh = _read_tsv(d / "coherence.tsv")
        marg = _read_tsv(d / "marg_topic.tsv")
        out["models"][m.n_topic] = {
            "topic_ass_equal": bool(np.array_equal(
                ass["Assignments"].to_numpy(), m.topic_ass["Assignments"].to_numpy().astype(int))),
            "cell_topic_max_abs": float(np.abs(np.load(d / "cell_topic.npy")
                                               - m.cell_topic.to_numpy()).max()),
            "topic_region_max_abs": float(np.abs(np.load(d / "topic_region.npy")
                                                 - m.topic_region.to_numpy()).max()),
            "metrics_max_rel": max(_rel(m.metrics[c].to_numpy(), met[c].to_numpy())
                                   for c in met.columns),
            "metrics_rel": {c: _rel(m.metrics[c].to_numpy(), met[c].to_numpy())
                            for c in met.columns},
            "coherence_max_rel": _rel(m.coherence["Mimno_2011"].to_numpy(),
                                      coh["Mimno_2011"].to_numpy()),
            "marg_topic_max_rel": _rel(m.marg_topic["Marg_Topic"].to_numpy(),
                                       marg["Marg_Topic"].to_numpy()),
            "fit_s": float(m.parameters.loc["time", "Parameter"]),
        }
    sel = evaluate_topic_models(models)
    ref_sel = int((parity / run / "topics" / "selected_model.txt").read_text().strip())
    out["selected"] = sel.n_topic
    out["selected_equal"] = sel.n_topic == ref_sel
    return out


def _bin_table(b):
    rows = [(t, idx, v) for t, s in b.items() for idx, v in s.iloc[:, 0].items()]
    return pd.DataFrame(rows, columns=["topic", "feature", "score"])


def check_binarize_dar(parity: Path, model=None, run: str = "A", k: int = 20):
    """Gates 8-15 (isolated: fed the reference topic model)."""
    from epione import scenic

    d = parity / run / "binarize_dar"
    m = load_reference_model(parity, run, k) if model is None else model
    out = {"timing_s": {}}
    t = time.time()
    bins = {
        "region_bin_topics_top_3k": scenic.binarize_topics(m, method="ntop", ntop=3000),
        "region_bin_topics_otsu": scenic.binarize_topics(m, method="otsu"),
        "cell_bin_topics_li": scenic.binarize_topics(m, target="cell", method="li", nbins=100),
    }
    out["timing_s"]["binarize_topics"] = time.time() - t
    for name, b in bins.items():
        ref = _read_tsv(d / f"{name}.tsv")
        mine = _bin_table(b)
        same = (len(ref) == len(mine) and (ref.topic.values == mine.topic.values).all()
                and (ref.feature.values == mine.feature.values).all())
        out[name] = {"order_equal": bool(same), "n": len(mine),
                     "score_max_abs": float(np.abs(ref.score.values - mine.score.values).max())
                     if same else None}

    t = time.time()
    imp = scenic.impute_accessibility(m, scale_factor=10 ** 6)
    out["timing_s"]["impute_accessibility"] = time.time() - t
    z = np.load(d / "imputed_acc.npz")["mtx"]
    rows_equal = _lines(d / "imputed_acc_rows.txt") == imp.feature_names
    out["impute"] = {"rows_equal": rows_equal,
                     "n_diff": int(np.count_nonzero(z != imp.mtx)) if rows_equal else None}
    del z

    t = time.time()
    norm = scenic.normalize_scores(imp, 10 ** 4)
    hvf = scenic.find_highly_variable_features(norm, min_disp=0.05, min_mean=0.0125, max_mean=3,
                                               n_bins=20)
    out["timing_s"]["normalize_hvf"] = time.time() - t
    zn = np.load(d / "normalized_acc_first2000.npz", allow_pickle=True)
    out["normalize_first2000_max_rel"] = _rel(norm.mtx[:2000], zn["mtx"])
    del norm
    ref_hvf = _lines(d / "variable_regions.txt")
    out["hvf_equal"] = hvf == ref_hvf
    out["hvf_n"] = len(hvf)

    cd = pd.read_csv(parity / run / "inputs" / "cistopic_cell_data.tsv", sep="\t", index_col=0)
    t = time.time()
    dars = scenic.find_diff_features(imp, cd["Seurat_cell_type"], var_features=hvf)
    out["timing_s"]["find_diff_features"] = time.time() - t
    dar_out = {}
    for ct, df in dars.items():
        ref = _read_tsv(d / "DARs" / f"{ct.replace('/', '_')}.tsv", index_col=0)
        same = list(ref.index) == list(df.index)
        dar_out[ct] = {"n": len(df), "order_equal": same,
                       "log2fc_max_rel": _rel(df["Log2FC"], ref["Log2FC"]) if same else None,
                       "adjp_max_rel": _rel(df["Adjusted_pval"], ref["Adjusted_pval"])
                       if same else None}
    out["dars"] = dar_out

    sets = {
        "Topics_otsu": {t: bins["region_bin_topics_otsu"][t].index for t in bins["region_bin_topics_otsu"]},
        "Topics_top_3k": {t: bins["region_bin_topics_top_3k"][t].index
                          for t in bins["region_bin_topics_top_3k"]},
        "DARs_cell_type": {ct.replace("/", "_"): dars[ct].index for ct in dars if len(dars[ct])},
    }
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        scenic.write_region_sets(tmp, sets)
        n_files = n_equal = 0
        for sub in sets:
            for f in sorted((d / "region_sets" / sub).glob("*.bed")):
                n_files += 1
                mine = Path(tmp) / sub / f.name
                n_equal += int(mine.exists() and mine.read_bytes() == f.read_bytes())
        out["region_sets"] = {"files": n_files, "byte_equal": n_equal}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parity_dir", type=Path)
    ap.add_argument("--stages", nargs="+", default=["lda_S", "binarize_dar"])
    ap.add_argument("--fragments")
    ap.add_argument("--blacklist")
    ap.add_argument("--cell-data")
    ap.add_argument("--models-pkl", help="pickled list of TopicModel for run A (skips refitting)")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    res = {}
    if "matrix" in args.stages:
        res["matrix"] = check_matrix(args.parity_dir, args.fragments, args.blacklist, args.cell_data)
        print("matrix", res["matrix"], flush=True)
    if "lda_S" in args.stages:
        res["lda_S"] = check_lda(args.parity_dir, "S", (10, 20), 5, n_jobs=args.n_jobs)
        print("lda_S", json.dumps(res["lda_S"], default=str), flush=True)
    if "lda_A" in args.stages:
        models = None
        if args.models_pkl:
            with open(args.models_pkl, "rb") as fh:
                models = pickle.load(fh)
        res["lda_A"] = check_lda(args.parity_dir, "A", (10, 20, 30, 40), 500, models=models,
                                 n_jobs=args.n_jobs)
        print("lda_A", json.dumps(res["lda_A"], default=str), flush=True)
    if "binarize_dar" in args.stages:
        res["binarize_dar"] = check_binarize_dar(args.parity_dir)
        print("binarize_dar", json.dumps(res["binarize_dar"], default=str), flush=True)
    if args.out:
        args.out.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
