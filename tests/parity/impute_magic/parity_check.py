"""Compare epione.tl.add_impute_weights / impute_matrix with the ArchR reference.

Usage: python parity_check.py <ref_dir> [out_json]
<ref_dir> is the output directory of r_reference.R.
"""
import glob
import json
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd

import epione as epi


def load_ref(ref):
    mat = pd.read_csv(os.path.join(ref, "matDR.tsv.gz"), sep="\t", index_col=0)
    imp = {t: pd.read_csv(os.path.join(ref, f"{t}_imputed.tsv.gz"), sep="\t", index_col=0)
           for t in ("seed1", "seed2")}
    raw = pd.read_csv(os.path.join(ref, "raw_genescore.tsv.gz"), sep="\t", index_col=0)
    blocks = {}
    for f in sorted(glob.glob(os.path.join(ref, "seed1_rep*_block*_names.txt"))):
        stem = f[: -len("_names.txt")]
        r, b = (int(x) for x in os.path.basename(stem)[len("seed1_rep"):].split("_block"))
        names = open(f).read().split()
        w = np.fromfile(stem + "_weights.f64", dtype="<f8").reshape(len(names), len(names), order="F")
        blocks.setdefault(r, {})[b] = (names, w)
    return mat, imp, raw, blocks


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main(ref, out_json=None):
    mat, imp, raw, rblocks = load_ref(ref)
    cells = mat.index.to_numpy()
    pos = {c: i for i, c in enumerate(cells)}
    adata = ad.AnnData(X=raw.T.loc[cells].to_numpy(dtype=np.float64))
    adata.obs_names = cells
    adata.var_names = raw.index.to_numpy()
    adata.obsm["X_lsi"] = mat.to_numpy(dtype=np.float64)

    # Fixture LSI export == getReducedDims() inside addImputeWeights.
    fx = "/scratch/users/steorra/analysis/25_epione/epione_dev/fixture/tables/lsi.tsv"
    res = {}
    if os.path.exists(fx):
        lsi = pd.read_csv(fx, sep="\t", index_col=0).loc[cells]
        res["reduced_dims_max_abs_err"] = max_abs(lsi.to_numpy(), mat.to_numpy())

    r_imp = imp["seed1"].loc[adata.var_names, cells].T.to_numpy()
    scale = float(np.max(np.abs(r_imp)))

    def compare_weights(a, tag):
        errs, same = [], True
        for r, rep in enumerate(a.uns["impute_weights"]["weights"], start=1):
            for b, blk in enumerate(rep, start=1):
                names, w = rblocks[r][b]
                same &= set(cells[blk["cells"]]) == set(names)
                order = np.array([pos[n] for n in names])
                loc = {c: i for i, c in enumerate(blk["cells"])}
                perm = np.array([loc[o] for o in order])
                errs.append(max_abs(np.asarray(blk["weights"])[np.ix_(perm, perm)], w))
        res[f"{tag}_blocks_identical"] = bool(same)
        res[f"{tag}_weights_max_abs_err"] = max(errs)

    # (a) R draws injected
    inj = [[np.array([pos[n] for n in rblocks[r][b][0]]) for b in sorted(rblocks[r])]
           for r in sorted(rblocks)]
    t0 = time.time()
    epi.tl.add_impute_weights(adata, use_rep="X_lsi", blocks=inj, verbose=False)
    t1 = time.time()
    out = epi.tl.impute_matrix(adata)
    t2 = time.time()
    compare_weights(adata, "injected")
    res["injected_imputed_max_abs_err"] = max_abs(out, r_imp)
    res["injected_imputed_max_rel_err"] = res["injected_imputed_max_abs_err"] / scale
    res["timing_weights_s"], res["timing_impute_s"] = t1 - t0, t2 - t1

    # (b) free-running, R RNG emulation
    epi.tl.add_impute_weights(adata, use_rep="X_lsi", seed=1, rng="R", verbose=False)
    out_r = epi.tl.impute_matrix(adata)
    compare_weights(adata, "rngR")
    res["rngR_imputed_max_abs_err"] = max_abs(out_r, r_imp)
    res["rngR_imputed_max_rel_err"] = res["rngR_imputed_max_abs_err"] / scale

    # (c) free-running, numpy RNG: correlation class vs R seed-1 / seed-2 baseline
    def pearson_per_gene(a, b):
        a = a - a.mean(0); b = b - b.mean(0)
        return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))

    epi.tl.add_impute_weights(adata, use_rep="X_lsi", seed=1, rng="numpy", verbose=False)
    out_np = epi.tl.impute_matrix(adata)
    r_imp2 = imp["seed2"].loc[adata.var_names, cells].T.to_numpy()
    ok = r_imp.std(0) > 0
    res["numpy_median_gene_pearson"] = float(np.median(pearson_per_gene(out_np, r_imp)[ok]))
    res["numpy_min_gene_pearson"] = float(np.min(pearson_per_gene(out_np, r_imp)[ok]))
    res["archr_seed1_vs_seed2_median_gene_pearson"] = float(
        np.median(pearson_per_gene(r_imp2, r_imp)[ok]))
    res["archr_seed1_vs_seed2_min_gene_pearson"] = float(
        np.min(pearson_per_gene(r_imp2, r_imp)[ok]))
    res["n_genes"], res["n_cells"] = int(adata.n_vars), int(adata.n_obs)

    print(json.dumps(res, indent=2))
    if out_json:
        json.dump(res, open(out_json, "w"), indent=2)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
