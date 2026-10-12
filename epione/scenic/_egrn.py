"""Enhancer-driven GRN (eRegulon) inference, a port of the SCENIC+ GSEA approach.

For every transcription factor (TF) with a cistrome, region-to-gene (R2G) links whose region
carries the TF's motif form candidate eModules, one per binarisation of the R2G importances
(per-gene quantiles, per-gene top-n, BASC) and per R2G correlation sign. Each eModule's target
genes are tested by pre-ranked GSEA against the TF's TF-to-gene (TF2G) ranking, separately for
positively and negatively correlated TF2G links; the leading-edge genes and their regions are
kept, and eModules of the same TF and sign combination are merged into eRegulons.

Output reproduces ``scenicplus.cli.commands.infer_grn`` (SCENIC+ 1.0a2) exactly, including the
row order and the triplet ranking. Where SCENIC+ derives an order from Python ``set``
iteration, the same set operations are performed here, so the row order depends on
``PYTHONHASHSEED`` exactly as upstream does (all values are independent of it).

Speed: thresholds are computed per gene with sorted, vectorised group operations; eModule
membership comes from one sparse product per binarisation instead of per-TF set
intersections; GSEA permutations are generated once per ranking length (they do not depend
on the gene set) and scored in parallel with numba; row orders are rebuilt only for the
eModules that survive GSEA.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from ._egrn_basc import basc_thresholds_grouped
from ._egrn_gsea import gene_set_seed, gsea_batch, permutations, significance
from ._dem import p_adjust_bh
from ._np126 import argsort_np126

__all__ = ["build_egrn", "p_adjust_bh"]

REGIONS2GENES_COLUMNS = ["region", "target", "importance", "rho", "importance_x_rho", "importance_x_abs_rho"]


def _sort_descending(values: np.ndarray) -> np.ndarray:
    """``pd.Series.sort_values(ascending=False)`` order (no NaNs), with numpy 1.26 tie order."""
    idx = np.arange(len(values))[::-1]
    return idx[argsort_np126(values[::-1])][::-1]


# --------------------------------------------------------------------------- thresholds
def _groups(codes: np.ndarray, values: np.ndarray):
    """Sort values within groups; return (order, sorted values, starts, counts, group of row)."""
    order = np.lexsort((values, codes))
    sc = codes[order]
    starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
    counts = np.diff(np.r_[starts, len(sc)])
    return order, values[order], starts, counts


def _group_quantile(sorted_vals, starts, counts, q):
    # np.quantile(x, q) with the default 'linear' method, per group
    n = counts
    virtual = (n - 1) * q
    prev = np.floor(virtual)
    above = virtual >= n - 1
    prev_i = np.where(above, n - 1, prev.astype(np.int64))
    next_i = np.where(above, n - 1, prev_i + 1)
    a = sorted_vals[starts + prev_i]
    b = sorted_vals[starts + next_i]
    gamma = virtual - np.where(above, -1, prev_i)
    diff = b - a
    out = a + diff * gamma
    hi = gamma >= 0.5
    out[hi] = (b - diff * (1 - gamma))[hi]
    return out


def _thresholds(gene_codes, importances, methods):
    """Per-row thresholds for each method, in method order: list of (label, kind, thr)."""
    order, sv, starts, counts = _groups(gene_codes, importances)
    row_group = np.empty(len(importances), dtype=np.int64)
    row_group[order] = np.repeat(np.arange(len(starts)), counts)
    out = []
    for label, kind, param in methods:
        if kind == "quantile":
            g = _group_quantile(sv, starts, counts, param)
        elif kind == "top":
            idx = np.where(counts >= param, starts + counts - param, starts)
            g = sv[idx]
        elif kind == "basc":
            # BASC sees the group values in their original row order and sorts them itself
            g = basc_thresholds_grouped(importances[order], starts.astype(np.int64),
                                        counts.astype(np.int64), 100)
        else:  # pragma: no cover
            raise ValueError(kind)
        out.append((label, kind, g[row_group]))
    return out


def _binarizations(r2g: pd.DataFrame, rho_threshold, quantiles, top_n, basc, order_by):
    """Yield (context frozenset, row positions into r2g) in SCENIC+'s order."""
    methods = [(f"{q} quantile", "quantile", q) for q in quantiles]
    methods += [(f"Top {n} region-to-gene links per gene", "top", n) for n in top_n]
    if basc:
        methods.append(("BASC binarized", "basc", None))
    rho = r2g["rho"].to_numpy()
    for sign, sel in (("negative r2g", rho < -rho_threshold), ("positive r2g", rho > rho_threshold)):
        rows = np.flatnonzero(sel)
        if len(rows) == 0:
            continue
        genes = r2g["target"].to_numpy()[rows]
        _, gcodes = np.unique(genes, return_inverse=True)
        imp = r2g[order_by].to_numpy()[rows].astype(np.float64)
        for label, kind, thr in _thresholds(gcodes, imp, methods):
            passing = imp >= thr if kind == "top" else imp > thr
            if passing.sum() > 0:
                yield frozenset([label, sign]), rows[passing]


# --------------------------------------------------------------------------- eModules
class _EModule:
    __slots__ = ("tf", "ctx_id", "context", "genes")

    def __init__(self, tf, ctx_id, context, genes):
        self.tf, self.ctx_id, self.context, self.genes = tf, ctx_id, context, genes


def _cistrome_matrix(cistromes, tfs_with_tf2g):
    """(regions, TF names, regions x TF boolean CSC) restricted to TFs present in TF2G."""
    if hasattr(cistromes, "obs_names") and hasattr(cistromes, "var_names"):
        regions = np.asarray(cistromes.obs_names, dtype=object)
        tfs = np.asarray(cistromes.var_names, dtype=object)
        X = cistromes.X
    else:
        regions = np.asarray(cistromes.index, dtype=object)
        tfs = np.asarray(cistromes.columns, dtype=object)
        X = cistromes.to_numpy()
    X = sparse.csc_matrix(X).astype(bool)
    keep = np.isin(tfs, list(tfs_with_tf2g))
    return regions, tfs[keep], X[:, np.flatnonzero(keep)]


# --------------------------------------------------------------------------- main entry
def build_egrn(
    tf_to_gene: pd.DataFrame,
    region_to_gene: pd.DataFrame,
    cistromes,
    is_extended: bool = False,
    *,
    order_regions_to_genes_by: str = "importance",
    order_tfs_to_genes_by: str = "importance",
    gsea_n_perm: int = 1000,
    quantiles: Sequence[float] = (0.85, 0.90, 0.95),
    top_n_region_to_genes_per_gene: Sequence[int] = (5, 10, 15),
    binarize_using_basc: bool = True,
    rho_threshold: float = 0.05,
    min_target_genes: int = 10,
    nes_threshold: float = 0.0,
    adj_pval_threshold: float = 1.0,
    seed: int = 666,
    tf_region_rank: pd.DataFrame | None = None,
    ranking_db: str | None = None,
    triplet_rng: np.random.Generator | int = 123,
    _diagnostics: dict | None = None,
) -> pd.DataFrame:
    """Infer eRegulons (SCENIC+ GSEA approach) and return the eRegulon table.

    Parameters
    ----------
    tf_to_gene
        TF-to-gene adjacencies with columns ``TF, target, importance, regulation, rho``.
    region_to_gene
        Region-to-gene adjacencies with columns ``target, region, importance, rho,
        importance_x_rho, importance_x_abs_rho``.
    cistromes
        Regions x TFs boolean cistrome membership, as an AnnData (``obs_names`` = regions,
        ``var_names`` = TFs, ``var['motifs']`` for the triplet ranking) or a DataFrame.
    is_extended
        Whether the cistromes are extended (motif similarity / orthology) annotations;
        only affects the eRegulon names.
    quantiles, top_n_region_to_genes_per_gene, binarize_using_basc
        R2G binarisations (each one, per R2G correlation sign, gives a candidate eModule).
    rho_threshold
        |rho| threshold to split TF2G and R2G links into positive / negative.
    min_target_genes
        Minimum number of TF2G targets for a TF ranking and of leading-edge genes per eModule.
    nes_threshold, adj_pval_threshold
        eModules are kept when ``NES > nes_threshold`` and BH-adjusted p ``< adj_pval_threshold``.
    seed
        GSEA permutation seed (SCENIC+ CLI default 666).
    tf_region_rank, ranking_db
        TF-to-region scores for the triplet ranking: either a regions x TFs table of motif
        ranks (lower is better), or the path of a cisTarget rankings database (``.feather``)
        from which the best rank over each TF's motifs is taken. If neither is given, the
        ``triplet_rank`` column is omitted.
    triplet_rng
        Random generator (or seed) used to break ties in the triplet ranking. SCENIC+ keeps one
        module-level ``default_rng(123)`` per process; pass the same Generator to successive
        calls to reproduce running several eGRNs in one SCENIC+ process.

    Returns
    -------
    DataFrame with one row per (eRegulon, region, gene) and the SCENIC+ columns ``Region, Gene,
    importance_R2G, rho_R2G, importance_x_rho, importance_x_abs_rho, TF, is_extended,
    eRegulon_name, Gene_signature_name, Region_signature_name, importance_TF2G, regulation,
    rho_TF2G`` (+ ``triplet_rank``).
    """
    tf2g = tf_to_gene
    r2g = region_to_gene.reset_index(drop=True)
    regions_c, tfs, C = _cistrome_matrix(cistromes, set(tf2g["TF"]))

    # region codes shared by R2G rows and cistrome rows
    r2g_regions = r2g["region"].to_numpy(dtype=object)
    uniq, r2g_rcode = np.unique(r2g_regions, return_inverse=True)
    pos_in_uniq = pd.Index(uniq).get_indexer(regions_c)
    present = pos_in_uniq >= 0
    # regions x TF over the R2G region universe
    Crr = sparse.csr_matrix(
        (np.ones(present.sum(), dtype=np.int8), (pos_in_uniq[present], np.flatnonzero(present))),
        shape=(len(uniq), len(regions_c))) @ C.astype(np.int8)
    Crr = sparse.csr_matrix(Crr)
    targets = r2g["target"].to_numpy(dtype=object)
    # integer gene codes shared by R2G targets and TF2G targets
    gene_names, gcode_all = np.unique(np.concatenate([targets, tf2g["target"].to_numpy(dtype=object)]),
                                      return_inverse=True)
    tcode = gcode_all[:len(targets)]
    tf2g_code = gcode_all[len(targets):]

    # ---- eModules: one per (binarisation context, TF) with at least one link
    contexts, ctx_rows, emodules = [], [], []
    for ctx_id, (ctx, rows) in enumerate(_binarizations(
            r2g, rho_threshold, quantiles, top_n_region_to_genes_per_gene, binarize_using_basc,
            order_regions_to_genes_by)):
        contexts.append(ctx | frozenset(["Cistromes"]))
        ctx_rows.append(rows)
        R = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (np.arange(len(rows)), r2g_rcode[rows])),
                              shape=(len(rows), len(uniq)))
        M = sparse.csc_matrix(R @ Crr)  # rows x TF: link row carries the TF's motif
        for t in range(len(tfs)):
            hit = M.indices[M.indptr[t]:M.indptr[t + 1]]
            if len(hit):
                emodules.append(_EModule(tfs[t], ctx_id, contexts[-1], np.unique(tcode[rows[hit]])))
    relevant_tfs = {m.tf for m in emodules}
    region_sets_tf, ctx_index = {}, {}
    if _diagnostics is not None:
        _diagnostics["emodules"] = [
            (m.tf, "|".join(sorted(m.context)), r2g_regions[r].tolist(), targets[r].tolist())
            for m in emodules
            for r in [_ordered_rows(m, ctx_rows, r2g_regions, regions_c, C, tfs, region_sets_tf, ctx_index)]]
        _diagnostics["gsea"] = []
        module_index = {id(m): i for i, m in enumerate(emodules)}

    # ---- GSEA of each eModule against the TF's positive / negative TF2G ranking
    rel_mask = tf2g["TF"].isin(relevant_tfs).to_numpy()
    rel = tf2g.loc[rel_mask]
    rel_code = tf2g_code[rel_mask]
    posarr = np.full(len(gene_names), -1, dtype=np.int64)
    results = []  # (module, context, NES, pval, leading-edge gene set) in SCENIC+ order
    rs_seed = gene_set_seed(seed)
    for label, sel in (("positive tf2g", rel["rho"] > rho_threshold), ("negative tf2g", rel["rho"] < -rho_threshold)):
        sel = sel.to_numpy()
        sub = rel.loc[sel]
        sub_code = rel_code[sel]
        tf_vals, counts = np.unique(sub["TF"], return_counts=True)
        ok_tfs = set(tf_vals[counts >= min_target_genes])
        rankings = {}
        tf_arr = sub["TF"].to_numpy(dtype=object)
        vals_all = sub[order_tfs_to_genes_by].to_numpy()
        for tf, idx in pd.Series(np.arange(len(sub))).groupby(tf_arr, sort=False).indices.items():
            if tf in ok_tfs:
                idx = np.sort(idx)
                vals = vals_all[idx]
                o = _sort_descending(vals)
                rankings[tf] = (sub_code[idx][o], np.abs(vals[o]) ** 1)
        todo = [m for m in emodules if m.tf in ok_tfs]
        by_len = defaultdict(list)
        for i, m in enumerate(todo):
            by_len[len(rankings[m.tf][0])].append(i)
        es = np.empty(len(todo))
        esnull = np.empty((len(todo), gsea_n_perm))
        imx = np.empty(len(todo), dtype=np.int64)
        imn = np.empty(len(todo), dtype=np.int64)
        hits = [None] * len(todo)
        for n, idxs in by_len.items():
            perms = permutations(n, gsea_n_perm, rs_seed)
            by_tf = defaultdict(list)
            for i in idxs:
                by_tf[todo[i].tf].append(i)
            for tf, ii in by_tf.items():
                genes, c = rankings[tf]
                posarr[genes] = np.arange(n)
                tags = np.zeros((len(ii), n), dtype=np.int8)
                for k, i in enumerate(ii):
                    p = posarr[todo[i].genes]
                    tags[k, p[p >= 0]] = 1
                    hits[i] = np.flatnonzero(tags[k])
                posarr[genes] = -1
                e, en, a, b = gsea_batch(c, tags, perms)
                es[ii], esnull[ii], imx[ii], imn[ii] = e, en, a, b
        nes, pvals = significance(es, esnull) if len(todo) else (np.empty(0), np.empty(0))
        for i, m in enumerate(todo):
            genes = rankings[m.tf][0]
            e = es[i]
            if e > 0:
                le = hits[i][hits[i] <= imx[i]]
            elif e < 0:
                le = hits[i][hits[i] >= imn[i]]
            else:
                le = hits[i]
            results.append((m, m.context | frozenset([label]), nes[i], float(pvals[i]), genes[le]))
            if _diagnostics is not None:
                _diagnostics["gsea"].append((module_index[id(m)], "+" if label == "positive tf2g" else "-",
                                             float(nes[i]), float(pvals[i]), sorted(gene_names[genes[le]].tolist())))

    results = [r for r in results if not np.isnan(r[2]) and not np.isnan(r[3])]
    adj = p_adjust_bh([r[3] for r in results])
    if any(np.isnan(adj)):
        raise RuntimeError("adjusted p-values contain NaN")

    # ---- leading-edge subsetting with SCENIC+'s row order, for surviving eModules only
    kept = []
    for (m, ctx, n_es, p, le_genes), q in zip(results, adj):
        if not (q < adj_pval_threshold and n_es > nes_threshold):
            continue
        if len(np.intersect1d(le_genes, m.genes)) < min_target_genes:
            continue
        rows = _ordered_rows(m, ctx_rows, r2g_regions, regions_c, C, tfs, region_sets_tf, ctx_index)
        rows = rows[np.isin(tcode[rows], le_genes)]
        kept.append((m.tf, ctx, rows))

    # ---- merge per (TF2G sign, R2G sign) and TF, then format
    tuples_cols = [r2g[c].tolist() for c in REGIONS2GENES_COLUMNS]
    frames = []
    for tf2g_sign, r2g_sign in (("positive tf2g", "positive r2g"), ("positive tf2g", "negative r2g"),
                                ("negative tf2g", "positive r2g"), ("negative tf2g", "negative r2g")):
        group = [k for k in kept if tf2g_sign in k[1] and r2g_sign in k[1]]
        for tf in sorted({k[0] for k in group}):
            seq = [tuple(col[i] for col in tuples_cols) for k in group if k[0] == tf for i in k[2]]
            merged = list(set(seq))
            df = pd.DataFrame(merged, columns=["Region", "Gene", "importance", "rho", "importance_x_rho",
                                               "importance_x_abs_rho"])
            name = (f"{tf}_{'extended' if is_extended else 'direct'}_"
                    f"{'+' if tf2g_sign == 'positive tf2g' else '-'}/{'+' if r2g_sign == 'positive r2g' else '-'}")
            df["TF"] = tf
            df["is_extended"] = is_extended
            df["eRegulon_name"] = name
            df["Gene_signature_name"] = f"{name}_({len(set(df['Gene']))}g)"
            df["Region_signature_name"] = f"{name}_({len(set(df['Region']))}r)"
            frames.append(df)
    if not frames:
        raise RuntimeError("no eRegulon passed the filters")
    meta = pd.concat(frames)
    meta = meta.merge(right=tf2g.rename({"target": "Gene"}, axis=1), how="left", on=["TF", "Gene"],
                      suffixes=["_R2G", "_TF2G"])

    if tf_region_rank is None and ranking_db is not None:
        from ._egrn_triplet import tf_region_max_rank
        if not hasattr(cistromes, "var") or "motifs" not in cistromes.var:
            raise ValueError("cistromes.var['motifs'] is required to read the ranking database")
        motifs = {tf: m.split(",") for tf, m in zip(cistromes.var_names, cistromes.var["motifs"])}
        tf_region_rank = tf_region_max_rank(list(cistromes.obs_names), motifs, ranking_db)
    if tf_region_rank is not None:
        from ._egrn_triplet import triplet_rank
        meta["triplet_rank"] = triplet_rank(meta, tf_region_rank, rng=triplet_rng)
    return meta


def _ordered_rows(m, ctx_rows, r2g_regions, regions_c, C, tfs, region_sets_tf, ctx_index):
    """R2G rows of eModule ``m`` in SCENIC+'s order.

    SCENIC+ selects ``r2g_df.loc[list(set(regions_with_motif) & set(r2g_df.index))]``; the
    same sets are built here from the same sequences so that the iteration order matches.
    """
    tf = m.tf
    if tf not in region_sets_tf:
        t = int(np.flatnonzero(tfs == tf)[0])
        col = C[:, t]
        region_sets_tf[tf] = set(regions_c[col.indices[np.argsort(col.indices)]].tolist()) if col.nnz else set()
    if m.ctx_id not in ctx_index:
        rows = ctx_rows[m.ctx_id]
        regs = r2g_regions[rows]
        by_region = defaultdict(list)
        for r, i in zip(regs.tolist(), rows.tolist()):
            by_region[r].append(i)
        ctx_index[m.ctx_id] = (set(regs.tolist()), by_region)
    ctx_set, by_region = ctx_index[m.ctx_id]
    out = []
    for r in list(region_sets_tf[tf] & ctx_set):
        out.extend(by_region[r])
    return np.asarray(out, dtype=np.int64)
