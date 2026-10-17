"""
Peak-free enhancer–gene links by Bayesian evidence combination.

For an element ``e`` (any interval: a CRISPR element, a variant window, a tile or a
called peak) and a gene ``g`` whose TSS lies within ``max_distance``,

.. math::
    \\operatorname{logit} P(e \\to g) = \\underbrace{b_0 + f(\\log_{10} s_{eg})
        + \\beta_1 [\\text{nearest}] + \\beta_2 \\log r_{eg}}_{\\text{distance prior}}
        + \\sum_k \\beta_k x_k,

an additive (generalised-linear) combination of a log-prior and log-likelihood-ratio
terms, i.e. a discriminatively calibrated naive-Bayes evidence combination:

* **distance prior** — polymer contact ``P(s) ∝ s^-γ`` (fractal globule; Lieberman-Aiden
  et al. 2009, Mirny 2011) gives a log-prior linear in ``log s``; a natural cubic spline in
  ``log10 s`` absorbs loop-extrusion deviations (Fudenberg et al. 2016); ``r`` is the rank of
  ``g`` among protein-coding TSSs by distance to ``e`` (nearest-gene effect);
* **activity** — peak-free accessibility of ``e`` from fragments: ``n_e`` Tn5 insertions of the
  target cells in ``e``, background ``μ_e = N L_e / G``, Gamma–Poisson shrunk enrichment
  ``Â_e = (n_e + κ μ_e) / ((1 + κ) μ_e)``; feature ``log Â_e``;
* **ABC** (Fulco et al. 2019; Nasser et al. 2021) — ``A = max(Â - 1, 0)``,
  ``ABC_eg = A_e C(s_eg) / (A_e C(s_eg) + Σ_p A_p C(s_pg))`` over the called peaks within 5 Mb,
  ``C(s) = (max(s, 5 kb)/5 kb)^-1.024``; feature ``log(ABC + 1e-4)``;
* **promoter activity** of ``g`` (``log Â`` of TSS ± 500 bp; ENCODE-rE2G idea, Gschwind et al.
  2023);
* **covariation** (only with paired RNA) — partial correlation across metacells of the element's
  accessibility (insertions counted directly in ``e``) with the gene's expression, conditional on the
  top ``K`` principal components of the metacell embedding (closed-form residualisation);
  feature ``max(ρ, 0)``.

The coefficients are fitted on DEV CRISPRi data only (see ``E2G_MODELS``); ``model="full"``
uses covariation, ``model="atac"`` does not.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ..utils import console
from ._contact import (ABC_GAMMA, _genes_table, _glasso_kernels, _metacell_membership,
                       _peak_table, _threaded, powerlaw_contact)

CHROMS = tuple([f"chr{i}" for i in range(1, 23)] + ["chrX"])

#: DEV-fitted models: L2 logistic regression (C = 1) on K562 / HCT116 / WTC11 CRISPRi pairs
#: (ENCODE-rE2G benchmark tables; IGVF multiome pools igvf10 / igvf18 / igvf3).
E2G_MODELS: dict = {
 "full": {
  "features": [
   "nearest",
   "log_rank",
   "log_activity",
   "log_abc",
   "log_promoter",
   "rho_pos"
  ],
  "knots": [
   4.402431529041011,
   5.221121735295595,
   5.544577730955853,
   5.769937001869276,
   5.964063422445953
  ],
  "mean": [
   5.410744634918956,
   1.0411642458255561,
   0.1393475785151166,
   0.026910554163429138,
   0.06297551789077213,
   2.461806141253437,
   1.9772803350538906,
   -6.735180330700004,
   2.962015276477524,
   0.01381910451359288
  ],
  "scale": [
   0.5099629724979517,
   0.7977493544936409,
   0.1758474276176295,
   0.04563195055058387,
   0.24291892070641907,
   1.1306409429227056,
   0.9118981263238947,
   1.450794485884019,
   1.0093650299718269,
   0.04141121503677577
  ],
  "coef": [
   0.2132935996212031,
   0.45734218256169096,
   0.09003928405986618,
   -0.20365015319152213,
   0.07363419089669908,
   -0.7148952553436178,
   -0.6047690572911906,
   1.599959146933231,
   -0.16432549336544583,
   0.33649244896844716
  ],
  "intercept": -4.570675858077183,
  "C": 1.0,
  "n_confounders": 10,
  "n_train": 13275,
  "n_pos": 600,
  "trained_on": "DEV: K562 training + K562 new (igvf10), HCT116 (igvf18), WTC11 (igvf3); seed-1 features"
 },
 "atac": {
  "features": [
   "nearest",
   "log_rank",
   "log_activity",
   "log_abc"
  ],
  "knots": [
   4.402431529041011,
   5.221121735295595,
   5.544577730955853,
   5.769937001869276,
   5.964063422445953
  ],
  "mean": [
   5.410744634918956,
   1.0411642458255561,
   0.1393475785151166,
   0.026910554163429138,
   0.06297551789077213,
   2.461806141253437,
   1.9772803350538906,
   -6.735180330700004
  ],
  "scale": [
   0.5099629724979517,
   0.7977493544936409,
   0.1758474276176295,
   0.04563195055058387,
   0.24291892070641907,
   1.1306409429227056,
   0.9118981263238947,
   1.450794485884019
  ],
  "coef": [
   0.4627359278011705,
   0.2236546627848279,
   0.04231303176294659,
   -0.07459487013715708,
   0.08197902264536203,
   -0.7465358176374985,
   -0.7227178085742065,
   1.9206907940981526
  ],
  "intercept": -4.571971580281047,
  "C": 1.0,
  "n_confounders": 0,
  "n_train": 13275,
  "n_pos": 600,
  "trained_on": "DEV: K562 training + K562 new (igvf10), HCT116 (igvf18), WTC11 (igvf3); seed-1 features"
 }
}


# ---------------------------------------------------------------------------
# Peak-free insertion counting
# ---------------------------------------------------------------------------

def _atoms(elements: pd.DataFrame):
    """Disjoint atoms of possibly overlapping intervals, per chromosome.

    Returns ``{chrom: (bounds, atom_to_elem CSR (n_atoms × n_elements))}``."""
    out = {}
    n_el = len(elements)
    for ch, g in elements.groupby("chrom", sort=False):
        s = g["start"].to_numpy(np.int64); e = g["end"].to_numpy(np.int64)
        b = np.unique(np.r_[s, e])
        rows, cols = [], []
        for idx, a0, a1 in zip(g.index.to_numpy(), np.searchsorted(b, s), np.searchsorted(b, e)):
            rows.append(np.arange(a0, a1)); cols.append(np.full(a1 - a0, idx))
        rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(max(len(b) - 1, 0), n_el))
        out[ch] = (b, A)
    return out


def count_insertions(fragments, cell_names: Sequence[str], elements: pd.DataFrame,
                     n_jobs: Optional[int] = None):
    """Tn5 insertions (fragment start and end − 1) per cell in arbitrary intervals.

    Arguments:
        fragments: list of ``(fragment_file, barcode_map)``; ``barcode_map`` maps the file's
            barcodes to entries of ``cell_names`` (dict or Series), ``None`` = identity.
        cell_names: output rows.
        elements: DataFrame ``chrom, start, end`` (0-based, half-open; may overlap).

    Returns:
        ``(counts, totals)``: CSR ``(cells × elements)`` float32 insertion counts and the total
        insertions per cell on chr1–22, X.
    """
    from ..pp._fragment_io import BarcodeLookup, map_fragment_chunks, sorted_reduce
    cell_names = [str(c) for c in cell_names]
    pos = {c: i for i, c in enumerate(cell_names)}
    n_cells = len(cell_names)
    el = elements.reset_index(drop=True)
    atoms = _atoms(el)
    totals = np.zeros(n_cells, dtype=np.float64)
    mats = []
    for path, bmap in fragments:
        if bmap is None:
            keys, rows = cell_names, np.arange(n_cells)
        else:
            bm = pd.Series(bmap)
            bm = bm[bm.isin(pos)]
            keys, rows = list(map(str, bm.index)), np.array([pos[v] for v in bm.to_numpy()], dtype=np.int64)
        lookup = BarcodeLookup(keys)

        def per_chunk(ch):
            res = []
            if len(ch) == 0:
                return res
            loc = lookup(ch.barcode_names)[ch.barcode]
            brk = np.flatnonzero(ch.chrom[1:] != ch.chrom[:-1]) + 1
            tot = np.zeros(n_cells)
            for a, b in zip(np.r_[0, brk], np.r_[brk, len(ch)]):
                name = ch.chrom_names[ch.chrom[a]]
                if name not in CHROMS:
                    continue
                li = loc[a:b]
                m = li >= 0
                cell = rows[li[m]]
                p1 = ch.start[a:b][m]; p2 = ch.end[a:b][m] - 1
                np.add.at(tot, cell, 2.0)
                if name not in atoms:
                    continue
                bounds, A = atoms[name]
                if A.shape[0] == 0:
                    continue
                ps = np.concatenate([p1, p2]); cs = np.concatenate([cell, cell])
                k = np.searchsorted(bounds, ps, side="right") - 1
                ok = (k >= 0) & (k < A.shape[0])
                if not ok.any():
                    continue
                key = k[ok].astype(np.int64) * n_cells + cs[ok]
                kk, ww = sorted_reduce(key, np.ones(len(key), np.int64), int(A.shape[0]) * n_cells)
                res.append((name, kk, ww))
            res.append(("__tot__", tot, None))
            return res

        for runs in map_fragment_chunks(path, per_chunk, n_jobs=n_jobs):
            for name, kk, ww in runs:
                if name == "__tot__":
                    totals += kk
                    continue
                A = atoms[name][1]
                atom = kk // n_cells; cell = kk - atom * n_cells
                M = sp.csr_matrix((ww.astype(np.float32), (cell, atom)), shape=(n_cells, A.shape[0]))
                mats.append(M @ A)
    C = sp.csr_matrix((n_cells, len(el)), dtype=np.float32)
    for M in mats:
        C = C + M
    return C.tocsr(), totals


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _ns_basis(x: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    """Natural cubic spline basis (Hastie, Tibshirani & Friedman, ESL eq. 5.4–5.5) without the
    constant: columns ``x, N_3 … N_K``."""
    k = np.asarray(knots, dtype=np.float64)
    K = len(k)
    def d(j):
        return (np.maximum(x - k[j], 0) ** 3 - np.maximum(x - k[K - 1], 0) ** 3) / (k[K - 1] - k[j])
    cols = [x] + [d(j) - d(K - 2) for j in range(K - 2)]
    return np.column_stack(cols)


def _pairs(el: pd.DataFrame, genes: pd.DataFrame, max_distance: int):
    """(element, gene) pairs with |centre − TSS| ≤ max_distance (vectorised); rank of the gene
    among all genes by distance to the element (1 = nearest)."""
    ei, gi, dist, rank = [], [], [], []
    centres = ((el["start"] + el["end"]) // 2).to_numpy(np.int64)
    echrom = el["chrom"].to_numpy()
    for ch, g in genes.groupby("chrom", sort=False):
        idx = np.flatnonzero(echrom == ch)
        if len(idx) == 0:
            continue
        o = np.argsort(g["tss"].to_numpy(), kind="stable")
        tss = g["tss"].to_numpy(np.int64)[o]; grow = g.index.to_numpy()[o]
        c = centres[idx]
        lo = np.searchsorted(tss, c - max_distance, side="left")
        hi = np.searchsorted(tss, c + max_distance, side="right")
        n = hi - lo
        tot = int(n.sum())
        if tot == 0:
            continue
        e_rep = np.repeat(idx, n)
        cols = np.repeat(lo, n) + (np.arange(tot) - np.repeat(np.cumsum(n) - n, n))
        d = tss[cols] - np.repeat(c, n)
        order = np.lexsort((np.abs(d), e_rep))
        start = np.repeat(np.cumsum(n) - n, n)                     # group start in sorted order
        r = np.empty(tot, dtype=np.int64)
        r[order] = np.arange(tot) - start + 1
        ei.append(e_rep); gi.append(grow[cols]); dist.append(d); rank.append(r)
    if not ei:
        z = np.zeros(0, np.int64)
        return z, z, z, z
    return (np.concatenate(ei), np.concatenate(gi), np.concatenate(dist), np.concatenate(rank))


def e2g_features(
    adata: AnnData,
    *,
    gene_annotation,
    elements: Optional[pd.DataFrame] = None,
    fragments=None,
    rna: Optional[AnnData] = None,
    groupby: Optional[str] = None,
    target=None,
    max_distance: int = 1_000_000,
    corr_window: int = 250_000,
    n_confounders: int = 10,
    kappa: float = 1.0,
    use_rep: str = "X_iterative_lsi",
    n_metacells: int = 500,
    k_neighbors: int = 100,
    seed: int = 0,
    genes: Optional[Sequence[str]] = None,
    total_insertions: Optional[Union[str, np.ndarray]] = None,
    genome_size: Optional[float] = None,
    element_counts=None,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Feature table of :func:`e2g_links` (one row per element–gene pair).

    ``adata`` is cells × called peaks (counts); its peaks give the ABC denominator and, when
    ``elements`` is ``None``, the candidate elements (counts from ``adata.X``). With ``elements``,
    insertions are counted directly from ``fragments`` (list of ``(file, barcode_map)``), so the
    candidates do not depend on peak calls. ``total_insertions`` (obs key or array) gives each
    cell's genome-wide insertions for the background when counting from ``adata.X``; with
    fragments it is measured.
    """
    import time as _time
    t0 = _time.time()
    tick = (lambda m: console.info(f"[e2g] {m} ({_time.time() - t0:.1f}s)")) if verbose else (lambda m: None)
    gdf = _genes_table(gene_annotation)
    gdf = gdf[gdf["chrom"].isin(CHROMS)].reset_index(drop=True)
    peaks = _peak_table(adata.var_names)
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    if groupby is not None and target is not None:
        tg = [target] if isinstance(target, str) else list(target)
        tmask = adata.obs[groupby].astype(str).isin([str(t) for t in tg]).to_numpy()
    else:
        tmask = np.ones(adata.n_obs, dtype=bool)
    G = float(genome_size) if genome_size else 2.875e9          # chr1–22, X (hg38)
    # --- candidate elements and their per-cell counts ---
    tss_el = pd.DataFrame({"chrom": gdf["chrom"], "start": gdf["tss"] - 500, "end": gdf["tss"] + 500})
    if elements is None:
        el = peaks[["chrom", "start", "end"]].copy()
        if total_insertions is None:
            raise ValueError("counting from adata.X needs `total_insertions` (obs key or array)")
        tot = adata.obs[total_insertions].to_numpy(np.float64) if isinstance(total_insertions, str) \
            else np.asarray(total_insertions, np.float64)
        Ce = X
        # promoter windows: max over overlapping peaks (no fragments needed)
        Cp = None
    else:
        el = elements[["chrom", "start", "end"]].reset_index(drop=True).copy()
        both = pd.concat([el, tss_el], ignore_index=True)
        if element_counts is not None:            # cached (counts of [elements, TSS windows], totals)
            Call, tot = element_counts
        else:
            if fragments is None:
                raise ValueError("`elements` given: pass `fragments=[(file, barcode_map)]`")
            Call, tot = count_insertions(fragments, list(adata.obs_names), both, n_jobs=n_jobs)
        Ce = Call[:, :len(el)]; Cp = Call[:, len(el):]
    el = el.reset_index(drop=True)
    tick("counts")
    N = tot[tmask].sum()
    L_e = (el["end"] - el["start"]).to_numpy(np.float64)
    n_e = np.asarray(Ce[tmask].sum(axis=0)).ravel()
    mu_e = N * L_e / G
    A_hat = (n_e + kappa * mu_e) / ((1 + kappa) * mu_e)
    # peaks (ABC denominator)
    n_p = np.asarray(X[tmask].sum(axis=0)).ravel()
    mu_p = N * (peaks["end"] - peaks["start"]).to_numpy(np.float64) / G
    A_p = np.maximum((n_p + kappa * mu_p) / ((1 + kappa) * mu_p) - 1.0, 0.0)
    # promoter activity
    if Cp is not None:
        n_t = np.asarray(Cp[tmask].sum(axis=0)).ravel()
        mu_t = N * 1000.0 / G
        prom = (n_t + kappa * mu_t) / ((1 + kappa) * mu_t)
    else:
        prom = np.ones(len(gdf))
        pk_by = {c: d for c, d in peaks.groupby("chrom", sort=False)}
        Ap_full = (n_p + kappa * mu_p) / ((1 + kappa) * mu_p)
        for ch, g in gdf.groupby("chrom", sort=False):
            p = pk_by.get(ch)
            if p is None:
                continue
            p = p.sort_values("start")
            ps, pe, pi = p["start"].to_numpy(), p["end"].to_numpy(), p["peak_idx"].to_numpy()
            i0 = np.searchsorted(pe, g["tss"].to_numpy() - 500, side="right")
            i1 = np.searchsorted(ps, g["tss"].to_numpy() + 500, side="left")
            vals = [Ap_full[pi[a:b]].max() if b > a else 1.0 for a, b in zip(i0, i1)]
            prom[g.index.to_numpy()] = vals
    tick("activity")
    # --- pairs ---
    if genes is not None:
        keep_g = gdf["gene_name"].isin(set(genes)).to_numpy()
    else:
        keep_g = np.ones(len(gdf), bool)
    ei, gi, dist, rank = _pairs(el, gdf, max_distance)
    sel = keep_g[gi]
    ei, gi, dist, rank = ei[sel], gi[sel], dist[sel], rank[sel]
    s = np.abs(dist).astype(np.float64)
    # ABC with peak denominator + element numerator
    from ._contact import _abc_denominators
    den = _abc_denominators(peaks, gdf, A_p, ABC_GAMMA, 5000.0, 5_000_000)
    A_e = np.maximum(A_hat - 1.0, 0.0)
    num = A_e[ei] * powerlaw_contact(s, ABC_GAMMA, 5000.0)
    abc = num / (num + den[gi] + 1e-12)
    tick("pairs + ABC")
    # --- covariation ---
    rho = np.full(len(ei), np.nan)
    if rna is not None:
        if list(rna.obs_names) != list(adata.obs_names):
            raise ValueError("rna.obs_names must equal adata.obs_names")
        emb = np.asarray(adata.obsm[use_rep])
        M = _metacell_membership(emb, n_metacells, k_neighbors, seed)
        k = _glasso_kernels()
        from ._contact import _metacell_log_std
        R = rna.X.tocsr() if sp.issparse(rna.X) else sp.csr_matrix(rna.X)
        gcol = pd.Index(rna.var_names.astype(str)).get_indexer(gdf["gene_name"])
        n_mc = M.shape[0]
        # element accessibility: log CPM of metacell insertions with genome-wide totals
        Se = (M @ Ce).toarray() if sp.issparse(Ce) else M @ Ce
        mtot = M @ tot
        Ze = np.log1p(Se * (1e6 / np.maximum(mtot, 1.0))[:, None])
        Ze = (Ze - Ze.mean(0)) / np.where(Ze.std(0, ddof=1) > 0, Ze.std(0, ddof=1), np.inf)
        Zg_all = _metacell_log_std(R, M, 1e4, n_jobs=n_jobs)          # genes x mc
        Zg = np.zeros((len(gdf), n_mc)); ok = gcol >= 0
        Zg[ok] = Zg_all[gcol[ok]]
        ZT = np.vstack([Zg, Ze.T])
        if n_confounders > 0:
            E = np.asarray((M @ emb) / np.asarray(M.sum(axis=1)))
            E = E - E.mean(0)
            U, sv, _ = np.linalg.svd(E, full_matrices=False)
            Q, _ = np.linalg.qr(np.column_stack([np.ones(n_mc), U[:, :n_confounders]]))
            ZT = ZT - (ZT @ Q) @ Q.T
            sd = ZT.std(axis=1, ddof=1, keepdims=True)
            ZT = (ZT - ZT.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, np.inf)
        ZT = np.ascontiguousarray(np.nan_to_num(ZT))
        win = np.flatnonzero(s <= corr_window)
        pv = np.zeros(len(win))
        _threaded(k["pair_dot"], len(win), n_jobs, gi[win].astype(np.int64),
                  (len(gdf) + ei[win]).astype(np.int64), ZT, pv)
        rho[:] = 0.0
        rho[win] = pv
        tick("covariation")
    names = (el["chrom"] + ":" + el["start"].astype(str) + "-" + el["end"].astype(str)).to_numpy()
    out = pd.DataFrame({
        "element": names[ei], "chrom": el["chrom"].to_numpy()[ei],
        "start": el["start"].to_numpy()[ei], "end": el["end"].to_numpy()[ei],
        "gene": gdf["gene_name"].to_numpy()[gi], "tss": gdf["tss"].to_numpy()[gi],
        "distance": dist, "rank": rank,
        "log10_dist": np.log10(np.maximum(s, 1000.0)),
        "nearest": (rank == 1).astype(np.float64), "log_rank": np.log(rank),
        "log_activity": np.log(A_hat[ei]), "log_abc": np.log(abc + 1e-4),
        "log_promoter": np.log(prom[gi]), "rho": rho,
    })
    return out


FEATURE_SETS = {
    "prior": ["nearest", "log_rank"],
    "prior_act": ["nearest", "log_rank", "log_activity"],
    "prior_act_abc": ["nearest", "log_rank", "log_activity", "log_abc"],
    "prior_act_abc_prom": ["nearest", "log_rank", "log_activity", "log_abc", "log_promoter"],
    "full": ["nearest", "log_rank", "log_activity", "log_abc", "log_promoter", "rho_pos"],
}


def design_matrix(feat: pd.DataFrame, features: Sequence[str], knots: Sequence[float]) -> np.ndarray:
    """Spline(log10 distance) + the listed features (``rho_pos`` = max(ρ, 0), NaN → 0)."""
    cols = [_ns_basis(feat["log10_dist"].to_numpy(np.float64), knots)]
    for f in features:
        if f == "rho_pos":
            cols.append(np.maximum(np.nan_to_num(feat["rho"].to_numpy(np.float64)), 0.0)[:, None])
        else:
            cols.append(feat[f].to_numpy(np.float64)[:, None])
    return np.hstack(cols)


def e2g_score(feat: pd.DataFrame, model: dict) -> np.ndarray:
    """Posterior probability from a fitted model dict (``features, knots, mean, scale, coef,
    intercept``)."""
    Xd = design_matrix(feat, model["features"], model["knots"])
    Xd = (Xd - np.asarray(model["mean"])) / np.asarray(model["scale"])
    z = Xd @ np.asarray(model["coef"]) + float(model["intercept"])
    return 1.0 / (1.0 + np.exp(-z))


def e2g_links(adata: AnnData, *, gene_annotation, model: Union[str, dict] = "full", **kw) -> pd.DataFrame:
    """Enhancer–gene posterior probabilities (see module docstring).

    ``model``: ``"full"`` (needs ``rna=``), ``"atac"`` (no RNA) or a fitted model dict.
    All other keyword arguments go to :func:`e2g_features`. Returns the feature table with a
    ``score`` column (posterior probability).
    """
    m = E2G_MODELS[model] if isinstance(model, str) else model
    if "rho_pos" not in m["features"]:
        kw = dict(kw); kw["rna"] = None
    feat = e2g_features(adata, gene_annotation=gene_annotation, **kw)
    feat["score"] = e2g_score(feat, m)
    return feat
