"""Chromatin loops: calling (scHiCluster) and differential loops across cell groups (HCEA).

Loop calling runs scHiCluster's ``call_loop`` on the imputed ``.cool`` of every
cell (:func:`impute_cells`). It writes, per group, the pseudobulk matrices
``<group>.{Q,Q2,E,E2,T,T2}.cool`` (mean over cells of the imputed contacts Q,
their enrichment over the distance background E and over the local
background T, and the means of their squares) and the loops
``<group>.loop.bedpe`` / ``<group>.loop_summit.bedpe``.

Differential loops follow the Human Cell Epigenome Atlas (``fig4/07``):
a per-pixel one-way ANOVA across groups from those group means and mean
squares (:func:`loop_anova`), the union of the groups' loops
(:func:`merge_loops`), the ANOVA and group values at each loop
(:func:`loop_values`), and the selection of differential and group-specific
loops (:func:`differential_loops`, :func:`specific_loops`).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]
LOOP_COLUMNS = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]


# ---------------------------------------------------------------------------
# loop calling (scHiCluster)
# ---------------------------------------------------------------------------

def call_loops(adata, groupby: str, out_dir: PathLike, *, black_list: Optional[PathLike] = None,
               shuffle: bool = True, resolution: Optional[int] = None, dist: int = 5_050_000,
               cap: int = 5, pad: int = 5, gap: int = 2, min_cutoff: float = 1e-6,
               fdr_thres: float = 0.1, dist_thres: int = 20_000, size_thres: int = 1,
               cpu_per_job: int = 10, chunk_size: int = 200, **kwargs) -> Dict[str, Path]:
    """Call loops per cell group with scHiCluster (``hicluster loop``).

    Reads every cell's imputed ``.cool`` (:func:`impute_cells`, ``output='cool'``),
    writes the cell table (cell, cool, group) and runs
    :func:`schicluster.loop.snakemake.call_loop`: per-cell distance and local
    background normalisation, group sums, t-tests against both backgrounds,
    background filters and, with ``shuffle=True``, an FDR from shuffled
    cells (needs ``black_list``). Settings are scHiCluster's defaults.

    Returns:
        ``{group: Path('<out_dir>/<group>/<group>')}``, the prefix of the group's
        ``.{Q,Q2,E,E2,T,T2}.cool`` and ``.loop.bedpe`` / ``.loop_summit.bedpe``
        (the input of :func:`loop_anova` and :func:`merge_loops`).
    """
    from ._imputed_io import imputed_path
    from schicluster.loop.snakemake import call_loop

    info = adata.uns.get("hic", {})
    if "imputed_dir" not in info:
        raise ValueError("run impute_cells(output='cool') first")
    paths = [imputed_path(info["imputed_dir"], c) for c in adata.obs_names]
    if any(p.suffix != ".cool" for p in paths):
        raise ValueError("call_loops reads imputed .cool files; run impute_cells(output='cool')")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"cell_url": [str(p) for p in paths],
                          "cell_group": adata.obs[groupby].astype(str).to_numpy()},
                         index=adata.obs_names)
    table_path = out_dir / "cell_table.csv"
    table.to_csv(table_path, header=False)
    sizes_path = out_dir / "chrom_sizes.txt"
    import cooler
    cooler.Cooler(str(paths[0])).chromsizes.to_csv(sizes_path, sep="\t", header=False)
    res = int(resolution or info.get("resolution") or cooler.Cooler(str(paths[0])).binsize)
    call_loop(cell_table_path=str(table_path), output_dir=str(out_dir), chrom_size_path=str(sizes_path),
              shuffle=shuffle, chunk_size=chunk_size, dist=dist, cap=cap, pad=pad, gap=gap,
              resolution=res, min_cutoff=min_cutoff, cpu_per_job=cpu_per_job,
              black_list_path=None if black_list is None else str(black_list),
              fdr_thres=fdr_thres, dist_thres=dist_thres, size_thres=size_thres, **kwargs)
    return {g: out_dir / g / g for g in table["cell_group"].unique()}


# ---------------------------------------------------------------------------
# differential loops (HCEA)
# ---------------------------------------------------------------------------

def _group_prefixes(groups) -> Dict[str, list]:
    """``{name: prefix}`` or ``{name: [prefix, ...]}`` -> ``{name: [prefix, ...]}``."""
    return {str(k): [str(v)] if isinstance(v, (str, Path)) else [str(x) for x in v]
            for k, v in groups.items()}


def _anova_chrom(args):
    """One chromosome, one matrix: HCEA ``compute_anova`` (fig4/07)."""
    import cooler
    from scipy.sparse import csr_matrix, triu

    chrom, matrix, groups, n_bins, bkl, pad, min_d, max_d = args
    cov = np.zeros(n_bins)
    for xx, yy in bkl:
        cov[xx - pad:yy + pad] = 1   # as HCEA: a region within `pad` bins of the start masks nothing
    tot, last = 0, 0
    Esum, E2sum, Elast, E2last, ss_intra = [csr_matrix((n_bins, n_bins)) for _ in range(5)]
    for prefixes in groups:
        for p in prefixes:
            c1 = cooler.Cooler(f"{p}.{matrix}.cool")
            E = triu(c1.matrix(balance=False, sparse=True).fetch(chrom))
            E2 = triu(cooler.Cooler(f"{p}.{matrix}2.cool").matrix(balance=False, sparse=True).fetch(chrom))
            n = c1.info["group_n_cells"]
            Esum += E * n
            E2sum += E2 * n
            tot += n
        Egroup = Esum - Elast
        E2group = E2sum - E2last
        Egroup.data = Egroup.data ** 2 / (tot - last)
        ss_intra += (E2group - Egroup)
        Elast = Esum.copy()
        E2last = E2sum.copy()
        last = tot
    Esum.data = Esum.data ** 2 / tot
    ss_total = E2sum - Esum
    with np.errstate(divide="ignore", invalid="ignore"):
        ss_intra.data = 1 / ss_intra.data
        ss_total = ss_total.multiply(ss_intra)
        ss_total.data = (ss_total.data - 1) * (tot - len(groups)) / (len(groups) - 1)
    ss_total = ss_total.tocoo()
    d = ss_total.col - ss_total.row
    keep = (cov[ss_total.row] == 0) & (cov[ss_total.col] == 0) & (d > min_d) & (d < max_d)
    F = csr_matrix((ss_total.data[keep], (ss_total.row[keep], ss_total.col[keep])), (n_bins, n_bins))
    return chrom, matrix, F, tot


def loop_anova(groups: Mapping[str, Union[PathLike, Sequence[PathLike]]], out_dir: PathLike, *,
               matrices: str = "QET", chromosomes: Optional[Sequence[str]] = None,
               black_list: Optional[PathLike] = None, black_list_pad: int = 7,
               min_dist: int = 5, max_dist: int = 500, nproc: int = 1) -> Dict[str, Path]:
    """Per-pixel one-way ANOVA of loop matrices across groups (HCEA ``fig4/07``).

    For each pixel and each matrix (Q, E, T), the F statistic between groups,
    from every group's mean (``<prefix>.Q.cool``), mean of squares
    (``<prefix>.Q2.cool``) and cell count (``group_n_cells``):
    ``F = (SS_total / SS_within - 1) * (N - k) / (k - 1)``.

    Arguments:
        groups: ``{name: prefix}`` (``prefix.Q.cool`` etc., e.g. from
            :func:`call_loops`), or ``{name: [prefix, ...]}`` to pool several
            cell types into one group.
        out_dir: output folder; writes ``<out_dir>/{Q,E,T}pv.cool``.
        chromosomes: default: all chromosomes of the first group's cools.
        black_list: BED of regions to mask (HCEA:
            ``hg38_bismark_loop_blacklist.bed``); pixels within
            ``black_list_pad`` bins of a region are dropped.
        min_dist, max_dist: keep pixels with ``min_dist < j - i < max_dist`` bins
            (HCEA: 5 and 500 at 10 kb, i.e. 50 kb to 5 Mb).
        nproc: worker processes (one chromosome x matrix each).

    Returns:
        ``{matrix: Path}`` of the F-statistic cools; their metadata hold the
        groups, ``n_groups`` and ``n_cells`` (for :func:`differential_loops`).
    """
    import cooler

    from ._imputed_io import write_cool

    groups = _group_prefixes(groups)
    first = cooler.Cooler(f"{next(iter(groups.values()))[0]}.Q.cool")
    res = int(first.binsize)
    sizes = first.chromsizes
    chroms = list(chromosomes) if chromosomes is not None else list(sizes.index)
    n_bins = {c: int(first.extent(c)[1] - first.extent(c)[0]) for c in chroms}
    bkl = {c: np.zeros((0, 2), dtype=int) for c in chroms}
    if black_list is not None:
        b = pd.read_csv(black_list, sep="\t", header=None, comment="#", usecols=[0, 1, 2])
        for c in chroms:
            bkl[c] = b.loc[b[0] == c, [1, 2]].to_numpy() // res
    glist = list(groups.values())
    tasks = [(c, m, glist, n_bins[c], bkl[c], int(black_list_pad), int(min_dist), int(max_dist))
             for c in chroms for m in matrices]
    if nproc > 1:
        with ProcessPoolExecutor(int(nproc)) as ex:
            results = list(ex.map(_anova_chrom, tasks))
    else:
        results = [_anova_chrom(t) for t in tasks]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tot = results[0][3] if results else 0
    meta = {"groups": {k: v for k, v in groups.items()}, "n_groups": len(groups), "n_cells": int(tot),
            "generated_by": "epione.single.hic.loop_anova"}
    out = {}
    for m in matrices:
        mats = {c: F for c, mm, F, _ in results if mm == m}
        path = out_dir / f"{m}pv.cool"
        write_cool(path, mats, {c: int(sizes[c]) for c in chroms}, res, metadata=meta,
                   keep_zeros=True)
        out[m] = path
    return out


def merge_loops(loops: Mapping[str, Union[PathLike, pd.DataFrame]],
                chromosomes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Union of several groups' loop pixels (HCEA ``merged_loop``).

    Arguments:
        loops: ``{group: loop.bedpe}`` (or DataFrames with the six BEDPE columns).
        chromosomes: keep these, in this order (default: order of first appearance).

    Returns:
        One row per distinct pixel ``(chrom1, start1, start2)``, sorted by
        chromosome then positions, with the six BEDPE columns and a boolean
        column per group (``True`` if that group called the loop).
    """
    parts = []
    for g, src in loops.items():
        df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, sep="\t", header=None, comment="#")
        df = df.iloc[:, :6].copy()
        df.columns = LOOP_COLUMNS
        df["group"] = str(g)
        parts.append(df)
    allp = pd.concat(parts, ignore_index=True)
    called = pd.crosstab([allp["chrom1"], allp["start1"], allp["start2"]], allp["group"]) > 0
    merged = allp.drop(columns="group").drop_duplicates(subset=["chrom1", "start1", "start2"])
    merged = merged.sort_values(["chrom1", "start1", "start2"])
    order = list(chromosomes) if chromosomes is not None else list(dict.fromkeys(allp["chrom1"]))
    merged = pd.concat([merged[merged["chrom1"] == c] for c in order], ignore_index=True)
    key = pd.MultiIndex.from_frame(merged[["chrom1", "start1", "start2"]])
    called = called.reindex(key).reindex(columns=[str(g) for g in loops]).fillna(False).astype(bool)
    for g in called.columns:
        merged[g] = called[g].to_numpy()
    return merged


def _values_one(args):
    import cooler
    path, chroms, pos = args
    clr = cooler.Cooler(str(path))
    res = clr.binsize
    out = []
    for c in chroms:
        x, y = pos[c]
        mat = clr.matrix(balance=False, sparse=True).fetch(c).tocsr()
        out.append(np.asarray(mat[x // res, y // res]).ravel())
    return np.concatenate(out) if out else np.zeros(0)


def loop_values(loops: pd.DataFrame, cools: Mapping[str, PathLike], nproc: int = 1) -> pd.DataFrame:
    """Value of each cool at each loop pixel (HCEA ``load_Q``).

    Arguments:
        loops: from :func:`merge_loops` (``chrom1``, ``start1``, ``start2``),
            rows grouped by chromosome.
        cools: ``{column: cool}``, e.g. ``{t: f'{prefix}.Q.cool'}`` per group, or
            the F cools of :func:`loop_anova`.

    Returns:
        DataFrame indexed like ``loops`` with one column per cool.
    """
    chroms = list(dict.fromkeys(loops["chrom1"]))
    pos = {c: (loops.loc[loops["chrom1"] == c, "start1"].to_numpy(),
               loops.loc[loops["chrom1"] == c, "start2"].to_numpy()) for c in chroms}
    order = np.concatenate([np.flatnonzero((loops["chrom1"] == c).to_numpy()) for c in chroms])
    tasks = [(p, chroms, pos) for p in cools.values()]
    if nproc > 1:
        with ProcessPoolExecutor(int(nproc)) as ex:
            vals = list(ex.map(_values_one, tasks))
    else:
        vals = [_values_one(t) for t in tasks]
    out = pd.DataFrame(index=loops.index)
    for name, v in zip(cools, vals):
        col = np.empty(len(loops))
        col[order] = v
        out[name] = col
    return out


def differential_loops(loops: pd.DataFrame, anova: Mapping[str, PathLike], *,
                       n_groups: Optional[int] = None, n_cells: Optional[int] = None,
                       z_quantile: float = 0.15, fdr: Optional[float] = None,
                       min_dist: int = 100_000, max_dist: int = 4_950_000, edge_bins: int = 10,
                       chrom_sizes: Optional[Mapping[str, int]] = None, nproc: int = 1) -> pd.DataFrame:
    """ANOVA statistics at each loop and the HCEA selection of differential loops.

    Adds, per matrix in ``anova`` (from :func:`loop_anova`): ``<m>anova`` (F),
    ``<m>pval`` (F distribution with ``k - 1`` and ``N - k`` degrees of freedom)
    and ``<m>fdr`` (Benjamini-Hochberg over all loops). A loop is
    ``differential`` when the z-scores of ``log10 Qanova`` and ``log10 Tanova``
    over all loops both exceed ``norm.isf(z_quantile)`` (HCEA: 0.15), it lies
    ``min_dist``-``max_dist`` from the diagonal and ``edge_bins`` bins from the
    chromosome ends, and, if ``fdr`` is given, ``Qfdr``, ``Efdr`` and ``Tfdr``
    are all below it.

    ``n_groups`` / ``n_cells`` default to the values stored by :func:`loop_anova`.
    """
    import cooler
    from scipy.stats import f as fdist, norm, zscore
    from statsmodels.stats.multitest import multipletests

    first = cooler.Cooler(str(next(iter(anova.values()))))
    meta = first.info.get("metadata") or {}
    k = int(n_groups if n_groups is not None else meta["n_groups"])
    N = int(n_cells if n_cells is not None else meta["n_cells"])
    sizes = chrom_sizes if chrom_sizes is not None else first.chromsizes.to_dict()
    res = int(first.binsize)
    out = loops.copy()
    vals = loop_values(loops, {f"{m}anova": p for m, p in anova.items()}, nproc=nproc)
    for m in anova:
        F = vals[f"{m}anova"].to_numpy()
        out[f"{m}anova"] = F
        p = fdist.sf(F, k - 1, N - k)
        out[f"{m}pval"] = p
        ok = np.isfinite(p)
        q = np.full(len(p), np.nan)
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
        out[f"{m}fdr"] = q
    with np.errstate(divide="ignore", invalid="ignore"):
        thr = norm.isf(z_quantile)
        pos = (out["Qanova"] > 0) & (out["Tanova"] > 0)
        zq = pd.Series(np.nan, index=out.index)
        zt = pd.Series(np.nan, index=out.index)
        zq[pos] = zscore(np.log10(out.loc[pos, "Qanova"]))
        zt[pos] = zscore(np.log10(out.loc[pos, "Tanova"]))
    d = out["start2"] - out["start1"]
    end = out["chrom1"].map(sizes)
    keep = (zq > thr) & (zt > thr) & (d > min_dist) & (d < max_dist) \
        & (out["start1"] >= edge_bins * res) & (out["start2"] < end - edge_bins * res)
    if fdr is not None:
        keep &= (out["Qfdr"] < fdr) & (out["Efdr"] < fdr) & (out["Tfdr"] < fdr)
    out["differential"] = keep.fillna(False).to_numpy()
    return out


def specific_loops(q: pd.DataFrame, t: pd.DataFrame, *, groups: Optional[Mapping[str, Sequence[str]]] = None,
                   fc_q: float = 1.2, fc_t: float = 1.5) -> pd.DataFrame:
    """Group-specific loops by fold change (HCEA ``fig4/07``).

    A loop is specific to group ``g`` when its mean Q in ``g`` over the mean Q of
    all other columns is above ``fc_q`` and likewise for T above ``fc_t``.

    Arguments:
        q, t: loops x columns, from :func:`loop_values` on the groups' Q and T cools.
        groups: ``{name: [columns]}``; default one group per column.

    Returns:
        Boolean DataFrame, loops x groups.
    """
    groups = {c: [c] for c in q.columns} if groups is None else dict(groups)
    out = pd.DataFrame(index=q.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        for g, cols in groups.items():
            rest_q = q.loc[:, ~q.columns.isin(cols)].mean(axis=1)
            rest_t = t.loc[:, ~t.columns.isin(cols)].mean(axis=1)
            out[g] = ((q[cols].mean(axis=1) / rest_q) > fc_q) & ((t[cols].mean(axis=1) / rest_t) > fc_t)
    return out


def loop_apa(cool: PathLike, loops: Mapping[str, pd.DataFrame], *, window: int = 10,
             max_dist_bins: int = 505, chromosomes: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    """Aggregate peak analysis of loop sets on one cool (HCEA ``fig4/07``).

    Each chromosome's upper triangle is divided by its mean contact at each
    distance ``1 <= d < max_dist_bins`` (diagonal set to 1, farther pixels and
    the lower triangle to 0); around every loop a ``(2 window + 1)^2`` patch is
    min-max scaled to [0, 1]; patches are averaged per loop set. Works on the
    sparse matrix (same values as HCEA's dense computation).

    Returns:
        ``{name: (2 window + 1, 2 window + 1) array}``. The APA score of HCEA is
        ``a[window, window] / a[-5:, :5].mean()`` (centre over lower-left corner).
    """
    import cooler
    from scipy.sparse import triu

    clr = cooler.Cooler(str(cool))
    res = clr.binsize
    chroms = list(chromosomes) if chromosomes is not None else list(clr.chromnames)
    w = 2 * window + 1
    acc = {k: np.zeros((w, w)) for k in loops}
    offs = np.arange(-window, window + 1)
    for c in chroms:
        U = triu(clr.matrix(balance=False, sparse=True).fetch(c)).tocsr()
        n = U.shape[0]
        coo = U.tocoo()
        d = coo.col - coo.row
        keep = d < max_dist_bins
        sums = np.bincount(d[keep], weights=coo.data[keep], minlength=max_dist_bins)[:max_dist_bins]
        lengths = np.maximum(n - np.arange(max_dist_bins), 1)
        decay = sums / lengths                       # mean of each diagonal, zeros included
        for name, lp in loops.items():
            sub = lp.loc[lp["chrom1"] == c, ["start1", "start2"]].to_numpy() // res
            for x, y in sub:
                if x - window < 0 or y + window >= n:
                    continue
                rows, cols = x + offs, y + offs
                Q = U[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].toarray()
                dd = cols[None, :] - rows[:, None]
                with np.errstate(divide="ignore", invalid="ignore"):
                    E = np.where((dd >= 1) & (dd < max_dist_bins),
                                 Q / decay[np.clip(dd, 0, max_dist_bins - 1)], 0.0)
                E[dd == 0] = 1.0
                rng = E.max() - E.min()
                acc[name] += (E - E.min()) / rng if rng > 0 else 0
    return {k: acc[k] / max(len(loops[k]), 1) for k in loops}
