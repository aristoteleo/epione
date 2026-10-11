"""Differentially enriched motifs (pycistarget ``DEM``).

For a foreground set of regions against a background, run a per-motif
Wilcoxon rank-sum test on the cisTarget *scores* database, then apply BH
correction and a log2 fold change of the mean scores. Motif hits are the
foreground regions scoring above ``motif_hit_thr``, or above the Youden-optimal
ROC threshold when it is ``None``.

The background is sampled as upstream does:

* Promoter and non-promoter background regions are balanced to the
  foreground's promoter fraction, where a promoter is gene **Start** ±
  ``promoter_space``.
* Each list is shuffled with a fresh ``random.Random(seed)`` (CPython's
  Mersenne Twister) after sorting.

The statistics are written independently but use the same floating-point
operations as upstream's results: exact half-integer rank sums, float64 row
means accumulated in region order, and ``erf``-based normal tails. P-values,
fold changes and means are bit-identical on the SCENIC+ reference run.
"""
from __future__ import annotations

import logging
import math
import os
import random
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import pandas as pd

from ._cistarget import MotifEnrichmentResult, _annotate, _as_region_sets, _query_to_targets, _say
from ._motif_annotation import cistromes_per_region_set, load_motif_annotations
from ._ranking_db import (CisTargetDB, RegionIndex, RegionLike, as_region_frame,
                          frame_to_region_names, region_names_to_frame, target_to_query)

log = logging.getLogger("epione.scenic")


# ---------------------------------------------------------------------------
# numba kernels (same arithmetic as pycistarget.motif_enrichment_dem)
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def _mean_axis1(arr):
    """Row means as upstream's numba ``np.mean`` loop computes them: a
    sequential float64 accumulator over the row, divided by its length."""
    out = np.empty(arr.shape[0], dtype=np.float64)
    n = arr.shape[1]
    for i in numba.prange(arr.shape[0]):
        c = 0.0
        for j in range(n):
            c += arr[i, j]
        out[i] = c / n
    return out


@numba.njit(cache=True)
def _mean_axis0(arr):
    """Column means of a (regions × motifs) array with the same float64
    sequential accumulation order as :func:`_mean_axis1` on its transpose."""
    n, m = arr.shape
    acc = np.zeros(m, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            acc[j] += arr[i, j]
    return acc / n


def _sorted_rows_of_transpose(vals: np.ndarray, n_threads: int = 8, block: int = 256) -> np.ndarray:
    """``np.sort(vals.T, axis=1)`` with motif blocks sorted in parallel threads
    (numpy releases the GIL while sorting)."""
    from concurrent.futures import ThreadPoolExecutor
    n, m = vals.shape
    out = np.empty((m, n), dtype=vals.dtype)

    def work(j0):
        out[j0:j0 + block] = np.sort(vals[:, j0:j0 + block].T, axis=1)

    with ThreadPoolExecutor(max_workers=max(1, n_threads)) as ex:
        list(ex.map(work, range(0, m, block)))
    return out


@numba.njit(cache=True)
def _log2_fc(fg_mean, bg_mean):
    return np.log2((fg_mean + 10 ** -12) / (bg_mean + 10 ** -12))


@numba.njit(cache=True)
def _norm_sf(z):
    """Upper tail of the standard normal, via ``erf`` (libm)."""
    return (1.0 + math.erf(-z / math.sqrt(2.0))) / 2.0


@numba.njit(parallel=True, cache=True)
def _ranksums_sorted(Xs, Ys):
    """Same statistic from row-sorted inputs.

    The average-rank sum of ``x`` is ``N(N+1)/2`` minus that of ``y``. The
    rank of each ``y`` is ``#(< v) + (#(== v) + 1) / 2`` over both samples,
    found by binary search. All rank sums are exact half-integers, so ``s``,
    ``z`` and ``p`` equal upstream's to the last bit, with a sort per row
    (numpy, outside) in place of an argsort of the concatenation.
    """
    n = Xs.shape[0]
    n1 = Xs.shape[1]
    n2 = Ys.shape[1]
    z = np.zeros(n, dtype=np.float64)
    p = np.zeros(n, dtype=np.float64)
    for i in numba.prange(n):
        xs = Xs[i]
        ys = Ys[i]
        twice_ry = 0  # 2 * rank sum of y (an integer)
        for j in range(n2):
            v = ys[j]
            lx = np.searchsorted(xs, v, side="left")
            rx = np.searchsorted(xs, v, side="right")
            ly = np.searchsorted(ys, v, side="left")
            ry = np.searchsorted(ys, v, side="right")
            less = lx + ly
            eq = (rx - lx) + (ry - ly)
            twice_ry += 2 * less + eq + 1
        N = n1 + n2
        s = (N * (N + 1) - twice_ry) / 2.0
        expected = n1 * (n1 + n2 + 1) / 2.0
        zz = (s - expected) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        z[i] = zz
        p[i] = 2 * _norm_sf(abs(zz))
    return z, p


def p_adjust_bh(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values.

    Walks the p-values from largest to smallest, scaling each by
    ``n / rank`` and taking the running minimum (capped at 1). Ties receive
    the same adjusted value whatever their order.
    """
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    desc = np.argsort(p)[::-1]                      # largest first
    scale = float(n) / np.arange(n, 0, -1)          # n/n, n/(n-1), ..., n/1
    adj = np.minimum.accumulate(scale * p[desc])
    out = np.empty(n, dtype=np.float64)
    out[desc] = np.minimum(1, adj)
    return out


def _optimal_threshold_roc(fg: np.ndarray, bg: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fg = fg[fg > 0]
    bg = bg[bg > 0]
    scores = np.concatenate([fg, bg]).astype(float)
    labels = np.r_[np.ones(len(fg), dtype=int), np.zeros(len(bg), dtype=int)]
    fpr, tpr, thr = roc_curve(labels, scores)
    return thr[np.argmax(tpr - fpr)]


# ---------------------------------------------------------------------------
# foreground / background
# ---------------------------------------------------------------------------

def _overlaps_any(regions: pd.DataFrame, index: RegionIndex) -> np.ndarray:
    ti, _, ov = index.overlaps(regions)
    hit = np.zeros(len(regions), dtype=bool)
    hit[ti[ov > 0]] = True
    return hit


class _BackgroundSampler:
    """Region names and promoter flags computed once for a collection of
    region sets, so that repeated foreground/background draws (one per DEM
    comparison) only do set arithmetic and two shuffles."""

    def __init__(self, sets: Sequence[RegionLike], genome_annotation, promoter_space,
                 balance):
        self.names = [set(frame_to_region_names(as_region_frame(x))) for x in sets]
        self.is_prom = None
        if balance:
            if genome_annotation is None:
                raise ValueError("genome_annotation is required to balance promoters")
            prom = genome_annotation[["Chromosome", "Start"]].copy()
            prom["End"] = prom["Start"] + promoter_space
            prom["Start"] = prom["Start"] - promoter_space
            prom = prom[["Chromosome", "Start", "End"]].drop_duplicates()
            allr = sorted(set().union(*self.names))
            flags = _overlaps_any(region_names_to_frame(allr).reset_index(drop=True),
                                  RegionIndex(prom))
            self.is_prom = dict(zip(allr, flags.tolist()))

    def draw(self, fg_idx: Sequence[int], bg_idx: Sequence[int], max_bg_regions, seed):
        fg_names = set().union(*[self.names[i] for i in fg_idx])
        bg_names = set().union(*[self.names[i] for i in bg_idx]) - fg_names
        if max_bg_regions is not None:
            max_bg_regions = min(max_bg_regions, len(bg_names))
        fg_names = sorted(fg_names)
        bg_names = sorted(bg_names)
        if self.is_prom is not None:
            frac = sum(self.is_prom[r] for r in fg_names) / len(fg_names)
            n_allowed = int((max_bg_regions if max_bg_regions is not None else len(bg_names)) * frac)
            bg_prom = [r for r in bg_names if self.is_prom[r]]
            bg_noprom = [r for r in bg_names if not self.is_prom[r]]
            n_allowed = min(len(bg_prom), n_allowed)
            random.Random(seed).shuffle(bg_prom)
            bg_prom = bg_prom[:n_allowed]
            if max_bg_regions is not None:
                n_noprom = max_bg_regions - n_allowed
            else:
                n_noprom = len(bg_names) - n_allowed
            random.Random(seed).shuffle(bg_noprom)
            bg_names = [*bg_prom, *bg_noprom[:n_noprom]]
        elif max_bg_regions is not None:
            random.Random(seed).shuffle(bg_names)
        return (region_names_to_frame(fg_names).reset_index(drop=True),
                region_names_to_frame(bg_names).reset_index(drop=True))


def foreground_background_regions(
    foreground: Sequence[RegionLike],
    background: Sequence[RegionLike],
    *,
    max_bg_regions: Optional[int] = None,
    genome_annotation: Optional[pd.DataFrame] = None,
    balance_number_of_promoters: bool = True,
    promoter_space: int = 1_000,
    seed: int = 555,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Foreground and (sampled) background regions
    (pycistarget ``get_foreground_and_background_regions``).

    The background keeps upstream's order: sampled promoters first, then
    non-promoters. With ``balance_number_of_promoters=False`` upstream
    shuffles but never truncates (``background[0:max_bg_regions]`` is a
    no-op there), and that is kept.
    """
    fg, bg = list(foreground), list(background)
    smp = _BackgroundSampler(fg + bg, genome_annotation, promoter_space,
                             balance_number_of_promoters)
    return smp.draw(range(len(fg)), range(len(fg), len(fg) + len(bg)), max_bg_regions, seed)


# ---------------------------------------------------------------------------
# DEM
# ---------------------------------------------------------------------------

def dem_test(fg_scores: np.ndarray, bg_scores: np.ndarray, *, regions_by_motifs: bool = False,
             n_threads: Optional[int] = None) -> pd.DataFrame:
    """Per-motif Wilcoxon rank-sum, BH correction and log2FC of mean scores.

    ``fg_scores`` / ``bg_scores`` are ``(n_motifs, n_regions)`` score matrices
    (``(n_regions, n_motifs)`` with ``regions_by_motifs=True``). Returns
    ``pval``, ``Adjusted_pval``, ``Log2FC``, ``Mean_fg``, ``Mean_bg``, one row
    per motif in input order.
    """
    if n_threads is None:
        n_threads = numba.get_num_threads()
    if not regions_by_motifs:
        fg_scores, bg_scores = np.asarray(fg_scores).T, np.asarray(bg_scores).T
    fg = np.ascontiguousarray(fg_scores)
    bg = np.ascontiguousarray(bg_scores)
    _, p = _ranksums_sorted(_sorted_rows_of_transpose(fg, n_threads),
                            _sorted_rows_of_transpose(bg, n_threads))
    mfg = _mean_axis0(fg)
    mbg = _mean_axis0(bg)
    return pd.DataFrame({"pval": p, "Adjusted_pval": p_adjust_bh(p),
                         "Log2FC": _log2_fc(mfg, mbg), "Mean_fg": mfg, "Mean_bg": mbg})


def run_dem(
    region_sets: Mapping[str, Mapping[str, RegionLike]],
    scores_db: Union[CisTargetDB, str, os.PathLike],
    *,
    genome_annotation: Optional[pd.DataFrame] = None,
    motif_annotation: Union[str, os.PathLike, pd.DataFrame, None] = None,
    species: str = "homo_sapiens",
    max_bg_regions: Optional[int] = 500,
    balance_number_of_promoters: bool = True,
    promoter_space: int = 1_000,
    seed: int = 666,
    fraction_overlap: float = 0.4,
    adjpval_thr: float = 0.05,
    log2fc_thr: float = 1.0,
    mean_fg_thr: float = 0.0,
    motif_hit_thr: Optional[float] = 3.0,
    annotation_version: str = "v10nr_clust",
    annotation_to_use: Sequence[str] = ("Direct_annot", "Orthology_annot"),
    motif_similarity_fdr: float = 0.001,
    orthologous_identity_threshold: float = 0.0,
    keep_all_pvals: bool = True,
    verbose: bool = True,
) -> Dict[str, MotifEnrichmentResult]:
    """DEM for every region set against the other sets of its group
    (the scenicplus ``dem`` step).

    Parameters
    ----------
    region_sets
        ``{group: {set_name: regions}}``, e.g.
        ``{"Topics_otsu": {...}, "DARs_cell_type": {...}}``. Each set is tested
        against the union of the other sets of the same group. Results are
        named ``"{group}_{set}_vs_all"``.
    scores_db
        Scores database (:class:`CisTargetDB` or path to ``*.scores.feather``).
    genome_annotation
        Gene annotation with ``Chromosome``/``Start`` columns, used for promoter
        balancing (scenicplus' ``genome_annotation.tsv``).
    max_bg_regions, balance_number_of_promoters, promoter_space, seed
        Background sampling (scenicplus defaults).
    adjpval_thr, log2fc_thr, mean_fg_thr, motif_hit_thr
        Enrichment thresholds and the score threshold for motif hits.
    keep_all_pvals
        Keep every motif's p-value / adjusted p-value in
        ``result.extra["pvals"]``.

    Notes
    -----
    Score columns for every database region that overlaps any set are read
    once. Comparisons upstream would crash on are skipped with a warning:
    a foreground with no database overlap, or an empty background.
    """
    t0 = time.perf_counter()
    db = scores_db if isinstance(scores_db, CisTargetDB) else CisTargetDB(scores_db)
    if db.kind != "scores":
        raise ValueError("run_dem needs a scores database")
    groups = {g: _as_region_sets(s) for g, s in region_sets.items()}
    index = db.region_index
    union = set()
    for sets in groups.values():
        for df in sets.values():
            try:
                union.update(target_to_query(df, index, fraction_overlap)["Query"])
            except ValueError:
                pass
    _say(verbose, f"reading {len(union):,} database regions × {len(db.motifs):,} motifs")
    db.preload(union)
    _say(verbose, f"database read in {time.perf_counter() - t0:.1f} s")
    annot = None
    if motif_annotation is not False:
        annot = motif_annotation if isinstance(motif_annotation, pd.DataFrame) else \
            load_motif_annotations(motif_annotation, species=species, version=annotation_version,
                                   motif_similarity_fdr=motif_similarity_fdr,
                                   orthologous_identity_threshold=orthologous_identity_threshold)
    params = dict(species=species, adjpval_thr=adjpval_thr, log2fc_thr=log2fc_thr,
                  mean_fg_thr=mean_fg_thr, motif_hit_thr=motif_hit_thr,
                  annotation_version=annotation_version, annotation_to_use=list(annotation_to_use),
                  motif_similarity_fdr=motif_similarity_fdr,
                  orthologous_identity_threshold=orthologous_identity_threshold,
                  max_bg_regions=max_bg_regions, seed=seed, promoter_space=promoter_space,
                  balance_number_of_promoters=balance_number_of_promoters,
                  fraction_overlap=fraction_overlap)
    out = {}
    for g, sets in groups.items():
        keys = list(sets)
        sampler = _BackgroundSampler([sets[k] for k in keys], genome_annotation, promoter_space,
                                     balance_number_of_promoters)
        for i, fgk in enumerate(keys):
            name = f"{g}_{fgk}_vs_all"
            try:
                target_to_query(sets[fgk], index, fraction_overlap)
                fg_r, bg_r = sampler.draw([i], [j for j in range(len(keys)) if j != i],
                                          max_bg_regions, seed)
                res = _dem_one(name, fg_r, bg_r, db, index, annot, params, keep_all_pvals)
            except ValueError as e:
                log.warning("DEM: %s skipped (%s)", name, str(e).strip().splitlines()[0])
                continue
            out[name] = res
            _say(verbose, f"{name}: {len(res.motif_enrichment)} enriched motifs")
    _say(verbose, f"DEM done in {time.perf_counter() - t0:.1f} s")
    return out


def _dem_one(name, fg_r, bg_r, db, index, annot, params, keep_all):
    fq = target_to_query(fg_r, index, params["fraction_overlap"])
    bq = target_to_query(bg_r, index, params["fraction_overlap"])
    r2d = pd.concat([fq, bq]).drop_duplicates().reset_index(drop=True)
    fg_regions, fg_vals = db.load(fq["Query"])
    _, bg_vals = db.load(bq["Query"])
    stats = dem_test(fg_vals, bg_vals, regions_by_motifs=True)
    stats.index = db.motifs
    res = stats[["Log2FC", "Adjusted_pval", "Mean_fg", "Mean_bg"]]
    res = res.loc[(res["Adjusted_pval"] <= params["adjpval_thr"])
                  & (res["Log2FC"] >= params["log2fc_thr"])
                  & (res["Mean_fg"] >= params["mean_fg_thr"])]
    res = res.sort_values(["Log2FC", "Adjusted_pval"], ascending=[False, True])
    me = _annotate(res.copy(), annot, params["annotation_to_use"], params["annotation_version"])
    motif_pos = pd.Index(db.motifs).get_indexer(me.index)
    db_hits, thrs = {}, []
    for m, i in zip(me.index, motif_pos):
        thr = params["motif_hit_thr"]
        if thr is None:
            thr = _optimal_threshold_roc(fg_vals[:, i], bg_vals[:, i])
        db_hits[m] = fg_regions[fg_vals[:, i] > thr].tolist()
        thrs.append(thr)
    me["Motif_hit_thr"] = thrs
    me["Motif_hits"] = [len(db_hits[m]) for m in me.index]
    q2t = _query_to_targets(r2d)
    rs_hits = {}
    for m, hits in db_hits.items():
        t = set()
        for q in hits:
            t.update(q2t.get(q, ()))
        rs_hits[m] = sorted(t)
    cistromes = {"Database": cistromes_per_region_set(me, db_hits, params["annotation_to_use"]),
                 "Region_set": cistromes_per_region_set(me, rs_hits, params["annotation_to_use"])}
    extra = {"foreground_regions": fg_r, "background_regions": bg_r}
    if keep_all:
        extra["pvals"] = stats[["pval", "Adjusted_pval"]]
    return MotifEnrichmentResult(name, "DEM", me, {"Database": db_hits, "Region_set": rs_hits},
                                 cistromes, r2d, params, extra)
