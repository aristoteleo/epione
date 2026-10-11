"""cisTarget motif enrichment on region sets (pycistarget ``cisTarget`` +
ctxcore ``recovery``/``aucs``/``leading_edge``).

For each region set:

1. Map the regions onto database regions (more than ``fraction_overlap`` of
   either interval).
2. Compute the AUC of every motif's recovery curve over the top
   ``auc_threshold`` fraction of the genome-wide ranking.
3. Compute ``NES = (AUC - mean) / std`` over all motifs and keep
   ``NES >= nes_threshold``.
4. For the enriched motifs, take the leading edge against the mean + 2 sd
   recovery curve of all motifs. This gives the motif hits.
5. Annotate motifs with TFs and build cistromes.

Reproduces upstream bit for bit, without the per-set dense recovery-curve
matrix (n_motifs × 5 % of the genome) that dominates upstream's time and
memory:

* With unit weights the ctxcore AUC sum is an integer
  (``m * cutoff - sum(ranks < cutoff)``), so the vectorised form gives the
  same float64 after the same division.
* The mean + 2 sd recovery curve and the leading edge come from one numba
  pass over a counting-sorted list of (rank, motif) events. Per rank it uses
  the same arithmetic as numpy's ``mean(axis=0)``/``std(axis=0)`` on the
  full matrix, so results match bit for bit, and no curve is stored.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numba
import numpy as np
import pandas as pd

from ._motif_annotation import cistromes_per_region_set, load_motif_annotations
from ._ranking_db import CisTargetDB, RegionLike, as_region_frame, frame_to_region_names, target_to_query

log = logging.getLogger("epione.scenic")

LOGO_URL = "https://motifcollections.aertslab.org/{version}/logos/"


@dataclass
class MotifEnrichmentResult:
    """Result of :func:`run_cistarget` or :func:`run_dem` for one region set.

    Attributes
    ----------
    name
        Region set / comparison name.
    kind
        ``"cisTarget"`` or ``"DEM"``.
    motif_enrichment
        One row per enriched motif (pycistarget column layout).
    motif_hits
        ``{"Database": {motif: [db regions]}, "Region_set": {motif: [regions]}}``.
    cistromes
        Same layout, keys ``TF_(Nr)`` / ``TF_extended_(Nr)``.
    regions_to_db
        ``Target``/``Query`` mapping of the input regions onto the database.
    params
        Parameters (written as hdf5 metadata).
    """

    name: str
    kind: str
    motif_enrichment: pd.DataFrame
    motif_hits: Dict[str, Dict[str, List[str]]]
    cistromes: Dict[str, Dict[str, List[str]]]
    regions_to_db: pd.DataFrame
    params: Dict = field(default_factory=dict)
    extra: Dict = field(default_factory=dict)

    def __repr__(self):
        return (f"MotifEnrichmentResult({self.kind} {self.name!r}: "
                f"{len(self.motif_enrichment)} motifs, "
                f"{len(self.cistromes.get('Region_set', {}))} cistromes)")


# ---------------------------------------------------------------------------
# core statistics
# ---------------------------------------------------------------------------

def _rank_cutoff(auc_threshold: float, total_regions: int) -> int:
    rank_cutoff = int(round(auc_threshold * total_regions))
    if not 0 < rank_cutoff <= total_regions - 1:
        raise ValueError(f"auc_threshold {auc_threshold} gives an empty AUC window")
    return rank_cutoff - 1


@numba.njit(parallel=True, cache=True)
def _auc_numerators(rankings, cutoff):
    n, m = rankings.shape
    out = np.zeros(m, dtype=np.int64)
    for j in numba.prange(m):
        acc = 0
        for i in range(n):
            r = rankings[i, j]
            if r < cutoff:
                acc += cutoff - r
        out[j] = acc
    return out


def recovery_aucs(rankings: np.ndarray, total_regions: int, auc_threshold: float) -> np.ndarray:
    """AUC per motif for unit-weight regions (ctxcore ``aucs``).

    ``rankings`` is ``(n_regions, n_motifs)`` with 0-based ranks. With unit
    weights ctxcore's ``sum(diff(x) * cumsum(w))`` is the integer
    ``sum(cutoff - r)`` over ranks below the cutoff, so the result is the same
    float64.
    """
    cutoff = _rank_cutoff(auc_threshold, total_regions)
    maxauc = float((cutoff + 1) * float(rankings.shape[0]))
    return _auc_numerators(np.ascontiguousarray(rankings), cutoff).astype(np.float64) / maxauc


@numba.njit(cache=True)
def _rank_events(rankings, rank_threshold):
    """Counting sort of the (rank, motif) pairs inside the recovery window:
    motifs with a region at rank k are ``ev[start[k]:start[k + 1]]``."""
    n, m = rankings.shape
    start = np.zeros(rank_threshold + 1, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            r = rankings[i, j]
            if r < rank_threshold:
                start[r + 1] += 1
    for k in range(rank_threshold):
        start[k + 1] += start[k]
    ev = np.empty(start[rank_threshold], dtype=np.int32)
    pos = start[:-1].copy()
    for i in range(n):
        for j in range(m):
            r = rankings[i, j]
            if r < rank_threshold:
                ev[pos[r]] = j
                pos[r] += 1
    return start, ev


@numba.njit(parallel=True, cache=True)
def _leading_edge_kernel(start, ev, n_motifs, rank_threshold, enriched, block):
    """Running ``argmax(rcc - (mean + 2 sd))`` for the enriched motifs.

    ``rcc[m, k]`` is the number of regions of motif ``m`` with rank ``<= k``.
    Per rank, the mean and sd over motifs use numpy's arithmetic for
    ``rccs.mean(axis=0)`` / ``rccs.std(axis=0)``: an exact integer sum
    divided by n_motifs, then a sequential motif-order sum of squared
    deviations. Rank blocks run in parallel from count snapshots.
    """
    nb = (rank_threshold + block - 1) // block
    E = enriched.shape[0]
    snaps = np.zeros((nb, n_motifs), dtype=np.int64)
    c = np.zeros(n_motifs, dtype=np.int64)
    for b in range(nb):
        snaps[b] = c
        k1 = min(rank_threshold, (b + 1) * block)
        for e in range(start[b * block], start[k1]):
            c[ev[e]] += 1
    best_val = np.full((nb, E), -np.inf)
    best_idx = np.zeros((nb, E), dtype=np.int64)
    for b in numba.prange(nb):
        cc = snaps[b].copy()
        S = 0
        for j in range(n_motifs):
            S += cc[j]
        k0 = b * block
        k1 = min(rank_threshold, k0 + block)
        for k in range(k0, k1):
            for e in range(start[k], start[k + 1]):
                cc[ev[e]] += 1
                S += 1
            mean = S / n_motifs
            acc = 0.0
            for j in range(n_motifs):
                d = cc[j] - mean
                acc += d * d
            a2 = mean + 2.0 * np.sqrt(acc / n_motifs)
            for t in range(E):
                v = cc[enriched[t]] - a2
                if v > best_val[b, t]:
                    best_val[b, t] = v
                    best_idx[b, t] = k
    out_val = np.full(E, -np.inf)
    out_idx = np.zeros(E, dtype=np.int64)
    for b in range(nb):
        for t in range(E):
            if best_val[b, t] > out_val[t]:
                out_val[t] = best_val[b, t]
                out_idx[t] = best_idx[b, t]
    return out_idx


def _leading_edges(rankings: np.ndarray, enriched: np.ndarray, rank_threshold: int,
                   block: int = 2048) -> np.ndarray:
    """Rank at max of ``rcc - (mean + 2 sd)`` for the enriched motifs
    (ctxcore ``recovery`` + ``leading_edge``)."""
    start, ev = _rank_events(np.ascontiguousarray(rankings), rank_threshold)
    return _leading_edge_kernel(start, ev, rankings.shape[1], rank_threshold,
                                np.asarray(enriched, dtype=np.int64), block)


def _query_to_targets(r2d: pd.DataFrame) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for t, q in zip(r2d["Target"].tolist(), r2d["Query"].tolist()):
        out.setdefault(q, set()).add(t)
    return out


def _annotate(me: pd.DataFrame, annot: Optional[pd.DataFrame], annotation_to_use: Sequence[str],
              version: str) -> pd.DataFrame:
    if annot is not None:
        me = me.merge(annot[list(annotation_to_use)], how="left", left_index=True,
                      right_index=True)
    url = LOGO_URL.format(version=version)
    me.insert(0, "Logo", [f'<img src="{url}{m}.png" width="200" >' for m in me.index])
    return me


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def _as_region_sets(region_sets) -> Dict[str, pd.DataFrame]:
    if isinstance(region_sets, Mapping):
        return {str(k): as_region_frame(v) for k, v in region_sets.items()}
    return {"region_set": as_region_frame(region_sets)}


def run_cistarget(
    region_sets: Union[Mapping[str, RegionLike], RegionLike],
    rankings_db: Union[CisTargetDB, str, os.PathLike],
    *,
    motif_annotation: Union[str, os.PathLike, pd.DataFrame, None] = None,
    species: str = "homo_sapiens",
    fraction_overlap: float = 0.4,
    auc_threshold: float = 0.005,
    nes_threshold: float = 3.0,
    rank_threshold: float = 0.05,
    annotation_version: str = "v10nr_clust",
    annotation_to_use: Sequence[str] = ("Direct_annot", "Orthology_annot"),
    motif_similarity_fdr: float = 0.001,
    orthologous_identity_threshold: float = 0.0,
    motifs_to_use: Optional[Sequence[str]] = None,
    keep_all_aucs: bool = True,
    verbose: bool = True,
) -> Dict[str, MotifEnrichmentResult]:
    """cisTarget motif enrichment for one or more region sets.

    Equivalent to pycistarget's ``cisTarget(...).run_ctx(cisTargetDatabase(...))``
    with the scenicplus defaults. The database is read once for the union
    of all sets.

    Parameters
    ----------
    region_sets
        ``{name: regions}`` (BED-like frames or ``chr:start-end`` lists), or
        a single region set.
    rankings_db
        Rankings database: a :class:`CisTargetDB`, or the path to a
        ``*.rankings.feather`` file.
    motif_annotation
        motif2TF table (path, or a frame from :func:`load_motif_annotations`).
        ``None`` downloads the snapshot for ``species``/``annotation_version``.
        ``False`` skips annotation.
    fraction_overlap, auc_threshold, nes_threshold, rank_threshold
        As in pycistarget (``rank_threshold`` is a fraction of the genome).
    keep_all_aucs
        Store the AUC of every motif in ``result.extra["aucs"]``.

    Returns
    -------
    ``{name: MotifEnrichmentResult}``. Region sets without any database overlap
    are skipped with a warning. Upstream raises for them and aborts the whole
    run.
    """
    t0 = time.perf_counter()
    db = rankings_db if isinstance(rankings_db, CisTargetDB) else CisTargetDB(rankings_db)
    if db.kind != "rankings":
        raise ValueError("run_cistarget needs a rankings database")
    sets = _as_region_sets(region_sets)
    index = db.region_index
    r2d: Dict[str, pd.DataFrame] = {}
    for name, df in sets.items():
        try:
            r2d[name] = target_to_query(df, index, fraction_overlap=fraction_overlap)
        except ValueError:
            log.warning("cisTarget: %s has no region in the database; skipped", name)
    union = set()
    for v in r2d.values():
        union.update(v["Query"])
    _say(verbose, f"reading {len(union):,} database regions × {len(db.motifs):,} motifs")
    db.preload(union)
    _say(verbose, f"database read in {time.perf_counter() - t0:.1f} s")

    annot = None
    if motif_annotation is not False:
        annot = motif_annotation if isinstance(motif_annotation, pd.DataFrame) else \
            load_motif_annotations(motif_annotation, species=species, version=annotation_version,
                                   motif_similarity_fdr=motif_similarity_fdr,
                                   orthologous_identity_threshold=orthologous_identity_threshold)
    motif_mask = None
    if motifs_to_use is not None:
        use = set(motifs_to_use)
        motif_mask = np.fromiter((m in use for m in db.motifs), dtype=bool, count=len(db.motifs))

    params = dict(species=species, auc_threshold=auc_threshold, nes_threshold=nes_threshold,
                  rank_threshold=rank_threshold, annotation_version=annotation_version,
                  annotation_to_use=list(annotation_to_use),
                  motif_similarity_fdr=motif_similarity_fdr,
                  orthologous_identity_threshold=orthologous_identity_threshold,
                  fraction_overlap=fraction_overlap)
    out = {}
    for name, mapping in r2d.items():
        res = _cistarget_one(name, mapping, db, annot, params, motif_mask, keep_all_aucs)
        out[name] = res
        _say(verbose, f"{name}: {len(mapping['Query'].unique()):,} db regions, "
                      f"{len(res.motif_enrichment)} enriched motifs")
    _say(verbose, f"cisTarget done in {time.perf_counter() - t0:.1f} s")
    return out


def _cistarget_one(name, mapping, db, annot, params, motif_mask, keep_all_aucs):
    regions = np.asarray(list(dict.fromkeys(mapping["Query"].tolist())), dtype=object)
    sorted_regions, vals = db.load(regions)
    motifs = db.motifs
    if motif_mask is not None:
        vals, motifs = vals[:, motif_mask], motifs[motif_mask]
    total = db.total_regions
    aucs = recovery_aucs(vals, total, params["auc_threshold"])
    ness = (aucs - aucs.mean()) / aucs.std()
    enriched = np.flatnonzero(ness >= params["nes_threshold"])
    extra = {"aucs": pd.Series(aucs, index=motifs)} if keep_all_aucs else {}
    if len(enriched) == 0:
        me = pd.DataFrame({c: [] for c in ["Logo", "Region_set", *params["annotation_to_use"],
                                            "NES", "AUC", "Rank_at_max", "Motif_hits"]})
        return MotifEnrichmentResult(name, "cisTarget", me, {"Database": {}, "Region_set": {}},
                                     {"Database": {}, "Region_set": {}}, mapping, params, extra)
    rt = int(params["rank_threshold"] * total)
    rank_at_max = _leading_edges(vals, enriched, rt)
    db_hits, rs_hits = {}, {}
    q2t = _query_to_targets(mapping)
    for k, m in enumerate(enriched):
        col = vals[:, m]
        sel = np.flatnonzero(col <= rank_at_max[k])
        sel = sel[np.argsort(col[sel], kind="stable")]
        hits = sorted_regions[sel].tolist()
        db_hits[motifs[m]] = hits
        t = set()
        for q in hits:
            t.update(q2t.get(q, ()))
        rs_hits[motifs[m]] = sorted(t)
    me = pd.DataFrame({"Region_set": name, "NES": ness[enriched], "AUC": aucs[enriched],
                       "Rank_at_max": rank_at_max},
                      index=pd.Index(motifs[enriched], name=None))
    me = me.sort_values("NES", ascending=False, kind="stable")
    me = _annotate(me, annot, params["annotation_to_use"], params["annotation_version"])
    me["Motif_hits"] = [len(db_hits[m]) for m in me.index]
    db_hits = {m: db_hits[m] for m in me.index}
    rs_hits = {m: rs_hits[m] for m in me.index}
    cistromes = {"Database": cistromes_per_region_set(me, db_hits, params["annotation_to_use"]),
                 "Region_set": cistromes_per_region_set(me, rs_hits, params["annotation_to_use"])}
    return MotifEnrichmentResult(name, "cisTarget", me, {"Database": db_hits, "Region_set": rs_hits},
                                 cistromes, mapping, params, extra)


def _say(verbose, msg):
    if verbose:
        print(f"  └─ [scenic] {msg}", flush=True)
