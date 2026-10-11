"""cisTarget-style scores / rankings databases from Epione's own MOODS scan.

The official cisTarget databases score every region with Cluster-Buster
CRM scores over ~1.8 M SCREEN regions. Epione can build the equivalent for
any region set from the genome-wide MOODS hit database written by
:func:`epione.tl.build_motif_database`:

* **score** of a (region, motif) pair: the best MOODS log-odds score of a
  hit overlapping the region (both strands), ``0`` when there is none;
* **ranking** per motif: regions ordered by score, highest first. Ties,
  including all the zero-score regions, are broken by a seeded random
  permutation, as ``create_cisTarget_databases`` does.

The resulting :class:`~epione.scenic.CisTargetDB` objects plug into
:func:`~epione.scenic.run_cistarget` and :func:`~epione.scenic.run_dem`
directly, and can be written to Feather for reuse.
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._ranking_db import CisTargetDB, RegionLike, as_region_frame, frame_to_region_names


def motif_score_db(
    regions: RegionLike,
    motif_database: Union[str, os.PathLike, None] = None,
    *,
    genome_fasta: Union[str, os.PathLike, None] = None,
    motifs=None,
    p_value: float = 1e-4,
    pseudocounts: float = 1e-4,
    background: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    n_jobs: int = 1,
    name: str = "epione_moods",
    verbose: bool = True,
) -> CisTargetDB:
    """Region × motif max-score database from MOODS hits.

    Parameters
    ----------
    regions
        Regions to score (BED-like frame or ``chr:start-end`` names), such as
        the consensus peaks.
    motif_database
        Directory written by :func:`epione.tl.build_motif_database`
        (a genome-wide hit database). When given, the stored hits are used and
        no scan is run.
    genome_fasta, motifs, p_value, pseudocounts, background
        Otherwise the regions are scanned directly with MOODS. ``motifs`` is a
        list of Biopython / JASPAR motifs (``None`` fetches JASPAR2024 CORE
        vertebrates). Hits need log-odds p-value ``< p_value``.
    n_jobs
        Chromosomes scanned in parallel (direct scan only).

    Returns
    -------
    :class:`CisTargetDB` with ``kind="scores"`` (float32), regions in input
    order.
    """
    df = as_region_frame(regions).reset_index(drop=True)
    names = frame_to_region_names(df)
    if motif_database is not None:
        out_dir = os.path.abspath(os.path.expanduser(os.fspath(motif_database)))
        meta = json.loads(pathlib.Path(out_dir, "_meta.json").read_text())
        motif_names = list(meta["motif_names"])
        S = _scores_from_hit_db(df, out_dir, len(motif_names), int(meta.get("max_motif_len", 35)),
                                verbose)
    else:
        if genome_fasta is None:
            raise ValueError("give motif_database or genome_fasta")
        from ..tl._motif_database import _load_jaspar_motifs, _motif_name
        motifs = list(_load_jaspar_motifs() if motifs is None else motifs)
        motif_names = [_motif_name(m, i) for i, m in enumerate(motifs)]
        S = _scan_regions(df, os.fspath(genome_fasta), motifs, p_value, pseudocounts,
                          list(background), n_jobs, verbose)
    return CisTargetDB.from_matrix(S, names, motif_names, kind="scores", name=name)


def _scores_from_hit_db(df, out_dir, n_motifs, max_len, verbose):
    S = np.zeros((len(df), n_motifs), dtype=np.float32)
    chrom = df["Chromosome"].to_numpy()
    for c in pd.unique(chrom):
        f = os.path.join(out_dir, f"{c}.parquet")
        if not os.path.exists(f):
            continue
        hits = pd.read_parquet(f, columns=["motif_idx", "start", "end", "score"])
        if hits.empty:
            continue
        hs = hits["start"].to_numpy()
        he = hits["end"].to_numpy()
        hm = hits["motif_idx"].to_numpy().astype(np.int64)
        sc = hits["score"].to_numpy().astype(np.float32)
        rows = np.flatnonzero(chrom == c)
        ps = df["Start"].to_numpy()[rows]
        pe = df["End"].to_numpy()[rows]
        lo = np.searchsorted(hs, ps - max_len, side="right")
        hi = np.searchsorted(hs, pe, side="left")
        cnt = np.maximum(hi - lo, 0)
        tot = int(cnt.sum())
        if tot == 0:
            continue
        rep = np.repeat(np.arange(len(rows)), cnt)
        idx = lo[rep] + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        keep = he[idx] > ps[rep]
        rep, idx = rep[keep], idx[keep]
        np.maximum.at(S, (rows[rep], hm[idx]), sc[idx])
        if verbose:
            print(f"  └─ [scenic] {c}: {len(rows):,} regions, {int(keep.sum()):,} motif hits",
                  flush=True)
    return S


def _scan_chrom(args):
    """Scan the regions of one chromosome; returns (row ids, (n, n_motifs) max scores)."""
    chrom, starts, ends, rows, fasta, state = args
    import pyfaidx
    from ..tl._motif_database import _build_scanner_and_thresholds
    scanner = _build_scanner_and_thresholds(state["motifs"], p_value=state["p_value"],
                                            pseudocounts=state["pseudocounts"], bg=state["bg"])
    n = len(state["motifs"])
    mlen = np.asarray(state["motif_len"], dtype=np.int64)
    fa = pyfaidx.Fasta(fasta, as_raw=True, sequence_always_upper=True)
    seq_c = fa[chrom]
    # one scan over the concatenated region sequences, N-spaced
    gap = "N" * int(mlen.max())
    parts, offs, pos = [], [], 0
    for s, e in zip(starts, ends):
        offs.append(pos)
        parts.append(str(seq_c[int(s):int(e)]))
        pos += int(e - s) + len(gap)
    seq = gap.join(parts)
    offs = np.asarray(offs, dtype=np.int64)
    lens = (ends - starts).astype(np.int64)
    S = np.zeros((len(starts), n), dtype=np.float32)
    res = scanner.scan(seq)
    for k in range(2 * n):
        hits = res[k]
        if not hits:
            continue
        p = np.fromiter((h.pos for h in hits), dtype=np.int64, count=len(hits))
        v = np.fromiter((h.score for h in hits), dtype=np.float32, count=len(hits))
        r = np.searchsorted(offs, p, side="right") - 1
        ok = (r >= 0) & (p + mlen[k % n] <= offs[r] + lens[r])
        np.maximum.at(S[:, k % n], r[ok], v[ok])
    return rows, S


def _scan_regions(df, fasta, motifs, p_value, pseudocounts, bg, n_jobs, verbose):
    state = {"motifs": motifs, "motif_len": [m.length for m in motifs], "p_value": p_value,
             "pseudocounts": pseudocounts, "bg": bg}
    chrom = df["Chromosome"].to_numpy()
    jobs = []
    for c in pd.unique(chrom):
        rows = np.flatnonzero(chrom == c)
        jobs.append((str(c), df["Start"].to_numpy()[rows], df["End"].to_numpy()[rows], rows,
                     fasta, state))
    S = np.zeros((len(df), len(motifs)), dtype=np.float32)

    def collect(results):
        for (c, *_), (rows, Sc) in zip(jobs, results):
            S[rows] = Sc
            if verbose:
                print(f"  └─ [scenic] {c}: {len(rows):,} regions scanned", flush=True)

    if n_jobs in (None, 0, 1):
        collect(map(_scan_chrom, jobs))
    else:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import cpu_count, get_context
        with ProcessPoolExecutor(max_workers=cpu_count() if n_jobs == -1 else int(n_jobs),
                                 mp_context=get_context("fork")) as ex:
            collect(ex.map(_scan_chrom, jobs))
    return S


def scores_to_rankings(scores_db: CisTargetDB, seed: int = 555, name: Optional[str] = None
                       ) -> CisTargetDB:
    """Per-motif 0-based rankings (highest score first, ties broken by a
    seeded random permutation of the regions)."""
    if scores_db.kind != "scores" or scores_db._matrix is None:
        raise ValueError("need an in-memory scores database (motif_score_db)")
    S = scores_db._matrix
    n, m = S.shape
    rng = np.random.RandomState(seed)
    R = np.empty((n, m), dtype=np.int32 if n >= 2 ** 15 else np.int16)
    ar = np.arange(n, dtype=np.int64)
    for j in range(m):
        tie = rng.permutation(n)
        order = np.lexsort((tie, -S[:, j]))
        R[order, j] = ar
    return CisTargetDB.from_matrix(R, scores_db.regions, scores_db.motifs, kind="rankings",
                                   name=name or scores_db.name.replace("scores", "rankings"))
