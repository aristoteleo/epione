"""Build and query a local genome-wide motif hit database.

`build_motif_database` runs one full-genome MOODS scan against a PWM
collection at a fixed p-value and stores the resulting hits as per-chromosome
parquet files. `query_motif_database` returns a sparse peak × motif boolean
matrix by doing ``O(log n)`` interval queries against those files —
typically ~2-5 s on ~100 k peaks once the database is built.

Replaces the per-call MOODS scan that pychromvar does inside
:func:`epione.tl.add_motif_matrix`. When the same genome + motif set + p-value
are reused across multiple datasets (which is the norm), this moves the
motif-annotation step from ~100 s to under 5 s.

Disk layout::

    <out_dir>/
      _meta.json        # {genome_hash, motif_names, p_value, ...}
      chr1.parquet      # columns: motif_idx (uint16), start, end (int32),
      chr2.parquet      # score (float32), strand (int8: +1/-1)
      ...

Per-chrom parquet is sorted by ``start`` so interval queries use
``np.searchsorted`` on the start column without a full index.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _console(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  └─ [motif_db] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Scanner / motif plumbing (shared with _motif_matrix wrapper)
# ---------------------------------------------------------------------------

def _load_jaspar_motifs(
    motif_db: str = "JASPAR2024",
    motif_collection: str = "CORE",
    motif_tax_group: Sequence[str] = ("vertebrates",),
):
    from pyjaspar import jaspardb
    jdb = jaspardb(release=motif_db)
    return jdb.fetch_motifs(collection=motif_collection,
                             tax_group=list(motif_tax_group))


def _build_scanner_and_thresholds(
    motifs,
    p_value: float,
    pseudocounts: float,
    bg: Sequence[float],
):
    import MOODS.scan
    import MOODS.tools
    n = len(motifs)
    matrices = [None] * (2 * n)
    thresholds = [None] * (2 * n)
    for i, m in enumerate(motifs):
        c = (tuple(m.counts["A"]), tuple(m.counts["C"]),
             tuple(m.counts["G"]), tuple(m.counts["T"]))
        matrices[i]       = MOODS.tools.log_odds(c, bg, pseudocounts)
        matrices[i + n]   = MOODS.tools.reverse_complement(matrices[i])
        thresholds[i]     = MOODS.tools.threshold_from_p(matrices[i], bg, p_value)
        thresholds[i + n] = thresholds[i]
    scanner = MOODS.scan.Scanner(7)
    scanner.set_motifs(matrices=matrices, bg=list(bg), thresholds=thresholds)
    return scanner


def _motif_name(m, i):
    return f"{getattr(m, 'matrix_id', i)}_{getattr(m, 'name', i)}"


def _genome_fingerprint(genome_fasta: str, chroms: Sequence[str]) -> str:
    """Hash chromosome names + lengths — identifies the reference build
    without hashing the full 3 GB FASTA."""
    import pyfaidx
    fa = pyfaidx.Fasta(genome_fasta, as_raw=True, sequence_always_upper=True)
    parts = []
    for c in chroms:
        if c in fa.keys():
            parts.append(f"{c}:{len(fa[c])}")
    h = hashlib.sha256(";".join(parts).encode()).hexdigest()
    return h[:16]


# ---------------------------------------------------------------------------
# Build database (one-time)
# ---------------------------------------------------------------------------

def _scan_chromosome(
    chrom: str,
    genome_fasta: str,
    motifs_state: Dict,
    out_dir: str,
    chunk_bp: int,
    verbose: bool,
) -> Tuple[str, int, float]:
    """Worker entry point: scan one full chromosome, write
    ``{out_dir}/{chrom}.parquet``. Uses ``motifs_state`` to rebuild a local
    MOODS scanner (Scanner objects don't pickle cleanly)."""
    import pyfaidx
    t0 = time.perf_counter()
    scanner = _build_scanner_and_thresholds(
        motifs_state["motifs"],
        p_value=motifs_state["p_value"],
        pseudocounts=motifs_state["pseudocounts"],
        bg=motifs_state["bg"],
    )
    n = len(motifs_state["motifs"])

    fa = pyfaidx.Fasta(genome_fasta, as_raw=True, sequence_always_upper=True)
    seq_all = str(fa[chrom][:])
    L = len(seq_all)

    # Segment to keep peak memory bounded. overlap = max motif length so no
    # near-boundary hits get lost.
    max_motif_len = int(motifs_state["max_motif_len"])
    overlap = max_motif_len - 1
    motif_idx_all = []
    start_all     = []
    end_all       = []
    score_all     = []
    strand_all    = []
    # motif_len array for end-from-start
    motif_len = np.asarray(motifs_state["motif_len"], dtype=np.int32)

    for seg_start in range(0, L, chunk_bp):
        seg_end = min(seg_start + chunk_bp + overlap, L)
        seg = seq_all[seg_start:seg_end]
        if len(seg) < max_motif_len:
            continue
        hits_by_mat = scanner.scan(seg)
        # hits_by_mat: list of 2n lists of Hit objects (pos, score)
        for mat_idx in range(2 * n):
            hits = hits_by_mat[mat_idx]
            if not hits:
                continue
            j = mat_idx % n
            strand = +1 if mat_idx < n else -1
            mlen = motif_len[j]
            # Convert hits → numpy arrays. Deduplicate boundary overlap by
            # requiring hit-start in the core region (not the carry-over
            # from previous segment's overlap).
            for h in hits:
                pos = h.pos
                if pos + mlen > len(seg):
                    continue
                global_start = seg_start + pos
                if seg_start > 0 and pos < overlap:
                    # Already emitted by the previous segment's overlap-free zone.
                    continue
                motif_idx_all.append(j)
                start_all.append(global_start)
                end_all.append(global_start + mlen)
                score_all.append(float(h.score))
                strand_all.append(strand)

    if not motif_idx_all:
        # Still write an empty parquet so querying doesn't need exist-checks.
        df = pd.DataFrame({
            "motif_idx": np.empty(0, dtype=np.uint16),
            "start": np.empty(0, dtype=np.int32),
            "end":   np.empty(0, dtype=np.int32),
            "score": np.empty(0, dtype=np.float32),
            "strand": np.empty(0, dtype=np.int8),
        })
    else:
        df = pd.DataFrame({
            "motif_idx": np.asarray(motif_idx_all, dtype=np.uint16),
            "start":     np.asarray(start_all,     dtype=np.int32),
            "end":       np.asarray(end_all,       dtype=np.int32),
            "score":     np.asarray(score_all,     dtype=np.float32),
            "strand":    np.asarray(strand_all,    dtype=np.int8),
        })
        df.sort_values("start", kind="mergesort", inplace=True)

    out_path = os.path.join(out_dir, f"{chrom}.parquet")
    df.to_parquet(out_path, compression="zstd")
    elapsed = time.perf_counter() - t0
    return chrom, int(df.shape[0]), elapsed


def build_motif_database(
    genome_fasta: str,
    out_dir: Union[str, "os.PathLike"],
    motifs=None,
    *,
    motif_db: str = "JASPAR2024",
    motif_collection: str = "CORE",
    motif_tax_group: Sequence[str] = ("vertebrates",),
    p_value: float = 5e-5,
    pseudocounts: float = 0.0001,
    background: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    chrom_filter: Optional[Sequence[str]] = None,
    chunk_bp: int = 10_000_000,
    n_jobs: int = -1,
    force: bool = False,
    verbose: bool = True,
) -> str:
    """Scan the genome once with MOODS and write a per-chromosome motif hit
    database for later reuse by :func:`epione.tl.add_motif_matrix`.

    Parameters
    ----------
    genome_fasta
        Path to the reference FASTA.
    out_dir
        Output directory. Will be created if missing. Expected to be stable
        across runs (e.g. ``~/.epione/hg19_jaspar2024_5e5``).
    motifs
        Iterable of Biopython / JASPAR ``motif`` objects. If ``None``, motifs
        are fetched from JASPAR via :mod:`pyjaspar` using
        ``motif_db`` / ``motif_collection`` / ``motif_tax_group``.
    p_value
        Per-motif log-odds p-value threshold (MOODS).
    chunk_bp
        Scan each chromosome in segments of this many bp. Keeps peak memory
        bounded (scanner emits one Python Hit object per hit, and at 5e-5
        density a single chromosome can produce tens of millions).
    n_jobs
        Number of chromosomes scanned in parallel. ``-1`` uses all CPUs.
        One worker per chromosome; each builds its own Scanner.
    force
        Re-build even if ``_meta.json`` says the directory is already
        populated with a matching configuration.

    Returns
    -------
    ``str`` — the absolute ``out_dir`` path.
    """
    out_dir = os.path.abspath(os.path.expanduser(os.fspath(out_dir)))
    os.makedirs(out_dir, exist_ok=True)

    if motifs is None:
        _console(
            f"fetching motifs: {motif_db}/{motif_collection}/{motif_tax_group}",
            verbose,
        )
        motifs = _load_jaspar_motifs(motif_db, motif_collection, motif_tax_group)
    motifs = list(motifs)
    motif_names = [_motif_name(m, i) for i, m in enumerate(motifs)]
    motif_len = [m.length for m in motifs]
    max_motif_len = int(max(motif_len))

    import pyfaidx
    fa = pyfaidx.Fasta(genome_fasta, as_raw=True, sequence_always_upper=True)
    all_chroms = list(fa.keys())
    if chrom_filter is not None:
        chroms = [c for c in chrom_filter if c in all_chroms]
    else:
        # Drop haplotypes / alt contigs by default: keep canonical chr1..22,X,Y,M.
        chroms = [c for c in all_chroms
                  if "_" not in c and len(c) <= 5]
    if not chroms:
        chroms = all_chroms

    genome_hash = _genome_fingerprint(genome_fasta, chroms)
    meta_path = os.path.join(out_dir, "_meta.json")
    config = {
        "genome_hash":     genome_hash,
        "motif_names":     motif_names,
        "p_value":         float(p_value),
        "pseudocounts":    float(pseudocounts),
        "background":      list(map(float, background)),
        "motif_db":        motif_db,
        "motif_collection": motif_collection,
        "motif_tax_group": list(motif_tax_group),
        "chroms":          chroms,
    }

    if os.path.exists(meta_path) and not force:
        try:
            existing = json.loads(pathlib.Path(meta_path).read_text())
            same = all(existing.get(k) == config[k]
                       for k in ("genome_hash", "p_value", "motif_names"))
            if same:
                _console(f"database already built at {out_dir} — skipping "
                         f"(pass force=True to rebuild)", verbose)
                return out_dir
        except Exception:
            pass

    _console(
        f"scanning {len(chroms)} chromosomes × {len(motifs)} motifs "
        f"× 2 strands @ p={p_value:.0e}",
        verbose,
    )

    motifs_state = {
        "motifs":       motifs,
        "motif_len":    motif_len,
        "max_motif_len": max_motif_len,
        "p_value":      p_value,
        "pseudocounts": pseudocounts,
        "bg":           list(background),
    }

    t0 = time.perf_counter()
    if n_jobs in (None, 0, 1):
        for c in chroms:
            chrom, n_hits, elapsed = _scan_chromosome(
                c, genome_fasta, motifs_state, out_dir, chunk_bp, verbose,
            )
            _console(f"{chrom}: {n_hits:,} hits ({elapsed:.1f} s)", verbose)
    else:
        from multiprocessing import get_context, cpu_count
        max_workers = cpu_count() if n_jobs == -1 else int(n_jobs)
        ctx = get_context("fork")
        with ProcessPoolExecutor(max_workers=max_workers,
                                 mp_context=ctx) as ex:
            futures = {
                ex.submit(_scan_chromosome, c, genome_fasta,
                          motifs_state, out_dir, chunk_bp, verbose): c
                for c in chroms
            }
            for fut in as_completed(futures):
                chrom, n_hits, elapsed = fut.result()
                _console(f"{chrom}: {n_hits:,} hits ({elapsed:.1f} s)",
                         verbose)

    config["build_time_s"] = float(time.perf_counter() - t0)
    config["max_motif_len"] = int(max_motif_len)
    pathlib.Path(meta_path).write_text(json.dumps(config, indent=1))
    _console(
        f"done — motif database at {out_dir} "
        f"(total {config['build_time_s']:.1f} s)",
        verbose,
    )
    return out_dir


# ---------------------------------------------------------------------------
# Query database → sparse peak × motif bool
# ---------------------------------------------------------------------------

def query_motif_database(
    out_dir: Union[str, "os.PathLike"],
    peaks: pd.DataFrame,
    *,
    min_score: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Return ``(n_peaks, n_motifs)`` sparse boolean matrix from a
    motif database.

    ``peaks`` must have columns ``chrom, start, end`` — order is preserved
    as output row order.

    Also returns the array of motif names (column order of the matrix).
    """
    out_dir = os.path.abspath(os.path.expanduser(os.fspath(out_dir)))
    meta = json.loads(pathlib.Path(out_dir, "_meta.json").read_text())
    motif_names = np.asarray(meta["motif_names"], dtype=object)
    max_motif_len = int(meta.get("max_motif_len", 35))
    n_motifs = len(motif_names)
    n_peaks = len(peaks)

    peaks = peaks.reset_index(drop=True)
    peaks["_row"] = np.arange(n_peaks, dtype=np.int64)

    rows_out: List[np.ndarray] = []
    cols_out: List[np.ndarray] = []
    for chrom, grp in peaks.groupby("chrom", sort=False):
        parquet = os.path.join(out_dir, f"{chrom}.parquet")
        if not os.path.exists(parquet):
            continue
        db = pd.read_parquet(parquet, columns=["motif_idx", "start", "end", "score"])
        if db.empty:
            continue
        db_start = db["start"].to_numpy()
        db_end   = db["end"].to_numpy()
        db_motif = db["motif_idx"].to_numpy()
        if min_score is not None:
            keep = db["score"].to_numpy() >= float(min_score)
            db_start, db_end, db_motif = db_start[keep], db_end[keep], db_motif[keep]

        peak_start = grp["start"].to_numpy()
        peak_end   = grp["end"].to_numpy()
        peak_row   = grp["_row"].to_numpy()

        # A hit overlaps [peak_start, peak_end] iff
        #     db_start < peak_end  AND  db_end > peak_start
        # Because every hit has length <= max_motif_len, we also have
        #     db_end > peak_start  =>  db_start > peak_start - max_motif_len
        # giving a TIGHT candidate range via two binary searches, with
        # O(#candidates) verification instead of O(n_hits) per peak.
        lo = np.searchsorted(db_start, peak_start - max_motif_len,
                              side="right")
        hi = np.searchsorted(db_start, peak_end, side="left")
        for i in range(len(peak_row)):
            a, b = lo[i], hi[i]
            if a >= b:
                continue
            ends = db_end[a:b]
            mask = ends > peak_start[i]
            if not mask.any():
                continue
            rows_out.append(np.full(int(mask.sum()), peak_row[i], dtype=np.int64))
            cols_out.append(db_motif[a:b][mask].astype(np.int64))

    if rows_out:
        rows_arr = np.concatenate(rows_out)
        cols_arr = np.concatenate(cols_out)
        data = np.ones(rows_arr.shape, dtype=np.bool_)
        M = sp.coo_matrix((data, (rows_arr, cols_arr)),
                           shape=(n_peaks, n_motifs))
        # A peak can hit the same motif multiple times — binarize.
        M = (M.tocsr() > 0).astype(np.bool_)
    else:
        M = sp.csr_matrix((n_peaks, n_motifs), dtype=np.bool_)

    if verbose:
        _console(
            f"queried {n_peaks:,} peaks × {n_motifs:,} motifs → "
            f"{M.nnz:,} hits",
        )
    return M, motif_names
