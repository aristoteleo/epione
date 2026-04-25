"""Motif-enrichment helpers for the bulk pipeline.

Two entry points:

- :func:`run_homer_motifs` — thin wrapper around HOMER's ``findMotifsGenome.pl``
  Perl pipeline. Idempotent (skips when ``knownResults.txt`` exists).
- :func:`find_motifs_genome` — pure-Python reimplementation of HOMER's
  *known*-motif enrichment (peak → GC-matched background → PWM scan →
  binomial test). No Perl dependency. Reads HOMER's ``known.motifs``
  library format, so the output schema matches ``knownResults.txt`` and
  pairs directly with :func:`epione.pl.homer_motif_table`.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import binom

from ._env import build_env, ensure_dir, run_cmd


def _to_homer_peak_file(
    peaks: Union[pd.DataFrame, str, Path],
    out_file: Path,
    *,
    strand_default: str = "+",
) -> Path:
    """Write a HOMER 5-column peak file (``id chrom start end strand``).

    ``peaks`` may be either a DataFrame with ``chrom,start,end`` columns
    (``name`` / ``strand`` optional) or a path to an existing BED / TSV.
    Rows with NaN in ``chrom/start/end`` are dropped.
    """
    if isinstance(peaks, (str, Path)):
        df = pd.read_csv(
            peaks, sep="\t", header=None,
            names=["chrom", "start", "end", "name", "score"],
            usecols=[0, 1, 2, 3, 4],
        )
    else:
        df = peaks.copy()

    df = df.dropna(subset=["chrom", "start", "end"]).reset_index(drop=True)
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)

    if "strand" not in df.columns:
        df["strand"] = strand_default
    if "name" not in df.columns or df["name"].isna().all():
        df["name"] = [f"peak_{i}" for i in range(len(df))]

    df[["name", "chrom", "start", "end", "strand"]].to_csv(
        out_file, sep="\t", index=False, header=False,
    )
    return out_file


def run_homer_motifs(
    peaks: Union[pd.DataFrame, str, Path],
    genome: Union[str, Path],
    outdir: Union[str, Path],
    *,
    size: int = 200,
    length: str = "8,10,12",
    mask: bool = True,
    threads: int = 8,
    preparsed_dir: Optional[Union[str, Path]] = None,
    extra_args: Optional[Sequence[str]] = None,
    homer_bin_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    quiet: bool = False,
) -> Path:
    """Run HOMER ``findMotifsGenome.pl`` on a peak set.

    Arguments:
        peaks: either a DataFrame (must contain ``chrom,start,end``; a
            ``strand`` column is optional) or a path to a tab-separated
            file (BED or HOMER-format) that already has those columns.
        genome: path to the genome FASTA (e.g. ``/.../hg19.fa``). HOMER
            also accepts a genome alias like ``hg19`` when its config is
            configured, but passing a local FASTA is the most portable.
        outdir: directory that HOMER will write into (``knownResults.txt``,
            ``knownResults/known<i>.motif``, ``homerResults.*``…).
        size: window size (bp) centered on each peak; paper default 200.
        length: comma-separated motif lengths to search for de-novo
            discovery; same flag as HOMER's ``-len``.
        mask: when True, pass ``-mask`` to mask repeats (paper default).
        threads: threads to give HOMER via ``-p``.
        preparsed_dir: HOMER's GC-matched background cache dir. If None,
            defaults to ``<outdir>/preparsed`` so the cache is local.
        extra_args: any additional raw flags to append (e.g.
            ``('-h',)`` to switch from binomial to hypergeometric).
        homer_bin_dir: if HOMER isn't on ``PATH``, pass the directory
            containing ``findMotifsGenome.pl`` here. It is prepended to
            the subprocess PATH for the call.
        overwrite: rerun HOMER even if ``knownResults.txt`` exists.
        quiet: swallow HOMER's stdout (keeps stderr).

    Returns:
        ``Path`` to ``outdir``. If HOMER was skipped (cached result), the
        same path is returned.

    Example:
        >>> import epione as epi, pandas as pd
        >>> peaks = pd.read_csv('peaks.bed', sep='\t', header=None,
        ...                     names=['chrom','start','end'])
        >>> outdir = epi.bulk.run_homer_motifs(
        ...     peaks,
        ...     genome='/ref/hg19.fa',
        ...     outdir='/tmp/homer_out',
        ...     size=200, length='8,10,12', mask=True, threads=8)
        >>> fig, axes, top = epi.pl.homer_motif_table(outdir, top_n=5)
    """
    outdir = Path(outdir)
    ensure_dir(outdir)
    known = outdir / "knownResults.txt"
    if known.exists() and not overwrite:
        if not quiet:
            print(f"[run_homer_motifs] {known} exists — skipping HOMER run "
                  f"(pass overwrite=True to rerun)")
        return outdir

    # Resolve HOMER binary.
    env = build_env()
    if homer_bin_dir is not None:
        env["PATH"] = f"{Path(homer_bin_dir)}:{env.get('PATH', '')}"
    if not shutil.which("findMotifsGenome.pl", path=env.get("PATH")):
        raise RuntimeError(
            "findMotifsGenome.pl not found on PATH. Install HOMER with "
            "`mamba install -c bioconda -y homer` (and configure genomes "
            "with `configureHomer.pl`), or pass homer_bin_dir=<path>."
        )

    # Materialise peak input.
    peak_txt = outdir / "_peaks_homer.txt"
    _to_homer_peak_file(peaks, peak_txt)

    # Preparsed cache dir — HOMER's GC-matched background cache. Per-genome
    # and expensive to rebuild (minutes on first call, cached thereafter).
    pre = Path(preparsed_dir) if preparsed_dir else outdir / "preparsed"
    ensure_dir(pre)

    cmd: list[str] = [
        "findMotifsGenome.pl",
        str(peak_txt),
        str(genome),
        str(outdir),
        "-size", str(size),
        "-len",  str(length),
        "-p",    str(threads),
        "-preparsedDir", str(pre),
    ]
    if mask:
        cmd.append("-mask")
    if extra_args:
        cmd.extend(map(str, extra_args))

    if not quiet:
        print(f"[run_homer_motifs] {' '.join(cmd)}")

    if quiet:
        subprocess.run(cmd, env=env, check=True,
                       stdout=subprocess.DEVNULL)
    else:
        run_cmd(cmd, env=env)

    return outdir


# ======================================================================
# Pure-Python HOMER known-motif enrichment
# ======================================================================
#
# Re-implements the *known* branch of ``findMotifsGenome.pl`` so callers
# can run motif enrichment without a Perl dependency. De-novo motif
# discovery (``homerResults.txt``) is out of scope — keep using HOMER or
# MEME-STREME for that.
#
# Algorithm summary (matches HOMER Methods):
#   1. Extract ``size``-bp windows centred on each peak from a FASTA.
#   2. Sample random genomic windows of the same size, then subsample to
#      match the target GC-content distribution in fixed-width bins.
#   3. For each PWM in HOMER's ``known.motifs`` library (with its
#      pre-calibrated log-odds threshold), scan both strands and mark
#      whether each sequence contains ≥ 1 hit.
#   4. Compute per-motif enrichment with a right-tailed binomial test
#      using the background hit rate as the null.

# Base → int lookup; anything outside ACGT (e.g. N) becomes 4 and is
# treated as missing during scans (contributes -inf log-odds → can't hit).
_BASE_TO_INT = np.full(256, 4, dtype=np.int8)
for _i, _b in enumerate(b"ACGT"):
    _BASE_TO_INT[_b] = _i
    _BASE_TO_INT[_b + 32] = _i  # lowercase


def _encode_sequences(seqs: Sequence[str], length: int) -> np.ndarray:
    """Encode a list of equal-length strings to a (N, length) int8 array."""
    arr = np.zeros((len(seqs), length), dtype=np.int8)
    for i, s in enumerate(seqs):
        if len(s) != length:
            raise ValueError(f"sequence {i} length {len(s)} != {length}")
        arr[i] = np.frombuffer(s.encode("ascii"), dtype=np.uint8).view(np.int8)
    return _BASE_TO_INT[arr.view(np.uint8)]


@dataclass
class HomerMotif:
    """One motif loaded from a HOMER ``known.motifs`` library."""
    consensus: str
    name: str
    threshold: float              # log-odds sum; HOMER-calibrated
    pwm: np.ndarray               # shape (L, 4), rows are positions
    extra: str = ""               # trailing header fields (p-val, stats)


def _load_homer_motif_library(
    path: Union[str, Path],
) -> List[HomerMotif]:
    """Parse HOMER's ``known.motifs`` text format into a list of motifs."""
    motifs: List[HomerMotif] = []
    rows: List[List[float]] = []
    header: Optional[List[str]] = None
    with open(path) as fh:
        for ln in fh:
            if ln.startswith(">"):
                if header is not None and rows:
                    motifs.append(HomerMotif(
                        consensus=header[0][1:],
                        name=header[1] if len(header) > 1 else "?",
                        threshold=float(header[2]) if len(header) > 2 else 0.0,
                        pwm=np.asarray(rows, dtype=np.float32),
                        extra="\t".join(header[3:]) if len(header) > 3 else "",
                    ))
                rows = []
                header = ln.rstrip("\n").split("\t")
            elif ln.strip():
                rows.append([float(x) for x in ln.split()[:4]])
        if header is not None and rows:
            motifs.append(HomerMotif(
                consensus=header[0][1:],
                name=header[1] if len(header) > 1 else "?",
                threshold=float(header[2]) if len(header) > 2 else 0.0,
                pwm=np.asarray(rows, dtype=np.float32),
                extra="\t".join(header[3:]) if len(header) > 3 else "",
            ))
    return motifs


def _pwm_to_log_odds(pwm: np.ndarray, pseudocount: float = 0.001) -> np.ndarray:
    """Convert a probability PWM to natural-log-odds against a uniform bg.

    HOMER's calibrated threshold is in natural-log units (we verified
    against HOMER's internal ``ZOOPS`` threshold values), so use
    ``np.log`` not ``np.log2``."""
    p = np.clip(pwm, pseudocount, None)
    return np.log(p / 0.25).astype(np.float32)


def _revcomp_log_odds(lo: np.ndarray) -> np.ndarray:
    """Reverse-complement log-odds matrix: reverse rows and swap A<->T,
    C<->G (columns 0<->3 and 1<->2)."""
    return lo[::-1, [3, 2, 1, 0]]


def _dedup_peaks(peaks: pd.DataFrame, size: int) -> pd.DataFrame:
    """Remove peaks whose ``size``-bp window overlaps a kept peak on the
    same chromosome — matches HOMER's implicit peak-dedup behaviour (the
    published ``knownResults.txt`` header reports ~60% of raw peaks are
    retained for ``-size 200``). Without this our binomial p-values are
    ~2× more extreme than HOMER's simply because we use twice as many
    (highly-correlated) sequences."""
    if len(peaks) == 0:
        return peaks
    half = size // 2
    p = peaks.copy()
    p["_center"] = (p["start"].astype(int) + p["end"].astype(int)) // 2
    p = p.sort_values(["chrom", "_center"]).reset_index(drop=True)
    keep = np.ones(len(p), dtype=bool)
    last_center = {}
    for i, (chrom, c) in enumerate(zip(p["chrom"].values, p["_center"].values)):
        prev = last_center.get(chrom)
        if prev is not None and abs(c - prev) < size:
            keep[i] = False
        else:
            last_center[chrom] = c
    return p[keep].drop(columns="_center").reset_index(drop=True)


def _extract_peak_sequences(
    peaks: pd.DataFrame,
    genome_fa: Union[str, Path],
    size: int,
) -> Tuple[List[str], List[int]]:
    """Pull ``size``-bp windows centred on each peak midpoint.

    Returns ``(sequences, kept_indices)`` — indices let the caller align
    enrichment results back to the original peak order; sequences that
    would run off the chromosome end or contain only Ns are dropped."""
    import pyfaidx
    fa = pyfaidx.Fasta(str(genome_fa), as_raw=True,
                       sequence_always_upper=True)
    half = size // 2
    seqs: List[str] = []
    keep: List[int] = []
    for idx, row in enumerate(peaks.itertuples(index=False)):
        chrom = str(row.chrom)
        if chrom not in fa:
            continue
        center = (int(row.start) + int(row.end)) // 2
        s = center - half
        e = s + size
        if s < 0 or e > len(fa[chrom]):
            continue
        seq = fa[chrom][s:e]
        if seq.count("N") + seq.count("n") > size // 2:
            continue
        # Pad with 'N' if short (shouldn't happen after the length check, but
        # defensive for masked genomes).
        if len(seq) < size:
            seq = seq.ljust(size, "N")
        seqs.append(seq)
        keep.append(idx)
    return seqs, keep


def _sample_gc_matched_background(
    target_seqs: Sequence[str],
    genome_fa: Union[str, Path],
    size: int,
    *,
    n_bg: int = 50_000,
    n_gc_bins: int = 20,
    oversample: float = 4.0,
    seed: int = 0,
) -> List[str]:
    """Sample ``n_bg`` random genomic windows with the same GC-content
    distribution as ``target_seqs``.

    Draws ``oversample × n_bg`` uniform-random windows first (from the
    set of canonical chromosomes, weighted by length), then sub-samples
    per GC bin to match the target's GC histogram 1 : ``n_bg/len(target)``."""
    import pyfaidx
    rng = np.random.default_rng(seed)
    fa = pyfaidx.Fasta(str(genome_fa), as_raw=True,
                       sequence_always_upper=True)

    target_gc = np.asarray([
        (s.count("G") + s.count("C")) / max(size, 1) for s in target_seqs
    ])
    edges = np.linspace(0, 1, n_gc_bins + 1)
    edges[-1] += 1e-9
    target_hist, _ = np.histogram(target_gc, bins=edges)

    # Canonical chromosomes only (skip haplotypes / chrUn / random).
    chrom_names = [c for c in fa.keys()
                   if re.match(r"^chr([0-9]+|X|Y)$", c)]
    lens = np.asarray([len(fa[c]) for c in chrom_names], dtype=np.int64)
    if lens.sum() == 0:
        raise RuntimeError("no canonical chromosomes found in genome")
    chrom_p = lens / lens.sum()

    n_raw = int(n_bg * oversample)
    chrom_draws = rng.choice(len(chrom_names), size=n_raw, p=chrom_p)
    raw_seqs: List[str] = []
    raw_gc: List[float] = []
    for ci in chrom_draws:
        cname = chrom_names[ci]
        clen = lens[ci]
        if clen < size + 1:
            continue
        s = int(rng.integers(0, clen - size))
        e = s + size
        seq = fa[cname][s:e]
        if not seq or len(seq) < size:
            continue
        if seq.count("N") + seq.count("n") > size // 2:
            continue
        raw_seqs.append(seq)
        raw_gc.append((seq.count("G") + seq.count("C")) / size)
    raw_gc_arr = np.asarray(raw_gc)

    # Per-bin subsample so background hist matches target hist proportionally.
    scale = n_bg / max(len(target_seqs), 1)
    picked: List[int] = []
    for bi in range(n_gc_bins):
        in_bin = np.where((raw_gc_arr >= edges[bi]) & (raw_gc_arr < edges[bi + 1]))[0]
        want = int(round(target_hist[bi] * scale))
        if len(in_bin) == 0 or want == 0:
            continue
        k = min(want, len(in_bin))
        picked.extend(rng.choice(in_bin, size=k, replace=False).tolist())

    return [raw_seqs[i] for i in picked]


def _log_binom_sf(k: int, n: int, p: float) -> float:
    """Natural log of right-tail binomial survival ``log P(X >= k | n, p)``.

    scipy's ``binom.logsf`` underflows to ``-inf`` for the extreme tails
    that CUT&RUN motif enrichment produces (HOMER routinely reports
    ``Log P-value`` down to -2000 to -5000). This helper falls back to
    arbitrary-precision ``mpmath.betainc`` when scipy loses resolution,
    and further falls back to a Gaussian large-deviation bound when even
    mpmath underflows."""
    if k <= 0:
        return 0.0
    if k > n or p <= 0:
        return float("-inf")
    val = float(binom.logsf(k - 1, n, p))
    if np.isfinite(val) and val < 0:
        return val
    # ----- Extreme-tail fallback: arbitrary-precision incomplete-beta.
    try:
        import mpmath as _mp
        _mp.mp.dps = 50
        tail = _mp.betainc(k, n - k + 1, 0, p, regularized=True)
        if tail > 0:
            return float(_mp.log(tail))
    except Exception:
        pass
    # ----- Gaussian / Chernoff fallback: log(Phi(-Z)) for extreme Z.
    mean = n * p
    var = n * p * (1.0 - p)
    if var <= 0:
        return 0.0
    z = (k - 0.5 - mean) / np.sqrt(var)
    if z <= 0:
        return 0.0
    # log(Phi(-z)) ≈ -z^2/2 - log(z*sqrt(2*pi)) - log(1 + 1/z^2) for z >> 0
    return float(-0.5 * z * z - np.log(z * np.sqrt(2 * np.pi)))


def _extend_log_odds(log_odds: np.ndarray) -> np.ndarray:
    """Pad a (Lm, 4) log-odds matrix with a 5th column of -1e6 so N-base
    lookups by ``int8 value 4`` automatically sink the score to -inf. Lets
    us drop every ``np.where(col == 4, ...)`` guard in the scan loop."""
    Lm = log_odds.shape[0]
    out = np.full((Lm, 5), -1e6, dtype=np.float32)
    out[:, :4] = log_odds
    return out


def _scan_hits(
    seqs_int: np.ndarray,      # (N, L_seq) int8, values in {0,1,2,3,4}
    log_odds: np.ndarray,      # (L_mot, 4) float32
    threshold: float,
) -> np.ndarray:
    """Return a boolean array of length N marking which sequences contain
    ≥ 1 match (forward or reverse strand) for this PWM.

    Fast path vs the naive version: N bases are encoded as a 5th column in
    the log-odds matrix (value -1e6), so the inner loop is a single
    ``lo[j, seqs_int[:, j:j+n_pos]]`` lookup with in-place accumulation
    into one scratch array — no ``np.where``, no intermediate copies."""
    N, L = seqs_int.shape
    Lm = log_odds.shape[0]
    if Lm > L:
        return np.zeros(N, dtype=bool)
    n_pos = L - Lm + 1

    lo_f = _extend_log_odds(log_odds)
    lo_r = _extend_log_odds(_revcomp_log_odds(log_odds))

    # Forward strand.
    scores = lo_f[0, seqs_int[:, 0:n_pos]].astype(np.float32, copy=True)
    for j in range(1, Lm):
        scores += lo_f[j, seqs_int[:, j:j + n_pos]]
    fwd_hit = (scores.max(axis=1) >= threshold)

    # Reverse strand.
    scores = lo_r[0, seqs_int[:, 0:n_pos]].astype(np.float32, copy=True)
    for j in range(1, Lm):
        scores += lo_r[j, seqs_int[:, j:j + n_pos]]
    rc_hit = (scores.max(axis=1) >= threshold)

    return fwd_hit | rc_hit


# ---------------------------------------------------------------------
# Worker globals for multiprocessing (Pool initializer populates these).
# Using a module-level cache lets workers share the encoded sequence
# arrays via copy-on-write after fork instead of pickling them into every
# task; for 95k × 200bp sequences that saves ~20MB/task × N tasks.
# ---------------------------------------------------------------------
_WORKER_TGT: Optional[np.ndarray] = None
_WORKER_BG:  Optional[np.ndarray] = None


def _worker_init(tgt_int: np.ndarray, bg_int: np.ndarray) -> None:
    global _WORKER_TGT, _WORKER_BG
    _WORKER_TGT = tgt_int
    _WORKER_BG  = bg_int


def _worker_scan(task: Tuple[int, np.ndarray, float]) -> Tuple[int, int, int]:
    """Scan one motif across target + background; return (idx, t_hits, b_hits)."""
    idx, log_odds, threshold = task
    t = int(_scan_hits(_WORKER_TGT, log_odds, threshold).sum())
    b = int(_scan_hits(_WORKER_BG,  log_odds, threshold).sum())
    return idx, t, b


# ---------------------------------------------------------------------
# GPU path: conv1d-based batched PWM scanning on torch. 100x+ faster
# than the numpy loop for ~400 motifs / ~100k sequences (single H100:
# ~15 s end-to-end vs ~20 min on CPU).
# ---------------------------------------------------------------------

def _seqs_int_to_onehot_gpu(seqs_int: np.ndarray, device) -> "torch.Tensor":
    """Encode (N, L) int8 with {0..3} = ACGT and 4 = N, into (N, 4, L)
    float32 one-hot on ``device``. N bases become all-zero rows so any
    PWM contribution at that position is 0 (then we rely on a mask check
    below to reject sequences with too many Ns if needed).
    """
    import torch
    seqs = torch.from_numpy(seqs_int.astype(np.int64)).to(device)
    N, L = seqs.shape
    oh = torch.zeros((N, 5, L), dtype=torch.float32, device=device)
    oh.scatter_(1, seqs.unsqueeze(1), 1.0)
    return oh[:, :4, :].contiguous()  # drop the N-base slot


def _scan_hits_gpu_grouped(
    seqs_int: np.ndarray,
    motifs: Sequence[Tuple[np.ndarray, float]],   # (log_odds, threshold) per motif
    *,
    device: str = "cuda",
    batch_size: int = 64,
    verbose: bool = False,
) -> np.ndarray:
    """Return a length-``len(motifs)`` int array: # target sequences that
    contain ≥ 1 hit (fwd or rc) for each motif.

    Strategy:
      - Encode sequences to one-hot once, move to GPU.
      - Group motifs by length Lm. For each group:
          * stack fwd + rc log-odds as a (2*M_g, 4, Lm) conv weight
          * conv1d over the one-hot tensor -> (N, 2*M_g, n_pos)
          * max over position, split fwd/rc, OR, compare to threshold
          * sum over sequences -> (M_g,) hit counts
      - Motif groups are batched if M_g × n_pos × N is too big.
    """
    import torch
    import torch.nn.functional as F

    dev = torch.device(device)
    N, L = seqs_int.shape
    onehot = _seqs_int_to_onehot_gpu(seqs_int, dev)  # (N, 4, L)

    # Group motif indices by Lm
    by_len: dict[int, list[int]] = {}
    for i, (lo, _) in enumerate(motifs):
        by_len.setdefault(lo.shape[0], []).append(i)

    hit_counts = np.zeros(len(motifs), dtype=np.int64)
    for Lm, idxs in by_len.items():
        if Lm > L:
            continue

        # Prepare stacked weights for this length group + reverse complements.
        fwd = np.stack([motifs[i][0] for i in idxs], axis=0)            # (M, Lm, 4)
        rc  = np.stack([_revcomp_log_odds(motifs[i][0]) for i in idxs], 0)
        weights = np.concatenate([fwd, rc], axis=0)                    # (2M, Lm, 4)
        weights = np.transpose(weights, (0, 2, 1))                     # (2M, 4, Lm)
        W = torch.from_numpy(weights.astype(np.float32)).to(dev)
        thresholds = np.asarray([motifs[i][1] for i in idxs], dtype=np.float32)
        thr = torch.from_numpy(thresholds).to(dev)

        M = len(idxs)
        for bstart in range(0, M, batch_size):
            bend = min(bstart + batch_size, M)
            Wb = torch.cat([W[bstart:bend], W[M + bstart:M + bend]], dim=0)
            # conv1d: input (N, 4, L) × weight (2*Mb, 4, Lm) → (N, 2*Mb, n_pos)
            scores = F.conv1d(onehot, Wb)
            peak = scores.max(dim=2).values                            # (N, 2*Mb)
            Mb = bend - bstart
            fwd_peak = peak[:, :Mb]
            rc_peak  = peak[:, Mb:]
            thr_b = thr[bstart:bend]
            hits = ((fwd_peak >= thr_b) | (rc_peak >= thr_b))          # (N, Mb)
            counts = hits.sum(dim=0).to(torch.int64).cpu().numpy()
            for k, mi in enumerate(idxs[bstart:bend]):
                hit_counts[mi] = int(counts[k])
            if verbose:
                print(f"[gpu] Lm={Lm}  motifs {bstart}-{bend}/{M}  "
                      f"peakGPU mem≈{torch.cuda.memory_allocated(dev) >> 20} MB",
                      flush=True)

        del W, thr
        torch.cuda.empty_cache()

    del onehot
    torch.cuda.empty_cache()
    return hit_counts


def find_motifs_genome(
    peaks: Union[pd.DataFrame, str, Path],
    genome: Union[str, Path],
    motif_library: Union[str, Path],
    *,
    size: int = 200,
    n_bg: int = 50_000,
    n_gc_bins: int = 20,
    oversample: float = 4.0,
    seed: int = 0,
    threads: int = 8,
    backend: str = "auto",
    gpu_batch_size: int = 64,
    outdir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Pure-Python HOMER-style known-motif enrichment.

    No Perl dependency. Implements HOMER's *known* pipeline:
    peak-sequence extraction → GC-matched genomic background → PWM scan
    (both strands) with HOMER's pre-calibrated log-odds threshold →
    per-motif right-tailed binomial test. Produces a DataFrame with the
    same column schema as HOMER's ``knownResults.txt`` so the result
    feeds straight into :func:`epione.pl.homer_motif_table`.

    Arguments:
        peaks: DataFrame with ``chrom,start,end`` columns, or a BED/TSV
            path with those as the first three columns.
        genome: path to the genome FASTA (indexed by ``.fai``; will be
            auto-indexed on first call).
        motif_library: path to a HOMER ``known.motifs`` file. Default
            install puts the human library at
            ``$HOMER/data/knownTFs/vertebrates/known.motifs``.
        size: window size centred on each peak midpoint (HOMER default 200).
        n_bg: target number of GC-matched background sequences.
        n_gc_bins: GC histogram bins for matching (HOMER uses ~20).
        oversample: fraction of extra random windows to draw before GC
            sub-sampling (helps hit low-density GC bins).
        seed: RNG seed for background sampling (and reverse complements).
        threads: worker processes for the ``backend='cpu'`` path. Ignored on
            the GPU path.
        backend: ``'auto'`` (default) picks GPU when ``torch.cuda`` is
            available, else falls back to the CPU multiprocess path. Pass
            ``'gpu'`` or ``'cpu'`` to force.
        gpu_batch_size: number of motifs processed per ``conv1d`` call in
            the GPU path. Lower this if you hit GPU OOM on very long
            sequences; 64 fits comfortably on a 40 GB A100 / 80 GB H100
            for ~95 k × 200 bp inputs.
        outdir: if given, mimic HOMER's output layout: writes
            ``knownResults.txt`` and ``knownResults/known<i>.motif``
            files (for PWM logos), matching the schema
            :func:`epione.pl.homer_motif_table` expects.
        verbose: print progress lines.

    Returns:
        DataFrame with HOMER ``knownResults.txt`` columns:
        ``Motif Name``, ``Consensus``, ``P-value``, ``Log P-value``,
        ``q-value (Benjamini)``, ``# of Target Sequences with Motif``,
        ``% of Target Sequences with Motif``,
        ``# of Background Sequences with Motif``,
        ``% of Background Sequences with Motif``. Sorted by p-value.
    """
    # 1. Peaks → DataFrame.
    if isinstance(peaks, (str, Path)):
        peaks_df = pd.read_csv(
            peaks, sep="\t", header=None,
            names=["chrom", "start", "end", "name", "score"],
            usecols=[0, 1, 2, 3, 4],
        )
    else:
        peaks_df = peaks.copy()
    peaks_df = peaks_df.dropna(subset=["chrom", "start", "end"]).reset_index(drop=True)
    peaks_df["start"] = peaks_df["start"].astype(int)
    peaks_df["end"]   = peaks_df["end"].astype(int)

    # 2. Deduplicate + extract target + background sequences.
    n_in = len(peaks_df)
    peaks_df = _dedup_peaks(peaks_df, size)
    if verbose and len(peaks_df) < n_in:
        print(f"[find_motifs_genome] peak dedup: {n_in} -> {len(peaks_df)} "
              f"(overlap < {size} bp merged)")
    if verbose:
        print(f"[find_motifs_genome] extracting {len(peaks_df)} target windows "
              f"(size={size})")
    tgt_seqs, _ = _extract_peak_sequences(peaks_df, genome, size)
    if verbose:
        print(f"[find_motifs_genome]   kept {len(tgt_seqs)} after bounds / N-filter")
        print(f"[find_motifs_genome] sampling GC-matched background ({n_bg} seqs)")
    bg_seqs = _sample_gc_matched_background(
        tgt_seqs, genome, size,
        n_bg=n_bg, n_gc_bins=n_gc_bins, oversample=oversample, seed=seed,
    )
    if verbose:
        print(f"[find_motifs_genome]   kept {len(bg_seqs)} background seqs")

    tgt_int = _encode_sequences(tgt_seqs, size)
    bg_int  = _encode_sequences(bg_seqs,  size)

    # 3. Load motif library.
    motifs = _load_homer_motif_library(motif_library)
    if verbose:
        print(f"[find_motifs_genome] scanning {len(motifs)} motifs from "
              f"{motif_library}")

    # Prepare per-motif log-odds + threshold.
    T = len(tgt_seqs); B = len(bg_seqs)
    lo_thr: List[Tuple[np.ndarray, float]] = [
        (_pwm_to_log_odds(m.pwm), m.threshold) for m in motifs
    ]

    # Pick backend. 'auto' = GPU if torch.cuda.is_available(), else cpu-mp.
    chosen = backend
    if chosen == "auto":
        try:
            import torch  # noqa: F401
            chosen = "gpu" if torch.cuda.is_available() else "cpu"
        except Exception:
            chosen = "cpu"
    if verbose:
        print(f"[find_motifs_genome] backend={chosen}", flush=True)

    import time
    t0 = time.time()
    if chosen == "gpu":
        t_hits_arr = _scan_hits_gpu_grouped(
            tgt_int, lo_thr, batch_size=gpu_batch_size, verbose=verbose)
        b_hits_arr = _scan_hits_gpu_grouped(
            bg_int,  lo_thr, batch_size=gpu_batch_size, verbose=verbose)
        hits = np.stack([t_hits_arr, b_hits_arr], axis=1)
    else:
        tasks: List[Tuple[int, np.ndarray, float]] = [
            (i, lo, thr) for i, (lo, thr) in enumerate(lo_thr)
        ]
        hits = np.zeros((len(motifs), 2), dtype=np.int64)
        if threads and threads > 1:
            import multiprocessing as _mp
            with _mp.Pool(processes=threads,
                          initializer=_worker_init,
                          initargs=(tgt_int, bg_int)) as pool:
                for k, (idx, t_h, b_h) in enumerate(
                    pool.imap_unordered(_worker_scan, tasks, chunksize=4)
                ):
                    hits[idx] = (t_h, b_h)
                    if verbose and (k + 1) % 50 == 0:
                        elapsed = time.time() - t0
                        print(f"[find_motifs_genome]   scanned "
                              f"{k + 1}/{len(motifs)} ({elapsed:.1f}s)",
                              flush=True)
        else:
            _worker_init(tgt_int, bg_int)
            for k, task in enumerate(tasks):
                idx, t_h, b_h = _worker_scan(task)
                hits[idx] = (t_h, b_h)
                if verbose and (k + 1) % 50 == 0:
                    print(f"[find_motifs_genome]   scanned {k + 1}/{len(motifs)}",
                          flush=True)
    if verbose:
        print(f"[find_motifs_genome] scan done in {time.time()-t0:.1f}s",
              flush=True)

    rows = []
    for i, m in enumerate(motifs):
        t_hits, b_hits = int(hits[i, 0]), int(hits[i, 1])
        p_bg = max(b_hits / B, 1.0 / B) if B else 0.0
        if T == 0 or p_bg <= 0:
            log_p = 0.0
        else:
            log_p = _log_binom_sf(t_hits, T, p_bg)
        rows.append({
            "Motif Name": m.name,
            "Consensus": m.consensus,
            "Log P-value": log_p,
            "P-value": float(np.exp(log_p)) if log_p > -700 else 0.0,
            "# of Target Sequences with Motif": t_hits,
            "% of Target Sequences with Motif": 100 * t_hits / max(T, 1),
            "# of Background Sequences with Motif": b_hits,
            "% of Background Sequences with Motif": 100 * b_hits / max(B, 1),
            "_pwm": m.pwm,
        })

    result = pd.DataFrame(rows).sort_values("Log P-value").reset_index(drop=True)

    # Benjamini-Hochberg on the tail p-values.
    p = np.minimum(np.exp(np.clip(result["Log P-value"].values, -700, 0)), 1.0)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int); ranks[order] = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    qvals = np.empty(n); qvals[order] = np.minimum(q, 1.0)
    result["q-value (Benjamini)"] = qvals

    # 4. Optionally emit HOMER-shaped outputs (so epi.pl.homer_motif_table works).
    if outdir is not None:
        outdir = ensure_dir(outdir)
        pwm_dir = ensure_dir(outdir / "knownResults")
        # knownResults.txt in HOMER's column order
        out = result.drop(columns=["_pwm"]).copy()
        cols = [
            "Motif Name", "Consensus", "P-value", "Log P-value",
            "q-value (Benjamini)",
            "# of Target Sequences with Motif",
            "% of Target Sequences with Motif",
            "# of Background Sequences with Motif",
            "% of Background Sequences with Motif",
        ]
        # HOMER formats target/bg percents as strings with '%'
        out2 = out.copy()
        out2["% of Target Sequences with Motif"] = out2[
            "% of Target Sequences with Motif"].map(lambda v: f"{v:.2f}%")
        out2["% of Background Sequences with Motif"] = out2[
            "% of Background Sequences with Motif"].map(lambda v: f"{v:.2f}%")
        out2[cols].to_csv(outdir / "knownResults.txt",
                          sep="\t", index=False)

        # Write one ``known{i}.motif`` per row so epi.pl.homer_motif_table
        # can recover the PWM by rank. Header thresholds are placeholders
        # (0.0) because the plotter only consumes the probability rows.
        for i, (pwm, name, cons) in enumerate(zip(
            result["_pwm"], result["Motif Name"], result["Consensus"],
        )):
            with open(pwm_dir / f"known{i + 1}.motif", "w") as fh:
                fh.write(f">{cons}\t{name}\t0.0\n")
                for row in pwm:
                    fh.write("\t".join(f"{float(v):.3f}" for v in row) + "\n")

    return result.drop(columns=["_pwm"])
