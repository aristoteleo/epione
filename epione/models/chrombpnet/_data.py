"""ChromBPNet data recipe: regions, signal, hyper-parameters, batches.

Ports of the official helpers in kundajelab/chrombpnet (MIT licence,
Copyright (c) 2022 Kundaje Lab), commit ``09938fd``:

* ``training/utils/one_hot.py``, ``data_utils.py``, ``augment.py`` and
  ``data_generators/batchgen_generator.py`` -> :func:`one_hot_encode`,
  :func:`load_regions`, :class:`ChromBPNetBatches`;
* ``helpers/hyperparameters/find_bias_hyperparams.py`` and
  ``find_chrombpnet_hyperparams.py`` -> :func:`bias_hyperparams`,
  :func:`chrombpnet_hyperparams`;
* ``helpers/make_gc_matched_negatives/*`` -> :func:`gc_matched_nonpeaks`.

Two departures, both on the input side only:

* the per-base insertion signal can come from a fragments file directly
  (:class:`InsertionSignal`, via :class:`epione._insertions.InsertionReader`)
  instead of a bigWig. For 10x fragments the official pipeline builds its
  bigWig with cuts at a fragment's ``start`` and ``end`` (its auto-detected
  +4/-5 input shift plus the +4/-4 target shift), which is
  ``shift=(0, 0)``; that is the default here;
* region tables are pandas DataFrames in memory rather than files on disk.

Random draws use explicit :class:`numpy.random.RandomState` /
:class:`random.Random` objects. Upstream calls the global generators; the
legacy ``RandomState`` gives the same stream for the same seed and call
sequence, which is what the parity tests rely on.
"""
from __future__ import annotations

import math
import random as _random
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "NARROWPEAK_SCHEMA",
    "one_hot_encode",
    "read_narrowpeak",
    "BigWigSignal",
    "InsertionSignal",
    "open_signal",
    "filter_edge_regions",
    "load_regions",
    "bias_hyperparams",
    "chrombpnet_hyperparams",
    "genomewide_gc",
    "gc_matched_nonpeaks",
    "ChromBPNetBatches",
]

NARROWPEAK_SCHEMA = ["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"]


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------

def one_hot_encode(seqs: Sequence[str]) -> np.ndarray:
    """Upstream ``dna_to_one_hot``: ``(N, L, 4)`` int8, A/C/G/T channels,
    upper-cased first, anything else all zeros."""
    seqs = list(seqs)
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("all sequences must have the same length")
    lut = np.full(256, 4, dtype=np.int64)
    for i, b in enumerate("ACGT"):
        lut[ord(b)] = i
        lut[ord(b.lower())] = i
    codes = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)
    eye = np.identity(5, dtype=np.int8)[:, :-1]
    return eye[lut[codes]].reshape(len(seqs), L, 4)


def read_narrowpeak(path_or_df) -> pd.DataFrame:
    """A 10-column narrowPeak (``summit`` relative to ``start``), as upstream
    reads it. A DataFrame with those columns is passed through."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df
        if list(df.columns[:3]) != ["chr", "start", "end"]:
            df = df.copy()
            df.columns = NARROWPEAK_SCHEMA[: df.shape[1]]
        return df
    return pd.read_csv(path_or_df, header=None, sep="\t", names=NARROWPEAK_SCHEMA)


def _fasta(genome):
    import pyfaidx

    fasta = getattr(genome, "fasta", genome)
    if callable(fasta):
        fasta = fasta()
    if isinstance(fasta, pyfaidx.Fasta):
        return fasta
    return pyfaidx.Fasta(str(fasta))


def get_seq(regions: pd.DataFrame, genome, width: int) -> np.ndarray:
    """``data_utils.get_seq``: one-hot ``(N, width, 4)`` centred on summits."""
    fa = _fasta(genome)
    centers = (regions["start"] + regions["summit"]).to_numpy()
    seqs = [str(fa[c][(m - width // 2):(m + width // 2)])
            for c, m in zip(regions["chr"].to_numpy(), centers)]
    if not seqs:
        return np.zeros((0, width, 4), dtype=np.int8)
    return one_hot_encode(seqs)


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

class BigWigSignal:
    """Per-base counts from an unstranded insertion bigWig (upstream input)."""

    def __init__(self, path: Union[str, Path]):
        self.path = str(path)
        self._bw = None

    def _open(self):
        if self._bw is None:
            import pyBigWig
            self._bw = pyBigWig.open(self.path)
        return self._bw

    def chroms(self) -> Dict[str, int]:
        return dict(self._open().chroms())

    def values(self, chrom: str, start: int, end: int) -> np.ndarray:
        return np.nan_to_num(np.asarray(self._open().values(chrom, int(start), int(end))))

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_bw"] = None
        return d


class InsertionSignal:
    """Per-base Tn5 insertions read straight from a tabix-indexed fragments
    file -- no bigWig export.

    Each fragment ``[s, e)`` contributes one cut at ``s + shift[0]`` and one
    at ``e + shift[1]``; the duplicate-count column is ignored, as in the
    official ``reads_to_bigwig`` (two tags per fragment line).

    Parameters
    ----------
    fragments
        bgzipped, tabix-indexed fragments file.
    chrom_sizes
        ``{chrom: length}`` (or a Genome with ``.chrom_sizes``); needed for
        the edge filters upstream applies with ``bw.chroms()``.
    barcodes
        Cells to pool; ``None`` pools every fragment in the file, as the
        official pipeline does.
    shift
        Cut convention, see :mod:`epione._insertions`. ``(0, 0)`` matches
        the bigWig the official pipeline builds from 10x fragments.
    """

    def __init__(self, fragments, chrom_sizes, *, barcodes=None, shift=(0, 0)):
        self.fragments = str(fragments)
        cs = getattr(chrom_sizes, "chrom_sizes", chrom_sizes)
        if callable(cs):
            cs = cs()
        self._chrom_sizes = {str(k): int(v) for k, v in dict(cs).items()}
        self.barcodes = None if barcodes is None else set(barcodes)
        self.shift = (int(shift[0]), int(shift[1]))
        self._tbx = None

    def _open(self):
        if self._tbx is None:
            import pysam

            from epione._insertions import ensure_tabix_index
            ensure_tabix_index(self.fragments)
            self._tbx = pysam.TabixFile(self.fragments)
            self._contigs = set(self._tbx.contigs)
        return self._tbx

    def chroms(self) -> Dict[str, int]:
        return dict(self._chrom_sizes)

    def values(self, chrom: str, start: int, end: int) -> np.ndarray:
        start, end = int(start), int(end)
        out = np.zeros(end - start, dtype=np.float64)
        tbx = self._open()
        if chrom not in self._contigs:
            return out
        s0, s1 = self.shift
        # Widen the fetch so fragments whose shifted cut lands in the window
        # are returned even when the fragment itself does not overlap it.
        pad = max(abs(s0), abs(s1)) + 1
        bc = self.barcodes
        for line in tbx.fetch(chrom, max(0, start - pad), end + pad):
            f = line.split("\t", 4)
            if bc is not None and f[3] not in bc:
                continue
            for p in (int(f[1]) + s0, int(f[2]) + s1):
                if start <= p < end:
                    out[p - start] += 1
        return out

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_tbx"] = None
        return d


def open_signal(signal=None, *, bigwig=None, fragments=None, chrom_sizes=None,
                barcodes=None, shift=(0, 0)):
    """Normalise the ways a signal can be given into an object with
    ``values(chrom, start, end)`` and ``chroms()``."""
    if signal is not None and hasattr(signal, "values") and hasattr(signal, "chroms"):
        return signal
    if bigwig is not None or (isinstance(signal, (str, Path)) and str(signal).endswith((".bw", ".bigwig", ".bigWig"))):
        return BigWigSignal(bigwig or signal)
    frag = fragments or signal
    if frag is None:
        raise ValueError("give a bigwig= or fragments= signal")
    if chrom_sizes is None:
        raise ValueError("fragments input needs chrom_sizes=")
    return InsertionSignal(frag, chrom_sizes, barcodes=barcodes, shift=shift)


def get_cts(regions: pd.DataFrame, signal, width: int) -> np.ndarray:
    """``data_utils.get_cts``: per-base counts ``(N, width)`` at summits."""
    centers = (regions["start"] + regions["summit"]).to_numpy()
    if len(centers) == 0:
        return np.zeros((0, width))
    return np.array([signal.values(c, m - width // 2, m + width // 2)
                     for c, m in zip(regions["chr"].to_numpy(), centers)])


# ---------------------------------------------------------------------------
# Region filters and hyper-parameters
# ---------------------------------------------------------------------------

def filter_edge_regions(regions: pd.DataFrame, chrom_sizes: Mapping[str, int],
                        width: int) -> pd.DataFrame:
    """``param_utils.filter_edge_regions``: drop regions whose
    ``width``-bp window around the summit leaves the chromosome."""
    center = regions["start"] + regions["summit"]
    keep = (center - width // 2) >= 0
    regions = regions[keep]
    center = regions["start"] + regions["summit"]
    sizes = regions["chr"].map(lambda c: chrom_sizes.get(c, 0))
    return regions[~(center + width // 2 > sizes)]


def _split(regions: pd.DataFrame, fold: Mapping[str, Sequence[str]], *modes):
    chroms = [c for m in modes for c in fold[m]]
    return regions[regions["chr"].isin(chroms)]


def _counts_sum(regions, signal, outputlen) -> np.ndarray:
    return np.sum(get_cts(regions, signal, outputlen), axis=1) if len(regions) else np.zeros(0)


def bias_hyperparams(peaks, nonpeaks, signal, fold, *, inputlen: int = 2114,
                     outputlen: int = 1000, bias_threshold_factor: float = 0.5,
                     outlier_threshold: float = 0.9999, filters: int = 128,
                     n_dil_layers: int = 4, max_jitter: int = 0) -> dict:
    """``find_bias_hyperparams.main``.

    Returns the filtered non-peaks (train+valid kept below the peak-derived
    ceiling and outlier bounds; test untouched), the filtered peaks and the
    model parameters, with ``counts_loss_weight`` rounded to 2 decimals as
    upstream writes it to the params TSV and reads it back.
    """
    peaks, nonpeaks = read_narrowpeak(peaks), read_narrowpeak(nonpeaks)
    if peaks.empty or nonpeaks.empty:
        raise ValueError("peaks and nonpeaks must be non-empty")
    if inputlen < outputlen:
        raise ValueError("inputlen must be >= outputlen")
    sizes = signal.chroms()

    tr_peaks = _split(peaks, fold, "train", "valid")
    te_peaks = _split(peaks, fold, "test")
    tr_non = _split(nonpeaks, fold, "train", "valid")
    te_non = _split(nonpeaks, fold, "test")

    tr_non = filter_edge_regions(tr_non, sizes, inputlen)
    te_non = filter_edge_regions(te_non, sizes, inputlen)
    tr_peaks = filter_edge_regions(tr_peaks, sizes, inputlen)
    te_peaks = filter_edge_regions(te_peaks, sizes, inputlen)

    peak_cnts = _counts_sum(tr_peaks, signal, outputlen)
    nonpeak_cnts = _counts_sum(tr_non, signal, outputlen)

    counts_threshold = np.quantile(peak_cnts, 0.01) * bias_threshold_factor
    if not counts_threshold > 0:
        raise ValueError("counts threshold is 0 - all non peaks would be filtered")
    final_cnts = nonpeak_cnts[nonpeak_cnts < counts_threshold]
    if len(final_cnts) == 0:
        raise ValueError("upper-bound cut-off leaves no non-peaks for the bias model")

    upper = np.quantile(final_cnts, outlier_threshold)
    lower = np.quantile(final_cnts, 1 - outlier_threshold)
    tr_non = tr_non[(nonpeak_cnts < upper) & (nonpeak_cnts > lower)]

    clw = np.median(final_cnts[(final_cnts < upper) & (final_cnts > lower)]) / 10
    if clw == 0:
        raise ValueError("counts_loss_weight is 0")
    if clw < 1.0:
        clw = 1.0
    return {
        "nonpeaks": pd.concat([tr_non, te_non]),
        "peaks": pd.concat([tr_peaks, te_peaks]),
        "counts_threshold": float(counts_threshold),
        "counts_sum_min_thresh": float(lower),
        "counts_sum_max_thresh": float(upper),
        "params": {
            "counts_loss_weight": round(float(clw), 2),
            "filters": filters, "n_dil_layers": n_dil_layers,
            "inputlen": inputlen, "outputlen": outputlen,
            "max_jitter": max_jitter, "negative_sampling_ratio": 1.0,
        },
    }


def chrombpnet_hyperparams(peaks, nonpeaks, signal, fold, *, inputlen: int = 2114,
                           outputlen: int = 1000, max_jitter: int = 500,
                           negative_sampling_ratio: float = 0.1,
                           outlier_threshold: float = 0.9999, filters: int = 512,
                           n_dil_layers: int = 8,
                           rng: Optional[np.random.RandomState] = None) -> dict:
    """``find_chrombpnet_hyperparams.main`` (without the bias rescaling,
    which needs the model: see :func:`._train.adjust_bias_logcounts`).

    ``rng`` replaces upstream's unseeded global ``np.random.choice``.
    Returns filtered peaks, the training non-peaks (subsampled to the
    number of peaks with ``random_state=1`` as upstream), the train+valid
    non-peaks within the count bounds (for the bias rescaling), and params.
    """
    rng = rng if rng is not None else np.random.RandomState()
    peaks, nonpeaks = read_narrowpeak(peaks), read_narrowpeak(nonpeaks)
    sizes = signal.chroms()

    tr_peaks = _split(peaks, fold, "train", "valid")
    te_peaks = _split(peaks, fold, "test")
    tr_non = _split(nonpeaks, fold, "train", "valid")
    te_non = _split(nonpeaks, fold, "test")

    tr_peaks = filter_edge_regions(tr_peaks, sizes, inputlen + 2 * max_jitter)
    te_peaks = filter_edge_regions(te_peaks, sizes, inputlen)
    tr_non = filter_edge_regions(tr_non, sizes, inputlen)
    te_non = filter_edge_regions(te_non, sizes, inputlen)

    peak_cnts = _counts_sum(tr_peaks, signal, outputlen)
    nonpeak_cnts = _counts_sum(tr_non, signal, outputlen)

    if negative_sampling_ratio > 0:
        final_cnts = np.concatenate((peak_cnts, rng.choice(
            nonpeak_cnts, replace=False,
            size=int(negative_sampling_ratio * len(peak_cnts)))))
    else:
        final_cnts = peak_cnts
    upper = np.quantile(final_cnts, outlier_threshold)
    lower = np.quantile(final_cnts, 1 - outlier_threshold)

    keep_non = (nonpeak_cnts < upper) & (nonpeak_cnts > lower)
    tr_peaks = tr_peaks[(peak_cnts < upper) & (peak_cnts > lower)]
    tr_non_kept = tr_non[keep_non]
    if tr_non_kept.shape[0] > tr_peaks.shape[0]:
        train_non = tr_non_kept.sample(n=tr_peaks.shape[0], random_state=1)
    else:
        train_non = tr_non_kept

    clw = np.median(final_cnts[(final_cnts <= upper) & (final_cnts >= lower)]) / 10
    if clw == 0:
        raise ValueError("counts_loss_weight is 0")
    if clw < 1.0:
        clw = 1.0
    return {
        "peaks": pd.concat([tr_peaks, te_peaks]),
        "nonpeaks": pd.concat([train_non, te_non]),
        "bias_adjust_regions": tr_non_kept,
        "bias_adjust_counts": nonpeak_cnts[keep_non],
        "counts_sum_min_thresh": float(lower),
        "counts_sum_max_thresh": float(upper),
        "params": {
            "counts_loss_weight": round(float(clw), 2),
            "filters": filters, "n_dil_layers": n_dil_layers,
            "inputlen": inputlen, "outputlen": outputlen,
            "max_jitter": max_jitter,
            "negative_sampling_ratio": negative_sampling_ratio,
        },
    }


# ---------------------------------------------------------------------------
# GC-matched non-peaks
# ---------------------------------------------------------------------------

def genomewide_gc(genome, width: int = 2114, stride: int = 1000,
                  chroms: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """``get_genomewide_gc_bins.get_genomewide_gc``: GC fraction (rounded to
    2 decimals) of ``width``-bp bins every ``stride`` bp."""
    fa = _fasta(genome)
    rows = []
    div, rem = width // stride, width % stride
    sxd = div * stride
    for chrm in (chroms or list(fa.keys())):
        s = str(fa[chrm][:]).upper()
        cache = [0] * div
        runsum = 0
        for i in range(0, sxd, stride):
            c = s.count("C", i, i + stride) + s.count("G", i, i + stride)
            cache[i // stride % div] = c
            runsum += c
        for i in range(sxd, len(s) - rem, stride):
            left = 0
            if rem:
                left = s.count("C", i, i + rem) + s.count("G", i, i + rem)
                runsum += left
            rows.append((chrm, i - sxd, i - sxd + width, round(runsum / width, 2)))
            if div == 0:
                runsum = 0
            else:
                runsum -= cache[i // stride % div]
                right = s.count("C", i + rem, i + stride) + s.count("G", i + rem, i + stride)
                runsum += right
                cache[i // stride % div] = left + right
    return pd.DataFrame(rows, columns=["chr", "start", "end", "gc"])


def _peak_gc(peaks: pd.DataFrame, genome, chrom_sizes, width: int) -> pd.DataFrame:
    """``get_gc_content.main``."""
    fa = _fasta(genome)
    out = []
    for c, s, smt in zip(peaks["chr"], peaks["start"], peaks["summit"]):
        m = s + smt
        a, b = m - width // 2, m + width // 2
        if a < 0 or b > chrom_sizes[c]:
            continue
        seq = str(fa[c][a:b]).upper()
        out.append((c, a, b, round((seq.count("G") + seq.count("C")) / len(seq), 2)))
    return pd.DataFrame(out, columns=["chr", "start", "end", "gc"])


def _exclude(candidates: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """``bedtools intersect -v`` of candidate bins against merged intervals
    (half-open; a bin is dropped if it overlaps any interval by >= 1 bp)."""
    keep = np.ones(len(candidates), dtype=bool)
    for chrom, grp in intervals.groupby("chr"):
        iv = grp.sort_values("start")
        starts, ends = iv["start"].to_numpy(), iv["end"].to_numpy()
        # merge
        ms, me = [], []
        for a, b in zip(starts, ends):
            if ms and a <= me[-1]:
                me[-1] = max(me[-1], b)
            else:
                ms.append(a)
                me.append(b)
        ms, me = np.asarray(ms), np.asarray(me)
        idx = np.where(candidates["chr"].to_numpy() == chrom)[0]
        if not len(idx) or not len(ms):
            continue
        cs = candidates["start"].to_numpy()[idx]
        ce = candidates["end"].to_numpy()[idx]
        j = np.searchsorted(ms, ce, side="left") - 1  # last interval starting before ce
        hit = (j >= 0) & (me[np.clip(j, 0, None)] > cs)
        keep[idx[hit]] = False
    return candidates[keep]


def gc_matched_nonpeaks(peaks, genome, fold, chrom_sizes, *, inputlen: int = 2114,
                        stride: int = 1000, neg_to_pos_ratio_train: int = 2,
                        blacklist: Optional[pd.DataFrame] = None, seed: int = 1234,
                        candidates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """``chrombpnet prep nonpeaks``: GC-matched background regions.

    Candidate bins (:func:`genomewide_gc`) that overlap a peak or blacklist
    region padded by ``inputlen // 2`` are excluded; for every peak, a
    candidate with the same GC fraction (or the nearest available, stepping
    0.01 at random as upstream does) is drawn without replacement --
    ``neg_to_pos_ratio_train`` per train/valid peak, one per test peak.
    Returns a 10-column narrowPeak-like table with ``summit = inputlen//2``.
    """
    peaks = read_narrowpeak(peaks)
    sizes = dict(getattr(chrom_sizes, "chrom_sizes", chrom_sizes))
    if candidates is None:
        candidates = genomewide_gc(genome, inputlen, stride)
    excl = peaks[["chr", "start", "end"]].copy()
    if blacklist is not None:
        excl = pd.concat([excl, blacklist.iloc[:, :3].set_axis(["chr", "start", "end"], axis=1)])
    pad = inputlen // 2
    excl = excl.assign(start=(excl["start"] - pad).clip(lower=0),
                       end=np.minimum(excl["end"] + pad, excl["chr"].map(sizes).fillna(np.inf)).astype(int))
    cand = _exclude(candidates, excl)

    def remap(c):
        for m in ("train", "valid", "test"):
            if c in fold[m]:
                return f"chrom_{m}"
        return "ignore"

    negatives: Dict[str, Dict[float, list]] = {}
    for c, s, e, gc in cand.itertuples(index=False):
        key = remap(c)
        if key == "ignore":
            continue
        negatives.setdefault(key, {}).setdefault(float(gc), []).append((key, s, e, c))

    rnd = _random.Random(seed)

    def scale_gc(g):
        g = g + 0.01 if rnd.random() > 0.5 else g - 0.01
        g = round(g, 2)
        if g <= 0:
            g += 0.01
        if g >= 1:
            g -= 0.01
        return g

    used: Dict[str, Dict[float, list]] = {}

    def adjust(chrom, g):
        used.setdefault(chrom, {}).setdefault(g, [])
        while (g not in negatives[chrom]) or (len(used[chrom][g]) >= len(negatives[chrom][g])):
            g = scale_gc(g)
            used[chrom].setdefault(g, [])
        return g

    fg = _peak_gc(peaks, genome, sizes, inputlen)
    # Upstream loops forever when a split has fewer candidates than the
    # negatives it must draw; fail clearly instead.
    need: Dict[str, int] = {}
    for c in fg["chr"]:
        key = remap(c)
        if key != "ignore":
            need[key] = need.get(key, 0) + (neg_to_pos_ratio_train
                                            if key in ("chrom_train", "chrom_valid") else 1)
    for key, n_need in need.items():
        have = sum(len(v) for v in negatives.get(key, {}).values())
        if have < n_need:
            raise ValueError(f"{key}: {have} candidate background bins for {n_need} "
                             "non-peaks; lower neg_to_pos_ratio_train or the stride")
    out = []
    for c, s, e, gc in fg.itertuples(index=False):
        key = remap(c)
        if key == "ignore":
            continue
        reps = neg_to_pos_ratio_train if key in ("chrom_train", "chrom_valid") else 1
        for _ in range(reps):
            g = adjust(key, float(gc))
            k = rnd.randint(0, len(negatives[key][g]) - 1)
            while k in used[key][g]:
                g = adjust(key, g)
                k = rnd.randint(0, len(negatives[key][g]) - 1)
            used[key][g].append(k)
            _, ns, ne, nc = negatives[key][g][k]
            out.append((nc, int(ns), int(ne), g))
    neg = pd.DataFrame(out, columns=["chr", "start", "end", "gc"])
    res = neg[["chr", "start", "end"]].copy()
    for col in ["1", "2", "3", "4", "5", "6"]:
        res[col] = "."
    res["summit"] = inputlen // 2
    return res


# ---------------------------------------------------------------------------
# Loading and batching
# ---------------------------------------------------------------------------

def load_regions(regions: Optional[pd.DataFrame], genome, signal, inputlen: int,
                 outputlen: int, max_jitter: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """``data_utils.get_seq_cts_coords``: sequences ``(N, inputlen +
    2*jitter, 4)`` and counts ``(N, outputlen + 2*jitter)`` at summits."""
    if regions is None or len(regions) == 0:
        return (np.zeros((0, inputlen + 2 * max_jitter, 4), np.int8),
                np.zeros((0, outputlen + 2 * max_jitter)))
    seqs = get_seq(regions, genome, inputlen + 2 * max_jitter)
    cts = get_cts(regions, signal, outputlen + 2 * max_jitter)
    return seqs, cts


def _take_per_row(A, idx, n):
    all_idx = idx[:, None] + np.arange(n)
    return A[np.arange(all_idx.shape[0])[:, None], all_idx]


class ChromBPNetBatches:
    """``ChromBPNetBatchGenerator``: per-epoch jitter, revcomp, negative
    subsampling and shuffling, in upstream's order of random draws.

    Peaks carry ``max_jitter`` flanks and are randomly cropped each epoch;
    non-peaks are subsampled to ``negative_sampling_ratio x n_peaks`` (if
    the ratio is < 1); half of all examples are reverse-complemented; the
    result is shuffled when ``shuffle`` is set.
    """

    def __init__(self, peak_seqs, peak_cts, nonpeak_seqs, nonpeak_cts, *, inputlen,
                 outputlen, batch_size=64, negative_sampling_ratio=1.0, add_revcomp=False,
                 shuffle=False, rng: Optional[np.random.RandomState] = None):
        self.peak_seqs = peak_seqs if peak_seqs is not None and len(peak_seqs) else None
        self.peak_cts = peak_cts
        self.nonpeak_seqs = nonpeak_seqs if nonpeak_seqs is not None and len(nonpeak_seqs) else None
        self.nonpeak_cts = nonpeak_cts
        self.inputlen, self.outputlen = int(inputlen), int(outputlen)
        self.batch_size = int(batch_size)
        self.negative_sampling_ratio = float(negative_sampling_ratio)
        self.add_revcomp, self.shuffle = bool(add_revcomp), bool(shuffle)
        self.rng = rng if rng is not None else np.random.RandomState()
        self.crop_revcomp_data()

    def __len__(self) -> int:
        return math.ceil(self.seqs.shape[0] / self.batch_size)

    def _random_crop(self, seqs, cts):
        max_start = seqs.shape[1] - self.inputlen
        if seqs.shape[1] - self.inputlen != cts.shape[1] - self.outputlen:
            raise ValueError("sequence and count flanks differ")
        starts = self.rng.choice(range(max_start + 1), size=seqs.shape[0], replace=True)
        return (_take_per_row(seqs, starts, self.inputlen),
                _take_per_row(cts, starts, self.outputlen))

    def crop_revcomp_data(self) -> None:
        if self.peak_seqs is not None and self.nonpeak_seqs is not None:
            cp, cc = self._random_crop(self.peak_seqs, self.peak_cts)
            if self.negative_sampling_ratio < 1.0:
                n = int(self.negative_sampling_ratio * len(self.peak_seqs))
                keep = self.rng.choice(len(self.nonpeak_seqs), size=n, replace=False)
                self.seqs = np.vstack([cp, self.nonpeak_seqs[keep]])
                self.cts = np.vstack([cc, self.nonpeak_cts[keep]])
            else:
                self.seqs = np.vstack([cp, self.nonpeak_seqs])
                self.cts = np.vstack([cc, self.nonpeak_cts])
        elif self.peak_seqs is not None:
            self.seqs, self.cts = self._random_crop(self.peak_seqs, self.peak_cts)
        elif self.nonpeak_seqs is not None:
            # No copy: upstream aliases the stored arrays here, so the
            # in-place revcomp below accumulates across epochs. Kept as is.
            self.seqs = self.nonpeak_seqs
            self.cts = self.nonpeak_cts
        else:
            raise ValueError("both peak and non-peak arrays are empty")

        seqs, cts = self.seqs, self.cts
        if self.add_revcomp:
            pos = self.rng.choice(range(seqs.shape[0]), size=int(seqs.shape[0] * 0.5),
                                  replace=False)
            seqs[pos] = seqs[pos, ::-1, ::-1]
            cts[pos] = cts[pos, ::-1]
        if self.shuffle:
            perm = self.rng.permutation(seqs.shape[0])
            seqs, cts = seqs[perm], cts[perm]
        self.cur_seqs, self.cur_cts = seqs, cts

    def __getitem__(self, idx):
        sl = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        cts = self.cur_cts[sl]
        return self.cur_seqs[sl], cts, np.log(1 + cts.sum(-1, keepdims=True))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def on_epoch_end(self) -> None:
        self.crop_revcomp_data()
