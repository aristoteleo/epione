"""HINT-ATAC: bias-corrected ATAC signal, HMM footprints and differential TF activity.

A port of HINT-ATAC (Li, Schulz *et al.*, *Genome Biology* 2019) from the
Regulatory Genomics Toolbox (RGT 1.0.2, ``rgt-hint``; CostaLab/reg-gen,
GPL-3, the same licence as Epione). HINT works on bulk ATAC-seq BAM files
in four steps, and each one is a function here:

==========================  ===============================================
:class:`HintBias`           HINT's strand-specific 8-mer Tn5 bias tables
                            (k-mer counts estimated with a variable-order
                            Markov model). A :class:`~epione.tl.Tn5Bias`,
                            so it also plugs into the rest of Epione's
                            footprinting via ``tn5_bias(method="hint")``.
:func:`hint_signal`         Strand-specific cut counts, bias-corrected
                            signal and the normalised + slope tracks the
                            HMM reads, for one region.
:func:`hint_footprints`     ``rgt-hint footprinting --atac-seq``: Viterbi
                            decoding of the trained HMM (single-end model,
                            4 tracks; paired-end model, 8 tracks split at a
                            fragment length of 145 bp), footprint
                            post-processing and tag counts.
:func:`hint_match_motifs`   ``rgt-motifanalysis matching``: MOODS motif
                            matching in the footprints (MPBS).
:func:`hint_differential`   ``rgt-hint differential``: per-TF footprint
                            profiles, protection score, TC, TF activity,
                            z-score and p-value between two conditions.
==========================  ===============================================

Exactness
---------
Upstream is deterministic, and this port reproduces it rather than
approximating it (see ``tests/parity/hint_atac/MANIFEST.md``):

* The bias-corrected signal uses upstream's running sums in the same
  order of floating-point operations. They are *sequential* sums:
  ``sum()`` in Python >= 3.12 is compensated and would not match, so the
  kernels add explicitly.
* The 98th percentile is scipy 1.7's ``scoreatpercentile`` (``fraction``
  interpolation), inlined so the result does not depend on the installed
  scipy.
* The HMM log-likelihood is RGT's own full-covariance Gaussian density
  (Cholesky + triangular solve). Viterbi follows hmmlearn 0.2.2's
  ``_hmmc._viterbi``, including its first-maximum tie rule.
* ``differential`` accumulates in float32, as upstream does. The float32
  division by ``factor * n_sites`` reproduces numpy 1.x scalar casting,
  so the result is the same under numpy 2.

Upstream quirks are kept, because "matches rgt-hint" is the contract:

* the single-end model scales the reverse-strand slope with the standard
  deviation of the already-normalised *forward* slope;
* footprinting shifts reverse-strand cuts by -5 but ``differential`` by
  -4 (``read.aend + shift - 1``);
* the paired-end model silently skips a peak whose signal extraction
  fails (for instance within 29 bp of a chromosome start);
* ``differential`` does not orient the profile by motif strand.

Model files
-----------
The trained HMMs ship with Epione (``epione/data/hint``, exported from
RGT's pickles). The two ATAC bias tables (65,536 8-mers per strand) are
downloaded once from RGT's repository at a pinned commit, md5-checked,
and cached through the model registry (``hint_atac_bias_tables`` in
:func:`epione.models.list_pretrained`); set ``EPIONE_MODELS_DIR`` or
``EPIONE_DATA_DIR`` to choose where. Any
``rgt-hint estimation`` table pair can be passed instead.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from math import ceil, floor
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._tn5_bias import Tn5Bias, _fasta_path

__all__ = [
    "HintBias",
    "HintHMM",
    "HintDifferentialResult",
    "fetch_hint_file",
    "hint_differential",
    "hint_footprints",
    "hint_match_motifs",
    "hint_signal",
    "load_hint_bias_table",
    "load_hint_hmm",
]

PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------

#: CostaLab/reg-gen commit the model files are taken from (RGT 1.0.2).
RGT_COMMIT = "66f5fbb33f199424d09272c628287955e383f7bd"
_RGT_RAW = f"https://raw.githubusercontent.com/CostaLab/reg-gen/{RGT_COMMIT}/data/fp_hmms/"
_RGT_FILES = {
    "atac_bias_table_F.txt": "md5:75dcd7c1c21ae23ce45cb50d636e4ca9",
    "atac_bias_table_R.txt": "md5:0212670fc325736397ca85ac030c16ff",
}
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "hint"


def fetch_hint_file(name: str) -> str:
    """Path to one of RGT's HINT model files, downloaded on first use.

    Parameters
    ----------
    name
        ``'atac_bias_table_F.txt'`` or ``'atac_bias_table_R.txt'`` (about
        1 MB each).
    """
    if name not in _RGT_FILES:
        raise KeyError(f"unknown HINT file {name!r}; one of {sorted(_RGT_FILES)}")
    from epione.models.registry import get_pretrained

    return str(get_pretrained("hint_atac_bias_tables").fetch(name))


# ---------------------------------------------------------------------------
# Bias tables
# ---------------------------------------------------------------------------

_CODE = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate("ACGT"):
    _CODE[ord(_b)] = _i
_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def _revcomp(seq: str) -> str:
    # RGT's AuxiliaryFunctions.revcomp: complement A/C/G/T/N, reverse.
    return seq.translate(_COMP)[::-1]


def _kmer_codes(seq: str, k: int) -> np.ndarray:
    """2-bit index of every k-mer of ``seq`` (A,C,G,T = 0..3, first base most
    significant); -1 where the k-mer has a non-ACGT base."""
    n = len(seq) - k + 1
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    c = _CODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    bad = c < 0
    idx = np.zeros(n, dtype=np.int64)
    badk = np.zeros(n, dtype=bool)
    for t in range(k):
        idx = idx * 4 + np.where(bad[t:t + n], 0, c[t:t + n])
        badk |= bad[t:t + n]
    idx[badk] = -1
    return idx


@dataclass
class _KmerTable:
    """One strand's table: the ``{kmer: bias}`` dict upstream builds, plus a
    dense ``4**k`` array (NaN = absent) for vector lookups when every key is
    an ACGT k-mer of the same length."""

    table: Dict[str, float]
    k: int
    dense: Optional[np.ndarray]

    @classmethod
    def from_dict(cls, table: Mapping[str, float]) -> "_KmerTable":
        table = {str(k): float(v) for k, v in table.items()}
        if not table:
            raise ValueError("empty bias table")
        # RGT: k_nb = len(list(fBiasDict.keys())[0])
        k = len(next(iter(table)))
        dense = None
        if k <= 12 and all(len(key) == k and set(key) <= set("ACGT") for key in table):
            dense = np.full(4 ** k, np.nan)
            # the k-mers starting at 0, k, 2k... of the joined keys are the keys
            codes = _kmer_codes("".join(table), k)[::k]
            # fancy assignment keeps the last value of a repeated code, as
            # the dict did for a repeated key (the dict holds one per key)
            dense[codes] = np.fromiter(table.values(), dtype=np.float64, count=len(table))
        return cls(table, k, dense)

    def lookup(self, kmers: np.ndarray) -> np.ndarray:
        """Bias for k-mer codes, 1.0 (RGT's ``defaultKmerValue``) if absent."""
        out = np.ones(kmers.shape[0])
        ok = kmers >= 0
        vals = self.dense[kmers[ok]]
        vals[np.isnan(vals)] = 1.0
        out[ok] = vals
        return out


def load_hint_bias_table(path_f: PathLike, path_r: PathLike) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Read a forward / reverse bias table pair (``rgt-hint estimation``
    output: one ``kmer<TAB>bias`` per line), exactly as RGT's
    ``BiasTable.load_table`` does."""
    out = []
    for path in (path_f, path_r):
        tab: Dict[str, float] = {}
        with open(path) as fh:
            for line in fh:
                ll = line.strip().split("\t")
                tab[ll[0]] = float(ll[1])
        out.append(tab)
    return out[0], out[1]


class HintBias(Tn5Bias):
    """HINT-ATAC's strand-specific k-mer Tn5 bias.

    HINT corrects the forward- and reverse-strand cuts separately, each with
    its own table of k-mer cleavage preferences (8-mers centred on the cut;
    ``rgt-hint estimation`` fits them with a variable-order Markov model).

    Parameters
    ----------
    genome
        :class:`~epione.core.genome.Genome` or indexed FASTA path.
    table_f, table_r
        Forward / reverse table: a path to an ``rgt-hint estimation`` text
        file or a ``{kmer: bias}`` mapping. Default: HINT's published ATAC
        tables (downloaded once, md5-checked).

    Notes
    -----
    :meth:`track` returns the forward + reverse bias per base, which is what
    an unstranded consumer of :class:`~epione.tl.Tn5Bias` needs.
    :meth:`track_stranded` returns them separately.

    The cut convention: HINT counts a forward read's cut at
    ``reference_start + 5`` and a reverse read's at ``reference_end - 6``
    (footprinting shifts). On a fragments file that already carries the
    Tn5 +4/-5 adjustment those are the fragment ``start + 1`` and
    ``end - 1``, so :attr:`shift` is ``(1, -1)``.
    """

    name = "hint"
    shift = (1, -1)

    def __init__(self, genome, table_f=None, table_r=None) -> None:
        self.fasta = _fasta_path(genome)
        if (table_f is None) != (table_r is None):
            raise ValueError("pass both table_f and table_r, or neither")
        if table_f is None:
            table_f = fetch_hint_file("atac_bias_table_F.txt")
            table_r = fetch_hint_file("atac_bias_table_R.txt")
        if isinstance(table_f, (str, Path)):
            table_f, table_r = load_hint_bias_table(table_f, table_r)
        self.forward = _KmerTable.from_dict(table_f)
        self.reverse = _KmerTable.from_dict(table_r)
        # RGT reads k from the forward table only.
        self.k = self.forward.k

    # -- resources ---------------------------------------------------------

    def _fasta(self):
        if getattr(self, "_fa", None) is None:
            import pysam
            self._fa = pysam.FastaFile(self.fasta)
        return self._fa

    # -- the k-mer scan ----------------------------------------------------

    def _halves(self, paired: bool) -> Tuple[int, int]:
        k = self.k
        if paired:     # get_bc_signal_by_fragment_length: int(k/2) both sides
            h = int(k / 2.)
            return h, h
        return int(floor(k / 2.)), int(ceil(k / 2.))

    def kmer_bias(self, chrom: str, p1_w: int, p2_w: int, *, paired: bool = False,
                  fasta=None) -> Tuple[np.ndarray, np.ndarray]:
        """Forward / reverse bias for the cut sites ``[p1_w, p2_w)``.

        This is the scan inside RGT's ``bias_correction_atac``
        (``paired=False``) and ``get_bc_signal_by_fragment_length``
        (``paired=True``); the two differ only for odd ``k``. Near a
        chromosome end the result is shorter, as upstream's is.
        """
        fa = fasta if fasta is not None else self._fasta()
        a, b = self._halves(paired)
        p1_wk, p2_wk = p1_w - a, p2_w + b
        cur = str(fa.fetch(chrom, p1_wk, p2_wk - 1)).upper()
        rc = _revcomp(str(fa.fetch(chrom, p1_wk + 1, p2_wk)).upper())
        L = len(cur)
        k = self.k
        fast = (a == b and a + b == k and len(rc) == L and L >= k
                and self.forward.dense is not None and self.reverse.dense is not None
                and self.reverse.k == k)
        if fast:
            af = self.forward.lookup(_kmer_codes(cur, k))
            ar = self.reverse.lookup(_kmer_codes(rc, k)[::-1].copy())
            return af, ar
        # Literal transcription of the upstream loop (odd k, truncated
        # sequence at a chromosome end, or tables with non-ACGT keys).
        ft, rt = self.forward.table, self.reverse.table
        af, ar = [], []
        for i in range(b, L - a + 1):
            af.append(ft.get(cur[i - a:i + b], 1.0))
            ar.append(rt.get(rc[L - b - i:L + a - i], 1.0))
        return np.asarray(af, dtype=np.float64), np.asarray(ar, dtype=np.float64)

    def track_stranded(self, chrom: str, start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
        """Forward and reverse k-mer bias for cut sites ``[start, end)``."""
        if end <= start:
            raise ValueError(f"empty interval {chrom}:{start}-{end}")
        return self.kmer_bias(chrom, start, end)

    def track(self, chrom: str, start: int, end: int) -> np.ndarray:
        af, ar = self.track_stranded(chrom, start, end)
        out = np.ones(end - start)
        n = min(len(af), len(ar), end - start)
        out[:n] = af[:n] + ar[:n]
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"HintBias(k={self.k}, shift={self.shift})"


# ---------------------------------------------------------------------------
# Running-sum kernels (exact upstream operation order)
# ---------------------------------------------------------------------------

def _correct_py(n: np.ndarray, a: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    m = max(len(n) - 2 * half, 0)
    N = np.empty(m)
    s = 0.0
    for v in n[:window]:
        s += v
    last = n[0] if len(n) else 0.0
    for idx in range(m):
        i = idx + half
        N[idx] = s
        s -= last
        s += n[i + half]
        last = n[i - half + 1]
    out_len = max(len(a) - 2 * half, 0)
    out = np.empty(out_len)
    s = 0.0
    for v in a[:window]:
        s += v
    last = a[0] if len(a) else 0.0
    for idx in range(out_len):
        i = idx + half
        out[idx] = N[i - half] * (a[i] / s)
        s -= last
        s += a[i + half]
        last = a[i - half + 1]
    return out


try:  # numba gives the same IEEE operations (no fastmath), ~100x faster
    from numba import njit

    @njit(cache=True)
    def _correct_nb(n, a, window):  # pragma: no cover - same as _correct_py
        half = window // 2
        m = max(n.shape[0] - 2 * half, 0)
        N = np.empty(m)
        s = 0.0
        for j in range(min(window, n.shape[0])):
            s += n[j]
        last = n[0] if n.shape[0] > 0 else 0.0
        for idx in range(m):
            i = idx + half
            N[idx] = s
            s -= last
            s += n[i + half]
            last = n[i - half + 1]
        out_len = max(a.shape[0] - 2 * half, 0)
        out = np.empty(out_len)
        s = 0.0
        for j in range(min(window, a.shape[0])):
            s += a[j]
        last = a[0] if a.shape[0] > 0 else 0.0
        for idx in range(out_len):
            i = idx + half
            out[idx] = N[i - half] * (a[i] / s)
            s -= last
            s += a[i + half]
            last = a[i - half + 1]
        return out

    @njit(cache=True)
    def _viterbi_nb(log_start, log_trans, flp):  # pragma: no cover
        T, K = flp.shape
        lat = np.zeros((T, K))
        buf = np.empty(K)
        for i in range(K):
            lat[0, i] = log_start[i] + flp[0, i]
        for t in range(1, T):
            for i in range(K):
                for j in range(K):
                    buf[j] = log_trans[j, i] + lat[t - 1, j]
                best = -np.inf
                pos = 0
                for j in range(K):
                    if buf[j] > best:
                        best = buf[j]
                        pos = j
                lat[t, i] = buf[pos] + flp[t, i]
        states = np.empty(T, dtype=np.int64)
        best = -np.inf
        where = 0
        for i in range(K):
            if lat[T - 1, i] > best:
                best = lat[T - 1, i]
                where = i
        states[T - 1] = where
        for t in range(T - 2, -1, -1):
            for i in range(K):
                buf[i] = lat[t, i] + log_trans[i, where]
            best = -np.inf
            pos = 0
            for i in range(K):
                if buf[i] > best:
                    best = buf[i]
                    pos = i
            where = pos
            states[t] = where
        return states

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is optional
    _HAS_NUMBA = False


def _correct(n, a, window: int = 50) -> np.ndarray:
    """RGT's bias correction of one strand: ``N_w(i) * a(i) / sum_w(a)`` with
    both window sums (width ``window``, centred) kept as running sums."""
    n = np.ascontiguousarray(n, dtype=np.float64)
    a = np.ascontiguousarray(a, dtype=np.float64)
    if _HAS_NUMBA:
        return _correct_nb(n, a, window)
    return _correct_py(n, a, window)


def _argmax_first(x: np.ndarray) -> int:
    # hmmlearn's _argmax: strict '>' from -inf, so ties keep the first and an
    # all -inf/NaN row gives 0.
    best, pos = -np.inf, 0
    for i, v in enumerate(x):
        if v > best:
            best, pos = v, i
    return pos


def _viterbi_py(log_start, log_trans, flp):
    T, K = flp.shape
    lat = np.zeros((T, K))
    lat[0] = log_start + flp[0]
    for t in range(1, T):
        work = log_trans + lat[t - 1][:, None]          # work[j, i]
        best = np.array([work[_argmax_first(work[:, i]), i] for i in range(K)])
        lat[t] = best + flp[t]
    states = np.empty(T, dtype=np.int64)
    where = _argmax_first(lat[T - 1])
    states[T - 1] = where
    for t in range(T - 2, -1, -1):
        where = _argmax_first(lat[t] + log_trans[:, where])
        states[t] = where
    return states


# ---------------------------------------------------------------------------
# Normalisation (GenomicSignal.boyle_norm / hon_norm_atac / slope)
# ---------------------------------------------------------------------------

def _percentile(x: np.ndarray, per: float) -> float:
    """scipy 1.7 ``scoreatpercentile(x, per)`` ('fraction' interpolation)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.nan
    s = np.sort(x)
    idx = per / 100. * (s.shape[0] - 1)
    i = int(idx)
    if i == idx:
        return np.add.reduce(s[i:i + 1] * np.array(1)) / 1.0
    j = i + 1
    w = np.array([(j - idx), (idx - i)], float)
    return np.add.reduce(s[i:i + 2] * w) / w.sum()


def _boyle(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    pos = x[x > 0]
    if pos.size == 0:       # upstream: mean of empty slice is NaN -> unchanged
        return x
    return x / pos.mean()


def _hon(x, mean, std) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if std == 0:
        return x
    out = x.copy()
    nz = x != 0
    with np.errstate(over="ignore"):
        out[nz] = 1.0 / (1.0 + np.exp(-(x[nz] - mean) / std))
    return out


def savitzky_golay_coefficients(window_size: int, order: int = 2, deriv: int = 1) -> np.ndarray:
    """RGT's Savitzky-Golay coefficients (reversed derivative row of the
    Vandermonde pseudo-inverse)."""
    window_size, order = abs(int(window_size)), abs(int(order))
    half = (window_size - 1) // 2
    b = np.array([[k ** i for i in range(order + 1)] for k in range(-half, half + 1)])
    return np.linalg.pinv(b)[deriv][::-1]


def _slope(x, sg: np.ndarray) -> np.ndarray:
    full = np.convolve(np.asarray(x, dtype=np.float64), sg)
    h = len(sg) // 2
    return full[h:len(full) - h]


def _norm_hon(x, per: float) -> np.ndarray:
    x = _boyle(x)
    return _hon(x, _percentile(x, per), np.asarray(x).std())


# ---------------------------------------------------------------------------
# Cut counts
# ---------------------------------------------------------------------------

def _cut_counts(bam, chrom: str, lo: int, hi: int, fwd: int, rev: int,
                min_length: Optional[int] = None, max_length: Optional[int] = None):
    """Strand-specific Tn5 cuts in ``[lo, hi)``.

    A forward read cuts at ``reference_start + fwd``, a reverse read at
    ``reference_end + rev - 1``. ``min_length < |TLEN| <= max_length``
    selects fragments, as in RGT's ``get_bc_signal_by_fragment_length``.
    """
    nf = np.zeros(hi - lo)
    nr = np.zeros(hi - lo)
    for read in bam.fetch(chrom, lo, hi):
        if read.is_unmapped:
            continue
        if min_length is not None or max_length is not None:
            tl = abs(read.template_length)
            if min_length is not None and not tl > min_length:
                continue
            if max_length is not None and not tl <= max_length:
                continue
        if not read.is_reverse:
            c = read.reference_start + fwd
            if lo <= c < hi:
                nf[c - lo] += 1.0
        else:
            c = read.reference_end + rev - 1
            if lo <= c < hi:
                nr[c - lo] += 1.0
    return nf, nr


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------

@dataclass
class HintHMM:
    """A trained HINT footprinting HMM (Gaussian emissions, full covariance).

    Attributes
    ----------
    startprob, transmat, means, covars
        Parameters as hmmlearn stores them.
    fp_state
        The footprint state (``rgt-hint`` default: 6 for the single-end
        model, 8 for the paired-end one).
    """

    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    fp_state: int
    name: str = "custom"

    def __post_init__(self):
        from scipy import linalg
        self.startprob = np.asarray(self.startprob, dtype=np.float64)
        self.transmat = np.asarray(self.transmat, dtype=np.float64)
        self.means = np.asarray(self.means, dtype=np.float64)
        self.covars = np.asarray(self.covars, dtype=np.float64)
        n_dim = self.means.shape[1]
        self._chol = []
        for cv in self.covars:
            try:
                c = linalg.cholesky(cv, lower=True)
            except linalg.LinAlgError:
                c = linalg.cholesky(cv + 1.e-7 * np.eye(n_dim), lower=True)
            self._chol.append(c)
        with np.errstate(divide="ignore"):
            self._log_start = np.log(self.startprob)
            self._log_trans = np.log(self.transmat)

    @property
    def n_states(self) -> int:
        return self.means.shape[0]

    @property
    def n_features(self) -> int:
        return self.means.shape[1]

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Per-frame log density of every state (RGT
        ``_log_multivariate_normal_density_full``)."""
        from scipy import linalg
        n_samples, n_dim = X.shape
        log_prob = np.empty((n_samples, self.n_states))
        for c, (mu, cv_chol) in enumerate(zip(self.means, self._chol)):
            cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
            cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
            log_prob[:, c] = - .5 * (np.sum(np.array(cv_sol) ** 2, axis=1) +
                                     n_dim * np.log(2 * np.pi) + cv_log_det)
        return log_prob

    def decode(self, X) -> np.ndarray:
        """Viterbi path (hmmlearn 0.2.2 ``predict``). Raises ``ValueError``
        where hmmlearn's input check would: non-2-D, empty or non-finite."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != self.n_features:
            raise ValueError(f"HMM input has shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("HMM input contains NaN or inf")
        flp = self.log_likelihood(X)
        if _HAS_NUMBA:
            return _viterbi_nb(self._log_start, self._log_trans, flp)
        return _viterbi_py(self._log_start, self._log_trans, flp)


def load_hint_hmm(model: str = "paired") -> HintHMM:
    """One of HINT's trained ATAC HMMs: ``'single'`` (``atac_single.pkl``,
    7 states x 4 tracks) or ``'paired'`` (``atac_paired.pkl``, 9 x 8)."""
    if model not in ("single", "paired"):
        raise ValueError("model must be 'single' or 'paired'")
    return _hmm_from_npz(_DATA_DIR / f"atac_{model}_hmm.npz", name=f"atac_{model}")


def _hmm_from_npz(path, name=None) -> HintHMM:
    z = np.load(path)
    if name is None:
        name = Path(path).name.replace("_hmm.npz", "")
    return HintHMM(z["startprob"], z["transmat"], z["means"], z["covars"],
                   int(z["fp_state"]), name=name)


# ---------------------------------------------------------------------------
# Signal for one region
# ---------------------------------------------------------------------------

_WINDOW = 50


def _single_tracks(bam, fasta, bias: HintBias, chrom, p1, p2, fwd, rev, sg, per_norm=98.0):
    """``GenomicSignal.get_signal_atac`` (single-end model)."""
    a, b = bias._halves(paired=False)
    p1_w, p2_w = p1 - _WINDOW // 2, p2 + _WINDOW // 2
    p2_wk = p2_w + b
    out = {}
    if p1 <= 0 or p1_w <= 0 or p2_wk <= 0:
        bc_f, bc_r = _cut_counts(bam, chrom, p1, p2, fwd, rev)
    else:
        nf, nr = _cut_counts(bam, chrom, p1_w, p2_w, fwd, rev)
        af, ar = bias.kmer_bias(chrom, p1_w, p2_w, paired=False, fasta=fasta)
        out.update(raw_f=nf, raw_r=nr, bias_f=af, bias_r=ar)
        bc_f, bc_r = _correct(nf, af), _correct(nr, ar)
    out.update(bc_f=bc_f, bc_r=bc_r)
    hf = _norm_hon(bc_f, per_norm)
    hr = _norm_hon(bc_r, per_norm)
    sf, sr = _slope(hf, sg), _slope(hr, sg)
    sf = _hon(sf, _percentile(sf, per_norm), np.std(sf))
    # upstream quirk: the reverse slope is scaled by the std of the
    # already-normalised forward slope.
    sr = _hon(sr, _percentile(sr, per_norm), np.std(sf))
    out["X"] = np.array([hf, sf, hr, sr]).T
    return out


def _paired_tracks(bam, fasta, bias: HintBias, chrom, p1, p2, fwd, rev, sg, per_norm=98.0):
    """The paired-end model's input: bias-corrected f/r signal of fragments
    <= 145 bp and > 145 bp, each Boyle/Hon-normalised plus its slope.
    Returns ``None`` where upstream skips the peak."""
    a, b = bias._halves(paired=True)
    p1_w, p2_w = p1 - _WINDOW // 2, p2 + _WINDOW // 2
    if p1 <= 0 or p1_w <= 0 or p2_w + b <= 0:
        return None     # upstream returns one list, the unpacking fails
    try:
        af, ar = bias.kmer_bias(chrom, p1_w, p2_w, paired=True, fasta=fasta)
        out = {"bias_f": af, "bias_r": ar}
        cols = []
        for tag, lo, hi in (("max145", None, 145), ("min145", 145, None)):
            nf, nr = _cut_counts(bam, chrom, p1_w, p2_w, fwd, rev, lo, hi)
            bf, br = _correct(nf, af), _correct(nr, ar)
            out.update({f"raw_f_{tag}": nf, f"raw_r_{tag}": nr,
                        f"bc_f_{tag}": bf, f"bc_r_{tag}": br})
        for tag in ("max145", "min145"):
            for strand in ("f", "r"):
                h = _norm_hon(out[f"bc_{strand}_{tag}"], per_norm)
                cols += [h, _slope(h, sg)]
    except Exception:
        return None
    out["X"] = np.array(cols).T
    return out


def _open(bam, fasta):
    import pysam
    b = pysam.AlignmentFile(str(bam), "rb") if isinstance(bam, (str, Path)) else bam
    f = pysam.FastaFile(str(fasta)) if isinstance(fasta, (str, Path)) else fasta
    return b, f


def _detect_paired(bam_path, n: int = 1000) -> bool:
    import pysam
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for i, read in enumerate(bam.fetch(until_eof=True)):
            if read.is_paired:
                return True
            if i >= n:
                break
    return False


def hint_signal(
    bam: PathLike,
    chrom: str,
    start: int,
    end: int,
    genome=None,
    *,
    bias: Optional[HintBias] = None,
    paired_end: bool = True,
    forward_shift: int = 5,
    reverse_shift: int = -5,
    sg_window_size: int = 9,
    norm_per: float = 98.0,
) -> Dict[str, np.ndarray]:
    """HINT-ATAC's strand-specific signal for one region.

    Parameters
    ----------
    bam
        Indexed ATAC-seq BAM.
    chrom, start, end
        Region (0-based, half-open).
    genome
        :class:`~epione.core.genome.Genome` or FASTA path; may be omitted if
        ``bias`` is given.
    bias
        :class:`HintBias`; default HINT's ATAC tables on ``genome``.
    paired_end
        Build the paired-end model's tracks (fragments split at 145 bp)
        instead of the single-end ones.
    forward_shift, reverse_shift
        Cut position relative to a forward read's start / reverse read's end
        (``rgt-hint footprinting`` defaults).

    Returns
    -------
    dict of arrays
        ``raw_*`` cut counts over the region +/- 25 bp, ``bias_*`` k-mer
        bias over the same span, ``bc_*`` bias-corrected signal over the
        region, and ``X``, the HMM input (columns: single-end ``norm_f,
        slope_f, norm_r, slope_r``; paired-end the same for <= 145 bp, then
        for > 145 bp). Paired-end keys carry a ``_max145`` / ``_min145``
        suffix. Empty if HINT would skip the region.
    """
    if bias is None:
        if genome is None:
            raise ValueError("pass genome= or bias=")
        bias = HintBias(genome)
    b, f = _open(bam, bias.fasta)
    sg = savitzky_golay_coefficients(sg_window_size, 2, 1)
    fn = _paired_tracks if paired_end else _single_tracks
    out = fn(b, f, bias, chrom, int(start), int(end), forward_shift, reverse_shift, sg, norm_per)
    return out or {}


# ---------------------------------------------------------------------------
# Footprinting
# ---------------------------------------------------------------------------

def _read_bed(path_or_df) -> pd.DataFrame:
    """Regions as RGT's ``GenomicRegionSet.read`` sees them: swapped if
    start > end, zero-length and malformed lines skipped, then sorted by
    (chrom, start, end)."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
        cols = list(df.columns)
        ren = dict(zip(cols[:3], ["chrom", "start", "end"]))
        df = df.rename(columns=ren)
        rows = []
        for r in df.itertuples(index=False):
            d = r._asdict()
            rows.append((str(d["chrom"]), int(d["start"]), int(d["end"]),
                         d.get("name"), d.get("score"), d.get("strand")))
    else:
        rows = []
        with open(path_or_df) as fh:
            for line in fh:
                p = line.strip("\n").split()
                if len(p) < 3:
                    continue
                try:
                    s, e = int(p[1]), int(p[2])
                except ValueError:
                    continue
                name = p[3] if len(p) > 3 else None
                strand = p[5] if len(p) > 5 else None
                score = p[4] if len(p) > 4 else None
                rows.append((p[0], s, e, name, score, strand))
    out = []
    for chrom, s, e, name, score, strand in rows:
        if s > e:
            s, e = e, s
        if s == e:
            continue
        out.append((chrom, s, e, name, score, strand))
    df = pd.DataFrame(out, columns=["chrom", "start", "end", "name", "score", "strand"])
    # Python's stable sort on (chrom, initial, final), as GenomicRegion.__lt__
    order = sorted(range(len(df)), key=lambda i: (out[i][0], out[i][1], out[i][2]))
    return df.iloc[order].reset_index(drop=True)


def _calls_from_states(chrom, initial, final, states, fp_state, fp_max_size):
    fps = []
    start_pos, flag = 0, False
    n = len(states)
    for idx in range(n):
        k = initial + idx
        if flag:
            if states[idx] != fp_state:
                if k - start_pos < fp_max_size:
                    fps.append((chrom, start_pos, k))
                flag = False
        elif states[idx] == fp_state:
            flag, start_pos = True, k
    if flag and initial + n - start_pos < fp_max_size:
        fps.append((chrom, start_pos, final))      # upstream uses region.final
    return fps


def _footprint_chunk(args):
    (bam_path, bias, hmm, regions, paired, fwd, rev, sg_window, norm_per,
     fp_max_size, return_states) = args
    import pysam
    bam = pysam.AlignmentFile(str(bam_path), "rb")
    fasta = pysam.FastaFile(bias.fasta)
    sg = savitzky_golay_coefficients(sg_window, 2, 1)
    fps, states_out, skipped = [], [], []
    for chrom, s, e in regions:
        if paired:
            tr = _paired_tracks(bam, fasta, bias, chrom, s, e, fwd, rev, sg, norm_per)
        else:
            tr = _single_tracks(bam, fasta, bias, chrom, s, e, fwd, rev, sg, norm_per)
        st = None
        if tr is not None:
            try:
                st = hmm.decode(tr["X"])
            except ValueError:
                st = None
        if st is None:
            skipped.append((chrom, s, e))
            states_out.append(None)
            continue
        if return_states:
            states_out.append(st)
        fps.extend(_calls_from_states(chrom, s, e, st, hmm.fp_state, fp_max_size))
    return fps, states_out, skipped


def _intersect_original(a: List[list], b: List[tuple]) -> List[list]:
    """librgt ``intersectGenomicRegionSetsOriginal``: the two-pointer walk
    over sorted sets that keeps regions of ``a`` overlapping ``b``."""
    if not a or not b:
        return []
    out = []
    i = j = 0
    while True:
        ca, sa, ea = a[i][0], a[i][1], a[i][2]
        cb, sb, eb = b[j]
        if ca == cb and ((sa <= sb and ea > sb) or (sa > sb and sa < eb)):
            out.append(list(a[i]))
            if i < len(a) - 1:
                i += 1
            else:
                break
        else:
            if (ca, sa, ea) > (cb, sb, eb):
                if j < len(b) - 1:
                    j += 1
                else:
                    break
            else:
                if i < len(a) - 1:
                    i += 1
                else:
                    break
    return out


def _merge(regs: List[list]) -> List[list]:
    """``GenomicRegionSet.merge``: fuse regions that overlap (not those that
    only touch), in the current order."""
    if len(regs) < 2:
        return regs
    out = []
    prev = regs[0]
    for cur in regs[1:]:
        if prev[0] == cur[0] and (
                (prev[1] <= cur[1] and prev[2] > cur[1]) or (prev[1] > cur[1] and prev[1] < cur[2])):
            prev[1] = min(prev[1], cur[1])
            prev[2] = max(prev[2], cur[2])
        else:
            out.append(prev)
            prev = cur
    out.append(prev)
    return out


def hint_footprints(
    bam: PathLike,
    peaks,
    genome=None,
    *,
    paired_end: Optional[bool] = None,
    bias: Optional[HintBias] = None,
    hmm: Optional[HintHMM] = None,
    fp_state: Optional[int] = None,
    forward_shift: int = 5,
    reverse_shift: int = -5,
    sg_window_size: int = 9,
    norm_per: float = 98.0,
    fp_max_size: int = 50,
    fp_min_size: int = 5,
    fp_ext: int = 5,
    tc_ext: int = 100,
    output: Optional[PathLike] = None,
    n_jobs: int = 1,
    return_states: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Call TF footprints with HINT-ATAC (``rgt-hint footprinting --atac-seq``).

    For each peak, the bias-corrected strand-specific cut signal is
    normalised, its slope taken, and a trained HMM decodes every base; runs
    of the footprint state shorter than ``fp_max_size`` are footprints.
    Footprints are then merged, those shorter than ``fp_min_size`` extended
    by ``fp_ext`` on both sides, merged again, and given the number of reads
    within ``tc_ext`` bp of their centre.

    Parameters
    ----------
    bam
        Indexed ATAC-seq BAM (reads, not fragments: HINT uses the read
        strand and, for the paired-end model, the template length).
    peaks
        BED path or DataFrame (chrom, start, end) of the regions to search.
    genome
        :class:`~epione.core.genome.Genome` or FASTA path (may be omitted if
        ``bias`` is given).
    paired_end
        Use HINT's paired-end model (8 tracks: fragments <= 145 bp and
        > 145 bp). ``None`` (default) picks it when the BAM holds paired
        reads, which is what HINT's documentation recommends; ``rgt-hint``
        itself defaults to the single-end model unless ``--paired-end``.
    bias
        :class:`HintBias`; default HINT's ATAC tables.
    hmm, fp_state
        A custom :class:`HintHMM` (e.g. from ``rgt-hint training``) and its
        footprint state; default the trained ATAC model for the chosen mode.
    forward_shift, reverse_shift, sg_window_size, norm_per, fp_max_size, fp_min_size, fp_ext, tc_ext
        ``rgt-hint`` defaults.
    output
        If given, write the BED (and a ``.info`` file) in ``rgt-hint``'s
        format.
    n_jobs
        Worker processes (peaks are independent; the result does not depend
        on it).
    return_states
        Also return the per-peak Viterbi paths in ``.attrs['states']``.

    Returns
    -------
    pandas.DataFrame
        ``chrom, start, end, name, tag_count``, sorted. ``.attrs`` holds the
        numbers ``rgt-hint`` writes to its ``.info`` file, the model used,
        and the skipped peaks.

    Examples
    --------
    >>> fp = epi.tl.hint_footprints('Bcell.bam', 'peaks.bed', 'genome.fa')
    """
    import pysam
    if bias is None:
        if genome is None:
            raise ValueError("pass genome= or bias=")
        bias = HintBias(genome)
    if paired_end is None:
        paired_end = _detect_paired(bam)
    if hmm is None:
        hmm = load_hint_hmm("paired" if paired_end else "single")
    if fp_state is not None:
        hmm = HintHMM(hmm.startprob, hmm.transmat, hmm.means, hmm.covars,
                      int(fp_state), name=hmm.name)
    expected = 8 if paired_end else 4
    if hmm.n_features != expected:
        raise ValueError(f"the {'paired' if paired_end else 'single'}-end model needs an "
                         f"HMM over {expected} tracks; this one has {hmm.n_features}")
    bam_path = str(bam)
    if not os.path.exists(bam_path + ".bai") and not os.path.exists(
            os.path.splitext(bam_path)[0] + ".bai"):
        pysam.index(bam_path)

    peaks_df = _read_bed(peaks)
    regions = list(zip(peaks_df["chrom"], peaks_df["start"].astype(int), peaks_df["end"].astype(int)))
    if verbose:
        print(f"HINT-ATAC footprinting: {len(regions)} peaks, "
              f"{'paired' if paired_end else 'single'}-end model ({hmm.name})")

    n_jobs = max(1, int(n_jobs))
    chunks = [regions[i::n_jobs] for i in range(n_jobs)] if n_jobs > 1 else [regions]
    args = [(bam_path, bias, hmm, ch, paired_end, forward_shift, reverse_shift,
             sg_window_size, norm_per, fp_max_size, return_states) for ch in chunks]
    if n_jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(n_jobs) as pool:
            parts = pool.map(_footprint_chunk, args)
    else:
        parts = [_footprint_chunk(a) for a in args]
    fps, skipped = [], []
    states = [None] * len(regions)
    for c, (f_, st_, sk_) in enumerate(parts):
        fps.extend(f_)
        skipped.extend(sk_)
        if return_states:
            for m, st in enumerate(st_):
                states[c + m * n_jobs if n_jobs > 1 else m] = st

    # --- post-processing (Footprinting.post_processing) -------------------
    fps = sorted([list(f) for f in fps], key=lambda r: (r[0], r[1], r[2]))
    fps = _intersect_original(fps, regions)
    fps.sort(key=lambda r: (r[0], r[1], r[2]))
    fps = _merge(fps)
    for f in fps:
        if f[2] - f[1] < fp_min_size:
            f[1] = max(0, f[1] - fp_ext)
            f[2] = f[2] + fp_ext
    fps = _merge(fps)

    bamf = pysam.AlignmentFile(bam_path, "rb")
    sizes = dict(zip(bamf.references, bamf.lengths))
    fa = pysam.FastaFile(bias.fasta)
    sizes.update(dict(zip(fa.references, fa.lengths)))   # RGT reads the genome's chrom.sizes

    def _tc(chrom, a, b):
        try:
            # pysam truncates float coordinates; upstream passes floats.
            return sum(1 for _ in bamf.fetch(reference=chrom, start=int(a), end=int(b)))
        except Exception:
            return 0

    tcs = []
    for chrom, s, e in fps:
        mid = (s + e) / 2
        tcs.append(int(_tc(chrom, max(mid - tc_ext, 0), min(mid + tc_ext, sizes[chrom]))))
    out = pd.DataFrame(fps, columns=["chrom", "start", "end"])
    out["name"] = [f"{c}:{s}-{e}" for c, s, e in fps]
    out["tag_count"] = np.asarray(tcs, dtype=np.int64)

    n_tc = 0.0
    for chrom, s, e in regions:
        n_tc += _tc(chrom, s, e)
    out.attrs.update({
        "n_reads": int(pysam.AlignmentFile(bam_path, "rb").count(until_eof=True)),
        "n_peaks": len(regions),
        "n_tag_counts_in_peaks": n_tc,
        "n_footprints": len(out),
        "model": hmm.name,
        "paired_end": bool(paired_end),
        "skipped_peaks": skipped,
    })
    if return_states:
        out.attrs["states"] = {f"{c}:{s}-{e}": st for (c, s, e), st in zip(regions, states)}
    if output is not None:
        write_hint_bed(out, output)
        info = os.path.splitext(str(output))[0] + ".info"
        with open(info, "w") as fh:
            fh.write(f"Number of reads: {out.attrs['n_reads']}\n")
            fh.write(f"Number of peaks: {out.attrs['n_peaks']}\n")
            fh.write(f"Number of tag counts within peaks: {out.attrs['n_tag_counts_in_peaks']}\n")
            fh.write(f"Number of footprints: {out.attrs['n_footprints']}\n")
    if verbose:
        print(f"  {len(out)} footprints; {len(skipped)} peaks skipped")
    return out


def write_hint_bed(fp: pd.DataFrame, path: PathLike) -> None:
    """Write footprints in ``rgt-hint``'s BED layout."""
    with open(path, "w") as fh:
        for r in fp.itertuples(index=False):
            fh.write(f"{r.chrom}\t{r.start}\t{r.end}\t{r.name}\t{r.tag_count}\t.\t\n")


# ---------------------------------------------------------------------------
# Motif matching (rgt-motifanalysis matching)
# ---------------------------------------------------------------------------

def _read_motifs(motifs) -> List[Tuple[str, List[List[float]]]]:
    """Motifs as ``(name, 4 x L counts)``. Accepts a JASPAR file (names
    ``<matrix id>.<name>``, as RGT's ``jaspar_vertebrates`` folder names
    them), a directory of RGT ``.pwm`` files, or a mapping name -> matrix."""
    if isinstance(motifs, Mapping):
        return [(str(k), [list(map(float, row)) for row in np.asarray(v, dtype=float)])
                for k, v in motifs.items()]
    p = Path(motifs)
    out = []
    if p.is_dir():
        from MOODS import parsers
        for f in sorted(p.glob("*.pwm")):
            out.append((".".join(f.name.split(".")[:-1]), parsers.pfm(str(f))))
        return out
    import re
    name, rows = None, []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out.append((name, rows))
                parts = line[1:].split(None, 1)
                name = parts[0] if len(parts) == 1 else f"{parts[0]}.{parts[1].strip()}"
                rows = []
            else:
                body = line.split("[", 1)[1] if "[" in line else line
                rows.append([float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body)])
    if name is not None:
        out.append((name, rows))
    return out


def hint_match_motifs(
    regions,
    motifs,
    genome,
    *,
    fpr: float = 1e-4,
    pseudocounts: float = 1.0,
    remove_strand_duplicates: bool = False,
    rmdup: bool = False,
    output: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Motif-predicted binding sites (MPBS) in footprints, as
    ``rgt-motifanalysis matching`` finds them.

    Each PFM becomes a log-odds PSSM (flat background, ``pseudocounts``,
    log2), its threshold the score at false-positive rate ``fpr``, and
    MOODS scans both strands of every region.

    Parameters
    ----------
    regions
        BED path or DataFrame, typically :func:`hint_footprints` output.
    motifs
        JASPAR file, directory of RGT ``.pwm`` files, or mapping
        name -> 4 x L count matrix.
    genome
        :class:`~epione.core.genome.Genome` or FASTA path.
    remove_strand_duplicates, rmdup
        As the ``rgt-motifanalysis`` options of the same name.
    output
        Write the MPBS BED here.

    Returns
    -------
    pandas.DataFrame
        ``chrom, start, end, name, score, strand``.
    """
    try:
        from MOODS import scan, tools
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError("hint_match_motifs needs MOODS (pip install MOODS-python)") from exc
    import pysam

    mlist = []
    for name, pfm in _read_motifs(motifs):
        bg = tools.flat_bg(len(pfm))
        pssm = tools.log_odds(pfm, bg, pseudocounts, 2)
        thr = tools.threshold_from_p(pssm, bg, fpr)
        mlist.append((name, pssm, tools.reverse_complement(pssm), len(pfm[0]), thr))
    scanner = scan.Scanner(7)
    pssms, thrs = [], []
    for name, pssm, rc, L, thr in mlist:
        pssms += [pssm, rc]
        thrs += [thr, thr]
    scanner.set_motifs(pssms, tools.flat_bg(4), thrs)

    reg = _read_bed(regions)
    reg = reg[["chrom", "start", "end"]]
    if rmdup:
        reg = reg.drop_duplicates()
    fa = pysam.FastaFile(_fasta_path(genome))
    rows = []
    for chrom, s, e in reg.itertuples(index=False):
        seq = str(fa.fetch(chrom, int(s), int(e)))
        hits = []
        for i, res in enumerate(scanner.scan(seq)):
            name, _, _, L, _ = mlist[i // 2]
            strand = "-" if i % 2 == 1 else "+"
            for r in res:
                p1 = int(s) + r.pos
                hits.append((chrom, p1, p1 + L, name, r.score, strand))
        if remove_strand_duplicates and len(hits) > 1:
            hits = _remove_strand_duplicates(hits)
        rows.extend(hits)
    out = pd.DataFrame(rows, columns=["chrom", "start", "end", "name", "score", "strand"])
    if output is not None:
        with open(output, "w") as fh:
            for r in rows:
                fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]!r}\t{r[5]}\t\n")
    return out


def _remove_strand_duplicates(hits):
    # rgt-motifanalysis --remove-strand-duplicates, including its loop bounds
    seqs = sorted(hits, key=lambda h: (h[0], h[1], h[2]))
    new, cur, end = [], 0, len(seqs) - 1
    while cur < end:
        gr = seqs[cur]
        nxt = cur + 1
        while nxt < end:
            gr2 = seqs[nxt]
            if (gr[3] != gr2[3] or gr[0] != gr2[0] or gr[1] != gr2[1]
                    or gr[2] != gr2[2] or gr[5] == gr2[5]):
                break
            if float(gr[4]) < float(gr2[4]):
                gr = gr2
            nxt += 1
        new.append(gr)
        cur = nxt
    if cur == end:
        new.append(seqs[cur])
    return new


# ---------------------------------------------------------------------------
# Differential footprinting (rgt-hint differential)
# ---------------------------------------------------------------------------

@dataclass
class HintDifferentialResult:
    """Output of :func:`hint_differential`.

    Attributes
    ----------
    stats
        One row per motif: ``Num``, ``Protection_Score_<c>``, ``TC_<c>`` and,
        for two conditions, ``TF_Activity``, ``Z_score``, ``P_values`` (as
        ``<prefix>_statistics.txt``) plus ``log2FC`` (the volcano plot's
        fold change of activity scores). Sorted by ``P_values``.
    profiles
        Motif -> ``(n_conditions, window_size)`` float32 normalised
        footprint profiles (the ``Lineplots/*.txt`` values).
    factors
        Per-condition normalisation factor.
    conditions, window_size, motif_len
        Run parameters; motif length per motif.
    """

    stats: pd.DataFrame
    profiles: Dict[str, np.ndarray]
    factors: pd.Series
    conditions: List[str]
    window_size: int
    motif_len: Dict[str, int]
    sites: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def significant(self, fdr: float = 0.05) -> pd.DataFrame:
        """Motifs with ``P_values < fdr`` (what upstream highlights)."""
        return self.stats[self.stats["P_values"] < fdr]


def _bc_site_signal(bam, fasta, bias: HintBias, chrom, p1, p2, fwd, rev):
    """``DifferentialAnalysis.bias_correction``: forward + reverse
    bias-corrected signal over ``[p1, p2)``."""
    a, b = bias._halves(paired=False)
    p1_w, p2_w = p1 - _WINDOW // 2, p2 + _WINDOW // 2
    if p1 <= 0 or p1_w <= 0 or p1_w - a <= 0 or p2_w + b <= 0:
        nf, nr = _cut_counts(bam, chrom, p1, p2, fwd, rev)
        # upstream adds both strands into one list
        return nf + nr
    nf, nr = _cut_counts(bam, chrom, p1_w, p2_w, fwd, rev)
    af, ar = bias.kmer_bias(chrom, p1_w, p2_w, paired=False, fasta=fasta)
    bf, br = _correct(nf, af), _correct(nr, ar)
    n = min(len(bf), len(br))
    return bf[:n] + br[:n]


def _motif_signal(args):
    (bam_path, fasta_path, bias, sites, window, fwd, rev, bc) = args
    import pysam
    bam = pysam.AlignmentFile(str(bam_path), "rb")
    fasta = pysam.FastaFile(fasta_path) if bc else None
    signal = np.zeros(window)
    try:
        for chrom, s, e in sites:
            mid = (e + s) // 2
            p1, p2 = mid - window // 2, mid + window // 2
            if p1 <= 0:
                continue
            if bc:
                sig = _bc_site_signal(bam, fasta, bias, chrom, p1, p2, fwd, rev)
                if len(sig) != window:
                    continue
                signal = np.add(signal, np.array(sig))
            else:
                nf, nr = _cut_counts(bam, chrom, p1, p2, fwd, rev)
                signal += nf + nr
    except Exception:
        # upstream logs the exception and leaves this motif's row at zero
        return np.zeros(window), False
    return signal, True


def _site_pwm(fasta, sites: pd.DataFrame, window: int) -> pd.DataFrame:
    """Base counts around the sites, motif-oriented (upstream ``get_pwm``,
    used for the line-plot logo)."""
    pwm = {b: np.zeros(window) for b in "ACGTN"}
    aux = 1 if window % 2 else 0
    for chrom, s, e, strand in sites[["chrom", "start", "end", "strand"]].itertuples(index=False):
        mid = (s + e) // 2
        p1, p2 = mid - window // 2, mid + window // 2
        if p1 <= 0:
            continue
        if strand == "+":
            seq = str(fasta.fetch(chrom, p1, p2)).upper()
        elif strand == "-":
            seq = _revcomp(str(fasta.fetch(chrom, p1 + aux, p2 + aux)).upper())
        else:
            continue
        for i, ch in enumerate(seq):
            if ch in pwm:
                pwm[ch][i] += 1
    return pd.DataFrame({b: pwm[b] for b in "ACGT"})


def hint_differential(
    mpbs,
    bams: Sequence[PathLike],
    genome=None,
    *,
    conditions: Optional[Sequence[str]] = None,
    bias_correction: bool = True,
    bias: Optional[HintBias] = None,
    window_size: int = 200,
    forward_shift: int = 5,
    reverse_shift: int = -4,
    n_jobs: int = 1,
    verbose: bool = True,
) -> HintDifferentialResult:
    """Differential TF footprinting between conditions (``rgt-hint differential``).

    For every motif, the cut signal of all its binding sites (pooled over
    the conditions' MPBS, duplicates removed) is summed in a
    ``window_size`` window around each site centre, per condition, and
    normalised by a per-condition factor (median-of-ratios of total motif
    signal) and the number of sites. From the profile:

    * protection score ``PS = (N_R - N_C)/L + (N_L - N_C)/L``: flanking
      minus central signal, with ``L`` the motif length;
    * ``TC``: mean signal outside the motif;
    * ``TF_Activity = (PS + TC)_2 - (PS + TC)_1``, z-scored across motifs,
      with a two-sided normal p-value.

    Parameters
    ----------
    mpbs
        Motif-predicted binding sites: one BED path / DataFrame per condition
        (they are pooled), or a single one. Column 4 is the motif name,
        column 6 the strand (:func:`hint_match_motifs` output).
    bams
        One indexed BAM per condition.
    genome
        :class:`~epione.core.genome.Genome` or FASTA path.
    conditions
        Condition names; default the BAM basenames.
    bias_correction
        ``--bc``: use the bias-corrected signal (default HINT's ATAC tables,
        or ``bias=``) instead of raw cuts.
    window_size, forward_shift, reverse_shift
        ``rgt-hint differential`` defaults (note ``reverse_shift=-4`` there,
        -5 in footprinting).
    n_jobs
        Worker processes over motifs.

    Returns
    -------
    HintDifferentialResult
    """
    bams = [str(b) for b in bams]
    if conditions is None:
        conditions = [Path(b).stem for b in bams]
    conditions = list(conditions)
    if len(conditions) != len(bams):
        raise ValueError(f"{len(bams)} BAM files but {len(conditions)} condition names")
    if bias is None and genome is None:
        raise ValueError("pass genome= (or bias=)")
    fasta_path = bias.fasta if bias is not None else _fasta_path(genome)
    if bias_correction and bias is None:
        bias = HintBias(genome)

    if isinstance(mpbs, (str, Path, pd.DataFrame)):
        mpbs = [mpbs]
    allsites = pd.concat([_read_bed(m) for m in mpbs], ignore_index=True)
    allsites = allsites.iloc[sorted(range(len(allsites)), key=lambda i: (
        allsites.at[i, "chrom"], allsites.at[i, "start"], allsites.at[i, "end"]))].reset_index(drop=True)
    names = sorted(set(allsites["name"].astype(str)))
    upper = allsites["name"].astype(str).str.upper()

    jobs, per_motif = [], {}
    for name in names:
        sub = allsites[upper == name.upper()]          # by_names() is case-insensitive
        # remove_duplicates: unique (chrom, start, end, strand), then sorted
        sub = sub.drop_duplicates(subset=["chrom", "start", "end", "strand"])
        sub = sub.iloc[sorted(range(len(sub)), key=lambda i: (
            sub["chrom"].iat[i], sub["start"].iat[i], sub["end"].iat[i]))].reset_index(drop=True)
        per_motif[name] = sub
        sites = list(zip(sub["chrom"], sub["start"].astype(int), sub["end"].astype(int)))
        for b in bams:
            jobs.append((b, fasta_path, bias, sites, window_size, forward_shift,
                         reverse_shift, bias_correction))
    if verbose:
        print(f"HINT differential: {len(names)} motifs x {len(bams)} conditions"
              f" ({'bias-corrected' if bias_correction else 'raw'} signal)")
    if n_jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(int(n_jobs)) as pool:
            res = pool.map(_motif_signal, jobs)
    else:
        res = [_motif_signal(j) for j in jobs]

    C, M, W = len(bams), len(names), window_size
    signals = np.zeros((C, M, W), dtype=np.float32)
    for j in range(M):
        for i in range(C):
            sig, ok = res[j * C + i]
            signals[i, j, :] = sig
            if not ok:
                warnings.warn(f"signal extraction failed for {names[j]} in {conditions[i]}")
    motif_len = [int(per_motif[n]["end"].iat[0] - per_motif[n]["start"].iat[0]) for n in names]
    motif_num = [len(per_motif[n]) for n in names]

    # compute_factors (float32 throughout, as upstream)
    tot = np.sum(signals, axis=2)
    lg = np.log(tot + 0.01)
    lg = lg[:, ~np.isnan(lg).any(axis=0)]
    lg = lg - np.mean(lg, axis=0, keepdims=True)
    factors = np.exp(np.median(lg, axis=1))
    for i in range(C):
        for j in range(M):
            # numpy 1.x: float32 * int -> float64 scalar, which a float32
            # array then divides by after casting it back to float32.
            denom = np.float32(np.float64(factors[i]) * motif_num[j])
            signals[i, j, :] = signals[i, j, :] / denom

    half = W // 2
    ps_all, tc_all = [], []
    for j in range(M):
        L = motif_len[j]
        s = signals[:, j, :]
        nc = np.sum(s[:, half - L // 2:half + L // 2], axis=1)
        nr = np.sum(s[:, half + L // 2:half + L // 2 + L], axis=1)
        nl = np.sum(s[:, half - L // 2 - L:half - L // 2], axis=1)
        ps_all.append((nr - nc) / L + (nl - nc) / L)
        tc_all.append((np.sum(s, axis=1) - nc) / (W - L))

    stats = pd.DataFrame({"Motif": names, "Num": motif_num})
    for i, c in enumerate(conditions):
        stats[f"Protection_Score_{c}"] = [float(p[i]) for p in ps_all]
    for i, c in enumerate(conditions):
        stats[f"TC_{c}"] = [float(t[i]) for t in tc_all]
    if C == 2:
        from scipy.stats import norm, zscore
        a1 = np.array([float(p[0]) + float(t[0]) for p, t in zip(ps_all, tc_all)])
        a2 = np.array([float(p[1]) + float(t[1]) for p, t in zip(ps_all, tc_all)])
        act = np.subtract(a2, a1)
        z = zscore(act)
        pv = norm.sf(abs(z)) * 2
        stats["TF_Activity"] = act
        stats["Z_score"] = z
        stats["P_values"] = pv
        with np.errstate(divide="ignore", invalid="ignore"):
            stats["log2FC"] = np.log2(a2) - np.log2(a1)
        stats = stats.sort_values("P_values", kind="stable")
    stats = stats.set_index("Motif")
    return HintDifferentialResult(
        stats=stats,
        profiles={n: signals[:, j, :].copy() for j, n in enumerate(names)},
        factors=pd.Series(factors, index=conditions, name="factor"),
        conditions=conditions,
        window_size=W,
        motif_len=dict(zip(names, motif_len)),
        sites=per_motif,
    )
