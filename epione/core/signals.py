"""Rolling-window signal operations — pure-Python / NumPy.

This module replaces the former Cython extension ``epione.core.signals``
(``signals.pyx``). The kernels here are vectorised with NumPy / SciPy, so
epione installs and runs on any platform **without a C compiler** and
without shipping per-platform ``.so`` artefacts.

Public API (unchanged from the Cython version so ``from ...signals import *``
keeps working):

    fast_rolling_math      rolling max / min / mean / sum / prod
    shuffle_array          random within-window cut-site shuffling (background)
    tobias_footprint_array TOBIAS footprint score per bp
    FOS_score              Footprint-Occupancy-Score per bp
    OneSignal, SignalList  placeholder containers (work-in-progress upstream)

Original Cython implementation:
    @author Mette Bentsen / TOBIAS (MIT)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "OneSignal",
    "SignalList",
    "shuffle_array",
    "fast_rolling_math",
    "tobias_footprint_array",
    "FOS_score",
]


# ---------------------------------------------------------------------------
class OneSignal(np.ndarray):
    """Work in progress; placeholder for future development."""


class SignalList(list):
    """Work in progress; placeholder for future development."""

    def __init__(self, matrix=None, name=""):
        super().__init__()
        self.aggregate = ""
        self.name = name
        self.mat = matrix
        self.n = 0  # n regions

    def from_regions(self, regions, bigwig):
        """Read from regions and bigwig (assumes ``.signal`` in regions)."""

    def filter_outliers(self, lower=0.0, upper=1.0):
        """Filter rows based on outlier values."""
        max_values = np.max(self.mat, axis=1)
        upper_limit = np.percentile(max_values, [100 * upper])[0]
        return max_values <= upper_limit

    def aggregate(self, normalize=False, smooth=1):  # noqa: F811 - mirrors upstream stub
        """Make aggregate across all rows."""
        self.aggregate = ""
        return self.aggregate

    def correlate(self):
        """Placeholder."""

    def footprint(self):
        """Placeholder."""


# ---------------------------------------------------------------------------
def fast_rolling_math(arr, w, operation):
    """Rolling ``operation`` over ``arr`` with window size ``w``.

    Operations: ``"max"``, ``"min"``, ``"mean"``, ``"sum"``, ``"prod"``.
    Returns an array the same size as ``arr`` with ``NaN`` in the flanking
    positions for ``sum``/``mean`` (matching the original Cython kernel:
    the window sum is placed at index ``i + floor(w/2)``).
    """
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    L = arr.shape[0]
    lf = int(np.floor(w / 2.0))
    roll = np.full(L, np.nan, dtype=np.float64)

    if L == 0 or w <= 0:
        return roll

    if operation in ("sum", "mean"):
        if L >= w:
            csum = np.concatenate(([0.0], np.cumsum(arr)))
            wins = csum[w:] - csum[:-w]          # sum of arr[i:i+w], i = 0..L-w
            if operation == "mean":
                wins = wins / float(w)
            roll[lf:lf + wins.shape[0]] = wins
        return roll

    if operation in ("max", "min"):
        # Centred window of width w, clipped at the array borders — matches the
        # Cython kernel's behaviour (each position keeps at least its own value).
        from scipy.ndimage import maximum_filter1d, minimum_filter1d
        flt = maximum_filter1d if operation == "max" else minimum_filter1d
        # origin shifts the window so it spans [i-lf, i-lf+w) like the kernel.
        origin = lf - (w - 1) // 2
        return flt(arr, size=w, mode="nearest", origin=origin).astype(np.float64)

    if operation == "prod":
        out = np.empty(L, dtype=np.float64)
        rf = int(np.ceil(w / 2.0))
        for i in range(L):
            s, e = max(i - lf, 0), min(i - lf + w, L)
            out[i] = np.prod(arr[s:e]) if e > s else 1.0
        return out

    raise ValueError(f"unknown operation {operation!r}")


# ---------------------------------------------------------------------------
def shuffle_array(arr, no_rand, shift_options):
    """Shuffle non-zero values of ``arr`` within ``shift_options`` offsets.

    Produces ``no_rand`` randomised copies (rows) used to build a background
    distribution. Vectorised equivalent of the original double loop.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    shift_options = np.asarray(shift_options, dtype=np.int64)
    max_shift = int(max(abs(int(shift_options.min())), abs(int(shift_options.max()))))

    ext = np.concatenate((np.zeros(max_shift), arr, np.zeros(max_shift)))
    ext_len = ext.shape[0]
    nz = np.nonzero(ext)[0]
    no_shift = nz.shape[0]

    rand_mat = np.zeros((no_rand, ext_len), dtype=np.float64)
    if no_shift == 0:
        return rand_mat[:, max_shift:-max_shift] if max_shift else rand_mat

    rand_rel = np.random.choice(shift_options, size=(no_shift, no_rand))  # (no_shift, no_rand)
    cols = nz[:, None] + rand_rel                                          # target columns
    rows = np.broadcast_to(np.arange(no_rand)[None, :], cols.shape)
    vals = np.broadcast_to(ext[nz][:, None], cols.shape)
    np.add.at(rand_mat, (rows.ravel(), cols.ravel()), vals.ravel())

    return rand_mat[:, max_shift:-max_shift] if max_shift else rand_mat


# ---------------------------------------------------------------------------
def tobias_footprint_array(arr, flank_min, flank_max, fp_min, fp_max):
    """TOBIAS footprint score per bp (max over flank/footprint window sizes).

    Pure-Python port of the Cython kernel. For production footprint scoring
    prefer :class:`epione.tl.FootprintScorer`, which streams over a bigwig.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    L = arr.shape[0]
    scores = np.zeros(L, dtype=np.float64)
    pos_arr = np.where(arr > 0.0, arr, 0.0)
    neg_arr = np.where(arr < 0.0, arr, 0.0)
    # prefix sums for O(1) window sums
    pcs = np.concatenate(([0.0], np.cumsum(pos_arr)))
    ncs = np.concatenate(([0.0], np.cumsum(neg_arr)))

    def psum(s, e):
        return pcs[e] - pcs[s]

    def nsum(s, e):
        return ncs[e] - ncs[s]

    for i in range(L - 2 * flank_max - fp_max):
        for flank_w in range(flank_min, flank_max + 1):
            for footprint_w in range(fp_min, fp_max + 1):
                ls = i
                fs = i + flank_w
                rs = i + flank_w + footprint_w
                left_sum = psum(ls, ls + flank_w)
                fp_sum = nsum(fs, fs + footprint_w)
                right_sum = psum(rs, rs + flank_w)
                fp_mean = fp_sum / (1.0 * footprint_w)
                flank_mean = (right_sum + left_sum) / (2.0 * flank_w)
                fp_score = flank_mean - fp_mean
                seg = scores[fs:fs + footprint_w]
                np.maximum(seg, fp_score, out=seg)
    return scores


def FOS_score(arr, flank_min, flank_max, fp_min, fp_max):
    """Footprint-Occupancy-Score per bp (min over window sizes; lower = better)."""
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    L = arr.shape[0]
    scores = np.full(L, 10000.0, dtype=np.float64)
    cs = np.concatenate(([0.0], np.cumsum(arr)))

    def s(a, b):
        return cs[b] - cs[a]

    for i in range(L - 2 * flank_max - fp_max):
        for flank_w in range(flank_min, flank_max):
            for footprint_w in range(fp_min, fp_max):
                fs = i + flank_w
                rs = i + flank_w + footprint_w
                Lm = s(i, i + flank_w) / (flank_w * 1.0)
                Cm = s(fs, fs + footprint_w) / (footprint_w * 1.0)
                Rm = s(rs, rs + flank_w) / (flank_w * 1.0)
                if Cm < Rm and Cm < Lm and Lm > 0 and Rm > 0:
                    fos = (Cm + 1.0) / Lm + (Cm + 1.0) / Rm
                else:
                    fos = 10000.0
                seg = scores[fs:fs + footprint_w]
                np.minimum(seg, fos, out=seg)
    return scores
