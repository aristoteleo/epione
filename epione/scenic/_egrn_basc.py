"""BASC A binarisation threshold (Hopfensitz et al., IEEE/ACM TCBB 2012).

Used by the eGRN builder to binarise region-to-gene importances per gene, the way SCENIC+
does it: sort the values, keep at most the ``max_elements`` smallest, fit the optimal step
functions with 1..N-2 discontinuities by dynamic programming, score every discontinuity by
jump height / approximation error, and place the threshold between the two values around
the median location of the strongest discontinuities.

Floating-point contract: every mean and sum is a sequential left-to-right accumulation
starting from 0.0 and every square is ``d * d``; any division by zero or mean of an empty
slice aborts the binarisation, which then returns a threshold of 0. This reproduces the
SCENIC+ reference bit for bit. The cost table is filled once (O(N^3)) instead of being
recomputed inside the DP loop (O(N^4)); each entry is computed with the same operation order,
so the values are identical.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True)
def _seq_mean(v, a, b):
    # mean of v[a:b] accumulated left to right; caller guarantees b > a
    c = 0.0
    for k in range(a, b):
        c += v[k]
    return c / (b - a)


@numba.njit(cache=True)
def _basc_threshold_sorted(v):
    """Threshold for an ascending, already truncated vector ``v`` (len >= 3).

    Returns (threshold, ok); ok is False when the reference would raise.
    """
    n = v.shape[0]
    # cost[a, b]: squared deviation of v[a..b] from their mean
    cost = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        run = 0.0
        for b in range(a, n):
            run += v[b]
            mean = run / (b - a + 1)
            s = 0.0
            for k in range(a, b + 1):
                d = v[k] - mean
                s += d * d
            cost[a, b] = s
    C = np.zeros((n - 1, n), dtype=np.float64)
    for i in range(n):
        C[0, i] = cost[i, n - 1]
    ind = np.zeros((n - 2, n), dtype=np.int64)
    for j in range(1, n - 1):
        for i in range(0, n - j):
            cmin = np.inf
            dmin = -1
            for d in range(i, n - j):
                c = cost[i, d] + C[j - 1, d + 1]
                if c < cmin:
                    cmin = c
                    dmin = d
            C[j, i] = cmin
            ind[j - 1, i] = dmin
    m = n - 2
    P = np.zeros((m, n), dtype=np.int64)
    for j in range(m):
        z = j
        P[j, 0] = ind[z, 0]
        if j > 0:
            z -= 1
            for i in range(1, j + 1):
                P[j, i] = ind[z, P[j, i - 1] + 1]
                z -= 1
    last = n - 1
    best = np.empty(m, dtype=np.int64)
    for j in range(m):
        qmax = -1.0
        imax = -1
        for i in range(j + 1):
            p = P[j, i]
            # jump height between the means left and right of discontinuity i
            if i == 0 and j > 0:
                lo_a, lo_b, hi_a, hi_b = 0, p + 1, p + 1, P[j, i + 1] + 1
            elif i == j and j > 0:
                lo_a, lo_b, hi_a, hi_b = P[j, i - 1] + 1, p + 1, p + 1, last + 1
            elif i == 0 and j == 0:
                lo_a, lo_b, hi_a, hi_b = 0, p + 1, p + 1, last + 1
            else:
                lo_a, lo_b, hi_a, hi_b = P[j, i - 1] + 1, p + 1, p + 1, P[j, i + 1] + 1
            if hi_b > n:
                hi_b = n
            if lo_b > n:
                lo_b = n
            if hi_b <= hi_a or lo_b <= lo_a:
                return 0.0, False
            h = _seq_mean(v, hi_a, hi_b) - _seq_mean(v, lo_a, lo_b)
            zt = (v[p] + v[p + 1]) / 2
            e = 0.0
            for k in range(n):
                d = v[k] - zt
                e += d * d
            if e == 0.0:
                return 0.0, False
            q = h / e
            if q > qmax:
                qmax = q
                imax = p
        best[j] = imax
    vmed = int(np.floor(np.median(best)))
    return (v[vmed + 1] + v[vmed]) / 2, True


@numba.njit(cache=True)
def basc_threshold(x, max_elements=100):
    """BASC threshold of one vector, 0.0 when it cannot be binarised (len <= 2, constant)."""
    n = x.shape[0]
    if n <= 2:
        return 0.0
    same = True
    for k in range(1, n):
        if x[k] != x[0]:
            same = False
            break
    if same:
        return 0.0
    v = np.sort(x)
    if n > max_elements:
        v = v[:max_elements].copy()
    thr, ok = _basc_threshold_sorted(v)
    return thr if ok else 0.0


@numba.njit(parallel=True, cache=True)
def basc_thresholds_grouped(values, starts, counts, max_elements=100):
    """BASC threshold per group; ``values[starts[g]:starts[g]+counts[g]]`` is group g."""
    out = np.zeros(starts.shape[0], dtype=np.float64)
    for g in numba.prange(starts.shape[0]):
        s = starts[g]
        out[g] = basc_threshold(values[s:s + counts[g]], max_elements)
    return out
