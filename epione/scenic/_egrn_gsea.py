"""Pre-ranked GSEA with gene-set permutations, as used by the SCENIC+ eGRN builder.

This reproduces ``gseapy.algorithm.gsea_compute`` (gseapy 0.10.8, ``permutation_type
='gene_set'``, ``weighted_score_type=1``) for one gene set at a time:

* ``np.random.seed(seed)`` then one ``randint(2**31 - 1)`` gives the per-set seed; the null
  permutes the hit indicator with ``RandomState(that seed).shuffle`` once per permutation row.
  ``shuffle`` draws do not depend on the array content, so the permutations depend only on
  the ranking length and are generated once per length and shared by every gene set.
* The running sum is a sequential cumulative sum of ``hit * |r| * norm_hit - miss * norm_miss``
  and ``norm_hit = 1 / sum(hit * |r|)``. That sum is numpy's pairwise summation over the
  full row (zeros included), split into 8192-element blocks the way numpy 1.x buffered the
  reduction; it is evaluated here explicitly, so the result does not depend on the installed
  numpy.
* Nominal p-value and NES follow ``gsea_pval`` / ``normalize`` (masked means).
"""

from __future__ import annotations

import numba
import numpy as np

_NP_BUFSIZE = 8192  # numpy 1.x ufunc buffer size: reductions were summed per 8192-element block


@numba.njit(cache=True, error_model="numpy")
def _pw_leaf(a, lo, n):
    # numpy's pairwise_sum base cases (n <= 128): 8 accumulators
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[lo + i]
        return res
    r0 = a[lo]
    r1 = a[lo + 1]
    r2 = a[lo + 2]
    r3 = a[lo + 3]
    r4 = a[lo + 4]
    r5 = a[lo + 5]
    r6 = a[lo + 6]
    r7 = a[lo + 7]
    i = 8
    lim = n - (n % 8)
    while i < lim:
        r0 += a[lo + i]
        r1 += a[lo + i + 1]
        r2 += a[lo + i + 2]
        r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]
        r5 += a[lo + i + 5]
        r6 += a[lo + i + 6]
        r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


def _make_level(inner):
    @numba.njit(cache=False, error_model="numpy")
    def level(a, lo, n):
        if n <= 128:
            return _pw_leaf(a, lo, n)
        n2 = n // 2
        n2 -= n2 % 8
        return inner(a, lo, n2) + inner(a, lo + n2, n - n2)
    return level


# numpy's recursive pairwise_sum unrolled into a fixed chain (no recursion inside numba
# parallel regions): blocks are at most 8192 elements, i.e. at most 6 halvings above 128.
_pairwise = _pw_leaf
for _ in range(7):
    _pairwise = _make_level(_pairwise)


@numba.njit(cache=True, error_model="numpy")
def _np_sum(a, n):
    tot = _pairwise(a, 0, min(n, _NP_BUFSIZE))
    s = _NP_BUFSIZE
    while s < n:
        tot += _pairwise(a, s, min(_NP_BUFSIZE, n - s))
        s += _NP_BUFSIZE
    return tot


@numba.njit(cache=True, error_model="numpy")
def _row_es(c, tag, perm, use_perm, buf):
    """Enrichment score of one (possibly permuted) hit row.

    Returns (es, argmax, argmin) of the running sum, with numpy's NaN propagation.
    """
    n = c.shape[0]
    nhit = 0
    for j in range(n):
        t = tag[perm[j]] if use_perm else tag[j]
        buf[j] = t * c[j]
        nhit += t
    norm_tag = 1.0 / _np_sum(buf, n)
    norm_no_tag = 1.0 / (n - nhit)
    res = 0.0
    mx = -np.inf
    mn = np.inf
    imx = 0
    imn = 0
    has_nan = False
    first_nan = -1
    for j in range(n):
        t = tag[perm[j]] if use_perm else tag[j]
        term = (t * c[j]) * norm_tag - (1 - t) * norm_no_tag
        if j == 0:
            res = term
        else:
            res += term
        if res != res:
            if not has_nan:
                has_nan = True
                first_nan = j
        else:
            if res > mx:
                mx = res
                imx = j
            if res < mn:
                mn = res
                imn = j
    if has_nan:
        return np.nan, first_nan, first_nan
    es = mx if abs(mx) > abs(mn) else mn
    return es, imx, imn


@numba.njit(parallel=True, cache=True, error_model="numpy")
def gsea_batch(c, tags, perms):
    """Observed ES and null ES for several gene sets on the same ranking.

    c: (n,) |ranking metric|; tags: (m, n) int8 hit indicators in ranking order;
    perms: (n_perm, n) permutations. Returns es (m,), esnull (m, n_perm), argmax, argmin (m,).
    """
    m = tags.shape[0]
    n = c.shape[0]
    n_perm = perms.shape[0]
    es = np.empty(m, dtype=np.float64)
    imx = np.empty(m, dtype=np.int64)
    imn = np.empty(m, dtype=np.int64)
    esnull = np.empty((m, n_perm), dtype=np.float64)
    for s in numba.prange(m):
        buf = np.empty(n, dtype=np.float64)
        tag = tags[s]
        for i in range(n_perm):
            e, _, _ = _row_es(c, tag, perms[i], True, buf)
            esnull[s, i] = e
        e, a, b = _row_es(c, tag, perms[0], False, buf)
        es[s] = e
        imx[s] = a
        imn[s] = b
    return es, esnull, imx, imn


def gene_set_seed(seed: int) -> int:
    """The per-gene-set seed gseapy derives from ``seed`` (one set per call)."""
    rs = np.random.RandomState(seed)
    return int(rs.randint(np.iinfo(np.int32).max, size=1)[0])


@numba.njit(cache=True)
def _mt_shuffles(mt, pos, n, n_perm):
    """RandomState.shuffle(arange(n)) n_perm times, from MT19937 state (mt, pos).

    Legacy ``shuffle`` draws ``j = random_interval(i)`` for i = n-1 .. 1: the smallest bit
    mask covering i is applied to 32-bit outputs until the value is <= i.
    """
    out = np.empty((n_perm, n), dtype=np.int32)
    a = np.empty(n, dtype=np.int32)
    for r in range(n_perm):
        for k in range(n):
            a[k] = k
        for i in range(n - 1, 0, -1):
            mask = np.uint32(i)
            mask |= mask >> np.uint32(1)
            mask |= mask >> np.uint32(2)
            mask |= mask >> np.uint32(4)
            mask |= mask >> np.uint32(8)
            mask |= mask >> np.uint32(16)
            while True:
                if pos >= 624:
                    for kk in range(624):
                        y = (mt[kk] & np.uint32(0x80000000)) | (mt[(kk + 1) % 624] & np.uint32(0x7FFFFFFF))
                        v = mt[(kk + 397) % 624] ^ (y >> np.uint32(1))
                        if y & np.uint32(1):
                            v ^= np.uint32(0x9908B0DF)
                        mt[kk] = v
                    pos = 0
                y = mt[pos]
                pos += 1
                y ^= y >> np.uint32(11)
                y ^= (y << np.uint32(7)) & np.uint32(0x9D2C5680)
                y ^= (y << np.uint32(15)) & np.uint32(0xEFC60000)
                y ^= y >> np.uint32(18)
                value = y & mask
                if value <= np.uint32(i):
                    break
            j = np.int64(value)
            t = a[i]
            a[i] = a[j]
            a[j] = t
        out[r] = a
    return out


def permutations(n: int, n_perm: int, rs_seed: int) -> np.ndarray:
    """Row i is the index permutation ``RandomState(rs_seed).shuffle`` applies at call i."""
    st = np.random.RandomState(rs_seed).get_state()
    mt = np.asarray(st[1], dtype=np.uint32).copy()
    if n < 2:
        return np.zeros((n_perm, n), dtype=np.int32)
    return _mt_shuffles(mt, int(st[2]), n, n_perm)


def significance(es: np.ndarray, esnull: np.ndarray):
    """Nominal p-values and NES exactly as gseapy's ``gsea_pval`` and ``normalize``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        condlist = [es < 0, es >= 0]
        choicelist = [(esnull < es.reshape(len(es), 1)).sum(axis=1) / (esnull < 0).sum(axis=1),
                      (esnull >= es.reshape(len(es), 1)).sum(axis=1) / (esnull >= 0).sum(axis=1)]
        pvals = np.select(condlist, choicelist)
        pos = np.array(np.ma.MaskedArray(esnull, mask=(esnull < 0)).mean(axis=1))
        neg = np.array(np.ma.MaskedArray(esnull, mask=(esnull >= 0)).mean(axis=1))
        nes = np.where(es >= 0, es / pos, -es / neg)
    return nes, pvals
