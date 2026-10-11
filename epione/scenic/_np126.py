"""Bit-exact numpy 1.26 kernels for the SCENIC+ reference stack (AVX-512 CPUs).

pycisTopic's results depend on two numpy routines whose output changed
between numpy releases and differs between CPUs:

* ``np.argsort`` (default ``kind="quicksort"``) is not stable. Topic-region
  probabilities of binary scATAC data contain many exact ties, so the tie
  order decides which regions enter the Mimno 2011 coherence (tmtoolkit's top
  words). On AVX-512 machines numpy 1.26 dispatches float64 ``argsort`` to
  Intel's x86-simd-sort (``avx512-64bit-argsort.hpp``, BSD-3-Clause, vendored
  in ``numpy/core/src/npysort/x86-simd-sort``), which falls back to
  libstdc++'s ``std::sort`` on heavily tied segments.
* ``np.log`` on float64 calls Intel SVML's ``__svml_log8`` (BSD-3-Clause,
  ``numpy/core/src/umath/svml``), which differs from the C library's ``log``
  in the last bit for about 6 % of inputs; topic smoothing in
  ``binarize_topics`` then reorders nearly tied regions.

:func:`argsort_np126` re-implements the sort lane by lane (8-lane key/index
bitonic networks for blocks of at most 64 elements, median-of-8 pivot,
4x-unrolled vector-compress partition, ``std::sort`` introsort fallback) and
returns the same permutation, ties included. :func:`log_np126` re-implements
``__svml_log8`` with the same fused multiply-adds and constants; the one
hardware approximation it relies on (``vrcp14pd``, rounded to 5 fractional
bits) is replaced by the 16 exact switching points of that rounding, measured
on hardware and checked on 3.4e7 inputs. Both run on any CPU.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange, types
    from numba.core import cgutils
    from numba.extending import intrinsic
    from llvmlite import ir

    @intrinsic
    def _fma(typingctx, a, b, c):
        """Correctly rounded ``a * b + c`` (``llvm.fma.f64``)."""
        sig = types.float64(types.float64, types.float64, types.float64)

        def codegen(context, builder, signature, args):
            d = ir.DoubleType()
            fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(d, [d, d, d]),
                                                "llvm.fma.f64")
            return builder.call(fn, args)

        return sig, codegen

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba ships with scanpy
    _HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

# ``_mm512_set_epi64(v1, ..., v8)`` puts v8 in lane 0, so each network
# constant is the reversed macro list.
_IDX1 = np.array([3, 2, 1, 0, 7, 6, 5, 4], dtype=np.int64)   # NETWORK_64BIT_1
_REV = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int64)    # NETWORK_64BIT_2
_IDX3 = np.array([2, 3, 0, 1, 6, 7, 4, 5], dtype=np.int64)   # NETWORK_64BIT_3
_IDX4 = np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int64)   # NETWORK_64BIT_4
_INF = np.inf


@njit(cache=True, inline="always")
def _shuf(a):
    # _mm512_shuffle_pd(a, a, SHUFFLE_MASK(1, 1, 1, 1)): swap within pairs
    out = np.empty_like(a)
    for j in range(4):
        out[2 * j] = a[2 * j + 1]
        out[2 * j + 1] = a[2 * j]
    return out


@njit(cache=True, inline="always")
def _perm(a, idx):
    out = np.empty_like(a)
    for i in range(8):
        out[i] = a[idx[i]]
    return out


@njit(cache=True, inline="always")
def _vmin(a, b):
    # _mm512_min_pd(a, b) = a < b ? a : b
    out = np.empty_like(a)
    for i in range(8):
        out[i] = a[i] if a[i] < b[i] else b[i]
    return out


@njit(cache=True, inline="always")
def _vmax(a, b):
    # _mm512_max_pd(a, b) = a > b ? a : b
    out = np.empty_like(a)
    for i in range(8):
        out[i] = a[i] if a[i] > b[i] else b[i]
    return out


@njit(cache=True, inline="always")
def _cmp_merge(k1, k2, i1, i2, mask):
    mn = _vmin(k2, k1)
    mx = _vmax(k2, k1)
    k = np.empty(8)
    ix = np.empty(8, dtype=np.int64)
    for i in range(8):
        k[i] = mx[i] if (mask >> i) & 1 else mn[i]
        ix[i] = i1[i] if k[i] == k1[i] else i2[i]
    return k, ix


@njit(cache=True, inline="always")
def _coex(k1, k2, i1, i2):
    t1 = _vmin(k1, k2)
    t2 = _vmax(k1, k2)
    j1 = np.empty(8, dtype=np.int64)
    j2 = np.empty(8, dtype=np.int64)
    for i in range(8):
        if t1[i] == k1[i]:
            j1[i] = i1[i]
            j2[i] = i2[i]
        else:
            j1[i] = i2[i]
            j2[i] = i1[i]
    return t1, t2, j1, j2


@njit(cache=True)
def _sort_zmm(k, ix):
    k, ix = _cmp_merge(k, _shuf(k), ix, _shuf(ix), 0xAA)
    k, ix = _cmp_merge(k, _perm(k, _IDX1), ix, _perm(ix, _IDX1), 0xCC)
    k, ix = _cmp_merge(k, _shuf(k), ix, _shuf(ix), 0xAA)
    k, ix = _cmp_merge(k, _perm(k, _REV), ix, _perm(ix, _REV), 0xF0)
    k, ix = _cmp_merge(k, _perm(k, _IDX3), ix, _perm(ix, _IDX3), 0xCC)
    k, ix = _cmp_merge(k, _shuf(k), ix, _shuf(ix), 0xAA)
    return k, ix


@njit(cache=True)
def _bitonic_merge_zmm(k, ix):
    k, ix = _cmp_merge(k, _perm(k, _IDX4), ix, _perm(ix, _IDX4), 0xF0)
    k, ix = _cmp_merge(k, _perm(k, _IDX3), ix, _perm(ix, _IDX3), 0xCC)
    k, ix = _cmp_merge(k, _shuf(k), ix, _shuf(ix), 0xAA)
    return k, ix


@njit(cache=True)
def _merge_two(K, I, a, b):
    k1 = K[a]
    i1 = I[a]
    k2 = _perm(K[b], _REV)
    i2 = _perm(I[b], _REV)
    k3 = _vmin(k1, k2)
    k4 = _vmax(k1, k2)
    i3 = np.empty(8, dtype=np.int64)
    i4 = np.empty(8, dtype=np.int64)
    for i in range(8):
        if k3[i] == k1[i]:
            i3[i] = i1[i]
            i4[i] = i2[i]
        else:
            i3[i] = i2[i]
            i4[i] = i1[i]
    k4 = _perm(k4, _REV)
    i4 = _perm(i4, _REV)
    K[a], I[a] = _bitonic_merge_zmm(k3, i3)
    K[b], I[b] = _bitonic_merge_zmm(k4, i4)


@njit(cache=True)
def _merge_four(K, I, o):
    k2r = _perm(K[o + 2], _REV)
    k3r = _perm(K[o + 3], _REV)
    i2r = _perm(I[o + 2], _REV)
    i3r = _perm(I[o + 3], _REV)
    kt1 = _vmin(K[o], k3r)
    kt2 = _vmin(K[o + 1], k2r)
    km1 = _vmax(K[o], k3r)
    km2 = _vmax(K[o + 1], k2r)
    it1 = np.empty(8, dtype=np.int64)
    im1 = np.empty(8, dtype=np.int64)
    it2 = np.empty(8, dtype=np.int64)
    im2 = np.empty(8, dtype=np.int64)
    for i in range(8):
        if kt1[i] == K[o][i]:
            it1[i] = I[o][i]
            im1[i] = i3r[i]
        else:
            it1[i] = i3r[i]
            im1[i] = I[o][i]
        if kt2[i] == K[o + 1][i]:
            it2[i] = I[o + 1][i]
            im2[i] = i2r[i]
        else:
            it2[i] = i2r[i]
            im2[i] = I[o + 1][i]
    kt3 = _perm(km2, _REV)
    kt4 = _perm(km1, _REV)
    it3 = _perm(im2, _REV)
    it4 = _perm(im1, _REV)
    k0 = _vmin(kt1, kt2)
    k1 = _vmax(kt1, kt2)
    k2 = _vmin(kt3, kt4)
    k3 = _vmax(kt3, kt4)
    i0 = np.empty(8, dtype=np.int64)
    i1 = np.empty(8, dtype=np.int64)
    i2 = np.empty(8, dtype=np.int64)
    i3 = np.empty(8, dtype=np.int64)
    for i in range(8):
        if k0[i] == kt1[i]:
            i0[i] = it1[i]
            i1[i] = it2[i]
        else:
            i0[i] = it2[i]
            i1[i] = it1[i]
        if k2[i] == kt3[i]:
            i2[i] = it3[i]
            i3[i] = it4[i]
        else:
            i2[i] = it4[i]
            i3[i] = it3[i]
    K[o], I[o] = _bitonic_merge_zmm(k0, i0)
    K[o + 1], I[o + 1] = _bitonic_merge_zmm(k1, i1)
    K[o + 2], I[o + 2] = _bitonic_merge_zmm(k2, i2)
    K[o + 3], I[o + 3] = _bitonic_merge_zmm(k3, i3)


@njit(cache=True)
def _merge_eight(K, I):
    kr = [_perm(K[4 + j], _REV) for j in range(4)]   # 4r, 5r, 6r, 7r
    ir = [_perm(I[4 + j], _REV) for j in range(4)]
    kt = [np.empty(8) for _ in range(8)]
    it = [np.empty(8, dtype=np.int64) for _ in range(8)]
    km = [np.empty(8) for _ in range(4)]
    im = [np.empty(8, dtype=np.int64) for _ in range(4)]
    for j in range(4):
        # t_{j+1} = min(K[j], K[7-j] reversed)
        partner_k = kr[3 - j]
        partner_i = ir[3 - j]
        kt[j] = _vmin(K[j], partner_k)
        km[j] = _vmax(K[j], partner_k)
        for i in range(8):
            if kt[j][i] == K[j][i]:
                it[j][i] = I[j][i]
                im[j][i] = partner_i[i]
            else:
                it[j][i] = partner_i[i]
                im[j][i] = I[j][i]
    # t5..t8 = reversed m4, m3, m2, m1
    for j in range(4):
        kt[4 + j] = _perm(km[3 - j], _REV)
        it[4 + j] = _perm(im[3 - j], _REV)
    for a, b in ((0, 2), (1, 3), (4, 6), (5, 7), (0, 1), (2, 3), (4, 5), (6, 7)):
        kt[a], kt[b], it[a], it[b] = _coex(kt[a], kt[b], it[a], it[b])
    for j in range(8):
        K[j], I[j] = _bitonic_merge_zmm(kt[j], it[j])


@njit(cache=True)
def _small_sort(arr, arg, start, N):
    """``argsort_{8,16,32,64}_64bit`` on ``arg[start:start+N]`` (N <= 64)."""
    if N <= 8:
        nreg = 1
    elif N <= 16:
        nreg = 2
    elif N <= 32:
        nreg = 4
    else:
        nreg = 8
    half = nreg // 2 if nreg > 1 else 0
    # registers [0, half) are full loads, [half, nreg) masked (or reg 0 for N <= 8)
    K = [np.empty(8) for _ in range(nreg)]
    I = [np.empty(8, dtype=np.int64) for _ in range(nreg)]
    for r in range(nreg):
        for lane in range(8):
            p = 8 * r + lane
            if p < N:
                I[r][lane] = arg[start + p]
                K[r][lane] = arr[arg[start + p]]
            else:
                I[r][lane] = 0
                K[r][lane] = _INF
        K[r], I[r] = _sort_zmm(K[r], I[r])
    if nreg >= 2:
        for r in range(0, nreg, 2):
            _merge_two(K, I, r, r + 1)
    if nreg >= 4:
        for o in range(0, nreg, 4):
            _merge_four(K, I, o)
    if nreg == 8:
        _merge_eight(K, I)
    for r in range(nreg):
        for lane in range(8):
            p = 8 * r + lane
            if p < N:
                arg[start + p] = I[r][lane]


@njit(cache=True, inline="always")
def _partition_vec(arg, l_store, right, argvec, vec, pivot):
    """Compress lanes < pivot to ``arg[l_store:]`` and >= pivot to ``arg[right-n_ge:right]``."""
    n_ge = 0
    for lane in range(8):
        if vec[lane] >= pivot:
            n_ge += 1
    p = l_store
    q = right - n_ge
    for lane in range(8):
        if vec[lane] >= pivot:
            arg[q] = argvec[lane]
            q += 1
        else:
            arg[p] = argvec[lane]
            p += 1
    return n_ge


@njit(cache=True)
def _load(arr, arg, pos):
    av = arg[pos:pos + 8].copy()
    v = np.empty(8)
    for lane in range(8):
        v[lane] = arr[av[lane]]
    return av, v


@njit(cache=True)
def _partition(arr, arg, left, right, pivot, unroll):
    """``partition_avx512`` (unroll = 1) / ``partition_avx512_unrolled<4>``."""
    block = 8 * unroll
    for _ in range((right - left) % block):
        if not (arr[arg[left]] < pivot):
            right -= 1
            t = arg[left]
            arg[left] = arg[right]
            arg[right] = t
        else:
            left += 1
    if left == right:
        return left
    if unroll == 1 and right - left == 8:
        av, v = _load(arr, arg, left)
        n_ge = _partition_vec(arg, left, left + 8, av, v, pivot)
        return left + (8 - n_ge)

    lav = []
    lv = []
    rav = []
    rv = []
    for ii in range(unroll):
        a, v = _load(arr, arg, left + 8 * ii)
        lav.append(a)
        lv.append(v)
        a, v = _load(arr, arg, right - 8 * (unroll - ii))
        rav.append(a)
        rv.append(v)
    r_store = right - 8
    l_store = left
    left += block
    right -= block
    while right - left != 0:
        cav = []
        cv = []
        if (r_store + 8) - right < left - l_store:
            right -= block
            for ii in range(unroll):
                a, v = _load(arr, arg, right + 8 * ii)
                cav.append(a)
                cv.append(v)
        else:
            for ii in range(unroll):
                a, v = _load(arr, arg, left + 8 * ii)
                cav.append(a)
                cv.append(v)
            left += block
        for ii in range(unroll):
            n_ge = _partition_vec(arg, l_store, r_store + 8, cav[ii], cv[ii], pivot)
            l_store += 8 - n_ge
            r_store -= n_ge
    for ii in range(unroll):
        n_ge = _partition_vec(arg, l_store, r_store + 8, lav[ii], lv[ii], pivot)
        l_store += 8 - n_ge
        r_store -= n_ge
    for ii in range(unroll):
        n_ge = _partition_vec(arg, l_store, r_store + 8, rav[ii], rv[ii], pivot)
        l_store += 8 - n_ge
        r_store -= n_ge
    return l_store


# ---------------------------------------------------------------------------
# libstdc++ std::sort (introsort) — x86-simd-sort's fallback when quicksort
# stops making progress (heavily tied segments reach the depth limit).
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _lt(arr, i, j):
    return arr[i] < arr[j]


@njit(cache=True)
def _unguarded_linear_insert(arr, arg, last):
    val = arg[last]
    nxt = last - 1
    while arr[val] < arr[arg[nxt]]:
        arg[last] = arg[nxt]
        last = nxt
        nxt -= 1
    arg[last] = val


@njit(cache=True)
def _insertion_sort(arr, arg, first, last):
    if first == last:
        return
    for i in range(first + 1, last):
        if _lt(arr, arg[i], arg[first]):
            val = arg[i]
            for p in range(i, first, -1):
                arg[p] = arg[p - 1]
            arg[first] = val
        else:
            _unguarded_linear_insert(arr, arg, i)


@njit(cache=True)
def _push_heap(arr, arg, first, hole, top, value):
    parent = (hole - 1) // 2
    while hole > top and arr[arg[first + parent]] < arr[value]:
        arg[first + hole] = arg[first + parent]
        hole = parent
        parent = (hole - 1) // 2
    arg[first + hole] = value


@njit(cache=True)
def _adjust_heap(arr, arg, first, hole, length, value):
    top = hole
    second = hole
    while second < (length - 1) // 2:
        second = 2 * (second + 1)
        if _lt(arr, arg[first + second], arg[first + second - 1]):
            second -= 1
        arg[first + hole] = arg[first + second]
        hole = second
    if (length & 1) == 0 and second == (length - 2) // 2:
        second = 2 * (second + 1)
        arg[first + hole] = arg[first + second - 1]
        hole = second - 1
    _push_heap(arr, arg, first, hole, top, value)


@njit(cache=True)
def _heap_sort(arr, arg, first, last):
    """``std::__partial_sort(first, last, last)``: make_heap + sort_heap."""
    length = last - first
    if length >= 2:
        parent = (length - 2) // 2
        while True:
            _adjust_heap(arr, arg, first, parent, length, arg[first + parent])
            if parent == 0:
                break
            parent -= 1
    while last - first > 1:
        last -= 1
        value = arg[last]
        arg[last] = arg[first]
        _adjust_heap(arr, arg, first, 0, last - first, value)


@njit(cache=True)
def _move_median_to_first(arr, arg, result, a, b, c):
    if _lt(arr, arg[a], arg[b]):
        if _lt(arr, arg[b], arg[c]):
            m = b
        elif _lt(arr, arg[a], arg[c]):
            m = c
        else:
            m = a
    elif _lt(arr, arg[a], arg[c]):
        m = a
    elif _lt(arr, arg[b], arg[c]):
        m = c
    else:
        m = b
    t = arg[result]
    arg[result] = arg[m]
    arg[m] = t


@njit(cache=True)
def _unguarded_partition(arr, arg, first, last, pivot):
    while True:
        while _lt(arr, arg[first], arg[pivot]):
            first += 1
        last -= 1
        while _lt(arr, arg[pivot], arg[last]):
            last -= 1
        if not (first < last):
            return first
        t = arg[first]
        arg[first] = arg[last]
        arg[last] = t
        first += 1


@njit(cache=True)
def _std_sort(arr, arg, first, last):
    if first == last:
        return
    n = last - first
    lg = 0
    while (n >> (lg + 1)) > 0:
        lg += 1
    # __introsort_loop, recursion on the right part made explicit
    stack = [(first, last, 2 * lg)]
    while len(stack) > 0:
        f, l, depth = stack.pop()
        while l - f > 16:
            if depth == 0:
                _heap_sort(arr, arg, f, l)
                break
            depth -= 1
            mid = f + (l - f) // 2
            _move_median_to_first(arr, arg, f, f + 1, mid, l - 1)
            cut = _unguarded_partition(arr, arg, f + 1, l, f)
            stack.append((cut, l, depth))
            l = cut
    # __final_insertion_sort
    if last - first > 16:
        _insertion_sort(arr, arg, first, first + 16)
        for i in range(first + 16, last):
            _unguarded_linear_insert(arr, arg, i)
    else:
        _insertion_sort(arr, arg, first, last)


@njit(cache=True)
def _argsort_np126(arr):
    n = arr.shape[0]
    arg = np.arange(n, dtype=np.int64)
    if n <= 1:
        return arg
    max_iters = 2 * int(np.log2(n))
    stack = [(0, n - 1, max_iters)]
    while len(stack) > 0:
        left, right, iters = stack.pop()
        if iters <= 0:
            _std_sort(arr, arg, left, right + 1)
            continue
        if right + 1 - left <= 64:
            _small_sort(arr, arg, left, right + 1 - left)
            continue
        if right - left >= 8:
            size = (right - left) // 8
            s = np.empty(8)
            for j in range(8):
                s[j] = arr[arg[left + (8 - j) * size]]
            pivot = np.sort(s)[4]
        else:
            pivot = arr[arg[right]]
        smallest = _INF
        biggest = -_INF
        for p in range(left, right + 1):
            x = arr[arg[p]]
            if x < smallest:
                smallest = x
            if x > biggest:
                biggest = x
        unroll = 1 if (right + 1 - left) <= 8 * 4 * 8 else 4
        pivot_index = _partition(arr, arg, left, right + 1, pivot, unroll)
        if pivot != biggest:
            stack.append((pivot_index, right, iters - 1))
        if pivot != smallest:
            stack.append((left, pivot_index - 1, iters - 1))
    return arg


def argsort_np126(a) -> np.ndarray:
    """Permutation that numpy 1.26 ``np.argsort(a)`` returns on an AVX-512 CPU."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("argsort_np126 expects a 1-D array")
    if np.isnan(a).any():
        raise ValueError("argsort_np126 does not support NaN")
    return _argsort_np126(a)


# ---------------------------------------------------------------------------
# __svml_log8 (svml_z0_log_d_la.s)
# ---------------------------------------------------------------------------

_h = float.fromhex
# -log(mantissa of Rcp), indexed by the 4 leading mantissa bits of Rcp
_LOG_TBL = np.array([_h(v) for v in (
    "0x0.0p+0", "-0x1.f0a30c01162a6p-5", "-0x1.e27076e2af2e6p-4", "-0x1.5ff3070a793d4p-3",
    "-0x1.c8ff7c79a9a22p-3", "-0x1.1675cababa60ep-2", "-0x1.4618bc21c5ec2p-2",
    "-0x1.739d7f6bbd007p-2", "0x1.269621134db92p-2", "0x1.f991c6cb3b379p-3",
    "0x1.a93ed3c8ad9e3p-3", "0x1.5bf406b543db2p-3", "0x1.1178e8227e47cp-3",
    "0x1.9335e5d594989p-4", "0x1.08598b59e3a07p-4", "0x1.0415d89e74444p-5")])
_C256 = _h("0x1.c81cd309d7c70p-4")
_C320 = _h("-0x1.007357e93af62p-3")
_C384 = _h("0x1.249229cee81efp-3")
_C448 = _h("-0x1.55553fb28db06p-3")
_C512 = _h("0x1.9999999cc9f5cp-3")
_C576 = _h("-0x1.00000000c05bdp-2")
_C640 = _h("0x1.5555555555466p-2")
_C704 = _h("-0x1.fffffffffffc6p-2")
_LN2 = _h("0x1.62e42fefa39efp-1")
# smallest mantissa m in [1, 2) with round(32 * vrcp14(m)) / 32 <= k / 32, k = 31..16
_RCP_SWITCH = np.array([_h(v) for v in (
    "0x1.040fp+0", "0x1.0c97p+0", "0x1.15b4p+0", "0x1.1f70p+0", "0x1.29e6p+0",
    "0x1.3523p+0", "0x1.4143p+0", "0x1.4e5fp+0", "0x1.5c99p+0", "0x1.6c15p+0",
    "0x1.7d07p+0", "0x1.8f9cp+0", "0x1.a41ap+0", "0x1.bad0p+0", "0x1.d41cp+0",
    "0x1.f080p+0")])


@njit(cache=True, parallel=True)
def _log_np126(x, out):
    for i in prange(x.shape[0]):
        v = x[i]
        if not (v > 0.0) or v == np.inf:
            out[i] = math.log(v) if v >= 0.0 else np.nan
            continue
        f, e = math.frexp(v)            # v = f * 2**e, f in [0.5, 1)
        m = 2.0 * f                     # vgetmantpd: [1, 2)
        expo = float(e - 1)             # vgetexppd
        cnt = 0
        for t in _RCP_SWITCH:
            if t <= m:
                cnt += 1
        k = 32 - cnt
        rcp = k / 32.0                  # vrcp14pd + vrndscalepd(5 bits)
        r = _fma(rcp, m, -1.0)
        p1 = _fma(_C384, r, _C448)
        p0 = _fma(_C256, r, _C320)
        p3 = _fma(_C640, r, _C704)
        r2 = r * r
        p2 = _fma(_C512, r, _C576)
        if rcp < 0.75:
            expo = expo + 1.0
        r4 = r2 * r2
        p0 = _fma(r2, p0, p1)
        p2 = _fma(r2, p2, p3)
        idx = 0 if k == 32 else k - 16
        p0 = _fma(r4, p0, p2)
        p0 = _fma(r2, p0, r)
        s = _LOG_TBL[idx] + p0
        out[i] = _fma(_LN2, expo, s)


@njit(cache=True, inline="always")
def _rcp5(m):
    """``vrndscalepd(vrcp14pd(m), 5 fraction bits)`` for m in [1, 2), as k / 32."""
    cnt = 0
    for t in _RCP_SWITCH:
        if t <= m:
            cnt += 1
    return 32 - cnt


# __svml_log1p8 (svml_z0_log1p_d_la.s): same table and polynomial as log
@njit(cache=True, parallel=True)
def _log1p_np126(x, out):
    for i in prange(x.shape[0]):
        v = x[i]
        if not (v > -1.0) or v == np.inf or v != v:
            out[i] = math.log1p(v) if v >= -1.0 else np.nan
            continue
        y = 1.0 + v                       # 1 + x, rounded
        hi = 1.0 if 1.0 > v else v        # max(1, x)
        lo = 1.0 if 1.0 < v else v        # min(1, x)
        f, e = math.frexp(y)
        m = 2.0 * f
        expo = float(e - 1)
        xl = lo - (y - hi)                # rounding error of 1 + x
        xl = math.ldexp(xl, -(e - 1))     # vscalefpd: Xl * 2**-expo
        k = _rcp5(m)
        rcp = k / 32.0
        r = _fma(rcp, m, -1.0)
        r = _fma(xl, rcp, r)
        p1 = _fma(_C256, r, _C320)
        p2 = _fma(_C384, r, _C448)
        if rcp < 0.75:
            expo = expo + 1.0
        r2 = r * r
        p3 = _fma(_C512, r, _C576)
        r4 = r2 * r2
        p4 = _fma(_C640, r, _C704)
        p1 = _fma(r2, p1, p2)
        p3 = _fma(r2, p3, p4)
        p1 = _fma(r4, p1, p3)
        p1 = _fma(r2, p1, r)
        idx = 0 if k == 32 else k - 16
        res = _fma(_LN2, expo, _LOG_TBL[idx] + p1)
        if v == 0.0 and math.copysign(1.0, v) < 0:
            res = -0.0
        out[i] = res


_LOG10_TBL = np.array([_h(v) for v in (
    "0x0.0p+0", "-0x1.af5f92b00e610p-6", "-0x1.a30a9d609efeap-5", "-0x1.31b3055c47118p-4",
    "-0x1.8cf183886480dp-4", "-0x1.e3bc1ab0e19fep-4", "-0x1.1b3e71ec94f7bp-3",
    "-0x1.42c7e7fe3fc02p-3", "0x1.ffbfc2bbc7803p-4", "0x1.b721cd17157e3p-4",
    "0x1.715d0ce367afcp-4", "0x1.2e3a740b7800fp-4", "0x1.db11ed766abf4p-5",
    "0x1.5e3966b7e9295p-5", "0x1.cb38fccd8bfdbp-6", "0x1.c3d0837784c41p-7")])
_D256 = _h("0x1.8c2d828480370p-5")
_D320 = _h("-0x1.bd80d96029814p-5")
_D384 = _h("0x1.fc3f6f38b58a2p-5")
_D448 = _h("-0x1.287a63464dc80p-4")
_D512 = _h("0x1.63c62777f27d9p-4")
_D576 = _h("-0x1.bcb7b153c06a3p-4")
_D640 = _h("0x1.287a7636f428cp-3")
_D704 = _h("-0x1.bcb7b1526e4dbp-3")
_D768 = _h("0x1.bcb7b1526e50ep-2")
_LOG10_2 = _h("0x1.34413509f79ffp-2")


# __svml_log108 (svml_z0_log10_d_la.s)
@njit(cache=True, parallel=True)
def _log10_np126(x, out):
    for i in prange(x.shape[0]):
        v = x[i]
        if not (v > 0.0) or v == np.inf:
            out[i] = math.log10(v) if v >= 0.0 else np.nan
            continue
        f, e = math.frexp(v)
        m = 2.0 * f
        expo = float(e - 1)
        k = _rcp5(m)
        rcp = k / 32.0
        r = _fma(rcp, m, -1.0)
        q8 = _fma(_D512, r, _D576)
        q1 = _fma(_D256, r, _D320)
        q14 = _fma(_D384, r, _D448)
        r2 = r * r
        q12 = _fma(_D640, r, _D704)
        if rcp < 0.75:
            expo = expo + 1.0
        r4 = r2 * r2
        q1 = _fma(r2, q1, q14)
        q8 = _fma(r2, q8, q12)
        idx = 0 if k == 32 else k - 16
        q1 = _fma(r4, q1, q8)
        q1 = _fma(r, q1, _D768)
        q6 = _fma(q1, r, _LOG10_TBL[idx])
        out[i] = _fma(_LOG10_2, expo, q6)


def _apply(kernel, x, out=None):
    a = np.asarray(x, dtype=np.float64)
    if out is not None:
        if not (out.flags.c_contiguous and out.dtype == np.float64 and out.shape == a.shape):
            raise ValueError("out must be a C-contiguous float64 array of the input's shape")
        kernel(np.ascontiguousarray(a).reshape(-1), out.reshape(-1))
        return out
    flat = np.ascontiguousarray(a).ravel()
    res = np.empty_like(flat)
    kernel(flat, res)
    return res.reshape(a.shape)


def log1p_np126(x, out=None) -> np.ndarray:
    """Element-wise ``log1p`` as numpy 1.26 computes it on AVX-512 CPUs (SVML)."""
    return _apply(_log1p_np126, x, out)


def log10_np126(x) -> np.ndarray:
    """Element-wise ``log10`` as numpy 1.26 computes it on AVX-512 CPUs (SVML)."""
    return _apply(_log10_np126, x)


def log_np126(x) -> np.ndarray:
    """Element-wise natural log as numpy 1.26 computes it on AVX-512 CPUs."""
    a = np.asarray(x, dtype=np.float64)
    if not _HAVE_NUMBA:  # pragma: no cover
        raise ImportError("log_np126 requires numba")
    flat = np.ascontiguousarray(a).ravel()
    out = np.empty_like(flat)
    _log_np126(flat, out)
    return out.reshape(a.shape)


def argsort_compat(x, compat: str = "numpy126") -> np.ndarray:
    """``np.argsort(x)``: numpy 1.26 / AVX-512 order (``"numpy126"``) or installed numpy."""
    if compat == "numpy126":
        return argsort_np126(x)
    if compat == "native":
        return np.argsort(x)
    raise ValueError("compat must be 'numpy126' or 'native'")


def log_compat(x, compat: str = "numpy126") -> np.ndarray:
    """``np.log(x)``: numpy 1.26 / AVX-512 (``"numpy126"``) or installed numpy."""
    if compat == "numpy126":
        return log_np126(x)
    if compat == "native":
        return np.log(x)
    raise ValueError("compat must be 'numpy126' or 'native'")


# ---------------------------------------------------------------------------
# np.sum(x, axis=1) of a C-contiguous float64 matrix (numpy 1.26)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _pairwise_sum(a, lo, n):
    """``DOUBLE_pairwise_sum`` (``numpy/core/src/umath/loops_utils.h.src``)."""
    if n < 8:
        res = -0.0
        for i in range(n):
            res += a[lo + i]
        return res
    if n <= 128:
        r0 = a[lo]
        r1 = a[lo + 1]
        r2 = a[lo + 2]
        r3 = a[lo + 3]
        r4 = a[lo + 4]
        r5 = a[lo + 5]
        r6 = a[lo + 6]
        r7 = a[lo + 7]
        i = 8
        stop = n - (n % 8)
        while i < stop:
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
    n2 = n // 2
    n2 -= n2 % 8
    return _pairwise_sum(a, lo, n2) + _pairwise_sum(a, lo + n2, n - n2)


@njit(cache=True)
def _rowsum_np126(mat, out):
    # numpy 1.26 feeds the reduction through its 8192-element iterator buffer
    n = mat.shape[1]
    for r in range(mat.shape[0]):
        row = mat[r]
        acc = 0.0
        for s in range(0, n, 8192):
            acc += _pairwise_sum(row, s, min(8192, n - s))
        out[r] = acc


def rowsum_compat(mat, compat: str = "numpy126") -> np.ndarray:
    """``np.sum(mat, axis=1)`` for a float64 matrix: numpy 1.26 or installed numpy."""
    if compat == "native":
        return np.sum(mat, axis=1)
    if compat != "numpy126":
        raise ValueError("compat must be 'numpy126' or 'native'")
    mat = np.ascontiguousarray(mat, dtype=np.float64)
    out = np.empty(mat.shape[0])
    _rowsum_np126(mat, out)
    return out


# ---------------------------------------------------------------------------
# np.argpartition(x, kth) for one kth (numpy 1.26 introselect, scalar)
# ---------------------------------------------------------------------------

def argpartition_np126(v, kth: int) -> np.ndarray:
    """Permutation of numpy 1.26 ``np.argpartition(v, kth)`` (``npysort/selection.cpp``).

    numpy 2 switched to a SIMD selection on AVX-512 CPUs, which orders the
    elements on either side of ``kth`` differently. Only the median-of-3
    introselect path is reproduced (the median-of-medians fallback is reached
    only after ``2 * log2(n)`` unproductive rounds).
    """
    v = np.asarray(v, dtype=np.float64)
    num = v.shape[0]
    if kth < 0:
        kth += num
    t = list(range(num))
    less = lambda a, b: v[a] < v[b]

    low, high = 0, num - 1
    if kth - low < 3:                           # dumb_select
        for i in range(kth + 1):
            mi = i
            for k in range(i + 1, num):
                if v[t[k]] < v[t[mi]]:
                    mi = k
            t[i], t[mi] = t[mi], t[i]
        return np.array(t, dtype=np.int64)
    if kth == num - 1:
        mi = low
        for k in range(low + 1, num):
            if not (v[t[k]] < v[t[mi]]):
                mi = k
        t[kth], t[mi] = t[mi], t[kth]
        return np.array(t, dtype=np.int64)
    depth_limit = int(num).bit_length() - 1
    depth_limit *= 2
    while low + 1 < high:
        ll, hh = low + 1, high
        if not (depth_limit > 0 or hh - ll < 5):
            return np.argpartition(v, kth)      # median-of-medians path: not reproduced
        mid = low + (high - low) // 2
        if less(t[high], t[mid]):
            t[high], t[mid] = t[mid], t[high]
        if less(t[high], t[low]):
            t[high], t[low] = t[low], t[high]
        if less(t[low], t[mid]):
            t[low], t[mid] = t[mid], t[low]
        t[mid], t[low + 1] = t[low + 1], t[mid]
        depth_limit -= 1
        pivot = v[t[low]]
        while True:
            ll += 1
            while v[t[ll]] < pivot:
                ll += 1
            hh -= 1
            while pivot < v[t[hh]]:
                hh -= 1
            if hh < ll:
                break
            t[ll], t[hh] = t[hh], t[ll]
        t[low], t[hh] = t[hh], t[low]
        if hh >= kth:
            high = hh - 1
        if hh <= kth:
            low = ll
    if high == low + 1 and less(t[high], t[low]):
        t[high], t[low] = t[low], t[high]
    return np.array(t, dtype=np.int64)


def log1p_compat(x, compat: str = "numpy126", out=None) -> np.ndarray:
    if compat == "numpy126":
        return log1p_np126(x, out=out)
    if compat == "native":
        return np.log1p(x, out=out)
    raise ValueError("compat must be 'numpy126' or 'native'")


def log10_compat(x, compat: str = "numpy126") -> np.ndarray:
    if compat == "numpy126":
        return log10_np126(x)
    if compat == "native":
        return np.log10(x)
    raise ValueError("compat must be 'numpy126' or 'native'")
