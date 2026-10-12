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
returns the same permutation, ties included. It handles the dtypes numpy 1.26 sends
through x86-simd-sort (32- and 64-bit floats and integers; NaN-aware) and numpy's
introsort for small integer types, and is shared by every SCENIC+ block whose reference
output depends on ``argsort`` tie order: topics (Mimno coherence, DARs), TF-to-gene row
order, eGRN thresholds and triplet ranks, and AUCell rankings. :func:`log_np126` re-implements
``__svml_log8`` with the same fused multiply-adds and constants; the one
hardware approximation it relies on (``vrcp14pd``, rounded to 5 fractional
bits) is replaced by the 16 exact switching points of that rounding, measured
on hardware and checked on 3.4e7 inputs. Both run on any CPU.
"""
from __future__ import annotations

import math

import numba
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

# ---------------------------------------------------------------------------
# np.argsort (numpy 1.26, AVX-512: x86-simd-sort, Copyright (C) 2022 Intel
# Corporation, BSD-3-Clause; libstdc++ std::sort fallback)
# ---------------------------------------------------------------------------

_REV = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int64)
_N1 = np.array([3, 2, 1, 0, 7, 6, 5, 4], dtype=np.int64)
_N3 = np.array([2, 3, 0, 1, 6, 7, 4, 5], dtype=np.int64)
_N4 = np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int64)
_SH = np.array([1, 0, 3, 2, 5, 4, 7, 6], dtype=np.int64)


# ----------------------------------------------------------------------------- lane ops
@numba.njit(cache=True)
def _cmp_merge(k, a, p, mask):
    # cmp_merge<vtype1, vtype2>(in1=k, in2=perm(k), idx1=a, idx2=perm(a), mask)
    k2 = k[p]
    a2 = a[p]
    nk = np.empty_like(k)
    na = np.empty_like(a)
    for i in range(8):
        x = k[i]
        y = k2[i]
        mn = y if y < x else x
        mx = y if y > x else x
        t = mx if (mask >> i) & 1 else mn
        nk[i] = t
        na[i] = a[i] if t == x else a2[i]
    return nk, na


@numba.njit(cache=True)
def _sort_zmm(k, a):
    k, a = _cmp_merge(k, a, _SH, 0xAA)
    k, a = _cmp_merge(k, a, _N1, 0xCC)
    k, a = _cmp_merge(k, a, _SH, 0xAA)
    k, a = _cmp_merge(k, a, _REV, 0xF0)
    k, a = _cmp_merge(k, a, _N3, 0xCC)
    k, a = _cmp_merge(k, a, _SH, 0xAA)
    return k, a


@numba.njit(cache=True)
def _bitonic_merge(k, a):
    k, a = _cmp_merge(k, a, _N4, 0xF0)
    k, a = _cmp_merge(k, a, _N3, 0xCC)
    k, a = _cmp_merge(k, a, _SH, 0xAA)
    return k, a


@numba.njit(cache=True)
def _coex(k1, k2, a1, a2):
    # min/max with index move (ties keep a1 with the min lane)
    nk1 = np.empty_like(k1)
    nk2 = np.empty_like(k1)
    na1 = np.empty_like(a1)
    na2 = np.empty_like(a1)
    for i in range(8):
        x = k1[i]
        y = k2[i]
        mn = y if y < x else x
        mx = y if y > x else x
        nk1[i] = mn
        nk2[i] = mx
        if mn == x:
            na1[i] = a1[i]
            na2[i] = a2[i]
        else:
            na1[i] = a2[i]
            na2[i] = a1[i]
    return nk1, nk2, na1, na2


@numba.njit(cache=True)
def _merge_two(K, A, i, j):
    k1 = K[i]
    a1 = A[i]
    k2 = K[j][_REV]
    a2 = A[j][_REV]
    k3, k4, a3, a4 = _coex(k1, k2, a1, a2)
    k4 = k4[_REV]
    a4 = a4[_REV]
    k3, a3 = _bitonic_merge(k3, a3)
    k4, a4 = _bitonic_merge(k4, a4)
    K[i] = k3
    A[i] = a3
    K[j] = k4
    A[j] = a4


@numba.njit(cache=True)
def _merge_four(K, A, o):
    k2r = K[o + 2][_REV]
    k3r = K[o + 3][_REV]
    a2r = A[o + 2][_REV]
    a3r = A[o + 3][_REV]
    t1, m1, at1, am1 = _coex(K[o], k3r, A[o], a3r)
    t2, m2, at2, am2 = _coex(K[o + 1], k2r, A[o + 1], a2r)
    t3 = m2[_REV]
    t4 = m1[_REV]
    at3 = am2[_REV]
    at4 = am1[_REV]
    k0, k1, a0, a1 = _coex(t1, t2, at1, at2)
    k2, k3, a2, a3 = _coex(t3, t4, at3, at4)
    k0, a0 = _bitonic_merge(k0, a0)
    k1, a1 = _bitonic_merge(k1, a1)
    k2, a2 = _bitonic_merge(k2, a2)
    k3, a3 = _bitonic_merge(k3, a3)
    K[o], K[o + 1], K[o + 2], K[o + 3] = k0, k1, k2, k3
    A[o], A[o + 1], A[o + 2], A[o + 3] = a0, a1, a2, a3


@numba.njit(cache=True)
def _merge_eight(K, A):
    k4r = K[4][_REV]
    k5r = K[5][_REV]
    k6r = K[6][_REV]
    k7r = K[7][_REV]
    a4r = A[4][_REV]
    a5r = A[5][_REV]
    a6r = A[6][_REV]
    a7r = A[7][_REV]
    t1, m1, at1, am1 = _coex(K[0], k7r, A[0], a7r)
    t2, m2, at2, am2 = _coex(K[1], k6r, A[1], a6r)
    t3, m3, at3, am3 = _coex(K[2], k5r, A[2], a5r)
    t4, m4, at4, am4 = _coex(K[3], k4r, A[3], a4r)
    t5 = m4[_REV]
    t6 = m3[_REV]
    t7 = m2[_REV]
    t8 = m1[_REV]
    at5 = am4[_REV]
    at6 = am3[_REV]
    at7 = am2[_REV]
    at8 = am1[_REV]
    t1, t3, at1, at3 = _coex(t1, t3, at1, at3)
    t2, t4, at2, at4 = _coex(t2, t4, at2, at4)
    t5, t7, at5, at7 = _coex(t5, t7, at5, at7)
    t6, t8, at6, at8 = _coex(t6, t8, at6, at8)
    t1, t2, at1, at2 = _coex(t1, t2, at1, at2)
    t3, t4, at3, at4 = _coex(t3, t4, at3, at4)
    t5, t6, at5, at6 = _coex(t5, t6, at5, at6)
    t7, t8, at7, at8 = _coex(t7, t8, at7, at8)
    t1, at1 = _bitonic_merge(t1, at1)
    t2, at2 = _bitonic_merge(t2, at2)
    t3, at3 = _bitonic_merge(t3, at3)
    t4, at4 = _bitonic_merge(t4, at4)
    t5, at5 = _bitonic_merge(t5, at5)
    t6, at6 = _bitonic_merge(t6, at6)
    t7, at7 = _bitonic_merge(t7, at7)
    t8, at8 = _bitonic_merge(t8, at8)
    K[0], K[1], K[2], K[3], K[4], K[5], K[6], K[7] = t1, t2, t3, t4, t5, t6, t7, t8
    A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7] = at1, at2, at3, at4, at5, at6, at7, at8


# ----------------------------------------------------------------------------- base case
@numba.njit(cache=True)
def _argsort_small(arr, arg, lo, n, padval):
    """argsort_{8,16,32,64}_64bit on arg[lo:lo+n] (n <= 64)."""
    if n <= 0:
        return
    if n <= 8:
        nreg = 1
    elif n <= 16:
        nreg = 2
    elif n <= 32:
        nreg = 4
    else:
        nreg = 8
    K = np.empty((nreg, 8), dtype=arr.dtype)
    A = np.zeros((nreg, 8), dtype=np.int64)
    for r in range(nreg):
        for i in range(8):
            p = r * 8 + i
            if p < n:
                A[r, i] = arg[lo + p]
                K[r, i] = arr[A[r, i]]
            else:
                A[r, i] = 0
                K[r, i] = padval
        k, a = _sort_zmm(K[r].copy(), A[r].copy())
        K[r] = k
        A[r] = a
    if nreg >= 2:
        for r in range(0, nreg, 2):
            _merge_two(K, A, r, r + 1)
    if nreg >= 4:
        for r in range(0, nreg, 4):
            _merge_four(K, A, r)
    if nreg == 8:
        _merge_eight(K, A)
    for r in range(nreg):
        for i in range(8):
            p = r * 8 + i
            if p < n:
                arg[lo + p] = A[r, i]


# ----------------------------------------------------------------------------- partition
@numba.njit(cache=True)
def _partition_vec(arg, left, right, av, kv, pivot):
    # compress-store lanes < pivot at arg[left:], lanes >= pivot at arg[right - n_ge:right]
    n_ge = 0
    for i in range(8):
        if kv[i] >= pivot:
            n_ge += 1
    l = left
    r = right - n_ge
    for i in range(8):
        if kv[i] >= pivot:
            arg[r] = av[i]
            r += 1
        else:
            arg[l] = av[i]
            l += 1
    return n_ge


@numba.njit(cache=True)
def _load(arr, arg, pos):
    av = arg[pos:pos + 8].copy()
    kv = np.empty(8, dtype=arr.dtype)
    for i in range(8):
        kv[i] = arr[av[i]]
    return av, kv


@numba.njit(cache=True)
def _partition(arr, arg, left, right, pivot, unroll):
    """partition_avx512 (unroll=1) / partition_avx512_unrolled<4>; right is exclusive."""
    step = 8 * unroll
    for _ in range((right - left) % step):
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
        av, kv = _load(arr, arg, left)
        n_ge = _partition_vec(arg, left, left + 8, av, kv, pivot)
        return left + (8 - n_ge)
    VLa = np.empty((unroll, 8), dtype=np.int64)
    VLk = np.empty((unroll, 8), dtype=arr.dtype)
    VRa = np.empty((unroll, 8), dtype=np.int64)
    VRk = np.empty((unroll, 8), dtype=arr.dtype)
    for ii in range(unroll):
        av, kv = _load(arr, arg, left + 8 * ii)
        VLa[ii] = av
        VLk[ii] = kv
        av, kv = _load(arr, arg, right - 8 * (unroll - ii))
        VRa[ii] = av
        VRk[ii] = kv
    r_store = right - 8
    l_store = left
    left += step
    right -= step
    CA = np.empty((unroll, 8), dtype=np.int64)
    CK = np.empty((unroll, 8), dtype=arr.dtype)
    while right - left != 0:
        if (r_store + 8) - right < left - l_store:
            right -= step
            for ii in range(unroll):
                av, kv = _load(arr, arg, right + 8 * ii)
                CA[ii] = av
                CK[ii] = kv
        else:
            for ii in range(unroll):
                av, kv = _load(arr, arg, left + 8 * ii)
                CA[ii] = av
                CK[ii] = kv
            left += step
        for ii in range(unroll):
            n_ge = _partition_vec(arg, l_store, r_store + 8, CA[ii], CK[ii], pivot)
            l_store += 8 - n_ge
            r_store -= n_ge
    if unroll == 1:
        n_ge = _partition_vec(arg, l_store, r_store + 8, VLa[0], VLk[0], pivot)
        l_store += 8 - n_ge
        n_ge = _partition_vec(arg, l_store, l_store + 8, VRa[0], VRk[0], pivot)
        l_store += 8 - n_ge
        return l_store
    for ii in range(unroll):
        n_ge = _partition_vec(arg, l_store, r_store + 8, VLa[ii], VLk[ii], pivot)
        l_store += 8 - n_ge
        r_store -= n_ge
    for ii in range(unroll):
        n_ge = _partition_vec(arg, l_store, r_store + 8, VRa[ii], VRk[ii], pivot)
        l_store += 8 - n_ge
        r_store -= n_ge
    return l_store


# ----------------------------------------------------------------------------- std::sort
@numba.njit(cache=True)
def _lt(arr, i, j, nan_aware):
    x = arr[i]
    y = arr[j]
    if nan_aware:
        xn = x != x
        yn = y != y
        if not xn and not yn:
            return x < y
        if xn:
            return False
        return True
    return x < y


@numba.njit(cache=True)
def _lg(n):
    d = 0
    while n > 1:
        n >>= 1
        d += 1
    return d


@numba.njit(cache=True)
def _adjust_heap(arr, a, first, hole, length, value, nan_aware):
    top = hole
    child = hole
    while child < (length - 1) // 2:
        child = 2 * (child + 1)
        if _lt(arr, a[first + child], a[first + child - 1], nan_aware):
            child -= 1
        a[first + hole] = a[first + child]
        hole = child
    if (length & 1) == 0 and child == (length - 2) // 2:
        child = 2 * (child + 1)
        a[first + hole] = a[first + child - 1]
        hole = child - 1
    parent = (hole - 1) // 2
    while hole > top and _lt(arr, a[first + parent], value, nan_aware):
        a[first + hole] = a[first + parent]
        hole = parent
        parent = (hole - 1) // 2
    a[first + hole] = value


@numba.njit(cache=True)
def _heap_sort(arr, a, first, last, nan_aware):
    n = last - first
    if n < 2:
        return
    parent = (n - 2) // 2
    while True:
        _adjust_heap(arr, a, first, parent, n, a[first + parent], nan_aware)
        if parent == 0:
            break
        parent -= 1
    end = last
    while end - first > 1:
        end -= 1
        v = a[end]
        a[end] = a[first]
        _adjust_heap(arr, a, first, 0, end - first, v, nan_aware)


@numba.njit(cache=True)
def _std_sort(arr, a, first, last, nan_aware):
    """libstdc++ std::sort on index range a[first:last] with comparator arr[i] < arr[j]."""
    if last - first < 2:
        return
    stack = np.empty((128, 3), dtype=np.int64)
    sp = 0
    stack[0, 0] = first
    stack[0, 1] = last
    stack[0, 2] = 2 * _lg(last - first)
    sp = 1
    while sp > 0:
        sp -= 1
        f = stack[sp, 0]
        l = stack[sp, 1]
        depth = stack[sp, 2]
        while l - f > 16:
            if depth == 0:
                _heap_sort(arr, a, f, l, nan_aware)
                break
            depth -= 1
            mid = f + (l - f) // 2
            x, y, z = f + 1, mid, l - 1
            if _lt(arr, a[x], a[y], nan_aware):
                if _lt(arr, a[y], a[z], nan_aware):
                    m = y
                elif _lt(arr, a[x], a[z], nan_aware):
                    m = z
                else:
                    m = x
            elif _lt(arr, a[x], a[z], nan_aware):
                m = x
            elif _lt(arr, a[y], a[z], nan_aware):
                m = z
            else:
                m = y
            t = a[f]
            a[f] = a[m]
            a[m] = t
            lo = f + 1
            hi = l
            while True:
                while _lt(arr, a[lo], a[f], nan_aware):
                    lo += 1
                hi -= 1
                while _lt(arr, a[f], a[hi], nan_aware):
                    hi -= 1
                if not (lo < hi):
                    break
                t = a[lo]
                a[lo] = a[hi]
                a[hi] = t
                lo += 1
            stack[sp, 0] = lo
            stack[sp, 1] = l
            stack[sp, 2] = depth
            sp += 1
            l = lo
    # final insertion sort
    n = last - first
    lim = first + 16 if n > 16 else last
    for i in range(first + 1, lim):
        v = a[i]
        if _lt(arr, v, a[first], nan_aware):
            for k in range(i, first, -1):
                a[k] = a[k - 1]
            a[first] = v
        else:
            j = i
            while _lt(arr, v, a[j - 1], nan_aware):
                a[j] = a[j - 1]
                j -= 1
            a[j] = v
    if n > 16:
        for i in range(first + 16, last):
            v = a[i]
            j = i
            while _lt(arr, v, a[j - 1], nan_aware):
                a[j] = a[j - 1]
                j -= 1
            a[j] = v


# ----------------------------------------------------------------------------- driver
@numba.njit(cache=True, nogil=True)
def _argsort_simd(arr, padval, is_float):
    n = arr.shape[0]
    arg = np.arange(n).astype(np.int64)
    if n <= 1:
        return arg
    if is_float:
        for i in range(n):
            if arr[i] != arr[i]:
                _std_sort(arr, arg, 0, n, True)
                return arg
    stack = np.empty((256, 3), dtype=np.int64)
    stack[0, 0] = 0
    stack[0, 1] = n - 1
    stack[0, 2] = 2 * np.int64(np.log2(n))
    sp = 1
    samp = np.empty(8, dtype=arr.dtype)
    while sp > 0:
        sp -= 1
        left = stack[sp, 0]
        right = stack[sp, 1]
        iters = stack[sp, 2]
        if iters <= 0:
            _std_sort(arr, arg, left, right + 1, False)
            continue
        if right + 1 - left <= 64:
            _argsort_small(arr, arg, left, right + 1 - left, padval)
            continue
        if right - left >= 8:
            size = (right - left) // 8
            for i in range(8):
                samp[i] = arr[arg[left + (i + 1) * size]]
            pivot = np.sort(samp)[4]
        else:
            pivot = arr[arg[right]]
        smallest = arr[arg[left]]
        biggest = arr[arg[left]]
        for i in range(left, right + 1):
            v = arr[arg[i]]
            if v < smallest:
                smallest = v
            if v > biggest:
                biggest = v
        unroll = 1 if (right + 1 - left) <= 256 else 4
        pidx = _partition(arr, arg, left, right + 1, pivot, unroll)
        if pivot != biggest:
            stack[sp, 0] = pidx
            stack[sp, 1] = right
            stack[sp, 2] = iters - 1
            sp += 1
        if pivot != smallest:
            stack[sp, 0] = left
            stack[sp, 1] = pidx - 1
            stack[sp, 2] = iters - 1
            sp += 1
    return arg


@numba.njit(cache=True, nogil=True)
def _argsort_introsort(v):
    # numpy's npy_aquicksort (used for dtypes x86-simd-sort does not cover)
    num = v.shape[0]
    tosort = np.arange(num).astype(np.int64)
    if num < 2:
        return tosort
    pl = 0
    pr = num - 1
    stack = np.empty(256, np.int64)
    sp = 0
    depth = np.empty(128, np.int64)
    dp = 0
    cdepth = _lg(num) * 2
    while True:
        if cdepth < 0:
            _std_sort(v, tosort, pl, pr + 1, False)  # pragma: no cover (heapsort fallback differs)
        else:
            while (pr - pl) > 15:
                pm = pl + ((pr - pl) >> 1)
                if v[tosort[pm]] < v[tosort[pl]]:
                    tosort[pm], tosort[pl] = tosort[pl], tosort[pm]
                if v[tosort[pr]] < v[tosort[pm]]:
                    tosort[pr], tosort[pm] = tosort[pm], tosort[pr]
                if v[tosort[pm]] < v[tosort[pl]]:
                    tosort[pm], tosort[pl] = tosort[pl], tosort[pm]
                vp = v[tosort[pm]]
                pi = pl
                pj = pr - 1
                tosort[pm], tosort[pj] = tosort[pj], tosort[pm]
                while True:
                    pi += 1
                    while v[tosort[pi]] < vp:
                        pi += 1
                    pj -= 1
                    while vp < v[tosort[pj]]:
                        pj -= 1
                    if pi >= pj:
                        break
                    tosort[pi], tosort[pj] = tosort[pj], tosort[pi]
                pk = pr - 1
                tosort[pi], tosort[pk] = tosort[pk], tosort[pi]
                if pi - pl < pr - pi:
                    stack[sp] = pi + 1
                    stack[sp + 1] = pr
                    sp += 2
                    pr = pi - 1
                else:
                    stack[sp] = pl
                    stack[sp + 1] = pi - 1
                    sp += 2
                    pl = pi + 1
                cdepth -= 1
                depth[dp] = cdepth
                dp += 1
            for pi in range(pl + 1, pr + 1):
                vi = tosort[pi]
                vpp = v[vi]
                pj = pi
                pk = pi - 1
                while pj > pl and vpp < v[tosort[pk]]:
                    tosort[pj] = tosort[pk]
                    pj -= 1
                    pk -= 1
                tosort[pj] = vi
        if sp == 0:
            break
        sp -= 2
        pl = stack[sp]
        pr = stack[sp + 1]
        dp -= 1
        cdepth = depth[dp]
    return tosort


def argsort_np126(x: np.ndarray) -> np.ndarray:
    """``np.argsort(x)`` exactly as numpy 1.26 computes it on an AVX-512 CPU."""
    x = np.ascontiguousarray(x)
    if x.ndim != 1:
        raise ValueError("1-D input expected")
    dt = x.dtype
    if dt in (np.float32, np.float64):
        return _argsort_simd(x, dt.type(np.inf), True)
    if dt in (np.int32, np.uint32, np.int64, np.uint64):
        return _argsort_simd(x, np.iinfo(dt).max, False)
    if dt.kind in "iub" and dt.itemsize <= 2:
        return _argsort_introsort(x)
    return np.argsort(x, kind="quicksort")


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
    """``np.argsort(x)`` of a float64 array: numpy 1.26 / AVX-512 order (``"numpy126"``)
    or installed numpy."""
    if compat == "numpy126":
        return argsort_np126(np.asarray(x, dtype=np.float64))
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
