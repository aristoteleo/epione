"""Deterministic emulation of numpy 1.26's AVX-512 ``argsort``.

Duplicate of ``epione/scenic/_egrn_argsort.py`` from the SCENIC+ eGRN block (PR #57); the
copies are to be consolidated once both blocks are merged.  It is used here so that the
TF-to-gene table comes out in the reference row order: scenicplus sorts importances with
``np.argsort(kind="quicksort")`` and tied values are ordered by the sort implementation.

SCENIC+ breaks ties between equal scores by shuffling the values with a seeded permutation
and then calling ``np.argsort`` (quicksort, i.e. *unstable*). Which of several tied elements
comes first therefore depends on the sorting algorithm: on AVX-512 machines numpy 1.26 uses
x86-simd-sort (BSD-3-Clause, Intel), numpy 2.x ships a newer version with a different
order, and other CPUs use introsort. The SCENIC+ reference was produced with numpy 1.26.4 on
AVX-512 hardware.

This module re-implements that exact algorithm in scalar numba code (8-lane vectors as
small arrays): median-of-8 pivot, the unrolled compress-store partition, the key-value
bitonic sorting networks for blocks of at most 64 elements and the ``std::sort`` (libstdc++
introsort) fallback. The output is the same index permutation on any CPU and numpy version.
Only 32- and 64-bit keys go through x86-simd-sort; other dtypes use numpy's introsort.

The algorithm follows x86-simd-sort as vendored in numpy 1.26.4
(``avx512-64bit-argsort.hpp``, ``avx512-common-argsort.h``, ``avx512-64bit-keyvalue-networks.hpp``;
Copyright (C) 2022 Intel Corporation, BSD-3-Clause).
"""

from __future__ import annotations

import numba
import numpy as np

__all__ = ["argsort_np126"]

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
