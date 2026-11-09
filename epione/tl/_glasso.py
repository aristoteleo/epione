"""Graphical lasso, ported line by line from the R package ``glasso`` 1.11.

``glasso.f`` (J. Friedman, T. Hastie, R. Tibshirani; GPL-2) is translated
subroutine by subroutine -- ``glasso`` (block decomposition), ``connect`` /
``row`` (connected components of ``|S| > rho``), ``lasinv1`` (block
coordinate descent), ``setup``, ``lasso``, ``fatmul`` and ``inv`` -- keeping
the Fortran loop order and the order of every floating-point accumulation, so
that the result matches R's ``glasso::glasso(s, rho)`` to rounding.  Only the
code path Cicero uses is ported: ``approx = FALSE`` and a cold start
(``start = "cold"``); ``penalize.diagonal`` is supported.

Indices below follow the Fortran code (1-based in comments, 0-based in code);
matrices are indexed ``a[row, col]`` exactly like ``a(row, col)`` in Fortran.

``sklearn.covariance.graphical_lasso`` is not a substitute: it accepts only a
scalar penalty and uses a different solver and stopping rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

_EPS = 1.0e-7  # parameter(eps=1.0e-7) in lasinv1


@njit(cache=True, nogil=True)
def _row(nc, jr, nr, n, ss, rho, ie, kr, kr0):
    # subroutine row(nc,nr,jr,n,ss,rho,ie,na,kr): new neighbours of the nodes jr(1:nr)
    na = 0
    for l in range(nr):
        k = jr[l]
        for j in range(n):
            if ie[j] > 0:
                continue
            if j == k:
                continue
            if abs(ss[j, k]) <= rho[j, k]:
                continue
            kr[kr0 + na] = j
            na += 1
            ie[j] = nc
    return na


@njit(cache=True, nogil=True)
def _connect(n, ss, rho, ic, ir):
    # subroutine connect(n,ss,rho,nc,ic,ir,ie): breadth-first connected components
    ie = np.zeros(n, dtype=np.int64)
    nc = 0
    is_ = 0  # Fortran is=1
    for k in range(n):
        if ie[k] > 0:
            continue
        ir[is_] = k
        nc += 1
        ie[k] = nc
        ic[0, nc - 1] = is_
        is_ += 1
        na = _row(nc, ir[is_ - 1:is_], 1, n, ss, rho, ie, ir, is_)
        if na == 0:
            ic[1, nc - 1] = is_ - 1
            continue
        while True:
            nas = na
            iss = is_
            il = iss + nas - 1
            if il >= n - 1:  # Fortran: il .ge. n (1-based)
                break
            is_ = is_ + na
            na = _row(nc, ir[iss:iss + nas], nas, n, ss, rho, ie, ir, is_)
            if na == 0:
                break
        ic[1, nc - 1] = il
    return nc


@njit(cache=True, nogil=True)
def _setup(m, n, ss, rho, ww, vv, s, r):
    l = 0
    for j in range(n):
        if j == m:
            continue
        r[l] = rho[j, m]
        s[l] = ss[j, m]
        i = 0
        for k in range(n):
            if k == m:
                continue
            vv[i, l] = ww[k, j]
            i += 1
        l += 1


@njit(cache=True, nogil=True)
def _fatmul2(n, vv, x, s, z, mm):
    # subroutine fatmul(it=2,...): s = s - vv x   (vv symmetric)
    fac = 0.2
    l = 0
    for j in range(n):
        if x[j] == 0.0:
            continue
        mm[l] = j
        z[l] = x[j]
        l += 1
    if l > int(fac * n):
        # s = s - matmul(x, vv): column-wise sequential dot products
        for j in range(n):
            acc = 0.0
            for i in range(n):
                acc += x[i] * vv[i, j]
            s[j] = s[j] - acc
    else:
        for j in range(n):
            acc = 0.0
            for i in range(l):
                acc += vv[mm[i], j] * z[i]
            s[j] = s[j] - acc


@njit(cache=True, nogil=True)
def _lasso(rho, n, vv, s, thr, x, z, mm):
    _fatmul2(n, vv, x, s, z, mm)
    while True:
        dlx = 0.0
        for j in range(n):
            xj = x[j]
            x[j] = 0.0
            t = s[j] + vv[j, j] * xj
            a = abs(t) - rho[j]
            if a > 0.0:
                x[j] = (a if t >= 0.0 else -a) / vv[j, j]  # sign(abs(t)-rho(j), t)
            if x[j] == xj:
                continue
            d = x[j] - xj
            if abs(d) > dlx:
                dlx = abs(d)
            for i in range(n):
                s[i] = s[i] - d * vv[i, j]
        if dlx < thr:
            break


@njit(cache=True, nogil=True)
def _inv(n, ww, xs, wwi):
    nm1 = n - 1
    for j in range(n):
        for i in range(nm1):
            xs[i, j] = -xs[i, j]
    acc = 0.0
    for i in range(nm1):
        acc += xs[i, 0] * ww[i + 1, 0]
    wwi[0, 0] = 1.0 / (ww[0, 0] + acc)
    for i in range(nm1):
        wwi[i + 1, 0] = wwi[0, 0] * xs[i, 0]
    acc = 0.0
    for i in range(nm1):
        acc += xs[i, n - 1] * ww[i, n - 1]
    wwi[n - 1, n - 1] = 1.0 / (ww[n - 1, n - 1] + acc)
    for i in range(nm1):
        wwi[i, n - 1] = wwi[n - 1, n - 1] * xs[i, n - 1]
    for j in range(1, nm1):
        a1 = 0.0
        for i in range(j):
            a1 += xs[i, j] * ww[i, j]
        a2 = 0.0
        for i in range(j, nm1):
            a2 += xs[i, j] * ww[i + 1, j]
        wwi[j, j] = 1.0 / (ww[j, j] + a1 + a2)
        for i in range(j):
            wwi[i, j] = wwi[j, j] * xs[i, j]
        for i in range(j, nm1):
            wwi[i + 1, j] = wwi[j, j] * xs[i, j]


@njit(cache=True, nogil=True)
def _lasinv1(n, ss, rho, ipen, thr, maxit, ww, wwi):
    """Block coordinate descent on one connected component (ia = 0, is = 0)."""
    nm1 = n - 1
    shr = 0.0
    for j in range(n):
        for k in range(n):
            if j == k:
                continue
            shr += abs(ss[j, k])
    if shr == 0.0:
        for j in range(n):
            for k in range(n):
                ww[j, k] = 0.0
                wwi[j, k] = 0.0
        for j in range(n):
            ww[j, j] = ss[j, j] + rho[j, j] if ipen != 0 else ss[j, j]
            wwi[j, j] = 1.0 / max(ww[j, j], _EPS)
        return 0, 0.0
    shr = thr * shr / nm1
    vv = np.empty((nm1, nm1), dtype=np.float64).T  # Fortran-ordered scratch
    xs = np.zeros((n, nm1), dtype=np.float64).T
    s = np.empty(nm1)
    so = np.empty(nm1)
    x = np.empty(nm1)
    z = np.empty(nm1)
    mm = np.empty(nm1, dtype=np.int64)
    ro = np.empty(nm1)
    ws = np.empty(n)
    for j in range(n):
        for k in range(n):
            ww[j, k] = ss[j, k]
    for j in range(n):
        ww[j, j] = ss[j, j] + rho[j, j] if ipen != 0 else ss[j, j]
    niter = 0
    while True:
        dlx = 0.0
        for m in range(n):
            for i in range(nm1):
                x[i] = xs[i, m]
            for i in range(n):
                ws[i] = ww[i, m]
            _setup(m, n, ss, rho, ww, vv, s, ro)
            for i in range(nm1):
                so[i] = s[i]
            sv = 0.0  # sum(abs(vv)), column-major order
            for l in range(nm1):
                for i in range(nm1):
                    sv += abs(vv[i, l])
            _lasso(ro, nm1, vv, s, shr / sv, x, z, mm)
            l = 0
            for j in range(n):
                if j == m:
                    continue
                ww[j, m] = so[l] - s[l]
                ww[m, j] = ww[j, m]
                l += 1
            d = 0.0
            for i in range(n):
                d += abs(ww[i, m] - ws[i])
            if d > dlx:
                dlx = d
            for i in range(nm1):
                xs[i, m] = x[i]
        niter += 1
        if niter >= maxit:
            break
        if dlx < shr:
            break
    dl = dlx / nm1
    _inv(n, ww, xs, wwi)
    return niter, dl


@njit(cache=True, nogil=True)
def _glasso_core(sss, rrho, ipen, thr, maxit):
    nn = sss.shape[0]
    www = np.zeros((nn, nn), dtype=np.float64).T
    wwwi = np.zeros((nn, nn), dtype=np.float64).T
    ic = np.zeros((2, nn), dtype=np.int64)
    ir = np.zeros(nn, dtype=np.int64)
    nc = _connect(nn, sss, rrho, ic, ir)
    nniter = 0
    ddel = 0.0
    for kc in range(nc):
        n = ic[1, kc] - ic[0, kc] + 1
        if n <= 1:
            k = ir[ic[0, kc]]
            for i in range(nn):
                www[i, k] = 0.0
                www[k, i] = 0.0
                wwwi[i, k] = 0.0
                wwwi[k, i] = 0.0
            continue
        kb = ic[0, kc]
        ke = ic[1, kc]
        ss = np.empty((n, n), dtype=np.float64).T
        rho = np.empty((n, n), dtype=np.float64).T
        ww = np.empty((n, n), dtype=np.float64).T
        wwi = np.empty((n, n), dtype=np.float64).T
        for k in range(kb, ke + 1):
            ik = ir[k]
            for j in range(kb, ke + 1):
                ij = ir[j]
                ss[j - kb, k - kb] = sss[ij, ik]
                rho[j - kb, k - kb] = rrho[ij, ik]
                ww[j - kb, k - kb] = www[ij, ik]
                wwi[j - kb, k - kb] = wwwi[ij, ik]
        niter, dl = _lasinv1(n, ss, rho, ipen, thr, maxit, ww, wwi)
        nniter += niter
        ddel += dl
        for j in range(kb, ke + 1):
            k = ir[j]
            for i in range(nn):
                www[i, k] = 0.0
                www[k, i] = 0.0
                wwwi[i, k] = 0.0
                wwwi[k, i] = 0.0
        for k in range(kb, ke + 1):
            ik = ir[k]
            for j in range(kb, ke + 1):
                wwwi[ir[j], ik] = wwi[j - kb, k - kb]
        for k in range(kb, ke + 1):
            ik = ir[k]
            for j in range(kb, ke + 1):
                www[ir[j], ik] = ww[j - kb, k - kb]
    ddel = ddel / nc
    for j in range(nn):
        if www[j, j] != 0.0:
            continue
        www[j, j] = sss[j, j] + rrho[j, j] if ipen != 0 else sss[j, j]
        wwwi[j, j] = 1.0 / www[j, j]
    return www, wwwi, nniter, ddel


@dataclass
class GlassoResult:
    """Return value of :func:`glasso` (fields as in R's ``glasso()`` list)."""
    w: np.ndarray       #: estimated covariance matrix
    wi: np.ndarray      #: estimated inverse covariance matrix
    niter: int          #: number of sweeps (summed over connected components)
    del_: float         #: mean last-sweep change (``del`` in R)


def glasso(s, rho, thr: float = 1.0e-4, maxit: int = 10000,
           penalize_diagonal: bool = True) -> GlassoResult:
    """Graphical lasso of Friedman, Hastie & Tibshirani (2008) -- R ``glasso::glasso``.

    Parameters
    ----------
    s
        Square empirical covariance matrix.
    rho
        Scalar or square matrix of L1 penalties (``rho[i, j]`` penalises entry
        ``(i, j)`` of the inverse).
    thr, maxit, penalize_diagonal
        As in R (``thr = 1e-4``, ``maxit = 1e4``, ``penalize.diagonal = TRUE``).

    Returns
    -------
    :class:`GlassoResult` with ``w`` (covariance estimate) and ``wi`` (inverse).

    Notes
    -----
    Cold start, exact (``approx = FALSE``) path only -- the configuration
    Cicero uses. The arithmetic follows ``glasso.f`` operation by operation.
    """
    s = np.asfortranarray(np.asarray(s, dtype=np.float64))
    n = s.shape[0]
    if s.ndim != 2 or s.shape[1] != n:
        raise ValueError("s must be a square matrix")
    rho = np.asarray(rho, dtype=np.float64)
    if rho.ndim == 0 or rho.size == 1:
        rho = np.full((n, n), float(rho.reshape(-1)[0]))
    elif rho.ndim == 1:
        if rho.size != n:
            raise ValueError("Wrong number of elements in rho")
        rho = np.outer(np.sqrt(rho), np.sqrt(rho))
    if rho.shape != (n, n):
        raise ValueError("Wrong number of elements in rho")
    rho = np.asfortranarray(rho)
    w, wi, niter, d = _glasso_core(s, rho, 1 if penalize_diagonal else 0, float(thr), int(maxit))
    return GlassoResult(np.ascontiguousarray(w), np.ascontiguousarray(wi), int(niter), float(d))
