"""Minimum covariance determinant, as ``robust::covRob``'s ``mcd`` branch.

``robust::covRob(x)`` (robust 0.7.1) with ``estim="auto"`` uses
``robustbase::covMcd(x, cor = FALSE, control = covRob.control("mcd"))`` when
``n < 50000`` and ``p < 20`` (and the Donoho-Stahel rule does not apply), and
returns its *raw* centre and covariance. This module is a port of that path
(robustbase 0.95-1): the Fortran FAST-MCD routine ``rffastmcd`` (Rousseeuw and
Van Driessen) with its sub-dataset stages, C-steps and storage of the best
solutions, followed by the consistency factor ``.MCDcons`` and the small-sample
factor ``.MCDcnp2``.

FAST-MCD draws random subsets with R's ``unif_rand()``; :func:`cov_mcd` takes
an :class:`~epione.bulk.hic._r_rng.RRandom` and consumes exactly the same draws,
so the generator's state afterwards equals R's ``.Random.seed``. Exact-fit
situations (more than ``h`` observations on a hyperplane, a zero MAD, a
singular classical covariance) are not ported and raise
``NotImplementedError``. Parity: ``tests/parity/schicluster_compartment``
(gate D8).
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numba import njit

from ._r_rng import RRandom

_INT_MAX = 2146666666
_UNSET = 1234567.0

# status codes of _fastmcd
_OK, _EXACT_FIT, _NEED_DRAWS, _UNSUPPORTED = 0, 1, 2, 3


@njit(cache=True)
def _rffindq(aw, ncas, k, index):
    """k-th order statistic by partial sorting (Fortran ``rffindq``); k is 1-based."""
    for j in range(ncas):
        index[j] = j + 1
    lo = 1
    lr = ncas
    while lo < lr:
        ax = aw[k - 1]
        jnc = lo
        j = lr
        while jnc <= j:
            while aw[jnc - 1] < ax:
                jnc += 1
            while aw[j - 1] > ax:
                j -= 1
            if jnc <= j:
                t = index[jnc - 1]
                index[jnc - 1] = index[j - 1]
                index[j - 1] = t
                w = aw[jnc - 1]
                aw[jnc - 1] = aw[j - 1]
                aw[j - 1] = w
                jnc += 1
                j -= 1
        if j < k:
            lo = jnc
        if k < jnc:
            lr = j
    return aw[k - 1]


@njit(cache=True)
def _covar_rows(src, rows, nrow, p, cova, means):
    """``rfcovinit`` + ``rfadmit`` over ``rows[:nrow]`` (1-based) + ``rfcovar``."""
    sscp = np.zeros((p + 1, p + 1))
    for r in range(nrow):
        rec = src[rows[r] - 1]
        sscp[0, 0] = sscp[0, 0] + 1.0
        for j in range(p):
            sscp[0, j + 1] = sscp[0, j + 1] + rec[j]
            sscp[j + 1, 0] = sscp[0, j + 1]
        for i in range(p):
            for j in range(p):
                sscp[i + 1, j + 1] = sscp[i + 1, j + 1] + rec[i] * rec[j]
    nd = float(nrow)
    for i in range(p):
        means[i] = sscp[0, i + 1]
    for i in range(p):
        means[i] = means[i] / nd
    for i in range(p):
        for j in range(p):
            cova[i, j] = sscp[i + 1, j + 1]
    for i in range(p):
        for j in range(p):
            cova[i, j] = cova[i, j] - nd * means[i] * means[j]
            cova[i, j] = cova[i, j] / (nrow - 1)


@njit(cache=True)
def _sweep_det(cova, cinv, p, eps):
    """Copy ``cova`` into ``cinv``, sweep all pivots (``rfcovsweep``), sum log pivots.

    Returns ``(det, j)``; ``j >= 0`` is the first singular pivot (sweep stopped)."""
    for i in range(p):
        for j in range(p):
            cinv[i, j] = cova[i, j]
    det = 0.0
    for k in range(p):
        piv = cinv[k, k]
        det = det + np.log(piv)
        if piv < eps:
            return det, k
        d = cinv[k, k]
        for j in range(p):
            cinv[k, j] = cinv[k, j] / d
        for i in range(p):
            if i != k:
                b = cinv[i, k]
                for j in range(p):
                    cinv[i, j] = cinv[i, j] - b * cinv[k, j]
                cinv[i, k] = -b / d
        cinv[k, k] = 1.0 / d
    return det, -1


@njit(cache=True)
def _mahad_all(src, nn, p, means, cinv, ndist):
    for r in range(nn):
        t = 0.0
        for j in range(p):
            dj = src[r, j] - means[j]
            for k in range(p):
                t = t + dj * (src[r, k] - means[k]) * cinv[j, k]
        ndist[r] = t


@njit(cache=True)
def _plane_qorder(src, nn, p, cova, means, nhalf):
    """nhalf-th smallest distance to the plane of the smallest eigenvector."""
    w, v = np.linalg.eigh(cova)
    z = v[:, 0]
    d = np.empty(nn)
    for r in range(nn):
        s = 0.0
        for j in range(p):
            s = z[j] * (src[r, j] - means[j]) + s
        d[r] = abs(s)
    d.sort()
    return d[nhalf - 1]


@njit(cache=True)
def _fastmcd(dat, nhalff, krep, nmini, kmini, u):
    """Fortran ``rffastmcd`` for p >= 2 and the general-position path.

    Returns ``(status, initmean, initcov, object, n_draws)``."""
    n, p = dat.shape
    pos = 0
    nu = u.shape[0]
    k1, k2, k3 = 2, 2, 100
    nrep = krep
    kstep = k1
    percen = nhalff / n
    if p < 5:
        eps = 1.0e-12
    elif p <= 8:
        eps = 1.0e-14
    else:
        eps = 1.0e-16
    initmean = np.zeros(p)
    initcov = np.zeros((p, p))
    nmaxi = nmini * kmini
    km10 = 10 * kmini
    nsel = p + 1
    ngroup = 1
    fine = False
    final = False
    subdat = np.full((2, nmaxi), _INT_MAX, dtype=np.int64)
    mini = np.zeros(kmini, dtype=np.int64)
    part = krep > 0 and n >= 2 * nmini
    minigr = n
    nhalf = nhalff
    if part:
        kstep = k1
        ngroup = n // nmini
        if ngroup < kmini:
            mm = n // ngroup
            ii = n - ngroup * mm
            jj = ngroup - ii
            for j in range(jj):
                mini[j] = mm
            for j in range(jj, ngroup):
                mini[j] = mm + 1
            minigr = ngroup * mm + ii
        else:
            ngroup = kmini
            for j in range(kmini):
                mini[j] = nmini
            minigr = kmini * nmini
        nhalf = int(mini[0] * percen)
        nrep = krep // ngroup
        # rfrdraw: ngroup nonoverlapping subdatasets
        jndex = 0
        for k in range(1, ngroup + 1):
            for m in range(mini[k - 1]):
                if pos >= nu:
                    return _NEED_DRAWS, initmean, initcov, 0.0, pos
                nrand = int(u[pos] * (n - jndex)) + 1
                pos += 1
                jndex += 1
                if jndex == 1:
                    subdat[0, 0] = nrand
                    subdat[1, 0] = k
                else:
                    subdat[0, jndex - 1] = nrand + jndex - 1
                    subdat[1, jndex - 1] = k
                    for i in range(1, jndex):
                        if subdat[0, i - 1] > nrand + i - 1:
                            for j in range(jndex, i, -1):
                                subdat[0, j - 1] = subdat[0, j - 2]
                                subdat[1, j - 1] = subdat[1, j - 2]
                            subdat[0, i - 1] = nrand + i - 1
                            subdat[1, i - 1] = k
                            break
    else:
        minigr = n
        nhalf = nhalff
        kstep = k1
        replow = 0
        if nsel <= 6:
            replow = (500, 50, 22, 17, 15, 14)[nsel - 1]
        if krep == 0 or n <= replow:
            return _UNSUPPORTED, initmean, initcov, 0.0, pos  # all subsets: not ported
        nrep = krep

    mstock = np.full((10, p), _UNSET)
    cstock = np.full((10, p, p), _UNSET)
    m1stock = np.full((km10, p), _UNSET)
    c1stock = np.full((km10, p, p), _UNSET)
    mcdndex = np.zeros((10, 2, kmini))
    flag = np.ones(km10, dtype=np.int64)
    means = np.zeros(p)
    bmeans = np.zeros(p)
    med = np.zeros(p)
    mad = np.zeros(p)
    cova1 = np.zeros((p, p))
    cinv1 = np.zeros((p, p))
    cova2 = np.zeros((p, p))
    cinv2 = np.zeros((p, p))
    ndist = np.zeros(n)
    index1 = np.full(n, _INT_MAX, dtype=np.int64)
    index2 = np.full(n, _INT_MAX, dtype=np.int64)
    indexx = np.full(n, _INT_MAX, dtype=np.int64)
    temp = np.zeros(n, dtype=np.int64)
    dath = np.zeros((nmaxi, p))
    subndex = np.zeros((3 * nmini + 1) // 2 + 1, dtype=np.int64)
    am = np.zeros(n)
    am2 = np.zeros(n)

    # classical estimate: only its singularity matters here
    allrows = np.arange(1, n + 1)
    _covar_rows(dat, allrows, n, p, cova1, means)
    for j in range(p):
        if not cova1[j, j] > 0.0:
            return _EXACT_FIT, initmean, initcov, 0.0, pos
    det, jsing = _sweep_det(cova1, cinv1, p, eps)
    if jsing >= 0:
        return _EXACT_FIT, initmean, initcov, 0.0, pos

    obj = 0.0
    while True:  # label 5555
        obj = 10.0e25
        if (not part) or final:
            nn = n
        elif fine:
            nn = minigr
        else:
            nn = -1
        if fine or ((not part) and final):
            nrep = 10
            nsel = nhalf
            kstep = k2
            if final:
                nhalf = nhalff
                ngroup = 1
                npv = n * p
                if npv <= 100000:
                    kstep = k3
                elif npv <= 200000:
                    kstep = 10
                elif npv <= 300000:
                    kstep = 9
                elif npv <= 400000:
                    kstep = 8
                elif npv <= 500000:
                    kstep = 7
                elif npv <= 600000:
                    kstep = 6
                elif npv <= 700000:
                    kstep = 5
                elif npv <= 800000:
                    kstep = 4
                elif npv <= 900000:
                    kstep = 3
                elif npv <= 1000000:
                    kstep = 2
                else:
                    kstep = 1
                if n > 5000:
                    nrep = 1
            else:
                nhalf = int(minigr * percen)
        for i in range(nsel - 1):
            index1[i] = i + 1
            indexx[i] = i + 1
        index1[nsel - 1] = nsel - 1
        indexx[nsel - 1] = nsel - 1
        if not final:
            for i in range(10):
                for j in range(ngroup):
                    mcdndex[i, 0, j] = 10.0e25
                    mcdndex[i, 1, j] = 10.0e25
        if (not fine) and (not final):
            for j in range(p):
                for i in range(n):
                    am[i] = dat[i, j]
                    am2[i] = dat[i, j]
                # Fortran tests 2*n/2 .eq. n, which always holds: the even-n formula
                med1 = _rffindq(am, n, n // 2, index2)
                med2 = _rffindq(am2, n, (n + 2) // 2, index2)
                med[j] = (med1 + med2) / 2
                for i in range(n):
                    ndist[i] = abs(dat[i, j] - med[j])
                mad[j] = _rffindq(ndist, n, nhalff, index2)
                if mad[j] - 0.0 < eps:
                    return _EXACT_FIT, initmean, initcov, 0.0, pos
                for i in range(n):
                    dat[i, j] = (dat[i, j] - med[j]) / mad[j]
        if fine and (not final):
            for j in range(minigr):
                for k in range(p):
                    dath[j, k] = dat[subdat[0, j] - 1, k]
        kount = 0

        for ii in range(1, ngroup + 1):  # loop 1111
            if not fine:
                kount = 0
            if part and (not fine):
                nn = mini[ii - 1]
                kk = 0
                for j in range(minigr):
                    if subdat[1, j] == ii:
                        subndex[kk] = subdat[0, j]
                        kk += 1
                for j in range(mini[ii - 1]):
                    for k in range(p):
                        dath[j, k] = dat[subndex[j] - 1, k]
            for i in range(nn):
                index2[i] = i + 1
            for ix in range(n):
                indexx[ix] = index1[ix]
            usedat = (not part) or final
            src = dat if usedat else dath

            for i in range(1, nrep + 1):  # loop 1000
                pnsel = nsel
                deti = -1.0e300
                detimin1 = deti
                if (part and not fine) or ((not part) and (not final)):
                    nn_draw = mini[ii - 1] if part else n
                    for s in range(nsel):  # rfrangen
                        while True:
                            if pos >= nu:
                                return _NEED_DRAWS, initmean, initcov, 0.0, pos
                            num = int(u[pos] * nn_draw) + 1
                            pos += 1
                            dup = False
                            for t in range(s):
                                if index1[t] == num:
                                    dup = True
                                    break
                            if not dup:
                                break
                        index1[s] = num
                skip_group = False
                while True:  # label 9550
                    if (not fine) and part:
                        _covar_rows(dath, index1, pnsel, p, cova1, means)
                    if (not part) and (not final):
                        _covar_rows(dat, index1, pnsel, p, cova1, means)
                    if final:
                        if mstock[i - 1, 0] != _UNSET:
                            for jj in range(p):
                                means[jj] = mstock[i - 1, jj]
                                for kk in range(p):
                                    cova1[jj, kk] = cstock[i - 1, jj, kk]
                        else:
                            skip_group = True
                            break
                        if flag[i - 1] == 0:
                            return _UNSUPPORTED, initmean, initcov, 0.0, pos
                    if fine and (not final):
                        row = (ii - 1) * 10 + i - 1
                        if m1stock[row, 0] != _UNSET:
                            for jj in range(p):
                                means[jj] = m1stock[row, jj]
                                for kk in range(p):
                                    cova1[jj, kk] = c1stock[row, jj, kk]
                        else:
                            skip_group = True
                            break
                        if flag[row] == 0:
                            return _UNSUPPORTED, initmean, initcov, 0.0, pos
                    det, jsing = _sweep_det(cova1, cinv1, p, eps)
                    if jsing < 0:
                        break
                    # singular starting subset
                    if fine or final:
                        return _EXACT_FIT, initmean, initcov, 0.0, pos
                    qorder = _plane_qorder(src, nn, p, cova1, means, nhalf)
                    if abs(qorder - 0.0) < 10.0e-8:
                        return _EXACT_FIT, initmean, initcov, 0.0, pos
                    # extend the subsample by one random observation (prdraw)
                    srt = np.sort(index1[:pnsel])
                    for t in range(pnsel):
                        index1[t] = srt[t]
                    if pos >= nu:
                        return _NEED_DRAWS, initmean, initcov, 0.0, pos
                    jndex = pnsel
                    nrand = int(u[pos] * (nn - jndex)) + 1
                    pos += 1
                    jndex += 1
                    index1[jndex - 1] = nrand + jndex - 1
                    for t in range(1, jndex):
                        if index1[t - 1] > nrand + t - 1:
                            for jj in range(jndex, t, -1):
                                index1[jj - 1] = index1[jj - 2]
                            index1[t - 1] = nrand + t - 1
                            break
                    pnsel += 1
                if skip_group:
                    break

                _mahad_all(src, nn, p, means, cinv1, ndist)
                dist2 = _rffindq(ndist, nn, nhalf, index2)

                for step in range(1, kstep + 1):  # label 9555 / loop 400
                    for j in range(nhalf):
                        temp[j] = index2[j]
                    srt = np.sort(temp[:nhalf])
                    _covar_rows(src, srt, nhalf, p, cova1, means)
                    det, jsing = _sweep_det(cova1, cinv1, p, eps)
                    if jsing >= 0:
                        return _EXACT_FIT, initmean, initcov, 0.0, pos
                    if step >= 2 and det == detimin1:
                        break  # goto 5000
                    detimin1 = deti
                    deti = det
                    _mahad_all(src, nn, p, means, cinv1, ndist)
                    dist2 = _rffindq(ndist, nn, nhalf, index2)
                    if final and ((i == 1 and step == 1 and not fine) or det < obj):
                        _rffindq(ndist, nn, n // 2, index1)  # medi2 (not needed for the raw estimate)
                        obj = det
                        for a in range(p):
                            bmeans[a] = means[a]
                            for b in range(p):
                                cova2[a, b] = cova1[a, b]
                                cinv2[a, b] = cinv1[a, b]

                # label 5000: keep the 10 best solutions
                if not final:
                    iii = ii if (part and not fine) else 1
                    lll = 1 if flag[(iii - 1) * 10] == 1 else 2
                    to_part = (not fine) and part
                    for j in range(lll, 11):
                        if det <= mcdndex[j - 1, 1, iii - 1]:
                            insert = False
                            if det != mcdndex[j - 1, 1, iii - 1]:
                                insert = True
                            else:
                                for kkk in range(j, 11):
                                    if det == mcdndex[kkk - 1, 1, iii - 1]:
                                        for jjj in range(p):
                                            ref = (m1stock[(iii - 1) * 10 + kkk - 1, jjj] if to_part
                                                   else mstock[kkk - 1, jjj])
                                            if means[jjj] != ref:
                                                insert = True
                                                break
                                        if insert:
                                            break
                                        for a in range(p):
                                            for b in range(p):
                                                ref = (c1stock[(iii - 1) * 10 + kkk - 1, a, b] if to_part
                                                       else cstock[kkk - 1, a, b])
                                                if cova1[a, b] != ref:
                                                    insert = True
                                                    break
                                            if insert:
                                                break
                                        if insert:
                                            break
                            if insert:
                                if to_part:  # label 203
                                    base = (iii - 1) * 10
                                    for k in range(10, j, -1):
                                        c1stock[base + k - 1] = c1stock[base + k - 2]
                                        m1stock[base + k - 1] = m1stock[base + k - 2]
                                        mcdndex[k - 1, 0, iii - 1] = mcdndex[k - 2, 0, iii - 1]
                                        mcdndex[k - 1, 1, iii - 1] = mcdndex[k - 2, 1, iii - 1]
                                    for a in range(p):
                                        m1stock[base + j - 1, a] = means[a]
                                        for b in range(p):
                                            c1stock[base + j - 1, a, b] = cova1[a, b]
                                else:  # label 205
                                    for k in range(10, j, -1):
                                        cstock[k - 1] = cstock[k - 2]
                                        mstock[k - 1] = mstock[k - 2]
                                        mcdndex[k - 1, 0, iii - 1] = mcdndex[k - 2, 0, iii - 1]
                                        mcdndex[k - 1, 1, iii - 1] = mcdndex[k - 2, 1, iii - 1]
                                    for a in range(p):
                                        mstock[j - 1, a] = means[a]
                                        for b in range(p):
                                            cstock[j - 1, a, b] = cova1[a, b]
                                mcdndex[j - 1, 0, iii - 1] = i
                                mcdndex[j - 1, 1, iii - 1] = det
                            break  # goto 1000
        if part and not fine:
            fine = True
            continue
        if (not final) and ((part and fine) or (not part)):
            final = True
            continue
        break

    for j in range(p):
        initmean[j] = bmeans[j] * mad[j] + med[j]
    for i in range(p):
        for j in range(p):
            initcov[i, j] = cova2[i, j] * mad[i] * mad[j]
    for j in range(p):
        obj = obj + 2 * math.log(mad[j])
    return _OK, initmean, initcov, obj, pos


def _h_alpha_n(alpha: float, n: int, p: int) -> int:
    n2 = (n + p + 1) // 2
    return int(math.floor(2 * n2 - n + 2 * (n - n2) * alpha))


def _mcd_cons(p: int, alpha: float) -> float:
    """robustbase ``.MCDcons``: consistency factor of the raw MCD."""
    from scipy.special import gammainc
    from scipy.stats import chi2

    qalpha = chi2.ppf(alpha, p)
    ca = gammainc(p / 2 + 1, qalpha / 2) / alpha
    return 1.0 / ca


def _mcd_cnp2(p: int, n: int, alpha: float) -> float:
    """robustbase ``.MCDcnp2``: small-sample correction of the raw MCD (formula part)."""
    if alpha == 0.5 and 1 <= p <= 9:
        n_min = (1, 4, 6, 8, 10, 12, 14, 16, 18)[p - 1]
        n_max = (2, 6, 10, 13, 16, 18, 20, 20, 20)[p - 1]
        if n_min <= n <= n_max:
            raise NotImplementedError("covMcd's simulated small-sample factors (n <= 20) are not ported")
    if p > 2:
        c875 = np.array([[-0.455179464070565, -0.294241208320834],
                         [1.11192541278794, 1.09649329149811],
                         [2.0, 3.0]])
        c500 = np.array([[-1.42764571687802, -1.06141115981725],
                         [1.26263336932151, 1.28907991440387],
                         [2.0, 3.0]])
        y500 = np.log(-c500[0] / p ** c500[1])
        y875 = np.log(-c875[0] / p ** c875[1])
        A500 = np.column_stack([np.ones(2), -np.log(c500[2] * p ** 2)])
        A875 = np.column_stack([np.ones(2), -np.log(c875[2] * p ** 2)])
        co500 = np.linalg.solve(A500, y500)
        co875 = np.linalg.solve(A875, y875)
        f500 = 1 - math.exp(co500[0]) / n ** co500[1]
        f875 = 1 - math.exp(co875[0]) / n ** co875[1]
    elif p == 2:
        f500 = 1 - math.exp(0.673292623522027) / n ** 0.691365864961895
        f875 = 1 - math.exp(0.446537815635445) / n ** 1.06690782995919
    else:
        f500 = 1 - math.exp(0.262024211897096) / n ** 0.604756680630497
        f875 = 1 - math.exp(-0.351584646688712) / n ** 1.01646567502486
    if alpha <= 0.875:
        f = f500 + (f875 - f500) / 0.375 * (alpha - 0.5)
    else:
        f = f875 + (1 - f875) / 0.125 * (alpha - 0.875)
    return 1.0 / f


def cov_mcd(
    x,
    *,
    rng: RRandom,
    alpha: float = 0.5,
    nsamp: int = 500,
    nmini: int = 300,
    kmini: int = 5,
    use_correction: bool = True,
    return_details: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Raw MCD centre and covariance, equal to ``robust::covRob``'s ``mcd`` branch.

    Arguments:
        x: ``n × p`` data (rows with non-finite values are dropped), p >= 2.
        rng: R's random number generator in its current state (R continues
            its global stream; covRob does not reseed). It is advanced by the
            draws FAST-MCD makes.
        alpha, nsamp, nmini, kmini, use_correction: ``covRob.control("mcd")``
            defaults.
        return_details: also return ``{"h", "log_det", "calpha", "correct",
            "draws"}``.

    Returns:
        ``(center, cov)``: ``covMcd(x)$raw.center`` and ``$raw.cov`` (the
        covariance of the best h-subset times the consistency and
        small-sample factors).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x @ np.ones(x.shape[1]))]
    n, p = x.shape
    if p < 2:
        raise NotImplementedError("p = 1 (univariate MCD) is not ported")
    if n <= p + 1:
        raise ValueError("n <= p + 1: sample too small for MCD")
    h = _h_alpha_n(alpha, n, p)
    if h >= n:
        raise NotImplementedError("h = n (classical estimate) is not ported")
    state = rng.get_state()
    ndraw = 2 * nmini * kmini + 4 * nsamp * (p + 1) + 1000
    while True:
        u = rng.unif_rand(ndraw)
        status, mean, cov, logdet, used = _fastmcd(
            np.ascontiguousarray(x.copy()), int(h), int(nsamp), int(nmini), int(kmini), np.ascontiguousarray(u))
        if status != _NEED_DRAWS:
            break
        rng.set_state(state)
        ndraw *= 4
    rng.set_state(state)
    if used:
        rng.unif_rand(int(used))
    if status == _EXACT_FIT:
        raise NotImplementedError(
            "MCD exact fit (observations on a hyperplane, zero MAD or singular covariance) is not ported")
    if status == _UNSUPPORTED:
        raise NotImplementedError("this FAST-MCD path (all subsets or hyperplane solutions) is not ported")
    calpha = _mcd_cons(p, h / n)
    correct = _mcd_cnp2(p, n, alpha) if use_correction else 1.0
    raw_cov = (calpha * correct) * cov
    if return_details:
        return mean, raw_cov, {"h": h, "log_det": logdet, "calpha": calpha, "correct": correct,
                               "draws": int(used)}
    return mean, raw_cov
