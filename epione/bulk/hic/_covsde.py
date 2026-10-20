"""Stahel–Donoho robust covariance, as ``robust::covRob`` computes it.

``robust::covRob(x)`` (robust 0.7.1) picks the Donoho–Stahel estimator when
``(n < 1000 and p < 10) or (n < 5000 and p < 5)``. It then calls
``set.seed(21)`` and ``rrcov::CovSde(x, tune = 0.95, prob = 0.99, eps = 0.5)``
(rrcov 1.7-3), whose work is done by the Fortran routine ``rlds``:

1. **Outlyingness.** Draw ``nsamp`` random subsets of ``p`` observations. Each
   subset spans a hyperplane, with the subset centred on its last point; the
   unit normal of that hyperplane is a projection direction. For every
   direction, each observation's outlyingness is
   ``|x a - median(x a)| / MAD``. Keep the largest value per observation.
2. **Weights.** Pass outlyingness / ``sqrt(qchisq(tune, p))`` through a
   smooth weight (1 below 0.8, 0 above 1, a degree-6 polynomial in between).
3. **Estimate.** Weighted mean, and covariance with squared weights.

``CovSde`` finally scales the covariance so that the median Mahalanobis
distance equals ``qchisq(0.5, p)``.

This module reproduces those steps with the same random draws
(:class:`~epione.bulk.hic._r_rng.RRandom`) and the same order of floating-point
operations, so the estimate matches R to rounding. Parity: gate D6 in
``tests/parity/schicluster_compartment/MANIFEST.md``.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ._r_rng import RRandom

# R's qchisq(0.5, p) and qchisq(0.95, p), p = 1..9 (R 4.1.3, printed with %.17g).
# The Donoho-Stahel branch of covRob only runs for p < 10; scipy's chi2.ppf
# differs from R's qchisq by about one ulp, so R's values are used here.
_R_QCHISQ = {
    1: (0.45493642311957283, 3.841458820694124),
    2: (1.3862943611198906, 5.9914645471079799),
    3: (2.3659738843753386, 7.8147279032511792),
    4: (3.356693980033322, 9.487729036781154),
    5: (4.3514601910955282, 11.070497693516351),
    6: (5.3481206274471207, 12.591587243743978),
    7: (6.3458111955215166, 14.067140449340163),
    8: (7.3441214977017939, 15.507313055865451),
    9: (8.342832692252955, 16.918977604620451),
}

_TOLA, _TOLR, _BIG1, _BIG2 = 1e-15, 1e-8, 1e2, 1e15


def _qchisq(q: float, p: int) -> float:
    if p in _R_QCHISQ and q in (0.5, 0.95):
        return _R_QCHISQ[p][0 if q == 0.5 else 1]
    from scipy.stats import chi2
    return float(chi2.ppf(q, p))


# --- small-vector helpers: sequential sums, as the Fortran loops ------------

def _dprod(x, y) -> float:
    s = 0.0
    for a, b in zip(x, y):
        s += a * b
    return s


def _xnorma(v, tol) -> int:
    """``rlxnorma``: normalise ``v`` in place; 1 if its norm is <= ``tol``."""
    dn = math.sqrt(_dprod(v, v))
    if dn <= tol:
        return 1
    for i in range(len(v)):
        v[i] = v[i] / dn
    return 0


def _direction(pts) -> Optional[list]:
    """``rlvectora`` + ``rldirec`` (icent = 1): unit normal of the hyperplane
    through ``p`` points, or ``None`` if they are (numerically) degenerate.

    ``pts[k]`` is the k-th sampled observation (a length-p list).
    """
    p = len(pts)
    # xx(:, k) = observation k; centre the first p-1 on the last one
    cols = [list(pt) for pt in pts]
    last = cols[p - 1]
    m = p - 1
    for k in range(m):
        ck = cols[k]
        for i in range(p):
            ck[i] = ck[i] - last[i]
    # rlorthog(xx, nn = p, mm = p - 1)
    for j in range(m):
        if _xnorma(cols[j], _TOLA):
            return None
    for j in range(m - 1):
        if _xnorma(cols[j], _TOLR):
            return None
        cj = cols[j]
        for k in range(j + 1, m):
            ck = cols[k]
            dp = _dprod(cj, ck)
            for i in range(p):
                ck[i] = ck[i] - cj[i] * dp
    if _xnorma(cols[m - 1], _TOLA):  # tolb = tola
        return None
    # rlortdir(xx, mm = p): projector on the orthogonal complement, first
    # column whose norm reaches 1 / sqrt(p)
    tol = 1.0 / math.sqrt(float(p))
    dire = [0.0] * p
    dn = 0.0
    for k in range(p):
        for i in range(p):
            d = 0.0
            for j in range(m):
                d = d - cols[j][i] * cols[j][k]
            dire[i] = d
        dire[k] = dire[k] + 1.0
        dn = math.sqrt(_dprod(dire, dire))
        if dn >= tol:
            break
    return [d / dn for d in dire]


def _rwetml(x: np.ndarray) -> np.ndarray:
    """``RLRWETML`` weight function (vectorised; same operations per element)."""
    ax = np.abs(x)
    x2 = ax * ax
    x3 = x2 * ax
    x4 = x2 * x2
    x6 = x3 * x3
    poly = (-19.7187928669416 + 82.3045267489739 * x2
            + -105.4526748971229 * x4 + 42.8669410150906 * x6)
    return np.where(ax >= 1.0, 0.0, np.where(ax <= 0.8, 1.0, poly))


def _quntbi(p: float) -> float:
    """``RLQUNTBI``: rational approximation of the normal quantile."""
    c = (2.515517, 0.802853, 0.010328, 1.432788, 0.189269, 0.001308)
    p1 = 1.0 - p if p > 0.5 else p
    t = math.sqrt(-2.0 * math.log(p1))
    xz = (c[2] * t + c[1]) * t + c[0]
    xn = ((c[5] * t + c[4]) * t + c[3]) * t + 1.0
    x = t - xz / xn
    return -x if p < 0.5 else x


def _seqsum(v: np.ndarray) -> float:
    return float(np.cumsum(v)[-1]) if v.size else 0.0


def _rlds(x: np.ndarray, nsamp: int, maxres: int, c: float, rng: RRandom):
    """Fortran ``rlds`` (icent = 1): returns center, cov, weights, nresper."""
    n, p = x.shape
    k1 = (p - 1) + (n + 1) // 2
    k2 = (p - 1) + (n + 2) // 2
    cc = _quntbi((1 + (float(k1) / float(n))) / 2)
    z = np.full(n, -1.0)
    cols = [x[:, j].copy() for j in range(p)]
    rows = x.tolist()
    nresper = 0
    k = 0
    mid = n // 2
    while k < maxres and nresper < nsamp:
        k += 1
        # rlsubsamp: p distinct 1-based indices, int(n * u + 1)
        ind = []
        while len(ind) < p:
            i = int(n * rng.unif_rand() + 1.0)
            if i not in ind:
                ind.append(i)
        a = _direction([rows[i - 1] for i in ind])
        if a is None:
            continue
        nresper += 1
        b = np.zeros(n)
        for j in range(p):
            b = b + cols[j] * a[j]
        s = np.sort(b)
        bmed = s[mid]
        if 2 * mid >= n:
            bmed = (bmed + s[mid - 1]) / 2.0
        w = np.abs(b - bmed)
        ww = _seqsum(w) / n
        if ww >= _TOLA:
            ws = np.sort(w)
            bmad = (ws[k1 - 1] + ws[k2 - 1]) / 2
            bmad = bmad / cc
            if bmad >= _TOLR * ww:
                aux = np.abs(b - bmed) / bmad
                z = np.where(aux > z, aux, z)
            else:
                z = np.where(np.abs(b - bmed) > _BIG1 * bmad, _BIG2, z)
    wts = _rwetml(z / c)

    # rldonostah (icent = 1)
    sumw = _seqsum(wts)
    sumw2 = _seqsum(wts * wts)
    locat = np.array([_seqsum(wts * cols[j]) / sumw for j in range(p)])
    cov = np.empty((p, p))
    for j in range(p):
        dj = wts * (cols[j] - locat[j])
        for kk in range(p):
            cov[j, kk] = _seqsum(dj * wts * (cols[kk] - locat[kk])) / sumw2
    return locat, cov, wts, nresper, z


def cov_sde(
    x,
    *,
    tune: float = 0.95,
    prob: float = 0.99,
    eps: float = 0.5,
    nsamp: Optional[int] = None,
    maxres: Optional[int] = None,
    seed: Optional[int] = 21,
    return_details: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stahel–Donoho location and scatter, equal to ``robust::covRob``'s
    ``donostah`` branch (``set.seed(21); rrcov::CovSde(x)``).

    Arguments:
        x: ``n × p`` data (rows with non-finite values are dropped, as R does).
        tune: weight cut-off as a chi-square probability; the outlyingness is
            divided by ``sqrt(qchisq(tune, p))``.
        prob, eps: probability of drawing at least one outlier-free subset
            given a fraction ``eps`` of outliers; set the default number of
            subsets ``max(1000, ceil(log(1 - prob) / log(1 - (1 - eps)^(p + 1))))``.
        nsamp: number of valid subsets (overrides the default).
        maxres: maximum number of subsets drawn, including degenerate ones;
            default ``2 * nsamp``.
        seed: R ``set.seed`` value; covRob uses 21. ``None`` is not supported
            (R would continue its global stream).
        return_details: also return the weights, the outlyingness and the
            number of valid subsets.

    Returns:
        ``(center, cov)``: ``center`` of length ``p`` and the ``p × p``
        covariance, after CovSde's consistency correction
        ``median(mahalanobis) / qchisq(0.5, p)``. With ``return_details``, a
        third item ``{"weights": ..., "outlyingness": ..., "nresper": ...}``.
    """
    if seed is None:
        raise ValueError("seed=None (continue R's global stream) is not supported")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be a 2-D array")
    ok = np.isfinite(x @ np.ones(x.shape[1]))
    x = x[ok]
    n, p = x.shape
    if p < 2:
        raise ValueError("x must have at least two columns (as robust::covRob)")
    if n <= p + 1:
        raise ValueError("n <= p + 1: sample too small")
    if nsamp is None:
        nsamp = int(math.ceil(math.log(1 - prob) / math.log(1 - (1 - eps) ** (p + 1))))
    if nsamp != 0:
        nsamp = max(1000, int(nsamp))
    else:
        raise NotImplementedError("nsamp = 0 (all subsets) is not ported")
    nsamp = min(nsamp, 2**31 - 1)
    if maxres is None:
        maxres = 2 * nsamp
    maxres = min(int(maxres), 2**31 - 1)
    c = math.sqrt(_qchisq(tune, p))

    rng = RRandom(seed)
    center, cov, wts, nresper, z = _rlds(x, nsamp, maxres, c, rng)

    # CovSde: consistency correction from the Mahalanobis distances
    d = x - center
    inv = np.linalg.solve(cov, np.eye(p))
    mah = np.sum((d @ inv) * d, axis=1)
    cov = cov * (np.median(mah) / _qchisq(0.5, p))
    if return_details:
        return center, cov, {"weights": wts, "outlyingness": z, "nresper": nresper}
    return center, cov
