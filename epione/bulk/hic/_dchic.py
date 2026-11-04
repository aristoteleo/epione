"""Differential A/B compartments across conditions (dcHiC ``analyze``).

Port of the ``--pcatype analyze`` step of dcHiC (Chakraborty, Wang & Ay 2022,
*Nat. Commun.* 13, 6827; ay-lab/dcHiC ``144b4df``, ``dchicf.r``: ``pcanalyze``,
``calcen``, ``calchclust``; MIT), the statistic behind

.. code-block:: bash

    Rscript dchicf.r --file input.txt --pcatype analyze --dirovwt T --diffdir all

Per chromosome, the compartment scores (one column per sample) are quantile
normalised across all samples (``limma::normalizeQuantiles``), averaged per
condition, and every bin gets a Mahalanobis distance of its condition profile
from a shrunken centre. The covariance is ``robust::covRob`` of the condition
profiles, and each bin's quadratic form is scaled by the squared spread of its
profile. P-values are chi-square (``n_conditions - 1`` d.f.), refined once after
dropping the most extreme bins, and adjusted genome-wide.

The port follows R value for value. Between steps dcHiC writes its tables with
``write.table`` (15 significant digits, after ``round(x, 5)`` for the normalised
scores) and reads them back; the port repeats those round trips with R's
``round`` (``fround.c``) and R's number parser (``R_strtod``), and computes means,
row means and variances with R's long-double accumulation. ``robust::covRob``
picks its estimator from the data size; the ``pairwiseqc`` / ``pairwisegk``
branch (orthogonalised Gnanadesikan-Kettenring, ``rrcov::CovOgk``, used when
there are 20 or more conditions) is ported from ``rrcov`` 1.7-3 ``covOPW.c``;
the ``donostah`` branch (few conditions and bins, ``rrcov::CovSde``) is in
``epione.bulk.hic._covsde`` and the ``mcd`` branch (fewer than 20 conditions
with many bins, ``robustbase::covMcd``) in ``epione.bulk.hic._covmcd``. Those
two draw random subsets; one emulated R generator (``_r_rng``), seeded like
dcHiC's ``set.seed(--seed)``, is threaded through all ``covRob`` calls in
dcHiC's chromosome order, so the draws are R's.

Not ported yet (``NotImplementedError``): IHW, which dcHiC uses for the
adjusted p-value when any condition has replicates.

When every pairwise quadrant correlation of the condition profiles is equal
(it happens on small refinement subsets), the OGK eigenvectors are not unique
and R's result is fixed by its LAPACK build's rounding. The port then follows
the environment's LAPACK (``sys.prefix/lib/liblapack``; ``EPIONE_LAPACK``
overrides) and warns; with conda's OpenBLAS it reproduces R. ``padj_method="BH"`` is an explicit opt-in that replaces IHW with
Benjamini-Hochberg; it is a deviation from dcHiC. See
``tests/parity/schicluster_compartment/MANIFEST.md``.
"""
from __future__ import annotations

import math
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

_LD = np.longdouble
_MANIFEST = "tests/parity/schicluster_compartment/MANIFEST.md"


# --------------------------------------------------------------------------
# R numerics
# --------------------------------------------------------------------------

def _r_pow_di(x: float, n: int) -> float:
    """``R_pow_di``: integer power by repeated squaring."""
    xn = 1.0
    neg = n < 0
    n = abs(n)
    while True:
        if n & 1:
            xn *= x
        n >>= 1
        if n:
            x *= x
        else:
            break
    return 1.0 / xn if neg else xn


def r_round(x, digits: int = 5) -> np.ndarray:
    """R's ``round(x, digits)`` (``fround.c``, R 4.0.0 and later).

    Rounds to the nearer of the two decimal neighbours; exact ties go to the
    even last digit. Differs from :func:`numpy.round` (which scales and
    rounds half to even on the scaled value) in a few last-digit cases.
    """
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    ok = np.isfinite(x) & (x != 0.0)
    if not ok.any():
        return out
    dig = int(math.floor(digits + 0.5))
    xv = x[ok]
    sgn = np.where(xv < 0, -1.0, 1.0)
    ax = np.abs(xv)
    logb = np.frexp(ax)[1].astype(np.float64) - 1.0
    l10x = 0.301029995663981195213738894724 * (0.5 + logb)
    keep = (l10x + dig) > 15
    pow10 = _r_pow_di(10.0, dig)
    with np.errstate(over="ignore", invalid="ignore"):
        x10 = ax * pow10
        i10 = np.floor(x10)
        xd = i10 / pow10
        xu = np.ceil(x10) / pow10
        du = xu - ax
        dd = ax - xd
        up = (du < dd) | ((np.fmod(i10, 2.0) == 1) & (du == dd))
    res = np.where(keep, ax, np.where(up, xu, xd)) * sgn
    out[ok] = res
    return out


_NUM_RE = re.compile(r"^\s*([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?\s*$")


def _ld_fac(expn: int) -> _LD:
    """``fac`` of ``R_strtod5``: 10^|expn| by repeated squaring in long double."""
    n = abs(expn)
    p10, fac = _LD(10.0), _LD(1.0)
    while n:
        if n & 1:
            fac = fac * p10
        n >>= 1
        p10 = p10 * p10
    return fac


def _r_strtod_scalar(strings: Sequence[str]) -> np.ndarray:
    """Reference (per string) version of :func:`r_strtod`."""
    out = np.empty(len(strings), dtype=np.float64)
    facs: Dict[int, _LD] = {}
    for idx, s in enumerate(strings):
        s = s.strip()
        if s in ("NA", ""):
            out[idx] = np.nan
            continue
        low = s.lstrip("+-").lower()
        if low in ("nan",):
            out[idx] = np.nan
            continue
        if low in ("inf", "infinity"):
            out[idx] = -np.inf if s.startswith("-") else np.inf
            continue
        m = _NUM_RE.match(s)
        if m is None:
            raise ValueError(f"cannot parse number {s!r}")
        sign = -1.0 if m.group(1) == "-" else 1.0
        ip, fp = m.group(2), m.group(3) or ""
        digits = ip + fp
        ndigits = len(digits)
        if ndigits == 0:
            out[idx] = np.nan
            continue
        expn = -len(fp) + (int(m.group(4)) if m.group(4) else 0)
        ival = int(digits)
        if ival < (1 << 64):
            ans = _LD(np.uint64(ival))
        else:  # accumulate like R, rounding at every step
            ans = _LD(0.0)
            for ch in digits:
                ans = _LD(10.0) * ans + _LD(ord(ch) - 48)
        if expn + ndigits < -300:
            for _ in range(ndigits):
                ans = ans / _LD(10.0)
            expn += ndigits
        if expn < -307:
            fac = _LD(1.0)
            n, p10 = -expn, _LD(10.0)
            while n:
                if n & 1:
                    fac = fac / p10
                n >>= 1
                p10 = p10 * p10
            ans = ans * fac
        elif expn < 0:
            fac = facs.get(expn)
            if fac is None:
                fac = facs[expn] = _ld_fac(expn)
            ans = ans / fac
        elif ans != 0.0:
            fac = facs.get(expn)
            if fac is None:
                fac = facs[expn] = _ld_fac(expn)
            ans = ans * fac
        out[idx] = sign * float(np.float64(ans))
    return out


_TBL = [_LD(float(f"1e{k}")) for k in range(-1, 28)]  # format.c: tbl[i] = 1e(i-1)


def _r_scientific(r: float, digits: int = 15):
    """R ``scientific()`` (format.c, long-double branch) for ``r > 0``:
    ``(kpower, nsig, roundingwidens)``."""
    kp = int(math.floor(math.log10(r))) - digits + 1
    r_prec = _LD(r)
    if abs(kp) <= 27:
        if kp > 0:
            r_prec = r_prec / _TBL[kp + 1]
        elif kp < 0:
            r_prec = r_prec * _TBL[-kp + 1]
    else:
        r_prec = r_prec / np.power(_LD(10.0), _LD(kp))
    if r_prec < _TBL[digits]:
        r_prec = r_prec * _LD(10.0)
        kp -= 1
    alpha = float(np.float64(np.rint(r_prec)))
    nsig = digits
    for _ in range(digits):
        alpha /= 10.0
        if alpha == math.floor(alpha):
            nsig -= 1
        else:
            break
    if nsig == 0 and digits > 0:
        nsig = 1
        kp += 1
    kpower = kp + digits - 1
    rgt = min(max(digits - kpower, 0), 27)
    fuzz = 0.5 / float(np.float64(_TBL[1 + rgt]))
    widens = 0 < kpower <= 27 and _LD(r) < _TBL[kpower + 1] - _LD(fuzz)
    return kpower, nsig, widens


def _r_format_scalar(x: float) -> str:
    """Reference (per number) version of :func:`_r_format_many`."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    x = float(x)
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == 0.0:
        return "0"
    neg = 1 if x < 0 else 0
    kpower, nsig, widens = _r_scientific(abs(x))
    left = kpower + 1 - (1 if widens else 0)
    sleft = neg + (1 if left <= 0 else left)
    rgt = max(nsig - left, 0)
    mxsl = 1 + neg if left < 0 else sleft
    wF = mxsl + rgt + (rgt != 0)
    e = 2 if (left > 100 or left <= -99) else 1
    d = nsig - 1
    wE = neg + (d > 0) + d + 4 + e
    if wF <= wE:
        return f"{x:.{rgt}f}"
    return f"{x:.{d}e}"


_TBL_ARR = np.array(_TBL, dtype=_LD)
_LOG10 = np.frompyfunc(math.log10, 1, 1)


def _r_format_many(x) -> List[str]:
    """Numbers as ``write.table`` prints them (vectorised ``formatReal`` with
    15 digits; see :func:`_r_format_scalar`)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.empty(x.size, dtype=object)
    fin = np.isfinite(x) & (x != 0.0)
    out[np.isnan(x)] = "NA"
    out[np.isposinf(x)] = "Inf"
    out[np.isneginf(x)] = "-Inf"
    out[x == 0.0] = "0"
    if not fin.any():
        return out.tolist()
    idx = np.where(fin)[0]
    v = x[idx]
    r = np.abs(v)
    digits = 15
    kp = np.floor(_LOG10(r).astype(np.float64)).astype(np.int64) - digits + 1
    big = np.abs(kp) > 27
    r_prec = r.astype(_LD)
    small = ~big
    pos = small & (kp > 0)
    negk = small & (kp < 0)
    r_prec[pos] = r_prec[pos] / _TBL_ARR[kp[pos] + 1]
    r_prec[negk] = r_prec[negk] * _TBL_ARR[-kp[negk] + 1]
    if big.any():
        r_prec[big] = r_prec[big] / np.power(_LD(10.0), kp[big].astype(_LD))
    lo = r_prec < _TBL_ARR[digits]
    r_prec[lo] = r_prec[lo] * _LD(10.0)
    kp = kp - lo.astype(np.int64)
    alpha = np.rint(r_prec).astype(np.float64)
    nsig = np.full(v.size, digits, dtype=np.int64)
    active = np.ones(v.size, dtype=bool)
    for _ in range(digits):
        alpha = np.where(active, alpha / 10.0, alpha)
        intg = active & (alpha == np.floor(alpha))
        nsig -= intg.astype(np.int64)
        active = intg
    zero = nsig == 0
    nsig[zero] = 1
    kp = kp + zero.astype(np.int64)
    kpower = kp + digits - 1
    rgt_ = np.clip(digits - kpower, 0, 27)
    fuzz = (0.5 / _TBL_ARR[1 + rgt_].astype(np.float64)).astype(_LD)
    kidx = np.clip(kpower + 1, 0, 28)
    widens = (kpower > 0) & (kpower <= 27) & (r.astype(_LD) < _TBL_ARR[kidx] - fuzz)
    neg = (v < 0).astype(np.int64)
    left = kpower + 1 - widens.astype(np.int64)
    sleft = neg + np.where(left <= 0, 1, left)
    rgt = np.maximum(nsig - left, 0)
    mxsl = np.where(left < 0, 1 + neg, sleft)
    wF = mxsl + rgt + (rgt != 0)
    e = np.where((left > 100) | (left <= -99), 2, 1)
    d = nsig - 1
    wE = neg + (d > 0) + d + 4 + e
    fixed = wF <= wE
    vals = v.tolist()
    rg = rgt.tolist()
    dd = d.tolist()
    fx = fixed.tolist()
    out[idx] = [f"{a:.{b}f}" if f else f"{a:.{c}e}" for a, b, c, f in zip(vals, rg, dd, fx)]
    return out.tolist()


def _r_format(x: float) -> str:
    """One number as ``write.table`` prints it."""
    return _r_format_many([x])[0]


def r_strtod(strings: Sequence[str]) -> np.ndarray:
    """R's number parser (``R_strtod5``, as used by ``read.table``).

    Digits are accumulated in long double and scaled by a long-double power
    of ten, then rounded to double; ``NA`` gives NaN. Plain decimal strings
    are parsed vectorised, anything unusual goes through the per-string
    reference implementation.
    """
    arr = np.asarray(strings, dtype=str)
    if arr.size == 0:
        return np.zeros(0)
    arr = np.char.strip(arr)
    low = np.char.lower(arr)
    neg = np.char.startswith(arr, "-")
    body = np.char.lstrip(low, "+-")
    mant, _, expo = np.char.partition(body, "e").T
    ip, dot, fp = np.char.partition(mant, ".").T
    digits = np.char.add(ip, fp)
    nd = np.char.str_len(digits)
    okchars = np.char.isdigit(digits) & (nd > 0) & (nd <= 19)
    has_e = np.char.str_len(expo) > 0
    exp_body = np.char.lstrip(expo, "+-")
    okexp = ~has_e | (np.char.isdigit(exp_body) & (np.char.str_len(exp_body) > 0))
    okfrac = (np.char.str_len(fp) == 0) | np.char.isdigit(fp)
    fast = okchars & okexp & okfrac
    out = np.empty(arr.size, dtype=np.float64)
    if (~fast).any():
        slow = np.where(~fast)[0]
        out[slow] = _r_strtod_scalar(arr[slow].tolist())
    if fast.any():
        f = np.where(fast)[0]
        ival = digits[f].astype(np.uint64)
        ex = np.zeros(f.size, dtype=np.int64)
        he = has_e[f]
        if he.any():
            ex[he] = expo[f][he].astype(np.int64)
        expn = ex - np.char.str_len(fp[f]).astype(np.int64)
        tiny = (expn + nd[f] < -300) | (expn < -307)
        ans = ival.astype(_LD)
        res = np.empty(f.size, dtype=np.float64)
        for ue in np.unique(expn[~tiny]):
            m = (~tiny) & (expn == ue)
            if ue < 0:
                res[m] = (ans[m] / _ld_fac(int(ue))).astype(np.float64)
            elif ue > 0:
                a = ans[m]
                res[m] = np.where(a != 0, a * _ld_fac(int(ue)), a).astype(np.float64)
            else:
                res[m] = ans[m].astype(np.float64)
        if tiny.any():
            res[tiny] = _r_strtod_scalar(arr[f][tiny].tolist())
        res = np.where(neg[f] & ~tiny, -res, res)
        out[f] = res
    return out


def _roundtrip15(x) -> np.ndarray:
    """``write.table`` then ``read.table`` of doubles."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    return r_strtod(_r_format_many(flat)).reshape(x.shape)


def _seq_sum_ld(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Left-to-right long-double sum (R's ``LDOUBLE`` loops)."""
    x = np.asarray(x, dtype=_LD)
    if x.shape[axis] == 0:
        return np.zeros(np.delete(x.shape, axis), dtype=_LD)
    return np.add.accumulate(x, axis=axis).take(-1, axis=axis)


def r_mean(x) -> float:
    """R's ``mean`` of a double vector (long-double sum plus a correction pass)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    s = _seq_sum_ld(x) / _LD(n)
    if np.isfinite(np.float64(s)):
        t = _seq_sum_ld(x.astype(_LD) - s)
        s = s + t / _LD(n)
    return float(np.float64(s))


def _r_row_means_apply(x: np.ndarray) -> np.ndarray:
    """``apply(x, 1, mean)``."""
    return np.array([r_mean(r) for r in np.asarray(x, dtype=np.float64)])


def r_var(x) -> float:
    """R's ``var`` of a double vector (``cov.c``, two-pass, long double)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return float("nan")
    s = _seq_sum_ld(x)
    tmp = s / _LD(n)
    if np.isfinite(np.float64(tmp)):
        tmp = tmp + _seq_sum_ld(x.astype(_LD) - tmp) / _LD(n)
    xm = _LD(np.float64(tmp))
    dev = x.astype(_LD) - xm
    return float(np.float64(_seq_sum_ld(dev * dev) / _LD(n - 1)))


def _r_cov(x: np.ndarray) -> np.ndarray:
    """``cov(x)`` of a matrix (columns are variables)."""
    x = np.asarray(x, dtype=np.float64)
    n, p = x.shape
    xm = np.empty(p, dtype=np.float64)
    for j in range(p):
        s = _seq_sum_ld(x[:, j]) / _LD(n)
        if np.isfinite(np.float64(s)):
            s = s + _seq_sum_ld(x[:, j].astype(_LD) - s) / _LD(n)
        xm[j] = np.float64(s)
    dev = x.astype(_LD) - xm.astype(_LD)[None, :]
    out = np.empty((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(i + 1):
            out[i, j] = out[j, i] = np.float64(_seq_sum_ld(dev[:, i] * dev[:, j]) / _LD(n - 1))
    return out


_PN_A = (2.2352520354606839287, 161.02823106855587881, 1067.6894854603709582,
         18154.981253343561249, 0.065682337918207449113)
_PN_B = (47.20258190468824187, 976.09855173777669322, 10260.932208618978205,
         45507.789335026729956)
_PN_C = (0.39894151208813466764, 8.8831497943883759412, 93.506656132177855979,
         597.27027639480026226, 2494.5375852903726711, 6848.1904505362823326,
         11602.651437647350124, 9842.7148383839780218, 1.0765576773720192317e-8)
_PN_D = (22.266688044328115691, 235.38790178262499861, 1519.377599407554805,
         6485.558298266760755, 18615.571640885098091, 34900.952721145977266,
         38912.003286093271411, 19685.429676859990727)
_PN_P = (0.21589853405795699, 0.1274011611602473639, 0.022235277870649807,
         0.001421619193227893466, 2.9112874951168792e-5, 0.02307344176494017303)
_PN_Q = (1.28426009614491121, 0.468238212480865118, 0.0659881378689285515,
         0.00378239633202758244, 7.29751555083966205e-5)
_M_SQRT_32 = 5.656854249492380195206754896838
_M_1_SQRT_2PI = 0.398942280401432677939946059934


def _pnorm_lower1(x: float) -> float:
    """R's ``pnorm(x)`` (Cody, ``pnorm_both`` lower tail, not log)."""
    if math.isnan(x):
        return x
    eps = 2.220446049250313e-16 * 0.5
    y = abs(x)
    if y <= 0.67448975:
        if y > eps:
            xsq = x * x
            xnum = _PN_A[4] * xsq
            xden = xsq
            for i in range(3):
                xnum = (xnum + _PN_A[i]) * xsq
                xden = (xden + _PN_B[i]) * xsq
        else:
            xnum = xden = 0.0
        temp = x * (xnum + _PN_A[3]) / (xden + _PN_B[3])
        return 0.5 + temp

    def _del(X, temp):
        xsq = math.ldexp(math.trunc(math.ldexp(X, 4)), -4)
        de = (X - xsq) * (X + xsq)
        cum = math.exp(-xsq * math.ldexp(xsq, -1)) * math.exp(-math.ldexp(de, -1)) * temp
        return cum, 1.0 - cum

    if y <= _M_SQRT_32:
        xnum = _PN_C[8] * y
        xden = y
        for i in range(7):
            xnum = (xnum + _PN_C[i]) * y
            xden = (xden + _PN_D[i]) * y
        temp = (xnum + _PN_C[7]) / (xden + _PN_D[7])
        cum, ccum = _del(y, temp)
        return ccum if x > 0 else cum
    if -37.5193 < x < 8.2924:
        xsq = 1.0 / (x * x)
        xnum = _PN_P[5] * xsq
        xden = xsq
        for i in range(4):
            xnum = (xnum + _PN_P[i]) * xsq
            xden = (xden + _PN_Q[i]) * xsq
        temp = xsq * (xnum + _PN_P[4]) / (xden + _PN_Q[4])
        temp = (_M_1_SQRT_2PI - temp) / y
        cum, ccum = _del(x, temp)
        return ccum if x > 0 else cum
    return 1.0 if x > 0 else 0.0


def _pnorm_lower(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.array([_pnorm_lower1(v) for v in x.ravel().tolist()]).reshape(x.shape)


def _r_rank_average(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(x, method="average").astype(np.float64)


def normalize_quantiles(A) -> np.ndarray:
    """``limma::normalizeQuantiles(A, ties=TRUE)`` (limma 3.50).

    Every column is mapped onto the mean of the sorted columns; tied values
    get the interpolated mean of their ranks. A single column is returned
    unchanged. Missing values are not supported.

    Arguments:
        A: ``bins x samples`` array.

    Returns:
        Normalised ``float64`` array of the same shape.
    """
    A = np.array(A, dtype=np.float64, copy=True)
    if A.ndim != 2 or A.shape[1] == 1:
        return A
    if np.isnan(A).any():
        raise NotImplementedError("normalize_quantiles: missing values are not supported")
    n, p = A.shape
    S = np.sort(A, axis=0)
    m = np.asarray(_seq_sum_ld(S, axis=1) / _LD(p), dtype=np.float64)
    grid = np.arange(n, dtype=np.float64) / float(n - 1)
    for j in range(p):
        r = _r_rank_average(A[:, j])
        v = (r - 1.0) / float(n - 1)
        i = np.clip(np.searchsorted(grid, v, side="right") - 1, 0, n - 2)
        jj = i + 1
        y = m[i] + (m[jj] - m[i]) * ((v - grid[i]) / (grid[jj] - grid[i]))
        y = np.where(v == grid[i], m[i], y)
        y = np.where(v == grid[jj], m[jj], y)
        A[:, j] = y
    return A


def _scale_by_mean_abs(qn: np.ndarray) -> np.ndarray:
    """``qn / apply(abs(qn), 2, mean)`` with R's column-major recycling."""
    p = qn.shape[1]
    cm = np.array([r_mean(np.abs(qn[:, j])) for j in range(p)])
    flat = qn.ravel(order="F")
    div = np.resize(cm, flat.size)
    return (flat / div).reshape(qn.shape, order="F")


# --------------------------------------------------------------------------
# robust::covRob
# --------------------------------------------------------------------------

def _dsum(x: np.ndarray) -> float:
    """rrcov's pairwise ``dsum``."""
    x = np.asarray(x, dtype=np.float64)
    while x.size > 1:
        half = x.size // 2
        w = x[0:2 * half:2] + x[1:2 * half:2]
        if 2 * half < x.size:
            w[half - 1] = w[half - 1] + x[2 * half]
        x = w
    return float(x[0])


def _s_mad(x: np.ndarray):
    mu = float(np.median(x))
    return float(np.median(np.abs(x - mu))) * 1.4826, mu


def _scale_tau2(x: np.ndarray):
    C1, C2sq, Es2c = 4.5, 9.0, 0.9247153921761315
    medx = float(np.median(x))
    sigma0 = float(np.median(np.abs(x - medx)))
    w = np.abs(x - medx) / (C1 * sigma0)
    w2 = 1.0 - w * w
    w2 = ((np.abs(w2) + w2) / 2.0) ** 2
    tmpsum = _dsum(w2)
    mu = _dsum(x * w2) / tmpsum
    d = ((x - mu) / sigma0) ** 2
    d = np.minimum(d, C2sq)
    return sigma0 * math.sqrt(_dsum(d) / (x.size * Es2c)), mu


def _qc(x: np.ndarray, y: np.ndarray, scalefn) -> float:
    medx, medy = float(np.median(x)), float(np.median(y))
    onethree = int(np.sum(((x > medx) & (y > medy)) | ((x < medx) & (y < medy))))
    twofour = int(np.sum(((x > medx) & (y < medy)) | ((x < medx) & (y > medy))))
    r = (onethree - twofour) / (onethree + twofour)
    return math.sin(1.57079632679489661923 * r)


def _gk(x: np.ndarray, y: np.ndarray, scalefn) -> float:
    plus = scalefn(x + y)[0]
    minus = scalefn(x - y)[0]
    return (plus * plus - minus * minus) / 4.0


_LAPACK_FN: Dict[str, object] = {}
_SYS_LAPACK: List[object] = []


def _system_lapack():
    """The LAPACK of the Python environment (e.g. conda's OpenBLAS), if any.

    Preferred over the copy bundled in scipy wheels for :func:`_opw_eigvecs`:
    in degenerate cases the result depends on the build's rounding, and R
    (conda) uses the environment's OpenBLAS. ``EPIONE_LAPACK`` overrides the
    path.
    """
    import ctypes
    import sys
    if _SYS_LAPACK:
        return _SYS_LAPACK[0]
    lib = None
    cands = [os.environ.get("EPIONE_LAPACK", "")]
    cands += [os.path.join(sys.prefix, "lib", n) for n in ("liblapack.so.3", "liblapack.so", "liblapack.dylib")]
    for c in cands:
        if c and os.path.exists(c):
            try:
                cand = ctypes.CDLL(c)
                if all(hasattr(cand, f"{n}_") for n in ("dsptrd", "dstegr", "dopmtr")):
                    lib = cand
                    break
            except OSError:
                continue
    _SYS_LAPACK.append(lib)
    return lib


def _lapack(name: str, argtypes):
    """A LAPACK routine as a ctypes function: the environment's LAPACK if
    available, else ``scipy.linalg.cython_lapack``."""
    import ctypes
    fn = _LAPACK_FN.get(name)
    if fn is None:
        lib = _system_lapack()
        if lib is not None:
            fn = getattr(lib, f"{name}_")
            fn.restype = None
            fn.argtypes = argtypes
            _LAPACK_FN[name] = fn
            return fn
        import scipy.linalg.cython_lapack as cl
        cap = cl.__pyx_capi__[name]
        get_name = ctypes.pythonapi.PyCapsule_GetName
        get_name.restype = ctypes.c_char_p
        get_name.argtypes = [ctypes.py_object]
        get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
        get_ptr.restype = ctypes.c_void_p
        get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
        ptr = get_ptr(cap, get_name(cap))
        fn = ctypes.CFUNCTYPE(None, *argtypes)(ptr)
        _LAPACK_FN[name] = fn
    return fn


def _opw_eigvecs(U: np.ndarray) -> np.ndarray:
    """Eigenvectors of the symmetric ``U``, largest eigenvalue first, computed
    with the LAPACK sequence of ``covOPW.c`` (``dsptrd`` on the packed lower
    triangle, ``dstegr``, ``dopmtr``).

    When eigenvalues repeat (e.g. every quadrant correlation is 1, so ``U`` is
    all ones), the basis of the repeated eigenspace is arbitrary and changes
    the OGK result; following rrcov's calls reproduces rrcov's basis.
    """
    import ctypes
    p = U.shape[0]
    c_char = ctypes.c_char_p
    pi = ctypes.POINTER(ctypes.c_int)
    pd_ = ctypes.POINTER(ctypes.c_double)
    dsptrd = _lapack("dsptrd", [c_char, pi, pd_, pd_, pd_, pd_, pi])
    dstegr = _lapack("dstegr", [c_char, c_char, pi, pd_, pd_, pd_, pd_, pi, pi, pd_, pi, pd_, pd_,
                                pi, pi, pd_, pi, pi, pi, pi])
    dopmtr = _lapack("dopmtr", [c_char, c_char, c_char, pi, pi, pd_, pd_, pd_, pi, pd_, pi])

    def dp(a):
        return a.ctypes.data_as(pd_)

    def ip(a):
        return a.ctypes.data_as(pi)

    ap = np.empty(p * (p + 1) // 2, dtype=np.float64)
    for j in range(p):
        for i in range(j, p):
            ap[i + ((2 * p - j - 1) * j) // 2] = U[i, j]
    diag = np.zeros(p)
    off = np.zeros(p)
    tau = np.zeros(max(p - 1, 1))
    n = np.array([p], dtype=np.int32)
    info = np.zeros(1, dtype=np.int32)
    dsptrd(b"L", ip(n), dp(ap), dp(diag), dp(off), dp(tau), ip(info))
    if info[0] != 0:
        raise RuntimeError(f"dsptrd failed (info={info[0]})")
    lwork, liwork = 18 * p, 10 * p
    w = np.zeros(p)
    Z = np.zeros((p, p), dtype=np.float64, order="F")
    isuppz = np.zeros(2 * p, dtype=np.int32)
    work = np.zeros(max(p, lwork))
    iwork = np.zeros(liwork, dtype=np.int32)
    vl = np.zeros(1)
    vu = np.zeros(1)
    il = np.zeros(1, dtype=np.int32)
    iu = np.zeros(1, dtype=np.int32)
    abstol = np.zeros(1)
    m = np.zeros(1, dtype=np.int32)
    lw = np.array([lwork], dtype=np.int32)
    liw = np.array([liwork], dtype=np.int32)
    dstegr(b"V", b"A", ip(n), dp(diag), dp(off), dp(vl), dp(vu), ip(il), ip(iu), dp(abstol),
           ip(m), dp(w), dp(Z), ip(n), ip(isuppz), dp(work), ip(lw), ip(iwork), ip(liw), ip(info))
    if info[0] != 0:
        raise RuntimeError(f"dstegr failed (info={info[0]})")
    work2 = np.zeros(p)
    dopmtr(b"L", b"L", b"N", ip(n), ip(n), dp(ap), dp(tau), dp(Z), ip(n), dp(work2), ip(info))
    if info[0] != 0:
        raise RuntimeError(f"dopmtr failed (info={info[0]})")
    ws = np.sort(w)
    scale = max(float(np.abs(ws).max()), 1.0)
    if np.any(np.diff(ws) <= 1e-10 * scale):
        import warnings
        warnings.warn(
            "OGK (covRob 'pairwiseqc'): the pairwise correlation matrix has repeated "
            "eigenvalues, so the robust covariance is not unique; the basis LAPACK picks "
            "depends on its build (R's result does too).",
            RuntimeWarning, stacklevel=3,
        )
    return np.ascontiguousarray(Z[:, ::-1])


def _cov_opw(X: np.ndarray, niter: int, smrob: str, svrob: str):
    """rrcov ``covOPW.c``: orthogonalised pairwise covariance."""
    scalefn = {"s_mad": _s_mad, "scaleTau2": _scale_tau2}[smrob]
    rcovfn = {"qc": _qc, "gk": _gk}[svrob]
    Z = np.array(X, dtype=np.float64, copy=True)
    n, p = Z.shape
    As = []
    for _ in range(niter):
        d = np.array([scalefn(Z[:, j])[0] for j in range(p)])
        if np.any(np.abs(d) < 1e-12):
            raise ValueError("column with zero scale encountered (covOPW)")
        Z = Z * (1.0 / d)[None, :]
        U = np.eye(p)
        for i in range(1, p):
            for j in range(i):
                U[i, j] = U[j, i] = rcovfn(Z[:, i], Z[:, j], scalefn)
        E = _opw_eigvecs(U)
        Z = Z @ E
        As.append(d[:, None] * E)
    gam = np.empty(p)
    center = np.empty(p)
    for j in range(p):
        gam[j], center[j] = scalefn(Z[:, j])
    cov = np.diag(gam * gam)
    W = (Z - center[None, :]) / gam[None, :]
    dist = _seq_sum_ld(np.abs(W * W), axis=1).astype(np.float64)
    for A in reversed(As):
        cov = A @ cov
        cov = cov @ A.T
        center = A @ center
    return center, cov, dist


def _cov_ogk(x: np.ndarray, smrob: str, svrob: str, niter: int = 2, beta: float = 0.9) -> dict:
    """``rrcov::CovOgk`` with the C ``covOPW`` path."""
    from scipy.stats import chi2
    n, p = x.shape
    center, cov, dist2 = _cov_opw(x, niter, smrob, svrob)
    med = float(np.median(dist2))
    q05 = chi2.ppf(0.5, p)
    cdelta = med / q05
    raw_cov = cdelta * cov
    qq = (chi2.ppf(beta, p) * med) / q05
    wt = (dist2 < qq).astype(np.float64)
    swt = float(wt.sum())
    wcenter = np.asarray(_seq_sum_ld(x * wt[:, None], axis=0), dtype=np.float64) / swt
    Xc = np.sqrt(wt)[:, None] * (x - wcenter[None, :])
    wcov = (Xc.T @ Xc) / swt
    return {"cov": wcov, "center": wcenter, "raw_cov": raw_cov, "raw_center": center,
            "raw_dist": dist2, "raw_wt": wt}


def cov_rob(x, estim: str = "auto", rng=None) -> dict:
    """``robust::covRob(x)`` (robust 0.7.1).

    ``estim="auto"`` picks the estimator as robust does: ``donostah`` if
    ``(n < 1000 and p < 10) or (n < 5000 and p < 5)``, ``mcd`` if
    ``n < 50000 and p < 20``, otherwise ``pairwiseqc``.

    Arguments:
        x: ``n x p`` data matrix; rows with missing values are dropped.
        estim: ``"auto"``, ``"pairwiseqc"`` (OGK with MAD scale and quadrant
            correlation), ``"pairwisegk"`` (OGK with tau scale and
            Gnanadesikan-Kettenring), ``"donostah"`` (Stahel-Donoho,
            ``rrcov::CovSde`` under ``set.seed(21)``) or ``"mcd"``
            (``robustbase::covMcd``, raw estimate).
        rng: R's random number generator
            (:class:`epione.bulk.hic._r_rng.RRandom`) for the ``donostah`` and
            ``mcd`` branches, advanced as R's global stream would be. Default:
            a generator in the state of ``set.seed(123)`` (dcHiC's default
            seed).

    Returns:
        ``dict`` with ``cov`` and ``center`` (what ``covRob(...)$cov`` /
        ``$center`` return) and the chosen ``estim``; for the OGK branches also
        the raw estimate (``raw_cov``, ``raw_center``, ``raw_dist``).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    n, p = x.shape
    if p < 2:
        raise ValueError("data must have at least two columns to compute a covariance matrix")
    if n < p:
        raise ValueError("not enough observations")
    estim = estim.lower()
    if estim == "auto":
        if (n < 1000 and p < 10) or (n < 5000 and p < 5):
            estim = "donostah"
        elif n < 50000 and p < 20:
            estim = "mcd"
        else:
            estim = "pairwiseqc"
    if estim == "pairwiseqc":
        res = _cov_ogk(x, "s_mad", "qc")
    elif estim == "pairwisegk":
        res = _cov_ogk(x, "scaleTau2", "gk")
    elif estim in ("donostah", "mcd"):
        from ._r_rng import RRandom
        if rng is None:
            rng = RRandom(123)
        if estim == "donostah":
            from ._covsde import cov_sde
            center, cov = cov_sde(x, rng=rng)
        else:
            from ._covmcd import cov_mcd
            center, cov = cov_mcd(x, rng=rng)
        res = {"cov": np.asarray(cov, dtype=np.float64), "center": np.asarray(center, dtype=np.float64)}
    else:
        raise ValueError(f"unknown covRob estimator {estim!r}")
    res["estim"] = estim
    return res


# --------------------------------------------------------------------------
# dcHiC statistics
# --------------------------------------------------------------------------

def _calcen(df: np.ndarray, cls: str, rzscore: float, szscore: float) -> np.ndarray:
    """dcHiC ``calcen``: centres shrunk towards 0 for spread-out rows."""
    df = np.asarray(df, dtype=np.float64)
    n, p = df.shape
    D = np.abs(df[:, :, None] - df[:, None, :])  # [row, a, b]
    cs = np.asarray(_seq_sum_ld(D, axis=1), dtype=np.float64)  # colSums over a
    df_dist = np.sqrt(cs / float(p - 1))
    if cls == "rep":
        value = df_dist.ravel(order="F")
        mu, sd = r_mean(value), math.sqrt(r_var(value))
        z = (df_dist - mu) / sd
        pvl = r_round(_pnorm_lower(z - rzscore), 5)
    elif cls == "sample":
        z = np.empty_like(df_dist)
        for j in range(p):
            v = df_dist[:, j]
            mu, sd = r_mean(v), math.sqrt(r_var(v))
            z[:, j] = (v - mu) / sd
        pvl = r_round(_pnorm_lower(z - szscore), 5)
    else:
        raise ValueError(cls)
    pmax = pvl.max(axis=1)
    return r_round(df * (1.0 - pmax)[:, None], 5)


def _maha_scaled(diff: np.ndarray, sinv: np.ndarray, mmd: np.ndarray) -> np.ndarray:
    """``mahalanobis(x, c, solve(S) %*% diag(mmd_k), inverted=TRUE)`` per row."""
    M = sinv[None, :, :] * mmd[:, None, None]
    y = np.einsum("ki,kij->kj", diff, M)
    return np.asarray(_seq_sum_ld(y * diff, axis=1), dtype=np.float64)


def _bh(p: np.ndarray) -> np.ndarray:
    """``p.adjust(p, "BH")``."""
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    if n <= 1:
        return p.copy()
    o = np.argsort(-p, kind="stable")
    i = np.arange(n, 0, -1, dtype=np.float64)
    q = np.minimum.accumulate((float(n) / i) * p[o])
    out = np.empty(n)
    out[o] = np.minimum(1.0, q)
    return out


def _make_names_unique(names: Sequence[str]) -> List[str]:
    """``make.names(names, unique=TRUE)`` as ``data.frame(check.names=TRUE)`` does."""
    out = []
    for nm in names:
        s = re.sub(r"[^A-Za-z0-9._]", ".", str(nm))
        if not s or not (s[0].isalpha() or (s[0] == "." and not (len(s) > 1 and s[1].isdigit()))):
            s = "X" + s
        out.append(s)
    seen: Dict[str, int] = {}
    used = set(out)
    res = []
    for s in out:
        if s in seen:
            k = seen[s]
            while f"{s}.{k}" in used:
                k += 1
            seen[s] = k + 1
            new = f"{s}.{k}"
            used.add(new)
            res.append(new)
        else:
            seen[s] = 1
            res.append(s)
    return res


def _collate_key(name: str):
    """Sort key approximating R's ICU collation of file names (``list.files``).

    Punctuation sorts before digits, digits before letters, letters
    case-insensitively (lower case first on ties): ``chr1_...`` < ``chr10_...``.
    """
    prim = []
    for ch in name:
        if ch == "_":
            prim.append((1, 0))
        elif ch == "-":
            prim.append((2, 0))
        elif ch == ".":
            prim.append((3, 0))
        elif ch.isdigit():
            prim.append((5, ord(ch)))
        elif ch.isalpha():
            prim.append((6, ord(ch.lower())))
        else:
            prim.append((4, ord(ch)))
    tert = [0 if c.islower() or not c.isalpha() else 1 for c in name]
    return (prim, tert)


class _Sample:
    """One replicate: prefix, condition and its per-chromosome PC tables."""

    def __init__(self, prefix: str, cond: str, tables: Dict[str, pd.DataFrame],
                 texts: Optional[Dict[str, str]] = None):
        self.prefix = prefix
        self.cond = cond
        self.tables = tables      # chrom -> DataFrame(chr, start, end, PC) as R read it
        self.texts = texts or {}  # chrom -> source file path (for the _data copies)


def _write_table(df: pd.DataFrame, path: Path, header: bool = True) -> None:
    """``write.table(df, row.names=F, sep="\\t", quote=F)``."""
    cols = list(df.columns)
    with open(path, "w") as fh:
        if header:
            fh.write("\t".join(map(str, cols)) + "\n")
        conv = []
        for c in cols:
            s = df[c]
            if pd.api.types.is_integer_dtype(s.dtype):
                conv.append([str(int(v)) for v in s.tolist()])
            elif pd.api.types.is_float_dtype(s.dtype):
                conv.append(_r_format_many(s.to_numpy(np.float64)))
            else:
                conv.append([("NA" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))
                             for v in s.tolist()])
        for row in zip(*conv):
            fh.write("\t".join(row) + "\n")


def _key(df: pd.DataFrame) -> pd.Index:
    return pd.Index(df["chr"].astype(str) + "_" + df["start"].astype(str) + "_" + df["end"].astype(str))


def _pcanalyze(samples: List[_Sample], conditions: List[str], *, out: Optional[Path],
               rzscore: float, szscore: float, fdr: float, distclust: float, numberclust: float,
               refine: bool, rconf: float, padj_method: str, seed: int = 123) -> pd.DataFrame:
    from scipy.stats import chi2
    from ._r_rng import RRandom

    # dchicf.r calls set.seed(opt$seed) at start-up; the covRob calls below are the only
    # consumers of R's global random stream, in chromosome order (donostah reseeds with
    # 21 without restoring, mcd continues the stream).
    rng = RRandom(seed)

    data_prefix = [s.prefix for s in samples]               # data$prefix (input order)
    by_cond = {c: [s for s in samples if s.cond == c] for c in conditions}
    if any(not v for v in by_cond.values()):
        raise ValueError("every condition needs at least one sample")
    if any(len(v) > 1 for v in by_cond.values()) and padj_method == "auto":
        raise NotImplementedError(
            "dcHiC adjusts p-values with IHW when a condition has replicates; IHW is not "
            f"ported (see {_MANIFEST}). Pass padj_method='BH' to use Benjamini-Hochberg "
            "instead (a deviation from dcHiC)."
        )
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    # ---- per-condition combined tables ---------------------------------------
    sample_count = 0
    bed_rows: List[pd.DataFrame] = []
    cond_pcori: Dict[str, Dict[str, pd.DataFrame]] = {c: {} for c in conditions}
    chrom_list: List[str] = []
    for c in conditions:
        reps = by_cond[c]
        sample_count += len(reps)
        ddir = None
        if out is not None:
            ddir = out / f"{c}_data"
            ddir.mkdir(exist_ok=True)
            for s in reps:
                for chrom, src in s.texts.items():
                    shutil.copyfile(src, ddir / f"{s.prefix}_intra_{chrom}.pc.bedGraph")
                if not s.texts:
                    for chrom, t in s.tables.items():
                        _write_table(t, ddir / f"{s.prefix}_intra_{chrom}.pc.bedGraph", header=False)
        # chrom list = files of the last replicate of this condition (R's `chrom`)
        chrom_list = sorted(reps[-1].tables, key=lambda ch: _collate_key(f"{ch}.pc.bedGraph"))
        for chrom in chrom_list:
            have = [s for s in reps if chrom in s.tables]
            if not have:
                continue
            # replicate files in list.files order of "<prefix>_intra_<chrom>.pc.bedGraph"
            have = sorted(have, key=lambda s: _collate_key(f"{s.prefix}_intra_{chrom}.pc.bedGraph"))
            tabs = [s.tables[chrom] for s in have]
            for t in tabs:
                bed_rows.append(t[["chr", "start", "end"]])
            allk = pd.concat([_key(t).to_series() for t in tabs], ignore_index=True)
            cnt = allk.value_counts(sort=False)
            order = pd.unique(allk)
            keep = [k for k in order if cnt[k] == len(tabs)]
            first = pd.concat(tabs, ignore_index=True)
            first.index = _key(first)
            first = first[~first.index.duplicated()]
            bed = first.loc[keep, ["chr", "start", "end"]].reset_index(drop=True)
            mat = np.column_stack([t.set_index(_key(t)).loc[keep, "PC"].to_numpy(np.float64) for t in tabs])
            qn = _scale_by_mean_abs(normalize_quantiles(mat))
            names = [s.prefix for s in have]
            ori = bed.copy()
            qnd = bed.copy()
            for i, nm in enumerate(names):
                ori[nm] = mat[:, i]
                qnd[nm] = qn[:, i]
            if ddir is not None:
                _write_table(ori, ddir / f"intra_{chrom}_combined.pcOri.bedGraph")
                _write_table(qnd, ddir / f"intra_{chrom}_combined.pcQnm.bedGraph")
            # what the next step reads back
            back = ori.copy()
            for nm in names:
                back[nm] = _roundtrip15(back[nm].to_numpy())
            cond_pcori[c][chrom] = back

    allbed = pd.concat(bed_rows, ignore_index=True)
    allk = _key(allbed).to_series(index=None).reset_index(drop=True)
    counts = allk.value_counts(sort=False)
    allk_unique = pd.unique(allk)
    chr_of = dict(zip(allk, allbed["chr"].astype(str)))

    if out is not None:
        (out / "pcOri").mkdir(exist_ok=True)
        (out / "pcQnm").mkdir(exist_ok=True)

    # ---- across all samples ---------------------------------------------------
    combined_qnm: Dict[str, pd.DataFrame] = {}
    combined_ori: Dict[str, pd.DataFrame] = {}
    for chrom in chrom_list:
        vec = [k for k in allk_unique if chr_of[k] == chrom and counts[k] == sample_count]
        parts = []
        names: List[str] = []
        for i, c in enumerate(conditions):
            df = cond_pcori[c].get(chrom)
            if df is None:
                continue
            df = df.set_index(_key(df))
            sub = df.loc[vec]
            if not parts:
                parts.append(sub.reset_index(drop=True))
                names += list(df.columns)
            else:
                parts.append(sub.iloc[:, 3:].reset_index(drop=True))
                names += list(df.columns[3:])
        if not parts:
            continue
        cs = pd.concat(parts, axis=1)
        cs.columns = names
        vals = cs.iloc[:, 3:].to_numpy(np.float64)
        qn = r_round(_scale_by_mean_abs(normalize_quantiles(vals)), 5)
        ori = r_round(vals, 5)
        t_ori = cs.copy()
        t_qnm = cs.copy()
        t_ori.iloc[:, 3:] = ori
        t_qnm.iloc[:, 3:] = qn
        if out is not None:
            _write_table(t_ori, out / "pcOri" / f"intra_sample_{chrom}_combined.pcOri.bedGraph")
            _write_table(t_qnm, out / "pcQnm" / f"intra_sample_{chrom}_combined.pcQnm.bedGraph")
        for nm in names[3:]:
            t_ori[nm] = _roundtrip15(t_ori[nm].to_numpy())
            t_qnm[nm] = _roundtrip15(t_qnm[nm].to_numpy())
        combined_ori[chrom] = t_ori
        combined_qnm[chrom] = t_qnm

    # ---- significance ---------------------------------------------------------
    files = sorted(combined_qnm, key=lambda ch: _collate_key(f"intra_sample_{ch}_combined.pcQnm.bedGraph"))
    if out is not None:
        (out / "fdr_result").mkdir(exist_ok=True)
    res_list: List[pd.DataFrame] = []
    grp_ori_list: List[pd.DataFrame] = []
    rep_count = 0
    for chrom in files:
        df_intra = combined_qnm[chrom]
        df_ori = combined_ori[chrom]
        n = len(df_intra)
        sample_cols = list(df_intra.columns[3:])
        cen_cols, grp_cols, grp_ori_cols = [], [], []
        rep_count = 0
        for c in conditions:
            pref = [p for p in data_prefix if p in {s.prefix for s in by_cond[c]}]
            sub = df_intra[pref].to_numpy(np.float64)
            if len(pref) > 1:
                cen_cols.append(_calcen(sub, "rep", rzscore, szscore))
                grp_cols.append(_r_row_means_apply(sub))
                grp_ori_cols.append(_r_row_means_apply(df_ori[pref].to_numpy(np.float64)))
                rep_count += 1
            else:
                cen_cols.append(np.full((n, 1), r_mean(sub[:, 0])))
                grp_cols.append(sub[:, 0])
                grp_ori_cols.append(df_ori[pref[0]].to_numpy(np.float64))
        intra_cen = np.hstack([np.asarray(a).reshape(n, -1) for a in cen_cols])
        intra_grp = np.column_stack(grp_cols)
        grp_ori = pd.DataFrame(np.column_stack(grp_ori_cols), columns=conditions,
                               index=df_ori["chr"].astype(str) + "_" + df_ori["start"].astype(str))
        grp_ori_list.append(grp_ori)

        allvals = df_intra[sample_cols].to_numpy(np.float64)
        replicate_wt = None
        if rep_count > 0:
            dv = np.diag(_r_cov(allvals))
            sinv = np.diag(1.0 / dv)
            diff = allvals - intra_cen
            y = diff @ sinv
            rmaha = np.asarray(_seq_sum_ld(y * diff, axis=1), dtype=np.float64)
            replicate_wt = -chi2.logcdf(rmaha, df=allvals.shape[1] - 1)

        grp_cen = _calcen(intra_grp, "sample", rzscore, szscore)
        mmd = np.abs(intra_grp.max(axis=1) - intra_grp.min(axis=1))
        mmd = mmd * mmd
        dfree = intra_grp.shape[1] - 1
        diff = intra_grp - grp_cen
        S = cov_rob(intra_grp, rng=rng)["cov"]
        smaha = _maha_scaled(diff, np.linalg.inv(S), mmd)
        pval = chi2.sf(smaha, dfree)
        if refine:
            thr = chi2.sf(chi2.ppf(rconf, dfree), dfree)
            S = cov_rob(intra_grp[pval > thr], rng=rng)["cov"]
            smaha = _maha_scaled(diff, np.linalg.inv(S), mmd)
            pval = chi2.sf(smaha, dfree)

        res = df_intra.copy()
        grp_names = list(conditions)
        for i, c in enumerate(grp_names):
            res[f"\x00{c}"] = intra_grp[:, i]
        if replicate_wt is not None:
            res["replicate_wt"] = replicate_wt
        res["sample_maha"] = smaha
        res["pval"] = pval
        res.columns = _make_names_unique([c.lstrip("\x00") for c in res.columns])
        if out is not None:
            _write_table(res, out / "fdr_result" / f"differential.intra_sample_{chrom}_combined.pcQnm.bedGraph")
        res_list.append(res)

    if not res_list:
        raise ValueError("no chromosome has bins shared by all samples")
    intra_res = pd.concat(res_list, ignore_index=True)
    intra_res = intra_res.dropna().reset_index(drop=True)
    if rep_count > 0:
        intra_res["padj"] = _bh(intra_res["pval"].to_numpy())
        intra_res = intra_res.sort_values(["chr", "start"], kind="stable",
                                          key=lambda s: s.map(_collate_key) if s.name == "chr" else s
                                          ).reset_index(drop=True)
    else:
        intra_res["padj"] = _bh(intra_res["pval"].to_numpy())

    if distclust > -1:
        intra_res["dist_clust"] = _calchclust(intra_res, fdr, distclust)
    else:
        intra_res["dist_clust"] = 1.0
    rown = intra_res["chr"].astype(str) + "_" + intra_res["start"].astype(str)

    if out is not None:
        fr = out / "fdr_result"
        sel = (intra_res["padj"] < fdr) & (intra_res["dist_clust"] >= numberclust)
        _write_table(intra_res, fr / "differential.intra_sample_combined.pcQnm.bedGraph")
        if fdr > 0:
            _write_table(intra_res[sel], fr / "differential.intra_sample_combined.Filtered.pcQnm.bedGraph")
        grp = intra_res.drop(columns=[c for c in intra_res.columns if c in set(data_prefix)])
        _write_table(grp, fr / "differential.intra_sample_group.pcQnm.bedGraph")
        if fdr > 0:
            _write_table(grp[sel], fr / "differential.intra_sample_group.Filtered.pcQnm.bedGraph")
        gori = pd.concat(grp_ori_list)
        grp = grp.copy()
        conds = list(dict.fromkeys(conditions))
        grp[conds] = gori.loc[rown.to_numpy(), conds].to_numpy()
        _write_table(grp, fr / "differential.intra_sample_group.pcOri.bedGraph")
        if fdr > 0:
            _write_table(grp[sel], fr / "differential.intra_sample_group.Filtered.pcOri.bedGraph")
    return intra_res


def _calchclust(df: pd.DataFrame, fdr_thr: float, hclust_dist: float) -> np.ndarray:
    """dcHiC ``calchclust``: size of each significant bin's distance cluster."""
    from scipy.cluster.hierarchy import fcluster, linkage
    out = np.ones(len(df))
    sig = np.where(df["padj"].to_numpy() < fdr_thr)[0]
    if sig.size == 0:
        return out
    chrs = df["chr"].to_numpy()[sig]
    starts = df["start"].to_numpy(np.float64)[sig]
    sizes = []
    for chrom in pd.unique(chrs):
        st = starts[chrs == chrom] / 1e6
        if st.size == 1:
            sizes.append(np.ones(1))
            continue
        Z = linkage(st[:, None], method="complete")
        cl = fcluster(Z, t=hclust_dist / 1e6, criterion="distance")
        _, inv, cnt = np.unique(cl, return_inverse=True, return_counts=True)
        sizes.append(cnt[inv].astype(np.float64))
    out[sig] = np.concatenate(sizes)
    return out


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def differential_compartments(
    scores: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
    conditions: Union[Mapping[str, str], Sequence[str]],
    *,
    rzscore: float = 2.0,
    szscore: float = 0.0,
    refine: bool = True,
    rconf: float = 0.90,
    fdr: float = 0.1,
    distclust: float = -1,
    numberclust: float = 1,
    padj_method: str = "auto",
    out_dir: Optional[Union[str, Path]] = None,
    seed: int = 123,
) -> pd.DataFrame:
    """Differential compartments across conditions with dcHiC's statistic.

    Same computation as ``dchicf.r --pcatype analyze`` (see
    :func:`dchic_analyze` for the file-based driver), on compartment scores
    held in memory, e.g. the output of
    :func:`epione.single.hic.project_compartments`.

    Arguments:
        scores: either a wide table with ``chrom`` (or ``chr``), ``start``,
            ``end`` and one column per sample (NaN where a sample has no score
            for a bin), or a mapping ``sample -> table`` with ``chrom``,
            ``start``, ``end`` and one score column.
        conditions: ``sample -> condition``, or a list of conditions aligned
            with the sample columns. Conditions are tested in order of first
            appearance; samples of the same condition are replicates.
        rzscore: dcHiC ``--rzsc``, replicate z-score threshold (default 2).
        szscore: dcHiC ``--szsc``, sample z-score threshold (default 0).
        refine: dcHiC ``--refine``: re-estimate the covariance without the
            outlying bins and recompute (default ``True``).
        rconf: dcHiC ``--rconf`` (default 0.90).
        fdr: dcHiC ``--fdr``; only used for the ``Filtered`` files and the
            distance clustering (default 0.1).
        distclust: dcHiC ``--distclust`` (default -1, off).
        numberclust: dcHiC ``--numberclust`` (default 1).
        padj_method: ``"auto"`` follows dcHiC (Benjamini-Hochberg without
            replicates; IHW with replicates, which raises
            ``NotImplementedError``). ``"BH"`` forces Benjamini-Hochberg.
        seed: dcHiC ``--seed`` (default 123), the ``set.seed`` before the
            analysis; the robust covariance draws random subsets when it uses
            the Donoho-Stahel or MCD estimator (fewer than 20 conditions).
        out_dir: if given, the dcHiC result tree (``<cond>_data``, ``pcOri``,
            ``pcQnm``, ``fdr_result``) is written there.

    Returns:
        ``pandas.DataFrame`` with the content of dcHiC's
        ``differential.intra_sample_combined.pcQnm.bedGraph``: ``chr``,
        ``start``, ``end``, the normalised score of every sample, the mean per
        condition, ``replicate_wt`` (with replicates), ``sample_maha``,
        ``pval``, ``padj`` and ``dist_clust``.
    """
    if padj_method not in ("auto", "BH"):
        raise ValueError("padj_method must be 'auto' or 'BH'")
    if isinstance(scores, Mapping):
        tables = {}
        for name, t in scores.items():
            t = t.rename(columns={"chrom": "chr"})
            vcol = [c for c in t.columns if c not in ("chr", "start", "end")]
            if len(vcol) != 1:
                raise ValueError(f"table for {name!r} needs exactly one score column")
            tables[str(name)] = t[["chr", "start", "end", vcol[0]]].rename(columns={vcol[0]: "PC"})
        names = list(tables)
    else:
        wide = scores.rename(columns={"chrom": "chr"})
        names = [c for c in wide.columns if c not in ("chr", "start", "end")]
        tables = {str(n): wide[["chr", "start", "end", n]].rename(columns={n: "PC"}).dropna(subset=["PC"])
                  for n in names}
        names = [str(n) for n in names]
    if isinstance(conditions, Mapping):
        cond_of = {str(k): str(v) for k, v in conditions.items()}
    else:
        if len(conditions) != len(names):
            raise ValueError("conditions must align with the sample columns")
        cond_of = dict(zip(names, map(str, conditions)))
    missing = [n for n in names if n not in cond_of]
    if missing:
        raise ValueError(f"no condition for samples {missing}")
    samples = []
    for n in names:
        t = tables[n].copy()
        t["chr"] = t["chr"].astype(str)
        t["start"] = t["start"].astype(np.int64)
        t["end"] = t["end"].astype(np.int64)
        t["PC"] = t["PC"].astype(np.float64)
        per = {c: g.reset_index(drop=True) for c, g in t.groupby("chr", sort=False)}
        samples.append(_Sample(n, cond_of[n], per))
    conds = list(dict.fromkeys(cond_of[n] for n in names))
    out = Path(out_dir) if out_dir is not None else None
    return _pcanalyze(samples, conds, out=out, rzscore=rzscore, szscore=szscore, fdr=fdr,
                      distclust=distclust, numberclust=numberclust, refine=refine, rconf=rconf,
                      padj_method=padj_method, seed=seed)


def _read_bedgraph_r(path: Path) -> pd.DataFrame:
    """``read.table(path, h=F, as.is=T)`` of a 4-column PC bedGraph."""
    raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=str, comment=None)
    return pd.DataFrame({
        "chr": raw[0].astype(str),
        "start": raw[1].astype(np.int64),
        "end": raw[2].astype(np.int64),
        "PC": r_strtod(raw[3].tolist()),
    })


def dchic_analyze(
    input_file: Union[str, Path],
    workdir: Union[str, Path] = ".",
    *,
    diffdir: str = "sample",
    dirovwt: bool = False,
    rzscore: float = 2.0,
    szscore: float = 0.0,
    refine: bool = True,
    rconf: float = 0.90,
    fdr: float = 0.1,
    distclust: float = -1,
    numberclust: float = 1,
    padj_method: str = "auto",
    seed: int = 123,
) -> Optional[pd.DataFrame]:
    """Python drop-in for ``Rscript dchicf.r --file <input> --pcatype analyze``.

    Reads dcHiC's ``input.txt`` (matrix, bed, prefix, prefix.master; matrix and
    bed are not used here) and, for every prefix, the compartment scores in
    ``<workdir>/<prefix>_pca/intra_pca/<prefix>_mat/<chrom>.pc.bedGraph``
    (written by dcHiC's ``cis`` + ``select`` steps, or by any other caller).
    Writes the same files as dcHiC under
    ``<workdir>/DifferentialResult/<diffdir>/``.

    Arguments:
        input_file: dcHiC input table.
        workdir: directory holding the ``<prefix>_pca`` folders (dcHiC's
            working directory).
        diffdir: ``--diffdir``.
        dirovwt: ``--dirovwt``. As in dcHiC, nothing is done when
            ``DifferentialResult`` exists and this is ``False``.
        rzscore, szscore, refine, rconf, fdr, distclust, numberclust,
            padj_method, seed: as in :func:`differential_compartments`
            (``seed`` is ``--seed``).

    Returns:
        The combined table (see :func:`differential_compartments`), or
        ``None`` when the run was skipped.
    """
    workdir = Path(workdir)
    inp = Path(input_file)
    if not inp.is_absolute():
        inp = workdir / inp if not inp.exists() else inp
    data = pd.read_csv(inp, sep=r"\s+", header=None, dtype=str, comment="#")
    if data.shape[1] < 4:
        raise ValueError("input file needs 4 columns: matrix, bed, prefix, prefix.master")
    data = data.iloc[:, :4]
    data.columns = ["mat", "bed", "prefix", "prefix.master"]
    root = workdir / "DifferentialResult"
    if root.exists() and not dirovwt:
        print(f"[dchic_analyze] {root} exists and dirovwt=False; nothing done")
        return None
    samples = []
    for prefix, master in zip(data["prefix"], data["prefix.master"]):
        if (workdir / f"{prefix}_pca" / "inter_pca" / f"{prefix}_mat").is_dir():
            raise NotImplementedError("trans (inter_pca) differential compartments are not ported")
        mdir = workdir / f"{prefix}_pca" / "intra_pca" / f"{prefix}_mat"
        if not mdir.is_dir():
            continue
        tables, texts = {}, {}
        for f in os.listdir(mdir):
            if f.endswith(".pc.bedGraph"):
                chrom = f[: -len(".pc.bedGraph")]
                tables[chrom] = _read_bedgraph_r(mdir / f)
                texts[chrom] = mdir / f
        samples.append(_Sample(str(prefix), str(master), tables, texts))
    conds = list(dict.fromkeys(data["prefix.master"].astype(str)))
    out = root / diffdir
    return _pcanalyze(samples, conds, out=out, rzscore=rzscore, szscore=szscore, fdr=fdr,
                      distclust=distclust, numberclust=numberclust, refine=refine, rconf=rconf,
                      padj_method=padj_method, seed=seed)
