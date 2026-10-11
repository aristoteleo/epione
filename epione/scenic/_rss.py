"""Regulon specificity score (RSS; Suo et al., Cell Rep 2018), as in SCENIC+.

``RSS(regulon, group) = 1 - JSD(auc / sum(auc), onehot(group) / |group|)`` with the
Jensen-Shannon distance (square root of the divergence, natural log).

Floating-point contract (bit-identical to SCENIC+ with SciPy 1.12): every sum is numpy's
pairwise summation, and the relative entropy ``x * log(x / y)`` uses the C library ``log``
(SciPy's ``rel_entr``), called directly rather than through a vectorised log whose last bit
can differ between numpy/SciPy builds.
"""

from __future__ import annotations

import ctypes
import ctypes.util

import numba
import numpy as np
import pandas as pd

from ._egrn_gsea import _np_sum

__all__ = ["regulon_specificity_scores"]

_libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
_clog = _libm.log
_clog.restype = ctypes.c_double
_clog.argtypes = [ctypes.c_double]


@numba.njit(cache=False)
def _rel_entr(x, y):
    if x != x or y != y:
        return np.nan
    if x > 0 and y > 0:
        return x * _clog(x / y)
    if x == 0 and y >= 0:
        return 0.0
    return np.inf


@numba.njit(cache=False)
def _rss_kernel(A, groups, n_groups):
    n_reg, n = A.shape
    out = np.empty((n_groups, n_reg), dtype=np.float64)
    p = np.empty(n, dtype=np.float64)
    q = np.empty(n, dtype=np.float64)
    m = np.empty(n, dtype=np.float64)
    left = np.empty(n, dtype=np.float64)
    right = np.empty(n, dtype=np.float64)
    for g in range(n_groups):
        cnt = 0
        for i in range(n):
            if groups[i] == g:
                cnt += 1
        for i in range(n):
            q[i] = (1 if groups[i] == g else 0) / cnt
        sq = _np_sum(q, n)
        for i in range(n):
            q[i] = q[i] / sq
        for r in range(n_reg):
            s1 = _np_sum(A[r], n)
            for i in range(n):
                p[i] = A[r, i] / s1
            s2 = _np_sum(p, n)
            for i in range(n):
                p[i] = p[i] / s2
                m[i] = (p[i] + q[i]) / 2.0
                left[i] = _rel_entr(p[i], m[i])
                right[i] = _rel_entr(q[i], m[i])
            js = _np_sum(left, n) + _np_sum(right, n)
            out[g, r] = 1.0 - np.sqrt(js / 2.0)
    return out


def regulon_specificity_scores(auc: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """RSS of every regulon (columns of ``auc``) for every group (groups x regulons).

    Parameters
    ----------
    auc
        Cells x regulons AUC matrix.
    groups
        Group label per cell (aligned to ``auc.index``). Groups are reported in order of
        first appearance, as SCENIC+ does.
    """
    labels = groups.loc[auc.index]
    cell_types = list(labels.unique())
    codes = pd.Index(cell_types).get_indexer(labels.to_numpy())
    A = np.ascontiguousarray(auc.to_numpy(dtype=np.float64).T)
    out = _rss_kernel(A, codes.astype(np.int64), len(cell_types))
    return pd.DataFrame(out, index=cell_types, columns=auc.columns)
