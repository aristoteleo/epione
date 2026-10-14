"""Loop detection the way ``hicDetectLoops`` does it (HiCExplorer 3.7.6).

Port of ``hicexplorer/hicDetectLoops.py``, the obs/exp helpers it uses from
``hicexplorer/utilities.py`` and ``hicexplorer/lib/cnb.py`` (deeptools/HiCExplorer,
GPL-3.0, Copyright (c) Fidel Ramírez, Joachim Wolff and contributors;
Wolff et al., NAR 2020), and of ``fit_nbinom`` 1.2 (Joachim Wolff, GPL-3.0).
Used by ``loops(method="hicexplorer")``.

Per chromosome (pixels up to ``max_loop_distance``):

1. observed/expected per distance (``expected="mean"``: all pixels at a
   distance, HiCExplorer default);
2. for every distance, fit a negative binomial to the obs/exp values and keep
   pixels with obs/exp >= ``obs_exp_threshold`` whose upper-tail p-value is
   <= ``p_value_preselection`` and whose raw count >= ``peak_interactions_threshold``;
3. keep a candidate only if it is the maximum of its (2w+1)^2 neighbourhood;
4. test the peak (2p+1)^2 against the donut (bottom-left corner, horizontal,
   vertical strips) and the whole background with Wilcoxon rank-sum tests at
   ``p_value``.

HiCExplorer splits steps 2-4 across processes, but each candidate is handled
independently, so the result does not depend on the thread count.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, triu


# ------------------------------------------------------------- fit_nbinom 1.2
def fit_nbinom(X, initial_params=None):
    from scipy.optimize import fmin_l_bfgs_b as optim
    from scipy.special import factorial, gammaln

    infinitesimal = np.finfo(np.float64).eps

    def log_likelihood(params, *args):
        r, p = params
        X = args[0]
        N = X.size
        result = np.sum(gammaln(X + r)) \
            - np.sum(np.log(factorial(X))) \
            - N * (gammaln(r)) \
            + N * r * np.log(p) \
            + np.sum(X * np.log(1 - (p if p < 1 else 1 - infinitesimal)))
        return -result

    if initial_params is None:
        m = np.mean(X)
        v = np.var(X)
        size = (m ** 2) / (v - m) if v > m else 10
        p0 = size / ((size + m) if size + m != 0 else 1)
        initial_params = np.array([size, p0])
    bounds = [(infinitesimal, None), (infinitesimal, 1)]
    res = optim(log_likelihood, x0=initial_params, args=(X,), approx_grad=1, bounds=bounds)
    return {"size": res[0][0], "prob": res[0][1]}


def cnb_cdf(x, r, p):
    from scipy import special
    return special.betainc(r, x + 1, p)


# --------------------------------------------------------------- obs / exp
def _expected_interactions(sub):
    exp = np.zeros(sub.shape[0])
    row, col = sub.nonzero()
    dist = np.absolute(row - col)
    occ = np.arange(sub.shape[0] + 1, 1, -1)
    if len(dist) == 0:
        return None
    for i in range(dist.min(), dist.max() + 1, 1):
        exp[i] = np.sum(sub.data[dist == i])
    with np.errstate(divide="ignore", invalid="ignore"):
        exp = np.divide(exp, occ)
    exp[np.isnan(exp)] = 0
    exp[np.isinf(exp)] = 0
    return exp


def _to_eps(a):
    if len(a) == 0:
        return a
    a[np.flatnonzero(np.isnan(a))] = 0.000001
    a[np.flatnonzero(np.isinf(a))] = 0.000001
    return a


def _obs_exp_mean(sub):
    exp = _expected_interactions(sub)
    if exp is None:
        return None
    row, col = sub.nonzero()
    distance = np.ceil(np.absolute(row - col) / 2).astype(np.int32)
    out = sub.copy()
    if len(sub.data) > 0:
        dtype = type(sub.data[0])
        e = exp[distance]
        out.data = out.data.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            out.data = np.divide(out.data, e)
        out.data = _to_eps(out.data).astype(dtype)
    return out


def _obs_exp_non_zero(sub, ligation_factor):
    out = sub.copy()
    row, col = out.nonzero()
    n = len(row)
    dist = np.absolute(row - col)
    expected = np.zeros(out.shape[0])
    occ = np.zeros(out.shape[0])
    expected = np.bincount(dist, weights=out.data[:n], minlength=out.shape[0])[:out.shape[0]].astype(float)
    occ = np.bincount(dist, minlength=out.shape[0])[:out.shape[0]].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.divide(expected, occ)
    expected[np.isnan(expected)] = 0
    expected[np.isinf(expected)] = 0
    row_sums = np.array(out.sum(axis=1).T).flatten()
    total = out.sum()
    out.data = out.data.astype(np.float32)
    e = expected[dist]
    if ligation_factor:
        e = e * (row_sums[row] * row_sums[col] / total)
    with np.errstate(divide="ignore", invalid="ignore"):
        out.data[:n] = np.divide(out.data[:n], e)
    out.data[np.isnan(out.data)] = 0.000000001
    out.data[np.isinf(out.data)] = 0.000000001
    out.eliminate_zeros()
    return out


# -------------------------------------------------------------- loading
def _load_chrom(uri, chrom, max_distance, weight_name="weight"):
    """``hiCMatrix(pChrnameList=[chrom], pDistance=...)`` for a .cool."""
    import cooler

    clr = cooler.Cooler(uri)
    lo, hi = clr.extent(chrom)
    if max_distance is None or clr.binsize is None:
        matrix = clr.matrix(balance=False, sparse=True, as_pixels=False).fetch(chrom).tocsr()
    else:
        dist = max_distance // clr.binsize
        step = (hi - lo) // 32
        if step < 1:
            step = 1
        mat = lil_matrix((hi - lo, hi - lo), dtype=np.float32)
        for i0, i1 in cooler.util.partition(lo, hi, step):
            pix = clr.matrix(balance=False, as_pixels=True)[i0:i1, lo:hi]
            pix = pix[(pix["bin2_id"] - pix["bin1_id"]) < dist]
            mat[pix["bin1_id"] - lo, pix["bin2_id"] - lo] = pix["count"].astype(np.float32)
        matrix = mat.tocsr()
    bins = clr.bins().fetch(chrom)
    if weight_name in bins.columns:
        matrix.eliminate_zeros()
        if len(matrix.data) > 1:
            matrix.data = matrix.data.astype(float)
            cf = np.array(bins[weight_name].values).flatten()
            if np.sum(np.isnan(cf)) != len(cf):
                inst, feat = matrix.nonzero()
                f = cf[inst]
                f *= cf[feat]
                matrix.data *= f
    cut = [(str(c), int(s), int(e), 1.0) for c, s, e in bins[["chrom", "start", "end"]].values]
    mask = np.isnan(matrix.data)
    matrix.data[mask] = 0
    matrix.eliminate_zeros()
    # fillLowerTriangle
    from scipy import sparse
    if sparse.tril(matrix, k=-1).sum() == 0:
        matrix = matrix + triu(matrix, 1).T
    return matrix, cut


def _bin_size(cut):
    from collections import Counter
    chrom, start, end, _ = zip(*cut)
    diff = np.array(end) - np.array(start)
    if len(diff) == 1:
        return diff[0]
    return int(np.median(np.concatenate([
        np.diff([s for c, s, e, x in cut if c == cur]) for cur, nb in Counter(chrom).items() if nb > 1])))


# --------------------------------------------------------------- core
def _p_value_mask(oe_data, distance, pv_pre, obs_exp_thr, resolution):
    true_idx = []
    is_float = isinstance(pv_pre, float)
    for key in range(distance.min(), distance.max() + 1):
        m = distance == key
        vals = oe_data[m]
        pos = np.argwhere(m).flatten()
        # every distance in [min, max] is fitted, including empty ones, as in HiCExplorer
        data_obs_exp = np.array(vals)
        mask = data_obs_exp >= obs_exp_thr
        params = fit_nbinom(data_obs_exp)
        p_value = 1 - cnb_cdf(data_obs_exp[mask], params["size"], params["prob"])
        if is_float:
            md = p_value <= pv_pre
        else:
            md = p_value <= pv_pre[int(key * resolution)]
        j = 0
        for k, value in enumerate(mask):
            if value:
                if md[j]:
                    true_idx.append(pos[k])
                j += 1
    return true_idx


def _neighborhood_merge(cands, w, M):
    out = []
    x_max, y_max = M.shape
    for c in cands:
        sx = c[0] - w if (c[0] - w) > 0 else 0
        sy = c[1] - w if (c[1] - w) > 0 else 0
        ex = c[0] + w + 1 if c[0] + w + 1 < x_max else x_max
        ey = c[1] + w + 1 if c[1] + w + 1 < y_max else y_max
        nb = M[sx:ex, sy:ey].toarray().flatten()
        if len(nb) > 0 and np.max(nb) == M[c[0], c[1]]:
            out.append(c)
    return out


def _region_test(M, cands, w, p_value, pw):
    from scipy.stats import ranksums

    mask, pvals = [], []
    x_max, y_max = M.shape
    for c in cands:
        sx = c[0] - w if (c[0] - w) > 0 else 0
        sy = c[1] - w if (c[1] - w) > 0 else 0
        ex = c[0] + w + 1 if c[0] + w + 1 < x_max else x_max
        ey = c[1] + w + 1 if c[1] + w + 1 < y_max else y_max
        nb = M[sx:ex, sy:ey].toarray()
        if len(nb) == 0:
            mask.append(False)
            continue
        pr = np.array(np.where(nb == M[c[0], c[1]])).flatten()
        peak = nb[pr[0] - pw:pr[0] + pw + 1, pr[1] - pw:pr[1] + pw + 1].flatten()
        bg = []
        bg.extend(list(nb[:pr[0] - pw, :].flatten()))
        bg.extend(list(nb[pr[0] + pw + 1:, :].flatten()))
        bg.extend(list(nb[pr[0] - pw:pr[0] + pw + 1, pr[1] + pw + 1:].flatten()))
        bg.extend(list(nb[pr[0] - pw:pr[0] + pw + 1, :pr[1] - pw].flatten()))
        bg = np.array(bg)
        if len(bg) < w or len(peak) < w or np.mean(peak) < np.mean(bg) or np.max(peak) < np.max(bg):
            mask.append(False)
            continue
        hor = []
        hor.extend(nb[:pr[0] - pw, pr[1] - pw:pr[1] + pw + 1].flatten())
        hor.extend(nb[pr[0] + pw + 1:, pr[1] - pw:pr[1] + pw + 1].flatten())
        hor = np.array(hor).flatten()
        ver = []
        ver.extend(nb[pr[0] - pw:pr[0] + pw + 1, :pr[1] - pw].flatten())
        ver.extend(nb[pr[0] - pw:pr[0] + pw + 1, pr[1] + pw + 1:].flatten())
        ver = np.array(ver).flatten()
        blc = []
        blc.extend(nb[pr[0]:, :pr[1] - pw].flatten())
        blc.extend(nb[pr[0] + pw + 1:, pr[1] - pw:pr[1] + 1].flatten())
        accept = 0
        for data in (blc, hor, ver):
            _, s1 = ranksums(sorted(peak), sorted(data))
            if s1 <= p_value:
                accept += 1
        if accept >= 3:
            _, s = ranksums(sorted(peak), sorted(bg))
            if s <= p_value:
                mask.append(True)
                pvals.append(s)
                continue
        mask.append(False)
    return mask, pvals


def detect_loops_chrom(uri, chrom, *, window_size=5, peak_width=2, p_value_preselection=0.1,
                       peak_interactions_threshold=10, obs_exp_threshold=1.5, p_value=0.025,
                       max_loop_distance=2000000, expected="mean"):
    M, cut = _load_chrom(uri, chrom, max_loop_distance)
    if len(M.data) == 0 or M.shape[0] < 5 or M.shape[1] < 5:
        return []
    M = triu(M, format="csr")
    M.eliminate_zeros()
    inst, feat = M.nonzero()
    M.data[np.absolute(inst - feat) == 0] = 0
    M.eliminate_zeros()
    if expected == "mean":
        oe = _obs_exp_mean(M)
    elif expected == "mean_nonzero":
        oe = _obs_exp_non_zero(M, False)
    elif expected == "mean_nonzero_ligation":
        oe = _obs_exp_non_zero(M, True)
    else:
        raise ValueError("expected must be 'mean', 'mean_nonzero' or 'mean_nonzero_ligation'")
    if not isinstance(oe, csr_matrix):
        return []
    M.eliminate_zeros()
    oe.eliminate_zeros()
    if len(M.data) != len(oe.data):
        return []
    try:
        pv_pre = float(p_value_preselection)
    except (TypeError, ValueError):
        pv_pre = p_value_preselection
    inst, feat = oe.nonzero()
    distance = np.absolute(inst - feat)
    hard = M.data >= peak_interactions_threshold
    keep = np.zeros(len(distance), dtype=bool)
    keep[_p_value_mask(oe.data, distance, pv_pre, obs_exp_threshold, _bin_size(cut))] = True
    keep = np.logical_and(keep, hard)
    inst, feat = inst[keep], feat[keep]
    if len(feat) == 0:
        return []
    cands = np.array([*zip(inst, feat)])
    cands = _neighborhood_merge(cands, window_size, oe)
    if len(cands) == 0:
        return []
    cands = np.array(cands)
    mask, pvals = _region_test(oe, cands, window_size, p_value, peak_width)
    if len(mask) == 0:
        return []
    cands = cands[np.array(mask)]
    out = []
    for c, p in zip(cands, pvals):
        cx, sx, ex, _ = cut[c[0]]
        cy, sy, ey, _ = cut[c[1]]
        if max_loop_distance is not None and abs(int(sx) - int(sy)) > max_loop_distance:
            continue
        out.append((cx, sx, ex, cy, sy, ey, float(p)))
    return out


def detect_loops(uri, chromosomes: Optional[Sequence[str]] = None, **kw) -> pd.DataFrame:
    import cooler
    if kw.get("window_size", 5) <= kw.get("peak_width", 2):
        raise ValueError("window_size must be larger than peak_width")
    clr = cooler.Cooler(uri)
    chroms = list(clr.chromnames) if chromosomes is None else list(chromosomes)
    rows = []
    for c in chroms:
        rows.extend(detect_loops_chrom(uri, c, **kw))
    return pd.DataFrame(rows, columns=["chrom1", "start1", "end1", "chrom2", "start2", "end2", "pvalue"])
