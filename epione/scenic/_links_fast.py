"""Histogram gradient boosting / random forests for the fast SCENIC+ link modes.

``method="fast"`` in :func:`epione.scenic.tf_to_gene` and :func:`epione.scenic.region_to_gene`
replaces the bit-exact scikit-learn 1.3.2 trees (:mod:`epione.scenic._sktree`) with
histogram trees:

* every predictor column is binned once (at most 255 bins; columns with fewer distinct
  values keep one bin per value, so no split candidate is lost) and the bins are shared by
  all targets;
* a node's histogram is built from its samples or, when the column is sparser than the
  node, from the column's non-zero entries plus the node totals (the zero bin is the
  remainder), which is what makes TF-to-gene cheap on sparse expression;
* all targets are fitted in one numba ``prange`` loop.  Each target draws from its own
  generator seeded by ``(seed, target column)``, so the result does not depend on the
  number of threads or on which other targets are fitted.

The models keep GRNBoost2's semantics: least squares, depth-3 trees, learning rate,
row subsampling with arboreto's out-of-bag early-stopping rule, ``max_features`` drawn
per node (constant features are redrawn, as in scikit-learn), and impurity-decrease
("gain") importances aggregated the way scikit-learn aggregates them.  They are not meant
to reproduce scikit-learn's numbers; see ``tests/parity/scenic_links/MANIFEST.md`` for
how they compare with the exact path.
"""
from __future__ import annotations

import numba as nb
import numpy as np

__all__ = ["bin_columns", "csc_nonzero", "gbm_many", "rf_many"]

_MAX_BINS = 65535  # lossless for any column with fewer distinct values
_EPS = np.finfo(np.float64).eps
# count-trailing-zeros of an isolated low bit (de Bruijn multiplication)
_DEBRUIJN_MAGIC = np.uint64(0x03F79D71B4CB0A89)
_DEBRUIJN = np.zeros(64, dtype=np.int64)
for _i in range(64):
    _DEBRUIJN[(((1 << _i) * 0x03F79D71B4CB0A89) & 0xFFFFFFFFFFFFFFFF) >> 58] = _i
del _i  # scikit-learn's leaf rule: impurity <= EPSILON


# ---------------------------------------------------------------------------
# binning
# ---------------------------------------------------------------------------
@nb.njit(cache=True, nogil=True)
def _bin_one(col, max_bins, out_bins, edges_buf):
    """Bin one column; returns (n_bins, bin index of the value 0 or -1)."""
    n = col.shape[0]
    v = np.sort(col)
    nu = 1
    for i in range(1, n):
        if v[i] != v[i - 1]:
            nu += 1
    if nu <= max_bins:
        # one bin per distinct value: edges are the distinct values themselves
        k = 0
        edges_buf[0] = v[0]
        for i in range(1, n):
            if v[i] != v[i - 1]:
                k += 1
                edges_buf[k] = v[i]
        ne = nu
    else:
        # quantile edges over the sorted values (upper edge of each bin), deduplicated
        ne = 0
        for b in range(1, max_bins + 1):
            q = v[min(n - 1, (b * n) // max_bins - 1)]
            if ne == 0 or q > edges_buf[ne - 1]:
                edges_buf[ne] = q
                ne += 1
        if edges_buf[ne - 1] < v[n - 1]:
            edges_buf[ne - 1] = v[n - 1]
    # bin = first edge >= value
    for i in range(n):
        out_bins[i] = _first_ge(edges_buf, ne, col[i])
    return ne, _first_ge(edges_buf, ne, 0.0)


@nb.njit(cache=True, nogil=True)
def _first_ge(edges, ne, x):
    lo, hi = 0, ne - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if edges[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@nb.njit(cache=True, parallel=True)
def _bin_columns(X, max_bins, B, nbins, zbin):
    n, p = X.shape
    for j in nb.prange(p):
        col = np.empty(n, dtype=np.float64)
        for i in range(n):
            col[i] = X[i, j]
        out = np.empty(n, dtype=np.int64)
        edges = np.empty(min(max_bins, n) + 1, dtype=np.float64)
        ne, zb = _bin_one(col, max_bins, out, edges)
        for i in range(n):
            B[i, j] = out[i]
        nbins[j] = ne
        zbin[j] = zb


def bin_columns(X, max_bins: int = _MAX_BINS):
    """Bin the columns of ``X`` (cells x features) once.

    By default a column keeps one bin per distinct value (up to 65,535), so the split
    candidates are exactly scikit-learn's; above ``max_bins`` distinct values the bins are
    quantiles.

    Returns ``(B, n_bins, zero_bin)``: ``B`` is a Fortran-ordered uint8 matrix of bin
    indices (a split "bin <= b" is "value <= the b-th edge"), ``n_bins[j]`` the number of
    bins of column ``j`` and ``zero_bin[j]`` the bin holding the value 0 (only meaningful
    for non-negative columns that contain 0; -1 otherwise).
    """
    if not 2 <= max_bins <= 65535:
        raise ValueError("max_bins must be in [2, 65535]")
    X = np.asarray(X)
    n, p = X.shape
    B = np.empty((n, p), dtype=np.uint8 if max_bins <= 256 else np.uint16, order="F")
    nbins = np.empty(p, dtype=np.int64)
    zbin = np.empty(p, dtype=np.int64)
    _bin_columns(X, int(max_bins), B, nbins, zbin)
    if B.dtype != np.uint8 and (p == 0 or nbins.max() <= 256):
        B = np.asfortranarray(B.astype(np.uint8))
    # the zero bin is used to fill in the rows skipped by the sparse path, so it must hold
    # exactly the zeros: only for non-negative columns that contain 0
    ok = (X == 0).any(axis=0) & (X >= 0).all(axis=0) if n else np.zeros(p, dtype=bool)
    zbin = np.where(np.asarray(ok).ravel(), zbin, -1)
    return B, nbins, zbin.astype(np.int64)


def csc_nonzero(B, zbin):
    """Rows whose bin differs from the zero bin, per column, with their bins.

    Returns CSC ``(indptr, indices, values)``; columns without a zero bin get every row
    (they always use the dense path).
    """
    n, p = B.shape
    indptr = np.zeros(p + 1, dtype=np.int64)
    rows = []
    for j in range(p):
        r = np.flatnonzero(B[:, j] != zbin[j]) if zbin[j] >= 0 else np.arange(n)
        rows.append(r.astype(np.int32))
        indptr[j + 1] = indptr[j] + len(r)
    indices = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    values = np.empty(len(indices), dtype=B.dtype)
    for j in range(p):
        values[indptr[j]:indptr[j + 1]] = B[indices[indptr[j]:indptr[j + 1]], j]
    return indptr, indices, values


# ---------------------------------------------------------------------------
# random numbers (xorshift64*, splitmix64 seeding)
# ---------------------------------------------------------------------------
@nb.njit(cache=True, nogil=True)
def _splitmix(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15))
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@nb.njit(cache=True, nogil=True)
def _next(state):
    x = state[0]
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    state[0] = x
    return x * np.uint64(0x2545F4914F6CDD1D)


@nb.njit(cache=True, nogil=True)
def _below(state, k):
    """Integer in [0, k) for k < 2**32 (Lemire's multiply-shift; bias < k / 2**32)."""
    return np.int64(((_next(state) >> np.uint64(32)) * np.uint64(k)) >> np.uint64(32))


@nb.njit(cache=True, nogil=True)
def _seed_state(seed, key):
    st = np.empty(1, dtype=np.uint64)
    st[0] = _splitmix(np.uint64(seed) * np.uint64(0x100000001B3) + np.uint64(key))
    if st[0] == 0:
        st[0] = np.uint64(1)
    return st


# ---------------------------------------------------------------------------
# tree growing
# ---------------------------------------------------------------------------
@nb.njit(cache=True, nogil=True)
def _grow(B, nbins, zbin, indptr, indices, values, use_sparse, cols, y, w, wy, max_depth,
          max_features, state, samples, node_of, fperm, h, bm, stack, tree_imp,
          leaf_val_out):
    """Grow one regression tree on residuals ``y`` with sample weights ``w``.

    ``samples`` holds all rows (rows with ``w == 0`` ride along so their leaf is known).
    ``wy`` is ``w * y``.  The histogram ``h`` (bins x [weight, weighted residual, rows]) and
    the bin bitmap ``bm`` must be zero on entry and are left zero.
    Rows with ``w == 0`` keep ``node_of == -1``, so the sparse scan needs no weight test.
    Adds each split's weighted impurity decrease to ``tree_imp`` (unnormalised, not
    divided by the root weight) and writes every row's leaf value to ``leaf_val_out``.
    Returns (number of splits, root weight).
    """
    n = samples.shape[0]
    p = cols.shape[0]
    for i in range(n):
        samples[i] = i
        node_of[i] = 0 if w[i] > 0.0 else -1
    for f in range(p):
        fperm[f] = f
    sp = 0
    stack[0, 0] = 0
    stack[0, 1] = n
    stack[0, 2] = 0
    stack[0, 3] = 0  # node id
    sp = 1
    next_id = 1
    n_splits = 0
    root_w = 0.0
    while sp > 0:
        sp -= 1
        start = stack[sp, 0]
        end = stack[sp, 1]
        depth = stack[sp, 2]
        nid = stack[sp, 3]
        W = 0.0
        G = 0.0
        S2 = 0.0
        C = 0
        for i in range(start, end):
            s = samples[i]
            ws = w[s]
            if ws > 0.0:
                W += ws
                G += ws * y[s]
                S2 += ws * y[s] * y[s]
                C += 1
        if nid == 0:
            root_w = W
        value = G / W if W > 0.0 else 0.0
        impurity = S2 / W - value * value if W > 0.0 else 0.0
        best_gain = 0.0
        best_f = -1
        best_b = -1
        if depth < max_depth and C >= 2 and impurity > _EPS:
            base = G * G / W
            n_drawn = 0
            n_eval = 0
            # Fisher-Yates over the features until max_features non-constant ones are seen
            while n_eval < max_features and n_drawn < p:
                k = n_drawn + _below(state, p - n_drawn)
                tmp = fperm[n_drawn]
                fperm[n_drawn] = fperm[k]
                fperm[k] = tmp
                f = fperm[n_drawn]
                n_drawn += 1
                col = cols[f]
                zb = zbin[col]
                nnz = indptr[col + 1] - indptr[col] if use_sparse else n
                # node histogram; the bins hit are flagged in a bitmap so the split scan and
                # the reset cost O(bins / 64 + bins hit) instead of O(bins)
                if use_sparse and zb >= 0 and nnz < (end - start):
                    sw = 0.0
                    sg = 0.0
                    sc = 0
                    for t in range(indptr[col], indptr[col + 1]):
                        r = indices[t]
                        if node_of[r] == nid:
                            b = values[t]
                            bm[b >> 6] |= np.uint64(1) << np.uint64(b & 63)
                            h[b, 0] += w[r]
                            h[b, 1] += wy[r]
                            h[b, 2] += 1
                            sw += w[r]
                            sg += wy[r]
                            sc += 1
                    if C - sc > 0:
                        bm[zb >> 6] |= np.uint64(1) << np.uint64(zb & 63)
                        h[zb, 0] += W - sw
                        h[zb, 1] += G - sg
                        h[zb, 2] += C - sc
                else:
                    for i in range(start, end):
                        s = samples[i]
                        if node_of[s] >= 0:
                            b = B[s, col]
                            bm[b >> 6] |= np.uint64(1) << np.uint64(b & 63)
                            h[b, 0] += w[s]
                            h[b, 1] += wy[s]
                            h[b, 2] += 1
                # walk the flagged bins in order: best split, then reset
                WL = 0.0
                GL = 0.0
                seen = 0
                prev = 0
                fb = -1
                fg = 0.0
                for word in range((nbins[col] + 63) >> 6):
                    x = bm[word]
                    bm[word] = np.uint64(0)
                    while x != np.uint64(0):
                        low = x & (~x + np.uint64(1))
                        x ^= low
                        b = (word << 6) + _DEBRUIJN[(low * _DEBRUIJN_MAGIC) >> np.uint64(58)]
                        if seen > 0:
                            # candidate split between the previous flagged bin and this one
                            WR = W - WL
                            if WL > 0.0 and WR > 0.0:
                                GR = G - GL
                                gain = GL * GL / WL + GR * GR / WR - base
                                if gain > fg:
                                    fg = gain
                                    fb = prev
                        WL += h[b, 0]
                        GL += h[b, 1]
                        seen += 1
                        prev = b
                        h[b, 0] = 0.0
                        h[b, 1] = 0.0
                        h[b, 2] = 0.0
                if seen > 1:
                    # constant features (one bin) do not count towards max_features
                    n_eval += 1
                    if fb >= 0 and fg > best_gain:
                        best_gain = fg
                        best_f = f
                        best_b = fb
        if best_f < 0:
            for i in range(start, end):
                leaf_val_out[samples[i]] = value
            continue
        # split: rows with bin <= best_b go left
        col = cols[best_f]
        tree_imp[best_f] += best_gain
        n_splits += 1
        lo = start
        hi = end - 1
        while lo <= hi:
            if B[samples[lo], col] <= best_b:
                lo += 1
            else:
                tmp = samples[lo]
                samples[lo] = samples[hi]
                samples[hi] = tmp
                hi -= 1
        lid = next_id
        rid = next_id + 1
        next_id += 2
        for i in range(start, lo):
            if node_of[samples[i]] >= 0:
                node_of[samples[i]] = lid
        for i in range(lo, end):
            if node_of[samples[i]] >= 0:
                node_of[samples[i]] = rid
        stack[sp, 0] = lo
        stack[sp, 1] = end
        stack[sp, 2] = depth + 1
        stack[sp, 3] = rid
        sp += 1
        stack[sp, 0] = start
        stack[sp, 1] = lo
        stack[sp, 2] = depth + 1
        stack[sp, 3] = lid
        sp += 1
    return n_splits, root_w


@nb.njit(cache=True, nogil=True)
def _gbm_one(B, nbins, zbin, indptr, indices, values, use_sparse, cols, y, lr, n_estimators,
             max_features, max_depth, subsample, early_stop_window, seed, key, imp):
    """One least-squares GBM; fills ``imp`` (normalised importances), returns stages fitted."""
    n = y.shape[0]
    p = cols.shape[0]
    for f in range(p):
        imp[f] = 0.0
    anyy = False
    for i in range(n):
        if y[i] != 0.0:
            anyy = True
            break
    if not anyy or p == 0:
        return n_estimators
    state = _seed_state(seed, key)
    init = 0.0
    for i in range(n):
        init += y[i]
    init /= n
    raw = np.full(n, init)
    resid = np.empty(n)
    w = np.ones(n)
    leafv = np.empty(n)
    samples = np.empty(n, dtype=np.int64)
    node_of = np.empty(n, dtype=np.int32)
    wy = np.empty(n)
    fperm = np.empty(p, dtype=np.int64)
    nbmax = 1
    for f in range(p):
        nbmax = max(nbmax, nbins[cols[f]])
    h = np.zeros((nbmax, 3))
    bm = np.zeros((nbmax + 63) // 64 + 1, dtype=np.uint64)
    stack = np.empty((2 * max_depth + 4, 4), dtype=np.int64)
    tree_imp = np.empty(p)
    acc = np.zeros(p)
    oob_impr = np.empty(n_estimators)
    perm = np.arange(n)
    n_inbag = max(1, int(subsample * n))
    do_oob = subsample < 1.0
    n_rel = 0
    prev_loss = 0.0
    i = 0
    for i in range(n_estimators):
        if do_oob:
            # partial Fisher-Yates: the first n_inbag rows of a random permutation are in bag
            for k in range(n_inbag):
                j = k + _below(state, n - k)
                t = perm[k]
                perm[k] = perm[j]
                perm[j] = t
            for k in range(n):
                w[k] = 0.0
            for k in range(n_inbag):
                w[perm[k]] = 1.0
            if i == 0:
                prev_loss = 0.0
                m = 0
                for k in range(n):
                    if w[k] == 0.0:
                        d = y[k] - raw[k]
                        prev_loss += d * d
                        m += 1
                prev_loss = prev_loss / m if m > 0 else 0.0
        for k in range(n):
            resid[k] = y[k] - raw[k]
            wy[k] = w[k] * resid[k]
        for f in range(p):
            tree_imp[f] = 0.0
        n_splits, root_w = _grow(B, nbins, zbin, indptr, indices, values, use_sparse, cols,
                                 resid, w, wy, max_depth, max_features, state, samples, node_of,
                                 fperm, h, bm, stack, tree_imp, leafv)
        for k in range(n):
            raw[k] += lr * leafv[k]
        if n_splits > 0 and root_w > 0.0:
            for f in range(p):
                acc[f] += tree_imp[f] / root_w
            n_rel += 1
        if do_oob:
            loss = 0.0
            m = 0
            for k in range(n):
                if w[k] == 0.0:
                    d = y[k] - raw[k]
                    loss += d * d
                    m += 1
            loss = loss / m if m > 0 else 0.0
            oob_impr[i] = prev_loss - loss
            prev_loss = loss
            if early_stop_window > 0 and i >= early_stop_window - 1:
                s = 0.0
                for k in range(i - early_stop_window + 1, i + 1):
                    s += oob_impr[k]
                if s / early_stop_window < 0:
                    break
    if n_rel > 0:
        tot = 0.0
        for f in range(p):
            tot += acc[f]
        if tot > 0.0:
            for f in range(p):
                imp[f] = acc[f] / tot
    return i + 1


@nb.njit(cache=True, nogil=True)
def _rf_one(B, nbins, zbin, indptr, indices, values, use_sparse, cols, y, n_estimators, max_features,
            seed, key, imp):
    """One random forest (bootstrap, squared error, fully grown trees); fills ``imp``."""
    n = y.shape[0]
    p = cols.shape[0]
    for f in range(p):
        imp[f] = 0.0
    anyy = False
    for i in range(n):
        if y[i] != 0.0:
            anyy = True
            break
    if not anyy or p == 0:
        return
    state = _seed_state(seed, key)
    w = np.empty(n)
    leafv = np.empty(n)
    samples = np.empty(n, dtype=np.int64)
    node_of = np.empty(n, dtype=np.int32)
    wy = np.empty(n)
    fperm = np.empty(p, dtype=np.int64)
    nbmax = 1
    for f in range(p):
        nbmax = max(nbmax, nbins[cols[f]])
    h = np.zeros((nbmax, 3))
    bm = np.zeros((nbmax + 63) // 64 + 1, dtype=np.uint64)
    stack = np.empty((2 * n + 4, 4), dtype=np.int64)
    tree_imp = np.empty(p)
    acc = np.zeros(p)
    n_rel = 0
    for t in range(n_estimators):
        for k in range(n):
            w[k] = 0.0
        for k in range(n):
            w[_below(state, n)] += 1.0
        for k in range(n):
            wy[k] = w[k] * y[k]
        for f in range(p):
            tree_imp[f] = 0.0
        n_splits, root_w = _grow(B, nbins, zbin, indptr, indices, values, use_sparse, cols, y, w,
                                 wy, 1 << 30, max_features, state, samples, node_of, fperm,
                                 h, bm, stack, tree_imp, leafv)
        if n_splits > 0:
            tot = 0.0
            for f in range(p):
                tot += tree_imp[f]
            if tot > 0.0:
                for f in range(p):
                    acc[f] += tree_imp[f] / tot
                n_rel += 1
    if n_rel > 0:
        tot = 0.0
        for f in range(p):
            tot += acc[f]
        if tot > 0.0:
            for f in range(p):
                imp[f] = acc[f] / tot


@nb.njit(cache=True, parallel=True)
def _gbm_many(B, nbins, zbin, indptr, indices, values, use_sparse, Y, y_cols, keys, feat_ptr, feats,
              lr, n_estimators, max_features, max_depth, subsample, early_stop_window, seed,
              imp_out, n_est_out):
    nt = y_cols.shape[0]
    for t in nb.prange(nt):
        cols = feats[feat_ptr[t]:feat_ptr[t + 1]]
        y = np.empty(Y.shape[0])
        for i in range(Y.shape[0]):
            y[i] = Y[i, y_cols[t]]
        imp = np.empty(cols.shape[0])
        p = cols.shape[0]
        mf = max_features[t]
        n_est_out[t] = _gbm_one(B, nbins, zbin, indptr, indices, values, use_sparse, cols, y, lr,
                                n_estimators, mf, max_depth, subsample, early_stop_window,
                                seed, keys[t], imp)
        for f in range(p):
            imp_out[feat_ptr[t] + f] = imp[f]


@nb.njit(cache=True, parallel=True)
def _rf_many(B, nbins, zbin, indptr, indices, values, use_sparse, Y, y_cols, keys, feat_ptr, feats,
             n_estimators, max_features, seed, imp_out):
    nt = y_cols.shape[0]
    for t in nb.prange(nt):
        cols = feats[feat_ptr[t]:feat_ptr[t + 1]]
        y = np.empty(Y.shape[0])
        for i in range(Y.shape[0]):
            y[i] = Y[i, y_cols[t]]
        imp = np.empty(cols.shape[0])
        _rf_one(B, nbins, zbin, indptr, indices, values, use_sparse, cols, y, n_estimators,
                max_features[t], seed, keys[t], imp)
        for f in range(cols.shape[0]):
            imp_out[feat_ptr[t] + f] = imp[f]


def _n_feat(max_features, p):
    if max_features == "sqrt":
        return max(1, int(np.sqrt(p)))
    if isinstance(max_features, float):
        return max(1, int(max_features * p))
    return int(max_features)


def _with_threads(n_jobs, fn, *args):
    prev = nb.get_num_threads()
    nb.set_num_threads(max(1, min(int(n_jobs or 1), nb.config.NUMBA_NUM_THREADS)))
    try:
        return fn(*args)
    finally:
        nb.set_num_threads(prev)


def gbm_many(binned, Y, y_cols, keys, feat_ptr, feats, *, learning_rate, n_estimators,
             max_features, max_depth=3, subsample=1.0, early_stop_window=0, seed=666,
             n_jobs=1, use_sparse=True):
    """Fit one histogram GBM per target.

    ``binned`` is ``(B, n_bins, zero_bin, indptr, indices, values)``; target ``t`` predicts
    ``Y[:, y_cols[t]]`` from the binned columns ``feats[feat_ptr[t]:feat_ptr[t+1]]`` and
    draws its random numbers from ``(seed, keys[t])``.  Returns the concatenated
    normalised importances (aligned with ``feats``) and the stages fitted per target.
    """
    B, nbins, zbin, indptr, indices, values = binned
    feat_ptr = np.asarray(feat_ptr, dtype=np.int64)
    feats = np.asarray(feats, dtype=np.int64)
    p_t = np.diff(feat_ptr)
    mf = np.array([_n_feat(max_features, int(p)) for p in p_t], dtype=np.int64)
    imp = np.zeros(len(feats))
    n_est = np.zeros(len(y_cols), dtype=np.int64)
    _with_threads(n_jobs, _gbm_many, B, nbins, zbin, indptr, indices, values, bool(use_sparse), Y,
                  np.asarray(y_cols, dtype=np.int64), np.asarray(keys, dtype=np.int64),
                  feat_ptr, feats, float(learning_rate), int(n_estimators), mf, int(max_depth),
                  float(subsample), int(early_stop_window or 0), int(seed), imp, n_est)
    return imp, n_est


def rf_many(binned, Y, y_cols, keys, feat_ptr, feats, *, n_estimators, max_features, seed=666,
            n_jobs=1, use_sparse=True):
    """Fit one histogram random forest per target (see :func:`gbm_many`)."""
    B, nbins, zbin, indptr, indices, values = binned
    feat_ptr = np.asarray(feat_ptr, dtype=np.int64)
    feats = np.asarray(feats, dtype=np.int64)
    p_t = np.diff(feat_ptr)
    mf = np.array([_n_feat(max_features, int(p)) for p in p_t], dtype=np.int64)
    imp = np.zeros(len(feats))
    _with_threads(n_jobs, _rf_many, B, nbins, zbin, indptr, indices, values, bool(use_sparse), Y,
                  np.asarray(y_cols, dtype=np.int64), np.asarray(keys, dtype=np.int64),
                  feat_ptr, feats, int(n_estimators), mf, int(seed), imp)
    return imp
