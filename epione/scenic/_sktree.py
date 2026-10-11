"""scikit-learn 1.3.2 regression trees, gradient boosting and random forests in numba.

SCENIC+ scores TF-to-gene and region-to-gene links with the feature importances of
scikit-learn ``GradientBoostingRegressor`` / ``RandomForestRegressor`` models
(through arboreto).  The importances depend on the exact tree-growing path, and that
path changed after scikit-learn 1.3.2 (the GBM initial prediction and the node
impurity accumulation differ), so calling whatever scikit-learn is installed does not
reproduce the reference SCENIC+ output.  This module reimplements the 1.3.2 path:

* ``numpy.random.RandomState`` (MT19937, legacy seeding, ``uniform`` and masked
  ``randint``) and scikit-learn's ``rand_r`` xorshift generator;
* ``DepthFirstTreeBuilder`` + ``BestSplitter`` (introsort of the feature values,
  Fisher-Yates feature draws with constant-feature bookkeeping, final partition);
* the ``MSE`` and ``FriedmanMSE`` criteria with their summation order;
* ``BaseGradientBoosting._fit_stages`` for the least-squares loss (subsample mask,
  OOB scores, arboreto's early-stopping monitor) and the bootstrap of
  ``BaseForest.fit``;
* the numpy reductions (pairwise summation) used for the feature importances.

Every floating-point operation follows the order of the Cython source, so the
importances are bit-identical to scikit-learn 1.3.2 whatever version is installed.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np

__all__ = ["gbm_importances", "rf_importances", "build_tree"]

_RAND_R_MAX = 0x7FFFFFFF
_EPSILON = float(np.finfo(np.float64).eps)
_FEATURE_THRESHOLD = np.float32(1e-7)
_INF = np.inf
_TREE_LEAF = -1
_TREE_UNDEFINED = -2
_INT32_MAX = np.iinfo(np.int32).max

_CRIT_MSE = 0
_CRIT_FRIEDMAN = 1


# ---------------------------------------------------------------------------
# numpy.random.RandomState (MT19937) and scikit-learn rand_r
# ---------------------------------------------------------------------------
@nb.njit(cache=True)
def _mt_seed(seed, key, pos):
    """``RandomState(seed)`` for an integer seed (``mt19937_seed``)."""
    s = np.uint32(seed & 0xFFFFFFFF)
    for i in range(624):
        key[i] = s
        s = np.uint32((np.uint64(1812433253) * np.uint64(s ^ (s >> np.uint32(30))) + np.uint64(i + 1))
                      & np.uint64(0xFFFFFFFF))
    pos[0] = 624


@nb.njit(cache=True)
def _mt_twist(key):
    upper = np.uint32(0x80000000)
    lower = np.uint32(0x7FFFFFFF)
    matrix_a = np.uint32(0x9908B0DF)
    one = np.uint32(1)
    for i in range(624 - 397):
        y = (key[i] & upper) | (key[i + 1] & lower)
        key[i] = key[i + 397] ^ (y >> one) ^ (matrix_a if (y & one) else np.uint32(0))
    for i in range(624 - 397, 623):
        y = (key[i] & upper) | (key[i + 1] & lower)
        key[i] = key[i + (397 - 624)] ^ (y >> one) ^ (matrix_a if (y & one) else np.uint32(0))
    y = (key[623] & upper) | (key[0] & lower)
    key[623] = key[396] ^ (y >> one) ^ (matrix_a if (y & one) else np.uint32(0))


@nb.njit(cache=True)
def _mt_next32(key, pos):
    if pos[0] == 624:
        _mt_twist(key)
        pos[0] = 0
    y = key[pos[0]]
    pos[0] += 1
    y ^= y >> np.uint32(11)
    y ^= (y << np.uint32(7)) & np.uint32(0x9D2C5680)
    y ^= (y << np.uint32(15)) & np.uint32(0xEFC60000)
    y ^= y >> np.uint32(18)
    return y


@nb.njit(cache=True)
def _mt_double(key, pos):
    a = _mt_next32(key, pos) >> np.uint32(5)
    b = _mt_next32(key, pos) >> np.uint32(6)
    return (a * 67108864.0 + b) / 9007199254740992.0


@nb.njit(cache=True)
def _gen_mask(rng):
    mask = rng
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16
    mask |= mask >> 32
    return mask


@nb.njit(cache=True)
def _mt_bounded(key, pos, rng):
    """Value in ``[0, rng]`` as legacy ``randint`` draws it (masked rejection, rng < 2**32)."""
    if rng == 0:
        return 0
    mask = np.uint32(_gen_mask(np.int64(rng)))
    r = np.uint32(rng)
    while True:
        val = _mt_next32(key, pos) & mask
        if val <= r:
            return np.int64(val)


@nb.njit(cache=True)
def _rand_r(state):
    """scikit-learn ``our_rand_r`` (state is a 1-element uint32 array)."""
    if state[0] == 0:
        state[0] = np.uint32(1)
    s = state[0]
    s ^= np.uint32(s << np.uint32(13))
    s ^= np.uint32(s >> np.uint32(17))
    s ^= np.uint32(s << np.uint32(5))
    state[0] = s
    return np.int64(s % np.uint32(0x80000000))


@nb.njit(cache=True)
def _rand_int(low, high, state):
    return low + _rand_r(state) % (high - low)


# ---------------------------------------------------------------------------
# numpy pairwise summation (``np.add.reduce`` on a contiguous float64 vector)
# ---------------------------------------------------------------------------
@nb.njit(cache=True)
def _pairwise_leaf(a, lo, n):
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[lo + i]
        return res
    else:
        r0 = a[lo]
        r1 = a[lo + 1]
        r2 = a[lo + 2]
        r3 = a[lo + 3]
        r4 = a[lo + 4]
        r5 = a[lo + 5]
        r6 = a[lo + 6]
        r7 = a[lo + 7]
        i = 8
        m = n - (n % 8)
        while i < m:
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


@nb.njit(cache=True)
def _pairwise(a, lo, n):
    """numpy ``pairwise_sum`` (blocks of <= 128, 8 accumulators) with the recursion
    replaced by an explicit post-order stack (numba cannot cache recursive functions)."""
    if n <= 128:
        return _pairwise_leaf(a, lo, n)
    st_lo = np.empty(64, dtype=np.int64)
    st_n = np.empty(64, dtype=np.int64)
    st_s = np.empty(64, dtype=np.int64)
    vals = np.empty(64, dtype=np.float64)
    st_lo[0] = lo
    st_n[0] = n
    st_s[0] = 0
    sp = 1
    vp = 0
    while sp > 0:
        top = sp - 1
        flo = st_lo[top]
        fn = st_n[top]
        if fn <= 128:
            vals[vp] = _pairwise_leaf(a, flo, fn)
            vp += 1
            sp -= 1
            continue
        n2 = fn // 2
        n2 -= n2 % 8
        if st_s[top] == 0:
            st_s[top] = 1
            st_lo[sp] = flo
            st_n[sp] = n2
            st_s[sp] = 0
            sp += 1
        elif st_s[top] == 1:
            st_s[top] = 2
            st_lo[sp] = flo + n2
            st_n[sp] = fn - n2
            st_s[sp] = 0
            sp += 1
        else:
            vals[vp - 2] = vals[vp - 2] + vals[vp - 1]
            vp -= 1
            sp -= 1
    return vals[0]


@nb.njit(cache=True)
def _np_sum(a, n):
    """``np.sum(a[:n])`` bit for bit."""
    return 0.0 + _pairwise(a, 0, n)


# ---------------------------------------------------------------------------
# Introsort of (feature_values, samples) as in sklearn/tree/_splitter.pyx
# ---------------------------------------------------------------------------
@nb.njit(cache=True, inline="always")
def _swap(fv, samples, i, j):
    t = fv[i]
    fv[i] = fv[j]
    fv[j] = t
    s = samples[i]
    samples[i] = samples[j]
    samples[j] = s


@nb.njit(cache=True)
def _median3(fv, o, n):
    a = fv[o]
    b = fv[o + n // 2]
    c = fv[o + n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        return a
    elif b < c:
        if a < c:
            return a
        return c
    return b


@nb.njit(cache=True)
def _sift_down(fv, samples, o, start, end):
    root = start
    while True:
        child = root * 2 + 1
        maxind = root
        if child < end and fv[o + maxind] < fv[o + child]:
            maxind = child
        if child + 1 < end and fv[o + maxind] < fv[o + child + 1]:
            maxind = child + 1
        if maxind == root:
            break
        _swap(fv, samples, o + root, o + maxind)
        root = maxind


@nb.njit(cache=True)
def _heapsort(fv, samples, o, n):
    start = (n - 2) // 2
    end = n
    while True:
        _sift_down(fv, samples, o, start, end)
        if start == 0:
            break
        start -= 1
    end = n - 1
    while end > 0:
        _swap(fv, samples, o, o + end)
        _sift_down(fv, samples, o, 0, end)
        end -= 1


@nb.njit(cache=True)
def _sort(fv, samples, start, n, stack):
    """``sort(&fv[start], &samples[start], n)``; recursion replaced by a stack of disjoint ranges."""
    if n == 0:
        return
    maxd = 2 * int(math.log(float(n)) / math.log(2.0))
    sp = 0
    stack[0, 0] = start
    stack[0, 1] = n
    stack[0, 2] = maxd
    sp = 1
    while sp > 0:
        sp -= 1
        o = stack[sp, 0]
        n = stack[sp, 1]
        maxd = stack[sp, 2]
        while n > 1:
            if maxd <= 0:
                _heapsort(fv, samples, o, n)
                break
            maxd -= 1
            pivot = _median3(fv, o, n)
            i = 0
            lo = 0
            r = n
            while i < r:
                v = fv[o + i]
                if v < pivot:
                    if i != lo:
                        _swap(fv, samples, o + i, o + lo)
                    i += 1
                    lo += 1
                elif v > pivot:
                    r -= 1
                    _swap(fv, samples, o + i, o + r)
                else:
                    i += 1
            if lo > 1:
                stack[sp, 0] = o
                stack[sp, 1] = lo
                stack[sp, 2] = maxd
                sp += 1
            o += r
            n -= r


# ---------------------------------------------------------------------------
# One regression tree (DepthFirstTreeBuilder + BestSplitter + MSE/FriedmanMSE)
# ---------------------------------------------------------------------------
@nb.njit(cache=True)
def build_tree(X, cols, y, sw, rand_state, max_features, max_depth, criterion,
               samples, features, const_features, fv, sort_stack, node_stack,
               left, right, feat, thr, imp, nns, wnns, val):
    """Grow one tree; returns the node count.

    ``X`` is float32 (n_samples, n_features), any layout; ``y`` and ``sw`` float64.
    Node arrays (``left`` … ``val``) must hold at least ``2 * n_samples + 1`` entries.
    """
    n_total = X.shape[0]
    n_features = cols.shape[0]
    # Splitter.init
    j = 0
    weighted_n_samples = 0.0
    for i in range(n_total):
        if sw[i] != 0.0:
            samples[j] = i
            j += 1
        weighted_n_samples += sw[i]
    n_samples = j
    for f in range(n_features):
        features[f] = f

    node_count = 0
    sp = 0
    # stack records: start, end, depth, parent, is_left, n_constant ; impurity separately
    node_stack[0, 0] = 0
    node_stack[0, 1] = n_samples
    node_stack[0, 2] = 0
    node_stack[0, 3] = _TREE_UNDEFINED
    node_stack[0, 4] = 0
    node_stack[0, 5] = 0
    stack_imp = np.empty(node_stack.shape[0], dtype=np.float64)
    stack_imp[0] = _INF
    sp = 1
    first = True
    while sp > 0:
        sp -= 1
        start = node_stack[sp, 0]
        end = node_stack[sp, 1]
        depth = node_stack[sp, 2]
        parent = node_stack[sp, 3]
        is_left = node_stack[sp, 4]
        n_constant = node_stack[sp, 5]
        impurity = stack_imp[sp]
        n_node = end - start

        # criterion.init (node_reset)
        sum_total = 0.0
        sq_sum_total = 0.0
        w_node = 0.0
        for p in range(start, end):
            i = samples[p]
            w = sw[i]
            wy = w * y[i]
            sum_total += wy
            sq_sum_total += wy * y[i]
            w_node += w

        is_leaf = (depth >= max_depth or n_node < 2 or n_node < 2 or w_node < 0.0)
        if first:
            impurity = sq_sum_total / w_node
            impurity -= (sum_total / w_node) ** 2.0
            impurity = impurity / 1.0
            first = False
        is_leaf = is_leaf or impurity <= _EPSILON

        b_pos = end
        b_feature = 0
        b_thr = 0.0
        b_imp_left = _INF
        b_imp_right = _INF
        b_improvement = -_INF
        if not is_leaf:
            # ---------------- node_split_best ----------------
            best_proxy = -_INF
            f_i = n_features
            n_visited = 0
            n_found_constants = 0
            n_drawn_constants = 0
            n_known_constants = n_constant
            n_total_constants = n_known_constants
            while f_i > n_total_constants and (n_visited < max_features
                                               or n_visited <= n_found_constants + n_drawn_constants):
                n_visited += 1
                f_j = _rand_int(n_drawn_constants, f_i - n_found_constants, rand_state)
                if f_j < n_known_constants:
                    t = features[n_drawn_constants]
                    features[n_drawn_constants] = features[f_j]
                    features[f_j] = t
                    n_drawn_constants += 1
                    continue
                f_j += n_found_constants
                cur_f = features[f_j]
                c = cols[cur_f]
                v0 = X[samples[start], c]
                all_equal = True
                for p in range(start, end):
                    v = X[samples[p], c]
                    fv[p] = v
                    if v != v0:
                        all_equal = False
                # introsort leaves an all-equal range untouched (no element compares < or >
                # the pivot), so skipping it is exact; sparse TFs are often all-zero in a node
                if not all_equal:
                    _sort(fv, samples, start, end - start, sort_stack)
                if fv[end - 1] <= fv[start] + _FEATURE_THRESHOLD:
                    t = features[f_j]
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = t
                    n_found_constants += 1
                    n_total_constants += 1
                    continue
                f_i -= 1
                t = features[f_i]
                features[f_i] = features[f_j]
                features[f_j] = t
                # criterion.reset
                pos = start
                sum_left = 0.0
                sum_right = sum_total
                w_left = 0.0
                w_right = w_node
                p = start
                while p < end:
                    while p + 1 < end and fv[p + 1] <= fv[p] + _FEATURE_THRESHOLD:
                        p += 1
                    p_prev = p
                    p += 1
                    if p >= end:
                        continue
                    # min_samples_leaf = 1 always satisfied for p in (start, end)
                    new_pos = p
                    # criterion.update(new_pos)
                    if (new_pos - pos) <= (end - new_pos):
                        for q in range(pos, new_pos):
                            i = samples[q]
                            w = sw[i]
                            sum_left += w * y[i]
                            w_left += w
                    else:
                        sum_right = 0.0
                        sum_left = sum_total
                        w_right = 0.0
                        w_left = w_node
                        for q in range(end - 1, new_pos - 1, -1):
                            i = samples[q]
                            w = sw[i]
                            sum_left -= w * y[i]
                            w_left -= w
                    w_right = w_node - w_left
                    sum_right = sum_total - sum_left
                    pos = new_pos
                    # min_weight_leaf = 0
                    if w_left < 0.0 or w_right < 0.0:
                        continue
                    if criterion == _CRIT_FRIEDMAN:
                        tsl = 0.0 + sum_left
                        tsr = 0.0 + sum_right
                        diff = w_right * tsl - w_left * tsr
                        proxy = diff * diff / (w_left * w_right)
                    else:
                        pl = 0.0 + sum_left * sum_left
                        pr = 0.0 + sum_right * sum_right
                        proxy = pl / w_left + pr / w_right
                    if proxy > best_proxy:
                        best_proxy = proxy
                        th = fv[p_prev] / 2.0 + fv[p] / 2.0
                        if th == fv[p] or th == _INF or th == -_INF:
                            th = np.float64(fv[p_prev])
                        b_pos = p
                        b_feature = cur_f
                        b_thr = th
            if b_pos < end:
                # partition_samples_final (no missing values)
                p = start
                partition_end = end - 1
                while p < partition_end:
                    if X[samples[p], cols[b_feature]] <= b_thr:
                        p += 1
                    else:
                        t = samples[p]
                        samples[p] = samples[partition_end]
                        samples[partition_end] = t
                        partition_end -= 1
                # reset + update(best.pos)
                pos = start
                sum_left = 0.0
                sum_right = sum_total
                w_left = 0.0
                w_right = w_node
                new_pos = b_pos
                if (new_pos - pos) <= (end - new_pos):
                    for q in range(pos, new_pos):
                        i = samples[q]
                        w = sw[i]
                        sum_left += w * y[i]
                        w_left += w
                else:
                    sum_right = 0.0
                    sum_left = sum_total
                    w_right = 0.0
                    w_left = w_node
                    for q in range(end - 1, new_pos - 1, -1):
                        i = samples[q]
                        w = sw[i]
                        sum_left -= w * y[i]
                        w_left -= w
                w_right = w_node - w_left
                sum_right = sum_total - sum_left
                pos = new_pos
                # children_impurity
                sq_left = 0.0
                for q in range(start, pos):
                    i = samples[q]
                    w = sw[i]
                    yi = y[i]
                    sq_left += w * yi * yi
                sq_right = sq_sum_total - sq_left
                il = sq_left / w_left
                ir = sq_right / w_right
                il -= (sum_left / w_left) ** 2.0
                ir -= (sum_right / w_right) ** 2.0
                il = il / 1.0
                ir = ir / 1.0
                b_imp_left = il
                b_imp_right = ir
                if criterion == _CRIT_FRIEDMAN:
                    tsl = 0.0 + sum_left
                    tsr = 0.0 + sum_right
                    diff = (w_right * tsl - w_left * tsr) / 1.0
                    b_improvement = diff * diff / (w_left * w_right * w_node)
                else:
                    b_improvement = ((w_node / weighted_n_samples)
                                     * (impurity - (w_right / w_node * ir) - (w_left / w_node * il)))
            # constant feature bookkeeping
            for q in range(n_known_constants):
                features[q] = const_features[q]
            for q in range(n_found_constants):
                const_features[n_known_constants + q] = features[n_known_constants + q]
            n_constant = n_total_constants
            is_leaf = b_pos >= end or (b_improvement + _EPSILON < 0.0)

        node_id = node_count
        node_count += 1
        if parent != _TREE_UNDEFINED:
            if is_left:
                left[parent] = node_id
            else:
                right[parent] = node_id
        if is_leaf:
            left[node_id] = _TREE_LEAF
            right[node_id] = _TREE_LEAF
            feat[node_id] = _TREE_UNDEFINED
            thr[node_id] = _TREE_UNDEFINED
        else:
            feat[node_id] = b_feature
            thr[node_id] = b_thr
        imp[node_id] = impurity
        nns[node_id] = n_node
        wnns[node_id] = w_node
        val[node_id] = sum_total / w_node
        if not is_leaf:
            node_stack[sp, 0] = b_pos
            node_stack[sp, 1] = end
            node_stack[sp, 2] = depth + 1
            node_stack[sp, 3] = node_id
            node_stack[sp, 4] = 0
            node_stack[sp, 5] = n_constant
            stack_imp[sp] = b_imp_right
            sp += 1
            node_stack[sp, 0] = start
            node_stack[sp, 1] = b_pos
            node_stack[sp, 2] = depth + 1
            node_stack[sp, 3] = node_id
            node_stack[sp, 4] = 1
            node_stack[sp, 5] = n_constant
            stack_imp[sp] = b_imp_left
            sp += 1
    return node_count


@nb.njit(cache=True)
def _tree_importances(node_count, left, right, feat, imp, wnns, out):
    """``Tree.compute_feature_importances(normalize=False)`` into ``out`` (zeroed here)."""
    for f in range(out.shape[0]):
        out[f] = 0.0
    for nid in range(node_count):
        lc = left[nid]
        if lc != _TREE_LEAF:
            rc = right[nid]
            out[feat[nid]] += (wnns[nid] * imp[nid]
                               - wnns[lc] * imp[lc]
                               - wnns[rc] * imp[rc])
    root = wnns[0]
    for f in range(out.shape[0]):
        out[f] /= root


@nb.njit(cache=True)
def _leaf_value(X, cols, i, left, right, feat, thr, val):
    nid = 0
    while left[nid] != _TREE_LEAF:
        if X[i, cols[feat[nid]]] <= thr[nid]:
            nid = left[nid]
        else:
            nid = right[nid]
    return val[nid]


@nb.njit(cache=True)
def _ls_loss(y, raw, mask, want, buf):
    """``LeastSquaresError.__call__(y[m], raw[m], ones[m])`` with m = (mask == want)."""
    m = 0
    for i in range(y.shape[0]):
        if mask[i] == want:
            d = np.float64(y[i]) - raw[i]
            buf[m] = 1.0 * (d * d)
            m += 1
    ones = 0.0
    # ``sample_weight.sum()`` of m ones is exactly m
    ones = float(m)
    return 1.0 / ones * _np_sum(buf, m)


@nb.njit(cache=True, nogil=True)
def _gbm_fit(X, cols, y, init_raw, learning_rate, n_estimators, max_features, max_depth,
             subsample, n_inbag, early_stop_window, seed):
    """Least-squares GBM as ``GradientBoostingRegressor.fit`` (1.3.2) + arboreto monitor.

    Returns (sum of per-tree unnormalised importances over relevant trees,
    number of relevant trees, number of stages fitted).
    """
    n = X.shape[0]
    p = cols.shape[0]
    key = np.empty(624, dtype=np.uint32)
    pos = np.zeros(1, dtype=np.int64)
    _mt_seed(seed, key, pos)
    raw = np.full(n, init_raw)
    resid = np.empty(n, dtype=np.float64)
    sw = np.ones(n, dtype=np.float64)
    mask = np.ones(n, dtype=np.bool_)
    buf = np.empty(n, dtype=np.float64)
    rand = np.empty(n, dtype=np.float64)
    cap = 2 * n + 1
    samples = np.empty(n, dtype=np.int64)
    features = np.empty(p, dtype=np.int64)
    const_features = np.empty(p, dtype=np.int64)
    fv = np.empty(n, dtype=np.float32)
    sort_stack = np.empty((256, 3), dtype=np.int64)
    node_stack = np.empty((cap, 6), dtype=np.int64)
    left = np.empty(cap, dtype=np.int64)
    right = np.empty(cap, dtype=np.int64)
    feat = np.empty(cap, dtype=np.int64)
    thr = np.empty(cap, dtype=np.float64)
    imp = np.empty(cap, dtype=np.float64)
    nns = np.empty(cap, dtype=np.int64)
    wnns = np.empty(cap, dtype=np.float64)
    val = np.empty(cap, dtype=np.float64)
    tree_imp = np.empty(p, dtype=np.float64)
    acc = np.zeros(p, dtype=np.float64)
    hist = np.empty(n_estimators, dtype=np.float64)
    n_relevant = 0
    oob_scores = np.empty(n_estimators, dtype=np.float64)
    oob_impr = np.empty(n_estimators, dtype=np.float64)
    rand_state = np.zeros(1, dtype=np.uint32)
    do_oob = subsample < 1.0
    initial_loss = 0.0
    i = 0
    for i in range(n_estimators):
        if do_oob:
            for k in range(n):
                rand[k] = _mt_double(key, pos)
            n_bagged = 0
            for k in range(n):
                if rand[k] * (n - k) < (n_inbag - n_bagged):
                    mask[k] = True
                    n_bagged += 1
                else:
                    mask[k] = False
            if i == 0:
                initial_loss = _ls_loss(y, raw, mask, False, buf)
        for k in range(n):
            resid[k] = np.float64(y[k]) - raw[k]
            sw[k] = 1.0 * (1.0 if mask[k] else 0.0)
        rand_state[0] = np.uint32(_mt_bounded(key, pos, _RAND_R_MAX - 1))
        nc = build_tree(X, cols, resid, sw, rand_state, max_features, max_depth, _CRIT_FRIEDMAN,
                        samples, features, const_features, fv, sort_stack, node_stack,
                        left, right, feat, thr, imp, nns, wnns, val)
        for k in range(n):
            raw[k] += learning_rate * _leaf_value(X, cols, k, left, right, feat, thr, val)
        if nc > 1:
            _tree_importances(nc, left, right, feat, imp, wnns, tree_imp)
            if n_relevant == 0:
                for f in range(p):
                    acc[f] = tree_imp[f]
            else:
                for f in range(p):
                    acc[f] += tree_imp[f]
            hist[n_relevant] = tree_imp[0]
            n_relevant += 1
        if do_oob:
            oob_scores[i] = _ls_loss(y, raw, mask, False, buf)
            prev = initial_loss if i == 0 else oob_scores[i - 1]
            oob_impr[i] = prev - oob_scores[i]
            if early_stop_window > 0 and i >= early_stop_window - 1:
                lo = max(0, i - early_stop_window + 1)
                m = i + 1 - lo
                mean = _pairwise(oob_impr, lo, m) / m
                if mean < 0:
                    break
    if p == 1 and n_relevant > 0:
        # a (k, 1) stack is reduced pairwise by np.mean(axis=0)
        acc[0] = _np_sum(hist, n_relevant)
    return acc, n_relevant, i + 1


@nb.njit(cache=True, nogil=True)
def _rf_fit(X, cols, y, n_estimators, max_features, seed, out_acc):
    """``RandomForestRegressor`` (1.3.2, bootstrap, squared error); returns the sum of
    normalised per-tree importances over relevant trees and their number."""
    n = X.shape[0]
    p = cols.shape[0]
    key = np.empty(624, dtype=np.uint32)
    pos = np.zeros(1, dtype=np.int64)
    _mt_seed(seed, key, pos)
    seeds = np.empty(n_estimators, dtype=np.int64)
    for t in range(n_estimators):
        seeds[t] = _mt_bounded(key, pos, _INT32_MAX - 1)
    cap = 2 * n + 1
    sw = np.empty(n, dtype=np.float64)
    counts = np.empty(n, dtype=np.int64)
    samples = np.empty(n, dtype=np.int64)
    features = np.empty(p, dtype=np.int64)
    const_features = np.empty(p, dtype=np.int64)
    fv = np.empty(n, dtype=np.float32)
    sort_stack = np.empty((256, 3), dtype=np.int64)
    node_stack = np.empty((cap, 6), dtype=np.int64)
    left = np.empty(cap, dtype=np.int64)
    right = np.empty(cap, dtype=np.int64)
    feat = np.empty(cap, dtype=np.int64)
    thr = np.empty(cap, dtype=np.float64)
    imp = np.empty(cap, dtype=np.float64)
    nns = np.empty(cap, dtype=np.int64)
    wnns = np.empty(cap, dtype=np.float64)
    val = np.empty(cap, dtype=np.float64)
    tree_imp = np.empty(p, dtype=np.float64)
    hist = np.empty(n_estimators, dtype=np.float64)
    rand_state = np.zeros(1, dtype=np.uint32)
    n_relevant = 0
    for t in range(n_estimators):
        # bootstrap: RandomState(seed_t).randint(0, n, n)
        _mt_seed(seeds[t], key, pos)
        for k in range(n):
            counts[k] = 0
        for k in range(n):
            counts[_mt_bounded(key, pos, n - 1)] += 1
        for k in range(n):
            sw[k] = 1.0 * counts[k]
        # splitter: RandomState(seed_t).randint(0, RAND_R_MAX)
        _mt_seed(seeds[t], key, pos)
        rand_state[0] = np.uint32(_mt_bounded(key, pos, _RAND_R_MAX - 1))
        nc = build_tree(X, cols, y, sw, rand_state, max_features, _INT32_MAX, _CRIT_MSE,
                        samples, features, const_features, fv, sort_stack, node_stack,
                        left, right, feat, thr, imp, nns, wnns, val)
        if nc > 1:
            _tree_importances(nc, left, right, feat, imp, wnns, tree_imp)
            norm = _np_sum(tree_imp, p)
            if norm > 0.0:
                for f in range(p):
                    tree_imp[f] /= norm
            if n_relevant == 0:
                for f in range(p):
                    out_acc[f] = tree_imp[f]
            else:
                for f in range(p):
                    out_acc[f] += tree_imp[f]
            hist[n_relevant] = tree_imp[0]
            n_relevant += 1
    if p == 1 and n_relevant > 0:
        out_acc[0] = _np_sum(hist, n_relevant)
    return n_relevant


def _float32_mean(y):
    """``DummyRegressor(strategy="mean").fit(X, y).constant_`` for a float32 target."""
    return float(np.average(np.asarray(y).reshape(-1, 1), axis=0)[0])


def _n_features(max_features, p):
    if max_features == "sqrt":
        return max(1, int(np.sqrt(p)))
    if isinstance(max_features, float):
        return max(1, int(max_features * p))
    return int(max_features)


def gbm_importances(X, y, *, cols=None, learning_rate=0.01, n_estimators=500, max_features=0.1,
                    subsample=1.0, max_depth=3, seed=666, early_stop_window=None):
    """``GradientBoostingRegressor(...).fit(X[:, cols], y).feature_importances_`` (scikit-learn 1.3.2).

    ``X`` must already be float32 (any memory layout; Fortran order is fastest).
    ``cols`` selects the predictor columns without copying (default: all).
    ``early_stop_window`` enables arboreto's OOB monitor (only with ``subsample < 1``).
    Returns ``(feature_importances_, n_estimators_fitted)``.
    """
    X = np.asarray(X)
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    y = np.asarray(y, dtype=np.float32)
    n = X.shape[0]
    cols = np.arange(X.shape[1], dtype=np.int64) if cols is None else np.asarray(cols, dtype=np.int64)
    p = cols.shape[0]
    if not y.any():
        # zero target: zero residuals, single-node trees, OOB improvement 0 (monitor never stops)
        return np.zeros(p, dtype=np.float64), int(n_estimators)
    mf = _n_features(max_features, p)
    n_inbag = max(1, int(subsample * n))
    init_raw = float(np.float64(_float32_mean(y)))
    acc, n_rel, n_est = _gbm_fit(X, cols, y, init_raw, float(learning_rate), int(n_estimators), mf,
                                 int(max_depth), float(subsample), n_inbag,
                                 int(early_stop_window or 0), int(seed))
    if n_rel == 0:
        return np.zeros(p, dtype=np.float64), int(n_est)
    avg = acc / n_rel
    return avg / np.sum(avg), int(n_est)


def rf_importances(X, y, *, cols=None, n_estimators=1000, max_features="sqrt", seed=666):
    """``RandomForestRegressor(...).fit(X[:, cols], y).feature_importances_`` (scikit-learn 1.3.2)."""
    X = np.asarray(X)
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    y = np.asarray(y, dtype=np.float64)
    cols = np.arange(X.shape[1], dtype=np.int64) if cols is None else np.asarray(cols, dtype=np.int64)
    p = cols.shape[0]
    if not y.any():
        return np.zeros(p, dtype=np.float64)
    acc = np.zeros(p, dtype=np.float64)
    n_rel = _rf_fit(X, cols, y, int(n_estimators), _n_features(max_features, p), int(seed), acc)
    if n_rel == 0:
        return np.zeros(p, dtype=np.float64)
    avg = acc / n_rel
    return avg / np.sum(avg)
