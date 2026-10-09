"""Bit-faithful ports of the base-R routines ArchR heatmaps depend on.

* :class:`RRandom` — R's default RNG (Mersenne-Twister, ``set.seed`` scrambling,
  ``sample.kind = "Rejection"``) so ``sample.int`` / ``kmeans`` initialisation
  draws exactly the same indices as R.
* :func:`r_dist` — ``stats::dist(method = "euclidean")`` with R's summation order.
* :func:`r_hclust` — ``stats::hclust`` (Murtagh's NN-list Fortran + ``hcass2``).
* :func:`r_kmeans` — ``stats::kmeans`` Hartigan-Wong (AS 136, R's ``kmns.f``).
* :func:`r_reorder_dendrogram` — ``stats::reorder.dendrogram`` leaf order.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:  # numba ships with scanpy; fall back to pure Python when unavailable
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------

MERSENNE_TWISTER = "Mersenne-Twister"
LECUYER_CMRG = "L'Ecuyer-CMRG"
_M1 = 4294967087
_M2 = 4294944443


class RRandom:
    """R's uniform stream after ``set.seed(seed)``.

    ``kind`` is R's ``RNGkind``: ``"Mersenne-Twister"`` (R default) or
    ``"L'Ecuyer-CMRG"`` — which ArchR switches to whenever
    ``addArchRThreads(threads > 1)`` is called.
    """

    def __init__(self, seed: int, kind: str = MERSENNE_TWISTER):
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        self.kind = kind
        if kind == LECUYER_CMRG:
            st = []
            for _ in range(6):
                s = (69069 * s + 1) & 0xFFFFFFFF
                while s >= _M2:
                    s = (69069 * s + 1) & 0xFFFFFFFF
                st.append(s)
            self._lec = st
            return
        if kind != MERSENNE_TWISTER:
            raise ValueError(f"unsupported RNG kind {kind!r}")
        key = np.empty(625, dtype=np.uint32)
        for j in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            key[j] = s
        self._bitgen = np.random.MT19937()
        # key[0] is R's dummy[0] (mti); FixupSeeds sets it to 624 on seeding.
        self._bitgen.state = {
            "bit_generator": "MT19937",
            "state": {"key": key[1:].copy(), "pos": 624},
        }

    def _lecuyer(self) -> float:
        s = self._lec
        p1 = 1403580 * s[1] - 810728 * s[0]
        p1 %= _M1
        s[0], s[1], s[2] = s[1], s[2], p1
        p2 = 527612 * s[5] - 1370589 * s[3]
        p2 %= _M2
        s[3], s[4], s[5] = s[4], s[5], p2
        return ((p1 - p2) if p1 > p2 else (p1 - p2 + _M1)) * 2.328306549295727688e-10

    def unif_rand(self) -> float:
        if self.kind == LECUYER_CMRG:
            return self._lecuyer()
        v = float(self._bitgen.random_raw()) * 2.3283064365386963e-10
        i2_32m1 = 2.328306437080797e-10
        if v <= 0.0:
            return 0.5 * i2_32m1
        if 1.0 - v <= 0.0:
            return 1.0 - 0.5 * i2_32m1
        return v

    def _rbits(self, bits: int) -> int:
        v = 0
        for _ in range(0, bits + 1, 16):
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: float) -> int:
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample_int(self, n: int, size: int) -> np.ndarray:
        """``sample.int(n, size)`` without replacement, 1-based like R."""
        if size > n:
            raise ValueError("cannot take a sample larger than the population")
        x = list(range(n))
        out = np.empty(size, dtype=np.int64)
        nn = n
        for i in range(size):
            j = self.unif_index(nn)
            out[i] = x[j] + 1
            nn -= 1
            x[j] = x[nn]
        return out

    def sample(self, x, size: int) -> np.ndarray:
        """``sample(x, size)`` incl. R's length-1 numeric quirk."""
        x = np.asarray(x)
        if x.size == 1 and np.issubdtype(x.dtype, np.number) and x.item() >= 1:
            return self.sample_int(int(x.item()), size)
        return x[self.sample_int(x.size, size) - 1]


# ---------------------------------------------------------------------------
# rowMeans / colMeans (R accumulates in long double)
# ---------------------------------------------------------------------------

def r_row_means(m: np.ndarray) -> np.ndarray:
    """``rowMeans(m)``: column-sequential long-double accumulation."""
    m = np.asarray(m, dtype=np.float64)
    acc = np.zeros(m.shape[0], dtype=np.longdouble)
    for j in range(m.shape[1]):
        acc += m[:, j]
    return (acc / m.shape[1]).astype(np.float64)


def r_col_means(m: np.ndarray) -> np.ndarray:
    """``colMeans(m)``: row-sequential long-double accumulation."""
    return r_row_means(np.asarray(m, dtype=np.float64).T)


# ---------------------------------------------------------------------------
# dist / hclust
# ---------------------------------------------------------------------------

def r_dist(X: np.ndarray) -> np.ndarray:
    """Square Euclidean distance matrix between the rows of ``X``.

    Accumulates column-by-column like R's ``R_euclidean`` (NaN pairs skipped
    with R's ``count``-rescaling), so rounding matches ``stats::dist``.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    acc = np.zeros((n, n))
    cnt = np.zeros((n, n))
    has_nan = np.isnan(X).any()
    for j in range(p):
        col = X[:, j]
        dev = col[:, None] - col[None, :]
        if has_nan:
            ok = ~np.isnan(dev)
            acc += np.where(ok, dev * dev, 0.0)
            cnt += ok
        else:
            acc += dev * dev
    if has_nan:
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.where(cnt > 0, acc * np.where(cnt != p, p / np.maximum(cnt, 1), 1.0), np.nan)
    return np.sqrt(acc)


_HCLUST_METHODS = {
    "ward.D": 1, "single": 2, "complete": 3, "average": 4,
    "mcquitty": 5, "median": 6, "centroid": 7, "ward.D2": 8,
}


@njit(cache=True)
def _hclust_fortran(n, iopt, diss_full):
    # diss stored as full symmetric matrix (0-based); semantics follow hclust.f.
    INF = 1e300
    diss = diss_full.copy()
    membr = np.ones(n)
    flag = np.ones(n, dtype=np.bool_)
    nn = np.zeros(n, dtype=np.int64)
    disnn = np.zeros(n)
    ia = np.zeros(n, dtype=np.int64)
    ib = np.zeros(n, dtype=np.int64)
    crit = np.zeros(n)
    if iopt == 8:
        diss = diss * diss
    jm = 0
    jj = 0
    im = 0
    for i in range(n - 1):
        dmin = INF
        for j in range(i + 1, n):
            if dmin > diss[i, j]:
                dmin = diss[i, j]
                jm = j
        nn[i] = jm
        disnn[i] = dmin
    ncl = n
    while ncl > 1:
        dmin = INF
        for i in range(n - 1):
            if flag[i] and disnn[i] < dmin:
                dmin = disnn[i]
                im = i
                jm = nn[i]
        ncl -= 1
        i2 = min(im, jm)
        j2 = max(im, jm)
        ia[n - ncl - 1] = i2 + 1
        ib[n - ncl - 1] = j2 + 1
        if iopt == 8:
            dmin = math.sqrt(dmin)
        crit[n - ncl - 1] = dmin
        flag[j2] = False
        dmin = INF
        for k in range(n):
            if flag[k] and k != i2:
                xx = diss[i2, j2]
                d1 = diss[i2, k]
                d2 = diss[j2, k]
                if iopt == 1 or iopt == 8:
                    v = ((membr[i2] + membr[k]) * d1 + (membr[j2] + membr[k]) * d2
                         - membr[k] * xx)
                    v = v / (membr[i2] + membr[j2] + membr[k])
                elif iopt == 2:
                    v = min(d1, d2)
                elif iopt == 3:
                    v = max(d1, d2)
                elif iopt == 4:
                    v = (membr[i2] * d1 + membr[j2] * d2) / (membr[i2] + membr[j2])
                elif iopt == 5:
                    v = (d1 + d2) / 2.0
                elif iopt == 6:
                    v = ((d1 + d2) - xx / 2.0) / 2.0
                else:
                    v = ((membr[i2] * d1 + membr[j2] * d2
                          - membr[i2] * membr[j2] * xx / (membr[i2] + membr[j2]))
                         / (membr[i2] + membr[j2]))
                diss[i2, k] = v
                diss[k, i2] = v
                if i2 < k:
                    if v < dmin:
                        dmin = v
                        jj = k
                else:
                    if v < disnn[k]:
                        disnn[k] = v
                        nn[k] = i2
        membr[i2] = membr[i2] + membr[j2]
        disnn[i2] = dmin
        nn[i2] = jj
        for i in range(n - 1):
            if flag[i] and (nn[i] == i2 or nn[i] == j2):
                dmin = INF
                for j in range(i + 1, n):
                    if flag[j]:
                        if diss[i, j] < dmin:
                            dmin = diss[i, j]
                            jj = j
                nn[i] = jj
                disnn[i] = dmin
    return ia[: n - 1], ib[: n - 1], crit[: n - 1]


def _hcass2(n, ia, ib):
    ia = list(ia)
    ib = list(ib)
    iia = list(ia)
    iib = list(ib)
    for i in range(n - 2):
        k = min(ia[i], ib[i])
        for j in range(i + 1, n - 1):
            if ia[j] == k:
                iia[j] = -(i + 1)
            if ib[j] == k:
                iib[j] = -(i + 1)
    iia = [-v for v in iia]
    iib = [-v for v in iib]
    for i in range(n - 1):
        if iia[i] > 0 and iib[i] < 0:
            iia[i], iib[i] = iib[i], iia[i]
        if iia[i] > 0 and iib[i] > 0:
            k1, k2 = min(iia[i], iib[i]), max(iia[i], iib[i])
            iia[i], iib[i] = k1, k2
    order = [iia[n - 2], iib[n - 2]]
    for i in range(n - 2, 0, -1):
        for j in range(len(order)):
            if order[j] == i:
                order[j] = iia[i - 1]
                order.insert(j + 1, iib[i - 1])
                break
    merge = np.column_stack([iia, iib]).astype(np.int64)
    return merge, np.asarray([-v for v in order], dtype=np.int64)


class RHclust:
    """Container mirroring R's ``hclust`` object (1-based ``merge``/``order``)."""

    def __init__(self, merge, height, order, method):
        self.merge = merge
        self.height = height
        self.order = order
        self.method = method

    @property
    def order0(self) -> np.ndarray:
        return self.order - 1


def r_hclust(D: np.ndarray, method: str = "complete") -> RHclust:
    """``stats::hclust`` on a square distance matrix."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if n < 2:
        raise ValueError("must have n >= 2 objects to cluster")
    if n == 2:
        return RHclust(np.array([[-1, -2]]), np.array([D[0, 1]]), np.array([1, 2]), method)
    ia, ib, crit = _hclust_fortran(n, _HCLUST_METHODS[method], D)
    merge, order = _hcass2(n, ia, ib)
    return RHclust(merge, np.asarray(crit), order, method)


def r_reorder_dendrogram(hc: RHclust, wts: np.ndarray) -> np.ndarray:
    """Leaf order (0-based) of ``reorder(as.dendrogram(hc), wts, agglo.FUN = mean)``."""
    wts = np.asarray(wts, dtype=np.float64)
    merge = hc.merge
    nodes = {}

    def node(ref):
        if ref < 0:
            leaf = -ref - 1
            return [leaf], wts[leaf]
        return nodes[ref]

    for i, (a, b) in enumerate(merge, start=1):
        la, wa = node(a)
        lb, wb = node(b)
        kids = [(la, wa), (lb, wb)]
        # as.dendrogram puts children in merge order; reorder sorts by weight
        # (stable `order`).
        kids_sorted = sorted(kids, key=lambda t: t[1])
        leaves = kids_sorted[0][0] + kids_sorted[1][0]
        nodes[i] = (leaves, float(np.mean([wa, wb])))
    return np.asarray(nodes[len(merge)][0], dtype=np.int64)


def r_dendrogram_leaves(hc: RHclust) -> np.ndarray:
    """Leaf order of ``as.dendrogram(hc)`` (equals ``hc$order``)."""
    return hc.order0


# ---------------------------------------------------------------------------
# kmeans (Hartigan-Wong)
# ---------------------------------------------------------------------------



@njit(cache=True)
def _optra(a, m, n, c, k, ic1, ic2, nc, an1, an2, ncp, d, itran, live, indx):
    big = 1.0000000150474662e30
    for l in range(k):
        if itran[l] == 1:
            live[l] = m + 1
    for i in range(m):
        ip = i + 1
        indx += 1
        l1 = ic1[i]
        if nc[l1] != 1:
            if ncp[l1] != 0:
                de = 0.0
                for j in range(n):
                    df = a[i, j] - c[l1, j]
                    de = de + df * df
                d[i] = de * an1[l1]
            da = 0.0
            for j in range(n):
                db = a[i, j] - c[ic2[i], j]
                da = da + db * db
            l2 = ic2[i]
            ll = l2
            r2 = da * an2[l2]
            for l in range(k):
                if (ip >= live[l1] and ip >= live[l]) or l == l1 or l == ll:
                    continue
                rr = r2 / an2[l]
                dc = 0.0
                skip = False
                for j in range(n):
                    dd = a[i, j] - c[l, j]
                    dc = dc + dd * dd
                    if dc >= rr:
                        skip = True
                        break
                if skip:
                    continue
                r2 = dc * an2[l]
                l2 = l
            if r2 >= d[i]:
                ic2[i] = l2
            else:
                indx = 0
                live[l1] = m + ip
                live[l2] = m + ip
                ncp[l1] = ip
                ncp[l2] = ip
                al1 = float(nc[l1])
                alw = al1 - 1.0
                al2 = float(nc[l2])
                alt = al2 + 1.0
                for j in range(n):
                    c[l1, j] = (c[l1, j] * al1 - a[i, j]) / alw
                    c[l2, j] = (c[l2, j] * al2 + a[i, j]) / alt
                nc[l1] -= 1
                nc[l2] += 1
                an2[l1] = alw / al1
                an1[l1] = big
                if alw > 1.0:
                    an1[l1] = alw / (alw - 1.0)
                an1[l2] = alt / al2
                an2[l2] = alt / (alt + 1.0)
                ic1[i] = l2
                ic2[i] = l1
        if indx == m:
            return indx
    for l in range(k):
        itran[l] = 0
        live[l] -= m
    return indx


@njit(cache=True)
def _qtran(a, m, n, c, k, ic1, ic2, nc, an1, an2, ncp, d, itran, indx, imaxqtr):
    big = 1.0000000150474662e30
    icoun = 0
    istep = 0
    while True:
        for i in range(m):
            icoun += 1
            istep += 1
            if istep >= imaxqtr:
                return indx, -1
            l1 = ic1[i]
            l2 = ic2[i]
            if nc[l1] != 1:
                if istep <= ncp[l1]:
                    da = 0.0
                    for j in range(n):
                        db = a[i, j] - c[l1, j]
                        da = da + db * db
                    d[i] = da * an1[l1]
                if not (istep >= ncp[l1] and istep >= ncp[l2]):
                    r2 = d[i] / an2[l2]
                    dd = 0.0
                    transfer = True
                    for j in range(n):
                        de = a[i, j] - c[l2, j]
                        dd = dd + de * de
                        if dd >= r2:
                            transfer = False
                            break
                    if transfer:
                        icoun = 0
                        indx = 0
                        itran[l1] = 1
                        itran[l2] = 1
                        ncp[l1] = istep + m
                        ncp[l2] = istep + m
                        al1 = float(nc[l1])
                        alw = al1 - 1.0
                        al2 = float(nc[l2])
                        alt = al2 + 1.0
                        for j in range(n):
                            c[l1, j] = (c[l1, j] * al1 - a[i, j]) / alw
                            c[l2, j] = (c[l2, j] * al2 + a[i, j]) / alt
                        nc[l1] -= 1
                        nc[l2] += 1
                        an2[l1] = alw / al1
                        an1[l1] = big
                        if alw > 1.0:
                            an1[l1] = alw / (alw - 1.0)
                        an1[l2] = alt / al2
                        an2[l2] = alt / (alt + 1.0)
                        ic1[i] = l2
                        ic2[i] = l1
            if icoun == m:
                return indx, imaxqtr


@njit(cache=True)
def _kmns(a, c, k, iter_max, imaxqtr):
    m, n = a.shape
    big = 1.0000000150474662e30
    ic1 = np.zeros(m, dtype=np.int64)
    ic2 = np.zeros(m, dtype=np.int64)
    nc = np.zeros(k, dtype=np.int64)
    an1 = np.zeros(k)
    an2 = np.zeros(k)
    ncp = np.zeros(k, dtype=np.int64)
    d = np.zeros(m)
    itran = np.zeros(k, dtype=np.int64)
    live = np.zeros(k, dtype=np.int64)
    dt = np.zeros(2)
    ifault = 0
    for i in range(m):
        ic1[i] = 0
        ic2[i] = 1
        for il in range(2):
            dt[il] = 0.0
            for j in range(n):
                da = a[i, j] - c[il, j]
                dt[il] = dt[il] + da * da
        if dt[0] > dt[1]:
            ic1[i] = 1
            ic2[i] = 0
            tmp = dt[0]
            dt[0] = dt[1]
            dt[1] = tmp
        for l in range(2, k):
            db = 0.0
            skip = False
            for j in range(n):
                dc = a[i, j] - c[l, j]
                db = db + dc * dc
                if db >= dt[1]:
                    skip = True
                    break
            if skip:
                continue
            if db < dt[0]:
                dt[1] = dt[0]
                ic2[i] = ic1[i]
                dt[0] = db
                ic1[i] = l
            else:
                dt[1] = db
                ic2[i] = l
    for l in range(k):
        nc[l] = 0
        for j in range(n):
            c[l, j] = 0.0
    for i in range(m):
        l = ic1[i]
        nc[l] += 1
        for j in range(n):
            c[l, j] = c[l, j] + a[i, j]
    for l in range(k):
        if nc[l] == 0:
            return ic1, c, nc, 1, 0
    for l in range(k):
        aa = float(nc[l])
        for j in range(n):
            c[l, j] = c[l, j] / aa
        an2[l] = aa / (aa + 1.0)
        an1[l] = big
        if aa > 1.0:
            an1[l] = aa / (aa - 1.0)
        itran[l] = 1
        ncp[l] = -1
    indx = 0
    it = 0
    converged = False
    for ij in range(1, iter_max + 1):
        it = ij
        indx = _optra(a, m, n, c, k, ic1, ic2, nc, an1, an2, ncp, d, itran, live, indx)
        if indx == m:
            converged = True
            break
        indx, imaxqtr = _qtran(a, m, n, c, k, ic1, ic2, nc, an1, an2, ncp, d, itran, indx, imaxqtr)
        if imaxqtr < 0:
            ifault = 4
            converged = True
            break
        if k == 2:
            converged = True
            break
        for l in range(k):
            ncp[l] = 0
    if not converged:
        ifault = 2
        it = iter_max + 1
    for l in range(k):
        for j in range(n):
            c[l, j] = 0.0
    for i in range(m):
        ii = ic1[i]
        for j in range(n):
            c[ii, j] = c[ii, j] + a[i, j]
    for j in range(n):
        for l in range(k):
            c[l, j] = c[l, j] / float(nc[l])
    return ic1, c, nc, ifault, it


class RKmeans:
    def __init__(self, cluster, centers, size, iter, ifault, init_centers):
        self.cluster = cluster          # 1-based like R
        self.centers = centers
        self.size = size
        self.iter = iter
        self.ifault = ifault
        self.init_centers = init_centers


def r_kmeans_init(x: np.ndarray, k: int, rng: RRandom) -> np.ndarray:
    """Initial centres exactly as ``kmeans(x, k)`` draws them (``nstart = 1``).

    ``x[sample.int(m, k), ]``; if those rows contain duplicates R redraws from
    ``unique(x)`` (rows in first-appearance order). Exact float equality is
    used for duplicate detection (R compares 15-significant-digit strings).
    """
    m = x.shape[0]
    idx = rng.sample_int(m, k)
    cen = x[idx - 1]
    if len(np.unique(cen, axis=0)) < k:
        _, first = np.unique(x, axis=0, return_index=True)
        cn = x[np.sort(first)]
        if cn.shape[0] < k:
            raise ValueError("more cluster centers than distinct data points.")
        cen = cn[rng.sample_int(cn.shape[0], k) - 1]
    return np.ascontiguousarray(cen)


def r_kmeans(
    x: np.ndarray,
    centers: int,
    *,
    iter_max: int = 10,
    rng: Optional[RRandom] = None,
    seed: Optional[int] = None,
    init_centers: Optional[np.ndarray] = None,
) -> RKmeans:
    """``stats::kmeans(x, centers, algorithm = "Hartigan-Wong", nstart = 1)``.

    Initial centres are drawn with R's ``sample.int`` from ``rng`` (or a fresh
    ``set.seed(seed)`` stream) unless ``init_centers`` is given.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    m = x.shape[0]
    k = int(centers)
    if m < k:
        raise ValueError("more cluster centers than data points")
    if init_centers is None:
        if rng is None:
            rng = RRandom(1 if seed is None else seed)
        cen = r_kmeans_init(x, k, rng)
    else:
        cen = np.asarray(init_centers, dtype=np.float64)
    c = np.ascontiguousarray(cen.copy())
    if k == 1:
        cl = np.zeros(m, dtype=np.int64)
        return RKmeans(cl + 1, x.mean(axis=0, keepdims=True), np.array([m]), 1, 0, cen)
    imaxqtr = min(2147483647, 50 * m)
    ic1, c, nc, ifault, it = _kmns(x, c, k, int(iter_max), imaxqtr)
    if ifault == 1:
        raise ValueError("empty cluster: try a better set of initial centers")
    return RKmeans(ic1 + 1, c, nc, it, ifault, cen)
