# SPDX-License-Identifier: GPL-2.0-or-later
# Translation of parts of the igraph C core 0.10 (as vendored in R igraph 2.1.4), (c) The igraph
# development team; igraph_qsort from FreeBSD qsort.c, BSD-3-Clause.
"""Louvain community detection exactly as R ``igraph::cluster_louvain`` runs it.

Port of ``igraph_community_multilevel`` (igraph C core 0.10, vendored in the R
package igraph 2.1.4: ``src/vendor/cigraph/src/community/louvain.c``) together
with the pieces its result depends on:

* R igraph's RNG bridge (``rrandom.c``: 30-bit draws ``unif_rand() * 2^30``),
  igraph's bounded integers (``random.c``: 32-bit Lemire rejection) and
  ``igraph_vector_int_shuffle``;
* igraph's edge storage for undirected graphs (``from = max``, ``to = min``)
  and the order of ``igraph_incident(..., IGRAPH_ALL)``;
* ``igraph_qsort`` (FreeBSD qsort), whose order of equal keys decides the
  order in which a vertex's link weights are summed;
* ``graph_from_adjacency_matrix(mode = "undirected", weighted = TRUE)`` on
  Cicero's ``make_sparse_matrix`` (zero-weight self loops included).

With an :class:`epione.tl._rrng.RRandom` stream, CCAN memberships equal R's
draw for draw (GPL-2.0-or-later, as igraph).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

_TWO30 = 1073741824.0  # 0x40000000


class IgraphRRng:
    """igraph's RNG interface on top of R's ``unif_rand()`` stream."""

    def __init__(self, rrandom):
        self.r = rrandom

    def _get(self) -> int:                      # igraph_rng_R_get: bits = 30
        return int(self.r.unif_rand(1)[0] * _TWO30)

    def _uint32(self) -> int:                   # igraph_i_rng_get_random_bits(rng, 32)
        g1 = self._get()
        g2 = self._get()
        return ((g1 << 2) + (g2 >> 28)) & 0xFFFFFFFF

    def _bounded(self, rng: int) -> int:        # igraph_i_rng_get_uint32_bounded (Lemire)
        t = ((1 << 32) - rng) % rng
        while True:
            x = self._uint32()
            m = x * rng
            if (m & 0xFFFFFFFF) >= t:
                return m >> 32

    def integer(self, lo: int, hi: int) -> int:  # RNG_INTEGER(lo, hi)
        if hi == lo:
            return lo
        return lo + self._bounded(hi - lo + 1)


# ------------------------------------------------------------------ igraph_qsort (FreeBSD)
def _qsort(a: list, lo: int, n: int, key) -> None:
    """In-place port of igraph's ``local_qsort`` on ``a[lo:lo+n]`` with
    comparator ``key(x) - key(y)``; reproduces the order of equal keys."""
    def cmp(i, j):
        ki, kj = key(a[i]), key(a[j])
        return -1 if ki < kj else (1 if ki > kj else 0)

    def swap(i, j):
        a[i], a[j] = a[j], a[i]

    def vecswap(i, j, cnt):
        for t in range(cnt):
            swap(i + t, j + t)

    def med3(x, y, z):
        if cmp(x, y) < 0:
            return y if cmp(y, z) < 0 else (z if cmp(x, z) < 0 else x)
        return y if cmp(y, z) > 0 else (x if cmp(x, z) < 0 else z)

    def insertion(base, cnt):
        for pm in range(base + 1, base + cnt):
            pl = pm
            while pl > base and cmp(pl - 1, pl) > 0:
                swap(pl, pl - 1)
                pl -= 1

    if n < 2:
        return
    while True:
        swap_cnt = 0
        if n < 7:
            insertion(lo, n)
            return
        pm = lo + n // 2
        if n > 7:
            pl = lo
            pn = lo + n - 1
            if n > 40:
                d = n // 8
                pl = med3(pl, pl + d, pl + 2 * d)
                pm = med3(pm - d, pm, pm + d)
                pn = med3(pn - 2 * d, pn - d, pn)
            pm = med3(pl, pm, pn)
        swap(lo, pm)
        pa = pb = lo + 1
        pc = pd = lo + n - 1
        while True:
            while pb <= pc:
                r = cmp(pb, lo)
                if r > 0:
                    break
                if r == 0:
                    swap_cnt = 1
                    swap(pa, pb)
                    pa += 1
                pb += 1
            while pb <= pc:
                r = cmp(pc, lo)
                if r < 0:
                    break
                if r == 0:
                    swap_cnt = 1
                    swap(pc, pd)
                    pd -= 1
                pc -= 1
            if pb > pc:
                break
            swap(pb, pc)
            swap_cnt = 1
            pb += 1
            pc -= 1
        if swap_cnt == 0:
            insertion(lo, n)
            return
        pn = lo + n
        d1 = min(pa - lo, pb - pa)
        if d1 > 0:
            vecswap(lo, pb - d1, d1)
        d1 = min(pd - pc, pn - pd - 1)
        if d1 > 0:
            vecswap(pb, pn - d1, d1)
        d1 = pb - pa
        d2 = pd - pc
        if d1 <= d2:
            if d1 > 1:
                _qsort(a, lo, d1, key)
            if d2 > 1:
                lo, n = pn - d2, d2
                continue
            return
        else:
            if d2 > 1:
                _qsort(a, pn - d2, d2, key)
            if d1 > 1:
                n = d1
                continue
            return


# ------------------------------------------------------------------ graph
class _Graph:
    """Undirected igraph-style edge list: from = max, to = min (igraph_add_edges)."""

    def __init__(self, n: int, edges: Sequence[Tuple[int, int]]):
        self.n = n
        e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        self.frm = np.maximum(e[:, 0], e[:, 1])
        self.to = np.minimum(e[:, 0], e[:, 1])
        m = len(self.frm)
        eid = np.arange(m)
        # oi: edges ordered by (from, to), stable; ii: by (to, from), stable (igraph_vector_int_pair_order)
        oi = np.lexsort((eid, self.to, self.frm))
        ii = np.lexsort((eid, self.frm, self.to))
        self.out_list: List[np.ndarray] = [None] * n
        self.in_list: List[np.ndarray] = [None] * n
        fo = self.frm[oi]
        ti = self.to[ii]
        so = np.searchsorted(fo, np.arange(n + 1), side="left")
        si = np.searchsorted(ti, np.arange(n + 1), side="left")
        for v in range(n):
            self.out_list[v] = oi[so[v]:so[v + 1]]
            self.in_list[v] = ii[si[v]:si[v + 1]]

    def incident(self, v: int) -> List[int]:
        # igraph_i_incident for undirected graphs: out part, then in part (loops twice)
        return list(self.out_list[v]) + list(self.in_list[v])

    def other(self, e: int, v: int) -> int:
        return int(self.to[e]) if self.frm[e] == v else int(self.frm[e])


def _seq_sum(x) -> float:
    s = 0.0
    for v in x:
        s += v
    return s


def _multilevel_step(g: _Graph, weights: List[float], rng: IgraphRRng, resolution: float):
    vcount, ecount = g.n, len(g.frm)
    node_order = list(range(vcount))
    n = vcount
    while n > 1:                                         # igraph_vector_int_shuffle
        k = rng.integer(0, n - 1)
        n -= 1
        node_order[n], node_order[k] = node_order[k], node_order[n]
    membership = list(range(vcount))
    size = [1] * vcount
    w_in = [0.0] * vcount
    w_all = [0.0] * vcount
    communities_no = vcount
    weight_sum = 2.0 * _seq_sum(weights)
    frm, to = g.frm, g.to
    for i in range(ecount):
        f, t, w = int(frm[i]), int(to[i]), weights[i]
        w_all[f] += w
        w_all[t] += w
        if f == t:
            w_in[f] += 2 * w

    def modularity():
        res = 0.0
        m = weight_sum
        for i in range(vcount):
            if size[i] > 0:
                res += (w_in[i] - resolution * w_all[i] * w_all[i] / m) / m
        return res

    incident = [g.incident(v) for v in range(vcount)]
    q = modularity()
    while True:
        temp_communities_no = communities_no
        pass_q = q
        changed = False
        temp_membership = list(membership)
        if vcount > 1:
            i1 = rng.integer(0, vcount - 1)
            i2 = rng.integer(0, vcount - 1)
            node_order[i1], node_order[i2] = node_order[i2], node_order[i1]
        for i in range(vcount):
            ni = node_order[i]
            community = membership[ni]
            weight_all = weight_inside = weight_loop = 0.0
            links = []
            for e in incident[ni]:
                w = weights[e]
                other = g.other(e, ni)
                weight_all += w
                if other == ni:
                    weight_loop += w
                    links.append((community, 0.0))
                    continue
                tc = membership[other]
                if community == tc:
                    weight_inside += w
                links.append((tc, w))
            _qsort(links, 0, len(links), key=lambda x: x[0])
            lc: List[int] = []
            lw: List[float] = []
            last = -1
            for c, w in links:
                if c != last:
                    lc.append(c)
                    lw.append(w)
                    last = c
                else:
                    lw[-1] += w
            old_id = membership[ni]
            new_id = old_id
            membership[ni] = -1
            size[old_id] -= 1
            if size[old_id] == 0:
                communities_no -= 1
            w_all[old_id] -= weight_all
            w_in[old_id] -= 2 * weight_inside + weight_loop
            max_q_gain = 0.0
            max_weight = weight_inside
            for c, w in zip(lc, lw):
                q_gain = w - resolution * w_all[c] * weight_all / weight_sum
                if q_gain > max_q_gain:
                    new_id = c
                    max_q_gain = q_gain
                    max_weight = w
            membership[ni] = new_id
            if size[new_id] == 0:
                communities_no += 1
            size[new_id] += 1
            w_all[new_id] += weight_all
            w_in[new_id] += 2 * max_weight + weight_loop
            if new_id != old_id:
                changed = True
        q = modularity()
        if changed and (q > pass_q):
            continue
        membership = temp_membership
        communities_no = temp_communities_no
        break
    # igraph_reindex_membership: renumber by first appearance
    remap = {}
    for c in membership:
        if c not in remap:
            remap[c] = len(remap)
    membership = [remap[c] for c in membership]
    # shrink + simplify_multiple
    nn = max(membership) + 1 if membership else 0
    new_edges = [(membership[int(frm[i])], membership[int(to[i])]) for i in range(ecount)]
    a = np.array([max(x) for x in new_edges], dtype=np.int64)
    b = np.array([min(x) for x in new_edges], dtype=np.int64)
    keys = sorted(set(zip(a.tolist(), b.tolist())))
    kid = {k: i for i, k in enumerate(keys)}
    eids = [kid[(int(a[i]), int(b[i]))] for i in range(ecount)]
    new_w = [0.0] * len(weights)                     # igraph keeps the vector length
    for i in range(ecount):
        new_w[eids[i]] += weights[i]
    return membership, q, _Graph(nn, keys), new_w


def community_multilevel(n: int, edges: Sequence[Tuple[int, int]], weights: Sequence[float],
                         rng: IgraphRRng, resolution: float = 1.0) -> np.ndarray:
    """``igraph_community_multilevel`` (final level membership, 0-based)."""
    g = _Graph(n, edges)
    w = [float(x) for x in weights]
    level_membership = list(range(n))
    q = -1.0
    while True:
        step_vcount = g.n
        prev_q = q
        m, q, g2, w2 = _multilevel_step(g, w, rng, resolution)
        if g2.n == step_vcount or q < prev_q:
            break
        level_membership = [m[x] for x in level_membership]
        g, w = g2, w2
    return np.asarray(level_membership, dtype=np.int64)


def ccan_graph_edges(peaks_i: np.ndarray, peaks_j: np.ndarray, coaccess: np.ndarray, n: int):
    """Edge list of ``graph_from_adjacency_matrix(make_sparse_matrix(...), "undirected",
    weighted = TRUE)``: lower triangle (diagonal included) in column-major order, i.e.
    for each column j: the zero-weight loop (j, j), then rows i > j ascending."""
    a = np.maximum(peaks_i, peaks_j)
    b = np.minimum(peaks_i, peaks_j)
    off = a != b
    a, b, x = a[off], b[off], coaccess[off]
    # first occurrence of each unordered pair wins (make_sparse_matrix's duplicated())
    key = b * n + a
    _, first = np.unique(key, return_index=True)
    first = np.sort(first)
    a, b, x = a[first], b[first], x[first]
    cols = np.concatenate([np.arange(n), b])
    rows = np.concatenate([np.arange(n), a])
    vals = np.concatenate([np.zeros(n), x])
    o = np.lexsort((rows, cols))
    return list(zip(rows[o].tolist(), cols[o].tolist())), vals[o].tolist()
