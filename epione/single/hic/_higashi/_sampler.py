"""Higashi mini-batch construction: positive hyperedges, negative sampling and
the cell-dependent neighbour masks of stages 2/3.

Adapted from Higashi (https://github.com/ma-compbio/Higashi @ 42702a8,
``Higashi_wrapper.py``: ``check_nonzero``, ``generate_negative_cpu``,
``one_thread_generate_neg``, ``to_neighs_to_mask``; ``Higashi_backend/
Modules.py``: ``DataGenerator``). MIT License, Copyright (c) 2020 Ma Lab at
CMU; see ``LICENSE_HIGASHI``.

Upstream runs these per-sample Python loops in a pool of worker processes.
Here they are vectorised over the batch with NumPy; the sampling rule (the
proposal distributions, the 3x3 "is this pixel already a contact" test, the
50-trial limit, the reset-to-positive on an invalid distance) is unchanged,
so the distribution of batches is the same. Only the random streams differ.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import vstack


class DataGenerator:
    """Per-chromosome positive hyperedge stream (upstream ``DataGenerator``)."""

    def __init__(self, edges, edge_chrom, edge_weight, batch_size, num_list, rng, k=1):
        self.batch_size = int(batch_size)
        self.k = k
        self.num_list = list(num_list)
        self.rng = rng
        self.edges = [np.array(e) for e in edges]
        self.edge_weight = [np.array(w) for w in edge_weight]
        self.edge_chrom = [np.array(c) for c in edge_chrom]
        self.chrom_list = np.arange(len(self.num_list) - 1)
        self.size_list = []
        for i in range(len(self.num_list) - 1):
            self.size_list.append(len(self.edges[i]))
            if len(self.edges[i]) == 0:
                continue
            while len(self.edges[i]) <= self.batch_size:
                self.edges[i] = np.concatenate([self.edges[i], self.edges[i]])
                self.edge_weight[i] = np.concatenate([self.edge_weight[i], self.edge_weight[i]])
                self.edge_chrom[i] = np.concatenate([self.edge_chrom[i], self.edge_chrom[i]])
        self.pointer = np.zeros(int(np.max(self.chrom_list) + 1)).astype('int')

    def filter_edges(self, min_bin=0, max_bin=-1):
        for i in range(len(self.edges)):
            d = self.edges[i][:, 2] - self.edges[i][:, 1]
            mask = (d > min_bin) & (d < max_bin)
            self.edges[i] = self.edges[i][mask]
            self.edge_weight[i] = self.edge_weight[i][mask]
            self.edge_chrom[i] = self.edge_chrom[i][mask]

    def next_iter(self):
        chroms = self.rng.choice(self.chrom_list, size=self.k, replace=True)
        e_list, c_list, w_list = [], [], []
        batch_size = int(self.batch_size / self.k)
        for chrom in chroms:
            if len(self.edges[chrom]) == 0:
                continue
            self.pointer[chrom] += batch_size
            if self.pointer[chrom] > len(self.edges[chrom]):
                index = self.rng.permutation(len(self.edges[chrom]))
                self.edges[chrom] = self.edges[chrom][index]
                self.edge_weight[chrom] = self.edge_weight[chrom][index]
                self.edge_chrom[chrom] = self.edge_chrom[chrom][index]
                self.pointer[chrom] = batch_size
            sl = slice(self.pointer[chrom] - batch_size, min(self.pointer[chrom], len(self.edges[chrom])))
            e_list.append(self.edges[chrom][sl])
            c_list.append(self.edge_chrom[chrom][sl])
            w_list.append(self.edge_weight[chrom][sl])
        return np.concatenate(e_list), np.concatenate(c_list), np.concatenate(w_list), chroms


class ContactIndex:
    """Sorted keys of the stored entries of every cell's depth-normalised
    contact matrix, for the vectorised ``check_nonzero``."""

    def __init__(self, sparse_by_chrom, num_list):
        self.num_list = np.asarray(num_list)
        self.n_bins = []
        self.keys = []
        for mats in sparse_by_chrom:
            n = mats[0].shape[0]
            big = vstack(list(mats)).tocoo()
            self.n_bins.append(n)
            self.keys.append(np.sort(big.row.astype(np.int64) * n + big.col.astype(np.int64)))

    def check_nonzero(self, temp, chrom):
        """Upstream ``check_nonzero`` with ``mem_efficient_flag=True``: is any
        stored entry inside the (clipped) 3x3 window around the pixel?"""
        out = np.zeros(len(temp), dtype=bool)
        for c in np.unique(chrom):
            sel = np.where(chrom == c)[0]
            n = self.n_bins[c]
            off = self.num_list[c]
            cell = temp[sel, 0].astype(np.int64) - 1
            r = temp[sel, 1].astype(np.int64) - 1 - off
            col = temp[sel, 2].astype(np.int64) - 1 - off
            rs, re = np.maximum(0, r - 1), np.minimum(r + 2, n - 1)
            cs, ce = np.maximum(0, col - 1), np.minimum(col + 2, n - 1)
            keys = self.keys[c]
            found = np.zeros(len(sel), dtype=bool)
            for dr in (-1, 0, 1):
                rr = r + dr
                vr = (rr >= rs) & (rr < re)
                for dc in (-1, 0, 1):
                    cc = col + dc
                    v = vr & (cc >= cs) & (cc < ce) & ~found
                    if not v.any():
                        continue
                    k = (cell[v] * n + rr[v]) * n + cc[v]
                    pos = np.searchsorted(keys, k)
                    pos = np.minimum(pos, len(keys) - 1)
                    hit = keys[pos] == k
                    idx = np.where(v)[0]
                    found[idx[hit]] = True
            out[sel] = found
        return out


class NegativeSampler:
    """Vectorised port of upstream ``generate_negative_cpu``.

    ``pair_ratio`` is the probability of a "hard" negative (one node of the
    positive replaced); otherwise the whole triple is redrawn. ``steps == 1``
    restarts every trial from the positive (stage 1); later stages keep
    perturbing the current proposal.
    """

    def __init__(self, index: ContactIndex, start_end_dict, max_bin, neg_num, rng):
        self.index = index
        # upstream prepends a zero row so that 1-based node ids index directly
        self.sed = np.concatenate([np.zeros((1, 2)), start_end_dict], axis=0).astype(np.int64)
        self.max_bin = int(max_bin)
        self.neg_num = int(neg_num)
        self.rng = rng
        self.pair_ratio = 0.0
        self.steps = 1

    def _randint(self, start, end):
        return (self.rng.random(len(start)) * (end - start)).astype(np.int64) + start

    def generate(self, x, x_chrom):
        neg_num, max_bin = self.neg_num, self.max_bin
        n = len(x)
        sample = np.repeat(x.astype(np.int64), neg_num, axis=0)
        chrom = np.repeat(x_chrom, neg_num)
        change = self.rng.integers(0, x.shape[-1], n * neg_num)
        hard_all = self.rng.random(n * neg_num) <= self.pair_ratio
        temp = sample.copy()
        trial = np.zeros(len(temp), dtype=np.int64)
        failed = np.zeros(len(temp), dtype=bool)
        active = self.index.check_nonzero(temp, chrom)
        while active.any():
            idx = np.where(active)[0]
            if self.steps == 1:
                temp[idx] = sample[idx]
            trial[idx] += 1
            give_up = trial[idx] >= 50
            if give_up.any():
                failed[idx[give_up]] = True
                active[idx[give_up]] = False
                idx = idx[~give_up]
                if len(idx) == 0:
                    break
            hard = hard_all[idx]
            hi = idx[hard]
            if len(hi):
                ch = change[hi]
                start = self.sed[temp[hi, ch], 0].copy()
                end = self.sed[temp[hi, ch], 1].copy()
                is_bin = ch != 0
                if is_bin.any():
                    b = hi[is_bin]
                    other = temp[b, 2]  # upstream: ``other_bin = 2 if change == 1 else 2``
                    s = np.maximum(start[is_bin], other - max_bin)
                    e = np.minimum(end[is_bin], other + max_bin)
                    temp[b, ch[is_bin]] = self._randint(s, e) + 1
                if (~is_bin).any():
                    b = hi[~is_bin]
                    temp[b, 0] = self._randint(start[~is_bin], end[~is_bin]) + 1
            si = idx[~hard]
            if len(si):
                s, e = self.sed[temp[si, 0], 0], self.sed[temp[si, 0], 1]
                temp[si, 0] = self._randint(s, e) + 1
                s, e = self.sed[temp[si, 1], 0], self.sed[temp[si, 1], 1]
                temp[si, 1] = self._randint(s, e) + 1
                s2 = np.maximum(s, temp[si, 1] - max_bin)
                e2 = np.minimum(e, temp[si, 1] + max_bin)
                temp[si, 2] = self._randint(s2, e2) + 1
            temp[idx] = np.sort(temp[idx], axis=1)
            d = temp[idx, 2] - temp[idx, 1]
            bad = (d >= max_bin) | (d <= 1)
            temp[idx[bad]] = sample[idx[bad]]
            active[idx] = self.index.check_nonzero(temp[idx], chrom[idx])
        keep = ~failed
        return temp[keep], chrom[keep]


class NeighborMaskBuilder:
    """Stage-2/3 neighbour lists as a row-normalised sparse matrix
    (upstream ``one_thread_generate_neg`` graphsage branch +
    ``to_neighs_to_mask``)."""

    def __init__(self, sparse_by_chrom, num_list, rng):
        self.num_list = np.asarray(num_list)
        self.rng = rng
        self.set_matrices(sparse_by_chrom)

    def set_matrices(self, sparse_by_chrom):
        self.n_bins = [m[0].shape[0] for m in sparse_by_chrom]
        self.big = [vstack(list(m)).tocsr().astype(np.float32) for m in sparse_by_chrom]
        for b in self.big:
            b.sort_indices()

    def build(self, x, x_chrom, training):
        cell_ids = np.repeat(x[:, 0], 2)
        bin_ids = x[:, 1:].reshape(-1)
        remove_ids = x[:, 1:][:, ::-1].reshape(-1)
        nodes_chrom = np.repeat(x_chrom, 2)
        remove_or_not = self.rng.random(len(nodes_chrom))
        rows_out, cols_out, vals_out = [], [], []
        for c in np.unique(nodes_chrom):
            sel = np.where(nodes_chrom == c)[0]
            n = self.n_bins[c]
            off = self.num_list[c]
            r = (cell_ids[sel].astype(np.int64) - 1) * n + (bin_ids[sel].astype(np.int64) - 1 - off)
            sub = self.big[c][r]
            indptr, indices, data = sub.indptr, sub.indices, sub.data
            row_of = np.repeat(np.arange(len(sel)), np.diff(indptr))
            keep = np.ones(len(indices), dtype=bool)
            if training:
                nnz = np.diff(indptr)
                drop_row = (remove_or_not[sel] >= 0.6) & (nnz > 1)
                rem_local = remove_ids[sel].astype(np.int64) - 1 - off
                keep = ~(drop_row[row_of] & (indices == rem_local[row_of]))
            row_of, indices, data = row_of[keep], indices[keep], data[keep]
            data = np.log1p(data)
            rsum = np.bincount(row_of, weights=data, minlength=len(sel)).astype(np.float32)
            data = (data / rsum[row_of]).astype(np.float32)
            rows_out.append(sel[row_of])
            cols_out.append(indices.astype(np.int64) + 1 + off)
            vals_out.append(data)
        rows = np.concatenate(rows_out)
        cols = np.concatenate(cols_out)
        vals = np.concatenate(vals_out)
        unique_nodes, inv = np.unique(cols, return_inverse=True)
        return (torch.from_numpy(np.stack([rows, inv.reshape(-1)])), torch.from_numpy(vals),
                torch.from_numpy(unique_nodes))


class BatchMaker:
    """Assemble one training / validation batch (upstream ``one_thread_generate_neg``)."""

    def __init__(self, sampler: NegativeSampler, neighbor_builder: NeighborMaskBuilder, mode, rng):
        self.sampler = sampler
        self.nbr = neighbor_builder
        self.mode = mode
        self.rng = rng
        self.graphsage = False

    def make(self, edges_part, edges_chrom, edge_weight, training):
        neg_num = self.sampler.neg_num
        if neg_num == 0:
            y = np.ones((len(edges_part), 1))
            w = np.ones((len(edges_part), 1)) * edge_weight.reshape((-1, 1))
            x, x_chrom = edges_part, edges_chrom
        else:
            neg_list, neg_chrom = self.sampler.generate(edges_part, edges_chrom)
            if len(neg_list) == 0:
                raise RuntimeError("Higashi negative sampling produced no negatives")
            correction = 1.0 if self.mode == "classification" else 0.0
            y = np.concatenate([np.ones((len(edges_part), 1)), np.zeros((len(neg_list), 1))])
            w = np.concatenate([np.ones((len(edges_part), 1)) * edge_weight.reshape((-1, 1)),
                                np.ones((len(neg_list), 1)) * correction])
            x = np.concatenate([edges_part, neg_list])
            x_chrom = np.concatenate([edges_chrom, neg_chrom])
        index = self.rng.permutation(len(x))
        x, y, w, x_chrom = x[index], y[index], w[index], x_chrom[index]
        to_neighs = self.nbr.build(x, x_chrom, training) if self.graphsage else []
        return x, y, w, x_chrom, to_neighs
