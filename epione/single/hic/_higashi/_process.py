"""Higashi preprocessing: contacts -> hypergraph + node attributes.

Adapted from Higashi (https://github.com/ma-compbio/Higashi @ 42702a8,
``higashi/Process.py``: ``generate_chrom_start_end``, ``data2triplets``,
``split_intra_inter``, ``create_matrix``, ``create_matrix_one_chrom``,
``generate_feats_one``). MIT License, Copyright (c) 2020 Ma Lab at CMU; see
``LICENSE_HIGASHI``.

Everything is kept in memory (upstream writes ``node_feats.hdf5`` and
several ``.npy`` files to ``temp_dir``). The numerical steps, their order and
the random-number consumption are those of upstream, so with the same seed
the output is the same as upstream's ``process_data`` after
``np.random.seed(seed)`` (checked by parity gate G0). Not ported: batch-effect
correction (``batch_id`` / ``library_id``), co-assay signals, inter-chromosomal
contacts and the scHiCluster random-walk baseline (``random_walk``).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, hstack, vstack


@dataclass
class HigashiInput:
    """Preprocessed Higashi hypergraph (what upstream keeps in ``temp_dir``)."""
    chrom_list: List[str]
    resolution: int
    resolution_cell: int
    chrom_start_end: np.ndarray            # (n_chrom, 2), bin offsets
    num: np.ndarray                        # [n_cells, n_bins_chr1, ...]
    sparse_raw: List[list]                 # per chrom, per cell: raw symmetric csr (diag doubled)
    sparse_gcn: List[list]                 # per chrom, per cell: off-diag, depth-normalised csr (float32)
    bin_feats: List[np.ndarray]            # per chrom: pseudo-bulk rows (bin node attributes)
    cell_feats: List[np.ndarray]           # per chrom: SVD of smoothed cell maps (cell node attributes)
    sparsity: np.ndarray                   # (n_cells,)
    extra_cell_feats: np.ndarray           # (n_cells, n_chrom) intra reads per chrom
    start_end_dict: np.ndarray
    id2chrom: np.ndarray
    train_data: List[np.ndarray]
    train_weight: List[np.ndarray]
    train_chrom: List[np.ndarray]
    test_data: List[np.ndarray]
    test_weight: List[np.ndarray]
    test_chrom: List[np.ndarray]
    distance2weight: np.ndarray
    cell2weight: np.ndarray
    extra: Dict = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(self.num[0])


def generate_chrom_start_end(chrom_sizes: Dict[str, int], chrom_list: Sequence[str], res: int) -> np.ndarray:
    chrom_start_end = np.zeros((len(chrom_list), 2), dtype='int')
    for i, chrom in enumerate(chrom_list):
        n_bin = int(math.ceil(int(chrom_sizes[chrom]) / res))
        chrom_start_end[i, 1] = chrom_start_end[i, 0] + n_bin
        if i + 1 < len(chrom_list):
            chrom_start_end[i + 1, 0] = chrom_start_end[i, 1]
    return chrom_start_end


def data2triplets(data: pd.DataFrame, chrom_list, chrom_start_end, res, distance_filter=False):
    """Bin contacts and sum duplicates (upstream ``data2triplets``).

    ``distance_filter`` reproduces the per-cell-file ("higashi_v2") branch,
    which drops intra-chromosomal pairs closer than 2.5 kb (except 0). The
    ``data.txt`` ("higashi_v1") branch applies no filter.
    """
    if distance_filter:
        d = np.abs(data['pos2'] - data['pos1'])
        same = data['chrom1'] == data['chrom2']
        data = data[(((d >= 2500) | (d == 0)) & same) | (~same)].reset_index()
    pos1 = np.array(data['pos1'])
    pos2 = np.array(data['pos2'])
    bin1 = np.floor(pos1 / res).astype('int')
    bin2 = np.floor(pos2 / res).astype('int')
    chrom1, chrom2 = np.array(data['chrom1'].values), np.array(data['chrom2'].values)
    cell_id = np.array(data['cell_id'].values).astype('int')
    count = np.array(data['count'].values) if 'count' in data.columns else np.ones(len(data))
    new_chrom1 = np.ones_like(bin1, dtype='int') * -1
    new_chrom2 = np.ones_like(bin1, dtype='int') * -1
    for i, chrom in enumerate(chrom_list):
        mask = chrom1 == chrom
        new_chrom1[mask] = i
        bin1[mask] += chrom_start_end[i, 0]
        mask = chrom2 == chrom
        new_chrom2[mask] = i
        bin2[mask] += chrom_start_end[i, 0]
    data = np.stack([cell_id, new_chrom1, new_chrom2, bin1, bin2], axis=-1)
    mask = (data[:, 1] >= 0) & (data[:, 2] >= 0)
    count = count[mask]
    data = data[mask]
    unique, inv = np.unique(data, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    # upstream accumulates count[i] into a float32 array one by one
    new_count = np.bincount(inv, weights=count.astype(np.float64), minlength=len(unique)).astype('float32')
    return unique, new_count


def split_intra_inter(u_, n_):
    intra_ = u_[:, 1] == u_[:, 2]
    intra_data = u_[intra_]
    intra_count = n_[intra_]
    intra_data = intra_data[:, [0, 1, 3, 4]]
    bin1, bin2 = intra_data[:, 2], intra_data[:, 3]
    new_bin1 = np.minimum(bin1, bin2)
    new_bin2 = np.maximum(bin1, bin2)
    intra_data[:, 2] = new_bin1
    intra_data[:, 3] = new_bin2
    return intra_data, intra_count


def _create_matrix_one_chrom(c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num, per_cell_read,
                             res, res_cell):
    from scipy.ndimage import gaussian_filter
    scale_factor = int(res_cell / res)
    cell_adj, qc_list, sparse_list, sparse_list_for_gcn, read_count, a = [], [], [], [], [], []
    # group rows by cell, keeping upstream's within-cell row order
    order = np.argsort(temp[:, 0], kind='stable')
    cells_sorted = temp[order, 0]
    bounds = np.searchsorted(cells_sorted, np.arange(cell_num + 1))
    b = 0
    for i in range(cell_num):
        sel = order[bounds[i]:bounds[i + 1]]
        temp2 = temp[sel, 2:] - chrom_start_end[c, 0]
        temp2_scale = np.floor(temp2 / scale_factor).astype('int')
        temp_weight2 = temp_weight[sel]
        read_count.append(np.sum(temp_weight2))
        m1 = csr_matrix((temp_weight2, (temp2[:, 0], temp2[:, 1])), shape=(size, size), dtype='float32')
        qc_list.append(np.sum(temp2[:, 0] != temp2[:, 1]))
        m1 = m1 + m1.T
        sparse_list.append(m1)

        mask1 = (temp2[:, 1] - temp2[:, 0]) > 0
        read_count_norm = per_cell_read / (np.sum(temp_weight2[mask1]) + 1e-15)
        m2 = csr_matrix((temp_weight2[mask1], (temp2[mask1, 0], temp2[mask1, 1])), shape=(size, size),
                        dtype='float32')
        m2 = m2 + m2.T
        m2 += diags(np.array(m2.sum(axis=-1) == 0).reshape((-1)).astype('float32'))
        m2 = m2 * read_count_norm
        sparse_list_for_gcn.append(csr_matrix(m2).astype('float32'))

        m = csr_matrix((temp_weight2, (temp2_scale[:, 0], temp2_scale[:, 1])), shape=(cell_size, cell_size),
                       dtype='float32')
        m = m + m.T
        m = csr_matrix(gaussian_filter(m.astype(np.float32).toarray(), 1, order=0, truncate=1))
        m = m / (m.sum() + 1e-15)
        cell_adj.append(m)

        if res_cell != 1000000:
            scale_factor2 = int(1000000 / res)
            size_metric = int(math.ceil(size * res / 1000000))
            temp2_scale = np.floor(temp2 / scale_factor2).astype('int')
        else:
            size_metric = int(math.ceil(size * res / 1000000))
        a.append(len(np.unique(temp2_scale, axis=0)))
        b = int(size_metric * size_metric / 2)
    return (np.array(read_count).reshape((-1)), np.array(a), b, np.asarray(qc_list), sparse_list,
            sparse_list_for_gcn, cell_adj)


def _pseudo_bulk(temp_mask, temp_weight_mask, c, size, chrom_start_end):
    from sklearn.preprocessing import normalize
    bin_adj = csr_matrix((temp_weight_mask, (temp_mask[:, 2] - chrom_start_end[c, 0],
                                             temp_mask[:, 3] - chrom_start_end[c, 0])),
                         shape=(size, size), dtype='float32')
    bin_adj = np.array(bin_adj.todense())
    bin_adj = bin_adj + bin_adj.T + np.diag(np.sum(bin_adj, axis=-1) == 0)
    mean_, std_ = np.mean(bin_adj), np.std(bin_adj)
    np.clip(bin_adj, a_min=None, a_max=mean_ + 10 * std_, out=bin_adj)
    return normalize(bin_adj, axis=1, norm='l1')


def generate_feats_one(temp1, temp2, size, length, qc_list, random_state):
    """Cell node attributes for one chromosome (upstream ``generate_feats_one``)."""
    from sklearn.preprocessing import normalize
    if temp1.shape[0] <= 3:
        return np.eye(temp1.shape[0])
    results = []
    temp = None
    for temp in [temp1, temp2]:
        if len(temp.data) == 0:
            continue
        mask = np.array(np.sum(temp > 0, axis=0) > min(5, temp.shape[0] - 2)).reshape((-1))
        temp = temp[:, mask]
        size = min(size, temp.shape[-1] - 2)
        mean_, std_ = np.mean(temp.data), np.std(temp.data)
        np.clip(temp.data, a_min=None, a_max=mean_ + 10 * std_, out=temp.data)
        results.append(temp)
    if len(results) == 2:
        split = results[0].shape[1]
        temp = hstack(results)
        temp = normalize(temp, norm='l1', axis=1) * length
        temp = csr_matrix(temp)
        temp1, temp2 = temp[:, :split], temp[:, split:]
    else:
        temp1, temp2 = temp, None

    # Upstream calls ``TruncatedSVD(size, algorithm='randomized', n_iter=2)``
    # under scikit-learn 1.0.x. Later releases changed that estimator's sign
    # convention (always V-based), so we call ``randomized_svd`` the way
    # 1.0.x did: ``flip_sign=True``, i.e. U-based unless the problem is
    # transposed.
    from sklearn.utils.extmath import randomized_svd, safe_sparse_dot

    def _fit(x):
        u, s, vt = randomized_svd(x, size, n_iter=2, random_state=random_state)
        return u * s, vt

    if len(qc_list) - np.sum(qc_list) > 10:
        _, vt = _fit(temp1[qc_list])
        temp1 = np.asarray(safe_sparse_dot(temp1, vt.T))
        if temp2 is not None:
            _, vt = _fit(temp2[qc_list])
            temp2 = np.asarray(safe_sparse_dot(temp2, vt.T))
    else:
        temp1, _ = _fit(temp1)
        if temp2 is not None:
            temp2, _ = _fit(temp2)
    if temp2 is not None:
        temp1 = np.concatenate([temp1, temp2], axis=1)
    return temp1


def process_contacts(contacts: pd.DataFrame, chrom_sizes: Dict[str, int], chrom_list: Sequence[str],
                     resolution: int, resolution_cell: Optional[int] = None, random_state=0,
                     distance_filter: bool = False, verbose: bool = False) -> HigashiInput:
    """Build Higashi's hypergraph from a contact table.

    Arguments:
        contacts: columns ``cell_id`` (0-based, contiguous), ``chrom1``,
            ``pos1``, ``chrom2``, ``pos2`` and optionally ``count``.
        chrom_sizes: chromosome lengths in bp.
        chrom_list: chromosomes to model (only intra-chromosomal contacts are used).
        resolution: bin size in bp of the hypergraph / imputation.
        resolution_cell: bin size used for the cell-node attributes
            (default ``resolution``; must be a multiple of it).
        random_state: seed or ``np.random.RandomState``; consumed in upstream's
            order (TruncatedSVD per chromosome, then the 85/15 train/test split).
        distance_filter: drop pairs < 2.5 kb apart (upstream's per-cell-file
            input branch). Irrelevant for binned input.
    """
    rs = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)
    chrom_list = list(chrom_list)
    res = int(resolution)
    res_cell = int(resolution_cell or resolution)
    if res_cell % res:
        raise ValueError("resolution_cell must be a multiple of resolution")
    chrom_start_end = generate_chrom_start_end(chrom_sizes, chrom_list, res)

    unique, new_count = data2triplets(contacts, chrom_list, chrom_start_end, res, distance_filter=distance_filter)
    data, weight = split_intra_inter(unique, new_count)
    weight = weight.astype('float32')

    cell_num = int(np.max(data[:, 0]) + 1)
    total_reads, total_possible = np.zeros(cell_num), 0
    sparse_raw, sparse_gcn, cell_feats_reads, qc_all, cell_adj_by_chrom, bin_feats = [], [], [], [], [], []
    data_within_chrom_list, weight_within_chrom_list = [], []
    scale_factor = int(res_cell / res)
    for c in range(len(chrom_list)):
        mask = data[:, 1] == c
        temp = data[mask]
        temp_weight = weight[mask]
        data = data[~mask]
        weight = weight[~mask]
        size = int(chrom_start_end[c, 1] - chrom_start_end[c, 0])
        cell_size = int(math.ceil(size / scale_factor))
        data_within_chrom_list.append(np.copy(temp))
        weight_within_chrom_list.append(np.copy(temp_weight))
        per_cell_read = np.sum(temp_weight) / cell_num
        (read_count, a, b, qc, s_raw, s_gcn, cell_adj) = _create_matrix_one_chrom(
            c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num, per_cell_read, res, res_cell)
        bin_adj = _pseudo_bulk(temp, temp_weight, c, size, chrom_start_end)
        # upstream computes the QC threshold on the full-size pseudo-bulk
        q_size = np.sum(np.sum(bin_adj > 0, axis=-1) > 0.1 * len(bin_adj))
        if size >= 3000:
            import torch
            import torch.nn.functional as F
            sf = int(round(size / 3000))
            conv_filter = torch.ones(1, 1, 1, sf)
            conv_filter = conv_filter / torch.sum(conv_filter)
            bin_adj = F.conv2d(torch.from_numpy(bin_adj)[None, None, :, :].float(), conv_filter,
                               stride=[1, sf]).numpy()[0, 0]
        bin_feats.append(bin_adj)
        total_reads += a.reshape((-1))
        total_possible += float(b)
        cell_feats_reads.append(read_count)
        qc_all.append((np.asarray(qc) >= max(0.25 * q_size - 5, 0)).astype('bool'))
        sparse_raw.append(s_raw)
        sparse_gcn.append(s_gcn)
        cell_adj_by_chrom.append(cell_adj)
        if verbose:
            print("processed", chrom_list[c], flush=True)

    total_linear_chrom_size = 0.0
    chrom2celladj = {}
    for c in range(len(chrom_list)):
        cell_adj_all = cell_adj_by_chrom[c]
        idx = np.triu_indices(cell_adj_all[0].shape[0], k=1)
        rows1, rows2 = [], []
        for m in cell_adj_all:
            m_diag = m.diagonal()
            m_nodiag = m - diags(m.diagonal())
            rows1.append(csr_matrix(m_nodiag[idx]))
            rows2.append(csr_matrix(m_diag.reshape((1, -1))))
        chrom2celladj[c] = [vstack(rows1).tocsr(), vstack(rows2).tocsr()]
        total_linear_chrom_size += int(math.sqrt(list(chrom2celladj[c][0].shape)[-1]) * res_cell / 1000000)
    cell_adj_by_chrom = None
    last = chrom2celladj[len(chrom_list) - 1][0]
    if len(chrom_list) > 1:
        total_embed_size = min(max(int(last.shape[0] * 0.5), int(total_linear_chrom_size * 0.5)),
                               int(last.shape[0] * 0.65))
    else:
        total_embed_size = int(np.min(last.shape) * 0.8)
    total_embed_size = min(total_embed_size, 2400)
    cell_feats = []
    for c in range(len(chrom_list)):
        temp = chrom2celladj[c]
        length = int(np.sqrt(temp[0].shape[-1]) / 1000000 * res_cell)
        size = int(total_embed_size / total_linear_chrom_size * length) + 1
        cell_feats.append(generate_feats_one(temp[0], temp[1], size, length, qc_all[c], rs))

    sparsity = total_reads / total_possible
    extra_cell_feats = np.stack(cell_feats_reads, axis=-1)

    data = np.concatenate(data_within_chrom_list)
    weight = np.concatenate(weight_within_chrom_list)
    chrom_info = data[:, 1]
    data = data[:, [0, 2, 3]]
    data[:, 1:] += np.max(data[:, 0]) + 1
    num = [np.max(data[:, 0]) + 1]
    for cse in chrom_start_end:
        num.append(cse[1] - cse[0])
    num_arr = np.asarray(num)
    num0 = [0] + list(num)
    num_list = np.cumsum(num0)
    start_end_dict = np.zeros((num_list[-1], 2), dtype='int')
    id2chrom = np.zeros((num_list[-1] + 1), dtype='int')
    for i in range(len(num0) - 1):
        start_end_dict[num_list[i]:num_list[i + 1], 0] = num_list[i]
        start_end_dict[num_list[i]:num_list[i + 1], 1] = num_list[i + 1]
        id2chrom[num_list[i] + 1:num_list[i + 1] + 1] = i - 1

    mask = data[:, 1] != data[:, 2]
    weight = weight[mask]
    data = data[mask]
    chrom_info = chrom_info[mask]
    tr_d, tr_w, tr_c, te_d, te_w, te_c = [], [], [], [], [], []
    for c in range(len(chrom_list)):
        mask = (chrom_info == c) & ((data[:, 2] - data[:, 1]) >= 2)
        index = np.arange(int(np.sum(mask)))
        rs.shuffle(index)
        train_index = index[:int(0.85 * len(index))]
        test_index = index[int(0.85 * len(index)):]
        tr_d.append(data[mask][train_index] + 1)
        tr_c.append(chrom_info[mask][train_index])
        tr_w.append(weight[mask][train_index].astype('float32'))
        te_d.append(data[mask][test_index] + 1)
        te_c.append(chrom_info[mask][test_index])
        te_w.append(weight[mask][test_index].astype('float32'))

    distance = data[:, 2] - data[:, 1]
    info = pd.DataFrame({'dis': distance, 'weight': weight, 'cell': data[:, 0]})
    info1 = info.groupby(by='dis').mean().reset_index()
    max_bin1 = int(np.max(num0[2:]))
    distance2weight = np.zeros((max_bin1, 1), dtype='float32')
    distance2weight[np.array(info1['dis']), 0] = np.array(info1['weight'])
    info1 = info.groupby(by='cell').mean().reset_index()
    cell2weight = np.zeros((num0[1], 1), dtype='float32')
    cell2weight[np.array(info1['cell']), 0] = np.array(info1['weight'])

    return HigashiInput(
        chrom_list=chrom_list, resolution=res, resolution_cell=res_cell, chrom_start_end=chrom_start_end,
        num=num_arr, sparse_raw=sparse_raw, sparse_gcn=sparse_gcn, bin_feats=bin_feats, cell_feats=cell_feats,
        sparsity=sparsity, extra_cell_feats=extra_cell_feats, start_end_dict=start_end_dict, id2chrom=id2chrom,
        train_data=tr_d, train_weight=tr_w, train_chrom=tr_c, test_data=te_d, test_weight=te_w, test_chrom=te_c,
        distance2weight=distance2weight, cell2weight=cell2weight,
    )


def contacts_from_cools(cool_paths: Sequence[str], chrom_list: Sequence[str]) -> pd.DataFrame:
    """Intra-chromosomal pixels of per-cell coolers as a Higashi contact table.

    ``pos`` is the bin start, so binning at the cooler resolution recovers the
    pixel exactly.
    """
    import cooler
    frames = []
    for cid, p in enumerate(cool_paths):
        clr = cooler.Cooler(str(p))
        res = int(clr.binsize)
        for ch in chrom_list:
            off = int(clr.extent(ch)[0])
            px = clr.matrix(balance=False, as_pixels=True).fetch(ch)
            if len(px) == 0:
                continue
            b1 = px['bin1_id'].to_numpy() - off
            b2 = px['bin2_id'].to_numpy() - off
            keep = b2 >= b1
            frames.append(pd.DataFrame({
                'cell_id': cid, 'chrom1': ch, 'pos1': b1[keep].astype(np.int64) * res,
                'chrom2': ch, 'pos2': b2[keep].astype(np.int64) * res,
                'count': px['count'].to_numpy()[keep]}))
    return pd.concat(frames, ignore_index=True)
