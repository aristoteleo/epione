"""G0: Epione's Higashi preprocessing vs the reference's ``process_data`` (seed 0).

usage: python g0_preprocess.py <fixture_dir> <reference_temp_dir> <out.json>
"""
import json
import sys

import h5py
import numpy as np
import pandas as pd

import epione.utils  # noqa: F401  (import order: avoids a circular import in epione.single)
from epione.single.hic._higashi import process_contacts

fixture, ref_temp, out = sys.argv[1:4]
sizes = {l.split('\t')[0]: int(l.split('\t')[1]) for l in open(f'{fixture}/chrom_sizes.txt')}
chroms = list(sizes)
df = pd.read_table(f'{fixture}/data.txt', sep='\t')
d = process_contacts(df, sizes, chroms, 1_000_000, random_state=0)

res = {}


def exact(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    ok = a.shape == b.shape and np.array_equal(a, b)
    res.setdefault(name, []).append(bool(ok))
    return ok


def close(name, a, b, rtol=1e-6, atol=1e-9):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol)
    dev = float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else float('nan')
    res.setdefault(name, []).append(bool(ok))
    res.setdefault(name + '_maxabs', []).append(dev)
    return ok


cse = np.load(f'{ref_temp}/chrom_start_end.npy')
exact('chrom_start_end', d.chrom_start_end, cse)
with h5py.File(f'{ref_temp}/node_feats.hdf5', 'r') as f:
    exact('num', d.num, f['num'][()])
    exact('start_end_dict', d.start_end_dict, f['start_end_dict'][()])
    exact('id2chrom', d.id2chrom, f['id2chrom'][()])
    close('sparsity', d.sparsity, f['sparsity'][()])
    close('extra_cell_feats', d.extra_cell_feats, f['extra_cell_feats'][()])
    close('cell2weight', d.cell2weight, f['cell2weight'][()])
    close('distance2weight', d.distance2weight, f['distance2weight'][()])
    rel = []
    for i, c in enumerate(chroms):
        exact('train_data', d.train_data[i], f[f'train_data_{c}'][()])
        exact('test_data', d.test_data[i], f[f'test_data_{c}'][()])
        exact('train_chrom', d.train_chrom[i], f[f'train_chrom_{c}'][()])
        exact('test_chrom', d.test_chrom[i], f[f'test_chrom_{c}'][()])
        exact('train_weight', d.train_weight[i], f[f'train_weight_{c}'][()])
        exact('test_weight', d.test_weight[i], f[f'test_weight_{c}'][()])
        close('bin_feats', d.bin_feats[i], f[str(i)][()])
        a = np.asarray(d.cell_feats[i], dtype=np.float64)
        b = np.asarray(f['cell'][str(i)][()], dtype=np.float64)
        r = float(np.linalg.norm(a - b) / np.linalg.norm(b)) if a.shape == b.shape else float('inf')
        rel.append(r)
    res['cell_feats_relfro'] = rel
    res['cell_feats'] = [r <= 1e-3 for r in rel]

gcn_ref = np.load(f'{ref_temp}/sparse_nondiag_adj_nbr_1.npy', allow_pickle=True)
for i, c in enumerate(chroms):
    raw_ref = np.load(f'{ref_temp}/raw/{c}_sparse_adj.npy', allow_pickle=True)
    ok_raw, ok_pat, maxd = True, True, 0.0
    ok_gcn = True
    for k in range(d.n_cells):
        a, b = d.sparse_raw[i][k].tocsr(), raw_ref[k].tocsr()
        a.sort_indices(); b.sort_indices()
        ok_raw &= (a != b).nnz == 0 and a.dtype == b.dtype
        g1, g2 = d.sparse_gcn[i][k].tocsr().astype(np.float64), gcn_ref[i][k].tocsr().astype(np.float64)
        g1.sort_indices(); g2.sort_indices()
        same = (np.array_equal(g1.indptr, g2.indptr) and np.array_equal(g1.indices, g2.indices))
        ok_pat &= same
        if same:
            ok_gcn &= np.allclose(g1.data, g2.data, rtol=1e-6, atol=1e-9)
            maxd = max(maxd, float(np.max(np.abs(g1.data - g2.data))) if g1.nnz else 0.0)
    res.setdefault('sparse_raw', []).append(bool(ok_raw))
    res.setdefault('sparse_gcn_pattern', []).append(bool(ok_pat))
    res.setdefault('sparse_gcn', []).append(bool(ok_gcn and ok_pat))
    res.setdefault('sparse_gcn_maxabs', []).append(maxd)

summary = {k: (all(v) if isinstance(v, list) and v and isinstance(v[0], bool) else v) for k, v in res.items()}
gated = [k for k in summary if not k.endswith('_maxabs') and k != 'cell_feats_relfro']
summary['G0_pass'] = all(summary[k] for k in gated)
json.dump({'summary': summary, 'detail': res}, open(out, 'w'), indent=1, default=float)
for k in gated + ['G0_pass']:
    print(f'{k:22s} {summary[k]}')
print('cell_feats rel Frobenius max', max(res['cell_feats_relfro']))
print('sparse_gcn max|d|', max(res['sparse_gcn_maxabs']))
