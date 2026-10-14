"""Run Epione's Higashi on the shared fixture with the reference's config.

usage: python run_epi.py <fixture_dir> <run_dir> <seed>

Writes, in the reference's layout so one evaluator reads both sides:
  cell_embeddings_final.npy              stage-1 cell embeddings (end of stage 1)
  temp/<chrom>_exp_nbr_{0,5}_impute.hdf5 'coordinates' + 'cell_<i>' (raw imputed values)
  timing.json, history.json
"""
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

import epione.utils  # noqa: F401  (import order)
from epione.single.hic._higashi import Higashi, process_contacts

fixture, run_dir, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(os.path.join(run_dir, 'temp'), exist_ok=True)
sizes = {l.split('\t')[0]: int(l.split('\t')[1]) for l in open(os.path.join(fixture, 'chrom_sizes.txt'))}
chroms = list(sizes)
timing = {}
t0 = time.time()
df = pd.read_table(os.path.join(fixture, 'data.txt'), sep='\t')
data = process_contacts(df, sizes, chroms, 1_000_000, random_state=seed)
del df
timing['process'] = time.time() - t0

m = Higashi(data, dimensions=64, loss_mode='zinb', neighbor_num=5, local_transfer_range=1,
            minimum_distance=1_000_000, maximum_distance=-1, seed=seed)
t = time.time(); m.prep_model(); timing['prep_model'] = time.time() - t
t = time.time(); m.train_for_embeddings(); timing['stage1'] = time.time() - t
np.save(os.path.join(run_dir, 'cell_embeddings_final.npy'), m.fetch_cell_embeddings())


def writer(k):
    pairs = m.impute_pairs()
    files = {}
    for c in chroms:
        f = h5py.File(os.path.join(run_dir, 'temp', f'{c}_exp_nbr_{k}_impute.hdf5'), 'w')
        f.create_dataset('coordinates', data=pairs[c][1])
        files[c] = f

    def on_cell(i, vals):
        for c, v in vals.items():
            files[c].create_dataset(f'cell_{i}', data=v, compression='gzip', compression_opts=6)
    return files, on_cell


t = time.time(); m.train_for_imputation_nbr_0(); timing['stage2'] = time.time() - t
files, cb = writer(0)
t = time.time(); m.impute_no_nbr(cb); timing['impute_nbr0'] = time.time() - t
for f in files.values():
    f.close()
t = time.time(); m.train_for_imputation_with_nbr(); timing['stage3'] = time.time() - t
files, cb = writer(5)
t = time.time(); m.impute_with_nbr(cb); timing['impute_nbr5'] = time.time() - t
for f in files.values():
    f.close()
timing['total'] = time.time() - t0
json.dump(timing, open(os.path.join(run_dir, 'timing.json'), 'w'), indent=2)
json.dump(m.history, open(os.path.join(run_dir, 'history.json'), 'w'), indent=1)
np.save(os.path.join(run_dir, 'cell_neighbors.npy'), np.array([np.asarray(x) for x in m.cell_neighbor_list[1:]]))
print('DONE', timing)
