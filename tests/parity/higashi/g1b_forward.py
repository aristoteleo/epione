"""G1b / G1b-A1: Epione's network + imputation code with the reference's trained weights.

usage:
  (reference env)  python g1b_forward.py export <ref_run_dir>
  (Epione env)     python g1b_forward.py check <fixture_dir> <ref_run_dir> <out.json> [--a1]

``export`` loads the reference's pickled stage-1 model (needs the Higashi source on
PYTHONPATH) and writes the cell embeddings it produces. ``check`` rebuilds the
model in Epione from Epione's own preprocessing (``--a1``: with the reference's
stored cell-node attributes), loads the reference ``model_link`` state dicts with
``strict=True`` and compares embeddings, nearest-cell lists and imputations.
"""
import json
import os
import sys

import numpy as np

CELLS = [0, 700, 1429]

if sys.argv[1] == 'export':
    import torch
    run = sys.argv[2]
    import higashi.Higashi_wrapper  # noqa: F401  (registers the pickled classes)
    model = torch.load(os.path.join(run, 'temp/model/model.chkpt_stage1_model'), map_location='cuda')
    emb_nn = model.encode1.static_nn
    model.eval()
    with torch.no_grad():
        n_cells = int(emb_nn.num_list[1])
        ids = torch.arange(1, n_cells + 1).long().cuda()
        emb = emb_nn(ids).cpu().numpy()
    np.save(os.path.join(run, 'ref_stage1_cell_embeddings.npy'), emb)
    print('exported', emb.shape)
    sys.exit(0)

import h5py  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import epione.utils  # noqa: F401,E402
from epione.single.hic._higashi import Higashi, process_contacts  # noqa: E402

fixture, run, out = sys.argv[2:5]
a1 = '--a1' in sys.argv
sizes = {l.split('\t')[0]: int(l.split('\t')[1]) for l in open(os.path.join(fixture, 'chrom_sizes.txt'))}
chroms = list(sizes)
df = pd.read_table(os.path.join(fixture, 'data.txt'), sep='\t')
data = process_contacts(df, sizes, chroms, 1_000_000, random_state=0)
del df
if a1:
    with h5py.File(os.path.join(run, 'temp/node_feats.hdf5'), 'r') as f:
        data.cell_feats = [np.asarray(f['cell'][str(i)]) for i in range(len(chroms))]

m = Higashi(data, dimensions=64, loss_mode='zinb', neighbor_num=5, local_transfer_range=1,
            minimum_distance=1_000_000, maximum_distance=-1, seed=0, embedding_epoch=0, no_nbr_epoch=0,
            with_nbr_epoch=0, verbose=False)
m.prep_model()


def sd(stage):
    return torch.load(os.path.join(run, f'temp/model/model.chkpt_{stage}'), map_location=m.device)['model_link']


res = {'variant': 'A1 (reference cell attributes)' if a1 else 'original (Epione preprocessing)'}
m.higashi_model.load_state_dict(sd('stage1'), strict=True)
m.train_for_embeddings()  # 0 epochs: snapshot + embeddings with the loaded weights
emb = m.fetch_cell_embeddings()
ref_emb = np.load(os.path.join(run, 'ref_stage1_cell_embeddings.npy'))
res['a_cell_embedding_maxabs'] = float(np.max(np.abs(emb - ref_emb)))

m.train_for_imputation_nbr_0()
m.higashi_model.load_state_dict(sd('stage2'), strict=True)
m._snapshot('stage2')
m.train_for_imputation_with_nbr()
m.higashi_model.load_state_dict(sd('stage3'), strict=True)
m._snapshot('stage3')
wi = np.load(os.path.join(run, 'temp/weighted_info.npy'), allow_pickle=True)
ref_nbr = wi[0]
res['c_neighbor_lists_identical'] = bool(all(np.array_equal(np.asarray(ref_nbr[i]), np.asarray(m.cell_neighbor_list[i]))
                                             for i in range(1, m.cell_num + 1)))


def collect(fn):
    got = {}
    fn(lambda i, v: got.__setitem__(i, v), cells=CELLS)
    return got


for k, fn in ((0, m.impute_no_nbr), (5, m.impute_with_nbr)):
    got = collect(fn)
    worst = 0.0
    for c in chroms:
        with h5py.File(os.path.join(run, f'temp/{c}_exp_nbr_{k}_impute.hdf5'), 'r') as f:
            coords = np.asarray(f['coordinates'])
            assert np.array_equal(coords, m.impute_pairs()[c][1])
            for i in CELLS:
                worst = max(worst, float(np.max(np.abs(np.asarray(f[f'cell_{i}']) - got[i][c]))))
    res[f'{"b" if k == 0 else "c"}_impute_nbr{k}_maxabs'] = worst
res['G1b_pass'] = bool(res['a_cell_embedding_maxabs'] <= 1e-4 and res['b_impute_nbr0_maxabs'] <= 1e-4
                       and res['c_impute_nbr5_maxabs'] <= 1e-4 and res['c_neighbor_lists_identical'])
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps(res, indent=1))
