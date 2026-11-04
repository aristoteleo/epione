"""Compute the G2 metrics for one run directory (reference or Epione).

usage: python evaluate.py <fixture_dir> <run_dir> <out.json>
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import embedding_metrics, heldout_auroc, load_imputation_eval, pseudobulk_r  # noqa: E402

fixture, run_dir, out = sys.argv[1:4]
li = pickle.load(open(os.path.join(fixture, 'label_info.pickle'), 'rb'))
labels = np.asarray(li['cell type'])
sizes = {l.split('\t')[0]: int(l.split('\t')[1]) for l in open(os.path.join(fixture, 'chrom_sizes.txt'))}
chroms = list(sizes)
n_bins = {c: int(np.ceil(s / 1_000_000)) for c, s in sizes.items()}
heldout = dict(np.load(os.path.join(fixture, 'heldout.npz')))
full = dict(np.load(os.path.join(fixture, 'full_pixels.npz')))
res = {}
emb = np.load(os.path.join(run_dir, 'cell_embeddings_final.npy'))
em = embedding_metrics(emb, labels, np.asarray(li['repli_score'], float))
res.update({k: v for k, v in em.items() if not k.startswith('_')})
for k in (0, 5):
    ev, pb = load_imputation_eval(run_dir, k, chroms, n_bins, len(labels), heldout, labels, full)
    res[f'AUROC_nbr{k}'], _ = heldout_auroc(ev, heldout)
    res[f'PB_r_nbr{k}'], res[f'PB_r_nbr{k}_per_group'] = pseudobulk_r(pb)
t = os.path.join(run_dir, 'timing.json')
if os.path.exists(t):
    res['timing'] = json.load(open(t))
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps(res, indent=1))
