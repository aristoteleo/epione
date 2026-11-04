"""Diagnostic D1: as run_ref.py, with the reference's Adam forced to foreach=False.

Run the official Higashi (ma-compbio/Higashi @ 42702a8) on the shared fixture.

usage: run_ref_noforeach.py <fixture_dir> <run_dir> <seed> [--process-only]

The global numpy / torch / random seeds are set once before ``process_data`` so the
preprocessing (TruncatedSVD, train/test split) is reproducible for the G0 check.
Training itself is not reproducible (sampler workers draw from fresh entropy).

The sampler worker function is wrapped to log, once per (pid, steps, pair_ratio),
the *effective* globals the forked workers see.
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch

fixture, run_dir, seed = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3])
process_only = '--process-only' in sys.argv
os.makedirs(run_dir, exist_ok=True)
os.chdir(run_dir)  # Higashi writes ./tmp1 ./tmp2 from nvidia-smi
chroms = [l.split('\t')[0] for l in open(os.path.join(fixture, 'chrom_sizes.txt'))]
config = {
    "data_dir": fixture,
    "input_format": "higashi_v1",
    "structured": False,
    "temp_dir": os.path.join(run_dir, "temp"),
    "genome_reference_path": os.path.join(fixture, "chrom_sizes.txt"),
    "chrom_list": chroms,
    "resolution": 1000000,
    "resolution_cell": 1000000,
    "local_transfer_range": 1,
    "dimensions": 64,
    "loss_mode": "zinb",
    "rank_thres": 1,
    "embedding_name": "exp",
    "impute_list": chroms,
    "minimum_distance": 1000000,
    "maximum_distance": -1,
    "neighbor_num": 5,
    "cpu_num": 8,
    "gpu_num": 1,
    "impute_verbose": 200,
}
cfg_path = os.path.join(run_dir, "config.JSON")
json.dump(config, open(cfg_path, "w"), indent=2)

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

import higashi.Higashi_wrapper as HW
_A = HW.torch.optim.Adam
HW.torch.optim.Adam = lambda params, **kw: _A(params, foreach=False, **kw)

_orig = HW.one_thread_generate_neg
_seen = set()


def logged_generate_neg(*args, **kwargs):
    key = (os.getpid(), HW.steps, HW.pair_ratio, HW.neg_num, HW.max_bin)
    if key not in _seen:
        _seen.add(key)
        with open(os.path.join(run_dir, "worker_globals.log"), "a") as fh:
            fh.write("pid=%d steps=%s pair_ratio=%s neg_num=%s max_bin=%s\n" % key)
    return _orig(*args, **kwargs)


HW.one_thread_generate_neg = logged_generate_neg

timing = {}
t0 = time.time()
model = HW.Higashi(cfg_path)
model.process_data()
timing['process'] = time.time() - t0
if process_only:
    json.dump(timing, open(os.path.join(run_dir, "timing.json"), "w"), indent=2)
    sys.exit(0)

t = time.time(); model.prep_model(); timing['prep_model'] = time.time() - t
t = time.time(); model.train_for_embeddings(); timing['stage1'] = time.time() - t
emb = model.fetch_cell_embeddings()
np.save(os.path.join(run_dir, "cell_embeddings.npy"), emb)
t = time.time(); model.train_for_imputation_nbr_0(); timing['stage2'] = time.time() - t
t = time.time(); model.impute_no_nbr(); timing['impute_nbr0'] = time.time() - t
t = time.time(); model.train_for_imputation_with_nbr(); timing['stage3'] = time.time() - t
t = time.time(); model.impute_with_nbr(); timing['impute_nbr5'] = time.time() - t
# end-of-stage-1 cell embeddings (what fetch_cell_embeddings() returns after the full pipeline);
# the parity runs obtained the same array with ``g1b_forward.py export``
np.save(os.path.join(run_dir, "cell_embeddings_final.npy"), model.fetch_cell_embeddings())
timing['total'] = time.time() - t0
json.dump(timing, open(os.path.join(run_dir, "timing.json"), "w"), indent=2)
print("DONE", timing)
