"""Reference env only (TensorFlow + calico/scBasset). Export a trained official model.

usage: export_official.py PROCESSED_DIR RUN_DIR
Writes RUN_DIR/result.npz: embedding (get_cell_embedding), intercept (get_intercept),
test_ids, test_prob (eval-mode predictions on test_seqs.h5, float32).
"""
import sys
import h5py
import numpy as np
import tensorflow as tf
from scipy import sparse
from scbasset.utils import make_model, get_cell_embedding, get_intercept, generator

proc, run = sys.argv[1], sys.argv[2]
m_test = sparse.load_npz(f"{proc}/m_test.npz")
n_cells = m_test.shape[1]
model = make_model(32, n_cells, show_summary=False)
model.load_weights(f"{run}/best_model.h5")
ds = tf.data.Dataset.from_generator(
    generator(f"{proc}/test_seqs.h5", m_test),
    output_signature=(tf.TensorSpec(shape=(1344, 4), dtype=tf.int8),
                      tf.TensorSpec(shape=(n_cells), dtype=tf.int8))).batch(128)
p = model.predict(ds, verbose=0)
with h5py.File(f"{proc}/splits.h5", "r") as f:
    test_ids = f["test_ids"][:]
np.savez_compressed(f"{run}/result.npz", embedding=get_cell_embedding(model), intercept=get_intercept(model),
                    test_ids=test_ids, test_prob=p.astype(np.float32))
print("exported", p.shape)
