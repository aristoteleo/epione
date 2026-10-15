"""Reference env only. G1b: tf.keras.metrics.AUC(curve="ROC", multi_label=True) on 5 seeded random batches.

usage: keras_auc.py OUT.npz
"""
import sys
import numpy as np
import tensorflow as tf

rng = np.random.default_rng(0)
m = tf.keras.metrics.AUC(curve="ROC", multi_label=True)
P, Y = [], []
for _ in range(5):
    y = (rng.random((128, 300)) < 0.15).astype(np.float32)
    p = np.clip(rng.random((128, 300)) * 0.6 + 0.35 * y, 0, 1).astype(np.float32)
    m.update_state(y, p)
    P.append(p); Y.append(y)
np.savez(sys.argv[1], p=np.stack(P), y=np.stack(Y), auc=float(m.result().numpy()))
print(float(m.result().numpy()))
