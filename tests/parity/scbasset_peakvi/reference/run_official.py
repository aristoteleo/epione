"""Reference env only (TensorFlow). Run calico/scBasset bin/scbasset_train.py unchanged, after seeding.

Run with TF_GPU_ALLOCATOR=cuda_malloc_async on a shared GPU; SCBASSET_SRC is the calico/scBasset checkout.

usage: run_official.py SEED -- <scbasset_train.py args>
The official script sets no seed; only tf.keras.utils.set_random_seed(SEED) is added.
"""
import os, sys, runpy
seed = int(sys.argv[1]); assert sys.argv[2] == "--"
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
tf.keras.utils.set_random_seed(seed)
script = os.path.join(os.environ["SCBASSET_SRC"], "bin", "scbasset_train.py")  # calico/scBasset checkout
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
