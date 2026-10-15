# GPU acceleration

Epione's ATAC hot paths take a `device=` argument. The default, `device="cpu"`,
is the original NumPy / SciPy implementation and is unchanged. `device="cuda"`
runs the same computation with [PyTorch](https://pytorch.org) on an NVIDIA GPU;
`device="auto"` picks the GPU when one is visible and falls back to the CPU
otherwise.

```python
import epione as epi

epi.tl.iterative_lsi(adata, device="cuda")
epi.pp.neighbors(adata, use_rep="X_iterative_lsi", device="cuda")
epi.tl.add_motif_matrix(adata, genome_fasta=fasta, device="cuda")
epi.tl.compute_deviations(adata, device="cuda")
```

## What runs on the GPU

| Function | GPU part | Agreement with `device="cpu"` |
|---|---|---|
| {func}`epione.tl.lsi` | TF-IDF, L1, `log1p`, randomized SVD | same arithmetic and precision; the SVD is a port of sklearn's `randomized_svd` that draws the same random test matrix, so for a fixed `random_state` the embedding matches to rounding |
| {func}`epione.tl.iterative_lsi` | each round's randomized SVD; the kNN search of the intermediate clustering (exact) | one round: matches to rounding. With two or more rounds the CPU uses scanpy's approximate pynndescent search for the intermediate clustering, so later rounds can differ about as much as two CPU seeds do |
| {func}`epione.pp.neighbors` | exact brute-force kNN (float64 distances) | same neighbour sets as the exact CPU search |
| {func}`epione.tl.compute_deviations` | observed counts, background-peer statistics and the null, in cell batches | same float32 arithmetic; Z-scores agree to float32 rounding |
| {func}`epione.tl.add_motif_matrix` | the PWM scan: a float32 convolution of the MOODS log-odds matrices over the one-hot sequence, then exact float64 rescoring, in MOODS's order, of the few windows within rounding distance of the threshold | identical hit matrix (MOODS's edge rules are reproduced too) |
| {func}`epione.tl.peak_to_gene` | metacell aggregation, per-pair correlation | correlations agree to float32 rounding |

Gene scores ({func}`epione.tl.add_gene_score_matrix`) and footprints stay on the
CPU: their cost is reading the fragment file, not arithmetic.

The pre-registered CPU-versus-GPU parity gates, their results and a speed-up
table are in `tests/parity/gpu/MANIFEST.md`. On the ArchR hematopoiesis data
(10,705 cells) with one shared H100 against 16 CPU cores: LSI 8x, chromVAR
15-19x, motif scan 8x, iterative LSI 4-6x; LSI on 128,509 cells x 250,000
features 30x (1,270 s -> 42 s). The tutorial
{doc}`tutorials/single/t_gpu_acceleration` runs one pipeline both ways.

### Memory

Every GPU path works in batches sized to a memory budget: half of the GPU
memory that is free when the call starts. Set `EPIONE_GPU_MEMORY_GB` to fix the
budget, for example on a GPU shared with other jobs:

```bash
export EPIONE_GPU_MEMORY_GB=16
```

Sparse matrices that do not fit the budget are streamed to the GPU in row
blocks, so large datasets (10^5 cells and more) run on a single GPU.

## Installation

The torch backend needs only PyTorch (and MOODS for the motif scan):

```bash
pip install "epione[gpu]"
```

`pip` installs the default torch wheel for your platform. If you need a build
for a specific CUDA version, install torch first following
<https://pytorch.org/get-started/locally/>, for example

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "epione[gpu]"
```

The CUDA runtime ships inside the torch wheel; the machine needs an NVIDIA
driver recent enough for that CUDA version (`nvidia-smi` shows the highest CUDA
version the driver supports). Check the installation with

```python
import epione as epi
epi.core.resolve_device("auto")   # 'cuda:0' when the GPU is usable, else 'cpu'
```

## Optional: rapids-singlecell for graph steps

UMAP and Leiden have no torch implementation in Epione. When
[rapids-singlecell](https://rapids-singlecell.readthedocs.io) is installed they
can be dispatched to cuML / cuGraph with `backend="rapids"`:

```python
epi.pp.neighbors(adata, use_rep="X_iterative_lsi", backend="rapids")
epi.tl.umap(adata, backend="rapids")
epi.tl.clusters(adata, method="leiden", backend="rapids",
                leiden_kwargs=dict(resolution=1.0, random_state=0))
```

rapids-singlecell is **not** a dependency of Epione and is not part of the
`gpu` extra. Install it separately, with the RAPIDS libraries:

```bash
pip install "rapids-singlecell[rapids-cu12]"
```

(Recent releases ship their compiled CUDA kernels in the
`rapids-singlecell-cu12` wheel; inside an environment that already has RAPIDS,
such as the RAPIDS container, `pip install rapids-singlecell-cu12` is enough.)

Requirements of the RAPIDS wheels (cuDF, cuML, cuGraph, cuVS):

- Linux x86-64 or aarch64 with **glibc 2.28 or newer** (the wheels are
  `manylinux_2_28`; check with `ldd --version`). Older systems, such as
  CentOS / RHEL 7 with glibc 2.17, cannot install them natively; use the
  [RAPIDS container](https://hub.docker.com/r/rapidsai/base) instead, for
  example with Apptainer:

  ```bash
  apptainer build --sandbox rapids docker://rapidsai/base:25.12-cuda12-py3.12
  apptainer exec --nv rapids pip install --user rapids-singlecell-cu12 epione
  apptainer exec --nv rapids python my_script.py
  ```

- Python 3.12 or newer for rapids-singlecell 0.18.
- A CUDA 12 driver and a GPU of compute capability 7.0 (Volta) or newer.

Leiden and UMAP are stochastic, and cuGraph / cuML are different
implementations from igraph / umap-learn, so `backend="rapids"` results agree
with the CPU ones in the way two CPU seeds agree (similar ARI and
trustworthiness), not value for value.
