# scBasset and PeakVI parity, pre-registered

Written and committed before any Epione-vs-reference comparison. Gates are
never widened after results are seen. A change made after seeing results is
an amendment recorded below, with its reason, before it is run.

Before registration, 1-2 epoch timing probes ran on both sides to size the
compute budget. Their training logs were seen (epoch-1 training AUC was
0.610 for Epione and 0.609 for the official code). No gate quantity was
computed.

## Discovery

- **omicverse.** github.com/omicverse has no scBasset or PeakVI port, so
  nothing upstream is reused. Epione does not depend on omicverse.
- **PeakVI.** scvi-tools (BSD-3-Clause) `scvi.model.PEAKVI` is the authors'
  own implementation. `epi.tl.peakvi` is a thin wrapper around it, placed and
  shaped like `epi.tl.multivi` (#70).
- **scBasset.** The official code is TensorFlow/Keras (calico/scBasset,
  Apache-2.0, commit aed3a6f). scvi-tools ships a PyTorch
  `scvi.external.SCBASSET`. Read against the official code (scvi-tools
  1.4.2), it differs in the following ways:

  | aspect | official (calico) | scvi-tools 1.4.2 SCBASSET |
  |---|---|---|
  | tower filters | rounds the running product: 288, 323, 363, 407, 456, 512 | rounds each step: 288, 323, 362, 406, 456, 512 |
  | kernel init | Keras `he_normal` (truncated normal) everywhere | torch default for conv; kaiming normal for dense |
  | cell embedding init | `he_normal`, sd ≈ 0.25 | `randn`, sd 1 (4x larger) |
  | cell intercept init | 0 | `randn` |
  | stem max-pool | Keras `same` (no padding at 1344/3) | `padding=1`, which shifts the pooling windows by one |
  | Adam epsilon | 1e-7 (Keras) | 1e-8 |
  | peak split | 90/5/5 train/test/val, seed 10 | 90/10 train/val, no test set |
  | early stopping | training multi-label (per-cell) ROC AUC, patience 50 | per-batch flattened AUROC, patience 45 |
  | weights kept | best epoch (ModelCheckpoint) | last epoch |
  | TF activity | sigmoid of the intercept-free output | intercept-free logits, no sigmoid |

  None of these changes the paper's architecture class. Together they are a
  material difference in initialisation and training. Epione therefore
  rewrites the official code in PyTorch (`epione.models.scbasset`), layer for
  layer, without depending on scvi-tools for scBasset. The rewrite also loads
  the official `.h5` weights. The gates compare it with the official
  TensorFlow code. scvi-tools SCBASSET runs only as an information row.
- **Deliberate deviation in Epione's port.** Batches are drawn from a full
  random permutation of the training peaks each epoch. The official
  `tf.data` pipeline uses a 2,000-element shuffle buffer over
  genome-ordered peaks.
- **Not ported.** The batch-corrected variant (`scbasset_bc_train.py`) is
  left out.

## Data

- **Cells and labels.** 10x Multiome PBMC 10k ATAC, scGLUE release, as
  returned by `epi.datasets.pbmc10k_multiome`: 9,631 cells and Seurat-WNN
  `obs["cell_type"]` (19 types). This is the sample of scBasset's own
  multiome example.
- **Peak filter.** Exactly calico `examples/PBMC_multiome/make_anndata.ipynb`:
  peaks accessible in more than `int(0.05 × n_cells)` cells, on chr1-22, X and
  Y. This leaves 29,542 peaks.
- **Genome.** GRCh38, the GENCODE v41 FASTA behind `epi.data.hg38`.
- **Reference preprocessing.** The official `bin/scbasset_preprocess.py`,
  unchanged.
- **Shared input.** Both sides read the same `atac_ad.h5ad`. PeakVI uses the
  same matrix.

## Environments

- **Epione.** Python 3.11, torch 2.6.0+cu124, scvi-tools 1.4.2. No
  TensorFlow.
- **Reference.** A separate venv with Python 3.10, TensorFlow 2.15.1 (Keras
  2.15) and calico/scBasset at aed3a6f. It is outside the repository.
- **Hardware.** GPU runs use the one shared H100.
- **Seeding the official side.** `scbasset_train.py` sets no seed. The
  reference runner calls it unchanged after `tf.keras.utils.set_random_seed(seed)`
  and uses `TF_GPU_ALLOCATOR=cuda_malloc_async`.

## Gates

### G0 · preprocessing equality (exact)

| id | quantity | pass if |
|---|---|---|
| G0a | peak table (chr, start, end), Epione `peak_table(var)` vs official `peaks.bed` | identical |
| G0b | sequences, Epione `peak_sequences` vs official `all_seqs.h5` | identical at every A/C/G/T position. Positions that are not A/C/G/T are random on both sides, so they are counted and only checked to lie in 0-3. |
| G0c | binary targets, Epione `binarize(X)` rows vs official `m_train` / `m_val` / `m_test` | identical sparsity pattern, all values 1 |
| G0d | train / test / val peak ids vs official `splits.h5` | identical |

### G1 · metric code and architecture

| id | quantity | pass if |
|---|---|---|
| G1a | the parity metric module on synthetic data: kNN accuracy vs brute force, per-cell AUROC vs a Mann-Whitney implementation; ARI, AMI and silhouette are sklearn calls | \|Δ\| ≤ 1e-12 |
| G1b | Epione `KerasMultiLabelAUC` vs `tf.keras.metrics.AUC(curve="ROC", multi_label=True)`, accumulated over 5 random batches | \|Δ\| ≤ 1e-5 |
| G1c | official seed-0 `best_model.h5` loaded into Epione's `ScBassetNet` with `load_keras_h5` | eval-mode test-peak probabilities on CPU: max \|Δ\| ≤ 1e-4. Cell embedding and intercept: identical to official `get_cell_embedding` / `get_intercept`. |

### G2 · scBasset, Epione vs official (metric level)

- **Runs.** Seeds 0, 1, 2 on each side, same input.
- **Training.** Official defaults (batch 128, lr 0.01, patience 50 on the
  training AUC, best epoch kept), except that both sides cap training at
  **200 epochs**. The official default is 1,000 epochs. At about 30 s per
  epoch on the shared GPU, 1,000 epochs over 6 runs is about 50 GPU-hours, so
  the compute forces the cap. It applies to both sides alike, and Epione's
  default stays at 1,000.
- **Embedding.** The cell embedding is the final-layer kernel, used raw.

Metrics, computed by the same script for both sides:

- `sc.pp.neighbors(n_neighbors=15, random_state=0)`, then
  `sc.tl.leiden(resolution=1.0, flavor="igraph", n_iterations=2,
  directed=False, random_state=0)`. **ARI** and **AMI** of the clusters vs
  `cell_type`.
- **kNN accuracy.** Leave-one-out 15-NN (Euclidean, self excluded)
  majority vote of `cell_type`, ties to the smallest label code.
- **Silhouette.** sklearn `silhouette_score(embedding, cell_type)`,
  Euclidean.
- **Test AUROC.** Mean over cells of the per-cell sklearn `roc_auc_score` on
  the held-out test peaks. Eval-mode sigmoid outputs including the intercept.
  Cells with constant labels are skipped.

Rule per metric: |mean_Epione − mean_official| ≤ max(2 × SD, floor). SD is
the larger of the two sides' sample SDs (ddof = 1). Floors are 0.05 for ARI,
0.03 for AMI, 0.02 for kNN accuracy, 0.03 for silhouette and 0.01 for test
AUROC.

### P · PeakVI, Epione wrapper vs direct scvi-tools

| id | runs | pass if |
|---|---|---|
| P1 | CPU, seed 0, 10 epochs. `epi.tl.peakvi(adata, seed=0, max_epochs=10, accelerator="cpu")` vs `scvi.settings.seed = 0; PEAKVI.setup_anndata(adata); PEAKVI(adata).train(max_epochs=10, accelerator="cpu")` | latent max \|Δ\| = 0 |
| P2 | GPU, seeds 0-2 per side, scvi-tools default training (max 500 epochs, early stopping) | the G2 rule on ARI, AMI, kNN accuracy and silhouette; max \|Δ latent\| reported |

### G3 · fast mode

None was added, so there is nothing to gate.

### Information (no gate)

- scvi-tools `SCBASSET`, seeds 0-2 as time allows. Same sequences and binary
  targets, regions × cells as in the scvi-tools tutorial, 200-epoch cap,
  otherwise its defaults. The G2 metrics are reported next to the official
  values.
- For the tutorial: LSI on the same matrix (`epi.tl.lsi`, 50 components,
  first dropped).

## Results

(appended after the runs)
