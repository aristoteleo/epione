# scPair parity (metric level), pre-registered

Written before any Epione scPair code or run existed.

**Reference:** scTranslation's vendored scPair, at sctranslation@8bf1d63. It is an older copy of quon-titative-biology/scPair (MIT), and upstream is at c585949.

**Port:** `epione/translate/_scpair.py` adapts the MIT-licensed scPair code and keeps its notice. The ZINB log-likelihood follows scvi-tools (BSD-3), as scPair's does.

## How scTranslation runs scPair (from its runner and config)

Each run trains **two independent supervised predictors**, RNA→ATAC and ATAC→RNA, each an encoder plus a likelihood decoder. It then trains an embedding-to-embedding mapping, which `predict()` doesn't use.

| setting | value |
|---|---|
| encoder | FCNN [input, 800, 30] with layers Linear → LayerNorm → LeakyReLU → BatchNorm → Dropout(0.1), then an extra Linear(30, 30) |
| library-size branch | in the encoder when the output is ATAC (`infer_library_size_atac=True`) |
| RNA decoder | ZINB: mean = softmax(ReLU(Linear)) × library size (the observed library in training, `sample_factor_rna=True`); dispersion = exp(pre-softmax output); zero-inflation logits from a G×G Linear |
| ATAC decoder | Bernoulli: sigmoid(Linear) × sigmoid(inferred library) |
| optimiser | AdamW, learning rate 1e-3; weight decay 1e-9 on non-output weight matrices |
| batches | 130, shuffled each epoch |
| stopping | at most 1,000 epochs; early stop after 25 epochs without improvement on the full validation loss; best checkpoint kept |
| initialisation | Xavier-normal weights, zero biases |
| inputs | RNA as **raw counts** of the 3,000 HVGs (no log transform); ATAC as the peaks open in ≥ 49 cells, **binarised** inside scPair (`check_distribution`) for both input and target |
| predictions | RNA→ATAC: probabilities. ATAC→RNA: the ZINB *scale*, i.e. per-cell **proportions** summing to 1, because `predict()` passes library size 1 |
| truth files | RNA→ATAC: the **raw peak counts** of the test cells, not binarised (maximum 75). ATAC→RNA: the raw HVG counts |

**Vendored vs upstream.** The semantic diff (`ast.unparse`) shows only memory and engineering changes in upstream: gradient checkpointing, optional AMP, lazy GPU loading. The vendored copy also renames the modality "Gene Expression" to "Gene_Expression", consistently throughout. **No deviation that changes the trained model or its predictions was found**, unlike scButterfly's unshuffled loader.

**Epione differences, all in execution.**
- The mapping phase is omitted. It doesn't affect predictions; it only affects the optional cross-modal embeddings, which Epione doesn't expose yet. This saves time, so any speed comparison with the reference is reported with that caveat.
- Sparse matrices are densified batch by batch.

## Setting

| | |
|---|---|
| data | 10x Multiome PBMC 10k |
| split | fold0: 6,933 train, 771 validation, 1,927 test |
| seeds | 0, 1, 2 on both sides |
| RNA input | `inputs/rna_counts_hvg.h5ad` (raw counts) |
| ATAC input | `prepare_atac(inputs/atac_counts.h5ad, binarize=False, use_tfidf=False, scale=False)`: the ≥ 49-cell peaks as counts; scPair binarises them internally |

## Gates

The structure, tolerances and floors are those of the scButterfly and BABEL MANIFESTs.

- **G0 · preprocessing.**
  - rna2atac: Epione's filtered peak counts equal the reference truth exactly.
  - atac2rna: Epione's HVG counts equal the reference truth exactly.
- **G1 · metric implementation.** Epione's metrics on the reference predictions reproduce the reference `metrics.json` under `convention="sctranslation"`, with the same per-metric tolerances.
- **G2 · training parity.**
  - For every (metric, direction), pass if `|mean_Epione − mean_reference| ≤ max(2 × SD_reference, floor)`, using 3 seeds a side.
  - AUROC is not gated for atac2rna, so there are 23 gated pairs; G2 passes if all 23 pass.
  - If G2 fails, it is reported. Any amendment must be registered here before its run, must come from config, log or code facts, and is limited to one.
- **G3 · fast mode.**
  - Setting: `fast=True` is bf16 autocast with batch 520 (4×); the learning rate is unchanged.
  - Metric gate: the same rule as G2, against Epione's default.
  - Speed gate: the median default/fast ratio of training time is ≥ 1.5.
  - If it fails, fast mode is documented as **not equivalent** to the default, with no amendments.

## Results (2026-10-13)

| gate | result |
|---|---|
| G0 preprocessing | **PASS**: both truths identical |
| G1 metric implementation | **PASS**: max \|Δ\| ≤ 2e-6 for every metric; clustering exact |
| **G2 training parity** | **PASS: 23 / 23** |
| G3 fast mode | speed **FAIL** (median 1.17×, gate 1.5); metrics **FAIL** (20 / 23 pass) |

### G2: mean of 3 seeds, reference vs Epione

| direction | metric | reference | Epione | \|Δ\| | tol | result |
|---|---|---|---|---|---|---|
| rna2atac | PCC | 0.5313 ± 0.0018 | 0.5322 | 0.0010 | 0.0100 | pass |
| rna2atac | SPCC | 0.3583 ± 0.0000 | 0.3581 | 0.0001 | 0.0100 | pass |
| rna2atac | MSE | 0.6709 ± 0.0008 | 0.6710 | 0.0001 | 0.0335 | pass |
| rna2atac | MAE | 0.2320 ± 0.0003 | 0.2317 | 0.0003 | 0.0116 | pass |
| rna2atac | CORR | 0.1322 ± 0.0029 | 0.1276 | 0.0046 | 0.0100 | pass |
| rna2atac | MMD | 3.1883 ± 0.0115 | 3.1898 | 0.0016 | 0.3188 | pass |
| rna2atac | LISI | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0200 | pass |
| rna2atac | AUROC | 0.5685 ± 0.0000 | 0.5685 | 0.0000 | 0.0050 | pass |
| rna2atac | ARI | 0.4107 ± 0.0251 | 0.4372 | 0.0265 | 0.0501 | pass |
| rna2atac | AMI | 0.6704 ± 0.0110 | 0.6707 | 0.0003 | 0.0300 | pass |
| rna2atac | NMI | 0.6810 ± 0.0108 | 0.6805 | 0.0006 | 0.0300 | pass |
| rna2atac | HOM | 0.7371 ± 0.0117 | 0.7227 | 0.0144 | 0.0300 | pass |
| atac2rna | PCC | 0.9694 ± 0.0006 | 0.9694 | 0.0000 | 0.0100 | pass |
| atac2rna | SPCC | 0.4205 ± 0.0021 | 0.4165 | 0.0040 | 0.0100 | pass |
| atac2rna | MSE | 27.9965 ± 0.0011 | 27.9967 | 0.0002 | 1.3998 | pass |
| atac2rna | MAE | 0.2508 ± 0.0000 | 0.2508 | 0.0000 | 0.0125 | pass |
| atac2rna | CORR | 0.1521 ± 0.0027 | 0.1565 | 0.0044 | 0.0100 | pass |
| atac2rna | MMD | 3.6004 ± 0.0002 | 3.6004 | 0.0000 | 0.3600 | pass |
| atac2rna | LISI | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0200 | pass |
| atac2rna | ARI | 0.3726 ± 0.0287 | 0.3544 | 0.0183 | 0.0574 | pass |
| atac2rna | AMI | 0.6229 ± 0.0205 | 0.6194 | 0.0035 | 0.0410 | pass |
| atac2rna | NMI | 0.6343 ± 0.0199 | 0.6306 | 0.0037 | 0.0397 | pass |
| atac2rna | HOM | 0.6795 ± 0.0224 | 0.6746 | 0.0049 | 0.0448 | pass |

### G3

`fast=True` (bf16, batch 520) fails these pairs:

| direction | metric | default | fast | result |
|---|---|---|---|---|
| rna2atac | ARI | 0.4372 | 0.3552 | **fail** |
| rna2atac | AMI | 0.6707 | 0.6241 | **fail** |
| rna2atac | NMI | 0.6805 | 0.6372 | **fail** |

Training seconds, run concurrently:
- default: 792, 623, 789;
- fast: 770, 530, 658.

The speed gate fails, so `fast=True` is documented as **not equivalent** and **not faster** on this data. It is kept only as an option.

**Timing vs reference.** The reference took 966, 1692, 1200 s and Epione's default 792, 623, 789 s. The reference also trains the embedding-mapping phase, which Epione skips, so the times aren't compared as a speed-up.
