# multiDGD parity (metric level), pre-registered

Written before any Epione multiDGD run on the contract split. The reference
runs belong to agent A; their contract, inputs and outputs live outside the
repo, under `EPIONE_TRANSLATE_PARITY_DIR`:

- `CONTRACT.md`, `inputs/`, `split/fold0/`
- `ref/multiDGD/<rna2atac|atac2rna>/seed<k>/{pred,truth}.h5ad, metrics.json, run.json`

Reference implementation: scTranslation's vendored multiDGD at sctranslation@8bf1d63. It is a copy of upstream Center-for-Health-Data-Science/multiDGD@3887a98 (MIT). Ignoring comments, the training loop, prediction, GMM, representation, decoder and NB output modules are identical to upstream. scTranslation only adds an unused ADT output layer.

Reference schedule (`configs/default.yaml` and upstream defaults):
- batch 128, at most 500 epochs, early stopping as upstream (minimum 50 epochs, window 10);
- latent dimension 20, decoder 2 × 100 hidden units, then 3 layers per modality;
- learning rates 1e-4 (decoder), 1e-2 (representations), 1e-2 (GMM); Adam betas (0.5, 0.7), weight decay 1e-4.

## What the reference does, read from its code

These facts determine what can be compared. They were found before writing any Epione code.

1. **One Gaussian component.** The runner never sets `obs["observable"]`, so the GMM has a single component (the log says so).
2. **No covariates**, so there is no correction model.
3. **Unshuffled loaders.** The runner builds its loaders with `shuffle=False`. Upstream `DGD.train` shuffles both the training and the validation loader. This is the same deviation as scButterfly (amendment A3 there).
4. **Test representations are fitted on both modalities.** For the test cells, `DGD.test(n_epochs=20)` learns new representations from the test cells' concatenated RNA + ATAC counts. Because test and training sets have the same modalities, no modality translator is set, and the loss is the reconstruction of **both** modalities. The "RNA→ATAC" prediction is then the ATAC output decoded from that representation. **The target modality of each test cell is part of the input**, so the reference numbers measure joint reconstruction, not translation. Upstream multiDGD can fit a test representation from one modality only (its "modality translator" path), and that is translation.
5. **Predictions are softmax fractions.** `predict_from_representation` returns the decoder output: per-cell, per-modality softmax means that sum to 1 over features. They are not multiplied by a library size. The truth is raw counts. Hence MSE ≈ mean(truth²), and MMD and LISI compare distributions on different scales. Under this protocol, **MSE, MAE, MMD and LISI are degenerate**. They are gated as registered, but they carry no information about the model. The informative metrics are PCC, SPCC, CORR, AUROC and the four clustering scores.

## Setting

| | |
|---|---|
| data | 10x Multiome PBMC 10k (scGLUE release), 9,631 cells |
| split | fold0: 6,933 train, 771 validation, 1,927 test |
| RNA input | raw counts of the 3,000 contract HVGs. Epione: `epi.translate.prepare_counts`, which selects HVGs on log-normalised data, as the contract does, and returns the counts |
| ATAC input | raw counts, peaks open in ≥ 49 cells (99,272 peaks). Epione: the same `prepare_counts` call |
| seeds | 0, 1, 2 on both sides; the split is fixed |

Hyper-parameters are Epione's defaults, which equal the reference's.

Metrics are computed on the test cells, on Epione's predictions against the reference `truth.h5ad`. The tool is `epione.translate.evaluate` with `convention="sctranslation"`, the same convention as scButterfly amendment A2. The metric name mapping and the G1 tolerances are those of `translate_scbutterfly`.

## Epione inference modes

- `model.predict(source)` fits test representations on the **source modality only**, then decodes the target. This is the default: it is translation.
- `model.reconstruct(rna, atac)` fits them on **both** modalities, as scTranslation does, and returns both decoded modalities. It exists for this parity check and for joint reconstruction.

## Gates

**G0 · preprocessing.** `prepare_counts` on the raw GLUE files must equal the reference truth matrices on the test cells exactly (`array_equal`), with the same features in the same order.

**G1 · metric implementation.** `evaluate(ref pred, ref truth)` must reproduce the reference `metrics.json` for every seed and direction, with the scButterfly G1 tolerances. Clustering failures are reported but do not block.

**G2 · training parity.** This is the headline gate. Settings: Epione `reconstruct` (joint inference, the reference's protocol), `shuffle=False`, seeds 0–2.

For each direction and metric, take the mean over the 3 seeds on each side. Pass if

    |mean_Epione − mean_reference| ≤ max(2 × SD_reference, floor)

The floors are those of scButterfly:

| metric | floor |
|---|---|
| PCC, SPCC, CORR | 0.01 |
| AUROC | 0.005 |
| MSE, MAE | 5% of the reference mean |
| MMD | 10% of the reference mean |
| LISI | 0.02 |
| ARI, AMI, NMI, HOM | 0.03 |

- AUROC is gated for `rna2atac` only. For `atac2rna` it is reported but not gated.
- G2 passes if every gated (metric, direction) passes.

**Reported, not gated:**
- **G2-shuffled**: joint inference, `shuffle=True` (Epione's default and upstream's), against the reference.
- **G2-source**: Epione's default translation (`predict`, source-only inference), `shuffle=False`, against the reference. This quantifies how much of the reference's score came from seeing the target modality.

**G3 · fast mode.** `fast=True` means:
- bf16 autocast for the decoder's linear layers, with softmax and the NB log-likelihood in float32;
- training batch 512;
- decoder learning rate 2e-4;
- representation and GMM learning rates and the epoch schedule unchanged.

It is compared with Epione's default path (`shuffle=True`, source-only `predict`), with 3 seeds each, on the same split and the same GPU.
- **Metric gate**: the G2 rule, with SD_default (the SD of Epione's default seeds) in place of SD_reference, and the same floors.
- **Speed gate**: `median_k(train_seconds_default_k / train_seconds_fast_k) ≥ 1.5`.
- If either gate fails, `fast=True` is kept as registered and documented as not equivalent to the default. No amendment will be made to make it pass.

**Execution changes allowed in the default path.** These must not change what is computed. In the default path, Epione may:
- keep counts sparse and densify per batch;
- compute the NB log-likelihood as a dense zero-count term plus a sparse correction over nonzero counts. This is algebraically identical: for k = 0, lgamma(k + r) − lgamma(r) − lgamma(k + 1) = 0 and k·log(·) = 0.

They change summation order only. G2 decides whether they are acceptable.

## Results (all 3 reference seeds; 2026-10-13)

Runs: H100 shared with other jobs (it was at 100% utilisation throughout). The `shuffle=False`, default and fast runs used separate processes on the same card. Command: `parity_check.py gates`, with convention `sctranslation`. The full JSON stays outside the repo.

| gate | result |
|---|---|
| G0 preprocessing | **PASS**: both directions identical (max \|Δ\| 0), same features in the same order |
| G1 metric implementation, non-clustering (3 seeds × 2 directions) | **PASS**: max \|Δ\| ≤ 1.9e-15 for PCC, SPCC and AUROC; 1.7e-6 for MSE; 6e-5 for MMD |
| G1 clustering | one miss, reported and non-blocking: atac2rna seed 2 Leiden ARI 0.2107 (reference) vs 0.1813. Every other clustering value matches |
| **G2 training parity (joint inference, `shuffle=False`)** | **FAIL, 22/23 gated pairs**. atac2rna PCC: \|Δ\| = 0.0200 vs tolerance 0.0195 |
| G2-source (translation vs reference; not gated) | 22/23 within tolerance; atac2rna PCC outside it (\|Δ\| 0.0243) |
| G2-shuffled (Epione default vs reference; not gated) | 18/23 within tolerance |
| **G3 fast mode, metrics** | **PASS, 23/23** |
| **G3 fast mode, speed** | **PASS**: median 2.09× (2.66, 2.09, 1.51); gate ≥ 1.5 |

As registered, the G2 failure stands. No amendment was made. The failing pair is per-cell PCC of ATAC→RNA, under the protocol of §"What the reference does", point 5:
- reference seeds 0.917, 0.937, 0.926;
- Epione 0.887, 0.927, 0.906.

The Epione mean is 0.0005 outside the tolerance. Every other informative metric passes: SPCC, CORR, AUROC and the clustering scores. The degenerate ones (MSE, MAE, MMD, LISI) agree to 4 or more digits, as they must, because they are set by the truth's scale.

G2, joint inference, `shuffle=False`. Mean over 3 seeds:

| direction | metric | reference | Epione | \|Δ\| | tol | |
|---|---|---|---|---|---|---|
| rna2atac | PCC | 0.5466 ± 0.0120 | 0.5419 | 0.0047 | 0.0240 | pass |
| rna2atac | SPCC | 0.3388 ± 0.0087 | 0.3356 | 0.0031 | 0.0174 | pass |
| rna2atac | CORR | 0.1034 ± 0.0128 | 0.1012 | 0.0022 | 0.0256 | pass |
| rna2atac | AUROC | 0.5656 ± 0.0014 | 0.5652 | 0.0004 | 0.0050 | pass |
| rna2atac | ARI | 0.2350 ± 0.0166 | 0.2199 | 0.0151 | 0.0332 | pass |
| rna2atac | NMI | 0.5793 ± 0.0200 | 0.5743 | 0.0050 | 0.0400 | pass |
| atac2rna | PCC | 0.9266 ± 0.0098 | 0.9066 | 0.0200 | 0.0195 | **fail** |
| atac2rna | SPCC | 0.3229 ± 0.0114 | 0.3180 | 0.0049 | 0.0227 | pass |
| atac2rna | CORR | 0.0944 ± 0.0115 | 0.0893 | 0.0051 | 0.0230 | pass |
| atac2rna | AUROC | 0.5607 ± 0.0006 | 0.5604 | 0.0003 | 0.0050 | pass |
| atac2rna | ARI | 0.2048 ± 0.0052 | 0.1971 | 0.0077 | 0.0300 | pass |
| atac2rna | NMI | 0.5340 ± 0.0203 | 0.5408 | 0.0068 | 0.0406 | pass |

AMI, HOM, MSE, MAE, MMD and LISI also pass; the JSON has every value.

**Translation vs joint reconstruction.**
- G2-source is within 0.01 of G2 on most metrics; atac2rna PCC is 0.902 vs 0.907.
- With one mixture component and 20 inference steps, seeing the target modality adds little to these scores.
- Epione's default `predict` therefore does about as well as the reference's joint protocol on the benchmark metrics.

**Fast mode** (`fast=True`: bf16 decoder, batch 512, decoder learning rate 2e-4) passes both gates. All 3 fast seeds ran the full 500 epochs without early stopping; the default seeds ran 352, 500 and 462.

**Training time** (s, this shared GPU):
- reference: 1306, 848 and 2162;
- Epione default: 1563, 1697 and 1179;
- Epione `shuffle=False`: 712, 1043 and 1275;
- Epione fast: 588, 811 and 779.

The Epione default path is **not** faster than the reference on this contended card. The sparse NB likelihood saves arithmetic that is not the bottleneck here.
