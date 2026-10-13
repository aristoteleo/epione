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
