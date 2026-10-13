# scButterfly parity (metric level), pre-registered

Written before any Epione scButterfly run on the contract split. The reference
runs belong to agent A; their contract, inputs and outputs live outside the
repo, under `EPIONE_TRANSLATE_PARITY_DIR`:

- `CONTRACT.md`, `inputs/`, `split/fold0/`
- `ref/scbutterfly/<rna2atac|atac2rna>/seed<k>/{pred,truth}.h5ad, metrics.json, run.json`

Reference implementation: scTranslation's vendored scButterfly at sctranslation@8bf1d63 (upstream BioX-NKU/scButterfly@eb31e04, MIT).
Reference schedule (`configs/default.yaml`): batch 64, 100 + 100 pretraining epochs, 200 translator epochs, patience 50.

## Setting

| | |
|---|---|
| data | 10x Multiome PBMC 10k (scGLUE release), 9,631 cells |
| split | fold0: 6,933 train, 771 validation, 1,927 test |
| RNA input | `inputs/rna_lognorm_hvg.h5ad`, used as is |
| ATAC input | `epi.translate.prepare_atac(inputs/atac_counts.h5ad)` on all cells (binarise, peaks in ≥ 49 cells, TF-IDF, divide by the max), as in the reference runner |
| seeds | 0, 1, 2 on both sides; the split is fixed |

Hyper-parameters are Epione's defaults, which equal the reference config.

Metrics are computed on the test cells, on Epione's predictions against the reference `truth.h5ad`, with `epione.translate.evaluate`. Name mapping:

| contract | Epione |
|---|---|
| PCC | `PCC` |
| SPCC | `SPCC` |
| MSE | `MSE` |
| MAE | `MAE` |
| CORR | `CORR` |
| MMD | `MMD` |
| LISI | `iLISI` |
| AUROC | `AUROC` |
| ARI | `ARI` |
| AMI | `AMI` |
| NMI | `NMI` |
| HOM | `HOM` |

## Gates

**G0 · preprocessing.** Epione's prepared matrices must equal the reference `truth.h5ad` on the test cells:
- ATAC (`rna2atac` truth): `allclose(rtol=1e-5, atol=1e-7)`;
- RNA (`atac2rna` truth): exactly equal.

**G1 · metric implementation.** For every reference seed and direction, `epione.translate.evaluate(ref pred, ref truth)` must reproduce the reference `metrics.json`. This checks that Epione computes the same metrics, independently of training.

| metric | tolerance |
|---|---|
| PCC, SPCC, CORR, AUROC | ≤ 1e-4 |
| MSE, MAE | ≤ 1e-4 relative |
| MMD | ≤ 1e-3 relative (the reference computes in float32) |
| LISI | ≤ 1e-3 |
| ARI, AMI, NMI, HOM | ≤ 0.02. The Leiden result can differ across scanpy/igraph versions, so failures here are reported but do not block G2 |

**G2 · training parity.** This is the headline gate.

For each direction and metric, take the mean over the 3 seeds on each side. Pass if

    |mean_Epione − mean_reference| ≤ max(2 × SD_reference, floor)

where SD_reference is the sample SD of the 3 reference seeds.

| metric | floor |
|---|---|
| PCC, SPCC, CORR | 0.01 |
| AUROC | 0.005 |
| MSE, MAE | 5% of the reference mean |
| MMD | 10% of the reference mean |
| LISI | 0.02 |
| ARI, AMI, NMI, HOM | 0.03 |

- AUROC is gated for `rna2atac` only. For `atac2rna` it is reported but not gated, because AUROC of median-binarised log expression is not a meaningful target.
- G2 passes if every gated (metric, direction) passes. Every value is recorded, including failures.

**G3 · fast mode.** Compares `fast=True` (bf16 autocast, batch 256, learning rate 2e-3) with Epione's default path. Both use 3 seeds and the same split, and run on the same GPU.

- **Metric gate.** Same rule as G2, with SD_default (the SD of Epione's default seeds) in place of SD_reference, and the same floors.
- **Speed gate.** `median_k(train_seconds_default_k / train_seconds_fast_k) ≥ 1.5`. Seeds run back to back, default then fast, on the same device. A pilot of 8+8+8 epochs, run before this MANIFEST, measured about 1.7× per epoch on a shared H100.

## Results

Filled in after the runs; see below.
