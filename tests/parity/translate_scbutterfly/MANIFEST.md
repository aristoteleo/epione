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

## Results, G3 (fast mode as registered: bf16, batch 256, learning rate 2e-3)

Run on 2026-10-12 on the shared H100, seeds 0–2, interleaved default/fast per seed. Truth is Epione's prepared matrices (G0 checks that they equal the reference truth).

**Speed: PASS.**

| seed | default (s) | fast (s) | ratio |
|---|---|---|---|
| 0 | 843 | 500 | 1.69 |
| 1 | 817 | 389 | 2.10 |
| 2 | 666 | 409 | 1.63 |

The median ratio is 1.69 (gate ≥ 1.5). Default seed 2 stopped early at translator epoch 181; every other run used the full 100 + 100 + 200 epochs.

**Metrics: FAIL**, 6 of 22 gated (metric, direction) pairs.

| direction | metric | default | fast | \|Δ\| | tol | result |
|---|---|---|---|---|---|---|
| rna2atac | MMD | 0.0440 ± 0.0055 | 0.0324 ± 0.0017 | 0.0116 | 0.0109 | fail (fast lower = better) |
| atac2rna | SPCC | 0.3809 ± 0.0011 | 0.4000 ± 0.0029 | 0.0191 | 0.0100 | fail (fast better) |
| atac2rna | CORR | 0.2569 ± 0.0034 | 0.2937 ± 0.0075 | 0.0368 | 0.0100 | fail (fast better) |
| atac2rna | ARI | 0.4499 ± 0.0203 | 0.3516 ± 0.0135 | 0.0982 | 0.0406 | fail (fast worse) |
| atac2rna | AMI | 0.7460 ± 0.0046 | 0.7097 ± 0.0090 | 0.0363 | 0.0300 | fail (fast worse) |
| atac2rna | NMI | 0.7555 ± 0.0045 | 0.7224 ± 0.0086 | 0.0332 | 0.0300 | fail (fast worse) |

The other 16 gated pairs pass. RNA→ATAC PCC, AUROC, MSE and CORR all agree to within 0.002.

The registered fast mode is therefore **not** equivalent to the default path: its ATAC→RNA predictions cluster worse but correlate better per feature.

## Amendment A1 (registered after the G3 failure, before running it)

Hypothesis: the 4× fewer optimiser steps per epoch, combined with the doubled learning rate, change the optimum.

A1 fast mode:
- bf16 autocast;
- **batch 128**;
- **learning rate 1e-3** (unchanged).

It uses the same seeds, the same G3 metric rule and the same floors. The speed gate is relaxed to ≥ 1.3, since a smaller batch saves less. If A1 passes, it becomes `fast=True`. If it fails, `fast=True` keeps the registered setting and is documented as not equivalent to the default.

## Results, amendment A1 (bf16, batch 128, learning rate 1e-3)

**Speed: PASS**, median 1.42 (gate ≥ 1.3).

| seed | A1 (s) | default (s) | ratio |
|---|---|---|---|
| 0 | 594 | 843 | 1.42 |
| 1 | 577 | 817 | 1.42 |
| 2 | 864 | 666 | 0.77 |

For seed 2 the default run stopped early and the A1 run did not.

**Metrics: FAIL**, 2 of 22 gated pairs.

| direction | metric | default | A1 | \|Δ\| | tol | result |
|---|---|---|---|---|---|---|
| atac2rna | ARI | 0.4499 ± 0.0203 | 0.3690 ± 0.0113 | 0.0808 | 0.0406 | fail |
| atac2rna | AMI | 0.7460 ± 0.0046 | 0.7149 ± 0.0091 | 0.0311 | 0.0300 | fail |

The other 20 pass, including atac2rna SPCC, CORR and NMI, and rna2atac MMD, which failed under G3.

Both fast variants lower the Leiden ARI of ATAC→RNA predictions. Their common element is bf16 autocast, which is a likely cause but was not tested.

**Outcome, as pre-registered.** `fast=True` keeps the registered setting (bf16, batch 256, learning rate 2e-3). It is documented as faster but **not equivalent** to the default path. For results comparable to the reference, use the default.

## Amendment A2: metric convention (before any G1/G2 result was recorded)

A first look at reference seed 0 found two metrics that disagree.
- Epione's AUROC was 0.893 against the reference's 0.568, for rna2atac.
- Epione's CORR was 3.6e-3 higher, for atac2rna.

Every other metric matched to within 1e-4, the clustering metrics exactly.

Cause: scTranslation's `calculate_auroc` passes `(truth, pred)` to a function whose signature is `(imputed_data, data)`. That swaps the roles in two ways:
- **AUROC.** The *prediction* is binarised at its median, and the measured values are used as scores.
- **CORR.** The benchmark averages over features whose *prediction* is not constant. Features with constant measured values score 0, because of sklearn's `r_regression(force_finite=True)`.

Recomputing with the roles swapped reproduces the reference seed-0 `metrics.json` to 1e-15, in both directions. The contract's §4 describes the intended definition, not what the code does.

Changes:
- `epione.translate` keeps the stated definitions as the default (`convention="standard"`). It adds `convention="sctranslation"`, which reproduces the benchmark's code.
- G1 and G2 compare under `convention="sctranslation"`, so that both sides compute the same quantity.
- The standard-convention AUROC, AUPRC and CORR are also reported, for both sides.
- No tolerance or floor changes.
