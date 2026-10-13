# BABEL parity (metric level), pre-registered

Written before any Epione BABEL run on the contract split. The only runs before it were on synthetic data, for smoke-testing the code.

## Clean-room statement

The reference code, wukevin/babel and the copy vendored in sctranslation@8bf1d63, has **no license**. `epione/translate/_babel.py` was written from the paper alone: Wu, Yost, Chang & Zou, *PNAS* 2021, Materials and Methods. No reference code was read or copied.

The reference was used only as a **black box**:
- its hyper-parameters from scTranslation's `configs/default.yaml`: hidden 16, β (`lossweight`) 1.33, Adam, learning rate 0.01, batch 64, early stop 25;
- its output files, which show the spaces it predicts and is scored in.

| | |
|---|---|
| ATAC→RNA truth | the test cells' counts over the 3,000 HVGs, size-normalised to the **test set's own median**. Every cell sums to 636; the train median is 654 |
| ATAC→RNA predictions | positive reals summing to about 600 per cell, consistent with NB means in that space |
| RNA→ATAC truth | the binary peaks (99,272) |
| RNA→ATAC predictions | mostly in [0, 1], but **0.7% of entries exceed 1** (maximum 1.95). A sigmoid output, as the paper describes, cannot produce this. The cause is unknown without reading the code, so Epione follows the paper. A difference in MSE, MAE or MMD for rna2atac may come from this |

Choices where the paper gives no value are listed in the module docstring:
- BatchNorm placement;
- gradient-norm clip 5;
- `ReduceLROnPlateau` defaults;
- at most 500 epochs, keeping the best validation weights;
- mean-reduced losses;
- training cells shuffled each epoch;
- standardisation statistics taken from the training cells.

None of these were tuned against the reference.

## Setting

| | |
|---|---|
| data | 10x Multiome PBMC 10k (scGLUE release) |
| split | fold0: 6,933 train, 771 validation, 1,927 test |
| seeds | 0, 1, 2 on both sides |
| RNA input | `inputs/rna_counts_hvg.h5ad`, raw counts; BABEL normalises internally |
| ATAC input | `epi.translate.prepare_atac(inputs/atac_counts.h5ad, use_tfidf=False, scale=False)`: binary peaks open in ≥ 49 cells |
| truth | the reference `truth.h5ad`, checked against Epione's own construction in G0 |

**Reference quirk, recorded before the Epione runs.** Reference seed 0 for rna2atac looks degenerate: SPCC 0.005 vs 0.235 for seeds 1 and 2, and AUROC 0.499 vs 0.540. The pre-registered rule uses the 3-seed mean and SD as they are, so this seed widens some tolerances. Results are also reported against seeds 1–2 alone, for information.

## Gates

The same structure, tolerances and floors as `tests/parity/translate_scbutterfly/MANIFEST.md`.

- **G0 · preprocessing.**
  - rna2atac: the binary peaks of the test cells equal the reference truth.
  - atac2rna: `epi.translate.size_normalize(test counts)` equals the reference truth, `allclose(rtol=1e-5, atol=1e-7)`.
- **G1 · metric implementation.** Epione's metrics on the reference predictions reproduce the reference `metrics.json`, under `convention="sctranslation"`, with the same per-metric tolerances. This checks the metrics on a second method's output spaces.
- **G2 · training parity.** For every (metric, direction), pass if `|mean_Epione − mean_reference| ≤ max(2 × SD_reference, floor)`, using 3 seeds a side, with the same floors as scButterfly. AUROC is not gated for atac2rna. G2 passes if all 22 gated pairs pass.
- **G3 · fast mode.**
  - Setting: `fast=True` is bf16 autocast with batch 256; the learning rate is unchanged.
  - Metric gate: the same rule as G2, against Epione's default (3 seeds).
  - Speed gate: the median ratio of training time, default/fast, is ≥ 1.5.
  - If it fails, fast mode is documented as **not equivalent** to the default. There are no amendments to make it pass.

## Results

Filled in after the runs.
