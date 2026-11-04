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
- **G2 · training parity.** For every (metric, direction), pass if `|mean_Epione − mean_reference| ≤ max(2 × SD_reference, floor)`, using 3 seeds a side, with the same floors as scButterfly. AUROC is not gated for atac2rna. G2 passes if all 23 gated pairs pass (12 metrics × 2 directions, minus atac2rna AUROC).
- **G3 · fast mode.**
  - Setting: `fast=True` is bf16 autocast with batch 256; the learning rate is unchanged.
  - Metric gate: the same rule as G2, against Epione's default (3 seeds).
  - Speed gate: the median ratio of training time, default/fast, is ≥ 1.5.
  - If it fails, fast mode is documented as **not equivalent** to the default. There are no amendments to make it pass.

## Results (2026-10-13)

| gate | result |
|---|---|
| G0 preprocessing | **PASS**: rna2atac identical; atac2rna max \|Δ\| 3e-5 on values up to 410 |
| G1 metric implementation (3 seeds × 2 directions) | **PASS**: max \|Δ\| ≤ 7.6e-6 for every metric; clustering exact |
| **G2 training parity** | **FAIL: 6 / 23** gated pairs pass |
| G3 fast mode | speed **PASS** (median 2.32×); metrics **FAIL** (11 / 23 pass) |

### G2: mean of 3 seeds, reference vs Epione

| direction | metric | reference | Epione | \|Δ\| | tol | result |
|---|---|---|---|---|---|---|
| rna2atac | PCC | 0.3653 ± 0.0055 | 0.4966 | 0.1314 | 0.0110 | **fail** |
| rna2atac | SPCC | 0.1585 ± 0.1332 | 0.3411 | 0.1826 | 0.2664 | pass |
| rna2atac | MSE | 0.0650 ± 0.0001 | 0.0519 | 0.0131 | 0.0033 | **fail** |
| rna2atac | MAE | 0.0685 ± 0.0001 | 0.1003 | 0.0318 | 0.0034 | **fail** |
| rna2atac | CORR | 0.0136 ± 0.0036 | 0.1283 | 0.1147 | 0.0100 | **fail** |
| rna2atac | MMD | 1.8995 ± 0.0388 | 1.2350 | 0.6644 | 0.1899 | **fail** |
| rna2atac | LISI | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0200 | pass |
| rna2atac | AUROC | 0.5266 ± 0.0238 | 0.5652 | 0.0386 | 0.0476 | pass |
| rna2atac | ARI | 0.2577 ± 0.0214 | 0.3370 | 0.0793 | 0.0429 | **fail** |
| rna2atac | AMI | 0.5426 ± 0.0129 | 0.6074 | 0.0648 | 0.0300 | **fail** |
| rna2atac | NMI | 0.5603 ± 0.0121 | 0.6203 | 0.0600 | 0.0300 | **fail** |
| rna2atac | HOM | 0.6392 ± 0.0093 | 0.6787 | 0.0395 | 0.0300 | **fail** |
| atac2rna | PCC | 0.9671 ± 0.0014 | 0.9615 | 0.0056 | 0.0100 | pass |
| atac2rna | SPCC | 0.3536 ± 0.0004 | 0.3503 | 0.0033 | 0.0100 | pass |
| atac2rna | MSE | 1.6816 ± 0.2220 | 3.0149 | 1.3333 | 0.4439 | **fail** |
| atac2rna | MAE | 0.1668 ± 0.0024 | 0.1807 | 0.0139 | 0.0083 | **fail** |
| atac2rna | CORR | 0.1802 ± 0.0016 | 0.1698 | 0.0104 | 0.0100 | **fail** |
| atac2rna | MMD | 0.7310 ± 0.1686 | 0.2700 | 0.4610 | 0.3372 | **fail** |
| atac2rna | LISI | 0.2939 ± 0.0425 | 0.3536 | 0.0597 | 0.0849 | pass |
| atac2rna | ARI | 0.3910 ± 0.0368 | 0.2833 | 0.1076 | 0.0736 | **fail** |
| atac2rna | AMI | 0.6324 ± 0.0113 | 0.5507 | 0.0818 | 0.0300 | **fail** |
| atac2rna | NMI | 0.6443 ± 0.0105 | 0.5656 | 0.0788 | 0.0300 | **fail** |
| atac2rna | HOM | 0.6992 ± 0.0042 | 0.6219 | 0.0773 | 0.0300 | **fail** |

**What this says.** G0 and G1 show the two sides see the same inputs and truth and compute the same metrics. The gap therefore comes from the model or its training.
- **RNA→ATAC:** Epione's BABEL is clearly *better* (PCC 0.50 vs 0.37; per-peak CORR 0.13 vs 0.014; ARI 0.34 vs 0.26).
- **ATAC→RNA:** it is *worse* (ARI 0.28 vs 0.39; MSE 3.0 vs 1.7). The per-cell PCC is close (0.962 vs 0.967).

So it is not a faithful reproduction of the reference *numbers*, and it isn't uniformly worse either.

**Likely sources, none confirmable without reading the unlicensed code:**
1. **Training length.** Epione's runs stop early after 29–39 epochs (116–145 s); the reference trained for 250–632 s on the same GPU model, consistent with much longer training.
2. **Output post-processing.** 0.7% of the reference's RNA→ATAC predictions exceed 1, which no sigmoid output can produce.
3. **Unspecified details.** BatchNorm placement, the learning-rate schedule and its patience, and gradient clipping.

Each of these was fixed before the runs and was **not** tuned afterwards. Tuning until G2 passes would mean reverse-engineering the reference from its outputs.

**Outcome.** BABEL is offered as a reimplementation *of the paper*, documented as **not reproducing the scTranslation benchmark's BABEL numbers**. For results comparable to the published benchmark, use the original package.

### G3: fast mode (bf16, batch 256) vs default

Training seconds:
- default: 145, 116, 132;
- fast: 55, 78, 57;
- ratio: median 2.32.

| direction | metric | default | fast | result |
|---|---|---|---|---|
| rna2atac | PCC | 0.4966 | 0.4908 | pass |
| rna2atac | SPCC | 0.3411 | 0.3358 | pass |
| rna2atac | MSE | 0.0519 | 0.0525 | pass |
| rna2atac | MAE | 0.1003 | 0.1077 | pass |
| rna2atac | CORR | 0.1283 | 0.1096 | **fail** |
| rna2atac | MMD | 1.2350 | 1.4588 | **fail** |
| rna2atac | LISI | 0.0000 | 0.0000 | pass |
| rna2atac | AUROC | 0.5652 | 0.5643 | pass |
| rna2atac | ARI | 0.3370 | 0.2816 | pass |
| rna2atac | AMI | 0.6074 | 0.5705 | pass |
| rna2atac | NMI | 0.6203 | 0.5845 | pass |
| rna2atac | HOM | 0.6787 | 0.6462 | **fail** |
| atac2rna | PCC | 0.9615 | 0.9331 | **fail** |
| atac2rna | SPCC | 0.3503 | 0.3468 | pass |
| atac2rna | MSE | 3.0149 | 11.7714 | **fail** |
| atac2rna | MAE | 0.1807 | 0.2193 | **fail** |
| atac2rna | CORR | 0.1698 | 0.1420 | **fail** |
| atac2rna | MMD | 0.2700 | 0.9633 | **fail** |
| atac2rna | LISI | 0.3536 | 0.0514 | **fail** |
| atac2rna | ARI | 0.2833 | 0.2106 | pass |
| atac2rna | AMI | 0.5507 | 0.4794 | **fail** |
| atac2rna | NMI | 0.5656 | 0.4992 | **fail** |
| atac2rna | HOM | 0.6219 | 0.5680 | **fail** |

Per the pre-registered rule, `fast=True` is documented as **not equivalent** to the default. In particular, ATAC→RNA MSE is 11.8 vs 3.0 and iLISI 0.05 vs 0.35.

**Timing vs reference.** Epione default took 145, 116, 132 s; the reference took 534, 250, 632 s. The difference mostly reflects the earlier stopping, not faster code, so it is not reported as a speed-up.

## Amendment A1: match the reference training schedule (registered before the rerun)

The facts come from the reference run logs, which agent A saved under `work/sct_runs/BABEL/logs/train_s<k>.log` and `model/PBMC/<k>/net_history.json`, and from scTranslation's config. Only logged values and config values are used, no code.

| schedule fact | reference | source | Epione before A1 |
|---|---|---|---|
| optimiser and learning rate | Adam, 0.01 | config `optim: adam`, `lr: [0.01]`; paper | same |
| batch size | 64 | config `batchsize: [64]`; 122 batches per epoch = 109 train + 13 validation | same |
| early stopping | patience 25 on validation loss; the best checkpoint is kept | config `earlystop: 25`; log line "Stopping since valid_loss has not improved in the last 25 epochs"; `cp` column | same |
| **learning-rate schedule** | **constant 0.01** over all 49 / 44 / 43 epochs: the logged `lr` column never changes | `train_s{0,1,2}.log` | **ReduceLROnPlateau (factor 0.1, patience 10)** |
| epochs trained | 49 / 44 / 43, best epoch about 20–28 | logs | 39 / 29 / 35 |

**A1.** The only change: the learning rate stays at 0.01. `BabelConfig.lr_patience` is set to `None`, which means no scheduler; this becomes the default. All other settings are unchanged.

G2 and G3 are re-run with 3 seeds each and the same tolerances. If G2 still fails, it is reported as an unresolved discrepancy, with no further amendments.

**The reference outputs above 1.** All of them are in the **rna2atac** predictions, i.e. the ATAC head: 0.7% of entries, maximum 1.95. The atac2rna predictions are NB means in size-normalised counts, so values above 1 are expected there and aren't counted. An ATAC output through a sigmoid can't exceed 1, so the reference's ATAC head is not a plain sigmoid at prediction time. The paper says sigmoid, and Epione keeps the sigmoid.

## Results after A1 (final)

| gate | result |
|---|---|
| G0, G1 | PASS (unchanged) |
| **G2** | **FAIL: 6 / 23**. The same 17 pairs fail, with means equal to the pre-A1 runs to three decimals |
| G3 | speed PASS (median 1.92×); metrics FAIL (12 / 23 pass) |

**Why A1 changed nothing.** Epione stopped at 39 / 29 / 35 epochs, exactly as before. In the pre-A1 runs, the learning-rate cut only happened *after* each run's best epoch, and early stopping restores the best weights. So constant vs reduced learning rate yields the same model.

The real schedule difference is *when the validation loss stops improving*:
- Epione: best epoch about 4–14;
- reference: about 18–28 (log facts).

That is a property of the model and its optimisation, not of the schedule.

**Status: unresolved discrepancy.** Per the instruction, there are no further amendments. Epione's BABEL is a reimplementation of the paper:
- on RNA→ATAC it beats the reference implementation's scTranslation numbers on PCC, CORR and clustering;
- on ATAC→RNA it falls short on clustering, MSE and MAE.

It does **not** reproduce the reference within tolerance. Use the original package for results comparable to published scTranslation tables.

`fast=True` (bf16, batch 256) is **not equivalent** to the default (G3 metrics fail).

## Note: documented reference deviation (agent A's final report, contract §8)

scTranslation's vendored BABEL **drops upstream BABEL's per-chromosome ATAC encoder and decoder**: its `_infer_input_dim2` returns the total peak dimension. Epione follows the paper, with per-chromosome blocks. This plausibly explains why Epione's RNA→ATAC is *better* than the reference. Against the paper's PBMC numbers, the reference's RNA→ATAC is weak (NMI 0.56 vs 71.9, AUROC about 0.53).

This is recorded as a **documented deviation of the reference from the paper**, not a pass. G2 stays FAIL (6/23).

The ATAC→RNA side, where Epione is worse (ARI 0.28 vs 0.39, MSE 3.0 vs 1.7), is **still unexplained**. The reference's ATAC→RNA matches the paper (PCC 0.967 vs 96.71; MSE 1.68 vs 1.74).
