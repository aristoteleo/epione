# JAMIE parity (metric level), pre-registered

Written before any Epione JAMIE code or run existed.

**Reference:** scTranslation's vendored JAMIE, at sctranslation@8bf1d63, from daifengwanglab/JAMIE, GPL-3. JAMIE subclasses UnionCom (MIT).

**Port:** `epione/translate/_jamie.py` adapts JAMIE (GPL-3, the same licence as Epione) and the parts of UnionCom it uses (MIT), keeping both notices.

## How scTranslation runs JAMIE (runner, config, and the defaults it inherits)

| item | value |
|---|---|
| inputs | RNA: **raw counts** of the 3,000 HVGs; the runner's RNA scaling is commented out. ATAC: peaks open in ≥ 49 cells, then `sc.pp.scale` (zero-centred z-score) over **all** cells before the split |
| fitting | `fit_transform` on the **training cells only** (6,933); the validation split is unused |
| correspondence | no saved match file, so UnionCom's Prime-Dual matching runs on **geodesic** distances. `distance_mode` keeps UnionCom's default `geodesic` because the runner sets JAMIE's `dist_method`, not `distance_mode`. Distances use a kNN graph (k from 5 upwards until connected, at most max(40, 1% of n)) on the raw inputs; Prime-Dual runs 2,000 iterations with ρ 10, ε 0.01, delay 0 |
| model | `edModelVar` per modality, on PCA(512) of each modality, standardised by the global mean and SD of the training PCA scores |
| encoder | 512 → 1024 → 512 (BatchNorm, LeakyReLU, dropout 0) |
| latent | variational mean and log-variance, 32-d; modalities combined through the correspondence P = I |
| decoder | 32 → 512 → 1024 → 512 |
| loss | weights [1, 1, 1, 1] on: KL × 0.032 × sigmoid annealing (midpoint epoch 250); reconstruction MSE in PCA space; 32 × cosine/Euclidean alignment of each embedding to the combined one; the F-term ‖combined₀ − F·combined₁‖² with the Prime-Dual F |
| training | Adam, learning rate 1e-3; batch 128 drawn per step without replacement; 54 steps per epoch; gradient-norm clip 1 |
| stopping | at most 10,000 epochs, at least 500; then stop after 500 epochs without improving the epoch's best batch loss by more than 1e-8 |
| prediction | `decoder_to(μ(encoder_from(x)))` in eval mode, mapped back by the inverse standardisation and inverse PCA. Predictions therefore live in **raw-count space** for RNA and **z-scored space** for ATAC |
| truth files | ATAC→RNA: raw HVG counts. RNA→ATAC: z-scored peaks. Hence the count-scale MSE (contract §8) |

**A reference quirk, reproduced.** `edModelVar.refactor` returns `logvar`, the last modality's log-variance, instead of the list. The KL term's `logvars[i]` is therefore *row i* of that tensor, broadcast against modality i's means. Epione computes the KL the same way and documents it in the code.

**Epione differences, all in execution.**
- Geodesic distances use Dijkstra on the same kNN graph, not Floyd–Warshall. The shortest paths are identical; Floyd–Warshall is O(n³).
- GPU throughout.

## Setting

| | |
|---|---|
| data | 10x Multiome PBMC 10k |
| split | fold0: 6,933 train, 771 validation (unused, as in the reference), 1,927 test |
| seeds | 0, 1, 2 on both sides |
| RNA input | `inputs/rna_counts_hvg.h5ad` |
| ATAC input | the ≥ 49-cell peaks of `inputs/atac_counts.h5ad`, z-scored over all cells, as the reference runner does |

## Gates

The structure, tolerances and floors follow the earlier MANIFESTs.

- **G0.** Epione's RNA counts and z-scored ATAC equal the reference truth (`allclose(rtol=1e-5, atol=1e-6)`).
- **G1.** Epione's metrics on the reference predictions reproduce the reference `metrics.json` under `convention="sctranslation"`.
- **G2.** `|mean_Epione − mean_reference| ≤ max(2 × SD_reference, floor)`, using 3 seeds a side.
  - **Exempt from gating, reported only:**
    - rna2atac MSE: the reference ranges 76–864 across seeds, as agent A noted;
    - rna2atac MMD: degenerate at about 0, because extreme values inflate the kernel bandwidth;
    - atac2rna AUROC: as before.
  - That leaves **21 gated pairs**.
  - One amendment at most, registered here first and based on config, log or code facts.
- **G3.**
  - Setting: `fast=True` is bf16 autocast with batch 512, JAMIE's own default batch size; the learning rate is unchanged.
  - Metric gate: the same rule as G2, against Epione's default.
  - Speed gate: ≥ 1.5×.
  - If it fails, fast mode is documented as **not equivalent**, with no amendments.

## Results (2026-10-13)

**Vendored vs upstream** (`ast.unparse` diff against daifengwanglab/JAMIE@802148b). Upstream adds sparse-matrix support for P/F and refactors only. **No deviation that changes the model or its predictions.**

| gate | result |
|---|---|
| G0 preprocessing | **PASS**: both truths identical |
| G1 metric implementation | non-clustering **PASS** (max \|Δ\| ≤ 8e-4); clustering: **1 miss** (rna2atac seed 0 ARI: 0.4941 vs the reference's 0.4707; tolerance 0.02) |
| **G2 training parity** | **FAIL: 20 / 21** gated pairs pass |
| G3 fast mode | speed **PASS** (median 2.19×); metrics **FAIL** (13 / 21) |

### G2: mean of 3 seeds, reference vs Epione

| direction | metric | reference | Epione | \|Δ\| | tol | result |
|---|---|---|---|---|---|---|
| rna2atac | PCC | 0.1457 ± 0.0009 | 0.1452 | 0.0005 | 0.0100 | pass |
| rna2atac | SPCC | 0.1636 ± 0.0048 | 0.1707 | 0.0072 | 0.0100 | pass |
| rna2atac | MSE | 396.7626 ± 413.5406 | 341.7235 | 55.0391 | 827.0811 | (exempt) |
| rna2atac | MAE | 0.7013 ± 0.2029 | 0.7029 | 0.0015 | 0.4059 | pass |
| rna2atac | CORR | 0.0066 ± 0.0038 | 0.0052 | 0.0013 | 0.0100 | pass |
| rna2atac | MMD | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | (exempt) |
| rna2atac | LISI | 0.6955 ± 0.0233 | 0.6972 | 0.0017 | 0.0466 | pass |
| rna2atac | AUROC | 0.5601 ± 0.0025 | 0.5641 | 0.0039 | 0.0051 | pass |
| rna2atac | ARI | 0.4730 ± 0.0021 | 0.5047 | 0.0316 | 0.0300 | **fail** |
| rna2atac | AMI | 0.6887 ± 0.0026 | 0.7032 | 0.0145 | 0.0300 | pass |
| rna2atac | NMI | 0.6972 ± 0.0024 | 0.7113 | 0.0141 | 0.0300 | pass |
| rna2atac | HOM | 0.7267 ± 0.0036 | 0.7364 | 0.0097 | 0.0300 | pass |
| atac2rna | PCC | 0.9655 ± 0.0001 | 0.9655 | 0.0000 | 0.0100 | pass |
| atac2rna | SPCC | 0.3195 ± 0.0049 | 0.3165 | 0.0029 | 0.0100 | pass |
| atac2rna | MSE | 14.2418 ± 0.3328 | 14.0482 | 0.1937 | 0.7121 | pass |
| atac2rna | MAE | 0.2015 ± 0.0020 | 0.2004 | 0.0011 | 0.0101 | pass |
| atac2rna | CORR | 0.1581 ± 0.0034 | 0.1600 | 0.0019 | 0.0100 | pass |
| atac2rna | MMD | 0.0832 ± 0.0083 | 0.0886 | 0.0054 | 0.0166 | pass |
| atac2rna | LISI | 0.6559 ± 0.0427 | 0.6616 | 0.0057 | 0.0853 | pass |
| atac2rna | AUROC | 0.5561 ± 0.0009 | 0.5556 | 0.0005 | 0.0050 | (exempt) |
| atac2rna | ARI | 0.3416 ± 0.0426 | 0.3915 | 0.0499 | 0.0851 | pass |
| atac2rna | AMI | 0.5983 ± 0.0161 | 0.6118 | 0.0135 | 0.0321 | pass |
| atac2rna | NMI | 0.6104 ± 0.0149 | 0.6231 | 0.0127 | 0.0300 | pass |
| atac2rna | HOM | 0.6544 ± 0.0092 | 0.6602 | 0.0058 | 0.0300 | pass |

**The one failure: rna2atac Leiden ARI**, 0.505 vs 0.473 ± 0.002.
- **Environment effect.** Leiden itself differs between the two software environments. G1 shows Epione's environment gives ARI 0.494, 0.474 and 0.469 (mean 0.479, SD 0.013) on the *reference's own* predictions.
- **Same-environment comparison** (reported for information, not as a re-registered gate): Epione's mean is 0.026 from 0.479, inside the 0.03 floor.
- **Spread.** Epione's own seeds range 0.44–0.54.
- **AMI, NMI and HOM** for the same predictions pass.

The pre-registered rule compares against the reference `metrics.json`, so **G2 is recorded as FAIL (20/21)**. No amendment was made.

**Exempt pairs, as pre-registered.** rna2atac MSE: reference 397 ± 414, Epione 342. rna2atac MMD: both 0.0.

### G3

`fast=True` (bf16, batch 512) fails:

| direction | metric | default | fast | result |
|---|---|---|---|---|
| rna2atac | SPCC | 0.1707 | 0.1525 | **fail** |
| rna2atac | MAE | 0.7029 | 0.3731 | **fail** |
| rna2atac | CORR | 0.0052 | 0.0853 | **fail** |
| rna2atac | AMI | 0.7032 | 0.6664 | **fail** |
| rna2atac | NMI | 0.7113 | 0.6753 | **fail** |
| rna2atac | HOM | 0.7364 | 0.6938 | **fail** |
| atac2rna | SPCC | 0.3165 | 0.3013 | **fail** |
| atac2rna | MAE | 0.2004 | 0.2169 | **fail** |

Training seconds:
- default: 6153, 7221, 5556;
- fast: 2808, 2676, 2991.

`fast=True` is documented as **not equivalent**.

**Timing vs reference.** The reference took 5708, 1627, 10140 s and Epione's default 6153, 7221, 5556 s. Both are dominated by stopping on the training loss, and the shared GPU's load varied, so this is not reported as a speed comparison.
