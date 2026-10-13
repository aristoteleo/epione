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

**Vendored vs upstream** (daifengwanglab/JAMIE@802148b): checked with `ast.unparse` before the runs; see "Results".

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

## Results

Filled in after the runs.
