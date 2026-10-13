# Doublet detection: pre-registered gates

Written on 2026-10-13, before any Epione doublet code existed and before any run.

## Scope

| Function | What it is | Parity owned by |
|---|---|---|
| `epi.pp.archr_doublets` | Port of ArchR 1.0.3 `addDoubletScores` (MIT) | this MANIFEST (gates A1–A3) |
| `epi.pp.filter_doublets` | Removal step; one mode reproduces ArchR `filterDoublets` | this MANIFEST (gate F1) |
| `epi.pp.scdblfinder`, `epi.pp.amulet`, `epi.pp.clamulet` | Thin wrappers over `pyscdblfinder>=0.3.0` (omicverse/py-scDblFinder) | py-scDblFinder's own parity suite. Epione tests only the plumbing (gate W1). |
| `epi.pp.scrublet` | Already on main | not re-tested |

scDblFinder and AMULET are deliberately **not** reimplemented in Epione, because the omicverse org keeps a single port in py-scDblFinder.

## References

The reference environment is outside the repository: `epione_dev/bin/Rscript-archr` (R 4.4.3, ArchR 1.0.3, scDblFinder 1.23.4, xgboost 3.0.5.1). It is used read-only, and its scripts live in `epione_dev/doublet_ref/scripts/`.

- **Parity fixture.** The ArchR hematopoiesis tutorial project (`epione_dev/fixture/ArchR_base`): 3 samples, BMMC, CD34 BMMC and PBMC. ArchR runs on a *copy* of its Arrow files, because `addDoubletScores` writes into Arrow files.
- **Benchmark data.** GEO GSE162690 (Granja et al. 2021): 10x scATAC-seq of a mixture of 11 cell lines at low and high loading.
  - Genotype calls come from demuxlet (`DemuxletClassify` in the published `*-PeakMatrix-SE.rds`).
  - The cell sets are the published ones: 11,173 cells (Low) and 16,047 cells (High).
  - Truth is `DBL` = doublet and a cell-line name = singlet. `AMB` cells (5 and 16) are excluded from metrics.
  - Demuxlet cannot see doublets of two cells from the same line, so some "singlets" are homotypic doublets. With 11 lines this is a minority, but it caps the attainable precision for every method equally.

## Gate A: ArchR port vs ArchR 1.0.3

**Inputs, identical on both sides.** For each sample:
- the binarised 500-bp `TileMatrix` exported from ArchR's Arrow files, restricted to the project's cells;
- `nFrags`.

**Epione.** `epi.pp.archr_doublets` with defaults equal to ArchR's: `k=10`, `nTrials=5`, `dimsToUse=1:30`, `LSIMethod=1`, `scaleDims=FALSE`, `corCutOff=0.75`, `knnMethod="UMAP"`, and UMAP with `n_neighbors=40`, `min_dist=0.4`, euclidean metric. It runs per sample, as ArchR does.

**Known, accepted implementation differences.** These are why the gates are metric-level rather than exact:
- the RNG for cell sampling, tile down-sampling and UMAP initialisation;
- irlba vs a randomized/ARPACK truncated SVD;
- Seurat Louvain (resolution 2) inside the iterative LSI vs igraph Leiden at resolution 2. The `maxClusters=6` hclust (complete linkage) merge is kept;
- uwot vs umap-learn, including `umap_transform`.

**Noise baseline.** ArchR itself is re-run with seeds 1, 2 and 3. The seed enters `LSIParams$seed`, the doublet simulation seed and the UMAP `set.seed`, via namespace wrappers in the reference script; ArchR's code is otherwise unmodified. For each sample, the baseline B is the *lowest* pairwise agreement among the 3 ArchR runs.

Every gate threshold is `min(fixed, B)`, computed per sample. This rule was decided now and won't change after seeing results.

| Gate | Statistic (per sample, Epione seed 1 vs ArchR seed 1) | Fixed threshold |
|---|---|---|
| **A1** | Spearman ρ of `DoubletEnrichment` | 0.70 |
| **A2** | Jaccard of the cells that `filterDoublets(filterRatio=1)` removes, each side applied to its own enrichment | 0.50 |
| **A3** | Spearman ρ of `DoubletScore` | 0.60 |

A gate passes only if all 3 samples pass. Failures are recorded, not tuned away. If a gate fails, at most **one** amendment may be written here, before its run, as for the SCENIC+ and translate ports.

## Gate F1: `filter_doublets(method="archr")` vs ArchR `filterDoublets`

This step is deterministic, so the gate is **exact**. ArchR's `filterDoublets(cutEnrich=1, cutScore=-Inf, filterRatio=1)` is run on the ArchR project carrying ArchR's own `DoubletEnrichment`/`DoubletScore` (seed-1 run). Epione's `filter_doublets(method="archr")`, applied to the same values and `Sample`, must remove exactly the same set of cells in every sample.

The details to reproduce:
- a stable descending order;
- keeping `head(n · filterRatio · n / 1e5)`, where R truncates toward zero.

## Gate W1: wrapper plumbing (CI, no downloads)

A fake `pyscdblfinder` module, injected into `sys.modules`, checks that:
- AnnData counts reach `sc_dbl_finder` as features × cells;
- the obs-name ↔ fragment-barcode mapping (`barcode_key`, the `sample#barcode` convention) round-trips;
- per-sample splitting calls the backend once per sample;
- scores and classes land in `adata.obs` under the documented keys;
- a missing or too-old `pyscdblfinder` raises an informative `ImportError` that names the extra.

## Benchmark (descriptive; reported in full whatever it shows)

On GSE162690 Low and High, the following are reported:
- AUROC and AUPRC (average precision) per method;
- precision and recall at each method's own default call.

| Method | Source |
|---|---|
| `scrublet` | `epi.pp.scrublet` on the Epione 500-bp tile matrix |
| ArchR port, enrichment and score | `epi.pp.archr_doublets` on the same tile matrix |
| ArchR (published) | `DoubletEnrichment` in the GEO SE, from the authors' own run |
| AMULET | `epi.pp.amulet` (pyscdblfinder). If pyscdblfinder 0.3.0 isn't released when the benchmark runs, R `scDblFinder::amulet` 1.23.4 is reported instead and labelled as such. |
| scDblFinder-ATAC | `epi.pp.scdblfinder(aggregate_features=True, n_features=25, processing="normFeatures")`, with the same R fallback rule |
| clamulet | `epi.pp.clamulet`, with the same R fallback rule |
| Fisher combination | AMULET p-value and 1 − scDblFinder score, each clipped at 0.001, combined with Fisher's method as recommended by the scDblFinder vignette |

**Pre-registered gate on the benchmark: B1.** Per dataset, |AUPRC(ArchR port) − AUPRC(ArchR published)| ≤ 0.05. This checks that the port is as useful as ArchR on real ground truth, not just correlated with it on the fixture.

No method is declared "best" unless it leads on both datasets.

## Results

## Results, gate A as registered (2026-10-13): **FAIL**

Epione seed 1 vs ArchR seed 1. Each cell shows value / threshold, where the threshold is min(fixed, ArchR's worst seed pair).

| sample | cells | A1 enrichment ρ | A2 filter Jaccard | A3 score ρ |
|---|---|---|---|---|
| scATAC_BMMC_R1 | 4,932 | 0.571 / 0.609 **fail** | 0.455 / 0.500 **fail** | 0.656 / 0.600 pass |
| scATAC_CD34_BMMC_R1 | 3,275 | 0.619 / 0.636 **fail** | 0.196 / 0.132 pass | 0.564 / 0.576 **fail** |
| scATAC_PBMC_R1 | 2,453 | 0.373 / 0.387 **fail** | 0.558 / 0.500 pass | 0.349 / 0.352 **fail** |

- Every miss lies within 0.04 of ArchR's own worst seed-to-seed agreement.
- Both sides remove the same *number* of cells (243, 107 and 60), because that number is fixed by the n²/10⁵ rule.
- **Epione is more reproducible than ArchR.** Its own seed-to-seed enrichment ρ is 0.69–0.70 (BMMC), 0.78–0.79 (CD34) and 0.37–0.43 (PBMC). ArchR's worst pairs are 0.61, 0.64 and 0.39.

**Diagnostic, run after the result and not a gate (BMMC).**

| Comparison | Spearman ρ |
|---|---|
| ArchR's own LSI-space vs UMAP-space enrichment | 0.42 |
| Epione vs ArchR, LSI space | 0.65 |
| Epione vs ArchR, UMAP space | 0.57 |

The disagreement is mostly the stochasticity of the score itself, compounded by UMAP. A step-by-step re-check of the port found no algorithmic difference: feature selection, number of trials, down-sampling, projection and count mapping all match.

## Amendment A-1 (the one allowed), written before its run

**Rationale.** The registered statistic compares a *single* Epione–ArchR draw with the *minimum* of three ArchR–ArchR draws. With ArchR's self-agreement spanning 0.39–0.79 across samples, one draw is too noisy to be the test statistic.

**Amended statistic.** For each gate and sample, take the mean of the statistic over all 9 pairs (Epione seed i, ArchR seed j), with i, j ∈ {1, 2, 3}.

**Threshold.** Unchanged: min(fixed, ArchR's worst seed pair).

**Rule.** Pass/fail as before; all 3 samples must pass. The registered result above stays recorded as FAIL whatever A-1 shows.

### Seeding fix, found after the runs above

After the runs above, `archr_doublets` turned out to be non-deterministic for a fixed seed: igraph's Louvain draws from Python's global `random` module. The numbers above come from commit 5b7279d, before the fix. The fix seeds `random` locally and restores its state; `test_archr_doublets_is_deterministic` covers it.

Gate A was re-run on the fixed code: driver `gate_a_amend.py`, output `analysis/gate_a_amendment.json`. The registered statistic (Epione seed 1 vs ArchR seed 1) on the fixed code:

| sample | A1 | A2 | A3 | result |
|---|---|---|---|---|
| scATAC_BMMC_R1 | 0.574 | 0.429 | 0.641 | A1, A2 **fail** |
| scATAC_CD34_BMMC_R1 | 0.635 | 0.182 | 0.577 | A1 **fail** by 0.001; A2, A3 pass |
| scATAC_PBMC_R1 | 0.313 | 0.395 | 0.294 | **fail** all three |

## Results, amendment A-1 (fixed code): **FAIL**

Each cell is the mean over the 9 (Epione seed, ArchR seed) pairs / threshold.

| sample | A1 enrichment ρ | A2 filter Jaccard | A3 score ρ |
|---|---|---|---|
| scATAC_BMMC_R1 | 0.572 / 0.609 **fail** | 0.412 / 0.500 **fail** | 0.649 / 0.600 pass |
| scATAC_CD34_BMMC_R1 | 0.666 / 0.636 pass | 0.147 / 0.132 pass | 0.621 / 0.576 pass |
| scATAC_PBMC_R1 | 0.343 / 0.387 **fail** | 0.573 / 0.500 pass | 0.329 / 0.352 **fail** |

Gate A does not pass, either as registered or as amended. Epione's ArchR scores correlate with ArchR's slightly less well than ArchR's correlate with themselves across seeds; the gap in ρ is 0.02–0.05. The Epione docstring states that parity is metric-level and points here.

## Results, gate F1: **PASS (exact)**

`filter_doublets(method="archr")` on ArchR's own seed-1 scores removes the identical cell set that ArchR `filterDoublets` removes: 243 + 107 + 60 cells. `test_gate_f1_filter_matches_archr_exactly` re-checks this whenever `EPIONE_DOUBLETS_PARITY_DIR` is set.

## Results, benchmark as registered (published cell sets)

Truth is demuxlet DBL vs singlets (AMB excluded). AUPRC of a random score equals the doublet fraction: 0.030 (Low) and 0.066 (High).

| dataset | method | AUROC | AUPRC | called | precision | recall |
|---|---|---|---|---|---|---|
| Low | ArchR (published) | 0.489 | 0.038 | | | |
| Low | ArchR port, enrichment | 0.483 | 0.037 | 1,248 | 0.046 | 0.173 |
| Low | scrublet | 0.555 | 0.113 | 21 | 0.619 | 0.039 |
| Low | AMULET (pyscdblfinder branch) | 0.478 | 0.030 | 148 | 0.034 | 0.015 |
| Low | scDblFinder-ATAC (R 1.23.4) | 0.535 | **0.176** | 441 | 0.170 | 0.223 |
| Low | Fisher(AMULET, scDblFinder) | 0.515 | 0.107 | | | |
| Low | clamulet (R 1.23.4) | 0.352 | 0.031 | | | |
| High | ArchR (published) | 0.654 | 0.179 | | | |
| High | ArchR port, enrichment | 0.531 | 0.085 | 2,571 | 0.098 | 0.236 |
| High | scrublet | 0.678 | 0.217 | 780 | 0.308 | 0.225 |
| High | AMULET (pyscdblfinder branch) | 0.484 | 0.066 | 509 | 0.063 | 0.030 |
| High | scDblFinder-ATAC (R 1.23.4) | 0.628 | **0.228** | 2,260 | 0.165 | 0.351 |
| High | Fisher(AMULET, scDblFinder) | 0.595 | 0.116 | | | |

**Deviations from the plan, all disclosed.**
- **AMULET.** R `scDblFinder::amulet` did not finish in 4.5 h on the shared node; it was stopped. AMULET is therefore reported from the pyscdblfinder `feat/atac-mode` branch (0.3.0, not yet released), and the Fisher row uses that AMULET.
- **clamulet for High** was not run, because R clamulet took 1.5 h on Low.
- **Genome.** GSE162690 is aligned to **hg38**, checked at known TSSs. The tile grids were built on hg38 chromosome sizes. An earlier hg19-grid run, superseded, gave the same conclusions.

**Gate B1.** Low: |ΔAUPRC| = 0.001, **pass**. High: |ΔAUPRC| = 0.094, **fail**.

### Why the published cell sets are a biased benchmark (diagnostic, after the registered run)

The published cell sets were already filtered for doublets by the authors. Of the genotype doublets that pass ArchR's own QC (TSS ≥ 4, ≥ 1,000 fragments), the published High set keeps 13.9%; of the singlets, it keeps 90.9%. The doublets that remain are mostly ones ArchR-style scores miss.

That explains B1's failure as registered. ArchR's published scores were computed with all ~20.7k QC cells present, while the port was run only on the 16k filtered cells. On *identical* inputs (ArchR 1.0.3's own QC cells, High loading):

| scored on | ArchR port | ArchR 1.0.3, re-run | ArchR published |
|---|---|---|---|
| all 20,654 QC cells (28.9% doublets) | **0.825** | 0.776 | — |
| the published subset of them | 0.165 | 0.198 | 0.190 |

So the port is at least as good as ArchR on the same cells. B1 stays recorded as FAIL because the registered design compared different inputs.

## Supplementary benchmark: QC-only cells (post hoc, not pre-registered)

Given the bias above, every method was re-scored on QC-only cells: demuxlet-called barcodes with ≥ 1,000 unique fragments and no prior doublet removal, i.e. the cell set of the tutorial. Wrappers use the pyscdblfinder branch.

| dataset | cells | doublets | scrublet | ArchR port | scDblFinder | Fisher | AMULET | random |
|---|---|---|---|---|---|---|---|---|
| Low, AUPRC | 12,856 | 2,701 (21%) | **0.893** | 0.790 | 0.773 | 0.680 | 0.267 | 0.210 |
| Low, AUROC | | | **0.938** | 0.894 | 0.872 | 0.837 | 0.573 | 0.5 |
| High, AUPRC | 20,805 | 6,016 (29%) | **0.817** | 0.701 | 0.805 | 0.698 | 0.358 | 0.289 |
| High, AUROC | | | **0.900** | 0.829 | 0.868 | 0.822 | 0.579 | 0.5 |

**Findings.**
- Scrublet leads on both loadings. scDblFinder is close on High, and the ArchR port is close on Low.
- AMULET is weak, because these cells are shallow (median ≈ 5k fragments, against the 10–15k its authors recommend). Fisher-combining it with scDblFinder therefore lowers scDblFinder's AUPRC.
- This is one cell-line mixture. Cell lines are more distinct than cell types, which favours simulation-based methods, so these rankings should not be generalised.

The ArchR-port column was computed before the seeding fix. On the final code the executed tutorial gives 0.779 / 0.882 for Low (vs 0.790 / 0.894), with the same ranking.

**Runtime on the QC-only cells, High (20.8k cells, shared node).** Tile matrix 3,907 s; scrublet 1,042 s; ArchR port 219 s; scDblFinder 891 s; AMULET 148 s.
