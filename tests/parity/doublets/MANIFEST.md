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

None yet. The results are appended below only after the runs.
