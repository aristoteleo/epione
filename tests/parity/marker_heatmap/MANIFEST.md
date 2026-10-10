# Parity manifest — `epi.pl.marker_heatmap` / `epi.pl.peak2gene_heatmap`

Reference: ArchR 1.0.3 (`plotMarkerHeatmap`, `plotPeak2GeneHeatmap`; ComplexHeatmap
2.22.0) on the ArchR Hematopoiesis tutorial project (10,660 cells, 11 clusters,
hg19). The ArchR reference outputs are generated outside this repository (the
repository contains no R code); `parity_check.py` and `tests/test_marker_heatmap.py`
read them from `$EPIONE_ARCHR_PARITY_DIR/marker_heatmap/` and skip when it is unset. Gates were registered **before** the
first comparison and are never loosened (omicverse-rebuildr protocol).

## Deterministic port (same inputs fed to both implementations)

Marker statistics come from ArchR `getMarkerFeatures` (GeneScoreMatrix and
PeakMatrix, `groupBy = "Clusters"`, wilcoxon, bias = TSS + log10 nFrags) and are
handed unchanged to Epione. Nine `plotMarkerHeatmap` configurations cover
transpose, invert, `plotLog2FC`, `binaryClusterRows = FALSE`, `subsetMarkers`,
`clusterCols = FALSE`, `scaleRows = FALSE`, `log2Norm = FALSE` and an `abs()` /
`!` cutOff.

| Output | Class | Metric | Gate |
|---|---|---|---|
| `returnMatrix` values | 1 deterministic (standard) | max abs error | <= 1e-8 |
| `returnMatrix` row / column names and order | 1 | exact equality | identical |
| Drawn row order / column order (`row_order(draw(ht))`) | 1 | exact equality | identical |
| Row-label (anno_mark) labels | 1 | exact equality | identical |

Peak-to-gene: links, `seATAC` / `seRNA` KNN matrices and KNN membership come from
ArchR `addPeak2GeneLinks`. Two configurations: default (`k = 25`, `nPlot = 25000`,
`seed = 1`, ~66k links → subsampling path) and `k = 12`, `nPlot = 4000`, `seed = 7`.

| Output | Class | Metric | Gate |
|---|---|---|---|
| ATAC / RNA matrices with R's kmeans labels injected | 1 deterministic | max abs error | <= 1e-8 |
| `kmeansId`, column order, column groups, link rows (injected) | 1 | exact equality | identical |
| Free-running kmeans labels (Python Hartigan-Wong + R RNG) | 3 clustering | ARI vs R `k1$cluster` | >= 0.95 |
| Free-running matrices / ids | 1 | max abs error / equality | reported (gated only if labels identical) |

## End-to-end with Epione's own upstream steps (informational)

`epi.tl.find_marker_features` and `epi.tl.peak_to_gene` are different statistical
implementations from ArchR's `getMarkerFeatures` / `addPeak2GeneLinks`; their
parity is outside this port, so these numbers are **reported, not gated**.

| Output | Class | Metric |
|---|---|---|
| Marker Log2FC (GeneScore) | 5 ranked | Spearman vs ArchR |
| Per-cluster top-50 markers | 5 ranked | Jaccard vs ArchR |
| Heatmap marker set | 5 | Jaccard vs ArchR heatmap rows |
| Peak-gene correlation on shared pairs | 8 | Spearman vs ArchR |

## Results (ArchR 1.0.3 reference, 2026-10-09)

| Output | Value | Gate |
|---|---|---|
| `returnMatrix`, all 9 configurations | max abs error **0.0** (bit-identical) | pass |
| Row / column names, NaN pattern | identical (9/9) | pass |
| Drawn row / column order | identical (9/9), incl. ward.D2 + `reorder.dendrogram` | pass |
| anno_mark labels and positions | identical (9/9) | pass |
| P2G ATAC / RNA matrices (R kmeans injected) | max abs error **0.0**, both configs | pass |
| P2G kmeansId, column order, groups, links, kDF (injected) | identical, both configs | pass |
| P2G free-running kmeans | ARI **1.0** (labels identical), matrices max abs error 0.0 | pass |

Informational end-to-end (Epione upstream): gene-score Log2FC Spearman 0.79;
median per-cluster top-50 Jaccard 0.45; heatmap marker-set Jaccard 0.48
(2,519 ArchR vs 2,442 Epione); peak-gene correlation Spearman 0.89 on
1,097,714 shared pairs.

Exactness required porting R arithmetic: `rowMeans`/`colSums` accumulate in long
double, `matrixStats::rowVars` uses a refinement pass, `log2` comes from libm (numpy's
SIMD `log2` differs in the last ulp), and ArchR sessions use
`RNGkind("L'Ecuyer-CMRG")` after `addArchRThreads(threads > 1)`.
