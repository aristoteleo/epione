# Trajectory port — pre-registered parity gates

Reference: ArchR 1.0.3 (`addTrajectory`, `getTrajectory`, `plotTrajectoryHeatmap`,
`correlateTrajectories`, `plotTrajectory`) on the ArchR Hematopoiesis tutorial
project (10,660 cells; IterativeLSI 30 dims; UMAP), trajectory `C11 → C10 → C5`
(CD34⁺ progenitor → MPO⁺ GMP → CD14⁺ monocyte), `groupBy = "Clusters"`.

The ArchR reference outputs are generated outside this repository with ArchR
1.0.3 and read from `$EPIONE_ARCHR_PARITY_DIR/trajectory/` (the exported
GeneScoreMatrix / MotifMatrix tables in its `tables/` subdirectory).
`parity_trajectory.py` and `tests/test_trajectory.py` compare against them and
skip when the variable is not set.

All ported functions are deterministic (ArchR sets a seed but none of these
code paths draws random numbers), so every output is gated as class 1
(`deterministic-standard`, `atol = 1e-8`) unless stated otherwise. Discrete
outputs (cell sets, row orders, kNN indices) must match exactly.

Thresholds were written before the first comparison and are never widened.

| # | Output | R object | Class | Metric | Threshold |
|---|---|---|---|---|---|
| 1 | initial alignment time | `initialTime` (internal) | 1 deterministic-standard | max abs err | 1e-8 |
| 2 | fitted spline per dim | `matSpline` = `smooth.spline(...)[[2]]` | 1 deterministic-standard | max abs err | 1e-8 |
| 3 | nearest spline points | `nabor::knn(k = 3)` indices | exact | fraction identical | 1.0 |
| 4 | cells retained | `rownames(dfTrajectory)` | exact | set equality | 1.0 |
| 5 | `Distance`, `DistanceIdx` | `saveDF` frame | 1 deterministic-standard | max abs err | 1e-8 |
| 6 | pseudo-time | `cellColData$Trajectory` | 1 deterministic-standard | max abs err | 1e-8 |
| 7 | 5–6 for `embedding = "UMAP"` and `useAll = TRUE` | same | same | same | same |
| 8 | binned matrix | `assay(getTrajectory(...), "mat")` GeneScore / Motif / `groupEvery = 2` | 1 deterministic-standard | max abs err | 1e-8 |
| 9 | smoothed matrix | `assay(..., "smoothMat")` | 1 deterministic-standard | max abs err (NaN positions identical) | 1e-8 |
| 10 | bin labels | `colData(se)$label` | exact | fraction identical | 1.0 |
| 11 | heatmap matrix | `plotTrajectoryHeatmap(returnMatrix = TRUE)` GeneScore / Motif / non-default options | exact rows+order, values class 1 | max abs err | 1e-8 |
| 12 | correlations | `correlateTrajectories()$allMappings` (Correlation, VarAssay1/2, TStat) | 1 deterministic-standard | max abs err | 1e-8 |
| 13 | p-values / FDR | `Pval`, `FDR` | 1 deterministic-standard | max abs err | 1e-8 |
| 14 | correlated pairs | `$correlatedMappings` | exact | set equality | 1.0 |
| 15 | arrow path / scatter values | `plotTrajectory()` geom_path data and panel-2 data (no imputation) | 1 deterministic-standard | max abs err | 1e-8 |

Not gated (visual only): the `geom_smooth` (mgcv GAM) line of panel 2 of
`plotTrajectory` and ComplexHeatmap/ggplot rendering. MAGIC imputation inside
`plotTrajectory` is accepted as an input (`impute_weights=`) but gated by the
imputation port, not here.

## Results (appended after the run; thresholds above unchanged)

All 57 checks pass. Worst value per gate (LSI, UMAP and `useAll = TRUE` runs):

| # | Value | Pass |
|---|---|---|
| 1 | 0 (names identical) | ✅ |
| 2 | LSI 8.4e-11 / UMAP 1.7e-10 | ✅ |
| 3 | identical (kNN distances 1.8e-10) | ✅ |
| 4 | identical | ✅ |
| 5 | `Distance` 1.8e-10 / `DistanceIdx` 9.1e-13 | ✅ |
| 6 | 2.8e-14 | ✅ |
| 7 | as 5–6 | ✅ |
| 8 | GeneScore 8.9e-16, `groupEvery = 2` 1.8e-15, Motif 1.7e-13 | ✅ |
| 9 | GeneScore 8.9e-16, `groupEvery = 2` 1.8e-15, Motif 5.7e-14 | ✅ |
| 10 | identical | ✅ |
| 11 | rows identical; values ≤ 6.9e-15 (3 configurations) | ✅ |
| 12 | ≤ 2.7e-14 | ✅ |
| 13 | `Pval` 3.8e-15 / `FDR` 3.9e-15 | ✅ |
| 14 | identical (33 pairs) | ✅ |
| 15 | ≤ 2.8e-14 | ✅ |
