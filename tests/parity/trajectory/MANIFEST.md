# Trajectory port — pre-registered parity gates

Reference: ArchR 1.0.3 (`addTrajectory`, `getTrajectory`, `plotTrajectoryHeatmap`,
`correlateTrajectories`, `plotTrajectory`) on the ArchR Hematopoiesis tutorial
project (10,660 cells; IterativeLSI 30 dims; UMAP), trajectory `C11 → C10 → C5`
(CD34⁺ progenitor → MPO⁺ GMP → CD14⁺ monocyte), `groupBy = "Clusters"`.

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
