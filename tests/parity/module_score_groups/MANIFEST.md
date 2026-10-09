# Parity manifest — addModuleScore / plotGroups / confusionMatrix

Reference: ArchR 1.0.3 (R 4.4.3) on the ArchR Hematopoiesis tutorial project
(10,660 cells, 23,127-gene GeneScoreMatrix). Thresholds were registered before
the first comparison and are never loosened.

Taxonomy: omicverse-rebuildr `PARITY_TAXONOMY.md`.

| # | Output | Class | Metric | Threshold |
|---|--------|-------|--------|-----------|
| 1 | `confusionMatrix(Clusters, Sample)` counts + row/col order | 1 deterministic (strict) | max abs err; identical labels/order | `== 0` |
| 2 | addModuleScore feature binning (`rS` order, `Bins`) | 1 deterministic (strict) | exact equality | `== 0` |
| 3 | addModuleScore background features, R draws injected | 1 deterministic | exact set equality | `== 0` |
| 4 | addModuleScore per-cell scores, R draws injected | 1 deterministic (standard) | max abs err | `<= 1e-8` |
| 5 | addModuleScore per-cell scores, free-running (`seed=1`, R RNG emulation) | 1 deterministic (standard) | max abs err | `<= 1e-8` |
| 5b | (fallback if 5 cannot be exact) free-running scores | 6 ordinal | Pearson r | `>= 0.99` |
| 6 | plotGroups sampled cells (`maxCells`, `set.seed`) | 1 deterministic (strict) | exact set equality | `== 0` |
| 7 | plotGroups plotted values (after `quantCut`) | 1 deterministic (standard) | max abs err | `<= 1e-8` |
| 8 | ggGroup group order (ridges / violin) | 1 deterministic (strict) | exact equality | `== 0` |
| 9 | ridge density curves (`ggridges`, joint `bw.nrd0`) | 1 deterministic (bounded) | max abs err on density | `<= 1e-6` |
| 10 | violin densities (`stat_ydensity`, per-group `nrd0`, `scale="area"`) | 1 deterministic (bounded) | max abs err on density / violinwidth | `<= 1e-6` |
| 11 | boxplot statistics (lower/middle/upper/whiskers) | 1 deterministic (standard) | max abs err | `<= 1e-8` |

Row 5b is only reported if row 5 fails; it does not replace row 5.

## Results (ArchR 1.0.3, R 4.4.3, ggplot2 4.0.3, ggridges 0.5.7)

| # | Output | Value | Gate |
|---|--------|-------|------|
| 1 | confusionMatrix counts + order | 0 (identical order) | pass |
| 2 | feature totals with `sample_key="Sample"` vs Arrow `rowSums` | bitwise identical | pass |
| 3 | background features, R draws injected / free-running (10 configs) | identical (all 10) | pass |
| 4 | module scores, R draws injected | 3.6e-14 | pass |
| 5 | module scores, free-running `seed` | 3.6e-14 | pass |
| 6 | plotGroups sampled cells (10 cases) | identical | pass |
| 7 | plotGroups values after quantCut | 2.8e-14 | pass |
| 8 | group order (ridges / violin) | identical | pass |
| 9 | ridge densities | 5.9e-14 | pass |
| 10 | violin densities / violinwidth | 5.7e-14 | pass |
| 11 | box statistics | 1.8e-15 | pass |

Not part of a gate (convenience mode): without `sample_key`, feature totals are summed over
all cells at once; 20 of 23,127 genes change rank, which changes some background draws
(max |Δ| 0.55, min Pearson 0.9996 across configs).

Module-score references: "official" = unmodified ArchR, one call per multi-gene module;
"patched_all" = all modules in one call with ArchR's two module-assignment bugs fixed
(`length(idx[[z]])` -> `length(features[[z]])`, `drop = FALSE`). Where both run, they are
identical (asserted in `r_reference.R`).
