# Parity manifest — `var_deviations` / `correlate_matrices` / `positive_tf_regulators`

Pre-registered before the first R-vs-Python comparison. Thresholds are
read-only (omicverse-rebuildr PARITY_TAXONOMY); they are never widened.

Reference: ArchR 1.0.3 (R 4.4.3) on the ArchR Hematopoiesis tutorial project
(10,660 cells, 3 samples). R driver: `r_reference.R`; Python driver:
`py_parity.py`.

| # | Output | ArchR source | Class | Metric | Threshold |
|---|--------|--------------|-------|--------|-----------|
| 1 | `combinedVars` per motif (fixture `MotifMatrix`, JASPAR2020) | `getVarDeviations` → `.getRowVars` → `.combineVariances` | 1 (deterministic, standard) | max abs err | ≤ 1e-8 |
| 2 | `combinedMeans` per motif | same | 1 standard | max abs err | ≤ 1e-8 |
| 3 | variability rank | same | 1 | exact equality | 0 mismatches |
| 4 | same three outputs on the cisbp `MotifMatrix` | same | 1 standard | max abs err / exact rank | ≤ 1e-8 / 0 |
| 5 | kNN cell groups after overlap filter (R's sampled seed cells injected) | `.computeKNN` + `determineOverlapCpp` | 1 (integer) | exact set equality per group | 0 mismatches |
| 6 | group-mean matrices (GeneScore log2, Motif z) | `.getGroupMatrix` / k, `log2(x+1)` | 1 standard | max abs err | ≤ 1e-8 |
| 7 | `cor` per gene–motif pair (injected groups) | `.rowCorTest` (`cor.test`) | 1 standard | max abs err | ≤ 1e-8 |
| 8 | `-log10(pval)`, `-log10(padj)` (Bonferroni) (injected groups) | `.rowCorTest` | 1 bounded | max abs err | ≤ 1e-6 |
| 9 | `maxDelta` per motif (`getGroupSE` by Clusters) | tutorial workflow | 1 standard | max abs err | ≤ 1e-8 |
| 10 | positive TF regulator set (injected groups) | tutorial workflow | 7 (classification) | exact set equality | identical |
| 11 | free-running `cor` (Python's own seed-cell draw) | `correlateMatrices` | 8 (statistical inference) | Spearman on `cor` and on `-log10 padj` | ≥ 0.90 |
| 12 | free-running positive TF regulator set | tutorial workflow | 5 (ranked) | Jaccard | ≥ 0.80 |

Notes
- Inputs are identical on both sides: the IterativeLSI matrix exactly as
  returned by `getReducedDims(corCutOff = 0.75, dimsToUse = 1:30)`, the
  GeneScoreMatrix / MotifMatrix values exported from the same Arrow files.
- Row 11/12 additionally get a class-1 check if the Python port reproduces R's
  `set.seed(1); sample()` stream (`rng="R"` mode).
