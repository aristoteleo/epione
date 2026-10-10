# Parity manifest — `var_deviations` / `correlate_matrices` / `positive_tf_regulators`

Pre-registered before the first R-vs-Python comparison. Thresholds are
read-only (omicverse-rebuildr PARITY_TAXONOMY); they are never widened.

Reference: ArchR 1.0.3 (R 4.4.3) on the ArchR Hematopoiesis tutorial project
(10,660 cells, 3 samples). The ArchR reference outputs are generated outside this
repository with ArchR 1.0.3; the repository contains no R code. The Python driver
`py_parity.py` and `test_parity_with_archr` read them from
`$EPIONE_ARCHR_PARITY_DIR/var_dev_correlate/` and are skipped when the variable is
not set.

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

## Results (ArchR 1.0.3, R 4.4.3)

| # | Output | Value | Gate |
|---|--------|-------|------|
| 1 | `combinedVars` (JASPAR2020 / cisbp) | 2.8e-14 / 4.3e-14 | pass |
| 2 | `combinedMeans` (JASPAR2020 / cisbp) | 6.4e-16 / 7.1e-16 | pass |
| 3, 4 | variability rank (JASPAR2020 / cisbp) | 0 / 0 mismatches | pass |
| — | row-scaled LSI input (`scaleDims`) | 1.4e-14 | (input check) |
| 5 | seed cells (`rng="archr"`) / kNN groups after overlap filter | 0 of 500 / identical (492 groups) | pass |
| 6 | cluster means (`getGroupSE`) | 1.8e-14 | pass |
| 7 | `cor`, 825 gene–motif pairs | 8.3e-16 | pass |
| 8 | −log10 pval / −log10 padj | 3.8e-13 / 3.4e-13 | pass |
| 9 | `maxDelta` | 4.4e-14 | pass |
| 10 | positive TF regulator set (39 TFs) | identical, same row order | pass |
| 11 | `rng="numpy"`: Spearman on `cor` / on −log10 padj | 0.989 / 0.954 | pass |
| 12 | `rng="numpy"`: regulator-set Jaccard | 0.884 | pass |

With `rng="archr"` (default) the seed cells come from an emulation of R's RNG
(Mersenne-Twister and L'Ecuyer-CMRG, the latter being what ArchR uses after
`addArchRThreads(threads > 1)`), so the free-running result is identical to ArchR;
`rng="numpy"` is the only non-1:1 mode, by design. The JASPAR2020 MotifMatrix is named
by matrix ID, so the correlation / TF-regulator rows use the cisbp motif set from the
ArchR tutorial, computed in the same ArchR project.
