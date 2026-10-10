# Parity manifest — `epi.tl.add_impute_weights` / `epi.tl.impute_matrix`

Reference: ArchR 1.0.3 `addImputeWeights` + `imputeMatrix` (R/Imputation.R),
run on the ArchR Hematopoiesis tutorial project
(10,660 cells, IterativeLSI, GeneScoreMatrix) with
`sampleCells = 5000, nRep = 2, k = 15, td = 3, ka = 4, epsilon = 1, seed = 1,
threads = 1, useHdf5 = FALSE`.

Registered **before** the first comparison; thresholds are read-only.

| # | Output | Mode | Class | Metric | Threshold |
|---|--------|------|-------|--------|-----------|
| 1 | Reduced-dim input (`getReducedDims`, row z-scored) | deterministic | 1 (bounded) | max abs err | ≤ 1e-6 |
| 2 | Block membership per rep (sampled cell sets) | R draws injected | 1 | exact set equality | identical |
| 3 | Per-rep, per-block diffusion weights `Wt` | R draws injected | 1 (bounded) | max abs err | ≤ 1e-6 |
| 4 | Imputed GeneScoreMatrix, gene panel | R draws injected | 1 (bounded) | max abs err (rtol vs max |value|) | ≤ 1e-6 |
| 5 | Block membership, weights, imputed panel | free-running, `rng="R"` (built-in R Mersenne-Twister + `sample()` emulation) | 1 (bounded) | same as rows 2–4 | same as rows 2–4 |
| 6 | Imputed panel | free-running, `rng="numpy"` | 2 / 6 (stochastic, correlation) | median per-gene Pearson vs ArchR seed 1 | ≥ 0.95 (reported next to the ArchR seed 1 vs seed 2 baseline) |

Notes
- Classes follow `omicverse-rebuildr/PARITY_TAXONOMY.md`. Class-1 "bounded" tier
  (1e-6) is used because the kNN + four sparse products are accumulated in a
  different floating-point order than CHOLMOD.
- ArchR reference outputs are generated outside this repository with ArchR 1.0.3
  (the repository contains no R code). `parity_check.py` and
  `tests/test_impute_magic.py` read them from
  `$EPIONE_ARCHR_PARITY_DIR/impute_magic/` (plus the exported `lsi.tsv`); the
  parity test is skipped when the variable is unset.
- ArchR's `.safelapply` with `threads > 1` draws each replicate's permutation
  inside a `parallel::mclapply` fork, and the forks do not continue the
  parent's `set.seed()` stream (checked: `set.seed(1); mclapply(1:2, ...)`
  differs from `set.seed(1); lapply(1:2, ...)`). The ArchR reference run therefore pins
  `threads = 1`, the setting the `rng="R"` mode reproduces; for any other run
  the blocks can be passed explicitly through `blocks=`.

## Results (ArchR 1.0.3, Heme fixture, 10,660 cells × 99-gene panel)

| # | Output | Mode | Value | Gate |
|---|--------|------|-------|------|
| 1 | Reduced dims | deterministic | max abs err 0 | PASS |
| 2 | Block membership | R draws injected | identical (2 reps × 2 blocks) | PASS |
| 3 | Diffusion weights | R draws injected | max abs err 5.6e-17 | PASS (strict tier) |
| 4 | Imputed panel | R draws injected | max abs err 1.1e-14 (rel 2.2e-15) | PASS (strict tier) |
| 5 | Blocks / weights / imputed | free-running `rng="R"` | identical / 5.6e-17 / 1.1e-14 | PASS (strict tier) |
| 6 | Imputed panel | free-running `rng="numpy"` | median gene Pearson 0.917 (min 0.483) | **FAIL** (< 0.95) |

Row 6 is reported as failed against its pre-registered threshold. The threshold
was mis-specified: ArchR run with seed 1 vs ArchR run with seed 2 reaches only
a median gene Pearson of 0.913 (min 0.539), so no independent random blocking —
including ArchR's own — clears 0.95. The disagreement is a property of ArchR's
algorithm (kNN graphs are built inside random ~5,000-cell halves of the data),
not of the port; rows 2–5 show the port is exact whenever the blocking is the
same. The threshold is left unchanged and the pytest gate covers rows 1–5 only.
