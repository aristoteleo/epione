# Bulk Hi-C: cooltools and HiCExplorer parity (pre-registered)

Written before any comparison was run. Gate classes follow the omicverse-rebuildr
parity taxonomy (`PARITY_TAXONOMY.md`). Thresholds are fixed here and are not changed
after results are seen; any later change is recorded as a dated amendment below the
original text, with its reason, before the rerun.

## References (pinned)

| Package | Version | Licence | Role |
|---|---|---|---|
| cooltools | 0.7.1 | MIT | reference for the existing `epione.bulk.hic` wrappers |
| cooler | as resolved by conda-forge with cooltools 0.7.1 (recorded in results) | BSD-3 | matrix IO |
| HiCExplorer | 3.7.6 (GitHub tag 3.7.6 = master c2ad8630, bioconda 3.7.6) | GPL-3 | reference for the new `method="hicexplorer"` paths |
| HiCMatrix | as resolved by bioconda with HiCExplorer 3.7.6 (recorded in results) | GPL-3 | HiCExplorer's matrix object |

The PyPI `hicexplorer` release (2.2.1.1) is outdated; it is not used.

References run in a separate env outside the repo (`epione_dev/hic_ref_env`). HiCExplorer
is driven through its Python API, not its text outputs, because those round floats
(`%f` in `_tad_score.bm`, `%.12f` elsewhere). The Epione side is run in the same env for
the gates, so that the measurement isolates the algorithm from BLAS/numpy differences.
The Epione dev-env result is reported alongside, ungated.

## Fixtures

- **D** — Galaxy HiCExplorer lesson, Drosophila S2: `corrected_contact_matrix_dm3_large.h5`
  (Zenodo 16416373, md5 b98ea18d9abbf0b756c412672e17a950), converted to `.cool`.
- **R** — HiCExplorer test data at tag 3.7.6: `hicDifferentialTAD/GSM2644945_Untreated-R1.100000_chr1.cool`
  and `GSM2644947_Auxin2days-R1.100000_chr1.cool` (Rao 2017).
- **G** — HiCExplorer test data at tag 3.7.6: `hicDetectLoops/GSE63525_GM12878_insitu_primary_2_5mb.cool`.
- **P** — HiCExplorer test data at tag 3.7.6: `hicPCA/mm9_reduced_chr1.cool`.
- **C** — `GM12878_25kb_cooler_coarsen.cool` (Zenodo 16416373, md5 27889ebb59f691b7a10540720a432161),
  chromosomes chr17–chr22, for the cooltools audit (needs a `weight` column and loop-scale data).

## A. Audit of the existing wrappers against cooltools 0.7.1

Each Epione function is called with its defaults; the reference is the cooltools call
with the arguments the wrapper documents. Fixture C (and D where the function runs on it).

| Gate | Epione | Reference | Class | Threshold |
|---|---|---|---|---|
| A1 | `compartments` | `cooltools.eigs_cis` | 1, deterministic-strict | max abs err ≤ 1e-13; identical NaN mask |
| A2 | `saddle` | `cooltools.saddle` (sum / count) | 1, deterministic-strict | on cells with count > 0: max abs err ≤ 1e-13; cells with count = 0 must be NaN |
| A3 | `insulation` | `cooltools.insulation` | 1, deterministic-strict | scores ≤ 1e-13; `is_boundary_*` identical |
| A4 | `tad_boundaries` | rows of `cooltools.insulation` with `is_boundary_*` | exact set | identical (chrom, start, end) set |
| A5 | `loops` | `cooltools.dots` with the same expected | exact table | identical dot set and values ≤ 1e-13 |
| A6 | `pileup` | `cooltools.pileup` averaged over the feature axis | 1, deterministic-strict | ≤ 1e-13 |

A wrapper that fails is fixed to delegate correctly; the fix and the before/after values are
reported.

## B. HiCExplorer 3.7.6 ports

| Gate | Epione | Reference (Python API) | Fixture | Class | Threshold |
|---|---|---|---|---|---|
| B0 | `h5_to_cool` | `hicConvertFormat` h5 → cool | D | exact | identical bins and pixels (bin1, bin2, count) |
| B1 | `insulation(method="hicexplorer")` TAD-separation score, every window | `HicFindTads.compute_spectra_matrix` → `bedgraph_matrix` | D, R | 1, deterministic-standard | max abs err ≤ 1e-8; identical rows |
| B2 | `tad_boundaries` on B1 output | `find_boundaries` + `save_domains_and_boundaries` filter | D, R | exact set + 1 | identical boundary bins; p-values rtol ≤ 1e-8; delta ≤ 1e-8 |
| B3 | `tad_domains` on B2 output | `_domains.bed` intervals | D, R | exact | identical intervals |
| B4 | `compartments(method="hicexplorer")`, `dist_norm` and `lieberman`, gene-density phasing | `hicPCA` | P, D | 1, deterministic-bounded | per bin ≤ 1e-6 (LAPACK `eig`); same eigenvector selection order |
| B5 | `compare_matrices`, `diff` / `ratio` / `log2ratio` | `hicCompareMatrices` | R | 1, deterministic-strict | ≤ 1e-13 |
| B6 | `correlate_matrices`, pearson / spearman, with and without `range` and `log1p` | `hicCorrelate` correlation matrix | R | 1, deterministic-strict | ≤ 1e-12 |
| B7 | `differential_tads` | `hicDifferentialTAD` with `--threads 1` | R (domains from B3 on Untreated) | exact + 8 | identical accept/reject flags per mode; p-values rtol ≤ 1e-8 |
| B8 | `loops(method="hicexplorer")` | `hicDetectLoops` with its test-suite parameters | G | 7, classification | loop-set Jaccard ≥ 0.95 (target: exact); p-values Spearman ≥ 0.99 on shared loops |

B7 note: HiCExplorer's `hicDifferentialTAD` splits domains across threads and skips the
first/last domain of each thread chunk differently, so its output depends on `--threads`.
The port reproduces the single-thread semantics; the reference is run with `--threads 1`.

Speed is reported for every B gate, Epione vs HiCExplorer on one core. Any speed-up
must be an exact rewrite (same floating-point operations in the same order) or carry a
bound derived here before it is used; none is planned beyond vectorising Python loops
whose summation order is preserved.

## Results

(to be filled in after the runs)
