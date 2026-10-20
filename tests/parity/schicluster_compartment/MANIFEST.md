# Single-cell Hi-C compartments: scHiCluster, HCEA and dcHiC parity (pre-registered)

Written before any comparison was run. Gate classes follow the omicverse-rebuildr parity
taxonomy. Thresholds are fixed here and are not changed after results are seen; any later
change is recorded as a dated amendment below the original text, with its reason, before the
rerun.

## References (pinned)

| Reference | Version | Licence | Role |
|---|---|---|---|
| scHiCluster | ChrisMao0325/scHiCluster `c958838` (fork of zhoujt1994/scHiCluster, 1.3.5.dev36) | MIT | `get_cpg_profile`, `multiple_cell_compartment`, `compartment_strength`, `merge_cell_raw`, scool pixel builder |
| HumanCellEpigenomeAtlas | `fig4/04.compartment.ipynb` (Zhou, Wu, Liu et al.) | see repo | pseudobulk compartment model (fit on merged raw contacts, project each major type), saddle |
| dcHiC | ay-lab/dcHiC `144b4df`, `dchicf.r --pcatype analyze` | MIT | differential compartments |
| R / robust / rrcov / limma | R 4.1.3, robust 0.7.1, rrcov 1.7-3, limma 3.50.3 (mamba env `dchic`) | GPL | dcHiC's statistics |
| bedtools | 2.31.1 | MIT | `bedtools nuc` behind `get_cpg_profile` |

References run in the dev env `epione-hic` (Python) and the `dchic` env (R). The Epione side
runs in `epione-hic` for every gate.

## Fixtures

- **H** — 8,040 HCEA snm3C-seq cells, per-cell contact tables
  (`Hic_data/HCEA/sc_contact/*/*.3C.contact.rmbkl.tsv.gz`, columns chrom1 = 1, pos1 = 2,
  chrom2 = 5, pos2 = 6), hg38, 100 kb. Gate subset **H200**: 200 cells sampled with
  `random_state=0`, stratified over the three major types present (c0, c1, c7).
- **E** — published HCEA objects under the ENTEx project root: `merged_cool_raw/merged.mcool`
  (100 kb), `merged_cool_raw/L1/c*.mcool` (100 kb), `analysis/compartment/merged_comp.hdf`
  (`raw_binfilter`, `raw_pcloading`) and `analysis/compartment/L1/comp.raw.hdf`.
- **D** — dcHiC inputs of the HCEA run, `analysis/diff_comp/all/` (34 major types, one sample
  each, `input.txt` + `c*_100Kb_pca/intra_pca/*/chr*.pc.bedGraph`), copied to the dev
  folder. Reference = a fresh `Rscript dchicf.r --file input.txt --pcatype analyze --dirovwt T
  --diffdir all` in the `dchic` env. (Checked before this manifest: the fresh run equals the
  published `DifferentialResult/fdr_result` table exactly.)

## S. scHiCluster single-cell compartment score

| Gate | Epione | Reference | Fixture | Class | Threshold |
|---|---|---|---|---|---|
| S1 | `cpg_ratio` (pyfaidx) | `get_cpg_profile` (`bedtools nuc -pattern CG -C`) | hg38 main chromosomes, 100 kb | exact | identical rows; identical values |
| S2 | per-cell pixels written by `contacts_to_scool` | scHiCluster scool pixel builder (no blacklist) | H200 | exact | identical (bin1, bin2, count) per cell |
| S3 | `compartment_scores`, cool input | `multiple_cell_compartment(mode='cool')` | H200 scool | 1, deterministic-strict | max abs err ≤ 1e-12 |
| S4 | `compartment_scores`, contact-table input | `multiple_cell_compartment(mode='tsv')` | H200 | 1, deterministic-strict | max abs err ≤ 1e-12 |
| S5 | per-chromosome AA / BB / AB strength sums | `compartment_strength` per chromosome | H200 | 1, deterministic-strict | rel err ≤ 1e-12 |
| S6 | `merge_cells` (raw pseudobulk) | `merge_cell_raw` | H200, per major type | exact | identical pixels |

S5 note: upstream accumulates `scores += scores` inside the chromosome loop, where `scores` is
the per-chromosome result, so its genome-wide value is twice the last chromosome's. Epione
returns the sum over chromosomes; the gate compares per-chromosome values, and the deviation
is documented in the docstring.

## H. HCEA pseudobulk compartments (raw contacts)

| Gate | Epione | Reference | Fixture | Class | Threshold |
|---|---|---|---|---|---|
| H1 | `compartment_model` bin filter | `merged_comp.hdf` `raw_binfilter` | E merged | exact | identical mask |
| H2 | `compartment_model` loadings | `merged_comp.hdf` `raw_pcloading` | E merged | 2, deterministic-bounded (ARPACK start vector) | max abs err ≤ 1e-8; same PC chosen per chromosome |
| H3 | `project_compartments` | `L1/comp.raw.hdf` | E, every L1 type | 2, deterministic-bounded | max abs err ≤ 1e-8 |
| H4 | `compartment_saddle` (with `clip=None`, chr1–chr21, as in the notebook's `compsaddle`) | notebook `compsaddle` rerun in `epione-hic` | E, three L1 types | 1, deterministic-strict | saddle / count max rel err ≤ 1e-12 |

## D. dcHiC differential compartments (`--pcatype analyze`)

| Gate | Epione | Reference | Fixture | Class | Threshold |
|---|---|---|---|---|---|
| D1 | OGK robust covariance (`covRob` auto → `pairwiseqc`) | `robust::covRob(x)$cov` | D, per-chromosome group matrices; plus 3 random matrices | 1, deterministic-standard | max rel err ≤ 1e-10 |
| D2 | `normalize_quantiles` | `limma::normalizeQuantiles(ties=TRUE)` | D, plus a matrix with ties | 1, deterministic-strict | max abs err ≤ 1e-13 |
| D3 | `dchic_analyze` per-chromosome `pcOri` / `pcQnm` tables | `DifferentialResult/all/pcOri`, `pcQnm` | D | exact | identical bins; values identical as written (5 decimals) |
| D4 | `dchic_analyze` `sample_maha`, `pval`, `padj` | `fdr_result/differential.intra_sample_combined.pcQnm.bedGraph` | D | 1, deterministic-standard | rel err ≤ 1e-8 (for `pval`, `padj` ≥ 1e-300; below that, both ≤ 1e-300) ; identical padj < 0.1 set |
| D5 | written output files | the R output folder | D | exact | same file names, columns and row order |

Paths of dcHiC that are not ported in this round raise `NotImplementedError` with a pointer to
this manifest: IHW adjustment (used when a condition has replicates), and the `mcd` /
`donostah` branches of `covRob` (chosen when there are fewer than 20 conditions). They depend
on R's random number stream and get their own gates when ported.

Speed is reported for every gate, Epione vs reference on one core.

## Amendment H-1 (2026-10-20, after the first H2 run, before any rerun)

First run: H1 passed (identical mask, 24,478 of 28,760 bins kept); H2 failed (max abs err
0.0061, Pearson r 0.9995–0.9999 per chromosome, same PC chosen on every chromosome). Cause,
found by bisection on chr21: the published `raw_pcloading` was fitted on the correlation
matrix **without** the 5/95-percentile clip. With `clip=None` the chr21 loadings match to
2.3e-16 (both copies of the merged raw cool, `merged_cool_raw/merged.mcool` and
`analysis/compartment/merged.raw.mcool`, give the same result). The clip lines in the
notebook's cell 22 were evidently added after the stored model was produced; the projection
function `compute_comp` (cell 31) does clip.

Change: H2 calls `compartment_model(..., clip=None)`. The gate, its fixture and its threshold
are unchanged. The function's default follows whatever reproduces the published objects
(fit unclipped if H3 confirms the projection is clipped); this is documented in the docstring.

## Amendment D-1 (2026-10-20, before any CovSde comparison): Donoho–Stahel branch

Reason: the tutorial differential-compartment step compares three cell types, so per
chromosome `robust::covRob` sees n ≈ 1,000–2,500 bins × p = 3 and its `estim = "auto"` rule
((n < 1000 and p < 10) or (n < 5000 and p < 5)) chooses `donostah`, not `pairwiseqc`. That
branch is therefore ported now instead of raising `NotImplementedError`.

The branch is `set.seed(21)` followed by `rrcov::CovSde(x, tune = 0.95, prob = 0.99,
eps = 0.5)` (Fortran `rlds`, rrcov 1.7-3). The port needs R's uniform stream; the emulator
(`epione/bulk/hic/_r_rng.py`) was checked before this amendment as a building block, not a
gate: after `set.seed(21)` and `set.seed(123)`, 5,000 and 3,000 `runif` draws and
`.Random.seed` are bit-identical. R's `qchisq(0.5, p)` / `qchisq(0.95, p)` for p = 1..9 are
embedded, because `scipy.stats.chi2.ppf` differs from them by about one ulp.

| Gate | Epione | Reference | Fixture | Class | Threshold |
|---|---|---|---|---|---|
| D6 | `cov_sde` (center and cov) | `robust::covRob(x)` (`$center`, `$cov`; `estim = "auto"` → `donostah`) | 5 random matrices generated in R, shapes (300, 3), (900, 4), (2500, 3), (4000, 4), (800, 9); 3 columns (c0, c1, c7) of E `L1/comp.raw.hdf` on chr1 and on chr2 | 1, deterministic-standard | max rel err ≤ 1e-10 on cov and on center (relative to max abs of the reference entry block) |

`MCD` (5–19 conditions) and IHW (replicates) remain unported and keep raising
`NotImplementedError`.


## Results (D6), 2026-10-20

**PASS.** Max rel err 3.9e-15 (threshold 1e-10). `results/covsde_D6.json`;
reference script `covsde_make_ref.R`, driver `covsde_parity.py`.

| Matrix | n × p | R estim | valid subsets | center | cov (max rel err) | Epione s | R s |
|---|---|---|---|---|---|---|---|
| rand_300x3 | 300 × 3 | donostah | 1000 | identical | identical | 0.04 | 0.04 |
| rand_900x4 | 900 × 4 | donostah | 1000 | identical | identical | 0.06 | 0.14 |
| rand_2500x3 | 2500 × 3 | donostah | 1000 | identical | identical | 0.08 | 0.50 |
| rand_4000x4 | 4000 × 4 | donostah | 1000 | identical | 3.9e-15 | 0.12 | 0.87 |
| rand_800x9 | 800 × 9 | donostah | 4714 | identical | 1.7e-16 | 0.67 | 0.66 |
| real_chr1 | 2079 × 3 | donostah | 1000 | identical | 3.0e-16 | 0.07 | 0.35 |
| real_chr2 | 2235 × 3 | donostah | 1000 | identical | 3.7e-16 | 0.07 | 0.39 |

The center (weighted mean) is bit-identical on every matrix: same random subsets, directions,
outlyingness and weights as the Fortran `rlds`. The covariance differs by at most 4e-15,
which comes only from CovSde's final consistency factor (`median(mahalanobis(x, center,
cov)) / qchisq(0.5, p)`), where R uses its `solve` and `%*%` and Epione numpy's LAPACK/BLAS.

Run history: the first run exchanged the matrices as CSV text and passed with max rel err
4.6e-12, but no output was bit-identical. Tracing `rlds` (instrumented copy compiled with
`R CMD SHLIB` in the `dchic` env) showed identical subsets and arithmetic, but different
inputs: R's `read.csv` and pandas' default parser round some 17-digit strings to different
doubles. The fixture was switched to raw float64 files (same values, no text round trip);
gate and threshold unchanged.

## Results (D), 2026-10-20

**All five gates PASS** (`results/dchic_D.json`; driver `dchic_parity.py`, R side
`dchic_make_ref.R`). Fixture D: 35 major-type tracks of `analysis/diff_comp/all`, 24,478 bins.

| Gate | Measured | Threshold | Pass |
|---|---|---|---|
| D1 `cov_rob` vs `robust::covRob(x)$cov` (22 HCEA per-chromosome matrices + 3 random) | max rel err 2.1e-15; every matrix `pairwiseqc` on both sides | ≤ 1e-10 | ✅ |
| D2 `normalize_quantiles` vs `limma::normalizeQuantiles(ties=TRUE)` (22 HCEA + 4 incl. ties) | max abs err 0 (bit-identical) | ≤ 1e-13 | ✅ |
| D3 `pcOri` / `pcQnm` tables (44 global + 1,540 per-condition files) | identical bins, identical written values in all 1,584 files | exact | ✅ |
| D4 `sample_maha` / `pval` / `padj` | 2.8e-11 / 6.2e-9 / 6.2e-9 max rel err; 15,508 p-values below 1e-300 in both (0 in R's file); identical padj < 0.1 set (20,339 bins); normalised score columns identical | ≤ 1e-8 | ✅ |
| D5 file names, columns, row order | 2,382 files on both sides, all layouts identical | exact | ✅ |

Byte level: 2,354 of 2,382 output files are byte-identical. The 28 that differ are the
`fdr_result` tables, only in the last digits of `sample_maha` / `pval` / `padj`.

Speed (whole `analyze` step, wall clock, one BLAS thread): Epione 32 s, R 38 s.

What it took to get there, for the record:

- R's number round trips are reproduced exactly: `write.table` digits (`formatReal` /
  `scientific()`, which picks the digit count by rounding in long double, e.g. R writes
  `13.708020988103` where `%.15g` gives `13.7080209881029`), the `read.table` parser
  (`R_strtod5`), `round(x, 5)` (`fround.c`; `numpy.round` differs on 11,845 of 135,000 test
  values), `pnorm` (Cody), and long-double means / row means / variances.
- First full run: 20 chromosomes matched to ~1e-12, chr9 and chr22 differed by O(1). Cause:
  on the refinement subset of chr22 (119 bins × 35 types) every pairwise quadrant
  correlation is 1, so OGK's correlation matrix is all ones with a 34-fold zero eigenvalue.
  The eigenbasis there is arbitrary and is fixed by rounding noise (~1e-28) in LAPACK
  `dsptrd`. R's answer is therefore a property of its OpenBLAS build: with R's own
  `libopenblasp-r0.3.30.so` the port matches R to 2e-15, with the OpenBLAS bundled in the
  scipy wheel it does not. The port now calls `dsptrd` → `dstegr` → `dopmtr` like rrcov,
  using the environment's LAPACK (`sys.prefix/lib/liblapack*`, conda OpenBLAS 0.3.34 here,
  which agrees with 0.3.30 on this input) and falls back to scipy's; it emits a
  `RuntimeWarning` when the eigenvalues repeat (3 times on fixture D). This is not a gate
  change: the D4 threshold was met only after the fix, with the same threshold.
- Remaining unported: covRob `mcd` (fewer than 20 conditions where the `donostah` rule does not apply) and IHW
  (replicates); both raise `NotImplementedError`. `padj_method="BH"` is the documented
  opt-in deviation for replicates.

## Results (S, H) — 2026-10-20

Drivers: `s_gates.py` (fixture H200: 200 cells, c1 108 / c0 64 / c7 28) and `h_gates.py`;
raw numbers in `results/schicluster_S.json` and `results/hcea_H.json`. Env `epione-hic`
(cooler 0.10.4, cooltools 0.7.1, scHiCluster c958838).

| Gate | Measured | Threshold | Pass |
|---|---|---|---|
| S1 `cpg_ratio` | 28,760 rows identical, values identical (max err 0.0) | exact | ✅ |
| S2 scool pixels | 0 of 200 cells differ | exact | ✅ |
| S3 scores, cool input | max abs err 0.0; NaN mask identical | ≤ 1e-12 | ✅ |
| S4 scores, contact tables | max abs err 0.0; NaN mask identical | ≤ 1e-12 | ✅ |
| S5 per-chromosome strength | max rel err 0.0 (40 cells × 22 chromosomes) | ≤ 1e-12 | ✅ |
| S6 `merge_cells(double_diagonal=True)` vs `merge_cell_raw` | identical pixels, c0 / c1 / c7 | exact | ✅ |
| H1 bin filter | identical mask (24,478 of 28,760 bins) | exact | ✅ |
| H2 loadings (`clip=None`, Amendment H-1) | max abs err 1.2e-15; PC1 chosen on all 22 chromosomes, as published | ≤ 1e-8 | ✅ |
| H3 projection, 35 major types | max abs err 5.3e-14 (reference model); 9.8e-14 end to end (Epione fit → projection) | ≤ 1e-8 | ✅ |
| H4 saddle (c0, c1, c7; chr1–chr21; `clip=None`) | max rel err 6.4e-16 on sums and counts | ≤ 1e-12 | ✅ |

Notes. (1) The scHiCluster reference cannot write its NetCDF with current xarray (categorical
`chrom` coordinate); `s_gates.py` casts `chrom` to str in upstream's aggregation step only.
(2) Speed, one run each, Epione vs scHiCluster: S1 39 s vs 114 s (bedtools nuc); S2 54 s
(8 processes) vs 257 s (1 process); S4 22 s vs 44 s (8 processes each); S6 48 s vs 78 s. The
S3 reference timing is not meaningful (upstream reused its cached per-cell files from the
first, crashed run). (3) H3 first run (before Amendment H-1) already passed with the
reference model (5.3e-14); its end-to-end error was 0.088 because the fit was clipped.

## Amendment D-2 (2026-10-20, before any comparison): end-to-end three-condition run

D1–D5 used 35 conditions (covRob → OGK) and D6 tested the Donoho–Stahel estimator alone. The
tutorial runs dcHiC on three major types, where covRob picks Donoho–Stahel inside the full
`analyze` step. New gate, same classes and thresholds as D3–D5:

| Gate | Epione | Reference | Fixture | Class | Threshold |
|---|---|---|---|---|---|
| D7 | `dchic_analyze` | `dchicf.r --pcatype analyze --dirovwt T --diffdir all` | **D3t**: HCEA published raw scores of c0, c1, c7 (`L1/comp.raw.hdf`, bins with `raw_binfilter`), written as dcHiC `pc.bedGraph` inputs | as D3 / D4 / D5 | D3: exact; D4: rel err ≤ 1e-8, identical padj < 0.1 set; D5: exact |

## Results (D7), 2026-10-20

Raw numbers in `results/dchic_D7.json`. 270 output files with identical names, columns and row
order (242 byte-identical); `pcOri` / `pcQnm` tables identical; `sample_maha` rel err 9.8e-13,
`pval` / `padj` 1.2e-10; the same 5,877 bins at padj < 0.1. Epione 5.1 s, R 18.4 s. ✅
