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

## Amendment 1 (2026-10-13, before any gate-A or B8-on-C comparison)

Fixture C as downloaded names its chromosomes `1`..`22`, `X`, `Y`, `MT` (no `chr` prefix) and
has no `weight` column. The first gate-A run stopped at fixture loading (empty chromosome list),
before any comparison. Changes, gates and thresholds unchanged:

- chromosomes for A are `17`..`22`; for B8 on C, `21` and `22`;
- a copy of C is ICE-balanced once with `epione.bulk.hic.balance_cool` (cooler's
  `balance_cooler`, default parameters) to provide the `weight` column; both sides read
  that copy. B8 on C reads the same balanced copy.

## Amendment 2 (2026-10-13, before any B8 comparison)

"Its test-suite parameters" for B8 on G is made concrete as the parameters of
`test_hicDetectLoops.py::test_main_cool_chromosomes_single_core` (HiCExplorer 3.7.6):
`--maxLoopDistance 3000000 -pit 1 -w 5 -pw 2 -p 0.5 -pp 0.55 --chromosomes 1 2`. The first run
used the suite's other parameter set, whose default `--maxLoopDistance` (2 Mb) is below G's
2.5 Mb bin size, so HiCExplorer filtered out every pixel and wrote no file; the run stopped
there, before any comparison. Gate unchanged.

## Amendment 3 (2026-10-13, before any gate-A or B8-on-C comparison)

Balancing the whole 1 GB fixture C (25 kb, genome-wide) with cooler's sequential
`balance_cooler` did not finish in 30 min, and both runs were stopped before any comparison.
Fixture C is redefined as the sub-matrix of chromosomes `17`..`22` (cis and trans pixels among
them), written to a new `.cool` and ICE-balanced once with `balance_cool` defaults. Gates A and
B8-on-C read that file. Gates and thresholds unchanged.

## Amendment 4 (2026-10-13, before any B6 comparison)

Fixture R stores normalised values (median 1.7e-4, none above 1). `hicCorrelate` keeps only
pixel pairs whose two values sum to more than 1, so on R it keeps nothing and HiCExplorer itself
cannot compute a correlation. B6 moves to fixture **K**: HiCExplorer test data at tag 3.7.6,
`hicCorrectMatrix/gm12878_raw_values.cool` vs `hicCorrectMatrix/gm12878_KR.cool` (GM12878, 2.5 Mb,
1,254 identical bins; raw counts vs KR-corrected). Same cases and threshold. B5 stays on R.

## Amendment 5 (2026-10-13, after the two registered B8 cases, before the added one)

Both registered B8 cases were degenerate: HiCExplorer and Epione each found 0 loops, so the
gate passed without testing anything. On G, HiCExplorer 3.7.6 in this environment finds no loop
with `test_main_cool_chromosomes_single_core`'s parameters, although that test's expected file
holds one. On C, the cooler-balanced values (rows summing to about 1, single pixels far below 1)
all fall under `hicDetectLoops`' default `--peakInteractionsThreshold 10`, which assumes
count-scale input. Added case, same gate: **C_raw** = fixture C chromosomes `21`, `22`
**without** balancing (raw counts, no `weight` column), HiCExplorer default parameters.
The two degenerate results are reported as such.

A1 note recorded at the same time (gate unchanged): cooltools 0.7.1 `eigs_cis` without a phasing
track returns eigenvectors of arbitrary sign that change between repeated calls in one process
(two direct cooltools calls differ by up to 5.0; after sign alignment by about 4e-15). A1 as
registered therefore compares two random signs. A sign-aligned comparison is reported alongside.

## Note on A6 (2026-10-13, before any A6 comparison)

A6 piles up "the A5 dots". The first attempt failed in both the wrapper and the direct cooltools
call with "ObsExpSnipper is implemented for cis contacts only". The cause is a cooltools 0.7.1
bug: `snipping._extract_stack` unpacks any 2-character region name (here the view names `21`,
`22`) as a `(region1, region2)` pair. Epione's `pileup` now names its view regions
`chrom:0-size`, which avoids the bug and does not change any value; the reference call uses the
same names. A6 uses the first 200 A5 dots whose flank-padded window stays inside the chromosome.
Gate unchanged.

## Results (2026-10-13, environment: HiCExplorer 3.7.6, HiCMatrix 17.2, cooltools 0.7.1, cooler 0.10.4, numpy 1.26.4, scipy 1.17.1)

### A. Existing wrappers vs cooltools 0.7.1 (fixture C = GM12878 25 kb, chr17–22)

| Gate | Result | Notes |
|---|---|---|
| A1 compartments | **FAIL as registered** (max abs err 5.45) | cooltools `eigs_cis` without a phasing track gives a random sign per call: two direct cooltools calls also differ by 5.45. Sign-aligned Epione vs cooltools: 1.7e-14. The wrapper only delegates. |
| A2 saddle | PASS after fix | First run: cells with count 0 held uninitialised memory (6.5e-320). Fixed with `out=np.full(..., nan)`; finite cells were already exact (0). |
| A3 insulation | PASS (0; boundary flags identical) after fix | First run crashed: `balance_cool` did not store cooler's weight attributes (`ignore_diags`, ...), so `insulation(ignore_diags=None)` failed on any Epione-balanced cool. Fixed. |
| A4 tad_boundaries | PASS (753 / 753 identical) | |
| A5 loops (dots) | PASS (136 / 136 identical, values 0 err) | |
| A6 pileup | PASS (0) after fix | cooltools 0.7.1 bug: 2-character region names are unpacked as a region pair. Epione's `pileup` now names view regions `chrom:0-size`. Stack shape (n, D, D) confirms the wrapper's averaging axis. |

### B. HiCExplorer 3.7.6 ports

| Gate | Fixture | Result | Epione / HiCExplorer seconds |
|---|---|---|---|
| B0 h5_to_cool | D | PASS: identical bins, weights, 6,182,500 pixels | — |
| B1 TAD-separation score | D / R / R (Bonferroni) | PASS, max abs err 0 (11,189 / 1,889 rows) | 4.5 / 6.9 (D) |
| B2 boundaries | D / R / R (Bonferroni) | PASS, identical (820 / 141 / 119), p-value and delta err 0 | 5.2 / 7.7 (D) |
| B3 domains | D / R / R (Bonferroni) | PASS, identical (815 / 140 / 118) | — |
| B4 hicPCA | P dist_norm, P lieberman, P dist_norm + genes, D dist_norm + genes | PASS, identical to the 12 printed decimals | 1,080 / 910; 1,397 / 1,751; 646 / 964; 36 / 53 (96% of the time is the same `scipy.linalg.eig`; the run-to-run spread on the shared node is larger than any difference) |
| B5 compare_matrices | R, diff / ratio / log2ratio | PASS, max abs err 0 | 0.5 / 1.6–1.9 |
| B6 correlate_matrices | K, 5 cases | PASS, max abs err 0 | — |
| B7 differential_tads | R, 4 modes | PASS: 140 TADs, identical accept/reject flags, p-values within 9.5e-13 (text output) | 7.5–9 / 9–11 |
| B8 loops | G (test suite) / C balanced, defaults / C raw, defaults | PASS / PASS / PASS: 0 vs 0, 0 vs 0 (**both degenerate**, see amendment 5) and **140 vs 140 identical** | 0.5 / 2.0; 11 / 25; ~ / 31 |

B7 first run: the flags matched, but one p-value differed (the last TAD's left inter-TAD test,
0.728 vs 0.919). Cause: `hicDifferentialTAD` does not recompute the right-boundary index for a
chromosome's last TAD and reuses the previous TAD's value. The port now reproduces this; gate
unchanged; rerun passes.

Text-output precision: B7 and B8 p-values are read from HiCExplorer's text files, which hold
about 12 significant digits (relative differences 1e-13). An in-process check on B8 (chr21) gives
identical p-values and identical intermediate obs/exp matrices.

