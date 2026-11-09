# Iteration log — Cicero → `epione.tl`

One block per iteration (omicverse-rebuildr `templates/ITERATION_LOG.template.md` schema; parsed by
`engine/plot_evolution.py`). Benchmark: `scripts`-level `bench_iter.py` — `make_cicero_cds` + `run_cicero`
on fixture **F1** (`cicero_data`, 200 cells × 6,146 peaks), seed 42, 1 warm-up + 3 measured runs
(auto-extended to 5 when CV > 10 %), `OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS` pinned to the requested CPUs.
Parity metric plotted: gate 10 (Pearson of coaccess on matched pairs vs R, threshold 0.9999); gate 9
(max abs error, threshold 1e-8) is quoted in the notes.

Caveat on wall-clock: the CPU allocation (17 cores, sh04-05n13) was shared with other jobs of the same
user during the whole port, so single runs vary by up to ±25 %; speed-ups below 1.1× are within noise.

---

## Baseline — 2026-11-09 14:20:00

```yaml
iter: 0
status: baseline
action: null
admissibility: null
playbook_section: null
wall_clock_mean_s: 49.036
wall_clock_stddev_s: 5.0936
wall_clock_runs_s: [46.609, 57.9759, 48.4197, 45.7686, 46.4069]
warmup_run_s: 54.2882
parity_metric: 0.9999999999999615
parity_class: ordinal
parity_threshold: 0.9999
parity_passes: true
notes: |
  Line-by-line translation (commit 19b9184): glasso.f ported subroutine by subroutine to numba with
  the Fortran accumulation order; R's sample() via the bit-exact RRandom; stats::cov and mean()
  emulated with numpy long double (R's LDOUBLE); window/peak selection, rho, cov2cor, reconcile()
  and data.table key order mirrored. On F1 every deterministic gate passes: identical aggregation
  bins and sampled windows, identical mean distance parameter (bit-for-bit), coaccess max abs
  error 8.0e-16 with identical row order. Gate 16 (Louvain CCANs) is stochastic and is discussed
  in the report.
```

---

## iter 1 — 2026-11-09 14:27:00

```yaml
iter: 1
status: ACCEPT
action: cov_upper_triangle_rowwise
playbook_section: "§1 memoisation / loop restructuring"
admissibility: exact
admissibility_evidence: |
  R's cov_na_1 computes only j <= i and mirrors; each entry is the long-double running sum
  sum_k (x_ki - m_i)(x_kj - m_j) with k ascending. The rewrite forms the same double products for
  the upper triangle only and adds them to a long-double accumulator one row k at a time: the same
  additions in the same order, hence bit-identical (asserted in tests/test_cicero.py against a naive
  per-element loop, and equal to the chunked baseline on n = 3235, p = 200).
wall_clock_mean_s: 47.4022
wall_clock_stddev_s: 1.2502
wall_clock_runs_s: [47.2884, 48.7053, 46.2127]
warmup_run_s: 55.6421
speedup_vs_previous: 1.03
speedup_vs_baseline: 1.03
parity_metric: 0.9999999999999615
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  F1 has only 25 aggregated bins, so covariance is cheap there; on F2-sized windows (3235 bins,
  200 peaks) the micro-benchmark gives 3.38 s -> 1.26 s per covariance (2.7x).
```

---

## iter 2 — 2026-11-09 14:31:00

```yaml
iter: 2
status: ACCEPT
action: distance_parameter_two_pass_memo
playbook_section: "§1 memoisation"
admissibility: exact
admissibility_evidence: |
  find_distance_parameter() returns a non-number only for "No long edges", which depends on the
  window's peak coordinates alone; the other skips (<= 1 or > max_elements sites) are cheap as well.
  So R's sampling loop can be replayed first with these checks to fix the sequence of windows that
  yield a value, and the bisection (a deterministic function of the window) is run once per distinct
  window (windows are drawn with replacement). Same values, same order (gate 5 / 6 unchanged).
wall_clock_mean_s: 45.6849
wall_clock_stddev_s: 2.3476
wall_clock_runs_s: [45.9389, 43.2207, 47.8952]
warmup_run_s: 36.7299
speedup_vs_previous: 1.04
speedup_vs_baseline: 1.07
parity_metric: 0.9999999999999615
parity_delta_vs_baseline: 0.0
parity_passes: true
```

---

## iter 3 — 2026-11-09 14:37:00

```yaml
iter: 3
status: ACCEPT
action: thread_parallel_windows
playbook_section: "§4 embarrassingly parallel loops"
admissibility: exact
admissibility_evidence: |
  Windows are independent (R itself uses parallel::mclapply with mc.cores = 1). The per-window
  bisections (after iter 2 fixed which windows are used) and the per-window glasso models run in a
  ThreadPoolExecutor; the numba solver is compiled nogil and numpy long-double ufuncs release the
  GIL. Results are collected in window order, so the output is identical for any n_jobs.
wall_clock_mean_s: 4.8802
wall_clock_stddev_s: 0.4407
wall_clock_runs_s: [4.3285, 5.5459, 4.9589, 4.8308, 4.7369]
warmup_run_s: 5.2206
speedup_vs_previous: 9.36
speedup_vs_baseline: 10.05
parity_metric: 1.0
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  n_jobs = 8 (8 CPUs requested). Coaccess identical to iteration 2 (max abs error vs R 8.0e-16);
  (pearsonr itself rounds differently between runs: 1.0 here vs 1 - 4e-14 before, same values).
```

---

## iter 4 — 2026-11-09 14:55:00

```yaml
iter: 4
status: ACCEPT
action: equivalence_utilities_and_locale_collation
playbook_section: null
admissibility: exact
admissibility_evidence: |
  Equivalence step, not a speed rewrite. Ported the six remaining exported utilities (addendum gates
  19-24, pre-registered before their code) and found that R sorts factor levels / dcast / ddply groups
  with the locale's collation (en_US.UTF-8 in the reference: punctuation < digits < letters,
  case-insensitive, lowercase first), not byte order; _r_sorted() reproduces R's sort() on probe
  strings. data.table (assemble_connections) keeps byte order. Pipeline outputs unchanged.
wall_clock_mean_s: 5.1256
wall_clock_stddev_s: 0.3076
wall_clock_runs_s: [5.4804, 4.934, 4.9623]
warmup_run_s: 8.4858
speedup_vs_previous: 0.95
speedup_vs_baseline: 9.57
parity_metric: 1.0
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  n_jobs = 8. Gate 21 (aggregate_by_cell_bin) failed on row order before the collation fix and
  passes exactly after it; gates 19, 20, 22-24 passed at the first run.
```

---

## iter 5 — 2026-11-09 15:12:00

```yaml
iter: 5
status: REJECT_INADMISSIBLE
action: blas_covariance
playbook_section: "§2 bounded approximation"
admissibility: bounded
admissibility_evidence: |
  Replace R's long-double covariance by a double-precision BLAS product D'D/(n-1). Error per entry
  is bounded by the usual summation bound |S_blas - S_R| <= (gamma_n(double) + gamma_n(80-bit)) *
  sum_k |d_ki d_kj| with gamma_n = n u / (1 - n u). This is a (B) rewrite; the user's brief admits
  only provably equivalent (exact) rewrites, so it is rejected although every gate still passes.
perturbation_bound: |
  n = 3235 bins (F2): gamma_n ~ 3235 * 1.1e-16 = 3.6e-13 relative to sum |d d|.
wall_clock_mean_s: 4.6671
wall_clock_stddev_s: 0.1853
wall_clock_runs_s: [4.68, 4.4758, 4.8456]
warmup_run_s: 4.5395
speedup_vs_previous: 1.10
speedup_vs_baseline: 10.51
parity_metric: 1.0
parity_delta_vs_baseline: 0.0
parity_passes: true
math_reason_for_dip: |
  Not a visible dip: on F2 the coaccess max abs error grows from 4.4e-16 to 6.9e-15 and the mean
  distance parameter is no longer bit-identical (relative error 5.5e-16). F2 run_cicero: 32.6 s ->
  8.9 s with 8 threads.
```

### Decision

REJECT — bounded, not exact; superseded by iter 6, which gets most of the speed exactly.

---

## iter 6 — 2026-11-09 15:20:00

```yaml
iter: 6
status: ACCEPT
action: compiled_x87_long_double_covariance
playbook_section: "§1 same arithmetic, compiled"
admissibility: exact
admissibility_evidence: |
  R's LDOUBLE on x86-64 is the x87 80-bit format. A numba intrinsic emits LLVM x86_fp80 code:
  fpext(product) + running sum with one x87 rounding, exactly what GCC emits for `sum += prod`.
  The running sum is stored as hi = fptrunc(sum), lo = fptrunc(sum - hi); both conversions are
  exact because sum has 64 significant bits, so hi + lo reconstructs the 80-bit value. Products are
  formed in double as in R. Bit-identical to the numpy long-double path (test_r_cov_compiled_x87_
  equals_numpy_long_double, and F2 gates: mean distance parameter relative error 0, coaccess
  max abs error 4.4e-16 as before). Other platforms keep the numpy long-double path.
wall_clock_mean_s: 5.1964
wall_clock_stddev_s: 0.2281
wall_clock_runs_s: [4.9618, 5.4175, 5.2099]
warmup_run_s: 4.8102
speedup_vs_previous: 0.99
speedup_vs_baseline: 9.44
parity_metric: 1.0
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  F1 is glasso-bound (25 bins), so no change there. On F2 (3235 bins) a 200-site covariance drops
  from 1.27 s to 0.21 s and run_cicero from 32.6 s to 11.6 s with 8 threads (generate_cicero_models
  20.3 s -> 2.8 s), with outputs bit-identical to iterations 0-4.
```

---

## iter 7 — 2026-11-09 16:05:00

```yaml
iter: 7
status: ACCEPT
action: equivalence_louvain_on_r_rng
playbook_section: null
admissibility: exact
admissibility_evidence: |
  Equivalence step for the last stochastic output. R's cluster_louvain is igraph C core 0.10's
  igraph_community_multilevel drawing from R's unif_rand() through R igraph's 30-bit RNG bridge.
  Ported: the multilevel algorithm (louvain.c), igraph's bounded integers (Lemire, 32-bit),
  vector shuffle, the undirected edge store and igraph_incident order, igraph_qsort (FreeBSD; its
  order of equal keys fixes the order in which link weights are summed), and
  graph_from_adjacency_matrix on make_sparse_matrix (zero-weight self loops included). A second
  difference surfaced in find_ccan_cutoff: R 4.x round(x, 3) is not the correctly rounded decimal
  (round(0.10549999999999999656, 3) = 0.106); fround.c ported. With both, generate_ccans() gives
  the identical table (cutoff, peaks, CCAN ids) as R for seeds 42, 2017, 7, 11, 23 on F1 and F2.
wall_clock_mean_s: 5.1898
wall_clock_stddev_s: 0.0588
wall_clock_runs_s: [5.2451, 5.128, 5.1963]
warmup_run_s: 5.1803
speedup_vs_previous: 1.00
speedup_vs_baseline: 9.45
parity_metric: 1.0
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  Gate 16 (CCAN ARI vs R, seed 42) moves from 0.935 (F1, FAIL) / 0.995 (F2) to 1.0 / 1.0 with
  identical tables; python-igraph remains available as rng = "igraph".
```
