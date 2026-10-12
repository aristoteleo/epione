# Parity manifest — SCENIC+ block ③: search space, region-to-gene and TF-to-gene links

Reference: scenicplus 1.0a2 (`840dab85`), arboreto 0.1.6, scikit-learn 1.3.2, numpy 1.26.4,
run on the 10x multiome human cerebellum 3k tutorial dataset (2,313 multiome cells) with
the parameters of the scenicplus 1.0 Snakemake pipeline (seed 666, `PYTHONHASHSEED=0`):

* `get_search_space(upstream=(1000, 150000), downstream=(1000, 150000), extend_tss=(10, 10),
  use_gene_boundaries=False, remove_promoters=False)` → `search_space.tsv`
* `calculate_TFs_to_genes_relationships(method="GBM", seed=666)` (arboreto GRNBoost2:
  `GradientBoostingRegressor(learning_rate=0.01, n_estimators=5000, max_features=0.1,
  subsample=0.9, random_state=666)` with the OOB early-stopping monitor, window 25) →
  `tf_to_gene_adj_GBM.tsv`
* `calculate_regions_to_genes_relationships(importance_scoring_method="GBM"|"RF",
  correlation_scoring_method="SR", mask_expr_dropout=False)` →
  `region_to_gene_adj_{GBM,RF}.tsv`

Registered **before** the first comparison; thresholds are read-only.

| # | Output | Mode | Class | Metric | Threshold |
|---|--------|------|-------|--------|-----------|
| 1 | Search space (`Name`, `Gene`, `Distance`) | deterministic | 1 | file content | byte-identical rows, identical order |
| 2 | TF2G `importance` | stochastic, emulated RNG (MT19937 + sklearn `rand_r`) | 1 (emulated) | per (TF, target) relative error | ≤ 1e-8, identical link set |
| 3 | TF2G `rho` | deterministic | 1 | max abs err | ≤ 1e-8 |
| 4 | TF2G `regulation` | deterministic | 1 | exact | identical |
| 5 | TF2G row order | deterministic | 1 | exact | identical |
| 6 | R2G GBM `importance` | stochastic, emulated RNG | 1 (emulated) | per (region, target) relative error | ≤ 1e-8, identical link set |
| 7 | R2G RF `importance` | stochastic, emulated RNG | 1 (emulated) | per (region, target) relative error | ≤ 1e-8, identical link set |
| 8 | R2G `rho` (Spearman) | deterministic | 1 | max abs err (NaN positions identical) | ≤ 1e-8 |
| 9 | R2G `importance_x_rho`, `importance_x_abs_rho` | derived | 1 | relative error | ≤ 1e-8 |
| 10 | R2G `Distance`, row order | deterministic | 1 | exact | identical |
| 11 | Epione `n_jobs=1` vs `n_jobs=N` | determinism of the port | 1 | bitwise | identical |

Relative error for rows 2, 6, 7, 9 is `|a - b| / max(|b|, 1e-300)`, computed per link.

Notes

- Bit-exact importances need scikit-learn **1.3.2** tree-growing semantics. Newer releases
  differ: on the same input, 1.9.1 gives a different GBM initial prediction (1.3.2 fits
  `DummyRegressor` on the float32 target, so the initial mean is a float32 mean) and
  slightly different node impurities, so the subsampled GBM drifts at the third digit.
  Epione therefore does not call scikit-learn for these models. `epione.scenic._sktree`
  reimplements the 1.3.2 path (DepthFirstTreeBuilder, BestSplitter with its introsort and
  Fisher-Yates feature draws, `MSE` / `FriedmanMSE`, `rand_r`, the GBM stage loop with
  `_random_sample_mask` and OOB scores, bootstrap of `RandomForestRegressor`, and the
  numpy reductions used for feature importances) with numba, so the result does not depend
  on the installed scikit-learn.
- The cell order of the scenicplus matrices and the TF-list order come from Python set
  iteration under `PYTHONHASHSEED=0` upstream; they are inputs here (the parity run feeds
  the reference matrices in their reference order).
- The reference outputs are generated outside this repository. `parity_check.py` and
  `tests/test_scenic_links.py` read them from `$EPIONE_SCENIC_PARITY_DIR`
  (`A/scplus_inputs`, `A/tf2g`, `A/r2g`, `A/inputs`); the parity test is skipped when the
  variable is unset.

## Results (full reference run: 2,313 cells, 36,601 genes, 434,673 regions)

| # | Output | Value | Gate |
|---|--------|-------|------|
| 1 | Search space | 1,573,743 rows, byte-identical, same order | PASS |
| 2 | TF2G `importance` | 2,597,463 links, identical link set, 100 % bit-identical (max rel err 0) | PASS |
| 3 | TF2G `rho` | max abs err 5.5e-14 | PASS |
| 4 | TF2G `regulation` | 0 mismatches | PASS |
| 5 | TF2G row order | identical | PASS |
| 6 | R2G GBM `importance` | 1,573,743 links, 100 % bit-identical | PASS |
| 7 | R2G RF `importance` | 1,573,743 links, 100 % bit-identical | PASS |
| 8 | R2G `rho` | max abs err 4.4e-16, NaN positions identical | PASS |
| 9 | R2G `importance_x_rho`, `importance_x_abs_rho` | max rel err 7.5e-16 | PASS |
| 10 | R2G `Distance`, row order | identical (GBM and RF) | PASS |
| 11 | `n_jobs=1` vs `n_jobs=6` | identical (TF2G 40 genes, R2G GBM 40 genes, R2G RF 12 genes) | PASS |

Two order details had to be reproduced to pass rows 1 and 5:

- **Negative extended starts (row 1).** pyranges' join (NCLS, `ALL_POSITIVE_ORIENTATION`)
  stores an interval with a negative start as the reverse-orientation interval
  `[-end, -start)`. Genes within 150 kb of a chromosome start therefore only catch regions
  starting before `-ExtStart`; `get_search_space` applies the same rule (36 genes affected).
- **Tied importances (row 5).** scenicplus sorts TF2G links with
  `np.argsort(kind="quicksort")`; 4.7 % of the importances are tied, and their order
  depends on numpy's SIMD sort (numpy 1.26 AVX-512 x86-simd-sort for the reference; numpy
  2.x orders ties differently). Float keys are sorted with an emulation of the numpy 1.26
  AVX-512 argsort (`_links_argsort.py`, now merged into `_np126.py` with the topics and eGRN copies). The TF → TF rows are
  emitted in `PYTHONHASHSEED=0` set order with `self_link_order="upstream"`.

### Timing

Full runs were made as `srun --overlap` steps inside a shared allocation with other jobs
competing for the same cores (8 threads measured at ~2× a single thread), so wall-clock
times are not comparable to the reference's 24 dedicated workers. Single-core times on the
same machine are the fair comparison.

| Stage | scenicplus reference | Epione full run (8 threads, shared) | Single core, same inputs: scenicplus / Epione |
|---|---|---|---|
| Search space | 224 s | 18 s | 224 s / 17 s (13×) |
| TF2G (GRNBoost2) | 1,166 s × 24 workers | 2,286 s | 16 expressed genes 7.65 s / 7.21 s (1.06×); an all-zero gene 1.7 s / 0 s (6,586 such genes, ~40 % of the reference CPU time) |
| R2G GBM | 488 s × 24 workers | 2,490 s | 12 genes 5.19 s / 3.93 s (1.3×) |
| R2G RF | 3,186 s × 24 workers | 6,099 s | 6 genes 12.8 s / 8.3 s (1.55×) |

The tree models do the same work as scikit-learn's Cython code; reproducing its exact sample
order (introsort ties, Fisher-Yates feature draws) rules out algorithmic shortcuts such as
presorting. The per-tree speedup comes from removing the Python, joblib and dask overhead
around each tree.

## Fast modes (`method="fast"`, `method="corr"`): pre-registered validation

Registered **before** any fast-mode code was written or run; the gates above (rows 1–11,
`method="exact"`) are unchanged and still apply to the exact path.

`method="fast"` is not meant to reproduce scikit-learn. It is a histogram GBM (features
binned once and shared across targets, GRNBoost2-like semantics: least squares, depth-3
trees, learning rate 0.01, subsample 0.9 with arboreto's OOB early stopping for TF2G,
500 trees for R2G, gain importances, correlation sign). `method="corr"` (R2G only) ranks
regions by |correlation| alone. The question is whether the fast links are as good a
substitute for the exact links as the exact method is for itself.

### Why the coordinator's fixed thresholds can't be the whole gate

Before registering, the exact method was compared **with itself** across seeds
(seed 666 vs 667, 300 random expressed targets; metric definitions below):

| | median per-target Spearman | median top-k Jaccard |
|---|---|---|
| TF2G, exact vs exact | −0.11 | 0.27 (k = 50) |
| R2G GBM, exact vs exact | 0.92 | 0.67 (k = 10) |

GRNBoost2 with arboreto's monitor stops most targets at the 25-tree minimum (seen on a
probe of 12 genes: 25–55 trees). Each depth-3 tree samples 10 % of the 1,597 TFs per node,
so the ~50–150 TFs with non-zero importance are largely a random draw: the exact method
reproduces only a quarter of its own top-50 under another seed. A fixed "Spearman ≥ 0.8,
top-50 Jaccard ≥ 0.6" gate for TF2G would fail for the reference method itself. So each
gate below is `min(fixed threshold, exact-vs-exact spread)`. The fixed threshold applies
where the exact method itself meets it, and otherwise the fast mode must agree with the
reference at least as well as a second exact run with another seed does. The spread is
re-measured on the **full** data (exact seed 667 vs the reference seed 666), not taken
from the probe.

### Metrics

* Per-target Spearman: for each target, Spearman correlation of the importances over the
  union of the links either method reports (a link missing on one side counts as 0);
  targets with fewer than 3 such links are skipped; the median over targets is gated.
  R2G uses targets with ≥ 10 candidate regions.
* Top-k Jaccard: Jaccard of the top-k links by importance per target (TF2G k = 50; R2G
  k = 10, because the median gene has about 85 candidate regions, so its top 50 would be
  more than half of all candidates); median over targets.
* Downstream: `epione.scenic.build_egrn` (block ④, `PYTHONHASHSEED=0`) on the reference
  cistromes (`A/scplus_inputs`, direct and extended), with TF2G and R2G both from the
  method under test, compared with the reference eRegulons (`A/egrn`). TF-set Jaccard is
  the Jaccard of the sets of TFs that have at least one eRegulon. The target-gene
  Jaccard is computed per eRegulon name present in both, and its median is gated.

### Gates (`method="fast"`, seed 666, full data, vs the exact reference)

| # | Output | Gate |
|---|--------|------|
| F1 | TF2G median per-target Spearman | ≥ min(0.8, exact seed spread) |
| F2 | TF2G median top-50 Jaccard | ≥ min(0.6, exact seed spread) |
| F3 | R2G (GBM importance) median per-target Spearman | ≥ min(0.8, exact seed spread) |
| F4 | R2G (GBM importance) median top-10 Jaccard | ≥ min(0.6, exact seed spread) |
| F5 | eGRN TF-set Jaccard, direct and extended | ≥ min(0.8, exact seed spread) |
| F6 | eGRN median target-gene Jaccard, direct and extended | ≥ min(0.6, exact seed spread) |
| F7 | `fast` with `n_jobs=1` vs `n_jobs=N` | identical (determinism) |

Timing is reported, not gated: full TF2G and R2G wall time per mode with the thread
count, against `method="exact"` at the same thread count (target ≥ 10×).
`method="corr"` (R2G) is reported on F3, F4 and the downstream metrics without a pass/fail.
For downstream comparisons it is paired with the `fast` TF2G links.

**Amendment, made before the registered run (no gate changed).** The description above
says the fast R2G GBM uses 500 trees. During development (600- and 2,000-target subsets,
never the full data) the fast path with exact GBM settings was only about 1.9× faster than
`method="exact"` on R2G: accessibility has at most 33 distinct values per region, which
the exact splitter already sorts cheaply. So the fast R2G GBM takes the same shrinkage
budget in half the steps (`learning_rate=0.02`, 250 trees), about 3×; `learning_rate=0.1`
× 50 trees was tried and dropped (subset Spearman 0.75). The fast R2G random forest uses
100 trees instead of 1000. TF2G keeps the GRNBoost2 settings. TF2G bins are lossless: one
bin per distinct value, up to 65,535. With 255 quantile bins, the subset top-50 Jaccard
fell below the exact seed spread (0.235 vs 0.266). The registered gates F1–F7 are as
written above.

### Results (registered run; thresholds above unchanged)

`validate_fast.py` on the full data (2,313 cells; TF2G 30,015 expressed targets with links,
R2G 17,750 genes with ≥ 10 candidate regions), 8 threads, `PYTHONHASHSEED=0`. Exact
seed-spread baseline: `method="exact"` with seed 667 against the scenicplus reference
(seed 666).

| # | Metric | fast | exact seed spread | threshold | Gate |
|---|--------|------|-------------------|-----------|------|
| F1 | TF2G median per-target Spearman | −0.104 | −0.113 | ≥ −0.113 | PASS |
| F2 | TF2G median top-50 Jaccard | 0.282 | 0.266 | ≥ 0.266 | PASS |
| F3 | R2G median per-target Spearman | 0.891 | 0.920 | ≥ 0.8 | PASS |
| F4 | R2G median top-10 Jaccard | 0.667 | 0.667 | ≥ 0.6 | PASS |
| F5 | eGRN TF-set Jaccard, direct | 0.404 | 0.408 | ≥ 0.408 | **FAIL** |
| F5 | eGRN TF-set Jaccard, extended | 0.500 | 0.456 | ≥ 0.456 | PASS |
| F6 | eGRN median target-gene Jaccard, direct | 0.453 | 0.422 | ≥ 0.422 | PASS |
| F6 | eGRN median target-gene Jaccard, extended | 0.452 | 0.435 | ≥ 0.435 | PASS |
| F7 | `n_jobs=1` vs 8 (500 targets, TF2G and R2G); subset run equals full run | identical | | | PASS |

F5 (direct) misses by 0.005, about one TF: fast links give 67 TFs with direct eRegulons
(94 eRegulons), the exact seed-667 run 83 TFs (114), the reference 86 (125). Fewer
eRegulons, not different ones: the target-gene overlap of the eRegulons both have (F6) is
higher for fast than for the second exact run.

The seed-spread baseline is itself a finding. For two exact SCENIC+ runs that differ only in
the TF2G / R2G seed, the eRegulon TF sets have a Jaccard index of 0.41 (direct) and 0.46
(extended), and the median target-gene Jaccard of shared eRegulons is 0.42 / 0.43. The instability comes from GRNBoost2's
random feature draws on genes that stop at 25 trees.

`method="corr"` R2G (reported, no gate): Spearman 0.285, top-10 Jaccard 0.176; with fast
TF2G, eGRN TF-set Jaccard 0.386 / 0.397 and target-gene Jaccard 0.374 / 0.362 (direct /
extended).

#### Timing (full data, 8 threads, same shared allocation)

| Step | `method="exact"` | `method="fast"` | speedup | `method="corr"` |
|---|---|---|---|---|
| TF2G (GRNBoost2, 36,601 targets) | 1,758 s | 234 s | 7.5× | — |
| R2G GBM (18,549 genes, 1.57 M pairs) | 1,119 s | 212 s | 5.3× | 38 s (29×) |

The ≥ 10× target is not met. Single-thread profiling of TF2G: 50 ms per target (fast) vs
430 ms (exact), 8.6×, with early stopping at the same ~27 trees. The remaining cost is about
half histogram accumulation (~4 ns per non-zero entry, bound by the dependency chain on
the bin counters) and half per-draw overhead from scikit-learn's rule of redrawing constant
features. For R2G, the exact splitter is already cheap on few-valued accessibility (at
most 33 distinct values per region). R2G random forest (not gated; 600-target subset vs the
reference): 5.8× faster than exact, Spearman 0.984, top-10 Jaccard 0.818.
