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
  AVX-512 argsort (`_links_argsort.py`, shared with the eGRN block). The TF → TF rows are
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
