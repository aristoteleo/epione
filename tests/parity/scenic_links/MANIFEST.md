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
