# Parity manifest — SCENIC+ block ④: eGRN, AUCell, RSS

Reference: SCENIC+ 1.0a2 (`840dab85`), pycisTopic 2.0a0, ctxcore 0.2.0, pyscenic 0.12.1,
gseapy 0.10.8, numpy 1.26.4, pandas 1.5.3, run on the 10x multiome human cerebellum 3k
tutorial dataset (2,313 multiome cells). Every reference stage was run twice (S vs S2) and is
byte-identical. Thresholds were registered before the first comparison and are never loosened.

The references are produced outside this repository with the upstream packages; the
repository contains none of their code. `parity_scenic_egrn.py` and the parity tests read
`$EPIONE_SCENIC_PARITY_DIR/A/{tf2g,r2g,scplus_inputs,egrn,aucell_rss}` and are skipped when
the variable is not set.

Upstream call being reproduced (`infer_grn` / `calculate_auc` / `regulon_specificity_scores`):
order by importance, `gsea_n_perm=1000`, quantiles 0.85/0.90/0.95, top-n 5/10/15 per gene,
BASC on, rho dichotomisation of TF2G, R2G and eRegulons, `rho_threshold=0.05`,
`min_target_genes=10`, NES > 0, adjusted p < 1, merge on, `seed=666`; triplet ranking against
the hg38 screen v10 clustered rankings database; AUCell `auc_threshold=0.05`, no normalisation;
RSS over `Seurat_cell_type` on the direct and extended gene-based AUCs.

Taxonomy: omicverse-rebuildr `PARITY_TAXONOMY.md`. Every output here is class 1: the
stochastic parts (gseapy's `RandomState` permutations, the `default_rng(123)` tie-breaking in
AUCell ranking and in the triplet ranking) are reproduced draw for draw, so they are gated
like deterministic outputs.

Order conventions. Upstream builds several lists through Python `set`s of strings and tuples,
so its row order depends on `PYTHONHASHSEED` (the reference used `0`). The triplet ranking
breaks ties by row position, so `triplet_rank` depends on that order too. Rows 4 and 5 are
therefore evaluated in a process started with `PYTHONHASHSEED=0` and require the identical
row order. Row 3 is order-free and must hold for any hash seed.

| # | Output | Class | Metric | Threshold |
|---|--------|-------|--------|-----------|
| 1 | per-gene R2G thresholds (quantile 0.85/0.90/0.95, top 5/10/15, BASC), both rho signs | 1 deterministic (strict) | max abs err | `== 0` |
| 2 | GSEA per eModule (NES, nominal p, leading edge), `seed=666` | 1 emulated RNG (strict) | NES/p max abs err; leading-edge set equality | `== 0` |
| 3 | eRegulon table as a set of rows (all 15 columns except `triplet_rank`), direct and extended | 1 deterministic (strict) | multiset equality, floats bitwise | `== 0` |
| 4 | eRegulon table row order (`PYTHONHASHSEED=0`) | 1 deterministic (strict) | identical order | `== 0` |
| 5 | `triplet_rank` (`PYTHONHASHSEED=0`) | 1 emulated RNG (strict) | identical integers | `== 0` |
| 6 | AUCell gene-based AUC, direct and extended | 1 emulated RNG (standard) | max abs err; identical row/column labels | `<= 1e-8` |
| 7 | AUCell region-based AUC, direct and extended | 1 emulated RNG (standard) | max abs err; identical row/column labels | `<= 1e-8` |
| 8 | RSS (16 cell types × direct+extended gene-based eRegulons) | 1 deterministic (standard) | max abs err; identical labels | `<= 1e-8` |

Hard ceiling for any class-1 row: `1e-6`.

Speed: wall time of each stage against the CONTRACT baseline (run A, upstream with
`n_cpu=4`/joblib): `egrn_direct` 768 s, `egrn_extended` 374 s, `aucell_direct` 159 s,
`aucell_extended` 159 s, `rss` 0.9 s. Reported, not gated.

## Results

`parity_scenic_egrn.py` (with `--ranking-db` for the triplet motif ranks) was run in two
environments: Epione's (numpy 2.4.6, pandas 2.3.3, SciPy 1.17.1, numba 0.68) and the reference
one (numpy 1.26.4, pandas 1.5.3, SciPy 1.12.0, numba 0.59). The results are identical in both.

| # | Output | Value (direct / extended) | Gate |
|---|--------|---------------------------|------|
| 1 | R2G thresholds, 14 binarisations per run | max abs err 0 / 0 | pass |
| 2 | GSEA, 12,224 / 5,674 calls: NES, nominal p, leading edge, NaN pattern | max abs err 0 / 0; leading edges identical | pass |
| 3 | eRegulon table as a set of rows (28,242 rows / 125 eRegulons; 13,700 / 74) | identical | pass |
| 4 | row order, `PYTHONHASHSEED=0` | identical | pass |
| 5 | `triplet_rank`, `PYTHONHASHSEED=0` | identical | pass |
| 6 | AUCell gene-based AUC (2,313 cells) | max abs err 0 / 0 | pass |
| 7 | AUCell region-based AUC (434,673 regions ranked) | max abs err 0 / 0 | pass |
| 8 | RSS (16 cell types × 199 eRegulons) | max abs err 0 | pass |

Also checked: the eModules before GSEA (TF, context and md5 of the ordered region/gene rows
of all 8,586 / 3,954 candidates) are identical, and the TF-to-region motif ranks read from the
cisTarget database equal the reference `df_TF_region_max_rank`. With `PYTHONHASHSEED=1` row 3
still holds while rows 4 and 5 differ, as they do in SCENIC+ itself.

Extra reference dumps (`$EPIONE_SCENIC_PARITY_DIR/block4/`): per-call thresholds, eModule
digests, per-call GSEA results (exact float bits in `gsea_*_nes_pval.npy`) and the TF-region
rank tables, produced by replaying `infer_grn` with `n_cpu=1` (`block4/scripts/ref_extra.py`;
its eRegulon tables are byte-identical to `A/egrn/`).

Reproducibility notes (why exactness does not depend on the installed numpy/SciPy):

* SCENIC+ breaks ties by shuffling and then calling numpy's unstable `argsort`. On AVX-512 CPUs
  numpy 1.26 uses x86-simd-sort; numpy 2.x ships a newer version with a different tie order
  (36–61% of AUC values change, by up to 0.024; most triplet ranks change).
  `epione/scenic/_egrn_argsort.py` (now merged into `_np126.py`) emulates numpy 1.26's algorithm exactly. It was checked
  against `np.argsort` under numpy 1.26.4 on 672 tie-heavy arrays (int32/uint32/int64/float32/
  float64/int16, n = 2..100,000), with no mismatch.
* numpy 1.x summed reductions in 8,192-element blocks (TF rankings reach 12,630 genes); the
  GSEA normaliser emulates this explicitly.
* SciPy 1.17's `rel_entr` changes the last bit of 82 RSS values relative to SciPy 1.12; the RSS
  kernel calls the C library `log` as SciPy 1.12 did.
* GSEA permutations are drawn with an MT19937 implementation of legacy `RandomState.shuffle`,
  checked against numpy.

## Speed

Same data, same node (shared allocation). SCENIC+ ran with `n_cpu=4`; Epione used 8 numba
threads with the JIT cache warm. The first call compiles for about a minute.

| stage | SCENIC+ | Epione (numpy 2.4.6 env) | speed-up |
|---|---|---|---|
| eGRN direct (incl. ~93 s / 58 s reading the ranking DB for the triplet ranking) | 768 s | 136 s | 5.6x |
| eGRN extended (same) | 374 s | 59 s | 6.3x |
| AUCell direct + extended | 318 s | 74 s (one ranking of both matrices: 71 s, shared) | 4.3x |
| RSS | 0.9 s | 0.7 s | — |
