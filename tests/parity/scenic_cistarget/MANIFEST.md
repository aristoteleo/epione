# Parity manifest — `epione.scenic` cisTarget / DEM vs pycistarget 1.1 + ctxcore 0.2.0

Pre-registered before the first comparison; thresholds are read-only.

Reference: the SCENIC+ 1.0 pipeline (scenicplus 1.0a2 `840dab8`, pycistarget 1.1 `5aa5176`,
ctxcore 0.2.0 `e68a0b1`, numpy 1.26.4, numba 0.59.0) on the 10x multiome human cerebellum 3k
tutorial data. Region sets: `Topics_otsu`, `Topics_top_3k` (K = 20 cisTopic model) and
`DARs_cell_type`. Databases: `hg38_screen_v10_clust.regions_vs_motifs.{rankings,scores}.feather`,
motif annotation `motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl`. The reference was run with
`PYTHONHASHSEED=0`, one thread, and scenicplus CLI defaults:
- **cisTarget:** fraction_overlap 0.4, auc 0.005, nes 3.0, rank 0.05, Direct + Orthology annotation.
- **DEM:** max_bg 500, promoter balancing ±1000 bp around gene Start, seed 666, adjpval 0.05,
  log2fc 1.0, mean_fg 0, motif_hit_thr 3.0.

The reference outputs are generated outside this repository and read from
`$EPIONE_SCENIC_PARITY_DIR/A/{binarize_dar,cistarget,dem,scplus_inputs,inputs}`.
`parity_cistarget.py` and `tests/test_scenic_cistarget.py` compare against them, and skip when the
variable is unset.

## Gates

| # | Output | Class (rebuildr) | Metric | Threshold |
|---|---|---|---|---|
| 1 | region set → DB region mapping (`regions_to_db`, every set, cisTarget and DEM) | exact | set equality of (Target, Query) rows | identical |
| 2 | AUC of **every** motif, every region set | deterministic-strict | max abs err | 0 (bit-identical) |
| 3 | NES of the enriched motifs, every region set | deterministic-standard | max abs err | 1e-12 |
| 4 | enriched motif set, `Rank_at_max`, `Motif_hits` count | exact | equality | identical |
| 5 | motif hits per motif (`Database` and `Region_set` coordinates) | exact | set equality | identical |
| 6 | motif → TF annotation columns (`Direct_annot`, `Orthology_annot`) | exact, order-canonicalised | set of TFs per cell | identical |
| 7 | cistromes (name incl. `_(Nr)` size suffix → region set), `Database` and `Region_set` | exact | set equality | identical |
| 8 | DEM foreground regions; background regions **in sampling order** (CPython `random.Random(666).shuffle`) | exact | list equality | identical |
| 9 | DEM per-motif Wilcoxon p-value and BH-adjusted p-value (all motifs) | deterministic-standard | max abs err | 1e-12 |
| 10 | DEM enriched motifs, `Log2FC`, `Mean_fg`, `Mean_bg` | exact set; values deterministic-standard | set equality; max abs err | identical; 1e-12 |
| 11 | merged scenicplus cistromes (`cistromes_{direct,extended}`: TF × region membership, `motifs` per TF) | exact, order-canonicalised | set equality per TF | identical |

**Order canonicalisation (gates 6, 7, 11).** Upstream builds the TF annotation strings with
`', '.join(set(...))` and the cistrome rows from `set` unions, so their order depends on Python's
string hash seed. Upstream is irreproducible without `PYTHONHASHSEED`. These outputs are compared
as sets, and Epione emits them sorted.

The MOODS / JASPAR backend (Epione's own motif scanning turned into a rankings or scores DB) has no
upstream counterpart. Its gate is internal consistency: the same cisTarget / DEM code path gives
identical results whether the matrix is passed in memory or read back from the feather file it
writes.

## Results (appended after the run; thresholds above unchanged)

56 cisTarget region sets and 56 DEM comparisons (all the reference has; upstream skipped none).

| # | Value | Pass |
|---|---|---|
| 1 | cisTarget 56/56, DEM 56/56 identical | ✅ |
| 2 | max abs err 0 (5,876 motifs × 56 sets) | ✅ |
| 3 | 0 | ✅ |
| 4 | motif sets 56/56, `Rank_at_max` 56/56, `Motif_hits` 56/56 identical | ✅ |
| 5 | cisTarget 112/112, DEM 112/112 identical | ✅ |
| 6 | cisTarget 112/112, DEM 112/112 identical | ✅ |
| 7 | cisTarget 112/112, DEM 112/112 identical | ✅ |
| 8 | foreground 56/56; background order 56/56 identical (per chromosome, because the reference dump is `PyRanges.df`) | ✅ |
| 9 | p 0, adjusted p 0 | ✅ |
| 10 | motif sets 56/56; `Log2FC` / `Mean_fg` / `Mean_bg` / `Adjusted_pval` max abs err 0 | ✅ |
| 11 | direct (709 TFs) and extended (342 TFs): same TFs, same region membership, same motifs | ✅ |

**MOODS backend (internal consistency).** `test_feather_roundtrip_and_hdf5`: an in-memory database
and the same database written with `write_feather` and read back give identical
`motif_enrichment` and motif hits. ✅

**Cross-check of the hdf5 format.** `write_hdf5` was run on these results. Upstream
`pycistarget.input_output.read_hdf5` reads the file, and scenicplus'
`get_and_merge_cistromes` run on it reproduces the reference `cistromes_{direct,extended}`
exactly.

**Bugs found and fixed while reaching the gates** (none needed a threshold change):

1. The comparison script first parsed the reference TSVs with pandas' default float parser,
   which is not correctly rounded. That caused 1-ulp "differences"; it now uses `round_trip`.
2. Upstream's numba `np.mean` accumulates float32 scores in float64. Epione's first kernel
   used float32, which differed by up to 8.6e-6 in `Mean_fg`.

**Timing** (same node; upstream single-threaded as in the reference run):

| Step | upstream | Epione, 1 thread | Epione, 8 threads |
|---|---|---|---|
| cisTarget compute (56 sets) | 846 s | 92 s | 72 s |
| DEM compute (56 comparisons) | 581 s | 170 s | 80 s |
| database read | 185 s + 105 s | — | 37–250 s + 51–150 s (I/O, page-cache dependent) |
| `prepare_menr` | 42 s | 9 s | 9 s |
| peak RSS | 26.8 GB / 17.7 GB | — | 18.5 GB (both steps, one process) |
