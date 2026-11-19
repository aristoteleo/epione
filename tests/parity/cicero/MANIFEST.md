# Parity manifest — Cicero (Monocle 3 branch, v1.3.9) → `epione.tl`

Pre-registered before the first line of the Python port; thresholds are read-only.
Machine-readable copy: [`manifest.yaml`](manifest.yaml). Dependency audit: [`DISCOVERY.md`](DISCOVERY.md).

Reference: R 4.3.3, cicero 1.3.9 (cole-trapnell-lab/cicero-release@495ef0d, branch `monocle3`),
monocle3 1.3.1, glasso 1.11, run by [`r_reference_driver.R`](r_reference_driver.R).
`set.seed(seed)` is called immediately before `make_cicero_cds()` and again before `run_cicero()`.

Fixtures
- **F1** `cicero_data` (Cicero's example data; HSMM, hg19 chr18, 200 cells), binarised by `make_atac_cds`;
  UMAP from monocle3 computed once in R and fed to both sides; genome = full chr18.
- **F2** E8.5 heart multiome ATAC, chr19 only (16,787 cells), binarised; reduced coordinates =
  `emb_umap.harmony.rna.csv` on both sides; genome = mm10 chr19.
- **F3** same data, whole genome — runtime, run once.

Modes for the random parts: **rrng** (Python uses `RRandom(seed)`, the bit-exact R Mersenne-Twister),
**injected** (Python receives R's draws), **numpy** (free-running numpy RNG, distributional gate only).

| # | Output | Fixtures | Mode | Class | Metric | Threshold |
|---|---|---|---|---|---|---|
| 1 | aggregation bins + sampled windows | F1 F2 | rrng | det.-strict | exact equality | exact |
| 2 | kNN neighbour sets | F1 F2 | – | det.-strict | per-cell set equality | exact |
| 3 | aggregated, size-factor-normalised matrix (+ size factors) | F1 F2 | injected | det.-standard | max abs err | 1e-8 |
| 4 | glasso `w`, `wi` on identical `S`, `rho` | F1 F2 synthetic | – | det.-standard | max abs err; `wi` zero pattern | 1e-8; exact |
| 5 | distance parameter per sampled window | F1 F2 | injected | det.-bounded | max rel err | 1e-6 |
| 6 | mean distance parameter | F1 F2 | rrng | det.-bounded | rel err | 1e-6 |
| 7 | per-window model status | F1 F2 | rrng | det.-strict | identical | exact |
| 8 | set of (Peak1, Peak2) rows; NA pattern | F1 F2 | rrng | det.-strict | identical | exact |
| 9 | coaccess on matched pairs | F1 F2 | rrng | det.-standard | max abs err | 1e-8 |
| 10 | coaccess on matched pairs | F1 F2 | rrng | ordinal | Pearson, Spearman | ≥ 0.9999 |
| 11 | pairs with coaccess > 0.25 | F1 F2 | rrng | ranked | Jaccard | ≥ 0.999 |
| 12 | coaccess on matched pairs | F1 F2 | injected | det.-standard | max abs err | 1e-8 |
| 13 | CellOracle format | F1 F2 | – | det.-strict | columns `Peak1, Peak2, coaccess`; ids `chr_start_end`; both orientations | exact |
| 14 | `build_gene_activity_matrix` | F1 | same input | det.-standard | max abs err | 1e-8 |
| 15 | `normalize_gene_activities` | F1 | same input | det.-standard | max err / max\|R\| | 1e-8 |
| 16 | CCAN membership at R's cutoff | F1 F2 | same connections | clustering | ARI | ≥ 0.95 (*) |
| 17 | `find_ccan_cutoff` | F1 F2 | same connections | det.-bounded | \|Δ cutoff\| | ≤ 0.02 or within R-vs-R range |
| 18 | coaccess, free-running numpy RNG | F2 | numpy | stochastic | Pearson; Jaccard(>0.25) | ≥ median R-vs-R − 0.02 |

(*) Louvain (`igraph::cluster_louvain`) draws from R's RNG; python-igraph draws from Python's.
If the median R-vs-R ARI across seeds {2017, 7, 11, 23} is itself below 0.95, gate 16 becomes
ARI(Py, R) ≥ the minimum R-vs-R ARI. This rule is fixed here, before any Python result exists.

### Addendum 1 — pre-registered before porting the remaining exported utilities

The coverage audit (`AUDIT.md`) listed six more exported R functions. Gates for them were added
here and committed **before** their Python code; gates 1–18 are unchanged.

| # | Output | Fixture | Class | Metric | Threshold |
|---|---|---|---|---|---|
| 19 | `make_atac_cds` | F1 | det.-strict | peak / cell order, values, `chr bp1 bp2`, `num_genes_expressed` | exact |
| 20 | `aggregate_nearby_peaks` (distance 1000, 10000) | F1 | det.-strict | merged names, summed matrix | exact |
| 21 | `aggregate_by_cell_bin` (group = `timepoint`) | F1 | det.-standard | summed matrix, `Size_Factor`, `num_genes_expressed` | 1e-8 |
| 22 | `compare_connections` (maxgap 0, 1000) | F1 | det.-strict | logical vector | exact |
| 23 | `find_overlapping_coordinates` (single / multiple query, maxgap 0, 500) | F1 | det.-strict | returned peaks | exact |
| 24 | `df_for_coords` | F1 + R help examples | det.-strict | `chr bp1 bp2 Peak` | exact |

## Results

(appended after the runs; thresholds above unchanged)

Final code (commit after iteration 7), seed 42. F1 = `cicero_data`, F2 = E8.5 heart chr19.

| # | F1 | F2 | Pass |
|---|---|---|---|
| 1 | 25 bins and 115 sampled windows identical to R | 3,235 bins, 110 windows identical | ✅ |
| 2 | 0 cells with a different neighbour set | 0 | ✅ |
| 3 | matrix 0, size factors 0 | 0, 0 | ✅ |
| 4 | w 2.7e-15, wi 5.4e-9 (\|wi\| ≤ 5e3), zero pattern identical, iterations identical | 4.4e-15, 1.0e-15, identical | ✅ |
| 5 | 2.6e-15 | 1.3e-15 | ✅ |
| 6 | 0 (bit-identical mean) | 0 | ✅ |
| 7 | identical | identical | ✅ |
| 8 | 543,286 rows, same order; NA identical | 1,240,298 rows, same order; NA identical | ✅ |
| 9 | 8.0e-16 | 4.4e-16 | ✅ |
| 10 | Pearson 1 − 1e-14, Spearman 1 − 2e-12 | 1.0, 1 − 2e-13 | ✅ |
| 11 | 1.0 | 1.0 | ✅ |
| 12 | 8.0e-16 | 4.4e-16 | ✅ |
| 13 | ✅ | ✅ | ✅ |
| 14 | 8.9e-16; annotation identical | — | ✅ |
| 15 | 6.7e-16 | — | ✅ |
| 16 | ARI 1.0 (tables identical for seeds 42, 2017, 7, 11, 23); R-vs-R median 0.964 | ARI 1.0 (identical, 5 seeds); R-vs-R median 0.995 | ✅ |
| 17 | 0.46 = R | 0.27 = R | ✅ |
| 18 | — | Py(`rng="numpy"`) vs R(42): Pearson 0.9913, Jaccard 0.858; R(42) vs R(2017/7/11/23): Pearson 0.989–0.993 (median 0.9916), Jaccard 0.833–0.871 (median 0.868); thresholds 0.9716 / 0.8478 | ✅ |
| 19–24 | all identical (21: size factor 1.1e-16) | — | ✅ |

Gate 16 with the earlier python-igraph Louvain (iterations 0–6): F1 ARI 0.935 < 0.95 (**fail**),
F2 0.995. Fixed by iteration 7 (igraph Louvain on R's RNG + R's `round`), threshold unchanged.
