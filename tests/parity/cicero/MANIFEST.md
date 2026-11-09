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

## Results

(appended after the runs; thresholds above unchanged)
