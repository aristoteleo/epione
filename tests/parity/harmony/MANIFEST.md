# Parity manifest — `epi.tl.harmony` vs ArchR `addHarmony` (harmony 1.2.4)

Pre-registered before the first comparison; thresholds are read-only.

Reference: `set.seed(seed); addHarmony(proj, reducedDims = "IterativeLSI", groupBy = "Sample")`
(ArchR 1.0.3, harmony 1.2.4, R 4.4.3, OpenBLAS 0.3.32) on the ArchR Hematopoiesis tutorial
project (10,660 cells × 30 LSI dims, 3 samples). The ArchR reference outputs are generated
outside this repository with ArchR 1.0.3 and read from `$EPIONE_ARCHR_PARITY_DIR/harmony/`
(seed 42 run in its `seed42/` subdirectory); `parity_harmony.py` and `tests/test_harmony.py`
compare against them and skip when the variable is not set. The small references in this
directory (`small_*.tsv`) are R harmony 1.2.4 / `stats::kmeans` outputs on the deterministic
matrix built in `tests/test_harmony.py`.

| # | Output | Mode | Class (rebuildr) | Metric | Threshold |
|---|---|---|---|---|---|
| 1 | R RNG stream (`runif`, `sample.int`, `(int) runif(0, RAND_MAX)`, `.Random.seed`) | seed 1 | deterministic-strict | exact equality | 0 |
| 2 | `stats::kmeans(Z_cos, 100, iter.max = 25, nstart = 10)` centers | seed 1 | deterministic-standard | max abs err | 1e-8 |
| 3 | soft assignments `R` after init and after every harmony round | injected R centroids + R RNG state | deterministic-standard | max abs err | 1e-8 |
| 4 | `Z_corr` after every harmony round | injected | deterministic-standard | max abs err | 1e-8 |
| 5 | k-means rounds per harmony round, number of harmony rounds | injected + free | exact | equality | — |
| 6 | `objective_harmony` (float32 in C++) | injected | deterministic-standard, rtol mode | max rel err | 1e-6 |
| 7 | final Harmony embedding (`getReducedDims(proj, "Harmony")`) | free-running, seed 1 and seed 42 | deterministic-standard | max abs err | 1e-8 |
| 8 | final Harmony embedding | free-running, seed 1 | embedding | Procrustes similarity | ≥ 0.95 |
| 9 | 30-NN graph of the embedding | free-running, seed 1 | ranked | mean per-cell kNN Jaccard | ≥ 0.95 |
| 10 | Leiden clusters (same scanpy pipeline on both embeddings) | free-running, seed 1 | clustering | ARI | ≥ 0.95 |

Gates 8–10 are implied by gate 7. They are reported because they are the
class-appropriate metrics for an embedding.

## Results (appended after the run; thresholds above unchanged)

| # | Value | Pass |
|---|---|---|
| 1 | identical (state, runif, sample.int, runif(0, RAND_MAX)) | ✅ |
| 2 | 8.9e-16 | ✅ |
| 3 | init 3.2e-15; rounds 1–3: 4.0e-15 / 3.2e-14 / 5.7e-14 | ✅ |
| 4 | rounds 1–3: 3.7e-13 / 2.5e-13 / 4.0e-13 | ✅ |
| 5 | k-means rounds [13, 9, 5] = R; 3 harmony rounds = R | ✅ |
| 6 | 1.6e-9 (R printed with 9 significant digits) | ✅ |
| 7 | seed 1: 3.6e-13; seed 42: 2.6e-13 | ✅ |
| 8 | 1.000000 | ✅ |
| 9 | 1.000000 | ✅ |
| 10 | 1.000000 | ✅ |

Runtime (1 thread, same node): ArchR `addHarmony` 6.2 s, `epi.tl.harmony` 2.9 s.
