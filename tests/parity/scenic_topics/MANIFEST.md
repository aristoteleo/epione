# Parity manifest — `epione.scenic` topics block vs pycisTopic (SCENIC+ 1.0)

Pre-registered before the first comparison; thresholds are read-only.

Reference: pycisTopic 2.0a0 (`787ce42`) with the `lda` 3.0.0 collapsed Gibbs backend
(`aee65d3`), tmtoolkit 0.12.0, pyscenic 0.12.1, numpy 1.26.4, scipy 1.12.0, pandas 1.5.3,
`PYTHONHASHSEED=0`, one thread. Dataset: 10x multiome human cerebellum 3k (the SCENIC+ 1.0
tutorial dataset, 2,392 annotated cells, 436,234 consensus regions). The reference outputs
are generated outside this repository and read from `$EPIONE_SCENIC_PARITY_DIR` (run `A` =
canonical, run `S` = small LDA run with 5 Gibbs iterations); `parity_topics.py` and
`tests/test_scenic_topics.py` compare against them and skip when the variable is not set.

Each stage is checked twice: **isolated** (the Epione function is fed the reference's
upstream outputs) and, where cheap, **chained** (fed Epione's own upstream outputs).

| # | Output | Upstream call | Class (rebuildr) | Metric | Threshold |
|---|---|---|---|---|---|
| 1 | fragment count matrix (regions × cells), region and cell order | `create_cistopic_object_from_fragments(valid_bc, blacklist, split_pattern="-")` | deterministic-strict | exact equality | 0 |
| 2 | binary matrix, `cisTopic_nr_frag/acc` per cell and per region | same | deterministic-strict | exact equality | 0 |
| 3 | `lda` RNG stream: 131,072 uniforms + per-iteration `RandomState.shuffle` | `lda.LDA(random_state=555)` | deterministic-strict | exact equality | 0 |
| 4 | final topic assignment counts `nz_` (`topic_ass`) | `run_cgs_models`, S: K∈{10,20}, 5 it.; A: K∈{10,20,30,40}, 500 it. | stochastic, emulated | exact equality | 0 |
| 5 | `cell_topic` (topics × cells), `topic_region` (regions × topics) | same | stochastic, emulated | max abs err | 1e-8 |
| 6 | `loglikelihood`, `Arun_2010`, `Cao_Juan_2009`, `Mimno_2011` (model + per topic), `Marg_Topic` | same | deterministic-standard, rtol | max rel err | 1e-8 |
| 7 | automatically selected model (`evaluate_models(select_model=None)`) | same | exact | equality | — |
| 8 | binarized topics, regions (`ntop=3000`, `otsu`) and cells (`li`, `nbins=100`): selected features | `binarize_topics(smooth_topics=True)` | deterministic-strict | set + order equality | 0 |
| 9 | binarized topic scores | same | deterministic-standard | max abs err | 1e-8 |
| 10 | imputed accessibility (int32) and kept regions | `impute_accessibility(scale_factor=10**6)` | deterministic-strict | exact equality | 0 |
| 11 | normalized accessibility (first 2,000 regions) | `normalize_scores(10**4)` | deterministic-standard, rtol | max rel err | 1e-8 |
| 12 | highly variable regions | `find_highly_variable_features(min_disp=0.05, min_mean=0.0125, max_mean=3, n_bins=20)` | deterministic-strict | list equality | 0 |
| 13 | DARs per cell type: region list and order | `find_diff_features(variable="Seurat_cell_type", adjpval 0.05, log2fc log2(1.5))` | deterministic-strict | list equality | 0 |
| 14 | DAR `Log2FC`, `Adjusted_pval` | same | deterministic-standard, rtol | max rel err | 1e-8 |
| 15 | region-set BED files (`Topics_otsu`, `Topics_top_3k`, `DARs_cell_type`) | driver `write_region_sets` | deterministic-strict | byte equality | 0 |

Gate 10 is the one place where the reference result depends on the BLAS `sgemm` kernel
(float32 product truncated to int32). It is gated at exact equality anyway; if it fails,
the number of differing entries is reported instead of loosening the gate.

## Results (appended after the run; thresholds above unchanged)

Run A = SCENIC+ 1.0 tutorial parameters (4 models × 500 Gibbs sweeps); run S = same
data, 5 sweeps, K ∈ {10, 20}. Stages 8–15 were fed the reference K = 20 model; stages
1–2 started from the raw fragment file.

| # | Value | Pass |
|---|---|---|
| 1 | fragment matrix identical (436,234 × 2,392, 32.7 M non-zeros); region order and cell order identical | ✅ |
| 2 | binary matrix and all per-cell / per-region counts identical; log10 columns also bit-identical | ✅ |
| 3 | random stream reproduced through numpy's `RandomState` (assignments below are bit-identical) | ✅ |
| 4 | `topic_ass` identical for S (K = 10, 20) and A (K = 10, 20, 30, 40) | ✅ |
| 5 | `cell_topic`, `topic_region`: max abs err 0 (S and A, all K) | ✅ |
| 6 | Arun 2010, Cao Juan 2009, Mimno 2011 (model and per topic), log-likelihood, marginal topic distribution: max rel err 0 (S and A, all K) | ✅ |
| 7 | selected model K = 20 (S and A) | ✅ |
| 8 | `ntop` (60,000 rows), `otsu` (334,420), cell `li` (9,774): identical features in identical order | ✅ |
| 9 | scores: max abs err 0 | ✅ |
| 10 | imputed accessibility: 434,836 kept regions identical, 0 differing int32 entries (of 1.04 × 10⁹) | ✅ |
| 11 | normalized accessibility: max rel err 0 | ✅ |
| 12 | 70,220 highly variable regions, identical list | ✅ |
| 13 | DARs for 16 cell types (309,180 rows): identical regions in identical order | ✅ |
| 14 | `Log2FC`, `Adjusted_pval`: max rel err 0 | ✅ |
| 15 | 56 region-set BED files byte-identical | ✅ |

Gate 4–6 for run A: the full 4 × 500-sweep fit was run once; per-model assignments,
`cell_topic` and `topic_region` were compared directly, and the final metric code was
then re-applied to the fitted counts (`TopicModel.extra`) — `run_topic_models` computes
its metrics with exactly that code.

Bit-identity of the derived quantities needed six numpy-1.26-on-AVX-512 behaviours to
be reproduced (`epione/scenic/_np126.py`; every one is checked against numpy 1.26.4 in
`tests/test_scenic_topics.py`): the tie order of `np.argsort` (x86-simd-sort plus its
`std::sort` fallback), `np.argpartition` (scalar introselect), SVML `log`, `log1p` and
`log10` (the `vrcp14pd` approximation is replaced by its 16 measured switching points,
checked on 3.4 × 10⁷ inputs), and the 8,192-element buffered pairwise `np.sum`. Without
them (`compat="native"`), the coherence of tied regions, the order of nearly tied
binarized regions and the last bit of several floats follow the installed numpy instead.
The `cell_order="first_overlap"` order of `cistopic_matrix` reproduces PyRanges' NCLS
join order. Imputation (gate 10) relies on the BLAS `sgemm` kernel; both stacks used
OpenBLAS 0.3.34 on the same CPU.

Timing (same node; pycisTopic baseline = run A of the reference contract):

| stage | pycisTopic | Epione |
|---|---|---|
| fragment counts (`create_cistopic_object_from_fragments`) | 161.5 s | 53–68 s |
| LDA, 4 models × 500 sweeps, 4 cores | 10,061 s | 3,086 s |
| LDA, one sweep, 1 core, K = 10 / 20 / 40 | 5.2 / 12.1 / 21.0 s | 2.0 / 2.4 / 4.0 s |
| LDA setup (token lists + initial assignment), 1 core | 191 s | 1.9 s |
| binarize (ntop + otsu + li) | 8.4 s | 1.7–2.3 s |
| impute accessibility | 6.7 s | 5.9–8.6 s |
| normalize + HVF | 14.1 s | 11 s (`compat="native"`), 25.6 s (SVML `log1p` emulation) |
| DARs, 16 contrasts, 4 cores | 120.3 s | 19–20 s |

The node was shared with other jobs (load average ≈ 170 on 256 cores), so wall times
vary by ±30 %. The 4-model LDA wall time was measured with an earlier build of the
sampler; the single-core sweep timings use the final build and show a further ~25 %
gain at K = 20–40.
