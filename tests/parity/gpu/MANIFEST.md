# GPU acceleration (torch backend + optional rapids-singlecell backend): pre-registered parity gates

Written on 2026-10-15, before any CPU-versus-GPU comparison was run on any dataset.
The thresholds below are read-only. Results are appended in a separate section after the
run. A failed gate is reported as failed and is not re-tuned. An amendment (a change to a
gate, a dataset or a protocol) is dated, recorded in the "Amendments" section *before* the
rerun it affects, and reported together with the result it changed.

## What is compared

The reference is always Epione's own CPU implementation (`device="cpu"`, the default), on the
same inputs, in the same environment. The candidate is the same public function with
`device="cuda"` (torch) or `backend="rapids"` (rapids-singlecell). CPU results must not change
because of this work; gate G0 checks that.

| Function | GPU path | Class | Gates |
|---|---|---|---|
| (all of the below, `device="cpu"`) | unchanged | exact | G0 |
| `epi.tl.lsi` | TF-IDF + torch port of sklearn `randomized_svd` | deterministic linear algebra | G1, G2 |
| `epi.tl.iterative_lsi`, `iterations=1` | TF-logIDF + same SVD port | deterministic linear algebra | G3 |
| `epi.tl.iterative_lsi`, `iterations=2` | as above + exact GPU kNN inside the intermediate clustering | stochastic (graph -> Leiden -> feature selection) | G4 |
| `epi.pp.neighbors` (exact search) | brute-force torch kNN | deterministic search | G5 |
| `epi.tl.compute_deviations(method="analytical")` | torch peer statistics + matmul | deterministic linear algebra | G6 |
| `epi.tl.compute_deviations(method="sample")` | torch per-iteration null | deterministic linear algebra | G7 |
| `epi.tl.add_motif_matrix` (MOODS scan) | conv1d over one-hot sequence, MOODS log-odds matrices and thresholds | exact set | G8 |
| `epi.tl.peak_to_gene` | torch metacell aggregation + pair correlation | deterministic linear algebra | G9 |
| `epi.pp.neighbors(backend="rapids")` | rapids-singlecell (cuVS brute force) | deterministic search | G10 |
| `epi.tl.clusters(method="leiden", backend="rapids")` | rapids-singlecell (cuGraph Leiden) | stochastic graph | G11 |
| `epi.tl.umap(backend="rapids")` | rapids-singlecell (cuML UMAP) | stochastic embedding | G12 |
| speed | wall time CPU vs GPU | report | G13 |

Functions that the cost survey shows are dominated by something the GPU cannot speed up (for
example parsing a fragment file line by line) are not given a `device=` path; the survey and
the decision are reported with the results.

## Data

| ID | Dataset | Shape | Used for |
|---|---|---|---|
| D1 | ArchR hematopoiesis, Epione-built peak matrix (`heme_peaks.h5ad`, `epione_heme_data` contract): `X` int32 counts, `varm["motif"]` (746 JASPAR 2020 motifs), `varm["bg_peaks"]` (50 peers), `uns["peak_seq"]`, `obsm["X_iterative_lsi"]` | 10,705 cells x 232,272 peaks | G0-G2, G5-G13 |
| D1g | gene scores of the same cells (`heme_gene_scores.h5ad`) | 10,705 x 20,057 | G9 |
| D2 | same cells, 500 bp tile matrix (`cache/tiles_lsi.h5ad`) | 10,705 x 6,274,368 | G3, G4, G13 |
| D3 | GSE274113, all 13 readable replicate cCRE count matrices concatenated (EpiAgent hg38 cCRE vocabulary), restricted to the 250,000 cCREs accessible in the most cells | 128,509 x 250,000 | G13 (scaling) and G1/G2/G5 as a large-data replicate |

Reference outputs and timings are written outside the repository. `parity_check.py` reads them
from `$EPIONE_GPU_PARITY_DIR` and is skipped when the variable is unset.

## Protocol

- Same seeds on both sides. `tl.lsi` is called with `random_state=0` (it forwards to
  `randomized_svd`, whose own default draws an unseeded matrix). `iterative_lsi` with
  `seed=1`, `n_components=30`, `var_features=25000`, `resolution=2.0` (the D2 build
  parameters). chromVAR on the stored `varm["motif"]` / `varm["bg_peaks"]`.
- Embeddings are compared after the per-component sign is aligned (`|cos|`), because an SVD
  component is defined up to sign.
- The GPU is shared with another job; GPU timings are therefore upper bounds. CPU timings use a
  fixed number of cores, stated next to every number; BLAS / numba / OpenMP threads are set to
  that number.
- CPU seed-to-seed variability (used by the stochastic gates) is measured with seeds 0-4 of the
  CPU path; "CPU vs CPU" means all 10 pairs of those 5 seeds, and "GPU vs CPU" means the GPU run
  with seed s against the CPU run with the same seed s, s = 0-4.

## Gates

| # | Output | Metric | Threshold |
|---|---|---|---|
| G0 | `device="cpu"` outputs of every function in the table, before vs after this change (D1, small fixtures in CI) | equality | bit-identical |
| G1 | `tl.lsi` TF-IDF -> L1 -> log1p normalised matrix values (every stored value, D1) | max relative error | <= 1e-5 |
| G2 | `tl.lsi` `obsm["X_lsi"]` (D1; D3 replicate) | (a) per-component `|cos|` CPU vs GPU, minimum over components; (b) cosine of the largest principal angle between the two column spaces | (a) >= 0.999; (b) >= 0.999 |
| G3 | `iterative_lsi(iterations=1)` embedding (D2), and the kept-dims list | as G2 (a), (b); kept dims | (a), (b) >= 0.999; kept dims identical |
| G4 | `iterative_lsi(iterations=2)` final embedding (D2) | (a) Jaccard of the final variable-feature sets; (b) mean per-cell overlap of the 30 nearest neighbours (cosine) in the final embedding; each compared as GPU-vs-CPU (median over s = 0-4) against CPU-vs-CPU (median over the 10 seed pairs) | (a) and (b): GPU-vs-CPU >= CPU-vs-CPU - 0.02 |
| G5 | `pp.neighbors` exact kNN, `n_neighbors=30`, on `X_iterative_lsi`, metric `euclidean` and `cosine` (D1; D3 replicate euclidean) | (a) mean per-cell Jaccard of the neighbour sets; (b) max abs difference of the distances of neighbours present in both | (a) >= 0.99; (b) <= 1e-4 |
| G6 | `compute_deviations(method="analytical")` Z and raw deviations (D1) | Pearson r across cells per motif, minimum over motifs; max abs difference reported | r >= 0.9999 |
| G7 | `compute_deviations(method="sample")` Z and raw deviations (D1) | as G6 | r >= 0.9999 |
| G8 | `add_motif_matrix` peak x motif hit matrix, MOODS (pychromvar) vs GPU, all 232,272 peaks x 746 motifs (D1) | set identity of the (peak, motif) hits | identical (Jaccard = 1) |
| G9 | `peak_to_gene` table (D1 + D1g, `use_rep="X_iterative_lsi"`, defaults) | (a) same pair list; (b) max abs difference of `correlation`; (c) Jaccard of pairs with FDR < 0.05 | (a) identical; (b) <= 1e-4; (c) >= 0.999 |
| G10 | rapids `pp.neighbors` (brute force), as G5 (D1) | as G5 (a) | >= 0.99 |
| G11 | rapids Leiden on the CPU-built graph (D1, resolution 1.2) vs CPU igraph Leiden | ARI, GPU-vs-CPU (median over s = 0-4) against CPU-vs-CPU (median over seed pairs) | GPU-vs-CPU >= CPU-vs-CPU - 0.05 |
| G12 | rapids UMAP on the CPU-built graph (D1) | trustworthiness (k = 15, against `X_iterative_lsi`), GPU minus CPU (median over s = 0-4) | >= -0.02 |
| G13 | wall time of each function, CPU (16 cores) vs GPU (1 H100 shared, 6 cores), D1, D2, D3 | speed-up table | reported; no threshold |

If the rapids-singlecell backend cannot be run on this cluster (glibc 2.17; RAPIDS wheels need
glibc >= 2.28, so it runs only inside a container), G10-G12 are reported as "not run" and the
backend is covered by mocked dispatch tests only.

## Amendments

**A1 (2026-10-15, before any comparison was run).** Protocol details that the table left open,
fixed here before the first run:
- G11: CPU Leiden is `epi.tl.clusters(method="leiden", leiden_kwargs=dict(resolution=1.2,
  random_state=s, flavor="igraph", n_iterations=2, directed=False))` (the D1 build call);
  the rapids side passes the same `leiden_kwargs` except `n_iterations=-1`, which the
  dispatch maps to cuGraph's default 100 levels (igraph's 2 refinement passes have no cuGraph
  counterpart). The graph for G11/G12 is `epi.pp.neighbors(n_neighbors=30, metric="cosine",
  use_rep="X_iterative_lsi")` built once on the CPU and loaded by both sides.
- G5 (D3 replicate): both sides search the CPU run's `X_lsi`, so the comparison isolates the
  search.
- G1 compares the stored values of the two normalised CSR matrices after both are put in
  canonical (sorted-index) form; the sparsity patterns must be identical.

**A2 (2026-10-15, after the first G10 run, before its rerun).** G10 failed in the first run
(mean neighbour-set Jaccard 0.96667 = 29/30 in *every* row, distances of shared neighbours
equal to 5e-6). The cause is a layout difference, not the search: rapids-singlecell stores each
cell's self-match as an explicit zero entry in `.obsp["distances"]` (30 entries per row), while
`epi.pp.neighbors` stores the 29 neighbours without self. The rapids dispatch
(`epione.core._rapids.rapids_neighbors`) now drops stored diagonal entries so both backends
return the same layout. G10 is rerun on the fixed dispatch with the threshold unchanged; both
the failed first run and the rerun are reported. No other gate or threshold changes.

## Results (2026-10-15)

Environment: Python 3.11, torch 2.6.0+cu124, scikit-learn 1.9.1, scipy 1.17.1, numpy 2.4.6,
pychromvar 0.0.4 / MOODS 1.9.4.2, umap-learn 0.5.12. rapids side: RAPIDS 25.12 container
(`rapidsai/base:25.12-cuda12-py3.12`, glibc 2.39, cuML 25.12, CuPy 13.6) with
rapids-singlecell 0.18.1 (`rapids-singlecell-cu12` wheel). Reference for G0: `origin/main` at
b8f0364. Every gate was evaluated once on the final code, except G10 (see A2).

| # | Output | Threshold | Result | |
|---|---|---|---|---|
| G0 | `device="cpu"` outputs, main vs branch (28 output files: LSI, TF-IDF values, iterative LSI 1 and 2 rounds x 5 seeds, neighbours x 2 metrics, chromVAR x 2, motif scan, peak-to-gene, graph, Leiden x 5, UMAP x 5) | bit-identical | 28/28 bit-identical | PASS |
| G1 | `tl.lsi` normalised values | max rel err <= 1e-5 | 1.1e-14 (same sparsity pattern) | PASS |
| G2 | `tl.lsi` embedding, D1 | min \|cos\| and principal cos >= 0.999 | 1.000000 / 1.000000 | PASS |
| G2 | `tl.lsi` embedding, D3 (128,509 x 250,000, streamed) | same | 0.999999 / 1.000000 | PASS |
| G3 | `iterative_lsi(iterations=1)`, D2 | >= 0.999; kept dims identical | 1.000000 / 1.000000; identical | PASS |
| G4a | `iterative_lsi(iterations=2)` variable features, D2 | GPU-vs-CPU >= CPU-vs-CPU - 0.02 | Jaccard 0.9309 vs 0.9131 | PASS |
| G4b | `iterative_lsi(iterations=2)` 30-NN overlap, D2 | same | 0.6578 vs 0.0665 | PASS |
| G5 | `pp.neighbors` euclidean, D1 | Jaccard >= 0.99; \|dd\| <= 1e-4 | 1.00000 (min 1.0); 9.5e-7 | PASS |
| G5 | `pp.neighbors` cosine, D1 | same | 1.00000 (min 1.0); 3.4e-7 | PASS |
| G5 | `pp.neighbors` euclidean, D3 | same | 1.00000 (min 1.0); 4.8e-7 | PASS |
| G6 | chromVAR analytical, D1 (746 motifs) | min Pearson r >= 0.9999 | r(Z) and r(dev) 1.0000000 (7 d.p.); max \|dZ\| 3.8e-4; NaN masks identical | PASS |
| G7 | chromVAR sample, D1 | same | 1.0000000; max \|dZ\| 1.2e-3; NaN masks identical | PASS |
| G8 | motif hits, 232,272 peaks x 746 motifs | identical | 11,593,234 hits each, 0 differing entries | PASS |
| G9 | peak-to-gene (1,493,249 pairs) | identical pairs; \|dr\| <= 1e-4; FDR<0.05 Jaccard >= 0.999 | identical; 3.6e-6; 724,309 vs 724,311 pairs (Jaccard 0.999997) | PASS |
| G10 | rapids neighbours, first run | Jaccard >= 0.99 | 0.96667 in every row (stored self entry, A2) | FAIL |
| G10 | rapids neighbours after A2, euclidean / cosine | Jaccard >= 0.99 | 1.00000 / 1.00000 | PASS |
| G11 | rapids Leiden ARI | GPU-vs-CPU >= CPU-vs-CPU - 0.05 | 0.7362 vs 0.7547 (11 clusters each) | PASS |
| G12 | rapids UMAP trustworthiness (k=15) | rapids - CPU >= -0.02 | 0.7555 - 0.7607 = -0.0053 | PASS |

Observations (not gates):

- G4b: cell-level 30-NN overlap between two CPU seeds is only 0.07, because the seed also draws
  the 10,000 cells of the first round; cluster-level agreement between seeds is much higher
  (Leiden ARI 0.77 between CPU seeds 0 and 1; 0.85 between CPU and GPU at seed 1). The stored
  D2 embedding (built with seed 1 in an earlier session) overlaps the seed-1 CPU rerun by 0.60
  and the seed-1 GPU run by a similar 0.66.
- G5: the neighbour sets are identical, but the UMAP connectivities built from them differ
  (max 0.82; 7,332 rows euclidean, 10,702 rows cosine). The CPU path's sklearn search on
  float32 input returns a non-zero self-distance (up to 2.4e-7 euclidean, 4.2e-7 cosine) for
  10.6 % (euclidean) / 38.5 % (cosine) of the cells, which `umap.fuzzy_simplicial_set` takes as
  the nearest-neighbour distance rho; the GPU search sets the self-distance to exactly 0. This is
  a pre-existing property of the CPU path, left unchanged here (G0).

### G13: wall time

CPU: 16 cores of a shared 32-core allocation on an AMD EPYC 9754 node; BLAS / numba / OpenMP threads = 16;
another user's 16-thread aligner ran on the same node during the runs. GPU: one H100 80GB
**shared under MPS with a training job that kept it at ~100 % utilisation and ~49 GB**, 6 CPU
cores, `EPIONE_GPU_MEMORY_GB=16`. GPU times are therefore upper bounds. Seeded steps: median
over seeds 0-4 (the first GPU call of a process includes CUDA initialisation).

| Step | Data | CPU (16 cores) | GPU | Speed-up |
|---|---|---|---|---|
| `tl.lsi` | D1 10,705 x 232,272 | 41.8 s | 5.2 s | 8.1x |
| `tl.lsi` | D3 128,509 x 250,000, 1.73e9 nnz (streamed) | 1,270.5 s | 41.9 s | 30.3x |
| `iterative_lsi`, 1 round | D2 10,705 x 6.27 M tiles | 24.4 s | 4.3 s | 5.6x |
| `iterative_lsi`, 2 rounds | D2 | 35.2 s | 8.6 s | 4.1x |
| `pp.neighbors` k=30 euclidean / cosine (incl. CPU connectivities) | D1 | 1.2 / 2.5 s | 0.62 / 0.07 s | 1.9x / 34x |
| `pp.neighbors` k=30 euclidean | D3 | 15.2 s | 10.1 s | 1.5x |
| `compute_deviations` analytical | D1, 746 motifs | 157.6 s | 10.2 s | 15.5x |
| `compute_deviations` sample (50 iterations) | D1 | 388.0 s | 20.3 s | 19.1x |
| `add_motif_matrix` (fresh scan; CPU = pychromvar's single-threaded MOODS loop) | D1, 746 motifs | 221.2 s | 27.3 s | 8.1x |
| `peak_to_gene` | D1 + gene scores | 10.5 s | 4.1 s | 2.6x |
| `tl.umap`, `backend="rapids"` | D1 graph | 14.5 s | 0.28 s | 52x |
| `tl.clusters(leiden)`, `backend="rapids"` | D1 graph | 0.41 s | 1.23 s | 0.3x (igraph is faster) |

The tutorial `t_gpu_acceleration.ipynb` repeats the D1/D2 steps with the CPU at 6 threads on
the GPU node, an Intel Xeon Platinum 8462Y+ (LSI 19 s vs 2 s, chromVAR 236 s vs 10 s, ...).

Cost survey for functions that got no `device=` path: `add_gene_score_matrix` on D1 fragments
(39 M fragments) takes 142 s, of which 137 s is the Python loop that parses the fragment file
and 0.6 s the sparse `W @ matGS` product; footprint aggregation (`get_footprints`) is likewise
bound by reading fragments. A GPU would not change either.
