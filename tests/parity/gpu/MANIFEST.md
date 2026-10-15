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

## Results

(appended after the run)
