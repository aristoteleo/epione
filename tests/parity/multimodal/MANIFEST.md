# Multimodal parity and benchmark, pre-registered

Written before any Epione run of `combined_dims`, `wnn`, `multivi` or `tsne`.
References are produced outside the repo under `EPIONE_MULTIMODAL_PARITY_DIR`
(R for ArchR / Seurat runs there; no R in this repository).

Reference software: ArchR 1.0.3, Seurat 5.4.0 / SeuratObject 5.4.0 (R 4.4.3),
muon 0.1.9, scvi-tools 1.4.2, openTSNE 1.0.2.

## C · `epi.tl.combined_dims` vs ArchR `addCombinedDims`

Inputs: the reduced dims stored in the ArchR hematopoiesis fixture project
(`IterativeLSI` and `Harmony`, with ArchR's stored `corToDepth` and
`scaleDims`), exported to text. Configurations:

| id | reducedDims | dimWeights | dimsToUse | scaleDims | corCutOff |
|---|---|---|---|---|---|
| C1 | IterativeLSI, Harmony | default | default | default (stored) | 0.75 |
| C2 | IterativeLSI, Harmony | 1, 0.5 | default | default | 0.75 |
| C3 | IterativeLSI, Harmony | default | 1:20 | TRUE | 0.75 |
| C4 | IterativeLSI, Harmony | default | default | FALSE | 0.5 |

**Gate C:** for every configuration, the same shape and kept dims, and
`max |Epione − ArchR| ≤ 1e-10` (float64).

## W · `epi.tl.wnn` vs Seurat `FindMultiModalNeighbors`

`epi.tl.wnn` is reimplemented from Hao *et al.*, Cell 2021 (STAR Methods),
without Seurat code. Both sides get **identical** embeddings computed once in
Python on 10x Multiome PBMC 10k (`epi.datasets.pbmc10k_multiome`, 9,631 cells):

- RNA: library-size normalise to 1e4, log1p, 2,000 HVGs (`seurat_v3` on counts), scale, PCA 50; dims 1:50 used;
- ATAC: TF-IDF + truncated SVD 50 (`epi`'s LSI); dims 2:50 used.

Seurat: `FindMultiModalNeighbors(reduction.list = list("pca","lsi"),
dims.list = list(1:50, 2:50))` with default arguments (k.nn = 20).
muon: `mu.pp.neighbors` on the same embeddings, reported for information.

Gates (Epione vs Seurat):

| id | quantity | pass if |
|---|---|---|
| W1 | per-cell RNA modality weight | Pearson r ≥ 0.95 **and** median \|Δ\| ≤ 0.05 |
| W2 | 20-NN neighbour sets of the WNN graph | median per-cell Jaccard ≥ 0.70 |
| W3 | Leiden (scanpy, resolution 1.0, seed 0) on each WNN graph | ARI(Epione partition, Seurat-graph partition) ≥ 0.80 |
| W4 | cell-type ARI of those partitions vs `obs["cell_type"]` | \|ΔARI\| ≤ 0.05 |

Each gate is recorded pass/fail. A failure is reported, not tuned away; any
change made after seeing results is an amendment recorded below, with its
rationale, before its run.

## B · benchmark (no gate)

Same data. Joint representations compared: RNA-only (PCA), ATAC-only (LSI 2:50),
`joint_embedding`-style CCA, `combined_dims` (PCA + LSI), WNN, MultiVI
(default architecture, GPU, seed 0; seeds 1–2 if time allows).
Metrics on Leiden clusters of each representation's kNN graph vs
`obs["cell_type"]`: ARI, NMI; plus cell-type silhouette (on embeddings; for
WNN, which has no embedding, on its UMAP) and 20-NN cell-type purity. All
results are reported; no method is excluded after seeing results.

## T · `epi.tl.tsne` and `epi.io.read_10x_multiome`

- `tsne`: functional tests only (openTSNE is the reference implementation).
- `read_10x_multiome`: on the public cellranger-arc
  `pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5`, RNA and ATAC
  counts of the cells shared with the scGLUE h5ad must be **identical**
  (gene names / peak coordinates matched).

### WNN implementation choices, fixed before the first run

The paper leaves these details open. They are chosen a priori and are not
tuned against Seurat afterwards (any later change is an amendment):

- each modality's embedding is L2-normalised per cell; distances are Euclidean;
- modality kNN graphs use k = 20 and include the cell itself;
- within- and cross-modality predictions average the embeddings of those 20 neighbours;
- the kernel offset is the distance to the nearest neighbour other than the cell itself;
- bandwidth σ: among the 200 nearest cells, take the 20 with the highest
  Jaccard similarity of 20-NN sets, and use their mean distance;
- ε = 1e-4 in the within/cross affinity ratio; weights are a softmax over modalities;
- WNN candidates are the union of each modality's 200 nearest cells; the final
  graph keeps the 20 with the highest weighted affinity, the cell itself included;
- W2 compares neighbour sets with the cell itself removed on both sides.

## Results

### C · combined_dims: PASS (4/4)

| config | shape (Epione = ArchR) | max \|Δ\| |
|---|---|---|
| C1 | 10,660 × 60 | 7.1e-15 |
| C2 | 10,660 × 60 | 6.9e-15 |
| C3 | 10,660 × 40 | 8.9e-15 |
| C4 | 10,660 × 59 (1 depth-correlated dim filtered) | 7.1e-15 |

### W · WNN, first run (implementation choices as registered): FAIL (2 of 4 gates)

| gate | Epione vs Seurat | result |
|---|---|---|
| W1 | weight r = 0.761, median \|Δ\| = 0.088 | fail |
| W2 | median Jaccard = 0.407 | fail |
| W3 | partition ARI = 0.874 | pass |
| W4 | cell-type ARI 0.773 (Epione) vs 0.730 (Seurat) | pass |

For information, Epione vs muon: weight r = 0.41, Jaccard 0.31, partition ARI
0.74; muon vs Seurat was not gated. Mean RNA weight: Epione 0.550, Seurat
0.506, muon 0.467.

## Amendment A1 (registered after the first W run, before running it)

Re-reading the STAR Methods of Hao *et al.* showed that the registered
bandwidth choice contradicts the paper. The paper says, verbatim: *"For each
cell i, we therefore aim to identify the 20 cells in the dataset with the
lowest non-zero Jaccard similarity. … If more than 20 cells share the same
Jaccard value, we select the 20 with the furthest euclidean distance to cell
i. We take the average of the Euclidean distances from cell i to the 20
selected cells, and set this as the cell-specific kernel bandwidth."* The
registered choice took the 20 cells with the **highest** Jaccard among the 200
nearest.

A1 changes only σ, to follow the paper: the Jaccard similarity of 20-NN sets
is computed for every cell sharing at least one neighbour with cell i (the
cell itself excluded); σ is the mean distance to the 20 with the lowest
non-zero value, ties broken by the larger Euclidean distance. All other
choices and all gates are unchanged. If A1 still fails, the result is
reported as is and `epi.tl.wnn` keeps the paper's definition.

### W · WNN, amendment A1 (paper's bandwidth): FAIL (3 of 4 gates), reported as is

| gate | Epione vs Seurat | first run | A1 | result |
|---|---|---|---|---|
| W1 | weight Pearson r / median \|Δ\| | 0.761 / 0.088 | **0.939** / **0.011** | fail (r < 0.95; median \|Δ\| passes) |
| W2 | median 20-NN Jaccard | 0.407 | **0.900** | pass |
| W3 | partition ARI, Leiden seed 0 | 0.874 | 0.7997 | fail (by 0.0003) |
| W4 | cell-type ARI, Epione vs Seurat | 0.773 vs 0.730 | 0.905 vs 0.730 | fail (Epione higher by 0.175) |

Mean RNA weight: Epione 0.508, Seurat 0.506. As registered, A1 is the last
amendment: `epi.tl.wnn` keeps the paper's bandwidth and the failures stand.

**Post-hoc, not a gate:** W3 and W4 use a single Leiden seed. Over 10 seeds on the same two graphs:

| quantity | mean ± SD | range |
|---|---|---|
| partition ARI, Epione vs Seurat | 0.843 ± 0.061 | 0.775–0.982 |
| partition ARI, Seurat vs Seurat re-seeded | 0.868 ± 0.056 | 0.773–0.954 |
| cell-type ARI, Epione | 0.830 ± 0.078 | 0.748–0.906 |
| cell-type ARI, Seurat | 0.769 ± 0.057 | 0.730–0.920 |

The Epione–Seurat partition agreement is close to Seurat's own
seed-to-seed agreement, so W3/W4 were too tight for a single Leiden seed.
The remaining W1 gap (r = 0.94) means the per-cell weights still differ in
detail from Seurat's implementation, whose code was deliberately not consulted.

### T · reader, first run

- RNA: **identical** counts on all 29,095 genes (matched by Ensembl id) for the
  9,627 cells shared with scGLUE. Four scGLUE cells are not among the
  11,898 cellranger-arc 2.0.0 filtered barcodes.
- ATAC: **not comparable**. scGLUE's 107,194 peaks and cellranger-arc 2.0.0's
  143,887 share no exact coordinates; the peak sets were called differently.
  This is a flaw in the registered design, not a reader result.
- The check script also required all 9,631 scGLUE cells to be present, which
  is stricter than the registered gate; that condition is dropped.

## Amendment T-A1 (registered before running it)

The ATAC part of gate T is replaced by:
- **T-A1a (gate):** decode the same h5 independently with `h5py`
  (`matrix/data`, `indices`, `indptr`, `features/feature_type`, `features/id`).
  The reader's ATAC matrix and peak names must equal that decode exactly,
  and so must its RNA matrix.
- **T-A1b (information):** fraction of cells whose ATAC row sum equals
  `atac_peak_region_cutsites` and whose RNA row sum equals `gex_umis_count`
  in `per_barcode_metrics.csv`.
