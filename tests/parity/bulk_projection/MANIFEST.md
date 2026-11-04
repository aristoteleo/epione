# Bulk ATAC projection: parity with ArchR `projectBulkATAC` (pre-registered)

Written before any Epione run on the parity data. The reference side lives outside the repository, under `EPIONE_BULKPROJ_PARITY_DIR`. It holds the ArchR 1.0.3 outputs, the exported LSI model and the bulk matrix; `parity_check.py` reads it.

## Setting

| | |
|---|---|
| single cells | ArchR's hematopoiesis tutorial project: 10,660 cells, IterativeLSI on 500 bp tiles, 25,000 features, 30 dims, LSIMethod 2, scaleTo 10,000; UMAP with nNeighbors 30, minDist 0.5, cosine; 11 clusters |
| bulk | Corces et al. 2016 (GEO GSE74912, `GSE74912_ATACseq_All_Counts.txt.gz`, sha256 `31a7834d…183b791f`), 590,650 peaks × 132 samples, hg19. The **77 normal-donor samples** are used (Donor\*, 4983-\*, 6792-\*); the AML, cord-blood and BM-CD34 samples are left out |
| ArchR call | `set.seed(1); projectBulkATAC(proj, seATAC, reducedDims = "IterativeLSI", embedding = "UMAP", n = 250, threads = 1)` |
| Epione call | `epi.tl.project_bulk_atac(adata, bulk, model=LSIModel(flavor="archr", …), n_cells=250, seed=1, rng=…)`. The model holds ArchR's `svd$u`, `svd$d`, `rowSm`, `nCol` and `LSIFeatures` (closed intervals, `end = start + 499`) |

The bulk peaks enter ArchR as a GRanges built from the table's `Start`/`End` columns, and Epione receives the same numbers with `closed=True`.

## Gates

**G0 · feature mapping.** `bulk_to_features` must equal ArchR's `bulkMat` (the matrix after `subsetByOverlaps`, the count ordering and `!duplicated(subjectHits)`) exactly, for every one of the 77 samples, and the overlap ratio must be identical.

**G1 · LSI projection on identical pseudo-cells.** The reference script applies ArchR's sampling code to every sample and exports the simulated matrices with `.projectLSI`'s output. Epione projects those same matrices:
- raw coordinates: `max|Δ| ≤ 1e-8 × max|coord|`;
- row-scaled coordinates (`.scaleDims`): `max|Δ| ≤ 1e-8`.

**G2 · end-to-end bit-exact sampling.** With `rng="archr", seed=1`, Epione's `obsm['X_lsi']` must equal ArchR's `simulatedReducedDims` from the real `projectBulkATAC` call, to the G1 scaled tolerance, for all 77 × 250 pseudo-cells. This holds only if the sampling, including ArchR's re-seeding inside `.projectLSI`, is reproduced exactly.

**G3 · independent sampling** (`rng="numpy"`, seed 1, against the ArchR run above). Each pseudo-cell gets the majority cluster of its 10 nearest single cells in ArchR's scaled LSI.
- G3a: per-sample majority cluster identical to ArchR's in ≥ 95% of samples.
- G3b: mean, over samples, of the total-variation distance between the two per-sample cluster-fraction vectors ≤ 0.10.
- G3c: let `d(s) = ‖centroid_Epione(s) − centroid_ArchR(s)‖` in scaled LSI, and `spread(s)` the mean distance of ArchR's pseudo-cells to their centroid. Require median of `d/spread` ≤ 0.25 and maximum ≤ 0.5.

**G4 · UMAP projection.** Epione cannot load the `uwot` model. It fits umap-learn (0.5.x) on ArchR's scaled LSI of the 10,660 cells, initialised from ArchR's UMAP coordinates (n_neighbors 30, min_dist 0.5, cosine, seed 1), and transforms the G2 pseudo-cells. Each pseudo-cell gets the majority cluster of its 10 nearest single cells in the respective layout: Epione's refitted single-cell coordinates, and ArchR's UMAP for ArchR's projection.
- G4a: per-pseudo-cell agreement of that cluster ≥ 0.85.
- G4b: per-sample majority agreement ≥ 95% of samples.
- G4c: Epione's re-projection correlation (5,000 cells, both axes) ≥ 0.8, ArchR's own warning threshold. ArchR's value is reported.

Coordinates are not compared, since uwot and umap-learn produce different layouts.

**G5 · deconvolution, synthetic ground truth.** This uses Epione's own heme build (`heme_peaks.h5ad`: 10,705 cells, 232,272 peaks, clusters C1–C11).
- Cells are split 50/50 within each cluster, with seed 0, into reference and test halves.
- 50 synthetic bulks are made by summing the counts of 2,000 test cells drawn with Dirichlet(α = 1) cluster proportions, seed 0.
- `deconvolve_bulk(reference, bulk, "Clusters", weight="cells")` is compared with the true cell proportions.
- Gates: pooled Pearson ≥ 0.90 and mean absolute error ≤ 0.05.

The Corces sorted populations are reported (dominant group per population) but not gated, because the cluster identities are tentative.

All results are recorded below, including failures.

## Results

Run on 2026-10-13. Reference: ArchR 1.0.3, uwot 0.2.4, R version 4.4.3 (2025-02-28), one thread. Epione: Python 3.11, numpy 2.4, umap-learn 0.5.12. Both on one core of the same node. **All gates pass.**

| Gate | Result | Value |
|---|---|---|
| G0 feature mapping | pass | 0 differing entries over 25,000 features × 77 samples; overlap ratio 0.98932 on both sides |
| G1 LSI projection, identical pseudo-cells | pass | raw max relative Δ 8.7e-15, scaled max Δ 1.8e-14 |
| G2 end-to-end, `rng="archr"` | pass | same 19,250 pseudo-cell names, max Δ 1.8e-14, so every draw is reproduced bit-exactly |
| G3a majority cluster per sample, `rng="numpy"` | pass | 1.00 (gate ≥ 0.95) |
| G3b mean TVD of cluster fractions | pass | 0.0065 (gate ≤ 0.10) |
| G3c centroid shift / spread | pass | median 0.093, max 0.135 (gates ≤ 0.25 / 0.5) |
| G4a UMAP nearest cluster, per pseudo-cell | pass | 0.985 (gate ≥ 0.85) |
| G4b UMAP nearest cluster, per-sample majority | pass | 1.00 (gate ≥ 0.95) |
| G4c re-projection correlation | pass | Epione 0.9986 / 0.9984; ArchR 0.9984 / 0.9980 |
| G5 deconvolution, 50 synthetic bulks | pass | Pearson 0.958 (≥ 0.90), MAE 0.0169 (≤ 0.05), median R² 0.838 |

Timings for 77 samples × 250 pseudo-cells, one core:

| | seconds |
|---|---|
| ArchR `projectBulkATAC` (with UMAP) | 76 |
| Epione, `rng="archr"`, with the umap-learn fit and transform | 80 |
| Epione, `rng="numpy"`, LSI only | 27 |
| feature mapping | 1.0 |
| deconvolution of 50 bulks | 4.3 |

Dominant ArchR cluster (LSI nearest neighbours, ArchR run) per sorted population. This is reported, not gated; cluster names are those of the ArchR fixture, not of Epione's heme build: B → C6, CD4T → C8, CD8T → C8, CLP → C3, CMP → C9, Ery → C9, GMP → C11, HSC → C11, LMPP → C11, MEP → C9, MPP → C11, Mono → C5, NK → C7.

Notes:
- ArchR's `.projectLSI` calls `set.seed(LSI$seed)`. In a one-thread run, every bulk sample after the first is therefore sampled from the same RNG state. `rng="archr"` reproduces this.
- uwot and umap-learn layouts differ, so G4 compares cluster assignments rather than coordinates.
