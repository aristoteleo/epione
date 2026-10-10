# ArchR Hematopoiesis tutorial dataset

The ArchR-parity tutorials (Harmony, MAGIC imputation, trajectories, marker
heatmaps, browser tracks, peak-annotation enrichment, TF regulators, module
scores) all run on the same data: the scATAC-seq hematopoiesis dataset that
ArchR uses in its own tutorial (Granja *et al.*, *Nat Biotechnol* 2019; three
samples: BMMC, CD34+ BMMC and PBMC, about 10,600 cells after QC).

Each tutorial compares Epione with ArchR, so the dataset is prepared once with
ArchR itself. The script downloads the fragment files with
`ArchR::getTutorialData("Hematopoiesis")` and runs the standard ArchR steps.

## Build it

Requirements: R >= 4.1 with ArchR (>= 1.0.3), `BSgenome.Hsapiens.UCSC.hg19`,
`JASPAR2020`, `TFBSTools` and `data.table`, plus a MACS2 or MACS3 executable.

```bash
export EPIONE_ARCHR_HEME=$HOME/epione_data/archr_heme
Rscript epione_guide/tutorials/single/scripts/prepare_archr_heme.R \
    $EPIONE_ARCHR_HEME 8 $(which macs2)
```

The run takes about an hour with 8 threads and needs about 15 GB of disk. It is
resumable: finished stages are recorded in `$EPIONE_ARCHR_HEME/.stages/`.

| Stage | ArchR calls |
|---|---|
| 1 | `getTutorialData`, `createArrowFiles(minTSS = 4, minFrags = 1000)`, `addIterativeLSI(iterations = 2, varFeatures = 25000, dimsToUse = 1:30)` |
| 2 | `addGeneScoreMatrix`, `addClusters(resolution = 0.4)`, `addGroupCoverages`, `addReproduciblePeakSet`, `addPeakMatrix` |
| 3 | JASPAR2020 CORE vertebrates `addMotifAnnotations`, `addBgdPeaks`, `addDeviationsMatrix` |
| 4 | `addUMAP(nNeighbors = 30, minDist = 0.5, metric = "cosine")` |
| 5 | export of plain tables for Python |

Every stage re-seeds with `set.seed(1)` before running.

## Layout

```
$EPIONE_ARCHR_HEME/
├── HemeFragments/        fragment files (+ .tbi)
├── ArrowFiles/
├── ArchR_proj/           the ArchR project
├── tables/               plain-text / Matrix Market exports read by the tutorials
│   ├── cells.tsv         cell metadata incl. Sample, Clusters, UMAP1/2
│   ├── lsi.tsv           IterativeLSI embedding
│   ├── peaks.tsv         peak set
│   ├── PeakMatrix.mtx, GeneScoreMatrix.mtx  (+ _cells.txt, _features.tsv)
│   ├── MotifMatrix_z.tsv.gz, MotifMatrix_deviations.tsv.gz, MotifMatrix_features.tsv
│   ├── motif_matches.mtx, motif_matches_motifs.txt, bgd_peaks.tsv.gz
│   └── genes_hg19.tsv, exons_hg19.tsv, tss_hg19.tsv
├── archr_ref/<name>/     ArchR reference outputs for each tutorial's parity section
└── annotations/          downloaded annotation databases
```

## ArchR reference outputs

The "Parity vs ArchR" section of each tutorial reads ArchR's own results from
`archr_ref/<name>/`. Generate them with the reference script that ships with
the tutorial's parity test, for example:

```bash
Rscript tests/parity/harmony/ref_harmony.R $EPIONE_ARCHR_HEME
```

Each tutorial names its script at the start of the parity section. The parity
tests in `tests/` read the same variable and are skipped when it is not set.
