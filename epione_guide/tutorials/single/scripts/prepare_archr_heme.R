# Build the ArchR Hematopoiesis dataset used by the ArchR-parity tutorials.
#
# Usage:
#   Rscript prepare_archr_heme.R <out_dir> [threads=8] [path_to_macs2=macs2]
#
# Requires R >= 4.1 with ArchR (>= 1.0.3), BSgenome.Hsapiens.UCSC.hg19,
# JASPAR2020, TFBSTools, data.table, and a MACS2/MACS3 executable.
# Point the tutorials at the result with `export EPIONE_ARCHR_HEME=<out_dir>`.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript prepare_archr_heme.R <out_dir> [threads] [path_to_macs2]")
OUT <- normalizePath(args[1], mustWork = FALSE)
THREADS <- if (length(args) >= 2) as.integer(args[2]) else 8L
MACS <- if (length(args) >= 3) args[3] else "macs2"

suppressPackageStartupMessages({
  library(ArchR); library(Matrix); library(SummarizedExperiment); library(S4Vectors)
  library(BSgenome.Hsapiens.UCSC.hg19); library(data.table)
  library(JASPAR2020); library(TFBSTools)
})
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
setwd(OUT)

# Re-seed before every stage so each stage is reproducible on its own.
stage <- function(name) {
  cat("\n=====", name, "=====\n")
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(1)
  addArchRThreads(threads = THREADS)
  addArchRGenome("hg19")
}
PROJ <- file.path(OUT, "ArchR_proj")
done <- function(tag) file.exists(file.path(OUT, ".stages", tag))
mark <- function(tag) { dir.create(file.path(OUT, ".stages"), showWarnings = FALSE); file.create(file.path(OUT, ".stages", tag)) }

# 1. Download fragments, create Arrow files, iterative LSI.
if (!done("lsi")) {
  stage("download + arrows + LSI")
  inputFiles <- getTutorialData("Hematopoiesis")
  dir.create("ArrowFiles", showWarnings = FALSE)
  setwd("ArrowFiles")
  arrows <- createArrowFiles(file.path(OUT, inputFiles), names(inputFiles),
                             minTSS = 4, minFrags = 1000,
                             addTileMat = TRUE, addGeneScoreMat = FALSE, force = TRUE)
  setwd(OUT)
  proj <- ArchRProject(file.path(OUT, "ArrowFiles", basename(arrows)),
                       outputDirectory = PROJ, copyArrows = FALSE)
  proj <- addIterativeLSI(proj, useMatrix = "TileMatrix", name = "IterativeLSI",
                          iterations = 2, varFeatures = 25000, dimsToUse = 1:30,
                          LSIMethod = 2, seed = 1, force = TRUE)
  saveArchRProject(proj, load = FALSE)
  mark("lsi")
}

# 2. Gene scores, clusters, reproducible peak set, peak matrix.
if (!done("peaks")) {
  stage("gene scores + clusters + peaks")
  proj <- loadArchRProject(PROJ, showLogo = FALSE)
  proj <- addGeneScoreMatrix(proj, force = TRUE)
  proj <- addClusters(proj, reducedDims = "IterativeLSI", resolution = 0.4,
                      force = TRUE, knnAssign = 10, maxClusters = 25)
  proj <- addGroupCoverages(proj, groupBy = "Clusters", force = TRUE)
  proj <- addReproduciblePeakSet(proj, groupBy = "Clusters", pathToMacs2 = MACS,
                                 force = TRUE, threads = THREADS)
  proj <- addPeakMatrix(proj, force = TRUE)
  saveArchRProject(proj, load = FALSE)
  mark("peaks")
}

# 3. JASPAR2020 motif annotations, background peaks, chromVAR deviations.
if (!done("motifs")) {
  stage("motifs + chromVAR")
  proj <- loadArchRProject(PROJ, showLogo = FALSE)
  pfm <- TFBSTools::getMatrixSet(JASPAR2020::JASPAR2020,
    opts = list(collection = "CORE", tax_group = "vertebrates", all_versions = FALSE))
  proj <- addMotifAnnotations(proj, motifPWMs = TFBSTools::toPWM(pfm), force = TRUE, name = "Motif")
  proj <- addBgdPeaks(proj, method = "chromVAR", force = TRUE)
  proj <- addDeviationsMatrix(proj, peakAnnotation = "Motif", force = TRUE)
  saveArchRProject(proj, load = FALSE)
  mark("motifs")
}

# 4. UMAP.
if (!done("umap")) {
  stage("UMAP")
  proj <- loadArchRProject(PROJ, showLogo = FALSE)
  proj <- addUMAP(proj, reducedDims = "IterativeLSI", name = "UMAP",
                  nNeighbors = 30, minDist = 0.5, metric = "cosine", seed = 1, force = TRUE)
  saveArchRProject(proj, load = FALSE)
  mark("umap")
}

# 5. Export plain tables for the Python side.
stage("export tables")
proj <- loadArchRProject(PROJ, showLogo = FALSE)
dir.create("tables", showWarnings = FALSE)

cd <- as.data.frame(getCellColData(proj))
umap <- getEmbedding(proj, "UMAP")
cd$UMAP1 <- umap[rownames(cd), 1]; cd$UMAP2 <- umap[rownames(cd), 2]
fwrite(cbind(barcode = rownames(cd), cd), "tables/cells.tsv", sep = "\t")

lsi <- getReducedDims(proj, "IterativeLSI")
fwrite(cbind(barcode = rownames(lsi), as.data.frame(lsi)), "tables/lsi.tsv", sep = "\t")

ps <- getPeakSet(proj)
ps$cluster_name <- names(ps); names(ps) <- NULL
pdf <- as.data.frame(ps); pdf$peak <- paste0(pdf$seqnames, ":", pdf$start, "-", pdf$end)
fwrite(pdf, "tables/peaks.tsv", sep = "\t")

dump_mat <- function(name) {
  se <- getMatrixFromProject(proj, useMatrix = name)
  writeMM(as(assay(se), "CsparseMatrix"), file.path("tables", paste0(name, ".mtx")))
  writeLines(colnames(se), file.path("tables", paste0(name, "_cells.txt")))
  fwrite(as.data.frame(rowData(se)), file.path("tables", paste0(name, "_features.tsv")), sep = "\t")
}
dump_mat("PeakMatrix")
dump_mat("GeneScoreMatrix")

se <- getMatrixFromProject(proj, useMatrix = "MotifMatrix")
for (an in c("z", "deviations")) {
  fwrite(cbind(motif = rownames(se), as.data.frame(as.matrix(assay(se, an)))),
         paste0("tables/MotifMatrix_", an, ".tsv.gz"), sep = "\t")
}
fwrite(as.data.frame(rowData(se)), "tables/MotifMatrix_features.tsv", sep = "\t")

ma <- readRDS(getPeakAnnotation(proj, "Motif")$Matches)
writeMM(as(assay(ma), "CsparseMatrix"), "tables/motif_matches.mtx")
writeLines(colnames(ma), "tables/motif_matches_motifs.txt")

fwrite(as.data.frame(assay(getBgdPeaks(proj))), "tables/bgd_peaks.tsv.gz", sep = "\t")

ga <- getGeneAnnotation(proj)
fwrite(as.data.frame(ga$genes), "tables/genes_hg19.tsv", sep = "\t")
fwrite(as.data.frame(ga$exons), "tables/exons_hg19.tsv", sep = "\t")
fwrite(as.data.frame(ga$TSS), "tables/tss_hg19.tsv", sep = "\t")

print(sessionInfo())
cat("---DONE---\n")
