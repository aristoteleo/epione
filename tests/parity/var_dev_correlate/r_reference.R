# ArchR 1.0.3 reference for epione.tl.var_deviations / correlate_matrices /
# positive_tf_regulators.
# Usage: Rscript r_reference.R <fixture ArchR_base dir> <work dir> [threads]
suppressPackageStartupMessages({
  library(ArchR); library(Matrix); library(SummarizedExperiment); library(data.table)
  library(BSgenome.Hsapiens.UCSC.hg19); library(magrittr)
})
args <- commandArgs(trailingOnly = TRUE)
BASE <- args[1]; WD <- args[2]
THREADS <- if (length(args) >= 3) as.integer(args[3]) else 4L
set.seed(1); addArchRThreads(threads = THREADS); addArchRGenome("hg19")
dir.create(file.path(WD, "ref"), recursive = TRUE, showWarnings = FALSE)
setwd(WD)
OUT <- function(f) file.path(WD, "ref", f)

# ---- project copy (+ cisbp deviations, the ArchR-tutorial motif set) -------
PROJ <- file.path(WD, "ArchR_proj")
if (file.exists(file.path(PROJ, "Save-ArchR-Project.rds"))) {
  proj <- loadArchRProject(PROJ, showLogo = FALSE)
} else {
  proj <- saveArchRProject(loadArchRProject(BASE, showLogo = FALSE),
                           outputDirectory = PROJ, load = TRUE)
}
if (!"cisbpMatrix" %in% getAvailableMatrices(proj)) {
  proj <- addMotifAnnotations(proj, motifSet = "cisbp", annoName = "cisbp", force = TRUE)
  proj <- addDeviationsMatrix(proj, peakAnnotation = "cisbp", matrixName = "cisbpMatrix",
                              force = TRUE)
  proj <- saveArchRProject(proj, outputDirectory = PROJ, load = TRUE)
}

# ---- inputs, exported once so Python sees exactly the same numbers ----------
cells <- getCellNames(proj)
cd <- as.data.frame(getCellColData(proj))
fwrite(data.table(barcode = cells, Sample = cd[cells, "Sample"], Clusters = cd[cells, "Clusters"]),
       OUT("cells.tsv"), sep = "\t")
rD <- getReducedDims(proj, reducedDims = "IterativeLSI", corCutOff = 0.75, dimsToUse = 1:30)
fwrite(data.table(barcode = rownames(rD), as.data.frame(rD)), OUT("rD.tsv"), sep = "\t")
fwrite(data.table(dim = seq_along(proj@reducedDims$IterativeLSI$corToDepth$none),
                  none = proj@reducedDims$IterativeLSI$corToDepth$none,
                  scaled = proj@reducedDims$IterativeLSI$corToDepth$scaled,
                  scaleDims = proj@reducedDims$IterativeLSI$scaleDims),
       OUT("corToDepth.tsv"), sep = "\t")
fwrite(data.table(barcode = rownames(proj@reducedDims$IterativeLSI$matSVD),
                  as.data.frame(proj@reducedDims$IterativeLSI$matSVD)), OUT("lsi_raw.tsv"), sep = "\t")

ns <- sapply(getArrowFiles(proj), function(a) length(ArchR:::.availableCells(a, "cisbpMatrix")))
fwrite(data.table(sample = names(ns), n = ns), OUT("arrow_ncells.tsv"), sep = "\t")

dumpMotif <- function(mname, tag) {
  se <- getMatrixFromProject(proj, useMatrix = mname)
  z <- as.matrix(assay(se, "z"))[, cells]
  fwrite(data.table(motif = rownames(se), as.data.frame(z)), OUT(paste0(tag, "_z.tsv.gz")), sep = "\t")
}
dumpMotif("cisbpMatrix", "cisbp")
gs <- getMatrixFromProject(proj, useMatrix = "GeneScoreMatrix")
writeMM(as(assay(gs)[, cells], "CsparseMatrix"), OUT("genescore.mtx"))
fwrite(as.data.frame(rowData(gs)), OUT("genescore_features.tsv"), sep = "\t")

# ---- 1. getVarDeviations ---------------------------------------------------
for (m in c("MotifMatrix", "cisbpMatrix")) {
  v <- getVarDeviations(proj, name = m, plot = FALSE)
  fwrite(as.data.frame(v), OUT(paste0("vardev_", m, ".tsv")), sep = "\t")
}
p <- getVarDeviations(proj, name = "cisbpMatrix", plot = TRUE)
ggsave(OUT("plotVarDev_cisbp.png"), p, width = 5, height = 5, dpi = 150)

# ---- 2. correlateMatrices internals (seed cells, kNN groups) ---------------
set.seed(1)
idx <- sample(seq_len(nrow(rD)), 500, replace = !nrow(rD) >= 500)
writeLines(as.character(idx), OUT("knn_seed_idx.txt"))
knnObj <- ArchR:::.computeKNN(data = rD, query = rD[idx, ], k = 100)
keep <- ArchR:::determineOverlapCpp(knnObj, floor(0.8 * 100))
fwrite(as.data.table(knnObj), OUT("knn_all.tsv"), sep = "\t", col.names = FALSE)
writeLines(as.character(keep), OUT("knn_keep.txt"))

# ---- 3. correlateMatrices --------------------------------------------------
corGSM <- correlateMatrices(proj, useMatrix1 = "GeneScoreMatrix", useMatrix2 = "cisbpMatrix",
                            reducedDims = "IterativeLSI", seed = 1)
fwrite(as.data.frame(corGSM), OUT("correlateMatrices_GSM_cisbp.tsv"), sep = "\t")

# ---- 4. positive TF regulators (ArchR book, chapter 15.4) -------------------
seGroupMotif <- getGroupSE(ArchRProj = proj, useMatrix = "cisbpMatrix", groupBy = "Clusters")
seZ <- seGroupMotif[rowData(seGroupMotif)$seqnames == "z", ]
rowData(seZ)$maxDelta <- lapply(seq_len(ncol(seZ)), function(x) {
  matrixStats::rowMaxs(assay(seZ) - assay(seZ)[, x])
}) %>% Reduce("cbind", .) %>% matrixStats::rowMaxs()
fwrite(data.table(name = rowData(seZ)$name, maxDelta = rowData(seZ)$maxDelta,
                  as.data.frame(assay(seZ))), OUT("groupSE_cisbp_z.tsv"), sep = "\t")

corGSM$maxDelta <- rowData(seZ)[match(corGSM$cisbpMatrix_name, rowData(seZ)$name), "maxDelta"]
corGSM <- corGSM[order(abs(corGSM$cor), decreasing = TRUE), ]
corGSM <- corGSM[which(!duplicated(gsub("\\-.*", "", corGSM[, "cisbpMatrix_name"]))), ]
corGSM$TFRegulator <- "NO"
corGSM$TFRegulator[which(corGSM$cor > 0.5 & corGSM$padj < 0.01 &
                         corGSM$maxDelta > quantile(corGSM$maxDelta, 0.75))] <- "YES"
fwrite(as.data.frame(corGSM), OUT("positive_tf_regulators.tsv"), sep = "\t")
cat("YES:", sort(corGSM[corGSM$TFRegulator == "YES", 1]), "\n")

p <- ggplot(data.frame(corGSM), aes(cor, maxDelta, color = TFRegulator)) +
  geom_point() + theme_ArchR() + geom_vline(xintercept = 0, lty = "dashed") +
  scale_color_manual(values = c("NO" = "darkgrey", "YES" = "firebrick3")) +
  xlab("Correlation To Gene Score") + ylab("Max TF Motif Delta") +
  scale_y_continuous(expand = c(0, 0), limits = c(0, max(corGSM$maxDelta) * 1.05))
ggsave(OUT("positive_tf_regulators.png"), p, width = 5, height = 5, dpi = 150)
cat("---DONE---\n")
