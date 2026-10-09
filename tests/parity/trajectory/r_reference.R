# ArchR 1.0.3 reference outputs for the epione trajectory port.
# Usage: Rscript r_reference.R <ArchR project dir> <output dir>
suppressPackageStartupMessages({
  library(ArchR); library(data.table); library(SummarizedExperiment)
  library(JASPAR2020); library(TFBSTools)
})
args <- commandArgs(trailingOnly = TRUE)
projDir <- args[1]; out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
setwd(out)
set.seed(1); addArchRThreads(threads = 1); addArchRGenome("hg19")

TRAJ <- c("C11", "C10", "C5")
.getQuantiles <- ArchR:::.getQuantiles

writeDbl <- function(m, file) {
  con <- file(file, "wb"); writeBin(as.vector(as.double(m)), con, endian = "little"); close(con)
}
dumpMat <- function(m, prefix) {
  m <- as.matrix(m)
  writeDbl(m, paste0(prefix, ".f64"))
  writeLines(as.character(dim(m)), paste0(prefix, ".dim"))
  if (!is.null(rownames(m))) writeLines(rownames(m), paste0(prefix, ".rows"))
  if (!is.null(colnames(m))) writeLines(colnames(m), paste0(prefix, ".cols"))
}

proj <- loadArchRProject(projDir, showLogo = FALSE)
cd <- getCellColData(proj)
writeLines(rownames(cd), "cells.txt")
writeLines(as.character(cd$Clusters), "clusters.txt")
dumpMat(getReducedDims(proj, "IterativeLSI"), "lsi")
dumpMat(as.matrix(getEmbedding(proj, "UMAP")), "umap")

## ---- step-by-step addTrajectory intermediates (copied from ArchR 1.0.3) ----
traj_steps <- function(mat_all, groupBy, trajectory, pre = 0.9, post = 0.9, dof = 250, spar = 1) {
  groupDF <- data.frame(g = groupBy, row.names = rownames(mat_all))
  groupDF <- groupDF[groupDF[, 1] %in% trajectory, , drop = FALSE]
  mat <- mat_all[rownames(groupDF), , drop = FALSE]
  filterObj <- lapply(seq_along(trajectory), function(x) {
    groupsx <- rownames(groupDF)[groupDF[, 1] == trajectory[x]]
    matx <- mat[groupsx, , drop = FALSE]
    diffx <- sqrt(colSums((t(matx) - colMeans(matx))^2))
    idxKeep <- which(diffx <= quantile(diffx, pre))
    list(mat = matx[idxKeep, , drop = FALSE], groups = groupsx[idxKeep])
  })
  matFilter <- Reduce("rbind", lapply(filterObj, `[[`, "mat"))
  groupsFilter <- groupDF[Reduce("c", lapply(filterObj, `[[`, "groups")), , drop = FALSE]
  initialTime <- unlist(lapply(seq_along(trajectory), function(x) {
    groupsx <- rownames(groupsFilter)[groupsFilter[, 1] == trajectory[x]]
    matx <- matFilter[groupsx, , drop = FALSE]
    if (x != length(trajectory)) {
      meanx <- colMeans(matFilter[rownames(groupsFilter)[groupsFilter[, 1] == trajectory[x + 1]], , drop = FALSE])
      (1 - .getQuantiles(sqrt(colSums((t(matx) - meanx)^2)))) + x
    } else {
      meanx <- colMeans(matFilter[rownames(groupsFilter)[groupsFilter[, 1] == trajectory[x - 1]], , drop = FALSE])
      .getQuantiles(sqrt(colSums((t(matx) - meanx)^2))) + x
    }
  }))
  matSpline <- sapply(seq_len(ncol(matFilter)), function(x)
    stats::smooth.spline(x = initialTime, y = matFilter[names(initialTime), x], df = dof, spar = spar)[[2]])
  knnObj <- nabor::knn(data = matSpline, query = mat, k = 3)
  list(initialTime = initialTime, matSpline = matSpline, knnIdx = knnObj[[1]], knnDist = knnObj[[2]])
}

runCase <- function(tag, embedding = NULL, useAll = FALSE) {
  name <- paste0("Traj_", tag)
  saveDF <- file.path(out, paste0("dfTrajectory_", tag, ".rds"))
  p <- addTrajectory(proj, name = name, trajectory = TRAJ, groupBy = "Clusters",
                     reducedDims = "IterativeLSI", embedding = embedding, useAll = useAll,
                     saveDF = saveDF, force = TRUE, seed = 1)
  df <- readRDS(saveDF)
  fwrite(data.frame(cell = rownames(df), Distance = sprintf("%.17g", df$Distance),
                    DistanceIdx = sprintf("%.17g", df$DistanceIdx),
                    Trajectory = sprintf("%.17g", df$Trajectory)),
         paste0("trajectory_", tag, ".tsv"), sep = "\t")
  mat_all <- if (is.null(embedding)) getReducedDims(proj, "IterativeLSI") else as.matrix(getEmbedding(proj, embedding))
  st <- traj_steps(mat_all, as.character(cd$Clusters), TRAJ)
  dumpMat(matrix(st$initialTime, ncol = 1, dimnames = list(names(st$initialTime), NULL)), paste0("initialTime_", tag))
  dumpMat(st$matSpline, paste0("matSpline_", tag))
  dumpMat(st$knnIdx, paste0("knnIdx_", tag)); dumpMat(st$knnDist, paste0("knnDist_", tag))
  p
}

proj <- runCase("lsi")
proj <- runCase("umap", embedding = "UMAP")
proj <- runCase("lsi_useall", useAll = TRUE)

## ---- getTrajectory ----
trajGSM <- getTrajectory(proj, name = "Traj_lsi", useMatrix = "GeneScoreMatrix",
                         log2Norm = TRUE, trajectoryLabel = "Clusters", threads = 1)
dumpMat(assay(trajGSM, "smoothMat"), "trajGSM_smooth"); dumpMat(assay(trajGSM, "mat"), "trajGSM_mat")
writeLines(as.character(colData(trajGSM)$label), "trajGSM_labels.txt")
trajMM <- getTrajectory(proj, name = "Traj_lsi", useMatrix = "MotifMatrix",
                        log2Norm = FALSE, scaleTo = NULL, threads = 1)
dumpMat(assay(trajMM, "smoothMat"), "trajMM_smooth"); dumpMat(assay(trajMM, "mat"), "trajMM_mat")
trajGSM5 <- getTrajectory(proj, name = "Traj_lsi", useMatrix = "GeneScoreMatrix",
                          groupEvery = 2, smoothWindow = 6, threads = 1)
dumpMat(assay(trajGSM5, "smoothMat"), "trajGSM_ge2_smooth"); dumpMat(assay(trajGSM5, "mat"), "trajGSM_ge2_mat")

## ---- plotTrajectoryHeatmap(returnMatrix = TRUE) ----
dumpMat(plotTrajectoryHeatmap(trajGSM, varCutOff = 0.9, returnMatrix = TRUE), "heatGSM")
dumpMat(plotTrajectoryHeatmap(trajMM, varCutOff = 0.9, returnMatrix = TRUE), "heatMM")
dumpMat(plotTrajectoryHeatmap(trajGSM, varCutOff = 0.75, maxFeatures = 1000, limits = c(-2, 2),
                              grepExclude = "^chrX:", returnMatrix = TRUE), "heatGSM_opts")

## ---- correlateTrajectories (motif rows renamed TFNAME_ID) ----
sq <- TFBSTools::getMatrixSet(JASPAR2020::JASPAR2020,
  opts = list(collection = "CORE", tax_group = "vertebrates", all_versions = FALSE))
id2name <- setNames(TFBSTools::name(sq), TFBSTools::ID(sq))
fwrite(data.frame(id = names(id2name), name = unname(id2name)), "jaspar_id2name.tsv", sep = "\t")
rd <- rowData(trajMM)
rd$name <- paste0(toupper(id2name[as.character(rd$name)]), "_", rd$name)
rowData(trajMM) <- rd
cor <- correlateTrajectories(trajGSM, trajMM, corCutOff = 0.5, varCutOff1 = 0.8, varCutOff2 = 0.8)
am <- as.data.frame(cor$allMappings)
for (cc in c("Correlation", "VarAssay1", "VarAssay2", "TStat", "Pval", "FDR")) am[[cc]] <- sprintf("%.17g", am[[cc]])
fwrite(am, "correlate_all.tsv", sep = "\t")
fwrite(data.frame(name1 = cor$correlatedMappings$name1, name2 = cor$correlatedMappings$name2),
       "correlate_pf.tsv", sep = "\t")

## ---- plotTrajectory: arrow path + pseudotime/value frame ----
for (nm in c("Traj_lsi", "CEBPB")) {
  cb <- if (nm == "CEBPB") "GeneScoreMatrix" else "cellColData"
  pl <- plotTrajectory(proj, trajectory = "Traj_lsi", colorBy = cb, name = nm,
                       embedding = "UMAP", imputeWeights = NULL, addArrow = TRUE)
  arrow <- pl[[1]]$layers[[length(pl[[1]]$layers)]]$data
  fwrite(data.frame(x = sprintf("%.17g", arrow$x), y = sprintf("%.17g", arrow$y),
                    PseudoTime = sprintf("%.17g", arrow$PseudoTime), value = sprintf("%.17g", arrow$value)),
         paste0("plotTraj_arrow_", nm, ".tsv"), sep = "\t")
  d2 <- pl[[2]]$data
  fwrite(data.frame(x = sprintf("%.17g", d2$x), y = sprintf("%.17g", d2$y)),
         paste0("plotTraj_points_", nm, ".tsv"), sep = "\t")
  if (nm == "Traj_lsi") { pdf("archr_plotTrajectory.pdf", width = 8, height = 4); print(ggAlignPlots(pl[[1]], pl[[2]], type = "h")); dev.off() }
}
pdf("archr_trajectory_heatmap.pdf", width = 6, height = 8)
ComplexHeatmap::draw(plotTrajectoryHeatmap(trajGSM, varCutOff = 0.9, labelTop = 30))
dev.off()
cat(as.character(packageVersion("ArchR")), "\n", file = "archr_version.txt")
cat("---DONE---\n")
