# ArchR reference outputs for epione.pl.marker_heatmap / peak2gene_heatmap parity.
# Usage: Rscript ref_archr.R <fixture ArchR_base dir> <work dir>
suppressPackageStartupMessages({
  library(ArchR); library(magrittr); library(ComplexHeatmap); library(SummarizedExperiment); library(S4Vectors)
})
args <- commandArgs(trailingOnly = TRUE)
FX <- args[1]; W <- args[2]
set.seed(1); addArchRThreads(threads = 8); addArchRGenome("hg19")
dir.create(file.path(W, "ref"), showWarnings = FALSE, recursive = TRUE)
setwd(W)

wbin <- function(m, tag) {
  m <- as.matrix(m); storage.mode(m) <- "double"
  writeBin(as.vector(m), file.path("ref", paste0(tag, ".f64")), size = 8)
  writeLines(as.character(dim(m)), file.path("ref", paste0(tag, ".dim")))
  if (!is.null(rownames(m))) writeLines(rownames(m), file.path("ref", paste0(tag, ".rows")))
  if (!is.null(colnames(m))) writeLines(colnames(m), file.path("ref", paste0(tag, ".cols")))
}
wvec <- function(v, tag) writeLines(as.character(v), file.path("ref", paste0(tag, ".txt")))

pdir <- file.path(W, "ArchR_proj")
if (file.exists(file.path(pdir, "Save-ArchR-Project.rds"))) {
  proj <- loadArchRProject(pdir, showLogo = FALSE)
} else {
  proj <- saveArchRProject(loadArchRProject(FX, showLogo = FALSE), outputDirectory = pdir, load = TRUE)
}

## ---------------------------------------------------------------- markers
get_markers <- function(useMatrix) {
  f <- file.path("ref", paste0("se_", useMatrix, ".rds"))
  if (file.exists(f)) return(readRDS(f))
  se <- getMarkerFeatures(ArchRProj = proj, useMatrix = useMatrix, groupBy = "Clusters",
                          bias = c("TSSEnrichment", "log10(nFrags)"), testMethod = "wilcoxon")
  saveRDS(se, f); se
}
dump_se <- function(se, tag) {
  for (an in names(assays(se))) wbin(assay(se, an), paste0(tag, "__", an))
  rd <- as.data.frame(rowData(se))
  write.table(rd, file.path("ref", paste0(tag, "__rowData.tsv")), sep = "\t", quote = FALSE, row.names = FALSE)
  wvec(metadata(se)$Params$useMatrix, paste0(tag, "__useMatrix"))
  wvec(colnames(se), paste0(tag, "__groups"))
}
seGS <- get_markers("GeneScoreMatrix"); dump_se(seGS, "GS")
sePK <- get_markers("PeakMatrix");      dump_se(sePK, "PK")

markerGenes <- c("CD34", "GATA1", "PAX5", "MS4A1", "EBF1", "MME", "CD14", "CEBPB", "MPO",
                 "IRF8", "CD3D", "CD8A", "TBX21", "IL7R")
configs <- list(
  gs_default   = list(se = seGS, args = list(cutOff = "FDR <= 0.01 & Log2FC >= 1.25", labelMarkers = markerGenes, transpose = TRUE)),
  gs_rows      = list(se = seGS, args = list(cutOff = "FDR <= 0.01 & Log2FC >= 1.25")),
  gs_log2fc    = list(se = seGS, args = list(cutOff = "FDR <= 0.05 & abs(Log2FC) >= 1", plotLog2FC = TRUE, limits = c(-3, 3))),
  gs_nobinary  = list(se = seGS, args = list(cutOff = "FDR <= 0.01 & Log2FC >= 1.5", binaryClusterRows = FALSE, nLabel = 5)),
  gs_subset    = list(se = seGS, args = list(subsetMarkers = as.integer(c(5, 40, 400, 900, 1500, 2600, 5000, 7000, 9000, 12000, 15000, 20000)))),
  pk_default   = list(se = sePK, args = list(cutOff = "FDR <= 0.1 & Log2FC >= 0.5", transpose = TRUE)),
  pk_rows      = list(se = sePK, args = list(cutOff = "FDR <= 0.1 & Log2FC >= 0.5")),
  pk_invert    = list(se = sePK, args = list(cutOff = "FDR <= 0.1 & Log2FC <= -0.5", invert = TRUE)),
  pk_nocluster = list(se = sePK, args = list(cutOff = "FDR <= 0.05 & Log2FC >= 1 & !(MeanBGD > 0.2)", clusterCols = FALSE,
                                             scaleRows = FALSE, log2Norm = FALSE, limits = c(0, 0.5)))
)
for (nm in names(configs)) {
  cfg <- configs[[nm]]
  mat <- do.call(plotMarkerHeatmap, c(list(seMarker = cfg$se, returnMatrix = TRUE), cfg$args))
  wbin(mat, paste0("hm_", nm, "__mat"))
  ht <- do.call(plotMarkerHeatmap, c(list(seMarker = cfg$se), cfg$args))
  pdf(file.path("ref", paste0("hm_", nm, ".pdf")), width = 8, height = 10)
  d <- draw(ht)
  dev.off()
  ro <- row_order(d); co <- column_order(d)
  if (is.list(ro)) ro <- ro[[1]]
  if (is.list(co)) co <- co[[1]]
  wvec(unlist(ro), paste0("hm_", nm, "__row_order"))
  wvec(unlist(co), paste0("hm_", nm, "__col_order"))
  # anno_mark labels: row annotation (HeatmapList) or top annotation (transpose)
  env <- NULL
  hl <- if (is(ht, "HeatmapList")) ht@ht_list else list(ht)
  for (h in hl) {
    if (is(h, "HeatmapAnnotation")) env <- h@anno_list[[1]]@fun@var_env
    if (is(h, "Heatmap") && !is.null(h@top_annotation)) env <- h@top_annotation@anno_list[[1]]@fun@var_env
  }
  if (!is.null(env)) {
    wvec(unname(env$at2labels[as.character(env$at)]), paste0("hm_", nm, "__labels"))
    wvec(env$at, paste0("hm_", nm, "__label_at"))
  }
}

## ---------------------------------------------------------------- peak2gene
if (is.null(metadata(proj@peakSet)$Peak2GeneLinks)) {
  proj <- addPeak2GeneLinks(proj, reducedDims = "IterativeLSI", useMatrix = "GeneScoreMatrix", seed = 1)
  proj <- saveArchRProject(proj, outputDirectory = pdir, load = TRUE)
}
p2gAll <- metadata(proj@peakSet)$Peak2GeneLinks
wbin(as.matrix(as.data.frame(p2gAll)[, c("idxATAC", "idxRNA", "Correlation", "FDR", "VarQATAC", "VarQRNA")]), "p2g_all")
seATAC <- readRDS(metadata(p2gAll)$seATAC); seRNA <- readRDS(metadata(p2gAll)$seRNA)
wbin(assay(seATAC), "p2g_seATAC"); wbin(assay(seRNA), "p2g_seRNA")
wvec(paste0(rowRanges(seATAC)), "p2g_seATAC_peaks"); wvec(rowData(seRNA)$name, "p2g_seRNA_genes")
knn <- as(metadata(seRNA)$KNNList, "list")
write.table(data.frame(group = rep(seq_along(knn), lengths(knn)), cell = unlist(knn)),
            file.path("ref", "p2g_knn.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
ccd <- getCellColData(proj, select = "Clusters")
write.table(data.frame(cell = rownames(ccd), Clusters = ccd$Clusters), file.path("ref", "cells_clusters.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

for (cfg in list(list(tag = "p2g_default", nPlot = 25000L, k = 25L, seed = 1),
                 list(tag = "p2g_small", nPlot = 4000L, k = 12L, seed = 7))) {
  res <- plotPeak2GeneHeatmap(proj, groupBy = "Clusters", returnMatrices = TRUE,
                              k = cfg$k, nPlot = cfg$nPlot, seed = cfg$seed)
  wbin(res$ATAC$matrix, paste0(cfg$tag, "__ATAC"))
  wbin(res$RNA$matrix, paste0(cfg$tag, "__RNA"))
  wvec(res$ATAC$kmeansId, paste0(cfg$tag, "__kmeansId"))
  wvec(res$ATAC$colData$groupBy, paste0(cfg$tag, "__colGroups"))
  wvec(rownames(res$ATAC$colData), paste0(cfg$tag, "__colOrder"))
  wvec(res$Peak2GeneLinks$peak, paste0(cfg$tag, "__peak"))
  wvec(res$Peak2GeneLinks$gene, paste0(cfg$tag, "__gene"))
  # Re-run the RNG-dependent block exactly as plotPeak2GeneHeatmap does to expose kmeans / sampling draws.
  p2g <- p2gAll[which(p2gAll$Correlation >= 0.45 & p2gAll$FDR <= 0.0001), , drop = FALSE]
  p2g <- p2g[which(p2g$VarQATAC > 0.25), ]
  p2g <- p2g[which(p2g$VarQRNA > 0.25), ]
  mATAC <- ArchR:::.rowZscores(assay(seATAC)[p2g$idxATAC, ])
  set.seed(cfg$seed)
  k1 <- kmeans(mATAC, cfg$k)
  wvec(k1$cluster, paste0(cfg$tag, "__k1cluster"))
  if (nrow(mATAC) > cfg$nPlot) {
    nPK <- cfg$nPlot * table(k1$cluster) / length(k1$cluster)
    splitK <- split(seq_len(nrow(mATAC)), k1$cluster)
    kDF <- lapply(seq_along(splitK), function(x) {
      idx <- sample(splitK[[x]], floor(nPK[x])); DataFrame(k = rep(x, length(idx)), idx = idx)
    }) %>% Reduce("rbind", .)
  } else {
    kDF <- DataFrame(k = k1$cluster, idx = seq_len(nrow(mATAC)))
  }
  write.table(as.data.frame(kDF), file.path("ref", paste0(cfg$tag, "__kDF.tsv")), sep = "\t", quote = FALSE, row.names = FALSE)
}
cat("R:", R.version.string, " ArchR:", as.character(packageVersion("ArchR")),
    " ComplexHeatmap:", as.character(packageVersion("ComplexHeatmap")), "\n")
cat("---DONE---\n")
