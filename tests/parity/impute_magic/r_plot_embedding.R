# ArchR plotEmbedding with MAGIC imputation, for the tutorial's visual comparison.
# Usage: Rscript r_plot_embedding.R <ArchR_project_dir> <out_dir>
suppressPackageStartupMessages({ library(ArchR) })
args <- commandArgs(trailingOnly = TRUE)
addArchRThreads(threads = 1); addArchRVerbose(FALSE)
proj <- loadArchRProject(args[1], showLogo = FALSE)
proj <- addImputeWeights(proj, sampleCells = 5000, nRep = 2, k = 15, td = 3, ka = 4,
                         epsilon = 1, useHdf5 = FALSE, threads = 1, seed = 1)
markers <- c("CD34", "GATA1", "PAX5", "MS4A1", "CD14", "CD3D")
ps <- plotEmbedding(proj, colorBy = "GeneScoreMatrix", name = markers, embedding = "UMAP",
                    imputeWeights = getImputeWeights(proj), plotAs = "points", size = 0.3)
ps <- lapply(ps, function(p) p + ggplot2::theme(legend.position = "none",
  plot.title = ggplot2::element_text(size = 7)))
png(file.path(args[2], "archr_plotEmbedding_imputed.png"), width = 1800, height = 330, res = 110)
print(ggAlignPlots(plotList = ps, type = "h"))
dev.off()
cat("---DONE---\n")
