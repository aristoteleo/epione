# ArchR 1.0.3 reference outputs for epione.pl.plot_browser_track parity.
# Usage: Rscript r_reference.R <ArchR_project_dir> <out_dir> [threads]
suppressPackageStartupMessages({
  library(ArchR); library(data.table); library(grid)
})
args <- commandArgs(trailingOnly = TRUE)
projDir <- args[1]; outDir <- args[2]
threads <- if (length(args) >= 3) as.integer(args[3]) else 4L
dir.create(outDir, showWarnings = FALSE, recursive = TRUE)
setwd(outDir)
addArchRThreads(threads = threads); addArchRGenome("hg19")

proj <- loadArchRProject(projDir, showLogo = FALSE)
genes <- c("CD34", "GATA1", "MS4A1", "LYZ", "CD3D")
tileSize <- 250L; maxCells <- 500L; minCells <- 25L; normMethod <- "ReadsInTSS"
groupBy <- "Clusters"; seed <- 1L
ga <- getGeneAnnotation(proj)

# Co-accessibility loops (in-memory only; nothing is written to Arrow files).
proj <- addCoAccessibility(proj, reducedDims = "IterativeLSI", seed = seed,
                           threads = threads, verbose = FALSE)
loops <- getCoAccessibility(proj, corCutOff = 0.5, resolution = 1000, returnLoops = TRUE)
lp <- loops[[1]]
fwrite(data.frame(chrom = as.character(seqnames(lp)), start = start(lp), end = end(lp),
                  value = mcols(lp)$value), "loops.tsv", sep = "\t")

geneRegion <- function(sym) {
  region <- ga$genes[which(tolower(mcols(ga$genes)$symbol) %in% tolower(sym))]
  region <- GenomicRanges::resize(region, 1, "start")
  strand(region) <- "*"
  extendGR(region, upstream = 50000, downstream = 50000)
}

# Replicates the maxCells sampling block of ArchR:::.groupRegionSumArrows so the
# exact cells drawn under set.seed(seed) can be exported and injected in Python.
sampledCells <- function(proj) {
  cellGroups <- getCellColData(proj, groupBy, drop = TRUE)
  keep <- names(table(cellGroups)[table(cellGroups) >= minCells])
  ccd <- getCellColData(proj)[cellGroups %in% keep, , drop = FALSE]
  splitGroups <- split(rownames(ccd), ccd[, groupBy])
  unlist(lapply(seq_along(splitGroups), function(x) {
    if (length(splitGroups[[x]]) > maxCells) sample(splitGroups[[x]], maxCells) else splitGroups[[x]]
  }))
}

summ <- list()
for (g in genes) {
  region <- geneRegion(g)
  fwrite(data.frame(gene = g, chrom = as.character(seqnames(region)),
                    start = start(region), end = end(region)),
         paste0(g, "_region.tsv"), sep = "\t")

  set.seed(seed); useCells <- sampledCells(proj)
  writeLines(useCells, paste0(g, "_cells.txt"))

  set.seed(seed)
  raw <- ArchR:::.groupRegionSumArrows(ArchRProj = proj, groupBy = groupBy,
    normMethod = normMethod, useGroups = NULL, sampleLabels = "Sample",
    maxCells = maxCells, minCells = minCells, region = region,
    tileSize = tileSize, threads = 1)
  fwrite(raw, paste0(g, "_raw.tsv"), sep = "\t")

  set.seed(seed)
  pb <- ArchR:::.bulkTracks(ArchRProj = proj, region = region, tileSize = tileSize,
    groupBy = groupBy, sampleLabels = "Sample", threads = 1, maxCells = maxCells,
    minCells = minCells, normMethod = normMethod, geneAnnotation = ga)
  fwrite(pb$data, paste0(g, "_bulk.tsv"), sep = "\t")

  # Consistency check: the injected-cell recomputation equals ArchR's own draw.
  p2 <- proj; p2@cellColData <- p2@cellColData[useCells, , drop = FALSE]
  raw2 <- ArchR:::.groupRegionSumArrows(ArchRProj = p2, groupBy = groupBy,
    normMethod = normMethod, sampleLabels = "Sample", maxCells = .Machine$integer.max,
    minCells = minCells, region = region, tileSize = tileSize, threads = 1)
  summ[[g]] <- data.frame(gene = g, n_rows = nrow(raw),
                          injected_max_abs = max(abs(raw2$y - raw$y)))

  pf <- ArchR:::.featureTracks(features = getPeakSet(proj), region = region,
                               facetbaseSize = 7, hideX = TRUE, title = "Peaks")
  fwrite(pf$data[, c("seqnames", "start", "end")], paste0(g, "_features.tsv"), sep = "\t")

  pg <- ArchR:::.geneTracks(geneAnnotation = ga, region = region, facetbaseSize = 7, title = "Genes")
  gd <- pg$data
  if ("cluster" %in% colnames(gd)) {
    fwrite(gd[, c("seqnames", "start", "end", "strand", "symbol", "cluster")],
           paste0(g, "_genes.tsv"), sep = "\t")
  }

  pl <- ArchR:::.loopTracks(loops = loops, region = region, facetbaseSize = 7,
                            hideX = TRUE, hideY = TRUE, title = "Loops")
  if ("x" %in% colnames(pl$data)) {
    fwrite(data.frame(pl$data)[, c("x", "y", "id", "value")], paste0(g, "_loops.tsv"), sep = "\t")
  }

  set.seed(seed)
  gg <- plotBrowserTrack(proj, geneSymbol = g, groupBy = groupBy, loops = loops,
                         tileSize = tileSize, maxCells = maxCells, minCells = minCells,
                         normMethod = normMethod, verbose = FALSE, logFile = "browser.log")
  png(paste0(g, "_archr.png"), width = 6, height = 7, units = "in", res = 110)
  grid::grid.newpage(); grid::grid.draw(gg[[1]]); dev.off()
}
fwrite(do.call(rbind, summ), "summary.tsv", sep = "\t")
pal <- paletteDiscrete(values = gtools::mixedsort(unique(getCellColData(proj, groupBy, drop = TRUE))))
fwrite(data.frame(group = names(pal), color = unname(pal)), "palette.tsv", sep = "\t")
cat("---DONE---\n")
