# ArchR reference outputs for epione.tl.peak_anno_enrichment parity.
# Usage: Rscript r_reference.R <fixture_dir> <out_dir> <ArchR-Hg19-v1.Anno>
suppressPackageStartupMessages({
  library(ArchR); library(Matrix); library(SummarizedExperiment); library(S4Vectors)
  library(data.table); library(GenomicRanges)
})
args <- commandArgs(trailingOnly = TRUE)
FX <- args[1]; OUT <- args[2]; ANNO <- args[3]
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
setwd(OUT)
set.seed(1); addArchRThreads(threads = 8); addArchRGenome("hg19")

proj <- loadArchRProject(file.path(FX, "ArchR_base"), showLogo = FALSE)
# addPeakAnnotations / addArchRAnnotations write .rds files into the project
# output directory; redirect that (in memory only) so the fixture stays read-only.
proj@projectMetadata$outputDirectory <- file.path(OUT, "proj_out")
dir.create(file.path(OUT, "proj_out"), showWarnings = FALSE)

wtab <- function(x, f) fwrite(as.data.frame(x), f, sep = "\t", row.names = FALSE)
dump_se <- function(se, prefix) {
  for (an in names(assays(se))) {
    m <- as.matrix(assay(se, an))
    wtab(cbind(feature = rownames(se), as.data.frame(m)), paste0(prefix, "_", an, ".tsv.gz"))
  }
}

## 1. Marker peaks (input to the enrichment test)
seMarker <- getMarkerFeatures(
  ArchRProj = proj, useMatrix = "PeakMatrix", groupBy = "Clusters",
  bias = c("TSSEnrichment", "log10(nFrags)"), testMethod = "wilcoxon"
)
rd <- as.data.frame(rowData(seMarker))
wtab(rd, "markers_rowdata.tsv")
for (an in names(assays(seMarker))) {
  wtab(as.data.frame(as.matrix(assay(seMarker, an))), paste0("markers_", an, ".tsv.gz"))
}

## 2. Motif enrichment, background = all peaks / matched bgd peaks
cut <- "FDR <= 0.1 & Log2FC >= 0.5"
eAll <- peakAnnoEnrichment(seMarker, proj, peakAnnotation = "Motif", cutOff = cut)
dump_se(eAll, "enrich_motif_all")
eBgd <- peakAnnoEnrichment(seMarker, proj, peakAnnotation = "Motif", cutOff = cut, background = "bgdPeaks")
dump_se(eBgd, "enrich_motif_bgd")
eLoose <- peakAnnoEnrichment(seMarker, proj, peakAnnotation = "Motif", cutOff = "FDR <= 0.5 | Log2FC >= 2")
dump_se(eLoose, "enrich_motif_or")
bgd <- getBgdPeaks(proj)
wtab(as.data.frame(assay(bgd)), "bgd_peaks.tsv.gz")

## 3. Heatmap matrices (returnMatrix = TRUE)
hm <- list(
  default   = list(),
  n7_t      = list(n = 7, transpose = TRUE),
  n5_nocc   = list(n = 5, clusterCols = FALSE),
  nobin     = list(n = 7, binaryClusterRows = FALSE),
  pmax30    = list(n = 10, pMax = 30, cutOff = 5),
  pmax100_t = list(n = 10, pMax = 100, cutOff = 5, transpose = TRUE)
)
for (nm in names(hm)) {
  m <- do.call(plotEnrichHeatmap, c(list(seEnrich = eAll, returnMatrix = TRUE), hm[[nm]]))
  wtab(cbind(row = rownames(m), as.data.frame(m)), paste0("heatmap_", nm, ".tsv"))
}
m <- plotEnrichHeatmap(eBgd, n = 7, cutOff = 2, returnMatrix = TRUE)
wtab(cbind(row = rownames(m), as.data.frame(m)), "heatmap_bgd_n7.tsv")

## 4. hclust order reference (R's Fortran hclust + hcass2) on random inputs
set.seed(42)
ord <- lapply(1:30, function(i) {
  n <- sample(3:15, 1); p <- sample(2:20, 1)
  x <- matrix(round(runif(n * p) * 100, sample(0:3, 1)), n, p)
  wtab(as.data.frame(x), sprintf("hclust_in_%02d.tsv", i))
  hclust(dist(x))$order
})
writeLines(vapply(ord, paste, "", collapse = ","), "hclust_orders.txt")

## 5. addPeakAnnotations with custom BED region sets
tss <- getGeneAnnotation(proj)$TSS
prom <- resize(tss, width = 2001, fix = "center")
fwrite(data.frame(seqnames(prom), start(prom), end(prom)), "regions_promoters.bed", sep = "\t", col.names = FALSE)
cs <- getChromSizes(proj)
cs <- cs[as.character(seqnames(cs)) %in% paste0("chr", 1:22)]
set.seed(7)
mk_rand <- function(n, w) {
  ch <- sample(seq_along(cs), n, replace = TRUE, prob = as.numeric(end(cs)))
  st <- vapply(ch, function(i) sample.int(end(cs)[i] - w, 1), 1L)
  data.frame(as.character(seqnames(cs))[ch], st, st + w)
}
fwrite(mk_rand(20000, 500), "regions_random500.bed", sep = "\t", col.names = FALSE)
fwrite(mk_rand(3000, 5000), "regions_random5k.bed", sep = "\t", col.names = FALSE)
fwrite(data.frame("chr1", 1L, 2L), "regions_nohit.bed", sep = "\t", col.names = FALSE)
regs <- c(promoters = "regions_promoters.bed", random500 = "regions_random500.bed",
          random5k = "regions_random5k.bed", nohit = "regions_nohit.bed")
proj <- addPeakAnnotations(proj, regions = regs, name = "Region", force = TRUE)
rm_ <- getMatches(proj, "Region")
writeMM(as(assay(rm_) * 1, "CsparseMatrix"), "region_matches.mtx")
writeLines(colnames(rm_), "region_matches_names.txt")
eReg <- peakAnnoEnrichment(seMarker, proj, peakAnnotation = "Region", cutOff = cut)
dump_se(eReg, "enrich_region_all")

## 6. addArchRAnnotations from the ArchR hg19 database file
for (col in c("EncodeTFBS", "ATAC")) {
  proj <- addArchRAnnotations(proj, db = ANNO, collection = col, name = col, force = TRUE)
  am <- getMatches(proj, col)
  writeMM(as(assay(am) * 1, "CsparseMatrix"), paste0("archr_", col, "_matches.mtx"))
  writeLines(colnames(am), paste0("archr_", col, "_names.txt"))
  wtab(cbind(name = colnames(am), as.data.frame(colData(am))), paste0("archr_", col, "_info.tsv"))
  e <- peakAnnoEnrichment(seMarker, proj, peakAnnotation = col, cutOff = cut)
  dump_se(e, paste0("enrich_", col, "_all"))
}

## 7. customEnrichment on an arbitrary peak subset
peaks <- getPeakSet(proj)
p1 <- peaks[names(peaks) == "C1"]
wtab(data.frame(seqnames = as.character(seqnames(p1)), start = start(p1), end = end(p1)), "custom_ranges.tsv")
ce <- customEnrichment(ranges = p1, matches = getMatches(proj, "Motif"))
wtab(ce, "custom_all.tsv")
ce2 <- customEnrichment(ranges = p1, matches = getMatches(proj, "Motif"), bgdPeaks = getBgdPeaks(proj))
wtab(ce2, "custom_bgd.tsv")

## 8. Raw (unrounded) -log10 p for precision diagnostics
mt <- assay(getMatches(proj, "Motif"))
idx <- which(as.matrix(assay(seMarker, "FDR")) [, 1] <= 0.1 & as.matrix(assay(seMarker, "Log2FC"))[, 1] >= 0.5)
cf <- Matrix::colSums(mt[idx, , drop = FALSE]); bf <- Matrix::colSums(mt)
raw <- -phyper(cf - 1, bf, nrow(mt) - bf, length(idx), lower.tail = FALSE, log.p = TRUE) / log(10)
wtab(data.frame(feature = colnames(mt), raw = sprintf("%.17g", raw)), "raw_mlog10p_group1.tsv")

writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
cat("---DONE---\n")
