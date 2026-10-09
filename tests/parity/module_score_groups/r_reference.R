# ArchR 1.0.3 reference dump for epione.tl.add_module_score / epione.pl.plot_groups /
# epione.tl.confusion_matrix. Usage: Rscript r_reference.R <ArchRProject dir> <out dir>
suppressPackageStartupMessages({
  library(ArchR); library(Matrix); library(data.table); library(ggplot2)
})
args <- commandArgs(trailingOnly = TRUE)
projDir <- args[1]; out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
addArchRThreads(threads = 1); addArchRGenome("hg19"); addArchRVerbose(FALSE)
proj <- loadArchRProject(projDir, showLogo = FALSE)
cells <- getCellNames(proj)

# ---- confusionMatrix ------------------------------------------------------
cm <- confusionMatrix(proj$Clusters, proj$Sample)
fwrite(cbind(row = rownames(cm), as.data.frame(as.matrix(cm))), file.path(out, "cm_clusters_sample.tsv"), sep = "\t")

# ---- addModuleScore -------------------------------------------------------
features <- list(
  B = c("MS4A1", "CD79A", "PAX5"),
  T = c("CD3D", "CD3E", "CD2"),
  Mono = c("CD14", "LYZ", "CEBPB"),
  Erythroid = c("GATA1", "KLF1", "HBB"),
  Progenitor = c("CD34")
)
writeLines(jsonlite::toJSON(features, auto_unbox = FALSE), file.path(out, "modules.json"))

# Arrow-level row sums (what ArchR bins on) in ArchR's own feature order.
rS <- ArchR:::.getRowSums(ArrowFiles = getArrowFiles(proj), useMatrix = "GeneScoreMatrix")
fdf <- ArchR:::.getFeatureDF(head(getArrowFiles(proj), 2), subGroup = "GeneScoreMatrix")
fwrite(data.frame(seqnames = as.character(rS$seqnames), idx = rS$idx, rowSums = rS$rowSums),
       file.path(out, "gs_rowsums.tsv"), sep = "\t")
fwrite(data.frame(seqnames = as.character(fdf$seqnames), idx = fdf$idx, name = fdf$name),
       file.path(out, "gs_featuredf.tsv"), sep = "\t")

# Instrumented copy of addModuleScore that also records the background draws.
src <- deparse(ArchR::addModuleScore)
hit <- grep(".getPartialMatrix(", src, fixed = TRUE)
stopifnot(length(hit) == 1)
src <- append(src, "assign('.bgd', c(get('.bgd', envir = .GlobalEnv), list(idxBgd)), envir = .GlobalEnv)", after = hit - 1)
amsDump <- eval(parse(text = src)); environment(amsDump) <- asNamespace("ArchR")
.bgd <- list()

# ArchR 1.0.3 has two bugs on the name-based path of addModuleScore:
#  (1) featuresUse$Module gets one entry per *module* (length(idx[[z]]), idx is per
#      *gene*), so several multi-gene modules error or are silently mislabelled;
#  (2) m[seq_along(idxFgd), ] drops to a vector for one-gene modules -> colMeans error.
# "official" = unmodified ArchR, one call per multi-gene module (the only correct case).
# "patched"  = minimal two-token fix below, run on the full multi-module list.
patch <- function(src) {
  a <- grep("rep(names(features)[z], length(idx[[z]]))", src, fixed = TRUE); stopifnot(length(a) == 1)
  src[a] <- sub("length(idx[[z]])", "length(features[[z]])", src[a], fixed = TRUE)
  b <- grep("Matrix::colMeans(m[seq_along(idxFgd), ])", src, fixed = TRUE); stopifnot(length(b) == 1)
  src[b] <- gsub("seq_along(idxFgd), ])", "seq_along(idxFgd), , drop = FALSE])", src[b], fixed = TRUE)
  src
}
srcP <- patch(src)
amsDumpP <- eval(parse(text = srcP)); environment(amsDumpP) <- asNamespace("ArchR")
writeLines(srcP, file.path(out, "addModuleScore_patched.R"))
run_ms <- function(fun, feats, nBin, nBgd, seed) {
  p <- fun(proj, useMatrix = "GeneScoreMatrix", name = "Module", features = feats,
           nBin = nBin, nBgd = nBgd, seed = seed, threads = 1, verbose = FALSE)
  cd <- as.data.frame(getCellColData(p))
  cd[, grep("^Module\\.", colnames(cd)), drop = FALSE]
}
dump_ms <- function(fun, feats, nBin, nBgd, seed, tag, check = NULL) {
  .bgd <<- list()
  res <- run_ms(fun, feats, nBin, nBgd, seed)
  if (!is.null(check)) stopifnot(isTRUE(all.equal(as.matrix(check), as.matrix(res), tolerance = 0)))
  fwrite(cbind(barcode = rownames(res), res), file.path(out, paste0("module_scores_", tag, ".tsv")), sep = "\t")
  modOrder <- levels(factor(paste0("Module.", names(feats))))
  bg <- data.table(module = rep(modOrder, lengths(.bgd)), match = unlist(.bgd))
  fwrite(bg, file.path(out, paste0("module_bgd_", tag, ".tsv")), sep = "\t")
  invisible(res)
}
for (cfg in list(c(25, 100, 1), c(25, 25, 7))) {
  sfx <- sprintf("nbin%d_nbgd%d_seed%d", cfg[1], cfg[2], cfg[3])
  for (m in names(features)[lengths(features) >= 2]) {
    off <- run_ms(ArchR::addModuleScore, features[m], cfg[1], cfg[2], cfg[3])
    dump_ms(amsDump, features[m], cfg[1], cfg[2], cfg[3], paste0("official_", m, "_", sfx), check = off)
    dump_ms(amsDumpP, features[m], cfg[1], cfg[2], cfg[3], paste0("patchcheck_", m, "_", sfx), check = off)
  }
  dump_ms(amsDumpP, features, cfg[1], cfg[2], cfg[3], paste0("patched_all_", sfx))
}

# ---- plotGroups -----------------------------------------------------------
dump_pg <- function(p, tag) {
  d <- p$data
  fwrite(data.frame(x = as.character(d$x), y = d$y), file.path(out, paste0("pg_", tag, "_data.tsv")), sep = "\t")
  writeLines(levels(d$x), file.path(out, paste0("pg_", tag, "_levels.txt")))
  b <- ggplot_build(p)
  for (i in seq_along(b$data)) {
    di <- as.data.frame(b$data[[i]])
    keep <- vapply(di, function(col) is.atomic(col), logical(1))
    fwrite(di[, keep, drop = FALSE], file.path(out, paste0("pg_", tag, "_layer", i, ".tsv")), sep = "\t")
  }
}
gs_genes <- c("CD34", "GATA1", "MS4A1", "LYZ")
sampled_cells <- function(groupBy, maxCells, seed) {
  set.seed(seed)
  gn <- getCellColData(proj, groupBy, drop = FALSE)[, 1]; names(gn) <- rownames(getCellColData(proj))
  sg <- split(names(gn), gn)
  unlist(lapply(sg, function(x) if (length(x) > maxCells) sample(x, maxCells) else x), use.names = FALSE)
}
writeLines(sampled_cells("Clusters", 1000, 1), file.path(out, "pg_cells_clusters_seed1.txt"))
writeLines(sampled_cells("Sample", 500, 2), file.path(out, "pg_cells_sample_seed2.txt"))
for (plotAs in c("ridges", "violin")) {
  set.seed(1)
  pl <- plotGroups(proj, groupBy = "Clusters", colorBy = "GeneScoreMatrix", name = gs_genes,
                   imputeWeights = NULL, plotAs = plotAs)
  for (g in gs_genes) {
    # names in pl carry the requested feature names
    dump_pg(pl[[g]], paste0("gs_", g, "_", plotAs))
  }
  set.seed(2)
  p <- plotGroups(proj, groupBy = "Sample", colorBy = "cellColData", name = "TSSEnrichment",
                  plotAs = plotAs, maxCells = 500)
  dump_pg(p, paste0("tss_sample_", plotAs))
}
cat("ggplot2", as.character(packageVersion("ggplot2")), "ggridges", as.character(packageVersion("ggridges")), "\n")
cat("---DONE---\n")
