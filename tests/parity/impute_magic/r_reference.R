# ArchR reference for epione.tl.add_impute_weights / impute_matrix.
# Usage: Rscript r_reference.R <ArchR_project_dir> <out_dir>
# Runs addImputeWeights with threads = 1 (sequential R RNG) and useHdf5 = FALSE
# (nothing is written into the project) and dumps every intermediate needed
# for the Python parity check.
suppressPackageStartupMessages({ library(ArchR); library(Matrix); library(data.table) })
args <- commandArgs(trailingOnly = TRUE)
projDir <- args[1]; outDir <- args[2]
dir.create(outDir, showWarnings = FALSE, recursive = TRUE)
addArchRThreads(threads = 1); addArchRVerbose(FALSE)

proj <- loadArchRProject(projDir, showLogo = FALSE)

matDR <- getReducedDims(proj, reducedDims = "IterativeLSI", corCutOff = 0.75)
fwrite(cbind(barcode = rownames(matDR), as.data.frame(matDR)),
       file.path(outDir, "matDR.tsv.gz"), sep = "\t")

se <- getMatrixFromProject(proj, useMatrix = "GeneScoreMatrix")
gs <- assay(se); rownames(gs) <- rowData(se)$name
panel <- readLines(file.path(outDir, "panel_genes.txt"))
gs <- gs[panel, , drop = FALSE]

dumpWeights <- function(iW, tag) {
  W <- iW$Weights
  for (r in seq_along(W)) for (b in seq_along(W[[r]])) {
    m <- as.matrix(W[[r]][[b]])
    stem <- file.path(outDir, sprintf("%s_rep%d_block%d", tag, r, b))
    writeLines(rownames(m), paste0(stem, "_names.txt"))
    writeBin(as.vector(m), paste0(stem, "_weights.f64"), size = 8, endian = "little")
  }
}

runOne <- function(tag, seed, sampleCells = 5000, nRep = 2, dumpW = TRUE) {
  t0 <- Sys.time()
  p <- addImputeWeights(proj, reducedDims = "IterativeLSI", sampleCells = sampleCells,
                        nRep = nRep, k = 15, td = 3, ka = 4, epsilon = 1,
                        useHdf5 = FALSE, threads = 1, seed = seed)
  t1 <- Sys.time()
  iW <- getImputeWeights(p)
  if (dumpW) dumpWeights(iW, tag)
  imp <- imputeMatrix(as(gs, "dgCMatrix"), iW, threads = 1)
  t2 <- Sys.time()
  fwrite(cbind(gene = rownames(imp), as.data.frame(as.matrix(imp))),
         file.path(outDir, sprintf("%s_imputed.tsv.gz", tag)), sep = "\t")
  cat(sprintf("TIMING %s weights=%.2fs impute=%.2fs\n", tag,
              as.numeric(t1 - t0, units = "secs"), as.numeric(t2 - t1, units = "secs")))
}

runOne("seed1", 1)
runOne("seed2", 2, dumpW = FALSE)
fwrite(cbind(gene = rownames(gs), as.data.frame(as.matrix(gs))),
       file.path(outDir, "raw_genescore.tsv.gz"), sep = "\t")
cat(as.character(packageVersion("ArchR")), "\n", file = file.path(outDir, "archr_version.txt"))
cat("---DONE---\n")
