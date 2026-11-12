#!/usr/bin/env Rscript
# Fixture F1: Cicero's own example data (cicero_data, HSMM, hg19 chr18), as in the Cicero
# vignette: make_atac_cds(binarize = TRUE), then monocle3 LSI + UMAP computed ONCE here and
# frozen, so both R and Python aggregate cells on identical reduced coordinates.
# Writes the shared fixture format (see read_fixture() in r_reference_driver.R):
#   peaks.txt, cells.txt, x.i.bin (int32, 0-based row), x.p.bin (float64 col ptr),
#   x.x.bin (float64 values), shape.txt, coords.csv, genome.tsv
# Usage: Rscript make_fixture_F1.R <outdir>
suppressPackageStartupMessages({library(cicero); library(Matrix)})
out <- commandArgs(TRUE)[1]; dir.create(out, showWarnings = FALSE, recursive = TRUE)
data("cicero_data"); data("human.hg19.genome")
input_cds <- make_atac_cds(cicero_data, binarize = TRUE)
set.seed(2017)
input_cds <- detect_genes(input_cds)
input_cds <- estimate_size_factors(input_cds)
input_cds <- preprocess_cds(input_cds, method = "LSI")
input_cds <- reduce_dimension(input_cds, reduction_method = "UMAP", preprocess_method = "LSI")
umap <- reducedDims(input_cds)$UMAP
m <- as(exprs(input_cds), "CsparseMatrix")
writeLines(rownames(m), file.path(out, "peaks.txt")); writeLines(colnames(m), file.path(out, "cells.txt"))
writeBin(as.integer(m@i), file.path(out, "x.i.bin"), size = 4)
writeBin(as.double(m@p), file.path(out, "x.p.bin"), size = 8)
writeBin(as.double(m@x), file.path(out, "x.x.bin"), size = 8)
writeLines(as.character(c(nrow(m), ncol(m), length(m@x))), file.path(out, "shape.txt"))
write.csv(data.frame(cell = rownames(umap), UMAP_1 = sprintf("%.17g", umap[, 1]), UMAP_2 = sprintf("%.17g", umap[, 2])),
          file.path(out, "coords.csv"), row.names = FALSE, quote = FALSE)
g <- subset(human.hg19.genome, V1 == "chr18")
write.table(g, file.path(out, "genome.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
# promoter annotation exactly as the Cicero vignette builds it (for the gene-activity gates)
data(gene_annotation_sample)
pos <- subset(gene_annotation_sample, strand == "+"); pos <- pos[order(pos$start), ]
pos <- pos[!duplicated(pos$transcript), ]; pos$end <- pos$start + 1
neg <- subset(gene_annotation_sample, strand == "-"); neg <- neg[order(neg$start, decreasing = TRUE), ]
neg <- neg[!duplicated(neg$transcript), ]; neg$start <- neg$end - 1
gene_annotation_sub <- rbind(pos, neg)[, c(1:3, 8)]; names(gene_annotation_sub)[4] <- "gene"
write.table(gene_annotation_sub, file.path(out, "promoters.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("F1", nrow(m), "peaks x", ncol(m), "cells; nnz", length(m@x), "\n")
