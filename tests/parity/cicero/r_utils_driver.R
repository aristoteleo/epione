#!/usr/bin/env Rscript
# R reference for addendum gates 19-24 (exported Cicero utilities) on fixture F1 (cicero_data).
# Usage: Rscript r_utils_driver.R <ref_F1_dir_with_conns> <out_dir>
suppressPackageStartupMessages({library(cicero); library(Matrix); library(data.table)})
a <- commandArgs(TRUE); ref <- a[1]; out <- a[2]; dir.create(out, showWarnings = FALSE, recursive = TRUE)
wmat <- function(m, pre) {
  m <- as(m, "CsparseMatrix")
  writeLines(rownames(m), file.path(out, paste0(pre, "_rows.txt"))); writeLines(colnames(m), file.path(out, paste0(pre, "_cols.txt")))
  writeBin(as.integer(m@i), file.path(out, paste0(pre, "_i.i32")), size = 4)
  writeBin(as.double(m@p), file.path(out, paste0(pre, "_p.f64")), size = 8)
  writeBin(as.double(m@x), file.path(out, paste0(pre, "_x.f64")), size = 8)
}
data("cicero_data"); data("cell_data")
fwrite(cicero_data, file.path(out, "cicero_data.tsv"), sep = "\t")
fwrite(cell_data, file.path(out, "cell_data.tsv"), sep = "\t")

# gate 19: make_atac_cds
cds <- make_atac_cds(cicero_data, binarize = TRUE)
wmat(exprs(cds), "atac")
fwrite(data.table(peak = rownames(fData(cds)), chr = fData(cds)$chr, bp1 = fData(cds)$bp1, bp2 = fData(cds)$bp2,
                  num_cells_expressed = fData(cds)$num_cells_expressed), file.path(out, "atac_fdata.tsv"), sep = "\t")
fwrite(data.table(cell = colnames(cds), num_genes_expressed = pData(cds)$num_genes_expressed), file.path(out, "atac_pdata.tsv"), sep = "\t")
cds_nb <- make_atac_cds(cicero_data, binarize = FALSE); wmat(exprs(cds_nb), "atac_counts")

# gate 20: aggregate_nearby_peaks
for (d in c(1000, 10000)) wmat(exprs(aggregate_nearby_peaks(cds, distance = d)), paste0("anp_", d))

# gate 21: aggregate_by_cell_bin (group = timepoint)
pData(cds)$timepoint <- cell_data$timepoint[match(colnames(cds), cell_data$cell)]
ab <- aggregate_by_cell_bin(cds, "timepoint")
M <- exprs(ab)
writeLines(rownames(M), file.path(out, "abc_rows.txt")); writeLines(colnames(M), file.path(out, "abc_cols.txt"))
writeBin(as.double(as.matrix(M)), file.path(out, "abc.f64"), size = 8)
fwrite(data.table(bin = colnames(ab), Size_Factor = sprintf("%.17g", pData(ab)$Size_Factor),
                  num_genes_expressed = pData(ab)$num_genes_expressed, n = pData(ab)$n), file.path(out, "abc_pdata.tsv"), sep = "\t")

# gate 22: compare_connections
p1 <- readLines(file.path(ref, "conns_peak1.txt")); p2 <- readLines(file.path(ref, "conns_peak2.txt"))
co <- readBin(file.path(ref, "conns_coaccess.f64"), "double", n = length(p1), size = 8)
conns <- data.frame(Peak1 = p1, Peak2 = p2, coaccess = co, stringsAsFactors = FALSE)
c1 <- conns[seq(1, nrow(conns), by = 50), ]
set.seed(1)
alt <- conns[sample(nrow(conns), 5000), c("Peak1", "Peak2")]
shift <- function(p, s) { d <- df_for_coords(p); paste(d$chr, d$bp1 + s, d$bp2 + s, sep = "_") }
alt$Peak1 <- shift(alt$Peak1, 600); alt$Peak2 <- shift(alt$Peak2, -300)
fwrite(c1[, 1:2], file.path(out, "cc_conns1.tsv"), sep = "\t"); fwrite(alt, file.path(out, "cc_alt.tsv"), sep = "\t")
for (g in c(0, 1000)) writeLines(as.character(compare_connections(c1, alt, maxgap = g)), file.path(out, paste0("cc_maxgap", g, ".txt")))

# gate 23: find_overlapping_coordinates
pk <- rownames(cds)
q1 <- "chr18:10,100-1246234"
qs <- c("chr18:10,100-50000", "chr18:2,000,000-2,100,000", "chr18:99999999-100000000", "chr18_157883_158536")
for (g in c(0, 500)) {
  writeLines(find_overlapping_coordinates(pk, q1, maxgap = g), file.path(out, paste0("foc_single_", g, ".txt")))
  writeLines(find_overlapping_coordinates(pk, qs, maxgap = g), file.path(out, paste0("foc_multi_", g, ".txt")))
}

# gate 24: df_for_coords
d <- df_for_coords(c(pk, "chr1:2,039-30,239", "chrX:28884:101293", "chr18-8,575,097-8,839,855"))
fwrite(d, file.path(out, "dfc.tsv"), sep = "\t")
cat("utils done\n")
