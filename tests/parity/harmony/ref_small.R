# Small, data-free reference used by tests/test_harmony.py (no fixture needed).
# Usage: Rscript ref_small.R <out_dir>
suppressPackageStartupMessages(library(harmony))
out <- commandArgs(trailingOnly = TRUE)[1]
i <- 1:300
x <- cbind(sin(i / 5) + (i %% 3 == 0) * 0.8, cos(i / 11), sin(i / 17) * cos(i / 3), (i %% 7) / 7 - (i %% 3 == 1) * 0.5)
batch <- c("b1", "b2", "b3")[(i %% 3) + 1]
set.seed(7)
km <- kmeans(x, centers = 6, iter.max = 25, nstart = 10)
write.table(format(km$centers, digits = 17), file.path(out, "small_kmeans_centers.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
set.seed(3)
h <- RunHarmony(x, data.frame(batch = batch), "batch", nclust = 10, verbose = FALSE)
write.table(format(h, digits = 17), file.path(out, "small_harmony.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
set.seed(3)
h2 <- RunHarmony(x, data.frame(batch = batch), "batch", nclust = 10, lambda = NULL, verbose = FALSE)
write.table(format(h2, digits = 17), file.path(out, "small_harmony_lambda_auto.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
