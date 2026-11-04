# Reference values for gates D1 (robust::covRob) and D2 (limma::normalizeQuantiles).
# Usage: Rscript dchic_make_ref.R <workdir>
# <workdir>/d1_in/*.bin, d2_in/*.bin: raw little-endian float64, column-major, with
# <name>.dim holding "nrow ncol". Writes d1_ref/<name>.bin (cov, p x p) and
# d2_ref/<name>.bin (normalised matrix) plus <name>.estim for D1.
suppressMessages(library(robust))
args <- commandArgs(TRUE)
wd <- args[1]
rd <- function(f) {
  d <- scan(sub("\\.bin$", ".dim", f), quiet = TRUE)
  matrix(readBin(f, "double", n = d[1] * d[2], endian = "little"), d[1], d[2])
}
dir.create(file.path(wd, "d1_ref"), showWarnings = FALSE)
dir.create(file.path(wd, "d2_ref"), showWarnings = FALSE)
for (f in list.files(file.path(wd, "d1_in"), pattern = "\\.bin$", full.names = TRUE)) {
  x <- rd(f)
  t0 <- proc.time()[["elapsed"]]
  r <- covRob(x)
  el <- proc.time()[["elapsed"]] - t0
  nm <- basename(f)
  writeBin(as.vector(r$cov), file.path(wd, "d1_ref", nm), endian = "little")
  writeLines(c(r$estim, sprintf("%.4f", el)), file.path(wd, "d1_ref", sub("\\.bin$", ".estim", nm)))
}
for (f in list.files(file.path(wd, "d2_in"), pattern = "\\.bin$", full.names = TRUE)) {
  x <- rd(f)
  q <- limma::normalizeQuantiles(x, ties = TRUE)
  writeBin(as.vector(q), file.path(wd, "d2_ref", basename(f)), endian = "little")
}
