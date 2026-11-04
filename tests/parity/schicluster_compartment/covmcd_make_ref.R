# Reference for gate D8: robust::covRob(x) in its mcd branch, after set.seed(seed).
# usage: Rscript covmcd_make_ref.R <dir> <cases.tsv>
# cases.tsv: name <tab> seed, one line per case; <dir>/<name>.bin (raw float64,
# column-major) and <dir>/<name>.dim ("n p") hold the matrix.
# Writes <dir>/<name>_s<seed>.{center,cov,seed}.bin and <dir>/timings_R.tsv.
args <- commandArgs(TRUE)
suppressMessages(library(robust))
d <- args[1]
cases <- read.table(args[2], sep = "\t", col.names = c("name", "seed"), colClasses = c("character", "integer"))
tim <- data.frame(name = character(), seed = integer(), estim = character(), seconds = numeric())
for (i in seq_len(nrow(cases))) {
  nm <- cases$name[i]; s <- cases$seed[i]
  dm <- scan(file.path(d, paste0(nm, ".dim")), quiet = TRUE)
  x <- matrix(readBin(file.path(d, paste0(nm, ".bin")), "double", dm[1] * dm[2]), dm[1], dm[2])
  set.seed(s)
  t <- system.time(r <- covRob(x))[["elapsed"]]
  pre <- file.path(d, paste0(nm, "_s", s))
  writeBin(as.double(r$center), paste0(pre, ".center.bin"))
  writeBin(as.double(r$cov), paste0(pre, ".cov.bin"))
  writeBin(as.integer(.Random.seed), paste0(pre, ".seed.bin"))
  tim[nrow(tim) + 1, ] <- list(nm, s, r$estim, t)
}
write.table(tim, file.path(d, "timings_R.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
print(tim)
