# Gate D6 reference: robust::covRob on the D6 matrices (run in the `dchic` env, in the work dir).
# Matrices travel as raw little-endian float64, column-major, with a "<name>.dim" file
# (n p), so R and Python see identical bits (CSV parsing differs between R and pandas).
suppressMessages(library(robust))
set.seed(2026)
shapes <- list(c(300,3), c(900,4), c(2500,3), c(4000,4), c(800,9))
for (s in shapes) {
  n <- s[1]; p <- s[2]
  x <- matrix(rnorm(n*p), n, p) %*% matrix(runif(p*p, -1, 1), p, p)
  k <- floor(0.08*n); x[1:k, ] <- x[1:k, ] + matrix(rnorm(k*p, 6, 2), k, p)
  name <- sprintf("rand_%dx%d", n, p)
  writeBin(as.vector(x), paste0(name, ".bin"), size = 8, endian = "little")
  write(c(n, p), paste0(name, ".dim"), ncolumns = 2)
}
names <- c(sprintf("rand_%dx%d", sapply(shapes, `[`, 1), sapply(shapes, `[`, 2)), "real_chr1", "real_chr2")
for (name in names) {
  d <- scan(paste0(name, ".dim"), quiet = TRUE)
  x <- matrix(readBin(paste0(name, ".bin"), "double", n = d[1] * d[2], size = 8, endian = "little"), d[1], d[2])
  t0 <- proc.time()[["elapsed"]]
  r <- covRob(x)
  el <- proc.time()[["elapsed"]] - t0
  writeBin(as.vector(r$center), paste0(name, ".center.bin"), size = 8, endian = "little")
  writeBin(as.vector(r$cov), paste0(name, ".cov.bin"), size = 8, endian = "little")
  cat(name, d[1], d[2], r$estim, sprintf("%.3f", el), "\n")
}
