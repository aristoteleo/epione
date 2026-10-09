# R reference for epi.tl.harmony parity (ArchR 1.0.3 addHarmony -> harmony 1.2.4).
# Usage: Rscript ref_harmony.R <ArchR_project_dir> <out_dir> [seed]
suppressPackageStartupMessages({
  library(ArchR); library(harmony); library(data.table)
})
args <- commandArgs(trailingOnly = TRUE)
projDir <- args[1]; outDir <- args[2]
seed <- if (length(args) >= 3) as.integer(args[3]) else 1L
dir.create(outDir, showWarnings = FALSE, recursive = TRUE)
addArchRThreads(threads = 1); addArchRVerbose(FALSE)

proj <- loadArchRProject(projDir, showLogo = FALSE)
X <- getReducedDims(proj, reducedDims = "IterativeLSI")
meta <- data.frame(getCellColData(proj, select = "Sample")[rownames(X), , drop = FALSE])

fwrite(cbind(barcode = rownames(X), as.data.frame(X)), file.path(outDir, "input_lsi.tsv.gz"), sep = "\t")
fwrite(data.frame(barcode = rownames(meta), Sample = meta$Sample), file.path(outDir, "input_meta.tsv"), sep = "\t")

# ---- 1. RNG self-check (R's Mersenne-Twister + Rejection sampling) ----------
set.seed(seed)
rng <- list(runif = runif(7), sample = sample.int(nrow(X), 100L),
            runif_big = as.integer(runif(5, 0, 2147483647)))
set.seed(seed)
rng$seed_state <- .Random.seed
saveRDS(rng, file.path(outDir, "rng_check.rds"))
fwrite(data.table(runif = sprintf("%.17g", rng$runif)), file.path(outDir, "rng_runif.tsv"))
writeLines(as.character(rng$sample), file.path(outDir, "rng_sample.txt"))
writeLines(as.character(rng$runif_big), file.path(outDir, "rng_runif_big.txt"))
writeLines(as.character(rng$seed_state), file.path(outDir, "rng_seed_state.txt"))

# ---- 2. Free-running ArchR addHarmony ------------------------------------
set.seed(seed)
t0 <- Sys.time()
proj2 <- addHarmony(proj, reducedDims = "IterativeLSI", groupBy = "Sample",
                    name = "Harmony", force = TRUE, verbose = FALSE)
t_archr <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
H <- getReducedDims(proj2, reducedDims = "Harmony")
H <- H[rownames(X), , drop = FALSE]
fwrite(cbind(barcode = rownames(H), as.data.frame(H)), file.path(outDir, "archr_harmony.tsv.gz"), sep = "\t")

# ---- 3. Instrumented replay of RunHarmony.default (same defaults) --------
N <- nrow(X); K <- min(round(N / 30), 100)
sigma <- rep(0.1, K); theta0 <- 2; lambda_vec <- c(0, rep(1, nlevels(as.factor(meta$Sample))))
phi <- Matrix::t(Matrix::sparse.model.matrix(~0 + as.factor(meta$Sample)))
N_b <- Matrix::rowSums(phi); B_vec <- nlevels(as.factor(meta$Sample))
theta <- rep(theta0, B_vec) * (1 - exp(-(N_b / (K * 0))^2))
Hcls <- get("harmony", envir = asNamespace("harmony"))
set.seed(seed)
ho <- new(Hcls)
ho$setup(Matrix::t(X), phi, sigma, theta, lambda_vec, 0.2, 20L, 1e-3, 1e-2, K, 0.05, B_vec, FALSE)
Zcos <- ho$Z_cos
# k-means on the same Z_cos with an identical RNG state, for the stand-alone k-means check
state_before_kmeans <- .Random.seed
km <- kmeans(t(Zcos), centers = K, iter.max = 25, nstart = 10)
.Random.seed <<- state_before_kmeans
assign(".Random.seed", state_before_kmeans, envir = .GlobalEnv)
ho$init_cluster_cpp()
fwrite(as.data.table(km$centers), file.path(outDir, "kmeans_centers.tsv.gz"), sep = "\t")
fwrite(as.data.table(t(ho$Y)), file.path(outDir, "Y_init.tsv.gz"), sep = "\t")
fwrite(as.data.table(t(ho$R)), file.path(outDir, "R_init.tsv.gz"), sep = "\t")
states <- list(get(".Random.seed", envir = .GlobalEnv))
iters <- 0L
for (it in 1:10) {
  ho$cluster_cpp(); ho$moe_correct_ridge_cpp(); iters <- it
  states[[length(states) + 1]] <- get(".Random.seed", envir = .GlobalEnv)
  fwrite(as.data.table(t(ho$R)), file.path(outDir, sprintf("R_iter%02d.tsv.gz", it)), sep = "\t")
  fwrite(as.data.table(t(ho$Z_corr)), file.path(outDir, sprintf("Zcorr_iter%02d.tsv.gz", it)), sep = "\t")
  if (ho$check_convergence(1)) break
}
writeLines(as.character(states[[1]]), file.path(outDir, "rng_state_after_init.txt"))
writeLines(c(sprintf("%.9g", ho$objective_harmony)), file.path(outDir, "objective_harmony.txt"))
writeLines(c(sprintf("%.9g", ho$objective_kmeans)), file.path(outDir, "objective_kmeans.txt"))
writeLines(as.character(ho$kmeans_rounds), file.path(outDir, "kmeans_rounds.txt"))
Zfinal <- t(ho$Z_corr); rownames(Zfinal) <- rownames(X)
cat("replay vs addHarmony max|diff|:", max(abs(Zfinal - H)), "\n")
cat("harmony iterations:", iters, " K:", K, " N:", N, " d:", ncol(X), "\n")
cat("addHarmony seconds:", t_archr, "\n")
writeLines(c(paste0("iterations\t", iters), paste0("K\t", K), paste0("archr_seconds\t", t_archr),
             paste0("replay_vs_archr\t", max(abs(Zfinal - H)))), file.path(outDir, "summary.tsv"))
print(sessionInfo())
cat("---DONE---\n")
