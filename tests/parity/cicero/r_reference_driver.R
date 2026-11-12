#!/usr/bin/env Rscript
# R reference driver for the Cicero -> epione parity gates (see MANIFEST.md / manifest.yaml).
#
# Runs Cicero 1.3.9 (Monocle 3 branch) on a fixture directory and dumps every intermediate
# that a gate needs, in plain binary/text files readable without rpy2/pyreadr.
#
# Usage: Rscript r_reference_driver.R <fixture_dir> <out_dir> <seed> [full|light] [ccan_seeds]
#   full  : F1/F2 -- all intermediates, glasso probes, per-window distance parameters, run_cicero() identity check
#   light : F3    -- canonical pipeline + timings + coaccess table only
#
# run_cicero() is executed as its four documented steps (estimate_distance_parameter ->
# mean -> generate_cicero_models -> assemble_connections) with run_cicero's own arguments,
# so each step can be timed and dumped. In `full` mode run_cicero() itself is also called
# with the same seed and its table is checked to be identical.
suppressPackageStartupMessages({library(cicero); library(Matrix); library(data.table)})
a <- commandArgs(TRUE)
fx <- a[1]; out <- a[2]; seed <- as.integer(a[3]); mode <- ifelse(length(a) >= 4, a[4], "full")
ccan_seeds <- if (length(a) >= 5) as.integer(strsplit(a[5], ",")[[1]]) else integer(0)
dir.create(out, showWarnings = FALSE, recursive = TRUE)
lg <- function(...) message(format(Sys.time()), " | ", ...)
timings <- list(); tic <- function() proc.time()[["elapsed"]]
rec <- function(name, t0) { timings[[name]] <<- tic() - t0; lg(name, " ", round(timings[[name]], 2), " s") }
wd <- function(x, f) writeBin(as.double(x), file.path(out, f), size = 8)
wi <- function(x, f) writeBin(as.integer(x), file.path(out, f), size = 4)

# ---------------------------------------------------------------- fixture
read_fixture <- function(d) {
  sh <- as.numeric(readLines(file.path(d, "shape.txt")))
  i <- readBin(file.path(d, "x.i.bin"), "integer", n = sh[3], size = 4)
  p <- readBin(file.path(d, "x.p.bin"), "double", n = sh[2] + 1, size = 8)
  x <- readBin(file.path(d, "x.x.bin"), "double", n = sh[3], size = 8)
  peaks <- readLines(file.path(d, "peaks.txt")); cells <- readLines(file.path(d, "cells.txt"))
  m <- sparseMatrix(i = i + 1L, p = as.integer(p), x = x, dims = sh[1:2], dimnames = list(peaks, cells))
  co <- read.csv(file.path(d, "coords.csv"), row.names = 1, colClasses = c("character", "numeric", "numeric"))
  g <- read.table(file.path(d, "genome.tsv"), sep = "\t", stringsAsFactors = FALSE)
  list(m = m, coords = as.matrix(co[cells, ]), genome = g)
}
t0 <- tic(); F <- read_fixture(fx); rec("read_fixture", t0)
pk <- cicero:::split_peak_names(rownames(F$m))
peakinfo <- data.frame(site_name = rownames(F$m), chr = pk[, 1], bp1 = as.numeric(pk[, 2]),
                       bp2 = as.numeric(pk[, 3]), row.names = rownames(F$m))
cellinfo <- data.frame(cells = colnames(F$m), row.names = colnames(F$m))
input_cds <- suppressWarnings(new_cell_data_set(F$m, cell_metadata = cellinfo, gene_metadata = peakinfo))
input_cds <- detect_genes(input_cds)
input_cds <- estimate_size_factors(input_cds)
lg("input ", nrow(input_cds), " peaks x ", ncol(input_cds), " cells; genome ", paste(F$genome$V1, collapse = ","))
writeLines(c(R.version.string, paste("cicero", packageVersion("cicero")), paste("monocle3", packageVersion("monocle3")),
             paste("glasso", packageVersion("glasso")), paste("FNN", packageVersion("FNN")),
             paste("igraph", packageVersion("igraph")), paste("seed", seed), paste("mode", mode)),
           file.path(out, "versions.txt"))

# ---------------------------------------------------------------- kNN (gate 2)
k <- 50
if (mode == "full") {
  t0 <- tic(); nn <- FNN::knn.index(as.data.frame(F$coords), k = k - 1); rec("knn_index", t0)
  wi(t(nn), "knn_index.i32")                       # row-major n_cells x (k-1), 1-based
}

# ---------------------------------------------------------------- make_cicero_cds (gates 1, 3)
set.seed(seed)
t0 <- tic()
res <- make_cicero_cds(input_cds, reduced_coordinates = F$coords, k = k, return_agg_info = TRUE)
rec("make_cicero_cds", t0)
cicero_cds <- res[[1]]; agg_map <- res[[2]]
writeLines(colnames(cicero_cds), file.path(out, "agg_cells.txt"))   # chosen bins, in R's order
fwrite(agg_map, file.path(out, "agg_map.tsv"), sep = "\t")
cm <- as(exprs(cicero_cds), "CsparseMatrix")                          # peaks x bins, normalised
wi(cm@i, "cicero_x.i.i32"); wd(cm@p, "cicero_x.p.f64"); wd(cm@x, "cicero_x.x.f64")
writeLines(as.character(c(nrow(cm), ncol(cm), length(cm@x))), file.path(out, "cicero_x.shape.txt"))
wd(pData(cicero_cds)$Size_Factor, "cicero_size_factor_final.f64")
wd(pData(cicero_cds)$num_genes_expressed, "cicero_num_genes_expressed.f64")

# ---------------------------------------------------------------- run_cicero, step by step
gc_df <- F$genome
set.seed(seed)
t0 <- tic()
dps <- estimate_distance_parameter(cicero_cds, window = 500000, maxit = 100, sample_num = 100,
                                   distance_constraint = 250000, distance_parameter_convergence = 1e-22,
                                   genomic_coords = gc_df)
rec("estimate_distance_parameter", t0)
mean_dp <- mean(unlist(dps)); wd(unlist(dps), "distance_parameters.f64"); wd(mean_dp, "mean_distance_parameter.f64")
lg("mean distance parameter ", sprintf("%.17g", mean_dp), " from ", length(dps), " windows")
t0 <- tic()
models <- generate_cicero_models(cicero_cds, distance_parameter = mean_dp, window = 500000, genomic_coords = gc_df)
rec("generate_cicero_models", t0)
t0 <- tic(); conns <- assemble_connections(models, silent = FALSE); rec("assemble_connections", t0)
timings[["run_cicero_steps_total"]] <- timings$estimate_distance_parameter + timings$generate_cicero_models +
  timings$assemble_connections
st <- vapply(models, function(m) if (is.character(m)) m else "glasso", "")
ni <- vapply(models, function(m) if (is.list(m)) as.numeric(m$niter) else NA_real_, 0)
np_ <- vapply(models, function(m) if (is.list(m)) nrow(m$w) else NA_real_, 0)
fwrite(data.table(window = names(models), status = st, niter = ni, n_peaks = np_), file.path(out, "models.tsv"), sep = "\t")
writeLines(as.character(conns$Peak1), file.path(out, "conns_peak1.txt"))
writeLines(as.character(conns$Peak2), file.path(out, "conns_peak2.txt"))
wd(conns$coaccess, "conns_coaccess.f64")
lg(nrow(conns), " connection rows; ", sum(is.na(conns$coaccess)), " NA")

if (mode == "full") {
  # ---- per-sampled-window distance parameters, replaying estimate_distance_parameter's loop
  #      with Cicero's own internals (gate 5); must reproduce `dps` exactly.
  fd <- fData(cicero_cds)
  cds2 <- cicero_cds
  fData(cds2)$chr <- gsub("chr", "", fData(cds2)$chr)
  fData(cds2)$bp1 <- as.numeric(as.character(fData(cds2)$bp1))
  fData(cds2)$bp2 <- as.numeric(as.character(fData(cds2)$bp2))
  grs <- cicero:::generate_windows(500000, gc_df)
  set.seed(seed)
  it <- 0; calced <- 0; rows <- list()
  while (100 > calced & it < 500) {
    it <- it + 1
    win <- sample(seq_len(length(grs)), 1)
    wr <- cicero:::get_genomic_range(grs, cds2, win)
    n <- nrow(exprs(wr)); dp <- NA_real_; status <- "ok"
    if (n <= 1) status <- "le1" else if (n > 200) status <- "gt_max" else {
      dm <- cicero:::calc_dist_matrix(wr)
      r <- cicero:::find_distance_parameter(dm, wr, maxit = 100, null_rho = 0, 0.75,
                                            distance_constraint = 250000, distance_parameter_convergence = 1e-22)
      if (is.numeric(r)) { dp <- r; calced <- calced + 1 } else status <- r
    }
    rows[[it]] <- data.table(it = it, win = win, n_peaks = n, status = status, dp = sprintf("%.17g", dp))
  }
  samp <- rbindlist(rows); fwrite(samp, file.path(out, "sampled_windows.tsv"), sep = "\t")
  ok <- samp$status == "ok"
  stopifnot(identical(as.numeric(samp$dp[ok]), as.numeric(sprintf("%.17g", unlist(dps)))))
  lg("replayed ", it, " sampled windows; matches estimate_distance_parameter")

  # ---- glasso probes (gate 4): same S and rho as Cicero builds them, R glasso outputs
  gl <- which(st == "glasso"); probes <- unique(gl[unique(round(seq(1, length(gl), length.out = min(40, length(gl)))))])
  pr <- list(); off <- 0
  con_s <- file(file.path(out, "probe_S.f64"), "wb"); con_r <- file(file.path(out, "probe_rho.f64"), "wb")
  con_w <- file(file.path(out, "probe_w.f64"), "wb"); con_wi <- file(file.path(out, "probe_wi.f64"), "wb")
  for (w in probes) {
    wr <- cicero:::get_genomic_range(grs, cds2, w)
    dm <- cicero:::calc_dist_matrix(wr)
    rho <- cicero:::get_rho_mat(dm, mean_dp, 0.75)
    vals <- as.matrix(exprs(wr)); S <- cov(t(vals)); diag(S) <- diag(S) + 1e-4
    G <- glasso::glasso(S, rho)
    stopifnot(isTRUE(all.equal(unname(G$w), unname(models[[w]]$w), tolerance = 0)))
    writeBin(as.double(S), con_s); writeBin(as.double(rho), con_r)
    writeBin(as.double(G$w), con_w); writeBin(as.double(G$wi), con_wi)
    pr[[length(pr) + 1]] <- data.table(window = names(models)[w], n = nrow(S), offset = off, niter = G$niter)
    off <- off + nrow(S)^2
  }
  close(con_s); close(con_r); close(con_w); close(con_wi)
  fwrite(rbindlist(pr), file.path(out, "probes.tsv"), sep = "\t")
  # synthetic edge cases: block-diagonal S (connect() components), singleton, zero off-diagonal
  set.seed(1)
  syn <- list()
  X <- matrix(rnorm(40 * 12), 40); S1 <- cov(X); S1[1:6, 7:12] <- S1[7:12, 1:6] <- 0.001; R1 <- matrix(0.05, 12, 12)
  syn[[1]] <- list(S = S1, rho = R1)
  S2 <- diag(c(1, 2, 3, 4)); syn[[2]] <- list(S = S2, rho = matrix(0.1, 4, 4))
  X3 <- matrix(rnorm(30 * 8), 30); S3 <- cov(X3); R3 <- matrix(0.02, 8, 8); R3[1, -1] <- R3[-1, 1] <- 10
  syn[[3]] <- list(S = S3, rho = R3)
  X4 <- matrix(rpois(60 * 25, 2), 60); S4 <- cov(X4); diag(S4) <- diag(S4) + 1e-4
  D4 <- as.matrix(dist(sort(runif(25, 0, 5e5)))); syn[[4]] <- list(S = S4, rho = cicero:::get_rho_mat(D4, 0.5, 0.75))
  sy <- list(); off <- 0
  con_s <- file(file.path(out, "syn_S.f64"), "wb"); con_r <- file(file.path(out, "syn_rho.f64"), "wb")
  con_w <- file(file.path(out, "syn_w.f64"), "wb"); con_wi <- file(file.path(out, "syn_wi.f64"), "wb")
  for (j in seq_along(syn)) {
    G <- glasso::glasso(syn[[j]]$S, syn[[j]]$rho)
    writeBin(as.double(syn[[j]]$S), con_s); writeBin(as.double(syn[[j]]$rho), con_r)
    writeBin(as.double(G$w), con_w); writeBin(as.double(G$wi), con_wi)
    sy[[j]] <- data.table(case = j, n = nrow(syn[[j]]$S), offset = off, niter = G$niter); off <- off + nrow(syn[[j]]$S)^2
  }
  close(con_s); close(con_r); close(con_w); close(con_wi)
  fwrite(rbindlist(sy), file.path(out, "synthetic.tsv"), sep = "\t")

  # ---- run_cicero() itself, same seed: must give the identical table
  set.seed(seed)
  t0 <- tic(); conns_rc <- run_cicero(cicero_cds, gc_df, sample_num = 100); rec("run_cicero", t0)
  stopifnot(identical(conns_rc$Peak1, conns$Peak1), identical(conns_rc$Peak2, conns$Peak2),
            identical(conns_rc$coaccess, conns$coaccess))
  lg("run_cicero() identical to the step-by-step run")
}

# ---------------------------------------------------------------- CCANs (gates 16, 17)
if (length(ccan_seeds)) {
  cc_rows <- list(); memb <- list()
  for (s in ccan_seeds) {
    set.seed(s); t0 <- tic()
    cut <- cicero:::find_ccan_cutoff(conns, 2)
    ccan <- generate_ccans(conns, coaccess_cutoff_override = cut)
    rec(paste0("generate_ccans_seed", s), t0)
    cc_rows[[length(cc_rows) + 1]] <- data.table(seed = s, cutoff = cut, n_ccan = length(unique(ccan$CCAN)), n_peaks = nrow(ccan))
    fwrite(ccan, file.path(out, paste0("ccan_seed", s, ".tsv")), sep = "\t")
  }
  cc <- rbindlist(cc_rows); fwrite(cc, file.path(out, "ccan_cutoffs.tsv"), sep = "\t")
  # all seeds at the first seed's cutoff (membership noise floor at a fixed cutoff)
  for (s in ccan_seeds) {
    set.seed(s); ccan <- generate_ccans(conns, coaccess_cutoff_override = cc$cutoff[1])
    fwrite(ccan, file.path(out, paste0("ccan_fixedcut_seed", s, ".tsv")), sep = "\t")
  }
}

# ---------------------------------------------------------------- gene activity (gates 14, 15)
if (file.exists(file.path(fx, "promoters.tsv"))) {
  ann <- read.delim(file.path(fx, "promoters.tsv"), stringsAsFactors = FALSE)
  t0 <- tic()
  input_cds <- annotate_cds_by_site(input_cds, ann)
  unnorm_ga <- build_gene_activity_matrix(input_cds, conns)
  rec("build_gene_activity_matrix", t0)
  unnorm_ga <- unnorm_ga[!Matrix::rowSums(unnorm_ga) == 0, !Matrix::colSums(unnorm_ga) == 0]
  num_genes <- pData(input_cds)$num_genes_expressed; names(num_genes) <- row.names(pData(input_cds))
  t0 <- tic(); ga <- normalize_gene_activities(unnorm_ga, num_genes); rec("normalize_gene_activities", t0)
  fwrite(data.table(peak = rownames(fData(input_cds)), gene = fData(input_cds)$gene,
                    overlap = fData(input_cds)$overlap), file.path(out, "annotated_sites.tsv"), sep = "\t", na = "NA")
  for (nm in c("unnorm_ga", "ga")) {
    M <- as.matrix(get(nm))
    writeLines(rownames(M), file.path(out, paste0(nm, "_genes.txt"))); writeLines(colnames(M), file.path(out, paste0(nm, "_cells.txt")))
    wd(M, paste0(nm, ".f64"))                         # column-major genes x cells
  }
}

fwrite(data.table(step = names(timings), seconds = unlist(timings)), file.path(out, "timings.tsv"), sep = "\t")
lg("done")
