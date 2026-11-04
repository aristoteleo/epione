# Parity manifest — `epi.tl.cnv(method="epianeufinder")` vs R epiAneufinder 1.1.5

Pre-registered before the first comparison. The thresholds below are read-only:
if a gate fails, the cause is diagnosed and reported, and the threshold is not changed.

## Reference

* R package: epiAneufinder 1.1.5 (github.com/colomemaria/epiAneufinder, commit
  `47293f7`, GPL-3), R 4.4, BSgenome.Hsapiens.UCSC.hg38, installed in a conda
  env outside this repository.
* Call: `epiAneufinder(input=<fragments.tsv>, blacklist="hg38-blacklist.v2.bed",
  windowSize=1e5, genome="BSgenome.Hsapiens.UCSC.hg38", exclude=c("chrX","chrY","chrM"),
  test="AD", k=4, minsize=1, minsizeCNV=0, uq=0.9, lq=0.1, minFrags=20000,
  threshold_cells_nbins=0.05, threshold_blacklist_bins=0.85, gc_correction=TRUE)`,
  followed by `split_subclones(res, tree_depth=k)` and `karyotype_measures(res)`.
  A small driver also dumps the intermediate objects (window table, raw count matrix,
  GC-corrected matrix, per-cell per-chromosome `getbp` output, pruned breakpoints).
* Python: `epi.tl.cnv(...)` with the same parameters and the UCSC `hg38.fa`.
* Datasets
  * **D1** `sample_data/sample.tsv` shipped with epiAneufinder (SNU601 subset, 1.68 M
    fragments).
  * **D2** SNU601 gastric cancer cell line 10x scATAC (SRA PRJNA674903, the dataset of
    the epiAneufinder paper), fragments made from the submitted `possorted_bam.bam`.
    R run on the full cell set if it finishes in the allotted time, otherwise on a
    fixed random subset of cells (seed and size recorded in the results).
  * **D3** `test="KS"` on D1 (non-default test statistic).

## Gates

| # | Output | Data | Class (rebuildr) | Metric | Threshold |
|---|---|---|---|---|---|
| G0a | windows kept (chrom, start, end) after blacklist + N filter, and bins/cells kept after the zero filters | D1, D2 | exact | set equality | identical |
| G0b | GC / AT / N fraction per window | D1, D2 | deterministic-standard | max abs err | 1e-12 |
| G0c | raw count matrix (windows × cells) | D1, D2 | exact | equality | identical |
| G0d | GC-corrected integer counts | D1, D2 | deterministic (integer after rounding) | fraction of equal entries; max abs diff | ≥ 0.9999; ≤ 1 |
| G1a | `getbp` breakpoints per cell × chromosome (before pruning), given R's GC-corrected counts as input | D1, D2 | exact | equality of the ordered bp vectors | identical in ≥ 99.9 % of cell×chrom pairs |
| G1b | AD distances of those breakpoints | D1, D2 | deterministic-standard | max rel err | 1e-10 |
| G1c | pruned breakpoints per cell, end-to-end | D1, D2 | ranked / set | mean per-cell Jaccard with ±1-bin positional tolerance | ≥ 0.99 |
| G2a | per-cell per-bin CNV state (0/1/2), end-to-end | D1, D2 | classification | fraction of agreeing calls | ≥ 0.99 |
| G2b | per-cell per-bin CNV state given R's GC-corrected counts | D1, D2 | classification | fraction of agreeing calls | ≥ 0.999 |
| G2c | per-cell per-bin CNV state, KS test | D3 | classification | fraction of agreeing calls | ≥ 0.99 |
| G3a | subclones `split_subclones` (ward.D, cutree k = 2..6) | D1, D2 | clustering | ARI per k | ≥ 0.95 |
| G3b | karyotype measures (genome-wide and per-chromosome aneuploidy / heterogeneity) | D1, D2 | deterministic-standard | max abs err | 1e-6 |

Notes fixed in advance:

* G1a/G1b/G2b isolate the segmentation and state assignment from GC correction, which is
  done by `stats::loess` in R and by scikit-misc's `loess` (the same netlib `dloess`
  code) in Python; G0d measures that step on its own.
* G0d/G2a allow for integer rounding of `x * mean(x) / fit` flipping at a .5 boundary
  when the two loess implementations differ in the last bits.
* Positions are bin indices within the chromosome after filtering, as in R.
* Performance: wall time of the full pipeline (R `ncores` = Python `n_jobs`, same node,
  same thread count) is reported next to the gates; it is not a gate.

## Results (appended after the run; thresholds above unchanged)

All gates pass on all three datasets. D2 ran on the full cell set (3,160 cells with
≥ 20,000 fragments, the same set in R and Python, in the same column order).

| Gate | Output | Threshold | D1 sample (AD) | D2 SNU601 (AD) | D3 sample (KS) | Pass |
|---|---|---|---|---|---|---|
| G0a | windows, cells after filters | identical | identical (16 cells × 24558 windows) | identical (3160 cells × 26003 windows) | identical (16 cells × 24558 windows) | yes |
| G0b | GC / AT / N per window | max abs ≤ 1e-12 | 1.11e-16 | 1.11e-16 | 1.11e-16 | yes |
| G0c | raw counts | identical | identical | identical | identical | yes |
| G0d | GC-corrected counts | equal ≥ 0.9999, max diff ≤ 1 | 1.0 equal, max diff 0 | 1.0 equal, max diff 0 | 1.0 equal, max diff 0 | yes |
| G1a | `getbp` breakpoints per cell × chromosome (R input) | identical in ≥ 99.9 % | 1.0 of 352 | 1.0 of 69520 | 1.0 of 352 | yes |
| G1b | breakpoint distances | max rel ≤ 1e-10 | 5.00e-16 | 1.19e-15 | 2.46e-16 | yes |
| G1c | pruned breakpoints, end-to-end | Jaccard (±1 bin) ≥ 0.99 | 1.0 (exact positions 1.0) | 1.0 (exact positions 1.0) | 1.0 (exact positions 1.0) | yes |
| G2a | CNV state per cell × window, end-to-end | agreement ≥ 0.99 | 1.0 | 1.0 | 1.0 | yes |
| G2b | CNV state given R's GC-corrected counts | agreement ≥ 0.999 | 1.0 | 1.0 | 1.0 | yes |
| G3a | subclones (`split_subclones`) | ARI ≥ 0.95 | 1.0 (k = 2–6) | 1.0 (k = 2–6) | 1.0 (k = 2–6) | yes |
| G3b | karyotype measures | max abs ≤ 1e-6 | 5.00e-16 | 5.00e-16 | 9.75e-17 | yes |

Notes

* Windows, counts, GC-corrected counts, breakpoints and states are identical, not just within
  tolerance. The breakpoint distances agree to ~1e-15 relative (R exports them with 17
  significant digits).
* D3 (KS) first failed G1a (0.92) during development: `stats::ks.test` builds its statistic
  with R's `cumsum`, which accumulates in long double, so exact ties between candidate splits
  were broken differently. The port now accumulates in long double as R does; the threshold
  was not changed.
* Fragments for D2 were made from the submitted `possorted_bam.bam` of SRR12995620 with
  pysam (proper pairs, MAPQ ≥ 30, `CB` tag, Tn5 shift +4/−5, 10–5000 bp, duplicates
  collapsed); R and Python read the same file.

### Runtime (same node, SNU601, 3,160 cells × 26,003 windows)

| Step | R epiAneufinder 1.1.5 | `epi.tl.cnv` |
|---|---|---|
| windows + counting | 324 s (1 core) | 249 s (1 core) |
| GC correction | 393 s (24 cores) | included below |
| GC correction + segmentation + states | 3.8 h AD on 12 cores + 259 s pruning/states | 7.6 s (24 workers); 110 s (1 worker) |
| one cell, 1 core | 38.9 s (AD segmentation only) | 0.035 s (GC correction + segmentation + states) |
| subclones k = 2..6 + karyotype measures | 5,307 s | 11.8 s |
| **total to the CNV table** | **≈ 4.0 h** (mixed 1/24/12 cores) | **264 s** (24 workers), **381 s** (1 worker) |

The R run was split in two because forked `mclapply` workers of the first attempt were
OOM-killed on the shared allocation (R holds the whole fragments table in the parent
process); the second attempt resumed from R's own `counts_gc_corrected.rds`
(`reuse.existing=TRUE`) with 12 workers. The node was shared with other jobs, so R's workers
ran at ~75 % CPU; the single-cell timing (38.9 s vs. 0.035 s) is the like-for-like
comparison. R's karyogram (`ggsave` of 82 M tiles) took a further 24 min and is not counted.
