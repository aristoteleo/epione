# MATH.md — why the Cicero port reproduces R, and where it would not

No bounded (B) approximation is part of the port: every accepted rewrite is exact (E). This file
records the arguments behind "exact" and the one (B) candidate that was measured and rejected.

## 1. Random numbers

R draws with `sample()` (aggregation bins, sampled windows) and, inside `igraph::cluster_louvain`,
with `unif_rand()` through R igraph's RNG bridge.

* `RRandom(seed)` (epione.tl._rrng) is R's Mersenne-Twister with R's seed scrambling, and
  `unif_index(n)` is R ≥ 3.6's rejection sampler (`R_unif_index`, `sample.kind = "Rejection"`);
  `sample(seq_len(n), 1)` is `unif_index(n) + 1`, including the single draw R spends when `n == 1`.
* igraph (C core 0.10, vendored in R igraph 2.1.4) asks the R bridge for 30-bit words
  `(uint) (unif_rand() * 2^30)`, builds 32-bit words as `(g1 << 2) + (g2 >> 28)`, and maps them to
  `[0, range)` with Lemire's multiply-and-reject (`t = (2^32 - range) mod range`). `RNG_INTEGER(l, l)`
  draws nothing. Ported verbatim in `_louvain_r.IgraphRRng`.

Equal streams + the same consumption order ⇒ the same draws; gate 1 and the CCAN tables confirm it.

## 2. Floating point

R's numerics are reproduced operation by operation where the result could change a decision or a
reported number:

| R code | R arithmetic | Port |
|---|---|---|
| `mean()` (`summary.c`) | long-double sum / n, then long-double correction pass | `np.longdouble` `cumsum` (strictly sequential) |
| `cov()` (`cov.c`, `cov_na_1`) | two-pass long-double means; products in double; long-double running sums | means in `np.longdouble`; sums as compiled x87 `x86_fp80` additions (iter 6) |
| `glasso.f` | Fortran double, `-O2`, no FMA on x86-64 baseline | numba, same loop and accumulation order, no `fastmath` |
| `(1 - (1000/d)^s) * dp` | `R_pow` → libm `pow` | numba `math.pow` → the same libm `pow` |
| `stats::cov2cor` | `Is[i] * V[i,j] * Is[j]`, `Is = sqrt(1/diag)` | same order |
| `round(x, d)` (`fround.c`) | nearer of floor/ceil(x·10^d)/10^d, odd-floor ties | `_r_round` (not Python's `round`) |
| Louvain sums | order of `igraph_incident` + `igraph_qsort` on equal keys | same edge store, qsort ported |

**x87 emulation (iter 6).** On x86-64, `long double` is the 80-bit x87 format (64-bit significand);
GCC compiles `sum += prod` (prod a double) to `fld`/`fadd` with one rounding to 64 bits. The numba
intrinsic emits LLVM `fpext double → x86_fp80`, `fadd x86_fp80`, `fptrunc → double`, i.e. the same
instructions. The running sum `S` (≤ 64 significant bits) is kept as `hi = fl53(S)` and
`lo = fl53(S − hi)`: `S − hi` has at most 64 − 53 + 1 significant bits and is computed exactly in
x87, so `hi + lo = S` exactly and no information is lost between steps. Bit-identity with the
numpy long-double path is asserted in `tests/test_cicero.py`. On other platforms numpy's
`longdouble` *is* R's `LDOUBLE` (e.g. binary128 on aarch64), so that path is used there.

**What is not bit-identical.** glasso's iterations depend continuously on the covariance, and the
loop order is mirrored, but the compiled Fortran (gfortran intrinsics such as `matmul` and `sum`,
compiler scheduling) is not guaranteed to associate every sum the same way as the numba code. The
remaining differences — ≤ 8e-16 on coaccess, ≤ 5.4e-9 absolute (3.4e-11 relative) on `wi` entries
up to 5e3 in the F1 probe windows (25 bins, nearly singular covariances) — are far below every
gate; iteration counts were identical in all 88 probe windows. Distance parameters are the
end point of a bisection that runs to the double-precision resolution of the 0/non-zero boundary
of `wi`, so per-window values agree to ≤ 3e-15 relative and the mean was bit-identical on F1 and F2.

## 3. Rejected (B) candidate — BLAS covariance (iter 5)

Replacing the long-double covariance by `D'D / (n − 1)` in double:

|S_blas − S_R|_ij ≤ (γ_n^(53) + γ_n^(64)) · Σ_k |d_ki d_kj|,  γ_n^(p) = n u_p / (1 − n u_p),  u_p = 2^−p

(standard recursive-summation bound; BLAS blocking only lowers it). For F2, n = 3235 bins:
γ ≈ 3.6e-13 relative to Σ|d d|. Measured effect: coaccess max abs error 4.4e-16 → 6.9e-15, mean
distance parameter relative error 0 → 5.5e-16, all gates still passing, `run_cicero` 3.7× faster.
Rejected because the brief admits only provably equivalent rewrites; iteration 6 recovers most of
the speed exactly.

## 4. Locale

R sorts character vectors (factor levels, `sort()`, `reshape2::dcast`, `plyr::ddply` groups) with
the session's collation; the reference ran in `en_US.UTF-8` (glibc ≥ 2.28): punctuation (in the order
`_-,;:!?.'"()[]{}@*/\&#%`^+<=>|~$`) < digits < letters, case-insensitive, lowercase first on ties.
`_r_sorted()` implements that order for ASCII names. data.table (`assemble_connections`) and R's
radix `order()` on several keys use byte order, which the port keeps there. An R session in the C
locale would order some names (gene levels, merged peaks, `aggregate_by_cell_bin` rows) differently.
