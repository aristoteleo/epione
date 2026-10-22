# scHiCluster cell embedding at 100 kb: parity (pre-registered)

Written before any comparison was run. Thresholds are fixed here; later changes are dated
amendments below the original text, written before the rerun.

## Reference

scHiCluster `schicluster.embedding.calc_embedding.embedding` (ChrisMao0325/scHiCluster `c958838`,
CLI `hicluster embedding`; defaults `dim=50`, `dist=1,000,000`, `resolution=100,000`,
`scale_factor=100,000`, `norm_sig=False`). Per chromosome it takes the imputed contacts within
`dist` of the diagonal, scales them, runs a 50-component truncated SVD (ARPACK), concatenates the
chromosomes and runs a final truncated SVD.

## Fixture

**H200i**: the 200 HCEA cells of the compartment manifest (fixture H200), imputed at 100 kb with
`epi.single.hic.impute_cells` (scHiCluster `impute_chromosome`, `output_dist=10,000,000`), on
chr1–chr22. The reference reads the same imputed values from per-cell cools written from those
matrices (upper triangle).

## Gates

| Gate | Epione | Reference | Class | Threshold |
|---|---|---|---|---|
| E1 | per-chromosome band matrices (`embedding(method="schicluster")`, before SVD) | `make_chrom_matrix` | exact | identical arrays |
| E2 | final embedding | `embedding()` final decomposition | 2, deterministic-bounded (ARPACK start vector; component sign free) | after per-component sign alignment, max abs err ≤ 1e-6 × max abs value |

E1 also checks that imputation itself matches: the per-cell `.npz` written by `impute_cells`
equals scHiCluster's `impute_chromosome` output for the same cell and chromosome (exact).

## Amendment E-1 (2026-10-21, after the first E2 run, before any rerun)

First run: E1 passed (band matrices identical; imputation identical). E2 failed: max abs err
4.2e-5 × max abs value, above 1e-6. Diagnosis: the reference is not reproducible to 1e-6 itself.
Its matrices are float32 and `TruncatedSVD(algorithm="arpack")` starts from a random vector
(`random_state=None`, the global numpy RNG). Running scHiCluster's own SVD steps on the same band
matrices with global seeds 1, 2 and 3 gives max errors of 2.9e-5 to 3.7e-5 between runs; the
largest per-component differences are in components 31–44, whose singular values lie within 0.2 %
of a neighbour's. The 1e-6 threshold assumed a determinism the reference does not have.

Change: E2 passes when (a) the epione-vs-reference error is at most 2 × the largest error between
three reference runs with global seeds 1, 2, 3 (measured in the same driver), and (b) every
component's absolute Pearson correlation with the reference is ≥ 0.9999. E1 and the fixture are
unchanged.

## Results (E), 2026-10-21

Raw numbers in `results/embedding_E.json` (rerun after Amendment E-1).

| Gate | Measured | Threshold | Pass |
|---|---|---|---|
| E1 band matrices (22 chromosomes, 200 cells) | identical; imputation of the test cell identical to `impute_chromosome` | exact | ✅ |
| E2 final embedding (200 × 50) | max abs err 3.2e-5 × max value; reference vs itself 5.2e-5; min component \|r\| 0.99999999 | ≤ 2 × reference self-error, \|r\| ≥ 0.9999 | ✅ |

Speed: imputation of 200 cells × 22 chromosomes at 100 kb, 430 s on 16 processes. Embedding 70 s
(epione, one process) vs 16 s (scHiCluster, 16 processes).

## Amendment E-2 (2026-10-21, before any norm_sig comparison): gate E3

The lab's standard call is `hicluster embedding --dim 50 --dist 1000000 --resolution 100000
--scale_factor 100000 --norm_sig`. E2 tested only `norm_sig=False` (the CLI default). New gate,
same fixture (H200i) and same driver:

| Gate | Epione | Reference | Class | Threshold |
|---|---|---|---|---|
| E3 | `embedding(method="schicluster", norm_sig=True)` | `embedding(..., norm_sig=True)` final decomposition | 2, deterministic-bounded | as amended E2: max abs err (after per-component sign alignment) ≤ 2 × the largest error between three reference runs with global seeds 1, 2, 3 (`norm_sig=True`), and every component's \|r\| ≥ 0.9999 |

## Results (E, rerun with E3), 2026-10-21

Raw numbers in `results/embedding_E.json` (overwrites the first E results; E1 and E2 rerun in
the same driver).

| Gate | Measured | Threshold | Pass |
|---|---|---|---|
| E1 band matrices | identical; imputation of the test cell identical | exact | ✅ |
| E2 `norm_sig=False` | max abs err 8.7e-5 × max value (first run: 3.2e-5); reference vs itself 5.2e-5; min \|r\| 0.99999995 | ≤ 2 × reference self-error, \|r\| ≥ 0.9999 | ✅ |
| E3 `norm_sig=True` | max abs err 1.5e-4 × max value; reference vs itself 1.1e-4; min \|r\| 0.99999995 | ≤ 2 × reference self-error, \|r\| ≥ 0.9999 | ✅ |

E2 moves between runs because the reference's ARPACK start vector is random (`random_state=None`);
both runs are inside the bound.

## Amendment E-3 (2026-10-22): `accelerate=` in `impute_cells`

`impute_cells` gained `accelerate=True` (default), a faster implementation of `impute_chromosome`
that equals scHiCluster up to float32 rounding (max |diff| ~1e-7, `tests/test_single_hic_impute_fast.py`).
E1 checks that `impute_cells` writes scHiCluster's own output, so `e_gates.py` now passes
`accelerate=False`; the gates and recorded results are otherwise unchanged.
