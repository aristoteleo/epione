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
