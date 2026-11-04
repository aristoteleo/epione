# Single-cell Hi-C contact decay: scHiCluster and HCEA parity (pre-registered)

Written before any comparison was run. Thresholds are fixed here; later changes are dated
amendments below the original text, written before the rerun.

## References (pinned)

| Reference | Version | Role |
|---|---|---|
| scHiCluster | ChrisMao0325/scHiCluster `c958838` | `cool.contact_distance.compute_decay` (`hicluster contact-distance`) |
| HumanCellEpigenomeAtlas | `fig4/01.decay.ipynb`, `fig4/02.decay_compartment.ipynb` | short/long ratio; distance × compartment-score histograms |
| HCEA published objects | ENTEx `analysis/decay/cell_*_decay.hdf5` (86,689 cells × 132 bins), metadata column `Short/Long` (`5kCG100k3C_summary.csv.gz`), `analysis/compartment/L1/comp.impute.qnorm.hdf` | reference values |
| qnorm | 1.0.0 (PyPI) | quantile normalisation (HCEA used an earlier release) |

Python env `epione-hic` for both sides.

## Fixtures

- **H** — the 8,040 local HCEA contact tables (`Hic_data/HCEA/sc_contact`), all contained in
  the published decay tables. **H200** — the 200-cell subset of the compartment manifest.
- **E** — HCEA's `comp.impute.hdf` / `comp.impute.qnorm.hdf` (35 major types, rows with
  `raw_binfilter`).

## Gates

| Gate | Epione | Reference | Fixture | Class | Threshold |
|---|---|---|---|---|---|
| T1 | `contact_decay` (decay histogram + chromosome sparsity) | scHiCluster `compute_decay` | H200 | exact | identical integer tables |
| T2 | `contact_decay` decay histogram | published `cell_*_decay.hdf5` rows | H (8,040 cells) | exact | identical integer tables |
| T3 | `decay_ratio` | published metadata `Short/Long` | H (8,040 cells) | 1, deterministic-strict | max abs err ≤ 1e-12 |
| T4 | `quantile_normalize` | published `comp.impute.qnorm.hdf` from `comp.impute.hdf` | E | 1, deterministic-strict | max abs err ≤ 1e-10 (summation order of an older qnorm may differ) |
| T4b | `quantile_normalize` | `qnorm.quantile_normalize` 1.0.0 | E, plus a matrix with ties | 1, deterministic-strict | max abs err ≤ 1e-13 |
| T5 | `compartment_contact_hist` | `contact_hist` of `02.decay_compartment` (cell 11), verbatim | H200 with the cell's L1 column of `comp.impute.qnorm.hdf` | exact | identical histograms (distance × diff, distance × sum) summed per L1 |

Speed is reported for T1 and T5, Epione vs reference.

## Amendment T-1 (2026-10-20, after the first T2/T3 run, before any rerun)

First run: T1, T4, T4b, T5 passed; T2 failed (6,480 of 8,040 cells differ) and T3 failed (max
err 0.011), while `decay_ratio` applied to the *published* histograms reproduces `Short/Long`
to 4.4e-16. Cause, found on 40 cells: the published histograms count cis contacts on
chr1–chr22 and chrX only (no chrY); with that chromosome set all 40 cells are identical, with
chrY included none is, with chrX also removed none is. The contact files are byte-identical to
ENTEx's `contact/` copies. Change: T2/T3 pass `chrom_sizes` = chr1–22 + chrX (sizes from
`hg38.main.chrom.sizes`; the largest chromosome, which sets the bins, is unchanged). Gates,
fixture and thresholds unchanged.

## Results (T), 2026-10-20

Raw numbers in `results/decay_T.json` (rerun after Amendment T-1).

| Gate | Measured | Threshold | Pass |
|---|---|---|---|
| T1 decay + sparsity vs scHiCluster (H200) | identical (132 bins); Epione 6.1 s on 8 processes, reference 43.9 s on 1 | exact | ✅ |
| T2 decay vs published (8,040 cells) | 0 cells differ | exact | ✅ |
| T3 `decay_ratio` vs published `Short/Long` | max abs err 1.3e-15 | ≤ 1e-12 | ✅ |
| T4 `quantile_normalize` vs published `comp.impute.qnorm.hdf` | max abs err 1.1e-14 | ≤ 1e-10 | ✅ |
| T4b vs qnorm 1.0.0 (HCEA scores + a matrix with ties) | 0.0 | ≤ 1e-13 | ✅ |
| T5 `compartment_contact_hist` vs notebook `contact_hist` | identical for c0 / c1 / c7 (24.1 M contacts); 5.6 s vs 10.3 s on 8 processes | exact | ✅ |
