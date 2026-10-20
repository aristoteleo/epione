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
