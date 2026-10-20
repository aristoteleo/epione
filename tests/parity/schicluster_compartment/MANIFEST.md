# Single-cell Hi-C compartments: scHiCluster, HCEA and dcHiC parity (pre-registered)

Written before any comparison was run. Gate classes follow the omicverse-rebuildr parity
taxonomy. Thresholds are fixed here and are not changed after results are seen; any later
change is recorded as a dated amendment below the original text, with its reason, before the
rerun.

## References (pinned)

| Reference | Version | Licence | Role |
|---|---|---|---|
| scHiCluster | ChrisMao0325/scHiCluster `c958838` (fork of zhoujt1994/scHiCluster, 1.3.5.dev36) | MIT | `get_cpg_profile`, `multiple_cell_compartment`, `compartment_strength`, `merge_cell_raw`, scool pixel builder |
| HumanCellEpigenomeAtlas | `fig4/04.compartment.ipynb` (Zhou, Wu, Liu et al.) | see repo | pseudobulk compartment model (fit on merged raw contacts, project each major type), saddle |
| dcHiC | ay-lab/dcHiC `144b4df`, `dchicf.r --pcatype analyze` | MIT | differential compartments |
| R / robust / rrcov / limma | R 4.1.3, robust 0.7.1, rrcov 1.7-3, limma 3.50.3 (mamba env `dchic`) | GPL | dcHiC's statistics |
| bedtools | 2.31.1 | MIT | `bedtools nuc` behind `get_cpg_profile` |

References run in the dev env `epione-hic` (Python) and the `dchic` env (R). The Epione side
runs in `epione-hic` for every gate.

## Fixtures

- **H** — 8,040 HCEA snm3C-seq cells, per-cell contact tables
  (`Hic_data/HCEA/sc_contact/*/*.3C.contact.rmbkl.tsv.gz`, columns chrom1 = 1, pos1 = 2,
  chrom2 = 5, pos2 = 6), hg38, 100 kb. Gate subset **H200**: 200 cells sampled with
  `random_state=0`, stratified over the three major types present (c0, c1, c7).
- **E** — published HCEA objects under the ENTEx project root: `merged_cool_raw/merged.mcool`
  (100 kb), `merged_cool_raw/L1/c*.mcool` (100 kb), `analysis/compartment/merged_comp.hdf`
  (`raw_binfilter`, `raw_pcloading`) and `analysis/compartment/L1/comp.raw.hdf`.
- **D** — dcHiC inputs of the HCEA run, `analysis/diff_comp/all/` (34 major types, one sample
  each, `input.txt` + `c*_100Kb_pca/intra_pca/*/chr*.pc.bedGraph`), copied to the dev
  folder. Reference = a fresh `Rscript dchicf.r --file input.txt --pcatype analyze --dirovwt T
  --diffdir all` in the `dchic` env. (Checked before this manifest: the fresh run equals the
  published `DifferentialResult/fdr_result` table exactly.)

## S. scHiCluster single-cell compartment score

| Gate | Epione | Reference | Fixture | Class | Threshold |
|---|---|---|---|---|---|
| S1 | `cpg_ratio` (pyfaidx) | `get_cpg_profile` (`bedtools nuc -pattern CG -C`) | hg38 main chromosomes, 100 kb | exact | identical rows; identical values |
| S2 | per-cell pixels written by `contacts_to_scool` | scHiCluster scool pixel builder (no blacklist) | H200 | exact | identical (bin1, bin2, count) per cell |
| S3 | `compartment_scores`, cool input | `multiple_cell_compartment(mode='cool')` | H200 scool | 1, deterministic-strict | max abs err ≤ 1e-12 |
| S4 | `compartment_scores`, contact-table input | `multiple_cell_compartment(mode='tsv')` | H200 | 1, deterministic-strict | max abs err ≤ 1e-12 |
| S5 | per-chromosome AA / BB / AB strength sums | `compartment_strength` per chromosome | H200 | 1, deterministic-strict | rel err ≤ 1e-12 |
| S6 | `merge_cells` (raw pseudobulk) | `merge_cell_raw` | H200, per major type | exact | identical pixels |

S5 note: upstream accumulates `scores += scores` inside the chromosome loop, where `scores` is
the per-chromosome result, so its genome-wide value is twice the last chromosome's. Epione
returns the sum over chromosomes; the gate compares per-chromosome values, and the deviation
is documented in the docstring.

## H. HCEA pseudobulk compartments (raw contacts)

| Gate | Epione | Reference | Fixture | Class | Threshold |
|---|---|---|---|---|---|
| H1 | `compartment_model` bin filter | `merged_comp.hdf` `raw_binfilter` | E merged | exact | identical mask |
| H2 | `compartment_model` loadings | `merged_comp.hdf` `raw_pcloading` | E merged | 2, deterministic-bounded (ARPACK start vector) | max abs err ≤ 1e-8; same PC chosen per chromosome |
| H3 | `project_compartments` | `L1/comp.raw.hdf` | E, every L1 type | 2, deterministic-bounded | max abs err ≤ 1e-8 |
| H4 | `compartment_saddle` (with `clip=None`, chr1–chr21, as in the notebook's `compsaddle`) | notebook `compsaddle` rerun in `epione-hic` | E, three L1 types | 1, deterministic-strict | saddle / count max rel err ≤ 1e-12 |

## D. dcHiC differential compartments (`--pcatype analyze`)

| Gate | Epione | Reference | Fixture | Class | Threshold |
|---|---|---|---|---|---|
| D1 | OGK robust covariance (`covRob` auto → `pairwiseqc`) | `robust::covRob(x)$cov` | D, per-chromosome group matrices; plus 3 random matrices | 1, deterministic-standard | max rel err ≤ 1e-10 |
| D2 | `normalize_quantiles` | `limma::normalizeQuantiles(ties=TRUE)` | D, plus a matrix with ties | 1, deterministic-strict | max abs err ≤ 1e-13 |
| D3 | `dchic_analyze` per-chromosome `pcOri` / `pcQnm` tables | `DifferentialResult/all/pcOri`, `pcQnm` | D | exact | identical bins; values identical as written (5 decimals) |
| D4 | `dchic_analyze` `sample_maha`, `pval`, `padj` | `fdr_result/differential.intra_sample_combined.pcQnm.bedGraph` | D | 1, deterministic-standard | rel err ≤ 1e-8 (for `pval`, `padj` ≥ 1e-300; below that, both ≤ 1e-300) ; identical padj < 0.1 set |
| D5 | written output files | the R output folder | D | exact | same file names, columns and row order |

Paths of dcHiC that are not ported in this round raise `NotImplementedError` with a pointer to
this manifest: IHW adjustment (used when a condition has replicates), and the `mcd` /
`donostah` branches of `covRob` (chosen when there are fewer than 20 conditions). They depend
on R's random number stream and get their own gates when ported.

Speed is reported for every gate, Epione vs reference on one core.
