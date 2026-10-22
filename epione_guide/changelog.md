# Changelog

All notable changes to this project will be documented in this file.

The format roughly follows Keep a Changelog.


## Unreleased

### Added
- Single-cell Hi-C compartments from raw contacts (`epi.single.hic`): `cpg_ratio`, `contacts_to_scool`,
  `load_contact_tables`, `compartment_scores` (scHiCluster single-cell score), `merge_cells` (raw
  pseudobulks, `cis_only=`), and the Human Cell Epigenome Atlas recipe `compartment_model` →
  `project_compartments` / `compartment_saddle` / `oe_correlation`. scHiCluster outputs are reproduced
  exactly and HCEA's published model and 35 major-type scores to ≤ 1e-13
  (`tests/parity/schicluster_compartment`).
- `epi.bulk.hic.differential_compartments` / `dchic_analyze`: dcHiC's `dchicf.r --pcatype analyze` in
  Python, with `robust::covRob` (OGK, Donoho–Stahel and MCD branches, drawing from an emulation of R's
  random-number stream), `limma::normalizeQuantiles` and R's rounding; matches R on HCEA's 35 types,
  on three types and on six types (p-values ≤ 1e-8 relative, identical padj < 0.1 sets). IHW (used
  with replicates) raises `NotImplementedError`.
- Contact decay (`epi.single.hic`): `contact_decay` (scHiCluster `contact-distance`), `decay_ratio`
  (short/long ratio), `compartment_contact_hist`, `quantile_normalize`; identical to HCEA's published
  decay histograms (`tests/parity/schicluster_decay`).
- `epi.single.hic.embedding(method="schicluster")`: scHiCluster's `hicluster embedding` (per-chromosome
  band SVD, then SVD of the concatenation; `norm_sig=`). Band matrices identical to scHiCluster, final
  embedding within scHiCluster's own run-to-run variation (`tests/parity/schicluster_embedding`).
- `impute_cells(resolution=, window_size=, step_size=)`: cells loaded with `load_contact_tables` are
  imputed straight from raw contacts at any resolution, as `hicluster impute-cell` with a contact file.
- `impute_cells(accelerate=True)`: windowed random walks (`window_size` smaller than a chromosome, the
  10 kb recipe) run a faster implementation of scHiCluster's `impute_chromosome` (numba Gaussian, direct
  sparse kernels, one dense window accumulator); equal to scHiCluster up to float32 rounding. Whole-
  chromosome settings run scHiCluster unchanged.
- `epi.single.hic.gene_score`: scHiCluster's `hicluster gene-score --mode raw` (contacts inside each gene
  body) from contact tables, with `min_pos_dist=` to drop short contacts; identical to scHiCluster's
  `gene_score_raw`.
- Tutorials `t_sc_hic_compartment` and `t_sc_hic_decay` (13,166 HCEA snm3C-seq cells, six haematopoietic
  major types), `t_sc_hic_embedding` (6,000 of them), and a rebuilt Droplet Hi-C case study (Fig 1d
  by gene scores co-embedded with snRNA-seq through `epi.tl.integrate(method='cca')`).

### Fixed
- `impute_cells` wrote empty matrices with scHiCluster ≥ 1.3, which saves `.npz` instead of HDF; failures
  were silenced. It now reads both formats and warns when a cell fails.
- `epi.pl` contact plots: `title=''` hides the default title.
- `from epione.single import hic` (and `epi.single` before `epi.pl`) no longer fails with a circular
  import between `epione.utils` and `epione.single.atac`.
- `epi.tl.e2g_links`, `epi.tl.e2g_features`, `epi.tl.count_insertions` (Tn5 insertions in arbitrary
  intervals from fragment files), `epi.tl.contact_links`, `epi.tl.contact_gene_score`,
  `epi.tl.fit_contact_gamma`, `epi.tl.powerlaw_contact`.
- `epi.tl.umap(device="cuda")`: the spectral initialisation (torch LOBPCG) and the SGD epochs
  (cuML-style parallel updates) run on the GPU, with umap-learn's own graph pruning and schedule
  (1M cells: 36 s vs 21 min on the CPU).
- `epi.tl.add_gene_score_matrix(out_of_core="auto")`: an output larger than a quarter of the
  process memory limit is a memory-mapped array under `$TMPDIR` (same values).
- `epi.single.scan_peaks` (CellScan): a scATAC-native peak caller. Each window is scored by the number of
  distinct cells with a Tn5 insertion, against a depth-aware null and a Tn5-bias-corrected local background.
- `epi.single.call_peaks(adata, groupby, method=...)`: one entry point that returns a non-overlapping,
  fixed-width peak set ready for `epi.pp.make_peak_matrix`.
- `epi.tl.ot_batch`: batch correction by per-batch biwhitening of the counts and unbalanced
  entropic optimal transport to a Wasserstein barycenter; `epi.tl.batch_correct(adata, batch_key,
  method=...)` as the single entry point.

### Changed (breaking)
- **Default enhancer–gene / peak-to-gene links are now `epi.tl.link_peaks_to_genes`** (method `"e2g"`,
  `epi.tl.e2g_links`): a DEV-trained Bayesian evidence combination of a polymer-contact distance prior,
  peak-free accessibility (any interval, counted from fragments), ABC and — with RNA — metacell partial
  correlation. Output keeps the `peak_to_gene` columns plus `score` (posterior); `fdr` = 1 − `score`.
  The ArchR port `epi.tl.peak_to_gene` is unchanged (`method="peak_to_gene"`), as are
  `coaccessibility` and `contact_links`.
  - Held-out (fine-mapped eQTLs, OneK1K on 10x PBMC 10k multiome): AUPRC 0.291 vs ArchR 0.174,
    SnapATAC2 0.182, Epione `peak_to_gene` 0.156, Signac 0.152, Cicero 0.102, ABC 0.099.
  - **Not better than distance to TSS** (0.407 there; 0.506 vs 0.300 on GEUVADIS LCL eQTLs with
    GM12878 scATAC), because 73–75% of positive eQTL elements lie outside accessible chromatin; tie
    with ABC on GM12878 CRISPRi (16 positives). Report a distance baseline alongside links.
  - Speed: faster than `peak_to_gene` on the same data and threads (50k cells 16–18 s vs 18–31 s;
    100k cells 15 s vs 19 s).
- **`epi.tl.clusters` uses igraph's C Leiden by default** (`backend='igraph'`): the same objective
  as before (modularity ≡ RBConfiguration at the same resolution, weighted, iterated to convergence).
  On 50k / 100k-cell graphs the igraph partitions agree with leidenalg's as closely as leidenalg's
  own seeds do (ARI ≈ 0.6), with equal or higher modularity, and run 8-10x faster; leidenalg needs
  hours at 1M cells. The previous default is `backend='leidenalg'` (alias `'epione'`).
- `epi.pp.import_fragments(store_fragments="auto")` (new default): the in-memory fragment store is
  memory-mapped under `$TMPDIR` when it exceeds a quarter of the process memory limit (RAM or the
  cgroup / Slurm `--mem` limit). Results are identical; `True` keeps the previous behaviour.
- **Default peak caller is now CellScan** (`call_peaks(method='cellscan')`). The previous behaviour is
  `call_peaks(method='macs3')`, i.e. `epi.single.macs3` + `merge_peaks`, unchanged.
  - Held-out accuracy: on ENCODE HepG2 / MCF-7 / A549 scATAC with bulk ATAC + DNase truth, F1 of the top 50k
    peaks was 0.884 / 0.850 / 0.876 vs 0.875 / 0.837 / 0.866 for MACS3 + merge. It was also above ArchR,
    SnapATAC2, Signac, HMMRATAC and Genrich.
  - Speed: 10–30x faster (100k cells: 6 min vs 2 h 40 min, 8 threads).
  - Known weaknesses: less precise CTCF summits than Genrich / SnapATAC2 / MACS3; lower recall of peaks
    specific to a rare cluster than ArchR; peak memory 3–4x higher.
- **Default cell embedding is now BiRMT-2** (`epi.tl.birmt`, `obsm['X_birmt']`):
  `epi.tl.birmt` defaults to `version=2` (BiRMT v1 via `version=1`), and the
  `use_rep` defaults of `tl.harmony`, `tl.add_impute_weights`, `tl.peak_to_gene`,
  `tl.coaccessibility`, `tl.add_trajectory`, `tl.correlate_matrices`,
  `tl.positive_tf_regulators` and `pl.prepare_peak2gene_heatmap` change from
  `'X_iterative_lsi'` to `'X_birmt'`. Pass `use_rep='X_iterative_lsi'` to keep the
  previous behaviour; `tl.iterative_lsi` / `tl.lsi` are unchanged. Clustering stays
  `pp.neighbors` + `tl.clusters` on the embedding (BiRMT-2's own Markov-time-1
  partition is opt-in, `cluster=True`).
- **Default batch correction is now `ot_batch`** (`epi.tl.batch_correct(adata, batch_key)`,
  `obsm['X_ot_batch']`, computed from the count matrix). Harmony stays available unchanged as
  `epi.tl.harmony` / `batch_correct(method='harmony')`.
  - Held-out (SEA-AD MTG, Heart Cell Atlas v2, 10x PBMC multiome; scIB-style 0.6·bio + 0.4·batch):
    mean 0.651 vs 0.641 for the best competitor (Scanorama on the same biwhitened input), 0.622
    for BiRMT-2 + Harmony, 0.600 for iterative LSI + Harmony. The +0.010 margin is within
    run-to-run noise, so this is a tie with the best competitor: one win, one tie, one loss
    (PBMC, where LSI + Harmony is better).
  - Most of the gain comes from the biwhitened input. Per the ablations, the OT marginal coupling
    adds nothing beyond entropic soft assignment, while balanced OT is clearly worse.
  - Speed: faster than BiRMT-2 + Harmony on every evaluated set (140k cells: 139 s vs 253 s,
    8 threads); peak memory up to ~20 % higher than Harmony's pipeline on some sets.


### Performance
- `epi.tl.iterative_lsi` is matrix-free (TF-logIDF applied per row block inside the SVD products,
  bit-identical on the CPU; device blocks streamed within a budget on the GPU): +2 GiB instead of
  +22 GiB host memory at 100k cells, 1M cells within ~8 GiB of the input.
- `epi.tl.birmt` (BiRMT-2) works on the int32 sparsity pattern without value arrays or int64 index
  copies: bit-identical output with ~8 instead of ~36 bytes of extra memory per non-zero.
- `epi.tl.add_gene_score_matrix` no longer allocates a second dense output on the fragment-store
  path, and computes the `adata.X` path in row blocks.
