# Changelog

All notable changes to this project will be documented in this file.

The format roughly follows Keep a Changelog.


## Unreleased

### Added
- `epi.tl.e2g_links`, `epi.tl.e2g_features`, `epi.tl.count_insertions` (Tn5 insertions in arbitrary
  intervals from fragment files), `epi.tl.contact_links`, `epi.tl.contact_gene_score`,
  `epi.tl.fit_contact_gamma`, `epi.tl.powerlaw_contact`.
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

