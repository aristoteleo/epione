# Higashi parity (metric level), pre-registered

Written and committed before any Epione-vs-reference comparison. Before this file, the only Epione runs were smoke tests on the first 120 cells with 1–2 epochs, used to check that the code runs and to profile it. The reference seed-0 run had started but no metric had been computed on either side.

## Reference

| | |
|---|---|
| code | `ma-compbio/Higashi` @ `42702a8` (MIT), new `Higashi()` wrapper API, run unmodified from a source checkout |
| env | Python 3.9, numpy 1.23.5, scipy 1.7.3, pandas 1.3.4, scikit-learn 1.0.2, torch 2.0.1+cu118 (versions from upstream `setup.py` where pinned) |
| driver | `run_ref.py`: `np.random.seed(s)`, `torch.manual_seed(s)`, then `process_data → prep_model → train_for_embeddings → train_for_imputation_nbr_0 → impute_no_nbr → train_for_imputation_with_nbr → impute_with_nbr` |
| hardware | the same shared H100 as the Epione runs (`srun --overlap`), 8 CPU sampler workers |

## Fixture

Nagano *et al.* 2017 mouse ES cells, 1 Mb, the `.scool` used by `t_sc_hic_basics`. Cell selection follows that tutorial (GEO features table, `passed_qc == 1`, present in the scool, `total_contacts > 5000`) and keeps every cell of the three main phases rather than the tutorial's 35-per-phase subsample: **1,430 cells** (early-S 676, late-S/G2 406, G1 348). Chromosomes chr1–19, chrX. Only intra-chromosomal contacts are written, because Higashi uses no others.

**Held-out contacts.** For each cell, 10% of its off-diagonal pixels are removed (`numpy.random.default_rng(0)`) before training on both sides. Each removed pixel is paired with one pixel of the same cell, chromosome and genomic distance that has zero count in the full data: 2,081,470 positive/negative pairs. Both sides train on the remaining 90% (`data.txt`, Higashi's `higashi_v1` format) and are evaluated on the held-out pairs.

Both sides read the same `data.txt`. Shared config, upstream defaults otherwise:

| key | value |
|---|---|
| resolution, resolution_cell | 1,000,000 |
| dimensions | 64 |
| loss_mode | zinb (upstream's recommended mode) |
| neighbor_num | 5 |
| local_transfer_range | 1 |
| minimum_distance / maximum_distance | 1,000,000 / −1 |
| embedding / no_nbr / with_nbr epochs | 60 / 45 / 30 (upstream defaults) |
| structured | false (single-table read; see G0) |

**Seeds.** 0, 1, 2 on both sides.

## Known effective behaviours of the reference, reproduced on purpose

1. **Stage-1 pair ratio.** Upstream raises `pair_ratio` from 0 to 0.5 during stage 1, but it does so in the parent process after the sampler workers have been forked, so the workers keep 0.0 for the whole stage. This was checked on the running reference: every worker logged `steps=1 pair_ratio=0.0` through epoch 9, while the parent's value had reached 0.5 at epoch 5. Epione fixes the sampler's ratio when a stage starts: 0.0 in stage 1, 0.5 in stages 2–3.
2. **Duplicate optimiser parameters.** The stage-1 and stage-2 Adam optimisers receive the embedding parameters twice, so those parameters get two updates per step. Epione builds the optimisers the same way.
3. **Double activation.** The cell auto-encoder applies swish twice on its reconstruction path. Kept.
4. The negative-sampling bin-change rule uses `other_bin = 2 if change == 1 else 2`. Kept.

## Deviations (forced or performance-only)

- The batch construction (negative sampling and neighbour lists) runs vectorised in a background thread instead of 8 worker processes. The rule is unchanged: the proposals, the clipped 3×3 occupancy test, the 50-trial limit and the reset on an invalid distance. Only the random stream differs.
- The stage-2/3 neighbour aggregation is computed as a scatter-add, not a COO sparse `mm`. It gives the same sum without the coalescing sort. Neighbour embeddings are routed straight to the chromosome's auto-encoder, which gives identical values.
- Everything is held in memory instead of `temp_dir`; checkpoints are deep copies.
- Not ported: batch-effect correction, co-assay, inter-chromosomal contacts, the multi-GPU imputation split, the visualisation app. `transfer_range > 0` raises.

## Gates

Thresholds are fixed here and are not changed after results are seen. Any amendment is added below as a new section, before the run it governs, and the original result stays reported.

### G0 · preprocessing / hypergraph equality (seed 0)

Epione's `process_contacts(data.txt, random_state=0)` is compared with the reference's `process_data` output after `np.random.seed(0)` (`node_feats.hdf5`, `sparse_nondiag_adj_nbr_1.npy`, `raw/*_sparse_adj.npy`).

- **Exact:** `chrom_start_end`, `num`, `start_end_dict`, `id2chrom`; per chromosome, the train and test hyperedges (node ids and order), their chromosome labels and weights; the raw per-cell matrices.
- **rtol 1e-6, atol 1e-9:** the depth-normalised per-cell graphs (same sparsity pattern required), the bin-node attributes, `sparsity`, `extra_cell_feats`, `cell2weight`, `distance2weight`.
- **Cell-node attributes** (randomised TruncatedSVD): relative Frobenius error ≤ 1e-3 for each chromosome.

### G1 · metric implementation

The parity metric module (`metrics.py`, scikit-learn based) is checked against independent from-scratch implementations, on the reference seed-0 outputs:
- ARI and AMI from the contingency table;
- kNN accuracy by brute-force distances, with the same folds;
- per-cell AUROC by the Mann–Whitney rank formula;
- the distance-stratified pseudo-bulk Pearson correlation by an explicit loop.

Pass if every |Δ| ≤ 1e-9.

### G1b · architecture equivalence (forward pass with the reference weights)

The reference's trained stage-1/2/3 weights (`model.chkpt_stage{1,2,3}`, `model_link` state dicts) are loaded into Epione's modules with `strict=True`, built from Epione's own G0 preprocessing. Pass if max |Δ| ≤ 1e-4 (float32, torch 2.0.1 vs 2.6.0) for:
- (a) the stage-1 cell embeddings, against those the reference stage-1 model produces;
- (b) nbr-0 imputation of 3 cells (ids 0, 700, 1429), against the reference's `*_exp_nbr_0_impute.hdf5`;
- (c) nbr-5 imputation of the same cells, against `*_exp_nbr_5_impute.hdf5`.

For (c), the 5-nearest-cell lists must also be identical.

### G2 · training parity (3 seeds a side)

Rule, per metric:

    |mean_Epione − mean_reference| ≤ max(2 × SD_reference, floor)

where SD_reference is the sample SD of the 3 reference seeds.

**Embedding.** The cell embedding is the stage-1 cell-node embedding at the end of stage 1, which is what upstream's `fetch_cell_embeddings()` returns after the full pipeline.

| metric | definition | floor |
|---|---|---|
| ARI | KMeans(k=3, n_init=10, random_state=0) on the raw 64-d embedding vs phase | 0.03 |
| AMI | same clustering | 0.03 |
| kNN accuracy | KNeighborsClassifier(k=10), StratifiedKFold(5, shuffle, random_state=0), phase | 0.02 |

**Imputation.** For both nbr-0 (stage 2) and nbr-5 (stage 3), using the raw imputed values (upstream with `gpu_num < 2` writes them without per-cell rescaling).

| metric | definition | floor |
|---|---|---|
| held-out AUROC | per-cell AUROC of the imputed value, held-out contacts vs distance-matched zero pixels; mean over cells with ≥ 10 positives | 0.005 |
| pseudo-bulk r | per phase, the mean imputed map over the phase's cells vs the phase's raw pseudo-bulk (full data, held-out included); both z-scored within each (chromosome, distance) stratum over pairs at distance ≥ 1 bin; Pearson r over all pairs; mean over the 3 phases | 0.01 |

That makes 3 + 2 × 2 = **7 gated metrics**. G2 passes only if all 7 pass. Reported but not gated: the cross-validated R² of a ridge regression of `repli_score` on the embedding, wall-clock per stage, and epochs run before early stopping.

### G3 · fast mode

Not applicable: no fast mode is added. Fast-Higashi (ma-compbio/Fast-Higashi, a separate PARAFAC2-based method of about 3.9k lines) is not ported in this change.
