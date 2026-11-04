# Spatial ATAC: SpatialGlue parity (pre-registered)

Port: `epione.tl.spatialglue` / `epione.tl.spatialglue_preprocess`
(`epione/tl/_spatialglue.py`, AGPL-3.0-or-later, see licence below).
Clustering used by both arms: `epione.tl.spatial_domains(method="mclust")`.

Gates written 2026-10-14, before the gate run. A one-seed smoke run of the
harness (seed 2022) was used only to check that the script runs end to end;
the gates and tolerances below were fixed before any multi-seed numbers were
seen and are not to be widened.

## Reference

| item | value |
|---|---|
| upstream | https://github.com/JinmiaoChenLab/SpatialGlue, commit `7c976d811d27ace51ce47ae0ad94a068a7d222fa` (v1.1.5) |
| paper | Long et al., *Nature Methods* 21, 1658-1667 (2024) |
| pipeline | official Tutorial 4 "data integration for mouse brain Spatial-epigenome-transcriptome" (SpatialGlue_tutorials), run verbatim |
| dataset | `Dataset7_Mouse_Brain_ATAC` from Zenodo record 10362607 (`Data_SpatialGlue.zip`, md5 `b766948c1136bb3af2f3e6e85dc81cfe`, CC-BY-4.0): P22 mouse brain spatial ATAC-RNA-seq (Zhang et al., *Nature* 2023), 9,215 spots, 22,914 genes, 121,068 peaks |
| annotated domains | `obs["ATAC_clusters"]` and `obs["RNA_clusters"]` shipped in the dataset (the original authors' per-modality spot clusters). There is no manual tissue-domain annotation for this dataset, so these are the reference labels. |
| datatype | `Spatial-epigenome-transcriptome` -> 1600 epochs, weight factors [1, 5, 1, 1], spatial k = 6, lr 1e-4, dim 64 |
| clustering | PCA(20) of the joint embedding -> mclust `EEE`, G = 18 (tutorial setting), via pymclustR, seed 2020 |
| device | CPU (`srun --overlap`, 8 cores), torch 2.6.0 |
| seeds | 2022 (tutorial default), 2023, 2024 |

The official arm imports upstream unchanged from the clone. Its `mclust_R`
(rpy2 + R) is not called: both embeddings are clustered by the same
pymclustR routine, so the gates compare the integration model, not two
clustering back-ends. R is not used anywhere.

## Gates

| id | metric | pass if |
|---|---|---|
| G0 | max \|emb_official - emb_port\| of the L2-normalised joint embedding, per seed, same seed and inputs | <= 1e-5 for every seed |
| G1a | ARI(domains, `ATAC_clusters`), mean over 3 seeds | \|mean_port - mean_official\| <= max(2 x SD_official, 0.02) |
| G1b | ARI(domains, `RNA_clusters`) | same rule |
| G1c | NMI(domains, `ATAC_clusters`) | same rule |
| G1d | NMI(domains, `RNA_clusters`) | same rule |
| G1e | mean Moran's I of the one-hot domain labels on a 6-NN spot graph (spatial coherence) | same rule |
| G2 | ARI(port domains, official domains), per seed | >= 0.95 for every seed |

## Run

```
python tests/parity/spatial_atac/parity_spatialglue.py \
    --data <Data_SpatialGlue>/Dataset7_Mouse_Brain_ATAC \
    --ref-src <SpatialGlue clone> --out results.json --seeds 2022 2023 2024
```

## Results (2026-10-14, CPU, 3 seeds run in parallel)

9,196 spots after the tutorial's `filter_cells(min_genes=200)`.

Per seed (official = port for every number; G0 difference 0.0, i.e. bit-identical):

| seed | max \|d emb\| | max \|d alpha\| | ARI port vs official | ARI vs ATAC_clusters | ARI vs RNA_clusters | NMI vs ATAC_clusters | NMI vs RNA_clusters | Moran's I |
|---|---|---|---|---|---|---|---|---|
| 2022 | 0.0 | 0.0 | 1.000 | 0.2692 | 0.2452 | 0.3783 | 0.3805 | 0.7017 |
| 2023 | 0.0 | 0.0 | 1.000 | 0.2522 | 0.2508 | 0.3736 | 0.3878 | 0.6772 |
| 2024 | 0.0 | 0.0 | 1.000 | 0.2833 | 0.2520 | 0.3915 | 0.3917 | 0.6676 |

Gates:

| id | official mean (SD) | port mean | \|delta\| | tolerance | result |
|---|---|---|---|---|---|
| G0 | - | - | max 0.0 | 1e-5 | PASS |
| G1a ARI vs ATAC_clusters | 0.2683 (0.0156) | 0.2683 | 0.0 | 0.0312 | PASS |
| G1b ARI vs RNA_clusters | 0.2493 (0.0036) | 0.2493 | 0.0 | 0.02 | PASS |
| G1c NMI vs ATAC_clusters | 0.3811 (0.0093) | 0.3811 | 0.0 | 0.02 | PASS |
| G1d NMI vs RNA_clusters | 0.3867 (0.0057) | 0.3867 | 0.0 | 0.02 | PASS |
| G1e Moran's I | 0.6822 (0.0176) | 0.6822 | 0.0 | 0.0352 | PASS |
| G2 | - | - | min ARI 1.000 | >= 0.95 | PASS |

Training speed of the two arms is the same (200 epochs, 8 cores, run
alternately: official 69.7 / 79.8 s, port 68.7 / 64.2 s). The wall times in
the per-seed JSON differ because other jobs shared the node during the
port arm.

Context: ARI against the original per-modality clusters is modest (0.25-0.28)
for both arms. Those clusters come from one modality each and are noisy in
space (see the tutorial figure), so they are a reference partition, not
ground truth.

## Method choice (survey, 2026-10-14)

Spatial multi-omics integration methods with PyTorch code, checked for a
licence that allows a port into Epione (GPL-3.0):

| method | licence (GitHub) | note |
|---|---|---|
| SpatialGlue (Long et al. 2024, Nat Methods) | AGPL-3.0 | ported: most used, has a spatial ATAC-RNA tutorial and released data; AGPL combines with GPL-3.0 (section 13) |
| PRESENT (Li et al. 2024, Nat Commun) | MIT | ATAC-aware likelihoods; candidate for a second method |
| SpaMosaic | MIT | mosaic integration across sections |
| PRAGA, GraphST | AGPL-3.0 | |
| STAGATE / STAGATE_pyG | MIT | single-modality spatial domains |
| COSMOS, MISO | no licence file | cannot be ported |

omicverse (github.com/omicverse) has no spatial-ATAC reader or SpatialGlue
port; the clustering step reuses its standalone pymclustR (CRAN mclust port).

## Licence

SpatialGlue's `LICENSE.md` is AGPL-3.0 (its `setup.py` says MIT; the
stricter LICENSE file is followed). Epione is GPL-3.0. GPLv3 section 13
permits combining a GPLv3 work with an AGPLv3 work, with each part keeping
its own licence, so `epione/tl/_spatialglue.py` carries an
`SPDX-License-Identifier: AGPL-3.0-or-later` header and upstream
attribution. The rest of the spatial code (readers, graph, smoothing,
domains, plots) is original Epione code under GPL-3.0.
