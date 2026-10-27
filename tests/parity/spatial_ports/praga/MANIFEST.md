# PRAGA port: pre-registered parity manifest

Written 2026-10-27 before the PRAGA port was run on any data. Frozen by sha256 in
`MANIFEST.sha256`; gates must not change after results are seen.

## Upstream

| item | value |
|---|---|
| repo | https://github.com/Xubin-s-Lab/PRAGA, commit `4adb11c96fc7ddad800fa1787eadcc8b91b42784` |
| licence | AGPL-3.0 (LICENSE.md); parts from SpatialGlue (AGPL-3.0) and DeepDPM (MIT) -> vendored as an AGPL-3.0-or-later file, as the existing SpatialGlue port |
| paper | Huang et al., AAAI 2025, doi:10.1609/aaai.v39i1.32010 (P22: ARI 27.06 % on its own labels) |
| torch_geometric | not used upstream |
| replaced | R mclust (rpy2) -> pymclustR; hard-coded `cuda:1` -> `device` argument |
| exact rewrite | the per-spot 2-component sub-cluster GMM refit (identical for all spots of a label) is memoised per label |
| algorithm class | stochastic (GPU, autocast) -> embedding (4) + clustering (3) |

## Fixture

PRAGA's Zenodo 14591305 `Data/Mouse_Brain` (CC-BY-4.0; sha256 in `data/SOURCES.tsv`): P22
mouse-brain spatial ATAC-RNA (Zhang et al. 2023), `adata_RNA.h5ad` +
`adata_peaks_normalized.h5ad`; labels `MB_cluster.txt` (9,196 spots after the RNA QC,
14 classes; PRAGA's reference labels). Settings: upstream `run.sh` Mouse Brain line
(`data_type Spatial-epigenome-transcriptome, n_clusters 14, init_k 14, KNN_k 20,
RNA_weight 1, ADT_weight 10`, argparse defaults tau 2, alpha 0.9, cl_weight 1). Seeds:
upstream fixed seed 2024 shifted by s = 0..4. GPU H100; the official arm is upstream
`main.py`'s code path called from a runner that sets the device to `cuda:0` (the only GPU).

Domains: both arms' embeddings clustered identically,
`epi.tl.spatial_domains(method="mclust", use_pca=True, n_comps=20, G=14)` (upstream
`clustering(method="mclust", use_pca=True)` with R replaced by pymclustR). ARI/NMI vs
`MB_cluster.txt`.

## Gates

| id | kind | metric | pass if |
|---|---|---|---|
| G0 | exactness (reported) | max \|emb_off_s - emb_port_s\| | <= 1e-4 every seed |
| G1 | exactness (reported) | ARI(domains port_s, domains off_s) | >= 0.95 every seed |
| G2 | required | median_{s!=t} ARI(port_s, off_t) vs median_{s!=t} ARI(off_s, off_t) | cross >= seed-to-seed - 0.05 |
| G3 | required | Procrustes similarity, median_{s!=t}, cross vs seed-to-seed | cross >= seed-to-seed - 0.02 |
| G4 | required | mean ARI and NMI vs `MB_cluster.txt` | \|port - off\| <= max(2 SD_off, 0.02) each |

Pass = G2 & G3 & G4. Runtime: wall time per seed, both arms, same GPU.
