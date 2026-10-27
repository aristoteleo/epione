# CANDIES port: pre-registered parity manifest

Written 2026-10-27 before the CANDIES port was run on any data. Frozen by sha256 in
`MANIFEST.sha256`; gates must not change after results are seen.

## Upstream

| item | value |
|---|---|
| repo | https://github.com/YeLiu-Lab/CANDIES, commit `85ce567446721de83c09a6feed0f20d9744b031b` |
| licence | MIT (c) 2025 zouwanpeng -> vendor allowed; keep header + NOTICE |
| paper | Liu et al., *Adv Sci* 2026, doi:10.1002/advs.202523754 (MISAR E15.5 ARI 0.5597 reported) |
| replaced deps | torch_geometric `GCNConv` (on a torch sparse adjacency) -> plain torch; timm `Mlp` -> 2-layer MLP; POT `ot.dist` -> numpy; R mclust (rpy2) -> pymclustR |
| algorithm class | stochastic (GPU, `torch.use_deterministic_algorithms(True)` as upstream) -> embedding (4) + clustering (3) |

## Fixture

MISAR-seq E15.5-S1 (MultiSP redistribution, `paper/fig2/spatial/data/misar_multisp/E15_5-S1_{expr,atac}_gt.h5ad`,
sha256 in `data/SOURCES.tsv`): 1,949 spots, RNA + ATAC raw counts, `obsm["spatial"]` =
`obs[["array_col","array_row"]]` (tutorial). Labels `Annotation_for_Combined` (12).

Pipeline: upstream `tutorial/E18_mouse_brain.ipynb` (the MISAR-seq tutorial) cell by cell:
RNA `filter_genes(min_cells=10)`, HVG seurat_v3 3000, normalise, log1p, scale; ATAC HVG + `lsi(n_components=51,
use_highly_variable=False)`; RNA PCA(50); spatial graph (3-NN) + feature graph (20-NN correlation);
`encoder_ZINB(epochs=300, dim_output=64)`; `train_atac(epochs=600)`; `run_diff(k=3, batch_size=512,
hidden_size=256, learning_rate=1e-3, num_epoch=1000, diffusion_step=800, depth=6, head=16, patience=40,
bias=0.5)` denoising RNA conditioned on ATAC; `train_and_infer(epochs=100)`. The modality-selection cell
(diagnostic print, needs labels) is skipped in both arms. Seeds: the tutorial's fixed seeds (ZINB 41,
ATAC AE 2024, diffusion 2024, integration 2025) shifted by s = 0, 1, 2, 3, 4. GPU H100.

Domains: both arms' joint embeddings are clustered by the same routine,
`epi.tl.spatial_domains(method="mclust", use_pca=True, n_comps=20, G=12)` (upstream
`clustering(method="mclust", use_pca=True)` with R mclust replaced by pymclustR).

## Gates

| id | kind | metric | pass if |
|---|---|---|---|
| L0 | exactness (reported) | plain-torch GCNConv on a torch sparse adjacency vs PyG GCNConv, CPU, forward + grad | max abs diff == 0 |
| G0 | exactness (reported) | max \|emb_off_s - emb_port_s\| | <= 1e-4 every seed |
| G1 | exactness (reported) | ARI(domains port_s, domains off_s) | >= 0.95 every seed |
| G2 | required | median_{s!=t} ARI(port_s, off_t) vs median_{s!=t} ARI(off_s, off_t) | cross >= seed-to-seed - 0.05 |
| G3 | required | Procrustes similarity of embeddings, median_{s!=t}, cross vs seed-to-seed | cross >= seed-to-seed - 0.02 |
| G4 | required | mean ARI and NMI vs `Annotation_for_Combined` | \|port - off\| <= max(2 SD_off, 0.02) each |

Pass = G2 & G3 & G4. Runtime: wall time per seed, both arms, same GPU.

## Amendment A1 (2026-10-27 02:37, before any CANDIES run)

Reason: MISAR-seq E15.5 / E18.5 labels are the held-out set of the AEGLE evaluation.
Fixture becomes the DEV section **E13.5-S1** (1,777 spots, 12 `Annotation_for_Combined`
classes, G = 12). Pipeline, seeds, gates and thresholds unchanged.
