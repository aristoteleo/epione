# COSMOS port: pre-registered parity manifest

Written 2026-10-26 before any port code existed. Frozen by sha256 in
`MANIFEST.sha256`; the gates below must not be changed after results are seen.

## Upstream

| item | value |
|---|---|
| repo | https://github.com/Lin-Xu-lab/COSMOS, commit `56ea355be51e64d9253e2871b8bd447fdfd0d230` |
| licence | MIT (c) 2024 UT Southwestern Medical Center -> vendor allowed; keep header + NOTICE |
| paper | Zhou et al., *Nat Commun* 16, 2025, doi:10.1038/s41467-024-55204-y |
| torch_geometric use | `GCNConv`, `inits.reset/uniform` -> rewritten in plain torch (same op order) |
| algorithm class | stochastic training -> continuous embedding (4) + clustering (3); deterministic given seed on CPU |

## Fixture

`ATAC_RNA_Seq_MouseBrain_RNA_ATAC.h5` from Zenodo 13932144 (CC-BY-4.0, md5
`b1aa654e75150072a54d67414fb65483`): P22 mouse brain spatial ATAC-RNA (Zhang et al.
2023), 9,215 spots; `X_RNA` 3,000 genes, `X_ATAC` 50-d, `LayerName` = COSMOS manual
Allen labels (9 classes incl. `1_others`). This is the upstream tutorial
(`Tutorials/tutorial_ATAC_RNA_Seq_MouseBrain.ipynb`) input, used verbatim.

Settings (tutorial): `preprocessing_data(n_neighbors=10)`; `train(spatial_regularization_strength=0.01,
z_dim=50, lr=1e-3, wnn_epoch=500, total_epoch=1000, max_patience_bef=10, max_patience_aft=30,
min_stop=200, regularization_acceleration=True, edge_subset_sz=1000000)`.
Seeds: 20 (tutorial), 21, 22, 23, 24. Device: CPU (4 threads), torch 2.6.

Domains (both arms identically): the tutorial's Louvain + hand-made relabelling is
seed-specific and cannot be reproduced, so both embeddings are clustered by the same
routine: `epi.tl.spatial_domains(method="mclust", use_pca=True, n_comps=20, G=9)`
(pymclustR EEE, seed 2020). ARI/NMI against `LayerName`.

## Gates

Notation: off_s / port_s = official / port run with seed s (5 seeds).

| id | kind | metric | pass if |
|---|---|---|---|
| G0 | exactness (reported) | max \|emb_off_s - emb_port_s\| and max \|w_off_s - w_port_s\| (modality weights) | <= 1e-5 for every seed |
| G1 | exactness (reported) | ARI(domains port_s, domains off_s) | >= 0.95 for every seed |
| G2 | required | median over s != t of ARI(port_s, off_t) vs median over s != t of ARI(off_s, off_t) | cross >= seed-to-seed - 0.05 |
| G3 | required | Procrustes similarity (1 - scipy disparity) of embeddings, median over s != t, cross vs seed-to-seed | cross >= seed-to-seed - 0.02 |
| G4 | required | mean ARI(domains, LayerName) and mean NMI over seeds | \|port - off\| <= max(2 SD_off, 0.02) each |

The port passes if G2, G3 and G4 all pass. G0/G1 state whether it is exact.
Runtime: wall time of `train()` per seed, both arms, same node/threads.
