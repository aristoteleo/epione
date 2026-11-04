# SpaMosaic port: pre-registered parity manifest

Written 2026-10-27 before the SpaMosaic port was run on any data. Frozen by sha256 in
`MANIFEST.sha256`; gates must not change after results are seen.

## Upstream

| item | value |
|---|---|
| repo | https://github.com/JinmiaoChenLab/SpaMosaic, commit `cc1336755de8d1ddd1337b3cd67c9d41a61c2cb4` |
| licence | MIT (c) 2025 Jinmiao Lab -> vendored, header + NOTICE |
| paper | Yan et al., *Nat Genet* 2026, doi:10.1038/s41588-026-02573-3 |
| torch_geometric / torch_scatter | WLGCN propagation, add_remaining_self_loops, scatter_add -> plain torch; only the default `wlgcn` architecture is ported (HGT/GAT variants not) |
| other | Annoy MNN search kept (optional dep; exact sklearn fallback), harmony-pytorch optional, R mclust -> pymclustR |
| algorithm class | stochastic (GPU, upstream enables deterministic algorithms) -> embedding (4) + clustering (3) |

## Fixture (DEV sections only; E15.5/E18.5 are AEGLE held-out)

MISAR-seq E11.0-S1 + E13.5-S1, RNA + ATAC for both sections (= two bridge batches:
contrastive RNA-ATAC alignment within each batch plus cross-batch MNN edges), MultiSP
redistribution (sha256 in `data/SOURCES.tsv`). Preprocessing: upstream `RNA_preprocess(n_hvg=5000,
batch_corr=False)`, `Epigenome_preprocess(n_peak=50000, batch_corr=False)` (MB_vertical tutorial
settings), batch key = section. Model: `SpaMosaic(intra_knns=10, seed=s)`,
`train(net="wlgcn", lr=0.01, T=0.01, n_epochs=100)`, `infer_emb` (mean of modalities).
Seeds 1234..1238. GPU H100. Labels `Annotation_for_Combined` (13 classes over the two sections).

Domains: both arms' embeddings by the same routine, mclust `EEE`, G = 13, no PCA
(upstream `utils.clustering(algo="mclust")`), via pymclustR.

## Gates

| id | kind | metric | pass if |
|---|---|---|---|
| D1 | required (deterministic) | preprocessed inputs `obsm["dimred_bc"]` per modality and section | max abs diff <= 1e-8 |
| G0 | exactness (reported) | max \|emb_off_s - emb_port_s\| | <= 1e-4 every seed |
| G1 | exactness (reported) | ARI(domains port_s, domains off_s) | >= 0.95 every seed |
| G2 | required | median_{s!=t} ARI(port_s, off_t) vs median_{s!=t} ARI(off_s, off_t) | cross >= seed-to-seed - 0.05 |
| G3 | required | Procrustes similarity, median_{s!=t}, cross vs seed-to-seed | cross >= seed-to-seed - 0.02 |
| G4 | required | mean ARI and NMI vs labels | \|port - off\| <= max(2 SD_off, 0.02) each |

Pass = D1 & G2 & G3 & G4.
