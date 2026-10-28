# INSTINCT port: pre-registered parity manifest

Written 2026-10-27 before the INSTINCT port was run on any data. Frozen by sha256
in `MANIFEST.sha256`; gates must not change after results are seen.

## Upstream

| item | value |
|---|---|
| repo | https://github.com/yyLIU12138/INSTINCT, commit `5753b31fb1b9607511bb28d0904c89c69b70d3cf` |
| licence | MIT (c) 2024 Yuyao LIU -> vendor allowed; keep header + NOTICE |
| paper | Liu et al., *Nat Commun* 16 (2025), doi:10.1038/s41467-025-56535-0 |
| torch_geometric use | none (plain torch upstream) |
| other deps replaced | episcanpy filters -> scanpy; BiocPy GenomicRanges/IRanges/biocutils `peak_sets_alignment` -> numpy |
| algorithm class | deterministic preprocessing (1) + stochastic training (GPU) -> embedding (4) + clustering (3) |

## Fixture

MISAR-seq S1 sections E11.0, E13.5, E15.5, E18.5 (Jiang et al. 2023, CC-BY-4.0;
MultiSP redistribution `paper/fig2/spatial/data/misar_multisp/*_atac_gt.h5ad`, sha256 in
`data/SOURCES.tsv`): 1,258 + 1,777 + 1,949 + 2,129 spots, raw peak counts with
slice-specific peak sets; `obsm["spatial"]`; labels `obs["Annotation_for_Combined"]`.
This is the upstream `MouseBrain_Jiang2023/runINSTINCT.py` setting (their input files
are the same MISAR sections; the redistribution differs only in packaging).

Pipeline (upstream script): `peak_sets_alignment` -> obs-name suffix -> concat ->
`preprocess_CAS(use_fragment_count=True, min_cells_rate=0.03)` -> PCA(100,
random_state=1234) -> `create_neighbor_graph` -> `INSTINCT_Model(seed=s)` defaults
(hidden 50, latent 30, 500+500 steps, lr 1e-3/5e-4) -> `eval`. Seeds 1234..1238
(upstream uses 1234+j). GPU H100. The official arm caches the deterministic
preprocessing once (as the upstream script writes its h5ad files); the port arm runs
`epi.tl.instinct` end to end for every seed.

Domains (upstream `evaluation.py`): `GaussianMixture(n_components=16,
covariance_type="tied", random_state=1234)` on the latent space, same in both arms.
ARI/NMI against `Annotation_for_Combined` over all four slices.

## Gates

| id | kind | metric | pass if |
|---|---|---|---|
| D1 | required (deterministic) | merged peak names, merged count matrices, preprocessed TF-IDF matrix, spatial graphs | identical (names equal; max abs diff of matrices <= 1e-12) |
| D2 | required (deterministic) | PCA input `X_pca` | max abs diff <= 1e-8 |
| G0 | exactness (reported) | max \|emb_off_s - emb_port_s\| | <= 1e-4 every seed |
| G1 | exactness (reported) | ARI(domains port_s, domains off_s) | >= 0.95 every seed |
| G2 | required | median_{s!=t} ARI(port_s, off_t) vs median_{s!=t} ARI(off_s, off_t) | cross >= seed-to-seed - 0.05 |
| G3 | required | Procrustes similarity of embeddings, median_{s!=t}, cross vs seed-to-seed | cross >= seed-to-seed - 0.02 |
| G4 | required | mean ARI and NMI vs `Annotation_for_Combined` | \|port - off\| <= max(2 SD_off, 0.02) each |

Pass = D1 & D2 & G2 & G3 & G4. Runtime: preprocessing (alignment) and training wall
time, both arms.

## Amendment A1 (2026-10-27 02:37, before any INSTINCT result was looked at)

Reason: the coordinator reserved MISAR-seq E15.5 / E18.5 labels as the held-out set of
the AEGLE evaluation. The fixture becomes the two DEV sections **E11.0-S1 + E13.5-S1**
only (1,258 + 1,777 spots); GMM components = 13 (union of the two sections' label sets).
Everything else (pipeline, seeds, gates, thresholds) is unchanged. Two official seeds and
one port seed had already been run on the 4-section fixture; those outputs were moved
unopened to `runs_4slice_unused/` and are not used.
