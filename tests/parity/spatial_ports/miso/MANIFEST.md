# MISO re-implementation: pre-registered parity manifest

Written 2026-10-27 before the re-implementation was run on any data. Frozen by sha256
in `MANIFEST.sha256`; gates must not change after results are seen.

## Upstream

| item | value |
|---|---|
| repo | https://github.com/kpcoleman/miso, commit `1252c6f4b280a0fd303c2a3e131e68b379acb3a6` |
| licence | "Penn Software MISO": non-profit research use only, no distribution to commercial third parties -> NOT vendorable into GPL-3.0 Epione; clean-room re-implementation (paper + reading the code for hyper-parameters), no upstream code copied |
| paper | Coleman et al., *Nat Methods* 22 (2025), doi:10.1038/s41592-024-02574-2 |
| scope | omics modalities only (the histology ViT feature extractor is not re-implemented) |
| algorithm class | stochastic (GPU) -> embedding (4) + clustering (3) |

## Fixture

MISAR-seq E15.5-S1 RNA + ATAC (MultiSP redistribution, sha256 in `data/SOURCES.tsv`),
1,949 spots, labels `Annotation_for_Combined` (12). Inputs: upstream
`utils.preprocess(adata, "rna"/"atac")` (min_cells 10, log1p, dense) for the
official arm, `epi.tl.miso_features` for the re-implementation. Model: upstream
defaults (`Miso([rna, atac], ind_views="all", combs="all", sparse=False)`, 1000
epochs per layer). Seeds s = 100..104 set as in the upstream tutorial (numpy, torch,
random). Domains: each arm's own k-means (`n_clusters=12`, `random_state=100`). GPU H100.

## Gates

| id | kind | metric | pass if |
|---|---|---|---|
| G0 | exactness (reported) | max \|emb_off_s - emb_port_s\| | <= 1e-4 every seed |
| G1 | exactness (reported) | ARI(domains port_s, domains off_s) | >= 0.95 every seed |
| G2 | required | median_{s!=t} ARI(port_s, off_t) vs median_{s!=t} ARI(off_s, off_t) | cross >= seed-to-seed - 0.05 |
| G3 | required | Procrustes similarity, median_{s!=t}, cross vs seed-to-seed | cross >= seed-to-seed - 0.02 |
| G4 | required | mean ARI and NMI vs `Annotation_for_Combined` | \|port - off\| <= max(2 SD_off, 0.02) each |

Pass = G2 & G3 & G4. Runtime: wall time per seed, both arms, same GPU.

## Amendment A1 (2026-10-27 02:37, before any MISO run)

Reason: MISAR-seq E15.5 / E18.5 labels are the held-out set of the AEGLE evaluation.
Fixture becomes the DEV section **E13.5-S1** (1,777 spots, 12 classes, k-means k = 12).
Pipeline, seeds, gates and thresholds unchanged.
