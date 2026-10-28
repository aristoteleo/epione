# PRESENT port: pre-registered parity manifest

Written 2026-10-26 before any PRESENT port code existed. Frozen by sha256 in
`MANIFEST.sha256`; gates must not change after results are seen.

## Upstream

| item | value |
|---|---|
| repo | https://github.com/lizhen18THU/PRESENT, commit `c88a609b34aae9b84c2c8a7ffb6824bae5c78f23` (PyPI `bio-present`) |
| licence | MIT (c) 2024 Zhen Li -> vendor allowed; keep header + NOTICE |
| paper | Li et al., *Briefings in Bioinformatics* 27, bbag214 (2026), doi:10.1093/bib/bbag214 |
| torch_geometric use | `MessagePassing` GATConv copy, `ClusterData`/`ClusterLoader` (METIS mini-batches), `ToUndirected`, `dense_to_sparse` -> plain torch; METIS via `torch_sparse` only if installed, else a documented coordinate-bisection fallback |
| episcanpy use | `filter_features` / `filter_cells` (thin scanpy wrappers) -> scanpy calls |
| algorithm class | stochastic training (GPU) -> continuous embedding (4) + clustering (3) |

## Fixture

MISAR-seq E15.5-S1 (Jiang et al. 2023, CC-BY-4.0), MultiSP redistribution
`paper/fig2/spatial/data/misar_multisp/E15_5-S1_{expr,atac}_gt.h5ad` (sha256 in
`data/SOURCES.tsv`): 1,949 spots, 32,285 genes, 265,014 peaks (raw counts),
`obsm["spatial"]` = array (col,row). Labels: `obs["Annotation_for_Combined"]`
(12 classes, original authors' manual annotation).

Pipeline: upstream `PRESENT_function` (README "spatial multi-omics co-profiling"):
`adata_rna`, `adata_atac`, `gene_min_cells=1, num_hvg=3000, peak_min_cells_fraction=0.03,
nclusters=12, d_lat=50, k_neighbors=6, epochs=100, lr=1e-3, batch_size=320`, device CUDA.
Upstream does not seed; both arms call `PRESENT.Utils.setup_seed(s)` (port: the same
routine) right before the call. Seeds 0, 1, 2, 3, 4. GPU: H100 (job 48612073).

Domains: each arm's own output `obs["LeidenClusters"]` (upstream `run_leiden`
resolution search to 12 clusters). ARI/NMI against `Annotation_for_Combined`.

## Gates (same template as the other ports)

| id | kind | metric | pass if |
|---|---|---|---|
| L0 | exactness (reported) | plain-torch GATConv vs PRESENT GATConv, same seed, CPU: weights, forward (train+eval, dropout) and gradients | max abs diff == 0 |
| G0 | exactness (reported) | max \|emb_off_s - emb_port_s\| | <= 1e-4 every seed (not expected on GPU: CUDA scatter_add is non-deterministic) |
| G1 | exactness (reported) | ARI(domains port_s, domains off_s) | >= 0.95 every seed |
| G2 | required | median_{s!=t} ARI(port_s, off_t) vs median_{s!=t} ARI(off_s, off_t) | cross >= seed-to-seed - 0.05 |
| G3 | required | Procrustes similarity of embeddings, median_{s!=t}, cross vs seed-to-seed | cross >= seed-to-seed - 0.02 |
| G4 | required | mean ARI and mean NMI vs `Annotation_for_Combined` over seeds | \|port - off\| <= max(2 SD_off, 0.02) each |

Pass = G2 & G3 & G4. Runtime: wall time of the full call per seed, both arms, same GPU.

## Amendment A1 (2026-10-27 02:37, after the E15.5 gate run)

The E15.5 run above was completed (all required gates passed) before the coordinator
reserved E15.5 / E18.5 labels as the AEGLE held-out set; its labels were used only to
compare the port with the original, not to tune either. To avoid relying on held-out
labels, the same protocol (settings, seeds 0-4, gates and thresholds unchanged) is
repeated on the DEV section **E13.5-S1** (12 classes, nclusters = 12) and reported as
replication R1. The E15.5 result is kept as recorded.
