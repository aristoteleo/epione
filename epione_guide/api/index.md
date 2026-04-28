# API Reference

Public API of `epione`, organised by the v0.4 architecture.

The package is laid out along **two axes**: modality × scale at the
top of the tree (`bulk.atac`, `bulk.hic`, `single.atac`, `single.hic`),
plus stage-cutting cross-modality helpers (`pp` / `tl` / `pl`) and
purpose-driven infrastructure buckets (`io`, `upstream`, `core`,
`data`, `datasets`). New code should call the canonical paths shown
below; the v0.3 names (`epione.align`, `epione.hic`, `epione.sc_hic`,
`epione.utils.*`) still resolve via deprecation shims and will be
removed in v0.5.

## Modality × scale

Analysis tooling, organised by experiment type.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   epione.bulk.atac
   epione.bulk.hic
   epione.single.atac
   epione.single.hic
```

## Cross-modality

Stages that work on any modality once the data is loaded into
``AnnData`` / ``.cool`` / generic count-matrix form.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   epione.pp
   epione.tl
   epione.pl
```

## Infrastructure

Format I/O, pipeline orchestration, pure-Python utilities, and
reference / example data registries.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   epione.io
   epione.upstream
   epione.core
   epione.data
   epione.datasets
```

## Workflow guide

Common end-to-end workflows. Each row maps a task to the canonical
v0.4 entry points; deeper documentation lives on the per-module pages
linked above.

### Upstream — FASTQ to processed files

| Task | Functions |
|---|---|
| Verify CLI tools on PATH | {func}`epione.upstream.check_tools`, {func}`epione.upstream.tool_path`, {func}`epione.upstream.resolve_executable`, {func}`epione.upstream.build_env` |
| FASTQ trimming (fastp) | {func}`epione.upstream.trim_fastq_pair` |
| Reference genomes / indexes | {func}`epione.upstream.fetch_genome_fasta`, {func}`epione.upstream.fetch_genome_annotation`, {func}`epione.upstream.prepare_reference`, {func}`epione.upstream.ensure_aligner_index`, {func}`epione.upstream.ensure_chrom_sizes`, {func}`epione.upstream.ensure_fasta_index` |
| Aligners (CLI wrappers) | {mod}`epione.upstream.bowtie2`, {mod}`epione.upstream.bwa_mem2` |
| BAM ops | {func}`epione.upstream.sort_bam`, {func}`epione.upstream.index_bam`, {func}`epione.upstream.merge_bams`, {func}`epione.upstream.filter_bam` |
| BAM → coverage bigwig | {func}`epione.upstream.bam_to_bigwig` |
| Tn5 shift (ATAC) | {func}`epione.upstream.shift_atac_bam` |
| BAM → fragments | {func}`epione.upstream.bam_to_frags` |
| Bulk peak calling (MACS2) | {func}`epione.upstream.call_peaks_macs2` |
| Hi-C: BAM → pairs → .cool | {func}`epione.upstream.pairs_from_bam`, {func}`epione.upstream.pairs_to_cool` |
| Format helpers | {data}`epione.upstream.HIC_TOOLS`, {data}`epione.upstream.ATAC_TOOLS`, {data}`epione.upstream.RNA_TOOLS`, {data}`epione.upstream.MOTIF_TOOLS` |

### Preprocessing — load, QC, matrices

| Task | Functions |
|---|---|
| Import scATAC fragments | {func}`epione.pp.import_fragments`, {func}`epione.pp.ensure_tabix_index` |
| Per-cell QC | {func}`epione.pp.qc`, {func}`epione.pp.tss_enrichment` (alias `tsse`), {func}`epione.pp.nucleosome_signal`, {func}`epione.pp.frag_size_distr`, {func}`epione.pp.scrublet` |
| Concatenate samples | {func}`epione.pp.concat_samples` |
| Tile-matrix / peak-matrix / gene-activity matrix | {func}`epione.pp.add_tile_matrix`, {func}`epione.pp.make_peak_matrix`, {func}`epione.pp.make_gene_matrix` |
| Feature selection | {func}`epione.pp.select_features` |
| kNN graph | {func}`epione.pp.neighbors` |

### Embedding & clustering

| Task | Functions |
|---|---|
| LSI / iterative LSI | {func}`epione.tl.lsi`, {func}`epione.tl.iterative_lsi` |
| UMAP / t-SNE / PCA | {func}`epione.tl.umap`, {func}`epione.pl.umap`, {func}`epione.pl.tsne`, {func}`epione.pl.pca`, {func}`epione.pl.diffmap`, {func}`epione.pl.draw_graph` |
| Clustering (leiden / louvain / kmeans / RPH-kmeans / hierarchical) | {func}`epione.tl.clusters` |
| Marker features per cluster | {func}`epione.tl.find_marker_features` |
| Embedding scatter plot | {func}`epione.pl.embedding`, {func}`epione.pl.plot_embedding` |

### Multi-sample integration

| Task | Functions |
|---|---|
| Cross-batch integration | {func}`epione.tl.integrate`, {func}`epione.tl.joint_embedding` |
| CCA-based label transfer | {func}`epione.tl.transfer_labels` |

### chromVAR & motif analysis

| Task | Functions |
|---|---|
| Build / query motif database | {func}`epione.tl.build_motif_database`, {func}`epione.tl.query_motif_database` |
| Annotate peaks × motif matches | {func}`epione.tl.add_motif_matrix` |
| GC-matched background peaks | {func}`epione.tl.add_background_peaks` |
| chromVAR deviations / z-scores | {func}`epione.tl.compute_deviations` |
| Motif logo / matrix / clustering | {class}`epione.tl.MotifMatrix`, {class}`epione.tl.FormatMotifs`, {class}`epione.tl.ClusterMotifs`, {func}`epione.tl.format_motifs`, {func}`epione.tl.cluster_motifs` |
| Bulk motif enrichment (HOMER + pure-Python) | {func}`epione.bulk.atac.run_homer_motifs`, {func}`epione.bulk.atac.find_motifs_genome` |
| HOMER motif results table plot | {func}`epione.pl.homer_motif_table` |
| scATAC motif sequence annotation | {func}`epione.single.atac.add_dna_sequence`, {func}`epione.single.atac.match_motif` |

### Differential & peak-to-gene linkage

| Task | Functions |
|---|---|
| Differential peaks (DESeq2 / edgeR) | {func}`epione.tl.differential_peaks` |
| Volcano / MA / cumulative-distance plots | {func}`epione.pl.volcano`, {func}`epione.pl.ma_plot`, {func}`epione.pl.cumulative_distance` |
| ArchR-style peak-to-gene linkage | {func}`epione.tl.peak_to_gene`, {func}`epione.pl.plot_peak2gene` |
| Peak co-accessibility | {func}`epione.tl.coaccessibility` |
| Gene-activity score matrix | {func}`epione.tl.add_gene_score_matrix` |

### Per-barcode peak calling & pseudobulk

| Task | Functions |
|---|---|
| MACS3 per-cluster peak calling | {func}`epione.single.atac.macs3` |
| Merge per-sample peak sets | {func}`epione.single.atac.merge_peaks` |
| Pseudobulk (count-based) | {func}`epione.single.atac.pseudobulk` |
| Pseudobulk fragments → bigwig | {func}`epione.single.atac.pseudobulk_with_fragments` |
| Fragment file readers (Dask / parallel) | {func}`epione.single.atac.read_fragments_from_file`, {func}`epione.single.atac.read_fragments_with_dask_parallel` |
| Performance-backend helpers | {func}`epione.single.atac.check_performance_backends`, {func}`epione.single.atac.get_performance_recommendations`, {func}`epione.single.atac.install_performance_backend`, {func}`epione.single.atac.quick_install_pandarallel` |

### Footprint analysis (bulk ATAC)

| Task | Functions |
|---|---|
| ArchR-style footprint pipeline | {func}`epione.bulk.atac.footprint_archr`, {func}`epione.bulk.atac.bam_to_fragments_bulk` |
| TF aggregate footprint plot | {func}`epione.pl.plot_footprints` |
| Multi-scale footprint (scPrinter-style) | {func}`epione.pl.plot_multi_scale_footprint`, {func}`epione.pl.plot_multi_scale_footprint_region` |
| TOBIAS-style BINDetect (in-package port) | {class}`epione.tl.FootprintPlotter`, {class}`epione.tl.PlotTracks`, {func}`epione.tl.plot_tracks`, {func}`epione.tl.plot_heatmap` |

### BigWig matrix tooling (bulk)

| Task | Functions |
|---|---|
| BigWig collection class | {class}`epione.bulk.atac.bigwig`, {class}`epione.bulk.atac.plotloc` |
| TSS / TES / body matrix + plots | {func}`epione.bulk.atac.plot_matrix`, {func}`epione.bulk.atac.plot_matrix_line` |
| Score per BigWig bin (deeptools-style) | {func}`epione.bulk.atac.getScorePerBin`, {func}`epione.bulk.atac.mapReduce`, {func}`epione.bulk.atac.countReadsInRegions_wrapper`, {func}`epione.bulk.atac.countFragmentsInRegions_worker` |
| BigWig × gene table | {func}`epione.bulk.atac.gene_expression_from_bigwigs` |

### Hi-C — bulk

| Task | Functions |
|---|---|
| Build .cool from BAM | {func}`epione.upstream.pairs_from_bam`, {func}`epione.upstream.pairs_to_cool` |
| ICE balance | {func}`epione.bulk.hic.balance_cool` |
| Contact-matrix heatmap | {func}`epione.pl.plot_contact_matrix` |
| P(s) decay curve | {func}`epione.pl.plot_decay_curve` |
| Per-bin coverage / weight diagnostics | {func}`epione.pl.plot_coverage` |

### Hi-C — single-cell (scHiCluster)

| Task | Functions |
|---|---|
| Index per-cell .cool / .scool | {func}`epione.single.hic.load_cool_collection`, {func}`epione.single.hic.load_scool_cells` |
| scHiCluster impute (linear conv + RWR + top-k) | {func}`epione.single.hic.impute_cells`, {func}`epione.single.hic.impute_cell_chromosome` |
| Cell embedding (PCA on flattened imputed contacts) | {func}`epione.single.hic.embedding` |
| Cell scatter / per-cell heatmap | {func}`epione.pl.plot_embedding`, {func}`epione.pl.plot_cell_contacts` |

### TF networks (pySCENIC bridge)

| Task | Functions |
|---|---|
| Fragment filtering | {class}`epione.tl.FragmentFilter`, {func}`epione.tl.filter_fragments` |
| Network construction | {class}`epione.tl.NetworkBuilder`, {func}`epione.tl.create_network` |

### I/O — pure format readers / writers

| Task | Functions |
|---|---|
| GTF / GFF3 readers | {func}`epione.io.read_gtf`, {func}`epione.io.convert_gff_to_gtf`, {func}`epione.io.get_gene_annotation` |
| Peak / feature files | {func}`epione.io.read_features` |
| 10x ATAC matrix → AnnData | {func}`epione.io.read_ATAC_10x` |
| Pickle / h5ad cache helpers | {func}`epione.io.save`, {func}`epione.io.load`, {func}`epione.io.cached` |

### Core — pure-Python infrastructure

| Task | Functions |
|---|---|
| `Genome` class + dataset registry | {class}`epione.core.Genome`, {func}`epione.core.register_datasets` |
| Gene / annotation lookup | {class}`epione.core.Annotation`, {func}`epione.core.find_genes` |
| Peak ↔ region helpers | {func}`epione.core.distance_to_nearest_peak`, {func}`epione.core.filter_distal_peaks`, {func}`epione.core.classify_peaks_by_overlap`, {func}`epione.core.expression_matched_sample` |
| AnnData ↔ pandas shims | {func}`epione.core.obs_to_pandas`, {func}`epione.core.var_to_pandas` |
| Console / logger / PWMs / BED ops | {mod}`epione.core.console`, {mod}`epione.core.logger`, {mod}`epione.core.motifs`, {mod}`epione.core.regions`, {mod}`epione.core.utilities` |

### Data — reference registries

| Task | Functions |
|---|---|
| Pre-built `Genome` instances | {data}`epione.data.GRCh37`, {data}`epione.data.GRCh38`, {data}`epione.data.GRCm38`, {data}`epione.data.GRCm39`, {data}`epione.data.hg19`, {data}`epione.data.hg38`, {data}`epione.data.mm10`, {data}`epione.data.mm39` |

## Plotting helpers (overview)

| Plot | Function |
|---|---|
| Embedding scatter (PCA / UMAP / TSNE / Diffmap / draw_graph) | {func}`epione.pl.embedding`, {func}`epione.pl.umap`, {func}`epione.pl.pca`, {func}`epione.pl.tsne`, {func}`epione.pl.diffmap`, {func}`epione.pl.draw_graph`, {func}`epione.pl.plot_embedding` |
| Hi-C contact heatmap / decay / coverage / per-cell | {func}`epione.pl.plot_contact_matrix`, {func}`epione.pl.plot_decay_curve`, {func}`epione.pl.plot_coverage`, {func}`epione.pl.plot_cell_contacts` |
| Volcano / MA / cumulative distance | {func}`epione.pl.volcano`, {func}`epione.pl.ma_plot`, {func}`epione.pl.cumulative_distance` |
| Fragment QC | {func}`epione.pl.frag_size_distr`, {func}`epione.pl.fragment_histogram`, {func}`epione.pl.tss_enrichment`, {func}`epione.pl.plot_joint` |
| Footprint (per-TF, multi-scale) | {func}`epione.pl.plot_footprints`, {func}`epione.pl.plot_multi_scale_footprint`, {func}`epione.pl.plot_multi_scale_footprint_region` |
| Peak-to-gene arc plot | {func}`epione.pl.plot_peak2gene` |
| HOMER motif table | {func}`epione.pl.homer_motif_table` |
| Style / font setup | {func}`epione.pl.plot_set` |

## Deprecated paths (removed in v0.5)

These import paths still work in v0.4 but emit
{class}`DeprecationWarning`. Update tutorial / production code to the
canonical replacements.

| v0.3 path | v0.4 replacement |
|---|---|
| `epione.align` | {mod}`epione.upstream` |
| `epione.hic.balance_cool` | {func}`epione.bulk.hic.balance_cool` |
| `epione.hic.plot_contact_matrix` / `plot_decay_curve` / `plot_coverage` | {func}`epione.pl.plot_contact_matrix` etc. |
| `epione.hic.pairs_from_bam` / `pairs_to_cool` | {mod}`epione.upstream` |
| `epione.sc_hic.{load_cool_collection, load_scool_cells, impute_cells, impute_cell_chromosome, embedding}` | {mod}`epione.single.hic` |
| `epione.sc_hic.{plot_embedding, plot_cell_contacts}` | {mod}`epione.pl` |
| `epione.utils.{Genome, register_datasets, find_genes, Annotation, _sampling_helpers}` | {mod}`epione.core` |
| `epione.utils.{console, logger, motifs, regions, utilities, _compat}` | {mod}`epione.core` |
| `epione.utils.{read_gtf, read_ATAC_10x, read_features, get_gene_annotation, convert_gff_to_gtf, save, load, cached}` | {mod}`epione.io` |
| `epione.utils.{GRCh37, GRCh38, GRCm38, GRCm39, hg19, hg38, mm10, mm39}` | {mod}`epione.data` |
| `epione.utils.merge_peaks` | {func}`epione.single.atac.merge_peaks` |
