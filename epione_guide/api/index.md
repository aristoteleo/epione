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
| MAGIC imputation (ArchR `addImputeWeights` / `imputeMatrix`) | {func}`epione.tl.add_impute_weights`, {func}`epione.tl.impute_matrix` |
| Embedding scatter plot | {func}`epione.pl.embedding`, {func}`epione.pl.plot_embedding` |
| Gene-set module scores (ArchR `addModuleScore`) | {func}`epione.tl.add_module_score` |
| Label cross-tabulation (ArchR `confusionMatrix`) | {func}`epione.tl.confusion_matrix`, {func}`epione.pl.plot_confusion_matrix` |
| Per-group ridge / violin plots (ArchR `plotGroups`) | {func}`epione.pl.plot_groups` |

### Multi-sample integration

| Task | Functions |
|---|---|
| Cross-batch integration | {func}`epione.tl.integrate`, {func}`epione.tl.joint_embedding` |
| CCA-based label transfer | {func}`epione.tl.transfer_labels` |
| Harmony batch correction (ArchR `addHarmony`) | {func}`epione.tl.harmony`, {func}`epione.tl.run_harmony` |

### chromVAR & motif analysis

| Task | Functions |
|---|---|
| Build / query motif database | {func}`epione.tl.build_motif_database`, {func}`epione.tl.query_motif_database` |
| Annotate peaks × motif matches | {func}`epione.tl.add_motif_matrix` |
| GC-matched background peaks | {func}`epione.tl.add_background_peaks` |
| chromVAR deviations / z-scores | {func}`epione.tl.compute_deviations` |
| Motif variability ranking (ArchR `getVarDeviations`) | {func}`epione.tl.var_deviations`, {func}`epione.pl.var_dev` |
| Correlate two matrices over kNN groups (ArchR `correlateMatrices`) | {func}`epione.tl.correlate_matrices`, {func}`epione.tl.group_means` |
| Positive TF regulators (ArchR book recipe) | {func}`epione.tl.positive_tf_regulators`, {func}`epione.pl.tf_regulators` |
| Peak annotations (custom BED sets / ArchR ENCODE, ATAC, Codex databases) | {func}`epione.tl.add_peak_annotations`, {func}`epione.tl.add_archr_annotations` |
| Motif / annotation enrichment in marker peaks (ArchR `peakAnnoEnrichment`) | {func}`epione.tl.peak_anno_enrichment`, {func}`epione.tl.custom_enrichment`, {func}`epione.pl.enrich_heatmap` |
| Motif logo / matrix / clustering | {class}`epione.tl.MotifMatrix`, {class}`epione.tl.FormatMotifs`, {class}`epione.tl.ClusterMotifs`, {func}`epione.tl.format_motifs`, {func}`epione.tl.cluster_motifs` |
| Bulk motif enrichment (HOMER + pure-Python) | {func}`epione.bulk.atac.run_homer_motifs`, {func}`epione.bulk.atac.find_motifs_genome` |
| HOMER motif results table plot | {func}`epione.pl.homer_motif_table` |
| scATAC motif sequence annotation | {func}`epione.single.atac.add_dna_sequence`, {func}`epione.single.atac.match_motif` |

### SCENIC+ enhancer GRNs (`epione.scenic`)

| Task | Functions |
|---|---|
| cisTarget motif enrichment of region sets (pycistarget `cisTarget`) | {func}`epione.scenic.run_cistarget`, {func}`epione.scenic.recovery_aucs`, {class}`epione.scenic.MotifEnrichmentResult` |
| Differentially enriched motifs (pycistarget `DEM`) | {func}`epione.scenic.run_dem`, {func}`epione.scenic.dem_test`, {func}`epione.scenic.foreground_background_regions`, {func}`epione.scenic.p_adjust_bh` |
| cisTarget rankings / scores databases | {class}`epione.scenic.CisTargetDB`, {func}`epione.scenic.fetch_cistarget_database`, {func}`epione.scenic.target_to_query`, {class}`epione.scenic.RegionIndex`, {func}`epione.scenic.read_bed_regions` |
| Motif → TF annotation and cistromes | {func}`epione.scenic.load_motif_annotations`, {func}`epione.scenic.fetch_motif_annotation`, {func}`epione.scenic.cistromes_per_region_set`, {func}`epione.scenic.jaspar_annotation` |
| Cistromes for the eGRN step (scenicplus `prepare_menr`) and pycistarget hdf5 I/O | {func}`epione.scenic.merge_cistromes`, {func}`epione.scenic.write_hdf5`, {func}`epione.scenic.read_hdf5` |
| Database from Epione's MOODS + JASPAR scan | {func}`epione.scenic.motif_score_db`, {func}`epione.scenic.scores_to_rankings` |

### Differential & peak-to-gene linkage

| Task | Functions |
|---|---|
| Differential peaks (DESeq2 / edgeR) | {func}`epione.tl.differential_peaks` |
| Volcano / MA / cumulative-distance plots | {func}`epione.pl.volcano`, {func}`epione.pl.ma_plot`, {func}`epione.pl.cumulative_distance` |
| ArchR-style peak-to-gene linkage | {func}`epione.tl.peak_to_gene`, {func}`epione.pl.plot_peak2gene` |
| ArchR marker / peak-to-gene heatmaps | {func}`epione.pl.marker_heatmap`, {func}`epione.pl.marker_heatmap_matrix`, {func}`epione.pl.peak2gene_heatmap`, {func}`epione.pl.prepare_peak2gene_heatmap` |
| Peak co-accessibility | {func}`epione.tl.coaccessibility` |
| Gene-activity score matrix | {func}`epione.tl.add_gene_score_matrix` |
| ArchR-style pseudo-time trajectory | {func}`epione.tl.add_trajectory`, {func}`epione.tl.get_trajectory`, {func}`epione.tl.trajectory_heatmap_matrix`, {func}`epione.tl.correlate_trajectories`, {func}`epione.pl.plot_trajectory`, {func}`epione.pl.plot_trajectory_heatmap` |

### SCENIC+ enhancer-driven GRNs

| Task | Functions |
|---|---|
| Fragment counts in regions (pycisTopic `create_cistopic_object_from_fragments`) | {func}`epione.scenic.cistopic_matrix` |
| cisTopic LDA topic models + model selection (pycisTopic `run_cgs_models`, `evaluate_models`) | {func}`epione.scenic.run_topic_models`, {func}`epione.scenic.evaluate_topic_models`, {func}`epione.scenic.add_topic_model`, {class}`epione.scenic.TopicModel` |
| Topic binarization (pycisTopic `binarize_topics`) | {func}`epione.scenic.binarize_topics` |
| Imputed accessibility, highly variable regions, DARs (pycisTopic `diff_features`) | {func}`epione.scenic.impute_accessibility`, {func}`epione.scenic.normalize_scores`, {func}`epione.scenic.find_highly_variable_features`, {func}`epione.scenic.find_diff_features`, {class}`epione.scenic.ImputedFeatures` |
| Region sets → BED for cisTarget / DEM | {func}`epione.scenic.write_region_sets` |
| eRegulon inference, GSEA approach (SCENIC+ `infer_grn`) | {func}`epione.scenic.build_egrn` |
| TF-region-gene triplet ranking / cisTarget motif ranks | {func}`epione.scenic.triplet_rank`, {func}`epione.scenic.tf_region_max_rank` |
| AUCell eRegulon scores (rank once, score many) | {func}`epione.scenic.rank_cells`, {func}`epione.scenic.aucell`, {func}`epione.scenic.score_eregulons`, {func}`epione.scenic.eregulon_signatures` |
| Regulon specificity scores (RSS) | {func}`epione.scenic.regulon_specificity_scores` |

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

### SCENIC+ (enhancer-driven GRNs)

| Task | Functions |
|---|---|
| Gene annotation for the search space (Ensembl GTF) | {func}`epione.scenic.gene_annotation_from_gtf` |
| Region-to-gene search space | {func}`epione.scenic.get_search_space` |
| Region-to-gene links (GBM / RF importance + correlation) | {func}`epione.scenic.region_to_gene` |
| TF-to-gene links (GRNBoost2 + correlation) | {func}`epione.scenic.tf_to_gene` |

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
| Example datasets | {func}`epione.datasets.archr_hematopoiesis` (ArchR Hematopoiesis tutorial fragments) |

## Plotting helpers (overview)

| Plot | Function |
|---|---|
| Embedding scatter (PCA / UMAP / TSNE / Diffmap / draw_graph) | {func}`epione.pl.embedding`, {func}`epione.pl.umap`, {func}`epione.pl.pca`, {func}`epione.pl.tsne`, {func}`epione.pl.diffmap`, {func}`epione.pl.draw_graph`, {func}`epione.pl.plot_embedding` |
| Hi-C contact heatmap / decay / coverage / per-cell | {func}`epione.pl.plot_contact_matrix`, {func}`epione.pl.plot_decay_curve`, {func}`epione.pl.plot_coverage`, {func}`epione.pl.plot_cell_contacts` |
| Volcano / MA / cumulative distance | {func}`epione.pl.volcano`, {func}`epione.pl.ma_plot`, {func}`epione.pl.cumulative_distance` |
| Fragment QC | {func}`epione.pl.frag_size_distr`, {func}`epione.pl.fragment_histogram`, {func}`epione.pl.tss_enrichment`, {func}`epione.pl.plot_joint` |
| Footprint (per-TF, multi-scale) | {func}`epione.pl.plot_footprints`, {func}`epione.pl.plot_multi_scale_footprint`, {func}`epione.pl.plot_multi_scale_footprint_region` |
| Peak-to-gene arc plot | {func}`epione.pl.plot_peak2gene` |
| Grouped genome browser track (ArchR `plotBrowserTrack`) | {func}`epione.pl.plot_browser_track`, {func}`epione.pl.browser_track_coverage` |
| Marker / peak-to-gene heatmaps (ArchR) | {func}`epione.pl.marker_heatmap`, {func}`epione.pl.peak2gene_heatmap` |
| Peak-annotation enrichment heatmap | {func}`epione.pl.enrich_heatmap` |
| HOMER motif table | {func}`epione.pl.homer_motif_table` |
| Group distributions / confusion heatmap | {func}`epione.pl.plot_groups`, {func}`epione.pl.plot_confusion_matrix` |
| Motif variability / positive TF regulators | {func}`epione.pl.var_dev`, {func}`epione.pl.tf_regulators` |
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
