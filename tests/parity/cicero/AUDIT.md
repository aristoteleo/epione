## R function coverage audit

### Coverage summary

| Category | Ported | Total | % |
|---|---|---|---|
| Exported R functions | 10 | 19 | 52.6% |
| Internal helpers (reachable) | 1 | 26 | 3.8% |

_Python package exposes 55 unique names._

### Exported R functions

| R function | Python equivalent | Status |
|---|---|---|
| `aggregate_by_cell_bin` | `—` | ❌ MISSING |
| `aggregate_nearby_peaks` | `—` | ❌ MISSING |
| `annotate_cds_by_site` | `annotate_cds_by_site` | ✅ ported |
| `assemble_connections` | `assemble_connections` | ✅ ported |
| `build_gene_activity_matrix` | `build_gene_activity_matrix` | ✅ ported |
| `compare_connections` | `—` | ❌ MISSING |
| `df_for_coords` | `—` | ❌ MISSING |
| `estimate_distance_parameter` | `estimate_distance_parameter` | ✅ ported |
| `find_overlapping_ccans` | `find_overlapping_ccans` | ✅ ported |
| `find_overlapping_coordinates` | `—` | ❌ MISSING |
| `generate_ccans` | `generate_ccans` | ✅ ported |
| `generate_cicero_models` | `generate_cicero_models` | ✅ ported |
| `make_atac_cds` | `—` | ❌ MISSING |
| `make_cicero_cds` | `make_cicero_cds` | ✅ ported |
| `normalize_gene_activities` | `normalize_gene_activities` | ✅ ported |
| `plot_accessibility_in_pseudotime` | `—` | ❌ MISSING |
| `plot_connections` | `—` | ❌ MISSING |
| `ranges_for_coords` | `—` | ❌ MISSING |
| `run_cicero` | `run_cicero` | ✅ ported |

### Internal helpers reachable from exports

| R helper | File | Python equivalent | Status |
|---|---|---|---|
| `CustomTrack` | `fromGviz.R` | `—` | 🔸 missing-or-inlined |
| `agg_cells` | `aggregate.R` | `—` | 🔸 missing-or-inlined |
| `build_composite_gene_activity_matrix` | `activityScores.R` | `—` | 🔸 missing-or-inlined |
| `calc_dist_matrix` | `runCicero.R` | `—` | 🔸 missing-or-inlined |
| `find_ccan_cutoff` | `runCicero.R` | `—` | 🔸 missing-or-inlined |
| `find_distance_parameter` | `runCicero.R` | `—` | 🔸 missing-or-inlined |
| `generate_plotting_subset` | `plotting.R` | `—` | 🔸 missing-or-inlined |
| `generate_windows` | `runCicero.R` | `generate_windows` | ✅ ported |
| `get_colors` | `plotting.R` | `—` | 🔸 missing-or-inlined |
| `get_genomic_range` | `runCicero.R` | `—` | 🔸 missing-or-inlined |
| `get_rho_mat` | `runCicero.R` | `—` | 🔸 missing-or-inlined |
| `is_chr` | `utils.R` | `—` | 🔸 missing-or-inlined |
| `is_color` | `utils.R` | `—` | 🔸 missing-or-inlined |
| `make_bin_col` | `aggregate.R` | `—` | 🔸 missing-or-inlined |
| `make_ccan_graph` | `runCicero.R` | `—` | 🔸 missing-or-inlined |
| `make_gene_model_track` | `plotting.R` | `—` | 🔸 missing-or-inlined |
| `make_peak_track` | `plotting.R` | `—` | 🔸 missing-or-inlined |
| `make_sparse_matrix` | `utils.R` | `—` | 🔸 missing-or-inlined |
| `monocle_theme_opts` | `plotting.R` | `—` | 🔸 missing-or-inlined |
| `number_of_ccans` | `runCicero.R` | `—` | 🔸 missing-or-inlined |
| `plotBedpe` | `plotting.R` | `—` | 🔸 missing-or-inlined |
| `plotpair` | `plotting.R` | `—` | 🔸 missing-or-inlined |
| `reconcile` | `runCicero.R` | `—` | 🔸 missing-or-inlined |
| `signif_up` | `plotting.R` | `—` | 🔸 missing-or-inlined |
| `sparse_to_datatable` | `aggregate.R` | `—` | 🔸 missing-or-inlined |
| `split_peak_names` | `utils.R` | `—` | 🔸 missing-or-inlined |
