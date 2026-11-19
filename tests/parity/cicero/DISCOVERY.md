## Discovery — dependencies of `cicero-ref`

Reusing omicverse-org Python ports avoids duplicating upstream work.

### Imports

| R dep | omicverse match | Description |
|---|---|---|
| `assertthat` | — | (none) |
| `Biobase` | — | (none) |
| `BiocGenerics` | — | (none) |
| `data.table` | — | (none) |
| `dplyr` | — | (none) |
| `FNN` | — | (none) |
| `GenomicRanges` | — | (none) |
| `ggplot2` | — | (none) |
| `glasso` | — | (none) |
| `grDevices` | — | (none) |
| `igraph` | — | (none) |
| `IRanges` | — | (none) |
| `Matrix` | [`anndata-oom`](https://github.com/omicverse/anndata-oom) | Out-of-memory AnnData powered by Rust — a drop-in replacement for anndata.AnnDat… |
| `methods` | [`py-monocle2`](https://github.com/omicverse/py-monocle2) / [`py-dada2`](https://github.com/omicverse/py-dada2) | A pure-Python re-implementation of Monocle 2 (Qiu et al., Nature Methods 2017) f…; A pure-Python re-implementation of DADA2 (Callahan et al., Nature Methods 2016) … |
| `parallel` | — | (none) |
| `plyr` | — | (none) |
| `reshape2` | — | (none) |
| `S4Vectors` | — | (none) |
| `stats` | — | (none) |
| `stringi` | — | (none) |
| `stringr` | — | (none) |
| `tibble` | — | (none) |
| `tidyr` | — | (none) |
| `VGAM` | — | (none) |
| `utils` | — | (none) |

### Depends

| R dep | omicverse match | Description |
|---|---|---|
| `monocle3` | [`py-Monocle3`](https://github.com/omicverse/py-Monocle3) | omicverse wrapper for Bio-Babel/Monocle3-python (R monocle3 port). Re-exports th… |
| `Gviz` | — | (none) |

### Action items

- **3** of **27** R Imports/Depends/LinkingTo have a omicverse-org Python mirror.
- Add each match to `pyproject.toml::dependencies`.
- For deps **without** a mirror: either pure-Python equivalent in scipy/sklearn/statsmodels, OR scope as a future omicverse port.
### Decisions (Cicero for Monocle 3, cole-trapnell-lab/cicero-release@monocle3, v1.3.9)

Target check: `discover_omicverse_deps --check cicero` found no port under `omicverse` (110 repos)
or `aristoteleo` (150 repos). The port lives **inside epione** (`epione.tl`), not as a `py-cicero` wheel.

| R dep | Decision |
|---|---|
| `monocle3` (`py-Monocle3` exists) | **not used.** Cicero touches monocle3 only through `new_cell_data_set`, `detect_genes` and `estimate_size_factors` (5 lines). These are re-implemented on AnnData (cells × peaks) so epione does not gain a dependency. |
| `Matrix` / `methods` (false-positive matches) | scipy.sparse / plain Python. |
| `glasso` (Fortran, Friedman 2008) | **line-by-line port** of `glasso.f` 1.11 (`connect`/`row`/`lasinv1`/`setup`/`lasso`/`fatmul`/`inv`), numba-compiled. `sklearn.covariance.graphical_lasso` is NOT a substitute: scalar penalty only and a different solver / stopping rule. |
| `FNN::knn.index` (ANN kd-tree, exact) | `scipy.spatial.cKDTree.query` (exact Euclidean kNN, self excluded). |
| R RNG (`sample`) | `epione.tl._rrng.RRandom` (already in epione; bit-exact Mersenne-Twister + rejection sampling). |
| `stats::cov`, `cov2cor`, `mean` | numpy, with R's long-double two-pass `mean` emulated where it feeds a decision. |
| `igraph::cluster_louvain` | `python-igraph` `community_multilevel` (same C core; RNG differs → clustering-class gate). |
| `GenomicRanges` / `IRanges` | interval arithmetic in numpy / pandas (only overlap and range-merge are used). |
| `data.table`, `dplyr`, `plyr`, `reshape2`, `tibble`, `tidyr`, `stringr`, `stringi`, `assertthat` | pandas / numpy / plain Python. |
| `Gviz`, `ggplot2`, `grDevices` | plotting only (`plot_connections`, `plot_accessibility_in_pseudotime`): out of scope for this port. |
| `VGAM`, `Biobase`, `BiocGenerics`, `S4Vectors`, `parallel`, `utils` | not needed (`mclapply(mc.cores = 1)` → optional process pool). |
