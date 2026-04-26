"""Single-cell Hi-C analysis — companion to :mod:`epione.bulk.hic`.

Phase 1 covers scHiCluster (Zhou et al. 2019, PNAS): each per-cell
``.cool`` is densified by linear-conv + RWR + top-k, imputed matrices
are flattened into a cell × feature matrix, PCA gives the cell
embedding ready for scanpy ``neighbors`` / ``umap`` / ``leiden``
downstream.

Phase 2 (Chang 2024 Droplet Hi-C) adds the per-celltype reduction:
demux a multi-cell pairs file by barcode, build one ``.cool`` per
celltype, and quantify celltype × celltype similarity via cis
contact-vector correlation — the upstream of Chang 2024 Fig 1d/e/f.

Public API (flat — call as ``epi.single.hic.X``):
    * :func:`load_cool_collection`     — index per-cell ``.cool`` paths
    * :func:`load_scool_cells`         — index a multi-cell ``.scool`` bundle
    * :func:`impute_cell_chromosome`   — pure-numpy core (one matrix)
    * :func:`impute_cells`             — driver over a cell collection
    * :func:`embedding`                — PCA on flattened imputed contacts
    * :func:`demux_pairs_by_barcode`   — split multi-cell pairs by celltype
    * :func:`pseudobulk_by_celltype`   — demux + ``pairs_to_cool`` + balance
    * :func:`cluster_correlation`      — celltype × celltype Pearson r
"""
from __future__ import annotations

from ._io import load_cool_collection, load_scool_cells
from ._impute import impute_cell_chromosome, impute_cells
from ._embed import embedding
from ._demux import demux_pairs_by_barcode, pseudobulk_by_celltype
from ._correlation import cluster_correlation
# Plotting helpers live in :mod:`epione.pl` since v0.4 (PR 3); import
# them from there so ``epi.single.hic.plot_*`` still resolves.
from epione.pl._embedding import plot_embedding
from epione.pl._contact import plot_cell_contacts, plot_correlation_heatmap

__all__ = [
    "load_cool_collection",
    "load_scool_cells",
    "impute_cell_chromosome",
    "impute_cells",
    "embedding",
    "demux_pairs_by_barcode",
    "pseudobulk_by_celltype",
    "cluster_correlation",
    "plot_embedding",
    "plot_cell_contacts",
    "plot_correlation_heatmap",
]
