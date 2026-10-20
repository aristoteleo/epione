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

Higashi (Zhang, Zhou & Ma 2022, *Nat. Biotechnol.*) is available as an
alternative imputation / embedding method: ``impute_cells(...,
method="higashi")`` and ``embedding(..., method="higashi")`` train a
PyTorch port of ma-compbio/Higashi (MIT) on all cells jointly — a
hypergraph of cell and bin nodes scored by Hyper-SAGNN, a cell-dependent
GNN, and k-nearest-cell imputation. It needs ``torch`` (a GPU is
recommended). Defaults remain scHiCluster + PCA.

Public API (flat — call as ``epi.single.hic.X``):
    * :func:`load_cool_collection`     — index per-cell ``.cool`` paths
    * :func:`load_scool_cells`         — index a multi-cell ``.scool`` bundle
    * :func:`impute_cell_chromosome`   — pure-numpy core (one matrix)
    * :func:`impute_cells`             — driver over a cell collection
                                         (``method="schicluster"`` | ``"higashi"``)
    * :func:`embedding`                — PCA on flattened imputed contacts
                                         (``method="pca"`` | ``"higashi"``)
    * :func:`demux_pairs_by_barcode`   — split multi-cell pairs by celltype
    * :func:`pseudobulk_by_celltype`   — demux + ``pairs_to_cool`` + balance
    * :func:`cluster_correlation`      — celltype × celltype Pearson r

Compartments from raw contacts (scHiCluster + Human Cell Epigenome Atlas):
    * :func:`cpg_ratio`                — per-bin CpG density (phasing track)
    * :func:`contacts_to_scool`        — per-cell contact tables → raw ``.scool``
    * :func:`load_contact_tables`      — index contact tables without binning
    * :func:`compartment_scores`       — scHiCluster single-cell score
    * :func:`merge_cells`              — raw pseudobulk ``.cool`` per group
    * :func:`compartment_model`        — fit the HCEA PCA model on merged contacts
    * :func:`project_compartments`     — score each pseudobulk on that model
    * :func:`compartment_saddle`       — saddle matrix + AA/BB/AB/BA strength
    * :func:`oe_correlation`           — O/E and correlation matrix of one chromosome
"""
from __future__ import annotations

from ._io import load_cool_collection, load_scool_cells
from ._impute import impute_cell_chromosome, impute_cells
from ._embed import embedding
from ._demux import demux_pairs_by_barcode, pseudobulk_by_celltype
from ._correlation import cluster_correlation, cell_celltype_correlation
from ._scgad import scgad_score, load_refflat_genes
from ._compartment import (
    cpg_ratio,
    contacts_to_scool,
    load_contact_tables,
    compartment_scores,
    merge_cells,
    compartment_model,
    project_compartments,
    compartment_saddle,
    oe_correlation,
    write_bedgraph,
)
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
    "cell_celltype_correlation",
    "scgad_score",
    "load_refflat_genes",
    "cpg_ratio",
    "contacts_to_scool",
    "load_contact_tables",
    "compartment_scores",
    "merge_cells",
    "compartment_model",
    "project_compartments",
    "compartment_saddle",
    "oe_correlation",
    "write_bedgraph",
    "plot_embedding",
    "plot_cell_contacts",
    "plot_correlation_heatmap",
]
