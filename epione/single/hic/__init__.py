"""Single-cell Hi-C analysis — companion to :mod:`epione.bulk.hic`.

Phase 1 (this module) covers scHiCluster (Zhou et al. 2019, PNAS):
each per-cell ``.cool`` is densified by linear-conv + RWR + top-k,
imputed matrices are flattened into a cell × feature matrix, PCA
gives the cell embedding ready for scanpy ``neighbors`` / ``umap`` /
``leiden`` downstream.

Public API (flat — call as ``epi.single.hic.X``):
    * :func:`load_cool_collection` — index per-cell ``.cool`` paths
    * :func:`load_scool_cells`     — index a multi-cell ``.scool`` bundle
    * :func:`impute_cell_chromosome` — pure-numpy core (one matrix)
    * :func:`impute_cells`         — driver over a cell collection
    * :func:`embedding`            — PCA on flattened imputed contacts
"""
from __future__ import annotations

from ._io import load_cool_collection, load_scool_cells
from ._impute import impute_cell_chromosome, impute_cells
from ._embed import embedding
from ._plot import plot_embedding, plot_cell_contacts

__all__ = [
    "load_cool_collection",
    "load_scool_cells",
    "impute_cell_chromosome",
    "impute_cells",
    "embedding",
    "plot_embedding",
    "plot_cell_contacts",
]
