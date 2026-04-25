"""Single-cell Hi-C analysis — scHiCluster-style imputation + cell embedding.

Companion to :mod:`epione.hic` (bulk Hi-C). Where bulk works on a single
deeply-sequenced ``.cool``, sc-Hi-C operates on a *collection* of per-cell
cools (one per nucleus, typically <100k contacts each), too sparse to run
TAD/compartment callers on directly. The standard fix — Zhou et al. 2019
(scHiCluster) — is two-stage:

1. **Impute** each cell with linear convolution + random-walk-with-restart
   to densify the contact matrix (per chromosome, at 1 Mb resolution).
2. **Embed**: flatten the imputed matrices into a cell × feature matrix,
   run PCA, then standard scanpy neighbours/UMAP/clustering downstream.

Phase 1 (this module) covers:
    * :func:`load_cool_collection` — index a directory of per-cell ``.cool``
    * :func:`impute_cells`         — scHiCluster linear conv + RWR + top-k
    * :func:`embedding`            — concat imputed contacts → PCA → AnnData
    * :func:`plot_embedding`       — UMAP / PCA scatter coloured by metadata
"""
from __future__ import annotations

from .io import load_cool_collection
from .impute import impute_cell_chromosome, impute_cells
from .embed import embedding
from .plot import plot_embedding, plot_cell_contacts

__all__ = [
    "load_cool_collection",
    "impute_cell_chromosome",
    "impute_cells",
    "embedding",
    "plot_embedding",
    "plot_cell_contacts",
]
