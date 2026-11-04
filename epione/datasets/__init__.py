"""Example datasets for tutorials — fetch-on-demand, locally cached.

Mirrors ``scanpy.datasets`` style: each function downloads (or
returns the cached path of) a small / medium bio dataset suitable
for running an epione tutorial end-to-end. Cache lives under
``~/.cache/epione/datasets/``.

Phase-1 fetchers (added during the v0.4 refactor PR 4):

    * :func:`nagano2017_mES`         — 1 Mb scool of ~3.9k mouse-ES cells
      (Zenodo 3557682) + GEO cell-cycle metadata
    * :func:`drosophila_s2_hic`      — corrected Drosophila S2 Hi-C matrix
      (+ genes, histone bigWigs) from the Galaxy HiCExplorer lesson
      (Zenodo 16416373)
    * :func:`hicexplorer_test_matrices` — small Rao 2014/2017 ``.cool``
      matrices from HiCExplorer's test data (tag 3.7.6)
    * :func:`tobias_bcell_tcell`     — paired-end Bcell / Tcell
      ATAC bigwig + peak set (TOBIAS reference)
    * :func:`zhang2023_hippocampus_atac_rna` — human hippocampus spatial
      ATAC-RNA-seq, fragments + RNA + spatial folder (GEO GSE205055)
    * :func:`spatialglue_mouse_brain` — mouse brain spatial ATAC-RNA-seq as
      released with SpatialGlue (Zenodo 10362607)

Distinguished from :mod:`epione.data` (stable reference registries
like genomes / motif DBs) — *this* module is for one-off example
data anchored to specific tutorials.
"""
from __future__ import annotations

from ._archr import archr_hematopoiesis
from ._multiome import pbmc10k_multiome, pbmc10k_multiome_10x, pbmc3k_multiome_atac
from ._corces import corces2016_bulk_atac
from ._cellline_mix import granja2021_cellline_mix
from ._hic import drosophila_s2_hic, hicexplorer_test_matrices
from ._spatial import zhang2023_hippocampus_atac_rna, spatialglue_mouse_brain

__all__: list[str] = ["archr_hematopoiesis", "pbmc10k_multiome", "pbmc10k_multiome_10x", "pbmc3k_multiome_atac", "corces2016_bulk_atac",
                      "granja2021_cellline_mix", "drosophila_s2_hic", "hicexplorer_test_matrices",
                      "zhang2023_hippocampus_atac_rna", "spatialglue_mouse_brain"]
