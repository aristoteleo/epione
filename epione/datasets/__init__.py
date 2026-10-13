"""Example datasets for tutorials — fetch-on-demand, locally cached.

Mirrors ``scanpy.datasets`` style: each function downloads (or
returns the cached path of) a small / medium bio dataset suitable
for running an epione tutorial end-to-end. Cache lives under
``~/.cache/epione/datasets/``.

Phase-1 fetchers (added during the v0.4 refactor PR 4):

    * :func:`nagano2017_mES`         — 1 Mb scool of ~3.9k mouse-ES cells
      (Zenodo 3557682) + GEO cell-cycle metadata
    * :func:`drosophila_s2_hic`      — 10 kb Drosophila S2 cool from
      Galaxy HiCExplorer training (Zenodo 16416373)
    * :func:`tobias_bcell_tcell`     — paired-end Bcell / Tcell
      ATAC bigwig + peak set (TOBIAS reference)

Distinguished from :mod:`epione.data` (stable reference registries
like genomes / motif DBs) — *this* module is for one-off example
data anchored to specific tutorials.
"""
from __future__ import annotations

from ._archr import archr_hematopoiesis
from ._multiome import pbmc10k_multiome, pbmc10k_multiome_10x
from ._corces import corces2016_bulk_atac

__all__: list[str] = ["archr_hematopoiesis", "pbmc10k_multiome", "pbmc10k_multiome_10x", "corces2016_bulk_atac"]
