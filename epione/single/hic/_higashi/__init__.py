"""Higashi (Zhang, Zhou & Ma, *Nat. Biotechnol.* 2022) for single-cell Hi-C.

A PyTorch adaptation of https://github.com/ma-compbio/Higashi (MIT License,
Copyright (c) 2020 Ma Lab at CMU; full notice in ``LICENSE_HIGASHI``):
hypergraph representation of a scHi-C dataset (cell node + two bin nodes per
contact), Hyper-SAGNN hyperedge scoring with a ZINB likelihood, a
cell-dependent graph neural network and k-nearest-cell imputation.

Dropped upstream dependencies: bokeh / umap / seaborn / fbpca / Cython (the
visualisation app and unused analysis helpers) and the on-disk ``temp_dir``
pipeline. Used through :func:`epione.single.hic.impute_cells` and
:func:`epione.single.hic.embedding` with ``method="higashi"``.
"""
from ._process import HigashiInput, process_contacts, contacts_from_cools
from ._model import Higashi

__all__ = ["HigashiInput", "process_contacts", "contacts_from_cools", "Higashi"]
