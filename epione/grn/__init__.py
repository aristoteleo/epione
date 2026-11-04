"""Gene regulatory network inference.

One home for Epione's GRN methods:

* :func:`pando` — Pando (Fleck et al., *Nature* 2023): per-gene regression of
  expression on TF x binding-site accessibility, through the omicverse port
  py-Pando (``pip install pypando``).
* ``epione.grn.scenic`` — the SCENIC+ pipeline (enhancer-driven eRegulons). It
  stays importable as :mod:`epione.scenic`; this is the same module.
"""
from ._pando import pando, pando_annotation, seurat_lognormalize, signac_tfidf


def __getattr__(name):
    if name == "scenic":
        import importlib

        return importlib.import_module("epione.scenic")
    raise AttributeError(f"module 'epione.grn' has no attribute {name!r}")


__all__ = ["pando", "pando_annotation", "seurat_lognormalize", "signac_tfidf", "scenic"]
