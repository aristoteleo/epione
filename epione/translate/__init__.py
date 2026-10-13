"""Cross-modality translation between scRNA-seq and scATAC-seq.

Learn a mapping from paired multiome cells and translate RNA profiles to
chromatin accessibility (``rna2atac``) or accessibility to expression
(``atac2rna``). Every method sits behind one interface::

    rna_p, atac_p = epi.translate.prepare(rna, atac)
    model = epi.translate.fit(rna_p[train], atac_p[train], method="scbutterfly")
    atac_pred = model.predict(rna_p[test])            # RNA -> ATAC
    rna_pred = model.predict(atac_p[test])            # ATAC -> RNA

and a scib-style benchmark scores them on the metrics of the scTranslation
benchmark (clustering, regression, distribution, classification)::

    res = epi.translate.benchmark({"rna2atac": {"scButterfly": atac_pred}},
                                  {"rna2atac": atac_p[test]}, labels="cell_type")
    epi.translate.summarize(res); epi.translate.plot_benchmark(res)

Available methods: ``"scbutterfly"``, ``"babel"`` and ``"scpair"`` (prepare
the inputs each expects with ``prepare(..., method=...)``).

Methods need PyTorch (``pip install epione[translate]``); preprocessing and
metrics do not.
"""
from ._prep import prepare, prepare_rna, prepare_atac, tfidf, chrom_blocks, peak_chromosomes, size_normalize
from ._prep import prepare_counts
from ._base import TranslationModel, available_methods, fit, load

from ._bench import METRICS, evaluate, benchmark, summarize, plot_benchmark
from . import _metrics as metrics

__all__ = [
    "prepare",
    "prepare_rna",
    "prepare_atac",
    "tfidf",
    "chrom_blocks",
    "peak_chromosomes",
    "TranslationModel",
    "available_methods",
    "size_normalize",
    "fit",
    "load",
    "METRICS",
    "evaluate",
    "benchmark",
    "summarize",
    "plot_benchmark",
    "metrics",
    "prepare_counts",
]
