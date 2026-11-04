"""Pando gene regulatory networks on Epione data.

`Pando <https://github.com/quadbio/Pando>`_ (Fleck et al., *Nature* 2023)
regresses each target gene's expression on TF expression x accessibility of
TF-binding peaks near the gene. The algorithm lives in the omicverse port
`py-Pando <https://github.com/omicverse/py-Pando>`_ (``pip install pypando``),
which is parity-tested against R Pando 1.1.1. This module only connects it to
Epione objects:

* RNA / ATAC AnnData (or a MuData with ``rna`` and ``atac``), with Epione's
  ``chr:start-end`` peak names;
* the normalisations Pando expects (Seurat ``LogNormalize`` for RNA, Signac
  ``RunTFIDF`` for peaks), computed here when raw counts are passed;
* a Signac-style gene annotation built from Epione's genome registry
  (GENCODE GFF3), and the genome FASTA from the same registry.

Motif scanning stays in py-Pando: Pando scans exon-subtracted candidate
regions with ``TFBSTools::toPWM`` matrices against the regions' own base
composition, which :func:`epione.tl.add_motif_matrix` (whole peaks, MOODS
log-odds, even background) does not reproduce.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["pando", "pando_annotation", "signac_tfidf", "seurat_lognormalize"]

# Signac::GetGRangesFromEnsDb default biotypes (GENCODE calls lincRNA "lncRNA")
_BIOTYPES = ("protein_coding", "lncRNA", "lincRNA", "rRNA", "processed_transcript")


def _require_pypando():
    try:
        import pypando
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "epione.grn.pando needs py-Pando: pip install pypando (or pip install 'epione[pando]')"
        ) from err
    return pypando


def signac_tfidf(counts, scale_factor: float = 1e4):
    """``Signac::RunTFIDF(method = 1)`` on a cells x peaks count matrix.

    ``log1p(count / cell_total * n_cells / peak_total * scale_factor)``,
    computed on the non-zeros. Returns a CSR matrix.
    """
    X = sp.csr_matrix(counts, dtype=np.float64)
    cell_tot = np.asarray(X.sum(axis=1)).ravel()
    peak_tot = np.asarray(X.sum(axis=0)).ravel()
    idf = X.shape[0] / peak_tot
    X = sp.csr_matrix(sp.diags(1.0 / cell_tot) @ X)  # tf
    X = sp.csr_matrix(X @ sp.diags(idf))
    X.data = np.log1p(X.data * scale_factor)
    return X


def seurat_lognormalize(counts, scale_factor: float = 1e4):
    """``Seurat::NormalizeData(normalization.method = "LogNormalize")``."""
    X = sp.csr_matrix(counts, dtype=np.float64)
    tot = np.asarray(X.sum(axis=1)).ravel()
    X = sp.csr_matrix(sp.diags(1.0 / tot) @ X)
    X.data = np.log1p(X.data * scale_factor)
    return X


def _looks_like_counts(X) -> bool:
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    sample = data[: min(len(data), 100_000)]
    return bool(len(sample)) and np.all(sample >= 0) and np.allclose(sample, np.round(sample))


def pando_annotation(genome="hg38", biotypes: Sequence[str] = _BIOTYPES) -> pd.DataFrame:
    """Signac-style annotation (``Annotation(object)``) from a GENCODE GFF3/GTF.

    ``genome`` is an Epione genome name (``"hg38"``, ``"mm10"``, ...), a
    :class:`epione.core.genome.Genome`, or a path to a GFF3/GTF file. Rows are
    the exon / CDS / UTR features of genes of the given ``biotypes``, with
    columns ``seqnames, start, end, strand, tx_id, gene_name, gene_id,
    gene_biotype, type`` (``type`` in ``exon``, ``cds``, ``utr``) — what
    Pando reads: the gene extent (union of a gene's rows) and the exons.
    Parsing is done by ``pypando.annotation_from_gtf``.
    """
    pp = _require_pypando()
    return pp.annotation_from_gtf(_annotation_path(genome), biotypes=biotypes)


def _annotation_path(genome):
    if isinstance(genome, (str, os.PathLike)) and os.path.exists(str(genome)):
        return str(genome)
    return str(_genome_obj(genome).annotation)


def _genome_obj(genome):
    if isinstance(genome, str):
        from epione.utils import genome as registry

        g = getattr(registry, genome, None)
        if g is None:
            raise ValueError(f"unknown genome {genome!r}; pass a Genome object or a file path")
        return g
    return genome


def _fasta_path(genome):
    if isinstance(genome, (str, os.PathLike)) and os.path.exists(str(genome)):
        return str(genome)
    return str(_genome_obj(genome).fasta)


def _signac_peak_names(names) -> np.ndarray:
    out = []
    for n in map(str, names):
        m = re.match(r"^(.+)[:_-](\d+)[-_](\d+)$", n)
        if m is None:
            raise ValueError(f"cannot parse peak name {n!r}; expected 'chrom:start-end'")
        out.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    return np.asarray(out, dtype=object)


def pando(
    rna,
    atac=None,
    *,
    genome="hg38",
    annotation: Optional[pd.DataFrame] = None,
    genes: Optional[Sequence[str]] = None,
    rna_layer: Optional[str] = None,
    peak_layer: Optional[str] = None,
    normalize: bool = True,
    regions: Optional[pd.DataFrame] = None,
    exclude_exons: bool = True,
    pfm=None,
    motif_tfs: Optional[pd.DataFrame] = None,
    motif_bg="subject",
    p_cutoff: float = 5e-05,
    method: str = "glm",
    peak_to_gene_method: str = "Signac",
    upstream: int = 100000,
    downstream: int = 0,
    extend: int = 1000000,
    tf_cor: float = 0.1,
    peak_cor: float = 0.0,
    interaction_term: str = ":",
    alpha: float = 0.5,
    seed: Optional[int] = None,
    scan: str = "near_targets",
    find_modules: bool = True,
    p_thresh: float = 0.05,
    rsq_thresh: float = 0.1,
    nvar_thresh: int = 10,
    min_genes_per_module: int = 5,
    key_added: Optional[str] = "pando",
    verbose: bool = True,
    **infer_kwargs,
):
    """Infer a Pando GRN from paired RNA and ATAC.

    Runs py-Pando's ``initiate_grn`` → ``find_motifs`` → ``infer_grn``
    (→ ``find_modules``) with Pando's defaults.

    Parameters
    ----------
    rna, atac
        Cells x genes and cells x peaks AnnData of the same cells, or a MuData
        (``rna`` with modalities ``"rna"`` and ``"atac"``) with ``atac=None``.
        Peak names ``chr:start-end`` or ``chr-start-end``.
    genome
        Epione genome (``"hg38"``, ``"mm10"``, a ``Genome``) or a FASTA path;
        the FASTA is scanned for motifs, the matching GENCODE GFF3 gives the
        default ``annotation``.
    annotation
        Signac-style annotation; default :func:`pando_annotation` of ``genome``.
    genes
        Target genes; default the RNA's ``var["highly_variable"]`` genes.
    rna_layer, peak_layer
        Layers holding the matrices to use (default ``X``).
    normalize
        If the chosen matrices hold raw counts, apply ``LogNormalize`` (RNA)
        and Signac ``RunTFIDF`` (peaks) first — the ``data`` layers Pando reads.
    regions
        Optional candidate regions (e.g. ``pypando.data.phastcons_hg38()``).
    pfm, motif_tfs
        Motifs (``pypando.PFM``) and motif-to-TF table; default Pando's bundled
        collection (JASPAR 2020 + CIS-BP, 1,590 motifs).
    method, ...
        Passed to ``pypando.infer_grn`` (``"glm"``, ``"glmnet"``,
        ``"cv.glmnet"``, ``"xgb"``, ``"bagging_ridge"``, ``"bayesian_ridge"``).
    scan
        ``"near_targets"`` (default) scans only candidate regions a target gene
        can reach, against the base composition of *all* candidate regions
        (``pypando.region_background``). Each region is scanned independently
        given that background and regions are assigned to genes one at a time,
        so the network is identical to a full scan, at a fraction of the cost.
        ``"all"`` scans every region, as Pando does.
    key_added
        If set, the coefficient, goodness-of-fit and module tables are also
        stored in ``rna.uns[key_added]``.

    Returns
    -------
    pypando.GRNData
        Use ``pypando.coef(grn)``, ``pypando.gof(grn)``,
        ``pypando.NetworkModules(grn)``.
    """
    pp = _require_pypando()
    import anndata as ad

    if atac is None:
        mdata = rna
        rna, atac = mdata["rna"], mdata["atac"]
    host = rna
    cells = [c for c in rna.obs_names if c in set(atac.obs_names)]
    if not cells:
        raise ValueError("rna and atac share no cells")
    rna = rna[cells]
    atac = atac[cells]

    X_rna = rna.layers[rna_layer] if rna_layer else rna.X
    X_atac = atac.layers[peak_layer] if peak_layer else atac.X
    if normalize and _looks_like_counts(X_rna):
        X_rna = seurat_lognormalize(X_rna)
    if normalize and _looks_like_counts(X_atac):
        X_atac = signac_tfidf(X_atac)
    r = ad.AnnData(X=sp.csr_matrix(X_rna), obs=pd.DataFrame(index=np.asarray(cells, dtype=str)),
                   var=pd.DataFrame(index=np.asarray(rna.var_names, dtype=str)))
    a = ad.AnnData(X=sp.csr_matrix(X_atac), obs=pd.DataFrame(index=np.asarray(cells, dtype=str)),
                   var=pd.DataFrame(index=_signac_peak_names(atac.var_names)))

    if genes is None:
        if "highly_variable" not in rna.var:
            raise ValueError("pass `genes`, or flag highly variable genes in rna.var['highly_variable']")
        genes = list(rna.var_names[rna.var["highly_variable"].to_numpy()])
    if annotation is None:
        annotation = pando_annotation(genome)

    grn = pp.initiate_grn(r, a, annotation, regions=regions, exclude_exons=exclude_exons,
                          variable_features=genes)
    fasta = _fasta_path(genome)
    bg = motif_bg
    if scan == "near_targets":
        if isinstance(motif_bg, str) and motif_bg == "subject":
            bg = pp.region_background(grn, fasta)
        keep = pp.regions_near_genes(grn, genes, upstream=upstream, downstream=downstream,
                                     method=peak_to_gene_method, extend=extend)
        pp.subset_regions(grn, keep)
    elif scan != "all":
        raise ValueError("scan must be 'near_targets' or 'all'")
    grn = pp.find_motifs(grn, pfm=pfm, genome=fasta, motif_tfs=motif_tfs,
                         bg=bg, p_cutoff=p_cutoff, verbose=verbose)
    grn = pp.infer_grn(grn, genes=genes, method=method, peak_to_gene_method=peak_to_gene_method,
                       upstream=upstream, downstream=downstream, extend=extend, tf_cor=tf_cor,
                       peak_cor=peak_cor, interaction_term=interaction_term, alpha=alpha, seed=seed,
                       verbose=verbose, **infer_kwargs)
    if find_modules:
        grn = pp.find_modules(grn, p_thresh=p_thresh, rsq_thresh=rsq_thresh, nvar_thresh=nvar_thresh,
                              min_genes_per_module=min_genes_per_module, verbose=verbose)
    if key_added:
        host.uns[key_added] = {
            "coefs": pp.coef(grn),
            "gof": pp.gof(grn),
            "modules": pp.NetworkModules(grn).meta if find_modules else None,
            "params": {"method": method, "tf_cor": tf_cor, "peak_cor": peak_cor,
                       "upstream": upstream, "downstream": downstream},
        }
    return grn
