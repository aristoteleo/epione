"""Motif → TF annotation and cistrome assembly (pycistarget ``utils``).

The motif2TF tables (``motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl`` …)
list, per motif, the TFs annotated directly, by motif similarity, by
orthology, or by both. pycistarget joins the TFs of each motif with
``', '.join(set(...))``, so its order depends on Python's string hash
seed. Here the TFs are joined in sorted order. The TF *sets* are identical,
and so is every downstream result, because TF matching only depends on the
set (see :func:`motifs_per_tf`).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ANNOTATION_COLUMNS = ("Direct_annot", "Motif_similarity_annot", "Orthology_annot",
                      "Motif_similarity_and_Orthology_annot")

def _join(x) -> str:
    return ", ".join(sorted(set(x)))


@lru_cache(maxsize=8)
def _load_motif_annotations_cached(fname: str, motif_similarity_fdr: float,
                                   orthologous_identity_threshold: float) -> pd.DataFrame:
    df = pd.read_csv(fname, sep="\t", usecols=("#motif_id", "gene_name",
                                               "motif_similarity_qvalue",
                                               "orthologous_identity", "description"))
    df = df.rename(columns={"#motif_id": "MotifID", "gene_name": "TF",
                            "motif_similarity_qvalue": "MotifSimilarityQvalue",
                            "orthologous_identity": "OrthologousIdentity",
                            "description": "Annotation"})
    df = df[(df["MotifSimilarityQvalue"] <= motif_similarity_fdr)
            & (df["OrthologousIdentity"] >= orthologous_identity_threshold)]
    similar = df["Annotation"].str.contains("similar")
    ortho = df["Annotation"].str.contains("orthologous")
    parts = {
        "Direct_annot": df[df["Annotation"] == "gene is directly annotated"],
        "Motif_similarity_annot": df[similar & ~ortho],
        "Orthology_annot": df[~similar & ortho],
        "Motif_similarity_and_Orthology_annot": df[similar & ortho],
    }
    out = [p.groupby("MotifID")["TF"].agg(_join).rename(col) for col, p in parts.items()]
    return pd.concat(out, axis=1, sort=False)


def load_motif_annotations(
    fname: Optional[str] = None,
    species: str = "homo_sapiens",
    version: str = "v10nr_clust",
    motif_similarity_fdr: float = 0.001,
    orthologous_identity_threshold: float = 0.0,
) -> pd.DataFrame:
    """Motif × annotation-type table of comma-joined TF names.

    Parameters
    ----------
    fname
        Local motif2TF ``.tbl`` file. When ``None`` the Aerts-lab snapshot for
        ``species``/``version`` is downloaded (and cached) with pooch.
    motif_similarity_fdr, orthologous_identity_threshold
        Keep annotations with similarity q-value ``<=`` / orthologous identity
        ``>=`` these values.
    """
    if fname is None:
        from ._data import fetch_motif_annotation
        fname = fetch_motif_annotation(species=species, version=version)
    return _load_motif_annotations_cached(os.fspath(fname), float(motif_similarity_fdr),
                                          float(orthologous_identity_threshold))


def _split(cell) -> List[str]:
    if not isinstance(cell, str):
        return []
    return cell.split(", ")


def tf_list(motif_enrichment: pd.DataFrame,
            annotation: Sequence[str] = ANNOTATION_COLUMNS) -> List[str]:
    """All TFs named in the ``annotation`` columns (pycistarget ``get_TF_list``),
    sorted."""
    tfs = set()
    for col in annotation:
        if col in motif_enrichment:
            for cell in motif_enrichment[col]:
                tfs.update(_split(cell))
    return sorted(tfs)


def motifs_per_tf(motif_enrichment: pd.DataFrame, tf: str,
                  annotation: Sequence[str] = ANNOTATION_COLUMNS) -> List[str]:
    """Motifs whose annotation names ``tf`` (pycistarget ``get_motifs_per_TF``).

    Upstream matches with the regex ``tf + ',|' + tf + '$'``, which hits any
    annotated TF that *ends with* ``tf`` (``PAX5`` also matches ``SPAX5``).
    That rule is kept, so cistromes are identical. Returned sorted.
    """
    out = set()
    for col in annotation:
        if col in motif_enrichment:
            for motif, cell in motif_enrichment[col].items():
                if any(t.endswith(tf) for t in _split(cell)):
                    out.add(motif)
    return sorted(out)


def cistromes_per_region_set(
    motif_enrichment: pd.DataFrame,
    motif_hits: Mapping[str, Sequence[str]],
    annotation: Sequence[str] = ("Direct_annot", "Orthology_annot"),
) -> Dict[str, List[str]]:
    """TF → regions, named ``TF_(Nr)`` / ``TF_extended_(Nr)``
    (pycistarget ``get_cistromes_per_region_set``). Region lists are sorted."""
    def union(motifs):
        s = set()
        for m in motifs:
            if m in motif_hits:
                s.update(motif_hits[m])
        return sorted(s)

    cis: Dict[str, List[str]] = {}
    if "Direct_annot" in annotation:
        for tf in tf_list(motif_enrichment, ["Direct_annot"]):
            cis[tf] = union(motifs_per_tf(motif_enrichment, tf, ["Direct_annot"]))
    if "Direct_annot" not in annotation or len(annotation) > 1:
        # upstream takes the TF list from all annotation columns present
        for tf in tf_list(motif_enrichment):
            cis[tf + "_extended"] = union(motifs_per_tf(motif_enrichment, tf, annotation))
    return {f"{k}_({len(v)}r)": v for k, v in cis.items()}
