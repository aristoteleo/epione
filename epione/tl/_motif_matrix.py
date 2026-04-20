"""Peak × motif binary matrix via pychromvar's MOODS scan.

Thin wrapper around :func:`pychromvar.match_motif`. For each peak we
scan a sliding PWM against the peak sequence and record a hit when the
MOODS log-odds score exceeds the p-value threshold. Results go into
``adata.varm[key_added]`` (sparse bool ``n_peaks × n_motifs``).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def _console(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  └─ [motif_matrix] {msg}", flush=True)


def _pwms_from_jaspar(
    release: str = "JASPAR2024",
    collection: str = "CORE",
    tax_group: Sequence[str] = ("vertebrates",),
):
    """Fetch JASPAR motifs as Biopython motif objects (the form pychromvar
    expects)."""
    from pyjaspar import jaspardb
    jdb = jaspardb(release=release)
    return jdb.fetch_motifs(collection=collection, tax_group=list(tax_group))


def _as_pychromvar_compatible(adata: AnnData) -> AnnData:
    """pychromvar parses peak names with a single ``delimiter`` between
    all three fields. Accept epione's ``chrom:start-end`` by rewriting
    ``var_names`` to ``chrom-start-end`` (only if needed)."""
    names = list(adata.var_names)
    if any(":" in n for n in names[:5]):
        new = [n.replace(":", "-", 1) for n in names]
        adata = adata.copy()
        adata.var_names = new
    return adata


def _coerce_pwms_to_biopython(pwms):
    """Accept either (a) an iterable of Biopython / JASPAR ``motif``
    objects — passed straight through, or (b) a
    ``{name: 4xL probability matrix}`` dict — convert to minimal
    Biopython-compatible PFM objects so pychromvar's MOODS scan is happy.
    """
    from collections.abc import Mapping
    if pwms is None:
        return None
    if isinstance(pwms, Mapping):
        try:
            from Bio.motifs import Motif
            from Bio.motifs.matrix import FrequencyPositionMatrix
        except Exception as e:
            raise ImportError(
                "Biopython is required to pass PWMs as a {name: array} dict."
            ) from e
        out = []
        for name, mat in pwms.items():
            arr = np.asarray(mat, dtype=np.float64)
            if arr.shape[0] != 4:
                raise ValueError(
                    f"{name}: PWM must be 4×L (rows ACGT), got {arr.shape}")
            counts = {c: arr[i] for i, c in enumerate("ACGT")}
            m = Motif(alphabet="ACGT", counts=counts)
            m.matrix_id = name
            m.name = name
            out.append(m)
        return out
    return list(pwms)


def add_motif_matrix(
    adata: AnnData,
    genome_fasta: str,
    pwms=None,
    *,
    motif_db: str = "JASPAR2024",
    motif_collection: str = "CORE",
    motif_tax_group: Sequence[str] = ("vertebrates",),
    pvalue: float = 5e-5,
    background: str = "even",
    window: Optional[int] = None,
    key_added: str = "motif",
    verbose: bool = True,
) -> AnnData:
    """Scan peaks against PWMs; store sparse peak × motif binary matrix.

    Parameters
    ----------
    adata
        Peak-matrix ``AnnData`` where ``var_names`` are ``chrom:start-end``
        (``chrom-start-end`` is also accepted).
    genome_fasta
        Reference FASTA matching the peak coordinates.
    pwms
        Iterable of Biopython/JASPAR ``motif`` objects. If ``None``,
        motifs are fetched from JASPAR via :mod:`pyjaspar`.
    pvalue
        Per-motif log-odds p-value threshold (chromVAR default 5e-5).
    background
        Nucleotide background model passed through to pychromvar:
        ``"even"`` (0.25/base, default), ``"subject"`` (per-peak), or
        ``"genome"``.
    key_added
        - ``adata.varm[key_added]`` stores the sparse boolean matrix
          (same layout as :func:`pychromvar.match_motif` writes to
          ``adata.varm['motif_match']``).
        - ``adata.uns[key_added + '_names']`` stores motif names.
    """
    import pychromvar as pv
    if window is not None:
        import warnings
        warnings.warn(
            "window=%d ignored: pychromvar scans the full peak interval. "
            "Crop the peak set itself if you need a fixed window." % window,
            stacklevel=2,
        )
    adata_pv = _as_pychromvar_compatible(adata)
    if "peak_seq" not in adata_pv.uns:
        _console(f"reading sequences from {genome_fasta}", verbose)
        pv.add_peak_seq(adata_pv, genome_file=genome_fasta, delimiter="-")

    if pwms is None:
        _console(
            f"fetching motifs: {motif_db}/{motif_collection}/{motif_tax_group}",
            verbose,
        )
        pwms = _pwms_from_jaspar(motif_db, motif_collection, motif_tax_group)

    motifs = _coerce_pwms_to_biopython(pwms)
    _console(f"{len(motifs):,} motifs | pvalue={pvalue:.0e}", verbose)
    pv.match_motif(adata_pv, motifs=motifs, p_value=pvalue,
                   background=background,
                   genome_file=genome_fasta if background == "genome" else None)

    M = adata_pv.varm["motif_match"]
    if not sp.issparse(M):
        M = sp.csr_matrix(M)
    M = (M > 0).astype(np.bool_)
    motif_names = [f"{getattr(m, 'matrix_id', i)}_{getattr(m, 'name', i)}"
                   for i, m in enumerate(motifs)]

    adata.varm[key_added] = M
    adata.uns[f"{key_added}_names"] = np.asarray(motif_names, dtype=object)
    adata.uns[f"{key_added}_params"] = dict(
        db=motif_db, collection=motif_collection,
        tax_group=list(motif_tax_group),
        pvalue=float(pvalue), background=background,
    )
    # Carry the peak-sequence cache along so downstream add_background_peaks
    # can reuse it without re-reading the FASTA.
    if "peak_seq" in adata_pv.uns and "peak_seq" not in adata.uns:
        adata.uns["peak_seq"] = adata_pv.uns["peak_seq"]
    if "gc_bias" in adata_pv.var.columns and "gc_bias" not in adata.var.columns:
        adata.var["gc_bias"] = adata_pv.var["gc_bias"].values

    _console(
        f"{M.nnz:,} (peak, motif) hits | median per motif: "
        f"{int(np.median(np.asarray(M.sum(axis=0)).ravel())):,}",
        verbose,
    )
    return adata
