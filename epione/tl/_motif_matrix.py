"""Peak × motif binary matrix via pychromvar's MOODS scan.

Thin wrapper around :func:`pychromvar.match_motif`. For each peak we
scan a sliding PWM against the peak sequence and record a hit when the
MOODS log-odds score exceeds the p-value threshold. Results go into
``adata.varm[key_added]`` (sparse bool ``n_peaks × n_motifs``).
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def _console(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  └─ [motif_matrix] {msg}", flush=True)


def _save_cache(cache_file: str, M: sp.csr_matrix, motif_names, var_names):
    """Persist a peak × motif sparse matrix as .npz so the next run with
    the same var_names can skip MOODS entirely."""
    var_hash = hashlib.sha256(
        ";".join(map(str, var_names)).encode()
    ).hexdigest()[:16]
    out_parent = os.path.dirname(os.path.abspath(cache_file))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    np.savez_compressed(
        cache_file,
        data=M.data, indices=M.indices, indptr=M.indptr,
        shape=np.asarray(M.shape, dtype=np.int64),
        motif_names=np.asarray(motif_names, dtype=object),
        var_hash=np.asarray(var_hash),
    )


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
    genome_fasta: Optional[str] = None,
    pwms=None,
    *,
    motif_database: Optional[str] = None,
    cache_file: Optional[str] = None,
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
    motif_database
        Path to a pre-built genome-wide motif hit database produced by
        :func:`epione.tl.build_motif_database`. When provided, MOODS is not
        run — peaks are resolved by interval lookup against the database,
        which takes seconds instead of ~100 s. ``genome_fasta`` / ``pwms``
        / ``pvalue`` are ignored (the database's configuration wins).
    cache_file
        Path to a per-peak-set cache. On first run, the resulting motif
        matrix is dumped to this file (``.npz``). On later runs with the
        same ``adata.var_names`` hash the cache is loaded instead of
        re-scanning. Independent of ``motif_database``.
    key_added
        - ``adata.varm[key_added]`` stores the sparse boolean matrix
          (same layout as :func:`pychromvar.match_motif` writes to
          ``adata.varm['motif_match']``).
        - ``adata.uns[key_added + '_names']`` stores motif names.
    """
    # ---- Peak-set cache fast path -----------------------------------------
    if cache_file is not None:
        import hashlib
        var_hash = hashlib.sha256(
            ";".join(map(str, adata.var_names)).encode()
        ).hexdigest()[:16]
        if os.path.exists(cache_file):
            try:
                npz = np.load(cache_file, allow_pickle=True)
                if str(npz["var_hash"]) == var_hash:
                    _console(f"loading cache {cache_file}", verbose)
                    M = sp.csr_matrix(
                        (npz["data"].astype(np.bool_),
                         npz["indices"], npz["indptr"]),
                        shape=tuple(npz["shape"]),
                    )
                    adata.varm[key_added] = M
                    adata.uns[f"{key_added}_names"] = np.asarray(
                        npz["motif_names"], dtype=object)
                    _console(
                        f"{M.nnz:,} hits loaded from cache", verbose,
                    )
                    return adata
            except Exception as e:
                _console(f"cache load failed ({e}); rebuilding", verbose)

    # ---- Database fast path -----------------------------------------------
    if motif_database is not None:
        from ._motif_database import query_motif_database
        import re
        pat = re.compile(r"(?P<chrom>[^\s:_-]+(?:[._][^\s:_-]+)?)[:_-](?P<s>\d+)[-_](?P<e>\d+)$")
        rows = []
        for n in adata.var_names:
            m = pat.match(str(n))
            if m is None:
                raise ValueError(
                    f"cannot parse peak name {n!r}; expected 'chrom:start-end'."
                )
            rows.append((m.group("chrom"), int(m.group("s")), int(m.group("e"))))
        peaks_df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
        _console(f"querying database {motif_database}", verbose)
        M, motif_names = query_motif_database(motif_database, peaks_df,
                                              verbose=verbose)
        adata.varm[key_added] = M
        adata.uns[f"{key_added}_names"] = motif_names
        adata.uns[f"{key_added}_params"] = dict(
            source="motif_database", path=str(motif_database),
        )
        if cache_file is not None:
            _save_cache(cache_file, M, motif_names, adata.var_names)
        return adata

    # ---- Fresh MOODS scan (pychromvar) ------------------------------------
    import pychromvar as pv
    if genome_fasta is None:
        raise ValueError(
            "add_motif_matrix needs one of: genome_fasta=, motif_database=, "
            "or cache_file= (with a populated cache)."
        )
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
    if cache_file is not None:
        _save_cache(cache_file, M, motif_names, adata.var_names)
        _console(f"wrote cache {cache_file}", verbose)
    return adata
