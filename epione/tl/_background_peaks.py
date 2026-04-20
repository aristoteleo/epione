"""Background peak sets via :func:`pychromvar.get_bg_peaks`.

Wrapper around pychromvar's bias-matched peer sampler (a Python port of
chromVAR's ``getBackgroundPeaks``). For each peak it draws
``n_iterations`` peers with similar ``(log10 accessibility, GC)`` bias.
Stored as ``(n_peaks, n_iterations)`` int index matrix in
``adata.varm[key_added]`` — ready for :func:`epione.tl.compute_deviations`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from anndata import AnnData


def _console(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  └─ [bg_peaks] {msg}", flush=True)


def add_background_peaks(
    adata: AnnData,
    *,
    n_iterations: int = 50,
    genome_fasta: Optional[str] = None,
    gc_key: Optional[str] = None,
    n_jobs: int = -1,
    seed: int = 0,
    key_added: str = "bg_peaks",
    verbose: bool = True,
) -> AnnData:
    """Sample ``n_iterations`` bias-matched peer peaks per foreground peak.

    Parameters
    ----------
    adata
        Peak AnnData. ``adata.X`` is used to derive the accessibility
        bias (``log10(rowSums + 1)``).
    genome_fasta
        FASTA path — needed if GC content isn't already in ``adata.var``.
    gc_key
        Name of an existing GC column in ``adata.var``. If ``None`` and
        ``adata.var['gc_bias']`` is missing, GC is computed from
        ``genome_fasta`` via :func:`pychromvar.add_gc_bias`.
    n_jobs
        Passed to :func:`pychromvar.get_bg_peaks`; ``-1`` uses all cores.
    """
    import pychromvar as pv

    # Ensure GC bias is available. pychromvar stores it as
    # ``adata.var['gc_bias']``.
    if "gc_bias" not in adata.var.columns:
        if gc_key is not None and gc_key in adata.var.columns:
            adata.var["gc_bias"] = adata.var[gc_key].to_numpy()
        else:
            if genome_fasta is None:
                raise ValueError(
                    "add_background_peaks needs gc_key= or genome_fasta= "
                    "when adata.var['gc_bias'] is not already present."
                )
            if "peak_seq" not in adata.uns:
                _console(f"reading sequences from {genome_fasta}", verbose)
                # pychromvar expects var_names of form "chrom-start-end"
                names = list(adata.var_names)
                rewritten = any(":" in n for n in names[:5])
                if rewritten:
                    adata.var_names = [n.replace(":", "-", 1) for n in names]
                pv.add_peak_seq(adata, genome_file=genome_fasta,
                                delimiter="-")
                if rewritten:
                    adata.var_names = names
            _console("computing GC bias", verbose)
            pv.add_gc_bias(adata)

    # Rename our expected output key to what pychromvar writes
    # (``bg_peaks``), call, then alias under key_added if different.
    pv.get_bg_peaks(adata, niterations=n_iterations, n_jobs=n_jobs)
    if "bg_peaks" in adata.varm and key_added != "bg_peaks":
        adata.varm[key_added] = np.asarray(adata.varm["bg_peaks"],
                                            dtype=np.int64)
    else:
        adata.varm[key_added] = np.asarray(adata.varm.get(key_added,
                                            adata.varm["bg_peaks"]),
                                            dtype=np.int64)
    adata.uns[f"{key_added}_params"] = dict(
        n_iterations=int(n_iterations), seed=int(seed),
    )
    _console(
        f"built {adata.varm[key_added].shape[0]:,} peaks × "
        f"{adata.varm[key_added].shape[1]:,} bg peers",
        verbose,
    )
    return adata
