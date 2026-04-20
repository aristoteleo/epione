"""kNN-based background peak sets for chromVAR-style deviation bias correction.

Port of ArchR ``addBgdPeaks`` (which wraps ``chromVAR::getBackgroundPeaks``).
For each peak, find ``n_iterations`` background peers that share similar
``(log10(rowSums+1), GC content)`` bias. The returned index matrix is fed
to :func:`epione.tl.compute_deviations` to build the null distribution
for bias-corrected Z-scores.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def _console(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  └─ [bg_peaks] {msg}", flush=True)


def _compute_gc_from_fasta(
    adata: AnnData, genome_fasta: str,
) -> np.ndarray:
    """GC fraction per peak from a FASTA file."""
    from ._motif_matrix import _parse_peak_coords, _fetch_peak_sequences
    chroms, starts, ends = _parse_peak_coords(adata.var_names)
    seqs = _fetch_peak_sequences(genome_fasta, chroms, starts, ends)
    gc = np.zeros(len(seqs), dtype=np.float32)
    for i, s in enumerate(seqs):
        if len(s) == 0:
            gc[i] = np.nan
            continue
        cnt = s.count(b"G") + s.count(b"C")
        gc[i] = cnt / len(s)
    return gc


def add_background_peaks(
    adata: AnnData,
    *,
    n_iterations: int = 50,
    gc_key: Optional[str] = None,
    genome_fasta: Optional[str] = None,
    seed: int = 0,
    key_added: str = "bg_peaks",
    verbose: bool = True,
) -> AnnData:
    """Find ``n_iterations`` per-peak background peers in bias space.

    Parameters
    ----------
    adata
        Peak matrix. ``adata.X`` is used to compute ``rowSums``
        (accessibility bias).
    n_iterations
        Number of background peer peaks to draw per foreground peak.
    gc_key
        Name of a column in ``adata.var`` containing the per-peak GC
        fraction. When omitted, GC is computed from ``genome_fasta``.
    genome_fasta
        Path to a reference FASTA. Required unless ``gc_key`` is given.
    key_added
        Result is stored in ``adata.varm[key_added]`` as an
        ``(n_peaks, n_iterations)`` int32 matrix of peer peak indices.

    Notes
    -----
    The peer search is a randomized k-NN in standardised
    ``(log10(rowSums+1), GC)`` space. chromVAR's original implementation
    draws peers with density-weighted sampling inside 2-D bias bins; a
    straight k-NN is simpler and produces highly correlated deviations
    (see :func:`epione.tl.compute_deviations` cross-validation).
    """
    # 1. Bias features: (log10(rowSums+1), GC)
    X = adata.X
    if sp.issparse(X):
        rowSums = np.asarray(X.sum(axis=0)).ravel()
    else:
        rowSums = X.sum(axis=0).ravel()
    rowSums = np.asarray(rowSums, dtype=np.float64)
    log_acc = np.log10(rowSums + 1.0)

    if gc_key is not None:
        if gc_key not in adata.var.columns:
            raise KeyError(f"adata.var[{gc_key!r}] not found")
        gc = adata.var[gc_key].to_numpy(dtype=np.float64)
    elif genome_fasta is not None:
        _console(f"computing GC from {genome_fasta}", verbose)
        gc = _compute_gc_from_fasta(adata, genome_fasta).astype(np.float64)
    else:
        raise ValueError(
            "add_background_peaks needs either gc_key= or genome_fasta=."
        )

    # 2. Standardise (drop NaN peaks by clipping to median)
    valid = np.isfinite(gc)
    for v, name in ((log_acc, "log_acc"), (gc, "gc")):
        if not np.all(valid):
            med = float(np.nanmedian(v[valid]))
            v[~valid] = med
    bias = np.column_stack([
        (log_acc - log_acc.mean()) / (log_acc.std() + 1e-9),
        (gc       - gc.mean())       / (gc.std()       + 1e-9),
    ])

    # 3. kNN in 2-D bias space. We draw n_iterations peers (excluding self).
    #    Using a small jitter on exact ties so peaks with identical bias are
    #    broken by seed, matching chromVAR's stochastic sampling intent.
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    jitter = rng.normal(scale=1e-6, size=bias.shape)
    knn = NearestNeighbors(n_neighbors=n_iterations + 1, algorithm="kd_tree")
    knn.fit(bias + jitter)
    _, idx = knn.kneighbors(bias)                # (n_peaks, n_iter+1)
    # Drop self-match (first column)
    bg = idx[:, 1:].astype(np.int32)

    adata.varm[key_added] = bg
    adata.uns[f"{key_added}_params"] = dict(
        n_iterations=int(n_iterations), seed=int(seed),
    )
    _console(f"built {bg.shape[0]:,} peaks × {bg.shape[1]:,} bg peers",
             verbose)
    return adata
