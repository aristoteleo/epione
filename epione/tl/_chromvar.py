"""chromVAR-style per-cell motif deviation Z-scores.

Port of ArchR ``addDeviationsMatrix`` (which is itself
``chromVAR::computeDeviations``). For each motif, we compute:

.. code::

   observed  = annotation^T @ counts                       # per-cell
   expected  = (annotation^T @ expectation) * total_counts # per-cell
   obs_dev   = (observed - expected) / expected

and, using ``n_iterations`` random sets of background peers matched on
``(accessibility, GC)`` bias, compute a null distribution of deviations
whose mean/sd give the bias-corrected Z-score

.. code::

   z = (obs_dev - mean(bg_dev)) / sd(bg_dev)

Required inputs on ``adata`` (produced by
:func:`epione.tl.add_motif_matrix` and
:func:`epione.tl.add_background_peaks`):

- ``adata.varm['motif']``      — peak × motif boolean matrix
- ``adata.uns['motif_names']`` — motif names in column order
- ``adata.varm['bg_peaks']``   — peak × n_iterations int indices
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def _console(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  └─ [deviations] {msg}", flush=True)


def compute_deviations(
    adata: AnnData,
    *,
    motif_key: str = "motif",
    bg_key: str = "bg_peaks",
    key_added: str = "motif_deviations",
    layer: Optional[str] = None,
    chunk_size: int = 64,
    verbose: bool = True,
) -> AnnData:
    """Compute chromVAR-style motif deviation Z-scores.

    Parameters
    ----------
    adata
        Peak matrix (cells × peaks).
    motif_key
        ``adata.varm`` key for the peak × motif binary matrix.
    bg_key
        ``adata.varm`` key for the peak × n_iterations background peer
        matrix.
    layer
        Optional counts layer to use instead of ``adata.X``.
    chunk_size
        Number of motifs processed per numpy batch (affects memory use;
        default is fine for ~10k cells × 500 motifs).
    key_added
        Stores the per-cell Z-score matrix in
        ``adata.obsm[key_added]`` with shape ``(n_cells, n_motifs)``;
        also stores the uncorrected deviation in
        ``adata.obsm[key_added + '_raw']``. Motif names are available
        via ``adata.uns[motif_key + '_names']``.
    """
    if motif_key not in adata.varm:
        raise KeyError(f"adata.varm[{motif_key!r}] not found. "
                       "Run epi.tl.add_motif_matrix first.")
    if bg_key not in adata.varm:
        raise KeyError(f"adata.varm[{bg_key!r}] not found. "
                       "Run epi.tl.add_background_peaks first.")
    M = adata.varm[motif_key]
    if not sp.issparse(M):
        M = sp.csr_matrix(M)
    bg = np.asarray(adata.varm[bg_key], dtype=np.int64)
    n_peaks, n_motifs = M.shape
    n_iter = bg.shape[1]

    X = adata.layers[layer] if layer is not None else adata.X
    # Need X as (peaks, cells) sparse for fast chromVAR-style left-mul.
    if sp.issparse(X):
        Xpc = X.T.tocsr()
    else:
        Xpc = sp.csr_matrix(X.T)
    n_cells = Xpc.shape[1]

    _console(
        f"cells={n_cells:,} | peaks={n_peaks:,} | motifs={n_motifs:,} | "
        f"bg_iter={n_iter}",
        verbose,
    )

    # rowSums of counts (per peak) and per-cell totals
    rowSums = np.asarray(Xpc.sum(axis=1)).ravel().astype(np.float64)
    expectation = rowSums / max(rowSums.sum(), 1.0)             # (n_peaks,)
    counts_per_cell = np.asarray(Xpc.sum(axis=0)).ravel().astype(np.float64)

    Z   = np.zeros((n_cells, n_motifs), dtype=np.float32)
    DEV = np.zeros((n_cells, n_motifs), dtype=np.float32)

    # Iterate motifs in chunks so we can vectorise across motifs but cap
    # peak memory during the background expansion.
    motif_csc = M.tocsc()
    for m0 in range(0, n_motifs, chunk_size):
        m1 = min(m0 + chunk_size, n_motifs)
        if verbose:
            _console(f"motifs {m0:,} .. {m1:,}", verbose)
        for mj in range(m0, m1):
            col = motif_csc[:, mj]
            peak_idx = col.indices                              # (k,) foreground peaks
            k = peak_idx.size
            if k == 0:
                Z[:, mj]   = np.nan
                DEV[:, mj] = np.nan
                continue

            # --- Observed --------------------------------------------
            fg = sp.csr_matrix(
                (np.ones(k, dtype=np.float64),
                 (np.zeros(k, dtype=np.int64), peak_idx)),
                shape=(1, n_peaks),
            )
            fg_cts = fg @ Xpc
            if sp.issparse(fg_cts):
                fg_cts = fg_cts.toarray()
            observed = np.asarray(fg_cts).ravel()               # (n_cells,)
            fg_expect_frac = float(fg @ expectation)            # scalar
            expected = fg_expect_frac * counts_per_cell
            with np.errstate(divide="ignore", invalid="ignore"):
                obs_dev = np.where(expected > 0,
                                   (observed - expected) / expected,
                                   np.nan).astype(np.float32)

            # --- Background null -------------------------------------
            # For each background iteration, replace each foreground
            # peak i with bg[i, it]. Build a (n_iter × n_peaks) sparse
            # matrix of ones at these replacement positions, then
            # multiply against Xpc in one shot.
            cols = bg[peak_idx, :].ravel()                      # (k * n_iter,)
            rows = np.repeat(np.arange(n_iter), k)              # (k * n_iter,)
            data = np.ones(cols.shape, dtype=np.float64)
            S = sp.csr_matrix((data, (rows, cols)),
                              shape=(n_iter, n_peaks))
            S_Xpc = S @ Xpc
            if sp.issparse(S_Xpc):
                S_Xpc = S_Xpc.toarray()
            sampled          = np.asarray(S_Xpc)                # (n_iter, n_cells)
            sampled_fg_frac  = np.asarray(S @ expectation).ravel()
            sampled_expected = sampled_fg_frac[:, None] * counts_per_cell[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                sampled_dev = np.where(
                    sampled_expected > 0,
                    (sampled - sampled_expected) / sampled_expected,
                    np.nan,
                )
            mean_sd = np.nanmean(sampled_dev, axis=0)
            sd_sd   = np.nanstd(sampled_dev, axis=0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (obs_dev - mean_sd) / sd_sd
            DEV[:, mj] = (obs_dev - mean_sd).astype(np.float32)
            Z[:,   mj] = z.astype(np.float32)

    motif_names = np.asarray(adata.uns.get(f"{motif_key}_names",
                                           [f"M{j}" for j in range(n_motifs)]),
                             dtype=object)
    adata.obsm[key_added]                  = Z
    adata.obsm[f"{key_added}_raw"]         = DEV
    adata.uns[f"{key_added}_names"]        = motif_names
    adata.uns[f"{key_added}_params"]       = dict(
        motif_key=motif_key, bg_key=bg_key, layer=layer,
        n_iterations=int(n_iter),
    )
    _console(
        f"done. obsm[{key_added!r}] = {Z.shape} float32 Z-scores; "
        f"[{key_added}_raw] holds uncorrected deviations",
        verbose,
    )
    return adata
