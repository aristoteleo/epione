"""chromVAR-style per-cell motif deviation Z-scores.

Same math as ``chromVAR::computeDeviations`` / ``pychromvar.compute_deviations``
but every matrix operation is kept sparse wherever possible, and the
background permutation is expressed as a column-index remap of the
motif_match CSR — so each of the ``n_iterations`` null passes is a single
``sparse @ sparse`` matmul against the full peak/cell matrix.

Why this layout matters for memory and speed:

- Peak matrix ``X`` stays sparse as ``(n_peaks, n_cells) CSC``. No dense
  ``adata.X.todense()`` copy.
- Motif matrix ``M`` stays sparse as ``(n_motifs, n_peaks) CSR``.
- Foreground deviation: ``obs_fg = M @ X`` — one sparse matmul, densifies
  to ``(n_motifs, n_cells)`` only once (~32 MB at typical size).
- Background deviations: for each iteration ``it``, re-point M's column
  indices to their bg peers and do the same matmul. Running sum +
  sum-of-squares give mean / sd of the null without ever materialising
  the ``(n_iter, n_motifs, n_cells)`` tensor.

Peak math:

.. code::

    peak_frac = X.sum(axis=cells) / X.sum()       # (n_peaks,), global
    fg_fracs = M @ peak_frac                      # (n_motifs,)
    expected = fg_fracs[:, None] * cell_totals[None, :]
    obs_dev  = (M @ X - expected) / expected
"""
from __future__ import annotations

from typing import Optional

import numpy as np
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
    verbose: bool = True,
) -> AnnData:
    """Compute chromVAR-style motif deviation Z-scores.

    Parameters
    ----------
    adata
        Peak matrix (cells × peaks).
    motif_key
        ``adata.varm`` key for the peak × motif binary matrix
        (:func:`epione.tl.add_motif_matrix`).
    bg_key
        ``adata.varm`` key for the peak × ``n_iterations`` background
        peer index matrix (:func:`epione.tl.add_background_peaks`).
    layer
        Optional counts layer to use instead of ``adata.X``.
    key_added
        - ``adata.obsm[key_added]`` receives the ``(n_cells, n_motifs)``
          bias-corrected Z-score matrix.
        - ``adata.obsm[key_added + '_raw']`` receives the uncorrected
          deviation (``obs_dev - mean_bg``).

    Notes
    -----
    Uses a running sum / sum-of-squares accumulator for the background
    null distribution, so peak memory is ``O(n_motifs * n_cells)`` — a
    few hundred MB for a 750-motif × 10 k-cell run. This is independent
    of ``n_iterations``.
    """
    if motif_key not in adata.varm:
        raise KeyError(f"adata.varm[{motif_key!r}] not found. Run "
                       "epione.tl.add_motif_matrix first.")
    if bg_key not in adata.varm:
        raise KeyError(f"adata.varm[{bg_key!r}] not found. Run "
                       "epione.tl.add_background_peaks first.")

    M = adata.varm[motif_key]
    if not sp.issparse(M):
        M = sp.csr_matrix(M)
    # (n_motifs, n_peaks) CSR — keep sparse for all downstream matmuls.
    M_csr = M.T.astype(np.float32).tocsr()
    n_motifs, n_peaks = M_csr.shape

    bg = np.asarray(adata.varm[bg_key], dtype=np.int32)
    assert bg.shape[0] == n_peaks, (
        f"bg_peaks rows ({bg.shape[0]}) ≠ n_peaks ({n_peaks})"
    )
    n_iter = bg.shape[1]

    X = adata.layers[layer] if layer is not None else adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    # (n_peaks, n_cells) CSC for fast left-mul by CSR motif matrix.
    X_T = X.T.astype(np.float32).tocsc()
    n_cells = X_T.shape[1]

    total_counts = float(X_T.sum())
    peak_sums = np.asarray(X_T.sum(axis=1)).ravel().astype(np.float64)
    peak_frac = (peak_sums / max(total_counts, 1.0)).astype(np.float32)
    cell_totals = np.asarray(X_T.sum(axis=0)).ravel().astype(np.float32)

    _console(
        f"cells={n_cells:,} | peaks={n_peaks:,} | motifs={n_motifs:,} | "
        f"bg_iter={n_iter}",
        verbose,
    )

    # ---- Foreground deviation -----------------------------------------
    _console("observed deviations", verbose)
    obs_counts = np.asarray((M_csr @ X_T).todense(), dtype=np.float32)
    fg_fracs   = np.asarray(M_csr @ peak_frac).ravel().astype(np.float32)
    expected_fg = fg_fracs[:, None] * cell_totals[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        obs_dev = np.where(
            expected_fg > 0,
            (obs_counts - expected_fg) / expected_fg,
            np.nan,
        ).astype(np.float32)

    # ---- Background null ----------------------------------------------
    # bg column-index remap of M_csr is just:
    #     M_it.indices = bg[M_csr.indices, it]
    # which keeps data, indptr the same — so each iteration is a single
    # sparse matmul + an outer-product subtraction.
    sum_bg   = np.zeros((n_motifs, n_cells), dtype=np.float64)
    sumsq_bg = np.zeros((n_motifs, n_cells), dtype=np.float64)
    M_indices = M_csr.indices
    M_data    = M_csr.data
    M_indptr  = M_csr.indptr
    log_every = max(n_iter // 10, 1)
    for it in range(n_iter):
        if verbose and (it % log_every == 0):
            _console(f"background iteration {it + 1}/{n_iter}", verbose)
        new_indices = bg[M_indices, it].astype(np.int32)
        M_it = sp.csr_matrix(
            (M_data, new_indices, M_indptr),
            shape=(n_motifs, n_peaks),
        )
        obs_bg_it = np.asarray((M_it @ X_T).todense(), dtype=np.float32)
        fg_fracs_bg = np.asarray(M_it @ peak_frac).ravel().astype(np.float32)
        expected_bg = fg_fracs_bg[:, None] * cell_totals[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            dev_it = np.where(
                expected_bg > 0,
                (obs_bg_it - expected_bg) / expected_bg,
                np.nan,
            ).astype(np.float64)
        finite = np.isfinite(dev_it)
        dev_it = np.where(finite, dev_it, 0.0)
        sum_bg   += dev_it
        sumsq_bg += dev_it * dev_it

    mean_bg = (sum_bg / n_iter).astype(np.float32)
    var_bg  = (sumsq_bg / n_iter - (sum_bg / n_iter) ** 2)
    var_bg *= (n_iter / max(n_iter - 1, 1))               # Bessel
    std_bg = np.sqrt(np.clip(var_bg, 0.0, None)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(std_bg > 0, (obs_dev - mean_bg) / std_bg, np.nan).astype(np.float32)
    DEV = (obs_dev - mean_bg).astype(np.float32)

    # obsm convention: (n_cells, n_motifs)
    names = np.asarray(
        adata.uns.get(f"{motif_key}_names",
                      [f"M{j}" for j in range(n_motifs)]),
        dtype=object,
    )
    adata.obsm[key_added]          = Z.T
    adata.obsm[f"{key_added}_raw"] = DEV.T
    adata.uns[f"{key_added}_names"] = names
    adata.uns[f"{key_added}_params"] = dict(
        motif_key=motif_key, bg_key=bg_key, layer=layer,
        n_iterations=int(n_iter),
    )
    _console(
        f"done. obsm[{key_added!r}] = {Z.T.shape} float32 Z-scores; "
        f"[{key_added}_raw] holds uncorrected deviations",
        verbose,
    )
    return adata
