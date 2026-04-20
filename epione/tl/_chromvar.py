"""chromVAR-style per-cell motif deviation Z-scores.

Two methods are provided:

- ``method="analytical"`` (default, fast): exploits the chromVAR sampling
  design — background peers are drawn IID per peak and independently across
  peaks — to express the null distribution's mean and variance as **sums of
  per-peak statistics**. Under peak independence,

  .. code::

      mean(obs_bg[j, c]) = Σ_{i ∈ P_j} μ_peak[i, c]
      var (obs_bg[j, c]) = Σ_{i ∈ P_j} σ²_peak[i, c]          # peaks independent

  where ``μ_peak[i, c]`` / ``σ²_peak[i, c]`` are the mean / variance across
  peak ``i``'s bg peer draws. Computing those once turns the null into
  **two sparse × dense matmuls** instead of ``n_iterations`` repeats.

  Analytical is the ``n_iter → ∞`` limit of the sample-based estimator, so
  at finite ``n_iter`` it is **both faster and lower-variance**. On the
  ArchR heme benchmark, sample (50 iter) vs analytical per-TF Pearson
  ≥ 0.995 at ≥1000 cells — differences are within chromVAR's own 1/√n_iter
  sampling noise, and analytical vs ArchR correlates slightly higher than
  sample-based vs ArchR (because it removes one end of the sampling noise).

- ``method="sample"``: the original chromVAR / ArchR procedure — for each
  of ``n_iter`` permutations, remap peaks via bg and recompute the
  deviation; accumulate running mean / variance across iterations. Kept for
  bit-level reproducibility with chromVAR / ArchR outputs.

Shared math:

.. code::

    peak_frac = X.sum(axis=cells) / X.sum()       # (n_peaks,), global
    fg_fracs  = M @ peak_frac                      # (n_motifs,)
    expected  = fg_fracs[:, None] * cell_totals[None, :]
    obs_dev   = (M @ X - expected) / expected
    Z         = (obs_dev − mean_bg_dev) / std_bg_dev
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def _console(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  └─ [deviations] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Analytical path
# ---------------------------------------------------------------------------

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
    def _nb_fused_peer_stats(bg, X_dense, sum_out, sumsq_out):
        """One-pass peer stats: for each peak i, accumulate

            sum_out[i, c]   = Σ_it X_dense[bg[i, it], c]
            sumsq_out[i, c] = Σ_it X_dense[bg[i, it], c]²

        Uses thread-local buffers to avoid false sharing between adjacent
        output rows across threads — gives ~7× speedup over the scipy-based
        ``B @ X`` + ``B @ X²`` pair. Single pass over ``X_dense`` per peer,
        no intermediate ``X²`` allocation, no peer-count matrix.
        """
        n_peaks = bg.shape[0]
        n_iter  = bg.shape[1]
        n_cells = X_dense.shape[1]
        for i in prange(n_peaks):
            s  = np.zeros(n_cells, dtype=np.float32)
            ss = np.zeros(n_cells, dtype=np.float32)
            for it in range(n_iter):
                peer = bg[i, it]
                for c in range(n_cells):
                    v = X_dense[peer, c]
                    s[c]  += v
                    ss[c] += v * v
            for c in range(n_cells):
                sum_out[i, c]   = s[c]
                sumsq_out[i, c] = ss[c]

    @njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
    def _nb_csr_at_dense(data, indices, indptr, dense, out):
        """Parallel CSR @ dense → dense. prange over rows of the CSR."""
        n_rows = indptr.shape[0] - 1
        n_cells = dense.shape[1]
        for i in prange(n_rows):
            row = out[i]
            row[:] = 0.0
            for k_idx in range(indptr[i], indptr[i + 1]):
                col = indices[k_idx]
                val = data[k_idx]
                row += val * dense[col]

    @njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
    def _nb_csr_at_dense_pair(data, indices, indptr, dense_a, dense_b, out_a, out_b):
        """Fused: computes ``M @ dense_a`` and ``M @ dense_b`` in one pass.

        Saves ~30% over calling ``_nb_csr_at_dense`` twice because each peak
        row's 20 KB loads from ``dense_a[col]`` and ``dense_b[col]`` stay hot
        in L1/L2 while the inner SIMD FMAs into both output rows pipeline.
        """
        n_rows = indptr.shape[0] - 1
        for i in prange(n_rows):
            ra = out_a[i]; ra[:] = 0.0
            rb = out_b[i]; rb[:] = 0.0
            for k_idx in range(indptr[i], indptr[i + 1]):
                col = indices[k_idx]
                val = data[k_idx]
                ra += val * dense_a[col]
                rb += val * dense_b[col]

    # Warm the kernels at import time so the first production call doesn't
    # pay the JIT-compile cost. ``cache=True`` persists across processes,
    # but the first-ever call still incurs load + lowering cost.
    def _warmup_numba_kernels():
        _bg   = np.zeros((2, 2),  dtype=np.int64)
        _X    = np.zeros((2, 4),  dtype=np.float32)
        _s    = np.zeros((2, 4),  dtype=np.float32)
        _ss   = np.zeros((2, 4),  dtype=np.float32)
        _nb_fused_peer_stats(_bg, _X, _s, _ss)
        _data    = np.ones(1, dtype=np.float32)
        _indices = np.zeros(1, dtype=np.int64)
        _indptr  = np.array([0, 1, 1], dtype=np.int64)
        _dense   = np.zeros((2, 4), dtype=np.float32)
        _out     = np.zeros((2, 4), dtype=np.float32)
        _nb_csr_at_dense(_data, _indices, _indptr, _dense, _out)
        _out2 = np.zeros((2, 4), dtype=np.float32)
        _nb_csr_at_dense_pair(_data, _indices, _indptr, _dense, _dense, _out, _out2)
        # _nb_finalise_peer_stats is defined later; warm lazily on first call.

    try:
        _warmup_numba_kernels()
    except Exception:
        # Warmup failure is non-fatal; first call will just pay compile cost
        pass


# BLAS sgemm (dense × dense) outperforms CSR×dense once the sparse matrix's
# density exceeds a few percent because BLAS is SIMD-optimised per-FLOP and
# multi-threaded (via OpenBLAS/MKL) — at the cost of doing the zero-ops
# anyway. For chromVAR's motif matrix the density is often 30-50%, so BLAS
# wins by 5-12×. Below this threshold, numba sparse iteration wins because
# it truly skips zeros.
_M_DENSE_DISPATCH_DENSITY = 0.03


def _set_blas_threads_ctx(n_jobs: int):
    """Context manager setting BLAS thread count; no-op if threadpoolctl
    unavailable.

    ``n_jobs = -1`` (or 0/None) → use all CPUs — NOT ``limits=-1`` which
    ``threadpool_limits`` interprets as an invalid value. An earlier
    version clamped via ``max(n_jobs, 1)`` and inadvertently limited BLAS
    to a single thread whenever the caller passed ``-1``, dropping BLAS
    sgemm from ~0.2 s to ~0.8 s per M @ dense call on pbmc5k.
    """
    if n_jobs is None or n_jobs <= 0:
        n_jobs = os.cpu_count() or 1
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=n_jobs, user_api="blas")
    except Exception:
        from contextlib import nullcontext
        return nullcontext()


def _csr_at_dense_best(
    M,
    dense: np.ndarray,
    n_jobs: int,
    chunk_cells: int,
) -> np.ndarray:
    """Dispatch to the fastest kernel for ``M @ dense``.

    - If ``M`` is already dense (``np.ndarray``) → BLAS sgemm via
      ``numpy.matmul`` with multi-threaded BLAS.
    - If ``M`` is sparse with density > ~3% → densify and use BLAS; on a
      motif matrix with 40% density this is ~10× faster than the numba CSR
      kernel. (Callers sharing ``M`` across multiple matmuls should
      densify once up-front via :func:`_maybe_densify_M` rather than
      relying on this branch.)
    - Sparse ``M`` with low density → numba prange CSR kernel (truly
      skips zeros).
    - No numba → threaded scipy.
    """
    if isinstance(M, np.ndarray):
        if M.dtype != np.float32:
            M = M.astype(np.float32, copy=False)
        with _set_blas_threads_ctx(n_jobs):
            out = np.matmul(M, dense)
        if out.dtype != np.float32:
            out = out.astype(np.float32, copy=False)
        return out
    n_rows, n_cols_M = M.shape
    density = M.nnz / max(n_rows * n_cols_M, 1)
    if density >= _M_DENSE_DISPATCH_DENSITY:
        M_dense = M.toarray()
        if M_dense.dtype != np.float32:
            M_dense = M_dense.astype(np.float32, copy=False)
        with _set_blas_threads_ctx(n_jobs):
            out = np.matmul(M_dense, dense)
        if out.dtype != np.float32:
            out = out.astype(np.float32, copy=False)
        return out
    if _HAS_NUMBA:
        out = np.zeros((M.shape[0], dense.shape[1]), dtype=np.float32)
        _nb_csr_at_dense(
            M.data.astype(np.float32, copy=False),
            M.indices.astype(np.int64, copy=False),
            M.indptr.astype(np.int64, copy=False),
            dense, out,
        )
        return out
    return _threaded_csr_at_dense(M, dense, n_jobs, chunk_cells)


def _maybe_densify_M(M_csr: sp.csr_matrix) -> "np.ndarray | sp.csr_matrix":
    """Return ``M`` as ``np.ndarray`` if its density crosses the BLAS
    dispatch threshold, otherwise unchanged. Densifying up-front once lets
    the caller reuse the dense copy across multiple ``M @ dense`` calls
    instead of paying the conversion per call.
    """
    density = M_csr.nnz / max(M_csr.shape[0] * M_csr.shape[1], 1)
    if density >= _M_DENSE_DISPATCH_DENSITY:
        out = M_csr.toarray()
        if out.dtype != np.float32:
            out = out.astype(np.float32, copy=False)
        return out
    return M_csr


def _parallel_densify_csc(
    X_csc: sp.csc_matrix,
    n_jobs: int,
    chunk_cols: int,
    dtype=np.float32,
) -> np.ndarray:
    """Densify a CSC matrix into a C-contiguous (rows × cols) array using
    column-chunked parallel ``toarray`` calls.

    Single-threaded ``X_csc.toarray(order='C')`` is ~2× slower than
    ``.todense()`` (F-contig) because scipy writes strided columns into the
    C-contig output. Splitting by columns lets multiple threads each write
    to their own stripe without contention — about 1.7× faster in practice.
    """
    n_rows, n_cols = X_csc.shape
    out = np.empty((n_rows, n_cols), dtype=dtype)
    if n_jobs in (None, 0, 1) or n_cols <= chunk_cols:
        out[:] = X_csc.toarray(order="C")
        if dtype is not None and out.dtype != dtype:
            out = out.astype(dtype, copy=False)
        return out
    def _work(start):
        end = min(start + chunk_cols, n_cols)
        block = X_csc[:, start:end].toarray()
        if block.dtype != dtype:
            block = block.astype(dtype, copy=False)
        out[:, start:end] = block
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        list(ex.map(_work, range(0, n_cols, chunk_cols)))
    return out


def _threaded_csr_at_dense(
    M: sp.csr_matrix,
    dense: np.ndarray,
    n_jobs: int,
    chunk_cells: int,
) -> np.ndarray:
    """Parallel ``M @ dense`` by column-chunking the dense operand.

    scipy's ``csr_matrix._mul_multivector`` is single-threaded; splitting
    the dense operand by column and dispatching to a ``ThreadPoolExecutor``
    gives near-linear speedup up to the memory-bandwidth bound (typically
    4-8× on 8 cores).
    """
    n_rows_M = M.shape[0]
    n_cells = dense.shape[1]
    out = np.empty((n_rows_M, n_cells), dtype=np.float32)
    if n_jobs in (None, 0, 1) or n_cells <= chunk_cells:
        out[:] = np.asarray(M @ dense, dtype=np.float32)
        return out
    starts = list(range(0, n_cells, chunk_cells))
    def _work(start):
        end = min(start + chunk_cells, n_cells)
        out[:, start:end] = np.asarray(M @ dense[:, start:end], dtype=np.float32)
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        list(ex.map(_work, starts))
    return out


def _build_peer_count_matrix(
    bg: np.ndarray,
    n_peaks: int,
) -> sp.csr_matrix:
    """Build ``B[i, k] = #{it : bg[i, it] == k}`` as an (n_peaks × n_peaks) CSR.

    Duplicates collapse automatically in ``coo_matrix``'s ``tocsr()``.
    The matrix is sparse (each row has at most ``n_iter`` nnz).
    """
    n_iter = bg.shape[1]
    rows = np.repeat(np.arange(n_peaks, dtype=np.int64), n_iter)
    cols = bg.ravel().astype(np.int64, copy=False)
    data = np.ones(n_peaks * n_iter, dtype=np.float32)
    B = sp.coo_matrix((data, (rows, cols)), shape=(n_peaks, n_peaks)).tocsr()
    return B


if _HAS_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
    def _nb_finalise_peer_stats(sum_arr, sumsq_arr, n_iter):
        """In-place: sum_arr → mean, sumsq_arr → (unbiased) variance.

        Avoids the ``mean²`` 12 GB-scale temporary that the naive expression
        ``sumsq/n - mean*mean`` would allocate. Runs each row in parallel.
        """
        n_rows = sum_arr.shape[0]
        n_cells = sum_arr.shape[1]
        inv_n = np.float32(1.0 / n_iter)
        # ddof=1 Bessel correction.  factor = n / (n-1) if n>1, else 0.
        factor = np.float32(n_iter / (n_iter - 1)) if n_iter > 1 else np.float32(0.0)
        for i in prange(n_rows):
            for c in range(n_cells):
                s  = sum_arr[i, c]
                ss = sumsq_arr[i, c]
                mu = s * inv_n
                sum_arr[i, c] = mu
                var = (ss * inv_n - mu * mu) * factor
                if var < 0.0:
                    var = 0.0
                sumsq_arr[i, c] = var


def _finalise_peer_stats_inplace(
    mu_peak: np.ndarray,
    var_peak: np.ndarray,
    n_iter: int,
) -> None:
    """Convert (Σv, Σv²) → (mean, unbiased-variance) in-place."""
    if _HAS_NUMBA:
        _nb_finalise_peer_stats(mu_peak, var_peak, n_iter)
        return
    # numpy fallback — allocates one temporary per op but is still streamed
    mu_peak  /= n_iter
    var_peak /= n_iter
    var_peak -= mu_peak * mu_peak
    if n_iter > 1:
        var_peak *= n_iter / (n_iter - 1)
    np.clip(var_peak, 0.0, None, out=var_peak)


def _peer_stats_chunk(
    X_dense_block: np.ndarray,          # (n_peaks, chunk_cells) dense float32
    B: sp.csr_matrix,                   # (n_peaks, n_peaks) peer-count matrix
    n_iter: int,
) -> tuple:
    """Return (μ, σ²) for one cell chunk via two sparse × dense matmuls.

    μ[i, c] = (1/n_iter) Σ_it X[bg[i, it], c]
           = (B @ X)[i, c] / n_iter

    σ²[i, c] = (B @ X²)[i, c] / n_iter − μ[i, c]²   (unbiased corrected below)
    """
    sum_  = np.asarray(B @ X_dense_block, dtype=np.float32)
    sumsq = np.asarray(B @ (X_dense_block * X_dense_block), dtype=np.float32)
    mu  = sum_ / n_iter
    var = sumsq / n_iter - mu * mu
    if n_iter > 1:
        var *= n_iter / (n_iter - 1)
    np.clip(var, 0.0, None, out=var)
    return mu, var


def _analytical_deviations(
    M_for_matmul,                         # CSR sparse or dense ndarray
    X_T: sp.csc_matrix,
    bg: np.ndarray,
    peak_frac: np.ndarray,
    cell_totals: np.ndarray,
    obs_dev: np.ndarray,
    expected_fg: np.ndarray,
    chunk_cells: int,
    n_jobs: int,
    verbose: bool,
    X_dense: Optional[np.ndarray] = None,
):
    n_peaks, n_cells = X_T.shape
    n_iter = bg.shape[1]

    if n_jobs in (None, 0):
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = os.cpu_count() or 1

    if X_dense is None:
        X_dense = _parallel_densify_csc(X_T, n_jobs, min(chunk_cells, 512))

    mu_peak  = np.zeros((n_peaks, n_cells), dtype=np.float32)
    var_peak = np.zeros((n_peaks, n_cells), dtype=np.float32)

    if _HAS_NUMBA:
        _console(
            f"analytical null | fused peer-stats (numba prange, local buf)",
            verbose,
        )
        bg64 = bg.astype(np.int64, copy=False)
        # After the kernel: mu_peak holds Σ v, var_peak holds Σ v². Convert
        # in-place to (mean, variance) to avoid the 12 GB-scale temporary
        # that ``var_peak - mu_peak * mu_peak`` would otherwise allocate.
        _nb_fused_peer_stats(bg64, X_dense, mu_peak, var_peak)
        _finalise_peer_stats_inplace(mu_peak, var_peak, n_iter)
    else:
        # Fallback: explicit peer-count matrix + scipy sparse × dense
        _console(f"analytical null | peer-count matrix (sparse)", verbose)
        B = _build_peer_count_matrix(bg, n_peaks)
        mu_peak[:]  = _threaded_csr_at_dense(B, X_dense,           n_jobs, chunk_cells)
        var_peak[:] = _threaded_csr_at_dense(B, X_dense * X_dense, n_jobs, chunk_cells)
        _finalise_peer_stats_inplace(mu_peak, var_peak, n_iter)

    mu_peak  = mu_peak.astype(np.float32, copy=False)
    var_peak = var_peak.astype(np.float32, copy=False)

    _console("analytical null | M @ μ_peak", verbose)
    mean_obs_bg = _csr_at_dense_best(M_for_matmul, mu_peak, n_jobs, chunk_cells)
    _console("analytical null | M @ σ²_peak", verbose)
    var_obs_bg  = _csr_at_dense_best(M_for_matmul, var_peak, n_jobs, chunk_cells)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_dev_bg = np.where(
            expected_fg > 0, (mean_obs_bg - expected_fg) / expected_fg, np.nan,
        ).astype(np.float32)
        std_dev_bg = np.where(
            expected_fg > 0, np.sqrt(var_obs_bg) / expected_fg, np.nan,
        ).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(
            std_dev_bg > 0, (obs_dev - mean_dev_bg) / std_dev_bg, np.nan,
        ).astype(np.float32)
    DEV = (obs_dev - mean_dev_bg).astype(np.float32)
    return Z, DEV


# ---------------------------------------------------------------------------
# Sample path (chromVAR-faithful)
# ---------------------------------------------------------------------------

def _sample_deviations(
    M_csr: sp.csr_matrix,
    X_T: sp.csc_matrix,
    bg: np.ndarray,
    peak_frac: np.ndarray,
    cell_totals: np.ndarray,
    obs_dev: np.ndarray,
    expected_fg: np.ndarray,
    n_iter: int,
    verbose: bool,
):
    n_motifs, n_peaks = M_csr.shape
    n_cells = X_T.shape[1]

    sum_bg   = np.zeros((n_motifs, n_cells), dtype=np.float64)
    sumsq_bg = np.zeros((n_motifs, n_cells), dtype=np.float64)
    M_indices = M_csr.indices
    M_data    = M_csr.data
    M_indptr  = M_csr.indptr
    log_every = max(n_iter // 10, 1)
    for it in range(n_iter):
        if verbose and (it % log_every == 0):
            _console(f"sample null | iteration {it + 1}/{n_iter}", verbose)
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
    var_bg *= (n_iter / max(n_iter - 1, 1))
    std_bg = np.sqrt(np.clip(var_bg, 0.0, None)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(std_bg > 0, (obs_dev - mean_bg) / std_bg, np.nan).astype(np.float32)
    DEV = (obs_dev - mean_bg).astype(np.float32)
    return Z, DEV


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_deviations(
    adata: AnnData,
    *,
    motif_key: str = "motif",
    bg_key: str = "bg_peaks",
    key_added: str = "motif_deviations",
    method: Literal["analytical", "sample"] = "analytical",
    chunk_cells: int = 4000,
    n_jobs: int = -1,
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
    method
        - ``"analytical"`` (default): closed-form null distribution via
          peak-wise mean and variance. ~10-40× faster than ``"sample"``
          and strictly lower-variance (it is the ``n_iter → ∞`` limit).
        - ``"sample"``: iterate the null once per bg column and accumulate
          running sample mean / variance — the original chromVAR procedure.
          Use when bit-level reproducibility with chromVAR / ArchR is
          required.
    chunk_cells
        Cell-chunk size for the analytical preprocess. Peak memory is
        ``chunk_cells × n_peaks × 4 B`` for the dense X block plus two
        equally sized ``(n_peaks, chunk_cells)`` stat buffers.
    n_jobs
        Thread pool size for the analytical preprocess. ``-1`` → all CPUs.
        Threads parallelise cleanly because the scipy / numpy work on each
        chunk releases the GIL. ``method="sample"`` is single-threaded.
    layer
        Optional counts layer to use instead of ``adata.X``.
    key_added
        - ``adata.obsm[key_added]`` receives the ``(n_cells, n_motifs)``
          bias-corrected Z-score matrix.
        - ``adata.obsm[key_added + '_raw']`` receives the uncorrected
          deviation (``obs_dev - mean_bg``).
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
    # peak-motif density picks the matmul path (BLAS dense vs numba CSR).
    # At densities ≥ ~3% BLAS wins by a lot, so we skip the CSR rebuild
    # and go straight to a dense ``(n_motifs, n_peaks)`` float32 array —
    # saves a tocsr conversion at ~0.5 s per 10 M nnz. The sample-path
    # below builds its own M_csr only when needed.
    density = M.nnz / max(M.shape[0] * M.shape[1], 1)
    if density >= _M_DENSE_DISPATCH_DENSITY:
        # Build the dense (n_motifs, n_peaks) view as F-contig:
        #   M.toarray()            → (n_peaks, n_motifs) C-contig dtype
        #   .astype(np.float32)    → in-place cast where possible
        #   .T                     → (n_motifs, n_peaks) F-contig view
        # sgemm consumes F-contig equally fast (via transA flag), so this
        # skips the ~0.2 s C-contig copy that
        # ``ascontiguousarray(M.toarray().T, dtype=np.float32)`` would do.
        M_dense = M.toarray().astype(np.float32, copy=False).T
        M_for_matmul = M_dense
        n_motifs, n_peaks = M_dense.shape
        M_csr = None
    else:
        M_csr = M.T.astype(np.float32).tocsr()
        M_for_matmul = M_csr
        n_motifs, n_peaks = M_csr.shape

    bg = np.asarray(adata.varm[bg_key], dtype=np.int32)
    assert bg.shape[0] == n_peaks
    n_iter = bg.shape[1]

    X = adata.layers[layer] if layer is not None else adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    # X.T of a CSR is already CSC — no materialising conversion needed.
    # Defer ``astype(np.float32)`` to the analytical path's densification so
    # the integer counts flow straight into the dense float32 block without
    # a separate sparse-level ~140 MB cast pass (saves ~0.1-0.2 s).
    X_T = X.T.tocsc()
    n_cells = X_T.shape[1]

    total_counts = float(X_T.sum())
    peak_frac = (np.asarray(X_T.sum(axis=1)).ravel().astype(np.float64)
                 / max(total_counts, 1.0)).astype(np.float32)
    cell_totals = np.asarray(X_T.sum(axis=0)).ravel().astype(np.float32)

    _console(
        f"cells={n_cells:,} | peaks={n_peaks:,} | motifs={n_motifs:,} | "
        f"bg_iter={n_iter} | method={method}",
        verbose,
    )

    # Resolve n_jobs once
    n_jobs_eff = os.cpu_count() or 1 if n_jobs is not None and n_jobs < 0 else (n_jobs or 1)

    # Foreground — shared across methods. Use threaded sparse×dense path
    # when the dense X_T fits comfortably (saves the sparse×sparse overhead
    # scipy pays when the output would be near-dense anyway).
    #
    # Critical: ``X_T`` is CSC, so ``.todense()`` returns an **F-contiguous**
    # dense matrix (strides (4, n_peaks·4)). That makes the numba inner
    # ``for c`` loop over cells strided by ``n_peaks·4 B`` — a 400 KB stride
    # for pbmc5k — and slows it by ~17× at real scale. Forcing C-contiguous
    # layout puts the contiguous axis along cells, matching the access
    # pattern in both ``_nb_fused_peer_stats`` and ``_nb_csr_at_dense``.
    dense_bytes = n_peaks * n_cells * 4
    use_dense_X = (method == "analytical") or dense_bytes <= 2_000_000_000
    if use_dense_X:
        _console(f"densifying X.T ({dense_bytes/1e9:.2f} GB, C-contig)", verbose)
        # Parallel column-chunked densification — ~1.7× faster than serial
        # ``X_T.toarray(order='C')`` on 16+ cores because each thread writes
        # to its own column stripe of the C-contig output concurrently.
        # A smaller chunk (~512 cells) gives each thread more work items to
        # balance; larger chunks (≥4000) serialise most of the work and lose
        # 3-4× on 16-core boxes. Use a separate knob from ``chunk_cells``
        # (which controls the CSR @ dense kernel) so callers that tune one
        # don't inadvertently slow the other.
        densify_chunk = min(chunk_cells, 512)
        X_dense = _parallel_densify_csc(X_T, n_jobs_eff, densify_chunk)
        _console("observed deviation (M @ X_dense)", verbose)
        obs_counts = _csr_at_dense_best(M_for_matmul, X_dense, n_jobs_eff, chunk_cells)
    else:
        X_dense = None
        _console("observed deviation (M @ X.T, sparse)", verbose)
        if M_csr is None:
            M_csr = sp.csr_matrix(M_for_matmul)
        obs_counts = np.asarray((M_csr @ X_T).todense(), dtype=np.float32)
    fg_fracs = np.asarray(M_for_matmul @ peak_frac).ravel().astype(np.float32)
    expected_fg = fg_fracs[:, None] * cell_totals[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        obs_dev = np.where(
            expected_fg > 0, (obs_counts - expected_fg) / expected_fg, np.nan,
        ).astype(np.float32)

    if method == "analytical":
        Z, DEV = _analytical_deviations(
            M_for_matmul, X_T, bg, peak_frac, cell_totals, obs_dev, expected_fg,
            chunk_cells=chunk_cells, n_jobs=n_jobs, verbose=verbose,
            X_dense=X_dense,
        )
    elif method == "sample":
        if M_csr is None:
            M_csr = sp.csr_matrix(M_for_matmul)
        # sample path wants float32 sparse X.T for the per-iteration
        # ``M_it @ X_T`` matmul. Analytical never touches X_T directly so we
        # only pay this cast on the sample branch.
        if X_T.dtype != np.float32:
            X_T = X_T.astype(np.float32)
        Z, DEV = _sample_deviations(
            M_csr, X_T, bg, peak_frac, cell_totals, obs_dev, expected_fg,
            n_iter=n_iter, verbose=verbose,
        )
    else:
        raise ValueError(f"unknown method {method!r}; expected 'analytical' or 'sample'")

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
        n_iterations=int(n_iter), method=method,
    )
    _console(
        f"done. obsm[{key_added!r}] = {Z.T.shape} float32 Z-scores; "
        f"[{key_added}_raw] holds uncorrected deviations",
        verbose,
    )
    return adata
