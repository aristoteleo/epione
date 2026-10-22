"""Faster scHiCluster imputation for windowed random walks (``window_size`` < chromosome).

Same signature and output as
:func:`schicluster.impute.impute_chromosome.impute_chromosome`. When the
random walk runs in windows (the 10 kb recipe: ``window_size=30 Mb``,
``step_size=10 Mb``), it replaces upstream's slow steps:

* the Gaussian convolution scatters the nonzero contacts with numba instead
  of filtering a dense ``n_bins x n_bins`` copy;
* the random walk calls scipy's sparse kernels directly, with tight output
  bounds and the invariant ``rp * I`` hoisted out of the loop;
* windows are summed into one dense accumulator instead of sparse slice
  assignments, and the distance band is cut in one pass over the nonzeros
  (no ``np.triu_indices`` over the whole chromosome).

About 3x faster and 50-85 % less memory per chromosome at 10 kb; results
differ from upstream only by float32 rounding (max absolute difference
~1e-7). A window whose random-walk residual lands within rounding of ``tol``
can stop one iteration earlier or later than upstream; that window then
differs more. Every other setting (window covering the chromosome, ``rp=1``,
``chrom='all'``) runs upstream unchanged. Adapted from an optimisation of
scHiCluster's ``impute_chromosome`` (``scHiCluster_impute_chromosome``).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.sparse import _sparsetools, csr_matrix, diags, eye, save_npz


def _kernels():
    from . import _impute_fast_kernels as k
    return k


def random_walk(P, rp, tol):
    """``Q <- P ((1 - rp) Q + rp I)`` until ``||Q_new - Q||_F < tol`` (max 30 steps)."""
    n = P.shape[0]
    Q = P.copy()
    idx = np.int32
    Ap = np.asarray(P.indptr, dtype=idx)
    Aj = np.asarray(P.indices, dtype=idx)
    size = max(P.nnz * 2, 100_000)
    bufs = [(np.empty(n + 1, idx), np.empty(size, idx), np.empty(size, np.float32)) for _ in range(2)]
    diff_ip = np.empty(n + 1, idx)
    diff_ij = np.empty(size * 2, idx)
    diff_ix = np.empty(size * 2, np.float32)
    one_minus_rp = np.float32(1.0 - rp)
    rp_I = (rp * eye(n, dtype=np.float32)).tocsr()
    b = 0
    for _ in range(30):
        Qc = Q._with_data(Q.data * one_minus_rp, copy=False) + rp_I
        Bp = np.asarray(Qc.indptr, dtype=idx)
        Bj = np.asarray(Qc.indices, dtype=idx)
        bound = min(int(np.diff(Bp)[P.indices].sum()), n * n)
        if bound > bufs[0][1].size:
            bufs = [(np.empty(n + 1, idx), np.empty(bound, idx), np.empty(bound, np.float32))
                    for _ in range(2)]
        ip, ij, ix = bufs[b]
        _sparsetools.csr_matmat(n, n, Ap, Aj, P.data, Bp, Bj, Qc.data, ip, ij, ix)
        nnz = int(ip[-1])
        Q_new = csr_matrix((ix[:nnz], ij[:nnz], ip), shape=(n, n), copy=False)
        dbound = min(Q.nnz + nnz, n * n)
        if dbound > diff_ij.size:
            diff_ij = np.empty(dbound, idx)
            diff_ix = np.empty(dbound, np.float32)
        _sparsetools.csr_minus_csr(n, n, np.asarray(Q.indptr, dtype=idx), np.asarray(Q.indices, dtype=idx),
                                   Q.data, ip, ij[:nnz], ix[:nnz], diff_ip, diff_ij, diff_ix)
        delta = np.linalg.norm(diff_ix[:int(diff_ip[-1])])
        Q = Q_new
        if delta < tol:
            break
        b = 1 - b
    return Q.copy()


def _dense_to_csr(dense):
    k = _kernels()
    n_rows, n_cols = dense.shape
    indptr = np.empty(n_rows + 1, dtype=np.int32)
    k.dense_count_row_nnz(dense, indptr)
    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=dense.dtype)
    k.dense_fill_csr(dense, indptr, indices, data)
    return csr_matrix((data, indices, indptr), shape=(n_rows, n_cols), copy=False)


def _gaussian(A, std, truncate):
    """``gaussian_filter(A.toarray(), std, mode='mirror', truncate=truncate)`` for sparse ``A``."""
    r = int(std * truncate + 0.5)
    if r * 3 >= min(A.shape):       # repeated mirror folds: use scipy
        return csr_matrix(gaussian_filter(A.astype(np.float32).toarray(), std, order=0,
                                          mode="mirror", truncate=truncate))
    k = _kernels()
    x = np.arange(-r, r + 1).astype(np.float64)
    g = np.exp(-x ** 2 / (2 * std * std))
    g = (g / g.sum()).astype(np.float32)
    coo = A.tocoo()
    out = np.zeros(A.shape, dtype=np.float32)
    k.sparse_gauss_2d(coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.float32),
                      A.shape[0], A.shape[1], np.outer(g, g).astype(np.float32), r, out)
    return _dense_to_csr(out)


def _band(E, band_width):
    k = _kernels()
    n = E.shape[0]
    out_indptr = np.empty(n + 1, dtype=np.int32)
    out_indices = np.empty(E.nnz, dtype=np.int32)
    out_data = np.empty(E.nnz, dtype=E.data.dtype)
    nnz = k.filter_upper_band(np.asarray(E.indptr, dtype=np.int32), np.asarray(E.indices, dtype=np.int32),
                              E.data, band_width, out_indptr, out_indices, out_data)
    return csr_matrix((out_data[:nnz], out_indices[:nnz], out_indptr), shape=E.shape, copy=False)


def _n_bins(chrom, resolution, scool_url, chrom_size_path):
    if scool_url is not None:
        import cooler
        return int(cooler.Cooler(scool_url).chromsizes[chrom] // resolution + 1)
    sizes = pd.read_csv(chrom_size_path, sep="\t", index_col=0, header=None).squeeze(axis=1)
    return int(sizes.loc[chrom] // resolution + 1)


def impute_chromosome(chrom, resolution, output_path, scool_url=None, contact_path=None,
                      chrom_size_path=None, logscale=False, pad=1, std=1, rp=0.5, tol=0.01,
                      window_size=5000000000, step_size=10000000, output_dist=5000000000,
                      min_cutoff=0, chrom1=1, pos1=2, chrom2=5, pos2=6):
    """Drop-in for scHiCluster's ``impute_chromosome``; see the module docstring."""
    from schicluster.impute.impute_chromosome import impute_chromosome as upstream

    kw = dict(scool_url=scool_url, contact_path=contact_path, chrom_size_path=chrom_size_path,
              logscale=logscale, pad=pad, std=std, rp=rp, tol=tol, window_size=window_size,
              step_size=step_size, output_dist=output_dist, min_cutoff=min_cutoff,
              chrom1=chrom1, pos1=pos1, chrom2=chrom2, pos2=pos2)
    if rp == 1 or chrom == "all" or (scool_url is None and (contact_path is None or chrom_size_path is None)):
        return upstream(chrom, resolution, output_path, **kw)
    n_bins = _n_bins(chrom, resolution, scool_url, chrom_size_path)
    ws, ss = int(window_size // resolution), int(step_size // resolution)
    if ws >= n_bins:
        return upstream(chrom, resolution, output_path, **kw)
    try:
        _impute_windowed(chrom, resolution, output_path, n_bins, ws, ss, **kw)
    except Exception as e:  # unexpected scipy internals: fall back, never fail silently
        warnings.warn(f"fast imputation failed on {chrom} ({type(e).__name__}: {e}); "
                      "using scHiCluster's implementation", RuntimeWarning)
        upstream(chrom, resolution, output_path, **kw)


def _impute_windowed(chrom, resolution, output_path, n_bins, ws, ss, *, scool_url, contact_path,
                     chrom_size_path, logscale, pad, std, rp, tol, window_size, step_size,
                     output_dist, min_cutoff, chrom1, pos1, chrom2, pos2):
    if scool_url is not None:
        import cooler
        A = cooler.Cooler(scool_url).matrix(balance=False, sparse=True).fetch(chrom).tocsr()
    else:
        A = pd.read_csv(contact_path, sep="\t", header=None, index_col=None, comment="#")[[chrom1, pos1, chrom2, pos2]]
        A = A.loc[(A[chrom1] == chrom) & (A[chrom2] == chrom)]
        A[[pos1, pos2]] = (A[[pos1, pos2]] - 1) // resolution
        A = A.groupby(by=[pos1, pos2])[chrom1].count().reset_index()
        A = csr_matrix((A[chrom1].astype(np.int32), (A[pos1], A[pos2])), (n_bins, n_bins))
        A = A + A.T
    if logscale:
        A.data = np.log2(A.data + 1)
    A = A - diags(A.diagonal())
    if pad > 0:
        A = _gaussian(A, std=std, truncate=pad)
    A = A - diags(A.diagonal())

    # window masks, as upstream
    idx = (np.repeat(np.arange(ws), ws), np.tile(np.arange(ws), ws))
    keep = np.abs(idx[1] - idx[0]) < (output_dist // resolution + 1)
    idx = (idx[0][keep], idx[1][keep])
    s = idx[0] + idx[1]

    def _mask(m):
        return csr_matrix((np.ones(int(m.sum())), (idx[0][m], idx[1][m])), (ws, ws))

    mask1 = _mask(s < ws + ss)
    mask2 = _mask(s >= ((n_bins - ws) // ss * 2 + 1) * ss + 3 * ws - 2 * n_bins)
    mask0 = _mask((s < ws + ss) & (s >= ws - ss))

    k = _kernels()
    E_dense = np.zeros(A.shape, dtype=np.float32)
    last = n_bins - ws
    for ll in list(range(0, last, ss)) + [last]:
        B = A[ll:ll + ws, ll:ll + ws]
        B = B + diags((B.sum(axis=0).A.ravel() == 0).astype(int))
        P = diags(1 / B.sum(axis=0).A.ravel()).dot(B).astype(np.float32).tocsr()
        Etmp = random_walk(P, rp, tol)
        mask = mask1 if ll == 0 else mask2 if ll == last else mask0
        part = Etmp.multiply(mask).tocsr()
        k.dense_add_sparse_block(E_dense, ll, ll, part.indptr, part.indices, part.data)
    E = _dense_to_csr(E_dense)
    del E_dense

    E += E.T
    d = E.sum(axis=0).A.ravel()
    d[d == 0] = 1
    b = diags(1 / np.sqrt(d))
    E = b.dot(E).dot(b)
    E = _band(E.tocsr(), min(output_dist // resolution + 1, n_bins))
    if min_cutoff > 0:
        E = E.multiply(E > min_cutoff)
    save_npz(output_path, E)
