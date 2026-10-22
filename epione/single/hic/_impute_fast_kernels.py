"""Numba kernels of the fast scHiCluster imputation (imported only when needed)."""
from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True)
def dense_count_row_nnz(dense, indptr):
    n_rows, n_cols = dense.shape
    indptr[0] = 0
    for i in range(n_rows):
        cnt = 0
        for j in range(n_cols):
            if dense[i, j] != 0.0:
                cnt += 1
        indptr[i + 1] = indptr[i] + cnt


@numba.njit(cache=True)
def dense_fill_csr(dense, indptr, indices, data):
    n_rows, n_cols = dense.shape
    for i in range(n_rows):
        pos = indptr[i]
        for j in range(n_cols):
            v = dense[i, j]
            if v != 0.0:
                indices[pos] = j
                data[pos] = v
                pos += 1


@numba.njit(cache=True)
def dense_add_sparse_block(dense, ll_row, ll_col, sp_indptr, sp_indices, sp_data):
    ws = sp_indptr.shape[0] - 1
    for i in range(ws):
        row_base = ll_row + i
        for k in range(sp_indptr[i], sp_indptr[i + 1]):
            dense[row_base, ll_col + sp_indices[k]] += sp_data[k]


@numba.njit(cache=True)
def sparse_gauss_2d(A_row, A_col, A_data, n_rows, n_cols, kernel_2d, r, out):
    """Scatter each entry over its (2r+1)^2 Gaussian neighbourhood, with one
    mirror image per axis (scipy ``mode='mirror'`` while r < n / 3)."""
    for k in range(A_row.shape[0]):
        i = A_row[k]
        j = A_col[k]
        v = A_data[k]
        row_srcs = np.empty(3, dtype=np.int64)
        row_srcs[0] = i
        n_row_src = 1
        if 1 <= i <= r:
            row_srcs[n_row_src] = -i
            n_row_src += 1
        if n_rows - 1 - r <= i <= n_rows - 2:
            row_srcs[n_row_src] = 2 * (n_rows - 1) - i
            n_row_src += 1
        col_srcs = np.empty(3, dtype=np.int64)
        col_srcs[0] = j
        n_col_src = 1
        if 1 <= j <= r:
            col_srcs[n_col_src] = -j
            n_col_src += 1
        if n_cols - 1 - r <= j <= n_cols - 2:
            col_srcs[n_col_src] = 2 * (n_cols - 1) - j
            n_col_src += 1
        for si in range(n_row_src):
            src_row = row_srcs[si]
            for sj in range(n_col_src):
                src_col = col_srcs[sj]
                for di in range(-r, r + 1):
                    tr = src_row + di
                    if tr < 0 or tr >= n_rows:
                        continue
                    for dj in range(-r, r + 1):
                        tc = src_col + dj
                        if tc < 0 or tc >= n_cols:
                            continue
                        out[tr, tc] += v * kernel_2d[di + r, dj + r]


@numba.njit(cache=True)
def filter_upper_band(E_indptr, E_indices, E_data, band_width, out_indptr, out_indices, out_data):
    n = E_indptr.shape[0] - 1
    out_pos = 0
    out_indptr[0] = 0
    for i in range(n):
        for k in range(E_indptr[i], E_indptr[i + 1]):
            j = E_indices[k]
            if 0 <= j - i < band_width:
                out_indices[out_pos] = j
                out_data[out_pos] = E_data[k]
                out_pos += 1
        out_indptr[i + 1] = out_pos
    return out_pos
