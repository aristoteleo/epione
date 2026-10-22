"""Per-cell imputed-matrix files of :func:`impute_cells`, dense or sparse layout.

Dense layout (default): one ``n_bins x n_bins`` symmetric array per chromosome,
key = chromosome. Sparse layout (``impute_cells(sparse=True)``): scHiCluster's
upper triangle (diagonal included, within ``output_dist``) as CSR, keys
``<chrom>::data``, ``::indices``, ``::indptr``, ``::shape``, plus the marker
``__sparse__``. :class:`ImputedCell` reads both.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np

_MARK = "__sparse__"
_SEP = "::"


def save_imputed(path: Union[str, Path], mats: Dict[str, object], *, sparse: bool, compress: bool) -> None:
    """Write one cell. ``mats``: chromosome -> scipy upper triangle (sparse) or dense array."""
    save = np.savez_compressed if compress else np.savez
    if not sparse:
        save(path, **mats)
        return
    arrays = {_MARK: np.array(1, dtype=np.int8)}
    for c, U in mats.items():
        U = U.tocsr()
        U.sum_duplicates()
        arrays[f"{c}{_SEP}data"] = U.data.astype(np.float32)
        arrays[f"{c}{_SEP}indices"] = U.indices.astype(np.int32)
        arrays[f"{c}{_SEP}indptr"] = U.indptr.astype(np.int64)
        arrays[f"{c}{_SEP}shape"] = np.asarray(U.shape, dtype=np.int64)
    save(path, **arrays)


class ImputedCell:
    """One cell's imputed matrices.

    ``cell[chrom]`` returns the dense symmetric matrix (as the dense layout
    stores it); ``cell.upper(chrom)`` the CSR upper triangle and
    ``cell.values(chrom, rows, cols)`` single entries, without building the
    dense matrix.
    """

    def __init__(self, path: Union[str, Path]):
        self._z = np.load(path)
        self.sparse = _MARK in self._z.files
        if self.sparse:
            self.files = sorted({k.split(_SEP)[0] for k in self._z.files if _SEP in k})
        else:
            self.files = list(self._z.files)

    def __contains__(self, chrom) -> bool:
        return chrom in self.files

    def upper(self, chrom):
        from scipy.sparse import csr_matrix, triu
        if self.sparse:
            z = self._z
            return csr_matrix((z[f"{chrom}{_SEP}data"], z[f"{chrom}{_SEP}indices"], z[f"{chrom}{_SEP}indptr"]),
                              shape=tuple(z[f"{chrom}{_SEP}shape"]))
        return triu(csr_matrix(self._z[chrom])).tocsr()

    def __getitem__(self, chrom) -> np.ndarray:
        if not self.sparse:
            return self._z[chrom]
        U = self.upper(chrom).tocoo()
        arr = np.zeros(U.shape, dtype=np.float32)
        arr[U.row, U.col] = U.data
        arr[U.col, U.row] = U.data
        return arr

    def values(self, chrom, rows, cols) -> np.ndarray:
        """Entries ``(rows[k], cols[k])`` of the symmetric matrix, as float32."""
        if not self.sparse:
            return self._z[chrom][rows, cols].astype(np.float32)
        lo, hi = np.minimum(rows, cols), np.maximum(rows, cols)
        return np.asarray(self.upper(chrom)[lo, hi], dtype=np.float32).ravel()
