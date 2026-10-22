"""Per-cell imputed-matrix files of :func:`impute_cells`.

``output='cool'`` (default): one ``<cell>.cool`` per cell in scHiCluster's
imputed-cool format (``hicluster prepare-impute`` -> ``aggregate-chromosomes``):
bins over all chromosomes, cis pixels of the upper triangle within
``output_dist``, float32 ``count``. Readable by cooler, cooltools, HiGlass and
scHiCluster's downstream commands.

``output='npz'``: one ``<cell>.npz`` with a dense symmetric ``n_bins x n_bins``
array per chromosome (epione's earlier layout; also written by Higashi).

:class:`ImputedCell` reads both.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Mapping, Union

import numpy as np

PathLike = Union[str, Path]


def imputed_path(imputed_dir: PathLike, cell_id: str) -> Path:
    """The imputed file of one cell: ``<cell>.cool`` if present, else ``<cell>.npz``."""
    cool = Path(imputed_dir) / f"{cell_id}.cool"
    return cool if cool.exists() else Path(imputed_dir) / f"{cell_id}.npz"


def write_cool(path: PathLike, uppers: Mapping[str, object], chrom_sizes: Mapping[str, int],
               resolution: int, metadata: dict | None = None) -> None:
    """Write upper-triangle chromosome matrices (chromosome-relative bins) as one ``.cool``.

    Bins come from ``cooler.binnify(chrom_sizes, resolution)``; chromosomes in
    ``uppers`` are written in ``chrom_sizes`` order. Pixels beyond a
    chromosome's last bin (scHiCluster sizes matrices ``size // res + 1``) are
    dropped.
    """
    import cooler
    import pandas as pd

    sizes = pd.Series({str(k): int(v) for k, v in chrom_sizes.items()})
    bins = cooler.binnify(sizes, int(resolution))
    counts = bins.groupby("chrom", sort=False, observed=True).size()
    offsets = dict(zip(counts.index.astype(str), np.r_[0, np.cumsum(counts.to_numpy())][:-1]))
    n_per = dict(zip(counts.index.astype(str), counts.to_numpy()))

    def pixels():
        for chrom in sizes.index:
            if chrom not in uppers:
                continue
            U = uppers[chrom].tocoo()
            m = (U.row <= U.col) & (U.col < n_per[chrom]) & (U.data != 0)
            r, c, v = U.row[m].astype(np.int64), U.col[m].astype(np.int64), U.data[m].astype(np.float32)
            o = np.lexsort((c, r))
            off = offsets[chrom]
            yield pd.DataFrame({"bin1_id": r[o] + off, "bin2_id": c[o] + off, "count": v[o]})

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    cooler.create_cooler(str(tmp), bins, pixels(), ordered=True, dtypes={"count": np.float32},
                         metadata=metadata or {})
    os.replace(tmp, path)


def write_npz(path: PathLike, uppers: Mapping[str, object], *, compress: bool) -> None:
    """Dense symmetric matrix per chromosome (epione's ``.npz`` layout)."""
    mats: Dict[str, np.ndarray] = {}
    for chrom, U in uppers.items():
        U = U.tocoo()
        arr = np.zeros(U.shape, dtype=np.float32)
        arr[U.row, U.col] = U.data
        mats[chrom] = arr + arr.T - np.diag(np.diag(arr))
    (np.savez_compressed if compress else np.savez)(path, **mats)


class ImputedCell:
    """One cell's imputed matrices, from a ``.cool`` or a ``.npz``.

    ``cell.files``: imputed chromosomes. ``cell[chrom]``: dense symmetric
    matrix. ``cell.upper(chrom)``: CSR upper triangle. ``cell.values(chrom,
    rows, cols)``: single entries, without building the dense matrix from a
    ``.cool``.
    """

    def __init__(self, path: PathLike):
        path = Path(path)
        self.path = path
        self.is_cool = path.suffix == ".cool"
        if self.is_cool:
            import cooler
            self._clr = cooler.Cooler(str(path))
            meta = self._clr.info.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            self.files = list(meta.get("imputed_chromosomes", self._clr.chromnames))
        else:
            self._z = np.load(path)
            self.files = list(self._z.files)

    def __contains__(self, chrom) -> bool:
        return chrom in self.files

    def upper(self, chrom):
        from scipy.sparse import csr_matrix, triu
        if self.is_cool:
            M = self._clr.matrix(balance=False, sparse=True).fetch(chrom)
            return triu(M).tocsr().astype(np.float32)
        return triu(csr_matrix(self._z[chrom])).tocsr()

    def __getitem__(self, chrom) -> np.ndarray:
        if self.is_cool:
            return np.asarray(self._clr.matrix(balance=False).fetch(chrom), dtype=np.float32)
        return self._z[chrom]

    def values(self, chrom, rows, cols) -> np.ndarray:
        """Entries ``(rows[k], cols[k])`` of the symmetric matrix, as float32."""
        if not self.is_cool:
            return self._z[chrom][rows, cols].astype(np.float32)
        lo, hi = np.minimum(rows, cols), np.maximum(rows, cols)
        return np.asarray(self.upper(chrom)[lo, hi], dtype=np.float32).ravel()
