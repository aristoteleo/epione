"""Random-access Tn5 insertion counts from a tabix-indexed fragments file.

The one reader the rest of Epione should use when it needs per-base
insertions for a region and a set of cell groups. It reads only the
interval asked for, returns sparse ``cells x bp`` counts, collapses cells
into groups with a single sparse product, and is safe to hand to forked
workers. It started life inside :mod:`epione.models.scooby` and moved
here once footprinting needed the same thing.

Cut-site conventions
--------------------
Where the Tn5 cut "is" for a fragment ``[start, end)`` is a convention,
and Epione had three of them in use. They differ by one base on the
fragment-end cut, which does not matter for peak counts but does at the
single-base resolution footprinting works at -- a bias track and an
insertion track that disagree by 1 bp blur every footprint. The reader
therefore takes the convention as an explicit ``shift``:

==================  ==============  =========================================
``shift``           cuts            used by
==================  ==============  =========================================
``(0, 0)``          ``s``, ``e``    scPrinter / PRINT, scooby, snapATAC2
``(0, -1)``         ``s``, ``e-1``  ArchR, Epione's ``get_footprints``
==================  ==============  =========================================

Both assume fragments already carry the Tn5 +4/-5 adjustment, as
CellRanger output does. A bias model is trained under one of these, so
it is the bias model, not the caller, that should pick the shift; see
:class:`epione.tl.Tn5Bias`.

This module imports nothing else from Epione, so it stays usable in lean
model-training environments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp

__all__ = ["InsertionReader", "ensure_tabix_index", "groups_to_matrix"]

#: Cut convention of scPrinter / PRINT / scooby: insertions at s and e.
SHIFT_PRINT = (0, 0)
#: Cut convention of ArchR and Epione's get_footprints: s and e-1.
SHIFT_ARCHR = (0, -1)


def ensure_tabix_index(path: Union[str, Path]) -> Path:
    """Ensure a ``.tbi`` index sits next to a bgzipped fragments file.

    Mirrors ``epione.pp._data._ensure_tabix_index`` but without the
    package import, for the reason given in the module docstring.
    """
    path = Path(path)
    tbi = path.with_suffix(path.suffix + ".tbi")
    if tbi.exists():
        return tbi
    if path.suffix != ".gz":
        raise RuntimeError(
            f"{path} must be bgzipped (.tsv.gz) and tabix-indexed for "
            "random access"
        )
    import pysam

    try:
        pysam.tabix_index(
            str(path), preset="bed", force=True, keep_original=True
        )
    except Exception as exc:  # pragma: no cover - depends on local files
        raise RuntimeError(
            f"could not build tabix index for {path}: {exc}"
        ) from exc
    return tbi


def groups_to_matrix(
    groups: Sequence[Iterable[int]],
    n_cells: int,
    *,
    dtype=np.float32,
) -> sp.csr_matrix:
    """Build a ``[n_groups, n_cells]`` 0/1 membership matrix.

    Left-multiplying a per-cell insertion matrix by this collapses cells
    into group pseudobulks in one sparse product, which keeps the
    aggregation out of Python loops regardless of how many groups there
    are.

    A cell listed twice in the same group is counted once, matching the
    ``sum_duplicates`` behaviour of upstream's insertion construction.
    """
    rows: list[int] = []
    cols: list[int] = []
    for g, members in enumerate(groups):
        idx = np.unique(np.asarray(list(members), dtype=np.int64))
        if idx.size and (idx.min() < 0 or idx.max() >= n_cells):
            raise IndexError(
                f"group {g} references cell index outside [0, {n_cells})"
            )
        rows.append(np.full(idx.size, g, dtype=np.int64))
        cols.append(idx)
    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
    else:
        row = col = np.empty(0, dtype=np.int64)
    return sp.csr_matrix(
        (np.ones(row.size, dtype=dtype), (row, col)),
        shape=(len(groups), n_cells),
    )


class InsertionReader:
    """Random-access Tn5 insertion counts from a fragments BED.

    Parameters
    ----------
    fragment_file
        Path to a bgzipped, tabix-indexed ``fragments.tsv.gz``. The
        index is built on first use if missing.
    barcodes
        Cell barcodes in the row order the caller wants. Fragments whose
        barcode is absent are skipped, which doubles as the whitelist.
    chrom_sizes
        Optional ``{chrom: length}``, used to clip fetch windows.
    barcode_suffix
        Appended to each entry of ``barcodes`` before matching against
        the file. 10x/CellRanger fragments carry a ``-1`` suffix that
        spatial position lists usually drop, so pass ``'-1'`` when
        indexing by bare spot barcodes.
    shift
        ``(shift_start, shift_end)`` added to the two cut sites. See the
        module docstring: ``(0, 0)`` for PRINT/scooby, ``(0, -1)`` for
        ArchR. Neither undoes or applies the Tn5 +4/-5 adjustment.
    weight_by_count
        If ``True``, weight each insertion by the duplicate-count column
        of the fragments file. Default ``False``, matching scPrinter and
        scooby, which build insertion matrices with ``np.ones``.

    Notes
    -----
    The object holds an open ``pysam.TabixFile``. It is **not** safe to
    share across processes; in a torch ``DataLoader`` with
    ``num_workers > 0``, construct one reader per worker (see
    :meth:`reopen`).
    """

    def __init__(
        self,
        fragment_file: Union[str, Path],
        barcodes: Sequence[str],
        *,
        chrom_sizes: Optional[Mapping[str, int]] = None,
        barcode_suffix: str = "",
        shift: tuple[int, int] = (0, 0),
        weight_by_count: bool = False,
    ) -> None:
        self.fragment_file = str(fragment_file)
        self.barcodes = list(barcodes)
        self.barcode_suffix = barcode_suffix
        self.shift = (int(shift[0]), int(shift[1]))
        self.weight_by_count = bool(weight_by_count)
        self.chrom_sizes = dict(chrom_sizes) if chrom_sizes else {}

        self._bc_to_idx = {
            bc + barcode_suffix: i for i, bc in enumerate(self.barcodes)
        }
        self._tbx = None
        self._contigs: Optional[set[str]] = None

    @property
    def n_cells(self) -> int:
        return len(self.barcodes)

    def _ensure_open(self):
        if self._tbx is None:
            import pysam

            ensure_tabix_index(self.fragment_file)
            self._tbx = pysam.TabixFile(self.fragment_file)
            self._contigs = set(self._tbx.contigs)
        return self._tbx

    def reopen(self) -> "InsertionReader":
        """Drop the cached handle so the next read opens a fresh one.

        Call this in a ``DataLoader`` worker-init hook: a ``TabixFile``
        inherited across a fork yields corrupt reads.
        """
        self._tbx = None
        self._contigs = None
        return self

    def close(self):
        if self._tbx is not None:
            self._tbx.close()
            self._tbx = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tbx"] = None
        state["_contigs"] = None
        return state

    def window_insertions(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        dtype=np.float32,
    ) -> sp.csr_matrix:
        """Per-cell insertion counts over ``[start, end)``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape ``[n_cells, end - start]``. Entry ``(i, p)`` counts
            insertions of cell ``i`` at position ``start + p``.
        """
        if end <= start:
            raise ValueError(f"empty interval {chrom}:{start}-{end}")
        width = end - start
        tbx = self._ensure_open()

        if chrom not in self._contigs:
            return sp.csr_matrix((self.n_cells, width), dtype=dtype)

        # Fetch one extra base on the left: a fragment lying entirely to
        # the left of the window can still have its ``end`` insertion at
        # ``start`` exactly, and such a fragment does not overlap
        # [start, end) so tabix would not return it.
        fetch_start = max(0, start - 1)
        fetch_end = end
        if chrom in self.chrom_sizes:
            fetch_end = min(fetch_end, self.chrom_sizes[chrom])
            if fetch_end <= fetch_start:
                return sp.csr_matrix((self.n_cells, width), dtype=dtype)

        bc_to_idx = self._bc_to_idx
        shift_s, shift_e = self.shift
        weight_by_count = self.weight_by_count

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for line in tbx.fetch(chrom, fetch_start, fetch_end):
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            i = bc_to_idx.get(parts[3])
            if i is None:
                continue
            try:
                s = int(parts[1]) + shift_s
                e = int(parts[2]) + shift_e
            except ValueError:
                continue
            if weight_by_count:
                w = (
                    float(parts[4])
                    if len(parts) >= 5 and parts[4].isdigit()
                    else 1.0
                )
            else:
                w = 1.0
            # Two independent insertion sites; either may fall outside.
            if start <= s < end:
                rows.append(i)
                cols.append(s - start)
                vals.append(w)
            if start <= e < end:
                rows.append(i)
                cols.append(e - start)
                vals.append(w)

        if not rows:
            return sp.csr_matrix((self.n_cells, width), dtype=dtype)

        mat = sp.coo_matrix(
            (
                np.asarray(vals, dtype=dtype),
                (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
            ),
            shape=(self.n_cells, width),
        ).tocsr()
        mat.sum_duplicates()
        return mat

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}({Path(self.fragment_file).name}, "
            f"n_cells={self.n_cells})"
        )


    def group_insertions(
        self,
        chrom: str,
        start: int,
        end: int,
        groups: Union[Sequence[Iterable[int]], sp.spmatrix, None] = None,
        *,
        dtype=np.float32,
    ) -> np.ndarray:
        """Base-resolution insertion counts per cell group.

        Parameters
        ----------
        chrom, start, end
            Interval, half-open.
        groups
            Sequence of cell-index iterables, a ``[n_groups, n_cells]``
            membership matrix from :func:`groups_to_matrix`, or ``None``
            for a single group of every cell.

        Returns
        -------
        np.ndarray
            ``[n_groups, end - start]`` counts.
        """
        ins = self.window_insertions(chrom, start, end, dtype=dtype)
        if groups is None:
            return np.asarray(ins.sum(axis=0), dtype=dtype)
        G = (groups if sp.issparse(groups)
             else groups_to_matrix(groups, self.n_cells, dtype=dtype))
        if G.shape[1] != self.n_cells:
            raise ValueError(
                f"group matrix has {G.shape[1]} columns but reader has "
                f"{self.n_cells} cells"
            )
        return np.asarray((G @ ins).todense(), dtype=dtype)
