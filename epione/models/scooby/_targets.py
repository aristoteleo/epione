"""Tn5 insertion targets for sequence→profile models, read on demand.

Upstream scooby requires a genome-wide ``cells × base-pair`` sparse
insertion matrix in ``adata.obsm['insertion']``, produced by a forked
snapATAC2 (``snapatac2-scooby``). For a mouse brain section that matrix
is ~2.7e9 columns wide; materialising one per sample is the main reason
the upstream pipeline needs a dedicated preprocessing environment and a
lot of RAM.

Epione already stores fragments the other way round: ``import_fragments``
records the tabix-indexed BED path in ``uns['files']['fragments']`` and
re-reads windows on demand (see :mod:`epione.pp._data`). This module
applies the same idea to model targets, so a training window is built by
a tabix ``fetch`` over exactly the interval being trained on.

Why that is cheap here: insertion density is low. A 196,608 bp training
window of a spatial ATAC section with ~1e8 in-tissue fragments holds on
the order of 1e4 fragments *summed over all spots*, so the per-window
read is small and the Python-level loop is not the bottleneck.

The same arithmetic is a warning about resolution. At ~1.5e4 fragments
per spot over a 2.7e9 bp genome, a single spot contributes ~1 fragment
(2 insertions) to a 196 kb window. Single-spot targets are therefore
almost entirely zero, and useful profiles require aggregating many
spots. Grouping is consequently first-class in this module: every
reader method takes ``groups``, and per-cell training is just the
special case of one group per cell (optionally widened by a k-NN
neighbourhood, which is how upstream scooby does it).

Conventions reproduced from upstream scooby (``scooby.data.scdata`` and
the ``frags_to_insertions`` helper in its data-preparation notebooks),
so that targets built here can be compared against upstream numbers:

* A fragment ``[start, end)`` deposits **one** insertion at ``start``
  and **one** at ``end``. The duplicate-count column of the fragments
  file is ignored (upstream builds the matrix with ``np.ones``); pass
  ``weight_by_count=True`` to opt out of that.
* Counts are summed into ``bin_size`` (32 bp) bins — upstream spells
  this ``avg_pool1d(kernel_size=32, stride=32) * 32``.
* ATAC targets are then scaled by ``0.05`` and **not** clipped or
  power-transformed. The ``clip_soft`` / ``-1 + (1 + x) ** 0.75``
  pathway in upstream applies to RNA coverage only.

This module deliberately imports nothing from the rest of ``epione``.
A training environment needs torch and the Borzoi backbone but has no
reason to carry Epione's motif-scanning stack, and ``import epione``
pulls in optional native dependencies (MOODS) that such an environment
will not have. Keeping the target layer free-standing means it can be
imported — and unit-tested — anywhere pysam is available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp

# Upstream scooby's ATAC target scale factor (``normalize_atac=True``).
ATAC_SCALE = 0.05
# Borzoi/scooby target bin width in bp.
BIN_SIZE = 32

__all__ = [
    "ATAC_SCALE",
    "BIN_SIZE",
    "FragmentInsertionReader",
    "bin_sum",
    "ensure_tabix_index",
    "groups_to_matrix",
    "insertions_to_coverage",
]


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


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def bin_sum(x: np.ndarray, bin_size: int = BIN_SIZE) -> np.ndarray:
    """Sum adjacent positions into fixed-width bins.

    Equivalent to upstream's ``avg_pool1d(x, kernel_size=bin_size,
    stride=bin_size) * bin_size``, without the torch dependency.

    Parameters
    ----------
    x
        Array with positions along the last axis. Its length must be an
        exact multiple of ``bin_size``.
    bin_size
        Bin width in bp.

    Returns
    -------
    np.ndarray
        ``x`` with the last axis reduced by a factor of ``bin_size``.
    """
    x = np.asarray(x)
    width = x.shape[-1]
    if width % bin_size:
        raise ValueError(
            f"last axis ({width}) must be a multiple of bin_size ({bin_size}); "
            "resize the interval so that its width divides evenly"
        )
    return x.reshape(*x.shape[:-1], width // bin_size, bin_size).sum(-1)


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


def insertions_to_coverage(
    insertions: sp.spmatrix,
    *,
    bin_size: int = BIN_SIZE,
    scale: float = ATAC_SCALE,
    dtype=np.float32,
) -> np.ndarray:
    """Turn base-pair insertion counts into a binned, scaled target.

    This is the tail of upstream's ``_process_atac``: bin by summing,
    then apply the ATAC scale factor. No clipping or power transform —
    those belong to the RNA pathway.
    """
    dense = np.asarray(insertions.todense(), dtype=dtype)
    binned = bin_sum(dense, bin_size)
    if scale != 1.0:
        binned = binned * dtype(scale)
    return binned


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class FragmentInsertionReader:
    """Random-access Tn5 insertion coverage from a fragments BED.

    Parameters
    ----------
    fragment_file
        Path to a bgzipped, tabix-indexed ``fragments.tsv.gz``. The
        index is built on first use if missing.
    barcodes
        Cell barcodes in the row order the caller wants. Fragments whose
        barcode is absent are skipped, which doubles as the whitelist.
    chrom_sizes
        Optional ``{chrom: length}``. Used to clip fetch windows; also
        lets :meth:`pseudobulk` reject out-of-range intervals early.
    barcode_suffix
        Appended to each entry of ``barcodes`` before matching against
        the file. 10x/CellRanger fragments carry a ``-1`` suffix that
        spatial position lists usually drop, so pass ``'-1'`` when
        indexing by bare spot barcodes.
    shift
        ``(shift_start, shift_end)`` applied to the two insertion sites.
        Zero for CellRanger output, which has already applied the Tn5
        +4/-5 correction. Upstream scooby likewise imports with
        ``shift_left=0, shift_right=0``.
    weight_by_count
        If ``True``, weight each insertion by the duplicate-count column
        of the fragments file. Default ``False`` to match upstream,
        which builds the insertion matrix with ``np.ones``.

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

    # -- lifecycle ---------------------------------------------------------

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

    def reopen(self) -> "FragmentInsertionReader":
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

    # -- reads -------------------------------------------------------------

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

    def pseudobulk(
        self,
        chrom: str,
        start: int,
        end: int,
        groups: Union[Sequence[Iterable[int]], sp.spmatrix, None] = None,
        *,
        bin_size: int = BIN_SIZE,
        scale: float = ATAC_SCALE,
        dtype=np.float32,
    ) -> np.ndarray:
        """Binned, scaled insertion profile per cell group.

        Parameters
        ----------
        chrom, start, end
            Target interval. ``end - start`` must be a multiple of
            ``bin_size``.
        groups
            Either a sequence of cell-index iterables (one entry per
            output row), or a precomputed ``[n_groups, n_cells]``
            membership matrix from :func:`groups_to_matrix`. ``None``
            means one group containing every cell.
        bin_size, scale
            See :func:`insertions_to_coverage`.

        Returns
        -------
        np.ndarray
            Shape ``[n_groups, (end - start) // bin_size]``.
        """
        ins = self.window_insertions(chrom, start, end, dtype=dtype)

        if groups is None:
            agg = sp.csr_matrix(ins.sum(axis=0))
        else:
            G = (
                groups
                if sp.issparse(groups)
                else groups_to_matrix(groups, self.n_cells, dtype=dtype)
            )
            if G.shape[1] != self.n_cells:
                raise ValueError(
                    f"group matrix has {G.shape[1]} columns but reader has "
                    f"{self.n_cells} cells"
                )
            agg = G @ ins

        return insertions_to_coverage(
            agg, bin_size=bin_size, scale=scale, dtype=dtype
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"FragmentInsertionReader({Path(self.fragment_file).name}, "
            f"n_cells={self.n_cells})"
        )
