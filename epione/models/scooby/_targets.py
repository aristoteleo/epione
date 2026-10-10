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

This module imports only :mod:`epione._insertions` from the rest of
``epione``, and that module is itself dependency-free.
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

# Shared with Epione's footprinting; this module adds only scooby's
# target transform on top.
from epione._insertions import (  # noqa: F401  (re-exported)
    InsertionReader,
    ensure_tabix_index,
    groups_to_matrix,
)

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

class FragmentInsertionReader(InsertionReader):
    """:class:`epione._insertions.InsertionReader` plus scooby's targets.

    The reading itself lives in the shared reader; this subclass only
    adds :meth:`pseudobulk`, whose defaults (32 bp bins, x0.05) are
    upstream scooby's ATAC target transform. Cuts default to ``(s, e)``,
    the convention scooby inherits from snapATAC2.
    """

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
