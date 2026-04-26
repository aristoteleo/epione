"""Pairwise correlation across per-celltype Hi-C maps.

Chang 2024 Fig 1f shows a celltype × celltype Pearson correlation
heatmap built from cis contact-matrix vectors — celltype maps that
share compartment / TAD architecture cluster together. This module
turns a dict of ``{celltype: cool_path}`` (the output of
:func:`epione.single.hic.pseudobulk_by_celltype`) into the
correlation matrix that feeds
:func:`epione.pl.plot_correlation_heatmap`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def _resolve_uri(path: Union[str, Path], resolution: Optional[int]) -> str:
    p = str(path)
    if "::" in p:
        return p
    if resolution is None:
        return p
    return f"{p}::resolutions/{int(resolution)}"


def _flatten_cis(
    cool_path: Union[str, Path],
    *,
    chromosomes: Optional[Sequence[str]] = None,
    resolution: Optional[int] = None,
    balance: bool = True,
    log: bool = True,
) -> np.ndarray:
    """Flatten the upper-triangle cis pixels of a cool into a 1-D vector.

    Skips trans pixels (different chrom1 vs chrom2) — celltype
    correlation is dominated by cis structure (compartments + TADs +
    loops), and trans pixels add noise + memory.
    """
    import cooler

    clr = cooler.Cooler(_resolve_uri(cool_path, resolution))
    bins = clr.bins()[:]
    if chromosomes is not None:
        keep = bins["chrom"].isin(list(chromosomes)).to_numpy()
    else:
        keep = np.ones(len(bins), dtype=bool)

    out_chunks: list[np.ndarray] = []
    for chrom in (chromosomes or clr.chromnames):
        if chrom not in clr.chromnames:
            continue
        M = clr.matrix(balance=balance).fetch(chrom)
        if M.size == 0:
            continue
        # Keep only the strict upper triangle; main diag dominated
        # by self-ligation, lower triangle is symmetric.
        iu = np.triu_indices_from(M, k=1)
        v = M[iu].astype(np.float64, copy=False)
        if log:
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.log2(v + 1e-9)
            v = np.where(np.isfinite(v), v, np.nan)
        out_chunks.append(v)
    return np.concatenate(out_chunks) if out_chunks else np.zeros(0)


def cluster_correlation(
    cool_paths: Mapping[str, Union[str, Path]],
    *,
    chromosomes: Optional[Sequence[str]] = None,
    resolution: Optional[int] = None,
    balance: bool = True,
    log: bool = True,
) -> Tuple[pd.DataFrame, list[str]]:
    """Pairwise Pearson correlation across per-celltype contact maps.

    For each input cool, flattens the (balanced, log-transformed by
    default) cis upper-triangle into a 1-D vector; then computes the
    full pairwise Pearson correlation matrix. Bin pairs with NaN in
    *any* celltype are dropped before correlation so all pairs are
    over a common support.

    Arguments:
        cool_paths: ``{celltype: cool_path}`` (e.g. output of
            :func:`pseudobulk_by_celltype`).
        chromosomes: subset to use. Default = chromosomes present in
            *every* cool.
        resolution: bp resolution for ``.mcool`` (passed through to
            each cool).
        balance: read ICE-balanced contacts (default). Set ``False``
            for raw counts.
        log: log2-transform before correlating — standard for Hi-C
            because the contact-frequency distribution is heavy-tailed.

    Returns:
        ``(corr_df, names)`` — ``corr_df`` is a square
        ``DataFrame`` indexed by celltype name; ``names`` is the
        celltype order (handy for plotting).
    """
    names = list(cool_paths.keys())
    if not names:
        raise ValueError("cool_paths is empty")

    if chromosomes is None:
        import cooler
        seen = None
        for p in cool_paths.values():
            clr = cooler.Cooler(_resolve_uri(p, resolution))
            seen = set(clr.chromnames) if seen is None else seen & set(clr.chromnames)
        chromosomes = sorted(seen or [])

    vecs: list[np.ndarray] = []
    for nm in names:
        v = _flatten_cis(
            cool_paths[nm],
            chromosomes=chromosomes,
            resolution=resolution,
            balance=balance,
            log=log,
        )
        vecs.append(v)

    if len(set(v.shape[0] for v in vecs)) != 1:
        raise ValueError(
            "celltype cools have different bin counts on the chosen "
            "chromosomes — they must share the same cooler binning."
        )

    M = np.vstack(vecs)
    finite = np.all(np.isfinite(M), axis=0)
    M = M[:, finite]
    if M.shape[1] == 0:
        raise ValueError(
            "no shared finite cis pixels across celltypes — check "
            "balancing / chromosomes / resolution."
        )

    corr = np.corrcoef(M)
    df = pd.DataFrame(corr, index=names, columns=names)
    return df, names
