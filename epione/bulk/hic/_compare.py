"""Compare and correlate Hi-C matrices (HiCExplorer ``hicCompareMatrices`` /
``hicCorrelate``).

Ports of ``hicexplorer/hicCompareMatrices.py`` and the correlation part of
``hicexplorer/hicCorrelate.py`` (deeptools/HiCExplorer 3.7.6, GPL-3.0,
Copyright (c) Fidel Ramírez, Joachim Wolff and contributors).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.sparse import triu

from ._hicmatrix import HiCMat


def _uri(path, resolution):
    p = str(path)
    if "::" in p or resolution is None:
        return p
    return f"{p}::resolutions/{int(resolution)}"


def compare_matrices(
    cool_path1: Union[str, Path],
    cool_path2: Union[str, Path],
    *,
    operation: str = "log2ratio",
    normalize: bool = True,
    resolution: Optional[int] = None,
    out_path: Optional[Union[str, Path]] = None,
):
    """Pixel-wise ``diff`` / ``ratio`` / ``log2ratio`` of two Hi-C matrices.

    Same computation as ``hicCompareMatrices``: each matrix is loaded with its
    ``weight`` applied, optionally scaled to sum 1, then combined on the
    non-zero pixels (a pixel missing from the second matrix gives no ratio).
    Bins masked in either input are masked in the result.

    Arguments:
        cool_path1, cool_path2: ``.cool`` / ``.mcool::resolutions/N`` with the
            same bins and chromosome order.
        operation: ``"diff"`` (1 − 2), ``"ratio"`` (1 / 2) or ``"log2ratio"``.
        normalize: divide each matrix by its total before comparing
            (HiCExplorer's default; ``False`` is its ``--noNorm``).
        resolution: bp resolution for ``.mcool`` inputs.
        out_path: if given, also write the result as a ``.cool`` (float
            pixels, no ``weight`` column).

    Returns:
        ``(matrix, bins)``: the symmetric ``scipy.sparse.csr_matrix`` over all
        bins, with masked bins left empty, and the bins ``DataFrame``
        (``chrom``, ``start``, ``end``, ``masked``).
    """
    if operation not in ("diff", "ratio", "log2ratio"):
        raise ValueError("operation must be 'diff', 'ratio' or 'log2ratio'")
    h1 = HiCMat.from_cool(_uri(cool_path1, resolution))
    h2 = HiCMat.from_cool(_uri(cool_path2, resolution))
    if h1.matrix.shape != h2.matrix.shape:
        raise ValueError("the two matrices have different sizes")
    if list(h1.chrBinBoundaries.items()) != list(h2.chrBinBoundaries.items()):
        raise ValueError("the two matrices have different chromosome order")
    if normalize:
        h1.matrix.data = h1.matrix.data.astype(float) / h1.matrix.data.sum()
        h2.matrix.data = h2.matrix.data.astype(float) / h2.matrix.data.sum()
    nan_bins = sorted(set(h1.nan_bins).union(h2.nan_bins))
    if operation == "diff":
        new = h1.matrix - h2.matrix
    else:
        h2.matrix.data = float(1) / h2.matrix.data
        new = h1.matrix.multiply(h2.matrix)
        new.eliminate_zeros()
        if operation == "log2ratio":
            new.data = np.log2(new.data)
            new.eliminate_zeros()
    from scipy.sparse import csr_matrix
    h1.matrix = csr_matrix(new)
    h1.maskBins(nan_bins)
    h1.restoreMaskedBins()
    masked = np.zeros(h1.matrix.shape[0], dtype=bool)
    masked[np.asarray(h1.nan_bins, dtype=int)] = True
    bins = pd.DataFrame(h1.cut_intervals, columns=["chrom", "start", "end", "_x"]).drop(columns="_x")
    bins["masked"] = masked
    if out_path is not None:
        _write_float_cool(h1.matrix, bins[["chrom", "start", "end"]], out_path)
    return h1.matrix, bins


def _write_float_cool(matrix, bins, out_path):
    import cooler
    m = triu(matrix, format="csr")
    m.data[np.isnan(m.data)] = 0
    m.eliminate_zeros()
    m.sort_indices()
    coo = m.tocoo()
    pixels = pd.DataFrame({"bin1_id": coo.row.astype(np.int64), "bin2_id": coo.col.astype(np.int64),
                           "count": coo.data.astype(float)})
    cooler.create_cooler(str(out_path), bins=bins, pixels=pixels, dtypes={"count": float},
                         ordered=True)


def correlate_matrices(
    cool_paths: Sequence[Union[str, Path]],
    *,
    method: str = "pearson",
    log1p: bool = False,
    distance_range: Optional[tuple] = None,
    chromosomes: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    resolution: Optional[int] = None,
) -> pd.DataFrame:
    """Pairwise correlation of Hi-C matrices (``hicCorrelate``).

    Each matrix is restricted to intra-chromosomal pixels, the diagonal is
    zeroed, and only the upper triangle within ``distance_range`` is kept.
    Pixels are compared over the union of non-zero positions (a pixel that is
    zero in one matrix counts as zero), excluding bins masked in any input and
    positions where both values sum to at most 1 -- HiCExplorer's rule.

    Arguments:
        cool_paths: two or more ``.cool`` files with identical bins.
        method: ``"pearson"`` or ``"spearman"``.
        log1p: correlate ``log1p`` of the contacts.
        distance_range: ``(min_bp, max_bp)`` genomic distance window
            (HiCExplorer's ``--range min:max``).
        chromosomes: restrict to these chromosomes.
        labels: row/column names; default the file names.
        resolution: bp resolution for ``.mcool`` inputs.

    Returns:
        Symmetric ``DataFrame`` of correlation coefficients.
    """
    from scipy.stats import pearsonr, spearmanr

    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    corr = {"pearson": pearsonr, "spearman": spearmanr}[method]
    paths = [_uri(p, resolution) for p in cool_paths]
    if labels is None:
        labels = [Path(str(p).split("::")[0]).name for p in cool_paths]
    if len(labels) != len(paths):
        raise ValueError("labels must match the number of matrices")
    mats, all_mat, all_nan = [], None, []
    for p in paths:
        if chromosomes is not None and len(chromosomes) == 1:
            hm = HiCMat.from_cool(p, region=chromosomes[0])
        else:
            hm = HiCMat.from_cool(p)
            if chromosomes:
                hm.keepOnlyTheseChr(list(chromosomes))
            hm.filterOutInterChrCounts()
        hm.diagflat(0)
        bin_size = hm.getBinSize()
        all_nan = np.unique(np.concatenate([all_nan, hm.nan_bins]))
        m = triu(hm.matrix, k=0, format="csr")
        if distance_range is not None:
            min_dist, max_dist = int(distance_range[0]), int(distance_range[1])
            if max_dist < bin_size:
                raise ValueError(f"max distance must be larger than the bin size ({bin_size})")
            max_depth_in_bins = int(max_dist / bin_size)
            max_dist = max_dist // bin_size
            min_dist = min_dist // bin_size
            m = triu(m, k=0, format="csr") - triu(m, k=max_depth_in_bins, format="csr")
            m.eliminate_zeros()
            coo = m.tocoo()
            dist = coo.col - coo.row
            keep = np.flatnonzero((dist <= max_dist) & (dist >= min_dist))
            coo.data, coo.row, coo.col = coo.data[keep], coo.row[keep], coo.col[keep]
            m = coo.tocsr()
        else:
            m = triu(m, k=0, format="csr")
        if log1p:
            m.data = np.log1p(m.data)
        all_mat = m if all_mat is None else all_mat + m
        mats.append(m)
    keep_ids = np.delete(list(range(all_mat.shape[1])), all_nan.astype("int"))
    all_mat = all_mat[keep_ids, :][:, keep_ids]
    big = None
    for m in mats:
        m = m[keep_ids, :][:, keep_ids]
        v = (m + all_mat).data - all_mat.data
        big = v if big is None else np.vstack([big, v])
    big = np.ma.masked_invalid(big).T
    n = len(paths)
    res = np.zeros((n, n), dtype=float)
    rows, cols = np.triu_indices(n)
    for r, c in zip(rows, cols):
        if r == c:
            res[r, c] = 1
            continue
        sub = big[:, [r, c]]
        sub = sub[sub.sum(axis=1) > 1, :]
        res[r, c] = corr(sub[:, 0], sub[:, 1])[0]
    res = res + np.triu(res, 1).T
    return pd.DataFrame(res, index=list(labels), columns=list(labels))
