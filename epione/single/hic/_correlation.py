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


def cell_celltype_correlation(
    adata,
    celltype_cools: Mapping[str, Union[str, Path]],
    *,
    chromosomes: Optional[Sequence[str]] = None,
    resolution: Optional[int] = None,
    max_distance_bins: int = 20,
    z_score: bool = True,
) -> pd.DataFrame:
    """Per-cell × per-celltype Pearson correlation as a low-dim feature.

    For each cell in ``adata`` (which must have ``adata.uns['hic']``
    pointing at imputed ``.npz`` files via :func:`impute_cells`) and
    each celltype reference cool, compute the Pearson correlation
    between the cell's imputed cis upper-triangle (truncated to
    ``max_distance_bins``) and the celltype pseudobulk's. Returns a
    ``cell × celltype`` DataFrame — typically used as feature input
    to PCA/UMAP for the Chang 2024 Fig 1d-style cell embedding.

    With ``z_score=True`` (default) we per-row z-score the resulting
    matrix so each cell's features sum to 0 and have unit variance —
    this drops cell-depth artefacts and makes downstream UMAP
    discriminate fine subtype despite shared structure.

    Arguments:
        adata: AnnData from :func:`load_cool_collection` after
            :func:`impute_cells` has been run. Reads
            ``adata.uns['hic']['imputed_dir']``.
        celltype_cools: ``{celltype: cool_path}`` — typically the output
            of :func:`pseudobulk_by_celltype`. Cools must be balanced
            and share binsize with the imputed npz.
        chromosomes: subset; default = ``adata.uns['hic']['chromosomes']``.
        resolution: bp resolution (for ``.mcool`` URIs).
        max_distance_bins: cap the upper-triangle distance to this many
            bins (e.g. ``20`` × 100 kb = 2 Mb). Local TAD-scale signal
            dominates celltype identity and trans / very-long-range
            contacts add noise.
        z_score: per-cell z-score across celltype dimensions. Default
            ``True``.

    Returns:
        ``DataFrame`` indexed by cell barcode, columns = celltype name,
        values = (z-scored) Pearson r.
    """
    info = adata.uns.get("hic", {})
    imputed_dir = Path(info["imputed_dir"])
    if chromosomes is None:
        chromosomes = list(info.get("impute_params", {}).get(
            "chromosomes", info.get("chromosomes", [])))

    import cooler
    celltypes = list(celltype_cools.keys())

    def _cell_vec_from_arr(P: np.ndarray, max_dist: int) -> np.ndarray:
        n = P.shape[0]
        ii, jj = np.triu_indices(n, k=1)
        keep = (jj - ii) <= max_dist
        return P[ii[keep], jj[keep]].astype(np.float32)

    # Build celltype reference vectors.
    ct_vecs = {}
    for ct in celltypes:
        clr = cooler.Cooler(_resolve_uri(celltype_cools[ct], resolution))
        parts = []
        for ch in chromosomes:
            try:
                M = clr.matrix(balance=True, sparse=False).fetch(ch)
            except Exception:
                continue
            v = _cell_vec_from_arr(M, max_distance_bins)
            v = np.where(np.isfinite(v), v, 0)
            parts.append(np.log1p(np.maximum(v, 0)))
        if parts:
            ct_vecs[ct] = np.concatenate(parts)
    celltypes = [ct for ct in celltypes if ct in ct_vecs]

    cell_ids = list(adata.obs_names)
    feats = np.zeros((len(cell_ids), len(celltypes)), dtype=np.float32)
    for ci, cid in enumerate(cell_ids):
        npz_path = imputed_dir / f"{cid}.npz"
        if not npz_path.exists():
            continue
        z = np.load(npz_path)
        parts = []
        for ch in chromosomes:
            if ch in z.files:
                P = z[ch]
                parts.append(_cell_vec_from_arr(P, max_distance_bins))
        if not parts:
            continue
        cell_vec = np.concatenate(parts)
        for j, ct in enumerate(celltypes):
            ctv = ct_vecs[ct]
            if len(ctv) != len(cell_vec):
                continue
            m = np.isfinite(ctv) & np.isfinite(cell_vec)
            if m.sum() < 100:
                continue
            a, b = ctv[m], cell_vec[m]
            if a.std() == 0 or b.std() == 0:
                continue
            feats[ci, j] = float(np.corrcoef(a, b)[0, 1])

    if z_score:
        mu = feats.mean(axis=1, keepdims=True)
        sd = feats.std(axis=1, keepdims=True)
        feats = (feats - mu) / (sd + 1e-9)

    return pd.DataFrame(feats, index=cell_ids, columns=celltypes)
