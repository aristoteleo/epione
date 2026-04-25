"""scHiCluster (Zhou et al. 2019, PNAS) imputation for sc-Hi-C.

Each cell × chromosome contact matrix is densified through three steps:

1. **Linear convolution** with a (2``pad``+1)² sum-pooling kernel — pulls
   neighbouring contacts into each bin pair, smoothing single-contact
   noise without crossing chromosome boundaries.
2. **Random walk with restart** — a closed-form ``α(I - (1-α)W)⁻¹`` of
   the row-normalised matrix densifies sparse contacts via transitive
   neighbours. Equivalent to summing infinite RW with restart probability
   ``α`` (default 0.05).
3. **Top-k filter** — keep the largest ``top_pct`` quantile of values,
   zero the rest. Drops dense-everywhere noise so flattened features
   stay informative.

Per-cell imputed matrices are written to ``<out_dir>/<cell_id>.npz``
(scipy ``.npz`` archive, one ``arr_<chrom>`` per chromosome). The
filename layout mirrors the input ``.cool`` collection — one file per
cell — so :func:`embedding` can stream them back.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np


def _row_normalize(C: np.ndarray) -> np.ndarray:
    rowsum = C.sum(axis=1, keepdims=True)
    rowsum = np.where(rowsum > 0, rowsum, 1.0)
    return C / rowsum


def impute_cell_chromosome(
    C: np.ndarray,
    *,
    pad: int = 1,
    rwr_alpha: float = 0.05,
    top_pct: float = 0.05,
    log_transform: bool = True,
) -> np.ndarray:
    """scHiCluster imputation of a single chromosome contact matrix.

    Pure function — call this directly to verify the algorithm on a known
    matrix, or via :func:`impute_cells` for a whole cell collection.

    Arguments:
        C: ``(n, n)`` raw contact counts. Symmetric or upper-triangular —
            we symmetrise internally.
        pad: convolution radius. ``1`` (default) gives a 3×3 sum kernel.
            Set ``0`` to skip convolution.
        rwr_alpha: restart probability. ``0.05`` (paper default) gives
            ~20-step diffusion; smaller = longer-range smoothing.
        top_pct: keep the top ``top_pct`` quantile of values; zero the
            rest. ``0.05`` ≈ 5 % densest (paper default).
        log_transform: apply ``log(1 + p × scale)`` before the top-k cut
            so the percentile is taken on a more uniform distribution.

    Returns:
        ``(n, n)`` imputed matrix, symmetric, mostly-sparse after the
        top-k cut.
    """
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square (n,n); got {C.shape}")
    n = C.shape[0]
    if n == 0:
        return C.astype(np.float64)

    A = np.asarray(C, dtype=np.float64)
    # Symmetrise without double-counting a true-symmetric input.
    if not np.allclose(A, A.T):
        A = A + A.T - np.diag(np.diag(A))

    if pad > 0 and n > 1:
        # 2D sum-pool with a (2*pad+1) square kernel via cumulative-sum
        # tricks. scipy.signal.convolve2d works but is ~10x slower for
        # n=1000; the cumsum trick is also numpy-only.
        cum = np.zeros((n + 1, n + 1))
        cum[1:, 1:] = np.cumsum(np.cumsum(A, axis=0), axis=1)
        i, j = np.indices((n, n))
        i0 = np.clip(i - pad, 0, n)
        j0 = np.clip(j - pad, 0, n)
        i1 = np.clip(i + pad + 1, 0, n)
        j1 = np.clip(j + pad + 1, 0, n)
        A = cum[i1, j1] - cum[i0, j1] - cum[i1, j0] + cum[i0, j0]

    # Random walk with restart: P = α (I - (1-α) W)^-1
    W = _row_normalize(A)
    I = np.eye(n)
    try:
        P = rwr_alpha * np.linalg.solve(I - (1.0 - rwr_alpha) * W, I)
    except np.linalg.LinAlgError:
        # Numerical singular: fall back to lstsq.
        P, *_ = np.linalg.lstsq(I - (1.0 - rwr_alpha) * W, rwr_alpha * I,
                                rcond=None)
    # Symmetrise (RWR is not symmetric for non-symmetric W).
    P = 0.5 * (P + P.T)

    if log_transform:
        # Scale into a healthy range before log so percentile is stable.
        scale = n / max(A.sum(), 1.0)
        P = np.log1p(P * scale)

    if 0.0 < top_pct < 1.0:
        flat = P.ravel()
        if flat.size > 0:
            cutoff = np.quantile(flat, 1.0 - top_pct)
            P = np.where(P >= cutoff, P, 0.0)
    return P


def _cell_chromosome_matrix(cool_path: Path, chrom: str) -> np.ndarray:
    """Fetch a per-cell chrom contact matrix as a dense float64 array."""
    import cooler
    clr = cooler.Cooler(str(cool_path))
    return np.asarray(
        clr.matrix(balance=False, sparse=False).fetch(chrom),
        dtype=np.float64,
    )


def impute_cells(
    adata,
    out_dir: Union[str, Path],
    *,
    pad: int = 1,
    rwr_alpha: float = 0.05,
    top_pct: float = 0.05,
    chromosomes: Optional[Sequence[str]] = None,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """scHiCluster-impute every cell × chromosome in a collection.

    Reads each cell's ``cool_path`` (from ``adata.obs``), runs
    :func:`impute_cell_chromosome` per chromosome, and writes the result
    to ``<out_dir>/<cell_id>.npz``. ``adata.uns['hic']['imputed_dir']``
    is set so :func:`embedding` knows where to look.

    Arguments:
        adata: AnnData from :func:`load_cool_collection`.
        out_dir: output directory; created if missing.
        pad, rwr_alpha, top_pct: forwarded to
            :func:`impute_cell_chromosome`.
        chromosomes: subset to impute. Default = all chromosomes the
            collection was loaded with.
        overwrite: re-impute cells whose ``.npz`` already exists.
            Default ``False`` so re-running picks up new cells without
            redoing finished ones.
        progress: show a per-cell progress bar via ``tqdm`` if available.

    Returns:
        ``Path`` to ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = adata.uns.get("hic", {})
    chromosomes = list(chromosomes) if chromosomes is not None else list(
        info.get("chromosomes", [])
    )
    if not chromosomes:
        raise ValueError(
            "no chromosomes to impute — adata.uns['hic']['chromosomes'] "
            "is empty; pass chromosomes=... explicitly."
        )

    cell_ids = list(adata.obs_names)
    cool_paths = [Path(p) for p in adata.obs["cool_path"].astype(str)]

    iterator = zip(cell_ids, cool_paths)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(list(iterator), desc="impute_cells")
        except ImportError:
            pass

    for cell_id, cool_path in iterator:
        out_npz = out_dir / f"{cell_id}.npz"
        if out_npz.exists() and not overwrite:
            continue
        per_chrom: Dict[str, np.ndarray] = {}
        for chrom in chromosomes:
            C = _cell_chromosome_matrix(cool_path, chrom)
            P = impute_cell_chromosome(
                C, pad=pad, rwr_alpha=rwr_alpha, top_pct=top_pct,
            )
            per_chrom[chrom] = P.astype(np.float32)
        # Use np.savez (not _compressed) — random access on the embed
        # side is fast enough that the disk-space win isn't worth the
        # 5-10x slower write.
        np.savez(out_npz, **per_chrom)

    adata.uns.setdefault("hic", {})
    adata.uns["hic"]["imputed_dir"] = str(out_dir)
    adata.uns["hic"]["impute_params"] = {
        "pad": int(pad),
        "rwr_alpha": float(rwr_alpha),
        "top_pct": float(top_pct),
        "chromosomes": chromosomes,
    }
    return out_dir
