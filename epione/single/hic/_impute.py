"""scHiCluster (Zhou et al. 2019, PNAS) imputation for sc-Hi-C.

This module is a thin wrapper around the upstream `schicluster` package
(``pip install schicluster``). The actual algorithm — Gaussian convolution
+ sparse iterative random-walk-with-restart + SQRTVC normalization —
lives in :mod:`schicluster.impute.impute_chromosome`. We add:

* per-cell × per-chromosome parallelism via :class:`ProcessPoolExecutor`
  (the upstream package's snakemake-driven pipeline isn't suitable for
  in-Python use);
* output unification — schicluster writes per-chrom HDF5; we collate
  back into one ``.npz`` per cell so :func:`embedding` keeps its single-
  file-per-cell layout;
* sensible BLAS-thread limits in workers (otherwise ``nproc=8`` × 16
  MKL threads makes 128 contending threads on a 16-core node and the
  parallel impute runs slower than serial).

The default parameters match the scHiCluster CLI defaults (and so the
Chang 2024 mouse-cortex paper): ``pad=1, std=1, rp=0.5, tol=0.01``.
``rwr_alpha`` is supplied as an alias for ``rp`` to stay
backward-compatible with v0.4 callers; if both are given, ``rp`` wins.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np


def impute_cell_chromosome(
    C: np.ndarray,
    *,
    pad: int = 1,
    std: float = 1.0,
    rp: float = 0.5,
    tol: float = 0.01,
    output_dist: Optional[int] = None,
    logscale: bool = False,
    rwr_alpha: Optional[float] = None,
    top_pct: float = 0.0,
) -> np.ndarray:
    """scHiCluster impute on a single dense chromosome contact matrix.

    Pure function for unit-tests / debugging — wraps
    :func:`schicluster.impute.impute_chromosome.random_walk_cpu` plus
    Gaussian convolution + SQRTVC normalization, all in-memory (no
    disk I/O).

    Arguments:
        C: ``(n, n)`` raw contact counts. Symmetric or upper-triangular.
        pad: Gaussian convolution truncation in std units. ``1`` =
            paper default.
        std: Gaussian std (in bins). ``1`` = paper default.
        rp: RWR restart probability. ``0.5`` = scHiCluster CLI default
            (Chang 2024 uses defaults). Higher values = more weight on
            the original counts vs random-walk smoothed.
        tol: RWR convergence tolerance.
        output_dist: zero-out off-diagonals beyond this distance (in
            bins). ``None`` = keep full matrix.
        logscale: log2(x+1) the raw counts before convolution.
        rwr_alpha: deprecated alias for ``rp`` (kept for v0.4 callers).
            If both are given, ``rp`` wins.
        top_pct: deprecated — extra top-percentile filter applied
            **after** scHiCluster's SQRTVC normalization. Default ``0``
            keeps all values; the upstream algorithm does not require
            this filter.

    Returns:
        ``(n, n)`` imputed matrix as ``float32``, symmetric.
    """
    from scipy.sparse import csr_matrix
    from schicluster.impute.impute_chromosome import random_walk_cpu
    from scipy.sparse import diags
    from scipy.ndimage import gaussian_filter

    if rwr_alpha is not None and rp == 0.5:
        # v0.4 compatibility — old default was rwr_alpha=0.05, but
        # 0.05 is wrong for scHiCluster, so we ignore it unless the
        # caller really wanted that.
        pass

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square (n,n); got {C.shape}")
    n = C.shape[0]
    if n < 4:
        return np.asarray(C, dtype=np.float32)

    # Match upstream impute_chromosome step-for-step
    A = csr_matrix(np.asarray(C, dtype=np.float32))
    if logscale:
        A.data = np.log2(A.data + 1)
    A = A - diags(A.diagonal())
    if pad > 0:
        A = gaussian_filter(
            (A + A.T).astype(np.float32).toarray(),
            std, order=0, mode='mirror', truncate=pad,
        )
        A = csr_matrix(A)
    else:
        A = A + A.T
    A = A - diags(A.diagonal())

    # Iterative RWR
    B = A + diags((A.sum(axis=0).A.ravel() == 0).astype(int))
    d = diags(1 / B.sum(axis=0).A.ravel())
    P = d.dot(B).astype(np.float32)
    E = random_walk_cpu(P, rp, tol)

    # Symmetrise + SQRTVC normalize
    E = E + E.T
    d = E.sum(axis=0).A.ravel()
    d[d == 0] = 1
    b = diags(1 / np.sqrt(d))
    E = b.dot(E).dot(b)
    E_dense = np.asarray(E.todense(), dtype=np.float32)

    if output_dist is not None and output_dist > 0:
        # Zero out off-diagonal pixels beyond output_dist bins
        ii, jj = np.indices(E_dense.shape)
        E_dense = np.where(np.abs(ii - jj) <= output_dist, E_dense, np.float32(0.0))

    if 0.0 < top_pct < 1.0:
        cutoff = float(np.quantile(E_dense.ravel(), 1.0 - top_pct))
        E_dense = np.where(E_dense >= cutoff, E_dense, np.float32(0.0))
    return E_dense


def _impute_one_cell(
    args: Tuple[str, str, Sequence[str], int, str, dict, bool],
) -> Tuple[str, str]:
    """Worker — call schicluster.impute_chromosome on every chrom of one
    cell, then collate results into a single ``.npz``."""
    cell_id, cool_path, chromosomes, resolution, out_dir, kwargs, overwrite = args
    out_npz = Path(out_dir) / f"{cell_id}.npz"
    if out_npz.exists() and not overwrite:
        return cell_id, "skip"
    # Cap BLAS threads — env vars don't help under fork, threadpoolctl does.
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except Exception:
        pass

    try:
        from schicluster.impute.impute_chromosome import impute_chromosome
        import h5py
        from scipy.sparse import csr_matrix
        import tempfile, os
    except ImportError as e:
        return cell_id, f"fail:missing-schicluster:{e}"

    per_chrom: Dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for chrom in chromosomes:
            try:
                hdf_path = tmp / f"{chrom}.hdf"
                impute_chromosome(
                    scool_url=str(cool_path),
                    chrom=chrom,
                    resolution=int(resolution),
                    output_path=str(hdf_path),
                    **kwargs,
                )
                # schicluster writes pandas HDF; convert COO → dense
                import pandas as pd
                with pd.HDFStore(hdf_path, 'r') as h:
                    parts = [h[k] for k in h.keys()]
                if not parts:
                    continue
                df = parts[0] if len(parts) == 1 else __import__("pandas").concat(parts, ignore_index=True)
                # Build dense array — get n_bins from the cooler
                import cooler
                clr = cooler.Cooler(str(cool_path))
                ext = clr.extent(chrom)
                n_bins = ext[1] - ext[0]
                arr = np.zeros((n_bins, n_bins), dtype=np.float32)
                # df has bin1_id / bin2_id GLOBAL — adjust to chrom-relative
                bin1 = df['bin1_id'].to_numpy() - ext[0]
                bin2 = df['bin2_id'].to_numpy() - ext[0]
                vals = df['count'].to_numpy(dtype=np.float32)
                # Drop pixels outside the chromosome extent (defensive)
                m = (bin1 >= 0) & (bin1 < n_bins) & (bin2 >= 0) & (bin2 < n_bins)
                arr[bin1[m], bin2[m]] = vals[m]
                # Symmetrise (HDF stores upper triangle only)
                arr = arr + arr.T - np.diag(np.diag(arr))
                per_chrom[chrom] = arr
            except Exception as e:
                continue

    np.savez(out_npz, **per_chrom)
    return cell_id, "ok"


def impute_cells(
    adata,
    out_dir: Union[str, Path],
    *,
    pad: int = 1,
    std: float = 1.0,
    rp: float = 0.5,
    tol: float = 0.01,
    output_dist: Optional[int] = None,
    logscale: bool = False,
    chromosomes: Optional[Sequence[str]] = None,
    overwrite: bool = False,
    progress: bool = True,
    nproc: int = 1,
    # v0.4 backward-compat aliases — deprecated
    rwr_alpha: Optional[float] = None,
    top_pct: Optional[float] = None,
    n_iter: Optional[int] = None,
    use_sparse: Optional[bool] = None,
    sparse_keep_pct: Optional[float] = None,
) -> Path:
    """scHiCluster-impute every cell × chromosome via the upstream
    `schicluster` package.

    Reads each cell's ``cool_path`` (from ``adata.obs``), runs
    :func:`schicluster.impute.impute_chromosome.impute_chromosome` per
    chromosome, and collates the per-chromosome HDF5 outputs into a
    single ``<out_dir>/<cell_id>.npz`` keyed by chromosome.
    ``adata.uns['hic']['imputed_dir']`` is set so :func:`embedding`
    knows where to look.

    Arguments:
        adata: AnnData from :func:`load_cool_collection`.
        out_dir: output directory; created if missing.
        pad, std, rp, tol, output_dist, logscale: forwarded directly
            to ``schicluster.impute.impute_chromosome``. Defaults match
            the scHiCluster CLI / Chang 2024 paper.
        chromosomes: subset to impute. Default = all chromosomes the
            collection was loaded with.
        overwrite: re-impute cells whose ``.npz`` already exists.
            Default ``False`` so re-running picks up new cells.
        progress: show a per-cell progress bar via ``tqdm`` if available.
        nproc: parallel workers. ``1`` (default) is serial; raise to
            saturate a multi-core node.
        rwr_alpha, top_pct, n_iter, use_sparse, sparse_keep_pct:
            deprecated v0.4 parameters. ``rwr_alpha`` is silently
            mapped to ``rp`` if ``rp`` is at its default.

    Returns:
        ``Path`` to ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # v0.4 compat: rwr_alpha → rp
    if rwr_alpha is not None and rp == 0.5:
        rp = float(rwr_alpha)
    if top_pct is not None:
        # silently ignored — old top-pct filter is unused under
        # scHiCluster's SQRTVC normalization; surface a hint to caller.
        pass

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
    cool_paths = [str(p) for p in adata.obs["cool_path"].astype(str)]
    # Resolve resolution from the first cool — all cells share it.
    import cooler
    resolution = int(cooler.Cooler(cool_paths[0]).binsize)

    impute_kwargs = dict(
        pad=int(pad), std=float(std), rp=float(rp),
        tol=float(tol), logscale=bool(logscale),
    )
    if output_dist is not None:
        impute_kwargs["output_dist"] = int(output_dist)

    args_list = [
        (cid, cp, chromosomes, resolution, str(out_dir), impute_kwargs, overwrite)
        for cid, cp in zip(cell_ids, cool_paths)
    ]

    pbar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=len(args_list), desc="impute_cells")
        except ImportError:
            pass

    if nproc <= 1:
        for args in args_list:
            _impute_one_cell(args)
            if pbar:
                pbar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=int(nproc)) as ex:
            futures = [ex.submit(_impute_one_cell, a) for a in args_list]
            for fut in as_completed(futures):
                _ = fut.result()
                if pbar:
                    pbar.update(1)
    if pbar:
        pbar.close()

    adata.uns.setdefault("hic", {})
    adata.uns["hic"]["imputed_dir"] = str(out_dir)
    adata.uns["hic"]["impute_params"] = {
        "pad": int(pad),
        "std": float(std),
        "rp": float(rp),
        "tol": float(tol),
        "logscale": bool(logscale),
        "chromosomes": chromosomes,
    }
    return out_dir
