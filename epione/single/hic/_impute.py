"""sc-Hi-C imputation: scHiCluster (Zhou et al. 2019, PNAS; default) or Higashi.

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

import os
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
    args: Tuple[str, dict, Sequence[str], int, str, dict, bool],
) -> Tuple[str, str]:
    """Worker — call schicluster.impute_chromosome on every chrom of one
    cell, then collate results into a single ``.npz``.

    ``source`` is ``{"scool_url": <cool>}`` or the contact-table arguments of
    ``impute_chromosome`` (``contact_path``, ``chrom_size_path``, ``chrom1``,
    ``pos1``, ``chrom2``, ``pos2``)."""
    cell_id, source, chromosomes, resolution, out_dir, kwargs, overwrite = args
    cool_path = source.get("scool_url")
    kwargs = dict(kwargs)
    compress = bool(kwargs.pop("_compress", False))
    accelerate = bool(kwargs.pop("_accelerate", True))
    sparse = bool(kwargs.pop("_sparse", False))
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
        if accelerate:
            from ._impute_fast import impute_chromosome
        else:
            from schicluster.impute.impute_chromosome import impute_chromosome
        import h5py
        from scipy.sparse import csr_matrix
        import tempfile, os
    except ImportError as e:
        return cell_id, f"fail:missing-schicluster:{e}"

    per_chrom: Dict[str, object] = {}
    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for chrom in chromosomes:
            try:
                hdf_path = tmp / f"{chrom}.hdf"
                impute_chromosome(
                    **source,
                    chrom=chrom,
                    resolution=int(resolution),
                    output_path=str(hdf_path),
                    **kwargs,
                )
                sparse_path = Path(str(hdf_path) + ".npz")
                if sparse_path.exists():
                    # scHiCluster >= 1.3: scipy sparse upper triangle, chromosome-relative
                    from scipy.sparse import load_npz
                    E = load_npz(sparse_path).tocoo()
                elif cool_path is None:
                    raise FileNotFoundError(f"no imputation output at {sparse_path}")
                else:
                    # older scHiCluster: pandas HDF with global bin ids
                    import pandas as pd
                    import cooler
                    with pd.HDFStore(hdf_path, 'r') as h:
                        parts = [h[k] for k in h.keys()]
                    if not parts:
                        raise ValueError("empty imputation output")
                    df = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
                    ext = cooler.Cooler(str(cool_path)).extent(chrom)
                    n_bins = ext[1] - ext[0]
                    arr = np.zeros((n_bins, n_bins), dtype=np.float32)
                    bin1 = df['bin1_id'].to_numpy() - ext[0]
                    bin2 = df['bin2_id'].to_numpy() - ext[0]
                    vals = df['count'].to_numpy(dtype=np.float32)
                    m = (bin1 >= 0) & (bin1 < n_bins) & (bin2 >= 0) & (bin2 < n_bins)
                    from scipy.sparse import coo_matrix
                    E = coo_matrix((vals[m], (bin1[m], bin2[m])), shape=(n_bins, n_bins))
                if sparse:
                    # keep scHiCluster's upper triangle
                    per_chrom[chrom] = E.astype(np.float32).tocsr()
                else:
                    # dense symmetric matrix (both formats store the upper triangle)
                    arr = np.zeros(E.shape, dtype=np.float32)
                    arr[E.row, E.col] = E.data
                    arr = arr + arr.T - np.diag(np.diag(arr))
                    per_chrom[chrom] = arr
            except Exception as e:
                failed.append(f"{chrom}: {type(e).__name__}: {e}")
    from ._imputed_io import save_imputed
    save_imputed(out_npz, per_chrom, sparse=sparse, compress=compress)
    if failed:
        return cell_id, "fail:" + "; ".join(failed[:3])
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
    window_size: Optional[int] = None,
    step_size: Optional[int] = None,
    logscale: bool = False,
    chromosomes: Optional[Sequence[str]] = None,
    overwrite: bool = False,
    progress: bool = True,
    nproc: int = 1,
    compress: bool = False,
    resolution: Optional[int] = None,
    accelerate: bool = True,
    sparse: bool = False,
    # v0.4 backward-compat aliases — deprecated
    rwr_alpha: Optional[float] = None,
    top_pct: Optional[float] = None,
    n_iter: Optional[int] = None,
    use_sparse: Optional[bool] = None,
    sparse_keep_pct: Optional[float] = None,
    # method selection (default keeps the scHiCluster behaviour)
    method: str = "schicluster",
    random_state: int = 0,
    device=None,
    higashi_params: Optional[dict] = None,
    higashi_neighbors: bool = True,
) -> Path:
    """Impute every cell × chromosome: scHiCluster (default) or Higashi.

    ``method="schicluster"`` (default) runs the upstream `schicluster`
    package as described below. ``method="higashi"`` trains Higashi
    (Zhang, Zhou & Ma 2022; PyTorch port of ma-compbio/Higashi, see
    :mod:`epione.single.hic._higashi`) on all cells jointly and writes its
    imputation in the same ``<out_dir>/<cell_id>.npz`` layout; the
    scHiCluster-specific arguments are then ignored.

    Reads each cell's ``cool_path`` (binned cells) or ``contact_path`` (raw
    contact tables, binned at ``resolution``) from ``adata.obs``, runs
    :func:`schicluster.impute.impute_chromosome.impute_chromosome` per
    chromosome, and collates the outputs into a single
    ``<out_dir>/<cell_id>.npz`` keyed by chromosome.
    ``adata.uns['hic']['imputed_dir']`` is set so :func:`embedding`
    knows where to look.

    Arguments:
        adata: AnnData from :func:`load_cool_collection` /
            :func:`load_scool_cells`, or from :func:`load_contact_tables`.
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
        compress: write each cell's ``.npz`` compressed. Recommended at
            100 kb and finer with ``output_dist``: the stored matrices are
            dense, and a band of a few Mb is mostly zeros.
        window_size, step_size: scHiCluster's RWR window and step in bp
            (``--window_size``, ``--step_size``). ``None`` keeps upstream's
            defaults (5 Gb window, 10 Mb step): the window covers any
            chromosome, so RWR runs on whole chromosomes.
        resolution: bin size in bp. Cells loaded from raw contact tables
            (:func:`load_contact_tables`) are binned at this size by
            scHiCluster itself (``impute_chromosome(contact_path=...)``, as
            ``hicluster impute-cell --resolution``); default: the
            ``resolution`` given to ``load_contact_tables``. Contacts are
            counted as they are, with no de-duplication or minimum-distance
            filter, as upstream. Cells from cools are already binned:
            leave it ``None`` (the cools' bin size is used); any other value
            raises ``ValueError``.
        accelerate: use epione's faster implementation of scHiCluster's
            imputation (numba Gaussian, direct sparse kernels, one window
            accumulator): per cell about 3x faster at 10 kb (30 Mb windows,
            with 50-85 % less memory), 2.2x at 25 kb and 1.3-1.6x at 100 kb
            (whole-chromosome windows). Equal to scHiCluster up to float32
            rounding. ``False`` runs scHiCluster's own code.
        sparse: store scHiCluster's sparse upper triangle (within
            ``output_dist``) instead of dense symmetric matrices. Memory per
            worker and file size then scale with the band, not with
            ``n_bins^2``; use it at 25 kb and finer (at 10 kb chr1 is 2.5 GB
            dense). :func:`embedding` reads the band directly; other readers
            rebuild one dense chromosome at a time.
        rwr_alpha, top_pct, n_iter, use_sparse, sparse_keep_pct:
            deprecated v0.4 parameters. ``rwr_alpha`` is silently
            mapped to ``rp`` if ``rp`` is at its default.
        method: ``"schicluster"`` (default) or ``"higashi"``.
        random_state: Higashi seed (preprocessing SVD, train/test split,
            sampling, initialisation). GPU training is not bit-reproducible.
        device: torch device for Higashi (default: CUDA if available).
        higashi_params: overrides of Higashi's upstream defaults —
            ``dimensions`` (64), ``loss_mode`` (``"zinb"``),
            ``neighbor_num`` (5), ``local_transfer_range`` (1),
            ``minimum_distance`` (one bin, bp), ``maximum_distance``
            (-1 = none), ``embedding_epoch`` (60), ``no_nbr_epoch`` (45),
            ``with_nbr_epoch`` (30), ``optimizer`` (``"sequential"``; or
            ``"torch20_foreach"``). An epoch is 1,000 mini-batches.
        higashi_neighbors: write Higashi's imputation that borrows from the
            ``neighbor_num`` nearest cells (stage 3, upstream's final
            output; default) instead of the cell-only one (stage 2).

    Returns:
        ``Path`` to ``out_dir``. With ``method="higashi"`` also sets
        ``adata.obsm['X_higashi']`` (the stage-1 cell embedding) and saves
        the trained model as ``<out_dir>/higashi_model.pt``, which
        :func:`embedding` (``method="higashi"``) reuses.
    """
    if method == "higashi":
        from ._higashi_api import higashi_impute
        return higashi_impute(adata, out_dir, chromosomes=chromosomes, random_state=random_state,
                              device=device, higashi_params=higashi_params, neighbors=higashi_neighbors,
                              overwrite=overwrite, progress=progress)
    if method != "schicluster":
        raise ValueError(f"method must be 'schicluster' or 'higashi', got {method!r}")
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
    sizes_file = None
    if "cool_path" in adata.obs:
        # binned cells: the resolution is the cools' bin size
        import cooler
        cool_paths = [str(p) for p in adata.obs["cool_path"].astype(str)]
        binsize = int(cooler.Cooler(cool_paths[0]).binsize)
        if resolution is not None and int(resolution) != binsize:
            raise ValueError(
                f"cells are binned at {binsize:,} bp; resolution={int(resolution):,} needs cells "
                "binned at that size (contacts_to_scool) or raw contacts (load_contact_tables)")
        resolution = binsize
        sources = [{"scool_url": cp} for cp in cool_paths]
    elif "contact_path" in adata.obs:
        # raw contact tables (load_contact_tables): scHiCluster bins them at `resolution`
        if "chrom_sizes" not in info:
            raise ValueError("adata.uns['hic']['chrom_sizes'] missing; load cells with load_contact_tables")
        resolution = int(resolution if resolution is not None else info["resolution"])
        import tempfile
        fd, sizes_file = tempfile.mkstemp(suffix=".chrom.sizes", dir=out_dir)
        with os.fdopen(fd, "w") as fh:
            for c, n in info["chrom_sizes"].items():
                fh.write(f"{c}\t{int(n)}\n")
        cols = {k: int(v) for k, v in info.get("contact_columns", {}).items()}
        sources = [{"contact_path": cp, "chrom_size_path": sizes_file, **cols}
                   for cp in adata.obs["contact_path"].astype(str)]
    else:
        raise ValueError("adata.obs needs 'cool_path' (load_cool_collection / load_scool_cells) "
                         "or 'contact_path' (load_contact_tables)")

    impute_kwargs = dict(
        pad=int(pad), std=float(std), rp=float(rp),
        tol=float(tol), logscale=bool(logscale),
    )
    for name, value in (("output_dist", output_dist), ("window_size", window_size),
                        ("step_size", step_size)):
        if value is not None:
            impute_kwargs[name] = int(value)
    impute_kwargs["_compress"] = bool(compress)
    impute_kwargs["_accelerate"] = bool(accelerate)
    impute_kwargs["_sparse"] = bool(sparse)

    args_list = [
        (cid, src, chromosomes, resolution, str(out_dir), impute_kwargs, overwrite)
        for cid, src in zip(cell_ids, sources)
    ]

    pbar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=len(args_list), desc="impute_cells")
        except ImportError:
            pass

    statuses = []
    if nproc <= 1:
        for args in args_list:
            statuses.append(_impute_one_cell(args))
            if pbar:
                pbar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=int(nproc)) as ex:
            futures = [ex.submit(_impute_one_cell, a) for a in args_list]
            for fut in as_completed(futures):
                statuses.append(fut.result())
                if pbar:
                    pbar.update(1)
    if pbar:
        pbar.close()
    if sizes_file is not None:
        os.unlink(sizes_file)
    failed = [(c, st) for c, st in statuses if st.startswith("fail")]
    if failed:
        import warnings
        warnings.warn(f"impute_cells: {len(failed)} of {len(statuses)} cells had failures; "
                      f"first: {failed[0][0]} -> {failed[0][1]}", RuntimeWarning)

    adata.uns.setdefault("hic", {})
    adata.uns["hic"]["imputed_dir"] = str(out_dir)
    if "contact_path" in adata.obs and "cool_path" not in adata.obs:
        # record the resolution the raw contacts were imputed at (embedding reads it)
        sizes = info["chrom_sizes"]
        adata.uns["hic"]["resolution"] = int(resolution)
        adata.uns["hic"]["n_chrom_bins"] = {c: int(sizes[c] // resolution + 1) for c in chromosomes}
    adata.uns["hic"]["impute_params"] = {
        "pad": int(pad),
        "std": float(std),
        "rp": float(rp),
        "tol": float(tol),
        "logscale": bool(logscale),
        "chromosomes": chromosomes,
    }
    return out_dir
