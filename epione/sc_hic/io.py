"""Single-cell Hi-C input — index a directory of per-cell cools (or one
``.scool`` bundle) into an AnnData skeleton that downstream
:func:`impute_cells` / :func:`embedding` fill in.

We deliberately do NOT load contacts into memory here; per-cell ``.cool``
files stay on disk and are streamed during imputation. ``adata.X`` is
``None`` until :func:`embedding` flattens the imputed matrices.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


def load_cool_collection(
    cool_paths: Sequence[Union[str, Path]],
    *,
    cell_ids: Optional[Sequence[str]] = None,
    obs: Optional[pd.DataFrame] = None,
    chromosomes: Optional[Sequence[str]] = None,
):
    """Index a list of per-cell ``.cool`` files into a cell-level AnnData.

    Arguments:
        cool_paths: list of ``.cool`` paths, one per cell.
        cell_ids: parallel list of cell barcodes / IDs. Default: file
            stem (``Path(p).stem``), which works when filenames already
            encode the barcode.
        obs: optional cell metadata. Must align with ``cell_ids`` (same
            length, in the same order). Joined onto ``adata.obs`` so
            downstream plots can colour by cell-cycle / batch / cluster.
        chromosomes: subset of chromosomes to keep for downstream
            imputation/embedding. ``None`` (default) takes the
            intersection of chromosomes across all cells (so a cell with
            a missing chrom doesn't crash later steps).

    Returns:
        ``AnnData`` with shape ``(n_cells, 0)`` initially. ``obs`` carries
        ``cool_path`` (str) and any user-supplied metadata. ``uns['hic']``
        records the chosen ``chromosomes`` and the (assumed-uniform)
        ``resolution`` in bp.
    """
    import anndata as ad
    import cooler

    cool_paths = [Path(p) for p in cool_paths]
    if cell_ids is None:
        cell_ids = [p.stem for p in cool_paths]
    cell_ids = list(map(str, cell_ids))
    if len(cell_ids) != len(cool_paths):
        raise ValueError(
            f"cell_ids has {len(cell_ids)} entries but cool_paths has "
            f"{len(cool_paths)}; must match 1:1"
        )
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell_ids must be unique")

    # Probe the first cool for resolution + master chrom list, then
    # cross-check the rest. We tolerate cells missing some chroms (will
    # be intersected) but the binsize MUST match across cells.
    head = cooler.Cooler(str(cool_paths[0]))
    resolution = int(head.binsize)
    common: set = set(head.chromnames)
    for p in cool_paths[1:]:
        c = cooler.Cooler(str(p))
        if int(c.binsize) != resolution:
            raise ValueError(
                f"{p} has binsize {c.binsize} bp, expected {resolution} "
                "bp — all per-cell cools must share a resolution"
            )
        common &= set(c.chromnames)

    if chromosomes is None:
        # Preserve the order from the first cool so downstream feature
        # vectors are deterministic.
        chromosomes = [c for c in head.chromnames if c in common]
    else:
        chromosomes = list(chromosomes)
        missing = set(chromosomes) - common
        if missing:
            raise ValueError(
                f"requested chromosomes not present in every cell: {sorted(missing)}"
            )

    obs_df = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))
    obs_df["cool_path"] = [str(p) for p in cool_paths]
    if obs is not None:
        if len(obs) != len(cell_ids):
            raise ValueError(
                f"obs has {len(obs)} rows but {len(cell_ids)} cells"
            )
        obs_in = obs.copy()
        obs_in.index = obs_df.index
        for col in obs_in.columns:
            obs_df[col] = obs_in[col].values

    adata = ad.AnnData(X=np.zeros((len(cell_ids), 0), dtype=np.float32),
                       obs=obs_df)
    adata.uns["hic"] = {
        "chromosomes": list(chromosomes),
        "resolution": resolution,
        "n_chrom_bins": {
            ch: int(np.ceil(head.chromsizes[ch] / resolution))
            for ch in chromosomes
        },
    }
    return adata


def load_scool_cells(
    scool_path: Union[str, Path],
    *,
    cell_names: Optional[Sequence[str]] = None,
    obs: Optional[pd.DataFrame] = None,
    chromosomes: Optional[Sequence[str]] = None,
):
    """Index cells inside a multi-cell ``.scool`` HDF5 bundle.

    Two on-disk layouts are supported:

    * **Modern scool** (Bioinformatics 2021): cells live under
      ``/cells/<name>``. This is what ``cooler.fileops.list_scool_cells``
      enumerates; URIs become ``<path>::/cells/<name>``.
    * **Legacy multi-cool** (HiCMatrix / HiCExplorer pre-2021,
      including the Zenodo Nagano 2017 bundle): each cell is a
      *top-level* HDF5 group with the standard cooler layout. URIs
      become ``<path>::/<name>``.

    The wrapper auto-detects format by trying the modern layout first
    and falling back to legacy. Either way it forwards a list of
    cooler URIs to :func:`load_cool_collection`, so downstream
    :func:`impute_cells` / :func:`embedding` see one cell per row.

    Arguments:
        scool_path: path to the ``.scool`` HDF5 file.
        cell_names: subset of cells to load. ``None`` (default) loads
            every cell in the bundle.
        obs: cell metadata. Joined positionally with ``cell_names``.
        chromosomes: subset of chromosomes; passed through to
            :func:`load_cool_collection`. Use to drop ``*_random`` /
            ``chrUn`` etc. that are present in some Hi-C cools.

    Returns:
        AnnData of shape ``(n_cells, 0)`` with ``cool_path`` column
        holding the per-cell URI.
    """
    import h5py

    scool_path = str(scool_path)
    cell_uri_prefix = "::/cells/"
    try:
        import cooler.fileops
        available = cooler.fileops.list_scool_cells(scool_path)
        # ``list_scool_cells`` returns ``/cells/<name>`` style paths;
        # strip the ``/cells/`` prefix to expose just the cell name.
        available = [
            c.split("/cells/", 1)[-1] if "/cells/" in c else c
            for c in available
        ]
    except OSError:
        # Legacy multi-cool: each cell is a top-level group with the
        # standard ``bins / chroms / indexes / pixels`` layout.
        with h5py.File(scool_path, "r") as h:
            available = [
                k for k in h.keys()
                if isinstance(h[k], h5py.Group)
                and "bins" in h[k] and "pixels" in h[k]
            ]
        cell_uri_prefix = "::/"

    if cell_names is None:
        cell_names = available
    else:
        cell_names = list(map(str, cell_names))
        missing = [c for c in cell_names if c not in available]
        if missing:
            raise KeyError(
                f"{len(missing)} requested cell(s) not in scool — first "
                f"few: {missing[:5]}"
            )

    cool_paths = [f"{scool_path}{cell_uri_prefix}{c}" for c in cell_names]
    return load_cool_collection(
        cool_paths, cell_ids=cell_names, obs=obs,
        chromosomes=chromosomes,
    )
