"""Reader for 10x Genomics Multiome (cellranger-arc) output."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, os.PathLike]

_H5_NAMES = ("filtered_feature_bc_matrix.h5",)
_MTX_NAMES = ("filtered_feature_bc_matrix",)
_FRAG_NAMES = ("atac_fragments.tsv.gz", "fragments.tsv.gz")
_METRIC_NAMES = ("per_barcode_metrics.csv",)
_PEAK_RE = re.compile(r"^(?P<chrom>[^:]+)[:\-](?P<start>\d+)-(?P<end>\d+)$")


def _find(root: Path, names, suffix_ok: bool = True) -> Optional[Path]:
    for name in names:
        p = root / name
        if p.exists():
            return p
    if suffix_ok:
        # 10x public datasets prefix every file with the sample name.
        for name in names:
            hits = sorted(root.glob(f"*_{name}"))
            if hits:
                return hits[0]
    return None


def _peak_coords(names) -> pd.DataFrame:
    rows = []
    for n in names:
        m = _PEAK_RE.match(str(n))
        rows.append((m["chrom"], int(m["start"]), int(m["end"])) if m else (None, np.nan, np.nan))
    df = pd.DataFrame(rows, columns=["chrom", "chromStart", "chromEnd"], index=pd.Index(names))
    return df


def read_10x_multiome(
    path: PathLike,
    fragments: Optional[PathLike] = None,
    *,
    metrics: Optional[PathLike] = None,
    as_mudata: bool = False,
    backed: Optional[PathLike] = None,
):
    """Read paired RNA + ATAC from 10x Genomics Multiome (cellranger-arc) output.

    Accepts the cellranger-arc ``outs/`` directory, the
    ``filtered_feature_bc_matrix.h5`` file, or the
    ``filtered_feature_bc_matrix/`` folder (``matrix.mtx.gz``,
    ``features.tsv.gz``, ``barcodes.tsv.gz``). Files of 10x public datasets,
    which carry a sample-name prefix (``pbmc_..._filtered_feature_bc_matrix.h5``),
    are found too.

    Arguments:
        path: ``outs/`` directory, ``.h5`` file or matrix folder.
        fragments: ``atac_fragments.tsv.gz`` (tabix-indexed). Default: looked
            up next to ``path``. Recorded in ``atac.uns["files"]["fragments"]``,
            where the fragment-based tools (gene scores, footprints, PRINT)
            expect it; the file itself is not loaded.
        metrics: ``per_barcode_metrics.csv``. Default: looked up next to
            ``path``. Its columns are added to ``obs`` of both modalities.
        as_mudata: return a ``MuData`` with modalities ``"rna"`` and ``"atac"``
            instead of a ``(rna, atac)`` pair.
        backed: path of an ``.h5mu`` file. The data are written there and
            reopened out of core with :mod:`mudataoom` (implies ``as_mudata``).

    Returns:
        ``(rna, atac)`` AnnData objects with the same cells in the same order,
        raw counts in ``X``; or a ``MuData`` / ``MuDataOOM``. ATAC ``var`` has
        ``chrom``, ``chromStart`` and ``chromEnd`` parsed from the peak names.

    Examples:
        >>> rna, atac = epi.io.read_10x_multiome("sample/outs")
        >>> atac.uns["files"]["fragments"]
        'sample/outs/atac_fragments.tsv.gz'
    """
    import warnings

    import scanpy as sc

    def _read_h5(f):
        with warnings.catch_warnings():
            # 10x matrices repeat some gene symbols; they are made unique below.
            warnings.filterwarnings("ignore", message="Variable names are not unique")
            return sc.read_10x_h5(str(f), gex_only=False)

    path = Path(path)
    root = path if path.is_dir() else path.parent
    if path.is_file():
        adata = _read_h5(path)
    else:
        h5 = _find(path, _H5_NAMES)
        mtx = _find(path, _MTX_NAMES)
        if (path / "matrix.mtx.gz").exists() or (path / "matrix.mtx").exists():
            mtx, h5, root = path, None, path.parent
        if h5 is not None:
            adata = _read_h5(h5)
        elif mtx is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Variable names are not unique")
                adata = sc.read_10x_mtx(str(mtx), var_names="gene_symbols", gex_only=False, make_unique=False)
            root = mtx.parent
        else:
            raise FileNotFoundError(f"no filtered_feature_bc_matrix(.h5) under {path}")

    ftype = adata.var["feature_types"].astype(str)
    rna = adata[:, (ftype == "Gene Expression").to_numpy()].copy()
    atac = adata[:, (ftype == "Peaks").to_numpy()].copy()
    if rna.n_vars == 0 or atac.n_vars == 0:
        raise ValueError(f"expected Gene Expression and Peaks features, found {sorted(set(ftype))}")
    rna.var_names_make_unique()
    coords = _peak_coords(atac.var_names)
    for c in coords.columns:
        atac.var[c] = coords[c].to_numpy()

    if metrics is None:
        metrics = _find(root, _METRIC_NAMES)
    if metrics is not None:
        m = pd.read_csv(metrics, index_col=0)
        m = m.reindex(rna.obs_names)
        for a in (rna, atac):
            for c in m.columns:
                a.obs[c] = m[c].to_numpy()

    if fragments is None:
        fragments = _find(root, _FRAG_NAMES)
    if fragments is not None:
        atac.uns["files"] = {"fragments": str(fragments)}

    if backed is not None or as_mudata:
        import mudata as md

        mdata = md.MuData({"rna": rna, "atac": atac})
        if backed is None:
            return mdata
        import mudataoom as moom

        mdata.write(str(backed))
        return moom.read_h5mu(str(backed), backed="r")
    return rna, atac


__all__ = ["read_10x_multiome"]
