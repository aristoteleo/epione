"""Readers for spatial ATAC-seq and spatial ATAC-RNA co-profiling.

Covers the DBiT-seq family from the Rong Fan lab (spatial-ATAC-seq,
Deng et al. 2022 Nature; spatial ATAC-RNA-seq and CUT&Tag-RNA-seq,
Zhang et al. 2023 Nature), AtlasXomics DBiT releases, and any
platform that ships a fragments file plus a barcode -> position table.

Two position-table layouts are recognised:

* the 10x Space Ranger ``tissue_positions(_list).csv`` (with or without
  header): ``barcode, in_tissue, array_row, array_col,
  pxl_row_in_fullres, pxl_col_in_fullres``. DBiT data on GEO are usually
  released in this layout together with ``scalefactors_json.json`` and
  a tissue image (e.g. GSE205055, GSE308623);
* a bare DBiT barcode whitelist ``barcode, x, y`` (tab, comma or space
  separated, optional header, optional 4th ``in_tissue`` column), as in
  the ArchR-based workflows of the spatial-ATAC-seq papers.

The result follows the scanpy / squidpy convention: ``obsm["spatial"]``
holds ``(x, y) = (pxl_col, pxl_row)`` and the image and scale factors
go in ``uns["spatial"][library_id]``, so ``sc.pl.spatial``,
``squidpy.gr.spatial_neighbors`` and :func:`epione.pl.spatial` all
read it directly.
"""
from __future__ import annotations

import gzip
import json
import os
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, os.PathLike]

_POS_COLS = ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
_SUFFIX_RE = re.compile(r"-\d+$")


def _strip_suffix(bc) -> str:
    """Drop a 10x GEM-well suffix (``AAAC...-1`` -> ``AAAC...``)."""
    return _SUFFIX_RE.sub("", str(bc))


def _open_text(path: Path):
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt")
    return open(path, "r")


def read_spatial_positions(path: PathLike) -> pd.DataFrame:
    """Read a barcode -> spot position table.

    Arguments:
        path: Space Ranger ``tissue_positions_list.csv`` /
            ``tissue_positions.csv`` (optionally gzipped, header optional),
            ``tissue_positions.parquet``, or a DBiT whitelist with columns
            ``barcode, x, y`` (optionally a 4th ``in_tissue`` column).

    Returns:
        DataFrame indexed by barcode (GEM-well suffix removed) with columns
        ``in_tissue`` (bool), ``array_row``, ``array_col``,
        ``pxl_row_in_fullres`` and ``pxl_col_in_fullres``. For a bare
        whitelist the pixel columns equal the grid indices.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        df = df.set_index(df.columns[0])
    else:
        with _open_text(path) as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        if not lines:
            raise ValueError(f"{path} is empty")
        sep = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else None)
        rows = [ln.split(sep) for ln in lines]
        rows = [[c.strip().strip('"') for c in r] for r in rows]

        def _numeric(r):
            try:
                [float(c) for c in r[1:]]
                return True
            except ValueError:
                return False

        if not _numeric(rows[0]):
            rows = rows[1:]  # header
        ncol = len(rows[0])
        if any(len(r) != ncol for r in rows):
            raise ValueError(f"{path}: rows have different numbers of columns")
        bc = [r[0] for r in rows]
        vals = np.asarray([[float(c) for c in r[1:]] for r in rows])
        if ncol == 6:
            df = pd.DataFrame(vals, index=bc, columns=_POS_COLS)
        elif ncol in (3, 4):
            # DBiT whitelist: barcode, x(col), y(row)[, in_tissue]
            df = pd.DataFrame(index=bc)
            df["in_tissue"] = vals[:, 2] if ncol == 4 else 1
            df["array_row"] = vals[:, 1]
            df["array_col"] = vals[:, 0]
            df["pxl_row_in_fullres"] = vals[:, 1]
            df["pxl_col_in_fullres"] = vals[:, 0]
        else:
            raise ValueError(
                f"{path}: expected 6 columns (Space Ranger) or 3-4 (barcode, x, y[, in_tissue]); got {ncol}"
            )
    df = df.copy()
    df.index = pd.Index([_strip_suffix(b) for b in df.index], name="barcode")
    if df.index.duplicated().any():
        raise ValueError(f"{path}: duplicated barcodes after removing the '-N' suffix")
    df["in_tissue"] = df["in_tissue"].astype(int).astype(bool)
    for c in _POS_COLS[1:3]:
        df[c] = df[c].astype(int)
    for c in _POS_COLS[3:]:
        df[c] = df[c].astype(float)
    return df[_POS_COLS]


def _find_file(root: Path, stems) -> Optional[Path]:
    for stem in stems:
        for pat in (stem, f"{stem}.gz", f"*_{stem}", f"*_{stem}.gz"):
            hits = sorted(root.glob(pat))
            if hits:
                return hits[0]
    return None


def _read_image(path: Path) -> np.ndarray:
    from matplotlib.image import imread

    if path.suffix == ".gz":
        import io

        with gzip.open(path, "rb") as fh:
            return imread(io.BytesIO(fh.read()), format="png")
    return imread(str(path))


def read_spatial_images(spatial_dir: PathLike) -> dict:
    """Read ``scalefactors_json.json`` and the tissue images of a spatial folder.

    File names may carry a GEO sample prefix (``GSM..._tissue_hires_image.png``)
    and a ``.gz`` suffix.

    Returns:
        ``{"images": {"hires": ..., "lowres": ...}, "scalefactors": {...}}``
        in the layout of ``adata.uns["spatial"][library_id]``. Missing
        pieces are left out.
    """
    root = Path(spatial_dir)
    out: dict = {"images": {}, "scalefactors": {}}
    sf = _find_file(root, ["scalefactors_json.json"])
    if sf is not None:
        with _open_text(sf) as fh:
            out["scalefactors"] = json.load(fh)
    for key in ("hires", "lowres"):
        p = _find_file(root, [f"tissue_{key}_image.png"])
        if p is not None:
            out["images"][key] = _read_image(p)
    return out


def add_spatial(
    adata,
    positions: Union[PathLike, pd.DataFrame],
    *,
    spatial_dir: Optional[PathLike] = None,
    library_id: str = "spatial",
    tissue_only: bool = False,
    spot_pitch: Optional[float] = None,
):
    """Attach spot coordinates (and image) to an existing AnnData.

    Barcodes are matched after removing a GEM-well suffix, so fragments
    barcodes ``AAAC...-1`` match a position table without suffix and vice
    versa; ``obs_names`` are left unchanged.

    Arguments:
        adata: AnnData whose ``obs_names`` are spot barcodes.
        positions: path accepted by :func:`read_spatial_positions`, or its
            DataFrame output.
        spatial_dir: folder with ``scalefactors_json.json`` and
            ``tissue_{hires,lowres}_image.png``; stored in
            ``uns["spatial"][library_id]``.
        library_id: key under ``uns["spatial"]``.
        tissue_only: drop spots with ``in_tissue == 0``.
        spot_pitch: when there is no image, scale grid indices by this
            centre-to-centre distance (e.g. 50 for 50 um DBiT) so that
            ``obsm["spatial"]`` is in microns.

    Returns:
        The AnnData (a filtered copy when ``tissue_only`` drops spots or
        spots without a position).
    """
    pos = positions if isinstance(positions, pd.DataFrame) else read_spatial_positions(positions)
    key = pd.Index([_strip_suffix(b) for b in adata.obs_names])
    hit = key.isin(pos.index)
    if not hit.any():
        raise ValueError(
            "no obs_names match the position table; check that both use the same barcode orientation"
        )
    if not hit.all():
        adata = adata[hit].copy()
        key = key[hit]
    p = pos.loc[key]
    if tissue_only and not p["in_tissue"].all():
        m = p["in_tissue"].to_numpy()
        adata = adata[m].copy()
        p = p[m]
    for c in _POS_COLS[:3]:
        adata.obs[c] = p[c].to_numpy()
    xy = p[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
    if spot_pitch is not None:
        xy = p[["array_col", "array_row"]].to_numpy(dtype=float) * float(spot_pitch)
    adata.obsm["spatial"] = xy
    lib = {"images": {}, "scalefactors": {}}
    if spatial_dir is not None:
        lib = read_spatial_images(spatial_dir)
    lib.setdefault("metadata", {})
    adata.uns.setdefault("spatial", {})[library_id] = lib
    adata.obs["library_id"] = pd.Categorical([library_id] * adata.n_obs)
    return adata


def read_spatial_atac(
    fragments: PathLike,
    positions: Union[PathLike, pd.DataFrame],
    chrom_sizes,
    *,
    spatial_dir: Optional[PathLike] = None,
    library_id: str = "spatial",
    tissue_only: bool = True,
    min_num_fragments: int = 0,
    spot_pitch: Optional[float] = None,
    **kwargs,
):
    """Read spatial ATAC-seq (DBiT-style) into a spot-level AnnData.

    Fragments are scanned with :func:`epione.pp.import_fragments`,
    restricted to the barcodes of the position table (only in-tissue spots
    when ``tissue_only``), and the coordinates are attached with
    :func:`add_spatial`. Spots are the observations; downstream tile, peak
    and gene-score matrices (``epi.pp.add_tile_matrix``,
    ``epi.pp.make_peak_matrix``, ``epi.pp.make_gene_matrix``) and
    ``epi.pp.tsse`` work unchanged and keep ``obsm["spatial"]``.

    Arguments:
        fragments: tabix-indexed ``fragments.tsv.gz``.
        positions: barcode -> position table (see :func:`read_spatial_positions`).
        chrom_sizes: ``Genome`` or ``{chrom: length}``.
        spatial_dir: folder with scale factors and tissue images. Default:
            the folder containing ``positions`` when it holds a
            ``scalefactors_json.json``.
        library_id: key under ``uns["spatial"]``.
        tissue_only: keep only ``in_tissue`` spots (the tissue-on-spot mask).
        min_num_fragments: minimum unique fragments per spot (default 0:
            on-tissue spots are kept whatever their depth; filter later).
        spot_pitch: see :func:`add_spatial`.
        **kwargs: passed to :func:`epione.pp.import_fragments`.

    Returns:
        AnnData (spots x 0) with per-spot QC in ``obs``, ``obsm["spatial"]``,
        ``uns["spatial"]`` and ``uns["files"]["fragments"]``.
    """
    from ..pp import import_fragments

    pos = positions if isinstance(positions, pd.DataFrame) else read_spatial_positions(positions)
    if spatial_dir is None and not isinstance(positions, pd.DataFrame):
        cand = Path(positions).parent
        if _find_file(cand, ["scalefactors_json.json"]) is not None:
            spatial_dir = cand
    keep = pos[pos["in_tissue"]] if tissue_only else pos
    wl = set(keep.index) | {f"{b}-1" for b in keep.index}
    adata = import_fragments(
        fragments, chrom_sizes, whitelist=wl, min_num_fragments=min_num_fragments, **kwargs
    )
    return add_spatial(
        adata, pos, spatial_dir=spatial_dir, library_id=library_id,
        tissue_only=tissue_only, spot_pitch=spot_pitch,
    )


def _read_count_table(path: Path):
    """genes x spots (or spots x genes) dense text table -> AnnData (spots x genes)."""
    import anndata as ad
    import scipy.sparse as sp

    with _open_text(path) as fh:
        head = fh.readline()
    sep = "\t" if "\t" in head else ","
    df = pd.read_csv(path, sep=sep, index_col=0)
    df.columns = [str(c).strip('"') for c in df.columns]
    df.index = [str(i).strip('"') for i in df.index]

    def _looks_barcode(names) -> float:
        names = list(names)[:200]
        return np.mean([bool(re.fullmatch(r"[ACGTN]{8,}(-\d+)?", n)) for n in names])

    if _looks_barcode(df.columns) > _looks_barcode(df.index):
        df = df.T
    a = ad.AnnData(sp.csr_matrix(df.to_numpy(dtype=np.float32)))
    a.obs_names = df.index.astype(str)
    a.var_names = df.columns.astype(str)
    a.var_names_make_unique()
    return a


def read_spatial_atac_rna(
    fragments: Optional[PathLike],
    rna: Union[PathLike, "object"],
    positions: Union[PathLike, pd.DataFrame],
    chrom_sizes=None,
    *,
    spatial_dir: Optional[PathLike] = None,
    library_id: str = "spatial",
    tissue_only: bool = True,
    atac=None,
    as_mudata: bool = True,
    spot_pitch: Optional[float] = None,
    **kwargs,
):
    """Read spatial ATAC-RNA co-profiling (spatial ATAC-RNA-seq, DBiT-style).

    Both modalities share the spot barcodes of one DBiT grid; spots are
    aligned on the barcode without GEM-well suffix and the intersection is
    kept, in the order of the RNA table. Both modalities are then named by
    the ATAC barcode as written in the fragments file (e.g. with ``-1``),
    which fragment-based tools look up.

    Arguments:
        fragments: ATAC ``fragments.tsv.gz`` (ignored when ``atac`` is given).
        rna: RNA count table (genes x spots or spots x genes; ``.tsv``,
            ``.csv``, optionally gzipped), a 10x ``.h5`` / ``.h5ad`` file,
            or an AnnData.
        positions: barcode -> position table.
        chrom_sizes: ``Genome`` or ``{chrom: length}`` (needed with ``fragments``).
        spatial_dir, library_id, tissue_only, spot_pitch: see
            :func:`read_spatial_atac`.
        atac: an already-built ATAC AnnData (e.g. a peak matrix) to use
            instead of scanning ``fragments``.
        as_mudata: return ``MuData({"rna", "atac"})`` (default) or a
            ``(rna, atac)`` tuple.
        **kwargs: passed to :func:`epione.pp.import_fragments`.

    Returns:
        ``MuData`` with ``obsm["spatial"]`` on the MuData and on both
        modalities, or ``(rna, atac)``.
    """
    import anndata as ad

    pos = positions if isinstance(positions, pd.DataFrame) else read_spatial_positions(positions)
    if spatial_dir is None and not isinstance(positions, pd.DataFrame):
        cand = Path(positions).parent
        if _find_file(cand, ["scalefactors_json.json"]) is not None:
            spatial_dir = cand

    if isinstance(rna, ad.AnnData):
        rna_ad = rna.copy()
    else:
        rp = Path(rna)
        if rp.suffix == ".h5ad":
            rna_ad = ad.read_h5ad(rp)
        elif rp.suffix == ".h5":
            import scanpy as sc

            rna_ad = sc.read_10x_h5(str(rp))
            rna_ad.var_names_make_unique()
        else:
            rna_ad = _read_count_table(rp)
    common = dict(spatial_dir=spatial_dir, library_id=library_id,
                  tissue_only=tissue_only, spot_pitch=spot_pitch)
    rna_ad = add_spatial(rna_ad, pos, **common)
    if atac is None:
        if fragments is None or chrom_sizes is None:
            raise ValueError("give `fragments` and `chrom_sizes`, or a prebuilt `atac` AnnData")
        atac_ad = read_spatial_atac(fragments, pos, chrom_sizes, **common, **kwargs)
    else:
        atac_ad = add_spatial(atac.copy(), pos, **common)

    k_rna = pd.Index([_strip_suffix(b) for b in rna_ad.obs_names])
    k_atac = pd.Index([_strip_suffix(b) for b in atac_ad.obs_names])
    shared = k_rna[k_rna.isin(k_atac)]
    if len(shared) == 0:
        raise ValueError("RNA and ATAC share no spot barcodes")
    rna_ad = rna_ad[k_rna.get_indexer(shared)].copy()
    atac_ad = atac_ad[k_atac.get_indexer(shared)].copy()
    # Shared spot ids are the ATAC barcodes as spelled in the fragments
    # file, so fragment-based tools (tsse, tiles, gene scores) still match.
    rna_ad.obs_names = atac_ad.obs_names
    if not as_mudata:
        return rna_ad, atac_ad
    import mudata as md

    mdata = md.MuData({"rna": rna_ad, "atac": atac_ad})
    mdata.obsm["spatial"] = rna_ad.obsm["spatial"].copy()
    mdata.uns["spatial"] = rna_ad.uns["spatial"]
    return mdata
