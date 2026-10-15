"""Spatial ATAC / ATAC-RNA example data (DBiT-seq family, Rong Fan lab)."""
from __future__ import annotations

import os
import tarfile
from pathlib import Path
from typing import Optional, Union

GEO_SAMPLES = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6206nnn/"

ZHANG2023_HIPPOCAMPUS = {
    "GSM6206884_HumanBrain_50um_fragments.tsv.gz": (
        "GSM6206884/suppl/", "sha256:ee387d187c01a04c95114411166e100ea19f1d0b3b5948af4e2df26ef9ccc2fa"),
    "GSM6206884_HumanBrain_50um_spatial.tar.gz": (
        "GSM6206884/suppl/", "sha256:1e7596a538651894f070fa55607bcd1e0b383a6bcf64cf4898e5e7e67a126abd"),
    "GSM6206885_HumanBrain_50um_matrix.tsv.gz": (
        "GSM6206885/suppl/", "sha256:a45d110c233e463e78bec42c090e53005bc2f006b5a94dd55d0ec2ca480bf73a"),
}

SPATIALGLUE_ZENODO = "https://zenodo.org/api/records/10362607/files/Data_SpatialGlue.zip/content"
SPATIALGLUE_SHA256 = "sha256:456da38148fc6fd34762db193c90ef6742eb998d9d7ecbafdcaad6853f050b87"


def _cache(data_dir, name):
    import pooch

    d = Path(data_dir) if data_dir is not None else Path(pooch.os_cache("epione")) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def zhang2023_hippocampus_atac_rna(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    progressbar: bool = False,
) -> dict:
    """Spatial ATAC-RNA-seq of a human hippocampus section, 50 um DBiT pixels.

    Zhang, D. et al. "Spatial epigenome-transcriptome co-profiling of
    mammalian tissues". *Nature* 616, 113-122 (2023); GEO GSE205055, samples
    GSM6206884 (ATAC fragments, hg38, bgzipped) and GSM6206885 (RNA counts).
    A 50 x 50 barcode grid, 2,500 on-tissue spots. The three GEO files
    (fragments 456 MB, RNA 5 MB, spatial folder 52 MB) are fetched with
    :mod:`pooch`, checked against pinned sha256 sums, and the Space Ranger
    style ``spatial/`` folder (positions, scale factors, image) is unpacked.

    Returns:
        dict with ``fragments``, ``rna``, ``positions`` and ``spatial_dir``
        paths, ready for :func:`epione.io.read_spatial_atac_rna`.
    """
    import pooch

    d = _cache(data_dir, "zhang2023_hippocampus")
    p = {}
    for fname, (sub, h) in ZHANG2023_HIPPOCAMPUS.items():
        p[fname] = Path(pooch.retrieve(GEO_SAMPLES + sub + fname, known_hash=h, fname=fname,
                                       path=str(d), progressbar=progressbar))
    sdir = d / "HumanBrain_50um_spatial"
    if not (sdir / "tissue_positions_list.csv").exists():
        with tarfile.open(p["GSM6206884_HumanBrain_50um_spatial.tar.gz"]) as tf:
            members = [m for m in tf.getmembers()
                       if m.isfile() and not os.path.isabs(m.name) and ".." not in Path(m.name).parts]
            tf.extractall(d, members=members)
    return {
        "fragments": p["GSM6206884_HumanBrain_50um_fragments.tsv.gz"],
        "rna": p["GSM6206885_HumanBrain_50um_matrix.tsv.gz"],
        "positions": sdir / "tissue_positions_list.csv",
        "spatial_dir": sdir,
    }


def spatialglue_mouse_brain(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    dataset: str = "Dataset7_Mouse_Brain_ATAC",
    progressbar: bool = False,
):
    """Preprocessed mouse brain spatial ATAC-RNA-seq of the SpatialGlue paper.

    P22 mouse brain, spatial ATAC-RNA-seq (Zhang et al. 2023), 9,215 spots,
    as released by Long et al. (*Nature Methods* 2024) on Zenodo record
    10362607 (CC-BY-4.0). ``rna`` holds raw gene counts; ``atac`` the peak
    matrix with the authors' ``X_lsi``; both carry ``obsm["spatial"]`` and
    the original per-modality clusters (``obs["RNA_clusters"]``,
    ``obs["ATAC_clusters"]``). The 672 MB archive is downloaded once and
    only ``dataset`` is unpacked (other choices: ``Dataset8_Mouse_Brain_H3K4me3``,
    ``Dataset9_Mouse_Brain_H3K27ac``, ``Dataset10_Mouse_Brain_H3K27me3``).

    Returns:
        ``(rna, atac)`` AnnData objects with the same spots in the same order.
    """
    import zipfile

    import anndata as ad
    import pooch

    d = _cache(data_dir, "spatialglue")
    out = d / dataset
    files = ("adata_RNA.h5ad", "adata_peaks_normalized.h5ad")
    if not all((out / f).exists() for f in files):
        z = pooch.retrieve(SPATIALGLUE_ZENODO, known_hash=SPATIALGLUE_SHA256, fname="Data_SpatialGlue.zip",
                           path=str(d), progressbar=progressbar)
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for f in files:
                name = f"Data_SpatialGlue/{dataset}/{f}"
                with zf.open(name) as src, open(out / f, "wb") as dst:
                    dst.write(src.read())
    rna = ad.read_h5ad(out / files[0])
    atac = ad.read_h5ad(out / files[1])
    rna.var_names_make_unique()
    atac.var_names_make_unique()
    return rna, atac
