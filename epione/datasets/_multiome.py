"""10x Multiome PBMC 10k, as released by scGLUE (Cao & Gao, Nat Biotechnol 2022)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

GLUE_BASE_URL = "https://download.gao-lab.org/GLUE/dataset/"

PBMC10K_MULTIOME = {
    "10x-Multiome-Pbmc10k-RNA.h5ad": "sha256:708a48104091e2528028ad3b80962ad17aca35fc703d3410cb73c32a68a255a0",
    "10x-Multiome-Pbmc10k-ATAC.h5ad": "sha256:4d313cb3f56419a72d3143f7b1d35c3afb2c07c3631ed390a1c6ae0d3f4aac07",
}


def pbmc10k_multiome(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    progressbar: bool = False,
):
    """Paired scRNA-seq and scATAC-seq of 9,631 human PBMCs (10x Multiome).

    The 10x Genomics "pbmc_granulocyte_sorted_10k" sample in the scGLUE
    release: raw RNA counts (29,095 genes), raw ATAC peak counts (107,194
    peaks, ``var`` has ``chrom``/``chromStart``/``chromEnd``) and Seurat-WNN
    cell types in ``obs["cell_type"]``. This is the PBMC dataset of the
    scTranslation benchmark. Files are fetched with :mod:`pooch` and verified
    against pinned sha256 sums.

    Arguments:
        data_dir: Directory to store the two h5ad files in. Defaults to
            ``<pooch cache>/epione/pbmc10k_multiome``.
        progressbar: Show a download progress bar.

    Returns:
        ``(rna, atac)`` AnnData objects with the same cells in the same order.

    Examples:
        >>> rna, atac = epi.datasets.pbmc10k_multiome()
        >>> rna.shape, atac.shape
        ((9631, 29095), (9631, 107194))
    """
    import anndata as ad
    import pooch

    if data_dir is None:
        data_dir = Path(pooch.os_cache("epione")) / "pbmc10k_multiome"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for fname, known_hash in PBMC10K_MULTIOME.items():
        paths[fname] = pooch.retrieve(url=GLUE_BASE_URL + fname, known_hash=known_hash, fname=fname,
                                      path=str(data_dir), progressbar=progressbar)
    rna = ad.read_h5ad(paths["10x-Multiome-Pbmc10k-RNA.h5ad"])
    atac = ad.read_h5ad(paths["10x-Multiome-Pbmc10k-ATAC.h5ad"])
    return rna, atac


TENX_BASE_URL = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/"

PBMC10K_MULTIOME_10X = {
    "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5":
        "sha256:f6824171378787baab244f559b8b438f79db2eb39f78d17b2196f7ecd2c03549",
    "pbmc_granulocyte_sorted_10k_per_barcode_metrics.csv":
        "sha256:fd3e069b83e152145af234667b419c982968aca0df322a92adb71284d0d902cd",
}


def pbmc10k_multiome_10x(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    progressbar: bool = False,
) -> Path:
    """The cellranger-arc 2.0.0 output of the same 10x PBMC 10k Multiome sample.

    Downloads the filtered feature-barcode matrix (``.h5``, 192 MB; Gene
    Expression + Peaks) and ``per_barcode_metrics.csv`` (89 MB) from 10x
    Genomics, verified against pinned sha256 sums, and returns the folder.
    Read it with :func:`epione.io.read_10x_multiome`. The 2 GB fragment file
    is not downloaded.

    Arguments:
        data_dir: Folder for the files. Defaults to
            ``<pooch cache>/epione/pbmc10k_multiome_10x``.
        progressbar: Show a download progress bar.

    Returns:
        The folder containing the two files.

    Examples:
        >>> rna, atac = epi.io.read_10x_multiome(epi.datasets.pbmc10k_multiome_10x())
        >>> rna.shape, atac.shape
        ((11898, 36601), (11898, 143887))
    """
    import pooch

    if data_dir is None:
        data_dir = Path(pooch.os_cache("epione")) / "pbmc10k_multiome_10x"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname, known_hash in PBMC10K_MULTIOME_10X.items():
        pooch.retrieve(url=TENX_BASE_URL + fname, known_hash=known_hash, fname=fname,
                       path=str(data_dir), progressbar=progressbar)
    return data_dir


PBMC3K_BASE_URL = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/"

PBMC3K_ATAC = {
    "fragments": ("pbmc_granulocyte_sorted_3k_atac_fragments.tsv.gz",
                  "sha256:cb7725f828fe99c3660cdf24843347e91aec72507ae3ace3030a0c49e50352ee"),
    "fragments_index": ("pbmc_granulocyte_sorted_3k_atac_fragments.tsv.gz.tbi",
                        "sha256:365aa0ddb3d51e3618954752fd7e41a52c489591bd38c0c023e42475653fb7a2"),
    "peaks": ("pbmc_granulocyte_sorted_3k_atac_peaks.bed",
              "sha256:1b3fb82546659d51faf25957aec01abeb88b89d7c91f3019b3bce55f33a18ddb"),
    "per_barcode_metrics": ("pbmc_granulocyte_sorted_3k_per_barcode_metrics.csv",
                            "sha256:d8361253c64a67fd31ed9a226623d1d824b9acbc1bfe18c7c4b897de9d02c4f7"),
}


def pbmc3k_multiome_atac(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    progressbar: bool = False,
) -> dict:
    """ATAC files of the 10x Multiome PBMC 3k sample (cellranger-arc 2.0.0).

    Downloads the fragments file (468 MB) with its tabix index, the called
    peaks (2.3 MB) and the per-barcode metrics (with 10x's cell calls) from
    10x Genomics, verified against pinned sha256 sums. A small bulk-like
    ATAC sample used by the ChromBPNet tutorial.

    Arguments:
        data_dir: Folder for the files. Defaults to
            ``<pooch cache>/epione/pbmc3k_multiome``.
        progressbar: Show a download progress bar.

    Returns:
        ``{"fragments", "fragments_index", "peaks", "per_barcode_metrics"}``
        -> local paths.
    """
    import pooch

    if data_dir is None:
        data_dir = Path(pooch.os_cache("epione")) / "pbmc3k_multiome"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, (fname, known_hash) in PBMC3K_ATAC.items():
        out[key] = Path(pooch.retrieve(url=PBMC3K_BASE_URL + fname, known_hash=known_hash,
                                       fname=fname, path=str(data_dir), progressbar=progressbar))
    return out


PBMC10K_ATAC = {
    "fragments": (TENX_BASE_URL, "pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz",
                  "sha256:5075e32a0e9c6dded35b060bf90d6144375b150e131ffb0be121a93e3b5e1e38"),
    "fragments_index": (TENX_BASE_URL, "pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz.tbi",
                        "sha256:3a516291d0e6e5ddf9f651f470b6312b83eeb26f153ea12cfa9d082760a5e7f5"),
    "peaks": (TENX_BASE_URL, "pbmc_granulocyte_sorted_10k_atac_peaks.bed",
              "sha256:3975a4057f9caa3fb69ddaecc6ae9e530e77551717a1464c2d93ac9d73cb60ab"),
    "atac": (GLUE_BASE_URL, "10x-Multiome-Pbmc10k-ATAC.h5ad",
             PBMC10K_MULTIOME["10x-Multiome-Pbmc10k-ATAC.h5ad"]),
}


def pbmc10k_multiome_atac(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    progressbar: bool = False,
) -> dict:
    """ATAC fragments and cell types of the 10x Multiome PBMC 10k sample.

    Downloads, verified against pinned sha256 sums:

    * the cellranger-arc 2.0.0 fragments file (2.4 GB) with its tabix index
      and called peaks, from 10x Genomics;
    * scGLUE's ATAC AnnData of the same cells (154 MB), whose
      ``obs["cell_type"]`` holds the Seurat-WNN cell types (19 types, 9,631
      cells; barcodes match the fragments file).

    Deep enough (215 M fragments in cells) for footprinting per cell type;
    used by the PRINT and seq2PRINT tutorials.

    Arguments:
        data_dir: Folder for the files. Defaults to
            ``<pooch cache>/epione/pbmc10k_multiome``.
        progressbar: Show a download progress bar.

    Returns:
        ``{"fragments", "fragments_index", "peaks", "atac"}`` -> local paths.
    """
    import pooch

    if data_dir is None:
        data_dir = Path(pooch.os_cache("epione")) / "pbmc10k_multiome"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, (base, fname, known_hash) in PBMC10K_ATAC.items():
        out[key] = Path(pooch.retrieve(url=base + fname, known_hash=known_hash, fname=fname,
                                       path=str(data_dir), progressbar=progressbar))
    return out
