"""Download helpers for the Aerts-lab cisTarget resources (pooch-cached)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

CISTARGET_BASE = "https://resources.aertslab.org/cistarget/databases/"

# sha1 sums published next to each file on resources.aertslab.org (*.sha1sum.txt)
CISTARGET_DATABASES = {
    ("hg38", "screen_v10_clust", "rankings"): (
        "homo_sapiens/hg38/screen/mc_v10_clust/region_based/"
        "hg38_screen_v10_clust.regions_vs_motifs.rankings.feather",
        "sha1:1688a925f22d312769798258d990f13866bb4924"),
    ("hg38", "screen_v10_clust", "scores"): (
        "homo_sapiens/hg38/screen/mc_v10_clust/region_based/"
        "hg38_screen_v10_clust.regions_vs_motifs.scores.feather",
        "sha1:07b5e527d2ed082e081e439e68dffa77b5f6129c"),
}

MOTIF_ANNOTATIONS = {
    ("homo_sapiens", "v10nr_clust"): (
        "https://resources.aertslab.org/cistarget/motif2tf/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl",
        "sha256:81eb754118e27e854974301b1400fcf519489f8be5249239671fb288cb501c31"),
}


def _cache_dir(data_dir, sub):
    import pooch
    if data_dir is None:
        data_dir = os.environ.get("EPIONE_DATA_DIR") or pooch.os_cache("epione")
    d = Path(data_dir) / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_cistarget_database(
    assembly: str = "hg38",
    kind: str = "rankings",
    collection: str = "screen_v10_clust",
    data_dir: Optional[Union[str, os.PathLike]] = None,
    verify: bool = False,
    progressbar: bool = False,
) -> str:
    """Download an official region-based cisTarget database.

    Parameters
    ----------
    assembly, collection, kind
        For example ``("hg38", "screen_v10_clust", "rankings" | "scores")``.
        This is the database SCENIC+ uses for human data. The rankings file
        is 35 GB and the scores file 14 GB.
    data_dir
        Directory to store the files in (``<data_dir>/cistarget/``). Defaults to
        ``$EPIONE_DATA_DIR`` or the pooch cache.
    verify
        Check the published sha1 sum, also when the file already exists. Off by
        default because hashing 35 GB takes minutes. A download is always
        verified when the hash is known.

    Returns
    -------
    Local path of the ``.feather`` file.
    """
    import pooch
    key = (assembly, collection, kind)
    if key not in CISTARGET_DATABASES:
        raise KeyError(f"unknown database {key}; known: {sorted(CISTARGET_DATABASES)}")
    rel, sha = CISTARGET_DATABASES[key]
    d = _cache_dir(data_dir, "cistarget")
    fname = rel.rsplit("/", 1)[1]
    exists = (d / fname).exists()
    return pooch.retrieve(CISTARGET_BASE + rel, known_hash=sha if (verify or not exists) else None,
                          path=d, fname=fname, progressbar=progressbar)


def fetch_motif_annotation(species: str = "homo_sapiens", version: str = "v10nr_clust",
                           data_dir: Optional[Union[str, os.PathLike]] = None) -> str:
    """Download the motif2TF annotation table matching a cisTarget database."""
    import pooch
    url, sha = MOTIF_ANNOTATIONS[(species, version)]
    d = _cache_dir(data_dir, "cistarget")
    return pooch.retrieve(url, known_hash=sha, path=d, fname=url.rsplit("/", 1)[1],
                          progressbar=False)


def jaspar_annotation(motif_names: Sequence[str]) -> pd.DataFrame:
    """Motif → TF table for JASPAR-named motifs (``MA0035.4_GATA1``,
    ``MA0089.2_MAFG::NFE2L1``), in the layout :func:`run_cistarget` expects.
    The TF symbols after the matrix ID are the direct annotation, and
    heterodimers list both partners."""
    rows = {}
    for m in motif_names:
        tf = m.split("_", 1)[1] if "_" in m else m
        tfs = sorted({t.upper() for t in tf.replace("::", "+").split("+") if t})
        rows[m] = ", ".join(tfs)
    df = pd.DataFrame({"Direct_annot": pd.Series(rows)})
    df["Orthology_annot"] = pd.NA
    return df
