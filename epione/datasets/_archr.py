"""ArchR's Hematopoiesis tutorial fragments (Granja et al., Nat Biotechnol 2019)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

ARCHR_HEME_BASE_URL = "https://jeffgranja.s3.amazonaws.com/ArchR/TestData/HemeFragments/"

# Same files and URLs as ``ArchR::getTutorialData("Hematopoiesis")``;
# ArchR pins the md5 sums, which these sha256 sums were checked against.
ARCHR_HEME_FRAGMENTS = {
    "scATAC_BMMC_R1": "sha256:c2e2a7a45340f40d7e7345c5be101ee02cc085ec8439fd927f05b94e67e734e7",
    "scATAC_CD34_BMMC_R1": "sha256:5ae1f4a645f77c58ef1fb472addaf796069af16ac5c42b24ad9855c1af15da44",
    "scATAC_PBMC_R1": "sha256:c487ce44c9f180f96877ccbfa1d0bc41e5dce350a80f89e091ac0400134a0f56",
}


def archr_hematopoiesis(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    index: bool = True,
    progressbar: bool = False,
) -> dict:
    """Download the fragment files of ArchR's Hematopoiesis tutorial.

    Three 10x scATAC-seq samples (bone marrow, CD34+ bone marrow and
    peripheral blood; hg19) from Granja *et al.* 2019 — the dataset ArchR
    uses throughout its own tutorial. Files are fetched with :mod:`pooch`,
    verified against pinned sha256 sums and reused when already present.

    Arguments:
        data_dir: Directory to store the files in. They are written to
            ``<data_dir>/HemeFragments/``. Defaults to
            ``<pooch cache>/epione/archr_hematopoiesis``.
        index: Build a tabix index (``.tbi``) next to each file if missing.
        progressbar: Show a download progress bar.

    Returns:
        Dict mapping sample name to the local ``fragments.tsv.gz`` path, in
        the order BMMC, CD34+ BMMC, PBMC.

    Examples:
        >>> frags = epi.datasets.archr_hematopoiesis('data_archr_heme')
        >>> list(frags)
        ['scATAC_BMMC_R1', 'scATAC_CD34_BMMC_R1', 'scATAC_PBMC_R1']
    """
    import pooch

    if data_dir is None:
        data_dir = Path(pooch.os_cache("epione")) / "archr_hematopoiesis"
    frag_dir = Path(data_dir) / "HemeFragments"
    frag_dir.mkdir(parents=True, exist_ok=True)

    out = {}
    for sample, known_hash in ARCHR_HEME_FRAGMENTS.items():
        fname = f"{sample}.fragments.tsv.gz"
        path = pooch.retrieve(
            url=ARCHR_HEME_BASE_URL + fname,
            known_hash=known_hash,
            fname=fname,
            path=str(frag_dir),
            progressbar=progressbar,
        )
        if index and not os.path.exists(path + ".tbi"):
            import pysam
            pysam.tabix_index(path, preset="bed", force=True)
        out[sample] = Path(path)
    return out
