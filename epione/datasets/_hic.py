"""Bulk Hi-C example data: the Galaxy HiCExplorer lesson and HiCExplorer test matrices."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

GALAXY_HIC_URL = "https://zenodo.org/records/16416373/files/{}?download=1"

DROSOPHILA_S2_HIC = {
    "corrected_contact_matrix_dm3_large.h5":
        "sha256:2599f295435449a4ec042802a3ac59bfdb0dc87032fbde67f2f988fe38c1ba09",
    "dm3_genes.bed": "sha256:4a30e3237fa93391637821866c189191ce0e0ef77729e006f6f5251a5c90351b",
    "H3K27me3.bw": "sha256:cd7a42f6947821f7d4cc2e46f80c041134a2d676a2b017be0973b01125c973f7",
    "H3K36me3.bw": "sha256:96016795a05339b677bab00188e0356891e8b766178914592fea29b1e6400df5",
    "H4K16ac.bw": "sha256:7aab397740f2ff1b7ee1065201602a978d0504093cdefd86c1e373724e404722",
}

HICEXPLORER_TEST_URL = ("https://raw.githubusercontent.com/deeptools/HiCExplorer/3.7.6/"
                        "hicexplorer/test/test_data/{}")

HICEXPLORER_TEST_MATRICES = {
    "untreated": ("hicDifferentialTAD/GSM2644945_Untreated-R1.100000_chr1.cool",
                  "sha256:3326ffe1c55464b9c49af72697c59f01fc49f2d11e6f9b7c6f51b752a68a18b1"),
    "auxin": ("hicDifferentialTAD/GSM2644947_Auxin2days-R1.100000_chr1.cool",
              "sha256:6b3f3f4e8bf70b18832722cbf64195a0b267fdfaff2196f59cfb92b48f9b806c"),
    "gm12878_2_5mb": ("hicDetectLoops/GSE63525_GM12878_insitu_primary_2_5mb.cool",
                      "sha256:1f6e27d0005019874e6f8af012de8a6e1f76791c77f9862fd40e36e4602900f0"),
}


def _dir(data_dir, name):
    import pooch
    if data_dir is None:
        data_dir = Path(pooch.os_cache("epione")) / name
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return Path(data_dir)


def drosophila_s2_hic(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    tracks: bool = True,
    progressbar: bool = False,
) -> dict:
    """Drosophila S2 Hi-C from the Galaxy HiCExplorer training lesson (dm3).

    Downloads the lesson's corrected contact matrix (HiCExplorer ``.h5``,
    52 MB; Ramírez et al. 2018 / Galaxy Training Network) from Zenodo record
    16416373, converts it once to ``.cool`` with
    :func:`epione.bulk.hic.h5_to_cool` (masked bins get ``NaN`` weights so
    cooltools skips them), and optionally fetches the dm3 gene
    BED and three histone-mark bigWigs (H3K27me3, H3K36me3, H4K16ac) the
    lesson plots alongside the matrix. Every file is checked against a pinned
    sha256.

    Arguments:
        data_dir: Where to store the files. Defaults to
            ``<pooch cache>/epione/drosophila_s2_hic``.
        tracks: Also download the gene BED and the bigWigs (about 84 MB).
        progressbar: Show download progress bars.

    Returns:
        ``dict`` of paths: ``"cool"``, ``"h5"`` and, with ``tracks``,
        ``"genes"``, ``"H3K27me3"``, ``"H3K36me3"``, ``"H4K16ac"``.
    """
    import pooch

    d = _dir(data_dir, "drosophila_s2_hic")
    names = ["corrected_contact_matrix_dm3_large.h5"]
    if tracks:
        names += ["dm3_genes.bed", "H3K27me3.bw", "H3K36me3.bw", "H4K16ac.bw"]
    got = {}
    for n in names:
        got[n] = pooch.retrieve(url=GALAXY_HIC_URL.format(n), known_hash=DROSOPHILA_S2_HIC[n], fname=n,
                                path=str(d), progressbar=progressbar)
    cool = d / "corrected_contact_matrix_dm3_large.cool"
    if not cool.exists():
        from epione.bulk.hic import h5_to_cool
        tmp = d / "corrected_contact_matrix_dm3_large.cool.tmp"
        h5_to_cool(got["corrected_contact_matrix_dm3_large.h5"], tmp, nan_masked=True)
        os.replace(tmp, cool)
    out = {"cool": str(cool), "h5": got["corrected_contact_matrix_dm3_large.h5"]}
    if tracks:
        out["genes"] = got["dm3_genes.bed"]
        for m in ("H3K27me3", "H3K36me3", "H4K16ac"):
            out[m] = got[f"{m}.bw"]
    return out


def hicexplorer_test_matrices(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    progressbar: bool = False,
) -> dict:
    """Small ``.cool`` matrices from HiCExplorer's test data (tag 3.7.6).

    * ``"untreated"`` / ``"auxin"``: HCT-116 RAD21-mAID Hi-C, untreated and
      after 2 days of auxin (cohesin depleted), chr1 at 100 kb (Rao et al.,
      Cell 2017; GEO GSM2644945 / GSM2644947) -- 9.5 MB each.
    * ``"gm12878_2_5mb"``: GM12878 in situ Hi-C at 2.5 Mb (Rao et al., Cell
      2014; GSE63525) -- 1.6 MB.

    Arguments:
        data_dir: Where to store the files. Defaults to
            ``<pooch cache>/epione/hicexplorer_test``.
        progressbar: Show download progress bars.

    Returns:
        ``dict`` of ``.cool`` paths keyed as above.
    """
    import pooch

    d = _dir(data_dir, "hicexplorer_test")
    out = {}
    for key, (rel, h) in HICEXPLORER_TEST_MATRICES.items():
        out[key] = pooch.retrieve(url=HICEXPLORER_TEST_URL.format(rel), known_hash=h,
                                  fname=Path(rel).name, path=str(d), progressbar=progressbar)
    return out
