"""Bulk ATAC-seq of sorted human hematopoietic cells (Corces et al., Nat Genet 2016)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

CORCES2016_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74912/suppl/"
    "GSE74912_ATACseq_All_Counts.txt.gz"
)
CORCES2016_SHA256 = "sha256:31a7834dd78421f5b92fd408972e857ee7ff115789a1b8a4a2cb86c7183b791f"

# Normal-donor samples: cell type and donor from the GEO sample records
# (source name and ``donorid`` characteristic of GSE74912).
CORCES2016_NORMAL = {
    "4983-1A": ("HSC", "4983"),
    "4983-2A": ("MPP", "4983"),
    "4983-3A": ("LMPP", "4983"),
    "4983-4A": ("CMP", "4983"),
    "4983-4B": ("CMP", "4983"),
    "4983-5A": ("GMP", "4983"),
    "4983-5B": ("GMP", "4983"),
    "4983-6A": ("MEP", "4983"),
    "4983-6B": ("MEP", "4983"),
    "4983-7A": ("Mono", "4983"),
    "4983-7B": ("Mono", "4983"),
    "6792-1A": ("HSC", "6792"),
    "6792-1B": ("HSC", "6792"),
    "6792-2A": ("MPP", "6792"),
    "6792-3A": ("LMPP", "6792"),
    "6792-4A": ("CMP", "6792"),
    "6792-4B": ("CMP", "6792"),
    "6792-5A": ("GMP", "6792"),
    "6792-5B": ("GMP", "6792"),
    "6792-6A": ("MEP", "6792"),
    "6792-6B": ("MEP", "6792"),
    "6792-7A": ("Mono", "6792"),
    "6792-7B": ("Mono", "6792"),
    "Donor1022-Bcell-13-150120": ("B", "1022"),
    "Donor1022-CD4-9-150120": ("CD4T", "1022"),
    "Donor1022-CD8-10-150120": ("CD8T", "1022"),
    "Donor1022-Nkcell-11-150120": ("NK", "1022"),
    "Donor2596-CD4-9A-150303": ("CD4T", "2596"),
    "Donor2596-CD8-10B-150303": ("CD8T", "2596"),
    "Donor2596-CLP-14A-150303": ("CLP", "2596"),
    "Donor2596-CLP-14B-150303": ("CLP", "2596"),
    "Donor2596-CMP-4A-150303": ("CMP", "2596"),
    "Donor2596-CMP-4B-150303": ("CMP", "2596"),
    "Donor2596-Erythroblast-15A-150303": ("Ery", "2596"),
    "Donor2596-Erythroblast-15B-150303": ("Ery", "2596"),
    "Donor2596-Erythroblast-15C-150303": ("Ery", "2596"),
    "Donor2596-GMP-5A-150303": ("GMP", "2596"),
    "Donor2596-HSC-1A-150303": ("HSC", "2596"),
    "Donor2596-HSC-1B-150303": ("HSC", "2596"),
    "Donor2596-MEP-6A-150303": ("MEP", "2596"),
    "Donor2596-MPP-2A-150303": ("MPP", "2596"),
    "Donor2596-MPP-2B-150303": ("MPP", "2596"),
    "Donor2596-NK-11B-150303": ("NK", "2596"),
    "Donor2596-Nkcell-11A-150303": ("NK", "2596"),
    "Donor4983-Bcell-13A-150120": ("B", "4983"),
    "Donor4983-CD4-9A-150120": ("CD4T", "4983"),
    "Donor4983-CD8-10A-150120": ("CD8T", "4983"),
    "Donor4983-Nkcell-11A-150120": ("NK", "4983"),
    "Donor5483-Bcell-13A-150305": ("B", "5483"),
    "Donor5483-CD4-9A-150305": ("CD4T", "5483"),
    "Donor5483-CD4-9B-150305": ("CD4T", "5483"),
    "Donor5483-CD8-10B-150305": ("CD8T", "5483"),
    "Donor5483-CLP-14A-150305": ("CLP", "5483"),
    "Donor5483-Erythroblast-15A-150305": ("Ery", "5483"),
    "Donor5483-Erythroblast-15B-150305": ("Ery", "5483"),
    "Donor5483-Erythroblast-15C-150305": ("Ery", "5483"),
    "Donor5483-NK-11A-150305": ("NK", "5483"),
    "Donor5483-Nkcell-11B-150305": ("NK", "5483"),
    "Donor6926-CLP-14A-150318": ("CLP", "6926"),
    "Donor6926-Erythroblast-15A-150318": ("Ery", "6926"),
    "Donor6926-Erythroblast-15B-150318": ("Ery", "6926"),
    "Donor7256-1A-110614": ("HSC", "7256"),
    "Donor7256-1B-110614": ("HSC", "7256"),
    "Donor7256-2A-110614": ("MPP", "7256"),
    "Donor7256-2B-110614": ("MPP", "7256"),
    "Donor7256-3A-141106": ("LMPP", "7256"),
    "Donor7256-4A-110614": ("CMP", "7256"),
    "Donor7256-4B-141106": ("CMP", "7256"),
    "Donor7256-5A-141106": ("GMP", "7256"),
    "Donor7256-5B-141106": ("GMP", "7256"),
    "Donor7256-6A-141106": ("MEP", "7256"),
    "Donor7256-6B-141106": ("MEP", "7256"),
    "Donor7256-7A-141106": ("Mono", "7256"),
    "Donor7256-7B-141106": ("Mono", "7256"),
    "Donor5483-Bcell-13B-150305": ("B", "5483"),
    "Donor6926-CLP-14B-150318": ("CLP", "6926"),
    "Donor5483-CD8-10A-150305": ("CD8T", "5483"),
}

HEMATOPOIESIS_ORDER = ["HSC", "MPP", "LMPP", "CMP", "GMP", "MEP", "Ery", "Mono",
                       "CLP", "B", "CD4T", "CD8T", "NK"]


def _other_type(name: str) -> str:
    if "pHSC" in name:
        return "pHSC"
    if "LSC" in name:
        return "LSC"
    if "Leuk" in name:
        return "Blast"
    if name.startswith("BM-CD34"):
        return "CD34 Bone Marrow"
    if name.startswith("CB"):
        return "CD34 Cord Blood"
    return "unknown"


def corces2016_bulk_atac(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    normal_only: bool = True,
    progressbar: bool = False,
):
    """Bulk ATAC-seq counts of sorted hematopoietic populations (hg19).

    The peak-by-sample count table of Corces *et al.* 2016 (GEO GSE74912):
    590,650 peaks of 500 bp and 132 samples, among them 77 samples of 13
    sorted normal populations from 9 donors (HSC, MPP, LMPP, CMP, GMP, MEP,
    erythroblasts, monocytes, CLP, B, CD4 T, CD8 T, NK) plus AML samples. This
    is the bulk data ArchR projects onto its hematopoiesis scATAC-seq
    embedding. The file is fetched with :mod:`pooch` and verified against a
    pinned sha256.

    Arguments:
        data_dir: Download directory; defaults to
            ``<pooch cache>/epione/corces2016``.
        normal_only: Keep only the 77 normal-donor samples.
        progressbar: Show a download progress bar.

    Returns:
        AnnData, samples x peaks, integer counts in CSR. ``obs`` has
        ``cell_type`` and ``donor``, ``var`` has ``chrom``, ``start``,
        ``end``; peak names are ``chr:start-end``. Normal-sample cell types
        come from the GEO records; AML sample labels (``pHSC``, ``LSC``,
        ``Blast``) are derived from the sample names.

    Examples:
        >>> bulk = epi.datasets.corces2016_bulk_atac()
        >>> bulk.obs["cell_type"].value_counts()["HSC"]
        7
    """
    import numpy as np
    import pandas as pd
    import pooch
    import scipy.sparse as sp
    from anndata import AnnData

    if data_dir is None:
        data_dir = Path(pooch.os_cache("epione")) / "corces2016"
    path = pooch.retrieve(
        url=CORCES2016_URL, known_hash=CORCES2016_SHA256,
        fname="GSE74912_ATACseq_All_Counts.txt.gz", path=str(data_dir),
        progressbar=progressbar,
    )
    tab = pd.read_csv(path, sep="\t")
    samples = [c for c in tab.columns if c not in ("Chr", "Start", "End")]
    if normal_only:
        samples = [s for s in samples if s in CORCES2016_NORMAL]
    X = sp.csr_matrix(tab[samples].to_numpy(dtype=np.int32).T)
    var = pd.DataFrame({
        "chrom": tab["Chr"].astype(str).to_numpy(),
        "start": tab["Start"].astype(np.int64).to_numpy(),
        "end": tab["End"].astype(np.int64).to_numpy(),
    })
    var.index = var["chrom"] + ":" + var["start"].astype(str) + "-" + var["end"].astype(str)
    obs = pd.DataFrame(index=samples)
    obs["cell_type"] = [CORCES2016_NORMAL.get(s, (_other_type(s), ""))[0] for s in samples]
    obs["donor"] = [CORCES2016_NORMAL.get(s, ("", ""))[1] for s in samples]
    cats = HEMATOPOIESIS_ORDER + sorted(set(obs["cell_type"]) - set(HEMATOPOIESIS_ORDER))
    obs["cell_type"] = pd.Categorical(obs["cell_type"], categories=[c for c in cats if c in set(obs["cell_type"])])
    return AnnData(X=X, obs=obs, var=var)
