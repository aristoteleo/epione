"""10x scATAC-seq of a mixture of 11 cell lines with genotype-based doublet calls.

Granja et al. (ArchR paper, Nat Genet 2021), GEO GSE162690. The two samples
were loaded at a low and a high concentration to create doublets, and every
barcode was genotyped with demuxlet. This makes them a ground-truth benchmark
for doublet detection, used as such by the scDblFinder paper (Germain et al.,
F1000Research 2022).
"""
from __future__ import annotations

import gzip
import os
import shutil
from pathlib import Path
from typing import Optional, Union

GEO_SUPPL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE162nnn/GSE162690/suppl/"
UCSC_HG38_SIZES = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes"

_FILES = {
    "low": {
        "fragments": ("GSE162690_CellLine_LowLoading.fragments.tsv.gz",
                      "sha256:c81a26fa65320c62d34141a07beda1dac9c188b1d32955d9cefce8576ca5f01e"),
        "demuxlet": ("GSE162690_demuxlet_CellLineLow_mergedBAMs.best.txt.gz",
                     "sha256:283173e985d8a9de8da03f15a3b661d79fd46031d10fc5142aa595d4b1013b8a"),
    },
    "high": {
        "fragments": ("GSE162690_CellLine_HighLoading.fragments.tsv.gz",
                      "sha256:08f18fbaf1cc3f61ccb859fa728355089ec3353c8a614ef859b4fe16379ea8a0"),
        "demuxlet": ("GSE162690_demuxlet_CellLineHigh_mergedBAMs.best.txt.gz",
                     "sha256:5807e56dd00941f70e438783521fb02137c2ed22117bccde451d9cb84f4df947"),
    },
}
_HG38_SIZES_HASH = "sha256:e1d9152418038457a959e949d99c9caf7ae3e4f87cfbb4ffe7d8b9a54ba1202b"


def _bgzip_and_index(src: Path, dst: Path) -> Path:
    """Re-compress a (position-sorted) plain-gzip fragment file with BGZF + tabix."""
    import pysam
    if dst.exists() and Path(str(dst) + ".tbi").exists():
        return dst
    plain = dst.with_suffix("")  # .tsv
    with gzip.open(src, "rb") as fin, open(plain, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=16 << 20)
    pysam.tabix_compress(str(plain), str(dst), force=True)
    plain.unlink()
    pysam.tabix_index(str(dst), preset="bed", force=True)
    return dst


def granja2021_cellline_mix(
    loading: str = "low",
    data_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    progressbar: bool = False,
) -> dict:
    """Cell-line mixture scATAC-seq with demuxlet doublet calls (GSE162690).

    Arguments:
        loading: ``"low"`` (68 M fragments, about 3% genotype doublets among
            called cells) or ``"high"`` (about 7%).
        data_dir: Where to store the files. Defaults to
            ``<pooch cache>/epione/granja2021_cellline_mix``.
        progressbar: Show download progress bars.

    Returns:
        A dict with:

        - ``"fragments"``: path to a BGZF-compressed, tabix-indexed fragment
          file (hg38). The GEO file is plain gzip, so on first use it is
          re-compressed, which needs about 4 GB of temporary disk for
          ``"low"``.
        - ``"demuxlet"``: DataFrame indexed by barcode, with ``call``
          (``"SNG"``, ``"DBL"`` or ``"AMB"``), ``cell_line`` (the singlet's
          line, or the pair for doublets) and demuxlet's read counts.
        - ``"chrom_sizes"``: ``{chrom: length}`` for chr1–22, X and Y (UCSC
          hg38).

    Notes:
        Demuxlet cannot detect two cells of the *same* line in one droplet,
        so a minority of "singlets" are homotypic doublets.

    Examples:
        >>> d = epi.datasets.granja2021_cellline_mix("low")
        >>> d["demuxlet"]["call"].value_counts()
    """
    import pandas as pd
    import pooch

    key = loading.lower().replace("loading", "")
    if key not in _FILES:
        raise ValueError("loading must be 'low' or 'high'")
    if data_dir is None:
        data_dir = Path(pooch.os_cache("epione")) / "granja2021_cellline_mix"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    got = {}
    for what, (fname, h) in _FILES[key].items():
        got[what] = Path(pooch.retrieve(url=GEO_SUPPL + fname, known_hash=h, fname=fname,
                                        path=str(data_dir), progressbar=progressbar))
    sizes_f = pooch.retrieve(url=UCSC_HG38_SIZES, known_hash=_HG38_SIZES_HASH,
                             fname="hg38.chrom.sizes", path=str(data_dir), progressbar=False)
    frag = _bgzip_and_index(got["fragments"], data_dir / f"cellline_{key}.fragments.bgz.tsv.gz")

    dmx = pd.read_csv(got["demuxlet"], sep="\t")
    dmx = dmx[dmx["BARCODE"] != "."].set_index("BARCODE")
    best = dmx["BEST"].str.split("-", n=1, expand=True)
    dmx.insert(0, "call", best[0])
    lines = best[1].str.replace("CellLine_CellHash_Bulk_", "", regex=False)
    dmx.insert(1, "cell_line", lines.str.rsplit("-", n=1).str[0].where(dmx["call"] == "DBL", lines))
    sizes = pd.read_csv(sizes_f, sep="\t", header=None, index_col=0)[1]
    keep = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]
    return dict(fragments=frag, demuxlet=dmx, chrom_sizes={c: int(sizes[c]) for c in keep})
