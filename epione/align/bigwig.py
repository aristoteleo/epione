from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union


def bam_to_bigwig(
    bam: Union[str, Path],
    out_bw: Union[str, Path],
    *,
    bin_size: int = 10,
    normalize_using: Optional[str] = None,
    effective_genome_size: Optional[int] = None,
    scale_factor: Optional[float] = None,
    extend_reads: Optional[int] = None,
    center_reads: bool = False,
    min_mapping_quality: Optional[int] = None,
    ignore_for_normalization: Optional[Sequence[str]] = None,
    threads: int = 8,
) -> str:
    out_bw = Path(out_bw)
    out_bw.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import pyBigWig
    import pysam

    ignore_for_normalization = set(ignore_for_normalization or [])
    mapped_reads = 0
    chrom_sizes: Dict[str, int] = {}
    with pysam.AlignmentFile(str(bam), "rb") as bf:
        for sq in bf.header.get("SQ", []):
            chrom_sizes[sq["SN"]] = int(sq["LN"])
        for read in bf.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.is_qcfail or read.is_duplicate:
                continue
            if min_mapping_quality is not None and read.mapping_quality < min_mapping_quality:
                continue
            if read.reference_name in ignore_for_normalization:
                continue
            mapped_reads += 1

    if mapped_reads == 0:
        raise RuntimeError("No mapped reads remained after filtering; cannot write bigWig.")

    bw = pyBigWig.open(str(out_bw), "w")
    bw.addHeader(list(chrom_sizes.items()))

    with pysam.AlignmentFile(str(bam), "rb") as bf:
        for chrom, length in chrom_sizes.items():
            cov = np.zeros(length, dtype=np.float32)
            for read in bf.fetch(chrom):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                if read.is_qcfail or read.is_duplicate:
                    continue
                if min_mapping_quality is not None and read.mapping_quality < min_mapping_quality:
                    continue

                blocks = read.get_blocks()
                if not blocks:
                    continue

                if center_reads:
                    mid = (blocks[0][0] + blocks[-1][1]) // 2
                    s = max(0, mid)
                    e = min(length, mid + 1)
                    cov[s:e] += 1.0
                    continue

                if extend_reads is not None and len(blocks) == 1:
                    s, e = blocks[0]
                    if read.is_reverse:
                        s = max(0, e - int(extend_reads))
                    else:
                        e = min(length, s + int(extend_reads))
                    cov[s:e] += 1.0
                    continue

                for s, e in blocks:
                    s = max(0, s)
                    e = min(length, e)
                    if e > s:
                        cov[s:e] += 1.0

            starts = []
            ends = []
            values = []
            for s in range(0, length, int(bin_size)):
                e = min(length, s + int(bin_size))
                val = float(cov[s:e].mean())
                if normalize_using:
                    norm = str(normalize_using).upper()
                    if norm == "CPM":
                        val = val * 1e6 / mapped_reads
                    elif norm == "RPKM":
                        val = val * 1e9 / (mapped_reads * max(1, (e - s)))
                    elif norm == "BPM":
                        val = val * 1e6 / mapped_reads
                if effective_genome_size is not None and normalize_using and str(normalize_using).upper() == "RPGC":
                    val = val * float(effective_genome_size) / mapped_reads
                if scale_factor is not None:
                    val *= float(scale_factor)
                starts.append(s)
                ends.append(e)
                values.append(val)

            bw.addEntries([chrom] * len(starts), starts, ends=ends, values=values)

    bw.close()
    return str(out_bw)
