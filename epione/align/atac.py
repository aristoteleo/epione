from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from .samtools import index_bam


def shift_atac_bam(
    in_bam: Union[str, Path],
    out_bam: Union[str, Path],
    *,
    threads: int = 8,
    index: bool = True,
) -> str:
    import pysam

    out_bam = Path(out_bam)
    out_bam.parent.mkdir(parents=True, exist_ok=True)
    name_sorted = out_bam.with_suffix(".qname.tmp.bam")
    tmp_bam = out_bam.with_suffix(".shift.unsorted.bam")
    pysam.sort("-n", "-@", str(threads), "-o", str(name_sorted), str(in_bam))

    def _shift_one(read):
        if read.is_unmapped:
            return read
        shifted = read.__copy__()
        offset = 4 if not shifted.is_reverse else -5
        shifted.reference_start = max(0, shifted.reference_start + offset)
        return shifted

    def _sync_pair(read1, read2):
        if read1.is_unmapped or read2.is_unmapped or read1.reference_id != read2.reference_id:
            return
        read1.next_reference_id = read2.reference_id
        read2.next_reference_id = read1.reference_id
        read1.next_reference_start = read2.reference_start
        read2.next_reference_start = read1.reference_start
        left, right = (read1, read2) if read1.reference_start <= read2.reference_start else (read2, read1)
        left_end = left.reference_end or left.reference_start
        right_end = right.reference_end or right.reference_start
        template_len = max(left_end, right_end) - min(left.reference_start, right.reference_start)
        if template_len < 0:
            template_len = 0
        if left is read1:
            read1.template_length = template_len
            read2.template_length = -template_len
        else:
            read2.template_length = template_len
            read1.template_length = -template_len

    with pysam.AlignmentFile(str(name_sorted), "rb") as src, pysam.AlignmentFile(str(tmp_bam), "wb", template=src) as dst:
        pending = {}
        for read in src.fetch(until_eof=True):
            shifted = _shift_one(read)
            qname = shifted.query_name
            mate = pending.pop(qname, None)
            if mate is None:
                pending[qname] = shifted
                continue
            _sync_pair(mate, shifted)
            dst.write(mate)
            dst.write(shifted)
        for leftover in pending.values():
            dst.write(leftover)

    pysam.sort("-@", str(threads), "-o", str(out_bam), str(tmp_bam))
    try:
        os.remove(tmp_bam)
    except OSError:
        pass
    try:
        os.remove(name_sorted)
    except OSError:
        pass

    if index:
        index_bam(out_bam, threads=threads)
    return str(out_bam)
