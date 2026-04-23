from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence, Union

from ._common import CommandError, resolve_tool, run


def sort_bam(
    in_bam: Union[str, Path],
    out_bam: Union[str, Path],
    *,
    threads: int = 4,
    name_sort: bool = False,
) -> str:
    samtools = resolve_tool("samtools")
    cmd = [samtools, "sort"]
    if name_sort:
        cmd.append("-n")
    cmd += ["-@", str(threads), "-o", str(out_bam), str(in_bam)]
    run(cmd)
    return str(out_bam)


def index_bam(bam: Union[str, Path], *, threads: int = 4) -> str:
    samtools = resolve_tool("samtools")
    run([samtools, "index", "-@", str(threads), str(bam)])
    return str(bam) + ".bai"


def merge_bams(
    bam_files: Sequence[Union[str, Path]],
    out_bam: Union[str, Path],
    *,
    threads: int = 4,
    index: bool = True,
) -> str:
    if len(bam_files) == 0:
        raise ValueError("bam_files cannot be empty.")
    samtools = resolve_tool("samtools")
    out_bam = Path(out_bam)
    out_bam.parent.mkdir(parents=True, exist_ok=True)
    run([samtools, "merge", "-@", str(threads), "-o", str(out_bam), *map(str, bam_files)])
    if index:
        index_bam(out_bam, threads=threads)
    return str(out_bam)


def filter_bam(
    in_bam: Union[str, Path],
    out_bam: Union[str, Path],
    *,
    mapq: int = 30,
    proper_pair: bool = True,
    drop_secondary_supp: bool = True,
    drop_duplicates: bool = False,
    drop_qcfail: bool = False,
    drop_unmapped: bool = True,
    drop_mate_unmapped: bool = True,
    drop_chroms: Optional[Sequence[str]] = None,
    threads: int = 4,
) -> str:
    samtools = resolve_tool("samtools")
    out_bam = str(out_bam)

    include_flag = 0x2 if proper_pair else 0
    exclude_flag = 0
    if drop_secondary_supp:
        exclude_flag |= 0x100 | 0x800
    if drop_duplicates:
        exclude_flag |= 0x400
    if drop_qcfail:
        exclude_flag |= 0x200
    if drop_unmapped:
        exclude_flag |= 0x4
    if drop_mate_unmapped:
        exclude_flag |= 0x8

    if not drop_chroms:
        cmd = [samtools, "view", "-b"]
        if include_flag:
            cmd += ["-f", hex(include_flag)]
        if exclude_flag:
            cmd += ["-F", hex(exclude_flag)]
        cmd += ["-q", str(mapq), str(in_bam)]
        sort_cmd = [samtools, "sort", "-@", str(threads), "-o", out_bam, "-"]
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p2 = subprocess.run(sort_cmd, stdin=p1.stdout)
        if p1.stdout is not None:
            p1.stdout.close()
        p1.wait()
        if p1.returncode != 0 or p2.returncode != 0:
            raise CommandError("BAM filtering pipeline failed.")
        return out_bam

    drop_chroms = set(drop_chroms)
    cmd = [samtools, "view", "-h"]
    if include_flag:
        cmd += ["-f", hex(include_flag)]
    if exclude_flag:
        cmd += ["-F", hex(exclude_flag)]
    cmd += ["-q", str(mapq), str(in_bam)]

    awk_cmd = ["awk", "-v", "drop=" + " ".join(sorted(drop_chroms))]
    awk_cmd.append(
        'BEGIN{n=split(drop, a, " "); for(i=1;i<=n;i++) bad[a[i]]=1} '
        '/^@/ {print; next} '
        '{if (!($3 in bad)) print}'
    )

    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(awk_cmd, stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.run([samtools, "view", "-b", "-"], stdin=p2.stdout, stdout=subprocess.PIPE)
    p4 = subprocess.run([samtools, "sort", "-@", str(threads), "-o", out_bam, "-"], input=p3.stdout)
    if p1.stdout is not None:
        p1.stdout.close()
    if p2.stdout is not None:
        p2.stdout.close()
    p1.wait()
    p2.wait()
    if any(p.returncode != 0 for p in (p1, p2, p3, p4)):
        raise CommandError("BAM filtering pipeline with chromosome exclusion failed.")
    return out_bam
