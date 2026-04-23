from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

from ._common import remove_if_exists, resolve_tool, run, run_pipe


def align_fastq_to_bam(
    fq1: Union[str, Path],
    fq2: Optional[Union[str, Path]],
    out_bam: Union[str, Path],
    *,
    ref_index: Union[str, Path],
    threads: int = 8,
    rg: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    remove_duplicates: bool = False,
) -> str:
    out_bam = Path(out_bam)
    out_bam.parent.mkdir(parents=True, exist_ok=True)
    out_prefix = str(out_bam.with_suffix("")).rstrip(".")
    tmp_dir = out_bam.parent
    samtools = resolve_tool("samtools")
    aligner_bin = resolve_tool("bowtie2")

    if fq2 is None:
        coord_sorted = str(tmp_dir / (Path(out_prefix).name + ".coordsort.bam"))
        cmd = [aligner_bin, "-x", str(ref_index), "-U", str(fq1), "-p", str(threads)]
        if extra_args:
            cmd.extend(map(str, extra_args))
        run_pipe([
            cmd,
            [samtools, "view", "-b", "-"],
            [samtools, "sort", "-@", str(threads), "-o", coord_sorted, "-"],
        ])
        md_cmd = [samtools, "markdup", "-@", str(threads)]
        if remove_duplicates:
            md_cmd.append("-r")
        md_cmd += [coord_sorted, str(out_bam)]
        run(md_cmd)
        remove_if_exists(coord_sorted)
        return str(out_bam)

    name_sorted = str(tmp_dir / (Path(out_prefix).name + ".namesort.bam"))
    fixmate_bam = str(tmp_dir / (Path(out_prefix).name + ".fixmate.bam"))
    coord_sorted = str(tmp_dir / (Path(out_prefix).name + ".coordsort.bam"))

    cmd = [
        aligner_bin, "-x", str(ref_index), "-1", str(fq1), "-2", str(fq2), "-p", str(threads),
    ]
    if extra_args:
        cmd.extend(map(str, extra_args))

    run_pipe([
        cmd,
        [samtools, "view", "-b", "-"],
        [samtools, "sort", "-n", "-@", str(threads), "-o", name_sorted, "-"],
    ])
    run([samtools, "fixmate", "-m", name_sorted, fixmate_bam])
    run([samtools, "sort", "-@", str(threads), "-o", coord_sorted, fixmate_bam])
    md_cmd = [samtools, "markdup", "-@", str(threads)]
    if remove_duplicates:
        md_cmd.append("-r")
    md_cmd += [coord_sorted, str(out_bam)]
    run(md_cmd)
    remove_if_exists(name_sorted, fixmate_bam, coord_sorted)
    return str(out_bam)
