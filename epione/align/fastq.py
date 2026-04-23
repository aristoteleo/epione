from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from ._common import CommandError, resolve_tool, run


def trim_fastq_pair(
    fq1: Union[str, Path],
    fq2: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    sample_name: Optional[str] = None,
    method: str = "fastp",
    threads: int = 8,
    html_report: Optional[Union[str, Path]] = None,
    json_report: Optional[Union[str, Path]] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> Tuple[str, str]:
    fq1 = str(fq1)
    fq2 = str(fq2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_name = sample_name or Path(fq1).name.split("_R1")[0].split("_1")[0]

    if method == "fastp":
        fastp = resolve_tool("fastp")
        out1 = out_dir / f"{sample_name}_R1.trim.fastq.gz"
        out2 = out_dir / f"{sample_name}_R2.trim.fastq.gz"
        html_report = Path(html_report) if html_report is not None else out_dir / f"{sample_name}.fastp.html"
        json_report = Path(json_report) if json_report is not None else out_dir / f"{sample_name}.fastp.json"
        cmd = [
            fastp, "-i", fq1, "-I", fq2, "-o", str(out1), "-O", str(out2),
            "-w", str(threads), "-h", str(html_report), "-j", str(json_report),
        ]
        if extra_args:
            cmd.extend(map(str, extra_args))
        run(cmd)
        return str(out1), str(out2)

    if method == "trim_galore":
        trim_galore = resolve_tool("trim_galore")
        cmd = [
            trim_galore, "--paired", "--gzip", "--cores", str(max(1, threads)),
            "-o", str(out_dir), fq1, fq2,
        ]
        if extra_args:
            cmd.extend(map(str, extra_args))
        run(cmd)
        r1_val = out_dir / (Path(fq1).name.replace(".fastq.gz", "_val_1.fq.gz").replace(".fq.gz", "_val_1.fq.gz"))
        r2_val = out_dir / (Path(fq2).name.replace(".fastq.gz", "_val_2.fq.gz").replace(".fq.gz", "_val_2.fq.gz"))
        if not r1_val.exists() or not r2_val.exists():
            raise CommandError("trim_galore finished but expected output FASTQs were not found.")
        return str(r1_val), str(r2_val)

    raise ValueError("method must be 'fastp' or 'trim_galore'")
