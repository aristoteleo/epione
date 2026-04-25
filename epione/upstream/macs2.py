from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from ._common import resolve_tool, run


def call_peaks_macs2(
    bam: Union[str, Path],
    out_dir: Union[str, Path],
    name: str,
    *,
    control_bam: Optional[Union[str, Path]] = None,
    genome_size: str = "hs",
    format: str = "BAMPE",
    qvalue: float = 0.01,
    keep_dup: str = "all",
    call_summits: bool = True,
    nomodel: bool = True,
    shift: Optional[int] = None,
    extsize: Optional[int] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    macs2 = resolve_tool("macs2")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        macs2, "callpeak", "-t", str(bam), "-f", format, "-g", genome_size,
        "--keep-dup", keep_dup, "-q", str(qvalue), "-n", name, "--outdir", str(out_dir),
    ]
    if control_bam is not None:
        cmd += ["-c", str(control_bam)]
    if nomodel:
        cmd.append("--nomodel")
    if call_summits:
        cmd.append("--call-summits")
    if shift is not None:
        cmd += ["--shift", str(shift)]
    if extsize is not None:
        cmd += ["--extsize", str(extsize)]
    if extra_args:
        cmd += list(map(str, extra_args))
    run(cmd)

    prefix = out_dir / name
    return {
        "narrowPeak": str(prefix.with_name(prefix.name + "_peaks.narrowPeak")),
        "summits": str(prefix.with_name(prefix.name + "_summits.bed")),
        "xls": str(prefix.with_name(prefix.name + "_peaks.xls")),
    }
