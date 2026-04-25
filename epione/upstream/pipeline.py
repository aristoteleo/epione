from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import subprocess

from ._common import CommandError, remove_if_exists, resolve_tool


def bam_to_frags(
    bam: Union[str, Path],
    sample_name: str,
    out_frags_gz: Union[str, Path],
    *,
    bedtools_path: Optional[str] = None,
) -> str:
    bedtools = resolve_tool("bedtools", explicit=bedtools_path)
    samtools = resolve_tool("samtools")
    use_bgzip = shutil.which("bgzip") is not None

    bam = str(bam)
    out_frags_gz = str(out_frags_gz)
    stem = str(Path(out_frags_gz).with_suffix("").with_suffix(""))
    tmp_name_bam = stem + ".namesort.bam"
    bedpe = stem + ".bedpe"

    subprocess.run([samtools, "sort", "-n", "-@", str(os.cpu_count() or 2), "-o", tmp_name_bam, bam], check=True)
    with open(bedpe, "w") as fh:
        proc = subprocess.run(
            [bedtools, "bamtobed", "-bedpe", "-i", tmp_name_bam],
            stdout=fh,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise CommandError(f"bedtools bamtobed failed.\nstderr:\n{proc.stderr}")

    awk_script = (
        f'BEGIN{{OFS="\\t"}} $1==$4 {{s=($2<$5?$2:$5); e=($3>$6?$3:$6); '
        f'if(e>s) print $1,s,e,"{sample_name}"}}'
    )
    compressor = ["bgzip", "-c"] if use_bgzip else [resolve_tool("gzip"), "-c"]
    with open(out_frags_gz, "wb") as fout:
        pawk = subprocess.Popen(["awk", awk_script, bedpe], stdout=subprocess.PIPE)
        pgz = subprocess.Popen(compressor, stdin=pawk.stdout, stdout=fout)
        if pawk.stdout is not None:
            pawk.stdout.close()
        pgz.communicate()
        pawk.wait()
        if pawk.returncode != 0 or pgz.returncode != 0:
            raise CommandError("Fragment compression pipeline failed.")

    remove_if_exists(tmp_name_bam, bedpe)
    return out_frags_gz
def _epione_datasets():
    try:
        from epione.core.genome import register_datasets
    except Exception as e:
        raise RuntimeError("epione.utils._genome.register_datasets not available") from e
    return register_datasets()


def list_dataset_fastqs_tar() -> List[str]:
    ds = _epione_datasets()
    tar_path = Path(ds.fetch("atac_pbmc_500_fastqs.tar"))
    with tarfile.open(tar_path, "r") as tf:
        return [m.name for m in tf.getmembers() if m.isfile() and m.name.endswith((".fastq.gz", ".fq.gz"))]


def extract_fastqs_from_tar(members: Sequence[str], dest: Union[str, Path]) -> List[Path]:
    ds = _epione_datasets()
    tar_path = Path(ds.fetch("atac_pbmc_500_fastqs.tar"))
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []
    with tarfile.open(tar_path, "r") as tf:
        wanted = {m for m in members}
        for m in tf.getmembers():
            if m.isfile() and m.name in wanted:
                tf.extract(m, path=dest)
                out_paths.append(dest / m.name)
    return out_paths


def fetch_dataset_fastq_pairs(genome: str = "GRCh38", dest: Union[str, Path] = "./data") -> Dict[str, Tuple[Path, Path]]:
    ds = _epione_datasets()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = list_dataset_fastqs_tar()
    members = [f for f in files if f.startswith(f"{genome}/")]
    extracted = extract_fastqs_from_tar(members, dest)
    by_sample: Dict[str, Dict[str, Path]] = {}
    for p in extracted:
        name = p.name
        if "_R1" in name:
            sample = name.split("_R1")[0]
            by_sample.setdefault(sample, {})["R1"] = p
        elif "_R2" in name:
            sample = name.split("_R2")[0]
            by_sample.setdefault(sample, {})["R2"] = p
    out: Dict[str, Tuple[Path, Path]] = {}
    for sample, d in by_sample.items():
        if "R1" in d and "R2" in d:
            out[sample] = (d["R1"], d["R2"])
    return out
