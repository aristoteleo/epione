from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

from ._common import resolve_tool, run


def ensure_fasta_unzipped(fa_gz: Union[str, Path]) -> Path:
    fa_gz = Path(fa_gz)
    if fa_gz.suffix != ".gz":
        return fa_gz
    out = fa_gz.with_suffix("")
    if out.exists():
        return out
    with gzip.open(fa_gz, "rb") as fin, open(out, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return out


def ensure_fasta_index(fasta: Union[str, Path]) -> Path:
    fasta = Path(fasta)
    fai = fasta.with_suffix(fasta.suffix + ".fai")
    if fai.exists():
        return fai
    samtools = resolve_tool("samtools")
    run([samtools, "faidx", str(fasta)])
    return fai


def ensure_chrom_sizes(
    fasta: Optional[Union[str, Path]] = None,
    chrom_sizes: Optional[Union[str, Path]] = None,
) -> Path:
    if chrom_sizes is not None:
        chrom_sizes = Path(chrom_sizes)
        if chrom_sizes.exists():
            return chrom_sizes

    if fasta is None:
        raise ValueError("Provide either chrom_sizes or fasta.")

    fasta = Path(fasta)
    fai = ensure_fasta_index(fasta)
    out = Path(chrom_sizes) if chrom_sizes is not None else fasta.with_suffix(".chrom.sizes")
    with open(fai) as fin, open(out, "w") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            fout.write(f"{parts[0]}\t{parts[1]}\n")
    return out


def ensure_aligner_index(
    aligner: str,
    fasta: Union[str, Path],
    index_prefix: Union[str, Path, None] = None,
    overwrite: bool = False,
) -> str:
    fasta = Path(fasta)
    if aligner == "bwa-mem2":
        bwa_mem2 = resolve_tool("bwa-mem2")
        if index_prefix is None:
            prefix_path = fasta
        else:
            prefix_path = Path(index_prefix)
            prefix_path.parent.mkdir(parents=True, exist_ok=True)
        idx_candidates = [Path(str(prefix_path) + ext) for ext in [".0123", ".amb", ".ann", ".bwt"]]
        if overwrite or not all(p.exists() for p in idx_candidates):
            cmd = [bwa_mem2, "index"]
            if index_prefix is not None:
                cmd += ["-p", str(prefix_path)]
            cmd += [str(fasta)]
            run(cmd)
        return str(prefix_path)

    if aligner == "bowtie2":
        bowtie2_build = resolve_tool("bowtie2-build")
        if index_prefix is None:
            prefix = str(Path(fasta).with_suffix(""))
        else:
            prefix = str(index_prefix)
            Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        bt2_files = [
            Path(prefix + ext)
            for ext in [".1.bt2", ".2.bt2", ".3.bt2", ".4.bt2", ".rev.1.bt2", ".rev.2.bt2"]
        ]
        if overwrite or not all(p.exists() for p in bt2_files):
            run([bowtie2_build, str(fasta), prefix])
        return prefix

    raise ValueError("aligner must be 'bwa-mem2' or 'bowtie2'")


def prepare_reference(
    *,
    genome: Optional[str] = None,
    fasta: Optional[Union[str, Path]] = None,
    chrom_sizes: Optional[Union[str, Path]] = None,
    aligner: str = "bowtie2",
    index_prefix: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> Dict[str, str]:
    if fasta is None:
        if genome is None:
            raise ValueError("Provide either fasta or genome.")
        fasta = fetch_genome_fasta(genome)
    fasta = ensure_fasta_unzipped(fasta)
    fai = ensure_fasta_index(fasta)
    chrom_sizes = ensure_chrom_sizes(fasta=fasta, chrom_sizes=chrom_sizes)
    ref_index = ensure_aligner_index(
        aligner=aligner,
        fasta=fasta,
        index_prefix=index_prefix,
        overwrite=overwrite,
    )
    return {
        "fasta": str(fasta),
        "fai": str(fai),
        "chrom_sizes": str(chrom_sizes),
        "ref_index": str(ref_index),
    }


def fetch_genome_fasta(genome: str) -> Path:
    try:
        from epione.core.genome import register_datasets
    except Exception as e:
        raise RuntimeError("epione.core.genome.register_datasets not available") from e
    ds = register_datasets()
    key_map = {
        "GRCh37": "gencode_v41_GRCh37.fa.gz",
        "GRCh38": "gencode_v41_GRCh38.fa.gz",
        "GRCm38": "gencode_vM25_GRCm38.fa.gz",
        "GRCm39": "gencode_vM30_GRCm39.fa.gz",
    }
    if genome not in key_map:
        raise ValueError(f"Unsupported genome: {genome}")
    fa_gz = Path(ds.fetch(key_map[genome]))
    return ensure_fasta_unzipped(fa_gz)


def fetch_genome_annotation(genome: str) -> Path:
    try:
        from epione.core.genome import register_datasets
    except Exception as e:
        raise RuntimeError("epione.core.genome.register_datasets not available") from e
    ds = register_datasets()
    key_map = {
        "GRCh37": "gencode_v41_GRCh37.gff3.gz",
        "GRCh38": "gencode_v41_GRCh38.gff3.gz",
        "GRCm38": "gencode_vM25_GRCm38.gff3.gz",
        "GRCm39": "gencode_vM30_GRCm39.gff3.gz",
    }
    if genome not in key_map:
        raise ValueError(f"Unsupported genome: {genome}")
    return Path(ds.fetch(key_map[genome]))
