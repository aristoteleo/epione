"""FASTQ-to-cool Hi-C matrix construction via pairtools + cooler.

The canonical Hi-C upstream looks like::

    FASTQ (paired R1+R2, mapped independently with bowtie2 --local)
      → BAM          (name-sorted, merged R1+R2)
      → pairs.gz     (pairtools parse → sort → dedup → select UU|UR|RU)
      → .cool        (cooler cload pairs at binsize)
      → balanced     (cooler balance, in place)

This module covers the BAM → .cool half; the BAM is produced via
:mod:`epione.upstream.bowtie2`.

We shell out to the ``pairtools`` and ``cooler`` CLIs rather than use
their Python APIs directly because each stage streams multi-gigabyte
records and the CLI chain (``parse | sort | dedup | select``) is the
tested happy path in the cooler / pairtools docs. Every invocation
goes through :func:`epione.upstream._env.build_env` so subprocesses see
the active conda env's ``bin/`` regardless of how the caller's PATH
was set up.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Union

from ._env import build_env, resolve_executable


HIC_TOOLS: Sequence[str] = (
    "bowtie2",      # upstream mapping (shared with ATAC)
    "samtools",     # sort / merge the BAMs
    "pairtools",    # parse / sort / dedup / select pairs
    "cooler",       # load pairs into .cool, balance, dump
)


def _run(cmd, check=True, **kwargs):
    """Tiny wrapper so every shelled-out tool inherits the active
    env bin on PATH via :func:`build_env`."""
    env = kwargs.pop("env", build_env())
    return subprocess.run(cmd, env=env, check=check, **kwargs)


def pairs_from_bam(
    bam_path: Union[str, Path],
    chrom_sizes: Union[str, Path],
    out_pairs: Union[str, Path],
    *,
    min_mapq: int = 30,
    nproc: int = 4,
    assembly: str = "custom",
    select_expr: str = "(pair_type=='UU') or (pair_type=='UR') or (pair_type=='RU')",
    overwrite: bool = False,
) -> Path:
    """Turn a Hi-C BAM into a bgzip'd, indexed ``.pairs.gz`` file.

    Runs the canonical pairtools chain — ``parse → sort → dedup →
    select`` — streaming between stages via pipes to keep disk use
    bounded.

    Arguments:
        bam_path: input Hi-C BAM. Forward and reverse reads must have
            been mapped independently (typically with ``bowtie2 --local``)
            and then *name-sorted* merged — pairtools pairs R1 and R2 by
            read name.
        chrom_sizes: 2-column TSV of chromosome sizes. Pass the same
            file used by :mod:`epione.upstream.reference`.
        out_pairs: destination ``.pairs.gz`` path. The companion ``.px2``
            tabix index and ``.stats`` file are written alongside.
        min_mapq: minimum MAPQ filter applied during ``pairtools parse``.
            Default 30 matches the 4DN Hi-C spec.
        nproc: parallel threads for the parse / sort stages.
        assembly: assembly tag written into the pairs header (metadata
            only; use a human-readable string like ``'hg38'``).
        select_expr: pairtools filter expression kept after dedup. Default
            retains unique-unique, unique-rescued and rescued-unique pairs
            (4DN's recommended downstream set).
        overwrite: re-run even if ``out_pairs`` already exists.

    Returns:
        ``Path`` to the output ``.pairs.gz`` file.
    """
    bam_path = Path(bam_path)
    chrom_sizes = Path(chrom_sizes)
    out_pairs = Path(out_pairs)
    out_pairs.parent.mkdir(parents=True, exist_ok=True)

    if out_pairs.exists() and not overwrite:
        return out_pairs

    pairtools = resolve_executable("pairtools")
    tmp_dir = out_pairs.parent

    parse_cmd = [
        pairtools, "parse",
        "-c", str(chrom_sizes),
        "--assembly", assembly,
        "--drop-sam",
        "--drop-seq",
        "--min-mapq", str(min_mapq),
        "--output-stats", str(out_pairs.with_suffix(".parse.stats")),
        "--nproc-in", str(nproc),
        "--nproc-out", str(nproc),
        str(bam_path),
    ]
    sort_cmd = [
        pairtools, "sort",
        "--nproc", str(nproc),
        "--tmpdir", str(tmp_dir),
    ]
    dedup_cmd = [
        pairtools, "dedup",
        "--mark-dups",
        "--output-stats", str(out_pairs.with_suffix(".dedup.stats")),
        "--nproc-in", str(nproc),
        "--nproc-out", str(nproc),
    ]
    select_cmd = [
        pairtools, "select", select_expr,
        "-o", str(out_pairs),
    ]

    env = build_env()
    p_parse = subprocess.Popen(parse_cmd, stdout=subprocess.PIPE, env=env)
    p_sort  = subprocess.Popen(sort_cmd, stdin=p_parse.stdout,
                                stdout=subprocess.PIPE, env=env)
    p_parse.stdout.close()
    p_dedup = subprocess.Popen(dedup_cmd, stdin=p_sort.stdout,
                                stdout=subprocess.PIPE, env=env)
    p_sort.stdout.close()
    p_select = subprocess.Popen(select_cmd, stdin=p_dedup.stdout, env=env)
    p_dedup.stdout.close()

    rc = p_select.wait()
    # Also drain the earlier processes so errors surface.
    for p, name in ((p_parse, "parse"), (p_sort, "sort"),
                    (p_dedup, "dedup"), (p_select, "select")):
        if p.wait() != 0:
            raise RuntimeError(f"pairtools {name} failed (rc={p.returncode})")
    if rc != 0:
        raise RuntimeError(f"pairtools pipeline failed (rc={rc})")
    return out_pairs


def pairs_to_cool(
    pairs_path: Union[str, Path],
    chrom_sizes: Union[str, Path],
    out_cool: Union[str, Path],
    *,
    binsize: int = 10_000,
    assembly: Optional[str] = None,
    nproc: int = 4,
    overwrite: bool = False,
) -> Path:
    """Load a ``.pairs.gz`` file into a ``.cool`` contact matrix.

    Thin wrapper around ``cooler cload pairs`` — chosen over
    ``cooler.create_cooler()`` because the CLI streams pairs without
    materialising the whole sparse matrix in memory, which matters at
    kb-scale resolution on mammalian genomes.

    Arguments:
        pairs_path: input pairs from :func:`pairs_from_bam`.
        chrom_sizes: 2-column TSV of chromosome sizes (same one passed
            to :func:`pairs_from_bam`).
        out_cool: destination ``.cool`` path.
        binsize: genomic bin size in bp. 10 kb is the canonical
            "intra-chromosome TADs / compartments" resolution.
            Use 1 kb for loops, 50–100 kb for low-depth libraries.
        assembly: assembly tag written to the cool metadata.
        nproc: parallel threads for the bgzip read.
        overwrite: re-run even if ``out_cool`` already exists.

    Returns:
        ``Path`` to the output ``.cool``.
    """
    pairs_path = Path(pairs_path)
    chrom_sizes = Path(chrom_sizes)
    out_cool = Path(out_cool)
    out_cool.parent.mkdir(parents=True, exist_ok=True)

    if out_cool.exists() and not overwrite:
        return out_cool

    cooler_exe = resolve_executable("cooler")
    cmd = [
        cooler_exe, "cload", "pairs",
        "-c1", "2", "-p1", "3", "-c2", "4", "-p2", "5",
    ]
    if assembly:
        cmd.extend(["--assembly", assembly])
    cmd.extend([f"{chrom_sizes}:{binsize}", str(pairs_path), str(out_cool)])

    _run(cmd)
    return out_cool
