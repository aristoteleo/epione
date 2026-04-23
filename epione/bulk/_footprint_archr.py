"""ArchR-style footprint backend for bulk ATAC-seq.

The single-cell module :func:`epione.tl.get_footprints` already
implements the ArchR ``getFootprints`` algorithm (hexamer Tn5 bias
subtraction + per-site aggregation + Welford SE ribbons). For bulk
ATAC-seq we just need a thin adapter that:

1. Turns a BAM file into a tabix-indexed fragments.tsv.gz
   (``samtools`` + ``bedtools`` via the existing
   :func:`epione.bulk.bam_to_frags` helper).
2. Builds a one-row synthetic AnnData whose single "cell" is the whole
   bulk sample, so ``groupby='sample'`` aggregates every read.
3. Calls ``epi.tl.get_footprints`` and returns the same
   ``{motif_name: Footprint}`` mapping.

This makes ArchR-style footprinting a drop-in alternative to the
TOBIAS backend (:func:`epione.tl.atacorrect` +
:func:`epione.tl.score_bigwig` + :func:`epione.tl.bindetect`) for
bulk ATAC-seq data — no single-cell machinery required.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..utils import console
from ..utils._genome import Genome


def _run(cmd: list, **kw):
    p = subprocess.run(cmd, **kw)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")


def _ensure_tabix(frags_gz: Path) -> Path:
    """Ensure ``frags_gz`` has a tabix index next to it (bgzip + sort + tabix)."""
    frags_gz = Path(frags_gz)
    tbi = frags_gz.with_suffix(frags_gz.suffix + ".tbi")
    if tbi.exists():
        return tbi
    if shutil.which("tabix") is None:
        raise RuntimeError(
            "tabix not found in PATH — conda install -c bioconda tabix (or htslib)"
        )
    if shutil.which("bgzip") is None:
        raise RuntimeError(
            "bgzip not found in PATH — conda install -c bioconda tabix"
        )

    # tabix requires bgzip'd + sorted input. bam_to_frags already writes
    # bgzip'd; but rows aren't guaranteed sorted after the BEDPE step.
    # Sort + re-bgzip into a sibling file to be safe.
    sorted_gz = frags_gz.with_suffix(".sorted.tsv.gz")
    if not sorted_gz.exists():
        console.level2(f"Sorting + re-bgzipping {frags_gz.name}")
        tmp = frags_gz.with_suffix(".sorted.tsv")
        with tmp.open("w") as fout:
            _run(
                ["bash", "-c",
                 f"zcat '{frags_gz}' | sort -k1,1 -k2,2n"],
                stdout=fout,
            )
        _run(["bgzip", "-f", str(tmp)])
        # Replace original with sorted (keep original suffix expected by caller)
        tmp_gz = tmp.with_suffix(".tsv.gz")
        tmp_gz.replace(sorted_gz)
    # tabix index
    _run(["tabix", "-f", "-p", "bed", str(sorted_gz)])
    # Point the original path at the sorted one (symlink) so caller's
    # path still works, and also expose the .tbi at the original path.
    if frags_gz.exists() or frags_gz.is_symlink():
        frags_gz.unlink()
    frags_gz.symlink_to(sorted_gz.name)
    tbi_orig = frags_gz.with_suffix(frags_gz.suffix + ".tbi")
    if tbi_orig.exists() or tbi_orig.is_symlink():
        tbi_orig.unlink()
    tbi_orig.symlink_to(sorted_gz.name + ".tbi")
    return tbi_orig


def bam_to_fragments_bulk(
    bam_file: Union[str, Path],
    out_frags_gz: Union[str, Path],
    *,
    sample_name: str = "bulk",
) -> Path:
    """Bulk BAM → tabix-indexed fragments.tsv.gz.

    Wraps :func:`epione.bulk.bam_to_frags` and adds the tabix index step
    that :func:`epione.tl.get_footprints` needs.
    """
    from ._fastq2frags import bam_to_frags as _bam_to_frags
    frags_gz = Path(out_frags_gz)
    frags_gz.parent.mkdir(parents=True, exist_ok=True)
    if not frags_gz.exists():
        console.level1(f"BAM → fragments: {bam_file} → {frags_gz}")
        _bam_to_frags(str(bam_file), sample_name=sample_name,
                       out_frags_gz=str(frags_gz))
    _ensure_tabix(frags_gz)
    return frags_gz


def _synthetic_bulk_adata(frags_gz: Path, sample_name: str, genome: Genome):
    """Minimal AnnData that :func:`epi.tl.get_footprints` understands."""
    import anndata as ad
    from scipy import sparse as _sp
    adata = ad.AnnData(
        X=_sp.csr_matrix((1, 0), dtype=np.float32),
        obs=pd.DataFrame({"sample": [sample_name]}, index=[sample_name]),
        uns={
            "files": {"fragments": str(frags_gz)},
            "reference_sequences": dict(genome.chrom_sizes),
        },
    )
    return adata


def footprint_archr(
    bam_file: Union[str, Path],
    *,
    peak_bed: Union[str, Path, pd.DataFrame],
    genome: Union[Genome, str, Path],
    motif_database: Optional[Union[str, Path]] = None,
    motifs: Optional[Sequence[str]] = None,
    positions: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
    sample_name: str = "bulk",
    output_dir: Optional[Union[str, Path]] = None,
    flank: int = 250,
    flank_norm: int = 50,
    normalize: str = "Subtract",
    kmer_length: int = 6,
    bias_table: Optional[np.ndarray] = None,
    smooth: int = 11,
    min_cells_per_group: int = 1,
) -> dict:
    """ArchR-style footprint for a single bulk ATAC-seq BAM file.

    Parameters
    ----------
    bam_file
        Path to the input BAM (should be filtered / dedup'd; e.g. the
        output of :func:`epione.bulk.filter_bam`).
    peak_bed
        BED file of peaks (or a DataFrame with ``chrom/start/end``).
        Motif hits are restricted to peak-overlapping positions — this
        is the ArchR ``getPositions(proj, 'Motif')`` behaviour and the
        single biggest driver of real footprint amplitude.
    genome
        :class:`epione.utils._genome.Genome` object (or raw FASTA
        path). Used for the hexamer Tn5 bias track.
    motif_database
        Optional pre-built motif-hit database from
        :func:`epione.tl.build_motif_database`. With this you get exact
        PWM match coordinates (sharp footprints); without it you fall
        back to peak centres (coarser).
    motifs
        Subset of motif names to scan. Default: all motifs in the
        database. Use short names like ``['GATA1', 'CEBPA']`` — the
        database resolver does substring matching.
    positions
        Alternative to ``motif_database``: pass an explicit
        ``chrom/center/strand`` DataFrame or a
        ``{motif_name: DataFrame}`` mapping.
    sample_name
        Label used as the fragments file's barcode column (default
        ``'bulk'``). Only affects internal bookkeeping.
    output_dir
        Where to write intermediate fragments.tsv.gz (and later cache
        artifacts). Defaults to ``Path(bam_file).parent / 'epione_bulk_fp'``.
    flank, flank_norm, normalize, kmer_length, bias_table, smooth
        Forwarded to :func:`epione.tl.get_footprints` unchanged.
    min_cells_per_group
        Default 1 — we have a single-sample aggregate, so there's
        only one "cell".

    Returns
    -------
    dict
        ``{motif_name: epione.tl._footprint.Footprint}`` — ready to
        pass to :func:`epione.pl.plot_footprints`. Each ``Footprint``
        has a single group (``sample_name``).
    """
    if not isinstance(genome, Genome):
        # Allow passing a FASTA path for convenience; wrap into a
        # Genome so isinstance() in downstream helpers works.
        import pyfaidx
        fa_path = Path(str(genome))
        fa = pyfaidx.Fasta(str(fa_path))
        chrom_sizes = {c: len(fa[c]) for c in fa.keys()}
        # Genome requires annotation; bulk footprint doesn't use it, so
        # point it at the fasta path as a safe stand-in (callers never
        # read it from the Genome in this flow).
        genome = Genome(fasta=fa_path, annotation=fa_path,
                         chrom_sizes=chrom_sizes)

    bam_file = Path(bam_file)
    if output_dir is None:
        output_dir = bam_file.parent / "epione_bulk_fp"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frags_gz = bam_to_fragments_bulk(
        bam_file,
        output_dir / f"{bam_file.stem}.fragments.tsv.gz",
        sample_name=sample_name,
    )

    adata = _synthetic_bulk_adata(frags_gz, sample_name, genome)

    # Peak handling
    peaks_df = None
    if peak_bed is not None:
        if isinstance(peak_bed, pd.DataFrame):
            peaks_df = peak_bed
        else:
            p = Path(peak_bed)
            peaks_df = pd.read_csv(p, sep="\t", header=None,
                                    names=["chrom", "start", "end"],
                                    usecols=[0, 1, 2])

    from ..tl._footprint import get_footprints
    fp = get_footprints(
        adata,
        positions=positions,
        motifs=list(motifs) if motifs is not None else None,
        motif_database=motif_database,
        peaks=peaks_df,
        groupby="sample",
        genome=genome,
        flank=flank,
        flank_norm=flank_norm,
        normalize=normalize,
        kmer_length=kmer_length,
        bias_table=bias_table,
        smooth=smooth,
        min_cells_per_group=min_cells_per_group,
    )
    return fp


__all__ = ["footprint_archr", "bam_to_fragments_bulk"]
