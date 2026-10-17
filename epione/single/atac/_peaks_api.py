"""One entry point for building a cell-type-aware, non-overlapping peak set.

:func:`call_peaks` is the default way to go from fragments to the peak
set that :func:`epione.pp.make_peak_matrix`, motif annotation, chromVAR,
marker peaks and peak-to-gene linkage consume.

``method='cellscan'`` (default since this release)
    :func:`~epione.single.atac.scan_peaks` -- cells-as-replicates scan
    statistic with a Tn5-bias-corrected local background. Per-group peaks
    are merged by score into fixed-width summit-centred peaks.
``method='macs3'`` (previous default, unchanged behaviour)
    :func:`~epione.single.atac.macs3` per group followed by
    :func:`~epione.single.atac.merge_peaks` (ArchR / SnapATAC2-style
    summit +- ``half_width`` iterative overlap merge).

Both return the same table layout (``chrom, start, end, Peaks, group``
plus caller-specific score columns), so downstream code does not need to
know which caller produced it.

Why CellScan is the default
---------------------------
On held-out ENCODE scATAC data (HepG2, MCF-7, A549) with orthogonal
bulk truth (ENCODE bulk ATAC IDR + DNase-seq), the top 50k CellScan
peaks had higher F1 than MACS3-per-group + merge (0.884 vs 0.875,
0.850 vs 0.837, 0.876 vs 0.866) and than ArchR, SnapATAC2, Signac,
HMMRATAC and Genrich, with better split-half reproducibility and
better accuracy at 500-2,000 cells, while running 10-30x faster
(100k cells: 6 min vs 2 h 40 min on 8 threads). Known weaknesses:
CTCF summits are less precise than Genrich / SnapATAC2 / MACS3
(median 18-45 bp vs 8-19 bp from the motif), peaks specific to a rare
group are ranked lower than with ArchR's per-group quantile scores, and
peak memory is 3-4x higher (about 42 GB at 100k cells). Use
``method='macs3'`` to reproduce earlier results or ArchR-parity
analyses.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from epione.core.genome import Genome

__all__ = ["call_peaks"]


def call_peaks(
    adata,
    groupby: Optional[Union[str, Sequence]] = None,
    *,
    method: str = "cellscan",
    genome: Optional[Union[Genome, str, os.PathLike]] = None,
    chrom_sizes: Optional[dict] = None,
    half_width: int = 250,
    blacklist: Optional[Union[str, os.PathLike]] = None,
    key_added: str = "peaks",
    inplace: bool = True,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
) -> Optional[pd.DataFrame]:
    """Call peaks from fragments and return one non-overlapping peak set.

    Parameters
    ----------
    adata
        AnnData with ``uns['files']['fragments']`` (cells = ``obs_names``).
    groupby
        ``obs`` column (or per-cell labels) defining pseudobulk groups /
        cell types; ``None`` uses all cells as one group.
    method
        ``'cellscan'`` (default) or ``'macs3'`` (the previous default:
        :func:`macs3` + :func:`merge_peaks`, identical behaviour).
    genome
        :class:`~epione.core.genome.Genome` (e.g. ``epi.utils.hg38``) or
        a FASTA path. CellScan uses the FASTA for its Tn5 bias model
        (without it, bias correction is skipped); both methods take the
        chromosome sizes from it unless ``chrom_sizes`` is given.
    half_width
        Final peaks are ``[summit - half_width, summit + half_width)``.
    blacklist
        BED of regions to drop (applied by both methods).
    key_added
        The peak table is stored in ``adata.uns[key_added]``.
    **kwargs
        Passed to :func:`scan_peaks` (``method='cellscan'``) or
        :func:`macs3` (``method='macs3'``), e.g. ``qvalue``, ``window``,
        ``n_replicates``.

    Returns
    -------
    ``DataFrame`` with ``chrom, start, end, Peaks`` (``"chr:start-end"``,
    the label :func:`epione.pp.make_peak_matrix` takes as ``use_rep``),
    ``group`` and the caller's score columns, sorted by position; ``None``
    when ``inplace``.
    """
    method = method.lower()
    fasta = None
    if isinstance(genome, Genome):
        if chrom_sizes is None:
            chrom_sizes = genome.chrom_sizes
        if method == "cellscan":
            fasta = str(genome.fasta)
    elif genome is not None:
        fasta = str(genome)
    if chrom_sizes is None and fasta is not None:
        import pysam
        with pysam.FastaFile(fasta) as fa:
            chrom_sizes = dict(zip(fa.references, fa.lengths))
    if chrom_sizes is None:
        raise ValueError("pass genome= (Genome or FASTA) or chrom_sizes=")
    chrom_sizes = dict(chrom_sizes)

    if method == "cellscan":
        from ._scan_peaks import scan_peaks
        out = scan_peaks(adata, groupby=groupby, fasta=fasta, chrom_sizes=chrom_sizes,
                         half_width=half_width, blacklist=blacklist, n_jobs=n_jobs,
                         inplace=False, verbose=verbose, **kwargs)
        out = out.copy()
        out.insert(3, "Peaks", out["chrom"].astype(str) + ":" + out["start"].astype(str)
                   + "-" + out["end"].astype(str))
    elif method == "macs3":
        from ._call_peaks import macs3, merge_peaks
        res = macs3(adata, groupby=groupby, blacklist=blacklist, inplace=False,
                    verbose=verbose, **kwargs)
        per_group = res if isinstance(res, dict) else {"all": res}
        out = merge_peaks(per_group, chrom_sizes, half_width=half_width)
    else:
        raise ValueError("method must be 'cellscan' or 'macs3'")
    out = out.reset_index(drop=True)
    if inplace:
        adata.uns[key_added] = out
        return None
    return out
