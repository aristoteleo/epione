"""MACS3 peak calling + non-overlapping peak merging — no snapATAC2.

``epi.single.macs3`` runs MACS3 (via the standalone ``macs3`` PyPI
package, either its python API or a subprocess) on fragments either
globally or per-cluster. ``epi.single.merge_peaks`` reconciles the
per-group peaks into a single non-overlapping peak set (ArchR-style
summit ± half_width expansion with iterative overlap resolution).

Neither function depends on snapATAC2 — the fragment source is the
BED file recorded in ``adata.uns['files']['fragments']`` (as produced
by :func:`epi.pp.import_fragments`), and MACS3 is called as a
normal Python package.
"""
from __future__ import annotations

import gzip
import logging
import os
import shutil
import subprocess
import tempfile
from math import log
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from ..utils.genome import Genome
from ..utils import console


# ---------------------------------------------------------------------------
# Fragment → BED export per group
# ---------------------------------------------------------------------------

def _iter_fragments(path: str):
    """Yield ``(chrom, start, end, barcode, count)`` from a bgzipped or
    plain BED. No filtering.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                start = int(parts[1]); end = int(parts[2])
            except ValueError:
                continue
            cnt = int(parts[4]) if len(parts) >= 5 and parts[4].isdigit() else 1
            yield parts[0], start, end, parts[3], cnt


def _export_group_beds(
    fragment_file: str,
    group_assignments: dict,       # barcode → group
    outdir: str,
) -> dict:
    """Write one BED per group. Returns ``{group: path}``. Each line is
    ``chrom\\tstart\\tend`` — the format MACS3 callpeak -f BED expects.
    """
    handles = {}
    try:
        for chrom, start, end, bc, cnt in _iter_fragments(fragment_file):
            grp = group_assignments.get(bc)
            if grp is None:
                continue
            fh = handles.get(grp)
            if fh is None:
                fh = open(os.path.join(outdir, f"{grp}.bed"), "w")
                handles[grp] = fh
            # MACS3 BED format: chrom \t start \t end (tab-separated)
            fh.write(f"{chrom}\t{start}\t{end}\n")
    finally:
        for fh in handles.values():
            fh.close()
    return {g: os.path.join(outdir, f"{g}.bed") for g in handles}


# ---------------------------------------------------------------------------
# MACS3 runner
# ---------------------------------------------------------------------------

def _run_macs3(
    bed_path: str,
    outdir: str,
    name: str,
    qvalue: float,
    shift: int,
    extsize: int,
    nolambda: bool,
    call_broad_peaks: bool,
    broad_cutoff: float,
    gsize: Union[str, int] = "hs",
) -> pd.DataFrame:
    """Run ``macs3 callpeak`` as a subprocess and parse the output
    ``<name>_peaks.narrowPeak`` (or ``_broadPeak`` in broad mode)
    into a pandas DataFrame with ``chrom / start / end / name / score / …``.
    """
    cmd = [
        "macs3", "callpeak",
        "-t", bed_path,
        "-f", "BED",
        "-n", name,
        "--outdir", outdir,
        "--nomodel",
        "--shift", str(shift),
        "--extsize", str(extsize),
        "--qvalue", str(qvalue),
        "--keep-dup", "all",
        "-g", str(gsize),
    ]
    if nolambda:
        cmd.append("--nolambda")
    if call_broad_peaks:
        cmd += ["--broad", "--broad-cutoff", str(broad_cutoff)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"MACS3 callpeak failed for {name}:\n{res.stderr[-2000:]}"
        )

    peak_file = os.path.join(
        outdir,
        f"{name}_peaks.{'broadPeak' if call_broad_peaks else 'narrowPeak'}",
    )
    if not os.path.exists(peak_file):
        return pd.DataFrame(columns=["chrom", "start", "end", "name", "score",
                                     "strand", "signalValue", "pValue", "qValue", "peak"])
    cols = ["chrom", "start", "end", "name", "score", "strand",
            "signalValue", "pValue", "qValue"]
    if not call_broad_peaks:
        cols.append("peak")                              # summit offset
    df = pd.read_csv(peak_file, sep="\t", header=None, names=cols, comment="#")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def macs3(
    adata: AnnData,
    *,
    groupby: Optional[Union[str, list]] = None,
    qvalue: float = 0.05,
    call_broad_peaks: bool = False,
    broad_cutoff: float = 0.1,
    max_frag_size: Optional[int] = None,                 # unused (kept for API)
    selections: Optional[set] = None,
    nolambda: bool = False,
    shift: int = -100,
    extsize: int = 200,
    min_len: Optional[int] = None,                       # unused (kept for API)
    blacklist: Optional[Path] = None,
    key_added: str = "macs3",
    tempdir: Optional[Path] = None,
    inplace: bool = True,
    gsize: Union[str, int] = "hs",
    verbose: bool = True,
) -> Optional[dict]:
    """Run MACS3 per group on fragments recorded in
    ``adata.uns['files']['fragments']``.

    ``groupby=None`` — call peaks on all cells combined.
    ``groupby='leiden'`` — call peaks per cluster (one BED per Leiden
    cluster, one MACS3 invocation per BED, one DataFrame per cluster).

    ``gsize`` — MACS3 effective genome size. 'hs' / 'mm' accepted as
    shorthand; or pass an integer bp. Default 'hs' (human).

    The merged result (or the per-cluster dict) is written to
    ``adata.uns[key_added + '_pseudobulk']``.
    """
    frag_file = adata.uns.get("files", {}).get("fragments")
    if frag_file is None:
        raise ValueError("adata.uns['files']['fragments'] missing — "
                         "run epi.pp.import_fragments first")

    # Group assignment per barcode.
    bc_to_group: dict = {}
    if groupby is None:
        for bc in adata.obs_names:
            bc_to_group[bc] = "all"
    elif isinstance(groupby, str):
        if groupby not in adata.obs:
            raise KeyError(f"groupby={groupby!r} not in adata.obs")
        for bc, g in zip(adata.obs_names, adata.obs[groupby]):
            bc_to_group[bc] = str(g)
    else:
        for bc, g in zip(adata.obs_names, groupby):
            bc_to_group[bc] = str(g)

    if selections is not None:
        sel = set(map(str, selections))
        bc_to_group = {bc: g for bc, g in bc_to_group.items() if g in sel}

    console.level1(f"MACS3 peak calling: {len(set(bc_to_group.values()))} group(s)")

    with tempfile.TemporaryDirectory(dir=tempdir) as tmpdirname:
        # Export per-group BEDs.
        beds = _export_group_beds(str(frag_file), bc_to_group, tmpdirname)

        peaks: dict[str, pd.DataFrame] = {}
        for group, bed in sorted(beds.items()):
            console.level2(f"  calling peaks for group {group!r}")
            df = _run_macs3(
                bed_path=bed, outdir=tmpdirname, name=f"pk_{group}",
                qvalue=qvalue, shift=shift, extsize=extsize,
                nolambda=nolambda, call_broad_peaks=call_broad_peaks,
                broad_cutoff=broad_cutoff, gsize=gsize,
            )
            if blacklist is not None and len(df):
                mask = _bed_overlap_mask(df, str(blacklist))
                df = df.loc[~mask].reset_index(drop=True)
            peaks[group] = df

    if groupby is None:
        bulk = peaks.get("all", pd.DataFrame())
        out_key = f"{key_added}_pseudobulk" if inplace else None
        if inplace:
            adata.uns[out_key] = bulk
        return None if inplace else bulk
    if inplace:
        adata.uns[f"{key_added}_pseudobulk"] = peaks
        return None
    return peaks


def _bed_overlap_mask(peaks_df: pd.DataFrame, bed_path: str) -> np.ndarray:
    """Return a boolean mask over ``peaks_df`` rows marking peaks
    overlapping at least one interval in ``bed_path``.
    """
    from collections import defaultdict
    intervals = defaultdict(list)
    opener = gzip.open if bed_path.endswith(".gz") else open
    with opener(bed_path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                intervals[parts[0]].append((int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    for chrom in intervals:
        intervals[chrom].sort()

    mask = np.zeros(len(peaks_df), dtype=bool)
    for i, (chrom, s, e) in enumerate(zip(
        peaks_df["chrom"], peaks_df["start"], peaks_df["end"])):
        ivs = intervals.get(chrom)
        if not ivs:
            continue
        lo, hi = 0, len(ivs)
        while lo < hi:
            mid = (lo + hi) // 2
            if ivs[mid][0] >= int(e):
                hi = mid
            else:
                lo = mid + 1
        k = lo - 1
        while k >= 0 and ivs[k][0] < int(e):
            if ivs[k][1] > int(s):
                mask[i] = True
                break
            k -= 1
    return mask


# ---------------------------------------------------------------------------
# Merge peaks (ArchR-style summit expansion + iterative overlap resolution)
# ---------------------------------------------------------------------------

def merge_peaks(
    peaks: dict,
    chrom_sizes: Union[dict, Genome],
    half_width: int = 250,
) -> pd.DataFrame:
    """Merge per-group MACS3 peaks into a single non-overlapping peak set.

    Algorithm (matches the snapATAC2 / ArchR implementation):

    1. For each peak, re-centre at its summit and expand by
       ``half_width`` on each side — all merged peaks get the same
       fixed width ``2 * half_width``.
    2. Sort by ascending qValue (smaller = more significant).
    3. Walk the sorted list; accept each peak unless it overlaps a
       peak already accepted (keep the higher-significance one).
    """
    if isinstance(chrom_sizes, Genome):
        chrom_sizes = chrom_sizes.chrom_sizes
    chrom_sizes = dict(chrom_sizes)

    # Normalise input to a single DataFrame with a ``group`` column.
    frames = []
    for g, df in peaks.items():
        if df is None or len(df) == 0:
            continue
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        tmp = df.copy()
        tmp["group"] = g
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=["chrom", "start", "end", "Peaks"])
    combined = pd.concat(frames, ignore_index=True)

    # Compute summit-centred coordinates. MACS3 narrowPeak's ``peak`` col
    # is the summit offset relative to ``start``; broadPeak lacks it
    # (fall back to interval midpoint).
    if "peak" in combined.columns and combined["peak"].dtype != object:
        summit = combined["start"].astype(int) + combined["peak"].astype(int)
    else:
        summit = ((combined["start"].astype(int) + combined["end"].astype(int)) // 2)
    chroms = combined["chrom"].astype(str).values
    lo = (summit - half_width).astype(int).values
    hi = (summit + half_width).astype(int).values

    # Clip to chromosome bounds.
    for i, chrom in enumerate(chroms):
        size = chrom_sizes.get(chrom)
        if size is None:
            lo[i] = -1; hi[i] = -1
            continue
        lo[i] = max(lo[i], 0)
        hi[i] = min(hi[i], int(size))
    valid = lo >= 0

    # Sort by qValue (most significant first). narrowPeak stores
    # ``-log10(qvalue)`` → higher = more significant. If absent, fall
    # back to ``score``.
    if "qValue" in combined.columns:
        rank = -combined["qValue"].astype(float).fillna(0).values
    elif "score" in combined.columns:
        rank = -combined["score"].astype(float).fillna(0).values
    else:
        rank = np.arange(len(combined), dtype=float)

    order = np.argsort(rank, kind="stable")

    # Group by chromosome, keep peaks whose interval doesn't overlap
    # any previously-accepted one.
    from collections import defaultdict
    accepted_by_chrom: dict = defaultdict(list)
    out_rows = []
    for idx in order:
        if not valid[idx]:
            continue
        c = chroms[idx]; s = int(lo[idx]); e = int(hi[idx])
        taken = False
        for as_, ae_ in accepted_by_chrom[c]:
            if s < ae_ and e > as_:
                taken = True
                break
        if taken:
            continue
        accepted_by_chrom[c].append((s, e))
        out_rows.append({
            "chrom": c,
            "start": s,
            "end": e,
            "Peaks": f"{c}:{s}-{e}",
            "group": combined["group"].iloc[idx],
        })

    out = pd.DataFrame(out_rows)
    # Sort by genomic position for a tidy output.
    out = out.sort_values(["chrom", "start"]).reset_index(drop=True)
    console.level2(f"merged {len(out):,} non-overlapping peaks")
    return out
