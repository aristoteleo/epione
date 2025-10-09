#!/usr/bin/env python

"""
Footprint aggregation utilities inspired by ArchR's getFootprints/plotFootprints.

These routines operate directly on cutsite bigwig files (such as those produced
by `pseudobulk_with_fragments(..., bigwig_strategy="cutsite")`) and motif
regions, returning aggregate insertion profiles per condition & motif.

Bias correction (Tn5 k-mer) is not performed here; profiles represent the raw
average cutsite signal. The plotting helper mirrors ArchR's controls for
normalisation and smoothing.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyBigWig
import pysam

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..utils.regions import RegionList, OneRegion
from ..utils.motifs import MotifList
from ..utils import console

try:
    from tqdm import tqdm
except ImportError:  # fallback when tqdm unavailable
    def _progress_iter(iterable, **kwargs):
        return iterable
else:
    def _progress_iter(iterable, **kwargs):
        return tqdm(iterable, **kwargs)


def _ensure_dict(obj: Mapping[str, str] | Sequence[str]) -> Dict[str, str]:
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, Sequence):
        return {os.path.splitext(os.path.basename(path))[0]: path for path in obj}
    raise TypeError("score_files must be a mapping or sequence of paths.")


def _ensure_region_list(obj) -> RegionList:
    if isinstance(obj, RegionList):
        return obj
    rl = RegionList()
    if isinstance(obj, Iterable):
        for elem in obj:
            if isinstance(elem, OneRegion):
                rl.append(elem)
            elif isinstance(elem, (list, tuple)) and len(elem) >= 3:
                rl.append(OneRegion(list(elem)))
            else:
                raise ValueError(f"Unsupported motif position entry: {elem}")
        return rl
    raise TypeError("Positions must be RegionList or iterable of regions.")


def _running_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    window = min(window, arr.size)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(arr, kernel, mode="same")


def _normalise_profile(profile: np.ndarray, method: str, flank_norm: int, eps: float = 1e-9) -> np.ndarray:
    method = method.lower()
    if method == "none":
        return profile

    flank_norm = max(1, min(flank_norm, profile.size // 2))
    flank_idx = np.concatenate(
        [
            np.arange(0, flank_norm, dtype=int),
            np.arange(profile.size - flank_norm, profile.size, dtype=int),
        ]
    )
    baseline = np.mean(profile[flank_idx]) if flank_idx.size > 0 else 0.0

    if method == "subtract":
        return profile - baseline
    if method == "divide":
        denom = baseline if baseline > eps else eps
        return profile / denom
    raise ValueError("normMethod must be one of {'None','Subtract','Divide'}.")


@dataclass
class FootprintResult:
    motifs: List[str]
    conditions: List[str]
    flank: int
    profiles_: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    counts_: Dict[str, Dict[str, int]] = field(default_factory=dict)
    site_counts: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)

    def get_profile(
        self,
        motif: str,
        condition: str,
        norm_method: str = "None",
        flank_norm: Optional[int] = None,
        smooth_window: int = 1,
    ) -> np.ndarray:
        if motif not in self.profiles_:
            raise KeyError(f"Motif '{motif}' not found.")
        if condition not in self.profiles_[motif]:
            raise KeyError(f"Condition '{condition}' not found for motif '{motif}'.")

        profile = self.profiles_[motif][condition]
        norm_method = norm_method.lower()
        flank_norm = flank_norm if flank_norm is not None else max(1, self.flank // 2)
        out = _normalise_profile(profile, norm_method, flank_norm)
        if smooth_window > 1:
            out = _running_mean(out, smooth_window)
        return out

    def condition_counts(self, motif: str, condition: Optional[str] = None) -> int | Dict[str, int]:
        if motif not in self.counts_:
            raise KeyError(f"Motif '{motif}' not found.")
        if condition is None:
            return self.counts_[motif]
        return self.counts_[motif].get(condition, 0)


def get_footprints(
    score_files: Mapping[str, str] | Sequence[str],
    positions: Mapping[str, RegionList | Sequence],
    motif_file: Optional[str] = None,
    genome_fasta: Optional[str] = None,
    flank: int = 250,
    min_sites: int = 25,
    verbose: int = 2,
) -> FootprintResult:
    """
    Aggregate cutsite signals around motif instances per condition.

    Parameters
    ----------
    score_files :
        Mapping condition -> bigwig path, or sequence of paths (names inferred from basenames).
        Each bigwig must contain cutsite-level signal (1bp coverage).
    positions :
        Mapping motif name -> RegionList (or iterable) describing motif instances.
        Regions should carry start/end (and strand if available). Windows are centred at motif centres.
    motif_file :
        Optional motif file to derive consistent naming (JASPAR/MEME via MotifList).
        When provided, motif prefixes and valid motifs are taken from this file.
    genome_fasta :
        Optional FASTA path to verify chromosome boundaries. If provided, ensures windows stay within bounds.
    flank :
        Number of base pairs to include on each side of motif centre (window length = 2*flank+1).
    min_sites :
        Minimum motif occurrences required to retain an aggregate profile.
    verbose :
        Console verbosity (0=silent, 1=level1, 2=level2+).

    Returns
    -------
    FootprintResult
        Aggregated footprint profiles.
    """

    console.verbosity = verbose
    flank = int(flank)
    if flank < 1:
        raise ValueError("flank must be >= 1.")
    window_len = 2 * flank  # match ArchR/TOBIAS window convention

    cond_to_file = _ensure_dict(score_files)
    conditions = list(cond_to_file.keys())

    motif_map: Dict[str, RegionList] = {}
    for motif, sites in positions.items():
        rl = _ensure_region_list(sites)
        if len(rl) > 0:
            motif_map[str(motif)] = rl

    if motif_file:
        motif_list = MotifList().from_file(motif_file)
        for motif in motif_list:
            motif.set_prefix()
        valid_names = set(motif.prefix for motif in motif_list)
        motif_map = {name: rl for name, rl in motif_map.items() if name in valid_names}
        if verbose >= 1:
            console.level2(f"Filtered motifs using motif_file; retaining {len(motif_map)} motifs.")

    if not motif_map:
        raise ValueError("No motif positions available after filtering.")

    chrom_lengths = {}
    fasta_obj = None
    if genome_fasta:
        fasta_obj = pysam.FastaFile(genome_fasta)
        chrom_lengths = dict(zip(fasta_obj.references, fasta_obj.lengths))

    bigwig_handles = {}
    for cond, path in cond_to_file.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Bigwig file not found: {path}")
        bigwig_handles[cond] = pyBigWig.open(path)

    aggregates: Dict[str, Dict[str, np.ndarray]] = {
        motif: {cond: np.zeros(window_len, dtype=np.float64) for cond in conditions}
        for motif in motif_map.keys()
    }
    counts: Dict[str, Dict[str, int]] = {
        motif: {cond: 0 for cond in conditions}
        for motif in motif_map.keys()
    }
    site_counts: Dict[str, int] = {motif: 0 for motif in motif_map.keys()}

    motif_iter = motif_map.items()
    if verbose >= 1:
        motif_iter = _progress_iter(motif_iter, total=len(motif_map), desc="Motifs")

    for motif, sites in motif_iter:
        site_iter = sites
        if verbose >= 2:
            site_iter = _progress_iter(site_iter, total=len(sites), desc=f"{motif} sites")

        for region in site_iter:
            chrom = getattr(region, "chrom", region[0])
            start = int(getattr(region, "start", region[1]))
            end = int(getattr(region, "end", region[2]))
            strand = getattr(region, "strand", region[5] if len(region) > 5 else "+")

            if chrom_lengths:
                if chrom not in chrom_lengths:
                    continue
                if end > chrom_lengths[chrom]:
                    continue

            centre = (start + end) // 2
            window_start = centre - flank
            window_end = centre + flank

            if window_start < 0:
                continue
            if chrom_lengths and window_end > chrom_lengths[chrom]:
                continue

            windows = {}
            valid = True
            for cond, handle in bigwig_handles.items():
                values = handle.values(chrom, window_start, window_end)
                if values is None:
                    valid = False
                    break
                arr = np.array(values, dtype=np.float64)
                arr = np.nan_to_num(arr, nan=0.0)
                if arr.shape[0] != window_len:
                    valid = False
                    break
                if strand == "-":
                    arr = arr[::-1]
                windows[cond] = arr

            if not valid:
                continue

            site_counts[motif] += 1
            for cond in conditions:
                aggregates[motif][cond] += windows[cond]
                counts[motif][cond] += 1

    if fasta_obj is not None:
        fasta_obj.close()
    for handle in bigwig_handles.values():
        handle.close()

    avg_profiles: Dict[str, Dict[str, np.ndarray]] = {}
    for motif, cond_dict in aggregates.items():
        total_sites = site_counts[motif]
        if total_sites < min_sites:
            continue
        avg_profiles[motif] = {}
        for cond, arr in cond_dict.items():
            n = counts[motif][cond]
            profile = arr / n if n > 0 else np.zeros(window_len, dtype=np.float64)
            avg_profiles[motif][cond] = profile

    if not avg_profiles:
        raise ValueError("No motifs retained after applying min_sites threshold.")

    result = FootprintResult(
        motifs=list(avg_profiles.keys()),
        conditions=conditions,
        flank=flank,
        profiles_=avg_profiles,
        counts_={motif: counts[motif] for motif in avg_profiles.keys()},
        site_counts={motif: site_counts[motif] for motif in avg_profiles.keys()},
        metadata={
            "min_sites": min_sites,
            "window_len": window_len,
            "conditions": conditions,
            "total_sites": site_counts,
        },
    )
    return result


def plot_footprints(
    footprint_result: FootprintResult,
    normMethod: str = "None",
    plotName: str = "Footprints",
    output_dir: str = ".",
    addDOC: bool = False,
    smoothWindow: int = 5,
    flankNorm: Optional[int] = None,
    figsize: Tuple[float, float] = (6.0, 3.0),
) -> str:
    """
    Plot aggregate footprints akin to ArchR's plotFootprints.

    Parameters
    ----------
    footprint_result :
        Result returned by `get_footprints`.
    normMethod :
        "None", "Subtract", or "Divide".
    plotName :
        Base filename (without extension) for the generated PNG figure.
    output_dir :
        Directory where the plot will be saved.
    addDOC :
        If True, add a secondary axis showing total depth of coverage (sum of profiles).
    smoothWindow :
        Size of moving-average smoothing window.
    flankNorm :
        Baseline flank size for normalisation. Defaults to flank//2.
    figsize :
        Size of each subplot (width, height). Height scaled by number of motifs.

    Returns
    -------
    str
        Path to generated PNG file.
    """

    os.makedirs(output_dir, exist_ok=True)
    motifs = footprint_result.motifs
    conditions = footprint_result.conditions
    flank = footprint_result.flank

    n_motifs = len(motifs)
    if n_motifs == 0:
        raise ValueError("No motifs available for plotting.")

    width, height = figsize
    total_height = height * n_motifs
    fig, axes = plt.subplots(
        n_motifs,
        1,
        sharex=True,
        figsize=(width, total_height),
        squeeze=False,
    )

    x = np.arange(-flank, flank)
    norm_method = normMethod.lower()
    flank_norm = flankNorm if flankNorm is not None else max(1, flank // 2)

    for idx, motif in enumerate(motifs):
        ax = axes[idx, 0]
        for cond in conditions:
            profile = footprint_result.get_profile(
                motif,
                cond,
                norm_method=norm_method,
                flank_norm=flank_norm,
                smooth_window=smoothWindow,
            )
            ax.plot(
                x,
                profile,
                label=f"{cond} (n={footprint_result.counts_[motif][cond]})",
                linewidth=1.4,
            )

        ax.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylabel(motif)
        if norm_method == "none":
            ax.set_ylabel(f"{motif}\nCuts", rotation=0, labelpad=40)
        elif norm_method == "subtract":
            ax.set_ylabel(f"{motif}\nCuts (norm)", rotation=0, labelpad=40)
        else:
            ax.set_ylabel(f"{motif}\nRelative", rotation=0, labelpad=40)

        if addDOC:
            doc = sum(
                footprint_result.get_profile(
                    motif,
                    cond,
                    norm_method="none",
                    flank_norm=flank_norm,
                    smooth_window=smoothWindow,
                )
                for cond in conditions
            )
            ax2 = ax.twinx()
            ax2.plot(x, doc, color="grey", linewidth=1.0, alpha=0.3, label="DOC")
            ax2.set_ylabel("DOC", color="grey")
            ax2.tick_params(axis="y", colors="grey")

        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Distance to motif centre (bp)")
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{plotName}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def getFootprints(*args, **kwargs):
    """CamelCase alias mirroring ArchR naming."""
    return get_footprints(*args, **kwargs)


def plotFootprints(*args, **kwargs):
    """CamelCase alias mirroring ArchR naming."""
    return plot_footprints(*args, **kwargs)


__all__ = [
    "get_footprints",
    "plot_footprints",
    "FootprintResult",
    "getFootprints",
    "plotFootprints",
]
