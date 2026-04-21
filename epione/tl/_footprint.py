"""ArchR-style footprint analysis: per-group Tn5 cut density aggregation
around motif positions, with hexamer bias correction.

Produces three layers per motif (mirrors ArchR ``getFootprints`` output):

1. ``signal``          — raw Tn5 cut count per group × per bp in window
2. ``Tn5Bias``         — expected cut density from local hexamer × global
                         kmer bias table (genome-wide baseline)
3. ``normalizedSignal``— ``signal - Tn5Bias``  (or ``signal / Tn5Bias`` if
                         ``normalize='Divide'``)

Plotting mirrors ArchR ``plotFootprints`` — aggregate profile per group
with optional bias track below.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..utils import console
from ..utils._genome import Genome


# ---------------------------------------------------------------------------
# k-mer helpers
# ---------------------------------------------------------------------------

_BASE_TO_INT = np.full(256, 255, dtype=np.uint8)
for _b, _v in zip(b"ACGT", (0, 1, 2, 3)):
    _BASE_TO_INT[_b] = _v
    _BASE_TO_INT[ord(chr(_b).lower())] = _v


def _seq_to_codes(seq: str) -> np.ndarray:
    """ACGT string → numpy int codes (A=0, C=1, G=2, T=3, other=255)."""
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return _BASE_TO_INT[arr]


def _codes_to_kmers(codes: np.ndarray, k: int) -> np.ndarray:
    """Rolling k-mer encoded as base-4 integer. Invalid windows → -1."""
    n = len(codes) - k + 1
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    kmers = np.zeros(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    for i in range(k):
        col = codes[i : i + n]
        kmers = kmers * 4 + col.astype(np.int64)
        valid &= col < 4
    kmers = np.where(valid, kmers, -1)
    return kmers


# ---------------------------------------------------------------------------
# Tn5 bias table — genome-wide k-mer insertion rate
# ---------------------------------------------------------------------------

def compute_tn5_bias_table(
    fragment_file: Union[str, Path],
    genome_fasta: Union[str, Path],
    *,
    kmer_length: int = 6,
    max_fragments: int = 5_000_000,
    exclude_chroms: Sequence[str] = ("chrM", "chrMT", "M", "MT"),
    chroms: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Per-k-mer Tn5 bias table, defined as
    ``observed_rate / expected_rate`` where *expected* is the global
    cut-rate times the genome-wide k-mer frequency.

    Parameters
    ----------
    fragment_file
        Tabix-indexed fragments ``.tsv.gz`` (output of
        :func:`epione.pp.import_fragments` or similar).
    genome_fasta
        Matching genome FASTA (can be ``.gz`` — pyfaidx handles it).
    kmer_length
        k-mer size centred on the cut position (ArchR default: 6).
    max_fragments
        Stop scanning after this many fragments (2 cuts each). Set to
        ``math.inf`` for the full file.
    exclude_chroms
        Chromosomes skipped (default mitochondria).
    chroms
        Restrict to this chromosome list (default: all in tabix index).

    Returns
    -------
    ndarray of shape ``(4**kmer_length,)`` — ``bias[kmer_idx]`` is the
    insertion fold-enrichment of that hexamer relative to uniform. Save
    with ``np.savez_compressed`` for reuse.
    """
    import pyfaidx
    import pysam

    k = int(kmer_length)
    half_k = k // 2

    fa = pyfaidx.Fasta(str(genome_fasta))
    tbx = pysam.TabixFile(str(fragment_file))

    exclude = set(exclude_chroms)
    use_chroms = [c for c in (chroms or tbx.contigs) if c not in exclude]

    size = 4 ** k
    cut_counts = np.zeros(size, dtype=np.int64)
    kmer_counts = np.zeros(size, dtype=np.int64)

    n_frags = 0
    for chrom in use_chroms:
        if chrom not in fa:
            continue
        seq = str(fa[chrom]).upper()
        codes = _seq_to_codes(seq)

        # Genome-wide k-mer frequency for this chrom.
        kmer_arr = _codes_to_kmers(codes, k)
        valid_kmers = kmer_arr >= 0
        if valid_kmers.any():
            np.add.at(kmer_counts, kmer_arr[valid_kmers], 1)

        # Fragment cuts: 2 insertion events per fragment.
        if chrom not in tbx.contigs:
            continue
        for line in tbx.fetch(chrom):
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                s = int(parts[1]); e = int(parts[2])
            except ValueError:
                continue
            for cut in (s, e - 1):
                lo = cut - half_k
                hi = cut + (k - half_k)
                if 0 <= lo and hi <= len(codes):
                    sub = codes[lo:hi]
                    if (sub < 4).all():
                        idx = 0
                        for v in sub:
                            idx = idx * 4 + int(v)
                        cut_counts[idx] += 1
            n_frags += 1
            if n_frags >= max_fragments:
                break
        if n_frags >= max_fragments:
            break

    tbx.close()

    total_cuts = int(cut_counts.sum())
    total_kmers = int(kmer_counts.sum())
    if total_cuts == 0:
        raise ValueError("no fragments were scanned — check fragment file")
    if total_kmers == 0:
        raise ValueError("no valid k-mers in genome — check FASTA")

    expected = kmer_counts * (total_cuts / total_kmers)
    with np.errstate(divide="ignore", invalid="ignore"):
        bias = np.where(expected > 0, cut_counts / expected, 1.0)
    # Clip extreme outliers (rare kmers) for numerical stability.
    bias = np.clip(bias, 1e-3, 1e3).astype(np.float64)
    console.level2(
        f"Tn5 bias: scanned {n_frags:,} fragments, {total_cuts:,} cuts; "
        f"k={k}, bias range [{bias.min():.3f}, {bias.max():.3f}]"
    )
    return bias


# ---------------------------------------------------------------------------
# Footprint aggregation
# ---------------------------------------------------------------------------

@dataclass
class Footprint:
    """Per-motif aggregate footprint across groups (ArchR
    ``getFootprints`` output analogue)."""
    motif: str
    groups: List[str]
    signal: np.ndarray            # (n_groups, window) raw cut density / site
    Tn5Bias: Optional[np.ndarray] # (window,) expected cut density / site
    normalizedSignal: np.ndarray  # signal - Tn5Bias (or signal / Tn5Bias)
    n_sites: Dict[str, int]
    flank: int
    kmer_length: int
    normalize: str

    @property
    def positions(self) -> np.ndarray:
        """Genomic offsets covered by each column — ``[-flank, …, flank]``."""
        return np.arange(-self.flank, self.flank + 1)


def _parse_positions(pos_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a motif positions DataFrame: ensure ``chrom / center
    / strand`` columns."""
    df = pos_df.copy()
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for k in ("chrom", "chromosome", "seqname", "seqnames"):
        if k in lower:
            rename[lower[k]] = "chrom"; break
    for k in ("strand",):
        if k in lower:
            rename[lower[k]] = "strand"; break
    if "center" not in lower:
        if "start" in lower and "end" in lower:
            df = df.rename(columns=rename)
            df["center"] = ((df["start"].astype(int) + df["end"].astype(int)) // 2).astype(int)
        else:
            raise ValueError("positions needs either a ``center`` column or both ``start`` + ``end``")
    else:
        rename[lower["center"]] = "center"
        df = df.rename(columns=rename)
    if "strand" not in df.columns:
        df["strand"] = "+"
    return df[["chrom", "center", "strand"]].reset_index(drop=True)


def get_footprints(
    adata,
    *,
    positions: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
    groupby: str,
    genome: Union[Genome, str, Path, None] = None,
    flank: int = 250,
    normalize: Literal["None", "Subtract", "Divide"] = "Subtract",
    kmer_length: int = 6,
    bias_table: Optional[np.ndarray] = None,
    smooth: int = 0,
    min_cells_per_group: int = 10,
    motif_name: str = "motif",
) -> Dict[str, Footprint]:
    """Per-group Tn5 footprint aggregation around motif positions.

    Mirrors ArchR ``getFootprints`` — for every motif position, counts
    Tn5 insertion events in a ``±flank`` window per obs-group, and
    subtracts an expected profile derived from the local hexamer ×
    genome-wide Tn5 bias table.

    Parameters
    ----------
    adata
        AnnData / AnnDataOOM with
        ``uns['files']['fragments']`` (tabix-indexed) and a
        ``groupby`` column in ``obs`` (typically cluster / celltype).
    positions
        Motif positions. Either:

        * a DataFrame with ``chrom / center / strand`` (or
          ``chrom / start / end`` — ``center = (start+end)//2``), or
        * a dict ``{motif_name: positions_df}`` to aggregate multiple
          motifs in a single sweep of the fragments.
    groupby
        ``adata.obs`` column labelling the pseudo-bulk groups.
    genome
        ``Genome`` / FASTA path — required when ``normalize != 'None'``
        (bias track needs the genome sequence). Ignored when an
        explicit ``bias_table`` is supplied.
    flank
        Half-window around each motif centre (ArchR default 250).
    normalize
        ``'None'`` — return raw signal only.
        ``'Subtract'`` — ``signal - bias``  (ArchR default).
        ``'Divide'`` — ``signal / bias``.
    kmer_length
        k-mer size for Tn5 bias (ArchR default 6).
    bias_table
        Pre-computed hexamer bias (from
        :func:`compute_tn5_bias_table`). If ``None`` it's built on the
        fly (slower).
    smooth
        Rolling-mean window (bp). 0 = no smoothing.
    min_cells_per_group
        Groups with fewer cells than this are dropped.

    Returns
    -------
    dict
        ``{motif_name: Footprint}`` — call :func:`plot_footprints` to
        render.
    """
    import pyfaidx
    import pysam

    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise ValueError("adata.uns['files']['fragments'] missing — run "
                         "epi.pp.import_fragments first")
    frag_file = str(adata.uns["files"]["fragments"])

    # Normalise positions input.
    if isinstance(positions, pd.DataFrame):
        positions_dict = {motif_name: _parse_positions(positions)}
    else:
        positions_dict = {name: _parse_positions(df) for name, df in positions.items()}

    # Genome FASTA for bias track.
    do_bias = normalize.lower() != "none"
    fa = None
    if do_bias:
        if bias_table is None:
            if genome is None:
                raise ValueError("normalize != 'None' needs a Genome / fasta path or a pre-computed bias_table")
            gfasta = genome.fasta if isinstance(genome, Genome) else str(genome)
            bias_table = compute_tn5_bias_table(frag_file, gfasta, kmer_length=kmer_length)
        if genome is None:
            raise ValueError("genome is required to evaluate bias per motif position")
        gfasta = genome.fasta if isinstance(genome, Genome) else str(genome)
        fa = pyfaidx.Fasta(str(gfasta))

    k = int(kmer_length)
    half_k = k // 2

    # Groups — drop sparse ones, keep ArchR-like sort.
    group_series = adata.obs[groupby]
    if not isinstance(group_series, pd.Series):
        group_series = pd.Series(group_series)
    group_series = group_series.astype(str)
    group_series.index = list(adata.obs_names)
    group_counts = group_series.value_counts()
    keep_groups = sorted(g for g, n in group_counts.items() if n >= min_cells_per_group)
    if not keep_groups:
        raise ValueError("no groups with >= min_cells_per_group cells")
    bc_to_group = dict(group_series.items())

    window = 2 * flank + 1
    tbx = pysam.TabixFile(frag_file)

    results: Dict[str, Footprint] = {}
    for motif, pos_df in positions_dict.items():
        console.level1(
            f"footprint: {motif} — {len(pos_df):,} positions × {len(keep_groups)} groups"
        )
        signal = {g: np.zeros(window, dtype=np.float64) for g in keep_groups}
        n_sites = {g: 0 for g in keep_groups}
        bias_accum = np.zeros(window, dtype=np.float64)
        n_bias_sites = 0

        for chrom_name, group_df in pos_df.groupby("chrom", sort=False):
            if chrom_name not in tbx.contigs:
                continue
            seq_cache = None
            if do_bias and chrom_name in fa:
                seq_cache = _seq_to_codes(str(fa[chrom_name]).upper())
            for _, row in group_df.iterrows():
                center = int(row["center"])
                strand = str(row.get("strand", "+"))
                lo = center - flank
                hi = center + flank + 1
                if lo < 0:
                    continue

                # Bias profile — expected cut rate per bp from local k-mer.
                if do_bias and seq_cache is not None:
                    # Need codes[cut-half_k : cut+(k-half_k)] for each cut
                    # in [lo, hi). Equivalent to sliding kmers over
                    # codes[lo-half_k : hi + (k-half_k) - 1].
                    seq_lo = lo - half_k
                    seq_hi = hi + (k - half_k) - 1
                    if 0 <= seq_lo and seq_hi <= len(seq_cache):
                        sub = seq_cache[seq_lo:seq_hi]
                        kmers = _codes_to_kmers(sub, k)
                        if len(kmers) == window:
                            bp_bias = np.where(
                                kmers >= 0, bias_table[np.clip(kmers, 0, None)], 1.0
                            )
                            if strand == "-":
                                bp_bias = bp_bias[::-1]
                            bias_accum += bp_bias
                            n_bias_sites += 1

                # Cut density per group.
                for line in tbx.fetch(chrom_name, lo, hi):
                    parts = line.split("\t")
                    if len(parts) < 4:
                        continue
                    bc = parts[3]
                    g = bc_to_group.get(bc)
                    if g is None:
                        continue
                    try:
                        s = int(parts[1]); e = int(parts[2])
                    except ValueError:
                        continue
                    for cut in (s, e - 1):
                        if lo <= cut < hi:
                            idx = cut - lo if strand == "+" else (hi - 1 - cut)
                            signal[g][idx] += 1.0
                for g in keep_groups:
                    n_sites[g] += 1

        # Per-site average.
        sig_mat = np.vstack([
            signal[g] / max(n_sites[g], 1) for g in keep_groups
        ])
        bias_prof = bias_accum / max(n_bias_sites, 1) if n_bias_sites else None

        if smooth and smooth > 1:
            w = int(smooth)
            ker = np.ones(w, dtype=np.float64) / w
            sig_mat = np.vstack([np.convolve(row, ker, mode="same") for row in sig_mat])
            if bias_prof is not None:
                bias_prof = np.convolve(bias_prof, ker, mode="same")

        if normalize.lower() == "subtract" and bias_prof is not None:
            # Match rate levels: bias track (fold-enrichment) is dimensionless,
            # so scale it to the mean of signal's flanks before subtracting.
            flank_idx = np.r_[np.arange(0, flank // 2), np.arange(window - flank // 2, window)]
            sig_mean_flank = sig_mat[:, flank_idx].mean(axis=1, keepdims=True)
            bias_mean_flank = bias_prof[flank_idx].mean() or 1e-9
            bias_scaled = (bias_prof / bias_mean_flank) * sig_mean_flank
            normalized = sig_mat - bias_scaled
        elif normalize.lower() == "divide" and bias_prof is not None:
            normalized = sig_mat / np.maximum(bias_prof[None, :], 1e-9)
        else:
            normalized = sig_mat

        results[motif] = Footprint(
            motif=motif,
            groups=list(keep_groups),
            signal=sig_mat,
            Tn5Bias=bias_prof,
            normalizedSignal=normalized,
            n_sites=dict(n_sites),
            flank=flank,
            kmer_length=k,
            normalize=normalize,
        )

    tbx.close()
    return results


# Back-compat alias (ArchR name).
getFootprints = get_footprints


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_footprints(
    footprints: Union[Footprint, Mapping[str, Footprint]],
    *,
    motif: Optional[str] = None,
    groups: Optional[Sequence[str]] = None,
    palette: Optional[Sequence[str]] = None,
    show_bias: bool = True,
    figsize: Tuple[float, float] = (6.0, 4.5),
    show: bool = True,
):
    """ArchR-style footprint plot.

    One aggregate curve per group on the main axis; below it an
    optional Tn5 bias track (light grey). Returns ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt

    if isinstance(footprints, Footprint):
        fp = footprints
    else:
        if motif is None:
            if len(footprints) == 1:
                motif = next(iter(footprints))
            else:
                raise ValueError(
                    f"pass motif= one of {list(footprints)}"
                )
        fp = footprints[motif]

    g_order = list(groups) if groups is not None else fp.groups
    if palette is None:
        try:
            from ..pl._palette import ARCHR_STALLION
            palette = ARCHR_STALLION
        except Exception:
            palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {g: palette[i % len(palette)] for i, g in enumerate(g_order)}

    has_bias = show_bias and fp.Tn5Bias is not None
    if has_bias:
        fig, (ax, bias_ax) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw=dict(height_ratios=[3.5, 1.0], hspace=0.08),
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        bias_ax = None

    x = fp.positions
    y_key = fp.normalizedSignal if fp.normalize.lower() != "none" else fp.signal
    idx = {g: i for i, g in enumerate(fp.groups)}
    for g in g_order:
        if g not in idx:
            continue
        ax.plot(x, y_key[idx[g], :], color=colors[g], lw=1.2, label=g)

    ax.axvline(0, color="black", lw=0.4, ls=":")
    ax.set_ylabel(
        "Normalized insertions" if fp.normalize.lower() == "subtract"
        else "Signal / bias" if fp.normalize.lower() == "divide"
        else "Insertions / site"
    )
    ax.set_title(f"{fp.motif}  ({fp.n_sites.get(fp.groups[0], 0):,} sites)")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=1)
    ax.spines[["top", "right"]].set_visible(False)

    if has_bias:
        bias_ax.plot(x, fp.Tn5Bias, color="#888888", lw=1.0)
        bias_ax.axvline(0, color="black", lw=0.4, ls=":")
        bias_ax.set_xlabel("Distance to motif centre (bp)")
        bias_ax.set_ylabel("Tn5 bias")
        bias_ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.set_xlabel("Distance to motif centre (bp)")

    if show:
        plt.tight_layout()
    return fig, (ax if not has_bias else (ax, bias_ax))


plotFootprints = plot_footprints


__all__ = [
    "Footprint",
    "compute_tn5_bias_table",
    "get_footprints", "getFootprints",
    "plot_footprints", "plotFootprints",
]
