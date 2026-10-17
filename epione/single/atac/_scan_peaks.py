"""CellScan: a cell-aware, Tn5-bias-corrected scan-statistic peak caller.

MACS2/MACS3 on pseudobulk (what ArchR, SnapATAC2, Signac and
:func:`epione.single.macs3` do) pools all cells into one track and
forgets which cell an insertion came from: twenty insertions from two
deeply sequenced cells look exactly like twenty insertions from twenty
cells. In single-cell data each cell is an independent biological
replicate, so CellScan scores a window by the number of **distinct cells**
that have an insertion in it and asks how surprising that number is given
every cell's own depth and the local, Tn5-bias-corrected background.

Model
-----
Insertions (Tn5 cuts) are taken at ``start`` and ``end - 1`` of each
fragment (ArchR convention; fragments carry the +4/-5 shift). Within a
cell, duplicate cut positions count once. Let ``d_c`` be the number of
insertions of cell ``c`` and ``D = sum_c d_c``.

*Tn5 bias.* ``b(x) = 1/2 [r_s(k(x)) + r_e(k(x))]`` where ``k(x)`` is the
k-mer centred on ``x`` and ``r_s``/``r_e`` are the observed/expected
k-mer frequency ratios of fragment-start and fragment-end cuts, estimated
in background (non-candidate) regions so that the sequence content of
open chromatin is not mistaken for enzyme preference. ``r`` is normalised
so that ``sum_k f_genome(k) r(k) = 1``, i.e. ``b`` has genome mean 1;
``b = 0`` at ``N`` bases.

*Background.* For a window ``W`` (width ``w``) centred on a grid point,
the expected insertion count is ``m(W) = B(W) * max_L N(L) / B(L)`` where
``B(.)`` is the summed bias, ``N(.)`` the (cell-deduplicated) insertion
count, and ``L`` ranges over windows of the ``background`` sizes centred
on the same point plus the whole genome (``N_G / B_G``). This is MACS's
``max(lambda_BG, lambda_local)`` (Zhang *et al.* 2008), with every rate
measured per unit of Tn5 preference instead of per base.

*Cells as replicates.* Under the null an insertion of cell ``c`` falls
in ``W`` as a Poisson event with mean ``m(W) d_c / D``, so cell ``c`` is
present in ``W`` with probability ``p_c = 1 - exp(-m d_c / D)`` and the
number of distinct cells ``K(W)`` is Poisson-binomial with mean

.. math:: g(m) = \\sum_c \\left(1 - e^{-m d_c / D}\\right).

The p-value is the Poisson(``g(m)``) upper tail at the observed ``K``
(Le Cam's approximation; ``p_c`` is small for any real window). A window
whose insertions come from a few deep cells has small ``K`` relative to
its insertion count, and saturation (``g(m) < m``) is accounted for.

*Scan.* ``K(W)`` is evaluated on a grid (``step`` bp) by a two-pointer
sweep over the sorted insertions with a per-cell occupancy counter, so a
chromosome costs ``O(insertions + grid points)``. Grid points with
``-log10 p >= min_score`` form candidate regions; within a region, scan
maxima are taken greedily with non-maximum suppression at
``half_width`` (the scan statistic of Glaz, Naus & Wallenstein 2001, a
fixed-width version of Kulldorff's 1997 scan), and each maximum's summit
is refined to the base whose ``+-summit_half_width`` neighbourhood holds
the most distinct cells. Multiple testing: Benjamini-Hochberg over the
reported maxima with ``m_tests = G_eff / w`` independent windows.

Output peaks are summit-centred and ``2 * half_width`` wide with
iterative overlap removal by score, i.e. the same post-processing as
:func:`epione.single.merge_peaks`, so peak sets are directly comparable
with the MACS-based callers.

References
----------
Glaz J, Naus J, Wallenstein S (2001) *Scan Statistics*. Springer.
Kulldorff M (1997) A spatial scan statistic. *Commun Stat Theory Methods* 26:1481-1496.
Zhang Y *et al.* (2008) Model-based analysis of ChIP-Seq (MACS). *Genome Biol* 9:R137.
Benjamini Y, Hochberg Y (1995) *J R Stat Soc B* 57:289-300.
Le Cam L (1960) An approximation theorem for the Poisson binomial distribution. *Pacific J Math* 10:1181-1197.
Green B *et al.* (2012) Insertion site preference of Mu and Tn5 transposons. *Mob DNA* 3:3.
Martins AL *et al.* (2018) Universal correction of enzymatic sequence bias reveals molecular signatures of protein/DNA interactions. *Nucleic Acids Res* 46:e9.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from epione.utils import console

__all__ = ["scan_peaks"]

_CELL_BITS = 23                      # up to 8.4M cells per call
_SIDE_BITS = 1


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------

_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import numba as nb

    @nb.njit(nogil=True, cache=False)
    def encode_seq(raw):
        out = np.empty(raw.shape[0], np.uint8)
        for i in range(raw.shape[0]):
            c = raw[i]
            if c == 65 or c == 97:
                out[i] = 0
            elif c == 67 or c == 99:
                out[i] = 1
            elif c == 71 or c == 103:
                out[i] = 2
            elif c == 84 or c == 116:
                out[i] = 3
            else:
                out[i] = 4
        return out

    @nb.njit(nogil=True, cache=False)
    def kmer_at(seq, x, k, left):
        # code of seq[x-left : x-left+k], -1 if out of range or N
        a = x - left
        if a < 0 or a + k > seq.shape[0]:
            return -1
        code = 0
        for j in range(k):
            c = seq[a + j]
            if c > 3:
                return -1
            code = code * 4 + c
        return code

    @nb.njit(nogil=True, cache=False)
    def kmer_track(seq, k, left):
        n = seq.shape[0]
        out = np.full(n, -1, np.int32)
        mask = (1 << (2 * k)) - 1
        code = 0
        valid = 0
        # rolling code of the k-mer ending at position i (i.e. starting i-k+1)
        for i in range(n):
            c = seq[i]
            if c > 3:
                valid = 0
                code = 0
                continue
            code = ((code << 2) | c) & mask
            valid += 1
            if valid >= k:
                x = i - k + 1 + left       # the k-mer starts at x-left
                if x < n:
                    out[x] = code
        return out

    @nb.njit(nogil=True, cache=False)
    def count_kmers(kt, pos, side, excl_lo, excl_hi, n_k):
        # counts of k-mer codes at insertion positions (start / end side),
        # skipping positions inside excluded (sorted, disjoint) intervals.
        cs = np.zeros(n_k, np.float64)
        ce = np.zeros(n_k, np.float64)
        j = 0
        ne = excl_lo.shape[0]
        for i in range(pos.shape[0]):
            x = pos[i]
            while j < ne and excl_hi[j] <= x:
                j += 1
            if j < ne and excl_lo[j] <= x:
                continue
            code = kt[x]
            if code < 0:
                continue
            if side[i] == 0:
                cs[code] += 1.0
            else:
                ce[code] += 1.0
        return cs, ce

    @nb.njit(nogil=True, cache=False)
    def count_genome_kmers(kt, excl_lo, excl_hi, n_k):
        cg = np.zeros(n_k, np.float64)
        j = 0
        ne = excl_lo.shape[0]
        for x in range(kt.shape[0]):
            while j < ne and excl_hi[j] <= x:
                j += 1
            if j < ne and excl_lo[j] <= x:
                continue
            code = kt[x]
            if code >= 0:
                cg[code] += 1.0
        return cg

    @nb.njit(nogil=True, cache=False)
    def bias_bins(kt, seq, r, step, n_bins):
        # summed bias per grid bin; r is None-equivalent when r.shape[0]==0
        out = np.zeros(n_bins, np.float64)
        n = seq.shape[0]
        use = r.shape[0] > 0
        for x in range(n):
            b = 0.0
            if use:
                code = kt[x]
                if code >= 0:
                    b = r[code]
            else:
                if seq[x] <= 3:
                    b = 1.0
            out[x // step] += b
        return out

    @nb.njit(nogil=True, cache=False)
    def poisson_logsf(k, lam):
        # log P(X >= k), X ~ Poisson(lam); only used for k > lam (else ~0)
        if k <= 0:
            return 0.0
        if lam <= 0.0:
            return -1e300
        lt = -lam + k * math.log(lam) - math.lgamma(k + 1.0)
        s = 1.0
        t = 1.0
        j = 1
        while True:
            t *= lam / (k + j)
            s += t
            if t < 1e-12 * s or j > 100000:
                break
            j += 1
        v = lt + math.log(s)
        if v > 0.0:
            v = 0.0
        return v

    @nb.njit(nogil=True, cache=False)
    def interp_g(m, log_grid, g_vals):
        # g(m) by linear interpolation in log m
        if m <= 0.0:
            return 0.0
        lm = math.log(m)
        n = log_grid.shape[0]
        if lm <= log_grid[0]:
            return g_vals[0] * m / math.exp(log_grid[0])
        if lm >= log_grid[n - 1]:
            return g_vals[n - 1]
        lo = 0
        hi = n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if log_grid[mid] <= lm:
                lo = mid
            else:
                hi = mid
        f = (lm - log_grid[lo]) / (log_grid[hi] - log_grid[lo])
        return g_vals[lo] * (1.0 - f) + g_vals[hi] * f

    @nb.njit(nogil=True, cache=False)
    def _grow_i64(a, cap):
        b = np.empty(cap, np.int64)
        b[:a.shape[0]] = a
        return b

    @nb.njit(nogil=True, cache=False)
    def _grow_i32(a, cap):
        b = np.empty(cap, np.int32)
        b[:a.shape[0]] = a
        return b

    @nb.njit(nogil=True, cache=False)
    def _grow_f32(a, cap):
        b = np.empty(cap, np.float32)
        b[:a.shape[0]] = a
        return b

    @nb.njit(nogil=True, cache=False)
    def scan(pos, cell, n_cells, step, hw_bins, n_bins, cum_n, cum_b, bg_hw_bins,
             genome_rate, log_grid, g_vals, min_score, distinct, min_cells, sample_every):
        """Sweep grid points 0..n_bins; return sparse (bin, score, K, N, expected)
        plus (statistic, expected) at every ``sample_every``-th grid point
        (0 = none), used to fit the overdispersed null."""
        LOG10 = math.log(10.0)
        occ = np.zeros(n_cells, np.int32)
        cap = 1024
        o_bin = np.empty(cap, np.int64)
        o_sc = np.empty(cap, np.float32)
        o_k = np.empty(cap, np.int32)
        o_n = np.empty(cap, np.int32)
        o_e = np.empty(cap, np.float32)
        n_out = 0
        n_samp = (n_bins // sample_every + 1) if sample_every > 0 else 0
        s_stat = np.zeros(n_samp, np.int32)
        s_lam = np.full(n_samp, -1.0, np.float32)
        lo = 0
        hi = 0
        n_ins = pos.shape[0]
        k_cur = 0
        n_bg = bg_hw_bins.shape[0]
        for i in range(n_bins + 1):
            a_bin = i - hw_bins
            b_bin = i + hw_bins
            a = a_bin * step
            b = b_bin * step
            while hi < n_ins and pos[hi] < b:
                c = cell[hi]
                if occ[c] == 0:
                    k_cur += 1
                occ[c] += 1
                hi += 1
            while lo < hi and pos[lo] < a:
                c = cell[lo]
                occ[c] -= 1
                if occ[c] == 0:
                    k_cur -= 1
                lo += 1
            n_w = hi - lo
            stat = k_cur if distinct else n_w
            samp = sample_every > 0 and i % sample_every == 0
            if stat < min_cells and not samp:
                continue
            a0 = max(a_bin, 0)
            b0 = min(b_bin, n_bins)
            if b0 <= a0:
                continue
            bw = cum_b[b0] - cum_b[a0]
            if bw <= 0.0:
                continue
            rate = genome_rate
            for s in range(n_bg):
                la = max(i - bg_hw_bins[s], 0)
                lb = min(i + bg_hw_bins[s], n_bins)
                bl = cum_b[lb] - cum_b[la]
                if bl > 0.0:
                    r = (cum_n[lb] - cum_n[la]) / bl
                    if r > rate:
                        rate = r
            m = rate * bw
            lam = interp_g(m, log_grid, g_vals) if distinct else m
            if samp:
                s_stat[i // sample_every] = stat
                s_lam[i // sample_every] = lam
            if stat < min_cells or stat <= lam:
                continue
            sc = -poisson_logsf(stat, lam) / LOG10
            if sc < min_score:
                continue
            if n_out == cap:
                cap *= 2
                o_bin = _grow_i64(o_bin, cap)
                o_sc = _grow_f32(o_sc, cap)
                o_k = _grow_i32(o_k, cap)
                o_n = _grow_i32(o_n, cap)
                o_e = _grow_f32(o_e, cap)
            o_bin[n_out] = i
            o_sc[n_out] = sc
            o_k[n_out] = k_cur
            o_n[n_out] = n_w
            o_e[n_out] = lam
            n_out += 1
        return (o_bin[:n_out], o_sc[:n_out], o_k[:n_out], o_n[:n_out], o_e[:n_out],
                s_stat, s_lam)

    @nb.njit(nogil=True, cache=False)
    def nms(bins, scores, step, gap_bins, sup_bins, dip):
        """Group grid hits into regions (gaps <= gap_bins) and pick scan maxima
        greedily with non-maximum suppression radius sup_bins. With
        ``dip < 1`` a weaker maximum u is kept only if, towards every
        stronger kept maximum v, the scan falls below ``dip * score_u``
        somewhere between them (a gap in the hits counts as 0), i.e. u is a
        separate mode, not a shoulder of v. Returns a selection mask."""
        n = bins.shape[0]
        sel = np.zeros(n, np.bool_)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and bins[j + 1] - bins[j] <= gap_bins:
                j += 1
            m = j - i + 1
            idx = np.argsort(-scores[i:j + 1])
            taken = np.zeros(m, np.bool_)
            kept = np.empty(m, np.int64)
            n_kept = 0
            for t in range(m):
                u = idx[t]
                if taken[u]:
                    continue
                ok = True
                if dip < 1.0:
                    su = scores[i + u]
                    for q in range(n_kept):
                        v = kept[q]
                        a = min(u, v)
                        b = max(u, v)
                        low = su
                        for w in range(a, b):
                            if bins[i + w + 1] - bins[i + w] > 1:
                                low = 0.0
                                break
                            if scores[i + w + 1] < low:
                                low = scores[i + w + 1]
                        if low >= dip * su:
                            ok = False
                            break
                if not ok:
                    taken[u] = True
                    continue
                sel[i + u] = True
                kept[n_kept] = u
                n_kept += 1
                bu = bins[i + u]
                for v in range(m):
                    if abs(bins[i + v] - bu) <= sup_bins:
                        taken[v] = True
            i = j + 1
        return sel

    @nb.njit(nogil=True, cache=False)
    def refine_summits(pos, cell, n_cells, centres, radius, h):
        """For each centre, the position within +-radius whose +-h window holds
        the most distinct cells (median of ties)."""
        occ = np.zeros(n_cells, np.int32)
        out = np.empty(centres.shape[0], np.int64)
        n = pos.shape[0]
        for q in range(centres.shape[0]):
            c0 = centres[q]
            lo_i = np.searchsorted(pos, c0 - radius)
            hi_i = np.searchsorted(pos, c0 + radius)
            if hi_i <= lo_i:
                out[q] = c0
                continue
            best = -1
            best_first = c0
            best_last = c0
            # candidate summits: distinct insertion positions in range
            a = np.searchsorted(pos, pos[lo_i] - h)
            b = a
            k_cur = 0
            t = lo_i
            while t < hi_i:
                x = pos[t]
                while b < n and pos[b] <= x + h:
                    cc = cell[b]
                    if occ[cc] == 0:
                        k_cur += 1
                    occ[cc] += 1
                    b += 1
                while a < b and pos[a] < x - h:
                    cc = cell[a]
                    occ[cc] -= 1
                    if occ[cc] == 0:
                        k_cur -= 1
                    a += 1
                if k_cur > best:
                    best = k_cur
                    best_first = x
                    best_last = x
                elif k_cur == best:
                    best_last = x
                # skip same position
                while t < hi_i and pos[t] == x:
                    t += 1
            # clear occupancy
            for u in range(a, b):
                occ[cell[u]] = 0
            out[q] = (best_first + best_last) // 2
        return out

    _KERNELS = dict(encode_seq=encode_seq, kmer_track=kmer_track, count_kmers=count_kmers,
                    count_genome_kmers=count_genome_kmers, bias_bins=bias_bins, scan=scan,
                    nms=nms, refine_summits=refine_summits, poisson_logsf=poisson_logsf)
    return _KERNELS


# ---------------------------------------------------------------------------
# Insertion loading
# ---------------------------------------------------------------------------

def _load_insertions(fragment_file, barcodes, chroms, n_jobs):
    """Read fragments once; return ``{chrom: sorted unique int64 keys}`` with
    key = pos << (CELL+SIDE) | cell << SIDE | side, deduplicated on
    (pos, cell, side), and per-cell insertion counts (dedup on pos, cell)."""
    from epione.pp._fragment_io import BarcodeLookup, map_fragment_chunks, release_free_memory

    lookup = BarcodeLookup([str(b) for b in barcodes])
    chrom_index = {c: i for i, c in enumerate(chroms)}
    pieces = [[] for _ in chroms]
    shift = _CELL_BITS + _SIDE_BITS

    def fn(chunk):
        if len(chunk) == 0:
            return None
        row = lookup(chunk.barcode_names)[chunk.barcode]
        lut = np.asarray([chrom_index.get(c, -1) for c in chunk.chrom_names], np.int64)
        ci = lut[chunk.chrom] if len(lut) else np.full(len(chunk), -1)
        keep = (row >= 0) & (ci >= 0)
        if not keep.any():
            return None
        row = row[keep]; ci = ci[keep]
        s = chunk.start[keep]; e = chunk.end[keep] - 1
        out = {}
        for c in np.unique(ci):
            m = ci == c
            r = row[m].astype(np.int64)
            ks = (s[m] << shift) | (r << _SIDE_BITS)
            ke = (e[m] << shift) | (r << _SIDE_BITS) | 1
            out[int(c)] = np.concatenate([ks, ke])
        return out

    for res in map_fragment_chunks(fragment_file, fn, n_jobs=n_jobs):
        if res is None:
            continue
        for c, arr in res.items():
            pieces[c].append(arr)
    keys = {}

    def finish(ci):
        if not pieces[ci]:
            return ci, np.zeros(0, np.int64)
        arr = np.concatenate(pieces[ci])
        pieces[ci] = None
        arr.sort()
        if arr.size:
            keep = np.empty(arr.size, bool)
            keep[0] = True
            np.not_equal(arr[1:], arr[:-1], out=keep[1:])
            arr = arr[keep]
        return ci, arr

    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
        for ci, arr in pool.map(finish, range(len(chroms))):
            keys[chroms[ci]] = arr
    release_free_memory()
    return keys


def _split_keys(keys):
    shift = _CELL_BITS + _SIDE_BITS
    pos = keys >> shift
    cell = ((keys >> _SIDE_BITS) & ((1 << _CELL_BITS) - 1)).astype(np.int32)
    side = (keys & 1).astype(np.uint8)
    return pos, cell, side


def _dedup_pos_cell(pos, cell, side=None):
    """Keys are sorted by (pos, cell, side): drop the second side of equal
    (pos, cell) pairs so each cell counts once per base."""
    if pos.size == 0:
        return pos, cell, side
    keep = np.empty(pos.size, bool)
    keep[0] = True
    keep[1:] = (pos[1:] != pos[:-1]) | (cell[1:] != cell[:-1])
    return pos[keep], cell[keep], (None if side is None else side[keep])


def _g_table(depths, n_grid=3000):
    """Tabulate g(m) = sum_c (1 - exp(-m d_c / D)) on a log grid of m."""
    d = np.asarray(depths, np.float64)
    d = d[d > 0]
    D = d.sum()
    if D <= 0:
        return np.log(np.geomspace(1e-6, 1e9, n_grid)), np.zeros(n_grid)
    # compress depths into log bins for speed (weights = cell counts)
    edges = np.geomspace(d.min() * 0.999, d.max() * 1.001, 1025)
    idx = np.clip(np.searchsorted(edges, d) - 1, 0, 1023)
    w = np.bincount(idx, minlength=1024).astype(np.float64)
    sumd = np.bincount(idx, weights=d, minlength=1024)
    ok = w > 0
    dm = sumd[ok] / w[ok]
    w = w[ok]
    grid = np.geomspace(1e-6, max(D, 1.0) * 10, n_grid)
    g = np.empty(n_grid)
    for i in range(0, n_grid, 256):
        m = grid[i:i + 256, None]
        g[i:i + 256] = (w[None, :] * -np.expm1(-m * dm[None, :] / D)).sum(1)
    return np.log(grid), g


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_peaks(
    data,
    *,
    groupby: Optional[Union[str, Sequence]] = None,
    barcodes: Optional[Sequence[str]] = None,
    fasta: Optional[Union[str, os.PathLike]] = None,
    chrom_sizes: Optional[dict] = None,
    window: Union[int, Sequence[int]] = 300,
    step: int = 10,
    background: Sequence[int] = (10000,),
    kmer: Optional[int] = 7,
    statistic: str = "cells",
    null: str = "poisson",
    min_cells: int = 3,
    min_score: float = 2.0,
    half_width: int = 250,
    summit_half_width: int = 50,
    dip: float = 0.0,
    qvalue: float = 0.05,
    min_group_cells: int = 20,
    blacklist: Optional[Union[str, os.PathLike]] = None,
    key_added: str = "cellscan",
    n_jobs: Optional[int] = None,
    return_all: bool = False,
    inplace: bool = True,
    verbose: bool = True,
):
    """Call peaks with CellScan (cells as replicates, bias-corrected scan).

    Parameters
    ----------
    data
        An AnnData whose ``uns['files']['fragments']`` points at a
        fragments file (cells = ``obs_names``), or a fragments path
        (then pass ``barcodes``).
    groupby
        ``None`` scans all cells together; an ``obs`` column (or a
        per-cell label sequence) scans each group with its own cells and
        depths and merges the group peaks by score.
    fasta, chrom_sizes
        Genome FASTA (enables the Tn5 bias model) and chromosome sizes
        (default: from the FASTA index). Chromosomes absent from
        ``chrom_sizes`` are ignored.
    window, step
        Scan window width(s) and grid step in bp (rounded to a multiple
        of ``2 * step``). A sequence gives a variable-window scan: each
        grid point keeps its most significant scale, with a
        ``log10(n_scales)`` Bonferroni penalty on the score.
    background
        Local background window sizes (bp); the genome-wide rate is
        always included. ``()`` keeps only the genome-wide rate.
    kmer
        k-mer length of the Tn5 bias model (odd; ``None`` disables bias
        correction).
    statistic
        ``'cells'`` (distinct cells, default) or ``'insertions'``
        (cell-deduplicated insertion count with a Poisson null; the
        pseudobulk-style ablation).
    null
        ``'poisson'`` (default) or ``'nb'``: negative binomial with
        ``Var = mu + phi mu^2``, ``phi`` fitted by moments on background
        windows sampled every 500 bp (window-to-window overdispersion that
        the local background does not absorb).
    min_cells, min_score
        Minimum statistic and minimum ``-log10 p`` of a grid window to
        enter a candidate region.
    half_width
        Output peaks are ``[summit - half_width, summit + half_width)``;
        also the non-maximum-suppression radius.
    summit_half_width
        Half-width of the distinct-cell kernel used to place summits.
    dip
        Mode separation for several maxima in one candidate region: a
        weaker maximum is reported only if the scan drops below
        ``dip * its score`` between it and every stronger one (``1`` =
        plain non-maximum suppression, ``0`` = one summit per region).
    qvalue
        BH threshold of the returned peak set (``return_all=True`` keeps
        every candidate with its q-value).
    min_group_cells
        Groups with fewer cells are skipped.

    Returns
    -------
    DataFrame (``inplace=False``) with ``chrom, start, end, summit,
    score (-log10 p), qValue (-log10 q), n_cells, n_insertions, expected,
    group``; otherwise stored in ``adata.uns[key_added]``.
    """
    k = _kernels()
    adata = None
    if isinstance(data, (str, os.PathLike)):
        frag = str(data)
        if barcodes is None:
            raise ValueError("pass barcodes= when data is a fragments path")
        barcodes = [str(b) for b in barcodes]
    else:
        adata = data
        frag = adata.uns.get("files", {}).get("fragments")
        if frag is None:
            raise ValueError("adata.uns['files']['fragments'] missing")
        barcodes = [str(b) for b in (barcodes if barcodes is not None else adata.obs_names)]
    n_cells = len(barcodes)
    if n_cells >= (1 << _CELL_BITS):
        raise ValueError("too many cells for one call")

    if groupby is None:
        labels = np.zeros(n_cells, np.int64)
        group_names = ["all"]
    else:
        # list() also works for the lazy obs columns of backed (AnnDataOOM / snapATAC2) objects
        lab = (np.asarray([str(g) for g in adata.obs[groupby]]) if isinstance(groupby, str)
               else np.asarray([str(g) for g in groupby]))
        if len(lab) != n_cells:
            raise ValueError("groupby length does not match the cells")
        group_names, labels = np.unique(lab, return_inverse=True)
        group_names = [str(g) for g in group_names]

    if chrom_sizes is None:
        if fasta is None:
            raise ValueError("pass chrom_sizes= or fasta=")
        import pysam
        with pysam.FastaFile(str(fasta)) as fa:
            chrom_sizes = dict(zip(fa.references, fa.lengths))
    chroms = list(chrom_sizes)
    if null not in ("poisson", "nb"):
        raise ValueError("null must be 'poisson' or 'nb'")
    if statistic not in ("cells", "insertions"):
        raise ValueError("statistic must be 'cells' or 'insertions'")
    distinct = statistic == "cells"
    from epione.pp._fragment_io import resolve_n_jobs
    n_jobs = resolve_n_jobs(n_jobs)
    windows = [window] if np.isscalar(window) else list(window)
    hw_list = [max(1, int(round(w / (2 * step)))) for w in windows]
    hw_bins = max(hw_list)
    bg_hw = np.asarray([max(1, int(round(L / (2 * step)))) for L in background], np.int64)
    scale_pen = float(np.log10(len(hw_list)))
    use_bias = fasta is not None and kmer is not None
    if use_bias and kmer % 2 == 0:
        raise ValueError("kmer must be odd")

    if verbose:
        console.level1(f"CellScan: {n_cells:,} cells, {len(group_names)} group(s), "
                       f"window(s) {[2 * h * step for h in hw_list]} bp, bias={'k=%d' % kmer if use_bias else 'off'}")
    keys = _load_insertions(frag, barcodes, chroms, n_jobs)
    chroms = [c for c in chroms if keys[c].size]

    # Per-cell depth (insertions after (pos, cell) dedup) per chromosome.
    depth = np.zeros(n_cells, np.int64)
    for c in chroms:
        pos, cell, _ = _split_keys(keys[c])
        _, cell_d, _ = _dedup_pos_cell(pos, cell)
        depth += np.bincount(cell_d, minlength=n_cells)
    groups = []
    for gi, g in enumerate(group_names):
        members = labels == gi
        if members.sum() < min_group_cells:
            if verbose:
                console.level2(f"  skip group {g!r} ({int(members.sum())} cells)")
            continue
        groups.append((gi, g, members))

    fa = None
    if use_bias:
        import pysam
        fa = pysam.FastaFile(str(fasta))
        left = kmer // 2
        n_k = 4 ** kmer

    import threading
    fa_lock = threading.Lock()

    def seq_of(c):
        with fa_lock:                     # pysam.FastaFile is not thread-safe
            txt = fa.fetch(c, 0, chrom_sizes[c])
        return k["encode_seq"](np.frombuffer(txt.encode("ascii"), np.uint8))

    def group_arrays(c, members):
        pos, cell, side = _split_keys(keys[c])
        if members is not None:
            m = members[cell]
            pos, cell, side = pos[m], cell[m], side[m]
        return pos, cell, side

    def run_scan(c, members, glog, gval, genome_rate, cum_b, n_bins, min_sc, sample_every=0):
        pos, cell, side = group_arrays(c, members)
        pos, cell, _ = _dedup_pos_cell(pos, cell, side)
        if pos.size == 0:
            return None
        cnt = np.bincount(pos // step, minlength=n_bins)[:n_bins]
        cum_n = np.concatenate([[0.0], np.cumsum(cnt, dtype=np.float64)])
        outs = []
        for si, hw in enumerate(hw_list):
            # Bonferroni over scales: a scale's hit must reach min_sc + log10(n_scales)
            res = k["scan"](pos, cell, n_cells, step, hw, n_bins, cum_n, cum_b, bg_hw,
                            genome_rate, glog, gval, float(min_sc) + scale_pen, distinct,
                            int(min_cells), int(sample_every) if si == 0 else 0)
            outs.append(res)
        if len(outs) == 1:
            return pos, cell, outs[0]
        # variable-window scan (Kulldorff 1997): per grid point keep the most
        # significant scale, penalised by log10(number of scales)
        cat = [np.concatenate([o[j] for o in outs]) for j in range(5)]
        cat[1] = cat[1] - np.float32(scale_pen)
        order = np.lexsort((-cat[1], cat[0]))
        b = cat[0][order]
        first = np.r_[True, b[1:] != b[:-1]]
        idx = order[first]
        return pos, cell, tuple(c[idx] for c in cat) + outs[0][5:]

    # ------------------------------------------------------------------
    # Pass 1 (bias model): unbiased scan -> background mask -> k-mer ratios
    # ------------------------------------------------------------------
    pool = ThreadPoolExecutor(max_workers=n_jobs)
    unit_bias = {}

    def base_bins(c):
        n_bins = chrom_sizes[c] // step + 1
        if fa is not None:
            seq = seq_of(c)
        else:
            seq = None
        return n_bins, seq

    r_comb = None
    if use_bias:
        glog_all, gval_all = _g_table(depth)

        def pass1(c):
            n_bins, seq = base_bins(c)
            bb = k["bias_bins"](np.zeros(1, np.int32), seq, np.zeros(0), step, n_bins)
            return c, bb, seq is not None

        tot_b = 0.0
        for c, bb, _ in pool.map(pass1, chroms):
            unit_bias[c] = np.concatenate([[0.0], np.cumsum(bb)])
            tot_b += unit_bias[c][-1]
        rate0 = depth.sum() / max(tot_b, 1.0)

        def pass1b(c):
            n_bins = chrom_sizes[c] // step + 1
            out = run_scan(c, None, glog_all, gval_all, rate0, unit_bias[c], n_bins, 3.0)
            seq = seq_of(c)
            kt = k["kmer_track"](seq, kmer, left)
            if out is None:
                lo = hi = np.zeros(0, np.int64)
                pos = np.zeros(0, np.int64); side = np.zeros(0, np.uint8)
            else:
                bins = out[2][0]
                # exclusion: candidate grid points +-1 kb, merged
                lo = np.maximum(bins * step - 1000, 0)
                hi = bins * step + 1000
                if lo.size:
                    brk = np.r_[True, lo[1:] > hi[:-1]]
                    lo = lo[brk]
                    hi = np.maximum.reduceat(hi, np.flatnonzero(brk))
                pos, _, side = _split_keys(keys[c])
            cs, ce = k["count_kmers"](kt, pos, side, lo.astype(np.int64), hi.astype(np.int64), n_k)
            cg = k["count_genome_kmers"](kt, lo.astype(np.int64), hi.astype(np.int64), n_k)
            return cs, ce, cg

        CS = np.zeros(n_k); CE = np.zeros(n_k); CG = np.zeros(n_k)
        for cs, ce, cg in pool.map(pass1b, chroms):
            CS += cs; CE += ce; CG += cg
        fg = (CG + 1.0) / (CG.sum() + n_k)
        rs = ((CS + 1.0) / (CS.sum() + n_k)) / fg
        re_ = ((CE + 1.0) / (CE.sum() + n_k)) / fg
        r_comb = 0.5 * (rs / (fg * rs).sum() + re_ / (fg * re_).sum())
        if verbose:
            console.level2(f"  Tn5 bias: {kmer}-mer model from {int(CS.sum() + CE.sum()):,} "
                           f"background insertions; b range {r_comb.min():.2f}-{r_comb.max():.2f}")

    # ------------------------------------------------------------------
    # Bias prefix sums per chromosome (pass 2 input)
    # ------------------------------------------------------------------
    def bias_prefix(c):
        n_bins = chrom_sizes[c] // step + 1
        if r_comb is not None:
            seq = seq_of(c)
            kt = k["kmer_track"](seq, kmer, left)
            bb = k["bias_bins"](kt, seq, r_comb, step, n_bins)
            return c, np.concatenate([[0.0], np.cumsum(bb)])
        if c in unit_bias:
            return c, unit_bias[c]
        if fa is not None:
            seq = seq_of(c)
            bb = k["bias_bins"](np.zeros(1, np.int32), seq, np.zeros(0), step, n_bins)
            return c, np.concatenate([[0.0], np.cumsum(bb)])
        # no genome: uniform bias over the chromosome
        bb = np.full(n_bins, float(step))
        bb[-1] = chrom_sizes[c] - (n_bins - 1) * step
        return c, np.concatenate([[0.0], np.cumsum(bb)])

    cum_bias = dict(pool.map(bias_prefix, chroms))
    unit_bias.clear()
    total_bias = sum(v[-1] for v in cum_bias.values())
    eff_genome = total_bias          # ~ number of non-N bases (b has mean 1)

    # ------------------------------------------------------------------
    # Pass 2: scan per group and chromosome
    # ------------------------------------------------------------------
    frames = []
    for gi, g, members in groups:
        glog, gval = _g_table(depth[members])
        D = float(depth[members].sum())
        genome_rate = D / max(total_bias, 1.0)
        mem_arg = None if len(groups) == 1 and members.all() else members

        phi = 0.0
        if null == "nb":
            def sample(c, mem_arg=mem_arg, glog=glog, gval=gval, genome_rate=genome_rate):
                n_bins = chrom_sizes[c] // step + 1
                out = run_scan(c, mem_arg, glog, gval, genome_rate, cum_bias[c], n_bins,
                               np.inf, sample_every=max(1, 500 // step))
                return None if out is None else out[2][5:]
            ss, sl = [], []
            for r in pool.map(sample, chroms):
                if r is not None:
                    ok = r[1] > 0
                    ss.append(r[0][ok].astype(np.float64)); sl.append(r[1][ok].astype(np.float64))
            phi = _fit_dispersion(np.concatenate(ss), np.concatenate(sl))
            if verbose:
                console.level2(f"  NB null: dispersion phi = {phi:.4g} from "
                               f"{sum(len(x) for x in ss):,} background windows")

        def work(c, mem_arg=mem_arg, glog=glog, gval=gval, genome_rate=genome_rate, phi=phi):
            n_bins = chrom_sizes[c] // step + 1
            out = run_scan(c, mem_arg, glog, gval, genome_rate, cum_bias[c], n_bins, min_score)
            if out is None:
                return None
            pos, cell, (bins, sc, kk, nn, ee, _, _) = out
            if bins.size and phi > 0:
                stat = kk if distinct else nn
                sc = _nb_score(stat, ee, phi).astype(np.float32)
                keep = sc >= min_score
                bins, sc, kk, nn, ee = bins[keep], sc[keep], kk[keep], nn[keep], ee[keep]
            if bins.size == 0:
                return None
            sel = k["nms"](bins, sc.astype(np.float64), step, hw_bins, max(1, half_width // step),
                           float(dip))
            bins, sc, kk, nn, ee = bins[sel], sc[sel], kk[sel], nn[sel], ee[sel]
            centres = bins * step
            summits = k["refine_summits"](pos, cell, n_cells, centres.astype(np.int64),
                                          hw_bins * step, int(summit_half_width))
            return pd.DataFrame({"chrom": c, "summit": summits, "score": sc.astype(np.float64),
                                 "n_cells": kk, "n_insertions": nn, "expected": ee})

        parts = [p for p in pool.map(work, chroms) if p is not None]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        # BH with m = effective number of independent windows
        m_tests = max(sum(eff_genome / (2 * h * step) for h in hw_list) / len(hw_list), len(df))
        logp = -df["score"].to_numpy() * np.log(10.0)
        order = np.argsort(logp)
        ranks = np.empty(len(df)); ranks[order] = np.arange(1, len(df) + 1)
        logq = logp + np.log(m_tests) - np.log(ranks)
        # enforce monotonicity (step-up)
        lq_sorted = np.minimum.accumulate(logq[order][::-1])[::-1]
        logq[order] = np.minimum(lq_sorted, 0.0)
        df["qValue"] = -logq / np.log(10.0)
        df["group"] = g
        if verbose:
            n_sig = int((df["qValue"] >= -np.log10(qvalue)).sum())
            console.level2(f"  group {g!r}: {int(members.sum()):,} cells, "
                           f"{len(df):,} candidates, {n_sig:,} at q<={qvalue}")
        frames.append(df)
    pool.shutdown()
    if fa is not None:
        fa.close()

    cols = ["chrom", "start", "end", "summit", "score", "qValue", "n_cells",
            "n_insertions", "expected", "group"]
    if not frames:
        out = pd.DataFrame(columns=cols)
    else:
        allp = pd.concat(frames, ignore_index=True)
        if not return_all:
            allp = allp[allp["qValue"] >= -np.log10(qvalue)]
        out = _nonoverlap(allp, chrom_sizes, half_width)
        if blacklist is not None and len(out):
            from ._call_peaks import _bed_overlap_mask
            out = out.loc[~_bed_overlap_mask(out, str(blacklist))].reset_index(drop=True)
        out = out[cols]
    if verbose:
        console.level2(f"  {len(out):,} peaks")
    if inplace and adata is not None:
        adata.uns[key_added] = out
        return None
    return out


def _fit_dispersion(stat, lam):
    """Moment estimate of phi in Var(X) = mu + phi mu^2 from background
    windows (windows with Poisson -log10 p >= 2 are dropped as signal)."""
    from scipy.stats import poisson
    keep = poisson.sf(stat - 1, lam) > 1e-2
    s, l = stat[keep], lam[keep]
    if len(s) < 100:
        return 0.0
    return float(max(0.0, (((s - l) ** 2 - s).sum()) / (l ** 2).sum()))


def _nb_score(stat, mu, phi):
    """-log10 P(X >= stat), X ~ NB(mean mu, Var mu + phi mu^2)."""
    from scipy.stats import nbinom
    r = 1.0 / phi
    p = r / (r + np.asarray(mu, np.float64))
    return -nbinom.logsf(np.asarray(stat) - 1, r, p) / np.log(10.0)


def _nonoverlap(df, chrom_sizes, half_width):
    """Summit +- half_width, iterative overlap removal by descending score
    (ties by n_cells), sorted by position (as :func:`merge_peaks`)."""
    df = df.sort_values(["score", "n_cells"], ascending=False, kind="stable").reset_index(drop=True)
    start = (df["summit"].to_numpy() - half_width).clip(min=0)
    size = df["chrom"].map(chrom_sizes).to_numpy()
    end = np.minimum(df["summit"].to_numpy() + half_width, size)
    keep = np.zeros(len(df), bool)
    for c, idx in df.groupby("chrom", sort=False).indices.items():
        s = start[idx]; e = end[idx]
        # accepted intervals kept in a sorted array; overlap check by bisection
        acc_s = []
        acc_e = []
        import bisect
        for j, (a, b) in enumerate(zip(s, e)):
            p = bisect.bisect_left(acc_s, a)
            ok = True
            if p < len(acc_s) and acc_s[p] < b:
                ok = False
            if ok and p > 0 and acc_e[p - 1] > a:
                ok = False
            if ok:
                acc_s.insert(p, a); acc_e.insert(p, b)
                keep[idx[j]] = True
    out = df.loc[keep].copy()
    out["start"] = start[keep]
    out["end"] = end[keep]
    return out.sort_values(["chrom", "start"]).reset_index(drop=True)
