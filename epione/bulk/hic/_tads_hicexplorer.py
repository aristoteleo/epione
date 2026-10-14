"""TAD-separation score and boundary calling, ported from HiCExplorer 3.7.6.

Port of ``hicexplorer/hicFindTADs.py`` (deeptools/HiCExplorer, GPL-3.0,
Copyright (c) Fidel Ramírez, Joachim Wolff and contributors; Ramírez et al.,
Nat Commun 2018). Used by ``insulation(method="hicexplorer")``,
``tad_boundaries`` on its output and ``tad_domains``.

The algorithm and its quirks are reproduced as released, because the point of
the method switch is to give the numbers HiCExplorer users already have:

* the score is the mean z-score of the "diamond" between the left and right
  flanks of each bin, at window sizes ``min_depth + int(step * k ** 1.5)``;
* bins whose diamond is undefined at any window are dropped from the table;
* a consensus minimum is a ``peakdetect`` minimum of the across-window mean;
  its ``delta`` is the mean of the scores in a window around it *including* the
  minimum and its three right neighbours (the source comment says excluded);
* chromosome ends are added to the candidate list through an aliased array,
  so chromosome starts never become boundaries and every first domain starts
  at the first called boundary.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

from ._hicmatrix import HiCMat, enlarge_bins


def incremental_steps(min_win, max_win, step):
    out = []
    k = -1
    while True:
        k += 1
        inc = min_win + int(step * (k ** 1.5))
        if k > 1 and inc == out[-1]:
            continue
        if inc > max_win:
            break
        out.append(inc)
    return out


def _idx_at_distance(hm: HiCMat, idx, window_len):
    chrom, cut_start, cut_end, _ = hm.getBinPos(idx)
    left_start = max(0, cut_start - window_len)
    r = hm.getRegionBinRange(chrom, left_start, left_start + 1)
    left_idx = r[0]  # TypeError when None, as in HiCExplorer
    chr_end_pos = hm.get_chromosome_sizes()[chrom]
    right_end = min(chr_end_pos, cut_end + window_len) - 1
    right_idx = hm.getRegionBinRange(chrom, right_end, right_end)[0]
    return left_idx, right_idx


def _cut_weight(hm: HiCMat, cut, window_len, return_mean=False):
    if cut < 0 or cut > hm.matrix.shape[0]:
        return None
    try:
        left_idx, right_idx = _idx_at_distance(hm, cut, window_len)
    except TypeError:
        return None
    block = hm.matrix[left_idx:cut, cut:right_idx]
    if return_mean:
        if block.nnz == 0:
            return 0
        return block.todense().mean()
    return block.todense().A1


def default_depths(binsize):
    if binsize < 1000:
        return binsize * 30, binsize * 60, binsize * 4
    if binsize < 20000:
        return binsize * 10, binsize * 40, binsize * 2
    return binsize * 5, binsize * 10, binsize * 2


def tad_separation_score(hm: HiCMat, min_depth=None, max_depth=None, step=None,
                         chromosomes=None):
    """``HicFindTads.compute_spectra_matrix``; returns (table, zscore matrix, params)."""
    if chromosomes is not None:
        valid = [c for c in chromosomes if c in hm.chrBinBoundaries]
        hm.reorderChromosomes(valid)
    binsize = hm.getBinSize()
    d_min, d_max, d_step = default_depths(binsize)
    if max_depth is None:
        max_depth = d_max
    elif max_depth < binsize * 5:
        raise ValueError("max_depth must be at least 5 times the bin size")
    if min_depth is None:
        min_depth = d_min
    elif min_depth < binsize * 3:
        raise ValueError("min_depth must be at least 3 times the bin size")
    if step is None:
        step = d_step
    elif step < binsize:
        raise ValueError("step must be at least the bin size")
    if max_depth <= min_depth:
        raise ValueError("max_depth must be larger than min_depth")

    hm.diagflat(value=0)
    hm.maskBins(hm.nan_bins)
    orig = hm.cut_intervals[:]
    hm.convert_to_zscore_matrix(maxdepth=max_depth * 2.5, perchr=True)
    new_iv = enlarge_bins(hm.cut_intervals)
    if new_iv != orig:
        hm.set_cut_intervals(new_iv)
    binsize = hm.getBinSize()
    if int(min_depth / binsize) <= 1 or int(max_depth / binsize) <= 1:
        raise ValueError("min_depth / max_depth too small for this bin size")
    if int(step / binsize) == 0:
        raise ValueError(f"step must be larger than {binsize}")
    windows = incremental_steps(min_depth, max_depth, step)
    limit = 2 * int(max_depth / binsize)
    hm.matrix = sparse.triu(hm.matrix, k=0, format="csr") - sparse.triu(hm.matrix, k=limit, format="csr")
    hm.matrix.eliminate_zeros()

    rows, vals = [], []
    for chrom, (a, b) in hm.chrBinBoundaries.items():
        for cut in range(a, b):
            c, s, e, _ = hm.cut_intervals[cut]
            m = [_cut_weight(hm, cut, w, return_mean=True) for w in windows]
            if any(x is None or np.isnan(x) for x in m):
                continue
            vals.append(m)
            rows.append((c, s, e))
    mat = np.vstack(vals)
    table = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    table["start"] = table["start"].astype(int)
    table["end"] = table["end"].astype(int)
    for j, w in enumerate(windows):
        table[f"tad_separation_score_{w}"] = mat[:, j]
    table["tad_separation_score"] = mat.mean(axis=1)
    params = OrderedDict(step=step, minDepth=min_depth, maxDepth=max_depth, binsize=binsize,
                         windows=windows)
    return table, hm, params


def _peakdetect_min(y_axis, lookahead, chrom):
    """Minima from ``HicFindTads.peakdetect`` (Billauer's peakdet)."""
    max_peaks, min_peaks, dump = [], [], []
    x_axis = np.arange(len(y_axis))
    min_y, max_y = np.inf, -np.inf
    max_pos = min_pos = None
    search_for = None
    prev_chrom = None
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], y_axis[:-lookahead])):
        if not (-np.inf < y < np.inf):
            raise ValueError(f"infinite TAD-separation score at position {index}")
        if prev_chrom is None:
            prev_chrom = chrom[index]
        if prev_chrom != chrom[index]:
            min_y, max_y = np.inf, -np.inf
            max_pos = min_pos = None
            search_for = None
        prev_chrom = chrom[index]
        if y > max_y:
            max_y, max_pos = y, x
        if y < min_y:
            min_y, min_pos = y, x
        if y < max_y and max_y != np.inf and search_for != "min":
            if y_axis[index:index + lookahead].max() < max_y:
                max_peaks.append([max_pos, max_y])
                dump.append(True)
                max_y = min_y = y
                min_pos = x
                search_for = "min"
                continue
        if y > min_y and min_y != -np.inf and search_for != "max":
            if y_axis[index:index + lookahead].min() > min_y:
                min_peaks.append([min_pos, min_y])
                dump.append(False)
                min_y = max_y = y
                max_pos = x
                search_for = "max"
    if dump:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
    return min_peaks


def _delta_wrt_window(min_idx_list, avg, chrom, window_len=10):
    _, chr_start_idx = np.unique(chrom, return_index=True)
    chr_start_idx = np.sort(np.concatenate([chr_start_idx, [len(chrom) - 1]]))
    ranges = [(chr_start_idx[x], chr_start_idx[x + 1]) for x in range(len(chr_start_idx) - 1)]
    out = {}
    for mi in min_idx_list:
        near_border = True
        for a, b in ranges:
            if a < mi < b and mi - window_len >= a and mi + window_len < b:
                near_border = False
        if near_border:
            out[mi] = np.nan
        else:
            local = np.concatenate([avg[mi - window_len:mi + 3], avg[mi + 4:mi + window_len]])
            out[mi] = local.mean() - avg[mi]
    return out


def find_boundaries(table: pd.DataFrame, hm: HiCMat, params, *, delta=0.01,
                    threshold_comparisons=0.01, correct_for_multiple_testing="fdr",
                    min_boundary_distance=None) -> pd.DataFrame:
    """``find_boundaries`` + the filter in ``save_domains_and_boundaries``."""
    from scipy.stats import ranksums

    if correct_for_multiple_testing not in ("fdr", "bonferroni", "None", None):
        raise ValueError("correct_for_multiple_testing must be 'fdr', 'bonferroni' or None")
    mtc = "None" if correct_for_multiple_testing is None else correct_for_multiple_testing
    win_cols = [c for c in table.columns if c.startswith("tad_separation_score_")]
    matrix = table[win_cols].to_numpy()
    chrom = table["chrom"].to_numpy().astype(str)
    chr_start = table["start"].to_numpy().astype(int)
    chr_end = table["end"].to_numpy().astype(int)

    avg_bin = np.median(chr_end - chr_start)
    if min_boundary_distance is None:
        min_boundary_distance = avg_bin * 4
    lookahead = int(min_boundary_distance / avg_bin)
    if lookahead < 1:
        raise ValueError("min_boundary_distance must be at least one bin")
    avg = matrix.mean(axis=1)
    mins = _peakdetect_min(avg, lookahead, chrom)
    min_idx = [m[0] for m in mins]
    delta_of_min = _delta_wrt_window(min_idx, avg, chrom)
    if len(min_idx) == 0:
        raise ValueError("no TAD-separation score minima found; check delta / min_boundary_distance")

    window_len = params["minDepth"]
    pvals, kept = [], []
    for idx in min_idx:
        r = hm.getRegionBinRange(chrom[idx], chr_start[idx], chr_end[idx])
        if r is None:
            continue
        midx = r[0]
        kept.append(idx)
        try:
            li, ri = _idx_at_distance(hm, midx, window_len)
            left = _cut_weight(hm, li, window_len)
            right = _cut_weight(hm, ri, window_len)
        except TypeError:
            left = right = None
        boundary = _cut_weight(hm, midx, window_len)
        left = [] if left is None else left
        right = [] if right is None else right
        if len(left) == 0 and len(right) == 0:
            p = np.nan
        elif boundary is None or len(boundary) == 0 or len(left) == 0 or len(right) == 0:
            p = np.nan
        else:
            try:
                p = min(ranksums(boundary, left)[1], ranksums(boundary, right)[1])
            except ValueError:
                p = np.nan
        pvals.append(p)
    p_fdr = None
    if mtc == "fdr":
        pvals = np.array([e if ~np.isnan(e) else 1 for e in pvals])
        srt = sorted(pvals)
        largest = 0
        for i, p in enumerate(srt):
            if p <= threshold_comparisons * (i + 1) / len(srt):
                if p >= largest:
                    largest = p
        p_fdr = largest
    elif mtc == "bonferroni":
        pvals = np.array(pvals) * len(pvals)
        over = np.array([e > 1 if ~np.isnan(e) else False for e in pvals])
        if len(over) > 0:
            pvals[over] = 1
    pvalue_of_min = OrderedDict(zip(kept, pvals))

    # save_domains_and_boundaries: chromosome ends via the aliased index array
    _, chr_start_idx = np.unique(chrom, return_index=True)
    if len(np.unique(chrom)) == 1:
        chr_start_idx = [0]
        chr_end_idx = [len(chrom) - 1]
    else:
        chr_end_idx = chr_start_idx
        chr_end_idx[chr_end_idx == 0] = len(chrom)
        chr_end_idx -= 1
    cand = np.sort(np.concatenate([chr_start_idx, chr_end_idx, min_idx]).astype(int))
    thr = p_fdr if mtc == "fdr" else threshold_comparisons
    filtered = []
    for idx in cand:
        if idx not in delta_of_min:
            delta_of_min[idx] = np.nan
        if delta_of_min[idx] >= delta and idx in pvalue_of_min and pvalue_of_min[idx] <= thr:
            filtered.append(int(idx))
    chr_end_set = set(int(x) for x in np.atleast_1d(chr_end_idx))

    recs = []
    for k, b in enumerate(filtered):
        if b in chr_end_set:
            continue
        right_c = chr_start[b] + int((chr_end[b] - chr_start[b]) / 2)
        left_c = chr_start[b - 1] + int((chr_end[b - 1] - chr_start[b - 1]) / 2)
        if chrom[b] != chrom[b - 1]:
            continue
        nxt = None
        if k + 1 < len(filtered) and chrom[b] == chrom[filtered[k + 1]]:
            nxt = int(chr_start[filtered[k + 1]])
        recs.append((chrom[b], int(left_c), int(right_c), f"B{b:05d}", float(avg[b]),
                     float(delta_of_min[b]), float(pvalue_of_min[b]), int(chr_start[b]), nxt))
    out = pd.DataFrame(recs, columns=["chrom", "start", "end", "name", "tad_separation_score",
                                      "delta", "pvalue", "bin_start", "next_boundary_start"])
    out.attrs["hicexplorer"] = {"pvalue_threshold": thr, "correct_for_multiple_testing": mtc,
                                "delta": delta}
    return out


def domains_from_boundaries(boundaries: pd.DataFrame) -> pd.DataFrame:
    """``_domains.bed``: from each boundary's bin start to the next boundary's."""
    b = boundaries[boundaries["next_boundary_start"].notna()]
    out = pd.DataFrame({
        "chrom": b["chrom"].to_numpy(),
        "start": b["bin_start"].astype(int).to_numpy(),
        "end": b["next_boundary_start"].astype(int).to_numpy(),
        "tad_separation_score": b["tad_separation_score"].to_numpy(),
    })
    out.insert(3, "name", [f"ID_{boundaries.attrs.get('hicexplorer', {}).get('delta', '')}_{i + 1}"
                           for i in range(len(out))])
    return out.reset_index(drop=True)
