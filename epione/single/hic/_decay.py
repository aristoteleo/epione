"""Contact decay of single cells: scHiCluster and the HCEA recipe.

* :func:`contact_decay` — per-cell histogram of cis contact distances in
  log2 bins of 1/8 starting at 2.5 kb, plus the number of off-diagonal pixels
  per chromosome (scHiCluster ``hicluster contact-distance``).
* :func:`decay_ratio` — log2 ratio of short- to long-range contacts per cell
  (HCEA ``fig4/01.decay``; published as ``Short/Long``).
* :func:`compartment_contact_hist` — contacts of each group binned by distance
  and by the compartment scores of their two ends: the difference (A–B
  contacts have a large difference) and the mean (A–A high, B–B low)
  (HCEA ``fig4/02.decay_compartment``).
* :func:`quantile_normalize` — quantile normalisation with tied values mapped
  to the mean of their ranks (the ``qnorm`` package, used by HCEA to put the
  compartment scores of all cell types on one distribution).

Parity is tested in ``tests/parity/schicluster_decay``.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._compartment import CONTACT_COLUMNS, _map, _read_chrom_sizes, _read_contacts

#: HCEA's short (≈ 207 kb – 2.15 Mb) and long (≈ 10.1 – 107.6 Mb) windows, as
#: exclusive bounds on the decay-bin index (bin k starts at 2.5 kb · 2^(k/8)).
SHORT_BINS = (50, 78)
LONG_BINS = (95, 123)


def decay_bins(chrom_sizes) -> np.ndarray:
    """Edges of the contact-distance bins: ``2500 · 2^(k/8)``, up to the largest chromosome."""
    sizes = _read_chrom_sizes(chrom_sizes)
    nbins = np.floor(np.log2(sizes.to_numpy().max() / 2500) / 0.125)
    return 2500 * np.exp2(0.125 * np.arange(nbins + 1))


def _decay_one(args):
    path, columns, chroms, bins, resolution = args
    c1, p1, c2, p2 = (columns[k] for k in ("chrom1", "pos1", "chrom2", "pos2"))
    data = _read_contacts(path, columns)
    data = data.loc[(data[c1] == data[c2]) & data[c1].isin(chroms)]
    hist = np.histogram(np.abs(data[p2] - data[p1]), bins)[0]
    data = data.copy()
    data[[p1, p2]] = data[[p1, p2]] // resolution
    data = data.groupby(by=[c1, p1, p2])[c2].count().reset_index()
    sparsity = data.loc[data[p1] != data[p2], c1].value_counts()
    return hist, sparsity


def contact_decay(
    adata,
    *,
    chrom_sizes=None,
    resolution: int = 10_000,
    nproc: int = 1,
    progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Contact-distance histogram of every cell (scHiCluster ``contact-distance``).

    Cis contacts on the chromosomes of ``chrom_sizes`` are binned by
    ``|pos2 - pos1|`` into log2 bins of 1/8 that start at 2.5 kb
    (:func:`decay_bins`); shorter contacts are not counted. All contacts of
    the table are used (no de-duplication or distance filter).

    Arguments:
        adata: cells from :func:`load_contact_tables` (the histogram needs
            exact positions, so binned cools are not accepted).
        chrom_sizes: chromosomes to use and their sizes; the largest sets the
            number of bins. Default: ``adata.uns['hic']['chrom_sizes']``.
        resolution: bin size for the sparsity count.
        nproc: worker processes.
        progress: show a progress bar.

    Returns:
        ``(decay, sparsity)``: ``decay`` is cells × distance bins (integer
        counts, columns ``0 … n-1``); ``sparsity`` is cells × chromosomes, the
        number of distinct off-diagonal pixels at ``resolution``.
    """
    if "contact_path" not in adata.obs.columns:
        raise ValueError("contact_decay needs contact tables (load_contact_tables)")
    hic = adata.uns.get("hic", {})
    sizes = _read_chrom_sizes(chrom_sizes if chrom_sizes is not None else hic["chrom_sizes"])
    columns = hic.get("contact_columns", CONTACT_COLUMNS)
    bins = decay_bins(sizes)
    tasks = [(p, columns, list(sizes.index.astype(str)), bins, int(resolution))
             for p in adata.obs["contact_path"].astype(str)]
    hists, spars = _map(_decay_one, tasks, nproc, progress, desc="decay")
    decay = pd.DataFrame(np.vstack(hists), index=adata.obs_names)
    sparsity = pd.DataFrame({c: s for c, s in zip(adata.obs_names, spars)}).T
    sparsity = sparsity.reindex(index=adata.obs_names)
    return decay, sparsity


def decay_ratio(
    decay: pd.DataFrame,
    *,
    short: Tuple[int, int] = SHORT_BINS,
    long: Tuple[int, int] = LONG_BINS,
) -> pd.Series:
    """log2(short / long) contact ratio per cell (HCEA ``Short/Long``).

    Each cell's decay histogram is divided by its total, then averaged over
    the bins strictly between the bounds of ``short`` and of ``long``. The
    defaults are HCEA's: bins 51–77 (≈ 207 kb – 2.15 Mb) and 96–122
    (≈ 10.1 – 107.6 Mb).

    Arguments:
        decay: cells × bins from :func:`contact_decay`.
        short, long: exclusive bin-index bounds.

    Returns:
        ``Series`` of log2 ratios, indexed like ``decay``.
    """
    d = decay / decay.sum(axis=1).values[:, None]
    s = d.loc[:, (d.columns > short[0]) & (d.columns < short[1])].mean(axis=1)
    lng = d.loc[:, (d.columns > long[0]) & (d.columns < long[1])].mean(axis=1)
    return np.log2(s / lng).rename("short_long_ratio")


def quantile_normalize(data):
    """Quantile-normalise the columns of ``data`` (``qnorm.quantile_normalize``).

    Every column is mapped onto the mean of the sorted columns; tied values
    get the mean of the target over the ranks they occupy; NaNs are left as
    they are. Float64 throughout.

    Arguments:
        data: ``DataFrame`` or 2-D array, rows × samples.

    Returns:
        Same type and shape as ``data``.
    """
    is_df = isinstance(data, pd.DataFrame)
    values = np.asarray(data.values if is_df else data, dtype=np.float64)
    cols = np.ascontiguousarray(values.T)
    n_cols, n_rows = cols.shape
    target = np.zeros(n_rows)
    for start in range(0, n_cols, 8):          # qnorm sums columns in groups of 8
        total = np.zeros(n_rows)
        for j in range(start, min(start + 8, n_cols)):
            total += np.sort(cols[j])
        target += total
    target = target / max(n_cols, 1)
    out = cols.copy()
    for j in range(n_cols):
        column = out[j]
        idx = np.argsort(column)
        srt = column[idx]
        n = int(np.searchsorted(srt, np.nan))
        if n == 0:
            continue
        v = srt[:n]
        starts = np.r_[True, v[1:] != v[:-1]]
        if starts.all():
            column[idx[:n]] = target[:n]
            continue
        starts = np.flatnonzero(starts)
        counts = np.diff(starts, append=n)
        means = np.add.reduceat(target[:n], starts) / counts
        column[idx[:n]] = np.repeat(means, counts)
    out = out.T
    if is_df:
        return pd.DataFrame(out, index=data.index, columns=data.columns)
    return out


DIST_BINS = np.power(10, np.arange(5, 8.01, 3 / 100))
DIFF_BINS = np.arange(0, 30.01, 30 / 50)
SUM_BINS = np.arange(-15, 15.01, 30 / 50)


def _hist_one(args):
    (path, columns, score_by_chrom, resolution, dist_bins, diff_bins, sum_bins) = args
    c1, p1, c2, p2 = (columns[k] for k in ("chrom1", "pos1", "chrom2", "pos2"))
    data = _read_contacts(path, columns)
    data = data.loc[(data[c1] == data[c2]) & data[c1].isin(score_by_chrom.keys())]
    dist = np.abs(data[p1] - data[p2]).to_numpy()
    sl = np.empty(len(data))
    sr = np.empty(len(data))
    for chrom, idx in data.groupby(c1, sort=False).indices.items():
        s = score_by_chrom[chrom]
        bl = (data[p1].to_numpy()[idx] // resolution)
        br = (data[p2].to_numpy()[idx] // resolution)
        sl[idx] = s[np.clip(bl, 0, len(s) - 1)]
        sr[idx] = s[np.clip(br, 0, len(s) - 1)]
        sl[idx[(bl < 0) | (bl >= len(s))]] = np.nan
        sr[idx[(br < 0) | (br >= len(s))]] = np.nan
    keep = ~np.isnan(sl) & ~np.isnan(sr) & (dist > resolution)
    dist, sl, sr = dist[keep], sl[keep], sr[keep]
    hd = np.histogram2d(dist, np.abs(sl - sr), bins=[dist_bins, diff_bins])[0]
    hs = np.histogram2d(dist, (sl + sr) / 2, bins=[dist_bins, sum_bins])[0]
    return hd, hs


def compartment_contact_hist(
    adata,
    scores: pd.DataFrame,
    groupby: str,
    *,
    columns: Optional[Mapping[str, str]] = None,
    dist_bins: np.ndarray = DIST_BINS,
    diff_bins: np.ndarray = DIFF_BINS,
    sum_bins: np.ndarray = SUM_BINS,
    nproc: int = 1,
    progress: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Contacts per group binned by distance and by the compartment scores of their ends.

    For every cis contact farther apart than one bin and with both ends on
    scored bins, HCEA records the distance, the absolute score difference of
    the two ends and their mean, using the score track of the cell's own
    group. Histograms are summed over the cells of each group.

    Arguments:
        adata: cells from :func:`load_contact_tables`.
        scores: per-bin scores with ``chrom``, ``start``, ``end`` and one
            column per group (e.g. :func:`project_compartments`, often after
            :func:`quantile_normalize`); bins missing from the table (filtered)
            are skipped. The bin size is ``end - start`` of the first row.
        groupby: ``adata.obs`` column with each cell's group.
        columns: ``{group: score column}``, default the group name itself.
        dist_bins, diff_bins, sum_bins: histogram edges; the defaults are
            HCEA's (100 log bins over 100 kb–100 Mb; 50 bins over 0–30 and
            −15–15).
        nproc: worker processes (one cell per task).
        progress: show a progress bar.

    Returns:
        ``{group: {"diff": array, "sum": array, "n_cells": int}}``; ``diff`` is
        distance bins × difference bins, ``sum`` distance bins × mean bins.
    """
    hic = adata.uns.get("hic", {})
    ccols = hic.get("contact_columns", CONTACT_COLUMNS)
    res = int(scores["end"].iloc[0] - scores["start"].iloc[0])
    labels = adata.obs[groupby].astype(str)
    out = {}
    for g in labels.unique():
        col = (columns or {}).get(g, g)
        if col not in scores.columns:
            raise KeyError(f"no score column {col!r} for group {g!r}")
        by_chrom = {}
        for chrom, sub in scores.groupby("chrom", sort=False):
            n = int(sub["start"].max() // res) + 1
            s = np.full(n, np.nan)
            s[(sub["start"] // res).to_numpy()] = sub[col].to_numpy(np.float64)
            by_chrom[str(chrom)] = s
        paths = adata.obs.loc[labels == g, "contact_path"].astype(str).tolist()
        tasks = [(p, ccols, by_chrom, res, dist_bins, diff_bins, sum_bins) for p in paths]
        hd, hs = _map(_hist_one, tasks, nproc, progress, desc=f"hist {g}")
        out[g] = {"diff": np.sum(hd, axis=0), "sum": np.sum(hs, axis=0), "n_cells": len(paths)}
    return out
