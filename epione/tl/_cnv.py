"""Copy-number variation (CNV) calling from scATAC-seq.

``method="epianeufinder"`` is a port of the R package epiAneufinder 1.1.5
(Ramakrishnan, Symeonidi, Hanel et al., *Nat. Commun.* 14, 5846 (2023);
github.com/colomemaria/epiAneufinder, GPL-3). The pipeline is:

1. tile the genome into fixed windows, drop windows overlapping a blacklist and
   windows with >= 0.1 % N, and annotate each window with its GC content;
2. count fragments per window and cell, then drop cells with too few covered
   windows and windows that are empty in most cells;
3. correct each cell for GC bias with a LOESS fit of counts on GC content;
4. find breakpoints per cell and chromosome by recursive binary segmentation
   with the Anderson-Darling (default) or Kolmogorov-Smirnov distance;
5. keep the breakpoints whose distance has a positive z-score within the cell;
6. call each segment loss (0), normal (1) or gain (2) from its trimmed mean
   relative to the genome-wide mean.

The numerics follow the R code step by step: the AD statistic is accumulated
in the same order, R's ``quantile(type = 7)``, ``mean`` (long-double sum plus a
refinement pass), ``scale`` and ``round`` (half to even) are reproduced, and the
subclone split uses a port of R's ``hclust`` (Murtagh's nearest-neighbour
algorithm, ``ward.D``) and ``cutree``. The GC correction uses scikit-misc's
``loess``, which wraps the same netlib ``dloess`` code as ``stats::loess``.

Coordinates: the R package reads the start column of a fragments file as a
1-based coordinate, so a fragment ``(start, end)`` covers ``[start, end]`` and a
window ``k`` (0-based) covers ``[k*w + 1, (k+1)*w]``. The same convention is
used here so that counts agree with R exactly. ``var['start']`` is reported in
0-based BED convention (``k*w``).
"""
from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba ships with scanpy
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

__all__ = [
    "cnv",
    "cnv_make_windows",
    "cnv_count_matrix",
    "cnv_subclones",
    "cnv_karyotype_measures",
    "cnv_burden",
]

_STD_CHROM = re.compile(r"^(chr)?([0-9]+|X|Y|M|MT)$")


def _chrom_sort_key(c: str):
    m = _STD_CHROM.match(c)
    core = m.group(2)
    if core.isdigit():
        return (0, int(core))
    return (1, {"X": 0, "Y": 1, "M": 2, "MT": 2}[core])


def _standard_chroms(names) -> list:
    """``keepStandardChromosomes`` + ``sortSeqlevels`` for UCSC/NCBI style names."""
    return sorted([c for c in names if _STD_CHROM.match(c)], key=_chrom_sort_key)


# ---------------------------------------------------------------------------
# R numerics
# ---------------------------------------------------------------------------

def _r_mean(x: np.ndarray) -> float:
    """``mean()`` of a double vector in R: long-double sum / n, then one
    refinement pass ``s += sum(x - s) / n`` in long double."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return float("nan")
    ld = np.longdouble
    s = np.cumsum(x.astype(ld))[-1]
    if not np.isfinite(s):
        return float(s)
    s = s / ld(n)
    t = np.cumsum(x.astype(ld) - s)[-1]
    s = s + t / ld(n)
    return float(s)


def _r_mean_int(x: np.ndarray) -> float:
    """``mean()`` of an integer vector in R: long-double sum / n."""
    x = np.asarray(x)
    ld = np.longdouble
    return float(np.cumsum(x.astype(ld))[-1] / ld(x.size))


def _r_colmean(x: np.ndarray) -> float:
    """``colMeans`` (used by ``scale``): long-double sum / n, no refinement."""
    ld = np.longdouble
    return float(np.cumsum(np.asarray(x, dtype=np.float64).astype(ld))[-1] / ld(len(x)))


def _r_scale(x: np.ndarray) -> np.ndarray:
    """``scale(x, center=TRUE, scale=TRUE)`` for one column."""
    x = np.asarray(x, dtype=np.float64)
    c = x - _r_colmean(x)
    ld = np.longdouble
    ss = float(np.cumsum((c * c).astype(ld))[-1])
    sc = math.sqrt(ss / max(1, x.size - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return c / sc


def _r_quantile7(x: np.ndarray, probs) -> np.ndarray:
    """``quantile(x, probs, type = 7)`` with R's arithmetic."""
    xs = np.sort(np.asarray(x, dtype=np.float64))
    n = xs.size
    out = np.empty(len(probs))
    for k, p in enumerate(probs):
        index = 1.0 + max(n - 1, 0) * p
        lo = math.floor(index)
        hi = math.ceil(index)
        qs = xs[lo - 1]
        if index > lo and xs[hi - 1] != qs:
            h = index - lo
            qs = (1.0 - h) * qs + h * xs[hi - 1]
        out[k] = qs
    return out


# ---------------------------------------------------------------------------
# Step 1 - windows
# ---------------------------------------------------------------------------

def _read_bed3(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], comment="#",
                     dtype={0: str, 1: np.int64, 2: np.int64})
    df.columns = ["chrom", "start", "end"]
    return df


def _fasta_path(genome) -> str:
    if genome is None:
        raise ValueError("`genome` (a FASTA path or an epione Genome) is required for GC content")
    if hasattr(genome, "fasta"):
        return str(genome.fasta)
    return str(genome)


def cnv_make_windows(
    genome,
    blacklist=None,
    window_size: int = 100_000,
    exclude: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Tile the standard chromosomes into windows and annotate GC content.

    Port of epiAneufinder ``makeWindows``: ``tileGenome(tilewidth=window_size,
    cut.last.tile.in.chrom=TRUE)`` on the standard chromosomes, minus every
    window that overlaps a blacklist interval, with ``GC``, ``AT`` and ``N``
    fractions per window.

    Parameters
    ----------
    genome
        Path to a FASTA file (indexed or indexable by pyfaidx) or an
        :class:`epione.core.genome.Genome`.
    blacklist
        BED file (or DataFrame with ``chrom, start, end``) of excluded regions.
    window_size
        Window width in bp.
    exclude
        Chromosomes to drop, e.g. ``["chrX", "chrY", "chrM"]``.

    Returns
    -------
    DataFrame with ``chrom, start`` (0-based), ``end``, ``tile`` (0-based tile
    index within the chromosome), ``GC``, ``AT``, ``N``.
    """
    from pyfaidx import Fasta

    w = int(window_size)
    fa = Fasta(_fasta_path(genome), as_raw=True, sequence_always_upper=False)
    chroms = [c for c in _standard_chroms(list(fa.keys())) if c not in set(exclude or ())]
    if blacklist is None:
        bl = pd.DataFrame(columns=["chrom", "start", "end"])
    elif isinstance(blacklist, pd.DataFrame):
        bl = blacklist[["chrom", "start", "end"]].copy()
    else:
        bl = _read_bed3(blacklist)
    out = []
    for c in chroms:
        L = len(fa[c])
        nt = (L + w - 1) // w
        tile = np.arange(nt, dtype=np.int64)
        ws1 = tile * w + 1                   # 1-based start
        we = np.minimum((tile + 1) * w, L)   # 1-based end
        keep = np.ones(nt, dtype=bool)
        b = bl[bl["chrom"] == c]
        if len(b):
            bs1 = b["start"].to_numpy(np.int64) + 1  # read_bed: 0-based -> 1-based
            be = b["end"].to_numpy(np.int64)
            o = np.argsort(bs1, kind="stable")
            bs1, be = bs1[o], be[o]
            pref_max_end = np.maximum.accumulate(be)
            idx = np.searchsorted(bs1, we, side="right")  # bl with start1 <= window end
            has = idx > 0
            ov = np.zeros(nt, dtype=bool)
            ov[has] = pref_max_end[idx[has] - 1] >= ws1[has]
            keep &= ~ov
        if not keep.any():
            continue
        seq = np.frombuffer(str(fa[c][:]).upper().encode("ascii"), dtype=np.uint8)
        bounds = (tile * w).astype(np.int64)
        gc = np.add.reduceat(((seq == 71) | (seq == 67)).astype(np.int64), bounds)
        at = np.add.reduceat(((seq == 65) | (seq == 84)).astype(np.int64), bounds)
        width = (we - ws1 + 1).astype(np.float64)
        GC = gc / width
        AT = at / width
        df = pd.DataFrame({
            "chrom": c, "start": ws1 - 1, "end": we, "tile": tile,
            "GC": GC, "AT": AT, "N": 1.0 - (GC + AT),
        })[keep]
        out.append(df)
    win = pd.concat(out, ignore_index=True)
    win["chrom"] = pd.Categorical(win["chrom"], categories=[c for c in chroms if c in set(win["chrom"])])
    return win


# ---------------------------------------------------------------------------
# Step 2 - counts
# ---------------------------------------------------------------------------

def _tile_lookup(windows: pd.DataFrame):
    """Per chromosome, an array tile -> window row (-1 when the tile was removed)."""
    look = {}
    for c, sub in windows.groupby("chrom", observed=True):
        t = sub["tile"].to_numpy()
        arr = np.full(int(t.max()) + 1, -1, dtype=np.int64)
        arr[t] = sub.index.to_numpy()
        look[c] = arr
    return look


def _hits(chrom_codes, cats, starts, ends, look, w):
    """Window hits of intervals ``[start, end]`` (1-based closed, as in R)."""
    rows_all, win_all = [], []
    for ci, c in enumerate(cats):
        if c not in look:
            continue
        sel = np.flatnonzero(chrom_codes == ci)
        if sel.size == 0:
            continue
        arr = look[c]
        s = starts[sel]
        e = ends[sel]
        lo = np.maximum(s - 1, 0) // w
        hi = np.maximum(e - 1, 0) // w
        span = int((hi - lo).max()) if sel.size else 0
        for off in range(span + 1):
            t = lo + off
            ok = (t <= hi) & (t < arr.size)
            wi = np.full(sel.size, -1, dtype=np.int64)
            wi[ok] = arr[t[ok]]
            m = wi >= 0
            rows_all.append(sel[m])
            win_all.append(wi[m])
    if not rows_all:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(rows_all), np.concatenate(win_all)


def _count_fragments(path, windows, w, min_frags, chunksize=10_000_000):
    look = _tile_lookup(windows)
    n_win = len(windows)
    bc_index: dict = {}
    bc_names: list = []
    n_rows = []                 # rows per barcode (all chromosomes)
    mats = []
    first = {}                  # barcode id -> (window, start, end, row) of first hit (R column order)
    row_offset = 0
    reader = pd.read_csv(path, sep="\t", header=None, comment="#", usecols=[0, 1, 2, 3],
                         names=["chrom", "start", "end", "barcode"],
                         dtype={"chrom": "category", "start": np.int64, "end": np.int64,
                                "barcode": str},
                         chunksize=chunksize)
    for chunk in reader:
        codes, uniq = pd.factorize(chunk["barcode"], sort=False)
        gid = np.empty(len(uniq), dtype=np.int64)
        for i, b in enumerate(uniq):
            j = bc_index.get(b)
            if j is None:
                j = len(bc_names)
                bc_index[b] = j
                bc_names.append(b)
                n_rows.append(0)
            gid[i] = j
        g = gid[codes]
        cnt = np.bincount(g, minlength=len(bc_names))
        if len(n_rows) < cnt.size:
            n_rows.extend([0] * (cnt.size - len(n_rows)))
        for j in np.flatnonzero(cnt):
            n_rows[j] += int(cnt[j])
        cat = chunk["chrom"].cat
        rows, wins = _hits(cat.codes.to_numpy(), list(cat.categories),
                           chunk["start"].to_numpy(), chunk["end"].to_numpy(), look, w)
        if rows.size:
            bcs = g[rows]
            mats.append(sp.coo_matrix((np.ones(rows.size, dtype=np.int32), (wins, bcs)),
                                      shape=(n_win, len(bc_names))).tocsc())
            st = chunk["start"].to_numpy()[rows]
            en = chunk["end"].to_numpy()[rows]
            o = np.lexsort((rows, en, st, wins, bcs))
            bo = bcs[o]
            firsts = np.flatnonzero(np.r_[True, bo[1:] != bo[:-1]])
            for k in firsts:
                b = int(bo[k])
                key = (int(wins[o[k]]), int(st[o[k]]), int(en[o[k]]), int(rows[o[k]]) + row_offset)
                if b not in first or key < first[b]:
                    first[b] = key
        row_offset += len(chunk)
    nb = len(bc_names)
    M = sp.csc_matrix((n_win, nb), dtype=np.int32)
    for m in mats:
        m.resize((n_win, nb))
        M = M + m
    n_rows = np.asarray(n_rows, dtype=np.int64)
    keep = [b for b in first if n_rows[b] >= min_frags]
    keep.sort(key=lambda b: first[b])
    keep = np.asarray(keep, dtype=np.int64)
    return M[:, keep].T.tocsr(), [bc_names[b] for b in keep], n_rows[keep]


def _count_peak_matrix(adata: AnnData, windows, w):
    look = _tile_lookup(windows)
    names = pd.Index(adata.var_names.astype(str))
    if {"chrom", "start", "end"}.issubset(adata.var.columns):
        chrom = adata.var["chrom"].astype(str).to_numpy()
        s = adata.var["start"].to_numpy(np.int64)
        e = adata.var["end"].to_numpy(np.int64)
    else:
        parts = pd.Series(names).str.replace(":", "-", regex=False).str.split("-", n=2, expand=True)
        chrom = parts[0].to_numpy()
        s = parts[1].astype(np.int64).to_numpy()
        e = parts[2].astype(np.int64).to_numpy()
    codes, cats = pd.factorize(chrom)
    rows, wins = _hits(codes, list(cats), s, e, look, w)
    A = sp.coo_matrix((np.ones(rows.size), (wins, rows)), shape=(len(windows), adata.n_vars)).tocsr()
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    C = (X @ A.T).tocsr()
    return C, list(adata.obs_names), np.asarray(X.sum(1)).ravel()


def cnv_count_matrix(
    fragments,
    windows: pd.DataFrame,
    window_size: int,
    min_frags: int = 20000,
):
    """Count fragments (or peak counts) per window and cell.

    Port of epiAneufinder ``generateCountMatrix`` / ``countInsertions``:
    a fragment adds one count to every window it overlaps (1-based closed
    coordinates, as R reads the file), cells need ``min_frags`` fragment rows,
    and only windows with ``N < 0.001`` are kept.

    Returns ``(counts cells x windows CSR, barcodes, n_fragments, windows)``.
    """
    if isinstance(fragments, AnnData):
        C, bcs, nfr = _count_peak_matrix(fragments, windows, window_size)
    else:
        C, bcs, nfr = _count_fragments(fragments, windows, window_size, min_frags)
    keepw = windows["N"].to_numpy() < 0.001
    return C[:, keepw], bcs, nfr, windows[keepw].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3 - GC correction
# ---------------------------------------------------------------------------

def _gc_correct(x: np.ndarray, gc: np.ndarray) -> np.ndarray:
    try:
        from skmisc.loess import loess
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "GC correction needs scikit-misc (pip install scikit-misc, or "
            "pip install 'epione[cnv]')"
        ) from exc
    xf = x.astype(np.float64)
    lo = loess(gc, xf)  # span 0.75, degree 2, gaussian, interpolate: R defaults
    lo.fit()
    fitted = np.asarray(lo.outputs.fitted_values)
    corr = _r_mean(xf) / fitted  # the R count matrix is double here
    with np.errstate(invalid="ignore", over="ignore"):
        y = np.rint(xf * corr)
    if not np.all(np.isfinite(y)):
        bad = ~np.isfinite(y)
        y[bad] = 0
    return y.astype(np.int64)


# ---------------------------------------------------------------------------
# Step 4 - breakpoints
# ---------------------------------------------------------------------------

@njit(cache=True)
def _seq_dist_ad(codes, D, minsize):
    """``seq_dist_ad`` with ``test='AD'`` for one segment (``codes`` are ranks
    of the values among the segment's distinct values)."""
    L = codes.size
    nb = (L - 1) // minsize + 1
    out = np.zeros(nb)
    cp = np.zeros(D, np.int64)
    cq = np.zeros(D, np.int64)
    for t in range(L):
        cq[codes[t]] += 1
    cp[codes[0]] += 1
    cur = 1  # x = seg[0:cur], y = seg[cur-1:L]
    for k in range(nb):
        i = 1 + k * minsize
        while cur < i:
            cp[codes[cur]] += 1
            cq[codes[cur - 1]] -= 1
            cur += 1
        if D < 2:
            out[k] = 0.0  # R: NaN -> 0
            continue
        n = float(i)
        m = float(L - i + 1)
        N = m + n
        sx = 0.0
        sy = 0.0
        mx = 0.0
        my = 0.0
        for j in range(D - 1):
            mx += cp[j]
            my += cq[j]
            lj = float(cp[j] + cq[j])
            bj = mx + my
            denom = N * (bj * (N - bj))
            ax = (N * mx) - (n * bj)
            ay = (N * my) - (m * bj)
            sx = sx + (lj * (ax * ax)) / denom
            sy = sy + (lj * (ay * ay)) / denom
        v = (sx / n) + (sy / m)
        if v != v:
            v = 0.0
        out[k] = v
    return out


def _seq_dist_ks(seg, minsize):
    """``seq_dist_ad`` with ``test='KS'``: ``stats::ks.test(x, y)$statistic``
    per split. R builds ``z <- cumsum(ifelse(order(w) <= n.x, 1/n.x, -1/n.y))``
    and its ``cumsum`` accumulates in long double, so the same is done here
    (``np.cumsum`` on ``longdouble`` is sequential); this matters because
    exact ties between candidate splits are frequent with count data."""
    seg = np.asarray(seg, dtype=np.float64)
    L = seg.size
    out = []
    for i in range(1, L + 1, minsize):
        x = seg[:i]
        y = seg[i - 1:]
        n, m = x.size, y.size
        w = np.concatenate([x, y])
        o = np.argsort(w, kind="stable")
        steps = np.where(o < n, 1.0 / n, -1.0 / m)
        z = np.cumsum(steps.astype(np.longdouble)).astype(np.float64)
        ws = w[o]
        brk = np.flatnonzero(np.diff(ws) != 0)
        if brk.size + 1 < n + m:
            z = z[np.r_[brk, n + m - 1]]
        out.append(np.abs(z).max())
    return np.asarray(out)


def _codes(x):
    u, inv = np.unique(x, return_inverse=True)
    return inv.astype(np.int64), u.size


def _getbp(seq, k=4, minsize=1, test="AD", minsize_cnv=0):
    """Port of epiAneufinder ``getbp`` for one chromosome of one cell.

    Returns ``(bp, dist)`` with 1-based bin positions ordered by decreasing
    distance (as ``res$bp`` / ``res$ad.distance``).
    """
    seq = np.asarray(seq)
    L = seq.size
    bp: list = []
    distbp: list = []
    for _ in range(k):
        cuts = sorted(set(p for p in bp if 1 <= p <= L))
        starts = [0] + [p - 1 for p in cuts if p - 1 > 0]
        bounds = starts + [L]
        segs = [(bounds[i], bounds[i + 1]) for i in range(len(starts))]
        nb_set = set()
        for b in bp:
            nb_set.update(range(b - minsize_cnv, b + minsize_cnv + 1))
        new_bp, new_d = [], []
        any_sig = False
        cand = []
        for (a, b) in segs:
            if test == "AD":
                codes, D = _codes(seq[a:b])
                d = _seq_dist_ad(codes, D, minsize)
            else:
                d = _seq_dist_ks(seq[a:b], minsize)
            pos = (int(np.argmax(d)) + 1) * minsize
            ln = b - a
            if pos > ln:
                pos = ln
            add = a if bp else 0
            cand.append((pos + add, float(d.max()), (pos + add) not in nb_set))
        for p, dm, sig in cand:
            if sig:
                any_sig = True
                new_bp.append(p)
                new_d.append(dm)
        if any_sig:
            bp.extend(new_bp)
            distbp.extend(new_d)
        else:
            break
    if minsize_cnv != 0:
        ends = set(range(1, minsize_cnv + 1)) | set(range(L - minsize_cnv, L + 1))
    else:
        ends = {1, L}
    pairs = [(b, d) for b, d in zip(bp, distbp) if b not in ends]
    order = sorted(range(len(pairs)), key=lambda i: -pairs[i][1])  # stable
    return [pairs[i][0] for i in order], [pairs[i][1] for i in order]


# ---------------------------------------------------------------------------
# Step 5/6 - pruning and states
# ---------------------------------------------------------------------------

def _prune(bps):
    """``threshold_dist_values``: keep breakpoints with a positive z-score."""
    if not bps:
        return []
    d = np.array([b[2] for b in bps], dtype=np.float64)
    z = _r_scale(d)
    with np.errstate(invalid="ignore"):
        keep = z > 0
    return [b for b, k in zip(bps, keep) if k]


def _assign_gainloss(x, cluster, uq=0.9, lq=0.1, is_int=True):
    """Port of epiAneufinder ``assign_gainloss``."""
    xf = np.asarray(x, dtype=np.float64)
    mu = _r_mean_int(x) if is_int else _r_mean(xf)
    cn = xf / mu
    cn[cn < 0] = 0
    g = _r_quantile7(xf, (0.01, 0.98))  # R uses the raw counts here
    labels = np.unique(cluster)
    cnmean = np.empty(labels.size)
    for li, lab in enumerate(labels):
        v = cn[cluster == lab]
        q = _r_quantile7(v, (lq, uq))
        y = v[(v >= q[0]) & (v <= q[1]) & (v >= g[0]) & (v <= g[1])]
        if y.size == 0 or y.sum() == 0:
            y = v
        cnmean[li] = _r_mean(y)
    z = _r_scale(cnmean)
    with np.errstate(invalid="ignore"):
        sig = (z >= -1) & (z <= 1)  # NA -> no replacement
    cnmean[sig] = _r_mean(cnmean)
    scaled = cnmean / _r_mean(cnmean)
    scaled[scaled > 2] = 2
    states = np.rint(scaled)
    lut = dict(zip(labels.tolist(), states.tolist()))
    return np.array([lut[c] for c in cluster.tolist()], dtype=np.int8)


def _process_cells(args):
    (X, chrom_bounds, chrom_names, gc, gc_correction, k, minsize, test,
     minsize_cnv, uq, lq) = args
    n_bins = X.shape[1]
    states = np.empty((X.shape[0], n_bins), dtype=np.int8)
    corrected = np.empty((X.shape[0], n_bins), dtype=np.int64)
    raw_bp, kept_bp = [], []
    for r in range(X.shape[0]):
        x = X[r]
        y = _gc_correct(x, gc) if gc_correction else x.astype(np.int64)
        corrected[r] = y
        bps = []
        for ci, (a, b) in enumerate(chrom_bounds):
            p, d = _getbp(y[a:b], k=k, minsize=minsize, test=test, minsize_cnv=minsize_cnv)
            bps.extend((ci, pp, dd) for pp, dd in zip(p, d))
        raw_bp.append(bps)
        # R orders by -dist (stable over the chromosome-major list) before pruning
        order = sorted(range(len(bps)), key=lambda i: -bps[i][2])
        pr = _prune([bps[i] for i in order])
        kept_bp.append(pr)
        cluster = np.empty(n_bins, dtype=np.int64)
        off = 0
        for ci, (a, b) in enumerate(chrom_bounds):
            ln = b - a
            cuts = sorted(pp for (cc, pp, _) in pr if cc == ci)
            edges = [1] + cuts + [ln + 1]
            lab = np.repeat(np.arange(1, len(edges)), np.diff(edges))
            cluster[a:b] = lab + off
            off += len(edges) - 1
        # GC-corrected counts are integer in R (as.integer), raw counts are double
        states[r] = _assign_gainloss(y, cluster, uq=uq, lq=lq, is_int=bool(gc_correction))
    return states, corrected, raw_bp, kept_bp


def cnv(
    fragments: Union[str, Path, AnnData],
    genome=None,
    blacklist=None,
    *,
    method: str = "epianeufinder",
    window_size: int = 100_000,
    test: str = "AD",
    exclude: Optional[Sequence[str]] = ("chrX", "chrY", "chrM"),
    min_frags: int = 20000,
    selected_cells: Optional[Sequence[str]] = None,
    threshold_cells_nbins: float = 0.05,
    threshold_blacklist_bins: float = 0.85,
    k: int = 4,
    minsize: int = 1,
    minsize_cnv: int = 0,
    uq: float = 0.9,
    lq: float = 0.1,
    gc_correction: bool = True,
    windows: Optional[pd.DataFrame] = None,
    adata: Optional[AnnData] = None,
    key_added: str = "cnv",
    n_jobs: int = 1,
    verbose: bool = True,
) -> AnnData:
    """Call copy-number gains and losses per cell from scATAC-seq.

    ``method="epianeufinder"`` ports epiAneufinder 1.1.5 (Ramakrishnan et al.,
    *Nat. Commun.* 2023); parameter names and defaults follow the R function
    ``epiAneufinder()`` (``windowSize`` -> ``window_size``, ``minFrags`` ->
    ``min_frags``, ``minsizeCNV`` -> ``minsize_cnv``). The defaults for
    ``exclude`` and ``window_size`` are those of the package's wrapper script.

    Parameters
    ----------
    fragments
        A 10x ``fragments.tsv[.gz]`` (columns chrom, start, end, barcode, ...),
        or a cells x peaks AnnData (peaks named ``chr-start-end`` /
        ``chr:start-end`` or with ``var`` columns chrom/start/end), which R
        epiAneufinder also accepts (with lower sensitivity, see its vignette).
    genome
        FASTA path or :class:`epione.core.genome.Genome` for GC content.
    blacklist
        BED of blacklisted regions (e.g. ENCODE ``hg38-blacklist.v2.bed``).
    method
        Only ``"epianeufinder"``.
    window_size
        Window (bin) size in bp.
    test
        ``"AD"`` (Anderson-Darling, default) or ``"KS"`` (Kolmogorov-Smirnov).
    exclude
        Chromosomes to drop.
    min_frags
        Minimum fragment rows per barcode (fragments input only).
    selected_cells
        Optional barcodes to keep before the cell filter.
    threshold_cells_nbins
        Keep cells with more than this fraction of non-zero windows.
    threshold_blacklist_bins
        Drop a window if at least this fraction of cells have zero counts.
    k
        Recursion depth: at most ``2**k`` segments per chromosome.
    minsize
        Step between tested breakpoints (1 = every bin).
    minsize_cnv
        Minimum distance (bins) between breakpoints.
    uq, lq
        Quantiles for the trimmed segment means.
    gc_correction
        LOESS correction of counts on GC content.
    windows
        Precomputed window table from :func:`cnv_make_windows` (skips step 1).
    adata
        Optional AnnData of the same cells. When given, ``adata.obs`` gets
        ``{key_added}_burden`` and ``{key_added}_subclone``-ready states in
        ``adata.obsm[f"X_{key_added}"]`` (NaN rows for filtered cells) and the
        bin table in ``adata.uns[key_added]``.
    n_jobs
        Worker processes for steps 3-6 (per-cell independent).

    Returns
    -------
    AnnData (cells x windows) with

    * ``X`` - CNV state per window (int8: 0 loss, 1 normal, 2 gain);
    * ``layers['counts']`` - raw counts, ``layers['gc_corrected']`` - corrected;
    * ``var`` - chrom, start (0-based), end, GC, AT, N;
    * ``obs`` - ``n_fragments``, ``cnv_burden`` (fraction of non-normal bins);
    * ``uns['cnv']`` - parameters, breakpoint tables (``breakpoints`` after
      pruning, ``breakpoints_raw`` before), timings.
    """
    if method != "epianeufinder":
        raise ValueError(f"unknown method {method!r}; available: 'epianeufinder'")
    if test not in ("AD", "KS"):
        raise ValueError("test must be 'AD' or 'KS'")
    t0 = time.perf_counter()
    log = print if verbose else (lambda *a, **k: None)
    if windows is None:
        log("[cnv] making windows ...")
        windows = cnv_make_windows(genome, blacklist, window_size, exclude=exclude)
    t_win = time.perf_counter()
    log(f"[cnv] counting into {len(windows)} windows ...")
    C, bcs, nfr, win = cnv_count_matrix(fragments, windows, window_size, min_frags)
    if C.shape[0] == 0:
        raise ValueError("The created count matrix is empty. Check input files and filters.")
    t_cnt = time.perf_counter()
    bcs = np.asarray(bcs, dtype=object)
    C = C.tocsr()
    if selected_cells is not None:
        sel = pd.Index(bcs).get_indexer(pd.Index(list(selected_cells)))
        sel = sel[sel >= 0]
        C, bcs, nfr = C[sel], bcs[sel], nfr[sel]
    n_win = C.shape[1]
    nz_cell = np.diff(C.indptr)
    keep_c = (n_win - nz_cell) < ((1 - threshold_cells_nbins) * n_win)
    C, bcs, nfr = C[keep_c], bcs[keep_c], nfr[keep_c]
    ncells = C.shape[0]
    nz_bin = np.bincount(C.indices, minlength=n_win) if C.nnz else np.zeros(n_win, int)
    keep_b = (ncells - nz_bin) < (threshold_blacklist_bins * ncells)
    C = C[:, keep_b]
    var = win[keep_b].reset_index(drop=True)
    var["chrom"] = var["chrom"].cat.remove_unused_categories()
    log(f"[cnv] {ncells} cells x {var.shape[0]} windows after filtering")
    X = C.toarray()
    chrom_names = list(var["chrom"].cat.categories)
    codes = var["chrom"].cat.codes.to_numpy()
    chrom_bounds = []
    for ci in range(len(chrom_names)):
        idx = np.flatnonzero(codes == ci)
        chrom_bounds.append((int(idx[0]), int(idx[-1]) + 1))
    gc = var["GC"].to_numpy()
    common = (chrom_bounds, chrom_names, gc, gc_correction, k, minsize, test, minsize_cnv, uq, lq)
    nb = max(1, min(int(n_jobs), ncells))
    blocks = np.array_split(np.arange(ncells), nb * 4 if nb > 1 else 1)
    blocks = [b for b in blocks if b.size]
    log(f"[cnv] GC correction + segmentation ({test}) on {nb} worker(s) ...")
    if nb == 1:
        results = [_process_cells((X[b],) + common) for b in blocks]
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=nb, mp_context=ctx) as ex:
            results = list(ex.map(_process_cells, [(X[b],) + common for b in blocks]))
    states = np.vstack([r[0] for r in results])
    corrected = np.vstack([r[1] for r in results])
    raw_bp = [x for r in results for x in r[2]]
    kept_bp = [x for r in results for x in r[3]]
    t_seg = time.perf_counter()

    def _bp_table(lst):
        rows = [(bcs[i], chrom_names[c], p, d) for i, l in enumerate(lst) for (c, p, d) in l]
        return pd.DataFrame(rows, columns=["cell", "chrom", "bp", "distance"])

    var.index = (var["chrom"].astype(str) + ":" + var["start"].astype(str) + "-"
                 + var["end"].astype(str)).to_numpy()
    var = var.drop(columns=["tile"])
    obs = pd.DataFrame({"n_fragments": nfr}, index=pd.Index(bcs.astype(str), name=None))
    out = AnnData(X=states, obs=obs, var=var)
    out.layers["counts"] = sp.csr_matrix(X)
    out.layers["gc_corrected"] = sp.csr_matrix(corrected)
    out.obs["cnv_burden"] = (states != 1).mean(1)
    out.uns["cnv"] = {
        "method": method,
        "params": dict(window_size=int(window_size), test=test, k=k, minsize=minsize,
                       minsize_cnv=minsize_cnv, uq=uq, lq=lq, min_frags=min_frags,
                       threshold_cells_nbins=threshold_cells_nbins,
                       threshold_blacklist_bins=threshold_blacklist_bins,
                       gc_correction=gc_correction,
                       exclude=list(exclude) if exclude is not None else []),
        "breakpoints": _bp_table(kept_bp),
        "breakpoints_raw": _bp_table(raw_bp),
        "timings": {"windows": t_win - t0, "counts": t_cnt - t_win,
                    "segmentation": t_seg - t_cnt, "total": t_seg - t0},
        "state_labels": {"0": "loss", "1": "normal", "2": "gain"},
    }
    if adata is not None:
        pos = pd.Index(out.obs_names).get_indexer(adata.obs_names)
        M = np.full((adata.n_obs, out.n_vars), np.nan, dtype=np.float32)
        ok = pos >= 0
        M[ok] = states[pos[ok]]
        adata.obsm[f"X_{key_added}"] = M
        adata.obs[f"{key_added}_burden"] = np.where(ok, out.obs["cnv_burden"].to_numpy()[np.maximum(pos, 0)], np.nan)
        adata.uns[key_added] = {"var": out.var.copy(), "params": out.uns["cnv"]["params"]}
    log(f"[cnv] done in {t_seg - t0:.1f}s")
    return out


# ---------------------------------------------------------------------------
# Downstream: subclones, karyotype measures, burden
# ---------------------------------------------------------------------------

@njit(cache=True)
def _hclust_ward_d(diss, n):
    """R ``hclust(method = "ward.D")`` (Fortran HCLUST, F. Murtagh), 0-based.

    ``diss`` is the condensed distance vector in R ``dist`` order (modified in
    place). Returns ``ia, ib, crit`` (1-based cluster ids as in R)."""
    INF = 1e300
    ia = np.zeros(n, np.int64)
    ib = np.zeros(n, np.int64)
    crit = np.zeros(n)
    membr = np.ones(n)
    nn = np.zeros(n, np.int64)
    disnn = np.zeros(n)
    flag = np.ones(n, np.bool_)

    def ioffst(i, j):  # 1-based i < j -> 0-based index
        return j + (i - 1) * n - (i * (i + 1)) // 2 - 1

    jm = 0
    im = 0
    jj = 0
    for i in range(1, n):
        dmin = INF
        for j in range(i + 1, n + 1):
            ind = ioffst(i, j)
            if dmin > diss[ind]:
                dmin = diss[ind]
                jm = j
        nn[i - 1] = jm
        disnn[i - 1] = dmin
    ncl = n
    while True:
        dmin = INF
        for i in range(1, n):
            if flag[i - 1] and disnn[i - 1] < dmin:
                dmin = disnn[i - 1]
                im = i
                jm = nn[i - 1]
        ncl -= 1
        i2 = min(im, jm)
        j2 = max(im, jm)
        ia[n - ncl - 1] = i2
        ib[n - ncl - 1] = j2
        crit[n - ncl - 1] = dmin
        flag[j2 - 1] = False
        dmin = INF
        for kk in range(1, n + 1):
            if flag[kk - 1] and kk != i2:
                ind1 = ioffst(i2, kk) if i2 < kk else ioffst(kk, i2)
                ind2 = ioffst(j2, kk) if j2 < kk else ioffst(kk, j2)
                d12 = diss[ioffst(i2, j2)]
                v = ((membr[i2 - 1] + membr[kk - 1]) * diss[ind1]
                     + (membr[j2 - 1] + membr[kk - 1]) * diss[ind2]
                     - membr[kk - 1] * d12)
                diss[ind1] = v / (membr[i2 - 1] + membr[j2 - 1] + membr[kk - 1])
                if i2 < kk:
                    if diss[ind1] < dmin:
                        dmin = diss[ind1]
                        jj = kk
                else:
                    if diss[ind1] < disnn[kk - 1]:
                        disnn[kk - 1] = diss[ind1]
                        nn[kk - 1] = i2
        membr[i2 - 1] = membr[i2 - 1] + membr[j2 - 1]
        disnn[i2 - 1] = dmin
        nn[i2 - 1] = jj
        for i in range(1, n):
            if flag[i - 1] and (nn[i - 1] == i2 or nn[i - 1] == j2):
                dmin = INF
                for j in range(i + 1, n + 1):
                    if flag[j - 1]:
                        ind = ioffst(i, j)
                        if diss[ind] < dmin:
                            dmin = diss[ind]
                            jj = j
                nn[i - 1] = jj
                disnn[i - 1] = dmin
        if ncl <= 1:
            break
    return ia[: n - 1], ib[: n - 1], crit[: n - 1]


def _hcass2(n, ia, ib):
    """R ``HCASS2``: merge matrix (R sign convention) and leaf order (1-based)."""
    iia = ia.copy()
    iib = ib.copy()
    for i in range(n - 2):
        k = min(ia[i], ib[i])
        for j in range(i + 1, n - 1):
            if ia[j] == k:
                iia[j] = -(i + 1)
            if ib[j] == k:
                iib[j] = -(i + 1)
    iia = -iia
    iib = -iib
    for i in range(n - 1):
        if iia[i] > 0 and iib[i] < 0:
            iia[i], iib[i] = iib[i], iia[i]
        if iia[i] > 0 and iib[i] > 0:
            k1, k2 = min(iia[i], iib[i]), max(iia[i], iib[i])
            iia[i], iib[i] = k1, k2
    order = [iia[n - 2], iib[n - 2]]
    for i in range(n - 2, 0, -1):
        for j in range(len(order)):
            if order[j] == i:
                order[j] = iia[i - 1]
                order.insert(j + 1, iib[i - 1])
                break
    order = [-o for o in order]
    merge = np.column_stack([iia, iib])
    return merge, np.asarray(order, dtype=np.int64)


def _r_hclust_ward_d(M: np.ndarray):
    """``hclust(dist(M), "ward.D")`` -> ``(merge, height, order)`` as in R."""
    M = np.asarray(M, dtype=np.float64)
    n = M.shape[0]
    sq = (M * M).sum(1)
    G = M @ M.T
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0)
    iu = np.triu_indices(n, 1)
    diss = np.sqrt(D2[iu])  # row-major upper triangle = R dist order
    ia, ib, crit = _hclust_ward_d(diss.copy(), n)
    merge, order = _hcass2(n, ia, ib)
    return merge, crit, order


def _r_cutree(merge: np.ndarray, k: int) -> np.ndarray:
    n = merge.shape[0] + 1
    if k >= n:
        return np.arange(1, n + 1)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    node_rep = {}
    for step in range(n - k):
        m1, m2 = merge[step]
        a = -m1 - 1 if m1 < 0 else node_rep[m1]
        b = -m2 - 1 if m2 < 0 else node_rep[m2]
        ra, rb = find(a), find(b)
        parent[rb] = ra
        node_rep[step + 1] = ra
    lab = {}
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = find(i)
        if r not in lab:
            lab[r] = len(lab) + 1
        out[i] = lab[r]
    return out


def cnv_subclones(cnv_adata: AnnData, n_clones: int = 2, *, key_added: str = "cnv_subclone") -> pd.Series:
    """Split cells into subclones (epiAneufinder ``split_subclones``).

    Ward (``ward.D``) hierarchical clustering of Euclidean distances between
    CNV profiles, cut into ``n_clones`` groups; R's ``hclust``/``cutree`` are
    ported so that tie-breaking matches. Writes ``obs[key_added]`` (category)
    and stores the dendrogram in ``uns['cnv']['hclust']``.
    """
    merge, height, order = _get_hclust(cnv_adata)
    lab = _r_cutree(merge, n_clones)
    s = pd.Series(pd.Categorical(lab.astype(str), categories=[str(i) for i in range(1, n_clones + 1)]),
                  index=cnv_adata.obs_names, name=key_added)
    cnv_adata.obs[key_added] = s
    return s


def _get_hclust(cnv_adata: AnnData):
    u = cnv_adata.uns.setdefault("cnv", {})
    h = u.get("hclust")
    if h is None or len(h["order"]) != cnv_adata.n_obs:
        X = cnv_adata.X.toarray() if sp.issparse(cnv_adata.X) else np.asarray(cnv_adata.X)
        merge, height, order = _r_hclust_ward_d(X)
        u["hclust"] = {"merge": merge, "height": height, "order": order, "method": "ward.D"}
        h = u["hclust"]
    return np.asarray(h["merge"]), np.asarray(h["height"]), np.asarray(h["order"])


def cnv_karyotype_measures(cnv_adata: AnnData, base_state: int = 1):
    """Aneuploidy and heterogeneity (epiAneufinder ``karyotype_measures``).

    Aneuploidy is the fraction of (cell, bin) calls different from
    ``base_state``; heterogeneity per bin is ``sum(sorted_counts * 0..m-1) / n``
    as in AneuFinder (Bakker et al. 2016). Returns ``{"genomewide": DataFrame,
    "per_chromosome": DataFrame}``.
    """
    X = cnv_adata.X.toarray() if sp.issparse(cnv_adata.X) else np.asarray(cnv_adata.X)
    n_cell = X.shape[0]
    ne = X != base_state
    het = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        occ = np.sort(np.unique(X[:, j], return_counts=True)[1])[::-1]
        het[j] = (occ * np.arange(occ.size)).sum() / n_cell
    chrom = cnv_adata.var["chrom"].astype(str).to_numpy()
    chroms = list(dict.fromkeys(chrom))
    per = pd.DataFrame({
        "Aneuploidy": [ne[:, chrom == c].mean() for c in chroms],
        "Heterogeneity": [het[chrom == c].mean() for c in chroms],
    }, index=chroms)
    gw = pd.DataFrame({"Aneuploidy": [ne.mean()], "Heterogeneity": [het.mean()]})
    return {"genomewide": gw, "per_chromosome": per}


def cnv_burden(cnv_adata: AnnData, base_state: int = 1) -> pd.Series:
    """Per-cell CNV burden (epiAneufinder ``cnv_burden_per_cell``)."""
    X = cnv_adata.X.toarray() if sp.issparse(cnv_adata.X) else np.asarray(cnv_adata.X)
    return pd.Series((X != base_state).mean(1), index=cnv_adata.obs_names, name="cnv_burden")
