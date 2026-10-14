"""A/B compartments the way ``hicPCA`` computes them (HiCExplorer 3.7.6).

Port of ``hicexplorer/hicPCA.py`` and the obs/exp helpers it uses from
``hicexplorer/utilities.py`` (deeptools/HiCExplorer, GPL-3.0, Copyright (c)
Fidel Ramírez, Joachim Wolff and contributors). Used by
``compartments(method="hicexplorer")``.

Per chromosome: observed/expected (``dist_norm``: mean of the non-zero pixels
at each distance, optionally Homer's ligation factor; ``lieberman``:
Lieberman-Aiden 2009), covariance of the dense obs/exp matrix, then
``scipy.linalg.eig``. As in hicPCA the eigenvectors are taken in LAPACK's
return order (not sorted by eigenvalue), the dist_norm values are stored as
float32, and the Lieberman distance is ``ceil(|i - j| / 2)``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.sparse import csr_matrix

from ._hicmatrix import HiCMat, enlarge_bins


def _nan_inf_to_zero_csr_dense(m):
    m = csr_matrix(m)
    if len(m.data):
        m.data[np.flatnonzero(np.isnan(m.data))] = 0.0
    m = m.todense()
    m = csr_matrix(m)
    if len(m.data):
        m.data[np.flatnonzero(np.isinf(m.data))] = 0
    return m.todense()


def obs_exp_non_zero(sub, ligation_factor=False):
    row, col = sub.nonzero()
    n = len(row)
    dist = np.absolute(row - col)
    expected = np.zeros(sub.shape[0])
    occ = np.zeros(sub.shape[0])
    # sequential accumulation in nonzero() order, as the source loop does
    # bincount sums in input order, like the source's sequential loop (exact)
    expected = np.bincount(dist, weights=sub.data[:n], minlength=sub.shape[0])[:sub.shape[0]].astype(float)
    occ = np.bincount(dist, minlength=sub.shape[0])[:sub.shape[0]].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.divide(expected, occ)
    expected[np.isnan(expected)] = 0
    expected[np.isinf(expected)] = 0
    row_sums = np.array(sub.sum(axis=1).T).flatten()
    total = sub.sum()
    sub.data = sub.data.astype(np.float32)
    exp_i = expected[dist]
    if ligation_factor:
        exp_i = exp_i * (row_sums[row] * row_sums[col] / total)
    with np.errstate(divide="ignore", invalid="ignore"):
        sub.data[:n] = np.divide(sub.data[:n], exp_i)
    sub.data[np.isnan(sub.data)] = 0
    sub.data[np.isinf(sub.data)] = 0
    sub.eliminate_zeros()
    return sub


def obs_exp_lieberman(sub, length_chromosome, chromosome_count):
    expected = np.zeros(sub.shape[0])
    row, col = sub.nonzero()
    expected = np.bincount(np.absolute(row - col), weights=sub.data[:len(row)],
                           minlength=sub.shape[0])[:sub.shape[0]].astype(float)
    cti = np.arange(float(len(expected)))
    cti *= np.int32(chromosome_count)
    cti -= np.int32(length_chromosome)
    cti *= np.int32(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.divide(expected, cti)
    distance = np.ceil(np.absolute(row - col) / 2).astype(np.int32)
    if len(sub.data) > 0:
        dtype = type(sub.data[0])
        e = expected[distance]
        sub.data = sub.data.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            sub.data = np.divide(sub.data, e)
        d = sub.data
        d[np.flatnonzero(np.isnan(d))] = 0.0
        d[np.flatnonzero(np.isinf(d))] = 0.0
        sub.data = d.astype(dtype)
    return sub


def _read_bed3(path):
    import gzip

    fh = open(path, "rb")
    if fh.read(2) == b"\x1f\x8b":
        fh.seek(0)
        fh = gzip.GzipFile(fileobj=fh)
    else:
        fh.seek(0)
    for raw in fh:
        line = raw.decode() if isinstance(raw, bytes) else raw
        if line.startswith(("#", "track", "browser")) or line.strip() == "":
            continue
        f = line.strip().split("\t")
        yield f[0], int(f[1]), int(f[2])


def _flip_by_gene_track(hm: HiCMat, vecs, bed_path):
    from scipy.stats import pearsonr

    occ = np.zeros(len(hm.cut_intervals))
    sizes = hm.get_chromosome_sizes()
    names = hm.getChrNames()
    for chrom, start, end in _read_bed3(bed_path):
        if chrom not in names:
            continue
        if start > sizes[chrom]:
            continue
        r = hm.getRegionBinRange(chrom, start, end)
        if r is None:
            continue
        occ[r[1]] += 1
    vecs = np.array(vecs).real.transpose()
    for chrom in names:
        a, b = hm.getChrBinRange(chrom)
        for ev in vecs:
            c = pearsonr(np.array(ev[a:b]).real, occ[a:b])
            if c[0] < 0:
                ev[a:b] = np.negative(ev[a:b])
    return np.array(vecs).transpose()


def _flip_by_bigwig(evs, bw, chrom, start, end, mark_type):
    for index, vector in enumerate(evs):
        pos = np.where(vector > 0)[0]
        neg = np.where(vector < 0)[0]
        pos_sum = neg_sum = pos_mean = neg_mean = 0
        if chrom in bw.chroms().keys():
            for i in pos:
                v = bw.stats(chrom, int(start[i]), int(end[i]))[0]
                if v:
                    pos_sum += v
            if pos_sum != 0:
                pos_mean = pos_sum / len(pos)
            for i in neg:
                v = bw.stats(chrom, int(start[i]), int(end[i]))[0]
                if v:
                    neg_sum += v
            if neg_sum != 0:
                neg_mean = neg_sum / len(neg)
            flip = (pos_mean < neg_mean) if mark_type == "active" else (pos_mean > neg_mean)
            if flip and neg_mean != 0 and pos_mean != 0:
                vector[pos] = np.negative(vector[pos])
                vector[neg] = np.negative(vector[neg])
        evs[index] = vector


def hicpca(hm: HiCMat, *, which=(1, 2), method="dist_norm", ligation_factor=False,
           extra_track: Optional[str] = None, histone_mark_type="active",
           ignore_masked_bins=False, chromosomes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if method not in ("dist_norm", "lieberman"):
        raise ValueError("method must be 'dist_norm' or 'lieberman'")
    if histone_mark_type not in ("active", "inactive"):
        raise ValueError("histone_mark_type must be 'active' or 'inactive'")
    if ignore_masked_bins:
        hm.maskBins(hm.nan_bins)
        hm.cut_intervals = enlarge_bins(hm.cut_intervals)
        hm._reindex()
    if chromosomes:
        hm.keepOnlyTheseChr(list(chromosomes))
    is_bw = extra_track is not None and str(extra_track).endswith((".bw", ".bigwig"))
    bw = None
    if is_bw:
        import pyBigWig
        bw = pyBigWig.open(str(extra_track), "r")
    length_chromosome = sum(b - a for a, b in hm.chrBinBoundaries.values())
    n_chr = len(hm.getChrNames())
    vecs, chroms, starts, ends = [], [], [], []
    for chrom in hm.getChrNames():
        a, b = hm.getChrBinRange(chrom)
        sub = hm.matrix[a:b, a:b]
        if method == "lieberman":
            oe = obs_exp_lieberman(sub, length_chromosome, n_chr)
        else:
            oe = obs_exp_non_zero(sub, ligation_factor)
        oe = csr_matrix(oe).todense()
        cov = np.cov(oe)
        cov = _nan_inf_to_zero_csr_dense(cov)
        _, eigs = linalg.eig(cov)
        c, s, e, _ = zip(*hm.cut_intervals[a:b])
        chroms += c
        starts += s
        ends += e
        sel = None
        for k in which:
            col = eigs[:, int(k) - 1:int(k)]
            sel = col if sel is None else np.hstack((sel, col))
        if is_bw:
            _flip_by_bigwig(sel.transpose(), bw, chrom, s, e, histone_mark_type)
        vecs += sel.tolist()
    if bw is not None:
        bw.close()
    if extra_track is not None and not is_bw:
        vecs = _flip_by_gene_track(hm, vecs, extra_track)
    arr = np.array(vecs)
    out = pd.DataFrame({"chrom": [str(x) for x in chroms],
                        "start": np.asarray(starts, dtype=int),
                        "end": np.asarray(ends, dtype=int)})
    for j, k in enumerate(which):
        out[f"E{int(k)}"] = np.real(arr[:, j]).astype(float)
    return out
