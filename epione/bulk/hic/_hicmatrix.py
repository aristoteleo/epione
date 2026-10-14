"""Minimal in-memory Hi-C matrix used by the ``method="hicexplorer"`` paths.

This is a port of the parts of HiCMatrix (deeptools/HiCMatrix, GPL-3.0,
Copyright (c) Fidel Ramírez, Joachim Wolff and contributors) that HiCExplorer
3.7.6 relies on: loading a ``.cool`` / HiCExplorer ``.h5`` with its correction
factors and masked bins, bin masking, interval lookup, and the per-diagonal
z-score / observed-over-expected transform. The operations are kept in the same
order and use the same scipy calls, so the numbers match HiCExplorer
bit-for-bit; see ``tests/parity/hic_tools/MANIFEST.md``.

It is private: public functions in :mod:`epione.bulk.hic` take paths and return
pandas / numpy / scipy objects.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, dia_matrix, diags, lil_matrix, triu


Interval = Tuple[str, int, int, float]


def _to_str(x):
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("ascii")
    return str(x)


class _ChromIndex:
    """Bin lookup by genomic position, replacing HiCMatrix's interval trees.

    ``lookup(chrom, pos)`` returns the id of the bin whose half-open interval
    ``[start, end)`` contains ``pos`` -- what ``sorted(tree[pos:pos + 1])[0].data``
    returns for the non-overlapping bins HiCMatrix stores -- or ``None``.
    """

    def __init__(self, intervals: Sequence[Interval]):
        self.boundaries: "OrderedDict[str, Tuple[int, int]]" = OrderedDict()
        self._starts = {}
        self._ends = {}
        self._ids = {}
        if len(intervals) == 0:
            return
        # Same bookkeeping as HiCMatrix.intervalListToIntervalTree, including
        # its quirk that the first chromosome is first registered as (0, 0).
        chr_start_id = 0
        previous = None
        per = OrderedDict()
        for i, iv in enumerate(intervals):
            chrom = iv[0]
            if previous != chrom:
                if previous is None:
                    previous = chrom
                self.boundaries[previous] = (chr_start_id, i)
                chr_start_id = i
                per.setdefault(chrom, [])
                previous = chrom
            per[chrom].append((int(iv[1]), int(iv[2]), i))
        self.boundaries[chrom] = (chr_start_id, len(intervals))
        for chrom, rows in per.items():
            rows.sort()
            arr = np.array(rows, dtype=np.int64)
            self._starts[chrom] = arr[:, 0]
            self._ends[chrom] = arr[:, 1]
            self._ids[chrom] = arr[:, 2]

    def __contains__(self, chrom):
        return chrom in self._starts

    def lookup(self, chrom: str, pos: int) -> Optional[int]:
        starts = self._starts.get(chrom)
        if starts is None:
            return None
        k = int(np.searchsorted(starts, pos, side="right")) - 1
        if k < 0 or self._ends[chrom][k] <= pos:
            return None
        return int(self._ids[chrom][k])


def enlarge_bins(bin_intervals: List[Interval]) -> List[Interval]:
    """Close gaps between consecutive bins (``hicexplorer.utilities.enlarge_bins``)."""
    chr_start = True
    for idx in range(len(bin_intervals) - 1):
        chrom, start, end, extra = bin_intervals[idx]
        chrom_next, start_next, end_next, extra_next = bin_intervals[idx + 1]
        if chr_start is True:
            start = 0
            chr_start = False
            bin_intervals[idx] = (chrom, start, end, extra)
        if chrom == chrom_next and end != start_next:
            middle = start_next - int((start_next - end) / 2)
            bin_intervals[idx] = (chrom, start, middle, extra)
            bin_intervals[idx + 1] = (chrom, middle, end_next, extra_next)
        if chrom != chrom_next:
            chr_start = True
    chrom, start, end, extra = bin_intervals[-1]
    bin_intervals[-1] = (chrom, start, end, extra)
    return bin_intervals


class HiCMat:
    """The subset of ``hicmatrix.HiCMatrix.hiCMatrix`` HiCExplorer needs."""

    def __init__(self):
        self.matrix: Optional[csr_matrix] = None
        self.cut_intervals: List[Interval] = []
        self.nan_bins = np.array([])
        self.correction_factors = None
        self.bin_size = None
        self.orig_bin_ids = []
        self.orig_cut_intervals = []
        self._index: Optional[_ChromIndex] = None

    # ------------------------------------------------------------------ IO
    @classmethod
    def from_cool(cls, uri: str, region: Optional[str] = None,
                  weight_name: str = "weight", apply_correction: bool = True) -> "HiCMat":
        """Load like ``hiCMatrix(uri)`` (or ``pChrnameList=[region]``)."""
        import cooler

        clr = cooler.Cooler(uri)
        nbins = int(clr.info["nbins"])
        if region is None:
            pix = clr.matrix(balance=False, sparse=True, as_pixels=True)[:]
            count_dtype = pix["count"].dtype
            used_dtype = np.int32 if np.iinfo(np.int32).max >= nbins else np.int64
            data = pix["count"].to_numpy().astype(count_dtype)
            rows = pix["bin1_id"].to_numpy().astype(used_dtype)
            cols = pix["bin2_id"].to_numpy().astype(used_dtype)
            matrix = csr_matrix((data, (rows, cols)), shape=(nbins, nbins), dtype=count_dtype)
            bins = clr.bins()[:]
            cf = bins[[weight_name]] if (apply_correction and weight_name in bins.columns) else None
            iv_df = bins[["chrom", "start", "end"]]
        else:
            matrix = clr.matrix(balance=False, sparse=True, as_pixels=False).fetch(region).tocsr()
            iv_df = clr.bins().fetch(region)
            cf = iv_df[weight_name] if weight_name in iv_df.columns else None
            iv_df = iv_df[["chrom", "start", "end"]]
        correction_factors = None
        if cf is not None and apply_correction:
            matrix.eliminate_zeros()
            if len(matrix.data) > 1:
                matrix.data = matrix.data.astype(float)
                correction_factors = np.array(cf.values).flatten()
                if np.sum(np.isnan(correction_factors)) != len(correction_factors):
                    instances, features = matrix.nonzero()
                    instances_factors = correction_factors[instances]
                    features_factors = correction_factors[features]
                    operator = "/" if weight_name in ("KR", "VC", "SQRT_VC") else "*"
                    instances_factors *= features_factors
                    if operator == "*":
                        matrix.data *= instances_factors
                    else:
                        matrix.data /= instances_factors
        out = cls()
        out.cut_intervals = [(_to_str(v[0]), v[1], v[2], 1.0) for v in iv_df.values]
        mask = np.isnan(matrix.data)
        matrix.data[mask] = 0
        matrix.eliminate_zeros()
        shape = min(matrix.shape)
        nan_ids = np.setdiff1d(np.arange(shape), matrix.indices)
        out.nan_bins = np.array([b for b in nan_ids if len(matrix[b, :].data) == 0])
        out.matrix = matrix
        out.correction_factors = correction_factors
        out._post_load()
        return out

    @classmethod
    def from_h5(cls, path: str) -> "HiCMat":
        """Load a HiCExplorer ``.h5`` matrix (PyTables, possibly Blosc-compressed)."""
        import tables

        with tables.open_file(str(path), "r") as f:
            parts = {p: getattr(f.root.matrix, p).read() for p in ("data", "indices", "indptr", "shape")}
            matrix = csr_matrix((parts["data"], parts["indices"], parts["indptr"]), shape=parts["shape"])
            chrom = [_to_str(c) for c in f.root.intervals.chr_list.read()]
            start = f.root.intervals.start_list.read()
            end = f.root.intervals.end_list.read()
            extra = f.root.intervals.extra_list.read()
            nan_bins = f.root.nan_bins.read() if hasattr(f.root, "nan_bins") else np.array([])
            cf = None
            if hasattr(f.root, "correction_factors"):
                cf = np.array(f.root.correction_factors.read())
                cf[np.isnan(cf)] = 0
                cf[np.isinf(cf)] = 0
        out = cls()
        out.cut_intervals = list(zip(chrom, start, end, extra))
        out.matrix = matrix
        out.nan_bins = nan_bins
        out.correction_factors = cf
        out._post_load()
        return out

    def _post_load(self):
        if self.nan_bins is None:
            self.nan_bins = np.array([])
        # fillLowerTriangle
        if sparse.tril(self.matrix, k=-1).sum() == 0:
            self.matrix = self.matrix + triu(self.matrix, 1).T
        self._reindex()

    # ----------------------------------------------------------- helpers
    def _reindex(self):
        self._index = _ChromIndex(self.cut_intervals)

    @property
    def chrBinBoundaries(self):
        return self._index.boundaries

    def getChrNames(self):
        return list(self._index.boundaries)

    def getChrBinRange(self, chrom):
        return self._index.boundaries[chrom]

    def getBinPos(self, idx):
        return self.cut_intervals[idx]

    def getRegionBinRange(self, chrom, startpos, endpos):
        if chrom not in self._index:
            return None
        a = self._index.lookup(chrom, int(startpos))
        b = self._index.lookup(chrom, int(endpos))
        if a is None or b is None:
            return None
        return a, b

    def get_chromosome_sizes(self):
        sizes = OrderedDict()
        for chrom, (_, end_bin) in self._index.boundaries.items():
            c, _, end, _ = self.cut_intervals[end_bin - 1]
            sizes[c] = end
        return sizes

    def getBinSize(self):
        if self.bin_size is None:
            chrom, start, end, _ = zip(*self.cut_intervals)
            diff = np.array(end) - np.array(start)
            if len(diff) == 1:
                self.bin_size = diff[0]
                return self.bin_size
            median = int(np.median(np.concatenate([
                np.diff([s for c, s, e, x in self.cut_intervals if c == cur])
                for cur, nb in Counter(chrom).items() if nb > 1])))
            self.bin_size = median
        return self.bin_size

    # ------------------------------------------------------- operations
    def diagflat(self, value=np.nan):
        M = self.matrix.shape[0]
        diagmatrix = dia_matrix((np.repeat(value, M), 0), shape=(M, M))
        self_diag = dia_matrix(([self.matrix.diagonal()], [0]), shape=(M, M))
        self.matrix = self.matrix - self_diag + diagmatrix
        return self.matrix

    def filterOutInterChrCounts(self):
        coo = self.matrix.tocoo()
        dist, _ = getDistList(coo.row, coo.col, self.cut_intervals)
        coo.data[dist == -1] = 0
        self.matrix = coo.tocsr()
        self.matrix.eliminate_zeros()
        return self.matrix

    def keepOnlyTheseChr(self, chromosome_list):
        for c in chromosome_list:
            if c not in self._index.boundaries:
                raise ValueError(f"Chromosome name not in matrix: {c}")
        self.restoreMaskedBins()
        n = self.matrix.shape[0]
        sel = np.zeros(n, dtype=bool)
        for chrom, (a, b) in self._index.boundaries.items():
            if chrom in chromosome_list:
                sel[a:b] = True
        sel_id = np.flatnonzero(sel)
        mat = self.matrix[sel_id, :][:, sel_id]
        self.cut_intervals = [self.cut_intervals[x] for x in sel_id]
        if self.correction_factors is not None:
            self.correction_factors = [self.correction_factors[x] for x in sel_id]
        if len(self.nan_bins):
            tmp = np.zeros(n)
            tmp[np.asarray(self.nan_bins, dtype=int)] = 1
            self.nan_bins = np.flatnonzero(tmp[sel_id] == 1)
        else:
            self.nan_bins = []
        self._reindex()
        self.matrix = mat
        return self.matrix

    def reorderChromosomes(self, new_order):
        order = []
        for chrom in new_order:
            a, b = self._index.boundaries[chrom]
            order.extend(range(a, b))
        n = self.matrix.shape[0]
        self.matrix = self.matrix[order, :][:, order]
        self.cut_intervals = [self.cut_intervals[x] for x in order]
        if len(self.nan_bins):
            tmp = np.zeros(n)
            tmp[np.asarray(self.nan_bins, dtype=int)] = 1
            self.nan_bins = np.flatnonzero(tmp[order] == 1)
        else:
            self.nan_bins = []
        self._reindex()

    def maskBins(self, bin_ids=None):
        if bin_ids is None or len(bin_ids) == 0:
            return
        if len(self.orig_bin_ids) > 0:
            M = self.matrix.shape[0]
            previous = self.orig_bin_ids[M:]
            bin_ids = np.unique(np.concatenate([previous, self.orig_bin_ids[bin_ids]]))
            self.restoreMaskedBins()
        if self.nan_bins is not None and len(self.nan_bins) > 0:
            bin_ids = np.unique(np.concatenate([self.nan_bins, bin_ids]))
            self.nan_bins = []
        rows = cols = np.delete(list(range(self.matrix.shape[1])), bin_ids)
        self.matrix = self.matrix[rows, :][:, cols]
        self.orig_bin_ids = np.concatenate([rows, bin_ids])
        new_iv = [self.cut_intervals[x] for x in rows]
        self.orig_cut_intervals = new_iv + [self.cut_intervals[int(x)] for x in bin_ids]
        self.cut_intervals = new_iv
        self._reindex()
        if self.correction_factors is not None:
            self.correction_factors = np.asarray(self.correction_factors)[rows]

    def restoreMaskedBins(self):
        if len(self.orig_bin_ids) == 0:
            return
        M = self.matrix.shape[0]
        N = len(self.orig_bin_ids) - M
        self.matrix = sparse.vstack([self.matrix, csr_matrix((N, M))])
        self.matrix = sparse.hstack([self.matrix, csr_matrix((M + N, N))], format="csr")
        rows = cols = np.argsort(self.orig_bin_ids)
        self.matrix = self.matrix[rows, :][:, cols]
        self.cut_intervals = [self.orig_cut_intervals[x] for x in rows]
        self._reindex()
        self.nan_bins = self.orig_bin_ids[M:]
        if self.correction_factors is not None:
            cf = np.concatenate([self.correction_factors, np.repeat(np.nan, N)])
            self.correction_factors = cf[rows]
        self.orig_bin_ids = []
        self.orig_cut_intervals = []

    def set_cut_intervals(self, new_intervals):
        self.cut_intervals = new_intervals
        self.orig_cut_intervals = new_intervals
        self.orig_bin_ids = list(range(len(new_intervals)))
        self.nan_bins = []
        self._reindex()

    def convert_to_obs_exp_matrix(self, maxdepth=None, zscore=False, perchr=False):
        """Per-diagonal z-score or obs/exp of the non-zero pixels (HiCMatrix)."""
        binsize = self.getBinSize()
        max_depth_in_bins = None
        if maxdepth:
            if maxdepth < binsize:
                raise ValueError(f"Please specify a maxDepth larger than bin size ({binsize})")
            max_depth_in_bins = int(float(maxdepth * 1.5) / binsize)
            self.matrix = triu(self.matrix, k=0, format="csr") - triu(self.matrix, k=max_depth_in_bins, format="csr")
        else:
            self.matrix = triu(self.matrix, k=0, format="csr")
        self.matrix.eliminate_zeros()
        depth = None
        if zscore is True:
            m_size = self.matrix.shape[0]
            depth = max_depth_in_bins if max_depth_in_bins is not None else m_size
            # dtype pinned to what scipy (and so HiCMatrix) produces today: int64 ones cast to float64
            diag_mat_ones = diags(np.repeat([1], m_size * depth).reshape(depth, m_size), list(range(depth)),
                                  dtype=np.float64)
            self.matrix += diag_mat_ones
        trasf = lil_matrix(self.matrix.shape)
        chr_sub = OrderedDict()
        civ = OrderedDict()
        csizes = OrderedDict()
        crange = OrderedDict()
        if perchr:
            for chrom in self.getChrNames():
                a, b = self.getChrBinRange(chrom)
                chr_sub[chrom] = self.matrix[a:b, a:b].tocoo()
                civ[chrom] = [self.cut_intervals[x] for x in range(a, b)]
                csizes[chrom] = [chr_sub[chrom].shape[0]]
                crange[chrom] = (a, b)
        else:
            chr_sub["all"] = self.matrix.tocoo()
            civ["all"] = self.cut_intervals
            csizes["all"] = np.array([v[1] - v[0] for v in self.chrBinBoundaries.values()])
            crange["all"] = (0, self.matrix.shape[0])
        for chrom, sub in chr_sub.items():
            if zscore is True:
                sub.data -= 1
            dist_list, _ = getDistList(sub.row, sub.col, fit_cut_intervals(civ[chrom]))
            dist_list[dist_list == -1] = -binsize
            dist_list = (np.array(dist_list).astype(float) / binsize).astype(int) + 1
            sum_counts = np.bincount(dist_list, weights=sub.data)
            distance_len = np.bincount(dist_list)
            mat_size = sub.shape[0]
            mu = {}
            std = {}
            for d1, sum_value in enumerate(sum_counts):
                if maxdepth and d1 == 0:
                    mu[d1] = np.nan
                    std[d1] = np.nan
                    continue
                if d1 == 0:
                    total_intra = mat_size ** 2 - sum([s ** 2 for s in csizes[chrom]])
                    diagonal_length = int(total_intra / 2)
                else:
                    diagonal_length = sum([s - (d1 - 1) for s in csizes[chrom] if s > (d1 - 1)])
                diagonal_length = max(diagonal_length, distance_len[d1])
                if diagonal_length == 0:
                    mu[d1] = np.nan
                else:
                    mu[d1] = np.float64(sum_value) / diagonal_length
                if zscore:
                    vsd = np.abs((sub.data[dist_list == d1] - mu[d1]) ** 2)
                    zsum = (diagonal_length - len(vsd)) * mu[d1] ** 2
                    std[d1] = np.sqrt((vsd.sum() + zsum) / diagonal_length)
            transf = np.zeros(len(sub.data))
            mu_arr = np.array([mu[i] for i in range(len(sum_counts))], dtype=np.float64)
            keep = np.ones(len(sub.data), dtype=bool) if depth is None else ~(dist_list > depth + 1)
            if zscore:
                std_arr = np.array([std[i] for i in range(len(sum_counts))], dtype=np.float64)
                s = std_arr[dist_list]
                z = s == 0
                with np.errstate(divide="ignore", invalid="ignore"):
                    vals = (sub.data - mu_arr[dist_list]) / s
                vals[z] = np.nan
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    vals = sub.data / mu_arr[dist_list]
            transf[keep] = vals[keep]
            sub.data = transf
            a, b = crange[chrom]
            trasf[a:b, a:b] = sub.tolil()
        self.matrix = trasf.tocsr()
        return self.matrix

    def convert_to_zscore_matrix(self, maxdepth=None, perchr=False):
        return self.convert_to_obs_exp_matrix(maxdepth=maxdepth, zscore=True, perchr=perchr)


def getDistList(rows, cols, cut_intervals):
    chrnames, starts, _, _ = zip(*cut_intervals)
    start_row = np.take(starts, rows)
    start_col = np.take(starts, cols)
    dist = start_col - start_row
    chr_id = np.unique(chrnames, return_inverse=True)[1]
    chr_diff = np.take(chr_id, rows) - np.take(chr_id, cols)
    dist[chr_diff != 0] = -1
    chrom_list = np.take(chrnames, rows)
    chrom_list[chr_diff != 0] = ""
    return dist, chrom_list


def fit_cut_intervals(cut_intervals):
    if len(cut_intervals) <= 1:
        return cut_intervals
    chrom, start, end, extra = zip(*cut_intervals)
    median = int(np.median(np.concatenate([
        np.diff([s for c, s, e, x in cut_intervals if c == cur])
        for cur, nb in Counter(chrom).items() if nb > 1])))
    diff = np.array(end) - np.array(start)
    if len(np.flatnonzero(diff != median)) > (len(diff) * 0.01):
        def snap(x, m):
            resi = [-1 * (x % m), -x % m]
            return x + resi[np.argmin(np.abs(resi))]
        start = [snap(x, median) for x in start]
        end = [snap(x, median) for x in end]
        cut_intervals = list(zip(chrom, start, end, extra))
    return cut_intervals
