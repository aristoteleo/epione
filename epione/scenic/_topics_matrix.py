"""Fragment counts in regions — reproduces pycisTopic's ``create_cistopic_object_from_fragments``.

:func:`cistopic_matrix` counts, for every cell barcode and region, the
fragments that overlap the region by at least one base (0-based half-open
coordinates, as PyRanges' ``join``), removes blacklisted regions and regions
without counts, and returns a cells × regions AnnData with pycisTopic's
per-cell and per-region statistics. The fragment file is streamed with
pyarrow, so memory scales with the number of non-zero counts rather than with
the number of fragments.

pycisTopic is distributed under a non-commercial academic license; this module
reimplements its behaviour (checked against its outputs) without using its code.
"""
from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._np126 import log10_compat

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba ships with scanpy
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


def _console(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  └─ [scenic.topics] {msg}", flush=True)


def _read_bed3(path_or_df) -> pd.DataFrame:
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.iloc[:, :3].copy()
    else:
        df = pd.read_csv(path_or_df, sep="\t", header=None, comment="#", usecols=[0, 1, 2],
                         dtype={0: str, 1: np.int64, 2: np.int64})
    df.columns = ["Chromosome", "Start", "End"]
    df["Chromosome"] = df["Chromosome"].astype(str)
    df["Start"] = df["Start"].astype(np.int64)
    df["End"] = df["End"].astype(np.int64)
    return df.reset_index(drop=True)


def _natural_key(name: str):
    import re

    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _parse_region_names(names: Sequence[str]) -> pd.DataFrame:
    chrom, rest = zip(*(n.split(":", 1) for n in names))
    start, end = zip(*(r.split("-", 1) for r in rest))
    return pd.DataFrame({"Chromosome": list(chrom), "Start": np.array(start, dtype=np.int64),
                         "End": np.array(end, dtype=np.int64)})


@njit(cache=True)
def _overlap_counts(f_start, f_end, f_cell, f_index, r_start_sorted, r_end_sorted, r_idx_sorted,
                    max_len, out_region, out_cell, out_frag, out_start, out_end):
    """Region/cell pairs for every fragment-region overlap (start < end' and start' < end)."""
    n = 0
    m = r_start_sorted.shape[0]
    for i in range(f_start.shape[0]):
        fs = f_start[i]
        fe = f_end[i]
        # regions with start < fe
        lo = 0
        hi = m
        while lo < hi:
            mid = (lo + hi) >> 1
            if r_start_sorted[mid] < fe:
                lo = mid + 1
            else:
                hi = mid
        j = lo - 1
        while j >= 0 and r_start_sorted[j] > fs - max_len:
            if r_end_sorted[j] > fs:
                if n >= out_region.shape[0]:
                    return -1
                out_region[n] = r_idx_sorted[j]
                out_cell[n] = f_cell[i]
                out_frag[n] = f_index[i]
                out_start[n] = fs
                out_end[n] = fe
                n += 1
            j -= 1
    return n


def _blacklist_mask(regions: pd.DataFrame, blacklist: pd.DataFrame) -> np.ndarray:
    """True for regions overlapping any blacklist interval."""
    hit = np.zeros(len(regions), dtype=bool)
    for chrom, grp in regions.groupby("Chromosome", sort=False):
        bl = blacklist[blacklist["Chromosome"] == chrom]
        if bl.empty:
            continue
        bs = bl["Start"].to_numpy()
        be = bl["End"].to_numpy()
        order = np.argsort(bs, kind="stable")
        bs, be = bs[order], be[order]
        max_len = int((be - bs).max())
        rs = grp["Start"].to_numpy()
        re_ = grp["End"].to_numpy()
        hi = np.searchsorted(bs, re_, side="left")  # blacklist starts < region end
        res = np.zeros(len(grp), dtype=bool)
        for k in range(len(grp)):
            j = hi[k] - 1
            while j >= 0 and bs[j] > rs[k] - max_len:
                if be[j] > rs[k]:
                    res[k] = True
                    break
                j -= 1
        hit[grp.index.to_numpy()] = res
    return hit


def _iter_fragment_batches(path: Union[str, Path], block_size: int):
    import pyarrow as pa
    import pyarrow.csv as pacsv

    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    skip = 0
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                skip += 1
            else:
                ncol = len(line.rstrip("\n").split("\t"))
                break
    names = ["Chromosome", "Start", "End", "Name", "Score", "Strand"][:ncol]
    if ncol < 4:
        raise ValueError(f"fragments file {path} needs at least 4 columns")
    types = {"Chromosome": pa.dictionary(pa.int32(), pa.string()), "Start": pa.int64(),
             "End": pa.int64(), "Name": pa.dictionary(pa.int32(), pa.string())}
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(skip_rows=skip, column_names=names, block_size=block_size),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(column_types=types, include_columns=names[:4]),
    )
    for batch in reader:
        yield batch


def cistopic_matrix(
    fragments: Union[str, Path],
    regions: Union[str, Path, pd.DataFrame],
    *,
    barcodes: Optional[Sequence[str]] = None,
    blacklist: Optional[Union[str, Path, pd.DataFrame]] = None,
    min_frag: int = 1,
    min_cell: int = 1,
    is_acc: int = 1,
    project: str = "cisTopic",
    split_pattern: str = "___",
    tag_cells: bool = True,
    cell_order: str = "first_overlap",
    compat: str = "numpy126",
    block_size: int = 1 << 26,
    verbose: bool = True,
):
    """Cells × regions fragment-count AnnData; reproduces ``create_cistopic_object_from_fragments``.

    Parameters
    ----------
    fragments
        10x-style fragments file (``chrom, start, end, barcode[, count]``,
        optionally gzipped).
    regions
        BED file or DataFrame (``Chromosome, Start, End``) of regions; the
        output keeps their order.
    barcodes
        Cell barcodes to keep (pycisTopic ``valid_bc``). Default: every
        barcode in the file.
    blacklist
        BED file / DataFrame of regions to exclude (any overlap).
    min_frag, min_cell, is_acc
        Minimum fragments per cell, minimum accessible cells per region, and
        the count at which a region counts as accessible.
    project, split_pattern, tag_cells
        Cell names become ``barcode + split_pattern + project`` when
        ``tag_cells`` (pycisTopic default).
    cell_order
        ``"first_overlap"`` (pycisTopic): cells ordered by their first
        fragment-region overlap, walking chromosomes in natural order, regions
        in input order and, within a region, fragments by start (ascending),
        end (descending) and file order, as PyRanges' NCLS join reports them. ``"barcodes"``: the order
        of ``barcodes``. The order matters downstream because the LDA sampler
        visits cells in matrix order.
    compat
        ``"numpy126"`` computes the ``log10`` QC columns as numpy 1.26 does on
        AVX-512 CPUs (SVML), the SCENIC+ reference stack; ``"native"`` uses
        the installed numpy.
    block_size
        Bytes per pyarrow read block.

    Returns
    -------
    AnnData with ``X`` = fragment counts (int32 CSR), ``layers["binary"]``,
    ``obs`` = ``cisTopic_nr_frag``, ``cisTopic_log_nr_frag``,
    ``cisTopic_nr_acc``, ``cisTopic_log_nr_acc``, ``Unique_nr_frag``,
    ``barcode``, ``sample_id`` and ``var`` = ``Chromosome``, ``Start``,
    ``End``, ``Width`` and the per-region counts.
    """
    import anndata as ad

    reg = _read_bed3(regions)
    reg_names = (reg["Chromosome"] + ":" + reg["Start"].astype(str) + "-"
                 + reg["End"].astype(str)).tolist()
    # PyRanges order: chromosomes naturally sorted, regions in input order
    chrom_rank = {c: k for k, c in enumerate(sorted(reg["Chromosome"].unique(), key=_natural_key))}
    pos_in_chrom = reg.groupby("Chromosome", sort=False).cumcount().to_numpy().astype(np.int64)
    region_key = (reg["Chromosome"].map(chrom_rank).to_numpy().astype(np.int64) << 24) | pos_in_chrom
    by_chrom = {}
    for chrom, grp in reg.groupby("Chromosome", sort=False):
        s = grp["Start"].to_numpy()
        e = grp["End"].to_numpy()
        idx = grp.index.to_numpy().astype(np.int64)
        order = np.argsort(s, kind="stable")
        by_chrom[chrom] = (s[order], e[order], idx[order], int((e - s).max()))

    if barcodes is not None:
        barcodes = [str(b) for b in barcodes]
        bc_pos = {b: i for i, b in enumerate(barcodes)}
    else:
        bc_pos = {}
        barcodes = []
    fixed = len(barcodes) > 0

    frag_per_bc = Counter()
    rows_parts, cols_parts, frag_parts, fstart_parts, fend_parts = [], [], [], [], []
    n_read = 0
    _console("counting fragments in regions", verbose)
    for batch in _iter_fragment_batches(fragments, block_size):
        n_read += batch.num_rows
        chrom_col = batch.column(0)
        name_col = batch.column(3)
        name_dict = name_col.dictionary.to_pylist()
        if fixed:
            dict_to_cell = np.array([bc_pos.get(b, -1) for b in name_dict], dtype=np.int64)
        else:
            dict_to_cell = np.empty(len(name_dict), dtype=np.int64)
            for k, b in enumerate(name_dict):
                if b not in bc_pos:
                    bc_pos[b] = len(barcodes)
                    barcodes.append(b)
                dict_to_cell[k] = bc_pos[b]
        name_codes = name_col.indices.to_numpy(zero_copy_only=False)
        cell = dict_to_cell[name_codes]
        keep = cell >= 0
        u, c = np.unique(cell[keep], return_counts=True)
        frag_per_bc.update(dict(zip(u.tolist(), c.tolist())))
        starts = batch.column(1).to_numpy()
        ends = batch.column(2).to_numpy()
        chrom_dict = chrom_col.dictionary.to_pylist()
        chrom_codes = chrom_col.indices.to_numpy(zero_copy_only=False)
        for code, chrom in enumerate(chrom_dict):
            if chrom not in by_chrom:
                continue
            sel = keep & (chrom_codes == code)
            if not sel.any():
                continue
            rs, re_, ridx, max_len = by_chrom[chrom]
            sel_idx = np.flatnonzero(sel)
            fs, fe, fc = starts[sel_idx], ends[sel_idx], cell[sel_idx]
            fidx = (n_read - batch.num_rows) + sel_idx.astype(np.int64)
            cap = max(1024, 4 * fs.size)
            while True:
                out_r = np.empty(cap, dtype=np.int64)
                out_c = np.empty(cap, dtype=np.int64)
                out_f = np.empty(cap, dtype=np.int64)
                out_s = np.empty(cap, dtype=np.int64)
                out_e = np.empty(cap, dtype=np.int64)
                n = _overlap_counts(fs, fe, fc, fidx, rs, re_, ridx, max_len, out_r, out_c, out_f,
                                    out_s, out_e)
                if n >= 0:
                    break
                cap *= 4
            if n:
                rows_parts.append(out_r[:n])
                cols_parts.append(out_c[:n])
                frag_parts.append(out_f[:n])
                fstart_parts.append(out_s[:n])
                fend_parts.append(out_e[:n])
    n_cells = len(barcodes)
    rows = np.concatenate(rows_parts) if rows_parts else np.empty(0, np.int64)
    cols = np.concatenate(cols_parts) if cols_parts else np.empty(0, np.int64)
    frags = np.concatenate(frag_parts) if frag_parts else np.empty(0, np.int64)
    fstarts = np.concatenate(fstart_parts) if fstart_parts else np.empty(0, np.int64)
    fends = np.concatenate(fend_parts) if fend_parts else np.empty(0, np.int64)
    del rows_parts, cols_parts, frag_parts, fstart_parts, fend_parts
    counts = sp.csr_matrix((np.ones(rows.size, dtype=np.int32), (rows, cols)),
                           shape=(len(reg_names), n_cells))
    counts.sum_duplicates()
    _console(f"{n_read:,} fragments read, {counts.nnz:,} non-zero region × cell counts", verbose)
    # pycisTopic only keeps barcodes with at least one fragment in a region
    if cell_order == "first_overlap":
        # PyRanges join: per region, overlapping fragments in NCLS order
        # (start ascending, end descending, then input order)
        walk = np.lexsort((frags, -fends, fstarts, region_key[rows]))
        seen_cells = cols[walk]
        _, first_pos = np.unique(seen_cells, return_index=True)
        ci = seen_cells[np.sort(first_pos)]
    elif cell_order == "barcodes":
        ci = np.flatnonzero(np.asarray(counts.getnnz(axis=0)) > 0)
    else:
        raise ValueError("cell_order must be 'first_overlap' or 'barcodes'")
    del frags, fstarts, fends
    if ci.size != n_cells or cell_order == "first_overlap":
        counts = counts[:, ci]
        frag_per_bc = Counter({k: frag_per_bc.get(int(old), 0) for k, old in enumerate(ci)})
        barcodes = [barcodes[i] for i in ci]
        n_cells = len(barcodes)

    keep_reg = np.ones(len(reg_names), dtype=bool)
    if blacklist is not None:
        keep_reg &= ~_blacklist_mask(reg, _read_bed3(blacklist))
    binary = (counts >= is_acc).astype(np.int32)
    keep_reg &= np.asarray(binary.getnnz(axis=1)) > 0
    ri = np.flatnonzero(keep_reg)
    counts = counts[ri]
    binary = binary[ri]
    names = [reg_names[i] for i in ri]

    cell_names = [b + split_pattern + project if tag_cells else b for b in barcodes]
    nr_frag = np.asarray(counts.sum(axis=0)).ravel()
    nr_acc = np.asarray(binary.sum(axis=0)).ravel()
    with np.errstate(divide="ignore"):
        obs = pd.DataFrame({
            "cisTopic_nr_frag": nr_frag,
            "cisTopic_log_nr_frag": log10_compat(nr_frag.astype(np.float64), compat),
            "cisTopic_nr_acc": nr_acc,
            "cisTopic_log_nr_acc": log10_compat(nr_acc.astype(np.float64), compat),
            "sample_id": project,
            "Unique_nr_frag": [frag_per_bc.get(i, 0) for i in range(n_cells)],
            "barcode": barcodes,
        }, index=cell_names)
    if min_frag != 1:
        ci = np.flatnonzero(nr_frag >= min_frag)
        counts, binary, obs = counts[:, ci], binary[:, ci], obs.iloc[ci]
    var = _parse_region_names(names)
    var.index = names
    var["Width"] = np.abs(var["End"] - var["Start"]).astype(np.int32)
    var["cisTopic_nr_frag"] = np.asarray(counts.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        var["cisTopic_log_nr_frag"] = log10_compat(var["cisTopic_nr_frag"].to_numpy(np.float64), compat)
    var["cisTopic_nr_acc"] = np.asarray(binary.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        var["cisTopic_log_nr_acc"] = log10_compat(var["cisTopic_nr_acc"].to_numpy(np.float64), compat)
    if min_cell != 1:
        ri = np.flatnonzero(var["cisTopic_nr_acc"].to_numpy() >= min_cell)
        counts, binary, var = counts[ri], binary[ri], var.iloc[ri]

    adata = ad.AnnData(X=sp.csr_matrix(counts.T, dtype=np.int32), obs=obs, var=var)
    adata.layers["binary"] = sp.csr_matrix(binary.T, dtype=np.int32)
    adata.uns["cistopic"] = {"project": project, "fragments": str(fragments)}
    return adata
