"""Shared fragment-file step: aligned read pairs -> 10x-style ``fragments.tsv.gz``.

A fragment is one read pair that passes the filters below, written as

    chrom  start  end  barcode  count

with 0-based half-open coordinates after the Tn5 shift: ``start`` is the
5' end of the forward mate plus 4 and ``end`` the 5' end of the reverse mate
(exclusive) minus 5. The 5' ends are unclipped: soft-clipped bases at a
read's 5' end count. This is the convention of Cell Ranger ATAC and
SnapATAC2 fragment files. Read pairs with the same ``(chrom, start, end, barcode)``
are PCR / optical duplicates of one fragment; they are collapsed into one
line and ``count`` holds how many pairs it had. Lines are sorted by
chromosome (reference order), start, end and barcode, bgzip-compressed and
tabix-indexed.

Pair filters (all configurable): both mates mapped and primary, flagged as a
proper pair, on the same chromosome, MAPQ >= ``min_mapq`` for both mates,
not QC-fail, carrying the barcode tag, and insert size (between the two
5' ends, before the shift) at most ``max_fragment_length``.

Everything here is pysam + numpy; no aligner is called.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from array import array
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


_FLAG_SKIP = 0x4 | 0x100 | 0x200 | 0x800  # unmapped, secondary, qcfail, supplementary


class FragmentCollector:
    """Accumulates shifted fragments in compact integer arrays.

    ``add`` takes one fragment; ``finalize`` sorts, collapses duplicates and
    returns numpy arrays. Barcodes are interned to integer ids so tens of
    millions of fragments fit in a few hundred MB.
    """

    def __init__(self, chroms: Sequence[str]) -> None:
        self.chroms = list(chroms)
        self.chrom_index = {c: i for i, c in enumerate(self.chroms)}
        self.barcodes: List[str] = []
        self.barcode_index: Dict[str, int] = {}
        self._c = array("i")
        self._s = array("q")
        self._e = array("q")
        self._b = array("i")

    def barcode_id(self, bc: str) -> int:
        i = self.barcode_index.get(bc)
        if i is None:
            i = len(self.barcodes)
            self.barcode_index[bc] = i
            self.barcodes.append(bc)
        return i

    def add(self, chrom_id: int, start: int, end: int, barcode: str) -> None:
        self._c.append(chrom_id)
        self._s.append(start)
        self._e.append(end)
        self._b.append(self.barcode_id(barcode))

    def add_arrays(self, chrom_id: np.ndarray, start: np.ndarray, end: np.ndarray, barcode_id: np.ndarray) -> None:
        self._c.extend(np.asarray(chrom_id, dtype=np.int32).tolist())
        self._s.extend(np.asarray(start, dtype=np.int64).tolist())
        self._e.extend(np.asarray(end, dtype=np.int64).tolist())
        self._b.extend(np.asarray(barcode_id, dtype=np.int32).tolist())

    def __len__(self) -> int:
        return len(self._c)

    def finalize(self) -> Dict[str, np.ndarray]:
        """Sorted, de-duplicated fragments as arrays ``chrom, start, end,
        barcode, count`` (``barcode`` holds ids into :attr:`barcodes`)."""
        c = np.frombuffer(self._c, dtype=np.int32) if len(self._c) else np.zeros(0, np.int32)
        s = np.frombuffer(self._s, dtype=np.int64) if len(self._s) else np.zeros(0, np.int64)
        e = np.frombuffer(self._e, dtype=np.int64) if len(self._e) else np.zeros(0, np.int64)
        b = np.frombuffer(self._b, dtype=np.int32) if len(self._b) else np.zeros(0, np.int32)
        return dedup_fragments(c, s, e, b, barcode_names=self.barcodes)


def _name_rank(barcode_names: Sequence[str]) -> np.ndarray:
    """Lexicographic rank of each barcode id."""
    order = np.argsort(np.asarray(list(barcode_names), dtype=object).astype(str), kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank


def _collapse(chrom, start, end, barcode, bkey, weight) -> Dict[str, np.ndarray]:
    n = chrom.size
    if n == 0:
        z = np.zeros(0, dtype=np.int64)
        return {"chrom": z.astype(np.int32), "start": z, "end": z, "barcode": z.astype(np.int32), "count": z}
    idx = np.lexsort((bkey, end, start, chrom))
    c, s, e, b, k, w = chrom[idx], start[idx], end[idx], barcode[idx], bkey[idx], weight[idx]
    new = np.ones(n, dtype=bool)
    new[1:] = (c[1:] != c[:-1]) | (s[1:] != s[:-1]) | (e[1:] != e[:-1]) | (k[1:] != k[:-1])
    first = np.flatnonzero(new)
    return {
        "chrom": c[first].astype(np.int32),
        "start": s[first].astype(np.int64),
        "end": e[first].astype(np.int64),
        "barcode": b[first].astype(np.int32),
        "count": np.add.reduceat(w, first).astype(np.int64),
    }


def dedup_fragments(
    chrom: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    barcode: np.ndarray,
    *,
    count: Optional[np.ndarray] = None,
    barcode_names: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Sort fragments and collapse identical ``(chrom, start, end, barcode)``.

    ``barcode`` is an integer id array; ``count`` (default 1 per row) is
    summed over collapsed rows. When ``barcode_names`` is given, rows at the
    same position are ordered by barcode *string* (as in Cell Ranger output)
    rather than by id.
    """
    chrom = np.asarray(chrom)
    start = np.asarray(start)
    end = np.asarray(end)
    barcode = np.asarray(barcode)
    weight = np.ones(chrom.size, dtype=np.int64) if count is None else np.asarray(count, dtype=np.int64)
    if barcode_names is not None and len(barcode_names) and barcode.size:
        bkey = _name_rank(barcode_names)[barcode]
    else:
        bkey = barcode
    return _collapse(chrom, start, end, barcode, bkey, weight)


def _bgzip_and_index(tsv: Union[str, Path], out_gz: Union[str, Path], threads: int = 1) -> str:
    """bgzip ``tsv`` into ``out_gz`` and build ``out_gz.tbi`` (BED preset)."""
    import pysam

    tsv, out_gz = str(tsv), str(out_gz)
    bgzip = shutil.which("bgzip")
    if bgzip is None:
        from ._env import _env_bin_candidate

        bgzip = _env_bin_candidate("bgzip")
    if bgzip is not None:
        with open(out_gz, "wb") as fo:
            subprocess.run([bgzip, "-c", "-@", str(max(1, threads)), tsv], stdout=fo, check=True)
        os.remove(tsv)
    else:
        pysam.tabix_compress(tsv, out_gz, force=True)
        os.remove(tsv)
    pysam.tabix_index(out_gz, preset="bed", force=True, zerobased=True)
    return out_gz


def write_fragments(
    frags: Mapping[str, np.ndarray],
    chroms: Sequence[str],
    barcodes: Sequence[str],
    out_gz: Union[str, Path],
    *,
    barcode_translation: Optional[Mapping[str, str]] = None,
    threads: int = 1,
    chunk_size: int = 2_000_000,
) -> str:
    """Write de-duplicated fragments (output of :func:`dedup_fragments`) as a
    bgzipped, tabix-indexed ``fragments.tsv.gz``.

    ``barcode_translation`` maps barcodes to the names written (e.g. 10x
    Multiome ATAC -> GEX barcodes); barcodes missing from the map are written
    unchanged.
    """
    out_gz = Path(out_gz)
    out_gz.parent.mkdir(parents=True, exist_ok=True)
    if not str(out_gz).endswith(".gz"):
        raise ValueError("out_gz must end with .gz")
    tsv = Path(str(out_gz)[:-3] + ".tmp.tsv")
    chroms_arr = np.asarray(list(chroms), dtype=object)
    names = list(barcodes)
    if barcode_translation:
        names = [barcode_translation.get(b, b) for b in names]
    names_arr = np.asarray(names, dtype=object)
    c, s, e, b, n = (frags["chrom"], frags["start"], frags["end"], frags["barcode"], frags["count"])
    # A translation that merges barcodes (or changes their sort order)
    # needs a re-sort / re-collapse under the new names.
    if barcode_translation and c.size:
        uniq, inv = np.unique(names_arr.astype(str), return_inverse=True)
        d = dedup_fragments(c, s, e, inv[b], count=n)
        c, s, e, b, n = d["chrom"], d["start"], d["end"], d["barcode"], d["count"]
        names_arr = uniq.astype(object)
    with open(tsv, "w") as fh:
        for i0 in range(0, c.size, chunk_size):
            sl = slice(i0, i0 + chunk_size)
            cols = (
                chroms_arr[c[sl]],
                s[sl].astype(str),
                e[sl].astype(str),
                names_arr[b[sl]],
                n[sl].astype(str),
            )
            fh.write("".join(f"{a}\t{x}\t{y}\t{z}\t{w}\n" for a, x, y, z, w in zip(*cols)))
    return _bgzip_and_index(tsv, out_gz, threads=threads)


def _unclipped_start(r) -> int:
    c = r.cigartuples
    clip = c[0][1] if c and c[0][0] == 4 else 0
    return max(0, r.reference_start - clip)


def _unclipped_end(r) -> Optional[int]:
    end = r.reference_end
    if end is None:
        return None
    c = r.cigartuples
    return end + (c[-1][1] if c and c[-1][0] == 4 else 0)


def pair_to_fragment(
    r1,
    r2,
    *,
    min_mapq: int = 30,
    shift_plus: int = 4,
    shift_minus: int = -5,
    max_fragment_length: Optional[int] = 2000,
    require_proper_pair: bool = True,
) -> Optional[Tuple[int, int, int]]:
    """``(reference_id, start, end)`` of one read pair, or ``None`` if the
    pair fails a filter. ``r1``/``r2`` are pysam ``AlignedSegment`` mates."""
    if (r1.flag & _FLAG_SKIP) or (r2.flag & _FLAG_SKIP):
        return None
    if require_proper_pair and not (r1.is_proper_pair and r2.is_proper_pair):
        return None
    if r1.reference_id != r2.reference_id:
        return None
    if r1.mapping_quality < min_mapq or r2.mapping_quality < min_mapq:
        return None
    if r1.is_reverse == r2.is_reverse:
        return None
    # Tn5 cut sites are the 5' ends of the two mates: the start of the
    # forward mate and the end of the reverse mate, *unclipped* (bases the
    # aligner soft-clipped at the 5' end, usually sequencing errors near the
    # read start, still mark where the read began). Taking min / max over
    # both mates instead would pick up the 3' ends of reads that run into
    # adapter on short fragments.
    fwd, rev = (r2, r1) if r1.is_reverse else (r1, r2)
    left = _unclipped_start(fwd)
    right = _unclipped_end(rev)
    if right is None or right <= left:
        return None
    if max_fragment_length is not None and right - left > max_fragment_length:
        return None
    start = left + shift_plus
    end = right + shift_minus
    if end <= start:
        return None
    return r1.reference_id, start, end


class PairStats:
    """Counters filled while pairing reads."""

    __slots__ = ("pairs", "both_mapped", "passing", "with_barcode", "fragments")

    def __init__(self) -> None:
        self.pairs = 0
        self.both_mapped = 0
        self.passing = 0
        self.with_barcode = 0
        self.fragments = 0

    def merge(self, other: "PairStats") -> None:
        for k in self.__slots__:
            setattr(self, k, getattr(self, k) + getattr(other, k))

    def as_dict(self) -> Dict[str, int]:
        return {k: getattr(self, k) for k in self.__slots__}


def _collect_from_reads(
    reads: Iterable,
    collector: FragmentCollector,
    stats: PairStats,
    *,
    barcode_tag: Optional[str],
    constant_barcode: Optional[str],
    min_mapq: int,
    shift_plus: int,
    shift_minus: int,
    max_fragment_length: Optional[int],
    require_proper_pair: bool,
    tid_to_cid: np.ndarray,
) -> None:
    pending: Dict[str, object] = {}
    for r in reads:
        if r.is_secondary or r.is_supplementary or not r.is_paired:
            continue
        name = r.query_name
        mate = pending.pop(name, None)
        if mate is None:
            pending[name] = r
            continue
        stats.pairs += 1
        if mate.is_unmapped or r.is_unmapped:
            continue
        stats.both_mapped += 1
        hit = pair_to_fragment(
            mate, r,
            min_mapq=min_mapq, shift_plus=shift_plus, shift_minus=shift_minus,
            max_fragment_length=max_fragment_length, require_proper_pair=require_proper_pair,
        )
        if hit is None:
            continue
        stats.passing += 1
        if constant_barcode is not None:
            bc = constant_barcode
        else:
            if r.has_tag(barcode_tag):
                bc = r.get_tag(barcode_tag)
            elif mate.has_tag(barcode_tag):
                bc = mate.get_tag(barcode_tag)
            else:
                continue
        stats.with_barcode += 1
        tid, s, e = hit
        cid = tid_to_cid[tid]
        if cid < 0:
            continue
        collector.add(int(cid), s, e, bc)
        stats.fragments += 1


def _chunk_worker(args):
    import pysam

    (bam, region, kw, chroms) = args
    collector = FragmentCollector(chroms)
    stats = PairStats()
    with pysam.AlignmentFile(bam, "rb") as fh:
        tid_to_cid = _tid_map(fh, chroms)
        _collect_from_reads(fh.fetch(region), collector, stats, tid_to_cid=tid_to_cid, **kw)
    d = collector.finalize()
    return d, collector.barcodes, stats


def _tid_map(fh, chroms: Sequence[str]) -> np.ndarray:
    pos = {c: i for i, c in enumerate(chroms)}
    return np.asarray([pos.get(r, -1) for r in fh.references], dtype=np.int64)


def collect_fragments(
    bam: Union[str, Path],
    *,
    barcode_tag: Optional[str] = "CB",
    constant_barcode: Optional[str] = None,
    min_mapq: int = 30,
    shift_plus: int = 4,
    shift_minus: int = -5,
    max_fragment_length: Optional[int] = 2000,
    require_proper_pair: bool = True,
    chroms: Optional[Sequence[str]] = None,
    threads: int = 1,
) -> Tuple[Dict[str, np.ndarray], List[str], List[str], PairStats]:
    """Read pairs of ``bam`` -> de-duplicated fragment arrays.

    Works on name-grouped (aligner output) and coordinate-sorted BAMs. An
    indexed coordinate-sorted BAM is processed one chromosome per worker
    when ``threads > 1``. Returns ``(fragments, chroms, barcodes, stats)``.
    """
    import pysam

    bam = str(bam)
    kw = dict(
        barcode_tag=barcode_tag, constant_barcode=constant_barcode, min_mapq=min_mapq,
        shift_plus=shift_plus, shift_minus=shift_minus, max_fragment_length=max_fragment_length,
        require_proper_pair=require_proper_pair,
    )
    if barcode_tag is None and constant_barcode is None:
        raise ValueError("give barcode_tag or constant_barcode")
    with pysam.AlignmentFile(bam, "rb") as fh:
        refs = list(fh.references)
        so = fh.header.to_dict().get("HD", {}).get("SO", "unknown")
        indexed = so == "coordinate" and _has_index(fh)
    chroms = list(chroms) if chroms is not None else refs

    if indexed and threads > 1:
        from multiprocessing import get_context

        jobs = [(bam, c, kw, chroms) for c in chroms if c in refs]
        with get_context("spawn").Pool(min(threads, len(jobs) or 1)) as pool:
            parts = pool.map(_chunk_worker, jobs)
        return _merge_parts(parts, chroms)

    collector = FragmentCollector(chroms)
    stats = PairStats()
    with pysam.AlignmentFile(bam, "rb", threads=max(1, min(threads, 4))) as fh:
        tid_to_cid = _tid_map(fh, chroms)
        _collect_from_reads(fh.fetch(until_eof=True), collector, stats, tid_to_cid=tid_to_cid, **kw)
    return collector.finalize(), chroms, collector.barcodes, stats


def _has_index(fh) -> bool:
    try:
        return fh.check_index()
    except (ValueError, AttributeError):
        return False


def _merge_parts(parts, chroms):
    index: Dict[str, int] = {}
    cols = {k: [] for k in ("chrom", "start", "end", "barcode", "count")}
    stats = PairStats()
    for d, bcs, st in parts:
        stats.merge(st)
        remap = np.asarray([index.setdefault(b, len(index)) for b in bcs], dtype=np.int32)
        for k in cols:
            v = d[k]
            if k == "barcode" and v.size:
                v = remap[v]
            cols[k].append(v)
    names = list(index)
    out = {k: np.concatenate(v) for k, v in cols.items()} if parts else _collapse(*(np.zeros(0, np.int64),) * 6)
    # parts are disjoint chromosomes; restore the global sort order
    if out["chrom"].size:
        out = dedup_fragments(out["chrom"], out["start"], out["end"], out["barcode"],
                              count=out["count"], barcode_names=names)
    return out, list(chroms), names, stats


def barcode_table(frags: Mapping[str, np.ndarray], barcodes: Sequence[str]):
    """Per-barcode ``n_fragments`` (unique) and ``n_read_pairs`` (sum of
    counts), sorted by ``n_fragments`` descending."""
    import pandas as pd

    nb = len(barcodes)
    uniq = np.bincount(frags["barcode"], minlength=nb)
    pairs = np.bincount(frags["barcode"], weights=frags["count"], minlength=nb).astype(np.int64)
    df = pd.DataFrame({"barcode": list(barcodes), "n_fragments": uniq, "n_read_pairs": pairs})
    df = df[df["n_fragments"] > 0]
    return df.sort_values(["n_fragments", "barcode"], ascending=[False, True]).reset_index(drop=True)


def bam_to_fragments(
    bam: Union[str, Path],
    out_gz: Union[str, Path],
    *,
    barcode_tag: Optional[str] = "CB",
    constant_barcode: Optional[str] = None,
    min_mapq: int = 30,
    shift_plus: int = 4,
    shift_minus: int = -5,
    max_fragment_length: Optional[int] = 2000,
    require_proper_pair: bool = True,
    chroms: Optional[Sequence[str]] = None,
    barcode_translation: Optional[Mapping[str, str]] = None,
    threads: int = 1,
) -> str:
    """Paired-end BAM -> tabix-indexed 10x-style ``fragments.tsv.gz``.

    The single-cell counterpart of :func:`epione.upstream.bam_to_frags` (which
    keeps every same-chromosome pair of a bulk BAM, unshifted, under one
    sample name). Here pairs are filtered (proper pair, both mates
    MAPQ >= ``min_mapq``, primary, barcode tag present), Tn5-shifted
    (``+shift_plus`` on the left end, ``shift_minus`` on the right end) and
    de-duplicated per barcode; the 5th column counts the read pairs of each
    fragment. Use ``constant_barcode="sample"`` for a bulk library.

    Accepts name-grouped (straight from the aligner) or coordinate-sorted
    BAMs; an indexed coordinate-sorted BAM is split by chromosome over
    ``threads`` worker processes. ``chroms`` restricts and orders the output
    chromosomes (default: BAM header order). Returns the output path.
    """
    frags, chroms_out, barcodes, _ = collect_fragments(
        bam, barcode_tag=barcode_tag, constant_barcode=constant_barcode, min_mapq=min_mapq,
        shift_plus=shift_plus, shift_minus=shift_minus, max_fragment_length=max_fragment_length,
        require_proper_pair=require_proper_pair, chroms=chroms, threads=threads,
    )
    return write_fragments(frags, chroms_out, barcodes, out_gz, barcode_translation=barcode_translation, threads=threads)


def read_fragments(path: Union[str, Path], *, nrows: Optional[int] = None):
    """Load a fragments file into a DataFrame (``chrom, start, end, barcode,
    count``); ``#`` header lines are skipped."""
    import pandas as pd

    return pd.read_csv(
        path, sep="\t", header=None, comment="#", nrows=nrows,
        names=["chrom", "start", "end", "barcode", "count"],
        dtype={"chrom": str, "start": np.int64, "end": np.int64, "barcode": str, "count": np.int64},
    )


__all__ = [
    "FragmentCollector",
    "dedup_fragments",
    "write_fragments",
    "pair_to_fragment",
    "collect_fragments",
    "bam_to_fragments",
    "barcode_table",
    "read_fragments",
]
