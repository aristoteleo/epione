"""Parallel, vectorised reader for ``fragments.tsv[.gz]`` files.

Every fragment-based step (``import_fragments``, ``tsse``,
``add_tile_matrix``, ``add_gene_score_matrix``) streams the whole fragment
file. Doing that line by line in Python dominates their runtime, so this
module splits the file into byte ranges that are decoded independently:

* **BGZF** (bgzip / tabix input) is a series of independent gzip blocks,
  so each worker locates the first block boundary at or after its byte
  offset, inflates its own blocks and keeps only the lines that *start*
  inside its range (dropping the partial first line, completing the last
  one from the following block).
* plain text files are split the same way without inflating.
* plain (non-BGZF) gzip cannot be split; it is inflated sequentially and
  only parsing is parallel.

Each chunk of complete lines is parsed by ``pyarrow.csv`` into numpy
columns (chromosome / barcode dictionary-encoded) and handed to a
caller-supplied reduction ``fn`` that runs in the worker thread. Every
expensive call (``zlib``, ``pyarrow``, numpy sorting / arithmetic)
releases the GIL, so ``n_jobs`` threads scale close to linearly.

Parsing follows the legacy line-by-line reader exactly: ``#`` lines and
lines with fewer than four columns or non-integer coordinates are
skipped, and the count column (5th) is used only when it is all digits
(otherwise 1). Chunks containing anything the fast parser could treat
differently (comments after the header, ``\\r``, spaces, ragged rows,
non-digit counts, invalid integers) are re-parsed by a Python
implementation of the legacy rules, so results are identical either way.
"""
from __future__ import annotations

import io
import os
import struct
import zlib

try:  # ISA-L inflate (~2.5x zlib, releases the GIL); optional, same bytes out
    from isal import isal_zlib as _inflater
except ImportError:  # pragma: no cover
    _inflater = zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterator, NamedTuple, Optional

import numpy as np

__all__ = ["FragmentChunk", "map_fragment_chunks", "resolve_n_jobs", "BarcodeLookup",
           "release_free_memory"]


def release_free_memory() -> None:
    """Hand memory freed by the worker threads back to the OS.

    Thousands of few-MB per-chunk arrays are freed from per-thread malloc
    arenas (and pyarrow's pool), which keep the pages: without this the
    process RSS stays GBs above what is live. Best effort (glibc only)."""
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def memory_limit_bytes() -> int:
    """Memory this process may use: the smallest of the machine's RAM and
    every enclosing cgroup limit (v1 ``memory.limit_in_bytes`` up the
    hierarchy, v2 ``memory.max``), e.g. a Slurm ``--mem`` step limit."""
    lim = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    lim = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    try:
        with open("/proc/self/cgroup") as fh:
            lines = fh.read().splitlines()
    except OSError:
        lines = []
    cands = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        ctrl, path = parts[1], parts[2]
        if "memory" in ctrl.split(","):
            root, fname = "/sys/fs/cgroup/memory", "memory.limit_in_bytes"
        elif ctrl == "":
            root, fname = "/sys/fs/cgroup", "memory.max"
        else:
            continue
        while True:
            cands.append(os.path.join(root + path, fname))
            if path in ("/", ""):
                break
            path = os.path.dirname(path)
    for f in cands:
        try:
            v = open(f).read().strip()
        except OSError:
            continue
        if v and v != "max":
            v = int(v)
            if v < (1 << 62):
                lim = v if lim is None else min(lim, v)
    return int(lim) if lim else 1 << 62


_BGZF_MAGIC = b"\x1f\x8b\x08\x04"
_GZIP_MAGIC = b"\x1f\x8b"
# Compressed bytes per BGZF task (~4x as much text, ~1.5M fragments).
_CHUNK_BYTES = 16 << 20


def resolve_n_jobs(n_jobs: Optional[int] = None) -> int:
    """Number of worker threads: ``n_jobs`` if positive, else every CPU
    this process may run on (``os.sched_getaffinity``, which honours
    Slurm / cgroup CPU allocations), capped by ``OMP_NUM_THREADS`` when
    that is set -- the same limit NumPy's BLAS and pyarrow obey."""
    if n_jobs is not None and int(n_jobs) > 0:
        return int(n_jobs)
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS / Windows
        n = os.cpu_count() or 1
    try:
        omp = int(os.environ.get("OMP_NUM_THREADS", "0"))
    except ValueError:
        omp = 0
    if omp > 0:
        n = min(n, omp)
    return max(1, n)


class FragmentChunk(NamedTuple):
    """Parsed columns of a block of fragment lines (file order kept)."""
    chrom: np.ndarray          # int32 codes into ``chrom_names``
    chrom_names: list          # list[str]
    start: np.ndarray          # int64
    end: np.ndarray            # int64
    barcode: np.ndarray        # int32 codes into ``barcode_names``
    barcode_names: object      # pyarrow.StringArray
    count: np.ndarray          # int64 (5th column, 1 when absent / non-digit)

    def __len__(self):
        return len(self.start)


class BarcodeLookup:
    """Map chunk barcode dictionaries to row indices of a fixed barcode
    list, reproducing ``{b: i for i, b in enumerate(barcodes)}`` (the
    *last* occurrence wins for duplicated barcodes). Pass ``str`` barcodes
    (e.g. ``[str(b) for b in ...]``) when the legacy code compared
    ``str(b)``."""

    def __init__(self, barcodes):
        import pyarrow as pa
        # Only string keys can ever equal a barcode read from the file.
        d = {b: i for i, b in enumerate(barcodes) if isinstance(b, str)}
        self._values = pa.array(list(d), type=pa.string())
        self._index = np.fromiter(d.values(), dtype=np.int64, count=len(d))

    def __call__(self, names) -> np.ndarray:
        """``names`` (pyarrow string array) → int64 indices, -1 if absent."""
        import pyarrow.compute as pc
        pos = pc.index_in(names, value_set=self._values)
        pos = pos.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)
        out = np.full(len(pos), -1, dtype=np.int64)
        hit = pos >= 0
        out[hit] = self._index[pos[hit]]
        return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _empty_chunk() -> FragmentChunk:
    import pyarrow as pa
    e = np.zeros(0, dtype=np.int64)
    return FragmentChunk(np.zeros(0, np.int32), [], e, e.copy(),
                         np.zeros(0, np.int32), pa.array([], type=pa.string()),
                         e.copy())


def _parse_python(text: bytes) -> FragmentChunk:
    """Legacy line-by-line rules (text mode, universal newlines)."""
    import pyarrow as pa
    chrom_ids: dict = {}
    bc_ids: dict = {}
    chroms, starts, ends, bcs, cnts = [], [], [], [], []
    for line in io.TextIOWrapper(io.BytesIO(text)):
        if not line or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        try:
            start = int(parts[1]); end = int(parts[2])
        except ValueError:
            continue
        cnt = int(parts[4]) if len(parts) >= 5 and parts[4].isdigit() else 1
        chroms.append(chrom_ids.setdefault(parts[0], len(chrom_ids)))
        bcs.append(bc_ids.setdefault(parts[3], len(bc_ids)))
        starts.append(start); ends.append(end); cnts.append(cnt)
    return FragmentChunk(
        np.asarray(chroms, dtype=np.int32), list(chrom_ids),
        np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64),
        np.asarray(bcs, dtype=np.int32), pa.array(list(bc_ids), type=pa.string()),
        np.asarray(cnts, dtype=np.int64),
    )


def _concat(pieces) -> np.ndarray:
    """Concatenate byte pieces into one uint8 array (numpy copies release
    the GIL, unlike ``bytes.join``)."""
    total = sum(len(p) for p in pieces)
    buf = np.empty(total, dtype=np.uint8)
    pos = 0
    for p in pieces:
        n = len(p)
        if n:
            buf[pos:pos + n] = np.frombuffer(p, dtype=np.uint8)
            pos += n
    return buf


def _find_newline(buf: np.ndarray, pos: int) -> int:
    """Index of the first ``\\n`` at or after ``pos`` (-1 if none)."""
    w = 1 << 16
    while True:
        i = bytes(buf[pos:pos + w]).find(b"\n")
        if i >= 0:
            return pos + i
        if pos + w >= len(buf):
            return -1
        w *= 4


def _count_values(dict_values) -> np.ndarray:
    """Legacy count rule on the distinct strings of the 5th column."""
    return np.asarray([int(v) if v.isdigit() else 1 for v in dict_values],
                      dtype=np.int64)


def _parse_text(buf: np.ndarray, n_cols: int) -> FragmentChunk:
    """Parse complete lines (uint8 buffer); pyarrow fast path, legacy
    Python fallback for anything pyarrow could read differently."""
    # Leading header / comment lines (Cell Ranger writes a ``#`` header).
    pos, n = 0, len(buf)
    while pos < n and buf[pos] == 35:            # '#'
        nl = _find_newline(buf, pos)
        if nl < 0:
            return _empty_chunk()
        pos = nl + 1
    if pos >= n:
        return _empty_chunk()
    view = buf[pos:]
    if n_cols < 4:
        return _parse_python(view.tobytes())
    import pyarrow as pa
    import pyarrow.csv as pacsv
    names = [f"c{i}" for i in range(n_cols)]
    dict_t = pa.dictionary(pa.int32(), pa.string())
    # Counts are read as (few) distinct strings so the legacy
    # ``isdigit()`` rule applies verbatim (' 3', '+3', '-3', '.', '' -> 1).
    types = {"c0": dict_t, "c1": pa.int64(), "c2": pa.int64(), "c3": dict_t}
    if n_cols >= 5:
        types["c4"] = dict_t
    try:
        tbl = pacsv.read_csv(
            pa.py_buffer(view),
            read_options=pacsv.ReadOptions(
                column_names=names, use_threads=False,
                block_size=max(len(view) + 1, 1 << 20)),
            parse_options=pacsv.ParseOptions(
                delimiter="\t", quote_char=False, double_quote=False,
                escape_char=False, newlines_in_values=False,
                ignore_empty_lines=True),
            convert_options=pacsv.ConvertOptions(
                column_types=types, include_columns=names[:min(n_cols, 5)],
                strings_can_be_null=False, quoted_strings_can_be_null=False),
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        return _parse_python(view.tobytes())
    if tbl.num_rows == 0:
        return _empty_chunk()
    tbl = tbl.combine_chunks()
    if tbl.column("c1").null_count or tbl.column("c2").null_count:
        return _parse_python(view.tobytes())
    ch = tbl.column("c0").chunk(0)
    chrom_names = ch.dictionary.to_pylist()
    if any(c.startswith("#") for c in chrom_names):   # comment inside the chunk
        return _parse_python(view.tobytes())
    start = tbl.column("c1").chunk(0).to_numpy()
    end = tbl.column("c2").chunk(0).to_numpy()
    if n_cols >= 5:
        cc = tbl.column("c4").chunk(0)
        count = _count_values(cc.dictionary.to_pylist())[cc.indices.to_numpy()]
    else:
        count = np.ones(len(start), dtype=np.int64)
    bc = tbl.column("c3").chunk(0)
    return FragmentChunk(
        ch.indices.to_numpy().astype(np.int32, copy=False),
        chrom_names,
        start, end,
        bc.indices.to_numpy().astype(np.int32, copy=False),
        bc.dictionary, count,
    )


# ---------------------------------------------------------------------------
# Byte-range splitting
# ---------------------------------------------------------------------------

def _bgzf_header(buf, off: int) -> Optional[tuple]:
    """Return ``(block_size, header_len)`` of a BGZF block at ``off``."""
    if len(buf) - off < 18 or buf[off:off + 4] != _BGZF_MAGIC:
        return None
    xlen = struct.unpack_from("<H", buf, off + 10)[0]
    p, end = off + 12, off + 12 + xlen
    if end > len(buf):
        return None
    while p + 4 <= end:
        si1, si2, slen = buf[p], buf[p + 1], struct.unpack_from("<H", buf, p + 2)[0]
        if si1 == 66 and si2 == 67 and slen == 2 and p + 6 <= end:
            bsize = struct.unpack_from("<H", buf, p + 4)[0] + 1
            if bsize < 12 + xlen + 8:
                return None
            return bsize, 12 + xlen
        p += 4 + slen
    return None


def _is_bgzf(path: str) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(64)
    return _bgzf_header(head, 0) is not None


def _find_block(fd: int, pos: int, size: int) -> int:
    """Offset of the first BGZF block starting at or after ``pos``
    (``size`` if none). A candidate is accepted only when the block that
    follows it also has a valid header (or it ends exactly at EOF), so a
    false magic match inside compressed data is practically impossible."""
    if pos <= 0:
        return 0
    win = 1 << 17
    while pos < size:
        buf = os.pread(fd, win + 64, pos)
        i = 0
        while True:
            i = buf.find(_BGZF_MAGIC, i)
            if i < 0 or i >= win:
                break
            hdr = _bgzf_header(buf, i)
            if hdr is not None:
                nxt = pos + i + hdr[0]
                if nxt == size or (nxt < size and _bgzf_header(os.pread(fd, 64, nxt), 0)):
                    return pos + i
            i += 1
        pos += win
    return size


def _inflate(fd: int, a: int, b: int) -> list:
    """Inflate every BGZF block in the compressed byte range ``[a, b)``
    (one ``bytes`` per block)."""
    raw = os.pread(fd, b - a, a)
    mv = memoryview(raw)
    outs = []
    off = 0
    while off < len(raw):
        hdr = _bgzf_header(raw, off)
        if hdr is None:
            raise ValueError(f"corrupt BGZF block at byte {a + off}")
        bsize, hlen = hdr
        data = _inflater.decompress(mv[off + hlen: off + bsize - 8], -15)
        crc, isize = struct.unpack_from("<II", raw, off + bsize - 8)
        if len(data) != isize or zlib.crc32(data) != crc:
            raise ValueError(f"BGZF checksum mismatch at byte {a + off}")
        outs.append(data)
        off += bsize
    return outs


def _first_line_tail(read_next: Callable[[int], tuple], pos: int, size: int) -> list:
    """Pieces from ``pos`` up to and including the first newline."""
    parts = []
    while pos < size:
        pieces, pos = read_next(pos)
        for data in pieces:
            nl = data.find(b"\n")
            if nl >= 0:
                parts.append(memoryview(data)[:nl + 1])
                return parts
            parts.append(data)
    return parts


def _range_text(path: str, bgzf: bool, a_raw: int, b_raw: int, size: int,
                fd: int) -> np.ndarray:
    """Complete lines whose first byte lies in this task's range (uint8).

    Task ranges tile the file; a task drops everything up to and including
    the first newline (it belongs to the previous task) unless it starts
    the file, and extends its last line into the next range."""
    if bgzf:
        a = _find_block(fd, a_raw, size)
        b = _find_block(fd, b_raw, size) if b_raw < size else size
        if a >= b:
            return np.zeros(0, np.uint8)
        pieces = _inflate(fd, a, b)

        def read_next(p):
            hdr = _bgzf_header(os.pread(fd, 64, p), 0)
            if hdr is None:
                raise ValueError(f"corrupt BGZF block at byte {p}")
            return _inflate(fd, p, p + hdr[0]), p + hdr[0]
    else:
        a, b = a_raw, min(b_raw, size)
        if a >= b:
            return np.zeros(0, np.uint8)
        pieces = [os.pread(fd, b - a, a)]

        def read_next(p):
            q = min(p + (1 << 16), size)
            return [os.pread(fd, q - p, p)], q

    if a > 0:
        for i, data in enumerate(pieces):
            nl = data.find(b"\n")
            if nl >= 0:
                pieces = [memoryview(data)[nl + 1:]] + pieces[i + 1:]
                break
        else:
            # A single line spans the whole range: owned by an earlier task.
            return np.zeros(0, np.uint8)
    if b < size:
        pieces.extend(_first_line_tail(read_next, b, size))
    return _concat(pieces)


def _count_columns(path: str, bgzf: bool, gz: bool) -> int:
    """Number of tab-separated columns on the first data line."""
    import gzip
    opener = gzip.open if gz else open
    with opener(path, "rb") as fh:
        for line in fh:
            if line.startswith(b"#") or not line.strip():
                continue
            return len(line.rstrip(b"\r\n").split(b"\t"))
    return 0


def _gzip_text_chunks(path: str, chunk: int) -> Iterator[bytes]:
    """Sequentially inflate a (multi-member) gzip file into chunks of
    complete lines."""
    carry = b""
    with open(path, "rb") as fh:
        d = zlib.decompressobj(31)
        while True:
            raw = fh.read(chunk)
            if not raw:
                break
            out = []
            while raw:
                out.append(d.decompress(raw))
                if d.eof:
                    raw = d.unused_data
                    d = zlib.decompressobj(31)
                else:
                    raw = b""
            buf = carry + b"".join(out)
            nl = buf.rfind(b"\n")
            if nl < 0:
                carry = buf
                continue
            carry = buf[nl + 1:]
            yield buf[:nl + 1]
        tail = carry + d.flush()
        if tail:
            yield tail


def map_fragment_chunks(
    path,
    fn: Callable[[FragmentChunk], object],
    *,
    n_jobs: Optional[int] = None,
    chunk_bytes: Optional[int] = None,
) -> Iterator:
    """Yield ``fn(chunk)`` for consecutive chunks of ``path`` in file order.

    ``fn`` runs in a pool of ``n_jobs`` threads (see
    :func:`resolve_n_jobs`); at most ``2 * n_jobs`` chunks are in flight,
    which bounds memory to a few hundred MB per thread.
    """
    path = str(path)
    n_jobs = resolve_n_jobs(n_jobs)
    chunk_bytes = int(chunk_bytes or _CHUNK_BYTES)
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        magic = fh.read(2)
    gz = magic == _GZIP_MAGIC
    bgzf = gz and _is_bgzf(path)
    n_cols = _count_columns(path, bgzf, gz)
    if size == 0 or n_cols == 0:
        return

    fd = os.open(path, os.O_RDONLY)
    try:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            if gz and not bgzf:
                def job_text(text):
                    return fn(_parse_text(np.frombuffer(text, dtype=np.uint8), n_cols))
                tasks = (pool.submit(job_text, t)
                         for t in _gzip_text_chunks(path, chunk_bytes))
            else:
                step = chunk_bytes if bgzf else chunk_bytes * 4
                bounds = list(range(0, size, step)) + [size]

                def job(i):
                    text = _range_text(path, bgzf, bounds[i], bounds[i + 1], size, fd)
                    return fn(_parse_text(text, n_cols))
                tasks = (pool.submit(job, i) for i in range(len(bounds) - 1))
            window = deque()
            for fut in tasks:
                window.append(fut)
                if len(window) >= 2 * n_jobs:
                    yield window.popleft().result()
            while window:
                yield window.popleft().result()
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Sparse (key, weight) reductions used by the streaming consumers
# ---------------------------------------------------------------------------

def sorted_reduce(keys: np.ndarray, w: np.ndarray, key_max: int,
                  runs: bool = False) -> tuple:
    """Sort integer ``keys`` (``0 <= keys < key_max``) and sum the
    non-negative integer weights ``w`` of equal keys.

    Returns ``(unique_keys int64, summed_w int64)``. When key and weight
    fit together in 63 bits they are packed into one int64 and sorted with
    ``np.sort`` (much faster than ``argsort`` + gather); otherwise an
    ``argsort`` path is used. Both give identical results. ``runs=True``
    selects a stable (timsort) sort, which is near-linear when the input
    is a concatenation of already sorted pieces.
    """
    keys = np.asarray(keys, dtype=np.int64)
    w = np.asarray(w, dtype=np.int64)
    if len(keys) == 0:
        return keys, w
    kbits = max(int(key_max - 1).bit_length(), 1)
    wbits = 63 - kbits
    # Packed path: each weight must fit in the low bits (sums are taken
    # after unpacking, in int64).
    if (wbits >= 1 and int(w.min()) >= 0 and int(w.max()) < (1 << wbits)
            and int(keys.min()) >= 0 and int(keys.max()) < key_max):
        packed = np.sort((keys << wbits) | w, kind="stable" if runs else None)
        k = packed >> wbits
        wv = packed & ((1 << wbits) - 1)
    else:
        order = np.argsort(keys, kind="stable")
        k = keys[order]
        wv = w[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    return k[starts], np.add.reduceat(wv, starts)
