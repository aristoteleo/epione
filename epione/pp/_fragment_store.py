"""In-memory per-cell fragment store (parse the fragment file once).

:func:`epi.pp.import_fragments` parses the fragment file a single time;
besides the per-cell QC it keeps every fragment of the imported cells in
a compact CSR-by-cell layout (ArchR Arrow / SnapATAC2
``obsm['fragment_paired']`` style)::

    indptr  int64  (n_cells + 1)   rows of cell i: indptr[i]:indptr[i+1]
    chrom   int8 / int16  chromosome code (``chrom_names``), file order kept
    start   int32  fragment start (0-based)
    len     uint16 end - start; 65535 marks the rare fragment >= 65535 bp
                   (or end < start) whose exact end is in long_pos / long_end
    count   uint8 / uint16 / int32   5th column (legacy ``isdigit`` rule)

The store lives in a process-wide registry; ``adata.uns['files']
['fragment_store']`` records its token (a plain string, so the AnnData
still writes to ``.h5ad``). ``tsse`` / ``add_tile_matrix`` /
``add_gene_score_matrix`` look it up and run numba kernels (row blocks on
``n_jobs`` threads) over
it instead of re-reading the file; cells are matched by barcode, so
``adata[keep].copy()`` and any re-ordering stay consistent. When the token
is missing or unknown (new process, AnnData read back from disk, store
released) they fall back to streaming the file -- same results.

Release the memory with :func:`release_fragment_store`.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

import numpy as np

__all__ = ["FragmentStore", "get_fragment_store", "release_fragment_store"]

_STORES: dict = {}


_COLUMNS = ("chrom", "start", "len", "count")
LONG = 65535          # ``len`` sentinel: end stored in (long_pos, long_end)


def allocate_columns(n: int, dtypes: dict, spill: bool):
    """Output columns for a new store: anonymous memory, or (``spill``)
    ``np.memmap`` files in a fresh directory under ``tempfile.gettempdir()``
    (``$TMPDIR``), so the fragments stay in the page cache / on disk and
    only count towards a process's RSS while a step is reading them."""
    if spill and n > 0:
        import tempfile
        try:
            d = tempfile.mkdtemp(prefix="epione_fragstore_")
            cols = {c: np.memmap(os.path.join(d, c + ".bin"), dtype=dtypes[c],
                                 mode="w+", shape=(n,)) for c in _COLUMNS}
            return cols, d
        except OSError:
            pass
    return {c: np.empty(n, dtype=dtypes[c]) for c in _COLUMNS}, None


class FragmentStore:
    """Fragments of a set of cells, grouped by cell (see module docs)."""

    def __init__(self, path, barcodes, chrom_names, indptr, columns: dict,
                 long_pos=None, long_end=None, spill_dir: Optional[str] = None):
        self.path = str(path)
        try:
            st = os.stat(self.path)
            self.file_sig = (st.st_size, st.st_mtime_ns)
        except OSError:
            self.file_sig = None
        self.barcodes = list(barcodes)
        self.chrom_names = list(chrom_names)
        self.indptr = indptr
        self.long_pos = np.zeros(0, np.int64) if long_pos is None else long_pos
        self.long_end = np.zeros(0, np.int32) if long_end is None else long_end
        self._index = None
        self.spill_dir = spill_dir
        if spill_dir is None:
            self._cols = dict(columns)
            self._meta = None
        else:
            # Keep only (file, dtype, n); map on access, unmapped when the
            # caller drops the array.
            self._cols = None
            self._meta = {c: (a.filename, a.dtype, a.shape[0]) for c, a in columns.items()}
            import shutil
            import weakref
            weakref.finalize(self, shutil.rmtree, spill_dir, True)

    def _col(self, name):
        if self._cols is not None:
            return self._cols[name]
        fn, dt, n = self._meta[name]
        return np.memmap(fn, dtype=dt, mode="r", shape=(n,))

    chrom = property(lambda self: self._col("chrom"))
    start = property(lambda self: self._col("start"))
    len = property(lambda self: self._col("len"))

    @property
    def end_args(self):
        """``(len, long_pos, long_end)``: the kernels' view of ``end``."""
        return self._col("len"), self.long_pos, self.long_end
    count = property(lambda self: self._col("count"))

    @property
    def n_fragments(self) -> int:
        return int(self.indptr[-1])

    @property
    def nbytes(self) -> int:
        return int(self.indptr.nbytes + self.long_pos.nbytes + self.long_end.nbytes
                   + sum(self._col(c).nbytes for c in _COLUMNS))

    def row_map(self, keys) -> np.ndarray:
        """Store cell of every key (-1 if absent). Reproduces the legacy
        ``{k: i for i, k in enumerate(keys)}`` lookups: for duplicated keys
        only the *last* occurrence is mapped; non-``str`` keys never match."""
        import pandas as pd
        if self._index is None:
            self._index = pd.Index(self.barcodes)
        keys = list(keys)
        last = {}
        for i, k in enumerate(keys):
            if isinstance(k, str):
                last[k] = i
        out = np.full(len(keys), -1, dtype=np.int64)
        if last:
            rows = np.fromiter(last.values(), dtype=np.int64, count=len(last))
            out[rows] = self._index.get_indexer(list(last))
        return out

    def covering_row_map(self, keys) -> Optional[np.ndarray]:
        """``row_map(keys)`` if every key that the legacy lookup would map
        is in the store, else ``None`` (the caller then streams the file:
        a barcode missing from the store may still have fragments there)."""
        rm = self.row_map(keys)
        keys = list(keys)
        last = {}
        for i, k in enumerate(keys):
            if isinstance(k, str):
                last[k] = i
        if last:
            rows = np.fromiter(last.values(), dtype=np.int64, count=len(last))
            if (rm[rows] < 0).any():
                return None
        return rm

    def chrom_lut(self, mapping: dict, default=-1, dtype=np.int64) -> np.ndarray:
        """``mapping[chrom]`` for every store chromosome code."""
        return np.asarray([mapping.get(c, default) for c in self.chrom_names], dtype=dtype)


def register(store: FragmentStore) -> str:
    token = f"fragstore:{uuid.uuid4().hex}"
    _STORES[token] = store
    return token


def get_fragment_store(adata) -> Optional[FragmentStore]:
    """The store recorded on ``adata`` if it is still alive and matches
    ``adata.uns['files']['fragments']`` (path, size, mtime); else ``None``."""
    try:
        files = adata.uns.get("files", {})
        token = files.get("fragment_store")
        path = files.get("fragments")
    except Exception:
        return None
    if not token or token not in _STORES:
        return None
    store = _STORES[token]
    if path is None or str(path) != store.path:
        return None
    try:
        st = os.stat(store.path)
        if store.file_sig is not None and (st.st_size, st.st_mtime_ns) != store.file_sig:
            return None
    except OSError:
        return None
    return store


def release_fragment_store(adata=None) -> None:
    """Free the in-memory fragment store of ``adata`` (all stores when
    ``adata`` is ``None``). Later steps then re-read the fragment file."""
    if adata is None:
        import shutil
        for st in _STORES.values():
            if st.spill_dir:
                shutil.rmtree(st.spill_dir, ignore_errors=True)
        _STORES.clear()
        return
    token = adata.uns.get("files", {}).get("fragment_store")
    st = _STORES.pop(token, None)
    if st is not None and st.spill_dir:
        import shutil
        shutil.rmtree(st.spill_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------

def _numba():
    import numba
    return numba


_KERNELS: dict = {}


def _kernels():
    """nogil numba kernels over a block of rows ``[r0, r1)``; the Python
    drivers below run blocks on a thread pool (cheaper to JIT than
    ``parallel=True`` and no oversubscription with other numba code)."""
    if _KERNELS:
        return _KERNELS
    nb = _numba()
    jit = nb.njit(nogil=True, cache=False)

    @jit
    def scatter_chunk(dest, bc, chrom, chrom_map, start, lens, count, long_rows, long_end,
                      o_chrom, o_start, o_len, o_count, long_off, o_long_pos, o_long_end):
        cur = dest.copy()
        j = 0
        q = long_off
        nl = long_rows.shape[0]
        for i in range(bc.shape[0]):
            p = cur[bc[i]]
            if p >= 0:
                o_chrom[p] = chrom_map[chrom[i]]
                o_start[p] = start[i]
                o_len[p] = lens[i]
                o_count[p] = count[i]
                cur[bc[i]] = p + 1
            if j < nl and long_rows[j] == i:
                if p >= 0:
                    o_long_pos[q] = p
                    o_long_end[q] = long_end[j]
                    q += 1
                j += 1

    @jit
    def qc_chunk(kind, bc, count, start, end, n_bc, with_store):
        """Per-barcode QC counts of one parsed chunk (+ store length column)."""
        nf = np.zeros(n_bc, dtype=np.int64)
        reads = np.zeros(n_bc, dtype=np.int64)
        nm = np.zeros(n_bc, dtype=np.int64)
        rows = np.zeros(n_bc, dtype=np.int64)
        n = bc.shape[0]
        lens = np.empty(n if with_store else 0, dtype=np.uint16)
        n_long = 0
        for i in range(n):
            b = bc[i]
            k = kind[i]
            if k == 1:
                nf[b] += 1
                reads[b] += count[i]
            elif k == 2:
                nm[b] += 1
            if with_store:
                rows[b] += 1
                ln = np.int64(end[i]) - np.int64(start[i])
                if ln < 0 or ln >= 65535:
                    lens[i] = 65535
                    n_long += 1
                else:
                    lens[i] = ln
        long_rows = np.empty(n_long, dtype=np.int64)
        if n_long:
            j = 0
            for i in range(n):
                if lens[i] == 65535:
                    long_rows[j] = i
                    j += 1
        return nf, reads, nm, rows, lens, long_rows

    @jit
    def _end(p, start, lens, lpos, lend):
        ln = lens[p]
        if ln != 65535:
            return np.int64(start[p]) + np.int64(ln)
        return np.int64(lend[np.searchsorted(lpos, p)])

    @jit
    def _bound(tss, a, b, pos, value, right):
        """searchsorted(tss[a:b], value, side) + a, walking from ``pos``
        (cuts of one cell come in genomic order, so the answer is usually
        a few steps away); falls back to bisection on long jumps."""
        steps = 0
        if right:
            while pos < b and tss[pos] <= value and steps < 16:
                pos += 1
                steps += 1
            while pos > a and tss[pos - 1] > value and steps < 16:
                pos -= 1
                steps += 1
            if steps >= 16:
                return a + np.searchsorted(tss[a:b], value, side="right")
        else:
            while pos < b and tss[pos] < value and steps < 16:
                pos += 1
                steps += 1
            while pos > a and tss[pos - 1] >= value and steps < 16:
                pos -= 1
                steps += 1
            if steps >= 16:
                return a + np.searchsorted(tss[a:b], value, side="left")
        return pos

    @jit
    def tss_pileup(r0, r1, indptr, chrom, start, lens, lpos, lend, row_map, tss_off, tss,
                   up, down, mx):
        nf = up + down + 1
        for i in range(r0, r1):
            s_cell = row_map[i]
            if s_cell < 0:
                continue
            prev_c = -1
            lo0 = hi0 = lo1 = hi1 = 0
            for p in range(indptr[s_cell], indptr[s_cell + 1]):
                c = chrom[p]
                a = tss_off[c]
                b = tss_off[c + 1]
                if a == b:
                    continue
                if c != prev_c:
                    prev_c = c
                    lo0 = hi0 = lo1 = hi1 = a
                s = np.int64(start[p])
                e = _end(p, start, lens, lpos, lend)
                for which in range(2):
                    if which == 0:
                        cut = s
                        lo0 = _bound(tss, a, b, lo0, cut - down, False)
                        hi0 = _bound(tss, a, b, hi0, cut + up, True)
                        lo, hi = lo0, hi0
                    else:
                        cut = e - 1
                        lo1 = _bound(tss, a, b, lo1, cut - down, False)
                        hi1 = _bound(tss, a, b, hi1, cut + up, True)
                        lo, hi = lo1, hi1
                    for k in range(lo, hi):
                        t = tss[k]
                        if s < t + down + 1 and e > t - up:
                            col = cut - (t - up)
                            if 0 <= col < nf:
                                mx[i, col] += 1

    @jit
    def _row_tiles(p0, p1, chrom, start, lens, lpos, lend, count, base, bin_size, fragment_mode, buf):
        """Packed (tile << 32 | weight) insertions of one cell -> buf[:m]."""
        m = 0
        for p in range(p0, p1):
            b0 = base[chrom[p]]
            if b0 < 0:
                continue
            w = np.int64(count[p])
            t1 = b0 + np.int64(start[p]) // bin_size
            t2 = b0 + (_end(p, start, lens, lpos, lend) - 1) // bin_size
            if fragment_mode:
                for t in range(t1, t2 + 1):
                    buf[m] = (t << 32) | w
                    m += 1
            else:
                buf[m] = (t1 << 32) | w
                m += 1
                if t2 != t1:
                    buf[m] = (t2 << 32) | w
                    m += 1
        return m

    @jit
    def _row_sorted(p0, p1, chrom, start, lens, lpos, lend, count, unit, base, bin_size,
                    fragment_mode, out):
        """Packed (tile << 32 | weight) insertions of one cell, sorted by tile,
        in out[:m]. A cell's fragments are stored in file (genomic) order, so
        the start tiles are already sorted and the end tiles nearly so: sort
        the end tiles by (capped) insertion sort and merge, falling back to a
        full sort whenever that assumption does not hold."""
        if fragment_mode:
            m = _row_tiles(p0, p1, chrom, start, lens, lpos, lend, count, base, bin_size,
                           True, out)
            out[:m] = np.sort(out[:m])
            return m
        n = p1 - p0
        A = np.empty(n, dtype=np.int64)
        B = np.empty(n, dtype=np.int64)
        na = 0
        nb_ = 0
        a_sorted = True
        for p in range(p0, p1):
            b0 = base[chrom[p]]
            if b0 < 0:
                continue
            w = np.int64(1) if unit else np.int64(count[p])
            t1 = b0 + np.int64(start[p]) // bin_size
            t2 = b0 + (_end(p, start, lens, lpos, lend) - 1) // bin_size
            A[na] = (t1 << 32) | w
            if na and t1 < (A[na - 1] >> 32):
                a_sorted = False
            na += 1
            if t2 != t1:
                B[nb_] = (t2 << 32) | w
                nb_ += 1
        m = na + nb_
        ok = a_sorted
        if ok:
            moves = 0
            cap = 8 * nb_ + 64
            for k in range(1, nb_):
                v = B[k]
                key = v >> 32
                j = k - 1
                while j >= 0 and (B[j] >> 32) > key:
                    B[j + 1] = B[j]
                    j -= 1
                    moves += 1
                B[j + 1] = v
                if moves > cap:
                    ok = False
                    break
        if not ok:
            out[:na] = A[:na]
            out[na:m] = B[:nb_]
            out[:m] = np.sort(out[:m])
            return m
        i = 0
        j = 0
        q = 0
        while i < na and j < nb_:
            if (A[i] >> 32) <= (B[j] >> 32):
                out[q] = A[i]
                i += 1
            else:
                out[q] = B[j]
                j += 1
            q += 1
        while i < na:
            out[q] = A[i]
            i += 1
            q += 1
        while j < nb_:
            out[q] = B[j]
            j += 1
            q += 1
        return m

    @jit
    def _row_len(p0, p1, chrom, start, lens, lpos, lend, base, bin_size, fragment_mode):
        m = 0
        for p in range(p0, p1):
            b0 = base[chrom[p]]
            if b0 < 0:
                continue
            t1 = np.int64(start[p]) // bin_size
            t2 = (_end(p, start, lens, lpos, lend) - 1) // bin_size
            if fragment_mode:
                if t2 >= t1:
                    m += t2 - t1 + 1
            else:
                m += 2 if t2 != t1 else 1
        return m

    @jit
    def tile_count(r0, r1, indptr, chrom, start, lens, lpos, lend, count, row_map, base, bin_size,
                   n_tiles, fragment_mode, nnz, bad):
        for i in range(r0, r1):
            s_cell = row_map[i]
            if s_cell < 0:
                continue
            p0, p1 = indptr[s_cell], indptr[s_cell + 1]
            buf = np.empty(_row_len(p0, p1, chrom, start, lens, lpos, lend, base, bin_size,
                                    fragment_mode), dtype=np.int64)
            m = _row_sorted(p0, p1, chrom, start, lens, lpos, lend, count, False, base,
                            bin_size, fragment_mode, buf)
            if m == 0:
                continue
            if (buf[0] >> 32) < 0 or (buf[m - 1] >> 32) >= n_tiles:
                bad[0] = 1
            u = 1
            for k in range(1, m):
                if (buf[k] >> 32) != (buf[k - 1] >> 32):
                    u += 1
            nnz[i] = u

    @jit
    def tile_fill(r0, r1, indptr, chrom, start, lens, lpos, lend, count, row_map, base, bin_size,
                  fragment_mode, out_indptr, out_indices, out_data):
        for i in range(r0, r1):
            s_cell = row_map[i]
            if s_cell < 0:
                continue
            p0, p1 = indptr[s_cell], indptr[s_cell + 1]
            buf = np.empty(_row_len(p0, p1, chrom, start, lens, lpos, lend, base, bin_size,
                                    fragment_mode), dtype=np.int64)
            m = _row_sorted(p0, p1, chrom, start, lens, lpos, lend, count, False, base,
                            bin_size, fragment_mode, buf)
            if m == 0:
                continue
            seg = buf
            q = out_indptr[i]
            cur = seg[0] >> 32
            acc = np.int64(0)
            for k in range(m):
                t = seg[k] >> 32
                if t != cur:
                    out_indices[q] = cur
                    out_data[q] = acc          # int64 -> int32 wraps like coo(int32)
                    q += 1
                    cur = t
                    acc = 0
                acc += seg[k] & 0xFFFFFFFF
            out_indices[q] = cur
            out_data[q] = acc

    @jit
    def gene_score(r0, r1, indptr, chrom, start, lens, lpos, lend, row_map, base, tile_size,
                   ceiling, Wp, Wg, Wv, out):
        for i in range(r0, r1):
            s_cell = row_map[i]
            if s_cell < 0:
                continue
            p0, p1 = indptr[s_cell], indptr[s_cell + 1]
            seg = np.empty(2 * (p1 - p0), dtype=np.int64)
            m = _row_sorted(p0, p1, chrom, start, lens, lpos, lend, start, True, base,
                            tile_size, False, seg)
            if m == 0:
                continue
            k = 0
            while k < m:
                t = seg[k] >> 32
                c = 0
                while k < m and (seg[k] >> 32) == t:
                    c += 1
                    k += 1
                if ceiling > 0 and c > ceiling:
                    c = ceiling
                b = np.float32(c)
                for q in range(Wp[t], Wp[t + 1]):
                    out[i, Wg[q]] += Wv[q] * b

    @jit
    def chrom_max_tile(r0, r1, indptr, chrom, start, lens, lpos, lend, tile_size, res):
        neg = 0
        for p in range(indptr[r0], indptr[r1]):
            c = chrom[p]
            s = np.int64(start[p])
            e = _end(p, start, lens, lpos, lend) - 1
            if s < 0 or e < 0:
                neg = 1
                continue
            t = max(s, e) // tile_size
            if t > res[c]:
                res[c] = t
        return neg

    _KERNELS.update(qc_chunk=qc_chunk, scatter_chunk=scatter_chunk, tss_pileup=tss_pileup,
                    tile_count=tile_count, tile_fill=tile_fill,
                    gene_score=gene_score, chrom_max_tile=chrom_max_tile)
    return _KERNELS


def _run_blocks(fn, n_rows: int, n_jobs, *args, weights=None):
    """Call ``fn(r0, r1, *args)`` over row blocks on ``n_jobs`` threads.
    ``weights`` (per-row work estimate, e.g. fragments per cell) balances
    the blocks."""
    from concurrent.futures import ThreadPoolExecutor
    from ._fragment_io import resolve_n_jobs
    nt = resolve_n_jobs(n_jobs)
    n_blocks = max(1, min(n_rows, 4 * nt))
    if weights is not None and n_rows:
        cw = np.cumsum(np.asarray(weights, dtype=np.float64))
        edges = np.searchsorted(cw, np.linspace(0, cw[-1], n_blocks + 1)[1:-1])
        edges = np.unique(np.r_[0, edges, n_rows])
    else:
        edges = np.linspace(0, n_rows, n_blocks + 1).astype(np.int64)
    blocks = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    if not blocks:
        return []
    fn(blocks[0][0], blocks[0][0], *args)          # JIT once, outside the pool
    with ThreadPoolExecutor(max_workers=nt) as pool:
        return list(pool.map(lambda ab: fn(ab[0], ab[1], *args), blocks))


def _row_weights(store, row_map):
    n = np.diff(store.indptr)
    return np.where(row_map >= 0, n[np.maximum(row_map, 0)], 0) + 1


# ---------------------------------------------------------------------------
# Consumers
# ---------------------------------------------------------------------------

def tss_pileup(store: FragmentStore, mx, row_map, tss_by_chrom: dict,
               up: int, down: int, n_jobs=None) -> None:
    """Store-backed equivalent of ``_metric._tss_pileup_stream``."""
    k = _kernels()
    segs = [np.asarray(tss_by_chrom.get(c, np.zeros(0, np.int64)), dtype=np.int64)
            for c in store.chrom_names]
    tss_off = np.zeros(len(segs) + 1, dtype=np.int64)
    tss_off[1:] = np.cumsum([len(s) for s in segs])
    tss = np.concatenate(segs) if segs else np.zeros(0, np.int64)
    _run_blocks(k["tss_pileup"], len(row_map), n_jobs, store.indptr, store.chrom,
                store.start, *store.end_args, row_map, tss_off, tss, int(up), int(down), mx,
                weights=_row_weights(store, row_map))


def tile_matrix(store: FragmentStore, row_map, chrom_offsets: dict, n_tiles: int,
                bin_size: int, *, counting_strategy: str, chrM: set, n_jobs=None):
    """Store-backed equivalent of ``_data._tile_matrix_stream`` (int32 CSR)."""
    import scipy.sparse as sp
    k = _kernels()
    base = store.chrom_lut({c: o for c, o in chrom_offsets.items() if c not in chrM})
    frag_mode = counting_strategy == "fragment"
    n_rows = len(row_map)
    cols = (store.indptr, store.chrom, store.start, *store.end_args, store.count)
    wts = _row_weights(store, row_map)
    nnz = np.zeros(n_rows, dtype=np.int64)
    bad = np.zeros(1, dtype=np.int64)
    _run_blocks(k["tile_count"], n_rows, n_jobs, *cols, row_map, base, int(bin_size),
                int(n_tiles), frag_mode, nnz, bad, weights=wts)
    if bad[0]:
        raise ValueError("column index exceeds matrix dimensions")
    total = int(nnz.sum())
    idx_dtype = np.int32 if max(total, n_tiles, n_rows) < np.iinfo(np.int32).max else np.int64
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(nnz, out=indptr[1:])
    indices = np.empty(total, dtype=idx_dtype)
    data = np.empty(total, dtype=np.int32)
    _run_blocks(k["tile_fill"], n_rows, n_jobs, *cols, row_map, base, int(bin_size),
                frag_mode, indptr, indices, data, weights=wts)
    X = sp.csr_matrix((data, indices, indptr.astype(idx_dtype, copy=False)),
                      shape=(n_rows, n_tiles))
    X.has_sorted_indices = True
    X.has_canonical_format = True
    return X


def build_gene_weights(store: FragmentStore, genes_sorted, chroms_with_genes, *,
                       tile_size, extend_upstream, extend_downstream, gene_upstream,
                       gene_downstream, use_gene_boundaries, gene_scale_factor, n_jobs=None):
    """Tile -> gene weight matrix (CSR rows = gene-score tiles) over every tile
    of every gene domain, plus the store-chrom -> tile-base lookup.

    Weights are computed with exactly the arithmetic of the per-chromosome
    implementation (a tile outside every domain has no entry; a populated
    tile gets the same float32 weight as before)."""
    k = _kernels()
    n_chrom = len(store.chrom_names)
    n_cells = len(store.indptr) - 1
    res_l = []

    def cmt(r0, r1, indptr, chrom, start, lens, lpos, lend, ts):
        r = np.full(n_chrom, -1, dtype=np.int64)
        neg = k["chrom_max_tile"](r0, r1, indptr, chrom, start, lens, lpos, lend, ts, r)
        res_l.append((r, neg))

    _run_blocks(cmt, n_cells, n_jobs, store.indptr, store.chrom, store.start,
                *store.end_args, int(tile_size))
    res = np.vstack([r for r, _ in res_l]) if res_l else np.full((1, n_chrom), -1)
    neg = np.asarray([n for _, n in res_l] or [0])
    max_tile = res.max(axis=0)
    base = np.full(n_chrom, -1, dtype=np.int64)
    chrom_code = {c: i for i, c in enumerate(store.chrom_names)}
    rows_t, rows_g, vals = [], [], []
    off = 0
    half = tile_size // 2
    for chrom in chroms_with_genes:
        code = chrom_code.get(chrom)
        if code is None or max_tile[code] < 0:
            continue
        n_t = int(max_tile[code]) + 1
        base[code] = off
        g_chr = genes_sorted[genes_sorted["chrom"] == chrom].reset_index(drop=True)
        is_plus = g_chr["strand"].to_numpy() == "+"
        g_start = g_chr["start"].to_numpy(np.int64)
        g_end = g_chr["end"].to_numpy(np.int64)
        body_lo = np.where(is_plus, g_start - gene_upstream,
                           g_start - gene_downstream).astype(np.int64)
        body_hi = np.where(is_plus, g_end + gene_downstream,
                           g_end + gene_upstream).astype(np.int64)
        width = (body_hi - body_lo).astype(np.float64)
        width = np.where(width > 0, width, 1.0)
        m = 1.0 / width
        m_lo, m_hi = float(m.min()), float(m.max())
        if m_hi > m_lo:
            gene_weight = 1.0 + m * (gene_scale_factor - 1.0) / (m_hi - m_lo)
        else:
            gene_weight = np.ones_like(m)
        gene_weight = np.clip(gene_weight, 1.0, gene_scale_factor).astype(np.float32)
        u_min, u_max = extend_upstream
        d_min, d_max = extend_downstream
        five_prime_max = np.where(is_plus, u_max, d_max).astype(np.int64)
        five_prime_min = np.where(is_plus, u_min, d_min).astype(np.int64)
        three_prime_max = np.where(is_plus, d_max, u_max).astype(np.int64)
        three_prime_min = np.where(is_plus, d_min, u_min).astype(np.int64)
        if use_gene_boundaries and len(g_chr) >= 2:
            prev_body_end = np.concatenate([[-np.inf], body_hi[:-1].astype(float)])
            next_body_start = np.concatenate([body_lo[1:].astype(float), [np.inf]])
        else:
            prev_body_end = np.full(len(g_chr), -np.inf)
            next_body_start = np.full(len(g_chr), np.inf)
        s_max_reach = (body_lo - five_prime_max).astype(float)
        dom_lo = np.maximum(prev_body_end + tile_size, s_max_reach)
        dom_lo = np.minimum((body_lo - five_prime_min).astype(float), dom_lo)
        e_max_reach = (body_hi + three_prime_max).astype(float)
        dom_hi = np.minimum(next_body_start - tile_size, e_max_reach)
        dom_hi = np.maximum((body_hi + three_prime_min).astype(float), dom_hi)
        dom_lo = np.maximum(dom_lo, 1.0).astype(np.int64)
        dom_hi = np.maximum(dom_hi.astype(np.int64), dom_lo + 1)

        tile_centres = np.arange(n_t, dtype=np.int64) * tile_size + half
        i0 = np.searchsorted(tile_centres, dom_lo, side="left")
        i1 = np.searchsorted(tile_centres, dom_hi, side="left")
        n_per = np.maximum(i1 - i0, 0)
        total = int(n_per.sum())
        if total:
            g_rep = np.repeat(np.arange(len(g_chr)), n_per)
            t_loc = i0[g_rep] + (np.arange(total) - (np.cumsum(n_per) - n_per)[g_rep])
            c = tile_centres[t_loc]
            x = np.maximum(0, np.maximum(body_lo[g_rep] - (c + half),
                                         (c - half) - body_hi[g_rep])).astype(np.float64)
            w = (np.exp(-x / 5000.0) + np.exp(-1.0)).astype(np.float32) * gene_weight[g_rep]
            rows_t.append(t_loc + off)
            rows_g.append(g_chr["gene_idx_global"].to_numpy(np.int64)[g_rep])
            vals.append(w)
        off += n_t
    n_tiles = off
    if rows_t:
        t_all = np.concatenate(rows_t); g_all = np.concatenate(rows_g); v_all = np.concatenate(vals)
        order = np.argsort(t_all, kind="stable")
        Wg = g_all[order]; Wv = v_all[order]
        Wp = np.zeros(n_tiles + 1, dtype=np.int64)
        np.cumsum(np.bincount(t_all, minlength=n_tiles), out=Wp[1:])
    else:
        Wg = np.zeros(0, np.int64); Wv = np.zeros(0, np.float32)
        Wp = np.zeros(n_tiles + 1, dtype=np.int64)
    return base, n_tiles, Wp, Wg, Wv


def gene_score(store: FragmentStore, row_map, n_genes: int, weights, *, tile_size,
               ceiling, n_jobs=None) -> np.ndarray:
    """Raw (cells x genes) float32 gene scores from the store: for every cell
    the ceiling-clipped insertion counts of each 500 bp tile times the
    tile -> gene weights, summed over tiles in increasing order -- the same
    float32 operations, in the same order, as ``W @ matGS`` per chromosome."""
    k = _kernels()
    base, n_tiles, Wp, Wg, Wv = weights
    out = np.zeros((len(row_map), n_genes), dtype=np.float32)
    if n_tiles == 0:
        return out
    _run_blocks(k["gene_score"], len(row_map), n_jobs, store.indptr, store.chrom,
                store.start, *store.end_args, row_map, base, int(tile_size), int(ceiling or 0),
                Wp, Wg, Wv, out, weights=_row_weights(store, row_map))
    return out
