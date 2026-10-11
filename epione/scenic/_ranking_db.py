"""cisTarget ranking / score databases and region → database-region mapping.

Reimplements the pieces of ``ctxcore`` (``FeatherRankingDatabase``,
``CisTargetDatabase.subset_to_pandas``) and ``pycistarget.utils``
(``target_to_query``, region-name helpers) that SCENIC+ uses, without
pyranges, polars or ctxcore.

A cisTarget database is a Feather v2 (Arrow IPC) file with one column per
region (``int16``/``int32`` rankings or ``float32`` scores) plus a
``motifs`` (or ``tracks``) column. The official hg38 screen v10 files hold
1.8 M region columns, so they are never read whole. :class:`CisTargetDB`
decompresses only the requested columns and keeps them in an in-memory cache,
so a list of region sets is read once and then sliced from memory.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

RegionLike = Union[pd.DataFrame, Sequence[str]]

_INDEX_COLUMNS = ("motifs", "tracks", "regions", "genes")


# ---------------------------------------------------------------------------
# Region names <-> coordinates
# ---------------------------------------------------------------------------

def region_names_to_frame(names: Sequence[str]) -> pd.DataFrame:
    """``chr:start-end`` names → ``Chromosome``/``Start``/``End`` frame
    (pycistarget ``region_names_to_coordinates``; names without ``:`` are
    dropped, as upstream does)."""
    s = pd.Series(list(names), dtype=object)
    s = s[s.str.contains(":", regex=False)] if len(s) else s
    if len(s) == 0:
        return pd.DataFrame({"Chromosome": pd.Series(dtype=object),
                             "Start": pd.Series(dtype=np.int64), "End": pd.Series(dtype=np.int64)})
    parts = s.str.split(":", n=1, expand=True)
    se = parts[1].str.split("-", n=1, expand=True)
    df = pd.DataFrame({
        "Chromosome": parts[0].to_numpy(),
        "Start": se[0].astype(np.int64).to_numpy(),
        "End": se[1].astype(np.int64).to_numpy(),
    }, index=s.to_numpy())
    return df


def frame_to_region_names(df: pd.DataFrame) -> List[str]:
    """``Chromosome``/``Start``/``End`` → ``chr:start-end`` names
    (pycistarget ``coord_to_region_names``)."""
    if len(df) == 0:
        return []
    return list(df["Chromosome"].astype(str) + ":" + df["Start"].astype(str)
                + "-" + df["End"].astype(str))


def as_region_frame(regions: RegionLike) -> pd.DataFrame:
    """Accept a BED-like frame (``Chromosome/Start/End`` or the first three
    columns) or a list of ``chr:start-end`` names."""
    if isinstance(regions, pd.DataFrame):
        if {"Chromosome", "Start", "End"}.issubset(regions.columns):
            df = regions[["Chromosome", "Start", "End"]]
        else:
            df = regions.iloc[:, :3]
            df.columns = ["Chromosome", "Start", "End"]
        return pd.DataFrame({
            "Chromosome": df["Chromosome"].astype(str).to_numpy(),
            "Start": df["Start"].astype(np.int64).to_numpy(),
            "End": df["End"].astype(np.int64).to_numpy(),
        })
    return region_names_to_frame(regions).reset_index(drop=True)


def read_bed_regions(path: Union[str, os.PathLike]) -> pd.DataFrame:
    """Read the first three columns of a BED file as a region frame."""
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                     comment="#", dtype={0: str, 1: np.int64, 2: np.int64})
    df.columns = ["Chromosome", "Start", "End"]
    return df


class RegionIndex:
    """Per-chromosome sorted interval index used for overlap joins.

    ``overlaps(target)`` returns every (target row, index row) pair of
    intervals that share at least one base (half-open coordinates), with the
    overlap length. That is what pyranges' ``join(report_overlap=True)``
    reports.
    """

    def __init__(self, regions: RegionLike):
        df = as_region_frame(regions)
        self.frame = df
        self._names = None
        self._by_chrom: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
        chrom = df["Chromosome"].to_numpy()
        start = df["Start"].to_numpy()
        end = df["End"].to_numpy()
        order = np.lexsort((start, chrom))
        chrom_s = chrom[order]
        bounds = np.flatnonzero(chrom_s[1:] != chrom_s[:-1]) + 1
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(order)]):
            if hi <= lo:
                continue
            idx = order[lo:hi]
            s, e = start[idx], end[idx]
            self._by_chrom[str(chrom_s[lo])] = (s, e, idx, int((e - s).max()))

    @property
    def names(self) -> np.ndarray:
        """``chr:start-end`` name of every indexed region (cached)."""
        if self._names is None:
            self._names = np.asarray(frame_to_region_names(self.frame), dtype=object)
        return self._names

    def overlaps(self, target: RegionLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tdf = as_region_frame(target)
        t_chrom = tdf["Chromosome"].to_numpy()
        t_start = tdf["Start"].to_numpy()
        t_end = tdf["End"].to_numpy()
        ti_all, qi_all, ov_all = [], [], []
        for chrom in pd.unique(t_chrom):
            if chrom not in self._by_chrom:
                continue
            qs, qe, qidx, maxlen = self._by_chrom[chrom]
            tsel = np.flatnonzero(t_chrom == chrom)
            ts, te = t_start[tsel], t_end[tsel]
            lo = np.searchsorted(qs, ts - maxlen, side="right")
            hi = np.searchsorted(qs, te, side="left")
            cnt = np.maximum(hi - lo, 0)
            tot = int(cnt.sum())
            if tot == 0:
                continue
            rep = np.repeat(np.arange(len(tsel)), cnt)
            offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            q = lo[rep] + offs
            keep = qe[q] > ts[rep]
            rep, q = rep[keep], q[keep]
            ov = np.minimum(te[rep], qe[q]) - np.maximum(ts[rep], qs[q])
            ti_all.append(tsel[rep])
            qi_all.append(qidx[q])
            ov_all.append(ov)
        if not ti_all:
            e = np.empty(0, dtype=np.int64)
            return e, e, e
        return np.concatenate(ti_all), np.concatenate(qi_all), np.concatenate(ov_all)


def target_to_query(
    target: RegionLike,
    query: Union[RegionLike, RegionIndex],
    fraction_overlap: float = 0.4,
) -> pd.DataFrame:
    """Map ``target`` regions onto ``query`` regions (pycistarget
    ``target_to_query``).

    A (target, query) pair is kept when the overlap covers more than
    ``fraction_overlap`` of either interval. Returns a ``Target``/``Query``
    frame of region names, one row per kept pair. Raises ``ValueError`` when
    nothing overlaps, as upstream does.
    """
    tdf = as_region_frame(target)
    qidx = query if isinstance(query, RegionIndex) else RegionIndex(query)
    ti, qi, ov = qidx.overlaps(tdf)
    if len(ti) == 0:
        raise ValueError(
            "None of the regions overlap with the database. Check the chromosome "
            "names of the input regions and that the right database is used.")
    qdf = qidx.frame
    t_len = (tdf["End"].to_numpy() - tdf["Start"].to_numpy())[ti]
    q_len = (qdf["End"].to_numpy() - qdf["Start"].to_numpy())[qi]
    keep = (ov / q_len > fraction_overlap) | (ov / t_len > fraction_overlap)
    ti, qi = ti[keep], qi[keep]
    t_names = np.asarray(frame_to_region_names(tdf), dtype=object)
    return pd.DataFrame({"Target": t_names[ti], "Query": qidx.names[qi]})


# ---------------------------------------------------------------------------
# Ranking / score database
# ---------------------------------------------------------------------------

class CisTargetDB:
    """A cisTarget rankings or scores database (regions × motifs).

    Parameters
    ----------
    source
        Path to a cisTarget Feather v2 database
        (``*.regions_vs_motifs.rankings.feather`` / ``.scores.feather``).
        Use :meth:`from_matrix` for an in-memory database, such as the one
        :func:`epione.scenic.motif_score_db` builds from MOODS hits.
    name
        Label used in logs.

    Notes
    -----
    Values are returned as ``(n_regions, n_motifs)`` arrays: region rows in
    sorted name order (ctxcore's column order) and motif columns in file
    order (ctxcore's row order). Region names drop an optional ``prefix__``,
    as pycistarget does.
    """

    def __init__(self, source: Union[str, os.PathLike], name: Optional[str] = None):
        import pyarrow as pa
        import pyarrow.ipc as ipc

        self.path = os.fspath(source)
        self.name = name or os.path.basename(self.path)
        with open(self.path, "rb") as fh:
            head = fh.read(6)
            fh.seek(-6, 2)
            tail = fh.read(6)
        if not head == tail == b"ARROW1":
            raise ValueError(f"{self.path} is not a Feather v2 (Arrow IPC) cisTarget database")
        reader = ipc.open_file(pa.memory_map(self.path))
        schema = reader.schema
        names = schema.names
        idx_col = None
        for i in range(len(names) - 1, -1, -1):
            if names[i] in _INDEX_COLUMNS:
                idx_col = i
                break
        if idx_col is None:
            raise ValueError(f"{self.path} has no motifs/tracks column")
        self._index_column = names[idx_col]
        dtypes = {str(t) for i, t in enumerate(schema.types) if i != idx_col}
        if len(dtypes) != 1:
            raise ValueError(f"mixed value dtypes in {self.path}: {sorted(dtypes)}")
        dt = dtypes.pop()
        self.dtype = {"int16": np.int16, "int32": np.int32, "float": np.float32}[dt]
        self.kind = "scores" if self.dtype == np.float32 else "rankings"
        cols = names[:idx_col] + names[idx_col + 1:]
        self._field_of = np.r_[np.arange(idx_col), np.arange(idx_col + 1, len(names))]
        prefix = cols[0].split("__")[0] if "__" in cols[0] else None
        self.prefix = prefix
        self.regions = np.asarray(
            [c.split("__")[1] for c in cols] if prefix is not None else cols, dtype=object)
        self.total_regions = len(self.regions)
        rd = ipc.open_file(pa.memory_map(self.path),
                           options=ipc.IpcReadOptions(included_fields=[idx_col],
                                                      use_threads=False))
        self.motifs = np.asarray(rd.get_batch(0).column(0).to_pylist(), dtype=object)
        self._pos = None
        self._index = None
        self._matrix = None            # in-memory database (from_matrix)
        self._cache_regions = np.empty(0, dtype=object)
        self._cache = np.empty((0, len(self.motifs)), dtype=self.dtype)

    # -- construction from memory --------------------------------------
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, regions: Sequence[str], motifs: Sequence[str],
                    kind: str = "rankings", name: str = "memory") -> "CisTargetDB":
        """Wrap a dense ``(n_regions, n_motifs)`` array as a database."""
        self = cls.__new__(cls)
        self.path = None
        self.name = name
        self.prefix = None
        self.kind = kind
        matrix = np.asarray(matrix)
        if kind == "rankings":
            matrix = matrix.astype(np.int16 if matrix.max(initial=0) < 2 ** 15 else np.int32,
                                   copy=False)
        else:
            matrix = matrix.astype(np.float32, copy=False)
        self.dtype = matrix.dtype.type
        self.regions = np.asarray(list(regions), dtype=object)
        self.motifs = np.asarray(list(motifs), dtype=object)
        if matrix.shape != (len(self.regions), len(self.motifs)):
            raise ValueError("matrix must be (n_regions, n_motifs)")
        self.total_regions = len(self.regions)
        self._matrix = matrix
        self._pos = None
        self._index = None
        self._cache_regions = np.empty(0, dtype=object)
        self._cache = np.empty((0, len(self.motifs)), dtype=self.dtype)
        return self

    def __repr__(self):
        return (f"CisTargetDB({self.name!r}, kind={self.kind}, "
                f"{self.total_regions:,} regions × {len(self.motifs):,} motifs)")

    # -- region lookup ---------------------------------------------------
    @property
    def region_index(self) -> RegionIndex:
        if self._index is None:
            self._index = RegionIndex(list(self.regions))
            if len(self._index.frame) == len(self.regions):
                self._index._names = self.regions
        return self._index

    def _positions(self, regions: np.ndarray) -> np.ndarray:
        if self._pos is None:
            self._pos = pd.Index(self.regions)
        pos = self._pos.get_indexer(regions)
        if (pos < 0).any():
            missing = regions[pos < 0][:5]
            raise KeyError(f"regions not in {self.name}: {list(missing)} ...")
        return pos

    def _read(self, regions: np.ndarray, chunk: int = 65536) -> np.ndarray:
        """Read the value columns of ``regions`` → ``(n, n_motifs)``."""
        pos = self._positions(regions)
        if self._matrix is not None:
            return self._matrix[pos]
        import pyarrow as pa
        import pyarrow.ipc as ipc
        out = np.empty((len(regions), len(self.motifs)), dtype=self.dtype)
        order = np.argsort(pos, kind="stable")
        fields = self._field_of[pos[order]]
        for a in range(0, len(order), chunk):
            sel = fields[a:a + chunk]
            rd = ipc.open_file(pa.memory_map(self.path),
                               options=ipc.IpcReadOptions(included_fields=sel.tolist(),
                                                          use_threads=True))
            batch = rd.get_batch(0)
            # included_fields returns columns in schema order == sel order
            for j, col in enumerate(batch.columns):
                out[order[a + j]] = col.to_numpy(zero_copy_only=False)
            del batch, rd
        return out

    def load(self, regions: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Values of ``regions`` (unknown names are ignored, as ctxcore does).

        Returns ``(sorted_region_names, values)`` with ``values`` of shape
        ``(n_regions, n_motifs)``. Columns that were read before are served
        from the cache.
        """
        req = np.asarray(sorted(set(regions) & self._region_set()), dtype=object)
        if len(req) == 0:
            return req, np.empty((0, len(self.motifs)), dtype=self.dtype)
        if len(self._cache_regions):
            pos = np.searchsorted(self._cache_regions, req)
            pos_c = np.minimum(pos, len(self._cache_regions) - 1)
            have = self._cache_regions[pos_c] == req
        else:
            have = np.zeros(len(req), dtype=bool)
        if not have.all():
            self.preload(req[~have])
            pos = np.searchsorted(self._cache_regions, req)
        return req, self._cache[pos]

    def preload(self, regions: Iterable[str]) -> "CisTargetDB":
        """Read ``regions`` into the cache (the "load once" path)."""
        want = set(regions) & self._region_set()
        if len(self._cache_regions):
            # (np.isin on object arrays falls back to an O(n*m) loop)
            want -= set(self._cache_regions.tolist())
        new = np.asarray(sorted(want), dtype=object)
        if len(new) == 0:
            return self
        vals = self._read(new)
        if len(self._cache_regions) == 0:
            self._cache_regions, self._cache = new, vals
        else:
            allr = np.concatenate([self._cache_regions, new])
            order = np.argsort(allr, kind="stable")
            self._cache_regions = allr[order]
            self._cache = np.concatenate([self._cache, vals])[order]
        return self

    def save_cache(self, path: Union[str, os.PathLike]) -> str:
        """Save the cached region values (``.npz``). Reloading them with
        :meth:`load_cache` skips the database read, which is useful when a
        notebook is re-run."""
        path = os.fspath(path)
        np.savez(path, regions=self._cache_regions.astype(str), values=self._cache,
                 motifs=self.motifs.astype(str), total_regions=self.total_regions)
        return path

    def load_cache(self, path: Union[str, os.PathLike]) -> "CisTargetDB":
        """Merge values saved with :meth:`save_cache` into the cache."""
        z = np.load(os.fspath(path), allow_pickle=False)
        if list(z["motifs"]) != list(self.motifs) or int(z["total_regions"]) != self.total_regions:
            raise ValueError(f"{path} was saved from a different database")
        regions = z["regions"].astype(object)
        if len(self._cache_regions):
            have = set(self._cache_regions.tolist())
            keep = np.fromiter((r not in have for r in regions), dtype=bool, count=len(regions))
            regions, vals = regions[keep], z["values"][keep]
            allr = np.concatenate([self._cache_regions, regions])
            order = np.argsort(allr, kind="stable")
            self._cache_regions = allr[order]
            self._cache = np.concatenate([self._cache, vals])[order]
        else:
            self._cache_regions, self._cache = regions, z["values"]
        return self

    def clear_cache(self):
        self._cache_regions = np.empty(0, dtype=object)
        self._cache = np.empty((0, len(self.motifs)), dtype=self.dtype)

    def _region_set(self) -> set:
        if not hasattr(self, "_rset") or self._rset is None:
            self._rset = set(self.regions)
        return self._rset

    # -- export ----------------------------------------------------------
    def write_feather(self, path: Union[str, os.PathLike], compression: str = "zstd") -> str:
        """Write the database as a cisTarget Feather v2 file
        (``<name>.regions_vs_motifs.{rankings,scores}.feather``)."""
        import pyarrow as pa
        import pyarrow.feather as pf
        if self._matrix is None:
            raise ValueError("only in-memory databases can be written")
        path = os.fspath(path)
        # cisTarget layout: one column per region (values over motifs) + "motifs".
        names = list(self.regions) + ["motifs"]
        arrays = [pa.array(self._matrix[i]) for i in range(len(self.regions))]
        arrays.append(pa.array(list(self.motifs), type=pa.string()))
        table = pa.Table.from_arrays(arrays, names=names)
        pf.write_feather(table, path, compression=compression)
        return path
