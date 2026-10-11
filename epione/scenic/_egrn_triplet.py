"""TF-region-gene triplet ranking of eRegulon links (SCENIC+ ``triplet_score``).

Each link gets three scores: TF-to-gene importance, region-to-gene importance and the
TF-to-region motif rank (best rank of the region over the TF's motifs in a cisTarget rankings
database). Each score is ranked with random tie-breaking, turned into a rank ratio, and the
three ratios are combined with order statistics (Aerts et al., Nat Biotechnol 2003);
``triplet_rank`` is the rank of the combined value.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numba
import numpy as np
import pandas as pd

from ._egrn_argsort import argsort_np126

__all__ = ["triplet_rank", "tf_region_max_rank", "read_ranking_db"]


@numba.njit(cache=True)
def _order_statistics(r):
    # combined rank ratio of a set of rank ratios (sorted in place)
    n = r.shape[0]
    if n == 0:
        return 1.0
    r.sort()
    w = np.zeros(n + 1, dtype=np.float64)
    w[0] = 1.0
    w[1] = r[n - 1]
    for k in range(2, n + 1):
        f = -1.0
        for j in range(k):
            f = -(f * (k - j) * r[n - k]) / (j + 1.0)
            w[k] = w[k] + (w[k - j - 1] * f)
    return w[n]


@numba.njit(cache=True)
def _aggregate(ratios):
    out = np.zeros(ratios.shape[1], dtype=np.float64)
    for i in range(ratios.shape[1]):
        out[i] = _order_statistics(ratios[:, i].copy())
    return out


def _rank_with_random_ties(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # descending ranks; ties broken by a random permutation (one draw per call)
    perm = rng.permutation(scores.shape[0])
    return perm[argsort_np126((-scores)[perm])].argsort().astype(np.uint32)


def triplet_rank(eregulons: pd.DataFrame, tf_region_rank: pd.DataFrame,
                 rng: np.random.Generator | int = 123) -> np.ndarray:
    """Triplet rank of every eRegulon link.

    Parameters
    ----------
    eregulons
        eRegulon table with ``TF, Region, importance_TF2G, importance_R2G``.
    tf_region_rank
        Regions x TFs motif ranks (lower = better), e.g. from :func:`tf_region_max_rank`.
    rng
        Generator (or seed for ``np.random.default_rng``). SCENIC+ draws the three tie-breaking
        permutations from one module-level ``default_rng(123)``; a fresh generator per call
        matches one eGRN per process.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(seed=rng)
    ri = tf_region_rank.index.get_indexer(eregulons["Region"].to_numpy())
    ci = tf_region_rank.columns.get_indexer(eregulons["TF"].to_numpy())
    if (ri < 0).any() or (ci < 0).any():
        raise KeyError("eRegulon regions/TFs missing from tf_region_rank")
    tf_to_region = tf_region_rank.to_numpy()[ri, ci]
    tf_to_gene = eregulons["importance_TF2G"].to_numpy()
    region_to_gene = eregulons["importance_R2G"].to_numpy()
    r1 = _rank_with_random_ties(-tf_to_region, rng)
    r2 = _rank_with_random_ties(tf_to_gene, rng)
    r3 = _rank_with_random_ties(region_to_gene, rng)
    n = r1.shape[0]
    ratios = np.array([(r2.astype(np.float64) + 1) / n, (r3.astype(np.float64) + 1) / n,
                       (r1.astype(np.float64) + 1) / n])
    return argsort_np126(_aggregate(ratios)).argsort()


# --------------------------------------------------------------------------- ranking DB
def _parse_regions(names: Sequence[str]):
    chrom, start, end = [], [], []
    for x in names:
        c, rest = x.split(":", 1)
        s, e = rest.split("-", 1)
        chrom.append(c)
        start.append(int(s))
        end.append(int(e))
    return np.asarray(chrom, dtype=object), np.asarray(start, dtype=np.int64), np.asarray(end, dtype=np.int64)


def map_regions(target: Sequence[str], query: Sequence[str], fraction_overlap: float = 0.4) -> pd.DataFrame:
    """Pairs (Target, Query) of overlapping regions where the overlap covers more than
    ``fraction_overlap`` of either region (pycistarget ``target_to_query``)."""
    tc, ts, te = _parse_regions(target)
    qc, qs, qe = _parse_regions(query)
    tn, qn = np.asarray(target, dtype=object), np.asarray(query, dtype=object)
    out_t, out_q = [], []
    for c in np.unique(tc):
        ti = np.flatnonzero(tc == c)
        qi = np.flatnonzero(qc == c)
        if len(qi) == 0:
            continue
        o = np.argsort(qs[qi], kind="stable")
        qi = qi[o]
        qstart = qs[qi]
        maxlen = int((qe[qi] - qs[qi]).max())
        for t in ti:
            lo = np.searchsorted(qstart, ts[t] - maxlen, side="left")
            hi = np.searchsorted(qstart, te[t], side="left")
            cand = qi[lo:hi]
            cand = cand[qe[cand] > ts[t]]
            if len(cand) == 0:
                continue
            ov = np.minimum(te[t], qe[cand]) - np.maximum(ts[t], qs[cand])
            fq = ov / (qe[cand] - qs[cand])
            ft = ov / (te[t] - ts[t])
            keep = (fq > fraction_overlap) | (ft > fraction_overlap)
            out_t.extend([tn[t]] * int(keep.sum()))
            out_q.extend(qn[cand[keep]].tolist())
    return pd.DataFrame({"Target": out_t, "Query": out_q})


def _db_layout(fname: str):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    names = ipc.open_file(pa.memory_map(fname, "r")).schema.names
    index_col = next(c for c in ("motifs", "tracks", "regions", "genes") if c in names)
    db_cols = [c for c in names if c != index_col]
    prefix = db_cols[0].split("__")[0] + "__" if "__" in db_cols[0] else ""
    return names, index_col, db_cols, prefix


def read_ranking_db(fname: str, regions: Sequence[str] | None = None):
    """Read a cisTarget ``regions_vs_motifs`` rankings feather file (Arrow IPC).

    Returns (motif names, region names, pyarrow Table). With ``regions`` (names without the
    database prefix), only those columns are read from disk.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    names, index_col, db_cols, prefix = _db_layout(fname)
    if regions is None:
        sel = db_cols
    else:
        want = set(prefix + r for r in regions)
        sel = [c for c in db_cols if c in want]
    pos = {c: i for i, c in enumerate(names)}
    opts = ipc.IpcReadOptions(included_fields=sorted([pos[c] for c in sel] + [pos[index_col]]), use_threads=True)
    tab = ipc.open_file(pa.memory_map(fname, "r"), options=opts).read_all()
    motifs = np.asarray(tab.column(index_col).to_pylist(), dtype=object)
    return motifs, [c[len(prefix):] for c in sel], tab.drop_columns([index_col])


def tf_region_max_rank(cistrome_regions: Sequence[str], tf_motifs: Mapping[str, Sequence[str]],
                       ranking_db: str, fraction_overlap: float = 0.4, chunk: int = 20000) -> pd.DataFrame:
    """Best (lowest) motif rank per cistrome region and TF from a cisTarget rankings DB.

    Cistrome regions are mapped to database regions by overlap (``fraction_overlap``); a
    region mapped to several database regions takes the best rank among them.
    """
    _, _, db_cols, prefix = _db_layout(ranking_db)
    db_regions = [c[len(prefix):] for c in db_cols]
    mapping = map_regions(list(cistrome_regions), db_regions, fraction_overlap)
    motifs, cols, tab = read_ranking_db(ranking_db, pd.unique(mapping["Query"]))
    mpos = pd.Index(motifs)
    tfs = list(tf_motifs.keys())
    rows = [mpos.get_indexer(list(tf_motifs[tf])) for tf in tfs]
    if any((r < 0).any() for r in rows):
        raise KeyError("motifs missing from the ranking database")
    parts = []
    for s in range(0, len(cols), chunk):
        block = np.column_stack([tab.column(prefix + c).to_numpy() for c in cols[s:s + chunk]])
        parts.append(np.stack([block[r].min(0) for r in rows], axis=1))
    best = np.concatenate(parts, axis=0)
    q = pd.Index(cols).get_indexer(mapping["Query"])
    out = pd.DataFrame(best[q], columns=tfs)
    out["region"] = mapping["Target"].to_numpy()
    return out.groupby("region").min().rename_axis("cistrome_region_coord")
