"""A/B compartments for single-cell Hi-C: scHiCluster and the HCEA recipe.

Two complementary scores, both computed from **raw** (not imputed) contacts.

* **Single-cell score** (:func:`compartment_scores`) — a port of
  ``schicluster.compartment`` (Zhou lab, MIT). For every cell and bin, the
  score is the contact-weighted mean CpG density of the bin's
  intra-chromosomal partners: an A-compartment bin mostly contacts CpG-rich
  (A) bins, a B bin mostly CpG-poor ones. It needs no eigendecomposition, so
  it is defined for a single sparse cell.
* **Pseudobulk score** (:func:`compartment_model`, :func:`project_compartments`)
  — the recipe of the Human Cell Epigenome Atlas (HCEA; Zhou, Wu, Liu *et al.*,
  ``fig4/04.compartment.ipynb``). A PCA model is fitted once on the contacts of
  all cells merged (coverage filter, distance-normalised observed/expected,
  correlation, PCA; the PC with the stronger saddle contrast is kept and its
  sign is fixed by CpG density). Each cell type's pseudobulk is then
  *projected* on the same loadings, so scores are on one scale and directly
  comparable across cell types. :func:`compartment_saddle` gives the saddle
  matrix and the AA / BB / AB / BA strength.

Helpers: :func:`cpg_ratio` (per-bin CpG density, as ``bedtools nuc``),
:func:`contacts_to_scool` (per-cell contact tables → one ``.scool``),
:func:`load_contact_tables` (index contact tables without converting them) and
:func:`merge_cells` (raw pseudobulk ``.cool`` per group).

Parity with scHiCluster and the HCEA notebook is tested in
``tests/parity/schicluster_compartment``.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

# Column layout of the 4DN-style contact tables written by the snm3C-seq
# pipeline (YAP / scHiCluster): chrom1 = 1, pos1 = 2, chrom2 = 5, pos2 = 6.
CONTACT_COLUMNS = {"chrom1": 1, "pos1": 2, "chrom2": 5, "pos2": 6}


# ---------------------------------------------------------------------------
# reference tracks
# ---------------------------------------------------------------------------

def _read_chrom_sizes(chrom_sizes) -> pd.Series:
    if isinstance(chrom_sizes, pd.Series):
        return chrom_sizes.astype(np.int64)
    if isinstance(chrom_sizes, Mapping):
        return pd.Series(dict(chrom_sizes), dtype=np.int64)
    s = pd.read_csv(chrom_sizes, sep="\t", index_col=0, header=None).squeeze(axis=1)
    return s.astype(np.int64)


def _resolve_uri(path: PathLike, resolution: Optional[int]) -> str:
    p = str(path)
    if "::" in p or resolution is None:
        return p
    return f"{p}::resolutions/{int(resolution)}"


def cpg_ratio(
    fasta_path: PathLike,
    *,
    chrom_sizes=None,
    bins: Optional[pd.DataFrame] = None,
    resolution: int = 100_000,
    chromosomes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-bin CpG density, the phasing track of scHiCluster compartments.

    Equivalent to ``schicluster.compartment.get_cpg_profile``, which runs
    ``bedtools nuc -pattern CG -C`` on the bins: the ratio is the number of
    ``CG`` dinucleotides (case-insensitive) over the number of non-``N`` bases
    in the bin, and ``0`` for a bin made only of ``N``.

    Arguments:
        fasta_path: indexed genome FASTA (``.fai`` next to it).
        chrom_sizes: chromosome sizes (path to a two-column file, a Series or
            a dict). Bins are made with ``cooler.binnify``. Ignored when
            ``bins`` is given.
        bins: explicit bin table (``chrom``, ``start``, ``end``), e.g.
            ``cooler.Cooler(path).bins()[:]``, so the track lines up with a
            cool bin for bin.
        resolution: bin size in bp when ``chrom_sizes`` is used.
        chromosomes: keep only these chromosomes (in this order).

    Returns:
        ``DataFrame`` with ``chrom``, ``start``, ``end``, ``cpg_ratio``.
    """
    import cooler
    import pyfaidx

    if bins is None:
        if chrom_sizes is None:
            raise ValueError("give either chrom_sizes or bins")
        sizes = _read_chrom_sizes(chrom_sizes)
        if chromosomes is not None:
            sizes = sizes.loc[list(chromosomes)]
        bins = cooler.binnify(sizes, int(resolution))
    bins = bins[["chrom", "start", "end"]].copy()
    bins["chrom"] = bins["chrom"].astype(str)
    if chromosomes is not None:
        bins = bins[bins["chrom"].isin(list(chromosomes))]
    bins = bins.reset_index(drop=True)

    fa = pyfaidx.Fasta(str(fasta_path), sequence_always_upper=True, as_raw=True)
    count = np.zeros(len(bins), dtype=np.int64)
    n_n = np.zeros(len(bins), dtype=np.int64)
    for chrom, idx in bins.groupby("chrom", sort=False).groups.items():
        seq = fa[chrom][:]
        for k in idx:
            s = seq[int(bins.at[k, "start"]):int(bins.at[k, "end"])]
            count[k] = s.count("CG")
            n_n[k] = s.count("N")
    seq_len = (bins["end"] - bins["start"]).to_numpy(np.int64)
    out = bins.copy()
    out["cpg_ratio"] = pd.Series(count / (seq_len - n_n)).fillna(0).to_numpy()
    return out


# ---------------------------------------------------------------------------
# per-cell contact tables
# ---------------------------------------------------------------------------

def _read_contacts(path: PathLike, columns: Mapping[str, int]) -> pd.DataFrame:
    c1, p1, c2, p2 = (columns[k] for k in ("chrom1", "pos1", "chrom2", "pos2"))
    try:
        df = pd.read_csv(path, sep="\t", header=None, index_col=None, comment="#",
                         usecols=[c1, p1, c2, p2],
                         dtype={c1: str, p1: np.int64, c2: str, p2: np.int64})
    except pd.errors.EmptyDataError:
        df = pd.DataFrame({c1: pd.Series([], dtype=str), p1: pd.Series([], dtype=np.int64),
                           c2: pd.Series([], dtype=str), p2: pd.Series([], dtype=np.int64)})
    return df


def _cell_pixels(path, columns, chrom_offset, resolution, min_pos_dist, remove_duplicates):
    """Upper-triangle pixel table of one cell (scHiCluster ``generate_scool``)."""
    c1, p1, c2, p2 = (columns[k] for k in ("chrom1", "pos1", "chrom2", "pos2"))
    df = _read_contacts(path, columns)
    if remove_duplicates:
        df = df.drop_duplicates(subset=[c1, p1, c2, p2])
    df = df[df[c1].isin(chrom_offset.keys()) & df[c2].isin(chrom_offset.keys())
            & (df[p1] > 0) & (df[p2] > 0)]
    cis = df[c1] == df[c2]
    df = df[((df[p1] - df[p2]).abs() > min_pos_dist) | ~cis]
    b1 = (df[c1].map(chrom_offset) + (df[p1] - 1) // resolution).to_numpy(np.int64)
    b2 = (df[c2].map(chrom_offset) + (df[p2] - 1) // resolution).to_numpy(np.int64)
    lo, hi = np.minimum(b1, b2), np.maximum(b1, b2)
    if lo.size == 0:
        return pd.DataFrame({"bin1_id": np.zeros(0, np.int64), "bin2_id": np.zeros(0, np.int64),
                             "count": np.zeros(0, np.int64)})
    pix = pd.DataFrame({"bin1_id": lo, "bin2_id": hi}).value_counts().reset_index()
    pix.columns = ["bin1_id", "bin2_id", "count"]
    return pix.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)


def _pixel_worker(args):
    return _cell_pixels(*args)


class _LazyPixels(Mapping):
    """``{cell: pixels}`` whose values are computed in a process pool on access.

    ``cooler.create_scool`` reads the cells in sorted order; the pool works at
    most ``4 * nproc`` cells ahead in that order, so memory stays bounded.
    """

    def __init__(self, tasks: Dict[str, tuple], nproc: int):
        self._tasks = tasks
        self._order = sorted(tasks)
        self._nproc = max(1, int(nproc))
        self._pool = None
        self._pending = {}
        self._submitted = 0

    def __iter__(self):
        return iter(self._tasks)

    def __len__(self):
        return len(self._tasks)

    def _fill(self):
        while (self._submitted < len(self._order)
               and len(self._pending) < 4 * self._nproc):
            k = self._order[self._submitted]
            self._pending[k] = self._pool.submit(_pixel_worker, self._tasks[k])
            self._submitted += 1

    def __getitem__(self, key):
        if self._nproc == 1:
            return _cell_pixels(*self._tasks[key])
        if self._pool is None:
            self._pool = ProcessPoolExecutor(self._nproc)
        self._fill()
        fut = self._pending.pop(key, None)
        out = fut.result() if fut is not None else _cell_pixels(*self._tasks[key])
        self._fill()
        return out

    def close(self):
        if self._pool is not None:
            for f in self._pending.values():
                f.cancel()
            self._pool.shutdown()


def contacts_to_scool(
    contact_paths: Sequence[PathLike],
    scool_path: PathLike,
    *,
    chrom_sizes,
    resolution: int = 100_000,
    cell_ids: Optional[Sequence[str]] = None,
    columns: Mapping[str, int] = CONTACT_COLUMNS,
    min_pos_dist: int = 2500,
    remove_duplicates: bool = True,
    nproc: int = 1,
) -> str:
    """Write per-cell contact tables into one raw ``.scool`` bundle.

    Same pixels as scHiCluster's ``hicluster generate-scool`` without
    blacklists: duplicated contacts are dropped, cis contacts closer than
    ``min_pos_dist`` are dropped (trans contacts are kept), positions are
    binned with ``(pos - 1) // resolution`` and each pair is stored once in
    the upper triangle.

    Arguments:
        contact_paths: one contact table per cell (tab separated, optionally
            gzipped, ``#`` comments allowed).
        scool_path: output ``.scool``.
        chrom_sizes: chromosome sizes; contacts on other chromosomes are
            dropped.
        resolution: bin size in bp.
        cell_ids: cell names, default the file name up to the first ``.``.
        columns: 0-based column numbers of ``chrom1``, ``pos1``, ``chrom2``,
            ``pos2``. The default fits snm3C-seq ``.3C.contact.tsv.gz`` tables.
        min_pos_dist: minimum distance in bp of a kept cis contact.
        remove_duplicates: drop contacts with identical coordinates first.
        nproc: worker processes for parsing.

    Returns:
        ``scool_path`` as a string. Load it with
        :func:`epione.single.hic.load_scool_cells`.
    """
    import cooler

    paths = [str(p) for p in contact_paths]
    if cell_ids is None:
        cell_ids = [Path(p).name.split(".")[0] for p in paths]
    cell_ids = [str(c) for c in cell_ids]
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell_ids must be unique")
    sizes = _read_chrom_sizes(chrom_sizes)
    bins = cooler.binnify(sizes, int(resolution))
    offsets = np.r_[0, np.cumsum(bins.groupby("chrom", sort=False, observed=True).size().to_numpy())][:-1]
    chrom_offset = dict(zip(sizes.index.astype(str), offsets.astype(np.int64)))
    tasks = {cid: (p, dict(columns), chrom_offset, int(resolution), int(min_pos_dist),
                   bool(remove_duplicates)) for cid, p in zip(cell_ids, paths)}
    pixels = _LazyPixels(tasks, nproc)
    try:
        cooler.create_scool(str(scool_path), bins, pixels, ordered=True,
                            dtypes={"count": np.int32})
    finally:
        pixels.close()
    return str(scool_path)


def load_contact_tables(
    contact_paths: Sequence[PathLike],
    *,
    chrom_sizes,
    resolution: int = 100_000,
    cell_ids: Optional[Sequence[str]] = None,
    obs: Optional[pd.DataFrame] = None,
    columns: Mapping[str, int] = CONTACT_COLUMNS,
    chromosomes: Optional[Sequence[str]] = None,
):
    """Index per-cell contact tables into a cell-level AnnData, without binning.

    The counterpart of :func:`load_cool_collection` for contact tables. Only
    :func:`compartment_scores` and :func:`merge_cells` read this layout
    (scHiCluster's ``mode='tsv'``); convert with :func:`contacts_to_scool`
    for everything else.

    Returns:
        ``AnnData`` of shape ``(n_cells, 0)`` with ``obs['contact_path']``
        and ``uns['hic']`` holding ``chromosomes``, ``resolution``,
        ``chrom_sizes`` and the column layout.
    """
    import anndata as ad

    paths = [str(p) for p in contact_paths]
    if cell_ids is None:
        cell_ids = [Path(p).name.split(".")[0] for p in paths]
    cell_ids = [str(c) for c in cell_ids]
    if len(cell_ids) != len(paths) or len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell_ids must be unique and match contact_paths 1:1")
    sizes = _read_chrom_sizes(chrom_sizes)
    if chromosomes is None:
        chromosomes = list(sizes.index.astype(str))
    obs_df = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))
    obs_df["contact_path"] = paths
    if obs is not None:
        if len(obs) != len(cell_ids):
            raise ValueError(f"obs has {len(obs)} rows but {len(cell_ids)} cells")
        for col in obs.columns:
            obs_df[col] = obs[col].to_numpy()
    adata = ad.AnnData(X=np.zeros((len(cell_ids), 0), dtype=np.float32), obs=obs_df)
    adata.uns["hic"] = {
        "chromosomes": list(chromosomes),
        "resolution": int(resolution),
        "chrom_sizes": {str(k): int(v) for k, v in sizes.items()},
        "contact_columns": dict(columns),
        "n_chrom_bins": {c: int(sizes[c] // resolution + 1) for c in chromosomes},
    }
    return adata


# ---------------------------------------------------------------------------
# scHiCluster single-cell compartment score
# ---------------------------------------------------------------------------

def _tsv_chrom_matrices(path, chroms, sizes, resolution, columns):
    """Per-chromosome raw matrices of a contact table (scHiCluster ``mode='tsv'``)."""
    from scipy.sparse import csr_matrix

    c1, p1, c2, p2 = (columns[k] for k in ("chrom1", "pos1", "chrom2", "pos2"))
    data = _read_contacts(path, columns)
    data = data.loc[(data[c1] == data[c2]) & data[c1].isin(chroms)]
    out = {}
    for chrom in chroms:
        n_bins = int(sizes[chrom] // resolution) + 1
        sub = data.loc[data[c1] == chrom]
        if sub.shape[0] == 0:
            out[chrom] = csr_matrix((n_bins, n_bins))
            continue
        sub = sub.copy()
        sub[[p1, p2]] = (sub[[p1, p2]] - 1) // resolution
        sub = sub.groupby(by=[p1, p2])[c1].count().reset_index()
        m = csr_matrix((sub[c1].astype(np.int32), (sub[p1], sub[p2])), (n_bins, n_bins))
        out[chrom] = m + m.T
    return out


def _chrom_score(matrix, cpg: np.ndarray, calc_strength: bool):
    """scHiCluster ``single_chrom_compartment`` for one chromosome."""
    from scipy.sparse import diags

    matrix = matrix - diags(matrix.diagonal(), dtype=np.float64)
    matrix = matrix + diags((matrix.sum(axis=0).A.ravel() == 0).astype(int), dtype=np.float64)
    matrix.data = matrix.data / np.repeat(matrix.sum(axis=0).A1, matrix.getnnz(axis=1))
    comp = matrix.dot(cpg[:, None])[:, 0]
    scores = _strength(matrix, comp, cpg) if calc_strength else None
    return comp, scores


def _strength(matrix, comp: np.ndarray, cpg: np.ndarray) -> np.ndarray:
    """scHiCluster ``compartment_strength``: summed O/E within / between A and B."""
    bin_filter = cpg > 0
    tmp = comp[bin_filter]
    a_pos = tmp > np.percentile(tmp, 80)
    b_pos = tmp < np.percentile(tmp, 20)
    E = matrix.tocoo()
    decay = np.array([E.diagonal(i).mean() for i in range(E.shape[0])])
    E.data = E.data / decay[np.abs(E.col - E.row)]
    E = E.tocsr()[np.ix_(bin_filter, bin_filter)]
    aa = E[np.ix_(a_pos, a_pos)].sum()
    bb = E[np.ix_(b_pos, b_pos)].sum()
    ab = E[np.ix_(a_pos, b_pos)].sum()
    return np.array([aa, bb, ab])


def _score_cell(args):
    (source, kind, chroms, cpg_by_chrom, calc_strength, sizes, resolution, columns) = args
    if kind == "cool":
        import cooler
        clr = cooler.Cooler(source)
        mats = {c: clr.matrix(balance=False, sparse=True).fetch(c) for c in chroms}
    else:
        mats = _tsv_chrom_matrices(source, chroms, sizes, resolution, columns)
    comps, per_chrom = [], []
    for c in chroms:
        m = mats[c]
        if cpg_by_chrom[c].shape[0] != m.shape[0]:
            raise ValueError(
                f"CpG track and contact matrix differ in size on {c} "
                f"({cpg_by_chrom[c].shape[0]} vs {m.shape[0]} bins). Contact tables are "
                "binned as scHiCluster does, with size // resolution + 1 bins, which is one "
                "more than the cool bins when the size is a multiple of the resolution; "
                "convert with contacts_to_scool instead")
        comp, sc = _chrom_score(m.tocsr(), cpg_by_chrom[c], calc_strength)
        comps.append(comp)
        per_chrom.append(sc)
    return np.concatenate(comps), (np.vstack(per_chrom) if calc_strength else None)


def compartment_scores(
    adata,
    cpg: pd.DataFrame,
    *,
    chromosomes: Optional[Sequence[str]] = None,
    calc_strength: bool = False,
    nproc: int = 1,
    dtype=np.float32,
    progress: bool = True,
):
    """scHiCluster single-cell compartment score from raw contacts.

    For every cell and chromosome: drop the diagonal, normalise each row of
    the raw contact matrix to sum to one (an empty row gets a self-loop of
    weight one), and multiply by the CpG-density vector. Bin *i*'s score is
    the CpG density of its contact partners, weighted by contact counts;
    high values mean A, low values B. This is
    ``schicluster.compartment.multiple_cell_compartment``.

    Arguments:
        adata: cells from :func:`load_cool_collection` / :func:`load_scool_cells`
            (raw per-cell cools, scHiCluster ``mode='cool'``) or
            :func:`load_contact_tables` (``mode='tsv'``).
        cpg: per-bin CpG density from :func:`cpg_ratio` with the same bins as
            the cells.
        chromosomes: chromosomes to score. Default: for cools, the cools'
            chromosomes that are in ``cpg`` (cool order); for contact tables,
            ``adata.uns['hic']['chromosomes']``.
        calc_strength: also return the per-cell AA / BB / AB sums of distance-
            normalised contacts between the top-20 % (A) and bottom-20 % (B)
            bins by score.
        nproc: worker processes (one cell per task).
        dtype: dtype of the returned score matrix.
        progress: show a progress bar.

    Returns:
        ``AnnData`` with one row per cell and one column per bin: ``X`` holds
        the scores, ``var`` the bins (``chrom``, ``start``, ``end``,
        ``cpg_ratio``), ``obs`` is copied from ``adata``. With
        ``calc_strength``, ``obsm['compartment_strength']`` is a DataFrame
        with ``AA``, ``BB``, ``AB`` summed over chromosomes and
        ``uns['compartment_strength_per_chrom']`` keeps the per-chromosome
        values (cells × chromosomes × 3).

    Note:
        scHiCluster's genome-wide strength is twice the *last* chromosome's
        value (its accumulator is overwritten inside the chromosome loop).
        Epione sums over chromosomes; the per-chromosome values are the same.
    """
    import anndata as ad

    cpg = cpg.copy()
    cpg["chrom"] = cpg["chrom"].astype(str)
    hic = adata.uns.get("hic", {})
    resolution = int(hic.get("resolution", 0)) or None
    if "contact_path" in adata.obs.columns:
        kind = "tsv"
        sources = adata.obs["contact_path"].astype(str).tolist()
        sizes = pd.Series(hic["chrom_sizes"], dtype=np.int64)
        columns = hic.get("contact_columns", CONTACT_COLUMNS)
        if chromosomes is None:
            chromosomes = list(hic["chromosomes"])
        chromosomes = [str(c) for c in chromosomes]
        bins = pd.concat([
            pd.DataFrame({"chrom": c,
                          "start": np.arange(0, sizes[c] // resolution + 1) * resolution})
            for c in chromosomes], ignore_index=True)
        bins["end"] = np.minimum(bins["start"] + resolution, bins["chrom"].map(sizes))
    elif "cool_path" in adata.obs.columns:
        import cooler
        kind = "cool"
        sources = adata.obs["cool_path"].astype(str).tolist()
        sizes, columns = None, None
        clr = cooler.Cooler(sources[0])
        resolution = int(clr.binsize)
        if chromosomes is None:
            chromosomes = [c for c in clr.chromnames if c in set(cpg["chrom"])]
        chromosomes = [str(c) for c in chromosomes]
        bins = clr.bins()[:][["chrom", "start", "end"]]
        bins["chrom"] = bins["chrom"].astype(str)
        bins = pd.concat([bins[bins["chrom"] == c] for c in chromosomes], ignore_index=True)
    else:
        raise ValueError("adata.obs needs 'cool_path' (cools / scool) or 'contact_path' "
                         "(contact tables)")

    cpg_by_chrom = {c: cpg.loc[cpg["chrom"] == c, "cpg_ratio"].to_numpy(np.float64)
                    for c in chromosomes}
    tasks = [(s, kind, chromosomes, cpg_by_chrom, bool(calc_strength), sizes, resolution,
              columns) for s in sources]

    rows, strengths = _map(_score_cell, tasks, nproc, progress, desc="compartment")
    X = np.vstack(rows).astype(dtype, copy=False)

    var = bins.copy()
    cpg_idx = cpg.set_index(["chrom", "start"])["cpg_ratio"]
    var["cpg_ratio"] = cpg_idx.reindex(list(zip(var["chrom"], var["start"]))).to_numpy()
    var.index = (var["chrom"] + ":" + var["start"].astype(str) + "-"
                 + var["end"].astype(str)).to_numpy()
    out = ad.AnnData(X=X, obs=adata.obs.copy(), var=var)
    out.uns["hic"] = dict(hic, resolution=resolution, chromosomes=list(chromosomes))
    out.uns["compartment"] = {"method": "schicluster", "input": kind}
    if calc_strength:
        per = np.stack(strengths)  # cells x chroms x 3
        out.obsm["compartment_strength"] = pd.DataFrame(
            per.sum(axis=1), index=out.obs_names, columns=["AA", "BB", "AB"])
        out.uns["compartment_strength_per_chrom"] = {
            "values": per, "chromosomes": list(chromosomes), "columns": ["AA", "BB", "AB"]}
    return out


def _map(fn, tasks, nproc, progress, desc):
    """Ordered map with an optional process pool and progress bar."""
    from tqdm.auto import tqdm

    results = []
    if nproc <= 1:
        it = map(fn, tasks)
        for r in tqdm(it, total=len(tasks), desc=desc, disable=not progress):
            results.append(r)
    else:
        with ProcessPoolExecutor(int(nproc)) as ex:
            for r in tqdm(ex.map(fn, tasks, chunksize=max(1, len(tasks) // (int(nproc) * 8))),
                          total=len(tasks), desc=desc, disable=not progress):
                results.append(r)
    if results and isinstance(results[0], tuple):
        return tuple(list(x) for x in zip(*results))
    return results


# ---------------------------------------------------------------------------
# raw pseudobulk
# ---------------------------------------------------------------------------

def _cell_csr(source, kind, n_bins, chrom_offset, resolution, columns, min_pos_dist,
              remove_duplicates):
    from scipy.sparse import csr_matrix

    if kind == "cool":
        import cooler
        pix = cooler.Cooler(source).pixels()[:]
    else:
        pix = _cell_pixels(source, columns, chrom_offset, resolution, min_pos_dist,
                           remove_duplicates)
    return csr_matrix((pix["count"].to_numpy(np.int64),
                       (pix["bin1_id"].to_numpy(), pix["bin2_id"].to_numpy())),
                      shape=(n_bins, n_bins))


def _merge_group(args):
    import cooler
    from scipy.sparse import triu

    (sources, kind, bins, chrom_offset, resolution, columns, min_pos_dist,
     remove_duplicates, double_diagonal, out_path) = args
    n = len(bins)
    total = None
    for s in sources:
        m = _cell_csr(s, kind, n, chrom_offset, resolution, columns, min_pos_dist,
                      remove_duplicates)
        total = m if total is None else total + m
    if double_diagonal:
        from scipy.sparse import diags
        total = total + diags(total.diagonal(), dtype=total.dtype)
    total = triu(total).tocoo()
    order = np.lexsort((total.col, total.row))
    pixels = pd.DataFrame({"bin1_id": total.row[order].astype(np.int64),
                           "bin2_id": total.col[order].astype(np.int64),
                           "count": total.data[order].astype(np.int64)})
    pixels = pixels[pixels["count"] != 0]
    cooler.create_cooler(str(out_path), bins=bins, pixels=pixels, ordered=True,
                         dtypes={"count": np.int64},
                         metadata={"n_cells": len(sources), "generated-by": "epione.merge_cells"})
    return str(out_path)


def merge_cells(
    adata,
    groupby: str,
    out_dir: PathLike,
    *,
    groups: Optional[Sequence[str]] = None,
    min_pos_dist: int = 2500,
    remove_duplicates: bool = True,
    double_diagonal: bool = False,
    nproc: int = 1,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Sum raw contacts of the cells in each group into one ``.cool``.

    Raw pseudobulks are the input of :func:`compartment_model` /
    :func:`project_compartments` (HCEA computes compartments on raw, not
    imputed, contacts). Per-cell cools are summed pixel by pixel, as
    ``cooler merge``; contact tables are binned like :func:`contacts_to_scool`
    first, as scHiCluster's ``merge_cell_raw``.

    Arguments:
        adata: cells from :func:`load_scool_cells` / :func:`load_cool_collection`
            or :func:`load_contact_tables`.
        groupby: ``adata.obs`` column with the group of each cell.
        out_dir: folder for ``<group>.cool``.
        groups: subset of groups, default all.
        min_pos_dist, remove_duplicates: contact-table filters (see
            :func:`contacts_to_scool`); ignored for cools.
        double_diagonal: add the diagonal a second time, as scHiCluster's
            ``merge_cell_raw`` does. Off by default; the diagonal does not
            enter any compartment score.
        nproc: groups merged in parallel.
        overwrite: rebuild existing files.

    Returns:
        ``{group: cool_path}``.
    """
    import cooler

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hic = adata.uns.get("hic", {})
    if "contact_path" in adata.obs.columns:
        kind, col = "tsv", "contact_path"
        sizes = pd.Series(hic["chrom_sizes"], dtype=np.int64)
        resolution = int(hic["resolution"])
        bins = cooler.binnify(sizes, resolution)
        columns = hic.get("contact_columns", CONTACT_COLUMNS)
    else:
        kind, col = "cool", "cool_path"
        clr = cooler.Cooler(adata.obs[col].iloc[0])
        bins = clr.bins()[:][["chrom", "start", "end"]]
        resolution, columns = int(clr.binsize), None
    sizes_per_chrom = bins.groupby("chrom", sort=False, observed=True).size()
    offsets = np.r_[0, np.cumsum(sizes_per_chrom.to_numpy())][:-1]
    chrom_offset = dict(zip(sizes_per_chrom.index.astype(str), offsets.astype(np.int64)))

    labels = adata.obs[groupby].astype(str)
    if groups is None:
        groups = sorted(labels.unique())
    out, tasks = {}, []
    for g in groups:
        path = out_dir / f"{g}.cool"
        out[str(g)] = str(path)
        if path.exists() and not overwrite:
            continue
        srcs = adata.obs.loc[labels == str(g), col].astype(str).tolist()
        if not srcs:
            raise ValueError(f"group {g!r} has no cells")
        tasks.append((srcs, kind, bins, chrom_offset, resolution, columns, int(min_pos_dist),
                      bool(remove_duplicates), bool(double_diagonal), path))
    if tasks:
        _map(_merge_group, tasks, nproc, progress=False, desc="merge")
    return out


# ---------------------------------------------------------------------------
# HCEA pseudobulk compartments
# ---------------------------------------------------------------------------

def _oe_corr(Q: np.ndarray, clip: Optional[Tuple[float, float]]):
    """Distance-normalised O/E (``E``) and its clipped correlation (``C``).

    ``E[i, j] = (Q[i, j] + 1e-5) / (mean of diagonal |i - j| + 1e-5)`` with a
    unit diagonal; ``C = corr(log2(E + 0.001))``, clipped to its
    ``clip`` percentiles.
    """
    n = Q.shape[0]
    decay = np.array([np.mean(np.diag(Q, i)) for i in range(n)])
    E = np.zeros(Q.shape)
    row, col = np.diag_indices(n)
    E[row, col] = 1
    for i in range(1, n):
        E[row[:-i], col[i:]] = (Q[row[:-i], col[i:]] + 1e-5) / (decay[i] + 1e-5)
    E = E + E.T
    C = np.corrcoef(np.log2(E + 0.001))
    if clip is not None:
        vmin, vmax = np.percentile(C, clip[0]), np.percentile(C, clip[1])
        C[C < vmin] = vmin
        C[C > vmax] = vmax
    return E, C


def _saddle_sums(E: np.ndarray, pc: np.ndarray, n_q: int):
    labels = pd.qcut(pc, n_q, labels=False)
    onehot = np.zeros((len(labels), n_q))
    onehot[np.arange(len(labels)), labels] = 1.0
    sad = onehot.T @ E @ onehot
    sizes = onehot.sum(axis=0)
    return sad, np.outer(sizes, sizes)


def _contrast(sad: np.ndarray, ws: int) -> float:
    return ((sad[:ws, :ws].sum() + sad[-ws:, -ws:].sum())
            / (sad[:ws, -ws:].sum() + sad[-ws:, :ws].sum()))


def _fetch_q(clr, chrom: str) -> np.ndarray:
    Q = clr.matrix(balance=False, sparse=True).fetch(chrom).toarray()
    return Q - np.diag(np.diag(Q))


def oe_correlation(
    cool_path: PathLike,
    chromosome: str,
    *,
    keep: Optional[np.ndarray] = None,
    resolution: Optional[int] = None,
    clip: Optional[Tuple[float, float]] = (5, 95),
) -> Tuple[np.ndarray, np.ndarray]:
    """O/E and correlation matrices of one chromosome, as used by the HCEA model.

    Arguments:
        cool_path: raw ``.cool`` / ``.mcool``.
        chromosome: chromosome to fetch.
        keep: boolean bin mask (e.g. ``model.loc[model.chrom == c, "keep"]``);
            default all bins.
        resolution: layer of an ``.mcool``.
        clip: percentile clip of the correlation matrix; ``None`` to skip.

    Returns:
        ``(E, C)`` over the kept bins: distance-normalised observed/expected
        with a unit diagonal, and the Pearson correlation of ``log2(E + 0.001)``.
    """
    import cooler

    Q = _fetch_q(cooler.Cooler(_resolve_uri(cool_path, resolution)), chromosome)
    if keep is not None:
        keep = np.asarray(keep, dtype=bool)
        Q = Q[keep][:, keep]
    return _oe_corr(Q, clip)


def compartment_model(
    cool_path: PathLike,
    cpg: pd.DataFrame,
    *,
    resolution: Optional[int] = None,
    chromosomes: Optional[Sequence[str]] = None,
    n_quantiles: int = 50,
    ws: int = 10,
    clip: Optional[Tuple[float, float]] = None,
    random_state: Optional[int] = 0,
) -> pd.DataFrame:
    """Fit the HCEA compartment model on a deep raw pseudobulk.

    Use the contacts of all cells merged (``merge_cells`` with one group, or
    any deep raw ``.cool``). Per chromosome:

    1. **Bin filter.** Drop the diagonal; keep bins whose total contacts
       exceed ``2 * median - p99`` of the non-empty bins.
    2. **O/E and correlation.** Divide each pixel by the mean of its diagonal,
       take ``log2`` and correlate the rows.
    3. **PCA.** Two components (ARPACK). Keep the one whose 50-quantile
       saddle has the higher ``(AA + BB) / (AB + BA)`` contrast over the
       outer ``ws`` quantiles; flip it so it correlates positively with CpG
       density (A > 0).

    Arguments:
        cool_path: raw pseudobulk ``.cool`` / ``.mcool`` (with ``resolution``).
        cpg: per-bin CpG density (:func:`cpg_ratio`) on the cool's bins.
        resolution: layer of an ``.mcool``.
        chromosomes: default the cool's chromosomes present in ``cpg``.
        n_quantiles, ws: saddle quantiles and corner width used to choose
            between PC1 and PC2.
        clip: percentile clip of the correlation matrix before the PCA.
            Default ``None``: HCEA's published model was fitted on the
            unclipped matrix, while :func:`project_compartments` clips at the
            5th / 95th percentiles. With these defaults both reproduce the
            published loadings and scores.
        random_state: seed of the ARPACK start vector.

    Returns:
        ``DataFrame`` with one row per bin of the chosen chromosomes:
        ``chrom``, ``start``, ``end``, ``coverage`` (contacts of the bin,
        diagonal excluded), ``keep`` (bin filter), ``loading``
        (PC loading, 0 for filtered bins) and ``pc`` (0 or 1, the component
        kept on that chromosome). Pass it to :func:`project_compartments`.
    """
    import cooler
    from scipy.stats import pearsonr
    from sklearn.decomposition import PCA

    clr = cooler.Cooler(_resolve_uri(cool_path, resolution))
    cpg = cpg.copy()
    cpg["chrom"] = cpg["chrom"].astype(str)
    if chromosomes is None:
        chromosomes = [c for c in clr.chromnames if c in set(cpg["chrom"])]
    bins = clr.bins()[:][["chrom", "start", "end"]]
    bins["chrom"] = bins["chrom"].astype(str)
    out = []
    for chrom in chromosomes:
        Q = _fetch_q(clr, chrom)
        rowsum = Q.sum(axis=0)
        nz = rowsum[rowsum > 0]
        p50, p99 = np.percentile(nz, 50), np.percentile(nz, 99)
        keep = rowsum > (p50 * 2 - p99)
        Q = Q[keep][:, keep]
        E, C = _oe_corr(Q, clip)
        pca = PCA(n_components=2, svd_solver="arpack", random_state=random_state)
        pc = pca.fit_transform(C)
        cpg_c = cpg.loc[cpg["chrom"] == chrom, "cpg_ratio"].to_numpy()[keep]
        r = []
        for k in range(2):
            s, c = _saddle_sums(E, pc[:, k], n_quantiles)
            r.append(_contrast(s / c, ws))
        k = 0 if r[0] > r[1] else 1
        sign = 1.0 if pearsonr(cpg_c, pc[:, k])[0] > 0 else -1.0
        b = bins[bins["chrom"] == chrom].copy()
        b["coverage"] = rowsum
        b["keep"] = keep
        b["loading"] = 0.0
        b.loc[b.index[keep], "loading"] = sign * pca.components_[k]
        b["pc"] = k
        out.append(b)
    return pd.concat(out, ignore_index=True)


def _project_one(args):
    import cooler

    path, model, clip = args
    clr = cooler.Cooler(path)
    scores = []
    for chrom, m in model.groupby("chrom", sort=False):
        keep = m["keep"].to_numpy()
        Q = _fetch_q(clr, chrom)[keep][:, keep]
        _, C = _oe_corr(Q, clip)
        scores.append((C - np.mean(C, axis=0)).dot(m.loc[m["keep"], "loading"].to_numpy()))
    return np.concatenate(scores)


def project_compartments(
    cool_paths: Union[Mapping[str, PathLike], Sequence[PathLike]],
    model: pd.DataFrame,
    *,
    resolution: Optional[int] = None,
    clip: Optional[Tuple[float, float]] = (5, 95),
    nproc: int = 1,
) -> pd.DataFrame:
    """Project raw pseudobulks on a :func:`compartment_model`.

    For each cool and chromosome: the model's bins, the same O/E and clipped
    correlation as the fit, then ``(C - colmeans(C)) @ loading``. Because
    every group uses the merged model's bins and loadings, the scores are on
    one scale (HCEA ``compute_comp``).

    Arguments:
        cool_paths: ``{name: path}`` (e.g. from :func:`merge_cells`) or a list
            (names = file stems).
        model: output of :func:`compartment_model`.
        resolution: layer of ``.mcool`` inputs.
        clip: percentile clip of the correlation matrix before projecting
            (HCEA: 5th / 95th percentiles, also when the model was fitted
            unclipped).
        nproc: cools processed in parallel.

    Returns:
        ``DataFrame`` with one row per kept bin: ``chrom``, ``start``, ``end``
        and one score column per cool (positive = A).
    """
    if not isinstance(cool_paths, Mapping):
        cool_paths = {Path(str(p).split("::")[0]).stem: p for p in cool_paths}
    names = list(cool_paths)
    tasks = [(_resolve_uri(cool_paths[n], resolution), model, clip) for n in names]
    res = _map(_project_one, tasks, nproc, progress=False, desc="project")
    out = model.loc[model["keep"], ["chrom", "start", "end"]].reset_index(drop=True)
    for n, v in zip(names, res):
        out[n] = v
    return out


def compartment_saddle(
    cool_path: PathLike,
    model: pd.DataFrame,
    *,
    resolution: Optional[int] = None,
    chromosomes: Optional[Sequence[str]] = None,
    n_quantiles: int = 50,
    ws: int = 10,
    clip: Optional[Tuple[float, float]] = (5, 95),
) -> dict:
    """HCEA saddle plot and compartment strength of one raw pseudobulk.

    Per chromosome, bins are cut into ``n_quantiles`` quantiles of the
    projected score; O/E contacts (as in :func:`compartment_model`) are summed
    over each pair of quantiles and divided by the number of bin pairs, after
    summing over chromosomes.

    Arguments:
        cool_path: raw pseudobulk ``.cool`` / ``.mcool``.
        model: output of :func:`compartment_model`.
        resolution: layer of an ``.mcool``.
        chromosomes: subset of the model's chromosomes.
        n_quantiles: quantiles per axis.
        ws: number of quantiles in each corner for the strength summary.
        clip: percentile clip used to compute the projected score. HCEA's
            ``compsaddle`` projects without clipping (``clip=None``).

    Returns:
        ``dict`` with ``saddle`` (mean O/E, quantiles ordered B → A),
        ``sum`` and ``count`` (``n_quantiles × n_quantiles``), ``scores``
        (the projected score per kept bin) and ``strength`` (Series with
        ``BB``, ``AA``, ``BA``, ``AB`` corner means and
        ``strength = (BB + AA) / (BA + AB)`` on corner sums).
    """
    import cooler

    clr = cooler.Cooler(_resolve_uri(cool_path, resolution))
    sad = np.zeros((n_quantiles, n_quantiles))
    count = np.zeros((n_quantiles, n_quantiles))
    scores = []
    for chrom, m in model.groupby("chrom", sort=False):
        if chromosomes is not None and chrom not in chromosomes:
            continue
        keep = m["keep"].to_numpy()
        Q = _fetch_q(clr, chrom)[keep][:, keep]
        E, C = _oe_corr(Q, clip)
        pc = (C - np.mean(C, axis=0)).dot(m.loc[m["keep"], "loading"].to_numpy())
        scores.append(pd.Series(pc, index=m.index[keep]))
        s, c = _saddle_sums(E, pc, n_quantiles)
        sad += s
        count += c
    mean = sad / count
    strength = pd.Series({
        "BB": mean[:ws, :ws].mean(), "AA": mean[-ws:, -ws:].mean(),
        "BA": mean[:ws, -ws:].mean(), "AB": mean[-ws:, :ws].mean(),
        "strength": _contrast(mean, ws),
    })
    return {"saddle": mean, "sum": sad, "count": count, "strength": strength,
            "scores": pd.concat(scores)}


def write_bedgraph(scores: pd.DataFrame, column: str, path: PathLike) -> str:
    """Write one score column of :func:`project_compartments` as a bedGraph."""
    scores[["chrom", "start", "end", column]].to_csv(path, sep="\t", header=False, index=False)
    return str(path)
