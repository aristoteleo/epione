"""Gene scores from raw single-cell contacts (scHiCluster ``gene-score --mode raw``)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from ._compartment import CONTACT_COLUMNS, _map, _read_contacts


def _read_genes(genes) -> pd.DataFrame:
    if isinstance(genes, pd.DataFrame):
        missing = {"chrom", "start", "end"} - set(genes.columns)
        if missing:
            raise ValueError(f"genes needs columns chrom, start, end; missing {sorted(missing)}")
        return genes.copy()
    df = pd.read_csv(genes, sep="\t", header=None, comment="#")
    df = df.rename(columns={0: "chrom", 1: "start", 2: "end"}).set_index(3)
    df.index.name = None
    return df


def _score_cell(args):
    path, columns, chroms, n_bins, gene_bins, resolution, min_pos_dist = args
    from scipy.sparse import csr_matrix

    c1, p1, c2, p2 = (columns[k] for k in ("chrom1", "pos1", "chrom2", "pos2"))
    df = _read_contacts(path, columns)
    df = df[(df[c1] == df[c2]) & df[c1].isin(chroms)]
    if min_pos_dist > 0:
        df = df[(df[p1] - df[p2]).abs() > min_pos_dist]
    out = []
    for chrom in chroms:
        x = df[df[c1] == chrom]
        n = n_bins[chrom]
        b1 = ((x[p1].to_numpy() - 1) // resolution).astype(np.int64)
        b2 = ((x[p2].to_numpy() - 1) // resolution).astype(np.int64)
        D = csr_matrix((np.ones(len(b1), np.int32), (b1, b2)), shape=(n, n))
        for lo, hi in gene_bins[chrom]:
            out.append(D[lo:hi + 1, lo:hi + 1].sum())
    return np.asarray(out, dtype=np.float32)


def gene_score(
    adata,
    genes: Union[pd.DataFrame, str, Path],
    *,
    resolution: int = 10_000,
    slop: int = 0,
    min_pos_dist: int = 0,
    nproc: int = 1,
    progress: bool = True,
):
    """Count each cell's cis contacts inside each gene body.

    The same score as scHiCluster ``hicluster gene-score --mode raw``: cis
    contacts are binned at ``resolution`` (``(pos - 1) // resolution``) and a
    gene scores every contact whose two ends fall in the bins it spans, from
    ``start // resolution`` to ``end // resolution``. Duplicated contacts are
    kept, as upstream.

    Arguments:
        adata: AnnData from :func:`load_contact_tables` (``obs['contact_path']``).
        genes: gene table with ``chrom``, ``start`` (0-based) and ``end``
            columns, indexed by gene id, or a BED file (chrom, start, end,
            gene id, ...). Other columns are kept in ``var``. Genes on
            chromosomes outside ``adata.uns['hic']['chromosomes']`` are dropped.
        resolution: bin size in bp of the score (scHiCluster: 10 kb).
        slop: bp added on each side of every gene (scHiCluster ``--slop``).
        min_pos_dist: drop cis contacts whose ends are at most this many bp
            apart. 0 (upstream) keeps all; most contacts in sparse droplet
            data are shorter than 1 kb and mostly measure coverage.
        nproc: worker processes.
        progress: show a progress bar.

    Returns:
        ``AnnData`` of cells × genes (``X``: float32 sparse counts) with the
        cell ``obs`` of ``adata``, the gene table as ``var`` and the settings
        in ``uns['hic']['gene_score']``.
    """
    import anndata as ad
    from scipy.sparse import csr_matrix

    if "contact_path" not in adata.obs:
        raise ValueError("gene_score reads obs['contact_path']; index cells with load_contact_tables")
    info = adata.uns["hic"]
    chroms = list(info["chromosomes"])
    sizes = info["chrom_sizes"]
    columns = dict(info.get("contact_columns", CONTACT_COLUMNS))
    res = int(resolution)
    var = _read_genes(genes)
    var = var[var["chrom"].astype(str).isin(chroms)]
    lo = (var["start"].astype(np.int64) - int(slop)).clip(lower=0) // res
    hi = (var["end"].astype(np.int64) + int(slop)) // res
    n_bins = {c: int(sizes[c] // res + 1) for c in chroms}
    hi = np.minimum(hi.to_numpy(), var["chrom"].map(n_bins).to_numpy() - 1)
    gene_bins, order = {}, []
    for c in chroms:
        m = (var["chrom"] == c).to_numpy()
        gene_bins[c] = list(zip(lo.to_numpy()[m].tolist(), hi[m].tolist()))
        order.append(np.flatnonzero(m))
    var = var.iloc[np.concatenate(order)] if order else var.iloc[:0]

    tasks = [(p, columns, chroms, n_bins, gene_bins, res, int(min_pos_dist))
             for p in adata.obs["contact_path"]]
    rows = _map(_score_cell, tasks, nproc, progress, "gene_score")
    X = csr_matrix(np.vstack(rows)) if rows else csr_matrix((0, len(var)), dtype=np.float32)
    obs = adata.obs.drop(columns="contact_path").copy()
    out = ad.AnnData(X=X, obs=obs, var=var.astype({"chrom": str}))
    out.var_names = out.var_names.astype(str)
    out.uns["hic"] = {**{k: v for k, v in info.items() if k != "n_chrom_bins"},
                      "gene_score": {"resolution": res, "slop": int(slop),
                                     "min_pos_dist": int(min_pos_dist)}}
    return out
