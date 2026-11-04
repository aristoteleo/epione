"""Preprocessing shared by the translation methods.

The defaults follow scButterfly's ``RNA_data_preprocessing`` /
``ATAC_data_preprocessing``: library-size normalisation, log1p and 3,000
highly variable genes for RNA; binarisation, a minimum-cell filter, TF-IDF
and division by the global maximum for ATAC. The ATAC steps are done on the
sparse matrix (the reference densifies the full peak-by-cell matrix).
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

_PEAK_RE = re.compile(r"^(?P<chrom>[^:_\-]+)[:_\-](?P<start>\d+)[:_\-](?P<end>\d+)$")


def _as_csr(X) -> sp.csr_matrix:
    if sp.issparse(X):
        return sp.csr_matrix(X)
    return sp.csr_matrix(np.asarray(X))


def peak_chromosomes(atac: AnnData) -> np.ndarray:
    """Chromosome of every peak, from ``var['chrom']`` or ``chr:start-end`` names."""
    for key in ("chrom", "chr", "seqnames", "Chromosome"):
        if key in atac.var:
            return atac.var[key].astype(str).to_numpy()
    out = []
    for name in atac.var_names:
        m = _PEAK_RE.match(str(name))
        if m is None:
            raise ValueError(
                f"cannot parse a chromosome from peak name {name!r}; "
                "add a var['chrom'] column"
            )
        out.append(m.group("chrom"))
    return np.asarray(out)


def chrom_blocks(atac: AnnData) -> list[int]:
    """Number of consecutive peaks per chromosome, in ``var`` order.

    scButterfly's ATAC encoder and decoder have one block per chromosome and
    split the peak axis by these counts, so peaks of a chromosome must be
    contiguous; :func:`prepare_atac` sorts them that way.
    """
    chroms = peak_chromosomes(atac)
    if len(chroms) == 0:
        return []
    change = np.flatnonzero(chroms[1:] != chroms[:-1]) + 1
    bounds = np.concatenate([[0], change, [len(chroms)]])
    names = chroms[bounds[:-1]]
    if len(set(names)) != len(names):
        raise ValueError("peaks of a chromosome are not contiguous; run prepare_atac first")
    return np.diff(bounds).astype(int).tolist()


def prepare_rna(
    rna: AnnData,
    *,
    normalize_total: bool = True,
    log1p: bool = True,
    n_top_genes: Optional[int] = 3000,
    layer: Optional[str] = None,
    output: str = "lognorm",
) -> AnnData:
    """Normalise, log-transform and select highly variable genes.

    ``output="counts"`` selects the same genes but returns their raw counts
    (for methods that model counts, such as BABEL).

    Same steps and scanpy calls as scButterfly's ``RNA_data_preprocessing``:
    ``normalize_total`` (median library size), ``log1p`` and
    ``highly_variable_genes(flavor="seurat")`` on the log data, ranking only
    genes with at least one count.
    """
    import scanpy as sc

    raw = _as_csr(rna.layers[layer] if layer else rna.X).astype(np.float32)
    out = AnnData(X=raw.copy(), obs=rna.obs.copy(), var=rna.var.copy())
    out.var_names_make_unique()
    out.layers["counts"] = raw
    if normalize_total:
        sc.pp.normalize_total(out)
    if log1p:
        sc.pp.log1p(out)
    if n_top_genes:
        # genes without counts would shift the mean bins of the dispersion ranking
        expressed = np.asarray((out.X > 0).sum(axis=0)).ravel() > 0
        if not expressed.all():
            out = out[:, expressed].copy()
        sc.pp.highly_variable_genes(out, n_top_genes=n_top_genes)
        out = out[:, out.var["highly_variable"].to_numpy()].copy()
    if output == "counts":
        out.X = out.layers["counts"].copy()
    elif output != "lognorm":
        raise ValueError(f"output must be 'lognorm' or 'counts', not {output!r}")
    del out.layers["counts"]
    out.uns["translate_prep"] = {
        "modality": "rna",
        "output": output,
        "normalize_total": normalize_total,
        "log1p": log1p,
        "n_top_genes": n_top_genes,
    }
    return out


def tfidf(X: sp.spmatrix) -> sp.csr_matrix:
    """TF-IDF as in scButterfly: ``(x / cell total) * log(1 + n_cells / peak total)``."""
    X = _as_csr(X).astype(np.float64)
    n_cells = X.shape[0]
    cell_tot = np.asarray(X.sum(axis=1)).ravel()
    peak_tot = np.asarray(X.sum(axis=0)).ravel()
    with np.errstate(divide="ignore"):
        idf = np.log1p(n_cells / peak_tot)
        inv_cell = np.where(cell_tot > 0, 1.0 / cell_tot, 0.0)
    out = sp.diags(inv_cell) @ X @ sp.diags(idf)
    return sp.csr_matrix(out)


def prepare_atac(
    atac: AnnData,
    *,
    binarize: bool = True,
    min_cells_frac: Optional[float] = 0.005,
    use_tfidf: bool = True,
    scale: bool = True,
    sort_peaks: bool = True,
    layer: Optional[str] = None,
) -> AnnData:
    """Binarise, filter rare peaks, TF-IDF and scale ATAC counts to [0, 1].

    Matches scButterfly's ``ATAC_data_preprocessing``: peaks open in fewer than
    ``ceil(min_cells_frac * n_cells)`` cells are dropped, TF-IDF is computed on
    the binary matrix and the result is divided by its global maximum. Peaks are
    then ordered by chromosome (stable within a chromosome) so that
    :func:`chrom_blocks` can split them.
    """
    X = _as_csr(atac.layers[layer] if layer else atac.X)
    var = atac.var.copy()
    if binarize:
        X = X.copy()
        X.data = (X.data > 0).astype(np.float32)
        X.eliminate_zeros()
    if min_cells_frac:
        n_open = np.asarray((X > 0).sum(axis=0)).ravel()
        keep = n_open >= np.ceil(min_cells_frac * X.shape[0])
        X = X[:, keep]
        var = var.iloc[np.flatnonzero(keep)]
    max_value = None
    if use_tfidf:
        X = tfidf(X)
    if scale:
        max_value = float(X.max()) if X.nnz else 1.0
        X = X / max_value
    out = AnnData(X=_as_csr(X).astype(np.float32), obs=atac.obs.copy(), var=var)
    if sort_peaks:
        chroms = peak_chromosomes(out)
        first = pd.Series(np.arange(len(chroms))).groupby(chroms).transform("min").to_numpy()
        order = np.lexsort((np.arange(len(chroms)), first))
        if np.any(order != np.arange(len(order))):
            out = out[:, order].copy()
    out.uns["translate_prep"] = {
        "modality": "atac",
        "binarize": binarize,
        "min_cells_frac": min_cells_frac,
        "tfidf": use_tfidf,
        "scale_max": max_value,
    }
    return out


def prepare(
    rna: AnnData,
    atac: AnnData,
    *,
    n_top_genes: Optional[int] = 3000,
    min_cells_frac: Optional[float] = 0.005,
    method: str = "scbutterfly",
    rna_kwargs: Optional[dict] = None,
    atac_kwargs: Optional[dict] = None,
) -> tuple[AnnData, AnnData]:
    """Preprocess a paired RNA / ATAC dataset for :func:`epione.translate.fit`.

    Parameters
    ----------
    rna, atac
        Raw counts for the same cells (``obs_names`` must match; they are
        aligned to the RNA order).
    n_top_genes
        Highly variable genes kept for RNA (``None`` keeps all).
    min_cells_frac
        Minimum fraction of cells in which a peak must be open.
    method
        The method the data are for. ``"scbutterfly"``: log-normalised RNA and
        TF-IDF-scaled ATAC. ``"babel"``: raw RNA counts of the same genes and
        binary ATAC (BABEL normalises RNA internally). ``"scpair"``: raw RNA
        counts and raw peak counts (scPair binarises peaks internally).
        ``"jamie"``: raw RNA counts and peak counts z-scored over all the given
        cells (dense), as the scTranslation runner prepares them.

    Returns
    -------
    ``(rna, atac)`` preprocessed AnnData objects with identical ``obs_names``.
    """
    common = rna.obs_names.intersection(atac.obs_names, sort=False)
    if len(common) == 0:
        raise ValueError("RNA and ATAC share no obs_names")
    if len(common) < max(rna.n_obs, atac.n_obs):
        rna = rna[common]
        atac = atac[common]
    elif not np.array_equal(rna.obs_names, atac.obs_names):
        atac = atac[rna.obs_names]
    rkw, akw = dict(rna_kwargs or {}), dict(atac_kwargs or {})
    if method in ("babel", "scpair"):
        rkw.setdefault("output", "counts")
        akw.setdefault("use_tfidf", False)
        akw.setdefault("scale", False)
        if method == "scpair":
            akw.setdefault("binarize", False)  # scPair binarises internally; its truth is counts
    elif method not in ("scbutterfly", "jamie"):
        raise ValueError(f"unknown method {method!r}")
    if method == "jamie":
        rkw.setdefault("output", "counts")
        akw.setdefault("binarize", False)
        akw.setdefault("use_tfidf", False)
        akw.setdefault("scale", False)
    rna_p = prepare_rna(rna, n_top_genes=n_top_genes, **rkw)
    atac_p = prepare_atac(atac, min_cells_frac=min_cells_frac, **akw)
    if method == "jamie":
        # scTranslation's JAMIE runner z-scores the peak counts over all cells
        import scanpy as sc

        atac_p = AnnData(X=np.asarray(atac_p.X.toarray(), dtype=np.float32), obs=atac_p.obs, var=atac_p.var,
                         uns=atac_p.uns)
        sc.pp.scale(atac_p, zero_center=True)
        atac_p.X = np.nan_to_num(atac_p.X).astype(np.float32)
        atac_p.uns["translate_prep"]["zscore"] = True
    return rna_p, atac_p


def size_normalize_matrix(X, target_sum: Optional[float] = None):
    """Scale each row of ``X`` to ``target_sum`` (default: the median row sum).

    Returns ``(CSR float32, target_sum)``.
    """
    X = sp.csr_matrix(X, dtype=np.float64)
    tot = np.asarray(X.sum(axis=1)).ravel()
    if target_sum is None:
        target_sum = float(np.median(tot))
    scale = np.divide(target_sum, tot, out=np.zeros_like(tot), where=tot > 0)
    return sp.csr_matrix(sp.diags(scale) @ X, dtype=np.float32), target_sum


def size_normalize(adata: AnnData, target_sum: Optional[float] = None) -> AnnData:
    """Scale each cell's counts to ``target_sum`` (default: the median count per cell).

    Puts measured RNA counts on the scale of BABEL's ATAC→RNA predictions
    (BABEL normalises each set of cells to its own median).
    """
    X, t = size_normalize_matrix(adata.X, target_sum)
    out = AnnData(X=X, obs=adata.obs.copy(), var=adata.var.copy())
    out.uns["size_normalize"] = {"target_sum": t}
    return out


def check_paired(rna: AnnData, atac: AnnData) -> None:
    if rna.n_obs != atac.n_obs or not np.array_equal(rna.obs_names, atac.obs_names):
        raise ValueError("rna and atac must hold the same cells in the same order")


def resolve_split(
    n: int,
    train_idx: Optional[Sequence[int]],
    val_idx: Optional[Sequence[int]],
    *,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Training and validation indices.

    Without ``val_idx`` the training cells are shuffled and the last
    ``val_frac`` held out, as scButterfly's ``load_data`` does (80 / 20).
    """
    train = np.arange(n) if train_idx is None else np.asarray(train_idx, dtype=int)
    if val_idx is not None:
        return train, np.asarray(val_idx, dtype=int)
    rng = np.random.default_rng(seed)
    perm = train[rng.permutation(len(train))]
    n_train = int(len(perm) * (1 - val_frac))
    return perm[:n_train], perm[n_train:]


def prepare_counts(
    rna: AnnData,
    atac: AnnData,
    *,
    n_top_genes: Optional[int] = 3000,
    min_cells_frac: Optional[float] = 0.005,
    sort_peaks: bool = True,
) -> tuple[AnnData, AnnData]:
    """Raw-count inputs for count-likelihood methods (multiDGD).

    Highly variable genes are selected as in :func:`prepare_rna` (on
    library-normalised, log-transformed data), but the returned RNA matrix
    holds the raw counts of those genes. ATAC keeps raw counts of the peaks
    with a nonzero count in at least ``ceil(min_cells_frac * n_cells)`` cells.

    Returns
    -------
    ``(rna, atac)`` count matrices (float32 CSR) with identical ``obs_names``.
    """
    common = rna.obs_names.intersection(atac.obs_names, sort=False)
    if len(common) == 0:
        raise ValueError("RNA and ATAC share no obs_names")
    rna, atac = rna[common], atac[common]
    hvg = prepare_rna(rna, n_top_genes=n_top_genes).var_names
    rna_c = AnnData(X=_as_csr(rna.X).astype(np.float32), obs=rna.obs.copy(), var=rna.var.copy())
    rna_c.var_names_make_unique()
    rna_c = rna_c[:, hvg].copy()
    atac_c = prepare_atac(atac, binarize=False, min_cells_frac=min_cells_frac, use_tfidf=False,
                          scale=False, sort_peaks=sort_peaks)
    atac_c.X = atac_c.X.astype(np.float32)
    rna_c.uns["translate_prep"] = {"modality": "rna", "counts": True, "n_top_genes": n_top_genes}
    atac_c.uns["translate_prep"] = {"modality": "atac", "counts": True, "min_cells_frac": min_cells_frac}
    return rna_c, atac_c
