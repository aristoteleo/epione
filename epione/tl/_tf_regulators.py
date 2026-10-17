"""chromVAR-downstream ArchR ports: motif variability, matrix correlation
over kNN cell groups, and positive-TF-regulator calling.

Ports of ArchR 1.0.3 (MIT) ``getVarDeviations`` (MatrixDeviations.R),
``correlateMatrices`` (IntegrativeAnalysis.R), ``getGroupSE`` (GroupExport.R)
and the "positive TF regulators" workflow of the ArchR book (ch. 15.4).
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse, stats

from ._r_rng import RRandom


def _console(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"   {msg}")


def _dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def _get_feature_matrix(adata, key: Optional[str] = None, layer: Optional[str] = None,
                        names: Optional[Sequence[str]] = None):
    """Return ``(X cells×features, feature names)`` from an AnnData.

    ``key`` selects an ``obsm`` entry (e.g. chromVAR ``'motif_deviations'``,
    names from ``uns[key + '_names']``); otherwise ``layer`` / ``X`` with
    ``var_names``.
    """
    if key is not None:
        X = adata.obsm[key]
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(), np.asarray(X.columns, dtype=object)
        if names is None:
            names = adata.uns.get(f"{key}_names")
        if names is None:
            raise KeyError(f"feature names for obsm[{key!r}] not found; pass names=")
        return X, np.asarray(names, dtype=object)
    X = adata.layers[layer] if layer is not None else adata.X
    return X, np.asarray(adata.var_names, dtype=object)


# --------------------------------------------------------------------------
# getVarDeviations
# --------------------------------------------------------------------------
def _combine_variances(means: np.ndarray, variances: np.ndarray, ns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ArchR ``.combineVariances`` (fishmethods::combinevar)."""
    variances = np.where(np.isnan(variances), 0.0, variances)
    ntot = ns.sum()
    combined_means = (means * ns).sum(axis=1) / ntot
    summed = (variances * (ns - 1) + means ** 2 * ns).sum(axis=1)
    combined_vars = (summed - ntot * combined_means ** 2) / (ntot - 1)
    return combined_vars, combined_means


def var_deviations(
    adata,
    key: str = "motif_deviations",
    *,
    names: Optional[Sequence[str]] = None,
    sample_key: Optional[str] = "Sample",
    key_added: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    r"""Rank motifs / annotations by the variability of their chromVAR Z-scores.

    Port of ArchR ``getVarDeviations``. ArchR stores, per Arrow file
    (= sample), the row mean and the unbiased row variance of the deviation
    Z-scores and pools them with ``.combineVariances``; this function does
    the same from the cell × motif Z-score matrix, so the ``combinedVars``
    column equals ArchR's.

    Arguments:
        adata: AnnData with the Z-scores in ``obsm[key]`` (as written by
            :func:`epione.tl.compute_deviations`) or a cell × motif
            :class:`pandas.DataFrame` in ``obsm``.
        key: ``obsm`` key of the Z-score matrix.
        names: motif names (default ``uns[key + '_names']``).
        sample_key: ``obs`` column defining the per-sample blocks ArchR pools
            over (its Arrow files). ``None`` treats all cells as one block —
            mathematically the same pooled variance.
        key_added: ``uns`` key for the result (default ``key + '_var'``).
        verbose: print progress.

    Returns:
        DataFrame with ``name``, ``combinedVars``, ``combinedMeans``, ``rank``
        (1 = most variable), sorted by decreasing variability; also stored in
        ``adata.uns[key_added]``.
    """
    Z, names = _get_feature_matrix(adata, key=key, names=names)
    Z = _dense(Z).astype(np.float64, copy=False)
    if sample_key is not None and sample_key in adata.obs:
        groups = pd.Categorical(adata.obs[sample_key])
        codes = groups.codes
        blocks = [np.flatnonzero(codes == i) for i in range(len(groups.categories))]
        blocks = [b for b in blocks if len(b) > 0]
    else:
        blocks = [np.arange(Z.shape[0])]

    means = np.empty((Z.shape[1], len(blocks)))
    variances = np.empty_like(means)
    ns = np.empty(len(blocks))
    for j, b in enumerate(blocks):
        zb = Z[b]
        means[:, j] = zb.mean(axis=0)
        variances[:, j] = zb.var(axis=0, ddof=1) if len(b) > 1 else np.nan
        ns[j] = len(b)

    cv, cm = _combine_variances(means, variances, ns)
    df = pd.DataFrame({"seqnames": "z", "idx": np.arange(1, len(names) + 1),
                       "name": names, "combinedVars": cv, "combinedMeans": cm})
    df = df.iloc[np.argsort(-cv, kind="stable")].reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    adata.uns[key_added or f"{key}_var"] = df
    _console(f"var_deviations: top = {', '.join(map(str, df['name'].head(5)))}", verbose)
    return df


# --------------------------------------------------------------------------
# correlateMatrices
# --------------------------------------------------------------------------
_NAME_RULES = {
    "underscore": r"_.*",
    "dash": r"-.*",
    "numeric": r"[0-9]+",
    "dot": r"\..*",
}


def _match_names(names: Iterable[str], remove: Optional[Sequence[str]]) -> np.ndarray:
    out = []
    remove = [r.lower() for r in (remove or [])]
    for n in names:
        s = str(n).upper()
        for rule in ("underscore", "dash", "numeric", "dot"):
            if rule in remove:
                s = re.sub(_NAME_RULES[rule], "", s)
        out.append(s)
    return np.asarray(out, dtype=object)


def _row_zscore(m: np.ndarray) -> np.ndarray:
    return (m - m.mean(axis=1, keepdims=True)) / m.std(axis=1, ddof=1, keepdims=True)


def _reduced_dims(adata, use_rep, dims_to_use, scale_dims, depth_key, cor_cutoff):
    rd = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    if scale_dims:
        rd = _row_zscore(rd)
    dims = np.arange(rd.shape[1]) if dims_to_use is None else np.asarray(dims_to_use) - 1
    dims = dims[dims < rd.shape[1]]
    if depth_key is not None and cor_cutoff is not None:
        depth = np.log10(np.asarray(adata.obs[depth_key], dtype=np.float64) + 1)
        cors = np.array([np.corrcoef(rd[:, d], depth)[0, 1] for d in dims])
        dims = dims[np.abs(cors) <= cor_cutoff]
    return rd[:, dims]


def _knn_groups(rd: np.ndarray, seed_idx: np.ndarray, k: int, overlap_cutoff: float) -> np.ndarray:
    """Exact Euclidean kNN of the seed cells (self included, like
    ``nabor::knn``) followed by ArchR ``determineOverlapCpp``."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean").fit(rd)
    dist, knn = nn.kneighbors(rd[seed_idx])
    # Stable tie-break on index, as nabor's kd-tree returns ties lowest-index first.
    order = np.lexsort((knn, dist), axis=1)
    knn = np.take_along_axis(knn, order, axis=1)

    n = knn.shape[0]
    rows = np.repeat(np.arange(n), k)
    ind = sparse.csr_matrix((np.ones(rows.size, dtype=np.int32), (rows, knn.ravel())),
                            shape=(n, rd.shape[0]))
    ov = (ind @ ind.T).toarray()
    cut = int(np.floor(overlap_cutoff * k))
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    for i in range(1, n):
        prev = ov[i, :i][keep[:i]]
        max_ov = prev.max() if prev.size else 0
        keep[i] = max(max_ov, 0) <= cut
    return knn[keep]


def _group_means(X, groups: np.ndarray, k: int) -> np.ndarray:
    """Sum of ``X`` (cells × features) over each kNN group, divided by k →
    groups × features (ArchR ``.getGroupMatrix`` / k)."""
    g = groups.shape[0]
    rows = np.repeat(np.arange(g), groups.shape[1])
    A = sparse.csr_matrix((np.ones(rows.size), (rows, groups.ravel())), shape=(g, X.shape[0]))
    M = A @ X
    return _dense(M) / k


def _cor_test(x: np.ndarray, y: np.ndarray, min_obs: int = 10) -> Tuple[float, float]:
    """R ``cor.test(x, y)`` (Pearson, two-sided) on complete pairs."""
    ok = ~(np.isnan(x) | np.isnan(y))
    if (~np.isnan(x)).sum() <= min_obs or (~np.isnan(y)).sum() <= min_obs:
        return np.nan, np.nan
    x, y = x[ok], y[ok]
    n = x.size
    xc, yc = x - x.mean(), y - y.mean()
    den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    if den == 0:
        return np.nan, np.nan
    r = float((xc * yc).sum() / den)
    r = min(1.0, max(-1.0, r))
    df = n - 2
    if abs(r) == 1.0:
        return r, 0.0
    t = np.sqrt(df) * r / np.sqrt(1 - r * r)
    p = float(2 * min(stats.t.cdf(t, df), stats.t.sf(t, df)))
    return r, p


def correlate_matrices(
    adata1,
    adata2=None,
    *,
    key1: Optional[str] = None,
    key2: Optional[str] = None,
    layer1: Optional[str] = None,
    layer2: Optional[str] = None,
    names1: Optional[Sequence[str]] = None,
    names2: Optional[Sequence[str]] = None,
    name1: str = "matrix1",
    name2: str = "matrix2",
    remove_from_name1: Optional[Sequence[str]] = ("underscore", "dash"),
    remove_from_name2: Optional[Sequence[str]] = ("underscore", "dash"),
    log2_norm1: bool = True,
    log2_norm2: bool = True,
    use_rep: str = "X_birmt",
    dims_to_use: Optional[Sequence[int]] = None,
    scale_dims: bool = False,
    depth_key: Optional[str] = None,
    cor_cutoff: Optional[float] = 0.75,
    k: int = 100,
    knn_iteration: int = 500,
    overlap_cutoff: float = 0.8,
    seed: int = 1,
    rng: str = "archr",
    seed_cells: Optional[Sequence[int]] = None,
    padj_method: str = "bonferroni",
    verbose: bool = True,
) -> pd.DataFrame:
    r"""Correlate two cell × feature matrices across kNN cell groups.

    Port of ArchR ``correlateMatrices``: features of the two matrices are
    paired by upper-cased name (after stripping ``_…`` / ``-…`` etc.),
    ``knn_iteration`` seed cells are drawn, each seed's ``k`` nearest
    neighbours in the reduced dimensions form a group, groups overlapping a
    previously kept group by more than ``overlap_cutoff·k`` cells are
    discarded, group means are (optionally) ``log2(x+1)``-transformed and
    each feature pair is tested with Pearson ``cor.test`` + Bonferroni.

    Arguments:
        adata1: AnnData holding matrix 1 (cells × features) and the reduced
            dimensions ``obsm[use_rep]``.
        adata2: AnnData holding matrix 2 with the same ``obs_names``
            (default ``adata1``, e.g. gene scores in ``X`` and chromVAR
            Z-scores in ``obsm``).
        key1, key2: optional ``obsm`` keys for each matrix (names from
            ``uns[key + '_names']`` or ``names1/names2``).
        layer1, layer2: optional layers (otherwise ``X``).
        name1, name2: prefixes of the output name columns (ArchR uses the
            matrix names, e.g. ``'GeneScoreMatrix'``, ``'MotifMatrix'``).
        remove_from_name1, remove_from_name2: name-stripping rules among
            ``'underscore'``, ``'dash'``, ``'numeric'``, ``'dot'``.
        log2_norm1, log2_norm2: ``log2(x+1)`` the group means unless any is
            negative (deviation matrices are left as is, like ArchR).
        use_rep: ``obsm`` key of the embedding used for kNN groups.
        dims_to_use: 1-based dimensions to keep (ArchR ``dimsToUse``).
        scale_dims: row-z-score the embedding first (ArchR ``scaleDims``;
            ArchR's IterativeLSI default is ``TRUE``).
        depth_key: ``obs`` column of fragment counts; dims whose |Pearson r|
            with ``log10(depth+1)`` exceeds ``cor_cutoff`` are dropped.
        cor_cutoff: see ``depth_key``.
        k: neighbours per group.
        knn_iteration: number of seed cells.
        overlap_cutoff: maximum allowed fractional overlap with kept groups.
        seed: random seed.
        rng: how the seed cells are drawn. ``'archr'`` (default) reproduces
            ``set.seed(seed); sample()`` bit-exactly under the
            ``L'Ecuyer-CMRG`` generator that ``addArchRThreads(threads > 1)``
            selects — i.e. the same groups as a standard multi-threaded ArchR
            session; ``'R'`` does the same for R's default Mersenne-Twister
            (single-threaded ArchR); ``'numpy'`` uses
            :func:`numpy.random.default_rng`.
        seed_cells: explicit 1-based seed-cell indices (overrides ``rng``).
        padj_method: ``'bonferroni'`` (ArchR) or ``'fdr_bh'``.
        verbose: print progress.

    Returns:
        DataFrame with ``{name1}_name``, ``{name2}_name``, ``cor``, ``padj``,
        ``pval``, the match names and feature indices; the kNN groups (0-based
        cell indices) are attached as ``df.attrs['groups']``.
    """
    adata2 = adata1 if adata2 is None else adata2
    X1, n1 = _get_feature_matrix(adata1, key1, layer1, names1)
    X2, n2 = _get_feature_matrix(adata2, key2, layer2, names2)
    if not np.array_equal(np.asarray(adata1.obs_names), np.asarray(adata2.obs_names)):
        X2 = X2[pd.Index(adata2.obs_names).get_indexer(adata1.obs_names)]

    m1 = _match_names(n1, remove_from_name1)
    m2 = _match_names(n2, remove_from_name2)
    shared = set(m1) & set(m2)
    if not shared:
        raise ValueError("Matching of matrix 1 and matrix 2 resulted in no mappings!")
    f1 = np.flatnonzero(np.isin(m1, list(shared)))
    f2 = np.flatnonzero(np.isin(m2, list(shared)))
    by_name2 = pd.Series(np.arange(f2.size)).groupby(m2[f2]).apply(list).to_dict()
    pairs = [(i, j) for i in range(f1.size) for j in by_name2[m1[f1[i]]]]
    pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])
    _console(f"correlate_matrices: testing {len(pairs):,} mappings", verbose)

    rd = _reduced_dims(adata1, use_rep, dims_to_use, scale_dims, depth_key, cor_cutoff)
    ncell = rd.shape[0]
    if seed_cells is not None:
        idx = np.asarray(seed_cells, dtype=np.int64)
    elif rng.lower() in ("archr", "r"):
        kind = "L'Ecuyer-CMRG" if rng.lower() == "archr" else "Mersenne-Twister"
        idx = RRandom(seed, kind).sample_int(ncell, knn_iteration, replace=not ncell >= knn_iteration)
    elif rng.lower() == "numpy":
        g = np.random.default_rng(seed)
        idx = g.choice(ncell, knn_iteration, replace=not ncell >= knn_iteration) + 1
    else:
        raise ValueError(f"rng must be 'archr', 'R' or 'numpy', got {rng!r}")
    groups = _knn_groups(rd, idx - 1, k, overlap_cutoff)
    _console(f"correlate_matrices: {groups.shape[0]} non-overlapping kNN groups", verbose)

    G1 = _group_means(X1[:, f1] if not sparse.issparse(X1) else sparse.csc_matrix(X1)[:, f1], groups, k)
    G2 = _group_means(X2[:, f2] if not sparse.issparse(X2) else sparse.csc_matrix(X2)[:, f2], groups, k)
    if log2_norm1 and not (G1 < 0).any():
        G1 = np.log2(G1 + 1)
    if log2_norm2 and not (G2 < 0).any():
        G2 = np.log2(G2 + 1)

    res = np.array([_cor_test(G1[:, a], G2[:, b]) for a, b in zip(pi, pj)])
    cor, pval = res[:, 0], res[:, 1]
    padj = _p_adjust(pval, padj_method)

    df = pd.DataFrame({
        f"{name1}_name": n1[f1][pi], f"{name2}_name": n2[f2][pj],
        "cor": cor, "padj": padj, "pval": pval,
        f"{name1}_matchName": m1[f1][pi], f"{name2}_matchName": m2[f2][pj],
        f"{name1}_idx": f1[pi] + 1, f"{name2}_idx": f2[pj] + 1,
    })
    df.attrs["groups"] = groups
    df.attrs["seed_cells"] = idx
    return df


def _p_adjust(p: np.ndarray, method: str) -> np.ndarray:
    out = np.full_like(p, np.nan, dtype=np.float64)
    ok = ~np.isnan(p)
    n = ok.sum()
    if method == "bonferroni":
        out[ok] = np.minimum(1.0, p[ok] * n)
    elif method in ("fdr_bh", "BH"):
        pv = p[ok]
        order = np.argsort(pv)[::-1]
        ranks = np.arange(n, 0, -1)
        adj = np.minimum.accumulate(pv[order] * n / ranks)
        res = np.empty(n)
        res[order] = np.minimum(adj, 1.0)
        out[ok] = res
    else:
        raise ValueError(f"unknown padj_method {method!r}")
    return out


# --------------------------------------------------------------------------
# getGroupSE + positive TF regulators
# --------------------------------------------------------------------------
def group_means(adata, groupby: str, *, key: Optional[str] = None, layer: Optional[str] = None,
                names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Per-group mean of a cell × feature matrix (ArchR ``getGroupSE``,
    ``divideN = TRUE``). Returns features × groups."""
    X, fn = _get_feature_matrix(adata, key, layer, names)
    groups = pd.Categorical(adata.obs[groupby])
    A = sparse.csr_matrix((np.ones(len(groups)), (groups.codes, np.arange(len(groups)))),
                          shape=(len(groups.categories), len(groups)))
    sums = _dense(A @ X)
    counts = np.bincount(groups.codes, minlength=len(groups.categories))
    return pd.DataFrame((sums / counts[:, None]).T, index=fn, columns=list(groups.categories))


def positive_tf_regulators(
    gene_adata,
    motif_adata=None,
    *,
    groupby: str = "Clusters",
    motif_key: Optional[str] = "motif_deviations",
    motif_names: Optional[Sequence[str]] = None,
    gene_layer: Optional[str] = None,
    cor_threshold: float = 0.5,
    padj_threshold: float = 0.01,
    delta_quantile: float = 0.75,
    gene_label: str = "GeneScoreMatrix",
    motif_label: str = "MotifMatrix",
    key_added: Optional[str] = "positive_tf_regulators",
    verbose: bool = True,
    **correlate_kwargs,
) -> pd.DataFrame:
    r"""Identify positive TF regulators (ArchR book, "Identification of
    Positive TF-Regulators").

    1. ``maxDelta`` per motif — the largest difference of the cluster-mean
       Z-score between any two clusters (``getGroupSE`` by ``groupby``).
    2. :func:`correlate_matrices` of the gene matrix (gene scores or
       integrated expression) against the motif Z-scores.
    3. Keep the strongest |r| pair per TF (motif names collapsed at ``-``),
       call ``YES`` when ``cor > cor_threshold``, ``padj < padj_threshold``
       and ``maxDelta`` above its ``delta_quantile`` quantile.

    Arguments:
        gene_adata: AnnData with the gene matrix in ``X`` (or ``gene_layer``)
            and the embedding used for kNN groups.
        motif_adata: AnnData holding the Z-scores (default ``gene_adata``).
        groupby: ``obs`` column defining the clusters for ``maxDelta``.
        motif_key: ``obsm`` key of the Z-scores; ``None`` uses ``X``.
        motif_names: names of the motifs (default ``uns[motif_key + '_names']``).
        gene_layer: optional layer of ``gene_adata``.
        cor_threshold, padj_threshold, delta_quantile: calling thresholds
            (ArchR defaults 0.5, 0.01, 0.75).
        gene_label, motif_label: column prefixes (ArchR matrix names).
        key_added: ``uns`` key of the returned table in ``gene_adata``.
        verbose: print progress.
        **correlate_kwargs: forwarded to :func:`correlate_matrices`
            (``use_rep``, ``k``, ``seed``, ``scale_dims`` …).

    Returns:
        One row per TF with ``cor``, ``padj``, ``pval``, ``maxDelta`` and
        ``TFRegulator`` (``'YES'``/``'NO'``), ordered by decreasing |cor|.
    """
    motif_adata = gene_adata if motif_adata is None else motif_adata
    gm = group_means(motif_adata, groupby, key=motif_key, names=motif_names)
    z = gm.to_numpy()
    max_delta = pd.Series(np.max(z, axis=1) - np.min(z, axis=1), index=gm.index)

    df = correlate_matrices(
        gene_adata, motif_adata, key1=None, layer1=gene_layer, key2=motif_key,
        names2=motif_names, name1=gene_label, name2=motif_label, verbose=verbose,
        **correlate_kwargs,
    )
    df["maxDelta"] = max_delta.reindex(df[f"{motif_label}_name"]).to_numpy()
    df = df.iloc[np.argsort(-np.abs(df["cor"].to_numpy()), kind="stable")]
    tf = df[f"{motif_label}_name"].astype(str).str.replace(r"-.*", "", regex=True)
    df = df[~tf.duplicated().to_numpy()].reset_index(drop=True)
    q = np.quantile(df["maxDelta"].dropna().to_numpy(), delta_quantile)
    yes = (df["cor"] > cor_threshold) & (df["padj"] < padj_threshold) & (df["maxDelta"] > q)
    df["TFRegulator"] = np.where(yes, "YES", "NO")
    if key_added:
        gene_adata.uns[key_added] = df
    _console(f"positive_tf_regulators: {int(yes.sum())} positive regulators: "
             f"{', '.join(df.loc[yes, gene_label + '_name'].astype(str).head(20))}", verbose)
    return df
