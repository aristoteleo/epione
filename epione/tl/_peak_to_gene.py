"""
Peak-to-Gene linkage (co-accessibility between peaks and gene expression).

Reference:
    Granja, J.M., Corces, M.R., Pierce, S.E. et al. ArchR is a scalable software
    package for integrative single-cell chromatin accessibility analysis.
    Nat Genet 53, 403-411 (2021).  See `addPeak2GeneLinks`.

Idea
----
Single cells are too sparse for peak-gene correlation. Instead:

1. Cluster the embedding (e.g. `adata.obsm["X_iterative_lsi"]`) into ``n_metacells``
   anchor cells, and aggregate each anchor's ``k_neighbors`` nearest cells into
   a pseudobulk. Do this for both the peak matrix and the gene (expression or
   gene-score) matrix.
2. For every peak, find genes whose TSS sits within ±``max_distance`` bp of the
   peak center.
3. Compute Pearson's ``r`` across the ``n_metacells`` pseudobulks for each such
   peak-gene pair.
4. Convert ``r`` to a p-value via the t-transform ``t = r * sqrt((n-2)/(1-r^2))``
   and Benjamini-Hochberg FDR-correct across all tested pairs.

Output
------
A `pandas.DataFrame` with one row per evaluated peak-gene pair, columns:

    peak, gene, chrom, peak_start, peak_end, gene_start, gene_end, tss,
    distance (signed: peak_center - tss), correlation, p_value, fdr.

Also stored in ``adata.uns[key_added]``.
"""
from __future__ import annotations

from typing import Optional, Union, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ..utils import console


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_peak_names(names) -> pd.DataFrame:
    """Parse ``chr:start-end`` (or ``chr-start-end``) into a coordinates frame."""
    s = pd.Series(list(names), dtype=object)
    # Accept ':' or the first '-' as the chrom/start separator
    norm = s.str.replace(r"^([^:]+):", r"\1|", regex=True)
    parts = norm.str.split("|", n=1, expand=True)
    if parts.shape[1] == 1:  # no colon; fall back to chr-start-end
        parts = s.str.split("-", n=2, expand=True)
        chrom = parts[0]
        start = parts[1].astype(int)
        end = parts[2].astype(int)
    else:
        chrom = parts[0]
        coords = parts[1].str.split("-", n=1, expand=True)
        start = coords[0].astype(int)
        end = coords[1].astype(int)
    out = pd.DataFrame(
        {"chrom": chrom.values, "start": start.values, "end": end.values},
        index=list(names),
    )
    return out


def _resolve_gene_annotation(ann) -> pd.DataFrame:
    """Normalise the gene_annotation argument to a DataFrame with the schema
    {gene_name, chrom, start, end, strand(optional)}.

    Accepts either a DataFrame already in that shape, or a path to a GTF file
    (parsed with pyranges)."""
    if isinstance(ann, pd.DataFrame):
        out = ann.copy()
    else:
        # Path-like: parse via pyranges
        import pyranges as pr
        gr = pr.read_gtf(str(ann))
        g = gr.df
        g = g[g["Feature"] == "gene"]
        out = pd.DataFrame({
            "gene_name": g.get("gene_name", g.get("gene_id")).values,
            "chrom":     g["Chromosome"].astype(str).values,
            "start":     g["Start"].astype(int).values,
            "end":       g["End"].astype(int).values,
            "strand":    g.get("Strand", "+").values,
        })
    # Canonicalise columns
    out.columns = [c.lower() for c in out.columns]
    for col in ("gene_name", "chrom", "start", "end"):
        if col not in out.columns:
            raise KeyError(f"gene_annotation missing required column {col!r}")
    if "strand" not in out.columns:
        out["strand"] = "+"
    out["chrom"] = out["chrom"].astype(str)
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)
    # TSS = start if +, end if -
    out["tss"] = np.where(out["strand"] == "-", out["end"], out["start"])
    out = out.drop_duplicates("gene_name", keep="first").reset_index(drop=True)
    return out


def _build_metacell_indices(
    emb: np.ndarray, n_metacells: int, k_neighbors: int, seed: int,
) -> np.ndarray:
    """Pick anchor cells uniformly, then their k-nearest neighbours in `emb`.

    Returns an int array of shape (n_metacells, k_neighbors) of row indices.
    """
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    n_cells = emb.shape[0]
    n_metacells = min(n_metacells, n_cells)
    k_neighbors = min(k_neighbors, n_cells)

    anchors = rng.choice(n_cells, size=n_metacells, replace=False)
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine").fit(emb)
    _, idx = nn.kneighbors(emb[anchors], return_distance=True)
    return idx.astype(np.int64)


def _aggregate(X, metacell_idx) -> np.ndarray:
    """Mean-aggregate X across the k neighbours of each metacell.

    Output shape: (n_metacells, n_features), dtype float32.
    """
    n_mc, k = metacell_idx.shape
    n_features = X.shape[1]
    out = np.zeros((n_mc, n_features), dtype=np.float32)

    if sp.issparse(X):
        X_csr = X.tocsr()
        for i in range(n_mc):
            sub = X_csr[metacell_idx[i]]
            out[i] = np.asarray(sub.mean(axis=0), dtype=np.float32).ravel()
    else:
        X_dense = np.asarray(X, dtype=np.float32)
        for i in range(n_mc):
            out[i] = X_dense[metacell_idx[i]].mean(axis=0)
    return out


def _zscore_columns(M: np.ndarray) -> np.ndarray:
    out = M - M.mean(axis=0, keepdims=True)
    sd = out.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return out / sd


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR-adjusted q-values."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.clip(q, 0, 1)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def peak_to_gene(
    adata: AnnData,
    *,
    rna: Optional[Union[AnnData, np.ndarray]] = None,
    gene_layer: Optional[str] = None,
    gene_obsm: Optional[str] = None,
    gene_names: Optional[list] = None,
    gene_annotation: Union[pd.DataFrame, str, None] = None,
    use_rep: str = "X_iterative_lsi",
    n_metacells: int = 500,
    k_neighbors: int = 100,
    max_distance: int = 250_000,
    min_correlation: float = 0.0,
    alpha: float = 0.05,
    cor_method: Literal["pearson"] = "pearson",
    seed: int = 0,
    key_added: str = "peak_to_gene",
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """ArchR-style peak-to-gene linkage via kNN-metacell correlation.

    Parameters
    ----------
    adata
        Cells x peaks AnnData. ``adata.var_names`` must be interval-style
        (``chr:start-end``) and ``adata.obsm[use_rep]`` must hold an embedding
        (typically the output of :func:`epione.tl.iterative_lsi`).
    rna
        Per-cell gene expression. May be ``None`` – in that case
        ``adata.obsm[gene_obsm]`` is used as the gene matrix, and ``gene_names``
        supplies the gene-id axis. When an ``AnnData``, its ``obs_names`` must
        match ``adata`` exactly and ``var_names`` are the gene ids.
    gene_layer
        If set and ``rna`` is an ``AnnData``, read ``rna.layers[gene_layer]``
        instead of ``rna.X``.
    gene_obsm, gene_names
        Fallback pair when RNA lives inside ``adata`` (e.g. a gene-score matrix
        stored in ``adata.obsm``).
    gene_annotation
        DataFrame with columns ``gene_name, chrom, start, end[, strand]`` or a
        path to a GTF. Strand defaults to ``+`` if absent.
    use_rep
        Embedding key for metacell construction. Defaults to the iterative-LSI
        output.
    n_metacells
        Number of anchor cells to form pseudobulks.
    k_neighbors
        Neighbours per metacell.
    max_distance
        Maximum peak-centre ↔ gene TSS distance (bp).
    min_correlation
        Filter out peak-gene pairs with |r| below this value after fitting.
    alpha
        FDR threshold retained in the summary (all pairs are still returned).
    seed
        RNG seed for metacell anchor sampling.
    key_added
        Stored at ``adata.uns[key_added]``.
    inplace
        If False, do not write to ``adata.uns``.
    verbose
        Print progress.

    Returns
    -------
    pandas.DataFrame with columns:
        ``peak, gene, chrom, peak_start, peak_end, gene_start, gene_end, tss,
        distance, correlation, t, p_value, fdr``.
    """
    if cor_method != "pearson":
        raise NotImplementedError("only cor_method='pearson' is implemented")

    # -- 1. Resolve embedding ------------------------------------------------
    if use_rep not in adata.obsm:
        raise KeyError(
            f"adata.obsm[{use_rep!r}] not found – run iterative_lsi first."
        )
    emb = np.asarray(adata.obsm[use_rep])

    # -- 2. Resolve gene matrix ---------------------------------------------
    if rna is not None:
        if isinstance(rna, AnnData):
            if list(rna.obs_names) != list(adata.obs_names):
                # Align RNA to adata.obs_names by intersection
                common = adata.obs_names.intersection(rna.obs_names)
                if len(common) < 0.5 * adata.n_obs:
                    raise ValueError(
                        "adata and rna share fewer than half their obs_names."
                    )
                rna = rna[common]
                adata = adata[common]
                emb = np.asarray(adata.obsm[use_rep])
            gene_X = rna.layers[gene_layer] if gene_layer else rna.X
            gene_names_arr = np.asarray(rna.var_names)
        else:
            gene_X = np.asarray(rna)
            if gene_names is None:
                raise ValueError("`gene_names` required when `rna` is an ndarray")
            gene_names_arr = np.asarray(gene_names)
    elif gene_obsm is not None:
        if gene_obsm not in adata.obsm:
            raise KeyError(f"adata.obsm[{gene_obsm!r}] not found")
        gene_X = adata.obsm[gene_obsm]
        if gene_names is None:
            raise ValueError("`gene_names` required when using `gene_obsm`")
        gene_names_arr = np.asarray(gene_names)
    else:
        raise ValueError(
            "Must pass either `rna=` or `gene_obsm=` + `gene_names=`."
        )

    if gene_X.shape[0] != adata.n_obs:
        raise ValueError(
            f"gene matrix has {gene_X.shape[0]} rows, expected {adata.n_obs}"
        )
    if gene_X.shape[1] != len(gene_names_arr):
        raise ValueError(
            f"gene matrix has {gene_X.shape[1]} cols but gene_names has "
            f"{len(gene_names_arr)}"
        )

    # -- 3. Peak + gene coordinates -----------------------------------------
    peak_df = _parse_peak_names(adata.var_names)
    peak_df["center"] = ((peak_df["start"] + peak_df["end"]) // 2).astype(int)
    peak_df["peak_idx"] = np.arange(len(peak_df))

    if gene_annotation is None:
        raise ValueError("`gene_annotation` is required.")
    gene_df = _resolve_gene_annotation(gene_annotation)
    # Restrict to genes present in the gene matrix
    gene_df = gene_df[gene_df["gene_name"].isin(gene_names_arr)].reset_index(drop=True)
    gene_df["gene_idx"] = gene_df["gene_name"].map(
        {g: i for i, g in enumerate(gene_names_arr)}
    ).astype(int)

    if verbose:
        console.info(
            f"[peak_to_gene] {len(peak_df):,} peaks | {len(gene_df):,} annotated genes | "
            f"{adata.n_obs:,} cells"
        )

    # -- 4. Build metacells + aggregated matrices ---------------------------
    if verbose:
        console.info(
            f"[peak_to_gene] Building {n_metacells} metacells × "
            f"{k_neighbors} neighbours from {use_rep}"
        )
    mc_idx = _build_metacell_indices(emb, n_metacells, k_neighbors, seed)
    n_mc = mc_idx.shape[0]

    if verbose:
        console.info("[peak_to_gene] Aggregating peak matrix")
    peak_mc = _aggregate(adata.X, mc_idx)           # (n_mc, n_peaks)
    if verbose:
        console.info("[peak_to_gene] Aggregating gene matrix")
    gene_mc = _aggregate(gene_X, mc_idx)            # (n_mc, n_genes)

    # -- 5. Build peak-gene pair list by proximity --------------------------
    if verbose:
        console.info(
            f"[peak_to_gene] Finding peak-gene pairs within ±{max_distance:,} bp"
        )
    # Group peaks and genes by chromosome, use sorted-search for the distance filter.
    peak_by_chrom = {c: d for c, d in peak_df.groupby("chrom", sort=False)}
    gene_by_chrom = {c: d for c, d in gene_df.groupby("chrom", sort=False)}

    peak_idx_list = []
    gene_idx_list = []
    distances_list = []
    tss_list = []

    for chrom, p in peak_by_chrom.items():
        g = gene_by_chrom.get(chrom)
        if g is None or len(g) == 0:
            continue
        # Sort genes by TSS, find peaks whose centre is within ±max_distance
        g_sorted = g.sort_values("tss").reset_index(drop=True)
        tss_sorted = g_sorted["tss"].to_numpy()
        g_idx_sorted = g_sorted["gene_idx"].to_numpy()

        centers = p["center"].to_numpy()
        pk_idx = p["peak_idx"].to_numpy()

        lo = np.searchsorted(tss_sorted, centers - max_distance, side="left")
        hi = np.searchsorted(tss_sorted, centers + max_distance, side="right")
        for i, (l, h) in enumerate(zip(lo, hi)):
            if l == h:
                continue
            n_hits = h - l
            peak_idx_list.append(np.full(n_hits, pk_idx[i], dtype=np.int64))
            gene_idx_list.append(g_idx_sorted[l:h].astype(np.int64))
            tss_hits = tss_sorted[l:h]
            tss_list.append(tss_hits)
            distances_list.append(centers[i] - tss_hits)

    if not peak_idx_list:
        if verbose:
            console.warn("[peak_to_gene] No peak-gene pairs within max_distance.")
        df = pd.DataFrame(columns=[
            "peak", "gene", "chrom", "peak_start", "peak_end", "gene_start",
            "gene_end", "tss", "distance", "correlation", "t", "p_value", "fdr",
        ])
        if inplace:
            adata.uns[key_added] = df
        return df

    peak_idx_arr = np.concatenate(peak_idx_list)
    gene_idx_arr = np.concatenate(gene_idx_list)
    tss_arr      = np.concatenate(tss_list)
    dist_arr     = np.concatenate(distances_list)

    if verbose:
        console.info(f"[peak_to_gene] {len(peak_idx_arr):,} candidate pairs")

    # -- 6. Correlation across metacells ------------------------------------
    if verbose:
        console.info("[peak_to_gene] Computing correlations")
    Pz = _zscore_columns(peak_mc)           # (n_mc, n_peaks)
    Gz = _zscore_columns(gene_mc)           # (n_mc, n_genes)

    # For each pair, r = (Pz[:, p] . Gz[:, g]) / (M-1).
    # Chunk along the pair axis to avoid holding the full
    # (n_mc, n_pairs) intermediate in RAM.
    n_pairs = len(peak_idx_arr)
    # Target ~256 MB float32 per chunk: chunk_size * n_mc * 4 <= 256 MB
    max_chunk = max(1, int(256 * 2**20 / max(n_mc, 1) / 4))
    chunk_size = min(n_pairs, max(max_chunk, 100_000))
    r = np.empty(n_pairs, dtype=np.float32)
    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        pcols = Pz[:, peak_idx_arr[start:end]]
        gcols = Gz[:, gene_idx_arr[start:end]]
        r_chunk = (pcols * gcols).sum(axis=0) / (n_mc - 1)
        r[start:end] = r_chunk.astype(np.float32)
    r = np.clip(r, -1.0, 1.0)

    df_deg = n_mc - 2
    r2 = np.clip(1 - r.astype(np.float64) ** 2, 1e-12, None)
    t = r * np.sqrt(df_deg / r2)
    import scipy.stats as ss
    p = 2 * ss.t.sf(np.abs(t), df_deg)

    fdr = _bh_fdr(p)

    # -- 7. Assemble output -------------------------------------------------
    peak_names_arr = np.asarray(adata.var_names)
    out = pd.DataFrame({
        "peak":       peak_names_arr[peak_idx_arr],
        "gene":       gene_names_arr[gene_idx_arr],
        "chrom":      peak_df["chrom"].to_numpy()[peak_idx_arr],
        "peak_start": peak_df["start"].to_numpy()[peak_idx_arr],
        "peak_end":   peak_df["end"].to_numpy()[peak_idx_arr],
        "tss":        tss_arr,
        "distance":   dist_arr,
        "correlation": r.astype(np.float32),
        "t":          t.astype(np.float32),
        "p_value":    p.astype(np.float32),
        "fdr":        fdr.astype(np.float32),
    })
    # Attach gene body coords
    gmap = gene_df.set_index("gene_name")[["start", "end"]]
    out = out.join(gmap.rename(columns={"start": "gene_start", "end": "gene_end"}),
                   on="gene")
    out = out[[
        "peak", "gene", "chrom", "peak_start", "peak_end",
        "gene_start", "gene_end", "tss", "distance",
        "correlation", "t", "p_value", "fdr",
    ]]

    if min_correlation > 0:
        out = out[out["correlation"].abs() >= min_correlation].reset_index(drop=True)

    if verbose:
        sig = (out["fdr"] < alpha).sum()
        console.info(
            f"[peak_to_gene] {len(out):,} pairs retained, "
            f"{sig:,} significant (FDR < {alpha})"
        )

    if inplace:
        adata.uns[key_added] = out
        adata.uns[key_added + "_params"] = {
            "use_rep": use_rep,
            "n_metacells": int(n_metacells),
            "k_neighbors": int(k_neighbors),
            "max_distance": int(max_distance),
            "min_correlation": float(min_correlation),
            "alpha": float(alpha),
            "seed": int(seed),
        }

    return out
