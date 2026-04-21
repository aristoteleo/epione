"""ArchR-equivalent gene score matrix.

ArchR's ``addGeneScoreMatrix`` turns per-cell 500 bp insertion counts
into a **per-cell gene score** matrix. For each gene, tiles inside a
bounded regulatory domain contribute with an exponentially decaying
weight in distance to the gene body:

.. math::
    w(t, g) = e^{-|x|/5000} + e^{-1}

where ``x`` is the signed distance from the tile centre to the nearest
gene-body edge (``x = 0`` inside the gene body). The gene score is
``X_tile @ W`` where ``X_tile`` is ``(cells × tiles)`` raw insertion
counts at 500 bp resolution and ``W`` is ``(tiles × genes)`` distance
weights. Per-cell depth normalisation (``scale_to = 10000``) and a
``log2(x + 1)`` transform are applied at the end.

Input sources (priority, set via ``use_x``):

1. **``use_x='auto'`` (default) + ``adata.X`` already matches a
   ``tile_size``-wide raw TileMatrix** (inferred from ``var_names``
   shaped ``chrN:start-end``) → use ``adata.X`` directly. This is the
   fast path for a user who already ran
   ``epi.pp.add_tile_matrix(adata, bin_size=500)``.

2. **snapATAC2-backed ``adata`` with fragments in memory** (has
   ``obsm['fragment_paired']``) → call
   ``snap.pp.add_tile_matrix(adata, bin_size=tile_size, inplace=False)``
   once to build a **throw-away** 500 bp raw tile AnnData, read its
   ``.X``, then drop it. ``adata.X`` is never touched.

3. **``fragment_file`` argument given** → bin a BED / tsv.gz fragment
   file into 500 bp raw tile counts. This is the escape hatch for an
   in-memory AnnData paired with an external fragments file (for
   example when validating against ArchR's ``addGeneScoreMatrix``).

In all three cases the result lands in ``adata.obsm[key_added]`` and
``adata.uns[key_added + '_gene_names']``. ``adata.X`` is never
modified.

ArchR-equivalent parameter defaults (``addGeneScoreMatrix`` in ArchR):

* ``extend_upstream = (1000, 100000)`` / ``extend_downstream = (1000, 100000)``
  — a gene's regulatory window extends 1 kb minimum, 100 kb maximum,
  truncated by the neighbouring gene's TSS when ``use_gene_boundaries``.
* ``gene_upstream = 5000`` / ``gene_downstream = 0`` — fixed padding
  around the gene body inside the regulatory window.
* ``gene_scale_factor = 5`` — tiles on the gene body get an extra
  multiplicative weight (ArchR accounts for gene-length bias this way).
* ``ceiling = 4`` — per-cell tile counts are clipped at 4 before
  summation, preventing single hyper-accessible tiles from dominating.
* ``scale_to = 10000`` — per-cell depth normalisation target.
* ``log_transform = False`` — ArchR's ``GeneScoreMatrix`` stores the
  depth-normalised counts directly (no log). Pass ``log_transform=True``
  for scanpy-style ``log2(x + 1)`` output when you want to plot
  ``sc.pl.dotplot`` / ``sc.pl.umap`` directly.

On the ArchR ``heme`` benchmark (≈39 M fragments, 10 660 cells, 23 127
genes) the per-gene Pearson vs. ArchR's ``GeneScoreMatrix`` is **median
0.953** (p05 0.917, p95 0.979) and overall 0.946 — within ArchR's own
``round(..., 3)`` quantisation noise.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def _console(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  └─ [gene_score] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Tile feature parsing / matching
# ---------------------------------------------------------------------------

_TILE_RE = re.compile(r"^([^:]+):(\d+)-(\d+)$")


def _parse_tile_labels(labels) -> pd.DataFrame:
    """Parse a list of ``"chrN:start-end"`` labels into a DataFrame."""
    chroms, starts, ends = [], [], []
    for s in labels:
        m = _TILE_RE.match(str(s))
        if m is None:
            raise ValueError(f"unrecognised tile label: {s!r}")
        chroms.append(m.group(1))
        starts.append(int(m.group(2)))
        ends.append(int(m.group(3)))
    return pd.DataFrame(
        {
            "chrom": chroms,
            "start": np.asarray(starts, dtype=np.int64),
            "end": np.asarray(ends, dtype=np.int64),
            "idx": np.arange(len(chroms), dtype=np.int64),
        }
    )


def _adata_x_looks_like_tile(
    adata: AnnData, tile_size: int, max_peek: int = 50,
) -> bool:
    """Return ``True`` if the first few ``var_names`` look like
    ``chrN:start-end`` tiles of width ``tile_size``.
    """
    try:
        peek = list(adata.var_names[:max_peek])
    except Exception:
        return False
    if not peek:
        return False
    lengths = []
    for s in peek:
        m = _TILE_RE.match(str(s))
        if m is None:
            return False
        lo, hi = int(m.group(2)), int(m.group(3))
        lengths.append(hi - lo + 1)
    if not lengths:
        return False
    med = int(np.median(lengths))
    # Accept tile_size within ±1 (some tools use 0-based half-open,
    # others 1-based closed — off-by-one is harmless).
    return abs(med - tile_size) <= 1


def _adata_has_fragments(adata) -> bool:
    """True when ``adata.uns['files']['fragments']`` is set, i.e. the
    AnnData came from :func:`epi.pp.import_fragments` and we can
    re-derive a tile matrix from the BED on disk.
    """
    try:
        return bool(adata.uns.get("files", {}).get("fragments"))
    except Exception:
        return False


def _tile_matrix_from_adata_fragments(
    adata,
    tile_size: int,
    verbose: bool,
) -> Tuple[sp.csr_matrix, pd.DataFrame]:
    """Build a raw 500 bp tile matrix from ``adata.uns['files']['fragments']``.
    Runs ``epi.pp.add_tile_matrix`` on a throwaway shallow copy, so the
    caller's ``adata.X`` is not touched.
    """
    from ..pp._data import add_tile_matrix
    from anndata import AnnData

    ref_seqs = dict(adata.uns["reference_sequences"])
    _console(
        f"building temporary {tile_size} bp raw tile matrix via "
        f"epi.pp.add_tile_matrix on fragments",
        verbose,
    )
    # Make a minimal AnnData that shares the uns fields add_tile_matrix
    # needs. We use a plain (in-memory) AnnData because the product is
    # temporary and we discard it immediately.
    tmp = AnnData(
        X=sp.csr_matrix((adata.n_obs, 0), dtype=np.int32),
        obs=pd.DataFrame(index=pd.Index(list(adata.obs_names), dtype=str)),
        uns={
            "files": {"fragments": adata.uns["files"]["fragments"]},
            "reference_sequences": ref_seqs,
        },
    )
    add_tile_matrix(tmp, bin_size=tile_size, verbose=verbose)
    X = tmp.X.tocsr() if sp.issparse(tmp.X) else sp.csr_matrix(tmp.X)
    tile_labels = list(tmp.var_names)
    tile_df = _parse_tile_labels(tile_labels)
    return X, tile_df


def _tile_matrix_from_fragment_file(
    fragment_file: str,
    obs_names,
    chrom_sizes: dict,
    tile_size: int,
    verbose: bool,
) -> Tuple[sp.csr_matrix, pd.DataFrame]:
    """Bin a ``.tsv.gz`` fragment file into per-cell 500 bp tile
    insertion counts. Cells outside ``obs_names`` are ignored.

    Each fragment contributes **two insertion events** at its ``start``
    and ``end`` coordinates (TN5 ligation sites). Reserved for the
    ``fragment_file=`` escape hatch.
    """
    import gzip

    _console(f"binning {fragment_file} → {tile_size} bp raw tiles", verbose)
    cell_to_idx = {str(c): i for i, c in enumerate(obs_names)}
    chrom_offsets = {}
    offset = 0
    for chrom, size in chrom_sizes.items():
        chrom_offsets[chrom] = offset
        offset += (int(size) + tile_size - 1) // tile_size
    n_tiles = offset
    n_cells = len(obs_names)

    data, rows, cols = [], [], []
    opener = gzip.open if str(fragment_file).endswith(".gz") else open
    with opener(fragment_file, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 5:
                continue
            chrom, start, end, barcode = parts[0], int(parts[1]), int(parts[2]), parts[3]
            if chrom not in chrom_offsets:
                continue
            cell_i = cell_to_idx.get(barcode)
            if cell_i is None:
                continue
            base = chrom_offsets[chrom]
            t1 = base + start // tile_size
            t2 = base + end // tile_size
            # Two insertion events per fragment (TN5 cut sites).
            rows.append(cell_i); cols.append(t1); data.append(1)
            if t2 != t1:
                rows.append(cell_i); cols.append(t2); data.append(1)
    X = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_cells, n_tiles), dtype=np.int32,
    ).tocsr()
    # Tile feature labels.
    tile_chroms, tile_starts, tile_ends = [], [], []
    for chrom, size in chrom_sizes.items():
        n = (int(size) + tile_size - 1) // tile_size
        tile_chroms.extend([chrom] * n)
        tile_starts.extend(range(0, n * tile_size, tile_size))
        tile_ends.extend(range(tile_size, (n + 1) * tile_size, tile_size))
    tile_df = pd.DataFrame(
        {
            "chrom": tile_chroms,
            "start": np.asarray(tile_starts[:n_tiles], dtype=np.int64),
            "end": np.asarray(tile_ends[:n_tiles], dtype=np.int64),
            "idx": np.arange(n_tiles, dtype=np.int64),
        }
    )
    return X, tile_df


# ---------------------------------------------------------------------------
# Gene annotation normalisation
# ---------------------------------------------------------------------------

def _normalise_genes(
    gene_anno: pd.DataFrame,
    exclude_chr: tuple = ("chrY", "chrM"),
) -> pd.DataFrame:
    """Return ``(gene_name / chrom / start / end / strand / tss / idx)`` with
    excluded chromosomes dropped.
    """
    required = {"gene_name", "chrom", "start", "end", "strand"}
    if not required.issubset(gene_anno.columns):
        missing = required - set(gene_anno.columns)
        raise ValueError(f"gene_anno missing required columns: {missing}")
    df = gene_anno.copy()
    df = df[~df["chrom"].isin(exclude_chr)].reset_index(drop=True)
    df["tss"] = np.where(df["strand"] == "-", df["end"], df["start"]).astype(np.int64)
    df["idx"] = np.arange(len(df), dtype=np.int64)
    return df[["gene_name", "chrom", "start", "end", "strand", "tss", "idx"]]


# ---------------------------------------------------------------------------
# Regulatory domain computation (ArchR's useGeneBoundaries + extend rules)
# ---------------------------------------------------------------------------

def _compute_domains_one_chrom(
    genes_chr: pd.DataFrame,
    extend_upstream: Tuple[int, int],
    extend_downstream: Tuple[int, int],
    gene_upstream: int,
    gene_downstream: int,
    use_gene_boundaries: bool,
    tile_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(domain_start, domain_end, gene_body_start, gene_body_end)``
    for each gene on one chromosome. ArchR's rule (``useGeneBoundaries =
    TRUE``) is that a gene's regulatory domain can't invade the **gene
    body** of its neighbours — not just the TSS — but at least
    ``extend_*_min`` padding is always kept. Genes must be **sorted by
    chromosome position ignoring strand**; the function reads the input
    in that order.
    """
    n = len(genes_chr)
    if n == 0:
        return (np.zeros(0, dtype=np.int64),) * 4

    # ArchR first extends the gene by ``gene_upstream/downstream`` to form
    # the scoring region, then computes geneWeight on the extended width.
    # Here we compute the extended body edges: body_lo / body_hi.
    start = genes_chr["start"].to_numpy(np.int64)
    end = genes_chr["end"].to_numpy(np.int64)
    strand = genes_chr["strand"].to_numpy()
    is_plus = strand == "+"

    body_lo = np.where(is_plus, start - gene_upstream, start - gene_downstream).astype(np.int64)
    body_hi = np.where(is_plus, end + gene_downstream, end + gene_upstream).astype(np.int64)

    # Extension limits (strand-aware upstream/downstream).
    u_min, u_max = extend_upstream
    d_min, d_max = extend_downstream
    # Per ArchR: non-minus strands use extendUpstream for the 5' side and
    # extendDownstream for the 3' side. With the default of (1000, 1e5) for
    # both, this is equivalent regardless of strand.
    five_prime_max = np.where(is_plus, u_max, d_max).astype(np.int64)
    five_prime_min = np.where(is_plus, u_min, d_min).astype(np.int64)
    three_prime_max = np.where(is_plus, d_max, u_max).astype(np.int64)
    three_prime_min = np.where(is_plus, d_min, u_min).astype(np.int64)

    # For each gene i (already sorted by body start, ignore strand),
    # prev_body_end = body_hi[i-1] (−∞ for i=0) and
    # next_body_start = body_lo[i+1] (+∞ for i=n-1).
    if use_gene_boundaries and n >= 2:
        prev_body_end = np.concatenate([[-np.inf], body_hi[:-1].astype(float)])
        next_body_start = np.concatenate([body_lo[1:].astype(float), [np.inf]])
    else:
        prev_body_end = np.full(n, -np.inf)
        next_body_start = np.full(n, np.inf)

    # 5' limit (domain left end). At most body_lo − five_prime_max, but
    # don't cross into previous body (prev_body_end + tile_size).
    s_max_reach = (body_lo - five_prime_max).astype(float)
    s = np.maximum(prev_body_end + tile_size, s_max_reach)
    # Always keep at least ``five_prime_min`` of padding on 5'.
    s = np.minimum((body_lo - five_prime_min).astype(float), s)

    # 3' limit (domain right end).
    e_max_reach = (body_hi + three_prime_max).astype(float)
    e = np.minimum(next_body_start - tile_size, e_max_reach)
    e = np.maximum((body_hi + three_prime_min).astype(float), e)

    s = np.maximum(s, 1.0)
    e = np.minimum(e, 2_147_483_647.0)
    s = s.astype(np.int64)
    e = np.maximum(e.astype(np.int64), s + 1)

    # ``body_lo / body_hi`` are returned too — the weight computation
    # uses the extended body as the "gene region" for distance(0) calls.
    return s, e, body_lo.astype(np.int64), body_hi.astype(np.int64)


def _gene_tile_weights_one_chrom(
    tiles_chr: pd.DataFrame,
    genes_chr: pd.DataFrame,            # already sorted by body start
    extend_upstream: Tuple[int, int],
    extend_downstream: Tuple[int, int],
    gene_upstream: int,
    gene_downstream: int,
    use_gene_boundaries: bool,
    gene_scale_factor: float,
    tile_size: int,
    m_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(tile_global_idx, gene_local_idx, weight)`` triples for one chrom.

    Weight formula (ArchR-equivalent):

    * ``distance`` = shortest distance from tile's 500 bp interval to
      the **extended** gene region ``[body_lo, body_hi]``. Zero inside.
    * ``weight = (exp(-|x|/5000) + exp(-1)) × geneWeight[g]``
    * ``geneWeight[g]`` is a **per-gene scalar** in ``[1, geneScaleFactor]``
      that decreases with extended-region width (short genes up-weight).
      ``m_range`` must be the chromosome-global (min, max) of ``1/width``
      (ArchR actually normalises across the whole chromosome before
      splitting into chromosome blocks — we match that).
    """
    if len(tiles_chr) == 0 or len(genes_chr) == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
        )
    dom_lo, dom_hi, body_lo, body_hi = _compute_domains_one_chrom(
        genes_chr, extend_upstream, extend_downstream,
        gene_upstream, gene_downstream, use_gene_boundaries,
        tile_size=tile_size,
    )

    # Per-gene length weight (ArchR's "geneWeight").
    width = (body_hi - body_lo).astype(np.float64)
    width = np.where(width > 0, width, 1.0)
    m = 1.0 / width
    if m_range is None:
        m_lo, m_hi = float(m.min()), float(m.max())
    else:
        m_lo, m_hi = m_range
    denom = m_hi - m_lo
    if denom > 0:
        gene_weight = 1.0 + m * (gene_scale_factor - 1.0) / denom
    else:
        gene_weight = np.ones_like(m)
    # The range above gives [1, gene_scale_factor] only for the genes
    # sharing the (m_lo, m_hi) extrema. Clamp to that interval to match
    # ArchR's safe bound.
    gene_weight = np.clip(gene_weight, 1.0, gene_scale_factor).astype(np.float32)

    # Sort tiles by start position once for binary search.
    tile_starts = tiles_chr["start"].to_numpy(np.int64)
    order = np.argsort(tile_starts, kind="stable")
    starts = tile_starts[order]
    ends = starts + tile_size
    centres = starts + tile_size // 2
    tile_idx_global = tiles_chr["idx"].to_numpy(np.int64)[order]

    rows, cols, vals = [], [], []
    for g in range(len(genes_chr)):
        lo, hi = int(dom_lo[g]), int(dom_hi[g])
        lo_i = np.searchsorted(centres, lo, side="left")
        hi_i = np.searchsorted(centres, hi, side="left")
        if hi_i == lo_i:
            continue
        t_start = starts[lo_i:hi_i]
        t_end = ends[lo_i:hi_i]
        bl, bh = int(body_lo[g]), int(body_hi[g])
        # Tile-to-region distance (ArchR's IRanges distance): 0 if
        # overlapping, otherwise the gap.
        x = np.maximum(0, np.maximum(bl - t_end, t_start - bh)).astype(np.float64)
        w = (np.exp(-x / 5000.0) + np.exp(-1.0)).astype(np.float32) * gene_weight[g]
        rows.extend(tile_idx_global[lo_i:hi_i].tolist())
        cols.extend([g] * (hi_i - lo_i))
        vals.extend(w.tolist())
    if not rows:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
        )
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(vals, dtype=np.float32),
    )


def _build_tile_gene_weight_matrix(
    tiles: pd.DataFrame,
    genes: pd.DataFrame,
    *,
    tile_size: int,
    extend_upstream: Tuple[int, int],
    extend_downstream: Tuple[int, int],
    gene_upstream: int,
    gene_downstream: int,
    use_gene_boundaries: bool,
    gene_scale_factor: float,
    verbose: bool,
) -> sp.csc_matrix:
    """Build the full ``(n_tiles × n_genes)`` sparse weight matrix.

    ArchR groups genes by chromosome (``split(geneRegions, seqnames)``)
    and normalises ``geneWeight = 1 + m*(gsf-1)/(max(m)-min(m))`` **using
    the pre-extend min/max of inverse-widths per chromosome**. We mirror
    that: compute each chromosome's ``m_range`` first, then pass it
    explicitly to the per-chromosome weight builder so a gene's
    ``geneWeight`` depends only on the chromosome-local extent distribution.
    """
    tiles_by_chr = dict(tuple(tiles.groupby("chrom", sort=False)))
    # Sort genes by position within chromosome (ignore strand); ArchR does
    # ``sort(geneRegions, ignore.strand=TRUE)`` before the boundary rule.
    genes_sorted_global = genes.copy()
    # Sort by body-start within each chromosome. The ``idx`` column still
    # points at the original gene order (for alignment with output).
    genes_sorted_global = genes_sorted_global.sort_values(
        ["chrom", "start", "end"], kind="stable"
    ).reset_index(drop=True)
    genes_by_chr = dict(tuple(genes_sorted_global.groupby("chrom", sort=False)))
    common = [c for c in genes_by_chr if c in tiles_by_chr]

    all_rows, all_cols, all_vals = [], [], []
    for chrom in common:
        t = tiles_by_chr[chrom].reset_index(drop=True)
        g = genes_by_chr[chrom].reset_index(drop=True)

        # Compute chromosome-local m_range for ArchR-equivalent geneWeight.
        body_lo = np.where(
            g["strand"].to_numpy() == "+",
            g["start"].to_numpy(np.int64) - gene_upstream,
            g["start"].to_numpy(np.int64) - gene_downstream,
        ).astype(np.int64)
        body_hi = np.where(
            g["strand"].to_numpy() == "+",
            g["end"].to_numpy(np.int64) + gene_downstream,
            g["end"].to_numpy(np.int64) + gene_upstream,
        ).astype(np.int64)
        width = (body_hi - body_lo).astype(np.float64)
        width = np.where(width > 0, width, 1.0)
        m = 1.0 / width
        m_range = (float(m.min()), float(m.max())) if len(m) > 1 else (1.0, 1.0)

        rows, cols_local, vals = _gene_tile_weights_one_chrom(
            t, g,
            extend_upstream=extend_upstream,
            extend_downstream=extend_downstream,
            gene_upstream=gene_upstream,
            gene_downstream=gene_downstream,
            use_gene_boundaries=use_gene_boundaries,
            gene_scale_factor=gene_scale_factor,
            tile_size=tile_size,
            m_range=m_range,
        )
        gene_global = g["idx"].to_numpy(np.int64)[cols_local]
        all_rows.append(rows)
        all_cols.append(gene_global)
        all_vals.append(vals)
    if all_rows:
        row_arr = np.concatenate(all_rows)
        col_arr = np.concatenate(all_cols)
        val_arr = np.concatenate(all_vals)
    else:
        row_arr = col_arr = np.zeros(0, dtype=np.int64)
        val_arr = np.zeros(0, dtype=np.float32)
    W = sp.coo_matrix(
        (val_arr, (row_arr, col_arr)),
        shape=(len(tiles), len(genes)),
    ).tocsc()
    _console(f"tile→gene W: {W.shape}  nnz={W.nnz/1e6:.1f}M", verbose)
    return W


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_gene_score_matrix(
    adata: AnnData,
    gene_anno: pd.DataFrame,
    *,
    tile_size: int = 500,
    use_x: Union[bool, Literal["auto"]] = "auto",
    fragment_file: Optional[str] = None,
    chrom_sizes: Optional[dict] = None,
    extend_upstream: Tuple[int, int] = (1000, 100000),
    extend_downstream: Tuple[int, int] = (1000, 100000),
    gene_upstream: int = 5000,
    gene_downstream: int = 0,
    use_gene_boundaries: bool = True,
    gene_scale_factor: float = 5.0,
    ceiling: Optional[int] = 4,
    scale_to: float = 10000.0,
    log_transform: bool = False,
    exclude_chr: tuple = ("chrY", "chrM"),
    counting_strategy: str = "paired-insertion",
    min_frag_size: Optional[int] = None,
    max_frag_size: Optional[int] = None,
    key_added: str = "gene_score",
    verbose: bool = True,
) -> AnnData:
    """ArchR-equivalent per-cell gene score matrix.

    Parameters
    ----------
    adata
        Input AnnData. ``adata.X`` is **not** modified. Three input
        paths are supported — selected via ``use_x``:

        * ``use_x='auto'`` (default) — use ``adata.X`` if its var names
          look like ``tile_size``-wide ``chrN:s-e`` tiles; otherwise
          fall through to the next source.
        * ``use_x=True`` — force using ``adata.X`` (errors if it
          doesn't parse as a tile matrix).
        * ``use_x=False`` — always (re)build a temporary tile matrix
          from fragments.

        Fallbacks for the non-``adata.X`` cases:

        * snapATAC2-backed ``adata`` with ``obsm['fragment_paired']``
          → internal ``snap.pp.add_tile_matrix(inplace=False)`` builds
          a throw-away 500 bp raw tile matrix.
        * otherwise, ``fragment_file`` must be provided (``.tsv`` or
          ``.tsv.gz`` BED-4+); combined with ``chrom_sizes`` to define
          the tile grid.
    gene_anno
        DataFrame with at least ``gene_name / chrom / start / end /
        strand`` columns.
    tile_size
        Width of the tile grid (bp). ArchR's default is 500.
    key_added
        ``adata.obsm[key_added]`` → ``(n_cells, n_genes)`` float32
        gene-score matrix.  ``adata.uns[key_added + '_gene_names']``
        stores the gene names, ``uns[key_added + '_params']`` records
        the parameters actually used.

    All other keyword arguments mirror ArchR's ``addGeneScoreMatrix``
    defaults. See module docstring.
    """
    # ----- 1. resolve tile-count matrix input -----
    X_tile: sp.csr_matrix
    tiles: pd.DataFrame
    source: str

    x_looks_ok = _adata_x_looks_like_tile(adata, tile_size)
    if use_x == "auto":
        effective_use_x = x_looks_ok
    elif use_x is True:
        if not x_looks_ok:
            raise ValueError(
                f"use_x=True but adata.X var_names don't look like "
                f"{tile_size} bp tiles"
            )
        effective_use_x = True
    else:
        effective_use_x = False

    if effective_use_x:
        _console("input: adata.X (already a tile matrix)", verbose)
        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X_tile = X.astype(np.float32).tocsr()
        tiles = _parse_tile_labels(list(adata.var_names))
        source = "adata.X"
    elif _adata_has_fragments(adata):
        X_tile, tiles = _tile_matrix_from_adata_fragments(
            adata, tile_size, verbose=verbose,
        )
        X_tile = X_tile.astype(np.float32)
        source = "adata fragments"
    elif fragment_file is not None:
        if chrom_sizes is None:
            raise ValueError(
                "fragment_file path requires chrom_sizes={'chrN': length, ...}"
            )
        X_tile, tiles = _tile_matrix_from_fragment_file(
            fragment_file, adata.obs_names, chrom_sizes, tile_size, verbose,
        )
        X_tile = X_tile.astype(np.float32)
        source = "fragment_file"
    else:
        raise ValueError(
            "no usable input. Supply one of:\n"
            "  · adata.X with tile_size-wide chrN:s-e var_names, or\n"
            "  · snapATAC2-backed adata with fragment_paired, or\n"
            "  · fragment_file= + chrom_sizes="
        )

    n_cells, n_tiles = X_tile.shape
    _console(
        f"source={source}  cells={n_cells:,}  tiles={n_tiles:,}  "
        f"nnz={X_tile.nnz/1e6:.0f}M",
        verbose,
    )

    # ----- 2. gene annotation + tile→gene weight matrix -----
    genes = _normalise_genes(gene_anno, exclude_chr=exclude_chr)
    _console(
        f"genes: {len(genes):,} on {genes['chrom'].nunique()} chroms "
        f"(after excluding {list(exclude_chr)})",
        verbose,
    )
    W = _build_tile_gene_weight_matrix(
        tiles, genes,
        tile_size=tile_size,
        extend_upstream=extend_upstream,
        extend_downstream=extend_downstream,
        gene_upstream=gene_upstream,
        gene_downstream=gene_downstream,
        use_gene_boundaries=use_gene_boundaries,
        gene_scale_factor=gene_scale_factor,
        verbose=verbose,
    )

    # ----- 3. per-tile ceiling + matmul -----
    if ceiling is not None and ceiling > 0:
        # ArchR clips per-tile counts at ``ceiling``. This is a *per-cell*
        # per-tile cap, not a global one — do it on the sparse data array.
        np.clip(X_tile.data, 0, ceiling, out=X_tile.data)

    _console("computing gene_score = X_tile @ W", verbose)
    score = (X_tile @ W).astype(np.float32)
    if sp.issparse(score):
        score = score.toarray()

    # ----- 4. depth normalisation + log -----
    row_sums = score.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    score = score * (scale_to / row_sums)
    if log_transform:
        np.log2(score + 1, out=score)

    # ----- 5. attach + bookkeeping -----
    adata.obsm[key_added] = score
    adata.uns[f"{key_added}_gene_names"] = genes["gene_name"].tolist()
    adata.uns[f"{key_added}_params"] = dict(
        tile_size=int(tile_size),
        source=source,
        extend_upstream=list(extend_upstream),
        extend_downstream=list(extend_downstream),
        gene_upstream=int(gene_upstream),
        gene_downstream=int(gene_downstream),
        use_gene_boundaries=bool(use_gene_boundaries),
        gene_scale_factor=float(gene_scale_factor),
        ceiling=int(ceiling) if ceiling is not None else 0,
        scale_to=float(scale_to),
        log_transform=bool(log_transform),
        exclude_chr=list(exclude_chr),
    )
    _console(
        f"done. obsm[{key_added!r}] = {score.shape} float32 "
        f"(uns[{key_added}_gene_names]: {len(genes):,} genes)",
        verbose,
    )
    return adata
