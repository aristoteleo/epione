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


def _gene_score_from_fragments(
    adata,
    genes: pd.DataFrame,
    *,
    tile_size: int,
    extend_upstream: Tuple[int, int],
    extend_downstream: Tuple[int, int],
    gene_upstream: int,
    gene_downstream: int,
    use_gene_boundaries: bool,
    gene_scale_factor: float,
    ceiling: Optional[int],
    verbose: bool,
) -> np.ndarray:
    """ArchR-faithful gene score computation — **chromosome-by-chromosome**,
    never materialising the full genome-wide tile matrix.

    For each chromosome:

    1. Scan the fragment BED once, tile-align every TN5 cut
       (``start // tile_size * tile_size``, same for ``end``).
    2. Build ``matGS`` = sparse ``(n_unique_tiles × n_cells)`` with
       only the tiles that actually saw an insertion as rows.
    3. Apply the ceiling to cap the per-cell per-tile count.
    4. For every gene on this chromosome, locate overlapping
       unique tiles via binary search, compute the distance-decayed
       weight × per-gene ``geneWeight`` and build a sparse
       ``(n_genes_on_chrom × n_unique_tiles)`` weight matrix ``W``.
    5. ``chrom_score = W @ matGS`` — one sparse × sparse matmul.
    6. Accumulate into the global ``gene_score`` matrix at the
       right gene columns.
    """
    import pysam

    fragment_file = adata.uns["files"]["fragments"]
    bc_to_idx = {str(b): i for i, b in enumerate(adata.obs_names)}
    n_cells = len(bc_to_idx)
    n_genes = len(genes)

    # Per-gene precomputed fields (body_lo / body_hi / geneWeight etc.)
    # — done once up front, shared across chromosomes.
    genes_sorted = genes.copy()
    genes_sorted["gene_idx_global"] = np.arange(n_genes, dtype=np.int64)
    genes_sorted = genes_sorted.sort_values(
        ["chrom", "start", "end"], kind="stable"
    ).reset_index(drop=True)

    # We'll build geneWeight per chromosome (ArchR normalises per-chrom).
    gene_score = np.zeros((n_cells, n_genes), dtype=np.float64)

    tbx = pysam.TabixFile(str(fragment_file))
    try:
        chroms_in_file = set(tbx.contigs)
    except Exception:
        chroms_in_file = set(adata.uns.get("reference_sequences", {}))

    chroms_with_genes = [c for c in genes_sorted["chrom"].unique() if c in chroms_in_file]

    try:
        for chrom in chroms_with_genes:
            g_chr = genes_sorted[genes_sorted["chrom"] == chrom].reset_index(drop=True)
            _console(
                f"chr={chrom}  genes={len(g_chr):,}",
                verbose,
            )

            # ------- 1. scan fragments on this chromosome -------
            rows_i = []          # tile index (into uniq_ins below)
            cols_i = []          # cell index
            tile_pos = []        # int start coordinate of each insertion
            try:
                it = tbx.fetch(chrom)
            except Exception:
                continue
            # First pass: collect raw tile-aligned positions + cell ids.
            pos_list = []
            cell_list = []
            for line in it:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                try:
                    start = int(parts[1]); end = int(parts[2])
                except ValueError:
                    continue
                bc = parts[3]
                cell_i = bc_to_idx.get(bc)
                if cell_i is None:
                    continue
                ts = (start // tile_size) * tile_size
                te = ((end - 1) // tile_size) * tile_size
                pos_list.append(ts)
                cell_list.append(cell_i)
                if te != ts:
                    pos_list.append(te)
                    cell_list.append(cell_i)

            if not pos_list:
                continue

            pos_arr = np.asarray(pos_list, dtype=np.int64)
            cell_arr = np.asarray(cell_list, dtype=np.int64)
            del pos_list, cell_list

            # ------- 2. map to unique tile positions -------
            uniq_ins, inv = np.unique(pos_arr, return_inverse=True)
            # Build (n_uniq_tiles × n_cells) sparse.
            matGS = sp.coo_matrix(
                (np.ones(len(pos_arr), dtype=np.float32), (inv, cell_arr)),
                shape=(len(uniq_ins), n_cells),
            ).tocsr()
            matGS.sum_duplicates()
            if ceiling is not None and ceiling > 0:
                np.clip(matGS.data, 0, ceiling, out=matGS.data)
            del pos_arr, cell_arr, inv

            # ------- 3. compute extended-body / domain for each gene -------
            strand = g_chr["strand"].to_numpy()
            is_plus = strand == "+"
            g_start = g_chr["start"].to_numpy(np.int64)
            g_end = g_chr["end"].to_numpy(np.int64)
            body_lo = np.where(is_plus,
                                g_start - gene_upstream,
                                g_start - gene_downstream).astype(np.int64)
            body_hi = np.where(is_plus,
                                g_end + gene_downstream,
                                g_end + gene_upstream).astype(np.int64)

            # geneWeight: 1 / extended_width, normalised to [1, gsf]
            width = (body_hi - body_lo).astype(np.float64)
            width = np.where(width > 0, width, 1.0)
            m = 1.0 / width
            m_lo, m_hi = float(m.min()), float(m.max())
            if m_hi > m_lo:
                gene_weight = 1.0 + m * (gene_scale_factor - 1.0) / (m_hi - m_lo)
            else:
                gene_weight = np.ones_like(m)
            gene_weight = np.clip(gene_weight, 1.0, gene_scale_factor).astype(np.float32)

            # ------- 4. domain with optional neighbour-boundary capping -------
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

            # ------- 5. per-gene overlap with uniq_ins -------
            # uniq_ins is the start coordinate of each populated tile.
            tile_centres = uniq_ins + tile_size // 2

            W_rows = []; W_cols = []; W_vals = []
            for g in range(len(g_chr)):
                lo = int(dom_lo[g]); hi = int(dom_hi[g])
                i0 = np.searchsorted(tile_centres, lo, side="left")
                i1 = np.searchsorted(tile_centres, hi, side="left")
                if i1 == i0:
                    continue
                c = tile_centres[i0:i1]
                bl, bh = int(body_lo[g]), int(body_hi[g])
                x = np.maximum(0, np.maximum(bl - (c + tile_size // 2), (c - tile_size // 2) - bh)).astype(np.float64)
                w = (np.exp(-x / 5000.0) + np.exp(-1.0)).astype(np.float32) * gene_weight[g]
                W_rows.extend([g] * (i1 - i0))
                W_cols.extend(range(i0, i1))
                W_vals.extend(w.tolist())
            if not W_rows:
                continue
            W = sp.coo_matrix(
                (np.asarray(W_vals, dtype=np.float32),
                 (np.asarray(W_rows, dtype=np.int64),
                  np.asarray(W_cols, dtype=np.int64))),
                shape=(len(g_chr), len(uniq_ins)),
            ).tocsr()

            # ------- 6. matmul (n_genes_chr × n_uniq_tiles) @ (n_uniq_tiles × n_cells) -------
            chrom_score = (W @ matGS).astype(np.float32)
            if sp.issparse(chrom_score):
                chrom_score = chrom_score.toarray()

            # ------- 7. accumulate at the correct global gene columns -------
            g_idx_global = g_chr["gene_idx_global"].to_numpy(np.int64)
            # chrom_score has shape (n_genes_chr, n_cells) — transpose for our output.
            gene_score[:, g_idx_global] += chrom_score.T
            del matGS, W, chrom_score
    finally:
        tbx.close()

    return gene_score.astype(np.float32)


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
    # ----- 1. decide execution path -----
    # The ArchR implementation walks fragments **chromosome by chromosome**
    # and only materialises the tiles that actually saw an insertion — for
    # hg19 that's ~1-5 M tiles (not 6.27 M). We mirror that path whenever
    # ``adata.uns['files']['fragments']`` is known, avoiding a full
    # genome-wide ``(n_cells, 6.27M)`` materialisation of ``adata.X``.
    #
    # ``use_x`` keeps the legacy ``adata.X`` path available for callers
    # that don't have the BED (e.g. already-pruned data).
    has_fragments = _adata_has_fragments(adata) or (fragment_file is not None)
    x_looks_ok = _adata_x_looks_like_tile(adata, tile_size)

    if use_x == "auto":
        use_fragments_path = has_fragments      # prefer fragment path when possible
    elif use_x is True:
        use_fragments_path = False
        if not x_looks_ok:
            raise ValueError(
                f"use_x=True but adata.X var_names don't look like "
                f"{tile_size} bp tiles"
            )
    else:
        use_fragments_path = True

    # Support a ``fragment_file=`` override (for in-memory adata without
    # the standard ``uns['files']['fragments']`` entry).
    if use_fragments_path and fragment_file is not None:
        # Temporarily stash the path so the chrom-by-chrom helper below
        # picks it up; we restore afterward.
        old_files = dict(adata.uns.get("files", {}))
        adata.uns["files"] = {**old_files, "fragments": str(fragment_file)}
        if chrom_sizes is not None and "reference_sequences" not in adata.uns:
            adata.uns["reference_sequences"] = dict(chrom_sizes)
        _restore_files = lambda: adata.uns.__setitem__("files", old_files)
    else:
        _restore_files = lambda: None

    # ----- 2. gene annotation -----
    genes = _normalise_genes(gene_anno, exclude_chr=exclude_chr)
    _console(
        f"genes: {len(genes):,} on {genes['chrom'].nunique()} chroms "
        f"(after excluding {list(exclude_chr)})",
        verbose,
    )

    if use_fragments_path:
        _console(
            "chrom-by-chrom ArchR path (fragments → uniq_tiles → W @ matGS)",
            verbose,
        )
        score = _gene_score_from_fragments(
            adata, genes,
            tile_size=tile_size,
            extend_upstream=extend_upstream,
            extend_downstream=extend_downstream,
            gene_upstream=gene_upstream,
            gene_downstream=gene_downstream,
            use_gene_boundaries=use_gene_boundaries,
            gene_scale_factor=gene_scale_factor,
            ceiling=ceiling,
            verbose=verbose,
        )
        source = "fragments (chrom-by-chrom)"
        _restore_files()
    else:
        # Legacy path: use ``adata.X`` as the tile matrix. Requires the
        # full (n_cells × n_tiles) to fit in RAM.
        _console("input: adata.X (already a tile matrix)", verbose)
        X = adata.X
        if type(X).__name__ in (
            "BackedArray", "_SubsetBackedArray",
            "TransformedBackedArray", "ScaledBackedArray",
        ):
            # Chunked materialise to sparse to avoid 125 GB dense allocation.
            parts = []
            for _, _, chunk in X.chunked():
                if sp.issparse(chunk):
                    parts.append(chunk)
                else:
                    parts.append(sp.csr_matrix(chunk))
            X = sp.vstack(parts).tocsr()
        elif not sp.issparse(X):
            X = sp.csr_matrix(X)
        X_tile = X.astype(np.float32).tocsr()
        tiles = _parse_tile_labels(list(adata.var_names))
        n_cells_x, n_tiles_x = X_tile.shape
        _console(
            f"source=adata.X  cells={n_cells_x:,}  tiles={n_tiles_x:,}  "
            f"nnz={X_tile.nnz/1e6:.0f}M",
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
        if ceiling is not None and ceiling > 0:
            np.clip(X_tile.data, 0, ceiling, out=X_tile.data)
        _console("computing gene_score = X_tile @ W", verbose)
        score = (X_tile @ W).astype(np.float32)
        if sp.issparse(score):
            score = score.toarray()
        source = "adata.X"

    # ----- 3. depth normalisation + log -----
    row_sums = score.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    score = (score * (scale_to / row_sums)).astype(np.float32)
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
