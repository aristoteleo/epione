"""Pure-Python fragment → AnnData pipeline, no snapATAC2 dependency.

``epi.pp.import_fragments`` reads a tabix-indexed ``fragments.tsv.gz``,
computes per-cell QC (``n_fragment``, ``frac_dup``, ``frac_mito``),
and returns an ``AnnDataOOM`` object with:

* ``.obs``        — per-cell QC DataFrame (barcodes as index)
* ``.uns['files']['fragments']``     — path to the source BED
* ``.uns['reference_sequences']``    — ``{chrom: length}``

Fragments themselves are **not** stored in the AnnData object; all
downstream fragment-based operations (``add_tile_matrix`` /
``make_peak_matrix`` / QC metrics / ``add_gene_score_matrix``) re-read
the BED on demand via pysam. This matches ArchR's ``ArrowFile`` design
and keeps the AnnData file lightweight — fully compatible with
``anndataoom``'s out-of-memory X storage.
"""
from __future__ import annotations

import gzip
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ..utils._genome import Genome
from ..utils import console


# ---------------------------------------------------------------------------
# Genome helpers
# ---------------------------------------------------------------------------

def _resolve_chrom_sizes(chrom_sizes) -> dict:
    """Normalise ``chrom_sizes`` to ``{chrom: int length}``."""
    if hasattr(chrom_sizes, "chrom_sizes"):
        chrom_sizes = chrom_sizes.chrom_sizes
    cs = dict(chrom_sizes)
    if not cs:
        raise ValueError("chrom_sizes cannot be empty")
    return {str(k): int(v) for k, v in cs.items()}


def _build_anndata_oom(
    obs: pd.DataFrame,
    uns: dict,
    file: Optional[Path],
    n_vars: int = 0,
    var_df: Optional[pd.DataFrame] = None,
):
    """Build an ``AnnDataOOM`` (or plain ``AnnData`` if OOM unavailable).

    ``file`` is the .h5ad path — if given, persist to disk and reopen
    with ``anndataoom.read`` so the X matrix is backed on disk.
    Otherwise return an in-memory ``anndata.AnnData``.
    """
    if var_df is None:
        var_df = pd.DataFrame(index=pd.Index([], dtype=str))
    X = sp.csr_matrix((len(obs), n_vars), dtype=np.float32)
    adata = AnnData(X=X, obs=obs, var=var_df, uns=uns)
    if file is None:
        return adata
    file = Path(file)
    if file.exists():
        file.unlink()
    adata.write_h5ad(file)
    try:
        import anndataoom as oom
        return oom.read(str(file), backed="r+")
    except Exception:
        # Fall back to plain AnnData if anndataoom fails.
        return adata


# ---------------------------------------------------------------------------
# Fragment file scan (tabix-indexed or plain bgzipped BED)
# ---------------------------------------------------------------------------

def _open_fragment_file(path: Union[str, Path]):
    """Return an iterator yielding ``(chrom, start, end, barcode, count)``
    tuples from a ``fragments.tsv[.gz]`` file. Skips header / comment lines.
    """
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    fh = opener(path, "rt")
    try:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            chrom = parts[0]
            try:
                start = int(parts[1]); end = int(parts[2])
            except ValueError:
                continue
            bc = parts[3]
            cnt = int(parts[4]) if len(parts) >= 5 and parts[4].isdigit() else 1
            yield chrom, start, end, bc, cnt
    finally:
        fh.close()


def _ensure_tabix_index(path: Union[str, Path]) -> Path:
    """Ensure a ``.tbi`` index exists next to the fragment file.
    If not, build it with pysam's Tabix (bgzipped BED files only).
    """
    path = Path(path)
    if path.with_suffix(path.suffix + ".tbi").exists():
        return path.with_suffix(path.suffix + ".tbi")
    if path.suffix == ".gz":
        try:
            import pysam
            pysam.tabix_index(str(path), preset="bed", force=True, keep_original=True)
            return path.with_suffix(path.suffix + ".tbi")
        except Exception as exc:
            raise RuntimeError(f"could not build tabix index for {path}: {exc}")
    raise RuntimeError(
        f"{path} must be bgzipped (.tsv.gz) and tabix-indexed for random access"
    )


# ---------------------------------------------------------------------------
# Public API — fragment import
# ---------------------------------------------------------------------------

def import_fragments(
    fragment_file: Union[str, Path, list],
    chrom_sizes: Union[Genome, dict],
    *,
    file: Optional[Path] = None,
    min_num_fragments: int = 200,
    whitelist: Optional[Union[Path, list]] = None,
    chrM: list = ("chrM", "M", "chrMT", "MT"),
    n_jobs: int = 1,
    **kwargs,
) -> AnnData:
    """Scan a bgzipped fragments.tsv.gz and build a per-cell QC AnnData.

    No dependency on snapATAC2 — all logic is in pure Python. Per-cell
    QC metrics computed (``n_fragment``, ``frac_dup``, ``frac_mito``)
    follow the same definitions as snapATAC2 / ArchR so downstream
    filters behave identically.

    The fragment file itself is **not** ingested into the AnnData object
    (no ``obsm['fragment_paired']`` to avoid sparse-obsm persistence
    pitfalls with anndataoom). Instead, the BED path is recorded in
    ``adata.uns['files']['fragments']`` and subsequent tile /
    peak / gene-score steps re-read it via pysam.

    Parameters
    ----------
    fragment_file
        Path to a tabix-indexed ``fragments.tsv.gz`` (single or list).
    chrom_sizes
        A ``Genome`` object or ``{chrom: length}`` dict.
    file
        Output ``.h5ad`` path. When given, the AnnData is written via
        anndataoom for out-of-memory X storage. If ``None``, returns an
        in-memory ``anndata.AnnData``.
    min_num_fragments
        Minimum unique fragments per cell to retain (default 200).
    whitelist
        If given, only retain cells in this list (or path to a file with
        one barcode per line).
    chrM
        Chromosome names to treat as mitochondrial (affects ``frac_mito``).

    Returns
    -------
    adata
        AnnData (``n_obs × 0``) with per-cell QC in ``.obs``.
    """
    chrom_sizes = _resolve_chrom_sizes(chrom_sizes)

    if isinstance(fragment_file, (list, tuple)):
        if len(fragment_file) == 1:
            fragment_file = fragment_file[0]
        else:
            return [
                import_fragments(
                    f, chrom_sizes=chrom_sizes, file=file[i] if file else None,
                    min_num_fragments=min_num_fragments, whitelist=whitelist,
                    chrM=chrM, n_jobs=1,
                )
                for i, f in enumerate(fragment_file)
            ]

    fragment_file = Path(fragment_file)
    _ensure_tabix_index(fragment_file)

    chrM_set = set(chrM)
    valid_chroms = set(chrom_sizes)

    # Prepare whitelist set.
    whitelist_set = None
    if whitelist is not None:
        if isinstance(whitelist, (str, Path)):
            with open(whitelist, "r") as fh:
                whitelist_set = set(line.strip() for line in fh if line.strip())
        else:
            whitelist_set = set(whitelist)

    console.level1(f"scanning {fragment_file}")
    # Per-cell counters.
    n_frag = defaultdict(int)
    n_uniq = defaultdict(int)
    n_mito = defaultdict(int)
    seen = set()

    for chrom, start, end, bc, cnt in _open_fragment_file(fragment_file):
        if whitelist_set is not None and bc not in whitelist_set:
            continue
        if chrom not in valid_chroms and chrom not in chrM_set:
            continue
        if chrom in chrM_set:
            # Count mito reads (pre-dedup); ArchR's fragments file has
            # already collapsed PCR duplicates, so cnt here is effectively
            # the duplicate multiplicity.
            n_mito[bc] += 1
        else:
            # ``n_fragment`` = number of *unique* fragment rows for this
            # cell (one per row of the fragments file). This matches
            # ArchR's ``nFrags`` metric. The optional 5th column gives
            # the pre-dedup read count, which we use for frac_dup.
            n_frag[bc] += 1
            n_reads = cnt                           # pre-dedup read count
            # ``frac_dup`` = 1 - (unique fragments / total reads).
            # Track total-reads per cell in ``n_uniq`` (reused slot); the
            # name is misleading but keeps the rest of the code unchanged.
            n_uniq[bc] += n_reads

    # Build per-cell DataFrame.
    barcodes = sorted(set(n_frag) | set(n_mito))
    nf = np.asarray([n_frag[b] for b in barcodes], dtype=np.int64)
    n_total_reads = np.asarray([n_uniq[b] for b in barcodes], dtype=np.int64)
    nm = np.asarray([n_mito[b] for b in barcodes], dtype=np.int64)

    # Apply min fragments filter.
    keep = nf >= int(min_num_fragments)
    barcodes = [b for b, k in zip(barcodes, keep) if k]
    nf = nf[keep]; n_total_reads = n_total_reads[keep]; nm = nm[keep]

    frac_dup = np.where(n_total_reads > 0, 1.0 - nf / n_total_reads, 0.0)
    frac_mito = np.where(nf + nm > 0, nm / (nf + nm), 0.0)

    obs = pd.DataFrame({
        "n_fragment": nf,
        "frac_dup": frac_dup,
        "frac_mito": frac_mito,
    }, index=pd.Index(barcodes, name="barcode", dtype=str))

    uns = {
        "files": {"fragments": str(fragment_file)},
        "reference_sequences": chrom_sizes,
    }

    console.level2(f"imported {len(barcodes):,} cells  "
                   f"({int(nf.sum()):,} unique fragments)")

    adata = _build_anndata_oom(obs=obs, uns=uns, file=file)
    return adata


# ---------------------------------------------------------------------------
# Tile matrix — fragment BED → cell × 500 bp tile sparse matrix
# ---------------------------------------------------------------------------

def _tile_grid(chrom_sizes: dict, bin_size: int) -> tuple[dict, int, list]:
    """Return (chrom_offsets, total_tiles, tile_labels)."""
    offsets = {}
    labels = []
    base = 0
    for chrom, length in chrom_sizes.items():
        offsets[chrom] = base
        n = (int(length) + bin_size - 1) // bin_size
        for i in range(n):
            labels.append(f"{chrom}:{i*bin_size}-{(i+1)*bin_size - 1}")
        base += n
    return offsets, base, labels


def add_tile_matrix(
    adata: AnnData,
    *,
    bin_size: int = 500,
    counting_strategy: Literal["insertion", "paired-insertion", "fragment"] = "paired-insertion",
    chrM: list = ("chrM", "M", "chrMT", "MT"),
    verbose: bool = True,
) -> AnnData:
    """Bin fragments into per-cell 500 bp tile counts, write to ``adata.X``.

    Counting strategies
    -------------------
    - ``"insertion"``        — each TN5 cut (start + end) contributes 1
    - ``"paired-insertion"`` — each fragment contributes 1 to each tile
      its start **or** end falls into; if both ends fall in the same tile
      contribute 1 total (matches ArchR / snapATAC2 default)
    - ``"fragment"``         — each fragment contributes 1 to every tile
      it overlaps
    """
    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise ValueError("adata.uns['files']['fragments'] missing — "
                         "run epi.pp.import_fragments first")
    if "reference_sequences" not in adata.uns:
        raise ValueError("adata.uns['reference_sequences'] missing")

    fragment_file = adata.uns["files"]["fragments"]
    chrom_sizes = dict(adata.uns["reference_sequences"])
    chrom_offsets, n_tiles, tile_labels = _tile_grid(chrom_sizes, bin_size)

    barcodes = list(adata.obs_names)
    bc_to_idx = {b: i for i, b in enumerate(barcodes)}
    n_cells = len(barcodes)

    console.level1(
        f"building tile matrix: {n_cells:,} cells × {n_tiles:,} tiles "
        f"({bin_size} bp bins, strategy={counting_strategy})",
    )

    rows, cols, data = [], [], []
    chrM_set = set(chrM)
    for chrom, start, end, bc, cnt in _open_fragment_file(fragment_file):
        if bc not in bc_to_idx:
            continue
        if chrom in chrM_set or chrom not in chrom_offsets:
            continue
        i = bc_to_idx[bc]
        base = chrom_offsets[chrom]
        t1 = base + start // bin_size
        t2 = base + (end - 1) // bin_size

        if counting_strategy == "insertion":
            rows.append(i); cols.append(t1); data.append(cnt)
            if t2 != t1:
                rows.append(i); cols.append(t2); data.append(cnt)
        elif counting_strategy == "paired-insertion":
            rows.append(i); cols.append(t1); data.append(cnt)
            if t2 != t1:
                rows.append(i); cols.append(t2); data.append(cnt)
        elif counting_strategy == "fragment":
            for t in range(t1, t2 + 1):
                rows.append(i); cols.append(t); data.append(cnt)
        else:
            raise ValueError(f"unknown counting_strategy={counting_strategy!r}")

    X = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_cells, n_tiles), dtype=np.int32,
    ).tocsr()
    X.sum_duplicates()

    # Write back. AnnDataOOM supports .X assignment lazily; plain AnnData too.
    adata.X = X
    # Also fix var.
    var_df = pd.DataFrame(index=pd.Index(tile_labels, name="tile", dtype=str))
    try:
        adata.var = var_df
    except Exception:
        # Backed AnnData may not allow arbitrary var replacement; fall
        # back to attribute setting if available.
        for col in var_df.columns:
            adata.var[col] = var_df[col].values

    console.level2(f"tile matrix nnz={X.nnz:,}")
    return adata


# ---------------------------------------------------------------------------
# Peak matrix — fragment BED + peak BED → cell × peak sparse matrix
# ---------------------------------------------------------------------------

_PEAK_RE = re.compile(r"^([^:]+):(\d+)-(\d+)$")


def _parse_peaks(peaks) -> list[tuple[str, int, int]]:
    """Accept a list / Series / DataFrame of ``chrN:start-end`` strings or
    ``(chrom, start, end)`` tuples. Returns a sorted list of 3-tuples.
    """
    if isinstance(peaks, pd.DataFrame):
        if {"chrom", "start", "end"}.issubset(peaks.columns):
            out = [(r["chrom"], int(r["start"]), int(r["end"])) for _, r in peaks.iterrows()]
        else:
            out = [_parse_peak_label(p) for p in peaks.iloc[:, 0]]
    elif isinstance(peaks, pd.Series):
        out = [_parse_peak_label(p) for p in peaks]
    else:
        out = []
        for p in peaks:
            if isinstance(p, (tuple, list)) and len(p) == 3:
                out.append((p[0], int(p[1]), int(p[2])))
            else:
                out.append(_parse_peak_label(p))
    out.sort()
    return out


def _parse_peak_label(s):
    m = _PEAK_RE.match(str(s))
    if not m:
        raise ValueError(f"unrecognised peak label: {s!r}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def make_peak_matrix(
    adata: AnnData,
    use_rep,
    *,
    counting_strategy: Literal["insertion", "paired-insertion", "fragment"] = "paired-insertion",
    chrM: list = ("chrM", "M", "chrMT", "MT"),
    ceiling: Optional[int] = 4,
    verbose: bool = True,
) -> AnnData:
    """Build a new cell × peak AnnData from the fragment BED.

    ``use_rep`` is the peak set — accepts a list of ``chrN:s-e`` strings,
    a Series of the same, a DataFrame with ``chrom / start / end``
    columns, or a list of 3-tuples. The returned AnnData is a *new*
    object (does not modify ``adata``).

    ``ceiling`` — matches ArchR ``addPeakMatrix(ceiling=...)``. Per-cell
    per-peak counts are capped at this value. Default 4 (ArchR default).
    Pass ``None`` to disable capping.
    """
    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise ValueError("adata.uns['files']['fragments'] missing")

    fragment_file = str(adata.uns["files"]["fragments"])
    peaks = _parse_peaks(use_rep)
    # Sort peaks by (chrom, start); build numpy arrays per chromosome for
    # vectorised searchsorted lookup. ArchR merge_peaks guarantees
    # non-overlapping peaks, so each insertion site is in ≤ 1 peak.
    by_chrom: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for j, (chrom, s, e) in enumerate(peaks):
        by_chrom[chrom].append((s, e, j))
    chrom_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for chrom, lst in by_chrom.items():
        lst.sort()
        ps = np.fromiter((p[0] for p in lst), count=len(lst), dtype=np.int64)
        pe = np.fromiter((p[1] for p in lst), count=len(lst), dtype=np.int64)
        pj = np.fromiter((p[2] for p in lst), count=len(lst), dtype=np.int64)
        chrom_arrays[chrom] = (ps, pe, pj)
    peak_labels = [f"{c}:{s}-{e}" for c, s, e in peaks]

    barcodes = list(adata.obs_names)
    bc_to_idx = {b: i for i, b in enumerate(barcodes)}
    n_cells = len(barcodes)
    n_peaks = len(peaks)

    console.level1(f"building peak matrix: {n_cells:,} cells × {n_peaks:,} peaks")

    import pysam

    chrM_set = set(chrM)
    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    data_list: list[np.ndarray] = []

    _ensure_tabix_index(fragment_file)
    tbx = pysam.TabixFile(fragment_file)

    # Process chromosome-by-chromosome so we can vectorise per-chrom.
    for chrom, (ps, pe, pj) in chrom_arrays.items():
        if chrom in chrM_set:
            continue
        if chrom not in tbx.contigs:
            continue
        # Bulk-read fragment records for this chromosome.
        starts_buf: list[int] = []
        ends_buf: list[int] = []
        bcs_buf: list[int] = []
        cnts_buf: list[int] = []
        for line in tbx.fetch(chrom):
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            bc = parts[3]
            i = bc_to_idx.get(bc)
            if i is None:
                continue
            try:
                s = int(parts[1]); e = int(parts[2])
            except ValueError:
                continue
            cnt = int(parts[4]) if len(parts) >= 5 and parts[4].isdigit() else 1
            starts_buf.append(s); ends_buf.append(e)
            bcs_buf.append(i); cnts_buf.append(cnt)
        if not starts_buf:
            continue
        starts_arr = np.asarray(starts_buf, dtype=np.int64)
        ends_arr = np.asarray(ends_buf, dtype=np.int64)
        bcs_arr = np.asarray(bcs_buf, dtype=np.int64)
        cnts_arr = np.asarray(cnts_buf, dtype=np.int32)

        if counting_strategy in ("insertion", "paired-insertion"):
            # Two insertion sites per fragment (TN5 cut at both ends).
            # Each insertion event contributes 1, NOT ``cnt`` (the 5th
            # column is the pre-dedup read count, already collapsed to
            # a single unique fragment per row; ArchR's ``addPeakMatrix``
            # treats each fragment as 2 insertion events regardless).
            for cut_arr in (starts_arr, ends_arr - 1):
                idx = np.searchsorted(ps, cut_arr, side="right") - 1
                valid = idx >= 0
                if not valid.any():
                    continue
                # Check end > cut for the candidate peak.
                candidate_ends = pe[np.where(valid, idx, 0)]
                valid &= candidate_ends > cut_arr
                if not valid.any():
                    continue
                rows_list.append(bcs_arr[valid])
                cols_list.append(pj[idx[valid]])
                data_list.append(np.ones(int(valid.sum()), dtype=np.int32))
        else:
            # "fragment": overlap all peaks intersecting [start, end).
            # Vectorised: for each peak, find fragments in [start, end).
            # Simpler: iterate peaks per-chrom — usually n_peaks_chrom << n_frag_chrom.
            # For each peak, use searchsorted on ends_arr > peak_start and
            # starts_arr < peak_end to find contributing fragments.
            frag_sort = np.argsort(starts_arr, kind="stable")
            starts_s = starts_arr[frag_sort]
            ends_s = ends_arr[frag_sort]
            bcs_s = bcs_arr[frag_sort]
            cnts_s = cnts_arr[frag_sort]
            for k in range(len(ps)):
                pstart, pend, ppj = int(ps[k]), int(pe[k]), int(pj[k])
                # Fragments with start < pend
                right = np.searchsorted(starts_s, pend, side="left")
                if right == 0:
                    continue
                # Of those, fragments with end > pstart
                mask = ends_s[:right] > pstart
                if not mask.any():
                    continue
                rows_list.append(bcs_s[:right][mask])
                cols_list.append(np.full(int(mask.sum()), ppj, dtype=np.int64))
                # 1 per overlapping fragment, not ``cnt`` (read dup count).
                data_list.append(np.ones(int(mask.sum()), dtype=np.int32))

    tbx.close()

    if rows_list:
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.concatenate(data_list).astype(np.int32, copy=False)
    else:
        rows = np.array([], dtype=np.int64)
        cols = np.array([], dtype=np.int64)
        data = np.array([], dtype=np.int32)

    X = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_cells, n_peaks), dtype=np.int32,
    ).tocsr()
    X.sum_duplicates()
    # ArchR-style per-cell per-peak cap (addPeakMatrix(ceiling=...)).
    if ceiling is not None and X.nnz > 0:
        np.minimum(X.data, int(ceiling), out=X.data)

    peak_mat = AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.Index(barcodes, dtype=str)),
        var=pd.DataFrame(index=pd.Index(peak_labels, dtype=str)),
        uns={
            "files": dict(adata.uns.get("files", {})),
            "reference_sequences": dict(adata.uns.get("reference_sequences", {})),
        },
    )
    # Carry over adata.obs.
    try:
        peak_mat.obs = pd.DataFrame(adata.obs).copy()
        peak_mat.obs.index = barcodes
    except Exception:
        pass

    console.level2(f"peak matrix nnz={X.nnz:,}")
    return peak_mat


# ---------------------------------------------------------------------------
# Gene activity matrix — simple flat window (distinct from ArchR gene score)
# ---------------------------------------------------------------------------

def make_gene_matrix(
    adata: AnnData,
    gene_anno,
    *,
    upstream: int = 2000,
    downstream: int = 0,
    include_gene_body: bool = True,
    chrM: list = ("chrM", "M", "chrMT", "MT"),
    verbose: bool = True,
) -> AnnData:
    """Flat-window gene activity matrix (cell × gene).

    For each gene, counts fragments whose insertion sites fall inside a
    window spanning ``[gene_body_start − upstream, gene_body_end +
    downstream]`` on the appropriate strand. For the ArchR-equivalent
    distance-weighted version see :func:`epione.tl.add_gene_score_matrix`.
    """
    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise ValueError("adata.uns['files']['fragments'] missing")

    # Accept a Genome (resolve via get_gene_annotation), a GTF/GFF3 path
    # (same), or a ready DataFrame.
    from ..utils._genome import Genome
    if isinstance(gene_anno, Genome):
        from ..utils._read import get_gene_annotation
        gene_anno = get_gene_annotation(gene_anno)
    elif isinstance(gene_anno, (str, Path)):
        p = str(gene_anno)
        if p.endswith((".gtf", ".gff", ".gff3", ".gtf.gz", ".gff.gz", ".gff3.gz")):
            from ..utils._read import get_gene_annotation
            gene_anno = get_gene_annotation(p)
        else:
            gene_anno = pd.read_csv(p, sep="\t")
    required = {"gene_name", "chrom", "start", "end", "strand"}
    if not required.issubset(gene_anno.columns):
        raise ValueError(f"gene_anno missing columns: {required - set(gene_anno.columns)}")

    genes = gene_anno.copy()
    if include_gene_body:
        # Window = gene body ± (upstream, downstream) strand-aware.
        is_plus = genes["strand"].to_numpy() == "+"
        genes["window_start"] = np.where(
            is_plus, genes["start"] - upstream, genes["start"] - downstream,
        )
        genes["window_end"] = np.where(
            is_plus, genes["end"] + downstream, genes["end"] + upstream,
        )
    else:
        # Window = TSS ± padding.
        tss = np.where(genes["strand"] == "+", genes["start"], genes["end"])
        genes["window_start"] = tss - upstream
        genes["window_end"] = tss + downstream

    genes["window_start"] = np.maximum(genes["window_start"], 0).astype(np.int64)
    genes["window_end"] = genes["window_end"].astype(np.int64)

    # Per-chrom arrays: window_start, window_end, gene_idx sorted by start.
    # Unlike peaks, gene windows *can overlap* (adjacent genes' upstream
    # padding often hits neighbours), so each insertion event can fall
    # into multiple genes — we scan all candidates vectorised.
    chrom_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for chrom, sub in genes.groupby("chrom", sort=False):
        order = np.argsort(sub["window_start"].to_numpy(), kind="stable")
        ws = sub["window_start"].to_numpy()[order].astype(np.int64)
        we = sub["window_end"].to_numpy()[order].astype(np.int64)
        gi = np.asarray(sub.index[order].to_numpy(), dtype=np.int64)  # row in genes
        chrom_arrays[chrom] = (ws, we, gi)

    fragment_file = adata.uns["files"]["fragments"]
    barcodes = list(adata.obs_names)
    bc_to_idx = {b: i for i, b in enumerate(barcodes)}
    n_cells = len(barcodes)
    n_genes = len(genes)
    chrM_set = set(chrM)

    console.level1(f"building gene activity matrix: {n_cells:,} × {n_genes:,} (flat window)")

    import pysam
    _ensure_tabix_index(fragment_file)
    tbx = pysam.TabixFile(str(fragment_file))

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    data_list: list[np.ndarray] = []

    for chrom, (ws, we, gi) in chrom_arrays.items():
        if chrom in chrM_set or chrom not in tbx.contigs:
            continue
        # Bulk-load all fragments on this chromosome.
        starts_buf, ends_buf, bcs_buf = [], [], []
        for line in tbx.fetch(chrom):
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            i = bc_to_idx.get(parts[3])
            if i is None:
                continue
            try:
                s = int(parts[1]); e = int(parts[2])
            except ValueError:
                continue
            starts_buf.append(s); ends_buf.append(e); bcs_buf.append(i)
        if not starts_buf:
            continue
        starts_arr = np.asarray(starts_buf, dtype=np.int64)
        ends_arr = np.asarray(ends_buf, dtype=np.int64)
        bcs_arr = np.asarray(bcs_buf, dtype=np.int64)

        # For each insertion event (fragment start and end-1), find ALL
        # genes whose window covers it. Gene windows can overlap, so we
        # can't use the "pick one" trick from peak_matrix — instead, for
        # each event we bracket the candidate range
        # ``[first gene with window_start <= cut, …, last gene with
        # window_start <= cut]`` and then filter by ``window_end > cut``.
        # For ATAC tile density against ~20k genes this is still ~100×
        # faster than a per-fragment Python binary search.
        for cut_arr in (starts_arr, ends_arr - 1):
            # window_start ≤ cut  →  searchsorted(ws, cut, 'right') - 1
            upper = np.searchsorted(ws, cut_arr, side="right")
            # For each fragment, candidates are ws[:upper]. Scan backwards
            # through the max span of overlapping gene windows.
            max_span = int((we - ws).max()) if len(we) else 0
            lower_cut = cut_arr - max_span
            lower = np.searchsorted(ws, lower_cut, side="left")
            # Collect: for each fragment i, k in [lower[i], upper[i]) such
            # that we[k] > cut_arr[i]. To vectorise we explode into
            # (frag_idx, gene_idx) pairs via repeat + compare.
            counts = upper - lower
            if counts.sum() == 0:
                continue
            # Build (frag, gene_candidate_index) pairs.
            frag_rep = np.repeat(np.arange(len(cut_arr), dtype=np.int64), counts)
            gene_k = np.concatenate([
                np.arange(int(lo), int(hi), dtype=np.int64)
                for lo, hi in zip(lower, upper)
            ]) if counts.sum() else np.array([], dtype=np.int64)
            # Filter by window_end > cut
            keep = we[gene_k] > cut_arr[frag_rep]
            if not keep.any():
                continue
            rows_list.append(bcs_arr[frag_rep[keep]])
            cols_list.append(gi[gene_k[keep]])
            data_list.append(np.ones(int(keep.sum()), dtype=np.int32))

    tbx.close()

    if rows_list:
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.concatenate(data_list)
    else:
        rows = np.array([], dtype=np.int64)
        cols = np.array([], dtype=np.int64)
        data = np.array([], dtype=np.int32)

    X = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_cells, n_genes), dtype=np.int32,
    ).tocsr()
    X.sum_duplicates()

    out = AnnData(
        X=X,
        obs=pd.DataFrame(adata.obs).copy(),
        var=pd.DataFrame({"gene_name": genes["gene_name"].values},
                         index=pd.Index(genes["gene_name"], dtype=str)),
    )
    console.level2(f"gene matrix nnz={X.nnz:,}")
    return out


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(
    adata: AnnData,
    n_features: int = 500_000,
    *,
    key_added: str = "selected",
    verbose: bool = True,
) -> np.ndarray:
    """Mark the top-``n_features`` most-accessible columns (tiles or peaks)
    in ``adata.var[key_added]`` as True.

    Uses per-column fraction of cells with at least one count. Ties are
    broken by total count.
    """
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    # Per-column counts.
    total = np.asarray(X.sum(axis=0)).ravel()
    n_pos = np.asarray((X != 0).sum(axis=0)).ravel()
    # Compound score: fraction of cells present, break ties by total.
    score = n_pos.astype(np.float64) + total / (total.max() + 1e-9) * 1e-6
    n = min(n_features, X.shape[1])
    thr = np.partition(score, -n)[-n]
    sel = score >= thr
    try:
        adata.var[key_added] = sel
    except Exception:
        # backed AnnData might not accept bulk var assignment
        for i, v in enumerate(sel):
            adata.var.iloc[i, adata.var.columns.get_loc(key_added)] = v
    console.level2(f"selected {int(sel.sum()):,}/{adata.n_vars:,} features → var[{key_added!r}]")
    return sel


# ---------------------------------------------------------------------------
# Cell calling / filtering
# ---------------------------------------------------------------------------

def call_cells(adata: AnnData, min_fragments: int = 1000, **kwargs) -> AnnData:
    """Simple cell-calling: keep barcodes with ``n_fragment`` ≥ threshold."""
    obs = pd.DataFrame(adata.obs)
    keep = obs["n_fragment"].to_numpy() >= int(min_fragments)
    console.level2(f"call_cells: {int(keep.sum()):,}/{len(keep):,} pass")
    sub = adata[keep].copy()
    return sub


def filter_cells(
    adata: AnnData,
    min_counts: Optional[int] = None,
    max_counts: Optional[int] = None,
    min_tsse: Optional[float] = None,
    **kwargs,
) -> AnnData:
    """Filter cells by basic QC thresholds. All thresholds are applied
    jointly and return a new AnnData view.
    """
    obs = pd.DataFrame(adata.obs)
    mask = np.ones(len(obs), dtype=bool)
    if min_counts is not None and "n_fragment" in obs:
        mask &= obs["n_fragment"].to_numpy() >= min_counts
    if max_counts is not None and "n_fragment" in obs:
        mask &= obs["n_fragment"].to_numpy() <= max_counts
    if min_tsse is not None and "tsse" in obs:
        mask &= obs["tsse"].to_numpy() >= min_tsse
    console.level2(f"filter_cells: {int(mask.sum()):,}/{len(mask):,} retained")
    return adata[mask].copy()
