"""Sampling and distance utilities for bulk peak / gene analyses.

These helpers underlie the OTX2 case-study notebooks (Fig 3d, 3f) and
are general enough to reuse for any TF-binding / DEG analysis.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def expression_matched_sample(
    target_values: np.ndarray,
    pool_values: np.ndarray,
    pool_ids: Optional[Sequence] = None,
    *,
    n_bins: int = 10,
    size_per_bin: Optional[int] = None,
    log1p: bool = True,
    replace: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Draw a random subset of ``pool_ids`` whose values match the distribution
    of ``target_values`` via equal-count quantile bins.

    The classic use case is selecting a set of "random background" genes with
    the same expression-strength distribution as a foreground set of interest
    (e.g. DEGs vs expression-matched random controls), so that differences in
    downstream statistics aren't driven by expression-level confounding.

    Arguments:
        target_values: values whose distribution we want to match (e.g. baseMean
            of activated genes). 1-D array-like of length N_target.
        pool_values: values of the candidate pool to sample from (e.g. baseMean
            of all expressed non-activated genes). 1-D array-like.
        pool_ids: identifiers of pool items, same length as ``pool_values``.
            If ``None``, returns the integer indices into ``pool_values``.
        n_bins: number of quantile bins used to stratify the target distribution.
            Defaults to 10 (deciles).
        size_per_bin: if given, sample this many items per bin regardless of
            the per-bin count in ``target_values``. Default ``None`` matches
            the target's per-bin counts 1:1.
        log1p: if True, apply ``log1p`` before computing bin edges; useful when
            values span several orders of magnitude (RPKM, read counts).
        replace: sampling with replacement within each bin. Default False.
        seed: RNG seed.

    Returns:
        Array of sampled IDs (or indices if ``pool_ids`` is None) of length
        approximately equal to ``len(target_values)`` (or ``size_per_bin *
        n_bins`` when ``size_per_bin`` is set). Bins where the pool is empty
        are skipped silently.

    Example:
        >>> rand = expression_matched_sample(
        ...     target_values=base_activated, pool_values=base_other,
        ...     pool_ids=other_gene_symbols, n_bins=10, seed=0,
        ... )
    """
    tv = np.asarray(target_values, dtype=float)
    pv = np.asarray(pool_values, dtype=float)
    if pool_ids is None:
        pool_ids = np.arange(len(pv))
    else:
        pool_ids = np.asarray(pool_ids)
    if len(pool_ids) != len(pv):
        raise ValueError("pool_ids must have same length as pool_values")

    tv_t = np.log1p(tv) if log1p else tv
    pv_t = np.log1p(pv) if log1p else pv

    edges = np.percentile(tv_t, np.linspace(0, 100, n_bins + 1))
    # Expand the outer edges so min/max of tv are not on bin boundaries.
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    tv_bins = np.digitize(tv_t, edges[1:-1])
    pv_bins = np.digitize(pv_t, edges[1:-1])

    rng = np.random.default_rng(seed)
    out = []
    for b in range(n_bins):
        cand = np.where(pv_bins == b)[0]
        if len(cand) == 0:
            continue
        need = size_per_bin if size_per_bin is not None else int((tv_bins == b).sum())
        k = need if replace else min(need, len(cand))
        if k == 0:
            continue
        pick = rng.choice(cand, size=k, replace=replace)
        out.extend(pool_ids[pick].tolist())
    return np.asarray(out)


def distance_to_nearest_peak(
    features: pd.DataFrame,
    peaks: pd.DataFrame,
    *,
    feature_col: str = "tss",
    peak_center: str = "center",
    chrom_col: str = "chrom",
    unit: str = "bp",
) -> np.ndarray:
    """Minimum absolute distance from each feature locus to the nearest peak
    on the same chromosome.

    This is the engine for TSS-to-nearest-peak cumulative distribution plots
    (e.g. OTX2 Fig 3d). Features on a chromosome with no peaks are skipped.

    Arguments:
        features: DataFrame with ``chrom_col`` and ``feature_col`` columns.
            Typically one row per gene where ``feature_col`` is the TSS.
        peaks: DataFrame with ``chrom_col`` and ``peak_center`` columns; may
            alternatively supply ``start`` + ``end`` together and leave
            ``peak_center`` as a string that doesn't exist - the function then
            derives peak midpoints from ``start`` + ``end``.
        feature_col: column in ``features`` holding the anchor position.
        peak_center: column in ``peaks`` holding peak midpoint. If missing,
            midpoints are computed from ``peaks['start'] + peaks['end']``.
        chrom_col: chromosome column name (shared between tables).
        unit: "bp" or "kb". The return array's units.

    Returns:
        1-D numpy array of non-negative distances. Features without any same-
        chromosome peaks are omitted, so the return length may be less than
        ``len(features)``.
    """
    if peak_center not in peaks.columns:
        if not {"start", "end"}.issubset(peaks.columns):
            raise ValueError(
                "peaks must have a peak_center column, or both 'start' and 'end'"
            )
        peaks = peaks.copy()
        peaks[peak_center] = (
            peaks["start"].astype(np.int64) + peaks["end"].astype(np.int64)
        ) // 2

    peaks = peaks.dropna(subset=[chrom_col, peak_center])
    peak_by_chrom = {
        c: np.sort(g[peak_center].astype(np.int64).to_numpy())
        for c, g in peaks.groupby(chrom_col)
    }
    features = features.dropna(subset=[chrom_col, feature_col])
    out = []
    for c, pos in zip(features[chrom_col].values, features[feature_col].astype(np.int64).values):
        centers = peak_by_chrom.get(c)
        if centers is None or len(centers) == 0:
            continue
        i = np.searchsorted(centers, pos)
        cands = []
        if i > 0:
            cands.append(abs(int(centers[i - 1]) - int(pos)))
        if i < len(centers):
            cands.append(abs(int(centers[i]) - int(pos)))
        if cands:
            out.append(min(cands))
    arr = np.asarray(out, dtype=np.float64)
    return arr / 1000.0 if unit == "kb" else arr


# ----------------------------------------------------------------------
# Peak-set utilities (distal filter + multi-set cluster labelling).
# ----------------------------------------------------------------------

def filter_distal_peaks(
    peaks: pd.DataFrame,
    features: pd.DataFrame,
    *,
    min_distance: int = 2500,
    chrom_col: str = "chrom",
    feature_col: str = "tss",
) -> pd.DataFrame:
    """Return the subset of ``peaks`` whose centre is at least
    ``min_distance`` bp from the nearest feature on the same chromosome.

    The classic use-case is "distal" vs "promoter-proximal" filtering:
    keep peaks whose centre is ≥ 2.5 kb from any annotated TSS. Peaks on
    chromosomes that have no features listed are retained by default (a
    peak with *no nearby feature at all* is trivially distal).

    Arguments:
        peaks: DataFrame with ``chrom_col``, ``start``, ``end`` columns
            (the centre is computed on the fly as the midpoint).
        features: DataFrame with ``chrom_col`` and ``feature_col`` columns
            (e.g. a GTF projection with a ``tss`` column).
        min_distance: minimum allowed distance in bp.
        chrom_col: chromosome column name (shared between tables).
        feature_col: column in ``features`` holding the anchor position.

    Returns:
        A filtered DataFrame preserving ``peaks``' schema + column order.

    Example:
        >>> gtf['tss'] = np.where(gtf.strand == '+', gtf.start, gtf.end)
        >>> distal = epi.utils.filter_distal_peaks(
        ...     peaks_4C, gtf[['chrom','tss']], min_distance=2500)
    """
    if len(peaks) == 0:
        return peaks.copy()

    ft = features.dropna(subset=[chrom_col, feature_col])
    feat_by_chrom = {
        str(c): np.sort(g[feature_col].astype(np.int64).to_numpy())
        for c, g in ft.groupby(chrom_col, sort=False)
    }

    centres = (
        (peaks["start"].astype(np.int64) + peaks["end"].astype(np.int64)) // 2
    ).to_numpy()
    chroms = peaks[chrom_col].astype(str).to_numpy()

    keep = np.ones(len(peaks), dtype=bool)
    for i, (c, pos) in enumerate(zip(chroms, centres)):
        arr = feat_by_chrom.get(c)
        if arr is None or len(arr) == 0:
            continue  # no features on this chrom -> definitionally distal
        k = int(np.searchsorted(arr, pos))
        d_bp = 1e18
        if k > 0:
            d_bp = min(d_bp, int(pos - arr[k - 1]))
        if k < len(arr):
            d_bp = min(d_bp, int(arr[k] - pos))
        if d_bp < min_distance:
            keep[i] = False
    return peaks[keep].reset_index(drop=True)


def _build_interval_index(df: pd.DataFrame, chrom_col: str = "chrom") -> dict:
    """Return ``{chrom: (starts_sorted, ends_sorted)}`` — used internally
    by :func:`classify_peaks_by_overlap` for fast membership checks."""
    out = {}
    for c, g in df.groupby(chrom_col, sort=False):
        s = g["start"].astype(np.int64).to_numpy()
        e = g["end"].astype(np.int64).to_numpy()
        order = np.argsort(s)
        out[str(c)] = (s[order], e[order])
    return out


def _overlaps_any(
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    idx: dict,
) -> np.ndarray:
    """Boolean array: does each query (chrom, start, end) overlap any
    interval stored in ``idx`` (as produced by ``_build_interval_index``)?

    Each query scans candidates whose stored ``start`` is strictly less
    than the query's ``end`` (via ``searchsorted``), then tests whose
    stored ``end`` exceeds the query's ``start``. This is O(k log N)
    where ``k`` is the number of candidates with smaller starts; for
    sparse peak sets (typical MACS2 output) ``k`` is small."""
    hit = np.zeros(len(chroms), dtype=bool)
    for i in range(len(chroms)):
        lo_idx = idx.get(chroms[i])
        if lo_idx is None:
            continue
        st, en = lo_idx
        k = int(np.searchsorted(st, ends[i]))
        if k == 0:
            continue
        if bool(np.any(en[:k] > starts[i])):
            hit[i] = True
    return hit


def classify_peaks_by_overlap(
    peak_sets,
    *,
    primary_order=None,
    chrom_col: str = "chrom",
    specific_suffix: str = "-specific",
    shared_separator: str = "/",
    shared_suffix: str = " shared",
) -> pd.DataFrame:
    """Combine N peak sets into one table whose ``cluster`` column names
    every overlap pattern (A-specific, A/B shared, A/B/C shared, …).

    Each peak appears **exactly once** in the output. Peaks are
    attributed to a "primary" set following ``primary_order`` — the
    first set in that list that contains the peak — so the result
    partitions (peak_sets[0] ∪ peak_sets[1] ∪ …) without duplication
    even when the same genomic interval is called in multiple sets.

    Arguments:
        peak_sets: ``{label: DataFrame}`` (or any Mapping). Each frame
            must have ``chrom_col``, ``start``, ``end``.
        primary_order: iteration order used to decide which set each
            peak is attributed to. Defaults to ``list(peak_sets.keys())``.
            A peak from set ``A`` that also overlaps ``B`` and ``C`` is
            assigned to cluster ``"A/B/C shared"`` (not counted again in
            ``B`` or ``C`` rows).
        chrom_col: chromosome column name (shared across inputs).
        specific_suffix: suffix for clusters with a single set, default
            ``"-specific"`` -> ``"A-specific"``.
        shared_separator: joiner for multi-set cluster names, default
            ``"/"`` -> ``"A/B"``.
        shared_suffix: suffix appended to multi-set cluster names, default
            ``" shared"`` -> ``"A/B shared"``.

    Returns:
        Concatenated DataFrame (all rows from all input sets, one row
        per genomic peak under its primary set) with an extra ``cluster``
        column naming the overlap pattern. Other columns are carried
        through from each input set; columns not shared across sets are
        filled with NaN where missing.

    Example:
        >>> out = epi.utils.classify_peaks_by_overlap(
        ...     {'4C': distal_4C, 'ESC': distal_ESC, 'dEC': distal_dEC},
        ...     primary_order=['4C','ESC','dEC'],
        ... )
        >>> out['cluster'].value_counts()
        4C-specific            ...
        4C/ESC shared          ...
        4C/ESC/dEC shared      ...
        ESC-specific           ...
        ESC/dEC shared         ...
        dEC-specific           ...
    """
    if not peak_sets:
        raise ValueError("peak_sets must contain at least one DataFrame")

    labels = list(peak_sets.keys()) if primary_order is None \
        else list(primary_order)
    missing = [L for L in labels if L not in peak_sets]
    if missing:
        raise ValueError(f"primary_order references missing sets: {missing}")

    # Pre-build an interval index per set.
    indices = {L: _build_interval_index(peak_sets[L], chrom_col=chrom_col)
               for L in labels}

    # For each primary set, drop rows that any earlier set already contains
    # (so each peak is counted once) and compute its overlap signature.
    out_frames = []
    earlier = []
    for L in labels:
        df = peak_sets[L].copy()
        if len(df) == 0:
            earlier.append(L); continue
        ch = df[chrom_col].astype(str).to_numpy()
        s  = df["start"].astype(np.int64).to_numpy()
        e  = df["end"].astype(np.int64).to_numpy()

        # Drop peaks already covered by an earlier set in primary_order.
        if earlier:
            already = np.zeros(len(df), dtype=bool)
            for E in earlier:
                already |= _overlaps_any(ch, s, e, indices[E])
            df = df[~already].reset_index(drop=True)
            ch = df[chrom_col].astype(str).to_numpy()
            s  = df["start"].astype(np.int64).to_numpy()
            e  = df["end"].astype(np.int64).to_numpy()

        # Signature: which later sets this peak overlaps with.
        sig_cols = {L: np.ones(len(df), dtype=bool)}  # primary set membership
        for later in labels[labels.index(L) + 1:]:
            sig_cols[later] = _overlaps_any(ch, s, e, indices[later])

        def _name(row_members):
            if sum(row_members) == 1:
                return f"{L}{specific_suffix}"
            shared = [lab for lab, flag in zip(sig_cols.keys(), row_members) if flag]
            return shared_separator.join(shared) + shared_suffix

        memberships = np.stack([sig_cols[lab] for lab in sig_cols], axis=1)
        cluster = np.array(
            [_name(r) for r in memberships], dtype=object
        )
        df["cluster"] = cluster
        out_frames.append(df)
        earlier.append(L)

    return pd.concat(out_frames, ignore_index=True)


# ----------------------------------------------------------------------
# Genomic-feature annotation of a peak set (ChIPseeker-style).
# ----------------------------------------------------------------------

def _merge_intervals(df: pd.DataFrame, chrom_col: str = "chrom") -> dict:
    """Return ``{chrom: (starts, ends)}`` of merged, sorted, non-overlapping
    intervals — collapsing an interval list to a flat covered set so an
    "overlaps any" test is fast and unambiguous."""
    out = {}
    for c, g in df.groupby(chrom_col, sort=False):
        s = g["start"].astype(np.int64).to_numpy()
        e = g["end"].astype(np.int64).to_numpy()
        if len(s) == 0:
            continue
        order = np.argsort(s)
        s, e = s[order], e[order]
        ms, me = [int(s[0])], [int(e[0])]
        for i in range(1, len(s)):
            if s[i] <= me[-1]:                 # overlapping / book-ended
                me[-1] = max(me[-1], int(e[i]))
            else:
                ms.append(int(s[i])); me.append(int(e[i]))
        out[str(c)] = (np.asarray(ms, dtype=np.int64),
                       np.asarray(me, dtype=np.int64))
    return out


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR for a small p-value vector (no statsmodels
    dependency)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def annotate_peaks(
    peaks: pd.DataFrame,
    gtf,
    *,
    promoter_upstream: int = 2000,
    promoter_downstream: int = 2000,
    precedence: Sequence[str] = ("promoter", "exon", "intron", "intergenic"),
    chrom_col: str = "chrom",
    gene_type: Optional[str] = "protein_coding",
    feature_col: str = "feature",
) -> pd.DataFrame:
    """Assign every peak a single genomic-feature class from a GTF.

    Classifies each peak as ``promoter`` / ``exon`` / ``intron`` /
    ``intergenic`` — the standard ChIP-seq / ATAC-seq peak annotation
    (the equivalent of ChIPseeker's ``annotatePeak``). A peak that
    overlaps several feature kinds is assigned the **highest-precedence**
    class only, so each peak is counted exactly once and the four
    classes partition the peak set.

    Knowing *where* a peak set sits — promoter-proximal vs distal /
    intronic / intergenic — is what turns a list of peaks into a
    regulatory statement, so this is a routine step after peak calling
    or differential-peak analysis.

    Arguments:
        peaks: DataFrame with ``chrom_col``, ``start``, ``end`` columns
            (e.g. a narrowPeak table or a differential-peak set).
        gtf: a GTF/GFF3 path (``.gz`` ok) or an :class:`~epione.utils.genome.Genome`
            — anything :func:`epione.utils.get_gene_annotation` accepts.
        promoter_upstream: bp upstream of a TSS counted as promoter
            (strand-aware). Default 2000.
        promoter_downstream: bp downstream of a TSS counted as promoter
            (strand-aware). Default 2000 — i.e. promoter = TSS +/- 2 kb.
        precedence: feature classes ordered highest-priority first. A peak
            overlapping more than one is assigned the earliest class in
            this tuple. Default ``promoter > exon > intron > intergenic``.
        chrom_col: chromosome column name in ``peaks``.
        gene_type: gene biotype filter passed to ``get_gene_annotation``
            (e.g. ``"protein_coding"``). Pass ``None`` for a GTF that has
            no biotype attribute (some refGene-style GTFs).
        feature_col: name of the output class column. Default ``"feature"``.

    Returns:
        A copy of ``peaks`` with one added column (``feature_col``) holding
        the feature class of each peak.

    Example:
        >>> ann = epi.utils.annotate_peaks(gained_peaks, 'genes.gtf')
        >>> ann['feature'].value_counts(normalize=True)
        intergenic    0.52
        intron        0.31
        promoter      0.12
        exon          0.05
    """
    known = {"promoter", "exon", "intron", "intergenic"}
    bad = [c for c in precedence if c not in known]
    if bad:
        raise ValueError(f"precedence has unknown classes {bad}; "
                         f"allowed: {sorted(known)}")

    out = peaks.copy()
    if len(peaks) == 0:
        out[feature_col] = pd.Series([], dtype=object)
        return out

    from epione.io._read import get_gene_annotation  # lazy: avoid import cycle

    genes = get_gene_annotation(gtf, feature="gene", gene_type=gene_type)
    exons = get_gene_annotation(gtf, feature="exon", gene_type=gene_type)

    plus = genes["strand"].astype(str).to_numpy() == "+"
    g_start = genes["start"].astype(np.int64).to_numpy()
    g_end = genes["end"].astype(np.int64).to_numpy()
    tss = np.where(plus, g_start, g_end)
    p_start = np.where(plus, tss - promoter_upstream, tss - promoter_downstream)
    p_end = np.where(plus, tss + promoter_downstream, tss + promoter_upstream)

    prom_df = pd.DataFrame({"chrom": genes["chrom"].astype(str),
                            "start": np.maximum(p_start, 0), "end": p_end})
    body_df = pd.DataFrame({"chrom": genes["chrom"].astype(str),
                            "start": g_start, "end": g_end})
    exon_df = pd.DataFrame({"chrom": exons["chrom"].astype(str),
                            "start": exons["start"].astype(np.int64),
                            "end": exons["end"].astype(np.int64)})

    idx_prom = _merge_intervals(prom_df)
    idx_exon = _merge_intervals(exon_df)
    idx_body = _merge_intervals(body_df)

    ch = peaks[chrom_col].astype(str).to_numpy()
    s = peaks["start"].astype(np.int64).to_numpy()
    e = peaks["end"].astype(np.int64).to_numpy()

    masks = {
        "promoter": _overlaps_any(ch, s, e, idx_prom),
        "exon": _overlaps_any(ch, s, e, idx_exon),
        "intron": _overlaps_any(ch, s, e, idx_body),   # gene body; demoted by precedence
        "intergenic": np.ones(len(peaks), dtype=bool),
    }
    cls = np.empty(len(peaks), dtype=object)
    for name in reversed(list(precedence)):            # low priority first; high overwrites
        cls[masks[name]] = name
    out[feature_col] = cls
    return out


def peak_feature_enrichment(
    foreground_peaks: pd.DataFrame,
    background_peaks: pd.DataFrame,
    gtf,
    *,
    classes: Sequence[str] = ("promoter", "exon", "intron", "intergenic"),
    chrom_col: str = "chrom",
    **annotate_kwargs,
) -> pd.DataFrame:
    """Test which genomic-feature classes are enriched in a foreground peak
    set relative to a background peak set (Fisher's exact test).

    The standard use is asking *where* a set of differential peaks
    concentrates: annotate the gained (or lost) peaks and a background set
    — typically the unchanged / non-differential peaks, which controls for
    the assay's overall genomic bias — and test each feature class with a
    2x2 Fisher's exact test.

    Arguments:
        foreground_peaks: the peak set of interest (e.g. gained / lost
            differential peaks). DataFrame with ``chrom_col/start/end``.
        background_peaks: the reference set (e.g. unchanged peaks, or all
            called peaks). Using the unchanged peaks as background tests
            for a *redistribution* on top of the assay's baseline bias.
        gtf: GTF/GFF3 path or Genome — passed to :func:`annotate_peaks`.
        classes: feature classes to report. Default all four.
        chrom_col: chromosome column name in both peak tables.
        **annotate_kwargs: forwarded to :func:`annotate_peaks`
            (``promoter_upstream``, ``gene_type``, ...).

    Returns:
        DataFrame, one row per feature class, sorted by p-value:
        ``feature``, ``fg_count``, ``fg_frac``, ``bg_count``,
        ``bg_frac``, ``enrichment`` (= ``fg_frac / bg_frac``;
        ``inf`` if the class is in the foreground but absent from the
        background, ``NaN`` if absent from both), ``odds_ratio``,
        ``pvalue``, ``fdr`` (Benjamini-Hochberg). ``enrichment > 1``
        with a small ``fdr`` is a positively enriched class.

    Example:
        >>> enr = epi.utils.peak_feature_enrichment(
        ...     gained_peaks, unchanged_peaks, 'genes.gtf')
        >>> enr[['feature', 'enrichment', 'pvalue', 'fdr']]
    """
    from scipy.stats import fisher_exact  # lazy

    fg = annotate_peaks(foreground_peaks, gtf, chrom_col=chrom_col,
                        **annotate_kwargs)
    bg = annotate_peaks(background_peaks, gtf, chrom_col=chrom_col,
                        **annotate_kwargs)
    fg_counts = fg["feature"].value_counts()
    bg_counts = bg["feature"].value_counts()
    n_fg, n_bg = len(fg), len(bg)

    rows = []
    for cl in classes:
        a = int(fg_counts.get(cl, 0)); b = n_fg - a
        c = int(bg_counts.get(cl, 0)); d = n_bg - c
        odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        fg_frac = a / n_fg if n_fg else np.nan
        bg_frac = c / n_bg if n_bg else np.nan
        if bg_frac and bg_frac > 0:
            enr = fg_frac / bg_frac
        elif fg_frac and fg_frac > 0:
            enr = np.inf            # in the foreground, absent from the background
        else:
            enr = np.nan            # absent from both -> undefined
        rows.append(dict(feature=cl, fg_count=a, fg_frac=fg_frac,
                         bg_count=c, bg_frac=bg_frac,
                         enrichment=enr, odds_ratio=odds, pvalue=p))
    res = pd.DataFrame(rows)
    res["fdr"] = _bh_fdr(res["pvalue"].to_numpy())
    return res.sort_values("pvalue").reset_index(drop=True)
