"""Differential peak / feature analysis for bulk count matrices.

Two backends with a unified API:

- ``pydeseq2``  — PyDESeq2 (Python port of DESeq2) [default].
- ``edgepy``    — edgeR QL GLM via inmoose.edgepy.

Both are triggered through :func:`differential_peaks`, which accepts either
an :class:`anndata.AnnData` (``obs`` × ``var``) or an explicit
(counts, metadata) pair and returns a single DataFrame with the columns

    baseMean, log2FoldChange, lfcSE, stat, pvalue, padj

regardless of backend, so downstream plotting code doesn't care which
engine ran.

Typical usage::

    import epione as epi
    res = epi.tl.differential_peaks(
        adata,                              # samples × peaks counts
        design='~condition',
        contrast=('condition', 'trt', 'ctrl'),
        backend='pydeseq2',
    )
    epi.pl.volcano(res, title='trt vs ctrl')
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# Columns the function promises to return, in order.
_CANONICAL_COLS = [
    "baseMean",
    "log2FoldChange",
    "lfcSE",
    "stat",
    "pvalue",
    "padj",
]


def differential_peaks(
    data=None,
    *,
    counts: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    metadata: Optional[pd.DataFrame] = None,
    design: str = "~condition",
    contrast: Optional[Sequence[str]] = None,
    backend: Literal["pydeseq2", "edgepy", "poisson"] = "pydeseq2",
    min_count: int = 10,
    min_samples: int = 1,
    alpha: float = 0.05,
    n_cpus: Optional[int] = None,
    quiet: bool = True,
    **backend_kwargs,
) -> pd.DataFrame:
    """Differential analysis on bulk count matrices.

    Arguments:
        data: optional :class:`anndata.AnnData` with ``X`` = (samples, features)
            integer counts and ``obs`` = per-sample metadata. Mutually
            exclusive with ``counts`` + ``metadata``.
        counts: DataFrame or ndarray, (samples, features). Used when
            ``data`` is ``None``.
        metadata: per-sample DataFrame aligned to ``counts``' rows.
            Required when ``data`` is ``None``.
        design: R-style formula over metadata columns (e.g. ``'~condition'``,
            ``'~batch + condition'``). Used directly by pyDESeq2, converted
            to a patsy design matrix for edgepy.
        contrast: ``(factor, level_a, level_b)`` — test ``level_a`` versus
            ``level_b``, reporting positive log2FoldChange when the feature
            is higher in ``level_a``. **Required.**
        backend: ``'pydeseq2'`` or ``'edgepy'`` (both need biological
            replicates), or ``'poisson'`` for the **no-replicate** case
            (one sample per condition) — a per-region exact Poisson /
            binomial test of the two conditions' counts against the
            library-size-expected ratio. Use ``'poisson'`` for n=1 TF
            ChIP-seq / CUT&RUN; ``pydeseq2`` / ``edgepy`` raise a clear
            error if called with a single sample per condition.
        min_count: drop features whose total count across all samples is
            below this (pre-filter; saves compute and avoids zero-inflation
            regressions in both backends).
        min_samples: drop features detected in fewer than this many samples
            (non-zero count).
        alpha: false-discovery rate for independent filtering (pyDESeq2 only).
        n_cpus: parallel workers where supported. ``None`` leaves the
            backend to decide.
        quiet: suppress backend's progress chatter.
        **backend_kwargs: forwarded to the selected backend's entry point.

    Returns:
        ``pandas.DataFrame`` indexed by feature ID with the columns
        ``baseMean, log2FoldChange, lfcSE, stat, pvalue, padj``. Features
        dropped by the pre-filter are absent from the result.

    Example:
        >>> res = epi.tl.differential_peaks(
        ...     adata, design='~condition',
        ...     contrast=('condition', 'trt', 'ctrl'),
        ...     backend='pydeseq2',
        ... )
        >>> res.sort_values('padj').head()
    """
    if contrast is None or len(contrast) != 3:
        raise ValueError(
            "contrast must be a 3-tuple (factor, level_a, level_b); "
            f"got {contrast!r}"
        )
    counts_df, meta = _normalise_inputs(data, counts, metadata)

    # Pre-filter: drop features with too few reads / too few samples detected.
    totals = counts_df.sum(axis=0).to_numpy()
    detected = (counts_df > 0).sum(axis=0).to_numpy()
    keep = (totals >= min_count) & (detected >= min_samples)
    if not keep.any():
        raise ValueError(
            f"No features pass the filter (min_count={min_count},"
            f" min_samples={min_samples}). Loosen the thresholds."
        )
    counts_df = counts_df.loc[:, keep]

    backend = backend.lower()
    if backend == "poisson":
        res = _run_poisson(counts_df, meta, contrast, **backend_kwargs)
    elif backend in ("pydeseq2", "edgepy"):
        _require_replicates(counts_df, meta, contrast, backend)
        if backend == "pydeseq2":
            res = _run_pydeseq2(
                counts_df, meta, design, contrast,
                alpha=alpha, n_cpus=n_cpus, quiet=quiet,
                **backend_kwargs,
            )
        else:
            res = _run_edgepy(
                counts_df, meta, design, contrast,
                quiet=quiet, **backend_kwargs,
            )
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Use 'pydeseq2', 'edgepy' or "
            "'poisson'."
        )

    # Ensure canonical column set + order. Missing columns become NaN.
    for c in _CANONICAL_COLS:
        if c not in res.columns:
            res[c] = np.nan
    return res[_CANONICAL_COLS]


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

def _normalise_inputs(
    data, counts, metadata,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Accept AnnData OR (counts, metadata) and return a (counts_df, meta)
    pair with counts_df = (samples, features) and meta aligned by row."""
    if data is not None:
        if counts is not None or metadata is not None:
            raise ValueError(
                "Pass either ``data`` OR ``counts`` + ``metadata``, not both."
            )
        X = data.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        counts_df = pd.DataFrame(
            np.asarray(X),
            index=data.obs_names,
            columns=data.var_names,
        )
        meta = data.obs.copy()
        return counts_df, meta

    if counts is None or metadata is None:
        raise ValueError(
            "Provide either ``data`` (AnnData) or both ``counts`` and"
            " ``metadata``."
        )
    if isinstance(counts, np.ndarray):
        counts_df = pd.DataFrame(counts, index=metadata.index)
    elif isinstance(counts, pd.DataFrame):
        counts_df = counts.copy()
    else:
        raise TypeError(
            f"counts must be DataFrame or ndarray; got {type(counts).__name__}"
        )
    if not counts_df.index.equals(metadata.index):
        # Try to reorder metadata to counts' order if the same set of labels.
        if set(counts_df.index) == set(metadata.index):
            metadata = metadata.loc[counts_df.index]
        else:
            raise ValueError(
                "counts' rows and metadata's rows do not refer to the same"
                " samples."
            )
    # pyDESeq2 / edgepy both break on duplicate feature names — fail loudly
    # so users don't get cryptic "cannot reindex duplicate labels" errors.
    dup_feat = counts_df.columns[counts_df.columns.duplicated()]
    if len(dup_feat):
        raise ValueError(
            f"counts has {len(dup_feat)} duplicate feature names, e.g. "
            f"{list(dup_feat[:5])!r}. Aggregate or drop them before calling "
            "differential_peaks (e.g. "
            "``counts = counts.groupby(counts.columns, axis=1).sum()``)."
        )
    dup_samp = counts_df.index[counts_df.index.duplicated()]
    if len(dup_samp):
        raise ValueError(
            f"counts has {len(dup_samp)} duplicate sample names, e.g. "
            f"{list(dup_samp[:5])!r}."
        )
    return counts_df, metadata.copy()


# ---------------------------------------------------------------------------
# Backend: pyDESeq2
# ---------------------------------------------------------------------------

def _run_pydeseq2(
    counts_df, meta, design, contrast,
    *, alpha, n_cpus, quiet, **kwargs,
) -> pd.DataFrame:
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
        from pydeseq2.default_inference import DefaultInference
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pyDESeq2 is required for backend='pydeseq2'. "
            "Install with: pip install pydeseq2"
        ) from e

    # pyDESeq2 strictly requires integer counts.
    counts_int = counts_df.round().astype(int)
    inference = DefaultInference(n_cpus=n_cpus) if n_cpus else None
    dds = DeseqDataSet(
        counts=counts_int,
        metadata=meta,
        design=design,
        inference=inference,
        quiet=quiet,
        **kwargs,
    )
    dds.deseq2()

    stats = DeseqStats(
        dds, contrast=list(contrast),
        alpha=alpha, quiet=quiet,
    )
    stats.summary()
    res = stats.results_df.copy()
    # pyDESeq2 already ships baseMean / log2FoldChange / lfcSE / stat / pvalue / padj.
    return res


# ---------------------------------------------------------------------------
# Backend: edgepy (inmoose)
# ---------------------------------------------------------------------------

def _run_edgepy(
    counts_df, meta, design, contrast,
    *, quiet, **kwargs,
) -> pd.DataFrame:
    try:
        import patsy
        from inmoose.edgepy import DGEList, glmLRT
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "inmoose is required for backend='edgepy'. "
            "Install with: pip install inmoose"
        ) from e

    # patsy.dmatrix needs the design info attached — keep the DesignMatrix,
    # don't convert to ndarray (edgepy reads ``.design_info`` internally).
    X = patsy.dmatrix(design, meta)
    col_names = list(X.design_info.column_names)

    # edgepy expects counts in (features, samples) orientation.
    counts_mat = counts_df.values.T.astype(np.float64)

    dge = DGEList(counts=counts_mat)

    # Apply a light-weight TMM normalisation so effective lib sizes reflect
    # composition differences (behaviour expected by downstream log-FC).
    tmm = _calc_norm_factors_tmm(counts_mat)
    dge.samples["lib_size"] = dge.samples["lib_size"].values * tmm
    dge.samples["norm_factors"] = tmm

    dge = dge.estimateGLMCommonDisp(design=X)
    dge = dge.estimateGLMTagwiseDisp(design=X)

    # NOTE: we use glmLRT rather than glmQLFTest — the latter has a bug in
    # inmoose 0.8.1 where internal pandas ops on the DGELRT subclass trigger
    # a spurious "log2FoldChange missing from results table" error.
    fit = dge.glmFit(design=X, **kwargs)

    contrast_vec = _build_edgepy_contrast(contrast, col_names)
    result = glmLRT(fit, contrast=contrast_vec.reshape(-1, 1))
    # Unwrap DGELRT to a plain DataFrame to avoid its opinionated _constructor.
    res = pd.DataFrame(result.values, columns=result.columns, index=result.index)
    res.index = counts_df.columns

    # Map logCPM -> baseMean proxy (convert log2-CPM back to linear CPM scale
    # for comparability with pyDESeq2's baseMean, which is mean of normalised
    # counts). Not identical but serves the same role in plots.
    if "logCPM" in res.columns:
        res["baseMean"] = 2.0 ** res["logCPM"]
    # Benjamini-Hochberg FDR for 'padj' (edgepy's LRT doesn't attach it).
    if "pvalue" in res.columns:
        res["padj"] = _bh_fdr(res["pvalue"].to_numpy())
    return res


def _build_edgepy_contrast(
    contrast: Sequence[str], col_names: Sequence[str],
) -> np.ndarray:
    """Translate ``(factor, level_a, level_b)`` into a contrast vector over
    the patsy design matrix columns. When patsy treatment-codes a factor
    with intercept, the reference level is absent from the design columns,
    so the contrast becomes ±1 on the single level that is present."""
    factor, level_a, level_b = contrast
    # patsy uses e.g. ``condition[T.trt]`` for treatment-coded levels and
    # ``condition[trt]`` for no-intercept / sum-coded levels. Try both.
    candidates_a = [f"{factor}[T.{level_a}]", f"{factor}[{level_a}]"]
    candidates_b = [f"{factor}[T.{level_b}]", f"{factor}[{level_b}]"]
    vec = np.zeros(len(col_names))
    a_found = b_found = False
    for name in candidates_a:
        if name in col_names:
            vec[col_names.index(name)] = 1.0
            a_found = True
            break
    for name in candidates_b:
        if name in col_names:
            vec[col_names.index(name)] = -1.0
            b_found = True
            break
    if not (a_found or b_found):
        raise ValueError(
            f"Could not locate contrast {contrast!r} in design columns"
            f" {col_names!r}. Check that the factor/level names match the"
            " metadata."
        )
    return vec


# ---------------------------------------------------------------------------
# Light-weight TMM + BH utilities (avoid dragging extra R deps)
# ---------------------------------------------------------------------------

def _calc_norm_factors_tmm(
    counts_mat: np.ndarray,
    *,
    logratio_trim: float = 0.3,
    sum_trim: float = 0.05,
) -> np.ndarray:
    """Compute TMM (Trimmed Mean of M-values) normalisation factors.

    ``counts_mat`` is (features, samples). Returns an array of length
    ``n_samples`` whose product is 1.

    Reference column is chosen as the sample whose upper-quartile count
    ratio is closest to the mean (Robinson & Oshlack, 2010).
    """
    x = counts_mat
    lib = x.sum(axis=0)
    valid = lib > 0
    if valid.sum() < 2:
        return np.ones(x.shape[1])

    ratio = x / np.maximum(lib, 1)
    upper_q = np.quantile(ratio, 0.75, axis=0)
    ref_col = int(np.argmin(np.abs(upper_q - upper_q.mean())))

    ref = ratio[:, ref_col]
    factors = np.ones(x.shape[1])
    for j in range(x.shape[1]):
        if j == ref_col or not valid[j]:
            continue
        obs = ratio[:, j]
        mask = (x[:, j] > 0) & (x[:, ref_col] > 0)
        if mask.sum() < 4:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            M = np.log2(obs[mask] / ref[mask])
            A = 0.5 * np.log2(obs[mask] * ref[mask])
        finite = np.isfinite(M) & np.isfinite(A)
        if finite.sum() < 4:
            continue
        M, A = M[finite], A[finite]
        lo_m, hi_m = np.quantile(M, [logratio_trim, 1 - logratio_trim])
        lo_a, hi_a = np.quantile(A, [sum_trim, 1 - sum_trim])
        keep = (M >= lo_m) & (M <= hi_m) & (A >= lo_a) & (A <= hi_a)
        if keep.sum() == 0:
            continue
        factors[j] = 2.0 ** np.mean(M[keep])

    # Normalise so the geometric mean of the factors is 1.
    gm = np.exp(np.mean(np.log(factors[factors > 0])))
    factors = factors / gm
    return factors


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR, matching scipy.stats.false_discovery_control."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    ranked = np.arange(1, n + 1, dtype=float)
    q = p[order] * n / ranked
    # Enforce monotonic non-increase from the largest p-value downwards.
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


# ---------------------------------------------------------------------------
# No-replicate (Poisson) backend
# ---------------------------------------------------------------------------

def _require_replicates(counts_df, meta, contrast, backend):
    """Raise a clear, actionable error if a count-based backend is asked to
    run with only one sample per condition (it cannot estimate dispersion)."""
    factor = contrast[0]
    if factor not in meta.columns:
        return
    per_level = meta.groupby(factor).size()
    a, b = contrast[1], contrast[2]
    if int(per_level.get(a, 0)) < 2 or int(per_level.get(b, 0)) < 2:
        raise ValueError(
            f"backend={backend!r} needs >=2 samples per condition to "
            f"estimate dispersion, but contrast levels have "
            f"{int(per_level.get(a,0))} ({a}) and {int(per_level.get(b,0))} "
            f"({b}) sample(s). For a no-replicate design use "
            "backend='poisson'."
        )


def _run_poisson(counts_df, meta, contrast, *, pseudocount: float = 0.5,
                 alternative: str = "two-sided", **_kwargs) -> pd.DataFrame:
    """No-replicate differential test.

    Pools (sums) the replicate counts of each condition and applies a
    per-region **exact binomial test** of the foreground count against the
    library-size-expected proportion — the standard one-vs-one ChIP-seq /
    ATAC-seq comparison when there is no replication to estimate dispersion.

    ``alternative`` sets the test direction and should match the question:

    - ``'two-sided'`` (default) — test for **any** change; use when the
      question is "what changed" / "is there a difference".
    - ``'less'`` — test only for a **decrease** in the first contrast
      level (``log2FoldChange < 0``); use for a directional question
      such as "which regions are *reduced* / *lost*".
    - ``'greater'`` — test only for an **increase** in the first contrast
      level; use for "which regions are *gained* / *increased*".

    A one-sided test is the correct, more powerful choice when the
    question itself is directional — it is not p-hacking, because the
    direction was fixed by the question before the data was seen.
    """
    from scipy.stats import binomtest
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(
            "alternative must be 'two-sided', 'less' or 'greater'")
    factor, level_a, level_b = contrast
    a_idx = meta.index[meta[factor].astype(str) == str(level_a)]
    b_idx = meta.index[meta[factor].astype(str) == str(level_b)]
    if len(a_idx) == 0 or len(b_idx) == 0:
        raise ValueError(
            f"contrast levels {level_a!r}/{level_b!r} not both present in "
            f"metadata column {factor!r}."
        )
    a = counts_df.loc[a_idx].sum(axis=0).to_numpy(dtype=float)
    b = counts_df.loc[b_idx].sum(axis=0).to_numpy(dtype=float)
    lib_a, lib_b = float(a.sum()), float(b.sum())
    if lib_a <= 0 or lib_b <= 0:
        raise ValueError("one condition has zero total counts.")
    p0 = lib_a / (lib_a + lib_b)
    a_int = np.rint(a).astype(np.int64)
    tot = np.rint(a + b).astype(np.int64)
    pvals = np.ones(len(a), dtype=float)
    for i in range(len(a)):
        if tot[i] > 0:
            pvals[i] = binomtest(int(a_int[i]), int(tot[i]), p0,
                                 alternative=alternative).pvalue
    log2fc = np.log2(((a + pseudocount) / lib_a) /
                     ((b + pseudocount) / lib_b))
    base = (a / lib_a + b / lib_b) * (0.5e6)        # mean CPM (baseMean role)
    return pd.DataFrame({
        "baseMean": base,
        "log2FoldChange": log2fc,
        "lfcSE": np.nan,
        "stat": np.nan,
        "pvalue": pvals,
        "padj": _bh_fdr(pvals),
    }, index=counts_df.columns)


# ---------------------------------------------------------------------------
# Building the region x sample quantification matrix
# ---------------------------------------------------------------------------

def count_reads_in_peaks(
    bam_files,
    peaks: pd.DataFrame,
    *,
    chrom_col: str = "chrom",
    start_col: str = "start",
    end_col: str = "end",
    min_mapq: int = 0,
) -> pd.DataFrame:
    """Count reads overlapping each peak region in each BAM file.

    The standard first step of a count-based differential ChIP-seq /
    ATAC-seq analysis: turn aligned reads + a peak set into the
    region x sample count matrix that :func:`differential_peaks` consumes
    (the ``bedtools multicov`` / ``featureCounts`` equivalent).

    Arguments:
        bam_files: a mapping ``{sample_name: bam_path}`` (each BAM
            coordinate-sorted with a ``.bai`` index), or a list of BAM
            paths (the path string is then used as the sample name).
        peaks: DataFrame with ``chrom_col`` / ``start_col`` / ``end_col``
            columns — typically a consensus / union peak set.
        chrom_col, start_col, end_col: the interval column names in ``peaks``.
        min_mapq: ignore reads below this mapping quality (0 = count all).

    Returns:
        DataFrame, rows = peaks (``peaks``' index), columns = samples,
        values = integer read counts. Transpose (``.T``) before passing to
        :func:`differential_peaks`, which expects samples x regions.

    Example:
        >>> counts = epi.tl.count_reads_in_peaks(
        ...     {'ctrl': 'ctrl.bam', 'trt': 'trt.bam'}, consensus_peaks)
        >>> res = epi.tl.differential_peaks(
        ...     counts=counts.T, metadata=meta,
        ...     contrast=('condition', 'trt', 'ctrl'), backend='poisson')
    """
    import pysam
    items = (list(bam_files.items()) if isinstance(bam_files, dict)
             else [(str(p), p) for p in bam_files])
    chroms = peaks[chrom_col].astype(str).to_numpy()
    starts = peaks[start_col].astype(np.int64).to_numpy()
    ends = peaks[end_col].astype(np.int64).to_numpy()
    n = len(peaks)
    cb = ((lambda r: (not r.is_unmapped) and r.mapping_quality >= min_mapq)
          if min_mapq > 0 else "all")
    out = {}
    for sample, path in items:
        bam = pysam.AlignmentFile(str(path), "rb")
        refs = set(bam.references)
        col = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if chroms[i] in refs:
                col[i] = bam.count(chroms[i], int(starts[i]), int(ends[i]),
                                   read_callback=cb)
        bam.close()
        out[str(sample)] = col
    return pd.DataFrame(out, index=peaks.index)


def _merge_intervals_table(df: pd.DataFrame,
                           chrom_col: str = "chrom") -> pd.DataFrame:
    """Merge overlapping/book-ended intervals into a non-overlapping union;
    return a chrom/start/end DataFrame sorted by (chrom, start)."""
    rows = []
    for c, g in df.groupby(chrom_col, sort=True):
        s = g["start"].astype(np.int64).to_numpy()
        e = g["end"].astype(np.int64).to_numpy()
        order = np.argsort(s)
        s, e = s[order], e[order]
        cs, ce = int(s[0]), int(e[0])
        for i in range(1, len(s)):
            if s[i] <= ce:
                ce = max(ce, int(e[i]))
            else:
                rows.append((str(c), cs, ce)); cs, ce = int(s[i]), int(e[i])
        rows.append((str(c), cs, ce))
    return pd.DataFrame(rows, columns=[chrom_col, "start", "end"])


def peak_signal_matrix(
    peak_files,
    *,
    value_col: str = "signalValue",
    chrom_col: str = "chrom",
    agg: str = "max",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a consensus-region x sample signal matrix from per-sample
    narrowPeak files — the quantification path when BAMs / bigWigs are not
    available.

    Merges every sample's peaks into one consensus interval set, then for
    each consensus region records each sample's per-peak signal (the
    narrowPeak ``signalValue`` fold-enrichment by default); a region a
    sample did not call gets 0. ``log2`` ratios of the resulting signal
    between conditions give a quantitative differential-occupancy readout —
    far more informative than a peak presence/absence overlap.

    Arguments:
        peak_files: mapping ``{sample_name: narrowPeak_path}``.
        value_col: which narrowPeak column to read as the per-peak signal —
            ``'signalValue'`` (fold-enrichment, default), ``'score'``,
            ``'pValue'`` or ``'qValue'``.
        chrom_col: chromosome column name in the returned ``regions`` table.
        agg: how to combine when several of one sample's peaks fall inside a
            consensus region — ``'max'`` (default), ``'sum'`` or ``'mean'``.

    Returns:
        ``(signal, regions)``. ``signal`` is a DataFrame
        (consensus regions x samples) of per-region signal; ``regions`` is
        the matching consensus interval table (``chrom`` / ``start`` /
        ``end``) with the same row index, ready for :func:`annotate_peaks`.

    Example:
        >>> signal, regions = epi.tl.peak_signal_matrix(
        ...     {'ctrl1': 'c1.narrowPeak', 'trt1': 't1.narrowPeak'})
        >>> import numpy as np
        >>> log2fc = np.log2((signal[['trt1']].mean(1) + 1) /
        ...                  (signal[['ctrl1']].mean(1) + 1))
    """
    _NP_COLS = ["chrom", "start", "end", "name", "score", "strand",
                "signalValue", "pValue", "qValue", "peak"]
    if value_col not in _NP_COLS[4:]:
        raise ValueError(f"value_col must be one of {_NP_COLS[4:]}")
    if not isinstance(peak_files, dict):
        raise TypeError("peak_files must be a {sample: path} mapping")
    try:
        aggfun = {"max": np.max, "sum": np.sum, "mean": np.mean}[agg]
    except KeyError:
        raise ValueError("agg must be 'max', 'sum' or 'mean'")

    frames = {}
    all_peaks = []
    for sample, path in peak_files.items():
        df = pd.read_csv(path, sep="\t", header=None, comment="#",
                         names=_NP_COLS, usecols=range(10))
        df = df[["chrom", "start", "end", value_col]].copy()
        df["chrom"] = df["chrom"].astype(str)
        df["start"] = df["start"].astype(np.int64)
        df["end"] = df["end"].astype(np.int64)
        frames[str(sample)] = df
        all_peaks.append(df[["chrom", "start", "end"]])

    merged = _merge_intervals_table(pd.concat(all_peaks, ignore_index=True))
    merged = merged.reset_index(drop=True)               # 0..N-1 positions

    # position lookup per chrom: sorted merged-region starts/ends.
    merged_by_chrom = {}
    for c, g in merged.groupby("chrom", sort=False):
        merged_by_chrom[c] = (g["start"].to_numpy(), g["end"].to_numpy(),
                              g.index.to_numpy())

    sig = np.zeros((len(merged), len(frames)), dtype=float)
    for col_j, (sample, df) in enumerate(frames.items()):
        acc = {}                                          # region_pos -> [values]
        for c, g in df.groupby("chrom", sort=False):
            ref = merged_by_chrom.get(c)
            if ref is None:
                continue
            mstart, mend, mpos = ref
            for pk in g.itertuples(index=False):
                # each input peak lies wholly inside exactly one merged region
                j = int(np.searchsorted(mstart, pk.start, side="right")) - 1
                if 0 <= j < len(mstart) and mstart[j] <= pk.start <= mend[j]:
                    acc.setdefault(int(mpos[j]), []).append(
                        float(getattr(pk, value_col)))
        for pos, vals in acc.items():
            sig[pos, col_j] = float(aggfun(vals))

    region_ids = [f"region_{i}" for i in range(len(merged))]
    signal = pd.DataFrame(sig, index=region_ids, columns=list(frames))
    regions = merged.rename(columns={"chrom": chrom_col})
    regions.index = region_ids
    return signal, regions


__all__ = [
    "differential_peaks",
    "count_reads_in_peaks",
    "peak_signal_matrix",
]
