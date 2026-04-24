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
    backend: Literal["pydeseq2", "edgepy"] = "pydeseq2",
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
        backend: ``'pydeseq2'`` or ``'edgepy'``.
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
    if backend == "pydeseq2":
        res = _run_pydeseq2(
            counts_df, meta, design, contrast,
            alpha=alpha, n_cpus=n_cpus, quiet=quiet,
            **backend_kwargs,
        )
    elif backend == "edgepy":
        res = _run_edgepy(
            counts_df, meta, design, contrast,
            quiet=quiet, **backend_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Use 'pydeseq2' or 'edgepy'."
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


__all__ = ["differential_peaks"]
