"""Peak merge utility — re-exports the pure-Python implementation from
:mod:`epione.single._call_peaks`. Accepts per-sample peak dicts in any of
the common formats (pandas / polars / bare DataFrame) and normalises
to the pandas pipeline used by the merger.
"""
from __future__ import annotations

import pandas as pd

from ._genome import Genome
from ..single._call_peaks import merge_peaks as _merge_peaks_impl


def merge_peaks(
    peaks,
    chrom_sizes: "dict[str, int] | Genome",
    half_width: int = 250,
) -> pd.DataFrame:
    """Merge peaks from different groups into a non-overlapping peak set.

    For each peak, re-centre at its summit and expand by ``half_width``
    on each side; walk by descending significance and drop overlaps.

    Parameters
    ----------
    peaks
        Either a dict ``{group: DataFrame}`` (one entry per pseudobulk)
        or a single DataFrame (treated as ``sample0``).
    chrom_sizes
        Chromosome sizes, or a :class:`Genome` instance.
    half_width
        Half-width of the merged peaks (final peaks have width
        ``2 * half_width``).

    Returns
    -------
    pd.DataFrame
        Non-overlapping peak set with ``chrom / start / end / Peaks``
        columns (``Peaks`` is the ``"chrN:s-e"`` label).
    """
    # Normalise the input: accept pandas / polars / single DF.
    try:
        import polars as pl
        _pl_DataFrame = pl.DataFrame
    except Exception:
        pl = None
        _pl_DataFrame = ()

    if isinstance(peaks, pd.DataFrame) or (pl is not None and isinstance(peaks, _pl_DataFrame)):
        peaks = {"sample0": peaks}
    elif not isinstance(peaks, dict):
        raise TypeError(
            "peaks must be dict[str, DataFrame] or a single DataFrame;"
            f" received {type(peaks).__name__}"
        )
    # Convert any polars DataFrames to pandas.
    normalised = {}
    for k, v in peaks.items():
        if pl is not None and isinstance(v, _pl_DataFrame):
            normalised[k] = v.to_pandas()
        else:
            normalised[k] = v

    required = {"chrom", "start", "end"}
    for k, df in normalised.items():
        if df is None or len(df) == 0:
            continue
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"sample {k} missing required columns: {sorted(missing)}"
            )

    return _merge_peaks_impl(normalised, chrom_sizes=chrom_sizes, half_width=half_width)
