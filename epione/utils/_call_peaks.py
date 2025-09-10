import snapatac2._snapatac2 as _snapatac2

from ._genome import Genome
import pandas as pd



def merge_peaks(
    peaks,
    chrom_sizes: dict[str, int] | Genome,
    half_width: int = 250,
):
    """Merge peaks from different groups.

    Merge peaks from different groups. It is typically used to merge
    results from :func:`~snapatac2.tools.macs3`.

    This function initially expands the summits of identified peaks by `half_width`
    on both sides. Following this expansion, it addresses the issue of overlapping
    peaks through an iterative process. The procedure begins by prioritizing the
    most significant peak, determined by the smallest p-value. This peak is retained,
    and any peak that overlaps with it is excluded. Subsequently, the same method
    is applied to the next most significant peak. This iteration continues until
    all peaks have been evaluated, resulting in a final list of non-overlapping
    peaks, each with a fixed width determined by the initial extension.

    Parameters
    ----------
    peaks
        Peak information from different groups.
    chrom_sizes
        Chromosome sizes. If a :class:`~snapatac2.genome.Genome` is provided,
        chromosome sizes will be obtained from the genome.
    half_width
        Half width of the merged peaks.

    Returns
    -------
    'polars.DataFrame'
        A dataframe with merged peaks.

    See Also
    --------
    macs3
    """
    import pandas as pd
    import polars as pl
    # 1) chrom_sizes: supports both Genome object and direct dictionary
    chrom_sizes = getattr(chrom_sizes, "chrom_sizes", chrom_sizes)

    # 2) peaks: supports both dict[str, (pd.DataFrame|pl.DataFrame)] and single DataFrame
    if isinstance(peaks, (pd.DataFrame, pl.DataFrame)):
        # single sample: wrap in dict
        peaks = {"sample0": peaks}
    elif not isinstance(peaks, dict):
        # common misuse (e.g. Series) give clear error
        raise TypeError(
            "peaks must be dict[str, DataFrame] or single DataFrame/Polars DataFrame;"
            f"received type: {type(peaks)}"
        )

    # 3) convert pandas to polars for each sample
    peaks = {
        k: (pl.from_pandas(v) if isinstance(v, pd.DataFrame) else v)
        for k, v in peaks.items()
    }

    # 4) basic column check (optional, but useful)
    required = {"chrom", "start", "end"}
    for k, df in peaks.items():
        cols = set(df.columns)
        missing = required - cols
        if missing:
            raise ValueError(f"Sample {k} is missing required columns: {sorted(missing)}")

    # 5) call underlying merge
    return _snapatac2.py_merge_peaks(peaks, chrom_sizes, half_width)