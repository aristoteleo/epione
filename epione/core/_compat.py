"""Compatibility helpers for running snapATAC2-backed AnnData through
code that was written against ``anndata.AnnData``.

snapATAC2 stores ``obs`` / ``var`` as a ``PyDataFrameElem`` whose API is
similar but not identical to pandas — ``.index`` / ``.columns`` / ``.copy``
don't exist on the bare object, and scanpy / seaborn plotters that
expect a pandas DataFrame crash. These helpers convert the on-disk
columns into a fresh in-memory pandas DataFrame without touching the
backend.
"""
from __future__ import annotations

from typing import Any


def obs_to_pandas(adata: Any):
    """Return ``adata.obs`` as a pandas DataFrame, on both plain
    ``anndata.AnnData`` and snapATAC2-backed AnnData.
    """
    import pandas as pd

    obs = adata.obs
    if isinstance(obs, pd.DataFrame):
        return obs
    # snapATAC2's backend: ``obs[:]`` materialises a sub-element that has
    # ``to_pandas()``.
    try:
        sliced = obs[:]
    except Exception:
        sliced = obs
    if hasattr(sliced, "to_pandas"):
        return sliced.to_pandas()
    if isinstance(sliced, pd.DataFrame):
        return sliced
    # Last resort: try to construct a DataFrame from whatever attribute
    # surface we can find.
    try:
        return pd.DataFrame(dict(sliced))
    except Exception as e:
        raise TypeError(
            f"don't know how to convert adata.obs of type {type(obs).__name__} "
            f"to pandas ({e})"
        )


def var_to_pandas(adata: Any):
    """Return ``adata.var`` as a pandas DataFrame — mirror of :func:`obs_to_pandas`."""
    import pandas as pd

    var = adata.var
    if isinstance(var, pd.DataFrame):
        return var
    try:
        sliced = var[:]
    except Exception:
        sliced = var
    if hasattr(sliced, "to_pandas"):
        return sliced.to_pandas()
    if isinstance(sliced, pd.DataFrame):
        return sliced
    try:
        return pd.DataFrame(dict(sliced))
    except Exception as e:
        raise TypeError(
            f"don't know how to convert adata.var of type {type(var).__name__} "
            f"to pandas ({e})"
        )
