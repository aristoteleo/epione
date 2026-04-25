"""Generic save/load helpers — pickle with cloudpickle fallback, and
AnnData ``.write`` / ``read_h5ad`` when the file has an ``.h5ad``
extension. Modelled on ``omicverse.utils.save`` / ``load`` for
convenient cache-and-resume patterns in tutorials.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Optional

from epione.core import console


def _is_h5ad(path: str) -> bool:
    return str(path).lower().endswith((".h5ad", ".h5"))


def save(obj: Any, path: str | Path) -> None:
    """Persist ``obj`` to ``path`` — AnnData via ``write_h5ad`` when
    the extension is ``.h5ad`` / ``.h5``, otherwise pickle (with a
    ``cloudpickle`` fallback for unpicklable objects such as lambdas).

    Parameters
    ----------
    obj
        Anything: AnnData, DataFrame, dict, ndarray, fitted model, …
    path
        Target path. Parent directory is created if missing.

    Examples
    --------
    >>> epi.utils.save(peak_mat, 'cache/peak_mat.h5ad')       # h5ad
    >>> epi.utils.save(macs3_dict, 'cache/macs3.pkl')         # pickle
    >>> epi.utils.save(sklearn_model, 'cache/model.pkl')      # pickle
    """
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    if _is_h5ad(str(path)):
        # AnnData / AnnDataOOM: use its own writer.
        try:
            obj.write(str(path))
            console.level2(f"saved AnnData → {path}")
            return
        except Exception:
            # Fall through to pickle if it's not actually an AnnData.
            pass

    try:
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
        console.level2(f"saved pickle → {path}")
    except Exception:
        import cloudpickle
        with open(path, "wb") as fh:
            cloudpickle.dump(obj, fh)
        console.level2(f"saved cloudpickle → {path}")


def load(path: str | Path, *, backend: Optional[str] = None) -> Any:
    """Inverse of :func:`save`.

    ``backend`` may be ``"h5ad"``, ``"pickle"``, ``"cloudpickle"`` or
    ``None`` (auto-detect by extension / payload).

    Examples
    --------
    >>> peak_mat = epi.utils.load('cache/peak_mat.h5ad')
    >>> macs3   = epi.utils.load('cache/macs3.pkl')
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    ext_is_h5ad = _is_h5ad(str(path))
    if backend is None:
        backend = "h5ad" if ext_is_h5ad else "pickle"

    if backend == "h5ad":
        import anndata as ad
        obj = ad.read_h5ad(str(path))
        console.level2(f"loaded AnnData ← {path}  {obj.shape}")
        return obj

    if backend == "pickle":
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            console.level2(f"loaded pickle ← {path}")
            return obj
        except Exception:
            backend = "cloudpickle"

    if backend == "cloudpickle":
        import cloudpickle
        with open(path, "rb") as fh:
            obj = cloudpickle.load(fh)
        console.level2(f"loaded cloudpickle ← {path}")
        return obj

    raise ValueError(f"invalid backend: {backend!r}")


def cached(path: str | Path, *, backend: Optional[str] = None):
    """Decorator: persist the return value of a zero-arg function to
    ``path`` on first call, re-load on subsequent calls.

    Examples
    --------
    >>> @epi.utils.cached('cache/peak_mat.h5ad')
    ... def build_peak_mat():
    ...     return epi.pp.make_peak_matrix(data, use_rep=merged_peaks['Peaks'])
    ...
    >>> peak_mat = build_peak_mat()   # first call builds + saves
    >>> peak_mat = build_peak_mat()   # subsequent calls load from disk
    """
    def _decorator(fn):
        def _wrapped(*a, **kw):
            if Path(path).exists():
                return load(path, backend=backend)
            out = fn(*a, **kw)
            save(out, path)
            return out
        _wrapped.__wrapped__ = fn
        _wrapped.__name__ = fn.__name__
        return _wrapped
    return _decorator
