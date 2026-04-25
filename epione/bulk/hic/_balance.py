"""Matrix balancing / correction via cooler's iterative correction (ICE).

A raw contact matrix encodes both the underlying chromatin contact
frequencies *and* per-bin systematic biases (mappability, GC,
restriction-site density). ICE iteratively rescales rows + columns
until every bin has the same marginal, so downstream analyses see
balanced contact probabilities rather than coverage artefacts.

We use :func:`cooler.balance_cooler` directly from the cooler Python
API here — it returns the per-bin weight vector, and we write it
back to the cool file's ``bins/weight`` column in place. That column
is automatically honoured by ``cooler.Cooler.matrix(balance=True)``
downstream.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union


def balance_cool(
    cool_path: Union[str, Path],
    *,
    mad_max: int = 5,
    min_nnz: int = 10,
    min_count: int = 0,
    ignore_diags: int = 2,
    tol: float = 1e-5,
    max_iters: int = 200,
    nproc: int = 4,
    force: bool = False,
) -> dict:
    """ICE-balance a ``.cool`` matrix in place.

    Writes the ``bins/weight`` column so ``Cooler.matrix(balance=True)``
    returns normalised contacts. Idempotent — if a ``weight`` column
    already exists, returns its stats without recomputing unless
    ``force=True``.

    Arguments:
        cool_path: path to the ``.cool`` written by :func:`pairs_to_cool`.
        mad_max: bins whose coverage is more than this many median-
            absolute-deviations below the median are masked. Default 5
            matches the cooler CLI default.
        min_nnz: bins with fewer than this many non-zero neighbours are
            masked (filters out unmappable regions).
        min_count: bins with fewer than this many total contacts are
            masked. Default 0 (no hard threshold).
        ignore_diags: number of diagonals (including the main diag) to
            exclude from balancing. 2 skips the main diag + its immediate
            neighbour, which are dominated by self-ligation artefacts.
        tol: iteration convergence threshold on the marginal variance.
        max_iters: hard cap on balancing iterations.
        nproc: parallel workers used by cooler.
        force: re-balance even if a ``weight`` column already exists.

    Returns:
        ``dict`` with keys ``converged``, ``n_iters``, ``n_masked``,
        ``scale``, ``var`` — the per-run diagnostics.
    """
    import numpy as np
    import cooler
    from cooler import balance_cooler

    cool_path = Path(cool_path)
    clr = cooler.Cooler(str(cool_path))

    existing = clr.bins()[:]
    if "weight" in existing.columns and not force:
        w = existing["weight"].to_numpy()
        return {
            "converged": True,
            "n_iters": 0,
            "n_masked": int(np.isnan(w).sum()),
            "scale": float(np.nanmean(w)) if np.any(~np.isnan(w)) else float("nan"),
            "var": float(np.nanvar(w)) if np.any(~np.isnan(w)) else float("nan"),
            "reused_existing": True,
        }

    bias, stats = balance_cooler(
        clr,
        mad_max=mad_max,
        min_nnz=min_nnz,
        min_count=min_count,
        ignore_diags=ignore_diags,
        tol=tol,
        max_iters=max_iters,
        map=map,  # sequential; pass multiprocessing.Pool.map for parallel
    )

    # Write the weight vector back into the file's bins table.
    import h5py
    with h5py.File(str(cool_path), "r+") as f:
        if "weight" in f["bins"]:
            del f["bins/weight"]
        f.create_dataset("bins/weight", data=bias)
    return {
        "converged": bool(stats.get("converged", True)),
        "n_iters": int(stats.get("n_iters", 0) or 0),
        "n_masked": int((bias != bias).sum()),  # NaN count
        "scale": float(stats.get("scale", float("nan"))),
        "var": float(stats.get("var", float("nan"))),
        "reused_existing": False,
    }
