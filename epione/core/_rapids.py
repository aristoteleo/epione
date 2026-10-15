"""Optional dispatch of graph steps to rapids-singlecell.

``backend="rapids"`` on :func:`epione.pp.neighbors`, :func:`epione.tl.umap`
and :func:`epione.tl.clusters` (``method="leiden"``) hands the call to
`rapids-singlecell <https://rapids-singlecell.readthedocs.io>`_, which runs
on cuML / cuGraph / cuVS. It is never imported unless asked for, and it is
not a dependency of Epione: install it yourself (``pip install
"rapids-singlecell[rapids-cu12]"``; the RAPIDS wheels need glibc >= 2.28).
The torch backend (``device="cuda"``) needs none of this.
"""
from __future__ import annotations

from types import MappingProxyType


def get_rsc():
    """Import :mod:`rapids_singlecell` or raise with install instructions."""
    try:
        import rapids_singlecell as rsc
    except ImportError as e:
        raise ImportError(
            "backend='rapids' needs rapids-singlecell, which is not installed "
            "(or could not import its RAPIDS dependencies: cupy, cuml, cugraph). "
            "Install it with `pip install \"rapids-singlecell[rapids-cu12]\"` "
            "on a Linux system with glibc >= 2.28 and a CUDA 12 driver, or use "
            "backend='epione' (optionally with device='cuda', which only needs "
            "torch). See the 'GPU acceleration' page of the Epione docs."
        ) from e
    return rsc


def rapids_neighbors(adata, *, n_neighbors=15, n_pcs=None, use_rep=None,
                     metric="euclidean", metric_kwds=MappingProxyType({}),
                     random_state=0, key_added=None, copy=False):
    """:func:`rapids_singlecell.pp.neighbors` with exact (brute-force) search.

    rapids-singlecell stores each cell's self-match as an explicit zero in
    ``.obsp['distances']``; it is dropped so the distance graph has the same
    layout as :func:`epione.pp.neighbors` (``n_neighbors - 1`` neighbours per
    row, self excluded). Connectivities are left as computed.
    """
    rsc = get_rsc()
    out = rsc.pp.neighbors(
        adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep,
        random_state=random_state, algorithm="brute", metric=metric,
        metric_kwds=dict(metric_kwds), key_added=key_added, copy=copy,
    )
    target = out if copy else adata
    key = "distances" if key_added is None else f"{key_added}_distances"
    if target is not None and key in getattr(target, "obsp", {}):
        target.obsp[key] = _drop_self(target.obsp[key])
    return out


def _drop_self(D):
    """Remove stored diagonal entries of a sparse distance graph."""
    import scipy.sparse as sp
    D = sp.csr_matrix(D)
    C = D.tocoo()
    keep = C.row != C.col
    return sp.csr_matrix((C.data[keep], (C.row[keep], C.col[keep])), shape=D.shape)


def rapids_umap(adata, *, min_dist=0.5, spread=1.0, n_components=2,
                maxiter=None, alpha=1.0, negative_sample_rate=5,
                init_pos="spectral", random_state=0, a=None, b=None,
                key_added=None, neighbors_key="neighbors", copy=False):
    """:func:`rapids_singlecell.tl.umap` on the existing neighbour graph."""
    rsc = get_rsc()
    if not isinstance(init_pos, str) or init_pos not in ("spectral", "random", "auto"):
        raise ValueError(
            "backend='rapids' supports init_pos='spectral' or 'random' only")
    return rsc.tl.umap(
        adata, min_dist=min_dist, spread=spread, n_components=n_components,
        maxiter=maxiter, alpha=alpha, negative_sample_rate=negative_sample_rate,
        init_pos=init_pos, random_state=random_state, a=a, b=b,
        key_added=key_added,
        neighbors_key=None if neighbors_key == "neighbors" else neighbors_key,
        copy=copy,
    )


def rapids_leiden(adata, *, resolution=1.0, random_state=0, key_added="leiden",
                  restrict_to=None, adjacency=None, n_iterations=-1,
                  use_weights=True, neighbors_key=None, obsp=None, copy=False,
                  **_ignored):
    """:func:`rapids_singlecell.tl.leiden` (cuGraph) on the existing graph.

    cuGraph's Leiden counts ``n_iterations`` as a maximum number of levels;
    ``-1`` (run to convergence in leidenalg / igraph) maps to its default 100.
    """
    rsc = get_rsc()
    n_it = 100 if (n_iterations is None or n_iterations < 0) else int(n_iterations)
    return rsc.tl.leiden(
        adata, resolution=resolution, random_state=random_state,
        restrict_to=restrict_to, key_added=key_added, adjacency=adjacency,
        n_iterations=n_it, use_weights=use_weights,
        neighbors_key=neighbors_key, obsp=obsp, copy=copy,
    )
