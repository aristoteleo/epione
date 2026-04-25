"""sc-Hi-C plotting — cell embedding scatter + per-cell contact heatmap.

These are minimal companion plots; for richer scanpy-style scatter
(legend handling, dotplot, etc.) feed the AnnData from
:func:`epione.sc_hic.embedding` directly into scanpy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def plot_embedding(
    adata,
    *,
    basis: str = "X_pca",
    components: Tuple[int, int] = (1, 2),
    color: Optional[str] = None,
    cmap: str = "tab10",
    s: float = 16.0,
    figsize: Tuple[float, float] = (5.0, 4.0),
    title: Optional[str] = None,
    ax=None,
):
    """Scatter cells in a chosen embedding (PCA / UMAP / TSNE).

    Arguments:
        adata: AnnData from :func:`epione.sc_hic.embedding`. Must
            contain ``adata.obsm[basis]``.
        basis: key into ``adata.obsm`` — defaults to ``X_pca``. Use
            ``X_umap`` after ``sc.tl.umap``.
        components: 1-indexed component pair to plot (matching scanpy's
            convention).
        color: ``adata.obs`` column to colour by. Categorical → discrete
            ``cmap``; numeric → continuous (overrides ``cmap`` if it's
            categorical-only like ``tab10``).
        cmap, s, figsize, title, ax: cosmetic.

    Returns:
        ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    if basis not in adata.obsm:
        raise KeyError(
            f"adata.obsm[{basis!r}] not found. Run "
            "epione.sc_hic.embedding() (sets X_pca) or scanpy's "
            "sc.tl.umap (sets X_umap) first."
        )
    coords = np.asarray(adata.obsm[basis])
    i, j = components[0] - 1, components[1] - 1
    x, y = coords[:, i], coords[:, j]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if color is None or color not in adata.obs.columns:
        ax.scatter(x, y, s=s, c="#3a6eb3", alpha=0.85, edgecolors="white",
                   linewidths=0.4)
    else:
        col = adata.obs[color]
        if str(col.dtype) == "category" or col.dtype == object:
            cats = list(col.astype("category").cat.categories)
            cm = colormaps.get_cmap(cmap)
            for k, cat in enumerate(cats):
                m = (col == cat).values
                ax.scatter(x[m], y[m],
                           s=s, color=cm(k % cm.N), label=str(cat),
                           alpha=0.85, edgecolors="white", linewidths=0.4)
            ax.legend(frameon=False, fontsize=8, loc="best",
                      title=color, title_fontsize=8)
        else:
            sc = ax.scatter(x, y, s=s, c=col.values,
                            cmap="viridis", alpha=0.85,
                            edgecolors="white", linewidths=0.4)
            fig.colorbar(sc, ax=ax, label=color, shrink=0.8)

    label_basis = basis.replace("X_", "").upper()
    ax.set_xlabel(f"{label_basis} {components[0]}")
    ax.set_ylabel(f"{label_basis} {components[1]}")
    ax.set_title(title or f"sc-Hi-C cells ({label_basis})")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return fig, ax


def plot_cell_contacts(
    adata,
    cell_id: str,
    *,
    chromosome: str,
    use_imputed: bool = True,
    log: bool = True,
    cmap: str = "YlOrRd",
    figsize: Tuple[float, float] = (5.0, 4.5),
    ax=None,
):
    """Per-cell contact heatmap for a single chromosome.

    Useful for sanity-checking imputation: the raw matrix should look
    speckled (sparse contacts), the imputed one should show a clear
    diagonal + off-diagonal density.

    Arguments:
        adata: AnnData from :func:`epione.sc_hic.load_cool_collection`.
        cell_id: row in ``adata.obs_names``.
        chromosome: e.g. ``'chr1'``.
        use_imputed: read the imputed ``.npz`` (requires
            :func:`impute_cells` to have run). ``False`` reads raw counts
            from the original ``.cool`` for comparison.
        log, cmap, figsize, ax: cosmetic.

    Returns:
        ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    if cell_id not in adata.obs_names:
        raise KeyError(f"cell_id {cell_id!r} not in adata.obs_names")

    if use_imputed:
        info = adata.uns.get("hic", {})
        imputed_dir = info.get("imputed_dir")
        if imputed_dir is None:
            raise ValueError(
                "adata.uns['hic']['imputed_dir'] not set — call with "
                "use_imputed=False or run impute_cells first"
            )
        z = np.load(Path(imputed_dir) / f"{cell_id}.npz")
        if chromosome not in z.files:
            raise KeyError(
                f"chromosome {chromosome!r} not in imputed file for "
                f"{cell_id}; available: {sorted(z.files)}"
            )
        mat = z[chromosome]
    else:
        import cooler
        cool_path = adata.obs.loc[cell_id, "cool_path"]
        clr = cooler.Cooler(str(cool_path))
        mat = np.asarray(clr.matrix(balance=False).fetch(chromosome),
                         dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    finite = mat[np.isfinite(mat) & (mat > 0)]
    vmin = float(np.quantile(finite, 0.01)) if finite.size else 0.0
    vmax = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    if log and vmax > 0:
        vmin = max(vmin, 1e-6)
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))
    else:
        norm = Normalize(vmin=vmin, vmax=vmax if vmax > 0 else 1.0)

    img = ax.imshow(mat, cmap=cmap, norm=norm, interpolation="none")
    ax.set_title(
        f"{cell_id} — {chromosome}"
        + (" (imputed)" if use_imputed else " (raw)")
    )
    ax.set_xlabel("bin")
    ax.set_ylabel("bin")
    fig.colorbar(img, ax=ax, shrink=0.8,
                 label="log contact" if log else "contact")
    return fig, ax
