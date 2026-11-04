"""Spot maps for spatial ATAC / ATAC-RNA data.

Reads the scanpy / squidpy layout (``obsm["spatial"]`` in full-resolution
pixels, image and scale factors in ``uns["spatial"][library_id]``), so data
from :func:`epione.io.read_spatial_atac`, ``sc.read_visium`` or squidpy work
alike. Spots are drawn as squares (DBiT pixels) or circles (Visium) of their
physical size; without an image the map is drawn on a blank canvas with the
y axis pointing down, as on the tissue image.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["spatial", "spatial_qc"]


def _library(adata, library_id):
    lib = adata.uns.get("spatial", {}) if hasattr(adata, "uns") else {}
    if not lib:
        return None, {}
    if library_id is None:
        library_id = next(iter(lib))
    return library_id, lib.get(library_id, {})


def _values(adata, key, layer=None):
    """obs column, var name (``X`` / layer) or ``"mod:var"`` on a MuData."""
    if ":" in key and hasattr(adata, "mod"):
        mod, k = key.split(":", 1)
        return _values(adata.mod[mod], k, layer)
    if key in adata.obs.columns:
        return adata.obs[key]
    if hasattr(adata, "mod"):
        for m in adata.mod.values():
            if key in m.var_names:
                return _values(m, key, layer)
    if key in adata.var_names:
        j = adata.var_names.get_loc(key)
        X = adata.layers[layer] if layer is not None else adata.X
        col = X[:, j]
        col = col.toarray().ravel() if sp.issparse(col) else np.asarray(col).ravel()
        return pd.Series(col, index=adata.obs_names, name=key)
    raise KeyError(f"{key!r} is neither an obs column nor a feature")


def spatial(
    adata,
    color: Union[str, Sequence[str], None] = None,
    *,
    library_id: Optional[str] = None,
    img_key: Optional[str] = "auto",
    layer: Optional[str] = None,
    shape: str = "auto",
    spot_scale: float = 1.0,
    cmap: str = "magma",
    palette=None,
    vmin: Union[float, str, None] = None,
    vmax: Union[float, str, None] = "p99",
    na_color: str = "lightgrey",
    alpha: float = 1.0,
    alpha_img: float = 1.0,
    crop: bool = True,
    ncols: int = 4,
    panel_size: float = 3.6,
    title: Union[str, Sequence[str], None] = None,
    ax=None,
    show: Optional[bool] = None,
    legend_loc: str = "right margin",
):
    """Plot spot values on the tissue.

    Arguments:
        adata: AnnData or MuData with ``obsm["spatial"]``.
        color: obs column(s) or feature name(s) (gene score, peak, motif in a
            chromVAR deviation AnnData, ``"rna:Gene"`` on a MuData). ``None``
            draws the spots only.
        library_id: key in ``uns["spatial"]`` (default: the first).
        img_key: ``"hires"``, ``"lowres"``, ``None`` (no image) or ``"auto"``
            (hires, else lowres, else none).
        shape: ``"square"`` (DBiT), ``"circle"`` (Visium) or ``"auto"``
            (square when ``obs`` has a DBiT-style full grid, i.e. no hexagonal
            offset).
        spot_scale: multiply the spot side / diameter.
        alpha, alpha_img: opacity of the spots and of the tissue image.
        vmin, vmax: limits for continuous values; ``"pNN"`` is a percentile.
        crop: zoom to the spots.
        ncols, panel_size: layout when several ``color`` keys are given.
        ax: draw a single panel into this axis.

    Returns:
        The list of axes when ``show`` is ``False``; otherwise shows the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle, Rectangle

    keys = [color] if (color is None or isinstance(color, str)) else list(color)
    titles = [title] if isinstance(title, str) or title is None else list(title)
    if len(titles) == 1 and len(keys) > 1:
        titles = titles * len(keys)
    lid, lib = _library(adata, library_id)
    sf = lib.get("scalefactors", {}) if lib else {}
    imgs = lib.get("images", {}) if lib else {}
    if img_key == "auto":
        img_key = "hires" if "hires" in imgs else ("lowres" if "lowres" in imgs else None)
    img = imgs.get(img_key) if img_key else None
    scale = float(sf.get(f"tissue_{img_key}_scalef", 1.0)) if img is not None else 1.0
    xy = np.asarray(adata.obsm["spatial"], dtype=float) * scale

    if "spot_diameter_fullres" in sf:
        diam = float(sf["spot_diameter_fullres"]) * scale
    else:
        from sklearn.neighbors import NearestNeighbors

        d, _ = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy[: min(2000, len(xy))])
        diam = float(np.median(d[:, 1]))
    if shape == "auto":
        shape = "square"
        if "array_col" in adata.obs and "array_row" in adata.obs:
            r = adata.obs["array_row"].to_numpy(int)
            c = adata.obs["array_col"].to_numpy(int)
            if len(r) and np.all((r + c) % 2 == (r[0] + c[0]) % 2) and len(set(c)) > 2:
                shape = "circle"  # Visium hex lattice: row + col keeps one parity
    side = diam * spot_scale

    if ax is not None:
        axes = [ax]
        fig = ax.figure
        if len(keys) > 1:
            raise ValueError("pass a single `color` key with `ax`")
    else:
        n = len(keys)
        nc = min(ncols, n)
        nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(panel_size * nc + (1.2 if n else 0), panel_size * nr),
                                squeeze=False)
        axes = list(axs.ravel())
        for a in axes[n:]:
            a.set_visible(False)
        axes = axes[:n]

    for a, key, tt in zip(axes, keys, titles if len(titles) == len(keys) else [None] * len(keys)):
        if img is not None:
            a.imshow(img, alpha=alpha_img)
        if shape == "square":
            patches = [Rectangle((x - side / 2, y - side / 2), side, side) for x, y in xy]
        else:
            patches = [Circle((x, y), side / 2) for x, y in xy]
        pc = PatchCollection(patches, linewidths=0, alpha=alpha)
        if key is None:
            pc.set_facecolor("tab:blue")
        else:
            v = _values(adata, key, layer)
            if isinstance(v.dtype, pd.CategoricalDtype) or v.dtype == object or v.dtype == bool:
                cat = pd.Categorical(v.astype(str) if v.dtype == bool else v)
                cats = list(cat.categories)
                ukey = f"{key}_colors"
                if palette is not None:
                    cols = list(palette.values()) if isinstance(palette, dict) else list(palette)
                elif ukey in getattr(adata, "uns", {}) and len(adata.uns[ukey]) >= len(cats):
                    cols = list(adata.uns[ukey])
                else:
                    base = plt.get_cmap("tab20" if len(cats) > 10 else "tab10")
                    cols = [base(i % base.N) for i in range(len(cats))]
                lut = {c: cols[i % len(cols)] for i, c in enumerate(cats)}
                pc.set_facecolor([lut[c] if c in lut else na_color for c in cat])
                handles = [plt.Line2D([], [], marker="s", ls="", color=lut[c], label=str(c)) for c in cats]
                if legend_loc == "right margin":
                    a.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
                             fontsize=7, ncol=1 if len(cats) <= 16 else 2, handletextpad=0.2)
            else:
                vals = v.to_numpy(dtype=float)

                def _lim(q, default):
                    if isinstance(q, str) and q.startswith("p"):
                        return float(np.nanpercentile(vals, float(q[1:])))
                    return default if q is None else float(q)

                lo, hi = _lim(vmin, np.nanmin(vals)), _lim(vmax, np.nanmax(vals))
                pc.set_array(np.ma.masked_invalid(vals))
                pc.set_cmap(cmap)
                pc.set_clim(lo, hi)
                fig.colorbar(pc, ax=a, fraction=0.046, pad=0.02, shrink=0.8)
        a.add_collection(pc)
        pad = side
        if crop or img is None:
            a.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
            a.set_ylim(xy[:, 1].max() + pad, xy[:, 1].min() - pad)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        a.set_title(tt if tt is not None else (key or (lid or "spots")), fontsize=10)
    if ax is None:
        fig.tight_layout()
    if show is False:
        return axes
    if show:
        plt.show()
    return axes


def spatial_qc(
    adata,
    keys: Sequence[str] = ("n_fragment", "tsse", "frac_mito", "frac_dup"),
    *,
    log1p: Sequence[str] = ("n_fragment",),
    ncols: int = 4,
    **kwargs,
):
    """Per-spot QC maps: fragments per spot, TSS enrichment, mito / duplicate fractions.

    Missing columns are skipped; columns in ``log1p`` are drawn as
    ``log10(1 + x)``. Extra arguments go to :func:`spatial`.
    """
    present = [k for k in keys if k in adata.obs.columns]
    if not present:
        raise KeyError(f"none of {list(keys)} in obs; run epi.pp.tsse / import with epi.io.read_spatial_atac")
    tmp_cols = []
    for k in present:
        if k in log1p:
            c = f"log10(1+{k})"
            adata.obs[c] = np.log10(1 + adata.obs[k].astype(float))
            tmp_cols.append(c)
    show_keys = [f"log10(1+{k})" if k in log1p else k for k in present]
    try:
        return spatial(adata, show_keys, ncols=ncols, **kwargs)
    finally:
        for c in tmp_cols:
            del adata.obs[c]
