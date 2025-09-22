"""Standalone embedding visualization function with removed relative dependencies."""

from __future__ import annotations

import collections.abc as cabc
from copy import copy
from numbers import Integral
from itertools import combinations, product
from typing import (
    Collection, Union, Optional, Sequence, Any, Mapping, List, Tuple, Literal
)
from warnings import warn
from functools import partial

import numpy as np
import pandas as pd
from anndata import AnnData
from cycler import Cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import pyplot as pl, colors, colormaps
from matplotlib import rcParams, patheffects
from matplotlib.colors import Colormap, Normalize, to_hex, to_rgba, is_color_like
import matplotlib


# Type definitions
ColorLike = Union[str, Tuple[float, ...]]
VBound = Union[float, str, callable]
_FontSize = Union[int, float, str]
_FontWeight = Union[int, str]


def embedding(
    adata: AnnData,
    basis: str,
    *,
    color: Union[str, Sequence[str], None] = None,
    gene_symbols: Optional[str] = None,
    use_raw: Optional[bool] = None,
    sort_order: bool = True,
    edges: bool = False,
    edges_width: float = 0.1,
    edges_color: Union[str, Sequence[float], Sequence[str]] = 'grey',
    neighbors_key: Optional[str] = None,
    arrows: bool = False,
    arrows_kwds: Optional[Mapping[str, Any]] = None,
    groups: Optional[str] = None,
    components: Union[str, Sequence[str]] = None,
    dimensions: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
    layer: Optional[str] = None,
    projection: Literal['2d', '3d'] = '2d',
    scale_factor: Optional[float] = None,
    color_map: Union[Colormap, str, None] = None,
    cmap: Union[Colormap, str, None] = None,
    palette: Union[str, Sequence[str], Cycler, None] = None,
    na_color: ColorLike = "lightgray",
    na_in_legend: bool = True,
    size: Union[float, Sequence[float], None] = None,
    frameon: Optional[bool] = None,
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = 'bold',
    legend_loc: str = 'right margin',
    legend_fontoutline: Optional[int] = None,
    colorbar_loc: Optional[str] = "right",
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    add_outline: Optional[bool] = False,
    outline_width: Tuple[float, float] = (0.3, 0.05),
    outline_color: Tuple[str, str] = ('black', 'white'),
    ncols: int = 4,
    hspace: float = 0.25,
    wspace: Optional[float] = None,
    title: Union[str, Sequence[str], None] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    marker: Union[str, Sequence[str]] = '.',
    arrow_scale: float = 6,
    arrow_width: float = 0.005,
    **kwargs,
) -> Union[Figure, Axes, None]:
    """
    Scatter plot for user specified embedding basis (e.g. umap, pca, tsne).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    basis : str
        Name of the obsm basis to use (e.g., 'umap', 'pca', 'tsne')
    color : str or list of str, optional
        Keys for annotations of observations/cells or variables/genes
    gene_symbols : str, optional
        Column name in .var DataFrame for gene symbols
    use_raw : bool, optional
        Whether to use .raw attribute of adata
    sort_order : bool
        Sort order for points by color values (default: True)
    edges : bool
        Whether to draw edges of graph (default: False)
    edges_width : float
        Width of edges
    edges_color : str or sequence
        Color of edges
    neighbors_key : str, optional
        Key to use for neighbors
    arrows : bool
        Whether to draw arrows (default: False)
    arrows_kwds : dict, optional
        Keywords for arrow plotting
    groups : str, optional
        Restrict to a subset of groups
    components : str or sequence, optional
        Components to plot
    dimensions : tuple or sequence of tuples, optional
        Dimensions to plot
    layer : str, optional
        Layer to use for coloring
    projection : {'2d', '3d'}
        Projection type
    scale_factor : float, optional
        Scaling factor for spatial coordinates
    color_map, cmap : str or Colormap, optional
        Colormap for continuous variables
    palette : str, sequence, or dict, optional
        Colors to use for categorical variables
    na_color : color-like
        Color for missing values
    na_in_legend : bool
        Include missing values in legend
    size : float or array, optional
        Point size
    frameon : bool, optional
        Draw frame around plot
    legend_fontsize : float, optional
        Font size for legend
    legend_fontweight : str or int
        Font weight for legend
    legend_loc : str
        Location of legend
    legend_fontoutline : int, optional
        Font outline width for legend
    colorbar_loc : str, optional
        Location of colorbar
    vmax, vmin, vcenter : float or str, optional
        Color scale limits
    norm : Normalize, optional
        Normalization for color scale
    add_outline : bool
        Add outline to points
    outline_width : tuple of float
        Width of outline
    outline_color : tuple of str
        Color of outline
    ncols : int
        Number of columns for multi-panel plots
    hspace, wspace : float
        Spacing between subplots
    title : str or list of str, optional
        Plot title(s)
    show : bool, optional
        Show the plot
    save : bool or str, optional
        Save the plot
    ax : Axes, optional
        Matplotlib axes object
    return_fig : bool, optional
        Return figure object
    marker : str or list of str
        Marker style
        
    Returns
    -------
    Figure, Axes, or None
        Matplotlib figure or axes if show=False
    """
    # Check projection
    if projection not in ['2d', '3d']:
        raise ValueError(f"projection must be '2d' or '3d', got {projection}")
    
    # Get basis values
    basis_values = _get_basis(adata, basis)
    dimensions = _components_to_dimensions(
        components, dimensions, projection=projection, total_dims=basis_values.shape[1]
    )
    args_3d = dict(projection='3d') if projection == '3d' else {}
    
    # Figure out if we're using raw
    if use_raw is None:
        use_raw = layer is None and adata.raw is not None
    if use_raw and layer is not None:
        raise ValueError(
            f"Cannot use both a layer and the raw representation. "
            f"Was passed: use_raw={use_raw}, layer={layer}."
        )
    if use_raw and adata.raw is None:
        raise ValueError(
            "`use_raw` is set to True but AnnData object does not have raw."
        )
    
    if isinstance(groups, str):
        groups = [groups]
    
    # Handle colormap
    if color_map is not None:
        if cmap is not None:
            raise ValueError("Cannot specify both `color_map` and `cmap`.")
        else:
            cmap = color_map
    
    if cmap is None:
        cmap = 'RdBu_r'
    
    if not isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
        cmap = copy(matplotlib.colormaps.get(cmap))
        cmap.set_bad(na_color)
    
    kwargs["cmap"] = cmap
    na_color = colors.to_hex(na_color, keep_alpha=True)
    
    if 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = 'none'
    
    # Turn arguments into lists
    color = [color] if isinstance(color, str) or color is None else list(color)
    marker = [marker] if isinstance(marker, str) else list(marker)
    
    if title is not None:
        title = [title] if isinstance(title, str) else list(title)
    
    # Turn vmax, vmin, vcenter, norm into sequences
    if isinstance(vmax, str) or not isinstance(vmax, cabc.Sequence):
        vmax = [vmax]
    if isinstance(vmin, str) or not isinstance(vmin, cabc.Sequence):
        vmin = [vmin]
    if isinstance(vcenter, str) or not isinstance(vcenter, cabc.Sequence):
        vcenter = [vcenter]
    if isinstance(norm, Normalize) or not isinstance(norm, cabc.Sequence):
        norm = [norm]
    
    # Handle size
    if 's' in kwargs and size is None:
        size = kwargs.pop('s')
    if size is not None:
        if (
            size is not None
            and isinstance(size, (cabc.Sequence, pd.Series, np.ndarray))
            and len(size) == adata.shape[0]
        ):
            size = np.array(size, dtype=float)
    else:
        size = 120000 / adata.shape[0]
    
    # Setup layout
    if wspace is None:
        wspace = 0.75 / rcParams['figure.figsize'][0] + 0.02
    
    if components is not None:
        color, dimensions = list(zip(*product(color, dimensions)))
    
    color, dimensions, marker = _broadcast_args(color, dimensions, marker)
    
    # Create figure and axes
    if (
        not isinstance(color, str)
        and isinstance(color, cabc.Sequence)
        and len(color) > 1
    ) or len(dimensions) > 1:
        if ax is not None:
            raise ValueError(
                "Cannot specify `ax` when plotting multiple panels"
            )
        fig, grid = _panel_grid(hspace, wspace, ncols, len(color))
    else:
        grid = None
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111, **args_3d)
    
    # Main plotting loop
    axs = []
    
    for count, (value_to_plot, dims) in enumerate(zip(color, dimensions)):
        color_source_vector = _get_color_source_vector(
            adata,
            value_to_plot,
            layer=layer,
            use_raw=use_raw,
            gene_symbols=gene_symbols,
            groups=groups,
        )
        color_vector, categorical = _color_vector(
            adata,
            value_to_plot,
            color_source_vector,
            palette=palette,
            na_color=na_color,
        )
        
        # Order points
        order = slice(None)
        if sort_order is True and value_to_plot is not None:
            if categorical is False:
                # For continuous values, sort by value
                if hasattr(color_vector, '__len__'):
                    arr = np.asarray(color_vector)
                    if np.issubdtype(arr.dtype, np.number):
                        order = np.argsort(-arr, kind="stable")[::-1]
            else:
                # For categorical, put NaN values at the bottom
                order = np.argsort(~pd.isnull(color_source_vector), kind="stable")
        
        # Apply ordering
        if isinstance(size, np.ndarray):
            size = np.array(size)[order]
        color_source_vector = color_source_vector[order]
        color_vector = color_vector[order]
        coords = basis_values[:, dims][order, :]
        
        # Get axes for this plot
        if grid:
            ax = pl.subplot(grid[count], **args_3d)
            axs.append(ax)
        if frameon ==False:
            ax.axis('off')
            #from ..pl._single import add_arrow
            add_arrow(ax,adata,basis,fontsize=legend_fontsize,arrow_scale=arrow_scale,arrow_width=arrow_width)
        elif frameon == 'small':
            ax.axis('off')
            #from ..pl._single import add_arrow
            add_arrow(ax,adata,basis,fontsize=legend_fontsize,arrow_scale=arrow_scale,arrow_width=arrow_width)
        
        # Handle frame
        if frameon == False:
            ax.axis('off')
        
        # Set title
        if title is None:
            if value_to_plot is not None:
                ax.set_title(value_to_plot)
            else:
                ax.set_title('')
        else:
            try:
                ax.set_title(title[count])
            except IndexError:
                print(f"Warning: Title list is shorter than number of panels. "
                      f"Using 'color' value for plot {count}.")
                ax.set_title(value_to_plot)
        
        # Handle normalization for continuous data
        if not categorical:
            vmin_float, vmax_float, vcenter_float, norm_obj = _get_vboundnorm(
                vmin, vmax, vcenter, norm, count, color_vector
            )
            normalize = _check_colornorm(
                vmin_float, vmax_float, vcenter_float, norm_obj
            )
        else:
            normalize = None
        
        # Make the scatter plot
        if projection == '3d':
            cax = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=color_vector,
                s=size,
                norm=normalize,
                marker=marker[count],
                **kwargs,
            )
        else:
            if add_outline:
                # Add outline by drawing multiple overlapping scatter plots
                bg_width, gap_width = outline_width
                point = np.sqrt(size)
                gap_size = (point + (point * gap_width) * 2) ** 2
                bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2
                bg_color, gap_color = outline_color
                
                kwargs_outline = kwargs.copy()
                kwargs_outline['edgecolor'] = 'none'
                alpha = kwargs_outline.pop('alpha', None)
                
                ax.scatter(coords[:, 0], coords[:, 1], s=bg_size, c=bg_color,
                          norm=normalize, marker=marker[count], **kwargs_outline)
                ax.scatter(coords[:, 0], coords[:, 1], s=gap_size, c=gap_color,
                          norm=normalize, marker=marker[count], **kwargs_outline)
                
                kwargs['alpha'] = 0.7 if alpha is None else alpha
            
            cax = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=color_vector,
                s=size,
                norm=normalize,
                marker=marker[count],
                **kwargs,
            )
        
        # Remove ticks
        ax.set_yticks([])
        ax.set_xticks([])
        if projection == '3d':
            ax.set_zticks([])
        
        # Set axis labels
        name = _basis2name(basis)
        axis_labels = [name + str(d + 1) for d in dims]
        
        ax.set_xlabel(axis_labels[0], loc='left', fontsize=legend_fontsize)
        ax.set_ylabel(axis_labels[1], loc='bottom', fontsize=legend_fontsize)
        if projection == '3d':
            ax.set_zlabel(axis_labels[2], labelpad=-7)
        
        ax.autoscale_view()
        
        # Add edges if requested
        if edges:
            _plot_edges(ax, adata, basis, edges_width, edges_color, neighbors_key)
        
        # Add arrows if requested
        if arrows:
            _plot_arrows(ax, adata, basis, arrows_kwds)
        
        if value_to_plot is None:
            continue
        
        # Handle legend font outline
        if legend_fontoutline is not None:
            path_effect = [
                patheffects.withStroke(linewidth=legend_fontoutline, foreground='w')
            ]
        else:
            path_effect = None
        
        # Add legend or colorbar
        if categorical or color_vector.dtype == bool:
            _add_categorical_legend(
                ax,
                color_source_vector,
                palette=_get_palette(adata, value_to_plot, palette),
                scatter_array=coords,
                legend_loc=legend_loc,
                legend_fontweight=legend_fontweight,
                legend_fontsize=legend_fontsize,
                legend_fontoutline=path_effect,
                na_color=na_color,
                na_in_legend=na_in_legend,
                multi_panel=bool(grid),
            )
        elif colorbar_loc is not None:
            if frameon in ['small', False]:
                # Small colorbar for minimal frames
                from matplotlib.ticker import MaxNLocator
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                labelsize = legend_fontsize * 0.75 if legend_fontsize is not None else None
                cax1 = inset_axes(ax, width="2%", height="30%", loc=4, borderpad=0)
                cb = pl.colorbar(cax, orientation="vertical", cax=cax1)
                cb.set_alpha(1)
                cb.ax.tick_params(labelsize=labelsize)
                cb.locator = MaxNLocator(nbins=3, integer=True)
                cb.update_ticks()
            else:
                pl.colorbar(
                    cax, ax=ax, pad=0.01, fraction=0.08, aspect=30,
                    location=colorbar_loc
                )
    
    if return_fig is True:
        return fig
    
    axs = axs if grid else ax
    
    if show is not False and save is None:
        pl.show()
    elif save is not None:
        pl.savefig(save)
    
    if show is False:
        return axs


# Helper functions

def _get_basis(adata: AnnData, basis: str) -> np.ndarray:
    """Get array for basis from anndata."""
    if basis in adata.obsm:
        return adata.obsm[basis]
    elif f"X_{basis}" in adata.obsm:
        return adata.obsm[f"X_{basis}"]
    else:
        raise KeyError(f"Could not find '{basis}' or 'X_{basis}' in .obsm")


def _basis2name(basis: str) -> str:
    """Convert basis name to display name."""
    return (
        'DC' if basis == 'diffmap'
        else 'tSNE' if basis == 'tsne'
        else 'UMAP' if basis == 'umap'
        else 'PC' if basis == 'pca'
        else basis.replace('draw_graph_', '').upper() if 'draw_graph' in basis
        else basis
    )


def _components_to_dimensions(
    components: Optional[Union[str, Collection[str]]],
    dimensions: Optional[Union[Collection[int], Collection[Collection[int]]]],
    *,
    projection: Literal["2d", "3d"] = "2d",
    total_dims: int,
) -> List[Collection[int]]:
    """Normalize components/dimensions args for embedding plots."""
    ndims = {"2d": 2, "3d": 3}[projection]
    
    if components is None and dimensions is None:
        dimensions = [tuple(i for i in range(ndims))]
    elif components is not None and dimensions is not None:
        raise ValueError("Cannot provide both dimensions and components")
    
    if components == "all":
        dimensions = list(combinations(range(total_dims), ndims))
    elif components is not None:
        if isinstance(components, str):
            components = [components]
        dimensions = [[int(dim) - 1 for dim in c.split(",")] for c in components]
    
    if all(isinstance(el, Integral) for el in dimensions):
        dimensions = [dimensions]
    
    for dims in dimensions:
        if len(dims) != ndims or not all(isinstance(d, Integral) for d in dims):
            raise ValueError(f"Invalid dimensions: {dims}")
    
    return dimensions


def _broadcast_args(*args):
    """Broadcast arguments to common length."""
    lens = [len(arg) for arg in args]
    longest = max(lens)
    if not (set(lens) == {1, longest} or set(lens) == {longest}):
        raise ValueError(f"Could not broadcast arguments with shapes: {lens}.")
    return [
        [arg[0] for _ in range(longest)] if len(arg) == 1 else arg
        for arg in args
    ]


def _panel_grid(hspace, wspace, ncols, num_panels):
    """Create panel grid for multiple plots."""
    from matplotlib import gridspec
    
    n_panels_x = min(ncols, num_panels)
    n_panels_y = np.ceil(num_panels / n_panels_x).astype(int)
    
    fig = pl.figure(
        figsize=(
            n_panels_x * rcParams['figure.figsize'][0] * (1 + wspace),
            n_panels_y * rcParams['figure.figsize'][1],
        ),
    )
    
    left = 0.2 / n_panels_x
    bottom = 0.13 / n_panels_y
    gs = gridspec.GridSpec(
        nrows=n_panels_y,
        ncols=n_panels_x,
        left=left,
        right=1 - (n_panels_x - 1) * left - 0.01 / n_panels_x,
        bottom=bottom,
        top=1 - (n_panels_y - 1) * bottom - 0.1 / n_panels_y,
        hspace=hspace,
        wspace=wspace,
    )
    return fig, gs


def _get_color_source_vector(
    adata, value_to_plot, use_raw=False, gene_symbols=None, layer=None, groups=None
):
    """Get array from adata that colors will be based on."""
    if value_to_plot is None:
        return np.broadcast_to(np.nan, adata.n_obs)
    
    # Check if in obs or var
    in_obs = value_to_plot in adata.obs.columns if hasattr(adata.obs, 'columns') else False
    in_var = value_to_plot in adata.var_names
    
    # Handle gene symbols
    if gene_symbols is not None and not in_obs and not in_var:
        try:
            value_to_plot = adata.var.index[adata.var[gene_symbols] == value_to_plot][0]
            in_var = value_to_plot in adata.var_names
        except (IndexError, KeyError):
            pass
    
    # Get the values
    if in_obs:
        values = adata.obs[value_to_plot]
    elif use_raw and in_var:
        values = adata.raw.obs_vector(value_to_plot)
    elif in_var:
        values = adata.obs_vector(value_to_plot, layer=layer)
    else:
        try:
            values = adata.obs_vector(value_to_plot, layer=layer)
        except (KeyError, AttributeError):
            raise KeyError(f"Could not find '{value_to_plot}' in adata.obs or adata.var_names")
    
    # Convert to categorical if string type
    if not isinstance(values.dtype, pd.CategoricalDtype):
        arr = np.asarray(values)
        if arr.dtype.kind in ("U", "S", "O"):
            if pd.unique(arr).size < arr.size:
                values = pd.Categorical(arr)
    
    if groups and isinstance(values.dtype, pd.CategoricalDtype):
        values = values.remove_categories(values.categories.difference(groups))
    
    return values


def _color_vector(adata, values_key, values, palette, na_color="lightgray"):
    """Map array of values to array of colors."""
    to_hex_fn = partial(to_hex, keep_alpha=True)
    
    if values_key is None:
        return np.broadcast_to(to_hex_fn(na_color), adata.n_obs), False
    
    if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == bool:
        if values.dtype == bool:
            values = pd.Categorical(values.astype(str))
        
        color_map = {
            k: to_hex_fn(v)
            for k, v in _get_palette(adata, values_key, palette=palette).items()
        }
        color_vector = pd.Categorical(values.map(color_map))
        
        if color_vector.isna().any():
            color_vector = color_vector.add_categories([to_hex_fn(na_color)])
            color_vector = color_vector.fillna(to_hex_fn(na_color))
        
        return color_vector, True
    else:
        return values, False


def _get_palette(adata, values_key: str, palette=None):
    """Get color palette for categorical data."""
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    # Get categories
    values = adata.obs[values_key]
    if isinstance(values.dtype, pd.CategoricalDtype):
        categories = values.cat.categories
    else:
        categories = pd.unique(values)
    
    n_cats = len(categories)
    
    # Use provided palette
    if palette is not None:
        if isinstance(palette, dict):
            return palette
        elif isinstance(palette, str):
            if palette in plt.colormaps():
                cmap = plt.get_cmap(palette)
                colors_list = [cmap(i / max(n_cats - 1, 1)) for i in range(n_cats)]
            else:
                raise ValueError(f"Unknown colormap: {palette}")
        elif isinstance(palette, (list, tuple)):
            colors_list = palette
        elif isinstance(palette, Cycler):
            cc = palette()
            colors_list = [next(cc)["color"] for _ in range(n_cats)]
        else:
            raise ValueError(f"Invalid palette type: {type(palette)}")
    else:
        # Check if colors are stored in adata.uns
        color_key = f"{values_key}_colors"
        if color_key in adata.uns:
            colors_list = adata.uns[color_key]
        else:
            # Generate default colors
            if n_cats <= 10:
                colors_list = plt.cm.tab10.colors[:n_cats]
            elif n_cats <= 20:
                colors_list = plt.cm.tab20.colors[:n_cats]
            else:
                cmap = plt.get_cmap('hsv')
                colors_list = [cmap(i / n_cats) for i in range(n_cats)]
    
    # Convert to hex and create mapping
    return {
        cat: to_hex(colors_list[i % len(colors_list)], keep_alpha=True)
        for i, cat in enumerate(categories)
    }


def _add_categorical_legend(
    ax, color_source_vector, palette, scatter_array, legend_loc,
    legend_fontweight, legend_fontsize, legend_fontoutline,
    na_color, na_in_legend, multi_panel
):
    """Add categorical legend to plot."""
    # Handle NaN values
    if na_in_legend and pd.isnull(color_source_vector).any():
        if "NA" not in color_source_vector:
            if not hasattr(color_source_vector, 'add_categories'):
                color_source_vector = pd.Categorical(color_source_vector)
            color_source_vector = color_source_vector.add_categories("NA").fillna("NA")
            palette = palette.copy()
            palette["NA"] = na_color
    
    # Get categories
    if color_source_vector.dtype == bool:
        cats = pd.Categorical(color_source_vector.astype(str)).categories
    elif hasattr(color_source_vector, 'categories'):
        cats = color_source_vector.categories
    else:
        cats = pd.Categorical(color_source_vector).categories
    
    # Adjust axes for multi-panel plots
    if multi_panel:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.91, box.height])
    
    # Add legend
    if legend_loc == 'right margin':
        for label in cats:
            ax.scatter([], [], c=palette[label], label=label)
        ax.legend(
            frameon=False,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            ncol=(1 if len(cats) <= 14 else 2 if len(cats) <= 30 else 3),
            fontsize=legend_fontsize,
        )
    elif legend_loc == 'on data':
        all_pos = (
            pd.DataFrame(scatter_array, columns=["x", "y"])
            .groupby(color_source_vector, observed=True)
            .median()
            .sort_index()
        )
        
        for label, x_pos, y_pos in all_pos.itertuples():
            ax.text(
                x_pos, y_pos, label,
                weight=legend_fontweight,
                verticalalignment='center',
                horizontalalignment='center',
                fontsize=legend_fontsize,
                path_effects=legend_fontoutline,
            )


def _get_vboundnorm(vmin, vmax, vcenter, norm, index, color_vector):
    """Get normalization bounds for color scale."""
    out = []
    
    for v_name, v in [('vmin', vmin), ('vmax', vmax), ('vcenter', vcenter)]:
        if len(v) == 1:
            v_value = v[0]
        else:
            try:
                v_value = v[index]
            except IndexError:
                print(f"Warning: Invalid {v_name} for plot {index + 1}")
                v_value = None
        
        if v_value is not None:
            if isinstance(v_value, str) and v_value.startswith('p'):
                # Interpret as percentile
                try:
                    v_value = np.nanpercentile(color_vector, q=float(v_value[1:]))
                except ValueError:
                    print(f"Warning: Invalid percentile format for {v_name}: {v_value}")
                    v_value = None
            elif callable(v_value):
                # Interpret as function
                v_value = v_value(color_vector)
                if not isinstance(v_value, float):
                    print(f"Warning: Function for {v_name} did not return a float")
                    v_value = None
        
        out.append(v_value)
    
    out.append(norm[0] if len(norm) == 1 else norm[index])
    return tuple(out)


def _check_colornorm(vmin, vmax, vcenter, norm):
    """Check and create color normalization."""
    if norm is not None:
        return norm
    
    if vcenter is not None:
        from matplotlib.colors import TwoSlopeNorm
        return TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    else:
        from matplotlib.colors import Normalize
        return Normalize(vmin=vmin, vmax=vmax)


def _plot_edges(ax, adata, basis, edges_width, edges_color, neighbors_key):
    """Plot edges between neighbors."""
    if neighbors_key is None:
        neighbors_key = 'neighbors'
    
    if neighbors_key not in adata.uns:
        print(f"Warning: No neighbors found at .uns['{neighbors_key}']")
        return
    
    # Get connectivity matrix
    neighbors = adata.uns[neighbors_key]
    if 'connectivities_key' in neighbors:
        conn_key = neighbors['connectivities_key']
    else:
        conn_key = 'connectivities'
    
    if conn_key not in adata.obsp:
        print(f"Warning: No connectivity matrix found at .obsp['{conn_key}']")
        return
    
    connectivity = adata.obsp[conn_key]
    
    # Get coordinates
    coords = _get_basis(adata, basis)
    
    # Draw edges
    from scipy.sparse import issparse
    if issparse(connectivity):
        connectivity = connectivity.tocoo()
        for i, j in zip(connectivity.row, connectivity.col):
            if i < j:  # Avoid drawing edges twice
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color=edges_color,
                    linewidth=edges_width,
                    alpha=0.3,
                    zorder=0,
                )


def _plot_arrows(ax, adata, basis, arrows_kwds):
    """Plot velocity arrows."""
    print("Arrow plotting not implemented in standalone version")
    pass


# Convenience functions that wrap embedding()

def umap(adata: AnnData, **kwargs) -> Union[Axes, List[Axes], None]:
    """Scatter plot in UMAP basis."""
    return embedding(adata, 'umap', **kwargs)


def tsne(adata: AnnData, **kwargs) -> Union[Axes, List[Axes], None]:
    """Scatter plot in tSNE basis."""
    return embedding(adata, 'tsne', **kwargs)


def pca(
    adata: AnnData,
    *,
    annotate_var_explained: bool = False,
    **kwargs
) -> Union[Axes, List[Axes], None]:
    """Scatter plot in PCA coordinates."""
    if not annotate_var_explained:
        return embedding(adata, 'pca', **kwargs)
    else:
        # Add variance explained to axis labels
        if 'pca' not in adata.obsm.keys() and 'X_pca' not in adata.obsm.keys():
            raise KeyError("Could not find PCA in .obsm")
        
        label_dict = {
            f'PC{i + 1}': f'PC{i + 1} ({round(v * 100, 2)}%)'
            for i, v in enumerate(adata.uns['pca']['variance_ratio'])
        }
        
        axs = embedding(adata, 'pca', show=False, save=False, **kwargs)
        
        if isinstance(axs, list):
            for ax in axs:
                ax.set_xlabel(label_dict.get(ax.xaxis.get_label().get_text(), 
                                            ax.xaxis.get_label().get_text()))
                ax.set_ylabel(label_dict.get(ax.yaxis.get_label().get_text(),
                                            ax.yaxis.get_label().get_text()))
        else:
            axs.set_xlabel(label_dict.get(axs.xaxis.get_label().get_text(),
                                         axs.xaxis.get_label().get_text()))
            axs.set_ylabel(label_dict.get(axs.yaxis.get_label().get_text(),
                                         axs.yaxis.get_label().get_text()))
        
        return axs


def diffmap(adata: AnnData, **kwargs) -> Union[Axes, List[Axes], None]:
    """Scatter plot in Diffusion Map basis."""
    return embedding(adata, 'diffmap', **kwargs)


def draw_graph(
    adata: AnnData,
    *,
    layout=None,
    **kwargs
) -> Union[Axes, List[Axes], None]:
    """Scatter plot in graph-drawing basis."""
    if layout is None:
        layout = str(adata.uns['draw_graph']['params']['layout'])
    basis = 'draw_graph_' + layout
    if f'X_{basis}' not in adata.obsm_keys():
        raise ValueError(f'Did not find {basis} in adata.obsm')
    return embedding(adata, basis, **kwargs)

def add_arrow(
    ax: Axes,
    adata,
    basis: str,
    fontsize: int = 12,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    arrow_scale: float = 5,
    arrow_width: float = 0.01,
) -> None:
    """
    Add coordinate arrows and labels to an embedding plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes to add arrows to.
    adata : AnnData
        Annotated data matrix containing the embedding.
    basis : str
        Name of the basis/embedding in adata.obsm.
    fontsize : int
        Font size for axis labels (default: 12).
    x_label : str, optional
        Label for x-axis. If None, uses basis+'1'.
    y_label : str, optional
        Label for y-axis. If None, uses basis+'2'.
    arrow_scale : float
        Scale factor for arrow size (default: 5).
    arrow_width : float
        Width of arrow lines (default: 0.01).
    """
    # Get basis name without 'X_' prefix
    basis_name = basis.replace('X_', '') if basis.startswith('X_') else basis
    
    # Default labels
    if x_label is None:
        x_label = basis_name + '1'
    if y_label is None:
        y_label = basis_name + '2'
    
    # Get embedding data
    if basis in adata.obsm:
        embedding_data = adata.obsm[basis]
    elif f'X_{basis}' in adata.obsm:
        embedding_data = adata.obsm[f'X_{basis}']
    else:
        print(f"Warning: Could not find basis {basis} in adata.obsm")
        return
    
    # Calculate ranges
    x_range = (embedding_data[:, 0].max() - embedding_data[:, 0].min()) / 6
    y_range = (embedding_data[:, 1].max() - embedding_data[:, 1].min()) / 6
    x_min = embedding_data[:, 0].min()
    y_min = embedding_data[:, 1].min()
    
    # Draw x-axis arrow
    ax.arrow(
        x=x_min - x_range/5,
        y=y_min,
        dx=x_range + x_range/arrow_scale,
        dy=0,
        width=arrow_width,
        color="k",
        head_width=y_range * 2/arrow_scale,
        head_length=x_range * 2/arrow_scale,
        overhang=0.5
    )
    
    # Draw y-axis arrow
    ax.arrow(
        x=x_min,
        y=y_min - y_range/5,
        dx=0,
        dy=y_range + y_range/arrow_scale,
        width=arrow_width,
        color="k",
        head_width=x_range * 2/arrow_scale,
        head_length=y_range * 2/arrow_scale,
        overhang=0.5
    )
    
    # Add labels
    ax.text(
        x=x_min,
        y=y_min - y_range/2,
        s=x_label,
        fontsize=fontsize,
        multialignment='center',
        verticalalignment='center'
    )
    ax.text(
        x=x_min - x_range/2,
        y=y_min,
        s=y_label,
        fontsize=fontsize,
        rotation='vertical',
        multialignment='center',
        horizontalalignment='center'
    )