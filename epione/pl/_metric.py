

from anndata import AnnData
import snapatac2._snapatac2 as internal
import numpy as np
import snapatac2
import matplotlib.pyplot as plt
from ..utils import console


def is_anndata(data) -> bool:
    return isinstance(data, AnnData) or isinstance(data, internal.AnnData) or isinstance(data, internal.AnnDataSet)


def frag_size_distr(
    adata: AnnData | np.ndarray,
    use_rep: str = "frag_size_distr",
    max_recorded_size: int = 1000,
    figsize: tuple = (4, 4), 
    ax: plt.Axes = None,
    title: str = "Fragment size distribution",
    xlabel: str = "Fragment size",
    ylabel: str = "Count",
    log_y: bool = False,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """ Plot the fragment size distribution.
    """
    from ..pp import frag_size_distr as pp_frag_size_distr
    if is_anndata(adata):
        if use_rep not in adata.uns or len(adata.uns[use_rep]) <= max_recorded_size:
            console.level2("Computing fragment size distribution...")
            pp_frag_size_distr(adata, add_key=use_rep, max_recorded_size=max_recorded_size)
        data = adata.uns[use_rep]
    else:
        data = adata
    data = data[:max_recorded_size+1]

    x, y = zip(*enumerate(data))
    # Make a line plot
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.grid(False)
    return fig, ax

    
