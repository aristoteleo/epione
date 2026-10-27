"""Corner coordinate arrows of epi.pl.embedding(frameon='small') / epi.pl.add_arrow."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import pytest

import epione as epi


def _adata(n=200):
    rng = np.random.default_rng(0)
    a = ad.AnnData(np.zeros((n, 1)), obs=pd.DataFrame({"g": pd.Categorical(rng.choice(list("ab"), n))},
                                                       index=[f"c{i}" for i in range(n)]))
    a.obsm["X_umap"] = rng.normal(size=(n, 2)) * [10, 1]
    a.obsm["X_pca"] = rng.normal(size=(n, 3))
    return a


def _texts(ax):
    return [t.get_text() for t in ax.texts]


def _arrows(ax):
    return [c for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation) and c.arrow_patch is not None]


def test_small_frame_draws_labelled_arrows():
    a = _adata()
    fig, ax = plt.subplots()
    epi.pl.umap(a, color="g", ax=ax, show=False)
    assert "UMAP 1" in _texts(ax) and "UMAP 2" in _texts(ax)
    assert len(_arrows(ax)) == 2
    plt.close(fig)


def test_arrows_have_equal_physical_length_whatever_the_data_range():
    a = _adata()
    fig, ax = plt.subplots(figsize=(6, 3))
    epi.pl.add_arrow(ax, basis="X_umap", length=0.5)
    fig.canvas.draw()
    bb = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    lens = []
    for an in _arrows(ax):
        (x1, y1), (x0, y0) = an.xy, an.xyann
        lens.append(abs(x1 - x0) * bb.width + abs(y1 - y0) * bb.height)
    assert lens[0] == pytest.approx(0.5, rel=0.01)
    assert lens[0] == pytest.approx(lens[1], rel=0.05)
    plt.close(fig)


def test_frameon_false_draws_no_arrows_and_true_keeps_frame():
    a = _adata()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    epi.pl.umap(a, color="g", ax=ax1, show=False, frameon=False)
    epi.pl.embedding(a, basis="X_umap", color="g", ax=ax2, show=False, frameon=True)
    assert len(_arrows(ax1)) == 0
    assert ax2.get_xlabel() == "UMAP 1" and ax2.get_ylabel() == "UMAP 2"
    plt.close(fig)


def test_labels_follow_components_and_basis():
    a = _adata()
    fig, ax = plt.subplots()
    epi.pl.embedding(a, basis="X_pca", components="2,3", color="g", ax=ax, show=False)
    assert "PC 2" in _texts(ax) and "PC 3" in _texts(ax)
    plt.close(fig)


@pytest.mark.parametrize("loc", ["lower left", "lower right", "upper left", "upper right"])
def test_loc_stays_inside_axes_and_points_positive(loc):
    fig, ax = plt.subplots()
    epi.pl.add_arrow(ax, basis="X_tsne", loc=loc)
    for an in _arrows(ax):
        (x1, y1), (x0, y0) = an.xy, an.xyann
        assert 0 <= min(x0, x1) and max(x0, x1) <= 1 and 0 <= min(y0, y1) and max(y0, y1) <= 1
        assert x1 >= x0 and y1 >= y0
    assert {"t-SNE 1", "t-SNE 2"} <= set(_texts(ax))
    plt.close(fig)


def test_old_positional_call_still_works():
    a = _adata()
    fig, ax = plt.subplots()
    epi.pl.add_arrow(ax, a, "X_umap", fontsize=9, arrow_scale=6, arrow_width=0.005)
    assert "UMAP 1" in _texts(ax)
    plt.close(fig)
