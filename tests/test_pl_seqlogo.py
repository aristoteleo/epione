"""plot_seqlogo / pfm_information."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from epione.pl import pfm_information, plot_seqlogo


def test_pfm_information_bounds():
    pfm = np.array([[10, 0, 0, 0], [2.5, 2.5, 2.5, 2.5], [5, 5, 0, 0]], float).T   # [4, w]
    h = pfm_information(pfm)
    assert h.shape == (3, 4)
    np.testing.assert_allclose(h.sum(1), [2.0, 0.0, 1.0], atol=1e-12)


def test_seqlogo_draws_one_patch_per_nonzero_letter():
    v = np.zeros((5, 4))
    v[0, 0], v[1, 3], v[2, 1], v[3, 2] = 1.0, -0.5, 0.2, 0.0
    ax = plot_seqlogo(v, start=100)
    assert len(ax.patches) == 3
    assert ax.get_xlim() == (99.5, 104.5)
    lo, hi = ax.get_ylim()
    assert lo < -0.5 and hi > 1.0


def test_seqlogo_accepts_4xL_and_rejects_bad_shape():
    ax = plot_seqlogo(np.ones((4, 6)))
    assert len(ax.patches) == 24
    with pytest.raises(ValueError):
        plot_seqlogo(np.ones((3, 5)))
