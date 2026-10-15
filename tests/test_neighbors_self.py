import numpy as np

from epione.pp._neighbors import _set_self_distance_zero


def test_self_residue_zeroed_and_first():
    idx = np.array([[0, 2, 1], [1, 0, 2], [0, 2, 1]])
    dist = np.array([[2e-7, 0.5, 0.9], [1e-7, 0.3, 0.4], [0.0, 1e-7, 0.8]], dtype=np.float32)
    i, d = _set_self_distance_zero(idx, dist)
    # row 2: 0 is an exact duplicate tying with self; self goes first
    np.testing.assert_array_equal(i, [[0, 2, 1], [1, 0, 2], [2, 0, 1]])
    np.testing.assert_array_equal(d[:, 0], 0)
    np.testing.assert_allclose(d[2], [0.0, 0.0, 0.8])


def test_missing_self_inserted():
    idx = np.array([[1, 2, 3], [1, 0, 2]])
    dist = np.array([[0.0, 0.2, 0.3], [0.0, 0.0, 0.5]], dtype=np.float32)
    i, d = _set_self_distance_zero(idx, dist)
    np.testing.assert_array_equal(i[0], [0, 1, 2])
    np.testing.assert_allclose(d[0], [0.0, 0.0, 0.2])
    np.testing.assert_array_equal(i[1], [1, 0, 2])


def test_neighbors_float32_no_self_residue():
    import anndata as ad
    import epione as epi
    X = np.random.default_rng(0).normal(size=(300, 20)).astype(np.float32) * 50
    a = ad.AnnData(np.zeros((300, 1), dtype=np.float32))
    a.obsm["E"] = X
    for metric in ("euclidean", "cosine"):
        epi.pp.neighbors(a, n_neighbors=10, use_rep="E", metric=metric, method="gauss")
        D = a.obsp["distances"]
        assert D.diagonal().max() == 0
        assert D.nnz == 300 * 9
