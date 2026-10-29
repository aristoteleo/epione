"""spatial_domains(method=...) wrappers around omicverse's spatial-domain models.

omicverse is not an Epione dependency: the smoke tests skip unless it (and
the method's own optional dependencies) is importable.
"""
import builtins
import importlib.util

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import epione as epi
from epione.tl import _spatial_ov as S


def _toy(n=14, n_peaks=300, seed=0):
    rng = np.random.default_rng(seed)
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r, c = r.ravel(), c.ravel()
    left = c < n // 2
    p = np.where(left[:, None], np.r_[np.full(n_peaks // 2, 0.3), np.full(n_peaks - n_peaks // 2, 0.03)],
                 np.r_[np.full(n_peaks // 2, 0.03), np.full(n_peaks - n_peaks // 2, 0.3)])
    X = sp.csr_matrix(rng.binomial(1, p).astype(np.float32))
    a = ad.AnnData(X)
    a.obs_names = [f"s{i}" for i in range(a.n_obs)]
    a.var_names = [f"p{j}" for j in range(n_peaks)]
    a.obs["array_row"], a.obs["array_col"] = r, c
    a.obsm["spatial"] = np.c_[c, r].astype(float) * 50.0
    a.obsm["X_birmt"] = np.c_[left * 2.0 - 1, rng.normal(size=(a.n_obs, 4)) * 0.1].astype(np.float32)
    a.obs["truth"] = left.astype(int)
    return a


def test_features_matrix_tfidf_and_binary():
    a = _toy()
    X, names = S._features(a, None, None, 50, "tfidf")
    assert X.shape == (a.n_obs, 50) and len(names) == 50 and np.isfinite(X).all() and X.min() >= 0
    B, _ = S._features(a, None, None, 50, "binary")
    assert sp.issparse(B) and set(np.unique(B.data)) <= {1.0}
    E, _ = S._features(a, "X_birmt", None, 50, "tfidf")
    np.testing.assert_array_equal(E, a.obsm["X_birmt"])
    with pytest.raises(ValueError):
        S._features(a, "X_birmt", None, 50, "binary")
    with pytest.raises(KeyError):
        S._features(a, "X_missing", None, 50, "tfidf")


def test_missing_omicverse_raises_clear_error(monkeypatch):
    real = builtins.__import__

    def fake(name, *args, **kw):
        if name == "omicverse" or name.startswith("omicverse."):
            raise ImportError("no omicverse")
        return real(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)
    a = _toy()
    for m in S.OV_DOMAIN_METHODS:
        with pytest.raises(ImportError, match="pip install omicverse"):
            epi.tl.spatial_domains(a, 2, method=m)


def test_dispatch_rejects_kwargs_for_plain_clustering():
    a = _toy()
    with pytest.raises(TypeError):
        epi.tl.spatial_domains(a, 2, use_rep="X_birmt", method="kmeans", seed=1)


def _have(*mods):
    try:
        return all(importlib.util.find_spec(m) is not None for m in mods)
    except (ImportError, ValueError):
        return False


def _ov_ok():
    if not _have("omicverse"):
        return False
    try:
        import omicverse  # noqa: F401
    except Exception:
        return False
    return True


NEEDS = {
    "stagate": ("torch_geometric",), "graphst": ("ot",), "spaceflow": ("torch_geometric", "gudhi"),
    "cellcharter": (), "cast": ("dgl",), "binary": ("torch_geometric",),
}


@pytest.mark.parametrize("method", list(S.OV_DOMAIN_METHODS))
def test_omicverse_methods_smoke(method):
    if not _ov_ok():
        pytest.skip("omicverse not importable")
    if not _have("torch", *NEEDS[method]):
        pytest.skip(f"{method} needs {NEEDS[method]}")
    a = _toy()
    obsp_before = set(a.obsp)
    fast = {"stagate": {"num_epoch": 30}, "graphst": {"epochs": 30}, "spaceflow": {"epochs": 30, "min_stop": 5},
            "cellcharter": {}, "cast": {"epochs": 5}, "binary": {"n_epochs": 30}}[method]
    if method != "cellcharter":
        fast["cluster_method"] = "kmeans"
    if method in ("stagate", "graphst", "cast", "binary"):
        fast["device"] = "cpu"
    lab = epi.tl.spatial_domains(a, 2, method=method, key_added="dom", **fast)
    assert a.obs["dom"].dtype.name == "category" and a.obs["dom"].nunique() == 2
    assert sorted(a.obs["dom"].cat.categories) == [1, 2]
    assert len(lab) == a.n_obs
    assert set(a.obsp) == obsp_before  # runs on a private copy
