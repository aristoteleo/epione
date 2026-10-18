"""epi.tl.clusters Leiden backends: igraph (default, modularity to
convergence) and the previous leidenalg default."""
import numpy as np
import pytest
from anndata import AnnData


@pytest.fixture
def graph_adata():
    import epione as epi
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=5.0, size=(4, 8))
    lab = np.repeat(np.arange(4), 150)
    X = (centers[lab] + rng.normal(size=(lab.size, 8))).astype(np.float32)
    ad = AnnData(obs={"g": lab.astype(str)})
    ad.obsm["X_rep"] = X
    epi.pp.neighbors(ad, use_rep="X_rep", n_neighbors=15)
    return ad


def test_default_backend_is_igraph_modularity(graph_adata):
    import epione as epi
    from sklearn.metrics import adjusted_rand_score
    from epione.tl._leiden import leiden as _leiden
    a, b, c = graph_adata.copy(), graph_adata.copy(), graph_adata.copy()
    epi.tl.clusters(a, method="leiden")
    _leiden(b, flavor="igraph", n_iterations=-1, directed=False, key_added="clusters")
    np.testing.assert_array_equal(a.obs["clusters"].to_numpy(), b.obs["clusters"].to_numpy())
    epi.tl.clusters(c, method="leiden", backend="leidenalg")
    d = graph_adata.copy()
    _leiden(d, key_added="clusters")                    # leidenalg, as before
    np.testing.assert_array_equal(c.obs["clusters"].to_numpy(), d.obs["clusters"].to_numpy())
    assert adjusted_rand_score(a.obs["g"], a.obs["clusters"]) > 0.9
    e = graph_adata.copy()
    epi.tl.clusters(e, method="leiden", backend="epione")      # alias of leidenalg
    np.testing.assert_array_equal(e.obs["clusters"].to_numpy(), d.obs["clusters"].to_numpy())
    with pytest.raises(ValueError):
        epi.tl.clusters(graph_adata.copy(), method="leiden", backend="nope")
