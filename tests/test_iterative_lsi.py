"""Smoke / correctness tests for epione.tl.iterative_lsi on synthetic data.

Run with:  pytest tests/test_iterative_lsi.py -q

The tests build a realistic-ish ATAC-like matrix (~800 open tiles/cell, 3 groups
with group-specific features) and verify that:
  1. the function returns the expected obsm/varm/uns entries;
  2. Leiden clustering on the resulting embedding recovers the ground truth
     (ARI > 0.9 on these clean synthetic inputs);
  3. iterating twice does not *degrade* the embedding relative to a single
     round -- i.e. iterative LSI is at least as good as a plain LSI.
"""
import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def _make_synthetic(n_cells=900, n_features=5000, n_groups=3, seed=0, mean_open=800):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_groups), n_cells // n_groups)
    feat_sets = np.array_split(np.arange(n_features), n_groups)
    rows, cols = [], []
    for i, lbl in enumerate(labels):
        k = rng.integers(int(mean_open * 0.6), int(mean_open * 1.4))
        base = rng.choice(n_features, size=k // 2, replace=False)
        own = rng.choice(feat_sets[lbl], size=k // 2, replace=False)
        idx = np.unique(np.concatenate([base, own]))
        rows.extend([i] * len(idx))
        cols.extend(idx.tolist())
    X = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(labels), n_features),
    ).tocsr()
    ad = AnnData(
        X=X,
        obs={"group": labels.astype(str),
             "n_fragment": np.asarray(X.sum(axis=1)).ravel()},
    )
    return ad


def _ari_via_leiden(adata, use_rep="X_iterative_lsi"):
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score
    tmp = adata.copy()
    sc.pp.neighbors(tmp, use_rep=use_rep, n_neighbors=20)
    sc.tl.leiden(tmp, resolution=0.3, flavor="igraph", directed=False,
                 n_iterations=2, random_state=0)
    return adjusted_rand_score(tmp.obs["group"], tmp.obs["leiden"])


def test_iterative_lsi_stores_expected_keys():
    import epione as ep
    ad = _make_synthetic(seed=0)
    ep.tl.iterative_lsi(ad, n_components=20, iterations=2, var_features=2500,
                        total_features=4500, resolution=0.3, n_neighbors=20,
                        sample_cells_pre=None, depth_col="n_fragment", seed=0)
    assert "X_iterative_lsi" in ad.obsm
    assert "X_iterative_lsi_loadings" in ad.varm
    assert "X_iterative_lsi" in ad.uns
    meta = ad.uns["X_iterative_lsi"]
    for key in ("stdev", "features_final", "depth_cor", "kept_dims",
                "iterations", "params"):
        assert key in meta, f"missing uns key: {key}"
    # `iterations` is now a dict of per-iteration arrays (h5ad-friendly).
    assert len(meta["iterations"]["iteration"]) == 2
    assert meta["params"]["iterations"] == 2


def test_iterative_lsi_recovers_structure():
    import epione as ep
    ad = _make_synthetic(seed=0)
    ep.tl.iterative_lsi(ad, n_components=20, iterations=2, var_features=2500,
                        total_features=4500, resolution=0.3, n_neighbors=20,
                        sample_cells_pre=None, depth_col="n_fragment", seed=0)
    ari = _ari_via_leiden(ad)
    assert ari > 0.9, f"Expected ARI > 0.9 on clean synthetic data, got {ari:.3f}"


def test_iterative_lsi_not_worse_than_single_round():
    import epione as ep
    ad = _make_synthetic(seed=0)
    ep.tl.iterative_lsi(ad, n_components=20, iterations=1, var_features=2500,
                        total_features=4500, resolution=0.3, n_neighbors=20,
                        sample_cells_pre=None, depth_col="n_fragment", seed=0,
                        key_added="X_lsi_1")
    ep.tl.iterative_lsi(ad, n_components=20, iterations=2, var_features=2500,
                        total_features=4500, resolution=0.3, n_neighbors=20,
                        sample_cells_pre=None, depth_col="n_fragment", seed=0,
                        key_added="X_lsi_2")
    ari_1 = _ari_via_leiden(ad, use_rep="X_lsi_1")
    ari_2 = _ari_via_leiden(ad, use_rep="X_lsi_2")
    # On clean synthetic data both should be near 1; iterative should not
    # regress substantially.
    assert ari_2 >= ari_1 - 0.02, f"iterative ARI={ari_2:.3f} regressed vs plain={ari_1:.3f}"


if __name__ == "__main__":
    test_iterative_lsi_stores_expected_keys()
    test_iterative_lsi_recovers_structure()
    test_iterative_lsi_not_worse_than_single_round()
    print("All tests passed.")
