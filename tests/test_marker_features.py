"""Tests for epione.tl.find_marker_features."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def _make_marker_adata(seed=0, n_cells=1200, n_groups=4,
                        n_features=500, n_markers_per_group=15):
    rng = np.random.default_rng(seed)
    n_cells = (n_cells // n_groups) * n_groups
    labels = np.repeat(np.arange(n_groups), n_cells // n_groups)
    # Each cell has a background low expression on all features + high
    # expression on the set of "marker" features of its group.
    feat_sets = [rng.choice(n_features, size=n_markers_per_group, replace=False)
                 for _ in range(n_groups)]
    X = np.zeros((n_cells, n_features), dtype=np.float32)
    # Baseline 10% of features are randomly open per cell (ATAC-like sparsity)
    X[rng.random(X.shape) < 0.08] = 1.0
    # Marker features: high probability open in group cells
    for g in range(n_groups):
        mask = labels == g
        for fi in feat_sets[g]:
            X[mask, fi] = (rng.random(mask.sum()) < 0.85).astype(np.float32)
            X[~mask, fi] = (rng.random((~mask).sum()) < 0.05).astype(np.float32)

    # Differences in total depth to test bias matching
    depth_scale = np.where(labels < 2, 0.7, 1.3)  # groups 0-1 shallow, 2-3 deep
    for i, s in enumerate(depth_scale):
        drop_mask = rng.random(n_features) > s
        X[i, drop_mask] = 0.0

    ad = AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({
            "group": labels.astype(str),
            "n_fragment": np.asarray(X.sum(axis=1)).ravel(),
        }, index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"f{j:04d}" for j in range(n_features)]),
    )
    planted = {f"{g}": [f"f{fi:04d}" for fi in feat_sets[g]] for g in range(n_groups)}
    return ad, planted


def test_find_marker_features_recovers_planted_sets():
    import epione as ep
    ad, planted = _make_marker_adata()

    markers = ep.tl.find_marker_features(
        ad,
        group_by="group",
        bias_vars=["n_fragment"],
        k_match=1,
        max_cells_per_group=500,
        seed=0,
    )
    assert "markers" in ad.uns

    # For each group, verify top-10 by log2_fc covers most of the planted set
    for g, expected in planted.items():
        top = markers[markers["group"] == g]
        top = top[(top["log2_fc"] > 0) & (top["fdr"] < 0.05)]
        top = top.reindex(top["log2_fc"].sort_values(ascending=False).index)
        top10 = set(top.head(len(expected))["feature"].tolist())
        recovered = top10 & set(expected)
        print(f"group {g}: {len(recovered)}/{len(expected)} planted markers in top-{len(expected)}")
        assert len(recovered) >= 0.7 * len(expected), (
            f"group {g}: only {len(recovered)}/{len(expected)} planted markers recovered"
        )


def test_find_marker_features_bias_reduces_depth_confounds():
    """A marker that is purely a depth artefact should be reduced (not
    necessarily zero) with bias-matched background — as long as the groups'
    depth distributions overlap."""
    import epione as ep
    rng = np.random.default_rng(1)
    n_cells = 600
    labels = np.array(["a"] * 300 + ["b"] * 300)
    # Same marginal per-feature probability for both groups...
    X = (rng.random((n_cells, 200)) < 0.3).astype(np.float32)
    # ...but group "a" has per-cell drop-out drawn from a distribution that
    # *overlaps* group "b"'s (0 drop-out). Here: a ~ Unif[0, 0.6].
    per_cell_drop = np.where(labels == "a", rng.random(n_cells) * 0.6, 0.0)
    drop = rng.random(X.shape) < per_cell_drop[:, None]
    X[drop] = 0.0
    ad = AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({
            "group": labels,
            "n_fragment": X.sum(axis=1),
        }, index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"f{j:03d}" for j in range(200)]),
    )

    # Without bias matching: many "false markers" from group b (higher depth)
    m_no = ep.tl.find_marker_features(ad, group_by="group", bias_vars=None,
                                       max_cells_per_group=300, verbose=False)
    n_sig_b_no = ((m_no["group"] == "b") & (m_no["fdr"] < 0.05)
                  & (m_no["log2_fc"] > 0)).sum()

    # With bias matching on n_fragment: should drop substantially
    m_yes = ep.tl.find_marker_features(ad, group_by="group",
                                        bias_vars=["n_fragment"],
                                        k_match=1, replace=True,
                                        max_cells_per_group=300, verbose=False)
    n_sig_b_yes = ((m_yes["group"] == "b") & (m_yes["fdr"] < 0.05)
                   & (m_yes["log2_fc"] > 0)).sum()

    print(f"significant group-b markers: without bias match = {n_sig_b_no}, "
          f"with bias match = {n_sig_b_yes}")
    # Bias matching should at least halve the false-positive count when depth
    # distributions overlap.
    assert n_sig_b_yes <= n_sig_b_no, "bias matching produced MORE markers?"
    assert n_sig_b_yes < 0.5 * max(n_sig_b_no, 1), (
        f"expected bias-matched markers to drop below half of unmatched "
        f"({n_sig_b_yes} vs {n_sig_b_no})"
    )


if __name__ == "__main__":
    test_find_marker_features_recovers_planted_sets()
    test_find_marker_features_bias_reduces_depth_confounds()
    print("ok")
