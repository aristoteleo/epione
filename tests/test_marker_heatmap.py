"""Tests for epione.pl.marker_heatmap / peak2gene_heatmap (ArchR ports)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import epione as epi
from epione.pl._heatmap_rcompat import (
    RRandom, r_dist, r_hclust, r_kmeans, r_reorder_dendrogram,
)
from epione.pl._marker_heatmap import (
    MarkerAssays, _binary_sort, eval_cutoff, peak2gene_heatmap_matrices,
)

# Values are exactly representable so R and Python see identical inputs.
M6 = np.array([
    [0.5, 1.25, -2, 3, 0.75], [4, -1.5, 2.25, 0, 1], [-3, 0.25, 1.5, 2, -0.5],
    [1, 2, 3, -1, 0.125], [2.5, -2.25, 0.5, 1.75, -1], [0, 3.5, -0.75, 0.5, 2],
])


# --------------------------------------------------------------- R compat
def test_r_rng_matches_r():
    # set.seed(1); sample.int(1000, 10) / runif(3) in R 4.4
    assert RRandom(1).sample_int(1000, 10).tolist() == [836, 679, 129, 930, 509, 471, 299, 270, 978, 187]
    r = RRandom(1)
    assert [r.unif_rand() for _ in range(3)] == [0.2655086631421, 0.37212389963679016, 0.5728533633518964]
    r = RRandom(2024)
    assert r.sample_int(50, 7).tolist() == [2, 37, 45, 17, 48, 32, 41]
    assert r.sample(np.array([3, 8, 21, 40]), 2).tolist() == [8, 40]


def test_r_hclust_complete_and_ward():
    hc = r_hclust(r_dist(M6), "complete")
    assert hc.merge.tolist() == [[-2, -5], [-1, -6], [-3, -4], [2, 3], [1, 4]]
    assert hc.order.tolist() == [2, 5, 1, 6, 3, 4]
    np.testing.assert_allclose(
        hc.height, [3.5968736424845398, 3.8324274291889728, 5.541040064825375,
                    6.4963932300931413, 7.6730046266114034], rtol=0, atol=1e-14)
    hw = r_hclust(r_dist(M6), "ward.D2")
    assert hw.order.tolist() == [2, 5, 1, 6, 3, 4]
    assert (r_reorder_dendrogram(hw, -M6.mean(axis=1)) + 1).tolist() == [2, 5, 6, 1, 4, 3]


def test_r_kmeans_hartigan_wong():
    x = np.vstack([M6, M6 + 0.3, M6 - 0.2])
    km = r_kmeans(x, 3, rng=RRandom(11))
    assert km.cluster.tolist() == [3, 1, 3, 2, 1, 3, 3, 1, 3, 2, 1, 3, 3, 1, 3, 2, 1, 3]


# --------------------------------------------------------------- cutOff
def test_cutoff_three_valued_logic():
    fdr = np.array([[0.001, np.nan], [0.5, 0.001]])
    lfc = np.array([[2.0, 3.0], [np.nan, -2.0]])
    got = eval_cutoff("FDR <= 0.01 & Log2FC >= 1", {"FDR": fdr, "Log2FC": lfc})
    np.testing.assert_array_equal(got, np.array([[1.0, np.nan], [0.0, 0.0]]))
    got = eval_cutoff("FDR <= 0.01 | abs(Log2FC) >= 2.5", {"FDR": fdr, "Log2FC": lfc})
    np.testing.assert_array_equal(got, np.array([[1.0, 1.0], [np.nan, 1.0]]))
    got = eval_cutoff("!(Log2FC > 0) & FDR < 1e-2", {"FDR": fdr, "Log2FC": lfc})
    np.testing.assert_array_equal(got, np.array([[0.0, 0.0], [0.0, 1.0]]))


def test_binary_sort_orders_rows_by_pattern():
    m = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    row_idx, col_idx, _ = _binary_sort(m, cluster_cols=False)
    assert row_idx.tolist() == [2, 1, 0, 3]
    assert col_idx.tolist() == [0, 1, 2]


# --------------------------------------------------------------- marker heatmap
def _synthetic_markers(n_feat=60, groups=("A", "B", "C", "D"), seed=0):
    rng = np.random.default_rng(seed)
    mean = rng.gamma(2.0, 1.0, size=(n_feat, len(groups)))
    for i in range(n_feat):
        mean[i, i % len(groups)] *= 8
    lfc = np.log2(mean / mean.mean(axis=1, keepdims=True))
    fdr = np.where(lfc > 1, 1e-4, 0.5)
    names = [f"G{i}" for i in range(n_feat)]
    df = lambda a: pd.DataFrame(a, columns=list(groups))
    return MarkerAssays({"Mean": df(mean), "Log2FC": df(lfc), "FDR": df(fdr)},
                        pd.DataFrame({"name": names}), "GeneScoreMatrix")


def test_marker_heatmap_matrix_basic():
    ma = _synthetic_markers()
    res = epi.pl.marker_heatmap_matrix(ma, cut_off="FDR <= 0.01 & Log2FC >= 1")
    mat = res.matrix
    assert list(mat.columns) == ["A", "B", "C", "D"]
    assert mat.to_numpy().max() <= 2 and mat.to_numpy().min() >= -2
    # binary sort: rows marked in the first displayed column come first
    first = mat.columns[res.col_order[0]]
    lead = res.pass_matrix.loc[mat.index[0], first]
    assert lead == 1
    assert len(res.label_markers) > 0 and len(res.label_index) == len(res.label_markers)
    same = epi.pl.marker_heatmap(ma, cut_off="FDR <= 0.01 & Log2FC >= 1", return_matrix=True)
    pd.testing.assert_frame_equal(same, mat)


def test_marker_heatmap_from_long_dataframe_and_plot(tmp_path):
    ma = _synthetic_markers()
    rows = []
    for g in ma.groups:
        for i, f in enumerate(ma.row_data["name"]):
            rows.append(dict(group=g, feature=f, mean_fg=ma.assays["Mean"][g][i],
                             log2_fc=ma.assays["Log2FC"][g][i], fdr=ma.assays["FDR"][g][i]))
    long = pd.DataFrame(rows)
    a = epi.pl.marker_heatmap(long, cut_off="FDR <= 0.01 & Log2FC >= 1", return_matrix=True)
    b = epi.pl.marker_heatmap(ma, cut_off="FDR <= 0.01 & Log2FC >= 1", return_matrix=True)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())
    import matplotlib
    matplotlib.use("Agg")
    for kw in (dict(), dict(transpose=True), dict(binary_cluster_rows=False), dict(invert=True)):
        fig, res = epi.pl.marker_heatmap(ma, cut_off="FDR <= 0.01 & abs(Log2FC) >= 1", show=False, **kw)
        fig.savefig(tmp_path / "hm.png")


# --------------------------------------------------------------- peak2gene heatmap
def _synthetic_p2g(n_peaks=300, n_genes=80, n_knn=40, n_cells=400, seed=0):
    rng = np.random.default_rng(seed)
    atac = np.log2(rng.gamma(2, 1, size=(n_peaks, n_knn)) + 1)
    rna = np.log2(rng.gamma(2, 1, size=(n_genes, n_knn)) + 1)
    links = pd.DataFrame({
        "idxATAC": rng.integers(1, n_peaks + 1, 500), "idxRNA": rng.integers(1, n_genes + 1, 500),
        "Correlation": rng.uniform(0.3, 0.9, 500), "FDR": rng.uniform(0, 2e-4, 500),
        "VarQATAC": rng.uniform(0, 1, 500), "VarQRNA": rng.uniform(0, 1, 500),
    })
    knn_index = [rng.choice(n_cells, 20, replace=False) for _ in range(n_knn)]
    cells = rng.choice(["C1", "C2", "C10"], n_cells)
    return dict(links=links, atac_knn=atac, rna_knn=rna, knn_index=knn_index, cell_groups=cells,
                peak_names=[f"chr1:{i}" for i in range(n_peaks)], gene_names=[f"g{i}" for i in range(n_genes)])


def test_peak2gene_heatmap_matrices_paths(tmp_path):
    inp = _synthetic_p2g()
    full = peak2gene_heatmap_matrices(**inp, k=5, n_plot=10_000, seed=1)
    n = full.atac.shape[0]
    assert full.atac.shape == full.rna.shape and full.atac.shape[1] == 40
    assert sorted(set(full.kmeans_id)) == [1, 2, 3, 4, 5]
    sub = peak2gene_heatmap_matrices(**inp, k=5, n_plot=n // 2, seed=1)
    assert sub.atac.shape[0] <= n // 2
    # injecting the free-running labels reproduces the free-running result
    inj = peak2gene_heatmap_matrices(**inp, k=5, n_plot=n // 2, seed=1,
                                     kmeans_cluster=sub.kmeans_cluster)
    pd.testing.assert_frame_equal(inj.atac, sub.atac)
    import matplotlib
    matplotlib.use("Agg")
    fig, _ = epi.pl.peak2gene_heatmap(**inp, k=5, show=False)
    fig.savefig(tmp_path / "p2g.png")


# --------------------------------------------------------------- ArchR parity
# ArchR 1.0.3 reference outputs are generated outside the repository; point
# EPIONE_ARCHR_PARITY_DIR at the directory that contains ``marker_heatmap/``.
_PARITY_ROOT = os.environ.get("EPIONE_ARCHR_PARITY_DIR")
PARITY_DIR = Path(_PARITY_ROOT) / "marker_heatmap" if _PARITY_ROOT else None


@pytest.mark.skipif(PARITY_DIR is None or not (PARITY_DIR / "ref" / "p2g_default__kDF.tsv").exists(),
                    reason="EPIONE_ARCHR_PARITY_DIR not set or ArchR reference outputs missing")
def test_archr_parity():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "parity_check", Path(__file__).parent / "parity" / "marker_heatmap" / "parity_check.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    ref = PARITY_DIR / "ref"
    for name, rec in pc.marker_parity(ref).items():
        assert rec["shape_equal"] and rec["rows_equal"] and rec["cols_equal"], name
        assert rec["nan_pattern_equal"] and rec["max_abs_err"] <= 1e-8, name
        assert rec["row_order_equal"] and rec["col_order_equal"], name
        assert rec.get("labels_equal", True), name
    for tag, rec in pc.p2g_parity(ref).items():
        inj = rec["injected"]
        assert inj["atac_max_abs_err"] <= 1e-8 and inj["rna_max_abs_err"] <= 1e-8, tag
        assert all(inj[k] for k in ("rows_equal", "col_order_equal", "kmeans_id_equal",
                                    "col_groups_equal", "links_equal", "kdf_equal")), tag
        assert rec["free"]["kmeans_ari"] >= 0.95, tag
