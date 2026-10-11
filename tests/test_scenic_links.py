"""Tests for epione.scenic links (search space, TF-to-gene, region-to-gene).

The frozen values below were produced with scikit-learn 1.3.2 / arboreto 0.1.6 /
scenicplus 1.0a2 (the SCENIC+ reference stack) on the synthetic inputs built here.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import epione.scenic as sc
from epione.scenic import _sktree as T

# scikit-learn 1.3.2 feature_importances_ on _toy() (float.hex)
SGBM = ['0x1.10e289b88517cp-1', '0x1.df7006c59b630p-7', '0x1.651c4d7f8cf75p-6', '0x1.db7ecdb852bc5p-4',
        '0x1.18becdc69d713p-6', '0x1.1053309dfa08cp-6', '0x1.ba55e5a59b117p-7', '0x1.cbf443ce0682bp-6',
        '0x1.f275ce3f2d5b2p-6', '0x1.4089cafb563e2p-6', '0x1.57fedd5a40750p-4', '0x1.5a2bacaed17fbp-7',
        '0x1.7cc59b8538d2ep-6', '0x1.4d7d70df56f1dp-5', '0x1.f6dca9e77704bp-6']
SGBM_N = 30
GBM = ['0x1.ef012438d2bb5p-2', '0x1.2ff1d8fa38d27p-5', '0x1.7c1af731c2afep-6', '0x1.32a37683c14e7p-3',
       '0x1.86de01bc83d96p-7', '0x1.496edb186da3ep-6', '0x1.3f2fc73a9bef2p-6', '0x1.e00cc7b08d71fp-6',
       '0x1.8a140c89a558bp-5', '0x1.d3c70e2b231d7p-7', '0x1.c42d6f4859ad7p-5', '0x1.82a81fc1be41ep-6',
       '0x1.eced85ea9a37dp-6', '0x1.264e3d3ae292bp-5', '0x1.302055710f57ap-6']
RF = ['0x1.914af22fc187bp-2', '0x1.5ce07961638cep-5', '0x1.2b90855eef551p-5', '0x1.f82874363db38p-4',
      '0x1.f916f807ae246p-6', '0x1.0c271a4414052p-5', '0x1.0eea13037a3c0p-5', '0x1.5e9b380383df4p-5',
      '0x1.8e69eb2150b14p-5', '0x1.6fa65abf5fce6p-6', '0x1.bee21afb1156dp-5', '0x1.a7776a46a133cp-6',
      '0x1.6c9427c273d7ap-5', '0x1.5d08f93a31f4ep-5', '0x1.c86d38d467e31p-6']

# scenicplus get_search_space on _toy_annotation() / _TOY_REGIONS
SEARCH_SPACE = [
    ["chr1:1000-1500", "GA", "[48750]"],
    ["chr1:405000-405500", "GC", "[0]"],
    ["chr1:560000-560500", "GC", "[-130250]"],
    ["chr1:405000-405500", "GB", "[0]"],
    ["chr1:560000-560500", "GB", "[140250]"],
    ["chr2:800000-800500", "GE", "[99750]"],
    ["chr2:960000-960500", "GE", "[-10250]"],
    ["chr2:5000-5500", "GD", "[-4750]"],
]
_TOY_REGIONS = ["chr1:1000-1500", "chr1:120000-120500", "chr1:180000-180500", "chr1:405000-405500",
                "chr1:560000-560500", "chr2:5000-5500", "chr2:150000-150500", "chr2:170000-170500",
                "chr2:800000-800500", "chr2:960000-960500"]


def _toy():
    rs = np.random.RandomState(2024)
    X = rs.poisson(0.6, size=(120, 15)).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 3] + rs.poisson(0.5, 120)).astype(np.float32)
    return X, y


def _toy_annotation():
    return pd.DataFrame({
        "Chromosome": ["chr1", "chr1", "chr1", "chr2", "chr2"],
        "Start": [50000, 400000, 401000, 10000, 900000],
        "End": [60000, 420000, 430000, 30000, 950000],
        "Strand": ["+", "-", "+", "-", "+"],
        "Gene": ["GA", "GB", "GC", "GD", "GE"],
        "Transcription_Start_Site": [50000, 420000, 401000, 30000, 900000],
    })


def _hex(a):
    return np.array([float.fromhex(v) for v in a])


# ---------------------------------------------------------------------------
# RNG and reductions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 666, 123456789, 2**31 - 2])
def test_mt19937_matches_numpy_randomstate(seed):
    key = np.empty(624, np.uint32)
    pos = np.zeros(1, np.int64)
    T._mt_seed(seed, key, pos)
    rs = np.random.RandomState(seed)
    st = rs.get_state()
    assert np.array_equal(key, st[1]) and pos[0] == st[2]
    u = rs.uniform(size=700)
    assert np.array_equal(np.array([T._mt_double(key, pos) for _ in range(700)]), u)
    assert T._mt_bounded(key, pos, 2**31 - 2) == rs.randint(0, 2**31 - 1)
    draws = rs.randint(0, 977, 300)
    assert np.array_equal(np.array([T._mt_bounded(key, pos, 976) for _ in range(300)]), draws)


def test_pairwise_sum_matches_numpy():
    rs = np.random.RandomState(1)
    for n in [1, 7, 8, 9, 127, 128, 129, 255, 1000, 4099]:
        a = rs.randn(n) * 10.0 ** rs.randint(-5, 5, size=n)
        assert T._np_sum(a, n) == np.sum(a)


# ---------------------------------------------------------------------------
# scikit-learn 1.3.2 models
# ---------------------------------------------------------------------------
def test_sgbm_importances_match_sklearn_132():
    X, y = _toy()
    imp, n = T.gbm_importances(X, y, learning_rate=0.01, n_estimators=5000, max_features=0.1,
                               subsample=0.9, early_stop_window=25)
    assert n == SGBM_N
    assert np.array_equal(imp, _hex(SGBM))


def test_gbm_importances_match_sklearn_132():
    X, y = _toy()
    imp, n = T.gbm_importances(X, y, learning_rate=0.01, n_estimators=500, max_features=0.1)
    assert n == 500
    assert np.array_equal(imp, _hex(GBM))


def test_rf_importances_match_sklearn_132():
    X, y = _toy()
    assert np.array_equal(T.rf_importances(X, y, n_estimators=100), _hex(RF))


def test_tree_structure_matches_installed_sklearn():
    from sklearn.tree import DecisionTreeRegressor
    rs = np.random.RandomState(0)
    X = rs.rand(300, 12).astype(np.float32)
    y = X[:, 0] * 3 + np.sin(X[:, 1] * 6) + rs.rand(300) * 0.1
    ref = DecisionTreeRegressor(random_state=5, max_features=4, max_depth=6).fit(X, y).tree_
    n, p = X.shape
    cap = 2 * n + 1
    key = np.empty(624, np.uint32)
    pos = np.zeros(1, np.int64)
    T._mt_seed(5, key, pos)
    st = np.array([T._mt_bounded(key, pos, 2**31 - 2)], dtype=np.uint32)
    arr = [np.empty(cap, np.int64), np.empty(cap, np.int64), np.empty(cap, np.int64), np.empty(cap),
           np.empty(cap), np.empty(cap, np.int64), np.empty(cap), np.empty(cap)]
    nc = T.build_tree(X, np.arange(p), y, np.ones(n), st, 4, 6, 0, np.empty(n, np.int64),
                      np.empty(p, np.int64), np.empty(p, np.int64), np.empty(n, np.float32),
                      np.empty((256, 3), np.int64), np.empty((cap, 6), np.int64), *arr)
    left, right, feat, thr, imp, nns, wnns, val = arr
    assert nc == ref.node_count
    assert np.array_equal(left[:nc], ref.children_left)
    assert np.array_equal(feat[:nc], ref.feature)
    assert np.array_equal(thr[:nc], ref.threshold)
    np.testing.assert_allclose(val[:nc], ref.value[:, 0, 0], rtol=1e-12, atol=1e-12)


def test_zero_target_short_circuit():
    X, _ = _toy()
    imp, n = T.gbm_importances(X, np.zeros(X.shape[0], np.float32), subsample=0.9,
                               n_estimators=5000, early_stop_window=25)
    assert n == 5000 and not imp.any()


# ---------------------------------------------------------------------------
# search space, TF2G, R2G
# ---------------------------------------------------------------------------
def test_search_space_matches_scenicplus():
    out = sc.get_search_space(_TOY_REGIONS, ["GA", "GB", "GC", "GD", "GE"], _toy_annotation())
    assert out.astype(str).values.tolist() == SEARCH_SPACE


def _toy_multiome():
    rs = np.random.RandomState(7)
    n = 80
    genes = [f"G{i}" for i in range(12)]
    expr = pd.DataFrame(rs.poisson(1.0, size=(n, 12)).astype(np.float32), columns=genes)
    expr["G5"] = expr["G0"] * 2 + rs.poisson(0.3, n)
    regions = [f"chr1:{1000 * i}-{1000 * i + 500}" for i in range(1, 30)]
    acc = rs.poisson(2.0, size=(n, len(regions))).astype(np.int32)
    ss = pd.DataFrame({"Name": regions, "Gene": [genes[i % 12] for i in range(len(regions))],
                       "Distance": ["[0]"] * len(regions)})
    return expr, acc, regions, ss


def test_tf_to_gene_schema_and_thread_invariance():
    expr, *_ = _toy_multiome()
    tfs = ["G0", "G1", "G2", "G3"]
    a = sc.tf_to_gene(expr, tfs, n_jobs=1)
    b = sc.tf_to_gene(expr, tfs, n_jobs=3)
    assert list(a.columns) == ["TF", "target", "importance", "regulation", "rho"]
    pd.testing.assert_frame_equal(a, b)
    self_rows = a[a.TF == a.target]
    assert sorted(self_rows.TF) == sorted(set(a.TF))
    assert (a.importance > 0).all()
    g5 = a[(a.target == "G5")].sort_values("importance", ascending=False)
    assert g5.TF.iloc[0] == "G0" and g5.regulation.iloc[0] == 1


def test_region_to_gene_schema_and_thread_invariance():
    expr, acc, regions, ss = _toy_multiome()
    a = sc.region_to_gene(expr, acc, ss, n_jobs=1, accessibility_var_names=regions)
    b = sc.region_to_gene(expr, acc, ss, n_jobs=4, accessibility_var_names=regions)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["target", "region", "importance", "rho", "importance_x_rho",
                               "importance_x_abs_rho", "Distance"]
    assert len(a) == len(ss)
    np.testing.assert_allclose(a.groupby("target").importance.sum(), 1.0)
    c = sc.region_to_gene(expr, acc, ss, importance="RF", n_jobs=2, accessibility_var_names=regions,
                          targets=["G1", "G2"])
    assert set(c.target) == {"G1", "G2"}


def test_hashseed0_set_order():
    items = ["TF%d" % i for i in range(50)]
    order = sc._links.hashseed0_set_order(items)
    assert sorted(order) == sorted(items)
    assert order == sc._links.hashseed0_set_order(items)
    both = sc._links.hashseed0_set_order(items, intersect_with=items[::2])
    assert sorted(both) == sorted(items[::2])


# ---------------------------------------------------------------------------
# parity against scenicplus reference outputs (generated outside the repository)
# ---------------------------------------------------------------------------
_PARITY = os.environ.get("EPIONE_SCENIC_PARITY_DIR")


@pytest.mark.skipif(not _PARITY or not (Path(_PARITY or ".") / "A" / "r2g").exists(),
                    reason="EPIONE_SCENIC_PARITY_DIR with scenicplus reference outputs not available")
def test_parity_with_scenicplus_subset():
    spec = importlib.util.spec_from_file_location(
        "parity_scenic_links", Path(__file__).parent / "parity" / "scenic_links" / "parity_check.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    root = Path(_PARITY)
    expr, acc, regions = pc.load(root)
    genes = list(expr.columns)
    ss = pc.check_search_space(root, regions, genes)
    assert ss["identical"]
    rs = np.random.RandomState(0)
    expressed = np.flatnonzero(expr.to_numpy().sum(0) > 0)
    targets = [genes[i] for i in sorted(rs.choice(expressed, 24, replace=False))]
    tf = pc.check_tf2g(root, expr, 4, targets)
    assert tf["same_link_set"] and tf["importance_max_rel"] <= 1e-8
    assert tf["rho_max_abs"] <= 1e-8 and tf["regulation_mismatch"] == 0
    for kind in ("GBM", "RF"):
        r = pc.check_r2g(root, expr, acc, regions, kind, 4, targets)
        assert r["same_rows_and_order"] and r["distance_identical"] and r["rho_nan_identical"]
        assert r["importance_max_rel"] <= 1e-8 and r["rho_max_abs"] <= 1e-8
        assert r["importance_x_rho_max_rel"] <= 1e-8
