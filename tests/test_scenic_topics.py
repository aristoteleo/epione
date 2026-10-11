"""Tests for ``epione.scenic`` topic block (pycisTopic port).

Unit tests run on small synthetic data. The parity tests compare with
pycisTopic reference outputs generated outside this repository; they run when
``EPIONE_SCENIC_PARITY_DIR`` points at them (see
``tests/parity/scenic_topics/MANIFEST.md``) and are skipped otherwise.
"""
from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats

from epione import scenic
from epione.scenic import _topics as T
from epione.scenic._np126 import argpartition_np126, argsort_np126, log_np126, rowsum_compat
from epione.scenic._topics_dar import _p_adjust_bh


# ---------------------------------------------------------------------------
# numpy 1.26 kernels
# ---------------------------------------------------------------------------

# sha256[:16] of numpy 1.26.4 ``np.argsort(x)`` (AVX-512 CPU) on the arrays below
_ARGSORT_NP126 = {
    (37, 4): "2dfe07fcaf0bcae1",
    (100, 3): "4a4773fc2e4350b7",
    (300, 5): "fa576fdc3c1b6495",
    (1000, 7): "ccf69edd461ca8bc",
    (5000, 2): "a767325cbe63c2c1",
    (20000, 11): "dbae4d02549212c0",
}


@pytest.mark.parametrize("n,nv", list(_ARGSORT_NP126))
def test_argsort_np126_matches_numpy126(n, nv):
    i = np.arange(n, dtype=np.int64)
    x = ((i * 7919 + (i * i) % 13) % nv) / 3.0
    perm = argsort_np126(x)
    assert np.all(np.diff(x[perm]) >= 0)
    assert sorted(perm.tolist()) == list(range(n))
    digest = hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest()[:16]
    assert digest == _ARGSORT_NP126[(n, nv)]


def test_argsort_np126_distinct_values():
    x = np.cos(np.arange(5000) * 1.3)
    assert np.array_equal(argsort_np126(x), np.argsort(x, kind="stable"))


# sha256[:12] of numpy 1.26.4 ``np.argpartition(x, kth)``
_ARGPART_NP126 = [(6, 1, "c4af4158cc4e"), (10, 5, "4ceedce3c373"), (20, 15, "02c09782038f"),
                  (30, 25, "616ab0a0181b"), (40, 35, "d437481d9242"), (57, 28, "21e3516e646e"),
                  (100, 98, "160f3a60f976"), (300, 150, "e797bf3a0fc0")]


@pytest.mark.parametrize("n,kth,digest", _ARGPART_NP126)
def test_argpartition_np126_matches_numpy126(n, kth, digest):
    i = np.arange(n)
    x = np.sin(i * 1.7) + (i % 5) * 0.1
    x[::7] = x[0]
    perm = argpartition_np126(x, kth)
    assert np.all(x[perm[:kth]] <= x[perm[kth]]) and np.all(x[perm[kth + 1:]] >= x[perm[kth]])
    assert hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest()[:12] == digest


def test_rowsum_np126_buffered_pairwise():
    x = np.cos(np.arange(3 * 20000) * 0.37).reshape(3, 20000) + 2.0
    got = rowsum_compat(x, "numpy126")
    assert np.allclose(got, x.sum(axis=1), rtol=1e-14, atol=0)
    # numpy 1.26 adds the pairwise sums of 8192-element chunks in sequence
    from epione.scenic._np126 import _pairwise_sum
    expected = [sum((_pairwise_sum(r, s, min(8192, r.size - s)) for s in range(0, r.size, 8192)), 0.0)
                for r in x]
    assert got.tolist() == expected


# numpy 1.26.4 ``np.log`` (SVML ``__svml_log8``) of the inputs below
_LOG_IN = [1e-300, 3.7e-50, 1e-12, 0.001234, 0.1, 0.33, 0.7, 0.93, 0.999, 1.0, 1.0001, 1.06,
           1.5, 2.0, 3.14159, 10.0, 12345.678, 6.02e23, 1.7e308, 5e-324]
_LOG_NP126 = [
    "-0x1.5963447f87fb5p+9", "-0x1.c7489fbb7295cp+6", "-0x1.ba18a998fffa0p+4",
    "-0x1.aca3bf5b5b0c2p+2", "-0x1.26bb1bbb55515p+1", "-0x1.1bd150529e2b4p+0",
    "-0x1.6d3c324e13f50p-2", "-0x1.293fe305fa975p-4", "-0x1.064670d979b73p-10", "0x0.0p+0",
    "0x1.a368d0657fcd4p-14", "0x1.dd56c1d883ee7p-5", "0x1.9f323ecbf984cp-2",
    "0x1.62e42fefa39efp-1", "0x1.250cf662abf9ap+0", "0x1.26bb1bbb55515p+1",
    "0x1.2d79559791e31p+3", "0x1.b6094e92cca28p+5", "0x1.62dd08fdc6f88p+9",
    "-0x1.74385446d71c3p+9",
]


def test_log_np126_matches_svml():
    got = log_np126(np.array(_LOG_IN))
    assert [float.hex(float(v)) for v in got] == _LOG_NP126
    assert np.allclose(got, np.log(_LOG_IN), rtol=1e-15, atol=0)


def test_log_np126_special_values():
    out = log_np126(np.array([0.0, -1.0, np.inf, np.nan]))
    assert out[0] == -np.inf and np.isnan(out[1]) and out[2] == np.inf and np.isnan(out[3])


# ---------------------------------------------------------------------------
# LDA collapsed Gibbs sampler
# ---------------------------------------------------------------------------

def _reference_lda(X, n_topics, n_iter, alpha, eta, seed):
    """Transcription of the sampler of ``lda`` 3.0.0 (``lda.py`` + ``_lda.pyx``).

    ``lda`` is (c) Allen B. Riddell and contributors, Mozilla Public License 2.0.
    """
    D, W = X.shape
    rands = np.random.RandomState(seed).rand(1024 ** 2 // 8)
    rng = np.random.RandomState(seed)
    rands = rands.copy()
    ii, jj = np.nonzero(X)
    ss = np.asarray(X[ii, jj]).ravel()
    DS = np.repeat(ii, ss)
    WS = np.repeat(jj, ss)
    nzw = np.zeros((n_topics, W), dtype=np.intc)
    ndz = np.zeros((D, n_topics), dtype=np.intc)
    nz = np.zeros(n_topics, dtype=np.intc)
    ZS = np.empty_like(WS)
    for i in range(len(WS)):
        z = i % n_topics
        ZS[i] = z
        ndz[DS[i], z] += 1
        nzw[z, WS[i]] += 1
        nz[z] += 1
    eta_v = np.repeat(eta, W).astype(float)
    alpha_v = np.repeat(alpha, n_topics).astype(float)
    eta_sum = 0.0
    for v in eta_v:
        eta_sum += v
    for _ in range(n_iter):
        rng.shuffle(rands)
        for i in range(len(WS)):
            w, d, z = WS[i], DS[i], ZS[i]
            nzw[z, w] -= 1
            ndz[d, z] -= 1
            nz[z] -= 1
            dist = np.empty(n_topics)
            cum = 0.0
            for k in range(n_topics):
                cum += (nzw[k, w] + eta_v[w]) / (nz[k] + eta_sum) * (ndz[d, k] + alpha_v[k])
                dist[k] = cum
            r = rands[i % len(rands)] * cum
            z_new = int(np.searchsorted(dist, r, side="left"))
            ZS[i] = z_new
            nzw[z_new, w] += 1
            ndz[d, z_new] += 1
            nz[z_new] += 1
    return nzw, ndz, nz


def _toy_binary(n_cells=60, n_regions=150, seed=0):
    rng = np.random.RandomState(seed)
    prog = rng.randint(0, 3, n_cells)
    p = np.full((n_cells, n_regions), 0.05)
    for c in range(3):
        p[prog == c, c * 50:(c + 1) * 50] = 0.6
    return sp.csr_matrix((rng.rand(n_cells, n_regions) < p).astype(np.int32))


def test_lda_sampler_matches_reference_transcription():
    X = _toy_binary()
    nzw, ndz, nz = T._fit_cgs(X, n_topics=4, n_iter=7, alpha=50 / 4, eta=0.1, random_state=555)
    rnzw, rndz, rnz = _reference_lda(X.toarray(), 4, 7, 50 / 4, 0.1, 555)
    assert np.array_equal(nzw, rnzw)
    assert np.array_equal(ndz, rndz)
    assert np.array_equal(nz, rnz)


def test_run_topic_models_outputs():
    X = _toy_binary()
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(X.shape[0])]
    a.var_names = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(X.shape[1])]
    models = scenic.run_topic_models(a, [3, 5, 8], n_iter=20, method="exact", verbose=False)
    assert [m.n_topic for m in models] == [3, 5, 8]
    m = models[0]
    assert m.cell_topic.shape == (3, 60) and m.topic_region.shape == (150, 3)
    assert np.allclose(m.cell_topic.sum(axis=0), 1)
    assert np.allclose(m.topic_region.sum(axis=0), 1)
    assert int(m.topic_ass["Assignments"].sum()) == X.sum()
    # threads give the same result as sequential fitting
    seq = scenic.run_topic_models(a, [3, 5, 8], n_iter=20, n_jobs=1, method="exact",
                                  verbose=False)
    assert np.array_equal(seq[1].topic_region.values, models[1].topic_region.values)
    best, table = scenic.evaluate_topic_models(models, return_metrics=True)
    # topics with fewer than min_topics_coh (5) topics are not candidates
    assert best.n_topic in (5, 8) and list(table.index) == [5, 8] and "combined" in table
    scenic.add_topic_model(a, best)
    assert a.obsm["X_topic"].shape == (60, best.n_topic)
    assert a.varm["topic"].shape == (150, best.n_topic)


def _toy_adata():
    X = _toy_binary(n_cells=90, n_regions=150, seed=1)
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(X.shape[0])]
    a.var_names = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(X.shape[1])]
    return a


def test_parallel_sampler_counts_and_reproducibility():
    a = _toy_adata()
    m1 = scenic.run_topic_models(a, [3, 4], n_iter=40, n_jobs=3, random_state=7, verbose=False)
    m2 = scenic.run_topic_models(a, [3, 4], n_iter=40, n_jobs=3, random_state=7, verbose=False)
    for x, y in zip(m1, m2):
        assert np.array_equal(x.topic_region.values, y.topic_region.values)
        nzw, ndz = x.extra["nzw"], x.extra["ndz"]
        assert (nzw >= 0).all() and (ndz >= 0).all()
        assert nzw.sum() == ndz.sum() == a.X.sum()
        assert np.array_equal(ndz.sum(axis=1), np.asarray(a.X.sum(axis=1)).ravel())
        assert np.array_equal(nzw.sum(axis=0), np.asarray(a.X.sum(axis=0)).ravel())
        assert x.parameters.loc["package", "Parameter"] == "epione-parallel"


def test_parallel_sampler_recovers_programs():
    from sklearn.metrics import adjusted_rand_score

    a = _toy_adata()
    truth = np.random.RandomState(1).randint(0, 3, 90)  # programs used by _toy_binary(seed=1)
    exact = scenic.run_topic_models(a, [3], n_iter=100, method="exact", verbose=False)[0]
    fast = scenic.run_topic_models(a, [3], n_iter=100, n_jobs=4, verbose=False)[0]
    for m in (exact, fast):
        assert adjusted_rand_score(truth, m.cell_topic.values.argmax(0)) > 0.9


def test_gpu_sampler_on_cpu_device():
    torch = pytest.importorskip("torch")
    a = _toy_adata()
    m = scenic.run_topic_models(a, [3], n_iter=60, method="gpu", device="cpu", verbose=False)[0]
    nzw, ndz = m.extra["nzw"], m.extra["ndz"]
    assert nzw.sum() == ndz.sum() == a.X.sum()
    from sklearn.metrics import adjusted_rand_score

    truth = np.random.RandomState(1).randint(0, 3, 90)
    assert adjusted_rand_score(truth, m.cell_topic.values.argmax(0)) > 0.9


def test_loglikelihood_reproduces_pycistopic_quirk():
    import math

    nzw = np.array([[0, 3, 1], [2, 0, 0]], dtype=np.intc)
    ndz = np.array([[1, 2], [3, 0]], dtype=np.intc)
    alpha, eta = 50.0, 0.1
    # pycisTopic.utils.loglikelihood, written out
    D, K, V = 2, 2, 3
    const_prior = (K * math.lgamma(alpha) - math.lgamma(alpha * K)) * D
    const_ll = (V * math.lgamma(eta) - math.lgamma(eta * V)) * K
    topic_ll = math.lgamma(1 + eta) - math.lgamma(eta * V + 3 + 1)  # overwritten below
    topic_ll = math.lgamma(2 + eta) - math.lgamma(eta * V + 2)
    doc_ll = math.lgamma(3 + alpha) - math.lgamma(alpha * K + 3)
    expected = doc_ll - const_prior + topic_ll - const_ll
    assert T._loglikelihood(nzw, ndz, alpha, eta) == expected


# ---------------------------------------------------------------------------
# Binarization, imputation, DARs
# ---------------------------------------------------------------------------

def _toy_model():
    X = _toy_binary()
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(X.shape[0])]
    a.var_names = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(X.shape[1])]
    return a, scenic.run_topic_models(a, [3], n_iter=30, verbose=False)[0]


def test_binarize_topics_methods():
    _, m = _toy_model()
    for method, kw in [("otsu", {}), ("ntop", {"ntop": 20}), ("li", {}), ("yen", {})]:
        b = scenic.binarize_topics(m, method=method, **kw)
        assert list(b) == ["Topic1", "Topic2", "Topic3"]
        for t, df in b.items():
            assert (np.diff(df.iloc[:, 0].values) <= 0).all()
    top = scenic.binarize_topics(m, method="ntop", ntop=20)
    # regions scoring above the 21st highest value (ties can make it fewer)
    assert all(0 < len(df) <= 20 for df in top.values())
    cells = scenic.binarize_topics(m, target="cell", method="li")
    assert all(df.index.str.startswith("c").all() for df in cells.values())


def test_impute_normalize_hvf_dar():
    a, m = _toy_model()
    imp = scenic.impute_accessibility(m, scale_factor=10 ** 6)
    assert imp.mtx.dtype == np.int32
    expected = (m.topic_region.to_numpy().astype(np.float32)
                @ m.cell_topic.to_numpy().astype(np.float32)) * np.float32(1e6)
    assert np.array_equal(imp.mtx, expected.astype(np.int32)[np.count_nonzero(
        expected.astype(np.int32), axis=1) > 0])
    norm = scenic.normalize_scores(imp, 10 ** 4)
    assert np.allclose(norm.mtx, np.log1p(imp.mtx / (imp.mtx.sum(axis=0) / 1e4)))
    hvf = scenic.find_highly_variable_features(norm, min_disp=0.0, min_mean=0.0, max_mean=1e9)
    assert set(hvf) <= set(imp.feature_names)
    groups = pd.Series(np.where(np.arange(60) % 2 == 0, "A", "B"), index=a.obs_names)
    dars = scenic.find_diff_features(imp, groups, adjpval_thr=1.0, log2fc_thr=-np.inf)
    assert set(dars) == {"A", "B"}
    # Wilcoxon p-values and fold changes equal scipy's per-region computation
    df = dars["A"].loc[imp.feature_names]
    fg = imp.mtx[:, ::2]
    bg = imp.mtx[:, 1::2]
    p = np.array([stats.ranksums(fg[i], bg[i]).pvalue for i in range(fg.shape[0])])
    assert np.allclose(df["Adjusted_pval"].values, _p_adjust_bh(p), rtol=1e-12, atol=0)
    lfc = np.log2((fg.mean(axis=1) + 1e-12) / (bg.mean(axis=1) + 1e-12))
    assert np.allclose(df["Log2FC"].values, lfc, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Fragment counts
# ---------------------------------------------------------------------------

def test_cistopic_matrix_counts(tmp_path):
    frags = [
        ("chr1", 90, 120, "AAA-1", 1),    # overlaps r0
        ("chr1", 100, 150, "BBB-1", 2),   # r0
        ("chr1", 100, 300, "AAA-1", 1),   # r0 and r1 (spans)
        ("chr1", 199, 201, "CCC-1", 1),   # r0 (end 200 exclusive => start 199 < 200) and r1
        ("chr1", 200, 210, "BBB-1", 1),   # r1 only (half-open)
        ("chr1", 500, 600, "AAA-1", 1),   # blacklisted region r2
        ("chr2", 10, 20, "BBB-1", 1),     # r3
        ("chr2", 10, 20, "ZZZ-1", 1),     # barcode not selected
    ]
    fpath = tmp_path / "fragments.tsv.gz"
    with gzip.open(fpath, "wt") as fh:
        fh.write("# comment\n")
        for f in frags:
            fh.write("\t".join(map(str, f)) + "\n")
    regions = pd.DataFrame({"Chromosome": ["chr1", "chr1", "chr1", "chr2", "chr2"],
                            "Start": [100, 200, 450, 0, 1000], "End": [200, 400, 650, 50, 1100]})
    blacklist = pd.DataFrame({"Chromosome": ["chr1"], "Start": [480], "End": [490]})
    a = scenic.cistopic_matrix(fpath, regions, barcodes=["AAA-1", "BBB-1", "CCC-1"],
                               blacklist=blacklist, project="p", split_pattern="-", verbose=False)
    assert list(a.var_names) == ["chr1:100-200", "chr1:200-400", "chr2:0-50"]
    # first overlap order: r0 fragments by start, end desc -> AAA(90-120), AAA(100-300), BBB, CCC
    assert list(a.obs_names) == ["AAA-1-p", "BBB-1-p", "CCC-1-p"]
    X = a.to_df()
    assert X.loc["AAA-1-p"].tolist() == [2, 1, 0]
    assert X.loc["BBB-1-p"].tolist() == [1, 1, 1]
    assert X.loc["CCC-1-p"].tolist() == [1, 1, 0]
    assert a.obs.loc["AAA-1-p", "Unique_nr_frag"] == 3
    assert a.obs.loc["BBB-1-p", "cisTopic_nr_frag"] == 3
    assert a.var.loc["chr1:100-200", "cisTopic_nr_acc"] == 3


# ---------------------------------------------------------------------------
# Parity with pycisTopic (reference outputs generated outside the repository)
# ---------------------------------------------------------------------------

_PARITY = os.environ.get("EPIONE_SCENIC_PARITY_DIR")
_needs_parity = pytest.mark.skipif(
    not _PARITY or not (Path(_PARITY or ".") / "S" / "topics").is_dir(),
    reason="EPIONE_SCENIC_PARITY_DIR with pycisTopic reference outputs not available")


def _load_parity():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "parity_scenic_topics", Path(__file__).parent / "parity" / "scenic_topics" / "parity_topics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@_needs_parity
def test_parity_lda_small_run():
    res = _load_parity().check_lda(Path(_PARITY), run="S", n_topics=(10, 20), n_iter=5)
    for k, r in res["models"].items():
        assert r["topic_ass_equal"], k
        assert r["cell_topic_max_abs"] <= 1e-8 and r["topic_region_max_abs"] <= 1e-8, k
        assert r["metrics_max_rel"] <= 1e-8 and r["coherence_max_rel"] <= 1e-8, k
        assert r["marg_topic_max_rel"] <= 1e-8, k
    assert res["selected_equal"]
