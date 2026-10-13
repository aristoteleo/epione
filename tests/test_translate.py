"""epione.translate: preprocessing, metrics (against independent computations),
the benchmark tables and a small CPU scButterfly run on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import epione.translate as T
from epione.translate import _metrics as M


# --------------------------------------------------------------- fixtures
def _synthetic(n=240, g=120, per_chrom=60, seed=0):
    rng = np.random.default_rng(seed)
    types = rng.integers(0, 3, n)
    peaks = [f"chr{c}:{i * 1000}-{i * 1000 + 500}" for c in (2, 1, 3) for i in range(per_chrom)]
    mu_r = rng.gamma(1.0, 1.0, (3, g))
    mu_a = rng.random((3, len(peaks))) * 0.4
    R = rng.poisson(mu_r[types] * 3).astype(np.float32)
    A = (rng.random((n, len(peaks))) < mu_a[types]).astype(np.float32) * rng.integers(1, 3, (n, len(peaks)))
    obs = pd.DataFrame({"cell_type": pd.Categorical(types.astype(str))}, index=[f"c{i}" for i in range(n)])
    rna = AnnData(sp.csr_matrix(R), obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))
    atac = AnnData(sp.csr_matrix(A.astype(np.float32)), obs=obs.copy(), var=pd.DataFrame(index=peaks))
    return rna, atac


# ---------------------------------------------------------- preprocessing
def test_tfidf_matches_dense_formula():
    rng = np.random.default_rng(1)
    X = (rng.random((30, 50)) < 0.3).astype(float)
    X[:, 0] = 1.0
    got = T.tfidf(sp.csr_matrix(X)).toarray()
    # scButterfly's TFIDF on the peaks x cells transpose
    C = X.T
    nf = C / C.sum(axis=0, keepdims=True)
    idf = np.log(1 + C.shape[1] / C.sum(axis=1)).reshape(-1, 1)
    np.testing.assert_allclose(got, (nf * idf).T, rtol=1e-12)


def test_prepare_atac_filters_scales_and_sorts():
    rna, atac = _synthetic()
    p = T.prepare_atac(atac, min_cells_frac=0.05)
    assert p.X.max() == pytest.approx(1.0)
    assert p.X.min() >= 0
    n_open = np.asarray((atac.X > 0).sum(0)).ravel()
    assert p.n_vars == int((n_open >= np.ceil(0.05 * atac.n_obs)).sum())
    chroms = T.peak_chromosomes(p)
    # peaks of each chromosome are contiguous, chromosomes in first-seen order
    assert list(dict.fromkeys(chroms)) == ["chr2", "chr1", "chr3"]
    assert sum(T.chrom_blocks(p)) == p.n_vars


def test_prepare_rna_and_pairing():
    rna, atac = _synthetic()
    r, a = T.prepare(rna, atac[::-1], n_top_genes=50)
    assert r.n_vars == 50
    assert np.array_equal(r.obs_names, a.obs_names)
    # log1p of normalised counts
    assert r.X.max() < 20


def test_chrom_blocks_rejects_interleaved():
    _, atac = _synthetic(per_chrom=3)
    atac = atac[:, [0, 3, 1]]
    with pytest.raises(ValueError):
        T.chrom_blocks(atac)


# ------------------------------------------------------------- metrics
@pytest.fixture(scope="module")
def pt():
    rng = np.random.default_rng(2)
    truth = rng.random((60, 40))
    pred = truth + rng.normal(0, 0.3, truth.shape)
    return pred, truth


def test_per_cell_and_feature_correlations(pt):
    from scipy.stats import pearsonr, spearmanr

    pred, truth = pt
    np.testing.assert_allclose(M.pearson_per_cell(pred, truth),
                               [pearsonr(p, t)[0] for p, t in zip(pred, truth)], rtol=1e-10)
    np.testing.assert_allclose(M.spearman_per_cell(pred, truth),
                               [spearmanr(p, t)[0] for p, t in zip(pred, truth)], rtol=1e-10)
    np.testing.assert_allclose(M.pearson_per_feature(pred, truth),
                               [pearsonr(p, t)[0] for p, t in zip(pred.T, truth.T)], rtol=1e-10)


def test_constant_profile_is_nan_and_skipped(pt):
    pred, truth = pt
    pred = pred.copy()
    pred[0] = 1.0
    r = M.pearson_per_cell(pred, truth)
    assert np.isnan(r[0]) and np.isfinite(r[1:]).all()
    s = M.regression_scores(pred, truth)
    assert s["PCC"] == pytest.approx(np.mean(r[1:]))
    # SPCC is divided by the number of cells with a defined PCC
    assert s["SPCC"] == pytest.approx(np.nansum(M.spearman_per_cell(pred, truth)) / 59)


def test_mse_mae(pt):
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    pred, truth = pt
    s = M.regression_scores(sp.csr_matrix(pred), truth)
    assert s["MSE"] == pytest.approx(np.mean([mean_squared_error(t, p) for p, t in zip(pred, truth)]))
    assert s["MAE"] == pytest.approx(np.mean([mean_absolute_error(t, p) for p, t in zip(pred, truth)]))


def test_mmd_against_explicit_kernel_sum():
    rng = np.random.default_rng(3)
    x, y = rng.normal(0, 1, (20, 3)), rng.normal(0.5, 1, (20, 3))
    Z = np.vstack([x, y])
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    bw = D.sum() / (len(Z) ** 2 - len(Z)) / 2.0 ** 2
    K = sum(np.exp(-D / (bw * 2.0**i)) for i in range(5))
    n = len(x)
    expect = np.mean(K[:n, :n] + K[n:, n:] - K[:n, n:] - K[n:, :n])
    assert M.mmd_rbf(x, y) == pytest.approx(expect, rel=1e-10)
    assert M.mmd_rbf(x, x) == pytest.approx(0.0, abs=1e-12)


def test_lisi_against_bruteforce():
    rng = np.random.default_rng(4)
    emb = rng.normal(0, 1, (80, 2))
    lab = rng.integers(0, 2, 80)
    got = M.lisi(emb, lab, perplexity=5)
    # brute force: bisection on beta for each cell over its 15 nearest neighbours
    D = np.sqrt(((emb[:, None] - emb[None]) ** 2).sum(-1))
    out = []
    for i in range(len(emb)):
        order = np.argsort(D[i])[1:16]
        d = D[i, order]
        lo, hi = 1e-6, 1e6
        for _ in range(200):
            beta = np.sqrt(lo * hi)
            P = np.exp(-d * beta)
            P /= P.sum()
            H = -(P[P > 0] * np.log(P[P > 0])).sum()
            if H > np.log(5):
                lo = beta
            else:
                hi = beta
        w = np.bincount(lab[order], weights=P, minlength=2)
        out.append(1 / (w**2).sum())
    np.testing.assert_allclose(got, out, rtol=2e-3)


def test_graph_knn_matches_networkx_dijkstra():
    import networkx as nx

    rng = np.random.default_rng(7)
    A = sp.random(60, 60, density=0.08, random_state=1, format="csr")
    A = (A + A.T).tocsr()
    A.data = rng.random(A.nnz) + 0.01
    A = sp.csr_matrix((A + A.T) / 2)
    idx, dist = M._graph_knn(A, 10)
    G = nx.from_scipy_sparse_array(A)
    for i in range(60):
        lengths = nx.single_source_dijkstra_path_length(G, i)
        lengths.pop(i)
        ref = sorted(lengths.values())[:10]
        got = dist[i][np.isfinite(dist[i])]
        np.testing.assert_allclose(got, ref, rtol=1e-12)
        if len(ref) < 10:
            assert (idx[i] == -1).sum() == 10 - len(ref)


def test_ilisi_mixed_vs_separated():
    rng = np.random.default_rng(5)
    truth = rng.normal(0, 1, (100, 10))
    for method in ("graph", "embedding"):
        assert M.ilisi(truth + rng.normal(0, 0.05, truth.shape), truth, method=method) > 0.6
        assert M.ilisi(truth + 50.0 * np.eye(10)[0], truth, method=method) < 0.05


def test_auroc_auprc_against_sklearn(pt):
    from sklearn.metrics import average_precision_score, roc_auc_score

    pred, truth = pt
    y = (truth > np.median(truth)).astype(int)
    assert M.auroc(pred, truth) == pytest.approx(roc_auc_score(y.ravel(), pred.ravel()))
    assert M.auprc(pred, truth) == pytest.approx(average_precision_score(y.ravel(), pred.ravel()))
    per_feat = np.mean([roc_auc_score(yy, pp) for yy, pp in zip(y.T, pred.T)])
    assert M.auroc(pred, truth, average="feature") == pytest.approx(per_feat)
    yb = (truth > 0.5).astype(int)
    assert M.auroc(pred, truth, binarize=0.5) == pytest.approx(roc_auc_score(yb.ravel(), pred.ravel()))


def test_cluster_scores_recover_clear_groups():
    pytest.importorskip("leidenalg")
    rng = np.random.default_rng(6)
    lab = np.repeat(np.arange(3), 60)
    X = rng.normal(0, 0.3, (180, 20)) + 6 * np.eye(20)[lab]
    s = M.cluster_scores(X, lab, n_pcs=10, resolution=0.3)
    assert s["ARI"] > 0.95 and s["NMI"] > 0.95 and s["HOM"] > 0.95


# ----------------------------------------------------------- benchmark
def _fake_results():
    rows = []
    for meth, v in [("good", 0.9), ("bad", 0.1)]:
        rows += [("rna2atac", meth, "PCC", "regression", True, v),
                 ("rna2atac", meth, "MSE", "regression", False, 1 - v),
                 ("rna2atac", meth, "AUROC", "classification", True, 0.5 + v / 2),
                 ("rna2atac", meth, "MMD", "distribution", False, 1.0)]
    return pd.DataFrame(rows, columns=["task", "method", "metric", "category", "higher_is_better", "value"])


def test_summarize_scaling_and_ties():
    s = T.summarize(_fake_results())
    assert s.loc[("rna2atac", "good"), ("score", "regression")] == pytest.approx(1.0)
    assert s.loc[("rna2atac", "bad"), ("score", "regression")] == pytest.approx(0.0)
    # MMD is equal for both methods -> both score 1 (no flip to 0)
    assert s.loc[("rna2atac", "bad"), ("score", "distribution")] == pytest.approx(1.0)
    assert s.loc[("rna2atac", "good"), ("score", "overall")] == pytest.approx(1.0)
    assert list(s.index.get_level_values(1)) == ["good", "bad"]


def test_plot_benchmark_draws():
    import matplotlib

    matplotlib.use("Agg")
    fig = T.plot_benchmark(_fake_results())
    assert fig.axes


# ------------------------------------------------------- scButterfly
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def trained():
    rna, atac = _synthetic()
    r, a = T.prepare(rna, atac, n_top_genes=80, min_cells_frac=0.01)
    tr, te = np.arange(180), np.arange(180, 240)
    m = T.fit(r[tr], a[tr], method="scbutterfly", device="cpu", seed=3, threads=2,
              rna_pretrain_epochs=3, atac_pretrain_epochs=3, translator_epochs=3)
    return m, r, a, te


def test_scbutterfly_predicts_both_directions(trained):
    m, r, a, te = trained
    pa = m.predict(r[te])
    pr = m.predict(a[te])
    assert pa.uns["translate"]["direction"] == "rna2atac"
    assert pa.shape == (60, a.n_vars) and pr.shape == (60, r.n_vars)
    assert list(pa.var_names) == list(a.var_names)
    assert pa.X.max() <= 1.0  # sigmoid output
    assert set(m.history) >= {"rna_pretrain", "atac_pretrain", "translator", "translator_val"}
    assert len(m.history["translator"]) == 3


def test_scbutterfly_is_seed_deterministic_on_cpu(trained):
    m, r, a, te = trained
    tr = np.arange(180)
    m2 = T.fit(r[tr], a[tr], method="scbutterfly", device="cpu", seed=3, threads=2,
               rna_pretrain_epochs=3, atac_pretrain_epochs=3, translator_epochs=3)
    np.testing.assert_array_equal(m.predict(r[te]).X.toarray(), m2.predict(r[te]).X.toarray())


def test_scbutterfly_save_load_roundtrip(trained, tmp_path):
    m, r, a, te = trained
    m.save(tmp_path / "m.pt")
    m2 = T.load(tmp_path / "m.pt")
    np.testing.assert_array_equal(m.predict(a[te]).X.toarray(), m2.predict(a[te]).X.toarray())


def test_predict_reorders_features_and_rejects_missing(trained):
    m, r, a, te = trained
    shuffled = r[te][:, r.var_names[::-1]]
    np.testing.assert_allclose(m.predict(shuffled, "rna2atac").X.toarray(),
                               m.predict(r[te], "rna2atac").X.toarray(), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        m.predict(r[te][:, :10], "rna2atac")


def test_network_shapes_follow_scbutterfly():
    from epione.translate._scbutterfly import ButterflyNet, ScButterflyConfig

    net = ButterflyNet(50, [30, 20], ScButterflyConfig())
    assert tuple(net.atac_encoder.first.weight.shape) == (2, 30, 32)
    assert net.atac_encoder.rest.linears[0].in_features == 64
    assert tuple(net.atac_decoder.last.weight.shape) == (2, 32, 30)
    assert net.atac_decoder.last_bn.num_features == 50
    assert isinstance(net.atac_decoder.last_act, torch.nn.Sigmoid)
    assert net.rna_encoder.linears[0].in_features == 50
    assert net.disc_a.linears[0].out_features == 1


def test_unknown_method():
    with pytest.raises(ValueError):
        T.fit(*_synthetic(), method="nope")



def test_chrom_linear_equals_per_chromosome_linears():
    from epione.translate._scbutterfly import ChromLinear

    torch.manual_seed(0)
    chroms = [5, 3, 7]
    enc = ChromLinear(chroms, 4, "in")
    dec = ChromLinear(chroms, 4, "out")
    x = torch.randn(6, sum(chroms))
    h = torch.randn(6, 4 * len(chroms))
    starts = np.r_[0, np.cumsum(chroms)[:-1]]
    ref_enc, ref_dec = [], []
    for j, (s0, c) in enumerate(zip(starts, chroms)):
        ref_enc.append(x[:, s0:s0 + c] @ enc.weight[j, :c, :] + enc.bias[j])
        ref_dec.append(h[:, 4 * j:4 * j + 4] @ dec.weight[j, :, :c] + dec.bias[j, :c])
    torch.testing.assert_close(enc(x), torch.cat(ref_enc, 1))
    torch.testing.assert_close(dec(h), torch.cat(ref_dec, 1))
    # padding gets no gradient
    enc(x).sum().backward()
    assert torch.all(enc.weight.grad[1, 3:, :] == 0)
    dec(h).sum().backward()
    assert torch.all(dec.weight.grad[1, :, 3:] == 0) and torch.all(dec.bias.grad[1, 3:] == 0)


def test_sctranslation_convention_swaps_roles(pt):
    from sklearn.metrics import roc_auc_score
    from sklearn.feature_selection import r_regression

    pred, truth = pt
    truth = truth.copy()
    truth[:, 0] = 0.3          # constant measured feature
    pred = pred.copy()
    pred[:, 1] = 0.5           # constant predicted feature
    y = (pred > np.median(pred)).astype(int)
    assert M.auroc(pred, truth, convention="sctranslation") == pytest.approx(roc_auc_score(y.ravel(), truth.ravel()))
    feats = [j for j in range(pred.shape[1]) if len(np.unique(pred[:, j])) > 1]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = [r_regression(truth[:, [j]], pred[:, j])[0] for j in feats]
    got = M.regression_scores(pred, truth, convention="sctranslation")["CORR"]
    assert got == pytest.approx(np.mean(ref))
    # the standard convention drops undefined features instead
    std = M.regression_scores(pred, truth)["CORR"]
    assert std == pytest.approx(np.nanmean(M.pearson_per_feature(pred, truth)))


# ------------------------------------------------------------------ BABEL
def test_size_normalize_and_babel_prepare():
    rna, atac = _synthetic()
    r, a = T.prepare(rna, atac, n_top_genes=50, method="babel")
    assert np.allclose(r.X.data, np.round(r.X.data))          # raw counts kept
    assert set(np.unique(a.X.data)) == {1.0}                    # binary peaks
    rs = T.prepare(rna, atac, n_top_genes=50)[0]
    assert list(r.var_names) == list(rs.var_names)             # same genes as scButterfly
    n = T.size_normalize(r)
    tot = np.asarray(r.X.sum(1)).ravel()
    np.testing.assert_allclose(np.asarray(n.X.sum(1)).ravel()[tot > 0], np.median(tot), rtol=1e-5)


def test_nb_nll_matches_scipy():
    from scipy.stats import nbinom
    from epione.translate._babel import nb_nll

    rng = np.random.default_rng(8)
    y = rng.integers(0, 30, 200).astype(float)
    mu = rng.gamma(2.0, 3.0, 200)
    th = rng.gamma(2.0, 1.0, 200) + 0.1
    got = nb_nll(torch.tensor(y), torch.tensor(mu), torch.tensor(th), eps=0.0).numpy()
    ref = -nbinom.logpmf(y, th, th / (th + mu))
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-8)


def test_block_linear_equals_per_block_linears():
    from epione.translate._nn import BlockLinear

    torch.manual_seed(1)
    m = BlockLinear(3, 4, 2)
    x = torch.randn(5, 12)
    ref = torch.cat([x[:, 4 * j:4 * j + 4] @ m.weight[j] + m.bias[j] for j in range(3)], 1)
    torch.testing.assert_close(m(x), ref)


@pytest.fixture(scope="module")
def babel_trained():
    rna, atac = _synthetic()
    r, a = T.prepare(rna, atac, n_top_genes=80, min_cells_frac=0.01, method="babel")
    m = T.fit(r[:180], a[:180], method="babel", device="cpu", seed=2, threads=2, max_epochs=4)
    return m, r, a


def test_babel_predicts_both_directions(babel_trained):
    m, r, a = babel_trained
    pa, pr = m.predict(r[180:]), m.predict(a[180:])
    assert pa.shape == (60, a.n_vars) and pr.shape == (60, r.n_vars)
    assert not sp.issparse(pa.X)                              # BABEL keeps dense outputs
    assert pa.X.min() >= 0 and pa.X.max() <= 1               # sigmoid ATAC head
    assert pr.X.min() > 0                                      # NB means
    assert len(m.history["val"]) == 4 and m.history["best_epoch"] <= 3


def test_babel_deterministic_and_save_load(babel_trained, tmp_path):
    m, r, a = babel_trained
    m2 = T.fit(r[:180], a[:180], method="babel", device="cpu", seed=2, threads=2, max_epochs=4)
    np.testing.assert_array_equal(m.predict(a[180:]).X, m2.predict(a[180:]).X)
    m.save(tmp_path / "b.pt")
    m3 = T.load(tmp_path / "b.pt")
    np.testing.assert_array_equal(m.predict(r[180:]).X, m3.predict(r[180:]).X)


def test_benchmark_per_method_truth_and_exclude():
    rng = np.random.default_rng(9)
    t1, t2 = rng.random((40, 12)), rng.random((40, 12)) * 100
    lab = np.repeat([0, 1], 20)
    mk = lambda x: AnnData(x.astype(np.float32), obs=pd.DataFrame({"cell_type": lab.astype(str)}, index=[f"c{i}" for i in range(40)]))
    res = T.benchmark({"t": {"a": mk(t1 + 0.1), "b": mk(t2 + 1)}}, {"t": {"a": mk(t1), "b": mk(t2)}},
                      labels="cell_type", categories=("regression",), exclude=("MSE", "MAE"))
    assert set(res["metric"]) == {"PCC", "SPCC", "CORR"}
    assert np.allclose(res.loc[res.metric == "PCC", "value"], 1.0)
