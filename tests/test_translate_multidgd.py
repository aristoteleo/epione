"""multiDGD in epione.translate: likelihood, prior, network shapes and the fit/predict API."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import epione.translate as T

torch = pytest.importorskip("torch")
from epione.translate import _multidgd as D  # noqa: E402


def _log_nb_dense(k, m, r):
    """The reference's logNBdensity, elementwise."""
    eps = 1e-10
    x = torch.lgamma(k + r) - torch.lgamma(r) - torch.lgamma(k + 1)
    inv = 1 / (r + m + eps)
    return x + k * torch.log(m * inv + eps) + r * torch.log(r * inv)


def _counts(n, f, density, seed):
    rng = np.random.default_rng(seed)
    X = sp.random(n, f, density=density, random_state=seed,
                  data_rvs=lambda s: rng.integers(1, 12, s)).tocsr().astype(np.float32)
    return X


def test_sparse_nb_equals_dense_density():
    X = _counts(41, 67, 0.15, 0)
    sc = D._SparseCounts(X, "cpu")
    idx = torch.tensor([5, 0, 40, 17, 3, 22])
    mu = torch.softmax(torch.randn(len(idx), 67, dtype=torch.float64), -1).float()
    r = torch.rand(1, 67) * 3 + 1
    k = torch.from_numpy(X[idx.numpy()].toarray())
    ref = _log_nb_dense(k, sc.library[idx][:, None] * mu, r)
    nz = sc.rows(idx)
    assert torch.allclose(D._nb_logprob(mu, r, sc.library[idx], nz), ref.sum(), rtol=1e-6)
    assert torch.allclose(D._nb_logprob(mu, r, sc.library[idx], nz, per_cell=True), ref.sum(1), rtol=1e-5, atol=1e-4)


def test_sparse_rows_gathers_requested_rows():
    X = _counts(30, 20, 0.3, 1)
    sc = D._SparseCounts(X, "cpu")
    idx = torch.tensor([29, 2, 2, 11])
    rows, cols, k, lgk1 = sc.rows(idx)
    dense = torch.zeros(len(idx), 20)
    dense[rows, cols] = k
    assert np.array_equal(dense.numpy(), X[idx.numpy()].toarray())
    assert torch.allclose(lgk1, torch.lgamma(k + 1))


def test_gmm_single_component_is_gaussian_plus_prior():
    torch.manual_seed(0)
    g = D.GaussianMixture(1, 4, mean_init=(2.0, 5.0), sd_init=(0.4, 1.0))
    z = torch.randn(7, 4)
    var = torch.exp(-g.neglogvar[0])
    logn = torch.distributions.Normal(g.mean[0], var.sqrt()).log_prob(z).sum(-1)
    assert torch.allclose(-g(z), logn + g._prior_log_prob(), atol=1e-4)


def test_decoder_shapes_follow_multidgd():
    cfg = D.MultiDGDConfig()
    dec = D.Decoder(20, [3000, 99272], cfg)
    main = [m for m in dec.main if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in main] == [(20, 100), (100, 100)]
    rna = [(m.in_features, m.out_features) for m in dec.out_modules[0].fc if isinstance(m, torch.nn.Linear)]
    atac = [(m.in_features, m.out_features) for m in dec.out_modules[1].fc if isinstance(m, torch.nn.Linear)]
    assert rna == [(100, 100), (100, 100), (100, 3000)]          # max(100, round(sqrt(3000))) = 100
    assert atac == [(100, 100), (100, 315), (315, 99272)]        # round(sqrt(99272)) = 315
    assert torch.allclose(dec.out_modules[0].dispersion, torch.full((1, 3000), 2.0))
    # L1 covers the Linear weights only (not biases, not the dispersions)
    expect = sum(p.abs().sum() for n, p in dec.named_parameters() if n.endswith(".weight"))
    assert torch.allclose(dec.l1(), expect)


def test_sd_mean_follows_reference_rule():
    assert D.MultiDGDConfig().sd_mean == 0.4
    assert D.MultiDGDConfig(n_components=4).sd_mean == 0.1


def _paired(n=220, g=36, p=90, seed=0):
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, 3, n)
    Xr = rng.poisson(0.5 + 4 * (lab[:, None] == np.arange(g) % 3), (n, g)).astype(np.float32)
    Xa = rng.poisson(0.1 + 1.5 * (lab[:, None] == np.arange(p) % 3), (n, p)).astype(np.float32)
    obs = pd.DataFrame({"cell_type": lab.astype(str)}, index=[f"c{i}" for i in range(n)])
    rna = AnnData(sp.csr_matrix(Xr), obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))
    atac = AnnData(sp.csr_matrix(Xa), obs=obs.copy(),
                   var=pd.DataFrame(index=[f"chr{1 + i // 45}:{i * 1000}-{i * 1000 + 500}" for i in range(p)]))
    return rna, atac


@pytest.fixture(scope="module")
def trained():
    rna, atac = _paired()
    model = T.fit(rna[:180], atac[:180], method="multidgd", device="cpu", seed=3,
                  max_epochs=40, train_minimum=5, stop_after=3, inference_epochs=5)
    return model, rna, atac


def test_multidgd_registered():
    assert "multidgd" in T.available_methods()


def test_predicts_fractions_both_directions(trained):
    model, rna, atac = trained
    pa = model.predict(rna[180:], device="cpu")
    pr = model.predict(atac[180:], device="cpu")
    assert pa.shape == (40, atac.n_vars) and pr.shape == (40, rna.n_vars)
    assert list(pa.var_names) == list(atac.var_names)
    assert np.allclose(np.asarray(pa.X).sum(1), 1, atol=1e-4)
    assert np.allclose(np.asarray(pr.X).sum(1), 1, atol=1e-4)
    assert model.history["epochs"] == len(model.history["loss"])


def test_reconstruct_uses_both_modalities(trained):
    model, rna, atac = trained
    jr, ja = model.reconstruct(rna[180:], atac[180:], device="cpu")
    assert jr.shape == (40, rna.n_vars) and ja.shape == (40, atac.n_vars)
    src = model.predict(rna[180:], device="cpu")
    assert not np.allclose(np.asarray(src.X), np.asarray(ja.X))


def test_seed_deterministic_on_cpu(trained):
    model, rna, atac = trained
    again = T.fit(rna[:180], atac[:180], method="multidgd", device="cpu", seed=3,
                  max_epochs=40, train_minimum=5, stop_after=3, inference_epochs=5)
    assert again.history["loss"] == model.history["loss"]
    assert np.allclose(np.asarray(again.predict(rna[180:], device="cpu").X),
                       np.asarray(model.predict(rna[180:], device="cpu").X))


def test_save_load_roundtrip(trained, tmp_path):
    model, rna, atac = trained
    f = tmp_path / "m.pt"
    model.save(str(f))
    back = T.load(str(f))
    assert np.allclose(np.asarray(back.predict(atac[180:], device="cpu").X),
                       np.asarray(model.predict(atac[180:], device="cpu").X), atol=1e-6)


def test_stop_window_matches_train_dgd():
    stop = D._StopWindow(train_minimum=2, stop_after=3)
    losses = [9, 9, 5, 4, 6, 3, 7, 8, 9, 10]
    hits = [e for e, l in enumerate(losses) if stop(e, l)]
    # window [5, 4, 6] after epoch 4; 3 continues -> [4, 6, 3]; 7 > all -> stop at epoch 6
    assert hits[0] == 6
    tie = D._StopWindow(0, 2)
    assert not any(tie(e, 1.0) for e in range(50))   # equal losses never stop


def test_warns_on_non_counts():
    rna, atac = _paired(n=60)
    rna.X = rna.X * 0.5
    with pytest.warns(UserWarning, match="raw counts"):
        T.fit(rna, atac, method="multidgd", device="cpu", seed=0, max_epochs=1)


def test_prepare_counts_keeps_raw_counts():
    rng = np.random.default_rng(0)
    n, g, p = 150, 300, 80
    rna = AnnData(sp.csr_matrix(rng.poisson(rng.gamma(0.5, 2, g), (n, g)).astype(np.float32)),
                  obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                  var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))
    Xa = rng.poisson(0.05, (n, p)).astype(np.float32)
    Xa[:, 0] = 0
    Xa[0, 0] = 3          # open in one cell: kept (threshold ceil(0.005 * 150) = 1)
    Xa[:, 1] = 0          # open in no cell: filtered
    atac = AnnData(sp.csr_matrix(Xa), obs=rna.obs.copy(),
                   var=pd.DataFrame(index=[f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(p)]))
    r, a = T.prepare_counts(rna, atac, n_top_genes=50)
    hvg = T.prepare_rna(rna, n_top_genes=50).var_names
    assert list(r.var_names) == list(hvg)
    assert np.array_equal(r.X.toarray(), rna[:, hvg].X.toarray())
    assert "chr1:1000-1500" not in a.var_names and "chr1:0-500" in a.var_names
    assert np.array_equal(a[:, "chr1:0-500"].X.toarray().ravel(), Xa[:, 0])
    assert math.isclose(float(a.X.max()), float(Xa.max()))      # counts are not rescaled
