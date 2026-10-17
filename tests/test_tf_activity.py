"""Synthetic checks of epione.tl.tf_activity (conditional count model + EB shrinkage)."""
import numpy as np
import scipy.sparse as sp
import pytest
from anndata import AnnData

from epione.tl import _tf_activity as T


def _simulate(n_cells=600, n_peaks=3000, n_motifs=6, effect=-0.7, gc_bias=0.0, seed=0):
    """Poisson counts with per-cell depth, optional cell-specific GC slope and a
    planted loss of motif-0 accessibility in the first half of the cells."""
    rng = np.random.default_rng(seed)
    gc = rng.uniform(0.3, 0.7, n_peaks)
    base = rng.gamma(1.0, 1.0, n_peaks)
    pi = base / base.sum()
    # motif 0 enriched in GC-rich peaks (so a GC bias alone would mimic activity)
    pm = np.clip(0.05 + 0.6 * (gc - 0.3), 0, 1)
    W = (rng.uniform(size=(n_peaks, n_motifs)) < np.column_stack([pm] + [np.full(n_peaks, 0.15)] * (n_motifs - 1))).astype(float)
    depth = rng.integers(800, 4000, n_cells)
    slope = rng.normal(0, gc_bias, n_cells)
    ko = np.zeros(n_cells, bool); ko[: n_cells // 2] = True
    logmu = np.log(pi)[None, :] + slope[:, None] * (gc - 0.5)[None, :]
    logmu = logmu + np.where(ko[:, None], effect * W[None, :, 0], 0.0)
    mu = np.exp(logmu); mu = mu / mu.sum(1, keepdims=True) * depth[:, None]
    Y = sp.csr_matrix(rng.poisson(mu))
    return Y, sp.csr_matrix(W), gc, ko


def _auroc(neg_score_pos, neg_score_neg):
    a = np.concatenate([neg_score_pos, neg_score_neg]); r = a.argsort().argsort() + 1
    n1 = len(neg_score_pos)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg_score_neg))


def test_bias_strata_balanced():
    rng = np.random.default_rng(1)
    s = T.bias_strata(np.column_stack([rng.normal(size=1000), rng.integers(0, 3, 1000)]), [4, 5])
    assert s.max() + 1 == 20
    assert np.bincount(T.bias_strata(rng.normal(size=1000), [10])).min() == 100


def test_null_calibration_poisson():
    Y, W, gc, _ = _simulate(effect=0.0)
    strata = T.bias_strata(np.column_stack([gc, np.log10(np.asarray(Y.sum(0)).ravel() + 1)]), [5, 5])
    r = T.tf_activity_core(Y, W, strata, null="theoretical", shrinkage="none", verbose=False)
    z = r["z0"]
    assert abs(z.mean()) < 0.05
    assert 0.9 < z.std() < 1.1          # multinomial score z is N(0,1) under Poisson


def test_recovers_planted_loss():
    Y, W, gc, ko = _simulate(effect=-0.5)
    strata = T.bias_strata(np.column_stack([gc, np.log10(np.asarray(Y.sum(0)).ravel() + 1)]), [5, 5])
    r = T.tf_activity_core(Y, W, strata, verbose=False)
    for key in ("z", "post_z", "post_mean"):
        s = r[key]
        assert _auroc(-s[ko, 0], -s[~ko, 0]) > 0.9
        assert abs(_auroc(-s[ko, 1], -s[~ko, 1]) - 0.5) < 0.1
    # one-step log fold change is close to the planted effect
    g = r["gamma"][:, 0]
    assert abs((g[ko].mean() - g[~ko].mean()) - (-0.5)) < 0.15


def test_strata_remove_cell_gc_bias():
    """A cell-specific GC slope with no motif effect fools a no-strata model
    (motif 0 is GC-rich) but not the GC-stratified one."""
    Y, W, gc, _ = _simulate(effect=0.0, gc_bias=3.0)
    rng = np.random.default_rng(0)
    logacc = np.log10(np.asarray(Y.sum(0)).ravel() + 1)
    flat = T.tf_activity_core(Y, W, np.zeros(Y.shape[1], np.int64), null="theoretical",
                              shrinkage="none", verbose=False)["z0"][:, 0]
    strat = T.tf_activity_core(Y, W, T.bias_strata(np.column_stack([gc, logacc]), [10, 4]),
                               null="theoretical", shrinkage="none", verbose=False)["z0"][:, 0]
    assert flat.std() > 2.0
    assert strat.std() < 1.3


def test_ash_shrinks_towards_zero_and_orders():
    Y, W, gc, ko = _simulate(effect=-0.5)
    strata = T.bias_strata(np.column_stack([gc, np.log10(np.asarray(Y.sum(0)).ravel() + 1)]), [5, 5])
    r = T.tf_activity_core(Y, W, strata, verbose=False)
    g, pm = r["gamma"], r["post_mean"]
    assert np.all(np.abs(pm) <= np.abs(g) + 1e-6)
    np.testing.assert_allclose(r["ash_weights"].sum(1), 1.0, atol=1e-8)
    # pure-noise motif: the fitted prior puts most mass at/near zero
    assert np.abs(pm[:, 3]).mean() < 0.5 * np.abs(g[:, 3]).mean()


def test_occupancy_weights():
    sites = dict(peak=np.array([0, 0, 1, 2]), motif=np.array([0, 0, 0, 1]),
                 score=np.array([10.0, 9.0, 7.0, 12.0]), thr_match=np.array([9.0, 11.0]), n_peaks=3)
    Wb = T.occupancy_weights(sites, binary=True).toarray()
    np.testing.assert_array_equal(Wb, [[1, 0], [0, 0], [0, 1]])
    Wo = T.occupancy_weights(sites, offset=0.0).toarray()
    p1, p2 = 1 / (1 + np.exp(-1.0)), 0.5
    assert Wo[0, 0] == pytest.approx(1 - (1 - p1) * (1 - p2), rel=1e-6)
    assert Wo[1, 0] == 0.0                      # unmatched peak dropped (matched_only)
    Wa = T.occupancy_weights(sites, offset=0.0, matched_only=False).toarray()
    assert Wa[1, 0] == pytest.approx(1 / (1 + np.exp(2.0)), rel=1e-6)


def test_anndata_wrapper():
    Y, W, gc, ko = _simulate(n_cells=200, n_peaks=1500, effect=-0.6)
    a = AnnData(Y.astype(np.float32))
    a.var["gc_bias"] = gc
    a.varm["motif"] = W
    T.tf_activity(a, n_bins=(5, 5), verbose=False)
    assert a.obsm["tf_activity"].shape == (200, 6)
    assert {"tf_activity_z", "tf_activity_lfc", "tf_activity_post_mean"} <= set(a.obsm)
    assert a.uns["tf_activity_params"]["n_strata"] == 25
    with pytest.raises(ValueError):
        T.tf_activity(a, n_bins=(5,), verbose=False)


def test_peak_sensitivity_does_not_mutate_inputs_and_helps():
    """Only a quarter of motif-0 peaks respond; the EB peak-sensitivity step
    should find them, improve KO detection, and leave the caller's W intact."""
    rng = np.random.default_rng(3)
    n, p = 1200, 5000
    pi = rng.gamma(1, 1, p); pi /= pi.sum()
    gc = rng.uniform(.3, .7, p)
    W = sp.csr_matrix((rng.uniform(size=(p, 4)) < 0.2).astype(np.float64))
    resp = (W[:, 0].toarray().ravel() > 0) & (rng.uniform(size=p) < 0.25)
    act = rng.normal(0, 0.6, n)
    ko = np.zeros(n, bool); ko[: n // 3] = True; act[ko] -= 0.8
    mu = np.exp(np.log(pi)[None, :] + act[:, None] * resp[None, :])
    mu = mu / mu.sum(1, keepdims=True) * rng.integers(1500, 5000, n)[:, None]
    Y = sp.csr_matrix(rng.poisson(mu))
    strata = T.bias_strata(np.column_stack([gc, np.log10(np.asarray(Y.sum(0)).ravel() + 1)]), [5, 5])
    W0 = W.copy()
    base = T.tf_activity_core(Y, W, strata, verbose=False)
    ref = T.tf_activity_core(Y, W, strata, peak_sensitivity="eb", dispersion="quasi", verbose=False)
    assert (W != W0).nnz == 0 and np.array_equal(W.indptr, W0.indptr)
    a0 = _auroc(-base["gamma"][ko, 0], -base["gamma"][~ko, 0])
    a1 = _auroc(-ref["gamma"][ko, 0], -ref["gamma"][~ko, 0])
    assert a1 > a0 + 0.005
    rows = np.repeat(np.arange(p), np.diff(W.indptr)); m0 = W.indices == 0
    b = ref["peak_sensitivity"]
    assert b[m0 & resp[rows]].mean() > 2 * b[m0 & ~resp[rows]].mean()
    # repeated calls are identical
    ref2 = T.tf_activity_core(Y, W, strata, peak_sensitivity="eb", dispersion="quasi", verbose=False)
    np.testing.assert_array_equal(ref["z"], ref2["z"])


def test_quasi_dispersion_paired_counts():
    Y, W, gc, _ = _simulate(effect=0.0)
    strata = T.bias_strata(np.column_stack([gc, np.log10(np.asarray(Y.sum(0)).ravel() + 1)]), [5, 5])
    for k, target in ((1, 1.0), (2, 2.0)):
        r = T.tf_activity_core(Y * k, W, strata, null="theoretical", dispersion="quasi",
                               shrinkage="none", verbose=False)
        assert 0.9 < r["z"].std() < 1.1
