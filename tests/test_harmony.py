"""Tests for epi.tl.harmony (port of R harmony 1.2.4 / ArchR addHarmony)."""
from __future__ import annotations

import math
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import epione as epi
from epione.tl._harmony import r_kmeans, run_harmony
from epione.tl._rrng import RRandom

HERE = Path(__file__).parent / "parity" / "harmony"
_PARITY_ROOT = os.environ.get("EPIONE_ARCHR_PARITY_DIR")
HEME_REF = Path(_PARITY_ROOT) / "harmony" if _PARITY_ROOT else None


def _small_inputs():
    # Deterministic matrix the small R harmony 1.2.4 references were computed on.
    i = np.arange(1, 301)
    x = np.array([[math.sin(k / 5) + (k % 3 == 0) * 0.8, math.cos(k / 11),
                   math.sin(k / 17) * math.cos(k / 3), (k % 7) / 7 - (k % 3 == 1) * 0.5] for k in i])
    batch = np.array(["b1", "b2", "b3"])[(i % 3)]
    return x, pd.DataFrame({"batch": batch})


def _read(name):
    return np.loadtxt(HERE / name)


# -- R RNG emulation ------------------------------------------------------

def test_rrandom_matches_r_set_seed_1():
    r = RRandom(1)
    # set.seed(1); runif(3)
    np.testing.assert_array_equal(
        r.unif_rand(3), [0.26550866314209998, 0.37212389963679016, 0.57285336335189641])
    # set.seed(42); sample.int(10, 10)
    np.testing.assert_array_equal(RRandom(42).sample_int(10, 10), [1, 5, 10, 8, 2, 4, 6, 9, 7, 3])


def test_rrandom_state_roundtrip():
    r = RRandom(123)
    r.unif_rand(1000)
    st = r.state()
    a = r.unif_rand(5)
    np.testing.assert_array_equal(RRandom.from_state(st).unif_rand(5), a)


# -- stats::kmeans and RunHarmony vs R (small reference, no fixture) -------

def test_r_kmeans_matches_r():
    x, _ = _small_inputs()
    c = r_kmeans(x, 6, RRandom(7), iter_max=25, nstart=10)
    np.testing.assert_allclose(c, _read("small_kmeans_centers.tsv"), rtol=0, atol=1e-10)


@pytest.mark.parametrize("fname,lam", [("small_harmony.tsv", 1.0),
                                       ("small_harmony_lambda_auto.tsv", None)])
def test_run_harmony_matches_r(fname, lam):
    x, meta = _small_inputs()
    h = run_harmony(x, meta, "batch", nclust=10, lambda_=lam, seed=3, verbose=False)
    np.testing.assert_allclose(h, _read(fname), rtol=0, atol=1e-8)


# -- behaviour -------------------------------------------------------------

def _two_batch_adata(n=400, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(4, 10)) * 3
    ct = rng.integers(0, 4, n)
    batch = rng.integers(0, 2, n)
    X = centers[ct] + rng.normal(scale=0.3, size=(n, 10)) + batch[:, None] * 2.0
    a = ad.AnnData(np.zeros((n, 1), dtype=np.float32))
    a.obsm["X_iterative_lsi"] = X
    a.obs["Sample"] = pd.Categorical(np.where(batch == 1, "s2", "s1"))
    a.obs["celltype"] = ct
    return a


def test_harmony_anndata_api_and_mixing():
    from sklearn.neighbors import NearestNeighbors

    a = _two_batch_adata()
    epi.tl.harmony(a, batch_key="Sample", use_rep="X_iterative_lsi", verbose=False)
    H = a.obsm["X_harmony"]
    assert H.shape == a.obsm["X_iterative_lsi"].shape
    assert a.uns["harmony"]["n_iter"] >= 1
    assert a.uns["harmony"]["params"]["nclust"] == round(400 / 30)

    def batch_mix(E):
        nn = NearestNeighbors(n_neighbors=16).fit(E).kneighbors(E, return_distance=False)[:, 1:]
        b = a.obs["Sample"].cat.codes.to_numpy()
        return (b[nn] != b[:, None]).mean()

    def celltype_purity(E):
        nn = NearestNeighbors(n_neighbors=16).fit(E).kneighbors(E, return_distance=False)[:, 1:]
        c = a.obs["celltype"].to_numpy()
        return (c[nn] == c[:, None]).mean()

    assert batch_mix(H) > batch_mix(a.obsm["X_iterative_lsi"]) + 0.2
    assert celltype_purity(H) > 0.95


def test_harmony_is_seed_deterministic():
    a, b = _two_batch_adata(), _two_batch_adata()
    epi.tl.harmony(a, use_rep="X_iterative_lsi", seed=5, verbose=False)
    epi.tl.harmony(b, use_rep="X_iterative_lsi", seed=5, verbose=False)
    np.testing.assert_array_equal(a.obsm["X_harmony"], b.obsm["X_harmony"])


def test_harmony_input_validation():
    a = _two_batch_adata()
    with pytest.raises(KeyError):
        epi.tl.harmony(a, batch_key="nope", use_rep="X_iterative_lsi", verbose=False)
    with pytest.raises(KeyError):
        epi.tl.harmony(a, use_rep="X_missing", verbose=False)
    with pytest.raises(ValueError):
        epi.tl.harmony(a, use_rep="X_iterative_lsi", lambda_=-1.0, verbose=False)


# -- full ArchR Hematopoiesis parity (skips without the R reference) -------

@pytest.mark.skipif(HEME_REF is None or not (HEME_REF / "archr_harmony.tsv.gz").exists(),
                    reason="set EPIONE_ARCHR_PARITY_DIR to the ArchR reference outputs")
def test_parity_vs_archr_addharmony():
    import sys
    sys.path.insert(0, str(HERE))
    import parity_harmony as P

    lsi, meta = P.load_inputs(HEME_REF)
    X = lsi.to_numpy()
    inj = P.check_injected(HEME_REF, X, meta)
    assert inj["R_init_max_abs"] < 1e-8
    assert all(s["R_max_abs"] < 1e-8 and s["Zcorr_max_abs"] < 1e-8 for s in inj["iters"])
    assert inj["kmeans_rounds_py"] == inj["kmeans_rounds_r"]
    assert inj["objective_max_rel"] < 1e-6
    free = P.check_free(HEME_REF, X, meta, 1, extended=False)
    assert free["final_max_abs"] < 1e-8
