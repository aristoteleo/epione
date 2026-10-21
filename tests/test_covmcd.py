"""Unit tests for epione.bulk.hic._covmcd (robust::covRob's mcd branch).

Reference values come from R 4.1.3, robust 0.7.1, robustbase 0.95-1:
``set.seed(seed); covRob(x)`` on the matrices built below (written to R as raw
float64). Full parity: tests/parity/schicluster_compartment (gate D8).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from epione.bulk.hic import _dchic as dc  # noqa: E402
from epione.bulk.hic._covmcd import cov_mcd  # noqa: E402
from epione.bulk.hic._r_rng import RRandom  # noqa: E402


def _x(n, p):
    g = np.random.default_rng(11)
    x = g.standard_normal((n, p)) @ g.standard_normal((p, p))
    x[: n // 10] += 3
    return x


CASES = {
    # n >= 2 * nmini: FAST-MCD on sub-datasets
    "part": dict(n=1200, p=5, seed=7,
                 center=[0.006435167726211999, 0.05175494852084206, -0.02350458880015155],
                 diag=[12.297458370524524, 4.485127879409783, 1.2995576704794596],
                 cov01=-3.3036577899281094, pos=490, words=[1357248211, 1055593191, 1988374607]),
    # n < 2 * nmini: the whole data at once (p = 12, so covRob does not use donostah)
    "whole": dict(n=500, p=12, seed=123,
                  center=[0.07720013330985426, -0.21318658388506057, 0.17264449499352752],
                  diag=[19.81307297062092, 15.79137843528552, 7.470329243960952],
                  cov01=-10.546651339232852, pos=344, words=[-1112854297, -918576034, -463054814]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_cov_mcd_matches_r(name):
    c = CASES[name]
    rng = RRandom(c["seed"])
    center, cov = cov_mcd(_x(c["n"], c["p"]), rng=rng)
    np.testing.assert_array_equal(center[:3], c["center"])
    np.testing.assert_allclose(np.diag(cov)[:3], c["diag"], rtol=1e-12)
    np.testing.assert_allclose(cov[0, 1], c["cov01"], rtol=1e-12)
    # symmetric up to rounding: Fortran scales entry (i, j) as cov[i, j] * mad[i] * mad[j]
    np.testing.assert_allclose(cov, cov.T, rtol=1e-14)
    seed = rng.random_seed()
    assert seed[1] == c["pos"]
    assert list(seed[2:5]) == c["words"]


def test_cov_rob_threads_the_stream():
    x = _x(1200, 5)
    a = RRandom(7)
    r = dc.cov_rob(x, rng=a)
    assert r["estim"] == "mcd"
    np.testing.assert_array_equal(r["center"][:3], CASES["part"]["center"])
    assert list(a.random_seed()[2:5]) == CASES["part"]["words"]   # R's .Random.seed after the call
    # a second call continues R's stream (no reseeding), like two covRob calls in a row
    r2 = dc.cov_rob(x, rng=a)
    b = RRandom(7)
    cov_mcd(x, rng=b)
    c2, v2 = cov_mcd(x, rng=b)
    np.testing.assert_array_equal(r2["cov"], v2)
    np.testing.assert_array_equal(a.random_seed(), b.random_seed())


def test_unported_paths_raise():
    x = np.ones((50, 3))
    x[:, 1] = np.arange(50)
    x[:, 2] = np.arange(50) ** 2
    with pytest.raises(NotImplementedError):          # zero MAD in column 0 -> exact fit
        cov_mcd(x, rng=RRandom(1))
