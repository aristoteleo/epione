"""Thermodynamic occupancy score (``epione.tl._occupancy``).

Synthetic checks of the maths: the neighbourhood log-sum-exp against a
brute-force sum, the per-site weight against its formula, insertion
counting against a direct count on a small fragments file, and recovery
of planted bound sites end to end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epione.tl import _occupancy as O

RNG = np.random.default_rng(11)


def _brute_lse(keys, centres, values, radius):
    out = np.empty(len(values))
    for i in range(len(values)):
        m = (keys == keys[i]) & (np.abs(centres - centres[i]) <= radius)
        out[i] = np.log(np.sum(np.exp(values[m])))
    return out


def test_neighbourhood_logsumexp_matches_brute_force():
    n = 400
    keys = RNG.choice(np.array(["A|chr1", "A|chr2", "B|chr1"]), n)
    centres = RNG.integers(0, 3000, n)
    values = RNG.normal(0, 3, n)
    for r in (0, 10, 150, 5000):
        got = O.neighbourhood_logsumexp(keys, centres, values, r)
        exp = values if r == 0 else _brute_lse(keys, centres, values, r)
        np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-12)


def test_neighbourhood_logsumexp_is_stable_with_extreme_values():
    # a cumulative-sum difference would return log(0) for the small sites
    keys = np.array(["A"] * 4)
    centres = np.array([0, 1000, 1010, 5000])
    values = np.array([800.0, -50.0, -60.0, 0.0])
    got = O.neighbourhood_logsumexp(keys, centres, values, 100)
    exp = np.array([800.0, np.logaddexp(-50, -60), np.logaddexp(-50, -60), 0.0])
    np.testing.assert_allclose(got, exp, rtol=1e-12)


def test_occupancy_from_counts_formula():
    sites = pd.DataFrame({"chrom": ["c1"] * 3, "start": [100, 150, 900],
                          "end": [110, 160, 910], "motif": ["T"] * 3,
                          "score": [10.0, 5.0, 8.0]})
    n_all = np.array([[20.0, 3.0, 0.0]])
    n_short = np.array([[5.0, 1.0, 0.0]])
    score, logw = O.occupancy_from_counts(sites, n_all, n_short, beta=0.5,
                                          short_weight=2.0, radius=100)
    lw = np.log1p(n_all + 2 * n_short) + 0.5 * np.log(2) * sites.score.values
    np.testing.assert_allclose(logw, lw)
    np.testing.assert_allclose(score[0, 0], np.logaddexp(lw[0, 0], lw[0, 1]))
    np.testing.assert_allclose(score[0, 1], score[0, 0])
    np.testing.assert_allclose(score[0, 2], lw[0, 2])  # alone in its neighbourhood


def test_thermodynamic_prior_and_accessibility_both_order_sites():
    sites = pd.DataFrame({"chrom": ["c1"] * 4, "start": [0, 1000, 2000, 3000],
                          "end": [10, 1010, 2010, 3010], "motif": ["T"] * 4,
                          "score": [8.0, 12.0, 8.0, 8.0]})
    n = np.array([[10.0, 10.0, 40.0, 10.0]])
    score, _ = O.occupancy_from_counts(sites, n, np.zeros_like(n), radius=0)
    assert score[0, 1] > score[0, 0]          # stronger motif, same accessibility
    assert score[0, 2] > score[0, 3]          # more accessible, same motif
    np.testing.assert_allclose(score[0, 0], score[0, 3])


# ---------------------------------------------------------------------------
# Insertion counting and end-to-end recovery on a synthetic fragments file
# ---------------------------------------------------------------------------

pysam = pytest.importorskip("pysam")


def _write_fragments(path, frags):
    frags = sorted(frags, key=lambda f: (f[0], f[1], f[2]))
    raw = str(path)[:-3]
    with open(raw, "w") as fh:
        for c, s, e, b in frags:
            fh.write(f"{c}\t{s}\t{e}\t{b}\t1\n")
    pysam.tabix_compress(raw, str(path), force=True)
    pysam.tabix_index(str(path), preset="bed", force=True)
    return str(path)


def test_site_insertion_counts_matches_direct_count(tmp_path):
    frags = []
    for _ in range(3000):
        s = int(RNG.integers(0, 5000)); L = int(RNG.integers(30, 400))
        frags.append(("chr1", s, s + L, RNG.choice(["a", "b", "c"])))
    f = _write_fragments(tmp_path / "f.tsv.gz", frags)
    centre = np.array([300, 310, 2500, 4900, 2500])
    chrom = np.array(["chr1"] * 5)
    groups = [0, 1, -1]                       # cell c unused
    n_all, n_short = O.site_insertion_counts(f, ["a", "b", "c"], groups, 2, chrom, centre,
                                             window=50, short_max=120, chunk_size=1)
    exp_all = np.zeros((2, 5)); exp_short = np.zeros((2, 5))
    gmap = {"a": 0, "b": 1}
    for c, s, e, b in frags:
        if b not in gmap:
            continue
        for x in (s, e - 1):
            for k, cen in enumerate(centre):
                if abs(x - cen) <= 50:
                    exp_all[gmap[b], k] += 1
                    if e - s <= 120:
                        exp_short[gmap[b], k] += 1
    np.testing.assert_array_equal(n_all, exp_all)
    np.testing.assert_array_equal(n_short, exp_short)


def test_tf_occupancy_recovers_planted_bound_sites(tmp_path):
    """Bound sites: stronger motif and an open, short-fragment-rich window;
    the score must rank them above unbound sites of similar accessibility
    better than accessibility or the PWM alone."""
    ad = pytest.importorskip("anndata")
    n_sites, spacing = 300, 2000
    centres = np.arange(n_sites) * spacing + 1000
    bound = RNG.random(n_sites) < 0.3
    pwm = RNG.normal(10, 2, n_sites) + 2.0 * bound          # log2 odds
    access = RNG.lognormal(2.5, 0.8, n_sites) * np.where(bound, 1.6, 1.0)
    frags = []
    for c, a, bnd in zip(centres, access, bound):
        for _ in range(RNG.poisson(a)):
            L = int(RNG.integers(40, 110)) if (bnd and RNG.random() < 0.5) else int(RNG.integers(40, 500))
            s = int(c + RNG.normal(0, 40) - L / 2)
            frags.append(("chr1", max(s, 0), max(s, 0) + L, f"cell{RNG.integers(0, 20)}"))
    f = _write_fragments(tmp_path / "frags.tsv.gz", frags)
    adata = ad.AnnData(obs=pd.DataFrame(index=[f"cell{i}" for i in range(20)]))
    adata.uns["files"] = {"fragments": f}
    sites = pd.DataFrame({"chrom": "chr1", "start": centres - 5, "end": centres + 5,
                          "tf": "T", "pwm": pwm})
    res = O.tf_occupancy(adata, sites, motif_key="tf", score_key="pwm", verbose=False)
    from sklearn.metrics import roc_auc_score
    au = roc_auc_score(bound, res.score[0])
    au_acc = roc_auc_score(bound, res.n_all[0])
    au_pwm = roc_auc_score(bound, pwm)
    assert au > max(au_acc, au_pwm)
    assert au > 0.8
    long = res.to_frame()
    assert len(long) == n_sites and {"occupancy_score", "n_short"} <= set(long.columns)


# ---------------------------------------------------------------------------
# Default entry point: epi.tl.tf_binding
# ---------------------------------------------------------------------------

def _small_adata(tmp_path):
    ad = pytest.importorskip("anndata")
    frags = []
    for _ in range(2000):
        s = int(RNG.integers(0, 6000)); L = int(RNG.integers(40, 400))
        frags.append(("chr1", s, s + L, f"c{RNG.integers(0, 12)}"))
    f = _write_fragments(tmp_path / "small.tsv.gz", frags)
    adata = ad.AnnData(obs=pd.DataFrame({"grp": ["a"] * 6 + ["b"] * 6}, index=[f"c{i}" for i in range(12)]))
    adata.uns["files"] = {"fragments": f}
    sites = pd.DataFrame({"chrom": "chr1", "start": [500, 520, 3000, 5000], "end": [510, 530, 3010, 5010],
                          "strand": ["+", "-", "+", "+"], "motif": ["A", "A", "B", "A"],
                          "score": [9.0, 11.0, 10.0, 12.0]})
    return adata, sites


def test_tf_binding_default_is_occupancy(tmp_path):
    from epione.tl import tf_binding, tf_occupancy, TF_BINDING_METHODS
    import inspect
    assert inspect.signature(tf_binding).parameters["method"].default == "occupancy"
    assert TF_BINDING_METHODS[0] == "occupancy"
    adata, sites = _small_adata(tmp_path)
    res = tf_binding(adata, sites, verbose=False)
    ref = tf_occupancy(adata, sites, verbose=False)
    assert isinstance(res, O.OccupancyResult)
    np.testing.assert_array_equal(res.score, ref.score)
    grp = tf_binding(adata, sites, groupby="grp", min_cells_per_group=1, verbose=False)
    assert grp.groups == ["a", "b"] and grp.score.shape == (2, 4)
    np.testing.assert_array_equal(grp.n_all.sum(0), ref.n_all[0])   # groups partition the cells


def test_tf_binding_print_dispatch(tmp_path, monkeypatch):
    """method='print' forwards to the unchanged PRINT function with the peaks and strand-aware sites."""
    from epione.tl import _print, tf_binding
    adata, sites = _small_adata(tmp_path)
    seen = {}

    def fake(adata_, regions, *, groupby, sites, **kw):
        seen.update(regions=regions, groupby=groupby, sites=sites, kw=kw)
        return "print-result"
    monkeypatch.setattr(_print, "tf_binding_score", fake)
    peaks = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [6000]})
    out = tf_binding(adata, sites, method="print", regions=peaks, groupby="grp", tile=5)
    assert out == "print-result" and seen["groupby"] == "grp" and seen["kw"] == {"tile": 5}
    assert list(seen["sites"].columns) == ["chrom", "start", "end", "strand"]
    with pytest.raises(ValueError, match="regions"):
        tf_binding(adata, sites, method="print", groupby="grp")
    with pytest.raises(ValueError, match="unknown method"):
        tf_binding(adata, sites, method="tobias")
