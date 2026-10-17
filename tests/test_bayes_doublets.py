"""Synthetic checks of epi.pp.bayes_doublets (generative doublet model)."""
import gzip

import numpy as np
import pytest
import scipy.sparse as sp

from epione.pp._bayes_doublets import (
    _fit_bayes,
    count_overlap_loci,
    loading_doublet_rate,
)


def _planted(n=1500, K=4, F=20000, rho=0.15, depth=1200, seed=0, profile_seed=None):
    """Singlets ~ Mult(N, p_k); doublets = sum of two cells of different
    types (binarised), library sizes log-normal."""
    prng = np.random.default_rng(seed if profile_seed is None else profile_seed)
    base = prng.gamma(0.3, 1.0, size=F)
    prof = np.array([base * prng.gamma(0.5, 2.0, size=F) for _ in range(K)])
    rng = np.random.default_rng(seed) if profile_seed is None else np.random.default_rng(seed + 1000)
    prof /= prof.sum(1, keepdims=True)
    y = (rng.random(n) < rho).astype(int)
    rows, pairs = [], []
    for i in range(n):
        na = int(np.exp(rng.normal(np.log(depth), 0.4)))
        if y[i]:
            k, l = rng.choice(K, 2, replace=False)
            nb = int(np.exp(rng.normal(np.log(depth), 0.4)))
            c = rng.multinomial(na, prof[k]) + rng.multinomial(nb, prof[l])
        else:
            k = l = int(rng.integers(K))
            c = rng.multinomial(na, prof[k])
        pairs.append((k, l))
        rows.append(sp.csr_matrix(np.minimum(c, 1)))
    X = sp.vstack(rows).tocsr().astype(np.float64)
    return X, np.asarray(X.sum(1)).ravel(), y, np.array(pairs)


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


@pytest.mark.parametrize("use_library", [False, True])
def test_planted_doublets_are_recovered(use_library):
    X, N, y, _ = _planted()
    r = _fit_bayes(X, N, rho0=0.1, n_comps=10, n_iter=0, use_overlap=False,
                   use_library=use_library, homotypic=False, verbose=False, n_jobs=2, loo=False, background=0.0)
    assert np.isfinite(r["llr"]).all()
    assert _auc(y, r["llr"]) > 0.97
    # posterior is a probability and is higher for doublets
    assert ((r["prob"] >= 0) & (r["prob"] <= 1)).all()
    assert r["prob"][y == 1].mean() > r["prob"][y == 0].mean() + 0.5


def test_doublet_parents_identified():
    X, N, y, pairs = _planted(seed=1)
    r = _fit_bayes(X, N, rho0=0.1, n_comps=10, n_iter=0, use_overlap=False, use_library=False,
                   homotypic=False, verbose=False, n_jobs=2, loo=False, background=0.0)
    # map clusters to planted types by majority among singlets
    lab = r["gamma"].argmax(1)
    K = r["gamma"].shape[1]
    typ = np.full(K, -1)
    for k in range(K):
        w = (lab == k) & (y == 0)
        if w.sum():
            typ[k] = np.bincount(pairs[w, 0]).argmax()
    bp = r["best_pair"]
    d = np.flatnonzero((y == 1) & (r["prob"] > 0.5) & (bp >= 0))
    got = np.sort(np.column_stack([typ[bp[d] // 100000], typ[bp[d] % 100000]]), 1)
    true = np.sort(pairs[d], 1)
    assert (got == true).all(1).mean() > 0.9


def test_loading_doublet_rate():
    # small loading: rho ~ lambda / 2 ~ 0.8 % per 1,000 cells
    assert abs(loading_doublet_rate(1000) - 0.008) < 0.001
    assert abs(loading_doublet_rate(10000) - 0.0835) < 0.005
    assert loading_doublet_rate(20000) > loading_doublet_rate(10000)


def test_count_overlap_loci(tmp_path):
    # cell A: 3 fragments overlapping [150, 180) -> 1 locus; a 4th overlapping fragment
    # extends the same interval (still 1). Then coverage drops and 3 overlap again -> 2nd locus.
    # cell B: at most 2 overlapping -> 0. chrM ignored.
    lines = [
        ("chr1", 100, 200, "A"), ("chr1", 120, 180, "A"), ("chr1", 150, 300, "A"),
        ("chr1", 170, 260, "A"),
        ("chr1", 1000, 1100, "A"), ("chr1", 1050, 1150, "A"), ("chr1", 1090, 1200, "A"),
        ("chr1", 100, 200, "B"), ("chr1", 150, 250, "B"), ("chr1", 240, 400, "B"),
        ("chrM", 1, 50, "B"), ("chrM", 2, 50, "B"), ("chrM", 3, 50, "B"),
        ("chr2", 10, 20, "A"), ("chr2", 20, 30, "A"), ("chr2", 30, 40, "A"),   # touching only
    ]
    lines.sort(key=lambda t: (t[0], t[1]))
    f = tmp_path / "frag.tsv.gz"
    with gzip.open(f, "wt") as fh:
        for c, s, e, b in lines:
            fh.write(f"{c}\t{s}\t{e}\t{b}\t1\n")
    r = count_overlap_loci(str(f), ["A", "B"], n_jobs=1)
    m = np.bincount(r["row"], minlength=2)
    assert m.tolist() == [2, 0]
    assert r["n_fragments"].tolist() == [10, 3]


def test_doublet_population_cluster_is_pruned():
    """A frequent heterotypic pair that forms its own cluster must not be
    taken as a singlet type: half of all doublets are (0, 1) pairs."""
    X, N, y, pairs = _planted(n=2000, rho=0.0, seed=3, profile_seed=7)
    # add 300 (0, 1) doublets of the same types
    Xd, Nd, yd, pd_ = _planted(n=2000, rho=1.0, seed=4, profile_seed=7)
    sel = np.flatnonzero((np.sort(pd_, 1) == [0, 1]).all(1))[:300]
    X = sp.vstack([X, Xd[sel]]).tocsr()
    N = np.concatenate([N, Nd[sel]])
    y = np.concatenate([y, np.ones(sel.size, int)])
    r = _fit_bayes(X, N, rho0=0.1, n_comps=10, n_iter=0, use_overlap=False, use_library=False,
                   homotypic=False, verbose=False, n_jobs=2, loo=False, background=0.0,
                   prune_doublet_types=True)
    assert _auc(y, r["llr"]) > 0.97


def test_public_api_writes_obs():
    import anndata as ad
    from epione.pp import bayes_doublets
    X, N, y, _ = _planted(n=800, seed=5)
    a = ad.AnnData(X.astype(np.float32))
    bayes_doublets(a, n_comps=10, n_jobs=2)
    for c in ["probability", "score", "class", "pair", "type"]:
        assert f"bayes_doublet_{c}" in a.obs
    assert 0 < a.uns["bayes_doublet"]["rho"] < 0.75
    assert _auc(y, a.obs["bayes_doublet_score"].values) > 0.95
