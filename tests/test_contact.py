"""Synthetic tests for the polymer-contact tools (epione.tl._contact).

* powerlaw_contact: plateau below s0, exact s^-gamma decay above.
* fit_contact_gamma: recovers a planted decay exponent. Accessibility is a
  sum of shared "domain" factors whose lengths are Pareto(beta) distributed;
  two loci at distance s share a domain with probability ∝ s^(1-beta), so the
  correlation decays as s^-(beta-1) (gamma = beta - 1).
* contact_links: ABC sums to one per gene over its window and reproduces the
  formula; the graphical-lasso partial correlation removes cell-type
  confounding that fools Pearson (a confounded peak tracks the cell type, the
  true enhancer drives within-type variation of the gene).
* contact_gene_score: weights follow C(s); ``compete`` shares each feature
  among genes; fragment and matrix paths agree.
"""
import gzip

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from epione.tl import contact_gene_score, contact_links, fit_contact_gamma, powerlaw_contact
from epione.tl._contact import _glasso_partial_corr


def test_powerlaw_contact():
    s = np.array([0, 1000, 5000, 10000, 50000, -50000])
    c = powerlaw_contact(s, gamma=1.0, s0=5000)
    np.testing.assert_allclose(c, [1, 1, 1, 0.5, 0.1, 0.1])
    c2 = powerlaw_contact(s, gamma=0.5, s0=5000)
    np.testing.assert_allclose(c2[3], 2 ** -0.5)
    with pytest.raises(ValueError):
        powerlaw_contact(s, gamma=-1)


def _domain_field(n_obs, positions, beta, rng, lmin=2000, coverage=20):
    """Sum of domain factors with Pareto(beta) lengths; ~`coverage` domains per locus."""
    lo, hi = positions.min() - 2e6, positions.max() + 2e6
    mean_len = lmin * beta / (beta - 1)
    n_dom = int(coverage * (hi - lo) / mean_len)
    centres = rng.uniform(lo, hi, n_dom)
    lengths = np.minimum(lmin * (1 - rng.random(n_dom)) ** (-1.0 / beta), 4e6)
    field = np.zeros((n_obs, len(positions)))
    a_all = np.searchsorted(positions, centres - lengths / 2)
    b_all = np.searchsorted(positions, centres + lengths / 2)
    hit = b_all > a_all
    for a, b in zip(a_all[hit], b_all[hit]):
        field[:, a:b] += rng.standard_normal((n_obs, 1))
    return field


def test_fit_contact_gamma_recovers_planted_exponent():
    """The estimate equals the log-log slope of the planted field's true
    mean correlation curve, and orders two planted exponents correctly."""
    rng = np.random.default_rng(1)
    n_obs = 300
    pos = np.arange(0, 20_000_000, 1000)
    est, truth = {}, {}
    for beta in (1.8, 2.4):
        f1 = _domain_field(n_obs, pos, beta, rng)
        f2 = _domain_field(n_obs, pos, beta, rng)
        field = np.hstack([f1, f2])
        field = (field - field.mean(0)) / (field.std(0) + 1e-9)
        # true mean correlation at grid distances 10 kb .. 500 kb
        ds = np.array([10, 20, 50, 100, 200, 500]) * 1000
        r = [np.mean((f1[:, :-k] * f1[:, k:]).sum(0)) for k in ds // 1000]
        f1s = (f1 - f1.mean(0)) / f1.std(0)
        r = [np.mean((f1s[:, :-k] * f1s[:, k:]).sum(0) / (n_obs - 1)) for k in ds // 1000]
        truth[beta] = -np.polyfit(np.log(ds), np.log(r), 1)[0]
        X = np.exp(0.2 * field + 3.0)  # positive "accessibility", log-linear in the field
        names = [f"chr1:{p}-{p + 500}" for p in pos] + [f"chr2:{p}-{p + 500}" for p in pos]
        ad = AnnData(X=X.astype(np.float32), var=pd.DataFrame(index=names))
        M = sp.identity(n_obs, format="csr")  # observations already are metacells
        res = fit_contact_gamma(ad, metacells=M, d_min=10_000, d_max=500_000, n_bins=8,
                                n_pairs=20_000, verbose=False)
        est[beta] = res["gamma_raw"]
        assert abs(res["background_r"]) < 0.02
        assert abs(est[beta] - truth[beta]) < 0.15, (beta, est[beta], truth[beta])
    assert est[1.8] < est[2.4]


def _two_type_multiome(seed=0, n_per=600):
    """Cell types A/B. Gene G is higher in A (type effect) and, within each
    type, follows enhancer E (true regulator). Peak D only tracks the type
    (confounded, no within-type link). Peaks at fixed distances."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per
    ctype = np.repeat([1.0, 0.0], n_per)
    # within-type continuous state for E (shared by neighbouring cells via the embedding)
    state = rng.standard_normal(n)
    emb = np.column_stack([ctype * 4 + 0.05 * rng.standard_normal(n),
                           state + 0.05 * rng.standard_normal(n),
                           rng.standard_normal(n) * 0.05])
    lamE = np.exp(0.8 * state)                   # E: within-type variation
    lamD = np.exp(1.5 * ctype)                   # D: type only
    lamG = np.exp(1.5 * ctype + 0.6 * state)     # G: type + E
    bg = rng.poisson(1.0, size=(n, 8)).astype(float)
    peaks = np.column_stack([rng.poisson(2 * lamE), rng.poisson(2 * lamD), bg])
    names = ["chr1:1000000-1000500",   # E, 20 kb from TSS
             "chr1:1040000-1040500",   # D, 20 kb from TSS (other side)
             ] + [f"chr1:{1_100_000 + 30_000 * i}-{1_100_500 + 30_000 * i}" for i in range(8)]
    rna = np.column_stack([rng.poisson(5 * lamG), rng.poisson(3.0, size=(n, 3))]).astype(np.float32)
    obs = pd.DataFrame({"ctype": np.where(ctype == 1, "A", "B")},
                       index=[f"c{i}" for i in range(n)])
    atac = AnnData(X=sp.csr_matrix(peaks.astype(np.float32)), obs=obs,
                   var=pd.DataFrame(index=names))
    atac.obsm["X_emb"] = emb
    rn = AnnData(X=sp.csr_matrix(rna), obs=obs.copy(),
                 var=pd.DataFrame(index=["G", "N1", "N2", "N3"]))
    ann = pd.DataFrame({"gene_name": ["G", "N1", "N2", "N3"], "chrom": "chr1",
                        "start": [1_020_250, 3_000_000, 5_000_000, 7_000_000],
                        "end": [1_030_000, 3_010_000, 5_010_000, 7_010_000], "strand": "+"})
    return atac, rn, ann


def test_contact_links_abc_formula():
    atac, rn, ann = _two_type_multiome()
    df = contact_links(atac, gene_annotation=ann, method="abc", gamma=1.0, s0=5000,
                       max_distance=5_000_000, abc_window=5_000_000, inplace=False,
                       verbose=False)
    g = df[df.gene == "G"]
    np.testing.assert_allclose(g["abc"].sum(), 1.0, rtol=1e-5)
    # formula check against a direct computation
    a = np.asarray(atac.X.sum(0)).ravel()
    a = a * 1e6 / a.sum()
    c = ((atac.var_names.str.split(":").str[1].str.split("-").str[0].astype(int) + 250) - 1_020_250)
    w = a * powerlaw_contact(c.to_numpy(), 1.0, 5000)
    exp = dict(zip(atac.var_names, w / w.sum()))
    np.testing.assert_allclose(g["abc"].to_numpy(), [exp[p] for p in g["peak"]], rtol=1e-5)
    assert np.allclose(g["score"], g["abc"])
    # activity restricted to a target group changes ABC, contact unchanged
    dfA = contact_links(atac, gene_annotation=ann, method="abc", gamma=1.0,
                        groupby="ctype", target="A", inplace=False, verbose=False)
    gA = dfA[dfA.gene == "G"].set_index("peak")
    gB = g.set_index("peak")
    assert gA.loc["chr1:1040000-1040500", "abc"] > gB.loc["chr1:1040000-1040500", "abc"]
    np.testing.assert_allclose(gA["contact"], gB.loc[gA.index, "contact"])


def test_contact_links_pcor_removes_confounding():
    atac, rn, ann = _two_type_multiome()
    kw = dict(rna=rn, gene_annotation=ann, use_rep="X_emb", n_metacells=200,
              k_neighbors=10, corr_window=250_000, seed=0, inplace=False, verbose=False)
    pear = contact_links(atac, method="pearson_contact", **kw).set_index(["peak", "gene"])
    pc = contact_links(atac, method="pcor", n_confounders=1, glasso_alpha=0.01, **kw)
    pc = pc.set_index(["peak", "gene"])
    E, D = ("chr1:1000000-1000500", "G"), ("chr1:1040000-1040500", "G")
    # Pearson: the confounded peak looks as good as (or better than) the enhancer
    assert pear.loc[D, "pearson"] > pear.loc[E, "pearson"]
    # partial correlation given the cell-type axis: enhancer >> confounded peak
    assert pc.loc[E, "pcor"] > 0.3
    assert abs(pc.loc[D, "pcor"]) < 0.2
    assert pc.loc[E, "pcor"] - pc.loc[D, "pcor"] > 0.4
    assert pc.loc[E, "score"] == pytest.approx(pc.loc[E, "pcor"])
    # combined ABC x pcor ranks E above D
    comb = contact_links(atac, method="abc_pcor", n_confounders=1, glasso_alpha=0.01,
                         eps=0.05, gamma=1.0, **kw).set_index(["peak", "gene"])
    assert comb.loc[E, "score"] > comb.loc[D, "score"]


def test_glasso_partial_corr_matches_inverse_without_penalty():
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.standard_normal((n, 3))
    y = z[:, 0] + z[:, 1] + 0.5 * rng.standard_normal(n)
    x2 = z[:, 1] + 0.3 * rng.standard_normal(n)  # y ⟂ x3 | x2? no: x3 indep
    Y = np.column_stack([y, z[:, 0], x2, z[:, 2]])
    Y = (Y - Y.mean(0)) / Y.std(0, ddof=1)
    rho = _glasso_partial_corr(Y, alpha=1e-4)
    P = np.linalg.inv(np.corrcoef(Y.T))
    d = np.sqrt(np.diag(P))
    exp = -P[0, 1:] / (d[0] * d[1:])
    np.testing.assert_allclose(rho, exp, atol=0.02)
    assert abs(rho[2]) < 0.05


def _gs_inputs():
    rng = np.random.default_rng(3)
    n = 40
    starts = np.arange(0, 400_000, 500)
    X = rng.poisson(0.3, size=(n, len(starts))).astype(np.float32)
    names = [f"chr1:{s}-{s + 500}" for s in starts]
    ad = AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame(index=[f"b{i}" for i in range(n)]),
                 var=pd.DataFrame(index=names))
    ann = pd.DataFrame({"gene_name": ["g1", "g2"], "chrom": "chr1", "start": [100_000, 250_000],
                        "end": [120_000, 260_000], "strand": ["+", "-"]})
    return ad, ann, X, starts


def test_contact_gene_score_weights():
    ad, ann, X, starts = _gs_inputs()
    contact_gene_score(ad, ann, gamma=1.0, s0=5000, window=100_000, ceiling=None,
                       scale_to=1.0, verbose=False)
    centres = starts + 250
    raw = []
    for tss in (100_000, 260_000):
        s = np.abs(centres - tss)
        w = np.where(s <= 100_000, powerlaw_contact(s, 1.0, 5000), 0.0)
        raw.append(X @ w)
    raw = np.column_stack(raw)
    exp = raw / raw.sum(1, keepdims=True)
    np.testing.assert_allclose(ad.obsm["contact_gene_score"], exp, rtol=1e-4)
    # compete: per-feature weights summed over genes are 1 where any gene contacts
    contact_gene_score(ad, ann, gamma=1.0, s0=5000, window=400_000, compete=True,
                       ceiling=None, scale_to=1.0, key_added="cmp", verbose=False)
    w1 = powerlaw_contact(np.abs(centres - 100_000), 1.0, 5000)
    w2 = powerlaw_contact(np.abs(centres - 260_000), 1.0, 5000)
    in1 = np.abs(centres - 100_000) <= 400_000
    in2 = np.abs(centres - 260_000) <= 400_000
    w1, w2 = w1 * in1, w2 * in2
    tot = w1 + w2
    raw = np.column_stack([X @ (w1 / tot), X @ (w2 / tot)])
    np.testing.assert_allclose(ad.obsm["cmp"], raw / raw.sum(1, keepdims=True), rtol=1e-4)


def test_contact_gene_score_fragments_match_matrix(tmp_path):
    rng = np.random.default_rng(5)
    cells = [f"AAAC{i:04d}-1" for i in range(30)]
    rows = []
    for ci, c in enumerate(cells):
        st = np.sort(rng.integers(0, 399_000, size=200))
        ln = rng.integers(60, 600, size=200)
        for s, l in zip(st, ln):
            rows.append(("chr1", int(s), int(s + l), c, 1))
    rows.sort(key=lambda r: (r[0], r[1]))
    f = tmp_path / "frags.tsv.gz"
    with gzip.open(f, "wt") as fh:
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    ann = pd.DataFrame({"gene_name": ["g1", "g2"], "chrom": "chr1", "start": [100_000, 250_000],
                        "end": [120_000, 260_000], "strand": ["+", "-"]})
    ad = AnnData(obs=pd.DataFrame(index=cells))
    contact_gene_score(ad, ann, fragment_file=str(f), gamma=1.0, window=150_000, n_jobs=1,
                       verbose=False)
    # tile matrix by hand: insertions at start and end-1
    tiles = np.arange(0, 400_000, 500)
    Xt = np.zeros((len(cells), len(tiles)), dtype=np.float32)
    ci = {c: i for i, c in enumerate(cells)}
    for _, s, e, c, _ in rows:
        a, b = s // 500, (e - 1) // 500
        Xt[ci[c], a] += 1
        if b != a:
            Xt[ci[c], b] += 1
    ad2 = AnnData(X=sp.csr_matrix(Xt), obs=pd.DataFrame(index=cells),
                  var=pd.DataFrame(index=[f"chr1:{t}-{t + 500}" for t in tiles]))
    contact_gene_score(ad2, ann, gamma=1.0, window=150_000, verbose=False)
    np.testing.assert_allclose(ad.obsm["contact_gene_score"], ad2.obsm["contact_gene_score"],
                               rtol=1e-4)


def test_numba_glasso_matches_sklearn():
    """The numba solver optimises the same objective as sklearn's graphical_lasso."""
    from sklearn.covariance import graphical_lasso
    from epione.tl._contact import _glasso_kernels
    rng = np.random.default_rng(7)
    for m, alpha in ((6, 0.05), (25, 0.1), (40, 0.01)):
        A = rng.standard_normal((300, m)) @ (np.eye(m) + 0.3 * rng.standard_normal((m, m)))
        R = np.corrcoef(A.T)
        _, P_ref = graphical_lasso(R, alpha=alpha, max_iter=500, tol=1e-8, enet_tol=1e-8)
        P, _ = _glasso_kernels()["glasso"](R.copy(), alpha, 500, 1e-8, 1000, 1e-8)
        np.testing.assert_allclose(P, P_ref, atol=1e-4, rtol=1e-3)


def test_contact_links_numba_equals_sklearn_solver():
    atac, rn, ann = _two_type_multiome()
    kw = dict(rna=rn, gene_annotation=ann, use_rep="X_emb", n_metacells=200, k_neighbors=10,
              n_confounders=1, glasso_alpha=0.02, method="pcor", inplace=False, verbose=False)
    a = contact_links(atac, solver="numba", n_jobs=2, **kw)
    b = contact_links(atac, solver="sklearn", **kw)
    np.testing.assert_allclose(a["pcor"], b["pcor"], atol=2e-3)


def test_contact_gene_score_store_path_matches(tmp_path):
    """epi.pp.import_fragments (in-memory store) path == streaming path == matrix path."""
    import epione as epi
    rng = np.random.default_rng(11)
    cells = [f"AAAC{i:04d}-1" for i in range(25)]
    rows = []
    for c in cells:
        st = np.sort(rng.integers(0, 399_000, size=300))
        for s, l in zip(st, rng.integers(60, 600, size=300)):
            rows.append(("chr1", int(s), int(s + l), c, 1))
    rows.sort(key=lambda r: (r[0], r[1]))
    import pysam
    plain = tmp_path / "frags.tsv"
    with open(plain, "w") as fh:
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    f = tmp_path / "frags.tsv.gz"
    pysam.tabix_compress(str(plain), str(f))
    ann = pd.DataFrame({"gene_name": ["g1", "g2", "g3"], "chrom": "chr1",
                        "start": [100_000, 250_000, 300_000], "end": [120_000, 260_000, 330_000],
                        "strand": ["+", "-", "+"]})
    ad = epi.pp.import_fragments(str(f), chrom_sizes={"chr1": 400_000}, whitelist=cells,
                                 min_num_fragments=0)
    kw = dict(gamma=1.0, s0=1000, window=100_000, distance="body", compete=True, verbose=False)
    contact_gene_score(ad, ann, key_added="store", **kw)
    assert ad.uns["store_params"]["source"] == "fragment store"
    ad2 = AnnData(obs=pd.DataFrame(index=list(ad.obs_names)))
    contact_gene_score(ad2, ann, fragment_file=str(f), key_added="stream", n_jobs=1, **kw)
    np.testing.assert_allclose(ad.obsm["store"], ad2.obsm["stream"], rtol=1e-5, atol=1e-6)
