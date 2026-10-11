"""Tests for the SCENIC+ eGRN / AUCell / RSS port (epione.scenic, block 4).

Unit tests compare each kernel with a direct transcription of the reference algorithm on
small inputs. The parity test against the SCENIC+ 1.0a2 reference outputs runs only when
``EPIONE_SCENIC_PARITY_DIR`` is set (see tests/parity/scenic_egrn/MANIFEST.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epione.scenic import _egrn as E
from epione.scenic._aucell import aucell, rank_cells
from epione.scenic._egrn_basc import basc_threshold
from epione.scenic._egrn_gsea import gene_set_seed, gsea_batch, permutations, significance
from epione.scenic._egrn_triplet import map_regions, triplet_rank
from epione.scenic._egrn_argsort import argsort_np126
from epione.scenic._rss import regulon_specificity_scores


# ----------------------------------------------------------------------------- argsort
# md5 of np.argsort(x) computed with numpy 1.26.4 on an AVX-512 CPU (the SCENIC+ reference
# environment); x = ((arange(n) * 7919 + 17) % 13).astype(dtype) has many ties.
_ANCHOR = {37: "0db07520fa8904144bb6aa36ec6927fb", 300: "c7bd6d974bfef37bfabda90a433059cd",
           5000: "7ad0ac77da5829d6dc1f22cd8e728b84"}


@pytest.mark.parametrize("dtype", ["int32", "float32", "float64", "int64", "uint32"])
@pytest.mark.parametrize("n", [37, 300, 5000])
def test_argsort_matches_numpy_126(dtype, n):
    x = ((np.arange(n, dtype=np.int64) * 7919 + 17) % 13).astype(dtype)
    got = argsort_np126(x)
    assert hashlib.md5(got.astype(np.int64).tobytes()).hexdigest() == _ANCHOR[n]


@pytest.mark.parametrize("dtype", ["int32", "float32", "float64", "int16"])
def test_argsort_is_a_sort(dtype):
    rng = np.random.default_rng(0)
    for n in (1, 2, 9, 64, 65, 257, 3000):
        x = rng.integers(0, 5, n).astype(dtype)
        o = argsort_np126(x)
        assert np.array_equal(np.sort(o), np.arange(n))
        assert np.all(np.diff(x[o].astype(float)) >= 0)
        u = rng.permutation(n).astype(dtype if dtype != "int16" else "int32")
        assert np.array_equal(argsort_np126(u), np.argsort(u))  # no ties: unique answer


# ----------------------------------------------------------------------------- GSEA
def _gseapy_es(gene_list, correl, gene_set, nperm, seed):
    # gseapy 0.10.8 enrichment_score (weighted_score_type=1, single=False, scale=False)
    N = len(gene_list)
    tag = np.isin(gene_list, gene_set).astype(int)
    cor = np.abs(correl) ** 1
    hit_ind = np.flatnonzero(tag).tolist()
    tag = np.tile(tag, (nperm + 1, 1))
    cor = np.tile(cor, (nperm + 1, 1))
    rs = np.random.RandomState(seed)
    for i in range(nperm):
        rs.shuffle(tag[i])
    Nhint = tag.sum(axis=1, keepdims=True)
    sum_correl_tag = np.sum(cor * tag, axis=1, keepdims=True)
    no_tag = 1 - tag
    norm_tag = 1.0 / sum_correl_tag
    norm_no_tag = 1.0 / (N - Nhint)
    RES = np.cumsum(tag * cor * norm_tag - no_tag * norm_no_tag, axis=1)
    mx, mn = RES.max(axis=1), RES.min(axis=1)
    es_vec = np.where(np.abs(mx) > np.abs(mn), mx, mn)
    return es_vec[-1], es_vec[:-1], hit_ind, RES[-1]


def test_gsea_matches_gseapy_transcription():
    rng = np.random.default_rng(1)
    n, nperm = 400, 200
    genes = np.array([f"g{i}" for i in range(n)])
    cor = np.sort(rng.gamma(1.0, 1.0, n))[::-1]
    rs = gene_set_seed(666)
    perms = permutations(n, nperm, rs)
    sets = [rng.choice(genes, size=k, replace=False) for k in (5, 30, 120)]
    tags = np.stack([np.isin(genes, s) for s in sets]).astype(np.int8)
    es, esnull, imx, imn = gsea_batch(np.abs(cor) ** 1, tags, perms)
    for k, s in enumerate(sets):
        e, en, hits, res = _gseapy_es(genes, cor, s, nperm, rs)
        assert es[k] == e
        assert np.array_equal(esnull[k], en)
        assert imx[k] == res.argmax() and imn[k] == res.argmin()
    nes, pv = significance(es, esnull)
    assert np.all(np.isfinite(nes)) and np.all((pv >= 0) & (pv <= 1))


def test_permutations_match_randomstate_shuffle():
    rs = gene_set_seed(666)
    p = permutations(257, 5, rs)
    st = np.random.RandomState(rs)
    for i in range(5):
        a = np.arange(257)
        st.shuffle(a)
        assert np.array_equal(p[i], a)


# ----------------------------------------------------------------------------- BASC
def _basc_reference(x, max_elements=100):
    # direct transcription of the BASC A threshold (O(N^4) cost function)
    def mean(v):
        c = 0.0
        for t in v:
            c += t
        return c / len(v)

    def ssum(v):
        c = 0.0
        for t in v:
            c += t
        return c

    if len(x) <= 2 or np.all(x == x[0]):
        return 0.0
    v = np.sort(x)[:max_elements]
    N = len(v)

    def cost(a, b):
        y = mean(v[a:b + 1])
        return ssum([(t - y) * (t - y) for t in v][a:b + 1])

    C = np.zeros((N - 1, N))
    C[0] = [cost(i, N - 1) for i in range(N)]
    ind = np.zeros((N - 2, N), dtype=int)
    for j in range(1, N - 1):
        for i in range(N - j):
            cmin, dmin = np.inf, -1
            for d in range(i, N - j):
                c = cost(i, d) + C[j - 1, d + 1]
                if c < cmin:
                    cmin, dmin = c, d
            C[j, i] = cmin
            ind[j - 1, i] = dmin
    P = np.zeros((N - 2, N), dtype=int)
    for j in range(N - 2):
        z = j
        P[j, 0] = ind[z, 0]
        if j > 0:
            z -= 1
            for i in range(1, j + 1):
                P[j, i] = ind[z, P[j, i - 1] + 1]
                z -= 1
    best = []
    for j in range(N - 2):
        qmax, imax = -1.0, -1
        for i in range(j + 1):
            p = P[j, i]
            if i == 0 and j > 0:
                h = mean(v[p + 1:P[j, i + 1] + 1]) - mean(v[:p + 1])
            elif i == j > 0:
                h = mean(v[p + 1:]) - mean(v[P[j, i - 1] + 1:p + 1])
            elif i == j == 0:
                h = mean(v[p + 1:]) - mean(v[:p + 1])
            else:
                h = mean(v[p + 1:P[j, i + 1] + 1]) - mean(v[P[j, i - 1] + 1:p + 1])
            z = (v[p] + v[p + 1]) / 2
            e = ssum([(t - z) * (t - z) for t in v])
            q = h / e
            if q > qmax:
                qmax, imax = q, p
        best.append(imax)
    vm = int(np.floor(np.median(best)))
    return (v[vm + 1] + v[vm]) / 2


def test_basc_matches_reference():
    rng = np.random.default_rng(2)
    for n in (3, 4, 7, 20, 31):
        x = np.concatenate([rng.gamma(0.5, 0.01, n - n // 4), rng.gamma(4, 0.05, n // 4)])
        assert basc_threshold(x) == _basc_reference(x)
    assert basc_threshold(np.array([1.0, 1.0, 1.0])) == 0.0
    assert basc_threshold(np.array([1.0, 2.0])) == 0.0


def test_group_quantile_matches_numpy():
    rng = np.random.default_rng(3)
    vals = rng.random(500)
    codes = rng.integers(0, 37, 500)
    order, sv, starts, counts = E._groups(codes, vals)
    for q in (0.85, 0.9, 0.95):
        g = E._group_quantile(sv, starts, counts, q)
        ref = [np.quantile(vals[codes == c], q) for c in np.unique(codes)]
        assert np.array_equal(g, np.asarray(ref))


def test_p_adjust_bh():
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.04])
    q = E.p_adjust_bh(p)
    assert np.allclose(q, [0.05, 0.05, 0.05, 0.2, 0.05])


# ----------------------------------------------------------------------------- AUCell
def _aucell_reference(X, sig_idx, auc_threshold, seed):
    rng = np.random.default_rng(seed)
    n_cells, n = X.shape
    cutoff = int(round(auc_threshold * n)) - 1
    out = np.zeros((n_cells, len(sig_idx)))
    for i in range(n_cells):
        perm = rng.permutation(n)
        rank = perm[argsort_np126((-X[i])[perm])].argsort()
        for j, idx in enumerate(sig_idx):
            r = rank[idx]
            x = np.sort(r[r < cutoff])
            x = np.concatenate([x, [cutoff]])
            y = np.arange(1, len(x))
            out[i, j] = np.sum(np.diff(x) * y) / float((cutoff + 1) * len(idx))
    return out


def test_aucell_matches_reference():
    rng = np.random.default_rng(4)
    X = rng.poisson(0.7, size=(30, 400)).astype(np.float32)
    feats = np.array([f"f{i}" for i in range(400)])
    sig_idx = [rng.choice(400, size=k, replace=False) for k in (5, 40, 120)]
    sigs = {f"s{j}": list(feats[idx]) for j, idx in enumerate(sig_idx)}
    r = rank_cells(X, [f"c{i}" for i in range(30)], feats, auc_threshold=0.05, n_jobs=2)
    got = aucell(r, sigs).to_numpy()
    assert np.array_equal(got, _aucell_reference(X, sig_idx, 0.05, 123))


# ----------------------------------------------------------------------------- RSS
def test_rss_matches_scipy():
    from scipy.spatial.distance import jensenshannon

    rng = np.random.default_rng(5)
    auc = pd.DataFrame(rng.random((50, 4)), columns=list("abcd"))
    grp = pd.Series(rng.choice(["x", "y", "z"], 50), index=auc.index)
    rss = regulon_specificity_scores(auc, grp)
    for ct in rss.index:
        lab = (grp == ct).astype(int)
        for r in auc.columns:
            ref = 1 - jensenshannon(auc[r] / auc[r].sum(), lab / lab.sum())
            assert abs(rss.loc[ct, r] - ref) < 1e-12


# ----------------------------------------------------------------------------- triplet
def test_triplet_rank_and_region_mapping():
    m = map_regions(["chr1:100-200", "chr1:1000-1100"], ["chr1:150-400", "chr1:190-260", "chr2:100-200"])
    assert m.values.tolist() == [["chr1:100-200", "chr1:150-400"]]
    tab = pd.DataFrame({"TF": ["A", "A", "B"], "Region": ["r1", "r2", "r1"],
                        "importance_TF2G": [1.0, 0.5, 0.2], "importance_R2G": [0.1, 0.3, 0.3]})
    rank = pd.DataFrame({"A": [3, 1], "B": [2, 2]}, index=["r1", "r2"])
    t = triplet_rank(tab, rank)
    assert sorted(t.tolist()) == [0, 1, 2]


# ----------------------------------------------------------------------------- eGRN smoke
def test_build_egrn_small():
    import anndata as ad

    rng = np.random.default_rng(6)
    genes = [f"G{i}" for i in range(60)]
    regions = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(400)]
    tfs = ["TF1", "TF2"]
    r2g = pd.DataFrame({"target": rng.choice(genes, 3000), "region": rng.choice(regions, 3000)})
    r2g = r2g.drop_duplicates().reset_index(drop=True)
    r2g["importance"] = rng.gamma(1, 0.1, len(r2g))
    r2g["rho"] = rng.uniform(-0.6, 0.6, len(r2g))
    r2g["importance_x_rho"] = r2g["importance"] * r2g["rho"]
    r2g["importance_x_abs_rho"] = r2g["importance"] * r2g["rho"].abs()
    tf2g = pd.DataFrame([(tf, g) for tf in tfs for g in genes], columns=["TF", "target"])
    tf2g["importance"] = rng.gamma(1, 1, len(tf2g))
    tf2g["rho"] = rng.uniform(-0.5, 0.5, len(tf2g))
    tf2g["regulation"] = np.sign(tf2g["rho"]).astype(int)
    cis = ad.AnnData(X=(rng.random((len(regions), 2)) < 0.4), obs=pd.DataFrame(index=regions),
                     var=pd.DataFrame(index=tfs))
    try:
        out = E.build_egrn(tf2g, r2g, cis, min_target_genes=3, gsea_n_perm=100)
    except RuntimeError:
        pytest.skip("no eRegulon on random data")
    assert {"Region", "Gene", "TF", "eRegulon_name", "importance_TF2G", "rho_TF2G"} <= set(out.columns)
    assert out["eRegulon_name"].str.match(r"TF\d_direct_[+-]/[+-]$").all()


# ----------------------------------------------------------------------------- parity
_PARITY = os.environ.get("EPIONE_SCENIC_PARITY_DIR")


@pytest.mark.skipif(not _PARITY or not (Path(_PARITY or ".") / "A" / "egrn").exists(),
                    reason="EPIONE_SCENIC_PARITY_DIR with SCENIC+ reference outputs not available")
def test_parity_with_scenicplus(tmp_path):
    script = Path(__file__).parent / "parity" / "scenic_egrn" / "parity_scenic_egrn.py"
    out = tmp_path / "parity.json"
    env = dict(os.environ, PYTHONHASHSEED="0")
    subprocess.run([sys.executable, str(script), _PARITY, str(out)], check=True, env=env)
    r = json.loads(out.read_text())
    for kind in ("direct", "extended"):
        e = r[f"egrn_{kind}"]
        assert e["set_equal"] and e["order_equal"] and e["triplet_equal"]
        if f"thresholds_{kind}_max_abs_err" in r:
            assert r[f"thresholds_{kind}_max_abs_err"] == 0
            assert r[f"emodules_{kind}_equal"]
            g = r[f"gsea_{kind}"]
            assert g["same_calls"] and g["leading_edge_equal"] and g["NES_nan_pattern_equal"]
            assert g["NES_max_abs_err"] == 0 and g["pval_max_abs_err"] == 0
        for mod in ("Gene_based", "Region_based"):
            a = r[f"auc_{kind}_{mod}"]
            assert a["labels_equal"] and a["max_abs_err"] <= 1e-8
    assert r["rss"]["labels_equal"] and r["rss"]["max_abs_err"] <= 1e-8
