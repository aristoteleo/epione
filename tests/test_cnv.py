"""Tests for epi.tl.cnv (epiAneufinder port).

Small references in tests/parity/epianeufinder/small_reference.json were
produced by R epiAneufinder 1.1.5 / R 4.4.3 (seq_dist_ad, getbp,
assign_gainloss, quantile, mean, hclust(ward.D), cutree).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epione.tl import _cnv as C

REF = json.loads((Path(__file__).parent / "parity" / "epianeufinder" / "small_reference.json").read_text())


def _scan(x, minsize, test):
    if test == "KS":
        return C._seq_dist_ks(np.asarray(x), minsize)
    codes, D = C._codes(np.asarray(x))
    return C._seq_dist_ad(codes, D, minsize)


@pytest.mark.parametrize("key,minsize,test", [("ad", 1, "AD"), ("ks", 1, "KS"), ("ad_minsize3", 3, "AD")])
def test_seq_dist_matches_r(key, minsize, test):
    got = _scan(REF["seq1"], minsize, test)
    exp = np.asarray(REF[key], dtype=float)
    assert got.shape == exp.shape
    np.testing.assert_allclose(got, exp, rtol=1e-12, atol=0)


@pytest.mark.parametrize("key,seq,k,minsize,test,mcnv", [
    ("g1", "seq1", 4, 1, "AD", 0),
    ("g2", "seq1", 3, 2, "AD", 3),
    ("g3", "seq2", 4, 1, "KS", 0),
])
def test_getbp_matches_r(key, seq, k, minsize, test, mcnv):
    bp, d = C._getbp(np.asarray(REF[seq]), k=k, minsize=minsize, test=test, minsize_cnv=mcnv)
    assert bp == REF[key]["bp"]
    np.testing.assert_allclose(d, REF[key]["ad.distance"], rtol=1e-12)


def test_assign_gainloss_matches_r():
    x = np.asarray(REF["seq1"], dtype=np.int64)
    got = C._assign_gainloss(x, np.asarray(REF["cluster"]), uq=0.9, lq=0.1, is_int=True)
    assert got.tolist() == [int(v) for v in REF["gainloss"]]


def test_r_numerics():
    x = np.asarray(REF["seq1"], dtype=float)
    q = C._r_quantile7(x / C._r_mean(x), (0.01, 0.1, 0.9, 0.98))
    np.testing.assert_array_equal(q, np.asarray(REF["q7"]))
    assert C._r_mean(x / 7) == REF["mean"][0]


def test_hclust_ward_d_and_cutree_match_r():
    M = np.asarray(REF["M"], dtype=float)
    merge, height, order = C._r_hclust_ward_d(M)
    np.testing.assert_array_equal(merge, np.asarray(REF["merge"]))
    np.testing.assert_allclose(height, REF["height"], rtol=1e-13)
    np.testing.assert_array_equal(order, REF["order"])
    ct = np.asarray(REF["cutree"])
    for j, k in enumerate(range(2, 7)):
        np.testing.assert_array_equal(C._r_cutree(merge, k), ct[:, j])


# ---------------------------------------------------------------------------
# end-to-end on a synthetic genome
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("cnv")
    sizes = {"chr1": 2_000_000, "chr2": 1_500_000, "chr3": 1_000_000, "chr4": 1_500_000,
             "chr5": 1_500_000, "chrUn_x": 50_000}
    with open(d / "g.fa", "w") as fh:
        for c, L in sizes.items():
            s = rng.choice(list("ACGT"), size=L, p=[0.3, 0.2, 0.2, 0.3])
            if c == "chr1":
                s[:20_000] = "N"
            fh.write(f">{c}\n")
            seq = "".join(s)
            for i in range(0, L, 80):
                fh.write(seq[i:i + 80] + "\n")
    pd.DataFrame({0: ["chr3"], 1: [500_000], 2: [520_000]}).to_csv(d / "bl.bed", sep="\t", header=False, index=False)
    rows = []
    n_cells = 24
    for ci in range(n_cells):
        tumour = ci < 12
        for c in ("chr1", "chr2", "chr3", "chr4", "chr5"):
            L = sizes[c]
            n = rng.poisson(L / 1000 * 3)
            starts = np.sort(rng.integers(1, L - 300, n))
            if tumour and c == "chr2":  # gain of the first half of chr2
                extra = np.sort(rng.integers(1, L // 2, n))
                starts = np.sort(np.r_[starts, extra])
            if tumour and c == "chr3":  # loss of the second half of chr3
                starts = starts[(starts < L // 2) | (rng.random(starts.size) < 0.25)]
            for s in starts:
                rows.append((c, int(s), int(s) + 150, f"cell{ci:02d}", 1))
    fr = pd.DataFrame(rows).sort_values([0, 1])
    fr.to_csv(d / "frag.tsv", sep="\t", header=False, index=False)
    return d


def test_cnv_end_to_end(synthetic):
    import epione as epi
    d = synthetic
    res = epi.tl.cnv(str(d / "frag.tsv"), genome=str(d / "g.fa"), blacklist=str(d / "bl.bed"),
                     window_size=20_000, exclude=None, min_frags=1000, verbose=False)
    assert res.n_obs == 24
    assert set(res.var["chrom"].astype(str)) == {"chr1", "chr2", "chr3", "chr4", "chr5"}
    # blacklisted and N windows are gone
    assert not ((res.var["chrom"] == "chr3") & (res.var["start"] == 500_000)).any()
    assert not ((res.var["chrom"] == "chr1") & (res.var["start"] < 20_000)).any()
    X = np.asarray(res.X)
    chr2a = ((res.var["chrom"] == "chr2") & (res.var["end"] <= 700_000)).to_numpy()
    chr3b = ((res.var["chrom"] == "chr3") & (res.var["start"] >= 550_000)).to_numpy()
    tum = np.array([int(c[4:]) < 12 for c in res.obs_names])
    assert (X[tum][:, chr2a] == 2).mean() > 0.8
    assert (X[tum][:, chr3b] == 0).mean() > 0.8
    assert (X[~tum] == 1).mean() > 0.9
    assert res.obs.loc[tum, "cnv_burden"].min() > res.obs.loc[~tum, "cnv_burden"].max()
    s = epi.tl.cnv_subclones(res, 2)
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(tum, s.astype(int)) == 1.0
    res.write_h5ad(d / "cnv.h5ad")  # uns is serialisable
    km = epi.tl.cnv_karyotype_measures(res)
    assert set(km) == {"genomewide", "per_chromosome"}
    # parallel run gives the same result
    res2 = epi.tl.cnv(str(d / "frag.tsv"), genome=str(d / "g.fa"), blacklist=str(d / "bl.bed"),
                      window_size=20_000, exclude=None, min_frags=1000, verbose=False, n_jobs=2)
    np.testing.assert_array_equal(np.asarray(res2.X), X)
    import matplotlib
    matplotlib.use("Agg")
    epi.pl.karyogram(res, groupby="cnv_subclone")
    epi.pl.cnv_profile(res, res.obs_names[0])


def test_cnv_peak_matrix_input(synthetic):
    import anndata as ad
    import scipy.sparse as sp
    import epione as epi
    d = synthetic
    win = epi.tl.cnv_make_windows(str(d / "g.fa"), str(d / "bl.bed"), 20_000)
    fr = pd.read_csv(d / "frag.tsv", sep="\t", header=None)
    # one "peak" per 5 kb, cells x peaks counts from the fragments
    fr["peak"] = fr[0] + "-" + ((fr[1] // 5000) * 5000 + 1).astype(str) + "-" + ((fr[1] // 5000) * 5000 + 4000).astype(str)
    tab = fr.groupby([3, "peak"]).size().unstack(fill_value=0)
    a = ad.AnnData(sp.csr_matrix(tab.to_numpy()), obs=pd.DataFrame(index=tab.index), var=pd.DataFrame(index=tab.columns))
    res = epi.tl.cnv(a, windows=win, window_size=20_000, verbose=False)
    assert res.n_obs == 24 and res.n_vars > 100
