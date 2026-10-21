"""Unit tests for the dcHiC ``analyze`` port (epione.bulk.hic._dchic).

Reference values were produced with R 4.1.3, robust 0.7.1, rrcov 1.7-3 and
limma 3.50.3 (conda env ``dchic``); the R-side parity gates live in
tests/parity/schicluster_compartment/dchic_parity.py.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from epione.bulk.hic import _dchic as dc


def _ogk_fixture():
    n, p = 30, 4
    X = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            X[i, j] = ((i * 7 + j * 13 + i * j * 3) % 17) / 4 + j / 10 + ((i * i) % 5) / 8
    return X


def test_normalize_quantiles_ties_matches_limma():
    A = np.array([[5, 2, 3, 4], [4, 1, 4, 2], [3, 4, 6, 8], [4, 2, 8, 4], [1, 2, 4, 3]], float)
    ref = np.array([[6.25, 3.5, 1.75, 3.75], [3.75, 1.75, 3.25, 1.75], [3.0, 6.25, 4.0, 6.25],
                    [3.75, 3.5, 6.25, 3.75], [1.75, 3.5, 3.25, 3.0]])
    np.testing.assert_array_equal(dc.normalize_quantiles(A), ref)
    one = np.array([[1.0], [3.0], [2.0]])
    np.testing.assert_array_equal(dc.normalize_quantiles(one), one)


def test_cov_rob_pairwise_matches_robust():
    X = _ogk_fixture()
    ref_qc = np.array([1.12435, -0.76749999999999996, 0.31480000000000002, 0.48504999999999998,
                       -0.76749999999999996, 1.20625, 0.056249999999999994, -0.095000000000000001,
                       0.31480000000000002, 0.056249999999999994, 1.3846499999999999, -0.099599999999999994,
                       0.48504999999999998, -0.095000000000000001, -0.099599999999999994,
                       1.8261499999999997]).reshape(4, 4)
    ref_gk = np.array([0.99845377604166663, -0.96183268229166663, 0.190673828125, 0.36368815104166669,
                       -0.96183268229166663, 1.1039225260416667, -0.070393880208333329,
                       -0.23201497395833337, 0.190673828125, -0.070393880208333329, 1.3333062065972225,
                       -0.21622721354166666, 0.36368815104166669, -0.23201497395833337,
                       -0.21622721354166666, 1.7862141927083333]).reshape(4, 4)
    np.testing.assert_allclose(dc.cov_rob(X, "pairwiseqc")["cov"], ref_qc, rtol=1e-12, atol=0)
    np.testing.assert_allclose(dc.cov_rob(X, "pairwisegk")["cov"], ref_gk, rtol=1e-12, atol=0)


def test_cov_rob_auto_dispatch():
    rng = np.random.default_rng(0)
    # p >= 20 -> pairwiseqc
    assert dc.cov_rob(rng.normal(size=(200, 20)))["estim"] == "pairwiseqc"
    # n >= 1000, 5 <= p < 20 -> mcd; n < 1000, p < 10 -> donostah
    r = dc.cov_rob(rng.normal(size=(1200, 6)))
    assert r["estim"] == "mcd" and r["cov"].shape == (6, 6)
    assert dc.cov_rob(rng.normal(size=(300, 3)))["estim"] == "donostah"
    with pytest.raises(ValueError):
        dc.cov_rob(rng.normal(size=(3, 5)))


def test_r_round_and_format_edge_cases():
    # exact ties at the 6th decimal go to the even neighbour; near-ties follow R
    x = np.array([0.000005, 0.000015, 2.675, 1.234565, -1.234565, 1e-320, 0.0])
    out = dc.r_round(x, 5)
    assert out[-1] == 0.0
    assert np.all(np.isfinite(out))
    assert dc._r_format(13.708020988102952) == "13.708020988103"   # R: 14 digits here
    assert dc._r_format(1e-5) == "1e-05"
    assert dc._r_format(0.0001) == "1e-04"
    assert dc._r_format(123456.0) == "123456"
    assert dc._r_format(float("nan")) == "NA"
    np.testing.assert_array_equal(dc.r_strtod(["NA", "-0", "1e5", ".5", "27.47848896103256"]),
                                  [np.nan, -0.0, 1e5, 0.5, dc._r_strtod_scalar(["27.47848896103256"])[0]])


def test_bh_matches_p_adjust():
    p = np.array([0.01, 0.04, 0.03, 0.04, 0.5, 0.2])
    # R: p.adjust(c(0.01,0.04,0.03,0.04,0.5,0.2), "BH")
    ref = np.array([0.06, 0.06, 0.06, 0.06, 0.5, 0.24])
    np.testing.assert_allclose(dc._bh(p), ref, rtol=1e-15)


def _synthetic_scores(n_cond=20, n_bins=150, chroms=("chr1", "chr2"), seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for c in chroms:
        base = np.sin(np.arange(n_bins) / 7.0)
        for b in range(n_bins):
            rows.append((c, b * 100_000, (b + 1) * 100_000, base[b]))
    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "base"])
    for k in range(n_cond):
        v = df["base"].to_numpy() + rng.normal(scale=0.2, size=len(df))
        if k == 0:
            v[30:40] = v[30:40] + 3.0   # condition-specific shift on chr1
        df[f"s{k}"] = np.round(v, 6)
    return df.drop(columns="base")


def test_differential_compartments_output(tmp_path):
    df = _synthetic_scores()
    samples = [c for c in df.columns if c[0] == "s" and c[1:].isdigit()]
    conds = [f"cond{k}" for k in range(len(samples))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = dc.differential_compartments(df, conds, out_dir=tmp_path / "all")
    assert list(res.columns[:3]) == ["chr", "start", "end"]
    assert list(res.columns[-4:]) == ["sample_maha", "pval", "padj", "dist_clust"]
    assert res.shape == (300, 3 + 2 * len(samples) + 4)
    assert res["padj"].between(0, 1).all()
    # the planted switch is the most significant region on chr1
    top = res.loc[res["chr"] == "chr1"].nsmallest(10, "pval")["start"] // 100_000
    assert top.between(30, 39).all()
    fr = tmp_path / "all" / "fdr_result"
    for f in ["differential.intra_sample_combined.pcQnm.bedGraph",
              "differential.intra_sample_combined.Filtered.pcQnm.bedGraph",
              "differential.intra_sample_group.pcQnm.bedGraph",
              "differential.intra_sample_group.pcOri.bedGraph"]:
        assert (fr / f).exists()
    back = pd.read_csv(fr / "differential.intra_sample_combined.pcQnm.bedGraph", sep="\t")
    assert list(back.columns) == list(res.columns)


def test_replicates_need_explicit_bh():
    df = _synthetic_scores(n_cond=20)
    samples = [c for c in df.columns if c[0] == "s" and c[1:].isdigit()]
    conds = [f"cond{k // 2}" for k in range(len(samples))]   # 10 conditions x 2 replicates
    with pytest.raises(NotImplementedError):
        dc.differential_compartments(df, conds)


def test_dchic_analyze_reads_dchic_layout(tmp_path):
    df = _synthetic_scores(n_cond=20)
    samples = [c for c in df.columns if c[0] == "s" and c[1:].isdigit()]
    lines = []
    for s in samples:
        d = tmp_path / f"{s}_pca" / "intra_pca" / f"{s}_mat"
        d.mkdir(parents=True)
        for chrom, g in df.groupby("chrom"):
            g[["chrom", "start", "end", s]].to_csv(d / f"{chrom}.pc.bedGraph", sep="\t", header=False, index=False)
        lines.append(f".\t.\t{s}\tc_{s}")
    (tmp_path / "input.txt").write_text("\n".join(lines) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = dc.dchic_analyze(tmp_path / "input.txt", tmp_path, diffdir="all", dirovwt=True)
        b = dc.differential_compartments(df, {s: f"c_{s}" for s in samples})
    np.testing.assert_allclose(a["pval"].to_numpy(), b["pval"].to_numpy(), rtol=1e-12)
    assert dc.dchic_analyze(tmp_path / "input.txt", tmp_path, diffdir="all") is None   # dirovwt=False
