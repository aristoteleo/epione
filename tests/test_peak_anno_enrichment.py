"""Tests for epione.tl.peak_anno_enrichment & friends (ArchR peakAnnoEnrichment port)."""
import math
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.stats import hypergeom

import epione as epi
from epione.pl._enrich_heatmap import r_hclust_complete
from epione.tl._peak_anno_enrichment import _phyper_log_upper, _r_round


def test_phyper_log_upper_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(500):
        N = int(rng.integers(10, 5000)); K = int(rng.integers(0, N)); n = int(rng.integers(0, N))
        k = int(rng.integers(0, min(K, n) + 1))
        a = _phyper_log_upper(k - 1, K, N - K, n)
        b = hypergeom(N, K, n).logsf(k - 1)
        if np.isfinite(b) and b < -1e-10:
            assert abs(a - b) <= 1e-8 * abs(b)
    assert _phyper_log_upper(-1, 10, 10, 5) == 0.0          # k = 0 -> P[X > -1] = 1
    assert _phyper_log_upper(5, 10, 10, 5) == -math.inf     # beyond support


def test_phyper_log_upper_extreme_tail_is_finite():
    # p ~ 1e-600 underflows in linear space; log-space must stay finite.
    v = _phyper_log_upper(799, 1000, 100000, 1000)
    assert np.isfinite(v) and v < -1000


def test_r_round_matches_r():
    # Reference values from R 4.4.3 round(x, 4).
    x = [0.12345, 1.00005, 2.5e-5, 12.34565, -0.0, 3.14159265]
    r = [0.1235, 1.0001, 0.0, 12.3456, -0.0, 3.1416]
    assert np.array_equal(_r_round(x, 4), np.array(r))


def test_r_hclust_order_known_case():
    # R 4.4.3: h <- hclust(dist(x)); h$order = 5 1 3 2 4; h$merge as below.
    x = np.array([[0, 0], [10, 0], [0, 1], [11, 1], [30, 0]], dtype=float)
    merge, height, order = r_hclust_complete(x)
    assert list(order + 1) == [5, 1, 3, 2, 4]
    assert merge.tolist() == [[-1, -3], [-2, -4], [1, 2], [-5, 3]]
    from scipy.cluster.hierarchy import linkage
    assert np.allclose(np.sort(height), np.sort(linkage(x, "complete")[:, 2]))


def _toy():
    peaks = [f"chr1:{i * 1000 + 1}-{i * 1000 + 500}" for i in range(20)]
    M = np.zeros((20, 3), dtype=bool)
    M[:5, 0] = True          # motif A in the first 5 peaks
    M[::2, 1] = True         # motif B in every other peak
    M[19, 2] = True
    adata = ad.AnnData(var=pd.DataFrame(index=peaks))
    adata.varm["motif"] = sparse.csr_matrix(M)
    adata.uns["motif_names"] = np.array(["A", "B", "C"])
    fdr = pd.DataFrame({"g1": [0.01] * 5 + [0.9] * 15, "g2": [0.9] * 20}, index=peaks)
    lfc = pd.DataFrame({"g1": [2.0] * 20, "g2": [2.0] * 20}, index=peaks)
    return adata, {"FDR": fdr, "Log2FC": lfc}


def test_enrichment_counts_and_pvalue():
    adata, markers = _toy()
    e = epi.tl.peak_anno_enrichment(markers, adata, verbose=False)
    assert list(e) == ["mlog10Padj", "mlog10p", "Enrichment", "BackgroundProporition",
                       "nBackground", "BackgroundFrequency", "CompareProportion", "nCompare",
                       "CompareFrequency", "feature"]
    assert e["CompareFrequency"].loc["A", "g1"] == 5
    assert e["nCompare"].loc["A", "g1"] == 5
    assert e["BackgroundFrequency"].loc["B", "g1"] == 10
    assert e["Enrichment"].loc["A", "g1"] == pytest.approx(4.0)
    p = -hypergeom(20, 5, 5).logsf(4) / math.log(10)
    assert e["mlog10p"].loc["A", "g1"] == pytest.approx(round(p, 4), abs=1e-12)
    assert e["mlog10Padj"].loc["A", "g1"] == pytest.approx(max(round(p, 4) - math.log10(9), 0))
    # no foreground peaks -> NaN proportions, p = 1
    assert e["nCompare"].loc["A", "g2"] == 0 and np.isnan(e["CompareProportion"].loc["A", "g2"])
    assert e["mlog10p"].loc["A", "g2"] == 0


def test_enrichment_bgd_counts_duplicates():
    adata, markers = _toy()
    adata.varm["bg_peaks"] = np.tile(np.array([[19, 19]]), (20, 1))
    e = epi.tl.peak_anno_enrichment(markers, adata, background="bgdPeaks", verbose=False)
    assert e["nBackground"].loc["A", "g1"] == 15          # 5 fg + 5 * 2 bg (duplicates kept)
    assert e["BackgroundFrequency"].loc["C", "g1"] == 10


def test_cutoff_or_and_nan():
    adata, markers = _toy()
    markers["FDR"].iloc[0, 0] = np.nan
    e = epi.tl.peak_anno_enrichment(markers, adata, cutoff="FDR <= 0.1 & Log2FC >= 0.5", verbose=False)
    assert e["nCompare"].loc["A", "g1"] == 4
    e = epi.tl.peak_anno_enrichment(markers, adata, cutoff="FDR <= 0.1 | Log2FC >= 1", verbose=False)
    assert e["nCompare"].loc["A", "g2"] == 20
    with pytest.raises(ValueError):
        epi.tl.peak_anno_enrichment(markers, adata, cutoff="__import__('os')", verbose=False)


def test_long_marker_table_input():
    adata, markers = _toy()
    long = pd.concat([pd.DataFrame({"group": g, "feature": markers["FDR"].index,
                                    "fdr": markers["FDR"][g].to_numpy(),
                                    "log2_fc": markers["Log2FC"][g].to_numpy()})
                      for g in ["g1", "g2"]])
    e1 = epi.tl.peak_anno_enrichment(long, adata, verbose=False)
    e2 = epi.tl.peak_anno_enrichment(markers, adata, verbose=False)
    pd.testing.assert_frame_equal(e1["mlog10p"], e2["mlog10p"])


def test_add_peak_annotations_closed_intervals(tmp_path):
    adata, _ = _toy()
    bed = tmp_path / "r.bed"
    bed.write_text("chr1\t500\t600\nchr1\t1501\t1501\nchr2\t1\t100000\n")
    epi.tl.add_peak_annotations(adata, {"set1": str(bed),
                                        "none": pd.DataFrame({"chrom": ["chr9"], "start": [1], "end": [2]})},
                                key_added="Region", verbose=False)
    M = adata.varm["Region"].toarray()
    assert list(adata.uns["Region_names"]) == ["set1"]       # zero-overlap sets dropped
    assert M[:, 0].nonzero()[0].tolist() == [0]                # peak 2 (1001-1500) does not touch 1501
    with pytest.raises(ValueError):
        epi.tl.add_peak_annotations(adata, {"x": pd.DataFrame({"chrom": ["chr9"], "start": [1], "end": [2]})})


def test_enrich_heatmap_matrix_shape():
    adata, markers = _toy()
    e = epi.tl.peak_anno_enrichment(markers, adata, verbose=False)
    m = epi.pl.enrich_heatmap(e, cutoff=0, return_matrix=True)
    assert m.max(axis=1).round(10).eq(100).all()
    assert m.index[0].startswith("A (")


# --- parity against ArchR on the Heme fixture ---------------------------------

FIXTURE = os.environ.get("EPIONE_ARCHR_FIXTURE",
                         "/scratch/users/steorra/analysis/25_epione/epione_dev/fixture")
REF = os.environ.get("EPIONE_PEAK_ANNO_REF",
                     "/scratch/users/steorra/analysis/25_epione/epione_dev/work/peak-anno-enrichment/ref")
ANNO = os.environ.get("EPIONE_ARCHR_ANNO",
                      "/scratch/users/steorra/analysis/25_epione/epione_dev/work/peak-anno-enrichment/anno/ArchR-Hg19-v1.Anno")


@pytest.mark.skipif(not (os.path.isdir(FIXTURE) and os.path.isdir(REF) and os.path.exists(ANNO)),
                    reason="ArchR Heme fixture / reference outputs not available")
def test_parity_vs_archr():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "parity", "peak_anno_enrichment"))
    import compare
    rep = compare.main([FIXTURE, REF, ANNO])
    for k, v in rep.items():
        if isinstance(v, dict) and "max_abs_err" in v:
            assert v["max_abs_err"] <= 1e-8, (k, v)
        if isinstance(v, dict):
            for flag in ("feature_identical", "dimnames_identical", "names_identical", "row_order_identical"):
                if flag in v:
                    assert v[flag], (k, flag)
            if "n_mismatch" in v:
                assert v["n_mismatch"] == 0, (k, v)
    assert rep["raw_mlog10p_max_rel_err"] <= 1e-10
    n_ok, n_tot = map(int, rep["hclust_order_identical"].split("/"))
    assert n_ok == n_tot
