"""Tests for epione.scenic cisTarget / DEM / cistrome merging.

Unit tests run on small synthetic databases and check against direct
transcriptions of the upstream (ctxcore / pycistarget) formulas. The parity
test against the SCENIC+ reference runs only when ``EPIONE_SCENIC_PARITY_DIR``
and ``EPIONE_SCENIC_DB_DIR`` are set (see tests/parity/scenic_cistarget).
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import epione.scenic as sc
from epione.scenic._cistarget import _leading_edges, _rank_cutoff


# ---------------------------------------------------------------------------
# helpers: literal transcriptions of the upstream algorithms
# ---------------------------------------------------------------------------

def _ref_aucs(rank, total, auc_threshold):
    """ctxcore.recovery.aucs with unit weights (weighted_auc1d)."""
    cutoff = int(round(auc_threshold * total)) - 1
    w = np.ones(rank.shape[0])
    maxauc = float((cutoff + 1) * w.sum())
    out = []
    for j in range(rank.shape[1]):
        r = rank[:, j]
        f = r < cutoff
        x = np.sort(r[f])
        x = np.concatenate((x, [cutoff])).astype(np.int64)
        y = w[f][np.argsort(r[f])].cumsum()
        out.append(np.sum(np.diff(x) * y) / maxauc)
    return np.array(out)


def _ref_rank_at_max(rank, total, rank_threshold, enriched):
    """ctxcore.recovery.recovery (rcc2d) + leading_edge."""
    n, m = rank.shape
    w = np.insert(np.ones(n), n, 0.0)
    rk = np.vstack([rank, np.full((1, m), total)])
    rccs = np.empty((m, rank_threshold))
    for j in range(m):
        rccs[j] = np.cumsum(np.bincount(rk[:, j], weights=w)[:rank_threshold])
    avg2std = rccs.mean(axis=0) + 2.0 * rccs.std(axis=0)
    return np.array([np.argmax(rccs[j] - avg2std) for j in enriched])


def _random_db(n_regions=3000, n_motifs=60, seed=0, plant=None):
    rng = np.random.default_rng(seed)
    regions = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(n_regions)]
    R = np.empty((n_regions, n_motifs), dtype=np.int32)
    for j in range(n_motifs):
        R[:, j] = rng.permutation(n_regions)
    if plant is not None:
        idx, motif = plant
        # put the planted regions at the top of one motif's ranking
        col = R[:, motif].copy()
        top = np.argsort(col)[: len(idx)]
        col[top], col[idx] = col[idx], col[top].copy()
        R[:, motif] = col
    motifs = [f"m{j}" for j in range(n_motifs)]
    return sc.CisTargetDB.from_matrix(R, regions, motifs, kind="rankings"), regions, R


# ---------------------------------------------------------------------------
# region mapping
# ---------------------------------------------------------------------------

def test_target_to_query_fraction_rule():
    q = pd.DataFrame({"Chromosome": ["chr1"] * 3, "Start": [100, 300, 1000], "End": [200, 400, 1100]})
    t = pd.DataFrame({"Chromosome": ["chr1", "chr1", "chr2"], "Start": [150, 395, 100],
                      "End": [450, 500, 200]})
    m = sc.target_to_query(t, q, fraction_overlap=0.4)
    pairs = set(zip(m["Target"], m["Query"]))
    # 150-450 covers 50% of 100-200 and 100% of 300-400; 395-500 covers 5% / 5%
    assert pairs == {("chr1:150-450", "chr1:100-200"), ("chr1:150-450", "chr1:300-400")}
    with pytest.raises(ValueError):
        sc.target_to_query(t.iloc[[2]], q)


def test_target_to_query_matches_bruteforce():
    rng = np.random.default_rng(3)
    def rand(n):
        s = rng.integers(0, 20000, n)
        return pd.DataFrame({"Chromosome": rng.choice(["chr1", "chr2"], n), "Start": s,
                             "End": s + rng.integers(1, 800, n)})
    t, q = rand(300), rand(400)
    got = set(zip(*sc.target_to_query(t, q, 0.4).values.T))
    exp = set()
    for _, a in t.iterrows():
        for _, b in q.iterrows():
            if a.Chromosome != b.Chromosome:
                continue
            ov = min(a.End, b.End) - max(a.Start, b.Start)
            if ov > 0 and (ov / (b.End - b.Start) > 0.4 or ov / (a.End - a.Start) > 0.4):
                exp.add((f"{a.Chromosome}:{a.Start}-{a.End}", f"{b.Chromosome}:{b.Start}-{b.End}"))
    assert got == exp


# ---------------------------------------------------------------------------
# recovery statistics
# ---------------------------------------------------------------------------

def test_aucs_and_leading_edge_bit_exact_vs_ctxcore_formulas():
    db, regions, R = _random_db(plant=(np.arange(40), 7))
    sub = R[:200]
    total = R.shape[0]
    aucs = sc.recovery_aucs(sub, total, 0.05)
    np.testing.assert_array_equal(aucs, _ref_aucs(sub, total, 0.05))
    enriched = np.array([0, 7, 13])
    rt = int(0.2 * total)
    for block in (7, 64, 4096):
        got = _leading_edges(sub, enriched, rt, block=block)
        np.testing.assert_array_equal(got, _ref_rank_at_max(sub, total, rt, enriched))


def test_run_cistarget_finds_planted_motif():
    planted = np.arange(0, 120)
    db, regions, R = _random_db(plant=(planted, 5))
    annot = pd.DataFrame({"Direct_annot": ["TFA, TFB"], "Orthology_annot": [np.nan]},
                         index=["m5"])
    sets = {"planted": [regions[i] for i in planted], "random": regions[1000:1120]}
    res = sc.run_cistarget(sets, db, motif_annotation=annot, auc_threshold=0.05,
                           rank_threshold=0.2, verbose=False)
    me = res["planted"].motif_enrichment
    assert me.index[0] == "m5"
    assert me.loc["m5", "Direct_annot"] == "TFA, TFB"
    # the planted regions sit at the top of m5, so they are all hits
    assert set(res["planted"].motif_hits["Database"]["m5"]) == set(sets["planted"])
    assert any(k.startswith("TFA_(") for k in res["planted"].cistromes["Region_set"])
    # NES is (AUC - mean) / std over all motifs
    a = res["planted"].extra["aucs"].to_numpy()
    assert me.loc["m5", "NES"] == ((a - a.mean()) / a.std())[5]


def test_motifs_per_tf_keeps_upstream_suffix_rule():
    me = pd.DataFrame({"Direct_annot": ["SPAX5", "PAX5, X", "Y"]}, index=["a", "b", "c"])
    from epione.scenic._motif_annotation import motifs_per_tf
    assert motifs_per_tf(me, "PAX5", ["Direct_annot"]) == ["a", "b"]


# ---------------------------------------------------------------------------
# DEM
# ---------------------------------------------------------------------------

def test_background_sampling_uses_cpython_shuffle():
    fg = ["chr1:100-200", "chr1:5000-5100"]
    bg = [f"chr1:{10000 + 1000 * i}-{10100 + 1000 * i}" for i in range(50)]
    ga = pd.DataFrame({"Chromosome": ["chr1", "chr1"], "Start": [150, 20050]})
    f, b = sc.foreground_background_regions([fg], [bg], max_bg_regions=10,
                                            genome_annotation=ga, seed=666)
    names = [f"{c}:{s}-{e}" for c, s, e in b.values]
    # 1 of 2 foreground regions is a promoter → int(10 * 0.5) = 5 promoter slots;
    # the gene at 20050 (± 1000) overlaps 3 background regions
    prom = sorted(["chr1:19000-19100", "chr1:20000-20100", "chr1:21000-21100"])
    nonprom = sorted(set(bg) - set(prom))
    random.Random(666).shuffle(prom)
    random.Random(666).shuffle(nonprom)
    assert names == prom[:3] + nonprom[:7]


def test_dem_statistics_match_scipy():
    from scipy.stats import ranksums
    rng = np.random.default_rng(0)
    fg = rng.random((20, 300)).astype(np.float32)
    bg = (rng.random((20, 200)) * 0.9).astype(np.float32)
    bg[:, :50] = fg[:, :50]          # ties
    st = sc.dem_test(fg, bg)
    p = np.array([ranksums(fg[i], bg[i]).pvalue for i in range(20)])
    np.testing.assert_allclose(st["pval"], p, rtol=1e-10)
    np.testing.assert_allclose(st["Mean_fg"], fg.mean(axis=1, dtype=np.float64), rtol=1e-6)


def test_p_adjust_bh():
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.04])
    q = sc.p_adjust_bh(p)
    n = len(p)
    o = np.argsort(p)
    exp = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(exp, 1)
    np.testing.assert_allclose(q, out)


# ---------------------------------------------------------------------------
# I/O and the MOODS backend
# ---------------------------------------------------------------------------

def test_feather_roundtrip_and_hdf5(tmp_path):
    planted = np.arange(0, 120)
    db, regions, R = _random_db(plant=(planted, 5))
    f = db.write_feather(tmp_path / "toy.regions_vs_motifs.rankings.feather")
    db2 = sc.CisTargetDB(f)
    assert db2.kind == "rankings" and list(db2.motifs) == list(db.motifs)
    sets = {"planted": [regions[i] for i in planted]}
    kw = dict(motif_annotation=pd.DataFrame({"Direct_annot": ["TFA"], "Orthology_annot": [np.nan]},
                                             index=["m5"]),
              auc_threshold=0.05, rank_threshold=0.2, verbose=False)
    a = sc.run_cistarget(sets, db, **kw)["planted"]
    b = sc.run_cistarget(sets, db2, **kw)["planted"]
    pd.testing.assert_frame_equal(a.motif_enrichment, b.motif_enrichment)
    assert a.motif_hits == b.motif_hits
    direct, extended = sc.merge_cistromes({"planted": a})
    assert list(direct.var_names) == ["TFA"]
    assert int(direct.X.sum()) == len(a.motif_hits["Region_set"]["m5"])
    if importlib.util.find_spec("tables") is not None:   # pycistarget hdf5 needs PyTables
        h5 = sc.write_hdf5({"planted": a}, tmp_path / "ctx.hdf5")
        back = sc.read_hdf5(h5)["planted"]
        assert back.motif_hits["Region_set"] == a.motif_hits["Region_set"]
        d2, _ = sc.merge_cistromes([h5])
        assert (d2.X != direct.X).nnz == 0
    # cache round trip
    db.preload(sets["planted"])
    db.save_cache(tmp_path / "c.npz")
    db3 = sc.CisTargetDB(f).load_cache(tmp_path / "c.npz")
    r1, v1 = db.load(sets["planted"])
    r3, v3 = db3.load(sets["planted"])
    assert list(r1) == list(r3) and np.array_equal(v1, v3)


def test_motif_score_db_from_moods_hits(tmp_path):
    d = tmp_path / "mdb"
    d.mkdir()
    (d / "_meta.json").write_text(json.dumps({"motif_names": ["MA1_A", "MA2_B"], "max_motif_len": 10}))
    pd.DataFrame({"motif_idx": np.array([0, 0, 1], dtype=np.uint16),
                  "start": np.array([105, 150, 950], dtype=np.int32),
                  "end": np.array([115, 160, 960], dtype=np.int32),
                  "score": np.array([5.0, 7.5, 3.0], dtype=np.float32),
                  "strand": np.array([1, -1, 1], dtype=np.int8)}).to_parquet(d / "chr1.parquet")
    regions = ["chr1:100-200", "chr1:900-1000", "chr1:5000-5100"]
    S = sc.motif_score_db(regions, d, verbose=False)
    names, v = S.load(regions)            # rows come back in sorted name order
    got = pd.DataFrame(v, index=names).loc[regions].to_numpy()
    np.testing.assert_array_equal(got, [[7.5, 0], [0, 3.0], [0, 0]])
    R = sc.scores_to_rankings(S)
    names, r = R.load(regions)
    r = pd.DataFrame(r, index=names).loc[regions].to_numpy()
    assert r[0, 0] == 0 and r[1, 1] == 0
    assert sorted(r[:, 0]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# parity against the SCENIC+ reference (opt-in)
# ---------------------------------------------------------------------------

_PARITY = os.environ.get("EPIONE_SCENIC_PARITY_DIR")
_DBDIR = os.environ.get("EPIONE_SCENIC_DB_DIR")


@pytest.mark.skipif(not (_PARITY and _DBDIR and Path(_PARITY, "A", "cistarget").exists()),
                    reason="EPIONE_SCENIC_PARITY_DIR / EPIONE_SCENIC_DB_DIR not available")
def test_parity_with_scenicplus():
    spec = importlib.util.spec_from_file_location(
        "parity_scenic_cistarget", Path(__file__).parent / "parity" / "scenic_cistarget" / "parity_cistarget.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out, ctx = mod.run_cistarget_parity()
    assert out["gate2_auc_maxerr"] == 0.0 and out["gate3_nes_maxerr"] <= 1e-12
    for k in ("gate1_regions_to_db", "gate4_or_10_motif_set", "gate4_rank_at_max",
              "gate5_motif_hits", "gate6_annotation", "gate7_cistromes"):
        a, b = out[k].split()[0].split("/")
        assert a == b, k
    out, dem = mod.run_dem_parity()
    assert out["gate9_pval_maxerr"] <= 1e-12 and out["gate9_padj_maxerr"] <= 1e-12
    for k in ("gate8_foreground", "gate8_background_order", "gate5_motif_hits", "gate7_cistromes"):
        a, b = out[k].split()[0].split("/")
        assert a == b, k
    menr = mod.run_menr_parity(ctx, dem)
    for kind in ("direct", "extended"):
        assert all(menr[f"gate11_{kind}"][k] for k in ("same_tfs", "same_membership", "same_motifs"))


def test_motif_score_db_direct_moods_scan(tmp_path):
    pytest.importorskip("MOODS")
    pytest.importorskip("pyfaidx")

    class _M:  # minimal JASPAR-like motif: counts per base + length
        def __init__(self, name, seq):
            self.matrix_id, self.name, self.length = name, name, len(seq)
            self.counts = {b: [10.0 if c == b else 0.1 for c in seq] for b in "ACGT"}

    rng = np.random.default_rng(1)
    bg = "".join(rng.choice(list("ACGT"), 3000))
    seq = bg[:500] + "GATAAGGATAAG" + bg[512:2000] + "CACGTGCACGTG" + bg[2012:]
    (tmp_path / "g.fa").write_text(">chr1\n" + "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")
    regions = ["chr1:450-650", "chr1:1950-2150", "chr1:2500-2700"]
    S = sc.motif_score_db(regions, genome_fasta=tmp_path / "g.fa",
                          motifs=[_M("GATA", "GATAAG"), _M("EBOX", "CACGTG")],
                          p_value=1e-3, verbose=False)
    names, v = S.load(regions)
    v = pd.DataFrame(v, index=names, columns=S.motifs).loc[regions]
    assert v.loc["chr1:450-650", "GATA_GATA"] > 0 and v.loc["chr1:1950-2150", "EBOX_EBOX"] > 0
    assert v.loc["chr1:450-650", "GATA_GATA"] > v.loc["chr1:2500-2700", "GATA_GATA"]


def test_ranksums_match_average_rank_definition():
    import math
    from scipy.stats import rankdata
    from epione.scenic._dem import _mean_axis1
    rng = np.random.default_rng(7)
    fg = np.round(rng.gamma(1.0, 2.0, (40, 900)), 1).astype(np.float32)   # many ties
    bg = np.round(rng.gamma(1.0, 1.8, (40, 120)), 1).astype(np.float32)
    fg[:, :30] = 0
    bg[:, :10] = 0
    n1, n2 = fg.shape[1], bg.shape[1]
    ref = []
    for i in range(fg.shape[0]):
        s = rankdata(np.concatenate([fg[i], bg[i]]))[:n1].sum()     # exact half-integers
        z = (s - n1 * (n1 + n2 + 1) / 2.0) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        ref.append(2 * (1.0 + math.erf(-abs(z) / math.sqrt(2.0))) / 2.0)
    st = sc.dem_test(fg, bg)
    np.testing.assert_allclose(st["pval"].to_numpy(), ref, rtol=1e-14, atol=0)
    acc = np.zeros(fg.shape[0])
    for j in range(n1):                     # float64, region order
        acc += fg[:, j]
    np.testing.assert_array_equal(st["Mean_fg"].to_numpy(), acc / n1)
    np.testing.assert_array_equal(_mean_axis1(fg), acc / n1)
    st2 = sc.dem_test(fg.T, bg.T, regions_by_motifs=True)
    pd.testing.assert_frame_equal(st, st2)
