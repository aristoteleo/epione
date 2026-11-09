"""Tests for the Cicero port (epione.tl.*cicero*, epione.tl.glasso).

Reference values: R 4.3.3, cicero 1.3.9 (Monocle 3 branch), monocle3 1.3.1,
glasso 1.11, on Cicero's example data ``cicero_data`` (fixture F1), seed 42;
packed by tests/parity/cicero/pack_small_reference.py. Gates and thresholds are
the pre-registered ones of tests/parity/cicero/MANIFEST.md.

The full-size gates (F1 all rows, F2 = E8.5 heart chr19) run when
``EPIONE_CICERO_PARITY_DIR`` points at the fixture/reference dumps
(layout: fixtures/F1, ref/F1_seed42, fixtures/F2, ref/F2_seed42).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import epione as epi
from epione.tl import _cicero as C
from epione.tl._rrng import RRandom

HERE = Path(__file__).parent / "parity" / "cicero"
REF = HERE / "reference" / "F1_cicero_data_seed42.npz"
SEED = 42


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REF, allow_pickle=False))


@pytest.fixture(scope="module")
def f1(ref):
    n_peaks, n_cells = ref["shape"]
    m = sp.csc_matrix((ref["x_x"], ref["x_i"], ref["x_p"]), shape=(n_peaks, n_cells))
    ad = AnnData(X=m.T.tocsr(), obs=pd.DataFrame(index=ref["cells"]), var=pd.DataFrame(index=ref["peaks"]))
    ad.obsm["X_umap"] = ref["coords"]
    genome = pd.DataFrame({0: ref["genome_chr"], 1: ref["genome_len"]})
    return ad, genome


@pytest.fixture(scope="module")
def cicero_cds(f1):
    ad, _ = f1
    return epi.tl.make_cicero_cds(ad, reduced_coordinates="X_umap", random_state=SEED, silent=True)


@pytest.fixture(scope="module")
def conns(cicero_cds, f1):
    _, genome = f1
    return epi.tl.run_cicero(cicero_cds, genome, random_state=SEED, silent=True)


# ---------------------------------------------------------------- glasso (gate 4)
def test_glasso_matches_r_synthetic(ref):
    for n, off in zip(ref["syn_n"], ref["syn_off"]):
        sl = slice(off, off + n * n)
        S = ref["syn_S"][sl].reshape(n, n, order="F")
        R = ref["syn_rho"][sl].reshape(n, n, order="F")
        g = epi.tl.glasso(S, R)
        W = ref["syn_w"][sl].reshape(n, n, order="F")
        WI = ref["syn_wi"][sl].reshape(n, n, order="F")
        assert np.abs(g.w - W).max() <= 1e-8
        assert np.abs(g.wi - WI).max() <= 1e-8
        assert ((g.wi == 0) == (WI == 0)).all()


def test_glasso_scalar_and_vector_rho():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    S = np.cov(X, rowvar=False)
    a = epi.tl.glasso(S, 0.1)
    b = epi.tl.glasso(S, np.full((6, 6), 0.1))
    np.testing.assert_array_equal(a.w, b.w)
    # zero penalty off the diagonal -> w reproduces S off-diagonal
    c = epi.tl.glasso(S, np.zeros((6, 6)))
    assert np.abs(c.w - S).max() < 1e-3
    assert np.allclose(c.w @ c.wi, np.eye(6), atol=1e-2)


# ---------------------------------------------------------------- R numerics helpers
def test_r_cov_is_sequential_long_double():
    rng = np.random.default_rng(1)
    v = rng.poisson(1.0, size=(700, 9)).astype(float) / rng.uniform(0.5, 2, size=(700, 1))
    got = C._r_cov(v)
    # naive loop reference (double products, long-double running sums)
    n = v.shape[0]
    vl = v.astype(np.longdouble)
    m = vl.sum(0) / n
    s = np.zeros(v.shape[1], dtype=np.longdouble)
    for k in range(n):
        s += vl[k] - m
    xm = (m + s / n).astype(float)
    d = v - xm
    acc = np.zeros((9, 9), dtype=np.longdouble)
    for k in range(n):
        acc += np.outer(d[k], d[k])
    np.testing.assert_array_equal(got, (acc / (n - 1)).astype(float))
    assert np.allclose(got, np.cov(v, rowvar=False), rtol=1e-12)


def test_generate_windows_matches_r_seq():
    w = epi.tl._cicero.generate_windows(500000, pd.DataFrame({0: ["chr2", "chr10"], 1: [1_000_000, 750_001]}))
    assert w["chrom"].tolist() == ["chr10"] * 4 + ["chr2"] * 4      # ddply: sorted chromosome names
    assert w["start"].tolist()[:4] == [1, 250001, 500001, 750001]
    assert w["name"].iloc[0] == "chr10_1_500000"


# ---------------------------------------------------------------- make_cicero_cds (gates 1, 3)
def test_make_cicero_cds_matches_r_bins(cicero_cds, ref):
    r_chosen = np.array([int(s.split("_")[1]) - 1 for s in ref["r_agg_cells"]])
    np.testing.assert_array_equal(cicero_cds.uns["cicero"]["chosen"], r_chosen)
    assert list(cicero_cds.obs_names) == list(ref["r_agg_cells"])
    assert {"num_genes_expressed", "Size_Factor"} <= set(cicero_cds.obs.columns)
    assert {"chr", "bp1", "bp2", "num_cells_expressed"} <= set(cicero_cds.var.columns)


def test_make_cicero_cds_injected_bins_and_agg_info(f1, ref, cicero_cds):
    ad, _ = f1
    r_chosen = np.array([int(s.split("_")[1]) - 1 for s in ref["r_agg_cells"]])
    inj, agg_map = epi.tl.make_cicero_cds(ad, reduced_coordinates="X_umap", chosen=r_chosen, silent=True,
                                          return_agg_info=True)
    assert (abs(inj.X - cicero_cds.X)).max() == 0
    assert list(agg_map.columns) == ["cell", "agg_cell"]
    assert len(agg_map) == 50 * len(r_chosen)


def test_make_cicero_cds_numpy_rng_runs(f1):
    ad, _ = f1
    c = epi.tl.make_cicero_cds(ad, reduced_coordinates="X_umap", rng="numpy", random_state=1, silent=True)
    assert c.n_obs > 1 and c.n_vars == ad.n_vars


# ---------------------------------------------------------------- run_cicero (gates 1, 6, 8, 9, 13)
def test_run_cicero_matches_r(conns, ref):
    assert conns.attrs["mean_distance_parameter"] == ref["r_mean_dp"][0]          # bit-identical
    np.testing.assert_allclose(conns.attrs["distance_parameters"], ref["r_dps"], rtol=1e-6, atol=0)
    assert len(conns) == ref["r_n_conns"][0]
    sub = conns.iloc[ref["r_conn_rows"]]                                         # R's row order
    assert (sub["Peak1"].astype(str).to_numpy() == ref["r_conn_p1"]).all()
    assert (sub["Peak2"].astype(str).to_numpy() == ref["r_conn_p2"]).all()
    a, b = sub["coaccess"].to_numpy(), ref["r_conn_co"]
    assert (np.isnan(a) == np.isnan(b)).all()
    ok = ~np.isnan(a)
    assert np.abs(a[ok] - b[ok]).max() <= 1e-8


def test_estimate_distance_parameter_injected_windows(cicero_cds, f1, ref):
    _, genome = f1
    dps = epi.tl.estimate_distance_parameter(cicero_cds, genomic_coords=genome,
                                             sample_windows=ref["r_sampled_win"].tolist())
    np.testing.assert_allclose(dps, ref["r_dps"], rtol=1e-6, atol=0)


def test_rrandom_stream_continues_like_one_r_session(f1):
    ad, genome = f1
    # one set.seed() for the whole session == passing one RRandom object along
    r = RRandom(SEED)
    c1 = epi.tl.make_cicero_cds(ad, reduced_coordinates="X_umap", random_state=r, silent=True)
    c2 = epi.tl.make_cicero_cds(ad, reduced_coordinates="X_umap", random_state=SEED, silent=True)
    np.testing.assert_array_equal(c1.uns["cicero"]["chosen"], c2.uns["cicero"]["chosen"])
    assert not np.array_equal(r.state(), RRandom(SEED).state())     # the stream advanced in place


def test_celloracle_format(conns, tmp_path):
    out = epi.tl.cicero_connections_to_celloracle(conns, path=tmp_path / "cicero_connections.csv")
    assert list(out.columns) == ["Peak1", "Peak2", "coaccess"]
    assert out["Peak1"].str.match(r"^chr\w+_\d+_\d+$").all()
    back = pd.read_csv(tmp_path / "cicero_connections.csv", index_col=0)
    assert list(back.columns) == ["Peak1", "Peak2", "coaccess"] and len(back) == len(conns)
    # chr:start-end / chr-start-end ids are normalised
    t = pd.DataFrame({"Peak1": ["chr1:10-20"], "Peak2": ["chr1-30-40"], "coaccess": [0.5]})
    o = epi.tl.cicero_connections_to_celloracle(t)
    assert o.iloc[0].tolist() == ["chr1_10_20", "chr1_30_40", 0.5]


# ---------------------------------------------------------------- CCANs (gates 16, 17)
def test_generate_ccans(conns, ref):
    cc = epi.tl.generate_ccans(conns, random_state=SEED, silent=True)
    assert cc.attrs["coaccess_cutoff"] == ref["r_ccan_cutoff"][0]              # 0.46, as R
    assert list(cc.columns) == ["Peak", "CCAN"]
    assert (cc.groupby("CCAN").size() > 2).all()
    ov = epi.tl.find_overlapping_ccans(cc)
    assert list(ov.columns) == ["CCAN1", "CCAN2"]
    assert (ov["CCAN1"] != ov["CCAN2"]).all()


# ---------------------------------------------------------------- gene activity (gates 14, 15)
def test_gene_activity_matches_r(f1, conns, ref):
    ad, _ = f1
    ad = ad.copy()
    prom = pd.DataFrame({"chromosome": ref["prom_chromosome"], "start": ref["prom_start"],
                         "end": ref["prom_end"], "gene": ref["prom_gene"]})
    epi.tl.annotate_cds_by_site(ad, prom)
    assert (ad.var["gene"].fillna("").astype(str).to_numpy() == ref["r_ann_gene"]).all()
    ga = epi.tl.build_gene_activity_matrix(ad, conns)
    R1 = pd.DataFrame(ref["r_unnorm_ga"], index=ref["r_unnorm_ga_genes"], columns=ref["r_unnorm_ga_cells"])
    P1 = pd.DataFrame(ga.X.T.toarray(), index=ga.var_names, columns=ga.obs_names)
    assert np.abs(P1.loc[R1.index, R1.columns].to_numpy() - R1.to_numpy()).max() <= 1e-8
    keep_g = np.asarray(ga.X.sum(0)).ravel() != 0
    keep_c = np.asarray(ga.X.sum(1)).ravel() != 0
    num_genes = pd.Series(np.asarray((ad.X > 0).sum(1)).ravel(), index=ad.obs_names)
    gn = epi.tl.normalize_gene_activities(ga[keep_c][:, keep_g].copy(), num_genes)
    R2 = pd.DataFrame(ref["r_ga"], index=ref["r_ga_genes"], columns=ref["r_ga_cells"])
    P2 = pd.DataFrame(gn.X.T.toarray(), index=gn.var_names, columns=gn.obs_names)
    assert np.abs(P2.loc[R2.index, R2.columns].to_numpy() - R2.to_numpy()).max() / np.abs(R2.to_numpy()).max() <= 1e-8
    zero = AnnData(X=sp.hstack([ga.X, sp.csr_matrix((ga.n_obs, 1))]).tocsr(), obs=ga.obs.copy(),
                   var=pd.DataFrame(index=list(ga.var_names) + ["ZERO"]))
    with pytest.raises(ValueError):
        epi.tl.normalize_gene_activities(zero, num_genes)        # all-zero gene: R errors too


# ---------------------------------------------------------------- full-size gates from dumps
PARITY_DIR = os.environ.get("EPIONE_CICERO_PARITY_DIR")


@pytest.mark.skipif(PARITY_DIR is None, reason="set EPIONE_CICERO_PARITY_DIR to the R dumps")
@pytest.mark.parametrize("fx", ["F1", "F2"])
def test_full_parity_gates(fx, tmp_path):
    sys.path.insert(0, str(HERE))
    import parity_check
    root = Path(PARITY_DIR)
    if not (root / "ref" / f"{fx}_seed42").exists():
        pytest.skip(f"no reference dump for {fx}")
    res = parity_check.main([str(root / "fixtures" / fx), str(root / "ref" / f"{fx}_seed42"),
                             str(tmp_path / "res.json"), "--threads", "4"])
    deterministic = [g for g in res["gates"] if g not in ("16", "17", "1a_bins", "1b_windows")]
    failed = [g for g in deterministic if not res["gates"][g].get("pass", True)]
    assert not failed, {g: res["gates"][g] for g in failed}
