"""Tests for the HiCExplorer-method paths of ``epione.bulk.hic``.

Synthetic matrices only (fast, CPU, no downloads). Parity with HiCExplorer
3.7.6 itself is checked by ``tests/parity/hic_tools/parity_check.py``; the
opt-in test at the bottom runs part of it when ``EPIONE_HIC_PARITY_DIR`` is set.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cooler")


BIN = 20_000
SIZE = 4_000_000
EDGES = [0, 1_300_000, 2_600_000, SIZE]


def _dense(seed=0, boost_block=None, loop=None):
    """Distance-decay matrix with three TAD blocks; optional boost / loop."""
    rng = np.random.default_rng(seed)
    n = SIZE // BIN
    i, j = np.indices((n, n))
    d = np.abs(i - j)
    lam = 400.0 / (d + 1.0) ** 1.1
    blk = np.searchsorted(np.array(EDGES[1:]) // BIN, np.arange(n), side="right")
    same = blk[:, None] == blk[None, :]
    lam = np.where(same, lam * 3.0, lam)
    if boost_block is not None:
        b = (blk[:, None] == boost_block) & (blk[None, :] == boost_block)
        lam = np.where(b, lam * 2.0, lam)
    if loop is not None:
        a, c = loop
        lam[a - 1:a + 2, c - 1:c + 2] += 60.0
        lam[c - 1:c + 2, a - 1:a + 2] += 60.0
    m = rng.poisson(lam).astype(float)
    m = np.triu(m) + np.triu(m, 1).T
    return m


def _write_cool(path, dense, chrom="chrT", weight=True):
    import cooler

    n = dense.shape[0]
    bins = pd.DataFrame({"chrom": chrom, "start": np.arange(n) * BIN,
                         "end": np.minimum((np.arange(n) + 1) * BIN, SIZE)})
    if weight:
        bins["weight"] = 1.0
    r, c = np.nonzero(np.triu(dense))
    pixels = pd.DataFrame({"bin1_id": r, "bin2_id": c, "count": dense[r, c].astype(np.int32)})
    cooler.create_cooler(str(path), bins=bins, pixels=pixels, ordered=True)
    return str(path)


@pytest.fixture(scope="module")
def tads_cool(tmp_path_factory):
    p = tmp_path_factory.mktemp("hicx") / "tads.cool"
    return _write_cool(p, _dense(seed=1))


def test_incremental_steps_formula():
    from epione.bulk.hic._tads_hicexplorer import incremental_steps
    # min + int(step * k ** 1.5) up to max. (hicFindTADs --help lists
    # 20k, 30k, 40k, 70k, 100k for these values; the code gives this.)
    assert incremental_steps(20_000, 150_000, 10_000) == [20_000, 30_000, 48_284, 71_961, 100_000, 131_803]


def test_enlarge_bins_docstring_example():
    from epione.bulk.hic._hicmatrix import enlarge_bins
    iv = [("chr1", 10, 50, 1), ("chr1", 50, 80, 2), ("chr2", 10, 60, 3), ("chr2", 70, 90, 4)]
    assert enlarge_bins(iv) == [("chr1", 0, 50, 1), ("chr1", 50, 80, 2), ("chr2", 0, 65, 3), ("chr2", 65, 90, 4)]


def test_tad_separation_score_and_boundaries(tads_cool):
    import epione.bulk.hic as h

    tab = h.insulation(tads_cool, method="hicexplorer", min_depth=60_000, max_depth=200_000, step=40_000)
    win = [c for c in tab.columns if c.startswith("tad_separation_score_")]
    assert len(win) >= 2 and "tad_separation_score" in tab.columns
    assert np.allclose(tab["tad_separation_score"], tab[win].mean(axis=1))
    # the score dips at the two block edges
    for edge in EDGES[1:3]:
        near = tab[(tab["start"] - edge).abs() <= 2 * BIN]["tad_separation_score"].min()
        assert near < tab["tad_separation_score"].median()
    b = h.tad_boundaries(tab, threshold_comparisons=0.05)
    assert {"chrom", "start", "end", "pvalue", "delta"} <= set(b.columns)
    hits = [np.min(np.abs(b["end"].to_numpy() - e)) for e in EDGES[1:3]]
    assert max(hits) <= 3 * BIN
    d = h.tad_domains(b)
    assert (d["end"] > d["start"]).all()


def test_tad_boundaries_recomputes_when_cache_is_cleared(tads_cool):
    import epione.bulk.hic as h
    from epione.bulk.hic import _insulation

    tab = h.insulation(tads_cool, method="hicexplorer", min_depth=60_000, max_depth=200_000, step=40_000)
    b1 = h.tad_boundaries(tab)
    _insulation._HICEXPLORER_CACHE.clear()
    b2 = h.tad_boundaries(tab)
    pd.testing.assert_frame_equal(b1, b2)


def test_compartments_hicexplorer_shapes_and_gene_phasing(tmp_path, tads_cool):
    import epione.bulk.hic as h

    bed = tmp_path / "genes.bed"
    # genes concentrated in the first block -> E1 should be positive there
    with open(bed, "w") as fh:
        for s in range(20_000, 1_200_000, 15_000):
            fh.write(f"chrT\t{s}\t{s + 1000}\n")
    for oe in ("dist_norm", "lieberman"):
        df = h.compartments(tads_cool, method="hicexplorer", n_eigs=2, obs_exp=oe, extra_track=str(bed))
        assert list(df.columns) == ["chrom", "start", "end", "E1", "E2"]
        assert len(df) == SIZE // BIN
        assert np.isfinite(df["E1"]).all()
    with pytest.raises(ValueError):
        h.compartments(tads_cool, method="hicexplorer", fasta_path="x.fa")
    with pytest.raises(ValueError):
        h.compartments(tads_cool, method="nope")


def test_compare_and_correlate(tmp_path):
    import epione.bulk.hic as h

    a = _write_cool(tmp_path / "a.cool", _dense(seed=2))
    b = _write_cool(tmp_path / "b.cool", _dense(seed=3))
    m, bins = h.compare_matrices(a, a, operation="log2ratio")
    assert np.allclose(m.data, 0)
    m, bins = h.compare_matrices(a, b, operation="diff", normalize=False, out_path=tmp_path / "d.cool")
    assert (tmp_path / "d.cool").exists() and len(bins) == SIZE // BIN
    r = h.correlate_matrices([a, a, b], method="pearson")
    assert r.shape == (3, 3) and np.isclose(r.iloc[0, 1], 1.0) and r.iloc[0, 2] < 1.0
    r2 = h.correlate_matrices([a, b], method="spearman", log1p=True, distance_range=(40_000, 1_000_000))
    assert np.isclose(r2.iloc[0, 1], r2.iloc[1, 0])
    with pytest.raises(ValueError):
        h.compare_matrices(a, b, operation="sum")


def test_differential_tads_detects_boosted_block(tmp_path):
    import epione.bulk.hic as h

    ctl = _write_cool(tmp_path / "ctl.cool", _dense(seed=4))
    tgt = _write_cool(tmp_path / "tgt.cool", _dense(seed=5, boost_block=1))
    dom = pd.DataFrame({"chrom": "chrT", "start": EDGES[:-1], "end": EDGES[1:]})
    res = h.differential_tads(tgt, ctl, dom, mode="intra-TAD")
    assert len(res) == 3
    assert bool(res.loc[1, "rejected"]) and res.loc[1, "pvalue_intra"] < 1e-6
    assert np.isnan(res.loc[0, "pvalue_left_inter"]) and np.isnan(res.loc[2, "pvalue_right_inter"])


def test_loops_hicexplorer_finds_injected_dot(tmp_path):
    import epione.bulk.hic as h

    a, c = 40, 90
    cool = _write_cool(tmp_path / "loop.cool", _dense(seed=6, loop=(a, c)))
    loops = h.loops(cool, method="hicexplorer", max_loop_distance=3_000_000)
    assert list(loops.columns) == ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "pvalue"]
    d = np.abs(loops["start1"] // BIN - a) + np.abs(loops["start2"] // BIN - c)
    assert len(loops) > 0 and d.min() <= 2
    with pytest.raises(ValueError):
        h.loops(cool, method="hicexplorer", window_size=2, peak_width=2)


def test_h5_to_cool_roundtrip(tmp_path):
    tables = pytest.importorskip("tables")
    import cooler
    import epione.bulk.hic as h
    from scipy.sparse import csr_matrix, triu

    dense = _dense(seed=7)[:50, :50]
    cf = np.linspace(0.5, 1.5, 50)
    cf[3] = np.nan
    corrected = dense * np.outer(1 / np.nan_to_num(cf, nan=1), 1 / np.nan_to_num(cf, nan=1))
    corrected[3, :] = 0
    corrected[:, 3] = 0
    m = csr_matrix(triu(csr_matrix(corrected)))
    h5 = tmp_path / "m.h5"
    with tables.open_file(str(h5), "w") as f:
        g = f.create_group("/", "matrix")
        for k, v in (("data", m.data), ("indices", m.indices), ("indptr", m.indptr),
                     ("shape", np.array(m.shape))):
            f.create_carray(g, k, obj=v)
        g = f.create_group("/", "intervals")
        f.create_carray(g, "chr_list", obj=np.array([b"chrT"] * 50))
        f.create_carray(g, "start_list", obj=np.arange(50) * BIN)
        f.create_carray(g, "end_list", obj=(np.arange(50) + 1) * BIN)
        f.create_carray(g, "extra_list", obj=np.ones(50))
        f.create_carray("/", "nan_bins", obj=np.array([3]))
        f.create_carray("/", "correction_factors", obj=cf)
    out = h.h5_to_cool(h5, tmp_path / "m.cool")
    clr = cooler.Cooler(out)
    bal = clr.matrix(balance=True, sparse=True)[:].toarray()
    ref = np.triu(corrected)
    ok = np.isfinite(bal) & (ref != 0)
    assert np.allclose(bal[ok], ref[ok], rtol=1e-12)


def test_methods_validate():
    import epione.bulk.hic as h
    with pytest.raises(ValueError):
        h.insulation("x.cool", method="nope")
    with pytest.raises(ValueError):
        h.loops("x.cool", method="nope")


@pytest.mark.skipif(not os.environ.get("EPIONE_HIC_PARITY_DIR"), reason="needs EPIONE_HIC_PARITY_DIR")
def test_parity_tads_and_conversion(tmp_path):
    pytest.importorskip("hicexplorer")
    root = Path(os.environ["EPIONE_HIC_PARITY_DIR"])
    out = tmp_path / "r.json"
    script = Path(__file__).parent / "parity" / "hic_tools" / "parity_check.py"
    subprocess.run([sys.executable, str(script), "--data", str(root / "data"),
                    "--hicexplorer-src", str(root / "HiCExplorer"), "--out", str(out),
                    "--work", str(tmp_path / "w"), "--gates", "B0,B1"], check=True)
    import json
    res = json.loads(out.read_text())
    failed = [k for k, v in res.items() if isinstance(v, dict) and v.get("pass_") is False]
    assert not failed, failed
