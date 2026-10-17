"""Synthetic tests for CellScan (:func:`epione.single.atac.scan_peaks`).

The fixture plants (a) open sites shared by many cells and (b) "jackpot"
sites where two very deep cells deposit as many insertions as a real site
gets from all cells, on top of uniform background. The distinct-cell
statistic must call (a) and not (b); the pseudobulk-style insertion
statistic calls both, which is the failure mode CellScan addresses. Also
checked: the Poisson tail kernel, g(m) saturation, summit placement, the
Tn5 bias model recovering a planted k-mer preference, and groupby.

Run with:  pytest tests/test_scan_peaks.py -q
"""
import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba")
pytest.importorskip("pyarrow")

import epione  # noqa: E402,F401
import epione.utils  # noqa: E402,F401  (import order)
from epione.single.atac import _scan_peaks as sp_mod  # noqa: E402
from epione.single.atac import scan_peaks  # noqa: E402

SIZE = 400_000


def _write(tmp_path, rows, name="frags.tsv"):
    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "bc", "n"])
    df = df.sort_values(["chrom", "start"])
    p = tmp_path / name
    df.to_csv(p, sep="\t", header=False, index=False)
    return str(p)


def _fixture(rng, n_cells=300, bg_per_cell=150, frac_open=0.35):
    cells = [f"C{i:04d}" for i in range(n_cells)]
    rows = []
    real = np.arange(20_000, SIZE - 20_000, 25_000)[:12]
    jackpot = real + 12_500
    for i, bc in enumerate(cells):
        for s in rng.integers(0, SIZE - 500, bg_per_cell):
            rows.append(("chr1", int(s), int(s + rng.integers(60, 400)), bc, 1))
        for c in real:
            if rng.random() < frac_open:
                s = int(c + rng.normal(0, 40))
                rows.append(("chr1", s - 30, s + int(rng.integers(40, 120)), bc, 1))
    # two deep cells: same total insertions at each jackpot as a real site
    n_real = int(n_cells * frac_open)
    for c in jackpot:
        for j in range(n_real):
            bc = cells[j % 2]
            s = int(c + rng.integers(-150, 150))
            rows.append(("chr1", s, s + int(rng.integers(40, 300)), bc, 1))
    return cells, rows, real, jackpot


def _hits(peaks, sites, tol=300):
    if len(peaks) == 0:
        return np.zeros(len(sites), bool)
    s = peaks["summit"].to_numpy()
    return np.array([np.any(np.abs(s - c) <= tol) for c in sites])


def test_poisson_tail_matches_scipy():
    st = pytest.importorskip("scipy.stats")
    k = sp_mod._kernels()
    for kk, lam in [(5, 1.0), (30, 10.0), (200, 150.0), (3, 0.01)]:
        ref = st.poisson.logsf(kk - 1, lam)
        assert math.isclose(k["poisson_logsf"](kk, lam), ref, rel_tol=1e-6, abs_tol=1e-9)


def test_g_table_saturation():
    depths = np.array([1, 10, 100, 1000.0])
    lg, g = sp_mod._g_table(depths)
    m = np.exp(lg)
    # small m: g(m) ~ m (each insertion from a new cell); large m: g -> n_cells
    i = np.searchsorted(m, 1e-3)
    assert math.isclose(g[i], m[i], rel_tol=1e-2)
    assert g[-1] == pytest.approx(4.0, rel=1e-3)
    exact = (1 - np.exp(-5.0 * depths / depths.sum())).sum()
    j = np.searchsorted(m, 5.0)
    assert g[j] == pytest.approx(exact, rel=2e-2)


def test_cells_beat_jackpots(tmp_path):
    rng = np.random.default_rng(0)
    cells, rows, real, jackpot = _fixture(rng)
    frag = _write(tmp_path, rows)
    kw = dict(barcodes=cells, chrom_sizes={"chr1": SIZE}, kmer=None, n_jobs=2,
              inplace=False, verbose=False)
    pk = scan_peaks(frag, **kw)
    assert _hits(pk, real).all()
    assert not _hits(pk, jackpot).any()
    # the pseudobulk-style statistic is fooled by the deep cells
    pk_ins = scan_peaks(frag, statistic="insertions", **kw)
    assert _hits(pk_ins, real).all()
    assert _hits(pk_ins, jackpot).sum() >= len(jackpot) // 2
    # summits land on the planted centres and peaks are fixed width
    s = pk["summit"].to_numpy()
    err = [np.min(np.abs(s - c)) for c in real]
    assert np.median(err) <= 30
    assert ((pk["end"] - pk["start"]) == 500).all()


def test_groupby_and_return_all(tmp_path):
    rng = np.random.default_rng(1)
    cells, rows, real, _ = _fixture(rng, n_cells=200)
    # group B gets an extra private site
    extra = 395_000 - 2_000
    labels = np.array(["A"] * 100 + ["B"] * 100)
    for i in range(100, 200):
        if rng.random() < 0.5:
            s = int(extra + rng.normal(0, 30))
            rows.append(("chr1", s, s + 80, cells[i], 1))
    frag = _write(tmp_path, rows)
    pk = scan_peaks(frag, barcodes=cells, groupby=labels, chrom_sizes={"chr1": SIZE},
                    kmer=None, n_jobs=2, inplace=False, verbose=False)
    assert _hits(pk, [extra]).all()
    assert set(pk.loc[_hits_mask(pk, extra), "group"]) == {"B"}
    allp = scan_peaks(frag, barcodes=cells, chrom_sizes={"chr1": SIZE}, kmer=None,
                      n_jobs=2, inplace=False, verbose=False, return_all=True)
    assert len(allp) >= len(scan_peaks(frag, barcodes=cells, chrom_sizes={"chr1": SIZE},
                                       kmer=None, n_jobs=2, inplace=False, verbose=False))


def _hits_mask(pk, c, tol=300):
    return (pk["summit"] - c).abs() <= tol


def test_bias_model_recovers_planted_preference(tmp_path):
    """Background cuts prefer a k-mer centre 'G'; the fitted ratios must
    rank G-centred 3-mers above the rest, and no peaks are called in pure
    background."""
    rng = np.random.default_rng(2)
    seq = rng.choice(list("ACGT"), SIZE)
    fa = tmp_path / "g.fa"
    with open(fa, "w") as fh:
        fh.write(">chr1\n")
        s = "".join(seq)
        for i in range(0, SIZE, 60):
            fh.write(s[i:i + 60] + "\n")
    import pysam
    pysam.faidx(str(fa))
    g_pos = np.flatnonzero(seq == "G")
    o_pos = np.flatnonzero(seq != "G")
    cells = [f"C{i:03d}" for i in range(200)]
    rows = []
    for bc in cells:
        n = 200
        # 75% of fragment starts and (end-1)s at G
        st = np.where(rng.random(n) < 0.75, rng.choice(g_pos, n), rng.choice(o_pos, n))
        ln = rng.integers(80, 300, n)
        for a, L in zip(st, ln):
            if a + L < SIZE:
                rows.append(("chr1", int(a), int(a + L), bc, 1))
    frag = _write(tmp_path, rows)
    k = sp_mod._kernels()
    raw = np.frombuffer("".join(seq).encode(), np.uint8)
    enc = k["encode_seq"](raw)
    kt = k["kmer_track"](enc, 3, 1)
    df = pd.read_csv(frag, sep="\t", header=None)
    pos = df[1].to_numpy()
    cs, ce = k["count_kmers"](kt, pos, np.zeros(len(pos), np.uint8),
                              np.zeros(0, np.int64), np.zeros(0, np.int64), 64)
    cg = k["count_genome_kmers"](kt, np.zeros(0, np.int64), np.zeros(0, np.int64), 64)
    ratio = (cs + 1) / (cs.sum() + 64) / ((cg + 1) / (cg.sum() + 64))
    centre_g = np.array([(code >> 2) & 3 == 2 for code in range(64)])
    assert ratio[centre_g].min() > ratio[~centre_g].max()
    pk = scan_peaks(frag, barcodes=cells, fasta=str(fa), kmer=3, n_jobs=2,
                    inplace=False, verbose=False)
    assert len(pk) <= 2


def test_variable_window_dip_and_nb(tmp_path):
    rng = np.random.default_rng(3)
    cells, rows, real, jackpot = _fixture(rng)
    # a broad (~1.5 kb) site with two shoulders: dip=0 must report one summit
    broad = 395_000 - 3_000
    for bc in cells:
        if rng.random() < 0.6:
            for _ in range(3):
                s = int(broad + rng.uniform(-700, 700))
                rows.append(("chr1", s, s + 60, bc, 1))
    frag = _write(tmp_path, rows)
    kw = dict(barcodes=cells, chrom_sizes={"chr1": SIZE}, kmer=None, n_jobs=2,
              inplace=False, verbose=False)
    ms = scan_peaks(frag, window=[100, 200, 400], **kw)
    assert _hits(ms, real).all() and not _hits(ms, jackpot).any()
    one = scan_peaks(frag, dip=0.0, return_all=True, **kw)
    many = scan_peaks(frag, dip=1.0, return_all=True, **kw)
    n_one = int(((one.summit - broad).abs() <= 900).sum())
    n_many = int(((many.summit - broad).abs() <= 900).sum())
    assert n_one == 1 and n_many >= 2
    nb = scan_peaks(frag, null="nb", **kw)
    assert _hits(nb, real).all()
    # the NB tail is never more significant than the Poisson tail
    assert np.all(sp_mod._nb_score(np.array([20, 50]), np.array([5.0, 5.0]), 0.3)
                  <= -np.array([sp_mod._kernels()["poisson_logsf"](20, 5.0),
                                sp_mod._kernels()["poisson_logsf"](50, 5.0)]) / np.log(10) + 1e-9)


# ---------------------------------------------------------------------------
# call_peaks: the default entry point (CellScan) and the MACS3 path
# ---------------------------------------------------------------------------

def _adata(tmp_path, cells, rows, labels=None):
    import anndata as ad
    import pysam
    frag = _write(tmp_path, rows)
    pysam.tabix_compress(frag, frag + ".gz", force=True)
    pysam.tabix_index(frag + ".gz", preset="bed", force=True, zerobased=True)
    a = ad.AnnData(obs=pd.DataFrame(index=cells))
    if labels is not None:
        a.obs["grp"] = labels
    a.uns["files"] = {"fragments": frag + ".gz"}
    return a


def test_call_peaks_default_is_cellscan_and_feeds_peak_matrix(tmp_path):
    import epione as epi
    from epione.single.atac import call_peaks
    rng = np.random.default_rng(4)
    cells, rows, real, jackpot = _fixture(rng, n_cells=200)
    labels = np.array(["A", "B"] * 100)
    a = _adata(tmp_path, cells, rows, labels)
    call_peaks(a, groupby="grp", chrom_sizes={"chr1": SIZE}, kmer=None, n_jobs=2, verbose=False)
    pk = a.uns["peaks"]
    direct = scan_peaks(a, groupby="grp", chrom_sizes={"chr1": SIZE}, kmer=None, n_jobs=2,
                        inplace=False, verbose=False)
    assert list(pk.columns[:4]) == ["chrom", "start", "end", "Peaks"]
    assert (pk["start"].to_numpy() == direct["start"].to_numpy()).all()
    assert (pk["Peaks"] == pk.chrom + ":" + pk.start.astype(str) + "-" + pk.end.astype(str)).all()
    assert _hits(pk, real).all() and not _hits(pk, jackpot).any()
    # downstream: the cell x peak matrix accepts the CellScan peak set as is
    pm = epi.pp.make_peak_matrix(a, use_rep=pk["Peaks"], verbose=False)
    assert pm.shape == (len(cells), len(pk))
    assert np.asarray(pm.X.sum(0)).ravel().min() > 0
    with pytest.raises(ValueError):
        call_peaks(a, method="nope", chrom_sizes={"chr1": SIZE})


def test_call_peaks_macs3_path_is_unchanged(tmp_path):
    import os
    import shutil
    import sys
    macs = os.path.join(os.path.dirname(sys.executable), "macs3")
    if not (os.path.exists(macs) or shutil.which("macs3")):
        pytest.skip("macs3 not installed")
    from epione.single.atac import call_peaks, macs3, merge_peaks
    rng = np.random.default_rng(5)
    cells, rows, real, _ = _fixture(rng, n_cells=200)
    a = _adata(tmp_path, cells, rows)
    got = call_peaks(a, method="macs3", chrom_sizes={"chr1": SIZE}, gsize=SIZE, inplace=False,
                     verbose=False)
    res = macs3(a, gsize=SIZE, inplace=False, verbose=False)
    ref = merge_peaks({"all": res}, {"chr1": SIZE}, half_width=250)
    pd.testing.assert_frame_equal(got.reset_index(drop=True), ref.reset_index(drop=True))
    assert _hits(got.assign(summit=(got.start + got.end) // 2), real).all()
