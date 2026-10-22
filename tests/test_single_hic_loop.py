"""Loops: loop_anova equals a one-way ANOVA on per-cell values; merging, values, selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cooler = pytest.importorskip("cooler")
h5py = pytest.importorskip("h5py")
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.stats import f_oneway  # noqa: E402

from epione.single import hic  # noqa: E402

RES = 10_000
SIZES = pd.Series({"chrA": 600_000, "chrB": 405_000})
N_CELLS = {"g1": 6, "g2": 9, "g3": 5}
BAND = 12          # pixels stored up to 12 bins from the diagonal


def _write(path, mats, n_cells):
    """Pseudobulk cool like scHiCluster's: upper triangle, group_n_cells attribute."""
    bins = cooler.binnify(SIZES, RES)
    off = dict(zip(SIZES.index, np.r_[0, np.cumsum((-(-SIZES // RES)).to_numpy())][:-1]))
    rows = []
    for c, M in mats.items():
        i, j = np.nonzero(np.triu(M))
        rows.append(pd.DataFrame({"bin1_id": i + off[c], "bin2_id": j + off[c], "count": M[i, j]}))
    px = pd.concat(rows).sort_values(["bin1_id", "bin2_id"])
    cooler.create_cooler(str(path), bins, px, ordered=True, dtypes={"count": np.float64})
    with h5py.File(path, "r+") as f:
        f.attrs["group_n_cells"] = n_cells


@pytest.fixture(scope="module")
def groups(tmp_path_factory):
    d = tmp_path_factory.mktemp("loops")
    rng = np.random.default_rng(0)
    cells = {}
    prefixes = {}
    for gi, (g, n) in enumerate(N_CELLS.items()):
        per = {}
        for c, size in SIZES.items():
            nb = int(-(-size // RES))
            band = np.triu(np.ones((nb, nb)), 1) - np.triu(np.ones((nb, nb)), BAND + 1)
            x = rng.gamma(2.0 + gi, 1.0, size=(n, nb, nb)) * band   # cells x bins x bins
            x[:, 10, 18] += 3.0 * gi                                  # a loop that differs
            per[c] = x
        cells[g] = per
        prefixes[g] = d / g
        for m in ("Q", "E", "T"):
            _write(f"{d / g}.{m}.cool", {c: v.mean(0) for c, v in per.items()}, n)
            _write(f"{d / g}.{m}2.cool", {c: (v ** 2).mean(0) for c, v in per.items()}, n)
        loops = pd.DataFrame({0: ["chrA", "chrA", "chrB"], 1: [100_000, 200_000 + gi * 10_000, 50_000],
                              2: [110_000, 210_000 + gi * 10_000, 60_000], 3: ["chrA", "chrA", "chrB"],
                              4: [180_000, 300_000, 120_000], 5: [190_000, 310_000, 130_000], 6: [1.0, 2.0, 3.0]})
        loops.to_csv(f"{d / g}.loop.bedpe", sep="\t", header=False, index=False)
    return d, cells, prefixes


def test_anova_equals_one_way_anova(groups, tmp_path):
    d, cells, prefixes = groups
    out = hic.loop_anova(prefixes, tmp_path / "anova", min_dist=0, max_dist=BAND + 1)
    assert set(out) == {"Q", "E", "T"}
    clr = cooler.Cooler(str(out["Q"]))
    assert clr.info["metadata"]["n_groups"] == 3 and clr.info["metadata"]["n_cells"] == 20
    for c in SIZES.index:
        F = clr.matrix(balance=False).fetch(c)
        nb = F.shape[0]
        for i, j in [(10, 18), (0, 5), (3, 4), (nb - BAND - 1, nb - 1)]:
            ref = f_oneway(*[cells[g][c][:, i, j] for g in N_CELLS]).statistic
            np.testing.assert_allclose(F[i, j], ref, rtol=1e-6)
        assert np.all(np.triu(F, BAND + 1) == 0)          # nothing beyond the band


def test_anova_distance_and_blacklist(groups, tmp_path):
    d, cells, prefixes = groups
    bl = tmp_path / "bl.bed"
    pd.DataFrame([["chrA", 300_000, 320_000]]).to_csv(bl, sep="\t", header=False, index=False)
    out = hic.loop_anova(prefixes, tmp_path / "a2", matrices="Q", black_list=bl, black_list_pad=2,
                         min_dist=2, max_dist=10)
    F = cooler.Cooler(str(out["Q"])).matrix(balance=False).fetch("chrA")
    i, j = np.nonzero(np.triu(F))
    assert ((j - i) > 2).all() and ((j - i) < 10).all()
    masked = np.zeros(F.shape[0], bool)
    masked[30 - 2:32 + 2] = True
    assert not (masked[i] | masked[j]).any()


def test_merge_values_differential_specific(groups, tmp_path):
    d, cells, prefixes = groups
    loops = hic.merge_loops({g: f"{p}.loop.bedpe" for g, p in prefixes.items()})
    # 3 shared + 3 group-specific positions of the second loop -> 5 distinct pixels
    assert len(loops) == 5 and list(loops["chrom1"].unique()) == ["chrA", "chrB"]
    assert loops[list(N_CELLS)].sum(axis=1).tolist() == [3, 1, 1, 1, 3]
    out = hic.loop_anova(prefixes, tmp_path / "a3", min_dist=0, max_dist=BAND + 1)
    q = hic.loop_values(loops, {g: f"{p}.Q.cool" for g, p in prefixes.items()})
    t = hic.loop_values(loops, {g: f"{p}.T.cool" for g, p in prefixes.items()})
    for g in N_CELLS:
        np.testing.assert_allclose(q.loc[0, g], cells[g]["chrA"][:, 10, 18].mean())
    res = hic.differential_loops(loops, out, z_quantile=0.5, min_dist=0, edge_bins=0)
    ref = f_oneway(*[cells[g]["chrA"][:, 10, 18] for g in N_CELLS])
    np.testing.assert_allclose(res.loc[0, "Qanova"], ref.statistic, rtol=1e-6)
    np.testing.assert_allclose(res.loc[0, "Qpval"], ref.pvalue, rtol=1e-6)
    assert {"Qfdr", "Efdr", "Tfdr", "differential"} <= set(res.columns)
    spec = hic.specific_loops(q, t, fc_q=1.0, fc_t=1.0)
    assert spec.shape == (5, 3)
    assert spec.loc[0, "g3"] and not spec.loc[0, "g1"]               # loop (10, 18) highest in g3


def test_apa_shape(groups):
    d, cells, prefixes = groups
    loops = hic.merge_loops({g: f"{p}.loop.bedpe" for g, p in prefixes.items()})
    apa = hic.loop_apa(f"{prefixes['g1']}.Q.cool", {"all": loops}, window=3, max_dist_bins=BAND + 1)
    assert apa["all"].shape == (7, 7) and np.isfinite(apa["all"]).all()


def test_apa_matches_hcea_dense(tmp_path):
    """Sparse loop_apa equals HCEA's dense computation (fig4/07 `apa`)."""
    rng = np.random.default_rng(0)
    n, dist = 80, 35
    sizes = pd.Series({"chrA": n * RES})
    M = np.triu(rng.gamma(2, 1, (n, n)) * (np.triu(np.ones((n, n))) - np.triu(np.ones((n, n)), 40)))
    i, j = np.nonzero(M)
    px = pd.DataFrame({"bin1_id": i, "bin2_id": j, "count": M[i, j]})
    cooler.create_cooler(str(tmp_path / "q.cool"), cooler.binnify(sizes, RES), px, ordered=True,
                         dtypes={"count": np.float64})
    loops = pd.DataFrame({"chrom1": ["chrA"] * 3, "start1": [15 * RES, 20 * RES, 30 * RES],
                          "start2": [30 * RES, 45 * RES, 52 * RES]})
    mine = hic.loop_apa(tmp_path / "q.cool", {"a": loops}, window=5, max_dist_bins=dist)["a"]
    Q = cooler.Cooler(str(tmp_path / "q.cool")).matrix(balance=False).fetch("chrA")
    decay = np.array([np.mean(np.diag(Q, k)) for k in range(dist)])
    E = np.zeros(Q.shape)
    r, c = np.diag_indices(n)
    E[r, c] = 1
    for k in range(1, dist):
        E[r[:-k], c[k:]] = Q[r[:-k], c[k:]] / decay[k]
    ref = np.zeros((11, 11))
    for x, y in loops[["start1", "start2"]].to_numpy() // RES:
        t = E[x - 5:x + 6, y - 5:y + 6]
        ref += (t - t.min()) / (t.max() - t.min())
    np.testing.assert_allclose(mine, ref / 3, atol=1e-12)


def test_call_loops_passes_imputed_cools(tmp_path, monkeypatch):
    """call_loops hands scHiCluster the cell table of imputed .cool files."""
    pytest.importorskip("schicluster")
    import anndata as ad
    import schicluster.loop.snakemake as sm

    imp = tmp_path / "imp"
    imp.mkdir()
    bins = cooler.binnify(SIZES, RES)
    for c in ("a", "b", "c"):
        cooler.create_cooler(str(imp / f"{c}.cool"), bins,
                             pd.DataFrame({"bin1_id": [0], "bin2_id": [1], "count": [1.0]}))
    adata = ad.AnnData(np.zeros((3, 0)), obs=pd.DataFrame({"type": ["x", "y", "x"]}, index=["a", "b", "c"]))
    adata.uns["hic"] = {"imputed_dir": str(imp), "resolution": RES}
    seen = {}
    monkeypatch.setattr(sm, "call_loop", lambda **kw: seen.update(kw))
    out = hic.call_loops(adata, "type", tmp_path / "loops", black_list="bl.bed", cpu_per_job=4)
    table = pd.read_csv(seen["cell_table_path"], header=None, index_col=0)
    assert table.index.tolist() == ["a", "b", "c"] and table[2].tolist() == ["x", "y", "x"]
    assert table[1].str.endswith(".cool").all()
    assert seen["resolution"] == RES and seen["black_list_path"] == "bl.bed" and seen["shuffle"]
    assert out == {"x": tmp_path / "loops" / "x" / "x", "y": tmp_path / "loops" / "y" / "y"}
