import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pysam = pytest.importorskip("pysam")
ad = pytest.importorskip("anndata")

from epione.pl._browser_track import (  # noqa: E402
    RRandom, archr_palette_discrete, browser_track_coverage, clip_coverage, gene_region,
    gene_track_data, loop_track_data, plot_browser_track,
)


def _write_fragments(tmp_path, rows):
    path = tmp_path / "s1.fragments.tsv"
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    path.write_text("".join("\t".join(map(str, r)) + "\n" for r in rows))
    return pysam.tabix_index(str(path), preset="bed", force=True)


@pytest.fixture
def toy(tmp_path):
    # Two groups, three cells. Coordinates are 0-based starts (fragment file).
    rows = [
        ("chr1", 1099, 1300, "AAA", 1),   # start1=1100 -> tile 1000
        ("chr1", 1149, 1400, "AAA", 1),   # same cell, same tile -> binarised
        ("chr1", 1099, 1250, "BBB", 1),   # tile 1000
        ("chr1", 1599, 2600, "CCC", 1),   # start1=1600 -> tile 1500 (end ignored)
    ]
    frag = _write_fragments(tmp_path, rows)
    obs = pd.DataFrame(
        {"Clusters": ["C1", "C1", "C2"], "Sample": ["s1"] * 3, "ReadsInTSS": [10.0, 30.0, 20.0]},
        index=["s1#AAA", "s1#BBB", "s1#CCC"],
    )
    return ad.AnnData(obs=obs), {"s1": frag}


def test_coverage_counts_binarised_starts(toy):
    adata, frags = toy
    cov = browser_track_coverage(adata, ("chr1", 1000, 2000), fragments=frags,
                                 tile_size=500, min_cells=1)
    c1 = cov[(cov.group == "C1") & (cov.x == 1000)].y.item()
    assert c1 == pytest.approx(2 * 1e4 / 40.0)
    c2 = cov[(cov.group == "C2") & (cov.x == 1500)].y.item()
    assert c2 == pytest.approx(1e4 / 20.0)
    # end of the CCC fragment (tile 2500) is not counted by default (ArchR 1.0.3)
    assert not ((cov.group == "C2") & (cov.x == 2500) & (cov.y > 0)).any()


def test_coverage_count_ends_option(toy):
    adata, frags = toy
    cov = browser_track_coverage(adata, ("chr1", 1000, 3000), fragments=frags,
                                 tile_size=500, min_cells=1, count_ends=True)
    assert cov[(cov.group == "C2") & (cov.x == 2500)].y.item() > 0


def test_coverage_layout_matches_archr(toy):
    adata, frags = toy
    cov = browser_track_coverage(adata, ("chr1", 1000, 2000), fragments=frags,
                                 tile_size=500, min_cells=1, norm_method="None")
    # regionTiles = 1000, 1500, 2000, 2500; C1 non-zero at 1000 (row 1),
    # C2 at 1500 (row 2). Rows: non-zero (column-major), row-1 (row 0 dropped
    # for C1), row+1 padding.
    assert list(cov.group) == ["C1", "C2", "C2", "C1", "C2"]
    assert list(cov.x) == [1000, 1500, 1000, 1500, 2000]
    assert list(cov.y) == [2.0, 1.0, 0.0, 0.0, 0.0]


def test_min_cells_and_injected_cells(toy):
    adata, frags = toy
    cov = browser_track_coverage(adata, ("chr1", 1000, 2000), fragments=frags,
                                 tile_size=500, min_cells=2)
    assert set(cov.group) == {"C1"}
    cov = browser_track_coverage(adata, ("chr1", 1000, 2000), fragments=frags,
                                 tile_size=500, min_cells=1, cells=["s1#AAA", "s1#CCC"])
    assert cov[(cov.group == "C1") & (cov.x == 1000)].y.item() == pytest.approx(1e4 / 10.0)


def test_max_cells_subsampling_is_seeded(toy):
    adata, frags = toy
    a = browser_track_coverage(adata, ("chr1", 1000, 2000), fragments=frags, tile_size=500,
                               min_cells=1, max_cells=1, seed=3)
    b = browser_track_coverage(adata, ("chr1", 1000, 2000), fragments=frags, tile_size=500,
                               min_cells=1, max_cells=1, seed=3)
    assert a.attrs["cells"] == b.attrs["cells"]
    assert len(a.attrs["cells"]) == 2


def test_r_rng_mersenne_twister_matches_r():
    # set.seed(1); sample(1000, 10)  /  set.seed(42); runif(3)  in R 4.4
    assert list(RRandom(1).sample(1000, 10) + 1) == [836, 679, 129, 930, 509, 471, 299, 270, 978, 187]
    r = RRandom(42)
    assert [r.unif_rand() for _ in range(3)] == [
        0.91480604349635541, 0.93707541329786181, 0.28613953478634357]
    assert list(RRandom(7).sample(70000, 5) + 1) == [4572, 51906, 63706, 66107, 19348]


def test_r_rng_lecuyer_matches_r():
    # RNGkind("L'Ecuyer-CMRG"); set.seed(1); runif(3) / sample(835, 5)
    k = "L'Ecuyer-CMRG"
    r = RRandom(1, k)
    assert [r.unif_rand() for _ in range(3)] == [
        0.6775328286287442, 0.42734572288764422, 0.9103805304875483]
    assert list(RRandom(1, k).sample(835, 5) + 1) == [371, 359, 271, 171, 822]
    assert list(RRandom(5, k).sample(1938, 4) + 1) == [412, 1692, 1302, 1333]


def test_clip_quantile():
    df = pd.DataFrame({"x": np.arange(1000.0), "y": np.arange(1000.0), "group": "a"})
    out, lim = clip_coverage(df)
    assert lim[1] == pytest.approx(np.quantile(np.arange(1000.0), 0.999))
    assert out.y.max() == pytest.approx(lim[1])


def test_palette_archr_order():
    pal = archr_palette_discrete([f"C{i}" for i in range(11, 0, -1)])
    assert list(pal)[:3] == ["C1", "C2", "C3"]
    assert pal["C9"] == "#D8A767"
    big = archr_palette_discrete([f"g{i}" for i in range(25)])
    assert len(set(big.values())) == 25 and big["g0"] == "#D51F26"


def test_gene_region_strand():
    genes = pd.DataFrame({"seqnames": ["chr1", "chr1"], "start": [1000, 5000],
                          "end": [2000, 9000], "strand": ["+", "-"], "symbol": ["A", "B"]})
    assert gene_region(genes, "A", 100, 200) == ("chr1", 900, 1200)
    assert gene_region(genes, "b", 100, 200) == ("chr1", 8900, 9200)


def test_gene_packing_rows():
    genes = pd.DataFrame({"seqnames": "chr1", "start": [100, 150, 5000],
                          "end": [400, 300, 5200], "strand": ["+", "-", "+"],
                          "symbol": ["A", "B", "C"]})
    g = gene_track_data(genes, ("chr1", 1, 10_000))
    assert list(g.symbol) == ["A", "B", "C"]
    assert list(g.cluster) == [1, 2, 1]


def test_loop_geometry():
    loops = pd.DataFrame({"chrom": "chr1", "start": [100, 100], "end": [300, 500], "value": [0.6, 0.9]})
    d = loop_track_data(loops, ("chr1", 1, 1000))
    l2 = d[d.id == "l2"]
    assert l2.x.iloc[0] == pytest.approx(100) and l2.x.iloc[-1] == pytest.approx(500)
    assert l2.y.min() == pytest.approx(-100, abs=0.2)
    assert d[d.id == "l1"].y.min() == pytest.approx(-50, abs=0.2)


def test_plot_returns_figure(toy):
    adata, frags = toy
    genes = pd.DataFrame({"chrom": "chr1", "start": [1200], "end": [1800], "strand": ["+"],
                          "symbol": ["G"]})
    fig, data = plot_browser_track(
        adata, ("chr1", 1000, 2000), fragments=frags, tile_size=500, min_cells=1,
        features=pd.DataFrame({"chrom": ["chr1"], "start": [1100], "end": [1300]}),
        loops=pd.DataFrame({"chrom": ["chr1"], "start": [1100], "end": [1900], "value": [0.7]}),
        gene_annotation=genes, show=False, return_data=True,
    )
    assert fig is not None and {"bulk", "features", "loops", "genes"} <= set(data)


_PARITY_ROOT = os.environ.get("EPIONE_ARCHR_PARITY_DIR")
PARITY = Path(_PARITY_ROOT) / "browser_track" if _PARITY_ROOT else None


@pytest.mark.skipif(PARITY is None or not (PARITY / "summary.tsv").exists(),
                    reason="EPIONE_ARCHR_PARITY_DIR not set or ArchR reference outputs missing")
def test_parity_with_archr():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bt_compare", Path(__file__).parent / "parity" / "browser_track" / "compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = mod.compare(PARITY, PARITY, PARITY / "fragments")
    assert res["palette_identical"]
    for g in mod.GENES:
        r = res[g]
        assert r["region_equal"]
        assert r["n_rows_py"] == r["n_rows_r"] and r["groups_equal"]
        assert r["x_max_abs"] == 0.0
        assert r["coverage_max_abs"] <= 1e-8
        assert r["bulk_clipped_max_abs"] <= 1e-8
        assert r["free_archr_rng_same_cells"] and r["free_archr_rng_max_abs"] <= 1e-8
        assert r["features_identical"]
        assert r.get("genes_identical", True)
        assert r.get("loops_max_abs", 0.0) <= 1e-6
