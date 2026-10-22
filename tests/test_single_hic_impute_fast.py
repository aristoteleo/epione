"""Fast windowed imputation (hic._impute_fast) against scHiCluster's impute_chromosome."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba")
cooler = pytest.importorskip("cooler")
pytest.importorskip("schicluster")
from schicluster.impute.impute_chromosome import impute_chromosome as upstream  # noqa: E402
from scipy.sparse import load_npz  # noqa: E402

from epione.single import hic  # noqa: E402
from epione.single.hic import _impute_fast  # noqa: E402
from epione.single.hic._imputed_io import ImputedCell  # noqa: E402

RES = 10_000
SIZES = pd.Series({"chrA": 4_003_000, "chrB": 2_501_000})
COLS = dict(chrom1=1, pos1=2, chrom2=3, pos2=4)
WINDOWED = dict(window_size=1_000_000, step_size=300_000, output_dist=505_000)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("fastimp")
    rng = np.random.default_rng(1)
    rows = []
    for chrom, size in SIZES.items():
        p1 = rng.integers(1, size - 1, 6000)
        p2 = np.minimum(size, p1 + rng.exponential(80_000, 6000).astype(int) + 1)
        rows += [(f"r{i}", chrom, a, chrom, b) for i, (a, b) in enumerate(zip(p1, p2))]
    contacts = d / "cell.pairs"
    with open(contacts, "w") as f:
        f.write("#columns: readID chrom1 pos1 chrom2 pos2\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    sizes_file = d / "chrom.sizes"
    SIZES.to_csv(sizes_file, sep="\t", header=False)
    bins = cooler.binnify(SIZES, RES)
    offs = dict(zip(SIZES.index, np.r_[0, np.cumsum((SIZES // RES + 1).to_numpy())][:-1]))
    df = pd.DataFrame(rows, columns=["id", "c1", "p1", "c2", "p2"])
    b1 = df["c1"].map(offs) + (df["p1"] - 1) // RES
    b2 = df["c2"].map(offs) + (df["p2"] - 1) // RES
    px = pd.DataFrame({"bin1_id": np.minimum(b1, b2), "bin2_id": np.maximum(b1, b2)}).value_counts().reset_index()
    px.columns = ["bin1_id", "bin2_id", "count"]
    cool = d / "cell.cool"
    cooler.create_cooler(str(cool), bins=bins, pixels=px.sort_values(["bin1_id", "bin2_id"]), ordered=True)
    return d, contacts, sizes_file, cool


def _both(tmp, chrom, src, **kw):
    a, b = tmp / f"up_{chrom}", tmp / f"fast_{chrom}"
    upstream(chrom, RES, str(a), **src, **kw)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="fast imputation failed")   # no silent fallback
        _impute_fast.impute_chromosome(chrom, RES, str(b), **src, **kw)
    return load_npz(f"{a}.npz").tocsr(), load_npz(f"{b}.npz").tocsr()


def _close(u, f):
    assert u.shape == f.shape
    du, df = u.toarray(), f.toarray()
    assert np.array_equal(du != 0, df != 0)
    np.testing.assert_allclose(df, du, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("pad", [1, 2])
@pytest.mark.parametrize("min_cutoff", [0, 1e-5])
def test_windowed_contacts_match(data, tmp_path, pad, min_cutoff):
    _, contacts, sizes_file, _ = data
    src = dict(contact_path=str(contacts), chrom_size_path=str(sizes_file), **COLS)
    for chrom in SIZES.index:
        _close(*_both(tmp_path, chrom, src, pad=pad, min_cutoff=min_cutoff, **WINDOWED))


def test_windowed_cool_matches(data, tmp_path):
    _, _, _, cool = data
    for chrom in SIZES.index:
        _close(*_both(tmp_path, chrom, {"scool_url": str(cool)}, pad=1, **WINDOWED))


def test_windowed_path_is_used(data, tmp_path, monkeypatch):
    _, contacts, sizes_file, _ = data
    calls = []
    real = _impute_fast._impute_windowed
    monkeypatch.setattr(_impute_fast, "_impute_windowed", lambda *a, **k: calls.append(1) or real(*a, **k))
    src = dict(contact_path=str(contacts), chrom_size_path=str(sizes_file), **COLS)
    _impute_fast.impute_chromosome("chrA", RES, str(tmp_path / "w"), **src, pad=1, **WINDOWED)
    _impute_fast.impute_chromosome("chrA", RES, str(tmp_path / "f"), **src, pad=1, window_size=500_000_000)
    _impute_fast.impute_chromosome("chrA", RES, str(tmp_path / "r"), **src, pad=1, rp=1, **WINDOWED)
    assert calls == [1, 1]          # rp=1 runs upstream


@pytest.mark.parametrize("pad,od", [(1, 500_000_000), (2, 505_000)])
def test_whole_chromosome_matches(data, tmp_path, pad, od):
    _, contacts, sizes_file, cool = data
    src = dict(contact_path=str(contacts), chrom_size_path=str(sizes_file), **COLS)
    for chrom in SIZES.index:
        _close(*_both(tmp_path, chrom, src, pad=pad, window_size=500_000_000, output_dist=od))
    _close(*_both(tmp_path, "chrA", {"scool_url": str(cool)}, pad=pad, window_size=500_000_000, output_dist=od))


def test_impute_cells_accelerate(data, tmp_path):
    _, contacts, _, _ = data
    out = {}
    for acc in (True, False):
        a = hic.load_contact_tables([contacts], chrom_sizes=SIZES, resolution=RES, columns=COLS,
                                    cell_ids=["c"])
        hic.impute_cells(a, tmp_path / str(acc), pad=1, accelerate=acc, progress=False, **WINDOWED)
        out[acc] = ImputedCell(tmp_path / str(acc) / "c.cool")
    for chrom in SIZES.index:
        np.testing.assert_allclose(out[True][chrom], out[False][chrom], rtol=1e-4, atol=1e-6)
