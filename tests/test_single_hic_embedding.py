"""impute_cells output formats and embedding(method="schicluster") on toy cells."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cooler = pytest.importorskip("cooler")
pytest.importorskip("schicluster")

import epione as epi  # noqa: E402

epi.pl  # noqa: B018
from epione.single import hic  # noqa: E402
from epione.single.hic._embed import _band_indices  # noqa: E402

RES = 100_000
SIZES = pd.Series({"chrA": 3_050_000, "chrB": 2_050_000})


@pytest.fixture(scope="module")
def cells(tmp_path_factory):
    d = tmp_path_factory.mktemp("emb")
    bins = cooler.binnify(SIZES, RES)
    rng = np.random.default_rng(0)
    paths = []
    for k in range(12):
        rows = []
        off = 0
        for chrom, n in bins.groupby("chrom", sort=False).size().items():
            for _ in range(300):
                i = int(rng.integers(0, n))
                j = min(n - 1, i + int(rng.exponential(3 + (k % 2) * 4)))
                rows.append((off + i, off + j))
            off += n
        px = pd.DataFrame(rows, columns=["bin1_id", "bin2_id"]).value_counts().reset_index()
        px.columns = ["bin1_id", "bin2_id", "count"]
        px = px.sort_values(["bin1_id", "bin2_id"])
        p = d / f"cell{k}.cool"
        cooler.create_cooler(str(p), bins=bins, pixels=px, ordered=True)
        paths.append(p)
    return d, paths


def test_impute_fills_every_chromosome(cells):
    d, paths = cells
    ad = hic.load_cool_collection(paths)
    hic.impute_cells(ad, out_dir=d / "imp", output_dist=1_000_000, compress=True, progress=False)
    z = np.load(d / "imp" / "cell0.npz")
    assert sorted(z.files) == ["chrA", "chrB"]
    for c in z.files:
        assert z[c].shape == (int(np.ceil(SIZES[c] / RES)),) * 2
        assert z[c].sum() > 0
        np.testing.assert_array_equal(z[c], z[c].T)
        # nothing beyond output_dist
        iu, ju = np.nonzero(z[c])
        assert np.abs(ju - iu).max() <= 1_000_000 // RES


def test_schicluster_embedding(cells):
    d, paths = cells
    ad = hic.load_cool_collection(paths)
    hic.impute_cells(ad, out_dir=d / "imp", output_dist=1_000_000, compress=True, progress=False)
    emb = hic.embedding(ad, method="schicluster", dim=5, dist=300_000)
    assert emb.obsm["X_schicluster"].shape == (12, 5)
    assert emb.uns["hic"]["embedding"]["method"] == "schicluster"
    iu, ju = _band_indices(31, 300_000, RES)
    assert (ju - iu).max() == 3 and (ju - iu).min() == 1


@pytest.fixture(scope="module")
def contact_tables(tmp_path_factory):
    """Raw contact tables in the default column layout (chrom 1, pos 2, chrom 5, pos 6)."""
    d = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(1)
    paths = []
    for k in range(6):
        rows = []
        for chrom, size in SIZES.items():
            for _ in range(400):
                p1 = int(rng.integers(1, size))
                p2 = min(size, p1 + int(rng.exponential(300_000)))
                rows.append(("r", chrom, p1, "+", "-", chrom, p2))
        p = d / f"cell{k}.contacts.tsv.gz"
        pd.DataFrame(rows).to_csv(p, sep="\t", header=False, index=False)
        paths.append(p)
    return d, paths


def test_impute_raw_contacts_at_any_resolution(contact_tables):
    from schicluster.impute.impute_chromosome import impute_chromosome
    from scipy.sparse import load_npz

    d, paths = contact_tables
    sizes_file = d / "sizes.tsv"
    SIZES.to_csv(sizes_file, sep="\t", header=False)
    ad = hic.load_contact_tables(paths, chrom_sizes=SIZES)  # loader default: 100 kb
    res = 200_000
    hic.impute_cells(ad, out_dir=d / "imp", resolution=res, output_dist=1_000_000, progress=False)
    assert ad.uns["hic"]["resolution"] == res
    z = np.load(d / "imp" / "cell0.npz")
    for c in SIZES.index:
        out = d / f"ref_{c}.hdf"
        impute_chromosome(chrom=c, resolution=res, output_path=str(out), contact_path=str(paths[0]),
                          chrom_size_path=str(sizes_file), output_dist=1_000_000,
                          chrom1=1, pos1=2, chrom2=5, pos2=6)
        E = load_npz(f"{out}.npz").toarray()
        np.testing.assert_array_equal(z[c], (E + E.T - np.diag(np.diag(E))).astype(np.float32))
        assert z[c].shape[0] == ad.uns["hic"]["n_chrom_bins"][c] == SIZES[c] // res + 1
    assert not list((d / "imp").glob("*.chrom.sizes"))
    emb = hic.embedding(ad, method="schicluster", dim=3, dist=400_000)
    assert emb.obsm["X_schicluster"].shape == (6, 3)


def test_impute_cool_rejects_other_resolution(cells):
    d, paths = cells
    ad = hic.load_cool_collection(paths)
    with pytest.raises(ValueError, match="binned at 100,000"):
        hic.impute_cells(ad, out_dir=d / "imp_bad", resolution=1_000_000, progress=False)
