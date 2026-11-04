"""hic.gene_score: identical to scHiCluster gene_score_raw on toy contact tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("schicluster")
from schicluster.draft.gene_score import gene_score_raw  # noqa: E402

from epione.single import hic  # noqa: E402

RES = 10_000
SIZES = pd.Series({"chrA": 2_005_000, "chrB": 1_503_000})
COLS = {"chrom1": 1, "pos1": 2, "chrom2": 3, "pos2": 4}


@pytest.fixture(scope="module")
def cells(tmp_path_factory):
    d = tmp_path_factory.mktemp("gs")
    rng = np.random.default_rng(0)
    paths = []
    for k in range(4):
        rows = []
        for chrom, size in SIZES.items():
            p1 = rng.integers(1, size, 1500)
            p2 = np.minimum(size, p1 + rng.exponential(30_000, 1500).astype(int))
            p2[:300] = p1[:300] + rng.integers(0, 900, 300)          # short contacts
            rows += [(f"r{k}_{i}", chrom, a, chrom, b) for i, (a, b) in enumerate(zip(p1, p2))]
        rows += [(f"t{k}", "chrA", 5_000, "chrB", 7_000)]               # trans: ignored
        p = d / f"cell{k}.pairs"
        with open(p, "w") as f:
            f.write("## pairs format v1.0\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")
        paths.append(p)
    genes = pd.DataFrame({"chrom": ["chrA", "chrA", "chrB", "chrB", "chrC"],
                          "start": [10_000, 400_000, 0, 900_123, 0],
                          "end": [95_000, 1_300_000, 45_000, 1_503_000, 50_000],
                          "gene_name": ["a1", "a2", "b1", "b2", "c1"]},
                         index=["g1", "g2", "g3", "g4", "g5"])
    adata = hic.load_contact_tables(paths, chrom_sizes=SIZES, resolution=RES, columns=COLS,
                                    cell_ids=[f"cell{k}" for k in range(4)])
    return paths, genes, adata


def _upstream(paths, genes):
    meta = genes[genes["chrom"].isin(SIZES.index)][["chrom", "start", "end"]].copy()
    meta.columns = [0, 1, 2]
    meta[1] //= RES
    meta[2] //= RES
    return np.array([gene_score_raw(str(p), SIZES, meta, RES, 1, 2, 3, 4) for p in paths], dtype=float)


def test_matches_schicluster(cells):
    paths, genes, adata = cells
    gs = hic.gene_score(adata, genes, resolution=RES)
    assert list(gs.var_names) == ["g1", "g2", "g3", "g4"]               # chrC dropped
    assert list(gs.var["gene_name"]) == ["a1", "a2", "b1", "b2"]
    np.testing.assert_array_equal(gs.X.toarray(), _upstream(paths, genes))
    assert gs.uns["hic"]["gene_score"] == {"resolution": RES, "slop": 0, "min_pos_dist": 0}


def test_min_pos_dist_and_bed(cells, tmp_path):
    paths, genes, adata = cells
    bed = tmp_path / "genes.bed"
    genes.reset_index()[["chrom", "start", "end", "index", "gene_name"]].to_csv(
        bed, sep="\t", header=False, index=False)
    full = hic.gene_score(adata, bed, resolution=RES)
    long = hic.gene_score(adata, bed, resolution=RES, min_pos_dist=1000)
    assert list(full.var_names) == ["g1", "g2", "g3", "g4"]
    assert (long.X.toarray() <= full.X.toarray()).all()
    assert (long.X.toarray() < full.X.toarray()).any()
    # filtering the files by hand gives the same counts
    kept = []
    for p in paths:
        df = pd.read_csv(p, sep="\t", header=None, comment="#")
        df = df[(df[1] != df[3]) | ((df[2] - df[4]).abs() > 1000)]
        q = tmp_path / p.name
        df.to_csv(q, sep="\t", header=False, index=False)
        kept.append(q)
    np.testing.assert_array_equal(long.X.toarray(), _upstream(kept, genes))


def test_requires_contact_tables():
    import anndata as ad
    with pytest.raises(ValueError, match="load_contact_tables"):
        hic.gene_score(ad.AnnData(np.zeros((2, 0))), pd.DataFrame({"chrom": [], "start": [], "end": []}))
