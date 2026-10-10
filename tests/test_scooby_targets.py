"""Target and fold layers of ``epione.models.scooby``.

These two layers import no torch by design, so they run in CI on the
core dependency set alone.

The target test checks ``FragmentInsertionReader`` against an
independent reference built the way upstream scooby does it (scprinter's
``frags_to_insertions``: one insertion at each fragment end, value 1,
duplicates summed), because the whole point of the bridge is to produce
numbers that can be compared with upstream's.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from epione.models import scooby as sb

pysam = pytest.importorskip("pysam")

CHROM_SIZES = {"chr1": 4096, "chr2": 2048}
N_CELLS = 12
BARCODES = [f"BC{i:03d}" for i in range(N_CELLS)]


@pytest.fixture(scope="module")
def fragments(tmp_path_factory):
    """A small fragments file plus the records that went into it."""
    rng = np.random.default_rng(0)
    recs = []
    for chrom, length in CHROM_SIZES.items():
        for _ in range(400):
            s = int(rng.integers(0, length - 20))
            e = min(s + int(rng.integers(20, 300)), length)
            if e <= s:
                continue
            recs.append((chrom, s, e, BARCODES[int(rng.integers(0, N_CELLS))],
                         int(rng.integers(1, 4))))
    recs += [
        # end lands exactly on the window start used below: the insertion
        # is inside the window although the fragment does not overlap it
        ("chr1", 900, 1024, BARCODES[0], 1),
        ("chr1", 2040, 2300, BARCODES[1], 1),      # runs past the window
        ("chr1", 1500, 1600, BARCODES[2], 2),
        ("chr1", 1200, 1300, "NOT_A_CELL", 1),     # must be ignored
        ("chr1", 1700, 1800, BARCODES[3], 1),      # duplicate position,
        ("chr1", 1700, 1800, BARCODES[3], 1),      # counts must add
    ]
    recs.sort(key=lambda r: (r[0], r[1], r[2]))

    d = tmp_path_factory.mktemp("frag")
    raw, gz = d / "f.tsv", d / "f.tsv.gz"
    raw.write_text(
        "# synthetic\n"
        + "".join(f"{c}\t{s}\t{e}\t{bc}\t{n}\n" for c, s, e, bc, n in recs)
    )
    pysam.tabix_compress(str(raw), str(gz), force=True)
    pysam.tabix_index(str(gz), preset="bed", force=True, keep_original=True)
    return gz, recs


@pytest.fixture(scope="module")
def reader(fragments):
    gz, _ = fragments
    return sb.FragmentInsertionReader(gz, BARCODES, chrom_sizes=CHROM_SIZES)


def reference_profile(recs, chrom, start, end, groups, bin_size=32,
                      scale=sb.ATAC_SCALE, weight_by_count=False):
    """Upstream-style reference: whole-genome matrix, then slice."""
    offsets, off = {}, 0
    for c, length in CHROM_SIZES.items():
        offsets[c] = off
        off += length

    idx = {bc: i for i, bc in enumerate(BARCODES)}
    rows, cols, vals = [], [], []
    for c, s, e, bc, cnt in recs:
        i = idx.get(bc)
        if i is None:
            continue
        w = float(cnt) if weight_by_count else 1.0
        for p in (s, e):
            if 0 <= p < CHROM_SIZES[c]:
                rows.append(i); cols.append(offsets[c] + p); vals.append(w)
    ins = sp.coo_matrix(
        (np.asarray(vals, np.float32),
         (np.asarray(rows, np.int64), np.asarray(cols, np.int64))),
        shape=(N_CELLS, off)).tocsr()
    ins.sum_duplicates()

    sub = ins[:, offsets[chrom] + start: offsets[chrom] + end]
    out = []
    for members in groups:
        g = np.unique(np.asarray(list(members), dtype=np.int64))
        dense = np.asarray(sub[g].sum(axis=0), dtype=np.float32).ravel()
        out.append(sb.bin_sum(dense, bin_size) * np.float32(scale))
    return np.vstack(out)


GROUPS = [[0], [1, 2], list(range(N_CELLS)), [3]]


@pytest.mark.parametrize("chrom,start,end", [
    ("chr1", 1024, 2048),
    ("chr2", 0, 2048),
])
def test_matches_reference(reader, fragments, chrom, start, end):
    _, recs = fragments
    got = reader.pseudobulk(chrom, start, end, GROUPS)
    exp = reference_profile(recs, chrom, start, end, GROUPS)
    assert got.shape == exp.shape
    np.testing.assert_allclose(got, exp)
    assert got.sum() > 0, "fixture produced an empty window"


def test_duplicate_count_column_ignored(reader, fragments):
    """Upstream builds insertions with np.ones; the 5th column is not a
    weight. Weighting by it must change the answer, which proves the
    default really is the unweighted convention."""
    gz, recs = fragments
    weighted = sb.FragmentInsertionReader(
        gz, BARCODES, chrom_sizes=CHROM_SIZES, weight_by_count=True)
    a = reader.pseudobulk("chr1", 1024, 2048, GROUPS)
    b = weighted.pseudobulk("chr1", 1024, 2048, GROUPS)
    assert not np.allclose(a, b)
    np.testing.assert_allclose(
        b, reference_profile(recs, "chr1", 1024, 2048, GROUPS,
                             weight_by_count=True))


def test_atac_scale_is_a_plain_multiply(reader):
    """ATAC targets are 0.05 x binned counts with no clip or power
    transform; upstream's clip_soft path is RNA-only."""
    raw = reader.pseudobulk("chr1", 1024, 2048, GROUPS, scale=1.0)
    np.testing.assert_allclose(raw, np.round(raw))  # integral counts
    np.testing.assert_allclose(
        raw * sb.ATAC_SCALE,
        reader.pseudobulk("chr1", 1024, 2048, GROUPS), rtol=0, atol=1e-6)


def test_insertion_at_exact_window_start(reader):
    """A fragment ending exactly at the window start contributes an
    insertion there, even though it does not overlap the window and so
    is not returned by a naive tabix fetch."""
    assert reader.pseudobulk("chr1", 1024, 2048, [[0]])[0, 0] > 0


def test_unknown_contig_is_empty_not_an_error(reader):
    z = reader.pseudobulk("chrNope", 0, 1024, GROUPS)
    assert z.shape == (len(GROUPS), 32)
    assert not z.any()


def test_group_matrix_path_matches_list_path(reader):
    G = sb.groups_to_matrix(GROUPS, N_CELLS)
    np.testing.assert_allclose(
        reader.pseudobulk("chr1", 1024, 2048, G),
        reader.pseudobulk("chr1", 1024, 2048, GROUPS))


def test_width_must_divide_into_bins(reader):
    with pytest.raises(ValueError):
        reader.pseudobulk("chr1", 1024, 1024 + 33, GROUPS)


def test_pickles_without_its_file_handle(reader):
    """DataLoader workers get the reader by pickle; an inherited pysam
    handle would return corrupt reads."""
    import pickle
    clone = pickle.loads(pickle.dumps(reader))
    np.testing.assert_allclose(
        clone.pseudobulk("chr1", 1024, 2048, GROUPS),
        reader.pseudobulk("chr1", 1024, 2048, GROUPS))


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------

FOLD_SIZES = {
    "chr1": 120_000_000, "chr2": 90_000_000, "chr19": 25_000_000,
    "chrM": 16_000, "chrY": 9_000_000,
}


@pytest.fixture(scope="module")
def folds(tmp_path_factory):
    out = tmp_path_factory.mktemp("folds") / "folds.bed"
    return sb.make_fold_bed(FOLD_SIZES, out, n_folds=4, block_windows=16,
                            seed=1), out


def test_fold_bed_shape(folds):
    df, _ = folds
    assert len(df) > 0
    assert set(df.columns) == {"chrom", "start", "end", "fold"}
    assert (df["end"] - df["start"] == sb.TARGET_LENGTH).all()
    assert {"chrM", "chrY"}.isdisjoint(set(df["chrom"]))
    assert df["fold"].nunique() == 4
    counts = df["fold"].value_counts()
    assert counts.max() / counts.min() < 1.6


def test_no_cross_fold_leakage(folds):
    """The invariant the whole function exists for: a training window's
    524 kb input must never cover a different fold's 197 kb target."""
    df, _ = folds
    leaks = 0
    for _, sub in df.groupby("chrom"):
        sub = sub.sort_values("start")
        ts, te = sub["start"].to_numpy(), sub["end"].to_numpy()
        fo = sub["fold"].to_numpy()
        for i in range(len(sub)):
            hit = (ts < te[i] + sb.FLANK) & (te > ts[i] - sb.FLANK)
            leaks += int((hit & (fo != fo[i])).sum())
    assert leaks == 0


def test_buffer_keeps_most_windows(folds):
    df, _ = folds
    naive = sum(FOLD_SIZES[c] // sb.TARGET_LENGTH
                for c in ("chr1", "chr2", "chr19"))
    assert len(df) > 0.4 * naive


def test_windows_stay_clear_of_contig_ends(folds):
    """Upstream cannot produce a target for a window ending within
    custom_read_length + 10 bp of a contig end: its trim degenerates to
    dense[100:0] and pooling raises."""
    df, _ = folds
    for _, r in df.iterrows():
        assert r["start"] >= 100
        assert r["end"] <= FOLD_SIZES[r["chrom"]] - 100


def test_block_too_small_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        sb.make_fold_bed(FOLD_SIZES, tmp_path / "x.bed", block_windows=1)


def test_expected_fragments_per_window():
    """The measurement that decides whether per-cell targets are viable.
    Spatial ATAC at ~1.5e4 fragments per spot gives ~1 per window."""
    one = sb.expected_fragments_per_window(14_900, 2.7e9, 1)
    assert 0.5 < one < 3
    assert sb.expected_fragments_per_window(14_900, 2.7e9, 2000) > 1000


# ---------------------------------------------------------------------------
# Embedding / grouping export
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def adata():
    ad = pytest.importorskip("anndata")
    import pandas as pd
    n = 50
    X = np.random.default_rng(0).normal(size=(n, 10)).astype(np.float32)
    X[:, 0] = np.arange(n) * 100.0   # LSI component 0 tracks depth
    a = ad.AnnData(
        X=np.zeros((n, 3), dtype=np.float32),
        obs=pd.DataFrame({"celltype": ["A"] * 20 + ["B"] * 25 + ["C"] * 5},
                         index=[f"BC{i:03d}" for i in range(n)]),
    )
    a.obsm["X_lsi"] = X
    return a


def test_export_embedding_drops_depth_component(adata, tmp_path):
    # Parquet is upstream scooby's embedding format but no engine is a
    # core Epione dependency.
    pytest.importorskip("pyarrow")
    out = tmp_path / "emb.pq"
    df = sb.export_embedding(adata, "X_lsi", out, drop_first=True, n_dims=5)
    arr = np.vstack(list(df["embedding"].values))
    assert arr.shape == (adata.n_obs, 5)
    assert not np.allclose(arr[:, 0], adata.obsm["X_lsi"][:, 0])
    assert list(df["obs_names"]) == list(map(str, adata.obs_names))
    assert out.exists()


def test_export_embedding_scale(adata, tmp_path):
    pytest.importorskip("pyarrow")
    df = sb.export_embedding(adata, "X_lsi", tmp_path / "e.pq",
                             drop_first=True, n_dims=5, scale=True)
    a = np.vstack(list(df["embedding"].values))
    np.testing.assert_allclose(a.mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(a.std(0), 1, atol=1e-5)


def test_export_embedding_rejects_unknown_key(adata, tmp_path):
    with pytest.raises(KeyError):
        sb.export_embedding(adata, "nope", tmp_path / "z.pq")


def test_groups_from_obs(adata):
    groups, names = sb.groups_from_obs(adata, "celltype")
    assert names == ["A", "B", "C"]
    assert [len(g) for g in groups] == [20, 25, 5]
    _, names2 = sb.groups_from_obs(adata, "celltype", min_cells=10)
    assert names2 == ["A", "B"]
