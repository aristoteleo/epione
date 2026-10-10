"""Benchmark parity: run upstream scooby's own target code on the same data.

Shape agreement proves nothing about a benchmark. One set of synthetic
fragments is turned into targets two ways —

  upstream  fragments -> genome-wide ``obsm['insertion']``
            -> ``onTheFlyMultiomeDataset._load_pseudobulk``
  epione    fragments -> tabix window read
            -> ``FragmentInsertionReader.pseudobulk``

— and the two must be numerically identical. Upstream's code is
executed rather than reimplemented, so drift in any convention the
bridge relies on fails here.

Skipped unless ``SCOOBY_SRC`` points at a clone of
https://github.com/gagneurlab/scooby.
"""
from __future__ import annotations

import os
import pathlib
import sys
import types

import numpy as np
import pytest
import scipy.sparse as sp

from epione.models import scooby as sb

pysam = pytest.importorskip("pysam")
pytest.importorskip("torch")
ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")

_SRC = os.environ.get("SCOOBY_SRC")
if not _SRC or not (pathlib.Path(_SRC) / "scooby").exists():
    pytest.skip(
        "set SCOOBY_SRC to a clone of gagneurlab/scooby to run parity tests",
        allow_module_level=True,
    )

# scooby.data imports pybigtools (a Rust build) for its bigwig-backed
# dataset, which this path does not touch.
sys.modules.setdefault("pybigtools", types.ModuleType("pybigtools"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from scooby.data.scdata import onTheFlyMultiomeDataset  # noqa: E402

CHROM_SIZES = {"chr1": 50_000, "chr2": 30_000}   # two contigs: exercise offsets
N_CELLS = 20
BARCODES = [f"BC{i:03d}" for i in range(N_CELLS)]
CRL, BIN, WIN = 90, 32, 1024
GROUPS = [list(range(0, 10)), list(range(10, 20)), [3], list(range(0, 20))]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rng = np.random.default_rng(11)
    # (cell, start) must be unique: snapATAC2's fragment_paired stores one
    # length per (cell, start), and scipy's csr constructor silently sums
    # colliding entries. That is a property of the upstream container, not
    # of the conventions under test.
    seen, recs = set(), []
    for chrom, length in CHROM_SIZES.items():
        for _ in range(6_000):
            s = int(rng.integers(0, length - 500))
            e = s + int(rng.integers(30, 500))
            if e >= length:
                continue
            bc = BARCODES[int(rng.integers(0, N_CELLS))]
            if (bc, chrom, s) in seen:
                continue
            seen.add((bc, chrom, s))
            recs.append((chrom, s, e, bc, int(rng.integers(1, 5))))
    recs.sort(key=lambda r: (r[0], r[1], r[2]))

    d = tmp_path_factory.mktemp("parity")
    raw, gz = d / "f.tsv", d / "f.tsv.gz"
    raw.write_text("".join(f"{c}\t{s}\t{e}\t{bc}\t{n}\n"
                           for c, s, e, bc, n in recs))
    pysam.tabix_compress(str(raw), str(gz), force=True)
    pysam.tabix_index(str(gz), preset="bed", force=True, keep_original=True)

    # Upstream side: snapATAC2 fragment_paired -> scprinter frags_to_insertions.
    offsets, off = {}, 0
    for c, length in CHROM_SIZES.items():
        offsets[c] = off
        off += length
    idx = {bc: i for i, bc in enumerate(BARCODES)}
    rows = [idx[r[3]] for r in recs]
    cols = [offsets[r[0]] + r[1] for r in recs]
    lens = [r[2] - r[1] for r in recs]
    paired = sp.csr_matrix(
        (np.asarray(lens, np.uint16),
         (np.asarray(rows, np.int64), np.asarray(cols, np.int64))),
        shape=(N_CELLS, off))
    paired.sort_indices()
    assert paired.nnz == len(recs), "fixture collided; comparison invalid"

    x = paired
    ins = sp.csr_matrix(
        (np.ones(len(x.indices) * 2, dtype="uint16"),
         np.stack([x.indices, x.indices + x.data], axis=-1).reshape(-1),
         x.indptr * 2),
        shape=x.shape)
    ins.sort_indices()
    ins.sum_duplicates()

    a = ad.AnnData(X=sp.csr_matrix((N_CELLS, 0), dtype=np.float32))
    a.obs_names = BARCODES
    a.obsm["insertion"] = ins
    a.uns["reference_sequences"] = pd.DataFrame(
        {"reference_seq_name": list(CHROM_SIZES),
         "reference_seq_length": [CHROM_SIZES[c] for c in CHROM_SIZES]})

    reader = sb.FragmentInsertionReader(gz, BARCODES, chrom_sizes=CHROM_SIZES)
    return a, reader


def upstream_profile(adata, chrom, t_start, t_end, groups,
                     normalize_atac=True):
    ds = object.__new__(onTheFlyMultiomeDataset)
    ds.adatas = {"atac": adata}
    ds.normalize_atac = normalize_atac
    ds.clip_soft = 5
    ds.custom_read_length = CRL
    cs = adata.uns["reference_sequences"].copy()
    cs["offset"] = np.insert(cs["reference_seq_length"].cumsum()[:-1].values,
                             0, 0)
    ds.chrom_sizes = cs.set_index("reference_seq_name").to_dict("index")

    size = ds.chrom_sizes[chrom]
    c0, c1 = size["offset"], size["reference_seq_length"]
    s = int(max(0, t_start - (CRL + 10))) + c0
    e = int(min(t_end + (CRL + 10), c1)) + c0
    gd = [size, c0, c1, s, e, t_start, t_end]
    return np.vstack([ds._load_pseudobulk(list(g), gd).numpy().ravel()
                      for g in groups])


@pytest.mark.parametrize("chrom,start", [
    ("chr1", 5_120), ("chr1", 20_480), ("chr1", 40_960),
    ("chr2", 4_096), ("chr2", 16_384),
])
def test_identical_to_upstream(data, chrom, start):
    adata, reader = data
    up = upstream_profile(adata, chrom, start, start + WIN, GROUPS)
    mine = reader.pseudobulk(chrom, start, start + WIN, GROUPS,
                             bin_size=BIN, scale=sb.ATAC_SCALE)
    assert up.shape == mine.shape
    assert np.abs(up - mine).max() == 0, "diverged from upstream"


def test_targets_are_discriminating(data):
    """A comparison of two all-zero arrays would also pass."""
    adata, _ = data
    up = upstream_profile(adata, "chr1", 5_120, 5_120 + WIN, GROUPS)
    assert up.sum() > 0
    assert (up > 0).mean() > 0.3


def test_normalize_atac_false_matches_unscaled(data):
    adata, reader = data
    raw = reader.pseudobulk("chr1", 20_480, 20_480 + WIN, GROUPS,
                            bin_size=BIN, scale=1.0)
    up = upstream_profile(adata, "chr1", 20_480, 20_480 + WIN, GROUPS,
                          normalize_atac=False)
    np.testing.assert_allclose(up, raw, atol=1e-6)
    assert raw.max() > 5, (
        "counts never exceed upstream's RNA clip_soft, so an "
        "accidentally-applied clip would not show up here")


def test_upstream_cannot_handle_a_contig_end_window(data):
    """A window ending at a contig end makes upstream's trim
    ``dense[100:0]`` — an empty array — and pooling then raises. The
    bridge still returns a correct profile, and make_fold_bed's
    edge_margin keeps real runs away from this. If upstream ever fixes
    it, this test fails and the workaround can be retired.
    """
    adata, reader = data
    start = CHROM_SIZES["chr1"] - WIN
    with pytest.raises(RuntimeError):
        upstream_profile(adata, "chr1", start, start + WIN, GROUPS)
    mine = reader.pseudobulk("chr1", start, start + WIN, GROUPS,
                             bin_size=BIN, scale=sb.ATAC_SCALE)
    assert mine.shape == (len(GROUPS), WIN // BIN)
    assert mine.sum() > 0
