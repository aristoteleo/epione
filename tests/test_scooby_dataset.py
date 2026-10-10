"""``ScoobyATACDataset`` — geometry, augmentation and worker safety.

Uses a reduced geometry (4096 bp input / 1024 bp target) so the test is
fast; the centring and augmentation invariants do not depend on scale.
"""
from __future__ import annotations

import numpy as np
import pytest

from epione.models import scooby as sb

torch = pytest.importorskip("torch")
pysam = pytest.importorskip("pysam")
pd = pytest.importorskip("pandas")

CHROM, CHROM_LEN, N_CELLS = "chr1", 200_000, 24
BARCODES = [f"BC{i:03d}" for i in range(N_CELLS)]
INPUT_LEN, TARGET_LEN, BIN = 4096, 1024, 32
FLANK = (INPUT_LEN - TARGET_LEN) // 2
GROUPS = [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))]
EMB_DIM = 8


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    rng = np.random.default_rng(7)
    d = tmp_path_factory.mktemp("ds")

    seq = "".join(rng.choice(list("ACGT"), size=CHROM_LEN))
    fa = d / "ref.fa"
    with open(fa, "w") as fh:
        fh.write(f">{CHROM}\n")
        for i in range(0, CHROM_LEN, 80):
            fh.write(seq[i:i + 80] + "\n")
    pysam.faidx(str(fa))

    recs = sorted(
        (CHROM, int(s := rng.integers(0, CHROM_LEN - 400)),
         int(min(s + rng.integers(40, 400), CHROM_LEN - 1)),
         BARCODES[int(rng.integers(0, N_CELLS))])
        for _ in range(20_000)
    )
    raw, gz = d / "f.tsv", d / "f.tsv.gz"
    raw.write_text("".join(f"{c}\t{s}\t{e}\t{bc}\t1\n" for c, s, e, bc in recs))
    pysam.tabix_compress(str(raw), str(gz), force=True)
    pysam.tabix_index(str(gz), preset="bed", force=True, keep_original=True)

    reader = sb.FragmentInsertionReader(gz, BARCODES,
                                        chrom_sizes={CHROM: CHROM_LEN})
    starts = np.arange(20_000, 60_000, TARGET_LEN)
    intervals = pd.DataFrame({"chrom": CHROM, "start": starts,
                              "end": starts + TARGET_LEN, "fold": "fold0"})
    emb = rng.normal(size=(N_CELLS, EMB_DIM)).astype(np.float32)
    return dict(reader=reader, fasta=fa, intervals=intervals, emb=emb, seq=seq)


def make_ds(fx, **kw):
    kw.setdefault("groups", GROUPS)
    return sb.ScoobyATACDataset(
        fx["reader"], fx["fasta"], fx["intervals"], fx["emb"],
        input_length=INPUT_LEN, target_length=TARGET_LEN, bin_size=BIN,
        seed=0, **kw)


def test_item_shapes(fixture):
    ds = make_ds(fixture)
    assert len(ds) == len(fixture["intervals"])
    seq, rc, tgt, emb = ds[0]
    assert tuple(seq.shape) == (INPUT_LEN, 4)
    assert tuple(tgt.shape) == (TARGET_LEN // BIN, len(GROUPS))
    assert tuple(emb.shape) == (len(GROUPS), EMB_DIM)
    assert not bool(rc)
    sums = seq.numpy().sum(1)
    assert np.all((sums == 1) | (sums == 0)), "not one-hot (N -> all zero)"


def test_target_sits_centred_in_the_input(fixture):
    """The model reads INPUT_LEN to predict the central TARGET_LEN; if
    these drift apart every prediction is offset."""
    ds = make_ds(fixture)
    seq, _, _, _ = ds[0]
    t0 = int(fixture["intervals"].iloc[0]["start"])
    ref = fixture["seq"]
    np.testing.assert_array_equal(
        seq.numpy()[FLANK:FLANK + TARGET_LEN],
        sb.one_hot_dna(ref[t0:t0 + TARGET_LEN]))
    np.testing.assert_array_equal(
        seq.numpy()[:FLANK], sb.one_hot_dna(ref[t0 - FLANK:t0]))


def test_targets_match_the_reader(fixture):
    ds = make_ds(fixture)
    _, _, tgt, _ = ds[0]
    t0 = int(fixture["intervals"].iloc[0]["start"])
    direct = fixture["reader"].pseudobulk(
        CHROM, t0, t0 + TARGET_LEN, GROUPS, bin_size=BIN,
        scale=sb.ATAC_SCALE)
    np.testing.assert_allclose(tgt.numpy(), direct.T)
    assert direct.sum() > 0


def test_group_embedding_is_the_member_mean(fixture):
    ds = make_ds(fixture)
    _, _, _, emb = ds[0]
    np.testing.assert_allclose(
        emb.numpy()[0], fixture["emb"][GROUPS[0]].mean(0), atol=1e-6)


def test_reverse_complement_flips_sequence_and_target(fixture):
    fwd, rcd = make_ds(fixture), make_ds(fixture, rc_aug=True)
    seen_rc = seen_fwd = False
    for i in range(len(rcd)):
        s, rc, t, _ = rcd[i]
        if bool(rc):
            seen_rc = True
            sf, _, tf, _ = fwd[i]
            np.testing.assert_array_equal(s.numpy(), sf.numpy()[::-1, ::-1])
            np.testing.assert_allclose(t.numpy(), tf.numpy()[::-1, :])
        else:
            seen_fwd = True
    assert seen_rc and seen_fwd


def test_per_cell_mode(fixture):
    sp = pytest.importorskip("scipy.sparse")
    knn = sp.random(N_CELLS, N_CELLS, density=0.3, format="csr",
                    random_state=0)
    knn.data[:] = 1.0
    ds = make_ds(fixture, groups=None, neighbors=knn, cell_sample_size=6)
    _, _, tgt, emb = ds[0]
    assert tuple(tgt.shape) == (TARGET_LEN // BIN, 6)
    assert tuple(emb.shape) == (6, EMB_DIM)


def test_exactly_one_sampling_mode(fixture):
    sp = pytest.importorskip("scipy.sparse")
    with pytest.raises(ValueError):
        make_ds(fixture, groups=None)
    with pytest.raises(ValueError):
        make_ds(fixture, neighbors=sp.eye(N_CELLS, format="csr"))


def test_embedding_row_order_is_checked(fixture):
    """A silent mismatch here trains every cell against another cell's
    coverage, so it must raise rather than broadcast."""
    bad = pd.DataFrame({"embedding": list(fixture["emb"]),
                        "obs_names": BARCODES[::-1]})
    with pytest.raises(ValueError):
        sb.ScoobyATACDataset(
            fixture["reader"], fixture["fasta"], fixture["intervals"], bad,
            groups=GROUPS, input_length=INPUT_LEN,
            target_length=TARGET_LEN, bin_size=BIN)


def test_dataframe_embedding_matches_array(fixture):
    good = pd.DataFrame({"embedding": list(fixture["emb"]),
                         "obs_names": BARCODES})
    ds_df = sb.ScoobyATACDataset(
        fixture["reader"], fixture["fasta"], fixture["intervals"], good,
        groups=GROUPS, input_length=INPUT_LEN, target_length=TARGET_LEN,
        bin_size=BIN)
    np.testing.assert_allclose(ds_df[0][3].numpy(), make_ds(fixture)[0][3].numpy())


def test_contig_edge_is_zero_padded(fixture):
    edge = pd.DataFrame({"chrom": [CHROM], "start": [0], "end": [TARGET_LEN],
                         "fold": ["fold0"]})
    ds = sb.ScoobyATACDataset(
        fixture["reader"], fixture["fasta"], edge, fixture["emb"],
        groups=GROUPS, input_length=INPUT_LEN, target_length=TARGET_LEN,
        bin_size=BIN)
    seq, _, _, _ = ds[0]
    assert tuple(seq.shape) == (INPUT_LEN, 4)
    assert not seq.numpy()[:FLANK].any()


@pytest.mark.parametrize("workers", [0, 2])
def test_dataloader_round_trip(fixture, workers):
    """num_workers > 0 forks; an inherited pysam handle would silently
    return corrupt reads, so both paths must agree."""
    ds = make_ds(fixture)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=2, num_workers=workers,
        worker_init_fn=sb.worker_init if workers else None)
    batch = next(iter(dl))
    assert tuple(batch[0].shape) == (2, INPUT_LEN, 4)
    assert tuple(batch[2].shape) == (2, TARGET_LEN // BIN, len(GROUPS))
    np.testing.assert_allclose(batch[2][0].numpy(), ds[0][2].numpy())
