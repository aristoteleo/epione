"""Benchmark parity at the model level: real geometry, real Borzoi weights.

``test_scooby_parity`` shows the numbers match; this shows the plumbing
does. A batch from ``ScoobyATACDataset`` at the production geometry
(524,288 bp in, 196,608 bp / 6,144 bins out) must drive upstream's own
``Scooby`` module and ``poisson_multinomial_torch`` loss forward and
backward, with the model's output shape equal to the dataset's target
shape.

Skipped unless ``SCOOBY_SRC`` is set and a CUDA device is present; also
downloads a Borzoi checkpoint, so point ``HF_HOME`` somewhere with space.
"""
from __future__ import annotations

import os
import pathlib
import sys
import types

import numpy as np
import pytest

from epione.models import scooby as sb

pysam = pytest.importorskip("pysam")
torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

_SRC = os.environ.get("SCOOBY_SRC")
if not _SRC or not (pathlib.Path(_SRC) / "scooby").exists():
    pytest.skip("set SCOOBY_SRC to a clone of gagneurlab/scooby",
                allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

sys.modules.setdefault("pybigtools", types.ModuleType("pybigtools"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from scooby.modeling import Scooby  # noqa: E402
from scooby.utils.utils import poisson_multinomial_torch  # noqa: E402

BORZOI = os.environ.get("BORZOI_CKPT", "johahi/borzoi-replicate-0")
CHROM, CHROM_LEN = "chr1", 2_000_000
N_CELLS, EMB_DIM = 40, 14
N_BINS = sb.TARGET_LENGTH // 32
GROUPS = [list(range(0, 20)), list(range(20, 40))]


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    rng = np.random.default_rng(5)
    d = tmp_path_factory.mktemp("model")

    fa = d / "ref.fa"
    with open(fa, "w") as fh:
        fh.write(f">{CHROM}\n")
        for i in range(0, CHROM_LEN, 100_000):
            blk = "".join(rng.choice(list("ACGT"),
                                     size=min(100_000, CHROM_LEN - i)))
            for j in range(0, len(blk), 80):
                fh.write(blk[j:j + 80] + "\n")
    pysam.faidx(str(fa))

    barcodes = [f"BC{i:03d}" for i in range(N_CELLS)]
    raw, gz = d / "f.tsv", d / "f.tsv.gz"
    starts = np.sort(rng.integers(0, CHROM_LEN - 600, size=400_000))
    with open(raw, "w") as fh:
        for s in starts:
            fh.write(f"{CHROM}\t{int(s)}\t{int(s) + int(rng.integers(40, 500))}"
                     f"\t{barcodes[int(rng.integers(0, N_CELLS))]}\t1\n")
    pysam.tabix_compress(str(raw), str(gz), force=True)
    pysam.tabix_index(str(gz), preset="bed", force=True, keep_original=True)

    reader = sb.FragmentInsertionReader(gz, barcodes,
                                        chrom_sizes={CHROM: CHROM_LEN})
    t0 = 800_000
    intervals = pd.DataFrame({"chrom": [CHROM], "start": [t0],
                              "end": [t0 + sb.TARGET_LENGTH],
                              "fold": ["fold0"]})
    emb = rng.normal(size=(N_CELLS, EMB_DIM)).astype(np.float32)
    ds = sb.ScoobyATACDataset(reader, fa, intervals, emb, groups=GROUPS,
                              rc_aug=False, seed=0)
    return ds[0]


def test_dataset_produces_production_geometry(batch):
    seq, _, tgt, emb = batch
    assert tuple(seq.shape) == (sb.INPUT_LENGTH, 4)
    assert tuple(tgt.shape) == (N_BINS, len(GROUPS))
    assert tuple(emb.shape) == (len(GROUPS), EMB_DIM)
    assert float(tgt.sum()) > 0


def test_plain_from_pretrained_leaves_meta_buffers():
    """The problem ``load_scooby`` exists to solve. Borzoi registers its
    relative-position tables with persistent=False, so they are in no
    checkpoint and transformers' meta-device init never materialises
    them; ``.to('cuda')`` then raises. If this ever starts passing zero,
    the workaround can be retired.
    """
    raw = Scooby.from_pretrained(
        BORZOI, cell_emb_dim=EMB_DIM, embedding_dim=1920, n_tracks=1,
        return_center_bins_only=True, disable_cache=True,
        use_transform_borzoi_emb=False)
    assert sum(1 for _, b in raw.named_buffers() if b.is_meta) > 0


@pytest.fixture(scope="module")
def model():
    m = sb.load_scooby(BORZOI, cell_emb_dim=EMB_DIM, n_tracks=1,
                       lora=True, train=True)
    assert sum(1 for _, b in m.named_buffers() if b.is_meta) == 0
    return m.to("cuda")


def test_lora_leaves_a_small_trainable_fraction(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert 0 < trainable < total
    assert trainable / total < 0.1


def test_forward_backward_and_training_step(batch, model):
    seq, _, tgt, emb = batch
    x = seq.unsqueeze(0).permute(0, 2, 1).to("cuda")
    y = tgt.unsqueeze(0).to("cuda")
    ce = emb.unsqueeze(0).to("cuda")

    model.train()
    with torch.autocast("cuda"):
        out = model(x, ce)
    assert tuple(out.shape) == tuple(y.shape)

    with torch.autocast("cuda"):
        loss = poisson_multinomial_torch(out, y, total_weight=0.2)
    assert torch.isfinite(loss).all()
    loss.backward()

    named = [(n, p) for n, p in model.named_parameters()
             if p.requires_grad and p.grad is not None]
    live = [n for n, p in named if torch.any(p.grad != 0)]
    dead = [n for n, p in named if not torch.any(p.grad != 0)]
    assert live
    # PEFT zero-inits lora_B, so dL/dlora_A = dL/dout * B = 0 at step 0.
    # Anything else without gradient is a genuinely disconnected tensor.
    assert not [n for n in dead if "lora_A" not in n], dead[:3]
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in model.base_model.cell_state_to_conv.parameters())

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    for _ in range(8):
        opt.zero_grad()
        with torch.autocast("cuda"):
            l = poisson_multinomial_torch(model(x, ce), y, total_weight=0.2)
        l.backward()
        opt.step()
        losses.append(l.item())
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


def test_cell_states_give_different_predictions(batch, model):
    """If conditioning did nothing the two groups would be identical."""
    seq, _, _, emb = batch
    x = seq.unsqueeze(0).permute(0, 2, 1).to("cuda")
    ce = emb.unsqueeze(0).to("cuda")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda"):
        o = model(x, ce).float()
    assert (o[0, :, 0] - o[0, :, 1]).abs().mean().item() > 0
