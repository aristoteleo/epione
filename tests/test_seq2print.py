"""seq2PRINT: exactness and checkpoint compatibility against scPrinter.

The upstream side is scPrinter's own ``scprinter/seq`` package, loaded
from a clone under a throwaway parent package (its top-level
``scprinter/__init__`` would pull in snapATAC2, MOODS and network
downloads). Every comparison is ``torch.equal`` / ``np.array_equal``.

Skipped unless torch is installed and ``SCPRINTER_SRC`` points at the
``scprinter/`` package directory of a scPrinter clone; the torch-only
tests further down run without it.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from epione.models import seq2print as s2p  # noqa: E402

_SRC = os.environ.get("SCPRINTER_SRC")
needs_upstream = pytest.mark.skipif(
    not (_SRC and (pathlib.Path(_SRC) / "seq" / "Models.py").exists()),
    reason="set SCPRINTER_SRC to the scprinter/ package dir of a scPrinter clone",
)

SMALL = dict(n_filters=64, n_layers=4)       # fast; same code paths as default


@pytest.fixture(scope="module")
def up(tmp_path_factory):
    """scPrinter's seq package, importable without its heavy parent."""
    root = tmp_path_factory.mktemp("upstream")
    pkg = root / "scp_up"
    (pkg / "seq").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "seq" / "__init__.py").write_text("")
    for f in ("Models.py", "Modules.py", "Functions.py", "minimum_footprint.py"):
        shutil.copy(pathlib.Path(_SRC) / "seq" / f, pkg / "seq" / f)
    # minimum_footprint.py imports ..utils.rz_conv_torch at module level
    # (used only by an uncalled helper); provide it, verbatim.
    import ast
    text = (pathlib.Path(_SRC) / "utils.py").read_text()
    lines = text.splitlines(keepends=True)
    fn = next(n for n in ast.parse(text).body
              if isinstance(n, ast.FunctionDef) and n.name == "rz_conv_torch")
    (pkg / "utils.py").write_text("import torch\n\n" + "".join(lines[fn.lineno - 1:fn.end_lineno]))
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))   # Models.py imports it
    sys.path.insert(0, str(root))
    import importlib
    return types.SimpleNamespace(
        Models=importlib.import_module("scp_up.seq.Models"),
        Modules=importlib.import_module("scp_up.seq.Modules"),
        mf=importlib.import_module("scp_up.seq.minimum_footprint"),
    )


def upstream_model(up, *, n_filters, n_layers, **lora):
    """scPrinter's construct_model_from_config with its defaults, verbatim."""
    M = up.Modules
    dilation = lambda x: 2 ** (x + 1)  # noqa: E731
    dna = M.DNA_CNN(n_filters=n_filters)
    hid = M.DilatedCNN(n_filters=n_filters, bottleneck=n_filters, n_layers=n_layers,
                       kernel_size=3, groups=8, activation=torch.nn.GELU(),
                       batch_norm=True, residual=True, rezero=False,
                       dilation_func=dilation, batch_norm_momentum=0.1,
                       inception=[True] * n_layers, inception_version=2)
    head = M.Footprints_head(n_filters, kernel_size=1, n_scales=99)
    dna_len = 800 + dna.conv.weight.shape[2] - 1
    for i in range(n_layers):
        dna_len += 2 * 1 * dilation(i)
    return up.Models.seq2PRINT(dna_cnn_model=dna, hidden_layer_model=hid,
                               profile_cnn_model=head, dna_len=dna_len,
                               output_len=800, **lora)


def one_hot(n, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 4, (n, L), generator=g)
    return torch.nn.functional.one_hot(idx, 4).permute(0, 2, 1).float()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def test_default_geometry():
    m = s2p.build_seq2print()
    assert (m.dna_len, m.output_len) == (1840, 800)
    pred, cov = m.eval()(one_hot(2, m.dna_len))
    assert pred.shape == (2, 99, 800) and cov.shape == (2,)


@needs_upstream
@pytest.mark.parametrize("config", [SMALL, {}], ids=["small", "default"])
def test_state_dict_keys_match_scprinter(up, config):
    mine = s2p.build_seq2print(**config)
    theirs = upstream_model(up, n_filters=config.get("n_filters", 1024),
                            n_layers=config.get("n_layers", 8))
    a, b = mine.state_dict(), theirs.state_dict()
    assert list(a) == list(b)
    assert all(a[k].shape == b[k].shape for k in a)
    assert mine.dna_len == theirs.dna_len


@needs_upstream
def test_forward_bit_identical(up):
    torch.manual_seed(0)
    mine = s2p.build_seq2print(**SMALL).eval()
    theirs = upstream_model(up, **SMALL).eval()
    theirs.load_state_dict(mine.state_dict(), strict=True)   # both directions work
    x = one_hot(3, mine.dna_len)
    modes = torch.tensor([0, 8, 18, 48, 98])
    with torch.no_grad():
        for kw in ({}, {"modes": modes}):
            a, b = mine(x, **kw), theirs(x, **kw)
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@needs_upstream
@pytest.mark.parametrize("with_cov", [False, True])
def test_lora_bit_identical(up, with_cov):
    """Cell-state LoRA on every layer, with and without coverage covariates."""
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(5, 6)).astype(np.float32)
    cov = rng.normal(size=(5, 2)).astype(np.float32) if with_cov else None

    torch.manual_seed(0)
    mine = s2p.build_seq2print(**SMALL)
    mine.add_lora(emb.copy(), coverages=None if cov is None else cov.copy(),
                  rank=4, hidden_dim=16)
    theirs = upstream_model(
        up, **SMALL, embeddings=emb.copy(), coverages=None if cov is None else cov.copy(),
        rank=4, hidden_dim=16, lora_dna_cnn=True, lora_dilated_cnn=True,
        lora_pff_cnn=True, lora_profile_cnn=True, lora_count_cnn=True)
    a, b = mine.state_dict(), theirs.state_dict()
    assert sorted(a) == sorted(b)
    theirs.load_state_dict(a, strict=True)

    # B starts at zero: a fresh LoRA model equals its backbone. Perturb it
    # so the comparison exercises the cell-specific path too.
    with torch.no_grad():
        for p in list(mine.parameters()):
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))
    theirs.load_state_dict(mine.state_dict(), strict=True)
    mine.eval(); theirs.eval()
    x, cells = one_hot(4, mine.dna_len, seed=3), torch.tensor([0, 1, 3, 4])
    with torch.no_grad():
        pa, ca = mine(x, cells)
        pb, cb = theirs(x, cells)
    assert torch.equal(pa, pb) and torch.equal(ca, cb)
    # and the cell state actually changes the prediction
    with torch.no_grad():
        p0, _ = mine(x[:1].repeat(2, 1, 1), torch.tensor([0, 4]))
    assert not torch.equal(p0[0], p0[1])


@needs_upstream
def test_lora_embedding_rescale_matches(up):
    """scPrinter rescales the shared embedding once per wrapped layer."""
    rng = np.random.default_rng(2)
    emb = rng.normal(size=(7, 5)).astype(np.float32)
    torch.manual_seed(0)
    mine = s2p.build_seq2print(**SMALL).add_lora(emb.copy(), rank=4, hidden_dim=16)
    torch.manual_seed(0)
    theirs = upstream_model(up, **SMALL, embeddings=emb.copy(), rank=4, hidden_dim=16,
                            lora_dna_cnn=True, lora_dilated_cnn=True, lora_pff_cnn=True,
                            lora_profile_cnn=True, lora_count_cnn=True)
    assert torch.equal(mine.embeddings.weight, theirs.embeddings.weight)
    assert not torch.allclose(mine.embeddings.weight, torch.from_numpy(emb))


# ---------------------------------------------------------------------------
# Footprint targets
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dispersion():
    from epione.tl import DispersionModel
    try:
        return DispersionModel.load()
    except Exception as exc:
        pytest.skip(f"cannot fetch dispersion model: {exc}")


@needs_upstream
def test_targets_bit_identical(up, dispersion):
    """The training targets, from Epione's DispersionModel vs scPrinter's
    loader + dispModel, on realistic-scale tracks."""
    import h5py
    ref_disp = {}
    with h5py.File(dispersion.path) as f:
        for k in f:
            if k == "version":
                continue
            g = f[k]
            ref_disp[k] = {n: g[n][()] for n in ("featureMean", "featureSD", "targetMean", "targetSD")}
            ref_disp[k]["modelWeights"] = [torch.from_numpy(g["modelWeights"][n][()]).float()
                                           for n in ("ELT1", "ELT2", "ELT3", "ELT4")]
    theirs = up.mf.dispModel(ref_disp)
    mine = s2p.footprint_target_model(dispersion)

    g = torch.Generator().manual_seed(0)
    bias = torch.rand(6, 1000, generator=g) * 3
    atac = torch.poisson(bias * 2, generator=g)
    atac[0, 450:550] = 0                       # a footprint
    atac[3] = 0                                # an empty sample
    for modes in (np.arange(2, 101), np.array([2, 10, 37, 100])):
        a = s2p.multiscaleFoot(atac, bias, modes, mine)
        b = up.mf.multiscaleFoot(atac, bias, modes, theirs)
        assert a.shape == (6, len(modes), 800)
        assert np.array_equal(a.numpy(), b.numpy(), equal_nan=True)


# ---------------------------------------------------------------------------
# Training machinery
# ---------------------------------------------------------------------------

def test_ema_matches_ema_pytorch():
    """scPrinter uses ema_pytorch.EMA(beta=0.9999, update_after_step=100,
    update_every=10); our in-house EMA must produce the same weights."""
    ema_pytorch = pytest.importorskip("ema_pytorch")
    from epione.models.seq2print._train import _EMA
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8), torch.nn.Linear(8, 2))
    ref = ema_pytorch.EMA(net, beta=0.9999, update_after_step=100, update_every=10)
    mine = _EMA(net)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    for step in range(1500):
        opt.zero_grad()
        net(torch.randn(16, 4)).pow(2).mean().backward()
        opt.step()
        ref.update()
        mine.update(net)
    for (n, a), b in zip(mine.model.state_dict().items(), ref.ema_model.state_dict().values()):
        assert torch.equal(a, b), n


def test_lora_freezes_backbone():
    m = s2p.build_seq2print(**SMALL)
    n_before = sum(p.numel() for p in m.parameters())
    m.add_lora(np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32),
               rank=4, hidden_dim=16)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(("A_embedding" in n or "B_embedding" in n) for n in trainable)
    assert sum(p.numel() for p in m.parameters()) > n_before


def test_loss_matches_scprinter_formula(dispersion):
    """seq2print_loss == the expression in scPrinter's seq2PRINT.fit."""
    from epione.models.seq2print._train import seq2print_loss
    tgt = s2p.footprint_target_model(dispersion)
    torch.manual_seed(0)
    m = s2p.build_seq2print(**SMALL).eval()
    X = one_hot(3, m.dna_len)
    bias = torch.rand(3, 1000) * 3
    y = torch.stack([torch.poisson(bias * 2), bias], dim=1)
    modes = np.array([5, 9, 40, 77])
    loss, lfp, lcov = seq2print_loss(m, (X, y), tgt, modes, device="cpu")
    fp = s2p.multiscaleFoot(y[:, 0], y[:, 1], modes, tgt)
    mask = ~torch.isnan(fp)
    p_fp, p_cov = m(X, None, modes=torch.as_tensor(modes - 2))
    want_fp = torch.nn.functional.mse_loss(p_fp[mask], fp[mask])
    want_cov = torch.nn.functional.mse_loss(torch.log1p(y[:, 0].sum(-1)), p_cov)
    assert torch.equal(lfp, want_fp) and torch.equal(lcov, want_cov)
    assert torch.equal(loss, want_fp + want_cov)


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    """Synthetic genome + fragments + a PRINT-convention bias stand-in."""
    pysam = pytest.importorskip("pysam")
    from epione._insertions import InsertionReader
    from epione.tl._tn5_bias import Tn5Bias

    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("s2p")
    L = 60_000
    seq = "".join(rng.choice(list("ACGT"), size=L))
    (d / "g.fa").write_text(">c1\n" + "\n".join(seq[i:i + 60] for i in range(0, L, 60)) + "\n")
    pysam.faidx(str(d / "g.fa"))
    bcs = [f"b{i}" for i in range(12)]
    recs = sorted(("c1", int(s := rng.integers(0, L - 400)), int(s + rng.integers(40, 300)),
                   bcs[int(rng.integers(0, 12))]) for _ in range(40_000))
    (d / "f.tsv").write_text("".join(f"{c}\t{s}\t{e}\t{b}\t1\n" for c, s, e, b in recs))
    pysam.tabix_compress(str(d / "f.tsv"), str(d / "f.tsv.gz"), force=True)
    pysam.tabix_index(str(d / "f.tsv.gz"), preset="bed", force=True)

    class FlatBias(Tn5Bias):
        name = "print"
        def track(self, chrom, start, end):
            return np.ones(end - start, np.float32)

    peaks = pd_peaks = None
    import pandas as pd
    pd_peaks = pd.DataFrame({"chrom": "c1", "summit": np.arange(3_000, 57_000, 900)})
    reader = InsertionReader(d / "f.tsv.gz", bcs, shift=FlatBias.shift)
    return dict(fasta=d / "g.fa", reader=reader, bias=FlatBias(), peaks=pd_peaks,
                seq=seq, groups=[list(range(0, 6)), list(range(6, 12))])


def test_dataset_geometry_and_augmentation(tiny_data):
    ds = s2p.Seq2PRINTDataset(tiny_data["peaks"], fasta=tiny_data["fasta"],
                              reader=tiny_data["reader"], bias=tiny_data["bias"],
                              dna_len=1840, min_coverage=None, verbose=False)
    X, y = ds.centred()[0]
    assert X.shape == (4, 1840) and y.shape == (2, 1000)
    s = int(ds.peaks.summit[0])
    want = s2p._dataset._LUT[np.frombuffer(tiny_data["seq"][s - 920:s + 920].encode(), np.uint8)].T
    assert np.array_equal(X.numpy(), want)            # DNA centred on the summit
    ins = tiny_data["reader"].group_insertions("c1", s - 500, s + 500)[0]
    assert np.array_equal(y[0].numpy(), ins)          # signal centred, same convention
    # reverse complement flips DNA channels+positions and the signal positions
    Xr, yr, _ = ds.window(0, rc=True)
    assert torch.equal(Xr, X.flip(0).flip(1)) and torch.equal(yr, y.flip(1))
    # jitter shifts DNA and signal together
    Xj, yj, _ = ds.window(0, jitter=0)
    assert torch.equal(Xj[:, 128:], X[:, :-128]) and torch.equal(yj[:, 128:], y[:, :-128])
    # augmentation is reproducible per (seed, epoch, index) and changes by epoch
    a = ds[3][0]; assert torch.equal(a, ds[3][0])
    ds.set_epoch(1); assert not all(torch.equal(ds[i][0], s2p.Seq2PRINTDataset.__getitem__(ds.centred(), i)[0]) for i in range(5))


def test_dataset_rejects_mismatched_cut_convention(tiny_data):
    from epione._insertions import InsertionReader
    bad = InsertionReader(tiny_data["reader"].fragment_file, tiny_data["reader"].barcodes, shift=(0, -1))
    with pytest.raises(ValueError, match="cut convention"):
        s2p.Seq2PRINTDataset(tiny_data["peaks"], fasta=tiny_data["fasta"], reader=bad,
                             bias=tiny_data["bias"], verbose=False)


def test_fit_runs_and_lora_finetunes(tiny_data, dispersion):
    """Smoke: a tiny bulk model trains, then LoRA fine-tunes per group."""
    m = s2p.build_seq2print(n_filters=32, n_layers=3)
    kw = dict(fasta=tiny_data["fasta"], reader=tiny_data["reader"], bias=tiny_data["bias"],
              dna_len=m.dna_len, min_coverage=None, verbose=False)
    ds = s2p.Seq2PRINTDataset(tiny_data["peaks"], **kw)
    tgt = s2p.footprint_target_model(dispersion)
    res = s2p.fit(m, ds, ds, epochs=2, batch_size=8, num_workers=0, device="cpu",
                  ema=False, target_model=tgt, verbose=False)
    assert len(res.history) == 2 and np.isfinite(res.history[-1]["valid_loss"])

    lora = res.model.add_lora(np.eye(2, dtype=np.float32), rank=2, hidden_dim=8)
    ds2 = s2p.Seq2PRINTDataset(tiny_data["peaks"], groups=tiny_data["groups"], **kw)
    assert len(ds2) == 2 * len(ds) and len(ds2[0]) == 3
    res2 = s2p.fit(lora, ds2, ds2, epochs=1, batch_size=8, lr=3e-5, num_workers=0,
                   device="cpu", ema=False, target_model=tgt, verbose=False)
    assert np.isfinite(res2.history[0]["valid_loss"])


def test_coverage_warming_trains_head_then_lora(tiny_data, dispersion):
    """During warming only the coverage-adjustment head moves; afterwards
    it is frozen and only the LoRA networks move (scPrinter's schedule)."""
    m = s2p.build_seq2print(n_filters=32, n_layers=3)
    kw = dict(fasta=tiny_data["fasta"], reader=tiny_data["reader"], bias=tiny_data["bias"],
              dna_len=m.dna_len, min_coverage=None, verbose=False)
    ds = s2p.Seq2PRINTDataset(tiny_data["peaks"], groups=tiny_data["groups"], **kw)
    m.add_lora(np.eye(2, dtype=np.float32), coverages=np.array([[-1.0], [1.0]], np.float32),
               rank=2, hidden_dim=8)
    tgt = s2p.footprint_target_model(dispersion)
    snap = lambda: {n: p.detach().clone() for n, p in m.named_parameters()}
    is_head = lambda n: "adjustment_" in n
    is_lora = lambda n: ("A_embedding" in n or "B_embedding" in n)

    before = snap()
    s2p.fit(m, ds, ds, epochs=1, coverage_warming=1, batch_size=8, num_workers=0,
            device="cpu", ema=False, target_model=tgt, verbose=False)
    after_warm = snap()
    moved = {n for n in before if not torch.equal(before[n], after_warm[n])}
    assert moved and all(is_head(n) for n in moved), sorted(moved)[:5]

    # A second call with no warming continues from here: LoRA moves, head frozen
    # only because fit() freezes it when leaving warming -- emulate that here.
    for n, p in m.named_parameters():
        if is_head(n):
            p.requires_grad = False
    s2p.fit(m, ds, ds, epochs=1, batch_size=8, lr=1e-2, num_workers=0, device="cpu",
            ema=False, target_model=tgt, verbose=False)
    after = snap()
    moved = {n for n in after_warm if not torch.equal(after_warm[n], after[n])}
    assert moved and all(is_lora(n) for n in moved), sorted(moved)[:5]


def test_fit_leaves_warming_within_one_call(tiny_data, dispersion):
    """Warming then LoRA in a single fit(): head params change only in the
    warming epoch, LoRA params only afterwards."""
    m = s2p.build_seq2print(n_filters=32, n_layers=3)
    kw = dict(fasta=tiny_data["fasta"], reader=tiny_data["reader"], bias=tiny_data["bias"],
              dna_len=m.dna_len, min_coverage=None, verbose=False)
    ds = s2p.Seq2PRINTDataset(tiny_data["peaks"], groups=tiny_data["groups"], **kw)
    m.add_lora(np.eye(2, dtype=np.float32), coverages=np.array([[-1.0], [1.0]], np.float32),
               rank=2, hidden_dim=8)
    res = s2p.fit(m, ds, ds, epochs=2, coverage_warming=1, batch_size=8, lr=1e-2,
                  num_workers=0, device="cpu", ema=False,
                  target_model=s2p.footprint_target_model(dispersion), verbose=False)
    h = res.to_frame()
    assert h.warming.tolist() == [True, False]
    final = dict(m.named_parameters())
    assert not any(p.requires_grad for n, p in final.items() if "adjustment_" in n)
    assert any(p.requires_grad for n, p in final.items() if "A_embedding" in n)


@pytest.mark.parametrize("with_lora", [False, True])
def test_pretrained_round_trip(tmp_path, monkeypatch, with_lora):
    """save_pretrained -> load_pretrained reproduces the model exactly,
    and the registry's md5s match what was written."""
    pytest.importorskip("safetensors")
    from epione.models.seq2print import _pretrained

    torch.manual_seed(0)
    m = s2p.build_seq2print(n_filters=32, n_layers=3)
    manifest = {"architecture": {"n_filters": 32, "n_layers": 3}}
    if with_lora:
        m.add_lora(np.eye(3, dtype=np.float32), coverages=np.ones((3, 1), np.float32),
                   rank=2, hidden_dim=8)
        with torch.no_grad():
            for p in m.parameters():
                if p.requires_grad:
                    p.add_(0.01 * torch.randn_like(p))
        manifest["lora"] = {"rank": 2, "hidden_dim": 8, "targets": list(s2p.LORA_TARGETS)}
    md5 = s2p.save_pretrained(m, tmp_path / "toy", manifest=manifest)

    monkeypatch.setitem(_pretrained.PRETRAINED, "toy", {"revision": "local", "files": md5})
    monkeypatch.setattr(_pretrained, "_fetch", lambda name, f: str(tmp_path / name / f))
    loaded = s2p.load_pretrained("toy")

    x = torch.nn.functional.one_hot(torch.randint(0, 4, (2, m.dna_len)), 4).permute(0, 2, 1).float()
    cells = torch.tensor([0, 2]) if with_lora else None
    with torch.no_grad():
        a, b = m.eval()(x, cells), loaded(x, cells)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert set(md5) == {"model.safetensors", "manifest.json"}
