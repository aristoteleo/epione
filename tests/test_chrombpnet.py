"""epione.models.chrombpnet: CPU, no downloads, no TensorFlow."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from epione.models import chrombpnet as cbp  # noqa: E402

# A small but valid geometry: 2 dilated layers, 64-bp input.
SMALL = dict(input_len=128, output_len=40, conv1_kernel_size=5, profile_kernel_size=7)


def _small(filters=8, n=2, **kw):
    return cbp.BPNet(filters=filters, n_dil_layers=n, **{**SMALL, **kw})


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

def test_official_geometry():
    for kind in ("bias", "chrombpnet"):
        m = cbp.build_bpnet(kind)
        p, c = m(torch.zeros(2, 4, 2114))
        assert p.shape == (2, 1000) and c.shape == (2, 1)


def test_bad_geometry_raises():
    with pytest.raises(ValueError):
        cbp.BPNet(filters=4, n_dil_layers=2, input_len=60, output_len=60,
                  conv1_kernel_size=5, profile_kernel_size=7)


def test_chrombpnet_combines_logits_and_logsumexp():
    torch.manual_seed(0)
    b, a = _small(), _small()
    m = cbp.ChromBPNet(b, a).eval()
    x = torch.randint(0, 2, (3, 4, 128)).float()
    p, c = m(x)
    pa, ca = a(x)
    pb, cb = b(x)
    assert torch.allclose(p, pa + pb)
    assert torch.allclose(c, torch.log(torch.exp(ca) + torch.exp(cb)), atol=1e-6)
    assert not any(q.requires_grad for q in m.bias.parameters())


def test_precise_profile_matches_float64():
    torch.manual_seed(1)
    m = cbp.glorot_init_(_small(filters=16))
    x = torch.randint(0, 2, (4, 4, 128)).float()
    with torch.no_grad():
        p32 = m(x)[0].double()
        import copy
        p64 = copy.deepcopy(m).double()(x.double())[0]
    assert (p32 - p64).abs().max() < 1e-5


# ---------------------------------------------------------------------------
# Keras h5 loader (files written here with h5py in the official layout)
# ---------------------------------------------------------------------------

def _conv_cfg(name, filters, k, dil=1, inbound=None):
    return {"class_name": "Conv1D", "config": {"name": name, "filters": filters,
            "kernel_size": [k], "dilation_rate": [dil]},
            "inbound_nodes": [[[inbound, 0, 0, {}]]] if inbound else []}


def _layers(net, prefix=""):
    L = [{"class_name": "InputLayer", "config": {"name": "sequence",
          "batch_input_shape": [None, net.input_len, 4]}, "inbound_nodes": []}]
    L.append(_conv_cfg(prefix + "bpnet_1st_conv", net.filters, net.conv1.kernel_size[0]))
    for i in range(1, net.n_dil_layers + 1):
        L.append(_conv_cfg(f"{prefix}bpnet_{i}conv", net.filters, 3, 2 ** i))
    L.append(_conv_cfg(prefix + "prof_out_precrop", 1, net.prof_conv.kernel_size[0]))
    crop = (net._prof_len - net.output_len) // 2
    L.append({"class_name": "Cropping1D", "config": {"name": prefix + "crop_prof",
              "cropping": [crop, crop]},
              "inbound_nodes": [[[prefix + "prof_out_precrop", 0, 0, {}]]]})
    L.append({"class_name": "Dense", "config": {"name": prefix + "logcount_predictions"},
              "inbound_nodes": []})
    return L


def _write_weights(grp, net, prefix=""):
    def conv(name, mod):
        g = grp.create_group(name).create_group(name)
        g["kernel:0"] = np.transpose(mod.weight.detach().numpy(), (2, 1, 0))
        g["bias:0"] = mod.bias.detach().numpy()
    conv(prefix + "bpnet_1st_conv", net.conv1)
    for i, m in enumerate(net.dil_convs, 1):
        conv(f"{prefix}bpnet_{i}conv", m)
    conv(prefix + "prof_out_precrop", net.prof_conv)
    g = grp.create_group(prefix + "logcount_predictions").create_group(prefix + "logcount_predictions")
    g["kernel:0"] = net.count_dense.weight.detach().numpy().T
    g["bias:0"] = net.count_dense.bias.detach().numpy()


def test_load_single_bpnet_h5(tmp_path):
    torch.manual_seed(2)
    net = cbp.glorot_init_(_small())
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    f = tmp_path / "bias.h5"
    with h5py.File(f, "w") as h:
        h.attrs["model_config"] = json.dumps({"class_name": "Functional",
                                              "config": {"layers": _layers(net)}})
        _write_weights(h.create_group("model_weights"), net)
    m = cbp.load_keras_h5(f)
    x = torch.randint(0, 2, (3, 4, 128)).float()
    with torch.no_grad():
        assert torch.allclose(m(x)[0], net(x)[0], atol=1e-6)
        assert torch.allclose(m(x)[1], net(x)[1], atol=1e-6)


def test_load_full_chrombpnet_h5(tmp_path):
    torch.manual_seed(3)
    b, a = cbp.glorot_init_(_small(filters=6)), cbp.glorot_init_(_small(filters=10))
    f = tmp_path / "full.h5"
    with h5py.File(f, "w") as h:
        layers = [
            {"class_name": "InputLayer", "config": {"name": "sequence",
             "batch_input_shape": [None, 128, 4]}, "inbound_nodes": []},
            {"class_name": "Functional", "config": {"name": "model", "layers": _layers(b)}},
            {"class_name": "Functional", "config": {"name": "model_wo_bias",
             "layers": _layers(a, "wo_bias_")}},
        ]
        h.attrs["model_config"] = json.dumps({"class_name": "Functional",
                                              "config": {"layers": layers}})
        wg = h.create_group("model_weights")
        _write_weights(wg.create_group("model"), b)
        _write_weights(wg.create_group("model_wo_bias"), a, "wo_bias_")
    m = cbp.load_keras_h5(f)
    assert isinstance(m, cbp.ChromBPNet) and m.accessibility.filters == 10
    x = torch.randint(0, 2, (2, 4, 128)).float()
    with torch.no_grad():
        assert torch.allclose(m(x)[0], cbp.ChromBPNet(b, a)(x)[0], atol=1e-6)


# ---------------------------------------------------------------------------
# data recipe
# ---------------------------------------------------------------------------

def test_one_hot_matches_upstream_convention():
    x = cbp.one_hot_encode(["ACGTN", "acgtx"])
    assert x.shape == (2, 5, 4)
    assert (x[0, :4] == np.eye(4)).all() and x[0, 4].sum() == 0
    assert (x[1] == x[0]).all()


def test_multinomial_nll_matches_scipy():
    from scipy.stats import multinomial
    rng = np.random.RandomState(0)
    logits = rng.randn(3, 6)
    counts = rng.poisson(3, (3, 6)).astype(float)
    ours = cbp.multinomial_nll(torch.tensor(logits), torch.tensor(counts)).item()
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    ref = -np.mean([multinomial.logpmf(c, c.sum(), q) for c, q in zip(counts, p)])
    assert ours == pytest.approx(ref, rel=1e-9)


def test_batches_crop_revcomp_and_negatives():
    rng = np.random.RandomState(0)
    ps = np.eye(4, dtype=np.int8)[rng.randint(0, 4, (20, 30))]
    pc = rng.poisson(1, (20, 14)).astype(float)
    ns = np.eye(4, dtype=np.int8)[rng.randint(0, 4, (50, 20))]
    nc = rng.poisson(1, (50, 4)).astype(float)
    b = cbp.ChromBPNetBatches(ps, pc, ns, nc, inputlen=20, outputlen=4, batch_size=8,
                              negative_sampling_ratio=0.5, add_revcomp=True, shuffle=True,
                              rng=np.random.RandomState(1))
    assert b.seqs.shape == (30, 20, 4) and b.cts.shape == (30, 4)  # 20 peaks + 10 negatives
    assert len(b) == 4
    X, y, yl = b[0]
    assert np.allclose(yl, np.log1p(y.sum(-1, keepdims=True)))
    b.on_epoch_end()
    assert b.cur_seqs.shape == (30, 20, 4)


def test_dinuc_shuffle_preserves_dinucleotides():
    rng = np.random.RandomState(0)
    s = cbp.one_hot_encode(["".join(rng.choice(list("ACGT"), 200))])[0]
    sh = cbp.dinuc_shuffle(s, 3, np.random.RandomState(1))

    def dinucs(x):
        t = x.argmax(1)
        return np.bincount(t[:-1] * 4 + t[1:], minlength=16)
    for k in range(3):
        assert (dinucs(sh[k]) == dinucs(s)).all()
        assert sh[k].argmax(1)[0] == s.argmax(1)[0]


class _ArraySignal:
    def __init__(self, arr):
        self.arr = arr

    def chroms(self):
        return {c: len(v) for c, v in self.arr.items()}

    def values(self, c, a, b):
        return self.arr[c][a:b].astype(float)


@pytest.fixture
def toy_genome(tmp_path):
    rng = np.random.RandomState(0)
    chroms = {c: "".join(rng.choice(list("ACGT"), 20000)) for c in ("chr1", "chr2", "chr3")}
    fa = tmp_path / "toy.fa"
    with open(fa, "w") as f:
        for c, s in chroms.items():
            f.write(f">{c}\n{s}\n")
    import pysam
    pysam.faidx(str(fa))
    sig = {c: rng.poisson(0.05, len(s)).astype(float) for c, s in chroms.items()}
    return fa, chroms, sig


FOLD = {"train": ["chr1"], "valid": ["chr2"], "test": ["chr3"]}


def _peaks(rng, n=6):
    rows = []
    for c in ("chr1", "chr2", "chr3"):
        for s in rng.randint(300, 19500, n):
            rows.append((c, int(s), int(s) + 100, ".", ".", ".", ".", ".", ".", 50))
    from epione.models.chrombpnet._data import NARROWPEAK_SCHEMA
    return pd.DataFrame(rows, columns=NARROWPEAK_SCHEMA)


def test_gc_nonpeaks_too_few_candidates_raises(toy_genome):
    fa, _, sig = toy_genome
    peaks = _peaks(np.random.RandomState(0), n=6)
    with pytest.raises(ValueError, match="candidate background bins"):
        cbp.gc_matched_nonpeaks(peaks, fa, FOLD, {c: len(v) for c, v in sig.items()},
                                inputlen=200, stride=5000, neg_to_pos_ratio_train=50)


def test_bias_hyperparams_and_gc_nonpeaks(toy_genome):
    fa, chroms, sig = toy_genome
    rng = np.random.RandomState(0)
    peaks = _peaks(rng)
    for c, row in peaks.groupby("chr"):
        for s in row["start"] + row["summit"]:
            sig[c][s - 10:s + 10] += 5  # peaks are enriched
    signal = _ArraySignal(sig)
    non = cbp.gc_matched_nonpeaks(peaks, fa, FOLD, signal.chroms(), inputlen=200,
                                  stride=100, neg_to_pos_ratio_train=2, seed=1)
    assert len(non) == 2 * 6 + 2 * 6 + 6
    assert (non["summit"] == 100).all()
    hp = cbp.bias_hyperparams(peaks, non, signal, FOLD, inputlen=200, outputlen=40)
    assert hp["params"]["counts_loss_weight"] >= 1.0
    assert set(hp["nonpeaks"]["chr"]) <= {"chr1", "chr2", "chr3"}
    cp = cbp.chrombpnet_hyperparams(peaks, non, signal, FOLD, inputlen=200, outputlen=40,
                                    max_jitter=20, rng=np.random.RandomState(0))
    assert cp["params"]["negative_sampling_ratio"] == 0.1


def test_train_bias_smoke(toy_genome):
    fa, chroms, sig = toy_genome
    rng = np.random.RandomState(0)
    peaks = _peaks(rng)
    for c, row in peaks.groupby("chr"):
        for s in row["start"] + row["summit"]:
            sig[c][s - 10:s + 10] += 5
    signal = _ArraySignal(sig)
    # official kernels (21 and 75) with 2 dilated layers: 256 in -> 150 profile
    non = cbp.gc_matched_nonpeaks(peaks, fa, FOLD, signal.chroms(), inputlen=256,
                                  stride=128, seed=1)
    res = cbp.train_bias(peaks, non, fa, signal, FOLD, inputlen=256, outputlen=100,
                         filters=4, n_dil_layers=2, epochs=2, device="cpu", verbose=False)
    assert len(res.history) == 2 and res.best_epoch in (0, 1)


def test_chrombpnet_bias_backend_stitches_windows(toy_genome):
    from epione.tl import ChromBPNetBias, tn5_bias
    fa, _, _ = toy_genome
    torch.manual_seed(0)
    net = cbp.glorot_init_(_small(filters=4))
    bias = tn5_bias(fa, method="chrombpnet", model=net)
    assert isinstance(bias, ChromBPNetBias) and bias.shift == (0, 0)
    t = bias.track("chr1", 1000, 1100)
    assert t.shape == (100,) and (t > 0).all()
    # equals a direct prediction over the same windows
    import pysam
    fa_ = pysam.FastaFile(str(fa))
    pad = (128 - 40) // 2
    seqs = [fa_.fetch("chr1", w - pad, w - pad + 128) for w in range(1000, 1100, 40)]
    p, c = cbp.predict(net, cbp.one_hot(seqs))
    ref = (cbp.softmax(p.astype(np.float64)) * np.exp(c.astype(np.float64))).reshape(-1)[:100]
    assert np.array_equal(t, ref.astype(np.float32))
    # N-padded past the chromosome start
    assert bias.track("chr1", 0, 40).shape == (40,)


def test_ism_shape():
    torch.manual_seed(0)
    net = cbp.glorot_init_(_small(filters=4))
    s = cbp.one_hot_encode(["".join(np.random.RandomState(0).choice(list("ACGT"), 128))])[0]
    out = cbp.ism(net, s, positions=[60, 61, 62])
    assert out.shape == (3, 4)
    for i, p in enumerate([60, 61, 62]):
        assert out[i, s[p].argmax()] == pytest.approx(0.0, abs=1e-6)
