"""seq2PRINT sequence TF binding scores (scPrinter's seq_tfbs_seq2print)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from epione.models import seq2print as s2p  # noqa: E402
from epione.models.seq2print import _tfbs  # noqa: E402

SMALL = dict(n_filters=16, n_layers=8)        # dna_len 1840, as the real model


def one_hot(n, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 4, (n, L), generator=g)
    return torch.nn.functional.one_hot(idx, 4).permute(0, 2, 1).float()


def small_model(seed=0):
    torch.manual_seed(seed)
    m = s2p.build_seq2print(**SMALL).eval()
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.running_mean.normal_(0, 0.1)
            mod.running_var.uniform_(0.5, 2.0)
    return m


# --- scPrinter, verbatim (scprinter/seq/interpretation/attributions.py and
# scripts/generate_TFBS_bigwig.py at 6edde74, MIT, Copyright (c) 2023 Broad
# Institute), the references the ports are checked against ---------------
class FootprintScaling(torch.nn.Module):
    def forward(self, x):
        return s2p.zscore2pval(x)


class JustSumWrapper(torch.nn.Module):
    def __init__(self, model, nth_output=None, specific_pos=None, weight=None, threshold=None):
        super().__init__()
        self.model = model
        self.nth_output = nth_output
        self.specific_pos = specific_pos
        self.weight = weight
        self.threshold = threshold
        self.relu = torch.nn.ReLU()
        self.scaling = FootprintScaling()

    def forward(self, X, *args, **kwargs):
        logits = self.model(X, *args, **kwargs)[0]
        if self.nth_output is not None:
            logits = logits[:, self.nth_output]
        if self.specific_pos is not None:
            logits = logits[:, :, self.specific_pos]
        logits = logits.reshape(X.shape[0], -1)
        logits = logits - torch.mean(logits, dim=-1, keepdims=True)
        logits = self.scaling(logits)
        if self.threshold is not None:
            logits = self.relu(logits - self.threshold)
        if self.weight is not None:
            logits = logits * self.weight
        y = logits
        return (y).sum(axis=-1, keepdims=True)


@torch.no_grad()
def forward_pass_model(feats, model):
    feats = torch.as_tensor(feats).float()[:, None, :]
    feats_rev = torch.flip(feats, [2])
    pred = torch.stack(
        [
            torch.sigmoid(model(feats)).cpu().detach()[:, 0, :],
            torch.flip(torch.sigmoid(model(feats_rev)).cpu().detach()[:, 0, :], [1]),
        ],
        axis=0,
    )
    pred = torch.mean(pred, 0)
    return pred
# -------------------------------------------------------------------------


def test_footprint_tf_head_equals_scprinter_wrapper():
    pytest.importorskip("grelu")
    from epione.models import grelu as eg

    m = small_model()
    x = one_hot(3, m.dna_len, seed=1)
    lm = eg.wrap(m, head="footprint_tf")
    ref = JustSumWrapper(m, nth_output=torch.arange(0, 30), threshold=0.301)
    with torch.no_grad():
        ours = lm(x).ravel()
        theirs = ref(x).ravel()
    torch.testing.assert_close(ours, theirs, rtol=0, atol=0)
    assert (theirs > 0).all()


def _nets_or_skip():
    try:
        return {h: _tfbs._load_net(h, torch.device("cpu")) for h in _tfbs.SEQ_TFBS_HEADS}
    except Exception as exc:  # offline
        pytest.skip(f"TFBS networks not available: {exc}")


def test_tfbs_network_pass_equals_scprinter():
    nets = _nets_or_skip()
    feats = np.random.default_rng(0).normal(size=(5, 1200)).astype(np.float32)
    for net in nets.values():
        ours = _tfbs._run_net(net, feats, torch.device("cpu"))
        theirs = forward_pass_model(feats, net).numpy()[..., 100:-100]
        assert ours.shape == (5, 800)
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-7)


def test_tf_binding_lora_states_and_bulk():
    pytest.importorskip("grelu")
    _nets_or_skip()
    rng = np.random.default_rng(0)
    m = small_model()
    m.add_lora(rng.normal(size=(3, 5)).astype(np.float32), rank=2, hidden_dim=8)
    torch.manual_seed(1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "B_embedding" in n:
                p.add_(torch.randn_like(p) * 0.05)
    m.eval()
    X = one_hot(4, m.dna_len, seed=2).numpy()
    regions = pd.DataFrame({"chrom": "chr1", "summit": [10_000, 20_000, 30_000, 40_000]})
    res = s2p.tf_binding(m, X, states=[0, 2], regions=regions, n_shuffles=4, batch_size=8,
                         keep_attributions=True, device="cpu")
    assert res.score.shape == (2, 4, 800)
    assert ((res.score >= 0) & (res.score <= 1)).all()
    assert not np.allclose(res.score[0], res.score[1])          # states differ
    # normalised attributions: median 0, (q95 - q05) = 1 over the central 800 bp, pooled
    a = res.attributions["coverage"][..., 520:-520]
    assert abs(np.median(a)) < 1e-6
    assert np.quantile(a, 0.95) - np.quantile(a, 0.05) == pytest.approx(1, abs=1e-5)
    # the state's collapsed model gives the same scores with the same normalisation
    res1 = s2p.tf_binding(m.collapse(2), X, n_shuffles=4, batch_size=8, norm=res.norm,
                          device="cpu")
    np.testing.assert_allclose(res1.score[0], res.score[1], atol=1e-5)
    # site scores read the track
    sites = pd.DataFrame({"window": [0, 3], "start": [10_000 - 5, 40_000 + 100],
                          "end": [10_000 + 5, 40_000 + 110]})
    ss = res.site_scores(sites)
    assert ss.shape == (2, 2)
    assert ss.iloc[0, 0] == pytest.approx(res.score[0, 0, 395:405].max())


def test_tf_binding_needs_states_for_lora():
    m = small_model()
    m.add_lora(np.zeros((2, 3), np.float32), rank=2, hidden_dim=4)
    with pytest.raises(ValueError):
        s2p.tf_binding(m, one_hot(1, m.dna_len).numpy())


def test_one_hot_regions(tmp_path):
    pytest.importorskip("pyfaidx")
    fa = tmp_path / "g.fa"
    fa.write_text(">chr1\n" + "ACGTN" * 40 + "\n")
    X = s2p.one_hot_regions(fa, pd.DataFrame({"chrom": ["chr1", "chr1"], "summit": [10, 100]}),
                            length=20)
    assert X.shape == (2, 4, 20)
    # window [0, 20) for summit 10: ACGTN ACGTN ...
    assert list(X[0].argmax(0)[:4]) == [0, 1, 2, 3] and X[0][:, 4].sum() == 0
    # summit 10 with length 20 starts at 0; a summit near the start pads with N
    Y = s2p.one_hot_regions(fa, pd.DataFrame({"chrom": ["chr1"], "summit": [3]}), length=20)
    assert Y[0][:, :7].sum() == 0 and Y[0][:, 7].argmax() == 0
