"""seq2PRINT: LoRA collapse and motif delta effects."""
from __future__ import annotations


import numpy as np
import pytest

torch = pytest.importorskip("torch")

from epione.models import seq2print as s2p  # noqa: E402

SMALL = dict(n_filters=32, n_layers=3)


def one_hot(n, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 4, (n, L), generator=g)
    return torch.nn.functional.one_hot(idx, 4).permute(0, 2, 1).float()


def small_model(seed=0):
    torch.manual_seed(seed)
    m = s2p.build_seq2print(**SMALL).eval()
    # random BN statistics so the network is not near-linear
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.running_mean.normal_(0, 0.1)
            mod.running_var.uniform_(0.5, 2.0)
    return m


def lora_model(with_cov, seed=0):
    rng = np.random.default_rng(seed)
    m = small_model(seed)
    m.add_lora(rng.normal(size=(3, 5)).astype(np.float32),
               coverages=rng.normal(size=(3, 1)).astype(np.float32) if with_cov else None,
               rank=2, hidden_dim=8)
    torch.manual_seed(seed + 1)
    with torch.no_grad():                      # B starts at zero; move it
        for n, p in m.named_parameters():
            if "B_embedding" in n or "adjustment" in n:
                p.add_(torch.randn_like(p) * 0.05)
    return m.eval()


# ---------------------------------------------------------------------------
# collapse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("with_cov", [False, True])
def test_collapse_equals_lora_forward(with_cov):
    """In float64, so the comparison tests the algebra, not float32 rounding
    (W x + (x A) B and (W + A B) x round differently)."""
    m = lora_model(with_cov).double()
    x = one_hot(2, m.dna_len).double()
    for s in range(m.n_states):
        c = m.collapse(s).eval()
        assert c.embeddings is None
        with torch.no_grad():
            a = m(x, cells=torch.full((2,), s))
            b = c(x)
        torch.testing.assert_close(a[0], b[0], rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(a[1], b[1], rtol=1e-9, atol=1e-9)


def test_collapse_needs_lora():
    with pytest.raises(ValueError):
        small_model().collapse(0)


# ---------------------------------------------------------------------------
# delta effects
# ---------------------------------------------------------------------------

def test_plant_motif_writes_consensus_at_centre():
    B = one_hot(3, 101, seed=8).numpy()
    pfm = np.array([[9, 0, 0, 0, 1], [0, 9, 0, 0, 1], [0, 0, 9, 0, 1], [0, 0, 0, 9, 1]], float)
    P = s2p.plant_motif(B, pfm)
    pad = (101 - 5) // 2
    for i in range(3):
        bases = P[i].argmax(0)
        assert list(bases[pad:pad + 4]) == [0, 1, 2, 3]
        assert bases[pad + 4] == B[i].argmax(0)[pad + 4]      # tie -> keep background
        assert np.array_equal(P[i][:, :pad], B[i][:, :pad])
        assert np.array_equal(P[i][:, pad + 5:], B[i][:, pad + 5:])
    assert P.sum(1).max() == 1.0


def test_delta_effects_zero_when_motif_already_present_and_lora_state():
    m = lora_model(False)
    B = one_hot(4, m.dna_len, seed=9).numpy()
    pad = (m.dna_len - 6) // 2
    present = B[0, :, pad:pad + 6] * 10                       # motif = background 0's centre
    B[:, :, pad:pad + 6] = B[0, :, pad:pad + 6]
    foot, cov = s2p.delta_effects(m, {"present": present, "A6": np.eye(4)[[0] * 6].T},
                                  B, state=0, amp=False)
    assert foot["present"].shape == (99, 800)
    assert np.abs(foot["present"]).max() == 0 and cov["present"] == 0
    assert np.abs(foot["A6"]).max() > 0
    f2, c2 = s2p.delta_effects(m.collapse(0), {"A6": np.eye(4)[[0] * 6].T}, B, amp=False)
    np.testing.assert_allclose(foot["A6"], f2["A6"], atol=1e-6)
