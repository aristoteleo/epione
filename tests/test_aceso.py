"""ACESO — torch-free invariants (panel sampling, thinning, tokenisation order, metrics).

The GPU stages (EpiAgent fine-tuning, scGPT encoder, scooby decoder, joint training) are
exercised by the smoke driver in the analysis repository; here only the pieces that define the
data the models see are tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from epione.models import aceso


def test_sentences_from_tfidf_order():
    T = sp.csr_matrix(np.array([[0.0, 3.0, 1.0, 0.0, 2.0], [5.0, 0.0, 0.0, 4.0, 0.0]]))
    s = aceso.sentences_from_tfidf(T)
    assert s == ["[5,8,6]", "[4,7]"]        # descending TF-IDF, +4 offset


def test_thin_counts_reduces_to_depth():
    rng = np.random.default_rng(0)
    M = sp.csr_matrix(rng.poisson(5, size=(200, 50)).astype(np.float32))
    T = aceso.thin_counts(M, np.full(200, 40.0), np.random.default_rng(1))
    tot = np.asarray(T.sum(1)).ravel()
    assert (T.toarray() <= M.toarray()).all()
    assert abs(tot.mean() - 40) < 2
    assert (T.data > 0).all()                # zeros eliminated


def _space(n_genes=300, target=50, pool=200):
    return aceso.PanelSpace(genes=np.array([f"g{i}" for i in range(n_genes)], dtype=object),
                            vocab_ids=np.arange(n_genes) + 10, target=np.arange(target),
                            pool=np.arange(n_genes - pool, n_genes))


@pytest.mark.parametrize("mode", ["fixed", "random", "mixed"])
def test_panel_sampler_modes(mode):
    sp_ = _space()
    s = aceso.PanelSampler(sp_, target_depths=np.full(10, 100.0), mode=mode, p_target=0.5, min_genes=20, max_genes=80)
    rng = np.random.default_rng(0)
    draws = [s.draw(rng) for _ in range(400)]
    is_target = np.array([np.array_equal(c, sp_.target) for c, _ in draws])
    if mode == "fixed":
        assert is_target.all() and all(sc == 1.0 for _, sc in draws)
    elif mode == "random":
        assert not is_target.any()
    else:
        assert 0.4 < is_target.mean() < 0.6
    for c, sc in draws:
        if not np.array_equal(c, sp_.target):
            assert 20 <= len(c) < 80 and np.isin(c, sp_.pool).all() and np.all(np.diff(c) > 0)
            assert sc > 0


def test_mgi_orthologs_one_to_one(tmp_path):
    rows = [(1, "mouse, laboratory", "A1"), (1, "human", "HA1"),          # 1:1
            (2, "mouse, laboratory", "B1"), (2, "mouse, laboratory", "B2"), (2, "human", "HB"),  # 2:1 -> dropped
            (3, "mouse, laboratory", "C1")]                                 # no human -> dropped
    df = pd.DataFrame(rows, columns=["DB Class Key", "Common Organism Name", "Symbol"])
    p = tmp_path / "hom.rpt"; df.to_csv(p, sep="\t", index=False)
    assert aceso.mgi_orthologs(str(p)) == {"A1": "HA1"}


def test_knn_smoothed_pearson_identity():
    rng = np.random.default_rng(0)
    obs = rng.poisson(1.0, size=(500, 30)).astype(float)
    nb = np.arange(30)[:, None]                 # k = 1: each cell's own profile
    r = aceso.knn_smoothed_pearson(obs, obs, nb)
    assert np.allclose(r, 1.0)


def test_pseudobulk_pearson_perfect_and_differential():
    rng = np.random.default_rng(0)
    obs = rng.poisson(2.0, size=(400, 60)).astype(float)
    g = np.repeat(["a", "b", "c"], 20)
    out = aceso.pseudobulk_pearson(obs, obs, g)
    assert all(abs(v - 1) < 1e-9 for v in out["per_group"].values())
    assert abs(out["differential"] - 1) < 1e-9


def test_cell_index_embedding_exact():
    e = aceso.cell_index_embedding(100_000)
    assert e.dtype == np.float32 and e.shape == (100_000, 1)
    assert np.array_equal(e[:, 0].astype(np.int64), np.arange(100_000))


def test_expert_lora_zero_init_and_per_expert_updates():
    torch = pytest.importorskip("torch")
    from epione.models.aceso._molora import ExpertLoRA
    torch.manual_seed(0)
    for base, x in [(torch.nn.Linear(6, 5), torch.randn(3, 7, 6)), (torch.nn.Conv1d(6, 5, 1), torch.randn(3, 6, 7))]:
        m = ExpertLoRA(base, n_experts=3, rank=2, alpha=2.0)
        assert torch.allclose(m(x), base(x))                       # zero-initialised B: equals the base layer
        with torch.no_grad():
            m.expert_lora_B.normal_()
        same = x[:1].expand(3, *x.shape[1:])
        y = m(same)
        assert not torch.allclose(y[0], y[1])                      # expert k updates batch element k
        ref = base(same[:1]) + m.scale * (
            (same[:1] @ m.expert_lora_A[0].T) @ m.expert_lora_B[0].T if m.kind == "linear"
            else torch.einsum("or,ri,bil->bol", m.expert_lora_B[0], m.expert_lora_A[0], same[:1]))
        assert torch.allclose(y[:1], ref, atol=1e-5)


def test_sliced_lora_adapts_only_given_rows_and_expert_rows():
    torch = pytest.importorskip("torch")
    from epione.models.aceso._backbone import SlicedLoRA
    from epione.models.aceso._molora import ExpertLoRA
    torch.manual_seed(0)
    base = torch.nn.Linear(6, 10)
    rows = torch.tensor([0, 1, 2, 7, 8, 9])                 # e.g. q and v rows of a fused qkv
    m = SlicedLoRA(base, rows)
    x = torch.randn(3, 4, 6)
    assert torch.allclose(m(x), base(x))                    # zero-init B
    with torch.no_grad():
        m.lora_B.normal_()
    d = m(x) - base(x)
    other = torch.tensor([3, 4, 5, 6])
    assert torch.allclose(d[..., other], torch.zeros_like(d[..., other]))
    assert d[..., rows].abs().sum() > 0
    e = ExpertLoRA(m, n_experts=3, rank=2, alpha=2.0, rows=rows)   # experts on top of the shared sliced LoRA
    assert e.expert_lora_B.shape == (3, len(rows), 2)
    assert torch.allclose(e(x), m(x))
    with torch.no_grad():
        e.expert_lora_B.normal_()
    de = e(x) - m(x)
    assert torch.allclose(de[..., other], torch.zeros_like(de[..., other])) and not torch.allclose(de[0], de[1])


def test_alphagenome_locon_keeps_length_and_starts_at_zero():
    torch = pytest.importorskip("torch")
    pytest.importorskip("alphagenome_pytorch")
    from alphagenome_pytorch.convolutions import StandardizedConv1d
    from epione.models.aceso._alphagenome import Locon
    torch.manual_seed(0)
    x = torch.randn(2, 6, 50)
    for conv in [StandardizedConv1d(6, 7, 5), torch.nn.Conv1d(6, 7, 15, padding="same"), torch.nn.Conv1d(6, 7, 1)]:
        m = Locon(conv, rank=3, alpha=3)
        assert torch.allclose(m(x), conv(x))                 # zero-initialised up conv
        with torch.no_grad():
            m.locon_up.weight.normal_()
        assert m(x).shape == conv(x).shape                   # same output length as the original
