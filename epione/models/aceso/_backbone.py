"""Sequence backbones for the ACESO decoders: Borzoi or Flashzoi.

Both are Borzoi (borzoi-pytorch ``Borzoi``; same conv tower, U-net and 8 x 1536 transformer); they
differ only in the attention layer:
  borzoi   ``johahi/borzoi-replicate-0``  relative-position attention with separate to_q / to_k / to_v
  flashzoi ``johahi/flashzoi-replicate-0`` (Hingerl et al.; ``flashed`` config) flash_attn MHA with
           rotary embeddings, grouped-query (8 query / 4 key-value heads) and one fused Wqkv
           projection (rows: q 0:1536, k 1536:2304, v 2304:3072). Needs ``flash_attn`` and bf16/fp16.

Upstream scooby's LoRA (``get_lora``) targets ``to_q``, ``to_v``, the feed-forward linears and the
non-separable conv layers. On Flashzoi its ``to_q|to_v`` pattern matches nothing, so the attention
adapters are added here as two separate LoRAs on Wqkv, one on the q rows and one on the v rows
(:class:`SlicedLoRA`, rank 8, alpha 8 = peft defaults used upstream) — the same adapted projections,
each with its own adapter, as upstream on Borzoi. No
other part of the model differs. scooby upstream (dbfe6cd) has no Flashzoi support; this mapping is
ours.
"""
from __future__ import annotations

import torch
import torch.nn as nn

BACKBONES = {"borzoi": "johahi/borzoi-replicate-0", "flashzoi": "johahi/flashzoi-replicate-0",
             "alphagenome": "gtca/alphagenome_pytorch"}     # AlphaGenome: see _alphagenome.py


def is_flashed(core) -> bool:
    return bool(getattr(core.config, "flashed", False))


def q_rows(mha) -> torch.Tensor:
    """output rows of flash_attn MHA.Wqkv that hold q"""
    return torch.arange(0, mha.num_heads * mha.head_dim)


def v_rows(mha) -> torch.Tensor:
    """output rows of flash_attn MHA.Wqkv that hold v (after k; grouped-query kv heads)"""
    q, kv = mha.num_heads * mha.head_dim, mha.num_heads_kv * mha.head_dim
    return torch.arange(q + kv, q + 2 * kv)


def unwrap_linear(m):
    """the nn.Linear / nn.Conv1d under any stack of adapter wrappers (peft base_layer, .base)"""
    while not isinstance(m, (nn.Linear, nn.Conv1d)):
        m = getattr(m, "base_layer", None) or getattr(m, "original_layer", None) or getattr(m, "base")
    return m


class SlicedLoRA(nn.Module):
    """LoRA on a subset of a Linear's output rows (any batch size): y = W x + s * scatter_rows(B A x)"""

    def __init__(self, base: nn.Linear, rows: torch.Tensor, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.register_buffer("rows", rows.long(), persistent=False)
        self.lora_A = nn.Parameter(torch.empty(rank, unwrap_linear(base).in_features))
        self.lora_B = nn.Parameter(torch.zeros(len(rows), rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)          # peft's lora_A init

    def forward(self, x):
        y = self.base(x)
        d = (x @ self.lora_A.to(x.dtype).T) @ self.lora_B.to(x.dtype).T
        return y.index_add(-1, self.rows.to(y.device), self.scale * d.to(y.dtype))


def attention_modules(core, blocks):
    """[(attention module, kind)] for transformer blocks ``blocks``; kind 'borzoi' or 'flashzoi'"""
    out = []
    for i in blocks:
        attn = core.transformer[i][0].fn[1]
        out.append((attn, "flashzoi" if hasattr(attn, "mha") else "borzoi"))
    return out


def add_attention_lora(model, train: bool = True, rank: int = 8, alpha: float = 8.0) -> int:
    """Flashzoi only: separate q and v LoRAs on every block's Wqkv (what upstream get_lora does to to_q/to_v)."""
    core = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    n = 0
    for attn, kind in attention_modules(core, range(len(core.transformer))):
        if kind != "flashzoi":
            continue
        base = attn.mha.Wqkv
        for p in base.parameters():
            p.requires_grad = False
        dev = base.weight.device
        lq = SlicedLoRA(base, q_rows(attn.mha), rank, alpha).to(dev)            # own adapter for q
        lv = SlicedLoRA(lq, v_rows(attn.mha), rank, alpha).to(dev)              # and for v
        for l in (lq, lv):
            l.lora_A.requires_grad = l.lora_B.requires_grad = train
        attn.mha.Wqkv = lv
        n += 1
    return n
