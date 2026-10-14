"""ACESO decoder variant — per-cell non-linear read-out (FiLM head) on scooby's shared trunk.

scooby reads each cell out of one shared Borzoi embedding E (1920 x bins) with a linear 1x1 conv,
y_c = softplus(w_c . E + b_c), (w_c, b_c) from the cell state by upstream ``cell_state_to_conv``.
The FiLM head adds a per-cell non-linear term over the same shared features (a light decoder on a
shared embedding, as SAM's mask decoder; modulation as DiT's adaLN):

  u     = LayerNorm(E)                                    (per bin, shared)
  h_c   = GELU( (W1 u) * (1 + gamma(z_c)) + beta(z_c) )    W1: 1920 -> hidden, shared
  y_c   = softplus( w_c . E + b_c + v(z_c) . h_c )

gamma, beta, v are cell-state generators whose last layer is zero-initialised, so at step 0 the
model equals scooby (v = 0); scooby is the special case v = 0. The trunk is scooby's (upstream
LoRA), run once per sequence. Only ATAC (``n_tracks = 1``).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FiLMConfig:
    hidden: int = 256
    gen_hidden: int = 256


class FiLMHeadDecoder(nn.Module):
    """same call interface as scooby / AcesoDecoder"""

    def __init__(self, scooby_peft: nn.Module, cell_emb_dim: int, config: FiLMConfig = FiLMConfig()):
        super().__init__()
        self.scooby, self.config = scooby_peft, config
        core = self.core
        assert core.n_tracks == 1 and not core.use_transform_borzoi_emb
        D, H = core.embedding_dim, config.hidden
        self.head_norm = nn.LayerNorm(D)
        self.head_in = nn.Linear(D, H)
        self.cell_state_to_film = nn.Sequential(nn.Linear(cell_emb_dim, config.gen_hidden), nn.GELU(),
                                                nn.Linear(config.gen_hidden, 3 * H))   # gamma, beta, v
        nn.init.zeros_(self.cell_state_to_film[-1].weight); nn.init.zeros_(self.cell_state_to_film[-1].bias)

    @property
    def core(self):
        m = self.scooby
        return m.base_model.model if hasattr(m, "base_model") and hasattr(m.base_model, "model") else m

    def forward_seq_to_emb(self, sequence):
        """[1, 4, L] -> [1, 1920, bins] (scooby's trunk)"""
        return self.core.forward_seq_to_emb(sequence)

    def forward_cell_embs_only(self, cell_emb):
        """[1, n, d] -> (w [1, n, 1920, 1], b [1, n], gamma, beta, v [1, n, H] each)"""
        w, b = self.core.forward_cell_embs_only(cell_emb)
        gamma, beta, v = self.cell_state_to_film(cell_emb).chunk(3, dim=-1)
        return w, b, gamma, beta, v

    def forward_convs_on_emb(self, seq_emb, w, b, gamma, beta, v, bins_to_predict=None):
        """seq_emb [1, 1920, bins] -> [1, bins, n]"""
        E = seq_emb[0] if bins_to_predict is None else seq_emb[0][:, bins_to_predict]       # [D, B]
        lin = torch.einsum("db,nd->nb", E, w[0, :, :, 0].to(E.dtype)).float() + b[0].float()[:, None]
        a = self.head_in(self.head_norm(E.transpose(0, 1)))                                # [B, H] shared
        h = F.gelu(a[None] * (1 + gamma[0][:, None, :].to(a.dtype)) + beta[0][:, None, :].to(a.dtype))  # [n, B, H]
        nl = torch.einsum("nbh,nh->nb", h, v[0].to(h.dtype)).float()
        return F.softplus(lin + nl).unsqueeze(0).permute(0, 2, 1)

    def forward(self, sequence, cell_emb):
        p = self.forward_cell_embs_only(cell_emb)
        return self.forward_convs_on_emb(self.forward_seq_to_emb(sequence), *p)

    def param_groups(self, lr: float, cellstate_lr: float, weight_decay: float):
        """upstream ``add_weight_decay`` grouping; cell-state generators in the cell-state (high-lr) group"""
        high, no_decay, decay = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                no_decay.append(p)
            elif "cell_state_to_" in name:
                high.append(p)
            else:
                decay.append(p)
        return [{"params": high, "weight_decay": weight_decay, "lr": cellstate_lr},
                {"params": no_decay, "weight_decay": 0.0, "lr": lr},
                {"params": decay, "weight_decay": weight_decay, "lr": lr}]


def build_film_decoder(cell_emb_dim: int, config: FiLMConfig = FiLMConfig(), device: str = "cuda",
                       checkpoint: str = "johahi/borzoi-replicate-0", train: bool = True) -> FiLMHeadDecoder:
    """scooby (upstream LoRA, ATAC track) + per-cell FiLM head"""
    from ._decoder import build_decoder
    dec = FiLMHeadDecoder(build_decoder(cell_emb_dim, device=device, checkpoint=checkpoint, train=train),
                          cell_emb_dim, config).to(device)
    for n, p in dec.named_parameters():
        if n.startswith(("head_", "cell_state_to_film")):
            p.requires_grad = train
    return dec
