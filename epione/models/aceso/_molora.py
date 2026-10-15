"""ACESO decoder — state-conditioned mixture of LoRA experts on Borzoi (MoLoRA).

scooby conditions Borzoi on a cell only at the output: one cell-independent (LoRA-tuned) trunk
embedding per sequence is read out by a 1x1 convolution whose weights a hypernetwork predicts from
the cell state. Every cell therefore sees the *same* sequence features. The ACESO decoder lets the
cell state change the sequence features themselves, at a cost that does not grow with the number
of cells:

  lower trunk (shared)   conv tower, U-net skips and the first ``depth - n_expert_blocks``
                         transformer blocks, with upstream scooby's LoRA; run once per sequence;
  upper trunk (experts)  the last ``n_expert_blocks`` transformer blocks, the U-net upsampling path
                         and the final 1x1 convolution, run once per expert (the K experts are the
                         batch dimension). Each expert adds its own low-rank update to the modules
                         upstream LoRA adapts there (to_q, to_v, the feed-forward linears and the
                         non-separable 1x1 convolutions); -> K sequence embeddings E_k;
  gate                   prototype gate (default): pi(z) = softmax(cos(z, p_k) / tau) over K learnable
                         prototypes p_k, initialised by k-means of the cell states (``init_prototypes``);
                         or ``gate="mlp"``: pi(z) = softmax(MLP(z)), last layer zero-initialised;
  cell read-out          upstream ``cell_state_to_conv`` hypernetwork (w_c, b_c) applied to the
                         mixed embedding: y_c = softplus(w_c . sum_k pi_ck E_k + b_c).

Expert updates are zero-initialised, so at step 0 the model equals scooby (all E_k identical) whatever the
gate. The MLP gate also starts uniform and, with identical experts, receives almost no gradient: in the
first Borzoi run its per-cell pi stayed within 0.001 of 1/K for 1,250 steps while the experts diverged (an
average of experts, not routing). The prototype gate routes cell groups to different experts from step 0,
so each expert is trained mainly by its own cells. A load-balancing penalty (squared coefficient of variation of the mean
gate over the cells of a batch; Shazeer et al. 2017) keeps all experts in use (``balance="aux"``). Its weight must
be set relative to the task loss: with scooby's Poisson-multinomial loss (~1e-4) the default 0.01 dominated and drove
every cell's gate to uniform (Flashzoi run 2026-10-15: per-cell entropy ln K from step ~5,800, prototypes orthogonal
to all states). ``balance="bias"`` is the auxiliary-loss-free alternative (Wang et al. 2024, arXiv:2408.15664; used in
DeepSeek-V3): a per-expert bias b_k, not trained by gradient, is added to the gate logits and moved after every
optimizer step by ``bias_rate * sign(1/K - load_k)`` (load = mean gate over the cells of the step). Deviation from the
paper: there the bias only enters the top-K selection; the gate here is dense (no selection), so it enters the weights.
Only ATAC (``n_tracks = 1``) is supported.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoLoRAConfig:
    n_experts: int = 4
    n_expert_blocks: int = 2          # transformer blocks (of 8) in the expert part
    rank: int = 8                     # upstream LoRA default r / alpha
    alpha: float = 8.0
    gate: str = "prototype"           # or "mlp"
    gate_tau: float = 0.1             # prototype gate temperature (cosine similarities in [-1, 1])
    gate_hidden: int = 128
    balance_weight: float = 0.01      # balance="aux" only
    balance: str = "aux"              # "aux" (CV^2 penalty) or "bias" (auxiliary-loss-free bias, Wang et al. 2024)
    bias_rate: float = 1e-3           # balance="bias": update rate u of the paper
    checkpoint_upper: bool = True     # activation checkpointing of the expert part (memory)


class ExpertLoRA(nn.Module):
    """K low-rank updates on a Linear (input [K, L, in]) or 1x1 Conv1d (input [K, in, L]); the k-th
    batch element gets the k-th update. ``base`` may itself be a peft LoRA layer."""

    def __init__(self, base: nn.Module, n_experts: int, rank: int, alpha: float, rows=None):
        super().__init__()
        from ._backbone import unwrap_linear
        inner = unwrap_linear(base)
        if isinstance(inner, nn.Linear):
            self.kind, d_in, d_out = "linear", inner.in_features, inner.out_features
        elif isinstance(inner, nn.Conv1d):
            assert inner.kernel_size == (1,) and inner.groups == 1, "experts only on pointwise convolutions"
            self.kind, d_in, d_out = "conv", inner.in_channels, inner.out_channels
        else:
            raise TypeError(type(inner))
        self.base, self.scale = base, alpha / rank
        # rows: adapt only these output rows (Flashzoi's fused Wqkv: q and v rows)
        self.register_buffer("rows", None if rows is None else rows.long(), persistent=False)
        self.expert_lora_A = nn.Parameter(torch.empty(n_experts, rank, d_in))
        self.expert_lora_B = nn.Parameter(torch.zeros(n_experts, d_out if rows is None else len(rows), rank))
        for k in range(n_experts):                       # peft's init of lora_A
            nn.init.kaiming_uniform_(self.expert_lora_A.data[k], a=5 ** 0.5)

    def forward(self, x):
        y = self.base(x)
        A, B = self.expert_lora_A.to(x.dtype), self.expert_lora_B.to(x.dtype)
        assert x.shape[0] == A.shape[0], "expert modules expect the K experts as the batch dimension"
        if self.kind == "linear":
            d = torch.einsum("klr,kor->klo", torch.einsum("kli,kri->klr", x, A), B)
        else:
            d = torch.einsum("kor,krl->kol", B, torch.einsum("kri,kil->krl", A, x))
        if self.rows is not None:                        # linear only (asserted at build)
            return y.index_add(-1, self.rows.to(y.device), self.scale * d.to(y.dtype))
        return y + self.scale * d


def _expert_targets(scooby, n_expert_blocks: int):
    """(parent, attribute, rows) of every module that gets expert updates: in the last ``n_expert_blocks``
    transformer blocks the attention q and v projections (Borzoi: to_q, to_v; Flashzoi: q rows and v rows
    of the fused Wqkv, one expert set each) and the two feed-forward linears (upstream LoRA targets), and the 1x1 conv_layer of
    upsampling_unet1/0 and final_joined_convs (upstream targets conv_layer except the separable blocks)."""
    from ._backbone import attention_modules, q_rows, v_rows
    t = []
    depth = len(scooby.transformer)
    for (attn, kind), i in zip(attention_modules(scooby, range(depth - n_expert_blocks, depth)),
                               range(depth - n_expert_blocks, depth)):
        if kind == "flashzoi":
            t += [(attn.mha, "Wqkv", q_rows(attn.mha)), (attn.mha, "Wqkv", v_rows(attn.mha))]   # applied in turn
        else:
            t += [(attn, "to_q", None), (attn, "to_v", None)]
        ff = scooby.transformer[i][1].fn
        t += [(ff, "1", None), (ff, "4", None)]
    for name in ["upsampling_unet1", "upsampling_unet0", "final_joined_convs"]:
        t.append((getattr(scooby, name)[0], "conv_layer", None))
    return t


class PrototypeGate(nn.Module):
    """pi(z) = softmax(cos(z, p_k) / tau); prototypes p_k learnable, k-means initialised"""

    def __init__(self, dim: int, n_experts: int, tau: float):
        super().__init__()
        self.tau = tau
        self.prototypes = nn.Parameter(torch.randn(n_experts, dim) / dim ** 0.5)

    @torch.no_grad()
    def init_kmeans(self, states, seed: int = 0):
        import numpy as np
        from sklearn.cluster import KMeans
        X = np.asarray(states, dtype=np.float32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        km = KMeans(self.prototypes.shape[0], n_init=10, random_state=seed).fit(X)
        self.prototypes.copy_(torch.as_tensor(km.cluster_centers_, dtype=self.prototypes.dtype))
        return np.bincount(km.labels_, minlength=self.prototypes.shape[0])

    def forward(self, z):
        z = F.normalize(z.float(), dim=-1)
        p = F.normalize(self.prototypes.float(), dim=-1)
        return z @ p.T / self.tau                                    # logits; softmax in the caller


class AcesoDecoder(nn.Module):
    """MoLoRA decoder around a (peft-wrapped) upstream scooby model; same call interface as scooby:
    ``forward(sequence, cell_emb)``, ``forward_seq_to_emb``, ``forward_cell_embs_only``,
    ``forward_convs_on_emb``."""

    def __init__(self, scooby_peft: nn.Module, cell_emb_dim: int, config: MoLoRAConfig = MoLoRAConfig()):
        super().__init__()
        self.scooby, self.config = scooby_peft, config
        core = self.core
        assert core.n_tracks == 1, "ACESO decoder: ATAC only"
        assert not core.use_transform_borzoi_emb
        K = config.n_experts
        self.alphagenome = getattr(core, "ag", None) is not None
        if self.alphagenome:
            from ._alphagenome import ag_expert_targets
            targets, depth = ag_expert_targets(core, config.n_expert_blocks), len(core.ag.tower.blocks)
        else:
            targets, depth = _expert_targets(core, config.n_expert_blocks), len(core.transformer)
        for parent, attr, rows in targets:
            base = parent._modules[attr]
            parent._modules[attr] = ExpertLoRA(base, K, config.rank, config.alpha, rows=rows)
        self.n_low = depth - config.n_expert_blocks
        if config.gate == "prototype":
            self.cell_state_to_gate = PrototypeGate(cell_emb_dim, K, config.gate_tau)
        else:
            self.cell_state_to_gate = nn.Sequential(nn.Linear(cell_emb_dim, config.gate_hidden), nn.GELU(),
                                                    nn.Linear(config.gate_hidden, K))
            nn.init.zeros_(self.cell_state_to_gate[-1].weight); nn.init.zeros_(self.cell_state_to_gate[-1].bias)
        self.last_gate = None
        self.register_buffer("gate_bias", torch.zeros(K))               # balance="bias"; stays 0 otherwise

    @property
    def core(self):
        m = self.scooby
        return m.base_model.model if hasattr(m, "base_model") and hasattr(m.base_model, "model") else m

    # ---- sequence path
    def _upper(self, x, x_unet1, x_unet0):
        c = self.core
        x = c.transformer[self.n_low:](x)
        x = x.permute(0, 2, 1)
        x = c.upsampling_unet1(x) + x_unet1
        x = c.separable1(x)
        x = c.upsampling_unet0(x) + x_unet0
        x = c.separable0(x)
        x = c.crop(x.permute(0, 2, 1))
        return c.final_joined_convs(x.permute(0, 2, 1))

    def forward_seq_to_emb(self, sequence):
        """[1, 4, L] (as scooby's forward) -> [K, 1920, bins]: one embedding per expert."""
        c, K = self.core, self.config.n_experts
        assert sequence.shape[0] == 1, "one sequence per call (as upstream training)"
        if self.alphagenome:                                     # AlphaGenome: lower trunk once, upper per expert
            trunk, pair, inter, L = c.lower(sequence, self.n_low)
            ex = lambda t: t.expand(K, *t.shape[1:])
            args = (ex(trunk), ex(pair), {k: ex(v) for k, v in inter.items()}, L, self.n_low)
            if self.training and self.config.checkpoint_upper:
                return torch.utils.checkpoint.checkpoint(c.upper, *args, use_reentrant=False)
            return c.upper(*args)
        x = c.conv_dna(sequence)
        x_unet0 = c.res_tower(x)
        x_unet1 = c.unet1(x_unet0)
        x = c._max_pool(x_unet1)
        x_unet1 = c.horizontal_conv1(x_unet1)
        x_unet0 = c.horizontal_conv0(x_unet0)
        x = c.transformer[:self.n_low](x.permute(0, 2, 1))
        args = (x.expand(K, -1, -1), x_unet1.expand(K, -1, -1), x_unet0.expand(K, -1, -1))
        if self.training and self.config.checkpoint_upper:
            return torch.utils.checkpoint.checkpoint(self._upper, *args, use_reentrant=False)
        return self._upper(*args)

    # ---- cell path
    def forward_cell_embs_only(self, cell_emb):
        """[1, n, d] -> (w [1, n, 1920, 1], b [1, n], pi [1, n, K])"""
        w, b = self.core.forward_cell_embs_only(cell_emb)
        pi = torch.softmax(self.cell_state_to_gate(cell_emb).float() + self.gate_bias, dim=-1)
        self.last_gate = pi
        return w, b, pi

    def forward_convs_on_emb(self, seq_emb, w, b, pi, bins_to_predict=None):
        """seq_emb [K, 1920, bins] -> [1, bins, n]"""
        E = seq_emb if bins_to_predict is None else seq_emb[:, :, bins_to_predict]
        H = torch.einsum("kdb,nd->knb", E, w[0, :, :, 0].to(E.dtype))
        out = torch.einsum("knb,nk->nb", H.float(), pi[0]) + b[0].float()[:, None]
        return F.softplus(out).unsqueeze(0).permute(0, 2, 1)

    def forward(self, sequence, cell_emb):
        p = self.forward_cell_embs_only(cell_emb)
        return self.forward_convs_on_emb(self.forward_seq_to_emb(sequence), *p)

    def init_prototypes(self, states, seed: int = 0):
        """prototype gate: k-means (K clusters, cosine via L2-normalised states) of ``states`` (cells x d)"""
        assert isinstance(self.cell_state_to_gate, PrototypeGate)
        self.cell_state_to_gate.init_kmeans(states, seed)

    # ---- regulariser / bookkeeping
    def balance_loss(self):
        """squared coefficient of variation of the mean gate over the cells of the last batch"""
        if self.last_gate is None or self.config.balance != "aux":
            return torch.zeros(())
        imp = self.last_gate.reshape(-1, self.config.n_experts).mean(0)
        return self.config.balance_weight * (imp.var(unbiased=False) / imp.mean().pow(2))

    @torch.no_grad()
    def update_gate_bias(self, load):
        """balance="bias": b_k += bias_rate * sign(mean load - load_k); ``load`` = mean gate per expert over the step"""
        if self.config.balance == "bias":
            load = load.to(self.gate_bias.device, torch.float32)
            self.gate_bias += self.config.bias_rate * torch.sign(load.mean() - load)

    def param_groups(self, lr: float, cellstate_lr: float, weight_decay: float):
        """upstream ``add_weight_decay`` grouping; the gate joins the cell-state (high-lr) group."""
        high, no_decay, decay = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:                                  # same order of checks as upstream
                no_decay.append(p)
            elif "cell_state_to_conv" in name or "cell_state_to_gate" in name:
                high.append(p)
            else:
                decay.append(p)
        return [{"params": high, "weight_decay": weight_decay, "lr": cellstate_lr},
                {"params": no_decay, "weight_decay": 0.0, "lr": lr},
                {"params": decay, "weight_decay": weight_decay, "lr": lr}]


def build_aceso_decoder(cell_emb_dim: int, config: MoLoRAConfig = MoLoRAConfig(), device: str = "cuda",
                        checkpoint=None, train: bool = True, backbone: str = "borzoi") -> AcesoDecoder:
    """scooby (upstream LoRA, ATAC track; Borzoi or Flashzoi) + MoLoRA experts and gate."""
    from ._decoder import build_decoder
    dec = AcesoDecoder(build_decoder(cell_emb_dim, device=device, checkpoint=checkpoint, train=train, backbone=backbone),
                       cell_emb_dim, config).to(device)
    for n, p in dec.named_parameters():
        if "expert_lora" in n or "cell_state_to_gate" in n:
            p.requires_grad = train
    return dec
