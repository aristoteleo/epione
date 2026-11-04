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

**Lower-trunk conditioning (optional, off by default).** With the layout above the conv tower is shared, so the
cell state cannot change *which* motifs are read, only how the shared features are weighted (Fig 3h: attributions
under a cell's own state versus another state differ by a factor 0.98-1.03). ``n_low_expert_blocks`` adds expert
LoRA to that many transformer blocks *below* the expert part, and ``expert_conv_tower`` adds expert Locon
(:class:`ExpertLocon`, LoRA for k > 1 convolutions) to ``conv_dna``, the res-tower convolutions, ``unet1`` and the
two horizontal 1x1 convolutions. Both are zero-initialised, so the model at step 0 is exactly the model without
them. **Cost:** as soon as either is on, everything from the first conditioned module downwards runs K times per
window (the K experts are the batch dimension) instead of once; with ``expert_conv_tower`` that is the whole trunk,
i.e. K trunk passes per window. The cost is per *window*, not per cell: all cells still share the K embeddings.
Note also that in training mode the BatchNorm layers of the conv tower then see K x the batch; at step 0 the K
inputs are identical so the statistics are unchanged, but once the experts diverge the statistics are pooled over
the experts.

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

**Gate repair (optional, all off by default).** Measured on the official Flashzoi checkpoint
(``flz_molora2spec_within_ft160k_mixed.sel.pt``) over the 1,948 E8.75 ventricular cardiomyocytes of Fig 4: per-cell
gate entropy 1.351 of ln 4 = 1.386 (3.86 effective experts), median max-pi 0.343, 1,862 of 1,948 cells routed to the
same expert by argmax, and pi varying by at most 0.008 along the spatial axis. pi is therefore very nearly the same
constant vector for every cell, so ``sum_k pi_k LoRA_k`` is one fixed linear combination: the K experts collapse into
a single *unconditional* adapter and the mixture contributes no cell-state resolution. Three options attack the
reason cos(z, p_k) barely differs between cells, and are meant to be combined:
  ``gate_topk``          hard top-k routing (k = 1 or 2) with a straight-through estimator instead of the dense
                         softmax, so a cell's output really depends on which expert it selects;
  ``proto_sep_weight``   a separation penalty on the prototypes (:meth:`AcesoDecoder.prototype_penalty`), so they do
                         not collapse onto each other / onto the mean state;
  :meth:`AcesoDecoder.set_gate_tau`  lets the trainer anneal the temperature during training (0.1 -> 0.02);
                         lowering tau at inference only does not work -- it moves cells between experts without
                         aligning them with anything, and costs accuracy.
The load-balancing bias (``balance="bias"``) equalises the *load* but may also be flattening per-cell routing, so it
is worth testing with ``bias_rate`` weakened or the bias reset, as long as no expert starves.

**VQ-MoLoRA (``gate_mode="vq"``, off by default).** A dense softmax over K *precomputed* expert embeddings is an
affine combination of K fixed functions with weights that move smoothly in z, so the prediction is a smooth
function of z and **cannot produce a step**: measured on the official Flashzoi model, the response along a
straight path between two cell states has median linear R2 0.947 over patterned enhancers. Only a map from z with
*finite range* can put a discontinuity in z at a fixed compute budget. ``gate_mode="vq"`` makes the gate a
quantiser (:class:`QuantisedGate`): one hard code per cell, straight-through (optionally the rotation trick of
Fifty et al., ICLR 2025), with EMA codebook updates, a commitment term, dead-code restart and a learnable
temperature. The per-cell read-out ``cell_state_to_conv`` stays continuous, so accessibility is still graded
*within* a code. Because each cell now needs exactly one expert embedding, only the codes present in a batch have
to be run: :meth:`AcesoDecoder.set_active` / the ``active`` argument of
:meth:`AcesoDecoder.forward_seq_to_emb` reduce the expert batch dimension to those codes, which is what keeps
K = 8-16 affordable. Load balancing then uses the LIMoE entropy pair (:meth:`AcesoDecoder.limoe_loss`) rather than
the bias, whose purpose (cancelling a shared preference in the dense weights) is a confound under hard routing.
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
    # ---- gate repair (all off by default; the gate is then exactly the dense softmax of before)
    affine_scale: float = 1.0         # gate_mode="affine" only: multiplies the head output before the 1/K offset
    gate_topk: int = 0                # 0 = dense softmax; k > 0 = keep the k largest pi per cell, renormalise,
                                      # straight-through estimator so the dense pi still receives the gradient
    proto_sep_weight: float = 0.0     # weight of the prototype separation penalty (see ``prototype_penalty``)
    proto_sep_margin: float = 0.0     # cosine above which a prototype pair starts to be penalised
    # ---- VQ-MoLoRA (off by default; gate_mode="softmax" is exactly the gate of before)
    gate_mode: str = "softmax"        # "vq": :class:`QuantisedGate`, one hard code per cell (argmax cosine)
                                      # "affine": UNCONSTRAINED coefficients c(z) = scale * head(z) + 1/K, no softmax,
                                      # no simplex, no balance pressure. The per-cell trunk output becomes an affine
                                      # combination of the K expert trunks instead of a convex one. Motivation
                                      # (conditioning_pilot/r6_affine_gate/): MoLoRA's measured failure was gate
                                      # collapse to a near-uniform simplex point, i.e. one unconditional adapter;
                                      # an unconstrained map has no simplex to collapse onto and at
                                      # zero-initialised head reproduces exactly that uniform model, so it is a
                                      # strict generalisation with a well-defined starting point.
    codebook_ema: float = 0.99        # EMA decay of the codebook update (0 = train the codes by gradient instead)
    commit_weight: float = 0.0        # beta of the commitment term ||z - sg(p_k(z))||^2 (in ``prototype_penalty``)
    dead_code_restart: int = 0        # restart codes with load < dead_code_thresh every N optimizer steps (0 = off)
    dead_code_thresh: float = 0.02    # the A5 "starved expert" threshold
    learn_tau: bool = False           # learnable log-temperature for the soft backward path (X-MoE)
    rotation_trick: bool = False      # Fifty et al. (ICLR 2025) rotation+rescale instead of plain straight-through
    # ---- lower-trunk conditioning (off by default; the model is then bit-identical to before)
    n_low_expert_blocks: int = 0      # transformer blocks below the expert part that also get expert LoRA,
                                      # counted downwards from the top of the lower stack: [n_low - this, n_low)
    expert_conv_tower: bool = False   # expert Locon on conv_dna / res_tower / unet1 / horizontal_conv0,1
    checkpoint_lower: bool = True     # activation checkpointing of the (then K-wide) lower trunk during training


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
        self.active = None                               # set by AcesoDecoder.set_active; None = all K experts

    def forward(self, x, active=None):
        y = self.base(x)
        active = self.active if active is None else active
        A, B = self.expert_lora_A, self.expert_lora_B
        if active is not None:
            A, B = A[active], B[active]
        A, B = A.to(x.dtype), B.to(x.dtype)
        assert x.shape[0] == A.shape[0], "expert modules expect the (active) experts as the batch dimension"
        if self.kind == "linear":
            d = torch.einsum("klr,kor->klo", torch.einsum("kli,kri->klr", x, A), B)
        else:
            d = torch.einsum("kor,krl->kol", B, torch.einsum("kri,kil->krl", A, x))
        if self.rows is not None:                        # linear only (asserted at build)
            return y.index_add(-1, self.rows.to(y.device), self.scale * d.to(y.dtype))
        return y + self.scale * d


class ExpertLocon(nn.Module):
    """K low-rank updates on a k > 1 Conv1d (input [K, in, L], the k-th batch element gets the k-th update).

    Same factorisation, initialisation and scale as :class:`epione.models.aceso._alphagenome.Locon` (a k-wide down
    convolution to ``rank`` channels, then a 1x1 up convolution, kaiming / zero init, scale alpha / rank), with the
    K experts carried in the channel dimension of two grouped convolutions (``groups = K``) so that expert k only
    ever sees batch element k. Stride, padding (including ``'same'``), dilation and kernel size are copied from the
    base layer, so the output shape is the base layer's. ``base`` may itself be a peft LoRA layer; the up
    convolution is zero-initialised, so the module returns exactly ``base(x)`` until it is trained.
    """

    def __init__(self, base: nn.Module, n_experts: int, rank: int, alpha: float):
        super().__init__()
        import math
        from ._backbone import unwrap_linear
        inner = unwrap_linear(base)
        assert isinstance(inner, nn.Conv1d) and inner.groups == 1, "ExpertLocon: plain (non-grouped) Conv1d only"
        k = inner.kernel_size[0]
        pad = inner.padding
        if isinstance(pad, str):
            assert pad != "same" or (k % 2 == 1 and inner.stride[0] == 1), "'same' Locon needs an odd kernel, stride 1"
        self.base, self.scale, self.K = base, alpha / rank, n_experts
        self.c_in, self.c_out, self.rank = inner.in_channels, inner.out_channels, rank
        self.expert_locon_down = nn.Conv1d(n_experts * inner.in_channels, n_experts * rank, kernel_size=k,
                                           stride=inner.stride[0], padding=pad, dilation=inner.dilation[0],
                                           groups=n_experts, bias=False)
        self.expert_locon_up = nn.Conv1d(n_experts * rank, n_experts * inner.out_channels, kernel_size=1,
                                         groups=n_experts, bias=False)
        # per-expert fan-in equals the single-expert layer's, so this is Locon's init per expert
        nn.init.kaiming_uniform_(self.expert_locon_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.expert_locon_up.weight)
        self._conv_kw = dict(stride=inner.stride[0], padding=pad, dilation=inner.dilation[0])
        self.active = None                               # set by AcesoDecoder.set_active; None = all K experts

    def forward(self, x, active=None):
        y = self.base(x)
        active = self.active if active is None else active
        na = self.K if active is None else len(active)
        assert x.shape[0] == na, "expert modules expect the (active) experts as the batch dimension"
        wd = self.expert_locon_down.weight                       # [K * rank, c_in, k]
        wu = self.expert_locon_up.weight                         # [K * c_out, rank, 1]
        if active is not None:
            wd = wd.view(self.K, self.rank, self.c_in, -1)[active].reshape(na * self.rank, self.c_in, -1)
            wu = wu.view(self.K, self.c_out, self.rank, 1)[active].reshape(na * self.c_out, self.rank, 1)
        h = F.conv1d(x.reshape(1, na * self.c_in, x.shape[-1]), wd, groups=na, **self._conv_kw)
        d = F.conv1d(h, wu, groups=na)
        return y + self.scale * d.reshape(na, self.c_out, d.shape[-1]).to(y.dtype)


def _conv_blocks(scooby):
    """the non-separable convolution blocks of the lower trunk, in forward order: conv_dna, the res tower,
    unet1 and the two horizontal convolutions (upstream scooby LoRA targets every ``conv_layer`` but the
    separable ones; these are exactly the lower-trunk members of that set)."""
    t = [scooby.conv_dna]
    t += [m for m in scooby.res_tower if hasattr(m, "conv_layer")]
    t += [m for m in scooby.unet1 if hasattr(m, "conv_layer")]
    t += [scooby.horizontal_conv1, scooby.horizontal_conv0]
    return [(m, "conv_layer") for m in t]


def _low_expert_targets(scooby, n_low: int, n_low_expert_blocks: int, expert_conv_tower: bool):
    """(parent, attribute, rows) of the LOWER-trunk modules that get expert updates: the same module kinds as
    :func:`_expert_targets` but in transformer blocks [n_low - n_low_expert_blocks, n_low), plus (optionally) the
    non-separable convolutions of the conv tower / U-net skips."""
    from ._backbone import attention_modules, q_rows, v_rows
    t = []
    if expert_conv_tower:
        t += [(parent, attr, None) for parent, attr in _conv_blocks(scooby)]
    blocks = range(max(n_low - n_low_expert_blocks, 0), n_low)
    for (attn, kind), i in zip(attention_modules(scooby, blocks), blocks):
        if kind == "flashzoi":
            t += [(attn.mha, "Wqkv", q_rows(attn.mha)), (attn.mha, "Wqkv", v_rows(attn.mha))]
        else:
            t += [(attn, "to_q", None), (attn, "to_v", None)]
        ff = scooby.transformer[i][1].fn
        t += [(ff, "1", None), (ff, "4", None)]
    return t


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


class QuantisedGate(PrototypeGate):
    """Vector-quantising gate: one hard code per cell, ``code(z) = argmax_k cos(z, p_k)`` (VQ-MoLoRA, P1 of
    ``paper/fig4/sharpness/ARCHITECTURE_SURVEY.md``).

    A dense softmax over K *precomputed* expert embeddings is an affine combination with smooth weights, so the
    prediction is a smooth function of z and cannot step: measured on the official Flashzoi model, the response
    along a straight path between two cell states has median linear R2 0.947 over patterned enhancers. Only a map
    with finite range can put a discontinuity in z, which is what this gate is.

    Hygiene, all from the VQ literature and all optional:
      * **spherical codebook** -- the cosine assignment on l2-normalised codes is ViT-VQGAN's recommendation
        (Yu et al., ICLR 2022) and is what :class:`PrototypeGate` already did;
      * **EMA codebook update** (van den Oord et al. 2017, App. A.1): ``N_k <- d N_k + (1-d) n_k``,
        ``m_k <- d m_k + (1-d) sum_{i: k(i)=k} z_i``, ``p_k <- m_k / N_k``, accumulated over all micro-batches of
        an optimizer step and applied by :meth:`codebook_update`. With EMA on, the codes are *not* also trained by
        the optimizer (``prototypes.requires_grad_(False)``, honoured by ``build_aceso_decoder``);
      * **commitment term** ``commit_weight * mean ||z_n - sg(p_k(z))||^2`` (returned by
        :meth:`AcesoDecoder.prototype_penalty`);
      * **dead-code restart** (Jukebox, Dhariwal et al. 2020, 3.1): a code whose EMA load falls below
        ``dead_code_thresh`` is reset to a random assigned state of the current batch. The threshold is the A5
        "starved expert" threshold, so the pre-registered read-out already covers it;
      * **learnable temperature** (X-MoE) for the soft backward path, clamped to [0.01, 1] so it can never anneal
        to zero;
      * **rotation trick** (Fifty et al., ICLR 2025) instead of plain straight-through.

    **Straight-through and the rotation trick.** The forward value is always the one-hot ``pi_hard``. Plain ST
    uses ``pi = pi_soft + (pi_hard - pi_soft).detach()``. The rotation trick replaces "copy the gradient" by
    ``q = sg[R (||e||/||z||)] z`` with ``R = I - 2 r r^T + 2 e_hat z_hat^T``, ``r = (z_hat + e_hat)/||z_hat + e_hat||``
    (Fifty et al., eq. 4), which maps z onto e exactly while preserving the angle and magnitude of the incoming
    gradient, and -- the point -- makes the gradient depend on *where in the Voronoi cell* z sits, so the task loss
    can move a code boundary. **Deviation from the published setting:** there, z and e are the encoder output and
    the codebook vector that flow downstream. Here what flows downstream is the K-dimensional simplex vector that
    mixes the expert embeddings, not a quantised state, so the rotation is applied in gate space,
    ``z := pi_soft``, ``e := pi_hard``; the operator and its properties are unchanged (the forward value is exactly
    ``pi_hard``, since R takes the unit vector of ``pi_soft`` to that of ``pi_hard``), and both vectors are
    non-negative so ``z_hat + e_hat`` never vanishes.
    """

    def __init__(self, dim: int, n_experts: int, tau: float, ema: float = 0.99, commit_weight: float = 0.0,
                 dead_thresh: float = 0.02, learn_tau: bool = False, rotation_trick: bool = False):
        import math
        super().__init__(dim, n_experts, tau)
        self.K, self.ema, self.commit_weight = n_experts, float(ema), float(commit_weight)
        self.dead_thresh, self.rotation_trick, self.learn_tau = float(dead_thresh), bool(rotation_trick), bool(learn_tau)
        if learn_tau:
            self.log_tau = nn.Parameter(torch.tensor(float(math.log(tau))))
        if self.ema > 0:
            self.prototypes.requires_grad_(False)
            self.register_buffer("ema_count", torch.ones(n_experts))
            self.register_buffer("ema_sum", self.prototypes.detach().clone())
        self.register_buffer("load_ema", torch.full((n_experts,), 1.0 / n_experts))
        self.register_buffer("load_acc", torch.zeros(n_experts))     # TRAINED cells since the last reset_load()
        self.register_buffer("load_acc_full", torch.zeros(n_experts))  # full pre-grouping draws (observe_load)
        self.register_buffer("pending_count", torch.zeros(n_experts))
        self.register_buffer("pending_sum", torch.zeros(n_experts, dim))
        # load measured on the FULL pre-grouping batch (see observe_load): with --code-grouped-batches the trained
        # cells are a biased subsample that drops exactly the rare codes, so the dead-code test must not use it
        self.register_buffer("pending_load", torch.zeros(n_experts))
        self.restarts = 0
        self.last_zn = self.last_codes = None
        self._zn_graph = self._codes = None

    def commitment(self):
        """``commit_weight * mean_n ||z_n - sg(p_k(n))||^2`` of the last :meth:`quantise` (0 before the first call).

        **Note on where the gradient goes.** In VQ-VAE this term trains the *encoder* to commit to its code. ACESO
        has no encoder here: z is frozen stage-1 data, so the term as written is constant w.r.t. every trainable
        parameter and is only a diagnostic. When the codes are trained by gradient (``codebook_ema == 0``) the
        stop-gradient is therefore placed on z instead, which is VQ-VAE's *codebook* loss and pulls each code
        towards its assigned cells -- the same number, with the gradient where it can act. With EMA on, the EMA
        update already does that job.
        """
        if not self.commit_weight or self._zn_graph is None:
            return torch.zeros((), device=self.prototypes.device)
        p = F.normalize(self.prototypes.float(), dim=-1)[self._codes]
        zn = self._zn_graph
        d = (zn - p.detach()) ** 2 if self.prototypes.requires_grad is False else (zn.detach() - p) ** 2
        return self.commit_weight * d.sum(-1).mean()

    # ---- temperature
    def tau_value(self):
        if self.learn_tau:
            return self.log_tau.exp().clamp(0.01, 1.0)
        return self.tau

    def assign(self, z):
        """-> (cosine similarities [n, K] on l2-normalised codes, integer codes [n], normalised z)"""
        zn = F.normalize(z.float(), dim=-1)
        p = F.normalize(self.prototypes.float(), dim=-1)
        cos = zn @ p.T
        return cos, cos.argmax(-1), zn

    def quantise(self, z, bias=None):
        """-> (pi straight-through one-hot [..., K], integer codes [...], soft pi [..., K])"""
        flat = z.reshape(-1, z.shape[-1])
        cos, codes, zn = self.assign(flat)
        logits = cos / self.tau_value()
        if bias is not None:
            logits = logits + bias
        soft = torch.softmax(logits, dim=-1)
        hard = F.one_hot(codes, self.K).to(soft.dtype)
        if self.rotation_trick:
            st = _rotate_to(soft, hard)
        else:
            st = soft + (hard - soft).detach()
        if self.training:
            self._accumulate(zn.detach(), codes.detach())
        self._zn_graph, self._codes = zn, codes
        shape = z.shape[:-1]
        return st.reshape(*shape, self.K), codes.reshape(*shape), soft.reshape(*shape, self.K)

    @torch.no_grad()
    def _accumulate(self, zn, codes):
        """accumulate the EMA statistics of this micro-batch; :meth:`codebook_update` applies them"""
        oh = F.one_hot(codes, self.K).to(zn.dtype)
        self.pending_count += oh.sum(0)
        self.pending_sum += oh.T @ zn
        self.load_acc += oh.sum(0)
        self.last_zn, self.last_codes = zn, codes

    @torch.no_grad()
    def set_codebook(self, P):
        """Replace the codebook AND re-seed every statistic derived from it. Use this for any out-of-band change
        of the codes (``--codebook-init``, a k-means re-initialisation); writing ``prototypes`` directly is a bug.

        ``codebook_update`` recomputes ``prototypes = ema_sum / ema_count``, and ``ema_sum`` was seeded in
        ``__init__`` from the prototypes *as constructed*. Overwriting ``prototypes`` alone therefore survives
        only until the first optimizer step, after which the EMA pulls them back (0.99 x old + 0.01 x data) to
        the constructed codebook -- under ``--grow-experts`` that is the K = 4 checkpoint's four prototypes tiled
        to K, i.e. a few distinct directions. Measured on the combined-data run: 16 of 16 codes live at step 20,
        2 of 16 from step 40 onwards (``heart/runs/flz_heart_emaseed_abandoned_step160``). ``load_ema`` is reset
        too, so the dead-code test does not inherit a load history from the discarded codebook.
        """
        P = torch.as_tensor(P, dtype=self.prototypes.dtype, device=self.prototypes.device)
        assert P.shape == self.prototypes.shape, (tuple(P.shape), tuple(self.prototypes.shape))
        self.prototypes.copy_(P)
        if self.ema > 0:
            self.ema_sum.copy_(self.prototypes.detach())
            self.ema_count.fill_(1.0)
        self.load_ema.fill_(1.0 / self.K)
        self.pending_count.zero_(); self.pending_sum.zero_(); self.pending_load.zero_()
        self.last_zn = self.last_codes = None

    @torch.no_grad()
    def init_kmeans(self, states, seed: int = 0):
        """k-means initialisation (as PrototypeGate) + the EMA re-seed of :meth:`set_codebook`"""
        sizes = super().init_kmeans(states, seed)
        self.set_codebook(self.prototypes.detach().clone())
        return sizes

    @torch.no_grad()
    def observe_load(self, z):
        """Record the code histogram of the FULL batch of states, before any code grouping drops cells.

        ``--code-grouped-batches`` keeps only the most frequent codes of each draw, so the trained cells are a
        biased subsample that systematically drops the deliberately rare codes (in the union codebook the
        within-cell-type codes legitimately carry well under 1% of the training cells). Measuring the load on the
        grouped subsample would therefore mark exactly those codes dead and restart them into the dense part of
        the state space, undoing the codebook construction. The restart pool ``last_zn`` is taken from here too,
        so a restart draws from the real state distribution.
        """
        zn = F.normalize(z.reshape(-1, z.shape[-1]).float(), dim=-1)
        codes = (zn @ F.normalize(self.prototypes.float(), dim=-1).T).argmax(-1)
        oh = F.one_hot(codes, self.K).to(zn.dtype).sum(0)
        self.pending_load += oh
        self.load_acc_full += oh
        self.last_zn, self.last_codes = zn, codes
        return codes

    @torch.no_grad()
    def codebook_update(self, step: int = 0, restart_every: int = 0, generator=None):
        """EMA update from the accumulated micro-batches + (optionally) dead-code restart. Returns #restarts."""
        n = self.pending_count
        tot = float(n.sum())
        if tot <= 0:
            return 0
        nl = self.pending_load if float(self.pending_load.sum()) > 0 else n   # full batch if observe_load was used
        self.load_ema.mul_(self.ema).add_((1 - self.ema) * nl / float(nl.sum()))
        self.pending_load.zero_()
        if self.ema > 0:
            self.ema_count.mul_(self.ema).add_((1 - self.ema) * n)
            self.ema_sum.mul_(self.ema).add_((1 - self.ema) * self.pending_sum)
            self.prototypes.data.copy_(self.ema_sum / self.ema_count.clamp_min(1e-6)[:, None])
        self.pending_count.zero_(); self.pending_sum.zero_()
        r = 0
        if restart_every and step > 0 and step % restart_every == 0 and self.last_zn is not None:
            dead = torch.nonzero(self.load_ema < self.dead_thresh).flatten()
            if len(dead):
                pick = torch.randint(0, self.last_zn.shape[0], (len(dead),), device=self.last_zn.device,
                                     generator=generator)
                self.prototypes.data[dead] = self.last_zn[pick].to(self.prototypes.dtype)
                if self.ema > 0:
                    self.ema_sum[dead] = self.prototypes.data[dead].clone()
                    self.ema_count[dead] = 1.0
                self.load_ema[dead] = 1.0 / self.K
                r = len(dead); self.restarts += r
        return r

    # ---- diagnostics
    # TWO load histograms, both worth reporting and not interchangeable:
    #   load_acc       codes of the cells actually TRAINED on. utilisation()/perplexity() read it, so they measure
    #                  how many experts receive gradient -- the number that matters for the rare within-cell-type
    #                  experts, and the one that revealed that --code-grouped-batches 2 trains ~3 of 16 experts.
    #   load_acc_full  codes of the full pre-grouping draw (observe_load). *_full() read it, so they measure the
    #                  population's code distribution, i.e. a property of the codebook.
    def _stats(self, acc):
        t = float(acc.sum())
        if t <= 0:
            return float("nan"), float("nan")
        p = (acc / t).clamp_min(1e-12)
        return float(torch.exp(-(p * p.log()).sum())), float((acc > 0).float().mean())

    def perplexity(self):
        """exp(H(load)) over the TRAINED cells since the last :meth:`reset_load` (K = uniform, 1 = collapsed)"""
        return self._stats(self.load_acc)[0]

    def utilisation(self):
        """share of experts that received gradient since the last :meth:`reset_load`"""
        return self._stats(self.load_acc)[1]

    def perplexity_full(self):
        """exp(H(load)) over the full pre-grouping draws (needs :meth:`observe_load`)"""
        return self._stats(self.load_acc_full)[0]

    def utilisation_full(self):
        """share of codes occupied in the full pre-grouping draws (needs :meth:`observe_load`)"""
        return self._stats(self.load_acc_full)[1]

    def reset_load(self):
        self.load_acc.zero_(); self.load_acc_full.zero_()


def _rotate_to(z, e, eps: float = 1e-8):
    """Fifty et al. (ICLR 2025) rotation trick: return a tensor equal to ``e`` whose gradient w.r.t. ``z`` is the
    (detached) rotation-and-rescale that takes z to e, so angle and magnitude of the incoming gradient survive and
    the gradient depends on where z sits. ``z``, ``e``: [n, d] with z, e != 0 and z_hat != -e_hat."""
    zn = z.norm(dim=-1, keepdim=True).clamp_min(eps)
    en = e.norm(dim=-1, keepdim=True).clamp_min(eps)
    zh, eh = z / zn, e / en
    r = zh + eh
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(eps)
    d = z.shape[-1]
    R = (torch.eye(d, device=z.device, dtype=z.dtype) - 2 * r[..., :, None] * r[..., None, :]
         + 2 * eh[..., :, None] * zh[..., None, :])
    scale = (en / zn).detach()
    return torch.einsum("nij,nj->ni", (R * scale[..., None]).detach(), z)


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
        self.deep_conv = bool(config.expert_conv_tower)
        self.n_low_expert = int(config.n_low_expert_blocks)
        self.deep = self.deep_conv or self.n_low_expert > 0
        if self.alphagenome:
            assert not self.deep, "lower-trunk conditioning is implemented for Borzoi / Flashzoi only"
            from ._alphagenome import ag_expert_targets
            targets, depth = ag_expert_targets(core, config.n_expert_blocks), len(core.ag.tower.blocks)
        else:
            targets, depth = _expert_targets(core, config.n_expert_blocks), len(core.transformer)
        n_low = depth - config.n_expert_blocks
        if self.deep:
            assert 0 <= self.n_low_expert <= n_low, f"n_low_expert_blocks must be in [0, {n_low}]"
            targets = _low_expert_targets(core, n_low, self.n_low_expert, self.deep_conv) + targets
        from ._backbone import unwrap_linear
        for parent, attr, rows in targets:
            base = parent._modules[attr]
            inner = unwrap_linear(base)
            if isinstance(inner, nn.Conv1d) and inner.kernel_size[0] > 1:
                parent._modules[attr] = ExpertLocon(base, K, config.rank, config.alpha)
            else:
                parent._modules[attr] = ExpertLoRA(base, K, config.rank, config.alpha, rows=rows)
        self.n_low = n_low
        assert config.gate_mode in ("softmax", "vq", "affine"), config.gate_mode
        self.vq = config.gate_mode == "vq"
        assert not self.vq or config.gate == "prototype", "gate_mode='vq' needs the prototype (cosine) gate"
        if self.vq:
            self.cell_state_to_gate = QuantisedGate(cell_emb_dim, K, config.gate_tau, ema=config.codebook_ema,
                                                    commit_weight=config.commit_weight,
                                                    dead_thresh=config.dead_code_thresh,
                                                    learn_tau=config.learn_tau, rotation_trick=config.rotation_trick)
        elif config.gate == "prototype":
            self.cell_state_to_gate = PrototypeGate(cell_emb_dim, K, config.gate_tau)
        else:
            self.cell_state_to_gate = nn.Sequential(nn.Linear(cell_emb_dim, config.gate_hidden), nn.GELU(),
                                                    nn.Linear(config.gate_hidden, K))
            nn.init.zeros_(self.cell_state_to_gate[-1].weight); nn.init.zeros_(self.cell_state_to_gate[-1].bias)
        self.last_gate = None
        self.last_gate_hard = None                                      # gate_topk > 0 / gate_mode="vq" only
        self.last_codes = None                                          # gate_mode="vq": integer code per cell
        self.register_buffer("gate_bias", torch.zeros(K))               # balance="bias"; stays 0 otherwise

    # ---- active-expert batching (VQ-MoLoRA): run only the codes present in the batch
    def set_active(self, active):
        """Tell every expert module which experts are in the batch dimension of the sequence path.

        ``active`` is a sorted LongTensor of distinct codes, or None for all K. **Why an attribute and not an
        argument:** the wrapped modules are called by the *backbone's own* forward (``c.transformer[...](h)``,
        ``c.res_tower(x)``, ...), which passes only the tensor, so there is nowhere to thread a keyword through.
        :meth:`forward_seq_to_emb` sets it for the duration of the pass; ``ExpertLoRA.forward`` /
        ``ExpertLocon.forward`` also accept an explicit ``active=`` keyword for direct and unit-test use.
        """
        for m in self.modules():
            if isinstance(m, (ExpertLoRA, ExpertLocon)):
                m.active = active
        return active

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

    def forward_seq_to_emb(self, sequence, active=None):
        """[1, 4, L] (as scooby's forward) -> [K, 1920, bins]: one embedding per expert.

        ``active`` (sorted LongTensor of distinct codes, VQ-MoLoRA) runs only those experts, so the sequence path
        costs ``len(active)`` passes instead of K and K > 4 stays affordable; the returned embedding then has
        ``len(active)`` rows and :meth:`forward_convs_on_emb` must be given the same ``active``. The selection is
        published to the expert modules by :meth:`set_active` and is left in place after the call, because
        activation checkpointing re-runs the forward during the backward pass.
        """
        c = self.core
        K = self.config.n_experts if active is None else len(active)
        if not self.training and self.deep and K > 1:
            # Full-length DNA times many experts exceeds MaxPool1d's CUDA
            # integer indexing range. Eval BN uses fixed statistics, so experts
            # can run independently without changing routing or predictions.
            codes = (torch.arange(K, device=sequence.device) if active is None else active)
            try:
                return torch.cat([self.forward_seq_to_emb(sequence, active=code[None])
                                  for code in codes], dim=0)
            finally:
                self.set_active(active)
        self.set_active(active)
        assert sequence.shape[0] == 1, "one sequence per call (as upstream training)"
        if self.alphagenome:                                     # AlphaGenome: lower trunk once, upper per expert
            trunk, pair, inter, L = c.lower(sequence, self.n_low)
            ex = lambda t: t.expand(K, *t.shape[1:])
            args = (ex(trunk), ex(pair), {k: ex(v) for k, v in inter.items()}, L, self.n_low)
            if self.training and self.config.checkpoint_upper:
                return torch.utils.checkpoint.checkpoint(c.upper, *args, use_reentrant=False)
            return c.upper(*args)
        if self.deep:
            args = self._lower_deep(sequence, K)
        else:
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

    def _lower_deep(self, sequence, K=None):
        """lower trunk with expert-conditioned stages: everything from the first conditioned module downwards
        carries the (active) experts in the batch dimension (K trunk passes per window when ``expert_conv_tower``)."""
        c = self.core
        K = self.config.n_experts if K is None else K
        ck = self.training and self.config.checkpoint_lower

        def run(m, *t):                      # checkpoint only stages that hold parameters
            if ck and any(True for _ in m.parameters(recurse=True)):
                return torch.utils.checkpoint.checkpoint(m, *t, use_reentrant=False)
            return m(*t)

        seq = sequence.expand(K, -1, -1) if self.deep_conv else sequence
        x = run(c.conv_dna, seq)
        for m in c.res_tower:
            x = run(m, x)
        x_unet0 = x
        for m in c.unet1:
            x = run(m, x)
        x_unet1 = x
        x = c._max_pool(x_unet1)
        x_unet1 = run(c.horizontal_conv1, x_unet1)
        x_unet0 = run(c.horizontal_conv0, x_unet0)
        h = x.permute(0, 2, 1)
        split = self.n_low - self.n_low_expert
        for blk in c.transformer[:split]:
            h = run(blk, h)
        if h.shape[0] != K:
            h = h.expand(K, -1, -1)
        for blk in c.transformer[split:self.n_low]:
            h = run(blk, h)
        ex = lambda t: t if t.shape[0] == K else t.expand(K, -1, -1)
        return ex(h), ex(x_unet1), ex(x_unet0)

    # ---- cell path
    def forward_cell_embs_only(self, cell_emb):
        """[1, n, d] -> (w [1, n, 1920, 1], b [1, n], pi [1, n, K])

        ``last_gate`` always holds the DENSE softmax (the load-balancing bias update and the entropy logging read
        it); with ``gate_topk > 0`` the returned pi is the renormalised top-k gate and ``last_gate_hard`` holds it.
        """
        w, b = self.core.forward_cell_embs_only(cell_emb)
        if self.vq:                       # VQ-MoLoRA: one hard code per cell; the integer codes go to last_codes
            st, codes, soft = self.cell_state_to_gate.quantise(cell_emb, self.gate_bias)
            self.last_gate, self.last_gate_hard, self.last_codes = soft, st, codes
            return w, b, st
        logits = self.cell_state_to_gate(cell_emb).float() + self.gate_bias
        if self.config.gate_mode == "affine":
            # unconstrained affine coefficients; the softmax is kept only for the entropy logging
            self.last_gate = torch.softmax(logits, dim=-1)
            self.last_gate_hard, self.last_codes = None, None
            return w, b, logits * self.config.affine_scale + 1.0 / self.config.n_experts
        pi = torch.softmax(logits, dim=-1)
        self.last_gate = pi
        self.last_codes = None
        k = self.config.gate_topk
        if k and k < self.config.n_experts:
            val, idx = pi.topk(k, dim=-1)
            hard = torch.zeros_like(pi).scatter_(-1, idx, val / val.sum(-1, keepdim=True))
            self.last_gate_hard = hard
            pi = pi + (hard - pi).detach()                   # straight-through: hard forward, dense backward
        else:
            self.last_gate_hard = None
        return w, b, pi

    def forward_convs_on_emb(self, seq_emb, w, b, pi, bins_to_predict=None, active=None):
        """seq_emb [K, 1920, bins] -> [1, bins, n]; with ``active`` the gate is restricted to the same experts"""
        E = seq_emb if bins_to_predict is None else seq_emb[:, :, bins_to_predict]
        if active is not None:
            pi = pi[..., active]
        assert pi.shape[-1] == E.shape[0], \
            f"gate width {pi.shape[-1]} != expert dimension {E.shape[0]} of the sequence embedding"
        H = torch.einsum("kdb,nd->knb", E, w[0, :, :, 0].to(E.dtype))
        out = torch.einsum("knb,nk->nb", H.float(), pi[0]) + b[0].float()[:, None]
        return F.softplus(out).unsqueeze(0).permute(0, 2, 1)

    def forward(self, sequence, cell_emb):
        p = self.forward_cell_embs_only(cell_emb)
        active = None
        if self.vq:                                   # only the codes present in this batch need a trunk pass
            active = torch.unique(self.last_codes.reshape(-1))
        return self.forward_convs_on_emb(self.forward_seq_to_emb(sequence, active=active), *p, active=active)

    def init_prototypes(self, states, seed: int = 0):
        """prototype gate: k-means (K clusters, cosine via L2-normalised states) of ``states`` (cells x d)"""
        assert isinstance(self.cell_state_to_gate, PrototypeGate)
        self.cell_state_to_gate.init_kmeans(states, seed)

    def set_codebook(self, P):
        """Replace the gate's codebook from outside (``--codebook-init``). For the quantised gate this also
        re-seeds the EMA statistics and the load history (see :meth:`QuantisedGate.set_codebook`); writing
        ``cell_state_to_gate.prototypes`` directly leaves them stale and the codebook is lost at the first
        optimizer step."""
        g = self.cell_state_to_gate
        if hasattr(g, "set_codebook"):
            return g.set_codebook(P)
        import torch as _t
        with _t.no_grad():
            g.prototypes.copy_(_t.as_tensor(P, dtype=g.prototypes.dtype, device=g.prototypes.device))

    def set_gate_tau(self, tau: float):
        """prototype gate: set the temperature (temperature annealing during training). Nothing caches it.
        Ignored when ``learn_tau`` is on (the temperature is then a parameter)."""
        assert isinstance(self.cell_state_to_gate, PrototypeGate), "temperature annealing needs the prototype gate"
        self.cell_state_to_gate.tau = float(tau)

    def gate_tau(self) -> float:
        g = self.cell_state_to_gate
        return float(g.tau_value()) if isinstance(g, QuantisedGate) else float(getattr(g, "tau", float("nan")))

    def prototype_penalty(self):
        """``proto_sep_weight`` x mean over the K (K - 1) / 2 unordered prototype pairs of
        relu(cos(p_j, p_k) - ``proto_sep_margin``)^2 -- keeps the prototypes from collapsing onto each other, which
        is what makes cos(z, p_k) nearly the same for every cell (dead gate) -- plus, with ``gate_mode="vq"``, the
        VQ commitment term (:meth:`QuantisedGate.commitment`). Added to the loss by the trainer."""
        out = torch.zeros((), device=self.gate_bias.device)
        if not isinstance(self.cell_state_to_gate, PrototypeGate):
            return out
        if self.config.proto_sep_weight:
            p = F.normalize(self.cell_state_to_gate.prototypes.float(), dim=-1)
            c = p @ p.T
            iu = torch.triu_indices(c.shape[0], c.shape[0], offset=1, device=c.device)
            out = out + self.config.proto_sep_weight * (c[iu[0], iu[1]] - self.config.proto_sep_margin).relu().pow(2).mean()
        if self.vq:
            out = out + self.cell_state_to_gate.commitment()
        return out

    def codebook_update(self, step: int = 0):
        """VQ-MoLoRA: EMA codebook update + dead-code restart; call once after every optimizer step. -> #restarts"""
        if not self.vq:
            return 0
        return self.cell_state_to_gate.codebook_update(step, self.config.dead_code_restart)

    def observe_load(self, z):
        """Record the code histogram of the full, pre-grouping batch of states (see QuantisedGate.observe_load).
        Call it in the training loop before any ``--code-grouped-batches`` subsetting. -> codes or None."""
        if not self.vq:
            return None
        return self.cell_state_to_gate.observe_load(z)

    def limoe_loss(self, soft_pi=None, marginal=None):
        """LIMoE (Mustafa et al. 2022) entropy pair, normalised by ln K: ``mean_c H(pi_c) - H(mean_c pi_c)``.

        Minimising it makes each cell's routing confident (first term) while keeping the batch marginal spread over
        the experts (second term); it replaces the load-balancing bias, which equalises the load without touching
        per-cell routing. ``marginal`` lets the caller pass the mean pi of the WHOLE optimizer step (Qiu et al. 2025:
        micro-batch scope destroys specialisation); the gradient then flows only through the current micro-batch's
        share of it, which is the correct partial derivative of the global term.
        """
        p = self.last_gate if soft_pi is None else soft_pi
        if p is None:
            return torch.zeros((), device=self.gate_bias.device)
        import math
        p = p.reshape(-1, self.config.n_experts).float()
        h_cell = -(p * p.clamp_min(1e-9).log()).sum(-1).mean()
        m = p.mean(0) if marginal is None else marginal
        h_marg = -(m * m.clamp_min(1e-9).log()).sum()
        return (h_cell - h_marg) / math.log(self.config.n_experts)

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


def deep_state_dict(state: dict, model: "AcesoDecoder") -> dict:
    """Map a state dict saved by a model WITHOUT lower-trunk conditioning onto one with it.

    Wrapping a module in :class:`ExpertLoRA` / :class:`ExpertLocon` pushes everything that was there one level
    down, under ``.base`` (Flashzoi's fused ``Wqkv`` is wrapped twice, for the q and the v rows, so its old keys
    move two levels). Keys that already match the model are left alone, so this is a no-op for a checkpoint that
    was saved by the same (deep) model, and for a model without lower-trunk conditioning.
    """
    want = set(model.state_dict().keys())
    if all(k in want for k in state):                       # same model that saved it: nothing to do
        return dict(state)
    exp = {n for n, m in model.named_modules() if isinstance(m, (ExpertLoRA, ExpertLocon))}

    def own_key(p):                                          # the parameter that identifies the wrapper at p
        for s in ("expert_lora_A", "expert_locon_down.weight"):
            if f"{p}.{s}" in want:
                return f"{p}.{s}"
        return None

    shift = {}
    for p in exp:
        if p.endswith(".base") and p[: -len(".base")] in exp:
            continue                                         # not the outermost wrapper of its chain
        depth = nnew = 0
        q = p
        while q in exp:
            k = own_key(q)
            depth += 1
            nnew += int(k is not None and k not in state)
            q += ".base"
        if nnew:
            shift[p] = nnew
    order = sorted(shift, key=len, reverse=True)

    def remap(k):
        for p in order:
            if k.startswith(p + "."):
                return f"{p}{'.base' * shift[p]}.{k[len(p) + 1:]}"
        return k

    return {remap(k): v for k, v in state.items()}


def grow_experts_state_dict(state: dict, model: "AcesoDecoder") -> dict:
    """Warm-start a K_new-expert model from a K_old-expert checkpoint (K_new a multiple of K_old) by TILING.

    Every expert-indexed tensor carries the experts in dimension 0 -- ``expert_lora_A/B`` as [K, ...], the grouped
    Locon weights as [K * rank, ...] / [K * c_out, ...] with the expert as the outer block, and the gate's
    ``prototypes`` / ``gate_bias`` as [K, ...] -- so new expert ``j`` is given a copy of old expert ``j % K_old``
    by indexing dimension 0 with ``arange(n_new) % n_old``. Each new expert therefore starts as an exact copy of a
    trained one rather than at its fresh (zero-update) initialisation, which keeps the warm start's learned expert
    behaviour. The copies are identical and only diverge once they are routed to different cells, so the codebook
    should be re-initialised (``--reinit-prototypes``) when growing K. Tensors whose shapes already match are left
    alone, so this is a no-op when K is unchanged.
    """
    want = model.state_dict()
    out = {}
    for k, v in state.items():
        w = want.get(k)
        if w is None or w.shape == v.shape:
            out[k] = v
            continue
        assert w.shape[1:] == v.shape[1:] and w.shape[0] % v.shape[0] == 0, \
            f"cannot grow {k}: checkpoint {tuple(v.shape)} -> model {tuple(w.shape)}"
        idx = torch.arange(w.shape[0]) % v.shape[0]
        out[k] = v[idx].clone()
    return out


def build_aceso_decoder(cell_emb_dim: int, config: MoLoRAConfig = MoLoRAConfig(), device: str = "cuda",
                        checkpoint=None, train: bool = True, backbone: str = "borzoi") -> AcesoDecoder:
    """scooby (upstream LoRA, ATAC track; Borzoi or Flashzoi) + MoLoRA experts and gate."""
    from ._decoder import build_decoder
    dec = AcesoDecoder(build_decoder(cell_emb_dim, device=device, checkpoint=checkpoint, train=train, backbone=backbone),
                       cell_emb_dim, config).to(device)
    for n, p in dec.named_parameters():
        if "expert_lora" in n or "expert_locon" in n or "cell_state_to_gate" in n:
            p.requires_grad = train
    if isinstance(dec.cell_state_to_gate, QuantisedGate) and dec.cell_state_to_gate.ema > 0:
        dec.cell_state_to_gate.prototypes.requires_grad_(False)   # the codebook is updated by EMA, not by AdamW
    return dec
