"""AlphaGenome backbone for the ACESO decoders (alphagenome-pytorch >= 0.3, Python >= 3.12).

AlphaGenome (Avsec et al., Nature 2026; PyTorch port genomicsxai/alphagenome-pytorch, weights
``gtca/alphagenome_pytorch`` converted from DeepMind's JAX checkpoint) = conv encoder down to 128 bp,
9-block transformer with pair (2 kb) activations, U-net decoder back to 1 bp, organism-aware output
embedders (1 bp: 1536 channels). Weights and derivatives fall under Google DeepMind's AlphaGenome Model
Terms (non-commercial).

:class:`AlphaGenomeScooby` puts upstream scooby's cell read-out on this trunk, with the same interface as
``scooby.modeling.Scooby`` (``forward_seq_to_emb``, ``forward_cell_embs_only``, ``forward_convs_on_emb``,
``n_tracks``, ``embedding_dim``), so the FiLM head works unchanged and the MoLoRA decoder has an
AlphaGenome variant here:

  input        the same 524,288 bp windows as the Borzoi runs (AlphaGenome's native context is 1 Mb);
               organism index 1 (mouse);
  embedding    1 bp embeddings averaged over 32 bp -> 1536 x 6,144 bins of the central 196,608 bp target
               (Borzoi's trunk ends at 32 bp; AlphaGenome's at 1 bp);
  decoder crop the U-net decoder and output embedders (local convolutions; RMSBatchNorm uses fixed running
               statistics, so positions are independent) are run on the target plus an 8,192 bp margin per
               side instead of the whole input: identical target embeddings (checked numerically), 2.5x less
               decoder memory/compute. Encoder and transformer see the whole input;
  adapters     the scooby LoRA recipe mapped onto AlphaGenome with the port's own adapters: LoRA on the
               transformer q_proj / v_proj and MLP fc1 / fc2, Locon (LoRA for Conv1d) on the encoder,
               decoder and output-embedder convolutions (the port's Locon, padding fix: see :class:`Locon`); rank 8, alpha 8 (scale 1, as peft's defaults
               upstream); everything else frozen; heads removed;
  read-out     upstream ``cell_state_to_conv`` (same layers, init and softplus read-out as scooby).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

AG_WEIGHTS = ("gtca/alphagenome_pytorch", "model_all_folds.safetensors")
TARGET_BP, BIN_BP, MARGIN_BP, MOUSE = 196_608, 32, 8_192, 1
LORA_TARGETS = ["q_proj", "v_proj", "mlp.fc1", "mlp.fc2"]
LOCON_TARGETS = ["encoder.", "decoder.", "embedder_1bp.", "embedder_128bp."]

batch_conv = torch.vmap(F.conv1d, chunk_size=1024)                  # as upstream scooby


class Locon(nn.Module):
    """alphagenome-pytorch 0.3.1 ``Locon`` (LoRA for Conv1d: k-wide down conv to ``rank`` channels, 1x1 up conv,
    kaiming / zero init, scale alpha / rank, frozen original), with the padding fixed so the low-rank branch has the
    original's output length: the port reads ``padding[0]``, which is the character 's' for ``padding='same'`` (DNA
    embedder) and 0 for ``StandardizedConv1d`` (it pads 'same' by hand in forward, ``pad_mode``), so it fails on
    both. Here the down conv uses 'same' whenever the original does (all such kernels are odd, so PyTorch's 'same'
    equals AlphaGenome's left/right split) and the original padding otherwise."""

    def __init__(self, original_layer: nn.Conv1d, rank: int = 4, alpha: float = 1):
        super().__init__()
        import math
        self.original_layer = original_layer
        for p in original_layer.parameters():
            p.requires_grad = False
        self.scale = alpha / rank
        k = original_layer.kernel_size[0]
        same = original_layer.padding == "same" or getattr(original_layer, "pad_mode", None) == "same"
        assert not same or (k % 2 == 1 and original_layer.stride[0] == 1), "'same' Locon needs odd kernel, stride 1"
        self.locon_down = nn.Conv1d(original_layer.in_channels, rank, kernel_size=k,
                                    stride=original_layer.stride[0], padding="same" if same else original_layer.padding,
                                    dilation=original_layer.dilation[0], bias=False)
        self.locon_up = nn.Conv1d(rank, original_layer.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.locon_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.locon_up.weight)

    def forward(self, x):
        return self.original_layer(x) + self.locon_up(self.locon_down(x)) * self.scale


def apply_locon(model: nn.Module, target_modules, rank: int, alpha: float):
    """the port's ``apply_locon`` (substring match on module names, Conv1d only, no wrapping inside adapters)
    with the padding-safe :class:`Locon`"""
    from alphagenome_pytorch.extensions.finetuning.adapters import _is_adapter_internal
    for name, module in list(model.named_modules()):
        if not any(t in name for t in target_modules) or not isinstance(module, nn.Conv1d):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if _is_adapter_internal(name, parent):
            continue
        setattr(parent, attr, Locon(module, rank=rank, alpha=alpha))
    return model


def _grad_ckpt(ag) -> bool:
    """Is the non-transformer trunk to be activation-checkpointed on this call?

    Same condition the port already applies to the transformer blocks: the backbone was built with
    ``gradient_checkpointing`` and gradients are actually being taken. Under ``torch.no_grad`` (training
    predictions, :mod:`._predict`, every benchmark run) nothing changes — same ops, same order, same memory.
    It matters for **input gradients**: a backward through the 524,288 bp conv encoder and the U-net decoder
    without checkpointing is what made attribution on this backbone need ~30 GB and OOM at 18 GB.
    """
    return bool(getattr(ag.tower, "gradient_checkpointing", False)) and torch.is_grad_enabled()


def _ck(fn, *args, on: bool):
    return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False) if on else fn(*args)


def _crop_bounds(L: int, res: int):
    """[start, end) at resolution ``res`` of the target window plus margin, inside an input of L bp"""
    s = (L - TARGET_BP) // 2 - MARGIN_BP
    e = s + TARGET_BP + 2 * MARGIN_BP
    assert s >= 0 and s % 128 == 0 and e % 128 == 0, "input must be centred on the target with >= margin of flank"
    return s // res, e // res


class AlphaGenomeScooby(nn.Module):
    n_tracks, embedding_dim, use_transform_borzoi_emb = 1, 1536, False

    def __init__(self, ag, cell_emb_dim: int):
        super().__init__()
        self.ag = ag
        self.cell_emb_dim = cell_emb_dim
        E = self.embedding_dim
        self.cell_state_to_conv = nn.Sequential(                    # upstream Scooby.__init__ verbatim
            nn.Linear(cell_emb_dim, 128), nn.GELU(), nn.Linear(128, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 1024), nn.GELU(), nn.Dropout(0.2), nn.Linear(1024, (E + 1) * self.n_tracks))
        for m in self.cell_state_to_conv:                            # upstream _init_weights (post_init) for Linear
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.cell_state_to_conv[-1].bias)

    # ---- trunk pieces (shared with the MoLoRA variant)
    def _org(self, B, device):
        return torch.full((B,), MOUSE, dtype=torch.long, device=device)

    def lower(self, sequence, n_blocks: int):
        """[1, 4, L] -> trunk after ``n_blocks`` transformer blocks (NLC), pair activations, cropped skips"""
        ag = self.ag
        x = sequence.transpose(1, 2)                                 # scooby's [B, 4, L] -> AlphaGenome's NLC
        L = x.shape[1]
        ck = _grad_ckpt(ag)
        trunk, inter = _ck(ag.encoder, x, on=ck)                     # 524 kb conv stack: the big one for input grads
        inter = {k: v[:, :, slice(*_crop_bounds(L, int(k.split("_")[-1])))] for k, v in inter.items()}
        trunk = trunk.transpose(1, 2) + ag.organism_embed(self._org(x.shape[0], x.device)).unsqueeze(1)
        pair = None
        for blk in ag.tower.blocks[:n_blocks]:
            if ck:
                trunk, pair = torch.utils.checkpoint.checkpoint(ag.tower._forward_block, blk, trunk, pair,
                                                                ag.dtype_policy.compute_dtype, use_reentrant=False)
            else:
                trunk, pair = ag.tower._forward_block(blk, trunk, pair, ag.dtype_policy.compute_dtype)
        return trunk, pair, inter, L

    def upper(self, trunk, pair, inter, L: int, from_block: int):
        """remaining transformer blocks + cropped decoder/embedders -> [B, 1536, 6144] (32 bp)"""
        ag = self.ag
        ck = _grad_ckpt(ag)
        for blk in ag.tower.blocks[from_block:]:
            if ck:
                trunk, pair = torch.utils.checkpoint.checkpoint(ag.tower._forward_block, blk, trunk, pair,
                                                                ag.dtype_policy.compute_dtype, use_reentrant=False)
            else:
                trunk, pair = ag.tower._forward_block(blk, trunk, pair, ag.dtype_policy.compute_dtype)
        org = self._org(trunk.shape[0], trunk.device)
        t = trunk.transpose(1, 2)[:, :, slice(*_crop_bounds(L, 128))]
        # The skip tensors are passed positionally, not captured in the closure, so the recomputed forward
        # reconnects their gradients the same way the uncheckpointed one does.
        ik = list(inter)
        e128 = _ck(lambda t_: ag.embedder_128bp(t_, org, channels_last=False), t, on=ck)
        dec = _ck(lambda t_, *v: ag.decoder(t_, dict(zip(ik, v))), t, *[inter[k] for k in ik], on=ck)
        e1 = _ck(lambda d_, s_: ag.embedder_1bp(d_, org, skip_x=s_, channels_last=False), dec, e128, on=ck)
        e1 = e1[:, :, MARGIN_BP:MARGIN_BP + TARGET_BP]                            # [B, 1536, crop bp]
        return F.avg_pool1d(e1, BIN_BP, BIN_BP)                                   # [B, 1536, 6144]

    # ---- scooby interface
    def forward_seq_to_emb(self, sequence):
        n = len(self.ag.tower.blocks)
        trunk, pair, inter, L = self.lower(sequence, n)
        return self.upper(trunk, pair, inter, L, n)

    def forward_cell_embs_only(self, cell_emb):                      # upstream Scooby verbatim
        bs, n, _ = cell_emb.shape
        w = self.cell_state_to_conv(rearrange(cell_emb, "b n d -> (b n) d"))
        w = rearrange(w, "(b n) (l d) -> b (n l) d", b=bs, l=self.n_tracks)
        b = w[:, :, -1:].reshape(bs, n * self.n_tracks)
        w = w[:, :, :-1].reshape(bs, n * self.n_tracks, self.embedding_dim, 1)
        return w, b

    def forward_convs_on_emb(self, seq_emb, w, b, bins_to_predict=None):  # upstream Scooby verbatim
        x = seq_emb if bins_to_predict is None else seq_emb[:, :, bins_to_predict]
        out = batch_conv(x, w.to(x.dtype), b.to(x.dtype))
        return F.softplus(out).permute(0, 2, 1)

    def forward(self, sequence, cell_emb):
        w, b = self.forward_cell_embs_only(cell_emb)
        return self.forward_convs_on_emb(self.forward_seq_to_emb(sequence), w, b)


def load_alphagenome_scooby(cell_emb_dim: int, device: str = "cuda", train: bool = True, weights=None,
                            rank: int = 8, alpha: float = 8.0, gradient_checkpointing: bool = True) -> AlphaGenomeScooby:
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.extensions.finetuning.adapters import apply_lora
    if weights is None:
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download(*AG_WEIGHTS)
    ag = AlphaGenome.from_pretrained(weights, device="cpu", gradient_checkpointing=gradient_checkpointing)
    ag.heads = nn.ModuleDict()                                        # prediction heads are not used
    for p in ag.parameters():
        p.requires_grad = False
    apply_lora(ag, LORA_TARGETS, rank=rank, alpha=alpha)
    apply_locon(ag, LOCON_TARGETS, rank=rank, alpha=alpha)
    model = AlphaGenomeScooby(ag, cell_emb_dim)
    for n, p in model.named_parameters():
        p.requires_grad = train and ("lora_" in n or "locon_" in n or n.startswith("cell_state_to_conv"))
    return model.to(device)


def ag_expert_targets(core: AlphaGenomeScooby, n_expert_blocks: int):
    """(parent, attribute, rows) for MoLoRA experts on AlphaGenome, the same module types as on Borzoi:
    q/v projections and MLP linears of the last ``n_expert_blocks`` transformer blocks, and the pointwise
    convolutions after them (128 bp embedder, decoder up-block pointwise convs, 1 bp embedder in/skip)."""
    ag, t = core.ag, []
    n = len(ag.tower.blocks)
    for blk in ag.tower.blocks[n - n_expert_blocks:]:
        t += [(blk["mha"], "q_proj", None), (blk["mha"], "v_proj", None),
              (blk["mlp"], "fc1", None), (blk["mlp"], "fc2", None)]
    t.append((ag.embedder_128bp, "project_in", None))
    for ub in ag.decoder.up_blocks:
        t.append((ub.pointwise, "conv", None))
    t += [(ag.embedder_1bp, "project_in", None), (ag.embedder_1bp, "project_skip", None)]
    return t
