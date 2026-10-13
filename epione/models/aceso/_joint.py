"""Stage 3 — end-to-end joint fine-tuning (ACESO component A).

After stages 1 (encoder distilled onto z) and 2 (decoder trained on the encoder's predictions),
the decoder's sequence-level loss is back-propagated into the state encoder, so the state is
optimised for predicting chromatin, not only for matching the EpiAgent embedding:

    loss = poisson_multinomial(decoder(seq, encoder(panel RNA)), single-cell targets)
           + lambda_align * [(1 - cos) + InfoNCE](encoder(panel RNA), z_EpiAgent)

The alignment term keeps the state in the EpiAgent space (and so the latent-perturbation operator
usable); ``lambda_align`` is chosen on validation. Panels are drawn per sequence by the same
:class:`PanelSampler` as stage 1, so panel-agnosticity (component B) is preserved.

Data: the stage-2 ``ScoobyATACDataset`` is built with ``embedding = cell index`` (n x 1), so each
item returns the indices of its 64 sampled cells; their panel RNA is read and thinned on the fly.
Cells with no detected panel gene are dropped together with their target columns.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

from ._encoder import StateEncoder, alignment_loss, _drop_empty
from ._panel import PanelSampler, PanelSpace
from ._decoder import DecoderConfig, make_optimizer, set_stabilise, trainable_state, profile_pearson


def cell_index_embedding(n_cells: int) -> np.ndarray:
    """``embedding`` for ScoobyATACDataset that returns the sampled cells' indices (exact in float32)."""
    if n_cells >= 2 ** 24:
        raise ValueError("too many cells for an exact float32 index")
    return np.arange(n_cells, dtype=np.float32)[:, None]


class AcesoModel:
    """State encoder + sequence decoder; ``states(rows, cols, depths)`` and ``profiles(seq, states)``."""

    def __init__(self, encoder: StateEncoder, decoder, space: PanelSpace):
        self.encoder, self.decoder, self.space = encoder, decoder, space

    def states(self, M: sp.csr_matrix, cols: np.ndarray):
        return self.encoder(self.encoder.tokenize(M, self.space.vocab_ids[cols]))

    def profiles(self, x, z):
        """x: (B, 4, L) one-hot; z: (B, cells, d) -> (B, bins, cells)"""
        return self.decoder(x, z)

    def state_dict(self):
        return {"encoder": self.encoder.state_dict(), "decoder": trainable_state(self.decoder)}

    def load_state_dict(self, sd):
        self.encoder.load_state_dict(sd["encoder"]); self.decoder.load_state_dict(sd["decoder"], strict=False)


@dataclass
class JointConfig(DecoderConfig):
    lr: float = 5e-5               # decoder LoRA (continues from stage 2)
    cellstate_lr: float = 1e-4     # decoder hypernetwork
    encoder_lr: float = 1e-5       # scGPT + head (continues from stage 1)
    lambda_align: float = 1.0
    tau: float = 0.07
    warmup: int = 200
    epochs: int = 5


def train_joint(model: AcesoModel, X: sp.spmatrix, Z: np.ndarray, sampler: PanelSampler, train_ds, val_ds,
                out_dir: str, config: JointConfig = JointConfig(), resume: bool = True, log: Callable = print):
    """Joint fine-tuning. ``train_ds``/``val_ds``: ScoobyATACDataset built with
    :func:`cell_index_embedding`; ``X``: reference panel-space counts (rows = dataset cell order);
    ``Z``: EpiAgent states of the same cells (alignment target)."""
    import torch
    from epione.models import scooby as sb
    from scooby.utils.utils import poisson_multinomial_torch, add_weight_decay
    c = config
    dev = model.encoder.device
    os.makedirs(out_dir, exist_ok=True); torch.manual_seed(c.seed)
    X = sp.csr_matrix(X, dtype=np.float32)
    Zn = torch.from_numpy((Z / np.linalg.norm(Z, axis=1, keepdims=True)).astype(np.float32))
    dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=c.workers,
                                     worker_init_fn=sb.worker_init, persistent_workers=True, prefetch_factor=4)
    vdl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, worker_init_fn=sb.worker_init)
    groups = add_weight_decay(model.decoder, lr=c.lr, cellstate_lr=c.cellstate_lr, weight_decay=c.weight_decay)
    groups.append({"params": list(model.encoder.net.parameters()), "lr": c.encoder_lr, "weight_decay": 0.0})
    total = c.max_steps or c.epochs * math.ceil(len(train_ds) / c.accum)
    opt, sched = make_optimizer(model.decoder, groups, c, total)
    params = [p for g in groups for p in g["params"]]
    rng = np.random.default_rng(c.seed)
    step, best = 0, -1.0
    ck = os.path.join(out_dir, "last.pt")
    if resume and os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        step, best = s["step"], s["best"]; rng = np.random.default_rng([c.seed, step]); log(f"resumed at step {step}")

    tcols, tdepth = model.space.target, sampler.target_depths

    def val_embed(ce):
        """validation states: target panel at target depth (deterministic)"""
        idx = ce.view(-1).long().numpy()
        M = sampler.counts(X, idx, tcols, 1.0, np.random.default_rng(1))
        M, kept = _drop_empty(M, np.arange(len(idx)))
        z = torch.zeros(len(idx), Zn.shape[1], device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z[kept] = model.states(M, tcols).float()
        return z[None]

    flog = open(os.path.join(out_dir, "train.log"), "a"); t0 = time.time()
    while step < total:
        for k, (x, rc, y, ce) in enumerate(dl):
            set_stabilise(model.decoder, step < c.warmup); model.encoder.net.train()
            idx = ce.view(-1).long().numpy()
            cols, scale = sampler.draw(rng)
            M, keep = _drop_empty(sampler.counts(X, idx, cols, scale, rng), np.arange(len(idx)))
            if len(keep) < 2:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model.states(M, cols)
                out = model.profiles(x.permute(0, 2, 1).to(dev, non_blocking=True), z[None])
            ld = poisson_multinomial_torch(out.float(), y[:, :, keep].to(dev).float(), total_weight=0.2)
            la = alignment_loss(z.float(), Zn[idx[keep]].to(dev), c.tau)
            loss = (ld + c.lambda_align * la) / c.accum
            loss.backward()
            if (k + 1) % c.accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 20 == 0:
                rec = dict(step=step, decoder_loss=round(ld.item(), 6), align_loss=round(la.item(), 4),
                           s_per_step=round((time.time() - t0) / 20, 2))
                log(json.dumps(rec)); flog.write(json.dumps(rec) + "\n"); flog.flush(); t0 = time.time()
            if step % c.eval_every == 0 or step == total:
                model.encoder.net.eval()
                r_all, r_cell = profile_pearson(model.decoder, vdl, dev, embed=val_embed)
                rec = dict(step=step, val_pearson_log=r_all, val_pearson_per_cell=r_cell)
                log(json.dumps(rec)); flog.write(json.dumps(rec) + "\n"); flog.flush()
                state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "step": step, "best": max(best, r_all), "config": asdict(c)}
                torch.save(state, ck)
                if r_all > best:
                    best = r_all; torch.save(state, os.path.join(out_dir, "best.pt"))
            if step >= total:
                break
    return model
