"""Stage 2 — sequence decoder: scooby conditioned on the predicted chromatin state.

Built through :mod:`epione.models.scooby` (targets/dataset bit-exact with upstream scooby's
``onTheFlyMultiomeDataset``). Settings follow upstream ``scripts/train_multiome.py`` @ dbfe6cd and
the scooby paper Methods: ``Scooby.from_pretrained('johahi/borzoi-replicate-0')`` + upstream
``get_lora``; 524,288 bp input, 6,144 x 32 bp target bins, ATAC insertions x 0.05; 64 cells per
sequence, no neighbours (single-cell targets); reverse-complement and +-3 bp shift augmentation;
``poisson_multinomial_torch`` (total weight 0.2); ``add_weight_decay`` groups with lr 2e-4 and
cell-state lr 4e-4 (paper; the released config has 1e-4), weight decay 1e-6; 1,000 warm-up steps
then linear decay to 0; bf16 autocast; effective batch 8 sequences via gradient accumulation;
BatchNorm frozen and dropout off during warm-up (paper: "froze ... at the start").

Stated deviations: ATAC only (``n_tracks=1``); the shift is applied to sequence and target together
(bridge semantics; upstream shifts the sequence only); the cell embedding is ACESO's predicted z.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class DecoderConfig:
    epochs: int = 40
    max_steps: int = 0
    accum: int = 8
    lr: float = 2e-4
    cellstate_lr: float = 4e-4
    weight_decay: float = 1e-6
    warmup: int = 1000
    cells_per_sequence: int = 64
    shift_aug: int = 3
    eval_every: int = 250
    workers: int = 6
    seed: int = 0
    checkpoint: str = "johahi/borzoi-replicate-0"


def MultiFragmentReader(readers: Sequence, names: Optional[Sequence[str]] = None):
    """Several samples' ``FragmentInsertionReader``s behind one global cell index (cells stacked in
    reader order), so cells are sampled across all samples as from upstream's merged insertion
    matrix. Returns a ``FragmentInsertionReader`` subclass instance, so ``pseudobulk`` and the
    dataset's other reader calls are inherited."""
    import scipy.sparse as sp
    from epione.models.scooby import FragmentInsertionReader

    class _Multi(FragmentInsertionReader):
        def __init__(self):            # no single fragment file: do not call the parent constructor
            self.readers = list(readers)
            self.barcodes = list(names) if names is not None else [b for r in self.readers for b in r.barcodes]

        @property
        def n_cells(self):
            return len(self.barcodes)

        def reopen(self):
            for r in self.readers:
                r.reopen()
            return self

        def window_insertions(self, chrom, start, end, *, dtype=np.float32):
            return sp.vstack([r.window_insertions(chrom, start, end, dtype=dtype) for r in self.readers]).tocsr()

    return _Multi()


def build_decoder(cell_emb_dim: int, device: str = "cuda", checkpoint: str = "johahi/borzoi-replicate-0", train: bool = True):
    """scooby with upstream LoRA, ATAC track only."""
    from epione.models import scooby as sb
    return sb.load_scooby(checkpoint, cell_emb_dim=cell_emb_dim, n_tracks=1, lora=True, train=train, device=device,
                          disable_cache=True, use_transform_borzoi_emb=False)


def set_stabilise(model, on: bool):
    """BatchNorm in eval mode and dropout off (on=True), otherwise normal training mode."""
    import torch.nn as nn
    model.train()
    if on:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Dropout)):
                m.eval()


def trainable_state(model) -> dict:
    """LoRA + cell-state hypernetwork weights (everything that is trained)."""
    return {k: v for k, v in model.state_dict().items() if "lora" in k or "cell_state_to_conv" in k}


def profile_pearson(model, loader, device: str = "cuda", embed: Optional[Callable] = None):
    """Validation as upstream: Pearson of log1p(pred) vs log1p(target), pooled and per cell.
    ``embed(ce)`` maps the dataset's per-cell payload to states (identity by default)."""
    import torch
    from scipy import stats
    model.eval(); P, T = [], []
    with torch.no_grad():
        for x, rc, y, ce in loader:
            e = ce.to(device) if embed is None else embed(ce)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x.permute(0, 2, 1).to(device), e).float()
            P.append(out.squeeze(0).cpu().numpy()); T.append(y.squeeze(0).numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    r_all = stats.pearsonr(np.log1p(P).ravel(), np.log1p(T).ravel())[0]
    r_cell = np.nanmean([stats.pearsonr(np.log1p(P[:, j]), np.log1p(T[:, j]))[0] for j in range(P.shape[1])])
    return float(r_all), float(r_cell)


def make_optimizer(model, params_groups, c, total: int):
    import torch
    opt = torch.optim.AdamW(params_groups)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-7, total_iters=c.warmup),
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=max(total - c.warmup, 1))],
        milestones=[c.warmup])
    return opt, sched


def train_decoder(model, train_ds, val_ds, out_dir: str, config: DecoderConfig = DecoderConfig(), resume: bool = True,
                  device: str = "cuda", log: Callable = print):
    """Train scooby on ``ScoobyATACDataset``s (``embedding`` = predicted z). Writes ``last.pt``,
    ``best.pt`` (trainable weights + optimiser state) and ``train.log``; resumable."""
    import torch
    from epione.models import scooby as sb
    from scooby.utils.utils import poisson_multinomial_torch, add_weight_decay
    c = config
    os.makedirs(out_dir, exist_ok=True); torch.manual_seed(c.seed)
    dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=c.workers,
                                     worker_init_fn=sb.worker_init, persistent_workers=True, prefetch_factor=4)
    vdl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, worker_init_fn=sb.worker_init)
    total = c.max_steps or c.epochs * math.ceil(len(train_ds) / c.accum)
    opt, sched = make_optimizer(model, add_weight_decay(model, lr=c.lr, cellstate_lr=c.cellstate_lr,
                                                        weight_decay=c.weight_decay), c, total)
    step, best = 0, -1.0
    ck = os.path.join(out_dir, "last.pt")
    if resume and os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"], strict=False); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        step, best = s["step"], s["best"]; log(f"resumed at step {step}")
    flog = open(os.path.join(out_dir, "train.log"), "a"); t0 = time.time()
    while step < total:
        for k, (x, rc, y, ce) in enumerate(dl):
            set_stabilise(model, step < c.warmup)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x.permute(0, 2, 1).to(device, non_blocking=True), ce.to(device))
            loss = poisson_multinomial_torch(out.float(), y.to(device).float(), total_weight=0.2) / c.accum
            loss.backward()
            if (k + 1) % c.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 20 == 0:
                rec = dict(step=step, loss=round(loss.item() * c.accum, 6), lr=sched.get_last_lr()[0],
                           s_per_step=round((time.time() - t0) / 20, 2))
                log(json.dumps(rec)); flog.write(json.dumps(rec) + "\n"); flog.flush(); t0 = time.time()
            if step % c.eval_every == 0 or step == total:
                r_all, r_cell = profile_pearson(model, vdl, device)
                rec = dict(step=step, val_pearson_log=r_all, val_pearson_per_cell=r_cell)
                log(json.dumps(rec)); flog.write(json.dumps(rec) + "\n"); flog.flush()
                state = {"model": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "step": step, "best": max(best, r_all), "config": asdict(c)}
                torch.save(state, ck)
                if r_all > best:
                    best = r_all; torch.save(state, os.path.join(out_dir, "best.pt"))
            if step >= total:
                break
    return model
