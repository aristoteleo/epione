"""Training and evaluation for seq2PRINT.

Replaces scPrinter's training path -- a JSON config, the
``seq2print_train`` CLI (~770 lines) and ``seq2PRINT.fit`` (~370 lines,
wandb-bound) -- with two functions, :func:`fit` and :func:`evaluate`,
that take a model and datasets and return plain results.

What is kept exactly, because a benchmark depends on it:

* the loss: masked MSE between predicted and PRINT footprint z-scores on
  ``n_modes`` (30) scales drawn at random per batch, plus MSE between
  predicted and observed ``log1p`` window coverage
  (:func:`seq2print_loss`);
* the four validation metrics scPrinter reports (:func:`evaluate`);
* the optimiser recipe: AdamW (lr 3e-3, weight decay 1e-3), bf16
  autocast (the targets are computed under it too, as in scPrinter), an
  exponential moving average of the weights with ema_pytorch's warm-up
  schedule (beta 0.9999, after 100 updates, every 10), gradient clipping
  at 5, early stopping on validation loss with patience 5.

What is not: there is no wandb, no config file, no globals; the EMA is a
few lines here rather than the ``ema_pytorch`` dependency.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ._footprint import footprint_target_model, multiscaleFoot

__all__ = ["TrainResult", "evaluate", "fit", "seq2print_loss"]

ALL_MODES = np.arange(2, 101)


def _unpack(batch, device):
    X, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
    cell = batch[2].to(device) if len(batch) > 2 else None
    return X, y, cell


def seq2print_loss(model, batch, target_model, modes, *, device="cuda"):
    """scPrinter's training loss for one batch.

    Parameters
    ----------
    batch
        ``(X, y[, cells])`` from :class:`~._dataset.Seq2PRINTDataset`;
        ``y[:, 0]`` insertions, ``y[:, 1]`` bias.
    target_model
        From :func:`~._footprint.footprint_target_model`.
    modes
        Footprint radii (2..100) to train on this batch.

    Returns
    -------
    ``(loss, footprint_loss, coverage_loss)`` -- tensors.
    """
    X, y, cell = _unpack(batch, device)
    modes = np.asarray(modes)
    target = multiscaleFoot(y[:, 0], y[:, 1], modes, target_model)
    mask = ~torch.isnan(target)
    coverage = torch.log1p(y[:, 0].sum(dim=-1))
    pred_fp, pred_cov = model(X, cell, modes=torch.as_tensor(modes - 2, device=X.device))
    loss_fp = F.mse_loss(pred_fp[mask], target[mask])
    loss_cov = F.mse_loss(coverage, pred_cov)
    return loss_fp + loss_cov, loss_fp, loss_cov


def _row_pearson(x, y):
    """Pearson per row (scPrinter ``batch_pearson_correlation``)."""
    x = x - x.mean(-1, keepdim=True)
    y = y - y.mean(-1, keepdim=True)
    return (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1))


class _RunningPearson:
    """Pearson of everything seen so far, from running sums."""

    def __init__(self):
        self.n = 0
        self.s = np.zeros(5)        # sum x, y, xx, yy, xy

    def update(self, x, y):
        x = x.detach().double().reshape(-1).cpu().numpy()
        y = y.detach().double().reshape(-1).cpu().numpy()
        self.n += x.size
        self.s += [x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()]

    def corr(self):
        n, (sx, sy, sxx, syy, sxy) = self.n, self.s
        cov = sxy / n - sx * sy / n ** 2
        return float(cov / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2)))


@torch.no_grad()
def evaluate(model, dataset, *, target_model=None, batch_size: int = 64,
             device: str = "cuda", num_workers: int = 0) -> Dict[str, float]:
    """scPrinter's validation metrics, on all 99 scales, centred windows.

    Returns
    -------
    dict
        ``loss`` -- masked footprint MSE;
        ``profile_pearson`` -- mean per-window Pearson of the flattened
        (scale x position) footprint (NaN targets as 0);
        ``footprint_pearson`` -- Pearson across all windows' values;
        ``coverage_pearson`` -- Pearson of predicted vs observed
        ``log1p`` coverage.
    """
    target_model = (target_model or footprint_target_model()).to(device)
    model = model.to(device).eval()
    ds = dataset.centred() if hasattr(dataset, "centred") else dataset
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers)
    loss, n, prof = 0.0, 0, []
    fp_r, cov_r = _RunningPearson(), _RunningPearson()
    for batch in loader:
        X, y, cell = _unpack(batch, device)
        pred_fp, pred_cov = model(X, cell)
        y_cov = torch.log1p(y[:, 0].sum(dim=-1))
        y_fp = multiscaleFoot(y[:, 0], y[:, 1], ALL_MODES, target_model)
        mask = ~torch.isnan(y_fp)
        y_fp = torch.nan_to_num(y_fp, nan=0)
        loss += F.mse_loss(pred_fp[mask].float(), y_fp[mask]).item()
        n += 1
        pf, yf = pred_fp.float().reshape(len(pred_fp), -1), y_fp.reshape(len(y_fp), -1)
        prof.append(_row_pearson(pf, yf).cpu())
        fp_r.update(pf, yf)
        cov_r.update(pred_cov.float(), y_cov)
    prof = torch.cat(prof)
    return {
        "loss": loss / max(n, 1),
        "profile_pearson": float(torch.nanmean(prof)),
        "footprint_pearson": fp_r.corr(),
        "coverage_pearson": cov_r.corr(),
    }


class _EMA:
    """Weight EMA reproducing ``ema_pytorch.EMA`` exactly, as scPrinter
    configures it (beta 0.9999, update_after_step 100, update_every 10).

    Details that matter for matching it:

    * the decay warms up as ``1 - (1 + t) ** -2/3`` (capped at beta),
      ``t`` counting calls after the first 100 -- a constant 0.9999 would
      leave the average stuck near its first copy in a short run;
    * the first call copies; later calls act every ``every`` calls, test
      divisibility on the pre-increment count and compute the decay on
      the post-increment one;
    * floating-point parameters *and buffers* (BatchNorm statistics) are
      averaged; integer buffers are never updated after the initial copy.
    """

    def __init__(self, model, beta=0.9999, after=100, every=10,
                 inv_gamma=1.0, power=2 / 3):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.beta, self.after, self.every = beta, after, every
        self.inv_gamma, self.power = inv_gamma, power
        self.step, self.initted = 0, False

    def _pairs(self, model):
        is_float = lambda x: torch.is_floating_point(x) or torch.is_complex(x)  # noqa: E731
        for (_, e), (_, m) in zip(self.model.named_parameters(), model.named_parameters()):
            if is_float(e):
                yield e, m
        for (_, e), (_, m) in zip(self.model.named_buffers(), model.named_buffers()):
            if is_float(e):
                yield e, m

    def decay(self) -> float:
        # In torch, as ema_pytorch does: the int step clamped with a float
        # min becomes float32, and float32 rounding of the decay is what
        # its averages are built from.
        epoch = (torch.tensor(self.step) - self.after - 1).clamp(min=0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        if epoch.item() <= 0:
            return 0.0
        return value.clamp(min=0.0, max=self.beta).item()

    @torch.no_grad()
    def update(self, model):
        step = self.step
        self.step += 1
        if not self.initted:
            for e, m in self._pairs(model):
                e.data.copy_(m.data)
            self.initted = True
            return
        if step % self.every:
            return
        if step <= self.after:
            for e, m in self._pairs(model):
                e.data.copy_(m.data)
            return
        w = 1.0 - self.decay()
        for e, m in self._pairs(model):
            e.data.lerp_(m.data, w)


@dataclass
class TrainResult:
    """Outcome of :func:`fit`."""

    model: torch.nn.Module                    # best weights (EMA if enabled)
    history: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.history)


def fit(
    model,
    train,
    valid,
    *,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 3e-3,
    weight_decay: float = 1e-3,
    n_modes: int = 30,
    amp: bool = True,
    ema: bool = True,
    patience: int = 5,
    grad_clip: float = 5.0,
    accumulate: int = 1,
    coverage_warming: int = 0,
    max_steps_per_epoch: Optional[int] = None,
    target_model=None,
    num_workers: int = 4,
    device: str = "cuda",
    seed: int = 0,
    checkpoint: Optional[str] = None,
    verbose: bool = True,
) -> TrainResult:
    """Train seq2PRINT with scPrinter's recipe.

    Parameters
    ----------
    model
        From :func:`~._model.build_seq2print` (bulk), or a trained model
        after :meth:`~._model.Seq2PRINT.add_lora` (LoRA fine-tuning --
        only the LoRA networks are trainable; scPrinter uses ``lr=3e-5``
        and ``accumulate=8`` there).
    train, valid
        :class:`~._dataset.Seq2PRINTDataset` s, typically the train and
        valid chromosomes of :func:`~._splits.chromosome_folds`.
    epochs, patience
        Upper bound and early-stopping patience on validation loss.
    n_modes
        Scales sampled per batch (scPrinter: 30 of 99).
    coverage_warming
        LoRA fine-tuning only (needs ``add_lora(..., coverages=...)``):
        train just the coverage-adjustment head for this many epochs
        first, at ``min(10 * lr, 5e-4)``, then freeze it and train the
        LoRA networks. Footprint z-scores grow with depth (roughly as
        sqrt(n)), so a backbone trained on pooled cells overshoots every
        thinner group's targets; absorbing that scale first stops the
        LoRA update from bending the *shape* of predictions to cut the
        MSE. scPrinter uses 5. (If early stopping fires during warming,
        warming ends at the next epoch -- scPrinter intends this but,
        because it resets the epoch counter after the check, never
        leaves warming in that case.)
    max_steps_per_epoch
        Cap steps per epoch (for quick runs); ``None`` = full pass.
    checkpoint
        Path to save the best ``state_dict`` to.
    target_model
        From :func:`~._footprint.footprint_target_model`; built if omitted.
        Note it is moved to ``device`` in place (``nn.Module.to``).

    Returns
    -------
    :class:`TrainResult` holding the best model and per-epoch metrics
    (``train_loss``, ``valid_*`` from :func:`evaluate`).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    target_model = (target_model or footprint_target_model()).to(device)

    head = getattr(model, "profile_cnn_model", None)
    adjust = ([*head.adjustment_footprint.parameters(), *head.adjustment_count.parameters()]
              if hasattr(head, "adjustment_footprint") else [])
    if coverage_warming and not adjust:
        raise ValueError("coverage_warming needs a model with coverage covariates "
                         "(add_lora(..., coverages=...))")
    trainable = {n: p.requires_grad for n, p in model.named_parameters()}

    def make_optimizer(rate):
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("model has no trainable parameters")
        return params, torch.optim.AdamW(params, lr=rate, weight_decay=weight_decay)

    warming = int(coverage_warming)
    if warming:
        for p in model.parameters():
            p.requires_grad = False
        for p in adjust:
            p.requires_grad = True
        params, opt = make_optimizer(min(lr * 10, 5e-4))
    else:
        params, opt = make_optimizer(lr)

    averager = _EMA(model) if ema else None
    use_amp = amp and str(device).startswith("cuda")
    amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported()
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True, generator=torch.Generator().manual_seed(seed),
        persistent_workers=num_workers > 0,
    )
    result = TrainResult(model=model)
    best, stale, best_state = np.inf, 0, None
    scored = model
    for epoch in range(epochs):
        if warming and epoch == warming:
            # Leave warming: LoRA trains, the adjustment head is frozen.
            for n, p in model.named_parameters():
                p.requires_grad = trainable[n]
            for p in adjust:
                p.requires_grad = False
            params, opt = make_optimizer(lr)
            stale = 0
        in_warming = bool(warming) and epoch < warming
        if hasattr(train, "set_epoch"):
            train.set_epoch(epoch)
        model.train()
        t0, run, steps = time.time(), [], 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            modes = rng.permutation(ALL_MODES)[:n_modes]
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss, lfp, lcov = seq2print_loss(model, batch, target_model, modes,
                                                 device=device)
                loss = loss / accumulate
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            if (i + 1) % accumulate == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if averager is not None and not in_warming:
                    averager.update(model)
            run.append((loss.item() * accumulate, lfp.item(), lcov.item()))
            steps += 1
            if max_steps_per_epoch and steps >= max_steps_per_epoch:
                break

        # scPrinter validates the raw model and, once the EMA is running,
        # the EMA too; a lagging EMA resets the patience counter, and the
        # EMA's loss is the one tracked.
        val = evaluate(model, valid, target_model=target_model,
                       batch_size=batch_size, device=device)
        scored = model
        if averager is not None and not in_warming and averager.initted:
            val_ema = evaluate(averager.model, valid, target_model=target_model,
                               batch_size=batch_size, device=device)
            if val_ema["loss"] > val["loss"]:
                stale = 0
            val, scored = val_ema, averager.model
        r = np.mean(run, axis=0) if run else (np.nan,) * 3
        rec = {"epoch": epoch, "warming": in_warming, "train_loss": r[0],
               "train_footprint_loss": r[1], "train_coverage_loss": r[2],
               "seconds": time.time() - t0,
               **{f"valid_{k}": v for k, v in val.items()}}
        result.history.append(rec)
        if verbose:
            print(f"epoch {epoch:3d}{' (warming)' if in_warming else ''}  train {r[0]:.4f}  "
                  f"valid {val['loss']:.4f}  profile r {val['profile_pearson']:.3f}  "
                  f"coverage r {val['coverage_pearson']:.3f}  ({rec['seconds']:.0f}s)",
                  flush=True)
        if val["loss"] < best:
            best, stale, result.best_epoch = val["loss"], 0, epoch
            best_state = {k: v.detach().clone() for k, v in scored.state_dict().items()}
            if checkpoint:
                # Save the live state dict, not the clones: cloning unties
                # Conv1dWrapper's shared weight/conv.weight and doubles the file.
                torch.save(scored.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= patience:
                if in_warming:
                    warming, stale = epoch + 1, 0   # end warming, keep training
                else:
                    break

    if warming:
        # Leave the model as it would be after warming, even if training
        # ended inside it: LoRA trainable, adjustment head frozen.
        for n, prm in model.named_parameters():
            prm.requires_grad = trainable[n]
        for prm in adjust:
            prm.requires_grad = False

    final = copy.deepcopy(scored)
    if best_state is not None:
        final.load_state_dict(best_state)
    result.model = final.eval()
    return result
