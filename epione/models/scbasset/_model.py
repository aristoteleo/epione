"""scBasset network and training, in PyTorch.

A rewrite of the official TensorFlow/Keras code (calico/scBasset,
Apache-2.0; ``scbasset/utils.py::make_model``, ``scbasset/basenji_utils.py``
and ``bin/scbasset_train.py``). No TensorFlow is imported. Layer for layer:

* stochastic reverse complement and stochastic shift (+-3 bp, pad 0.25),
  one draw per batch, training only;
* stem: GELU -> Conv1d(288, 17, same, no bias) -> BatchNorm -> MaxPool(3);
* tower: 6 x [GELU -> Conv1d(f, 5) -> BatchNorm -> MaxPool(2)], filters
  288 x 1.122^i rounded (288, 323, 363, 407, 456, 512);
* GELU -> Conv1d(256, 1) -> BatchNorm;
* flatten (Keras length-major order) -> GELU -> Dense(32, no bias) ->
  BatchNorm -> Dropout(0.2) -> GELU;
* Dense(n_cells) + sigmoid: its kernel is the cell embedding, its bias the
  cell intercept.

GELU is scBasset's ``sigmoid(1.702 x) x``; BatchNorm uses Keras' epsilon
1e-3 and momentum 0.9; all kernels use Keras ``he_normal`` and all biases
start at zero. Training is ``scbasset_train.py``: binary cross-entropy,
Adam(lr 0.01, beta 0.95/0.9995, epsilon 1e-7), batches of 128 peaks, early
stopping on the training multi-label ROC AUC (patience 50, min_delta 1e-6)
and the weights of the best epoch kept.
"""
from __future__ import annotations

import copy
import math
import random
import time
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._data import SEQ_LEN

_TRUNC = 0.87962566103423978  # std of a unit normal truncated at +-2


def _he_normal_(w: torch.Tensor, fan_in: int) -> None:
    std = math.sqrt(2.0 / fan_in) / _TRUNC
    nn.init.trunc_normal_(w, 0.0, std, -2 * std, 2 * std)


class GELU(nn.Module):
    """scBasset / Basenji GELU: ``sigmoid(1.702 x) * x``."""

    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


def _bn(n: int) -> nn.BatchNorm1d:
    return nn.BatchNorm1d(n, eps=1e-3, momentum=0.1)


class _MaxPoolSame(nn.Module):
    """Keras ``MaxPool1D(pool, padding="same")``: pad the end with -inf."""

    def __init__(self, pool: int):
        super().__init__()
        self.pool = pool

    def forward(self, x):
        r = x.shape[-1] % self.pool
        if r:
            x = F.pad(x, (0, self.pool - r), value=float("-inf"))
        return F.max_pool1d(x, self.pool)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel_size: int, pool_size: int = 1):
        super().__init__()
        self.act = GELU()
        self.conv = nn.Conv1d(cin, cout, kernel_size, padding="same", bias=False)
        _he_normal_(self.conv.weight, cin * kernel_size)
        self.bn = _bn(cout)
        self.pool = _MaxPoolSame(pool_size) if pool_size > 1 else nn.Identity()

    def forward(self, x):
        return self.pool(self.bn(self.conv(self.act(x))))


def tower_filters(filters_init: int = 288, filters_mult: float = 1.122, repeat: int = 6) -> list:
    """Basenji ``conv_tower`` filter counts (the running product is rounded)."""
    out, f = [], float(filters_init)
    for _ in range(repeat):
        out.append(int(np.round(f)))
        f *= filters_mult
    return out


class ScBassetNet(nn.Module):
    """The scBasset network. Input: integer-coded sequences ``(B, L)`` (0-3)."""

    def __init__(self, n_cells: int, bottleneck: int = 32, seq_len: int = SEQ_LEN,
                 filters_init: int = 288, filters_mult: float = 1.122, tower_repeat: int = 6,
                 filters_pre: int = 256, dropout: float = 0.2):
        super().__init__()
        self.n_cells, self.bottleneck, self.seq_len = n_cells, bottleneck, seq_len
        self.stem = ConvBlock(4, filters_init, 17, pool_size=3)
        fs = tower_filters(filters_init, filters_mult, tower_repeat)
        blocks, cin = [], filters_init
        for f in fs:
            blocks.append(ConvBlock(cin, f, 5, pool_size=2))
            cin = f
        self.tower = nn.Sequential(*blocks)
        self.pre = ConvBlock(cin, filters_pre, 1)
        L = math.ceil(seq_len / 3)
        for _ in range(tower_repeat):
            L = math.ceil(L / 2)
        self.flat_len = L
        self.act_d = GELU()
        self.dense = nn.Linear(filters_pre * L, bottleneck, bias=False)
        _he_normal_(self.dense.weight, filters_pre * L)
        self.bn_d = _bn(bottleneck)
        self.drop = nn.Dropout(dropout)
        self.act_out = GELU()
        self.final = nn.Linear(bottleneck, n_cells)
        _he_normal_(self.final.weight, bottleneck)
        nn.init.zeros_(self.final.bias)
        self._aug = random.Random(0)

    def seed_augmentation(self, seed: int) -> None:
        self._aug = random.Random(seed)

    @staticmethod
    def one_hot(codes: torch.Tensor) -> torch.Tensor:
        """``(B, L)`` codes -> ``(B, L, 4)`` float one-hot."""
        return F.one_hot(codes.long(), 4).float()

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # StochasticReverseComplement, then StochasticShift(3); one draw per batch.
        if self._aug.random() > 0.5:
            x = x.flip(-1).flip(1)
        shift = self._aug.randint(-3, 3)
        if shift > 0:
            x = torch.cat([torch.full_like(x[:, :shift], 0.25), x[:, :-shift]], 1)
        elif shift < 0:
            x = torch.cat([x[:, -shift:], torch.full_like(x[:, :(-shift)], 0.25)], 1)
        return x

    def embed_regions(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, L, 4)`` one-hot -> ``(B, bottleneck)`` sequence embedding."""
        if self.training:
            x = self._augment(x)
        h = self.pre(self.tower(self.stem(x.transpose(1, 2))))
        h = h.transpose(1, 2).reshape(h.shape[0], -1)  # Keras flattens length-major
        h = self.drop(self.bn_d(self.dense(self.act_d(h))))
        return self.act_out(h)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Accessibility logits ``(B, n_cells)``."""
        return self.final(self.embed_regions(self.one_hot(codes)))


# --------------------------------------------------------------------------
# Keras-equivalent multi-label ROC AUC (tf.keras.metrics.AUC(curve="ROC",
# multi_label=True), 200 thresholds, interpolation): the training metric
# scBasset's early stopping monitors.
# --------------------------------------------------------------------------

class KerasMultiLabelAUC:
    def __init__(self, n_labels: int, num_thresholds: int = 200, device="cpu"):
        eps = 1e-7
        inner = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
        self.thr = torch.tensor([0.0 - eps] + inner + [1.0 + eps], dtype=torch.float32, device=device)
        self.T = num_thresholds
        self.n = n_labels
        self.pos = torch.zeros(n_labels * (self.T + 1), dtype=torch.float64, device=device)
        self.neg = torch.zeros_like(self.pos)

    @torch.no_grad()
    def update(self, prob: torch.Tensor, target: torch.Tensor) -> None:
        # bin = number of thresholds strictly below the prediction (pred > thr)
        b = torch.searchsorted(self.thr, prob.float().contiguous(), right=False)
        flat = (torch.arange(self.n, device=prob.device) * (self.T + 1)).unsqueeze(0) + b
        t = target.bool()
        L = self.n * (self.T + 1)
        self.pos += torch.bincount(flat[t], minlength=L).double()
        self.neg += torch.bincount(flat[~t], minlength=L).double()

    def result(self) -> float:
        pos = self.pos.view(self.n, self.T + 1)
        neg = self.neg.view(self.n, self.T + 1)
        # tp[j] = #pos with bin > j, j = 0..T-1
        cpos = pos.flip(1).cumsum(1).flip(1)
        cneg = neg.flip(1).cumsum(1).flip(1)
        tp, fp = cpos[:, 1:], cneg[:, 1:]
        P, N = pos.sum(1, keepdim=True), neg.sum(1, keepdim=True)
        tpr = torch.where(P > 0, tp / P.clamp(min=1), torch.zeros_like(tp))
        fpr = torch.where(N > 0, fp / N.clamp(min=1), torch.zeros_like(fp))
        auc = ((fpr[:, :-1] - fpr[:, 1:]) * (tpr[:, :-1] + tpr[:, 1:]) / 2).sum(1)
        return float(auc.mean())


def _dense_rows(Y: sp.csr_matrix, idx: np.ndarray) -> np.ndarray:
    return Y[idx].toarray().astype(np.float32)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train(net: ScBassetNet, codes: np.ndarray, Y: sp.csr_matrix, train_idx: np.ndarray,
          val_idx: Optional[np.ndarray] = None, *, epochs: int = 1000, batch_size: int = 128,
          lr: float = 0.01, patience: int = 50, min_delta: float = 1e-6, seed: int = 0,
          device="auto", verbose: bool = True, log_every: int = 1) -> dict:
    """``scbasset_train.py``'s recipe. Returns the per-epoch history; ``net``
    ends with the weights of the epoch with the best training AUC."""
    dev = resolve_device(device) if isinstance(device, str) else device
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    net.seed_augmentation(seed)
    net.to(dev)
    codes_t = torch.as_tensor(np.ascontiguousarray(codes), dtype=torch.int8, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.95, 0.9995), eps=1e-7)
    hist = {"loss": [], "auc": [], "val_loss": [], "val_auc": [], "time": []}
    best_auc, best_ckpt, best_state, wait = -np.inf, -np.inf, None, 0
    for ep in range(epochs):
        t0 = time.time()
        net.train()
        metric = KerasMultiLabelAUC(net.n_cells, device=dev)
        order = np_rng.permutation(train_idx)
        tot, nb = 0.0, 0
        for s in range(0, len(order), batch_size):
            idx = np.sort(order[s:s + batch_size])
            x = codes_t[torch.as_tensor(idx, device=dev)]
            y = torch.from_numpy(_dense_rows(Y, idx)).to(dev, non_blocking=True)
            logits = net(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            metric.update(torch.sigmoid(logits.detach()), y)
            tot += float(loss.detach()) * len(idx)
            nb += len(idx)
        auc = metric.result()
        hist["loss"].append(tot / nb)
        hist["auc"].append(auc)
        if val_idx is not None and len(val_idx):
            vl, va = evaluate(net, codes_t, Y, val_idx, batch_size=batch_size, device=dev)
        else:
            vl, va = float("nan"), float("nan")
        hist["val_loss"].append(vl)
        hist["val_auc"].append(va)
        hist["time"].append(time.time() - t0)
        if auc > best_ckpt:  # ModelCheckpoint(save_best_only=True, monitor="auc")
            best_ckpt = auc
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
            hist["best_epoch"] = ep
        if verbose and (ep % log_every == 0):
            print(f"epoch {ep + 1}: loss {tot / nb:.4f} auc {auc:.4f} val_loss {vl:.4f} "
                  f"val_auc {va:.4f} ({hist['time'][-1]:.0f}s)", flush=True)
        if auc - min_delta > best_auc:  # EarlyStopping(min_delta, patience, mode="max")
            best_auc, wait = auc, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    hist["epochs_run"] = len(hist["loss"])
    return hist


@torch.no_grad()
def predict_logits(net: ScBassetNet, codes, idx=None, batch_size: int = 256, device="auto",
                   normalize: bool = False) -> np.ndarray:
    """Eval-mode logits ``(n_peaks, n_cells)``; ``normalize=True`` drops the
    cell intercept (scBasset's depth-normalised accessibility before the sigmoid)."""
    dev = resolve_device(device) if isinstance(device, str) else device
    net.eval().to(dev)
    codes_t = codes if torch.is_tensor(codes) else torch.as_tensor(np.ascontiguousarray(codes), dtype=torch.int8)
    if idx is None:
        idx = np.arange(codes_t.shape[0])
    out = []
    for s in range(0, len(idx), batch_size):
        x = codes_t[torch.as_tensor(idx[s:s + batch_size], device=codes_t.device)].to(dev)
        h = net.embed_regions(net.one_hot(x))
        z = h @ net.final.weight.T
        if not normalize:
            z = z + net.final.bias
        out.append(z.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, net.n_cells), np.float32)


@torch.no_grad()
def evaluate(net, codes_t, Y, idx, batch_size=128, device="auto"):
    dev = resolve_device(device) if isinstance(device, str) else device
    net.eval()
    metric = KerasMultiLabelAUC(net.n_cells, device=dev)
    tot = 0.0
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        x = codes_t[torch.as_tensor(b, device=codes_t.device)].to(dev)
        y = torch.from_numpy(_dense_rows(Y, b)).to(dev)
        logits = net(x)
        tot += float(F.binary_cross_entropy_with_logits(logits, y)) * len(b)
        metric.update(torch.sigmoid(logits), y)
    return tot / max(len(idx), 1), metric.result()


def load_keras_h5(net: ScBassetNet, path) -> ScBassetNet:
    """Load weights saved by the official code (``best_model.h5``,
    ``model.save_weights``) into ``net``, without TensorFlow."""
    import h5py

    def _s(x):
        return x.decode() if isinstance(x, bytes) else x

    with h5py.File(path, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f
        names = [_s(n) for n in g.attrs["layer_names"]]
        convs = [n for n in names if n.startswith("conv1d")]
        bns = [n for n in names if n.startswith("batch_normalization")]
        dens = [n for n in names if n.startswith("dense")]
        conv_mods = [net.stem.conv] + [b.conv for b in net.tower] + [net.pre.conv]
        bn_mods = [net.stem.bn] + [b.bn for b in net.tower] + [net.pre.bn, net.bn_d]
        if len(convs) != len(conv_mods) or len(bns) != len(bn_mods) or len(dens) != 2:
            raise ValueError("unexpected layers in the Keras file: "
                             f"{len(convs)} conv1d, {len(bns)} batchnorm, {len(dens)} dense")

        def W(name):
            grp = g[name]
            wn = [_s(w) for w in grp.attrs["weight_names"]]
            return {w.split("/")[-1].split(":")[0]: torch.from_numpy(np.asarray(grp[w])) for w in wn}

        for n, m in zip(convs, conv_mods):
            m.weight.data.copy_(W(n)["kernel"].permute(2, 1, 0))  # (k, in, out) -> (out, in, k)
        for n, m in zip(bns, bn_mods):
            w = W(n)
            m.weight.data.copy_(w["gamma"]); m.bias.data.copy_(w["beta"])
            m.running_mean.copy_(w["moving_mean"]); m.running_var.copy_(w["moving_variance"])
        d0, d1 = W(dens[0]), W(dens[1])
        net.dense.weight.data.copy_(d0["kernel"].T)
        net.final.weight.data.copy_(d1["kernel"].T)
        net.final.bias.data.copy_(d1["bias"])
    return net
