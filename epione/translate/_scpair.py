"""scPair: supervised cross-modality prediction between scRNA-seq and scATAC-seq.

Port of the prediction networks of scPair (Hu *et al.*, quon-titative-biology/scPair)
into Epione, with the configuration the scTranslation benchmark runs.

Each direction is an **independent supervised predictor**, trained on its own.
An encoder maps the source profile to a 30-d embedding, and a likelihood
decoder predicts the target:

* RNA → ATAC: a Bernoulli decoder, ``sigmoid(Linear(z)) · sigmoid(library)``,
  where the library-size factor comes from a second branch of the encoder.
* ATAC → RNA: a ZINB decoder. The mean is ``softmax(ReLU(Linear(z)))`` times
  the observed library size during training; the dispersion is ``exp`` of the
  pre-softmax output; the zero-inflation logits come from a G×G linear map.
  Predictions are the per-cell **proportions** (library size 1), as scPair's
  ``predict`` returns them.

Encoders are FCNN blocks (Linear → LayerNorm → LeakyReLU → BatchNorm → Dropout)
followed by an extra Linear. Training uses AdamW (learning rate 1e-3, weight
decay 1e-9 on non-output weight matrices), batches of 130 shuffled each epoch,
and early stopping with patience 25 on the full validation loss, keeping the
best checkpoint. scPair's second phase, a mapping between the two embedding
spaces, does not affect predictions and is not run here.

Adapted from scPair, distributed under the MIT License:

    MIT License

    Copyright (c) 2024 QUON-titative biology

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

The ZINB log-likelihood follows scvi-tools' ``log_zinb_positive`` (BSD-3-Clause,
Copyright (c) 2020 Romain Lopez, Adam Gayoso, Galen Xing, Yosef Lab), as scPair's does.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import TranslationModel, register
from ._nn import _Batches, _epoch_batches, _nullcontext


# ----------------------------------------------------------------------- losses
def log_zinb_positive(x, mu, theta, pi, eps: float = 1e-8):
    """Element-wise ZINB log-likelihood with mean ``mu``, inverse dispersion ``theta``, zero-inflation logits ``pi``."""
    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)
    case_zero = F.softplus(pi_theta_log) - softplus_pi
    case_non_zero = (-softplus_pi + pi_theta_log + x * (torch.log(mu + eps) - log_theta_mu_eps)
                     + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1))
    return (x < eps).float() * case_zero + (x > eps).float() * case_non_zero


# ---------------------------------------------------------------------- network
def _fcnn(dims, act, layernorm, batchnorm, dropout):
    layers = []
    for i in range(1, len(dims)):
        layers.append(nn.Linear(dims[i - 1], dims[i]))
        if layernorm:
            layers.append(nn.LayerNorm(dims[i]))
        layers.append(act)
        if batchnorm:
            layers.append(nn.BatchNorm1d(dims[i]))
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, din, cfg: "ScPairConfig", infer_library: bool):
        super().__init__()
        act = nn.LeakyReLU()
        self.hidden_rep = _fcnn([din] + cfg.hidden, act, cfg.layernorm, cfg.batchnorm, cfg.dropout_rate)
        self.extra_linear_rep = nn.Linear(cfg.hidden[-1], cfg.hidden[-1])
        self.hidden_lib = (nn.Sequential(_fcnn([din] + cfg.hidden, act, cfg.layernorm, cfg.batchnorm, cfg.dropout_rate),
                                         nn.Linear(cfg.hidden[-1], 1)) if infer_library else None)

    def forward(self, x):
        z = self.extra_linear_rep(self.hidden_rep(x))
        lib = self.hidden_lib(x) if self.hidden_lib is not None else None
        return z, lib


class BernoulliDecoder(nn.Module):
    """``sigmoid(Linear(z))`` scaled by ``sigmoid(inferred library)`` (scPair's ``Output_Module_Ber``)."""

    def __init__(self, dz, dout):
        super().__init__()
        self.output_layer = nn.Linear(dz, dout)

    def forward(self, z, latent_lib):
        p = torch.sigmoid(self.output_layer(z))
        return p * torch.sigmoid(latent_lib) if latent_lib is not None else p


class ZINBDecoder(nn.Module):
    """scPair's ``Output_Module_Raw`` with zero inflation and per-cell dispersion."""

    def __init__(self, dz, dout):
        super().__init__()
        self.output_mean = nn.Sequential(nn.Linear(dz, dout), nn.ReLU())
        self.output_dropout = nn.Linear(dout, dout)

    def forward(self, z, lib):
        px_mu = self.output_mean(z)
        scale = torch.softmax(px_mu, dim=-1)
        rate = lib * scale
        theta = torch.exp(px_mu)
        pi = self.output_dropout(px_mu)
        return scale, theta, rate, pi


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)


def _param_groups(modules, lr, weight_decay):
    """scPair's ``add_weight_decay``: no decay on biases/norm parameters or on ``output*`` layers."""
    decay, nodecay = [], []
    for mod in modules:
        for name, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            (nodecay if (p.ndim == 1 or name.split("_")[0] == "output") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay, "lr": lr},
            {"params": nodecay, "weight_decay": 0.0, "lr": lr}]


# ------------------------------------------------------------------------ config
@dataclass
class ScPairConfig:
    """Hyper-parameters; defaults are the scTranslation configuration of scPair."""

    hidden: list = field(default_factory=lambda: [800, 30])
    dropout_rate: float = 0.1
    layernorm: bool = True
    batchnorm: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-9
    batch_size: int = 130
    max_epochs: int = 1000
    patience: int = 25
    # execution only
    amp: bool = False
    eval_batch_size: int = 1024


FAST_OVERRIDES = dict(batch_size=520, amp=True, eval_batch_size=4096)


# ------------------------------------------------------------------------- model
@register("scpair")
class ScPair(TranslationModel):
    """scPair translator: one supervised predictor per direction.

    Expects **raw RNA counts** and ATAC peaks (binarised internally); see
    ``epi.translate.prepare(..., method="scpair")``. RNA→ATAC predictions are
    peak probabilities. ATAC→RNA predictions are per-cell expression
    **proportions** (each row sums to 1); multiply by a library size to get
    counts.

    ``fit(direction=...)`` trains only the requested predictor(s).

    Parameters
    ----------
    fast
        bf16 autocast on CUDA and batch 520. Validated separately; see
        ``tests/parity/translate_scpair/MANIFEST.md``.
    **kwargs
        Any :class:`ScPairConfig` field.
    """

    method = "scpair"
    directions = ("rna2atac", "atac2rna")
    default_sparsify = None

    def __init__(self, *, fast: bool = False, **kwargs):
        base = dict(FAST_OVERRIDES) if fast else {}
        base.update(kwargs)
        self.config = ScPairConfig(**base)
        self.fast = fast
        self.nets: dict = {}
        self.history: dict = {}

    @staticmethod
    def _binary(X):
        X = sp.csr_matrix(X, dtype=np.float32, copy=True)
        X.data = (X.data > 0).astype(np.float32)
        X.eliminate_zeros()
        return X

    def _build(self, direction, n_genes, n_peaks):
        cfg = self.config
        if direction == "rna2atac":
            enc, dec = Encoder(n_genes, cfg, infer_library=True), BernoulliDecoder(cfg.hidden[-1], n_peaks)
        else:
            enc, dec = Encoder(n_peaks, cfg, infer_library=False), ZINBDecoder(cfg.hidden[-1], n_genes)
        enc.apply(_init_weights)
        dec.apply(_init_weights)
        return nn.ModuleDict({"encoder": enc, "decoder": dec})

    def _loss(self, net, direction, x, y, device):
        cfg = self.config
        ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if (cfg.amp and device.type == "cuda") else _nullcontext
        with ac():
            z, lib = net["encoder"](x)
            if direction == "rna2atac":
                out = net["decoder"](z, lib)
            else:
                out = net["decoder"](z, y.sum(1, keepdim=True))
        with torch.autocast(device_type=device.type, enabled=False):
            if direction == "rna2atac":
                return F.binary_cross_entropy(out.float(), y)
            scale, theta, rate, pi = (t.float() for t in out)
            return -log_zinb_positive(y, rate, theta, pi).mean()

    # -- training -----------------------------------------------------------
    def _fit(self, rna, atac, train_idx, val_idx, device, seed, verbose):
        cfg = self.config
        Xr = sp.csr_matrix(rna.X, dtype=np.float32)
        Xa = self._binary(atac.X)
        self.n_genes, self.n_peaks = rna.n_vars, atac.n_vars
        wanted = self.directions if self.direction == "both" else (self.direction,)
        self.history = {}
        t0 = time.time()
        for direction in wanted:
            torch.manual_seed(seed)
            np.random.seed(seed)
            gen = torch.Generator().manual_seed(seed)
            net = self._build(direction, rna.n_vars, atac.n_vars).to(device)
            src, dst = (Xr, Xa) if direction == "rna2atac" else (Xa, Xr)
            data = _Batches(src, dst, device=device)
            opt = torch.optim.AdamW(_param_groups([net["encoder"], net["decoder"]], cfg.lr, cfg.weight_decay))
            best, best_state, no_improve, hist = np.inf, None, 0, {"val": []}
            for epoch in range(cfg.max_epochs):
                net.train()
                for x, y in data.iterate(_epoch_batches(train_idx, cfg.batch_size, gen, True)):
                    loss = self._loss(net, direction, x, y, device)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    # scPair scores the whole validation set in one pass; batching it
                    # changes nothing in eval mode (BatchNorm uses running statistics)
                    # both losses are means over entries: weight batches by their size
                    tot, n = 0.0, 0
                    for x, y in data.iterate(_epoch_batches(val_idx, cfg.eval_batch_size, gen, False)):
                        tot += float(self._loss(net, direction, x, y, device)) * x.shape[0]
                        n += x.shape[0]
                    v = tot / n
                hist["val"].append(v)
                if v < best:
                    best, no_improve = v, 0
                    best_state = copy.deepcopy(net.state_dict())
                else:
                    no_improve += 1
                if verbose and epoch % 10 == 0:
                    print(f"[scPair {direction}] epoch {epoch}: val {v:.5f}")
                if epoch > 1 and no_improve == cfg.patience:
                    break
            net.load_state_dict(best_state)
            net.eval()
            hist["best_epoch"] = int(np.argmin(hist["val"]))
            self.nets[direction] = net
            self.history[direction] = hist
        self.history["train_seconds"] = time.time() - t0
        self.history["epochs"] = {d: len(self.history[d]["val"]) for d in wanted}

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def _predict(self, X: sp.csr_matrix, direction: str, device) -> np.ndarray:
        if direction not in self.nets:
            raise ValueError(f"this scPair model was not trained for {direction!r}")
        net = self.nets[direction].to(device).eval()
        X = self._binary(X) if direction == "atac2rna" else sp.csr_matrix(X, dtype=np.float32)
        out = []
        bs = self.config.eval_batch_size
        for i in range(0, X.shape[0], bs):
            x = torch.from_numpy(X[i:i + bs].toarray()).to(device)
            z, lib = net["encoder"](x)
            if direction == "rna2atac":
                y = net["decoder"](z, lib)
            else:
                y = net["decoder"](z, torch.ones(x.shape[0], 1, device=device))[0]
            out.append(y.float().cpu().numpy())
        return np.vstack(out)

    def _state(self):
        return {"config": self.config.__dict__, "fast": self.fast, "n_peaks": self.n_peaks,
                "state_dicts": {d: {k: v.cpu() for k, v in n.state_dict().items()} for d, n in self.nets.items()}}

    @classmethod
    def _from_state(cls, state, n_genes):
        obj = cls(**state["config"])
        obj.fast = state["fast"]
        obj.n_genes, obj.n_peaks = n_genes, state["n_peaks"]
        for d, sd in state["state_dicts"].items():
            net = obj._build(d, n_genes, obj.n_peaks)
            net.load_state_dict(sd)
            obj.nets[d] = net.eval()
        return obj
