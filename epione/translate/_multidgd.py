"""multiDGD: a deep generative decoder for paired scRNA-seq and scATAC-seq.

Port of multiDGD (Schuster et al., *Nat Commun* 2024;
github.com/Center-for-Health-Data-Science/multiDGD) into Epione. multiDGD has
no encoder. Every cell has a learnable latent vector, a Gaussian mixture is
the prior over the latent space, and a decoder maps a latent to per-cell
negative-binomial means of every gene and peak (softmax fractions times the
cell's library size). The decoder and the mixture are updated after every
batch, the training latents once per epoch. New cells are embedded by
optimising a fresh latent per cell with the decoder fixed.

The network, losses (NB likelihood, mixture prior with its own priors, L1 on
the decoder weights), optimisers, schedule, early stopping and new-cell
inference follow the reference ``DGD`` / ``train_dgd`` /
``learn_new_representations``. The reimplementation differs in how it
executes, not in what it computes:

- counts stay sparse;
- the NB log-likelihood is a dense zero-count term plus a sparse correction
  over the nonzero counts (for k = 0 the lgamma and k·log terms vanish);
- gradients that are never applied (to the decoder during validation and
  inference) are not computed.

Translation: :meth:`MultiDGD.predict` fits the new latents on the **source
modality only** and decodes the other modality. :meth:`MultiDGD.reconstruct`
fits them on both modalities, as the scTranslation benchmark does, and is
reconstruction rather than translation.

Adapted from multiDGD, which is distributed under the MIT License:

    MIT License

    Copyright (c) 2023 viktoriaschuster

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
"""
from __future__ import annotations

import math
import random
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData

from ._base import TranslationModel, _device, register

_EPS = 1e-10  # the reference's under-/over-flow guard in the NB density


# ------------------------------------------------------------------ latent prior
class _Softball:
    """Mollified uniform prior on an m-ball (prior on the mixture means)."""

    def __init__(self, dim, radius, sharpness):
        self.dim, self.radius, self.sharpness = dim, radius, sharpness
        self._norm = math.lgamma(1 + dim * 0.5) - dim * (math.log(radius) + 0.5 * math.log(math.pi))

    def sample(self, n):
        # approximate uniform sample from the ball (same draws as the reference)
        with torch.no_grad():
            s = torch.randn((n, self.dim))
            s.div_(s.norm(dim=-1, keepdim=True))
            s.mul_((self.radius * torch.pow(torch.rand((n, 1)), 1.0 / self.dim)).expand(-1, self.dim))
        return s

    def log_prob(self, z):
        return self._norm - torch.log(1 + torch.exp(self.sharpness * (z.norm(dim=-1) / self.radius - 1)))


class GaussianMixture(nn.Module):
    """Diagonal Gaussian mixture over the latent space with priors on its parameters.

    ``forward`` returns the per-cell negative log density *plus* the negative
    log prior of the mixture parameters, as the reference does (so the prior
    is counted once per cell in a summed loss).
    """

    def __init__(self, n_comp, dim, mean_init=(2.0, 5.0), sd_init=(0.4, 1.0), weight_alpha=1.0):
        super().__init__()
        self.dim, self.n_comp = dim, n_comp
        self._mean_prior = _Softball(dim, mean_init[0], mean_init[1])
        self.mean = nn.Parameter(self._mean_prior.sample(n_comp))
        nlv0 = -2 * math.log(sd_init[0])
        self.neglogvar = nn.Parameter(torch.full((n_comp, dim), nlv0))
        self._nlv_prior = torch.distributions.Normal(nlv0, sd_init[1])
        self._alpha = weight_alpha
        self._dirichlet_constant = math.lgamma(n_comp * weight_alpha) - n_comp * math.lgamma(weight_alpha)
        self.weight = nn.Parameter(torch.ones(n_comp))
        self._pi_term = -0.5 * dim * math.log(2 * math.pi)

    def _prior_log_prob(self):
        p = self._dirichlet_constant
        if self._alpha != 1:
            p = p + (self._alpha - 1.0) * torch.softmax(self.weight, -1).log().sum()
        p = p + self._mean_prior.log_prob(self.mean).sum()
        p = p + self._nlv_prior.log_prob(self.neglogvar).sum()
        return p

    def forward(self, x):
        cov = torch.exp(-self.neglogvar)
        y = -(x.unsqueeze(-2) - self.mean).square().div(2 * cov).sum(-1)
        y = y + 0.5 * self.neglogvar.sum(-1) + self._pi_term + torch.log_softmax(self.weight, dim=0)
        y = torch.logsumexp(y, dim=-1) + self._prior_log_prob()
        return -y


# ------------------------------------------------------------------ decoder
class _OutputModule(nn.Module):
    """Modality head: ``n_hidden`` Linear layers (ReLU between), softmax, NB dispersion."""

    def __init__(self, in_features, out_features, n_hidden, hidden, width, r_init=2.0):
        super().__init__()
        layers = []
        if n_hidden > 0:
            for i in range(n_hidden):
                start = stop = width * hidden
                if i == n_hidden - 1:
                    start = width * max(hidden, round(np.sqrt(out_features)))
                    stop = out_features
                if i == 0:
                    start = in_features
                if i == n_hidden - 2:
                    stop = width * max(hidden, round(np.sqrt(out_features)))
                layers.append(nn.Linear(start, stop))
                if i < n_hidden - 1:
                    layers.append(nn.ReLU(True))
        else:
            layers.append(nn.Linear(in_features, out_features))
        self.fc = nn.ModuleList(layers)
        # dispersion r = exp(log_r) + 1 (>= 1)
        self.log_r = nn.Parameter(torch.full((1, out_features), math.log(r_init - 1)))

    def forward(self, x):
        for m in self.fc:
            x = m(x)
        return F.softmax(x.float(), dim=-1)

    @property
    def dispersion(self):
        return torch.exp(self.log_r) + 1


class Decoder(nn.Module):
    def __init__(self, latent_dim, n_features, cfg: "MultiDGDConfig"):
        super().__init__()
        main = []
        n_last = cfg.n_units if cfg.n_hidden > 0 else latent_dim
        for i in range(cfg.n_hidden):
            start = latent_dim if i == 0 else cfg.n_units
            stop = n_last if i == cfg.n_hidden - 1 else cfg.n_units
            main += [nn.Linear(start, stop), nn.ReLU(True)]
        self.main = nn.ModuleList(main)
        self.out_modules = nn.ModuleList(
            _OutputModule(n_last, n, cfg.n_hidden_modality, cfg.n_units, cfg.decoder_width, cfg.r_init)
            for n in n_features)

    def forward(self, z):
        for m in self.main:
            z = m(z)
        return [om(z) for om in self.out_modules]

    def l1(self):
        # the reference regularises every parameter whose name contains "weight"
        return sum(p.abs().sum() for n, p in self.named_parameters() if "weight" in n)


# ------------------------------------------------------------------ NB likelihood on sparse counts
class _SparseCounts:
    """A CSR count matrix on the device, with row gathering and lgamma(k + 1) cached."""

    def __init__(self, X: sp.csr_matrix, device):
        X = sp.csr_matrix(X, dtype=np.float32)
        X.sort_indices()
        self.shape = X.shape
        self.indptr = torch.from_numpy(X.indptr.astype(np.int64)).to(device)
        self.indices = torch.from_numpy(X.indices.astype(np.int64)).to(device)
        self.data = torch.from_numpy(X.data).to(device)
        self.lgk1 = torch.lgamma(self.data + 1)
        self.library = torch.from_numpy(np.asarray(X.sum(axis=1), dtype=np.float32).ravel()).to(device)

    def rows(self, idx: torch.Tensor):
        """(row-in-batch, column, count, lgamma(count + 1)) of the nonzeros of rows ``idx``."""
        start, stop = self.indptr[idx], self.indptr[idx + 1]
        lens = stop - start
        total = int(lens.sum())
        rows = torch.repeat_interleave(torch.arange(len(idx), device=idx.device), lens)
        offs = torch.cumsum(lens, 0) - lens
        pos = torch.arange(total, device=idx.device) - torch.repeat_interleave(offs - start, lens)
        return rows, self.indices[pos], self.data[pos], self.lgk1[pos]


def _nb_logprob(mu_frac, r, lib, nz, per_cell=False):
    """Sum of log NB(k; lib * mu_frac, r) over a batch, from sparse counts.

    Equals the reference ``logNBdensity`` summed over every entry: zero counts
    contribute ``r * log(r / (r + m + eps))``; nonzero counts add
    ``lgamma(k + r) - lgamma(r) - lgamma(k + 1) + k * log(m / (r + m + eps) + eps)``.
    """
    rows, cols, k, lgk1 = nz
    m = lib[:, None] * mu_frac
    dense = r * torch.log(r / (r + m + _EPS))
    m_nz = m[rows, cols]
    r_nz = r[0, cols]
    corr = (torch.lgamma(k + r_nz) - torch.lgamma(r_nz) - lgk1
            + k * torch.log(m_nz / (r_nz + m_nz + _EPS) + _EPS))
    if per_cell:
        return dense.sum(-1) + torch.zeros_like(lib).index_add_(0, rows, corr)
    return dense.sum() + corr.sum()


# ------------------------------------------------------------------ config
@dataclass
class MultiDGDConfig:
    """Hyper-parameters; defaults are multiDGD's (``default_parameters``, ``DGD.train``)."""

    latent_dim: int = 20
    n_components: int = 1  # the scTranslation runner passes no cell labels, hence one component
    n_hidden: int = 2
    n_hidden_modality: int = 3
    n_units: int = 100
    decoder_width: int = 1
    softball_scale: float = 2.0
    softball_hardness: float = 5.0
    sd_sd: float = 1.0
    dirichlet_a: float = 1.0
    r_init: float = 2.0
    batch_size: int = 128
    lr_decoder: float = 1e-4
    lr_representation: float = 1e-2
    lr_gmm: float = 1e-2
    betas: tuple = (0.5, 0.7)
    weight_decay: float = 1e-4
    l1: float = 1.0
    max_epochs: int = 500
    train_minimum: int = 50
    stop_after: int = 10
    # upstream DGD.train shuffles; scTranslation's runner does not
    shuffle: bool = True
    # new-cell inference (learn_new_representations)
    inference_epochs: int = 20
    inference_lr: float = 1e-2
    # execution only
    amp: bool = False
    eval_batch_size: int = 512

    @property
    def sd_mean(self) -> float:
        return round((2 * self.softball_scale) / (10 * self.n_components), 2)


FAST_OVERRIDES = dict(batch_size=512, lr_decoder=2e-4, amp=True)


class _StopWindow:
    """``train_dgd``'s early stopping: after ``train_minimum`` epochs, keep a window of the
    last ``stop_after`` validation losses and stop at the first epoch whose loss is
    above every loss in the window (ties continue). There is no rollback."""

    def __init__(self, train_minimum, stop_after):
        self.minimum, self.length, self.queue = train_minimum, stop_after, []

    def __call__(self, epoch, metric) -> bool:
        if epoch < self.minimum:
            return False
        if epoch < self.minimum + self.length:
            self.queue.append(metric)
            return False
        if any(metric <= q for q in self.queue):
            self.queue.pop(0)
            self.queue.append(metric)
            return False
        return True


def _adam(params, lr, cfg):
    return torch.optim.Adam(params, lr=lr, weight_decay=cfg.weight_decay, betas=tuple(cfg.betas))


# ------------------------------------------------------------------ model
@register("multidgd")
class MultiDGD(TranslationModel):
    """multiDGD translator (one generative model of both modalities).

    Input is raw counts (see :func:`epione.translate.prepare_counts`).
    Predictions are multiDGD's decoder output: per-cell fractions that sum to
    1 over the target features (multiply by a library size for expected
    counts).

    Parameters
    ----------
    fast
        bf16 autocast for the decoder's linear layers, batch 512 and decoder
        learning rate 2e-4. See ``tests/parity/translate_multidgd/MANIFEST.md``
        for whether it matches the default path.
    **kwargs
        Any :class:`MultiDGDConfig` field.
    """

    method = "multidgd"
    directions = ("rna2atac", "atac2rna")

    def __init__(self, *, fast: bool = False, **kwargs):
        base = dict(FAST_OVERRIDES) if fast else {}
        base.update(kwargs)
        self.config = MultiDGDConfig(**base)
        self.fast = fast
        self.decoder: Optional[Decoder] = None
        self.gmm: Optional[GaussianMixture] = None
        self.history: dict = {}

    # -- building blocks ----------------------------------------------------
    def _build(self, n_features, seed=None):
        cfg = self.config
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        # same construction order as the reference: GMM (draws the means), then decoder
        self.gmm = GaussianMixture(cfg.n_components, cfg.latent_dim,
                                   mean_init=(cfg.softball_scale, cfg.softball_hardness),
                                   sd_init=(cfg.sd_mean, cfg.sd_sd), weight_alpha=cfg.dirichlet_a)
        self.decoder = Decoder(cfg.latent_dim, list(n_features), cfg)
        self.n_features = list(n_features)

    def _autocast(self, device):
        if self.config.amp and device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return _nullcontext()

    def _recon(self, z, data, idx, mods, device, per_cell=False):
        """Negative NB log-likelihood of modalities ``mods`` for rows ``idx``."""
        with self._autocast(device):
            ys = self.decoder(z)
        out = 0.0
        for j in mods:
            om = self.decoder.out_modules[j]
            nz = data[j].rows(idx)
            out = out - _nb_logprob(ys[j], om.dispersion, data[j].library[idx], nz, per_cell)
        return out

    # -- training -----------------------------------------------------------
    def _fit(self, rna, atac, train_idx, val_idx, device, seed, verbose):
        cfg = self.config
        Xr = sp.csr_matrix(rna.X, dtype=np.float32)
        Xa = sp.csr_matrix(atac.X, dtype=np.float32)
        for name, X in (("rna", Xr), ("atac", Xa)):
            if X.nnz and (X.data.min() < 0 or not np.allclose(X.data, np.round(X.data))):
                warnings.warn(f"multiDGD models raw counts; the {name} matrix is not integer counts "
                              "(use epi.translate.prepare_counts)")
        self._build((Xr.shape[1], Xa.shape[1]), seed)
        dec, gmm = self.decoder.to(device), self.gmm.to(device)
        tr = [_SparseCounts(Xr[train_idx], device), _SparseCounts(Xa[train_idx], device)]
        va = [_SparseCounts(Xr[val_idx], device), _SparseCounts(Xa[val_idx], device)]
        n_tr, n_va = len(train_idx), len(val_idx)
        n_feat = sum(self.n_features)
        rep = nn.Parameter(torch.zeros(n_tr, cfg.latent_dim, device=device))
        rep_va = nn.Parameter(torch.zeros(n_va, cfg.latent_dim, device=device))
        opt_dec = _adam(dec.parameters(), cfg.lr_decoder, cfg)
        opt_rep = _adam([rep], cfg.lr_representation, cfg)
        opt_gmm = _adam(gmm.parameters(), cfg.lr_gmm, cfg)
        opt_va = _adam([rep_va], cfg.lr_representation, cfg)
        gen = torch.Generator().manual_seed(seed)
        both = (0, 1)
        hist = {k: [] for k in ("loss", "reconstruction", "distribution", "val_loss", "val_reconstruction")}
        stopper, stopped = _StopWindow(cfg.train_minimum, cfg.stop_after), None
        t0 = time.time()
        for epoch in range(cfg.max_epochs):
            dec.train()
            opt_rep.zero_grad(set_to_none=False)
            order = (torch.randperm(n_tr, generator=gen) if cfg.shuffle else torch.arange(n_tr)).to(device)
            tot = rec_s = dist_s = 0.0
            for b in range(0, n_tr, cfg.batch_size):
                idx = order[b:b + cfg.batch_size]
                opt_dec.zero_grad(set_to_none=True)
                opt_gmm.zero_grad(set_to_none=True)
                z = rep[idx]
                rec = self._recon(z, tr, idx, both, device)
                gerr = gmm(z).sum()
                loss = rec + gerr + cfg.l1 * dec.l1()
                loss.backward()
                opt_gmm.step()
                opt_dec.step()
                tot += loss.detach()
                rec_s += rec.detach()
                dist_s += gerr.detach()
            opt_rep.step()
            hist["loss"].append(float(tot) / (n_feat * n_tr))
            hist["reconstruction"].append(float(rec_s) / (n_feat * n_tr))
            hist["distribution"].append(float(dist_s) / (n_feat * n_tr))
            # validation: one step on the validation latents, decoder and mixture fixed
            dec.eval()
            opt_va.zero_grad(set_to_none=False)
            vrec = vtot = 0.0
            for p in list(dec.parameters()) + list(gmm.parameters()):
                p.requires_grad_(False)
            for b in range(0, n_va, cfg.batch_size):
                idx = torch.arange(b, min(b + cfg.batch_size, n_va), device=device)
                z = rep_va[idx]
                rec = self._recon(z, va, idx, both, device)
                l = rec + gmm(z).sum()
                l.backward()
                vrec += rec.detach()
                vtot += l.detach()
            for p in list(dec.parameters()) + list(gmm.parameters()):
                p.requires_grad_(True)
            opt_va.step()
            stop_metric = float(vrec)
            hist["val_loss"].append(float(vtot) / (n_feat * n_va))
            hist["val_reconstruction"].append(stop_metric / (n_feat * n_va))
            if verbose and epoch % 10 == 0:
                print(f"[multiDGD] epoch {epoch}: loss {hist['loss'][-1]:.4f} "
                      f"val {hist['val_reconstruction'][-1]:.4f}")
            if stopper(epoch, stop_metric):
                stopped = epoch
                break
        hist["train_seconds"] = time.time() - t0
        hist["stopped_at_epoch"] = stopped
        hist["epochs"] = len(hist["loss"])
        self.history = hist
        self.train_representation = rep.detach().cpu().numpy()
        dec.eval()

    # -- new cells ----------------------------------------------------------
    def _learn_representations(self, data, mods, n, device):
        """Fit one latent per cell to modalities ``mods`` with the decoder fixed."""
        cfg = self.config
        dec, gmm = self.decoder.to(device).eval(), self.gmm.to(device)
        for p in list(dec.parameters()) + list(gmm.parameters()):
            p.requires_grad_(False)
        bs = cfg.eval_batch_size
        try:
            # initialise every cell at the mixture mean that reconstructs it best
            means = gmm.mean.detach()
            best = torch.empty(n, dtype=torch.long, device=device)
            with torch.no_grad():
                for b in range(0, n, bs):
                    idx = torch.arange(b, min(b + bs, n), device=device)
                    losses = torch.stack([
                        self._recon(means[c].expand(len(idx), -1), data, idx, mods, device, per_cell=True)
                        for c in range(means.shape[0])], dim=1)
                    best[idx] = losses.argmin(1)
            z = nn.Parameter(means[best].clone())
            opt = torch.optim.Adam([z], lr=cfg.inference_lr, weight_decay=1e-4, betas=(0.5, 0.7))
            for _ in range(cfg.inference_epochs):
                opt.zero_grad(set_to_none=False)
                for b in range(0, n, bs):
                    idx = torch.arange(b, min(b + bs, n), device=device)
                    zi = z[idx]
                    (self._recon(zi, data, idx, mods, device) + gmm(zi).sum()).backward()
                opt.step()
        finally:
            for p in list(dec.parameters()) + list(gmm.parameters()):
                p.requires_grad_(True)
        return z.detach()

    @torch.no_grad()
    def _decode(self, z, mod, device):
        out = []
        for b in range(0, z.shape[0], self.config.eval_batch_size):
            with self._autocast(device):
                out.append(self.decoder(z[b:b + self.config.eval_batch_size])[mod].float().cpu().numpy())
        return np.vstack(out)

    def _predict(self, X: sp.csr_matrix, direction: str, device) -> np.ndarray:
        src, dst = (0, 1) if direction == "rna2atac" else (1, 0)
        data = [None, None]
        data[src] = _SparseCounts(X, device)
        z = self._learn_representations(data, (src,), X.shape[0], device)
        self.last_representation = z.cpu().numpy()
        return self._decode(z, dst, device)

    def predict(self, adata: AnnData, direction: Optional[str] = None, *, device=None,
                sparsify: Optional[float] = None) -> AnnData:
        """Translate cells to the other modality (latents fitted on the source modality only).

        Returns per-cell fractions over the target features. ``sparsify`` is
        off by default because the fractions are small (about 1 / n_features).
        """
        return super().predict(adata, direction, device=device, sparsify=sparsify)

    def reconstruct(self, rna: AnnData, atac: AnnData, *, device=None) -> tuple[AnnData, AnnData]:
        """Decode both modalities from latents fitted on **both** modalities.

        This is the scTranslation benchmark's protocol for multiDGD. Each
        cell's target modality is part of the input, so it is joint
        reconstruction, not translation.
        """
        from ._prep import check_paired

        check_paired(rna, atac)
        dev = _device(device)
        Xr = sp.csr_matrix(rna[:, self.rna_features].X, dtype=np.float32)
        Xa = sp.csr_matrix(atac[:, self.atac_features].X, dtype=np.float32)
        data = [_SparseCounts(Xr, dev), _SparseCounts(Xa, dev)]
        z = self._learn_representations(data, (0, 1), rna.n_obs, dev)
        self.last_representation = z.cpu().numpy()
        out = []
        for mod, feats, src in ((0, self.rna_features, rna), (1, self.atac_features, atac)):
            ad = AnnData(X=self._decode(z, mod, dev), obs=src.obs.copy(), var=pd.DataFrame(index=feats.copy()))
            ad.uns["translate"] = {"method": self.method, "direction": "reconstruct"}
            out.append(ad)
        return out[0], out[1]

    # -- persistence --------------------------------------------------------
    def _state(self):
        return {"config": self.config.__dict__, "fast": self.fast, "n_features": self.n_features,
                "decoder": {k: v.cpu() for k, v in self.decoder.state_dict().items()},
                "gmm": {k: v.cpu() for k, v in self.gmm.state_dict().items()}}

    @classmethod
    def _from_state(cls, state, n_genes):
        obj = cls(**state["config"])
        obj.fast = state["fast"]
        obj._build(state["n_features"])
        obj.decoder.load_state_dict(state["decoder"])
        obj.gmm.load_state_dict(state["gmm"])
        obj.decoder.eval()
        return obj


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
