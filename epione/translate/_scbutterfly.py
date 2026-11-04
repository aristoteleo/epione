"""scButterfly: dual-aligned VAE translation between scRNA-seq and scATAC-seq.

Port of scButterfly (Cao et al., *Nat Commun* 2024; github.com/BioX-NKU/scButterfly)
into Epione. The network, losses, three-phase schedule (RNA autoencoder
pretraining, ATAC autoencoder pretraining, then joint translator training with
two adversarial discriminators), early stopping and defaults follow the
reference ``Model`` / ``Butterfly`` classes. The reimplementation differs in
how it executes, not in what it computes: batches are densified from the
sparse matrices on the fly instead of holding a dense copy of the data,
checkpoints stay in memory instead of on disk, and the discriminator's soft
labels are built with tensor ops instead of a Python loop.

Adapted from scButterfly, which is distributed under the MIT License:

    MIT License

    Copyright (c) 2023 BioX-NKU

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

import copy
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from ._base import TranslationModel, register
from ._prep import chrom_blocks
from ._nn import ChromLinear, _Batches, _EarlyStop, _epoch_batches, _load_parts, _mean, _nullcontext


# ------------------------------------------------------------------ network
class NetBlock(nn.Module):
    """Input dropout ("noise"), then [Linear -> BatchNorm -> activation -> dropout] x n.

    No dropout after the last layer.
    """

    def __init__(self, dims, acts, dropout_rate, noise_rate):
        super().__init__()
        self.noise = nn.Dropout(noise_rate)
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList(acts)
        self.drops = nn.ModuleList()
        n = len(dims) - 1
        for i in range(n):
            lin = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_uniform_(lin.weight)
            self.linears.append(lin)
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))
            if i < n - 1:
                self.drops.append(nn.Dropout(dropout_rate))

    def forward(self, x):
        x = self.noise(x)
        n = len(self.linears)
        for i in range(n):
            x = self.acts[i](self.bns[i](self.linears[i](x)))
            if i < n - 1:
                x = self.drops[i](x)
        return x


class ChromEncoder(nn.Module):
    """ATAC encoder: the first layer is one Linear/BN/act/dropout per chromosome.

    BatchNorm is per feature, so one BatchNorm over the concatenated blocks
    equals scButterfly's per-chromosome BatchNorms.
    """

    def __init__(self, chroms, per_chrom, dims, acts, dropout_rate, noise_rate):
        super().__init__()
        self.noise = nn.Dropout(noise_rate)
        self.first = ChromLinear(chroms, per_chrom, "in")
        self.first_bn = nn.BatchNorm1d(per_chrom * len(chroms))
        self.first_act = copy.deepcopy(acts[0])
        self.first_drop = nn.Dropout(dropout_rate)
        self.rest = NetBlock([per_chrom * len(chroms)] + list(dims), acts[1:], dropout_rate, 0.0)

    def forward(self, x):
        x = self.noise(x)
        h = self.first_drop(self.first_act(self.first_bn(self.first(x))))
        return self.rest(h)


class ChromDecoder(nn.Module):
    """ATAC decoder: the last layer is one Linear/BN/act per chromosome (no dropout)."""

    def __init__(self, chroms, per_chrom, dims, acts, dropout_rate):
        super().__init__()
        hidden = list(dims) + [per_chrom * len(chroms)]
        self.body = nn.ModuleList()
        for i in range(len(hidden) - 1):
            lin = nn.Linear(hidden[i], hidden[i + 1])
            nn.init.xavier_uniform_(lin.weight)
            self.body.append(nn.ModuleDict({
                "lin": lin, "bn": nn.BatchNorm1d(hidden[i + 1]),
                "act": copy.deepcopy(acts[i]), "drop": nn.Dropout(dropout_rate),
            }))
        self.last = ChromLinear(chroms, per_chrom, "out")
        self.last_bn = nn.BatchNorm1d(sum(chroms))
        self.last_act = copy.deepcopy(acts[-1])

    def forward(self, x):
        for m in self.body:
            x = m["drop"](m["act"](m["bn"](m["lin"](x))))
        return self.last_act(self.last_bn(self.last(x)))


def _lbr(din, dout, act):
    lin = nn.Linear(din, dout)
    nn.init.xavier_uniform_(lin.weight)
    return nn.Sequential(lin, nn.BatchNorm1d(dout), act)


def _reparameterize(mu, logvar):
    return mu + torch.randn_like(logvar) * torch.exp(logvar / 2)


class SingleTranslator(nn.Module):
    """VAE bottleneck used only to pretrain one modality's autoencoder."""

    def __init__(self, din, dembed, acts):
        super().__init__()
        self.mu = _lbr(din, dembed, copy.deepcopy(acts[0]))
        self.logvar = _lbr(din, dembed, copy.deepcopy(acts[1]))
        self.dec = _lbr(dembed, din, copy.deepcopy(acts[2]))

    def forward(self, x, sample: bool):
        mu, lv = self.mu(x), self.logvar(x)
        z = _reparameterize(mu, lv) if sample else mu
        return self.dec(z), mu, lv


class Translator(nn.Module):
    """Shared latent space: either modality's embedding maps to both decoders."""

    def __init__(self, din_r, din_a, dembed, acts):
        super().__init__()
        self.r_mu = _lbr(din_r, dembed, copy.deepcopy(acts[0]))
        self.a_mu = _lbr(din_a, dembed, copy.deepcopy(acts[0]))
        self.r_lv = _lbr(din_r, dembed, copy.deepcopy(acts[1]))
        self.a_lv = _lbr(din_a, dembed, copy.deepcopy(acts[1]))
        self.r_dec = _lbr(dembed, din_r, copy.deepcopy(acts[2]))
        self.a_dec = _lbr(dembed, din_a, copy.deepcopy(acts[2]))

    def forward(self, x, source: str, sample: bool):
        mu, lv = (self.r_mu(x), self.r_lv(x)) if source == "rna" else (self.a_mu(x), self.a_lv(x))
        z = _reparameterize(mu, lv) if sample else mu
        return self.r_dec(z), self.a_dec(z), mu, lv


class ButterflyNet(nn.Module):
    def __init__(self, n_genes, chroms, cfg: "ScButterflyConfig"):
        super().__init__()
        act = lambda name: {"leakyrelu": nn.LeakyReLU(), "sigmoid": nn.Sigmoid(), "identity": nn.Identity()}[name]
        A = lambda names: [act(n) for n in names]
        self.rna_encoder = NetBlock([n_genes] + cfg.rna_encoder_dims, A(cfg.rna_encoder_acts), cfg.dropout_rate, cfg.rna_noise_rate)
        self.atac_encoder = ChromEncoder(chroms, cfg.atac_per_chrom, cfg.atac_encoder_dims, A(cfg.atac_encoder_acts), cfg.dropout_rate, cfg.atac_noise_rate)
        self.rna_decoder = NetBlock(cfg.rna_decoder_dims + [n_genes], A(cfg.rna_decoder_acts), cfg.dropout_rate, 0.0)
        self.atac_decoder = ChromDecoder(chroms, cfg.atac_per_chrom, cfg.atac_decoder_dims, A(cfg.atac_decoder_acts), cfg.dropout_rate)
        tacts = A(cfg.translator_acts)
        de, dr, da = cfg.embed_dim, cfg.rna_encoder_dims[-1], cfg.atac_encoder_dims[-1]
        self.r_translator = SingleTranslator(dr, de, tacts)
        self.a_translator = SingleTranslator(da, de, tacts)
        self.translator = Translator(dr, da, de, tacts)
        self.disc_r = NetBlock([dr, 1], [nn.Sigmoid()], 0.0, 0.0)
        self.disc_a = NetBlock([da, 1], [nn.Sigmoid()], 0.0, 0.0)


# ------------------------------------------------------------------- config
@dataclass
class ScButterflyConfig:
    """Hyper-parameters; defaults are scButterfly's (``construct_model``/``train_model``)."""

    rna_encoder_dims: list = field(default_factory=lambda: [256, 128])
    rna_decoder_dims: list = field(default_factory=lambda: [128, 256])
    atac_per_chrom: int = 32
    atac_encoder_dims: list = field(default_factory=lambda: [128])
    atac_decoder_dims: list = field(default_factory=lambda: [128])
    rna_encoder_acts: list = field(default_factory=lambda: ["leakyrelu", "leakyrelu"])
    rna_decoder_acts: list = field(default_factory=lambda: ["leakyrelu", "leakyrelu"])
    atac_encoder_acts: list = field(default_factory=lambda: ["leakyrelu", "leakyrelu"])
    atac_decoder_acts: list = field(default_factory=lambda: ["leakyrelu", "sigmoid"])
    translator_acts: list = field(default_factory=lambda: ["leakyrelu", "leakyrelu", "leakyrelu"])
    embed_dim: int = 128
    dropout_rate: float = 0.1
    rna_noise_rate: float = 0.5
    atac_noise_rate: float = 0.3
    lr: float = 1e-3
    discriminator_lr: float = 5e-3
    rna_pretrain_epochs: int = 100
    atac_pretrain_epochs: int = 100
    translator_epochs: int = 200
    patience: int = 50
    batch_size: int = 64
    loss_weight: tuple = (1.0, 2.0, 1.0)  # RNA recon, ATAC recon, discriminator
    kl_warmup: int = 50
    kl_mean: bool = True
    lock_encoder_and_decoder: bool = False
    discriminator_threshold: float = 1.35
    # scButterfly shuffles training cells every epoch; scTranslation's runner
    # does not (its DataLoader is built with shuffle=False)
    shuffle: bool = True
    # execution only (does not change the model)
    amp: bool = False
    compile: bool = False
    eval_batch_size: int = 512


FAST_OVERRIDES = dict(batch_size=256, lr=2e-3, amp=True, eval_batch_size=4096)


# ------------------------------------------------------------------- helpers
def _kl(mu, lv, mean=True):
    v = 1 + lv - mu.pow(2) - lv.exp()
    return -0.5 * torch.mean(v) if mean else torch.clamp(-0.5 * torch.sum(v), 0, 10000)


# ---------------------------------------------------------------------- model
@register("scbutterfly")
class ScButterfly(TranslationModel):
    """scButterfly translator (both directions are trained jointly).

    Parameters
    ----------
    fast
        Train with bf16 autocast on CUDA, batch 256 and learning rate 2e-3:
        about 1.7x faster on an H100. This is *not* equivalent to the default
        path. In the pre-registered check (PBMC 10k, 3 seeds; see
        ``tests/parity/translate_scbutterfly/MANIFEST.md``), RNA->ATAC
        accuracy matched, but ATAC->RNA predictions clustered worse (Leiden ARI
        0.35 vs 0.45) while correlating better per gene. Use the default for
        results comparable to the reference implementation.
    **kwargs
        Any :class:`ScButterflyConfig` field.
    """

    method = "scbutterfly"
    directions = ("rna2atac", "atac2rna")

    def __init__(self, *, fast: bool = False, **kwargs):
        base = dict(FAST_OVERRIDES) if fast else {}
        base.update(kwargs)
        self.config = ScButterflyConfig(**base)
        self.fast = fast
        self.net: Optional[ButterflyNet] = None
        self.history: dict = {}

    # -- training -----------------------------------------------------------
    def _fit(self, rna, atac, train_idx, val_idx, device, seed, verbose):
        cfg = self.config
        torch.manual_seed(seed)
        np.random.seed(seed)
        gen = torch.Generator().manual_seed(seed)
        chroms = chrom_blocks(atac)
        self.chroms = chroms
        net = ButterflyNet(rna.n_vars, chroms, cfg).to(device)
        self.net = net
        Xr = sp.csr_matrix(rna.X, dtype=np.float32)
        Xa = sp.csr_matrix(atac.X, dtype=np.float32)
        data = _Batches(Xr, Xa, device=device)
        use_amp = cfg.amp and device.type == "cuda"
        autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else _nullcontext
        fwd = net
        if cfg.compile:
            fwd = torch.compile(net)

        _mse, _bce = nn.MSELoss(), nn.BCELoss()

        def _f32(fn):
            # losses in float32 outside autocast (BCE is not autocast-safe)
            def call(p, y):
                with torch.autocast(device_type=device.type, enabled=False):
                    return fn(p.float(), y.float())
            return call

        mse, bce = _f32(_mse), _f32(_bce)
        w_r, w_a, w_d = cfg.loss_weight
        kl_r = 20.0 / rna.n_vars
        kl_a = 20.0 / atac.n_vars
        kl_all = kl_r + kl_a
        Adam = lambda m, lr: torch.optim.Adam(m.parameters(), lr=lr)
        opt = {
            "re": Adam(net.rna_encoder, cfg.lr), "ae": Adam(net.atac_encoder, cfg.lr),
            "rd": Adam(net.rna_decoder, cfg.lr), "ad": Adam(net.atac_decoder, cfg.lr),
            "rt": Adam(net.r_translator, cfg.lr), "at": Adam(net.a_translator, cfg.lr),
            "t": Adam(net.translator, cfg.lr),
            "dr": torch.optim.SGD(net.disc_r.parameters(), lr=cfg.discriminator_lr),
            "da": torch.optim.SGD(net.disc_a.parameters(), lr=cfg.discriminator_lr),
        }

        def step(names, loss):
            for k in names:
                opt[k].zero_grad(set_to_none=True)
            loss.backward()
            for k in names:
                opt[k].step()

        def warm(weight, epoch):
            return weight * epoch / cfg.kl_warmup if epoch < cfg.kl_warmup else weight

        hist = {k: [] for k in ("rna_pretrain", "rna_pretrain_val", "atac_pretrain", "atac_pretrain_val",
                                "translator", "translator_val", "discriminator")}

        # ---- phase 1/2: autoencoder pretraining -------------------------------
        def pretrain(kind, epochs):
            enc, dec, tr = ((net.rna_encoder, net.rna_decoder, net.r_translator) if kind == "rna"
                            else (net.atac_encoder, net.atac_decoder, net.a_translator))
            names = ("re", "rd", "rt") if kind == "rna" else ("ae", "ad", "at")
            loss_fn = mse if kind == "rna" else bce
            klw = kl_r if kind == "rna" else kl_a
            stopper = _EarlyStop(cfg.patience)
            for epoch in range(epochs):
                net.train()
                tr_losses = []
                w = warm(klw, epoch)
                for r, a in data.iterate(_epoch_batches(train_idx, cfg.batch_size, gen, cfg.shuffle)):
                    x = r if kind == "rna" else a
                    with autocast():
                        out, mu, lv = tr(enc(x), sample=True)
                        pred = dec(out)
                    rec = loss_fn(pred.float(), x)
                    loss = rec + w * _kl(mu.float(), lv.float())
                    step(names, loss)
                    tr_losses.append(rec.detach())
                net.eval()
                va = []
                with torch.no_grad():
                    for r, a in data.iterate(_epoch_batches(val_idx, cfg.batch_size, gen, False)):
                        x = r if kind == "rna" else a
                        with autocast():
                            out, mu, lv = tr(enc(x), sample=False)
                            pred = dec(out)
                        va.append(loss_fn(pred.float(), x).detach())
                tr_m, va_m = _mean(tr_losses), _mean(va)
                hist[f"{kind}_pretrain"].append(tr_m)
                hist[f"{kind}_pretrain_val"].append(va_m)
                stopper(va_m, net)
                if verbose and (epoch % 10 == 0):
                    print(f"[scButterfly] {kind} pretrain epoch {epoch}: train {tr_m:.4f} val {va_m:.4f}")
                if stopper.stop:
                    _load_parts(net, stopper.state, (enc, dec, tr))
                    break

        def forward_pair(r, a, sample):
            r2 = net.rna_encoder(r)
            a2 = net.atac_encoder(a)
            r2r, r2a, mu_r, lv_r = net.translator(r2, "rna", sample)
            a2r, a2a, mu_a, lv_a = net.translator(a2, "atac", sample)
            return r2, a2, r2r, r2a, a2r, a2a, mu_r, lv_r, mu_a, lv_a

        def disc_loss(r, a, sample):
            r2, a2, _, r2a, a2r, _, *_ = forward_pair(r, a, sample)
            n = r.shape[0]
            u = torch.rand(n, device=device, generator=None)
            # soft labels: >0.8 keep, (0.5,0.8] -> 0.8, (0.2,0.5] -> 0.2, <=0.2 keep
            lab = torch.where(u > 0.8, u, torch.where(u > 0.5, torch.full_like(u, 0.8),
                              torch.where(u > 0.2, torch.full_like(u, 0.2), u)))
            real = (lab > 0.5).unsqueeze(1)
            in_a = torch.where(real, a2, r2a)
            in_r = torch.where(real, r2, a2r)
            pa = net.disc_a(in_a).reshape(n).float()
            pr = net.disc_r(in_r).reshape(n).float()
            return bce(pa, lab) + bce(pr, lab)

        def gen_loss(r, a, klw):
            _, _, r2r, r2a, a2r, a2a, mu_r, lv_r, mu_a, lv_a = forward_pair(r, a, True)
            R2R, R2A = net.rna_decoder(r2r), net.atac_decoder(r2a)
            A2R, A2A = net.rna_decoder(a2r), net.atac_decoder(a2a)
            rec = w_r * (mse(R2R.float(), r) + mse(A2R.float(), r)) + w_a * (bce(R2A.float(), a) + bce(A2A.float(), a))
            kl = _kl(mu_r.float(), lv_r.float(), cfg.kl_mean) + _kl(mu_a.float(), lv_a.float(), cfg.kl_mean)
            return rec, kl, klw * kl + rec

        t0 = time.time()
        pretrain("rna", cfg.rna_pretrain_epochs)
        pretrain("atac", cfg.atac_pretrain_epochs)

        # ---- phase 3: translator + discriminators -----------------------------
        stopper = _EarlyStop(cfg.patience)
        gnames = ["t"] + ([] if cfg.lock_encoder_and_decoder else ["re", "ae", "rd", "ad"])
        for epoch in range(cfg.translator_epochs):
            net.train()
            tr_l, dl = [], []
            klw = warm(kl_all, epoch)
            for r, a in data.iterate(_epoch_batches(train_idx, cfg.batch_size, gen, cfg.shuffle)):
                with autocast():
                    ld = disc_loss(r, a, True)
                step(("dr", "da"), ld)
                with autocast():
                    ld = disc_loss(r, a, True)
                    rec, kl, lg = gen_loss(r, a, klw)
                ld_v = ld.item()  # the generator loss depends on it (one sync per step)
                if ld_v < cfg.discriminator_threshold:
                    lg = lg - w_d * ld
                step(gnames, lg)
                tr_l.append(rec.detach())
                dl.append(ld_v)
            net.eval()
            va = []
            with torch.no_grad():
                for r, a in data.iterate(_epoch_batches(val_idx, cfg.batch_size, gen, False)):
                    with autocast():
                        # the reference scores validation with sampled latents
                        rec, _, _ = gen_loss(r, a, klw)
                    va.append(rec.detach())
            tr_m, va_m = _mean(tr_l), _mean(va)
            hist["translator"].append(tr_m)
            hist["translator_val"].append(va_m)
            hist["discriminator"].append(float(np.mean(dl)))
            stopper(va_m, net)
            if verbose and (epoch % 10 == 0):
                print(f"[scButterfly] translator epoch {epoch}: train {tr_m:.4f} val {va_m:.4f}")
            if stopper.stop:
                _load_parts(net, stopper.state, (net.rna_encoder, net.atac_encoder, net.rna_decoder,
                                                 net.atac_decoder, net.translator, net.disc_a, net.disc_r))
                break
        hist["train_seconds"] = time.time() - t0
        self.history = hist
        net.eval()

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def _predict(self, X: sp.csr_matrix, direction: str, device) -> np.ndarray:
        net = self.net.to(device).eval()
        out = []
        bs = self.config.eval_batch_size
        for i in range(0, X.shape[0], bs):
            x = torch.from_numpy(X[i:i + bs].toarray()).to(device)
            if direction == "rna2atac":
                _, z, _, _ = net.translator(net.rna_encoder(x), "rna", False)
                y = net.atac_decoder(z)
            else:
                z, _, _, _ = net.translator(net.atac_encoder(x), "atac", False)
                y = net.rna_decoder(z)
            out.append(y.float().cpu().numpy())
        return np.vstack(out)

    def _state(self):
        return {"config": self.config.__dict__, "chroms": self.chroms, "fast": self.fast,
                "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()}}

    @classmethod
    def _from_state(cls, state, n_genes):
        obj = cls(**state["config"])
        obj.fast = state["fast"]
        obj.chroms = state["chroms"]
        obj.net = ButterflyNet(n_genes, obj.chroms, obj.config)
        obj.net.load_state_dict(state["state_dict"])
        obj.net.eval()
        return obj
