"""BABEL: cross-modality translation between scRNA-seq and scATAC-seq.

Clean-room reimplementation of BABEL (Wu, Yost, Chang & Zou, *PNAS* 2021,
"BABEL enables cross-modality translation between multiomic profiles at
single-cell resolution"). The reference code (github.com/wukevin/babel) has no
license, so no code was read or copied from it or from the copy vendored in the
scTranslation benchmark. Everything here follows the paper's Materials and
Methods. The reference was used only as a black box: its published
hyper-parameters (scTranslation ``configs/default.yaml``) and its outputs,
compared through the metrics in ``tests/parity/translate_babel``.

From the paper:

* **Latent space.** A shared, deterministic (non-variational) 16-dimensional
  latent space. Two encoders and two decoders are trained on all four paths
  (RNA→RNA, RNA→ATAC, ATAC→RNA, ATAC→ATAC).
* **RNA encoder.** genes → 64 → 16, each layer with PReLU.
* **RNA decoder.** 16 → 64 (PReLU), then two gene-wise heads: an exponential
  head for the negative-binomial mean and a softplus head for the dispersion.
* **ATAC encoder.** Per chromosome, peaks → 32 → 16 (PReLU). The blocks are
  concatenated and projected to the 16-d latent with PReLU.
* **ATAC decoder.** The inverse: 16 → 16·C (PReLU), then per chromosome
  16 → 32 (PReLU) → peaks, with a sigmoid output.
* **Loss.** ``L = NB(r | RNA) + β BCE(a | ATAC) + β BCE(a | RNA) + NB(r | ATAC)``
  with β = 1.33, where NB is the negative-binomial negative log-likelihood
  given in the paper.
* **RNA input and target.** The input is size-normalised to the median count
  per cell, log-transformed, standardised and clipped at the 0.5% / 99.5%
  quantiles of the whole matrix. The NB target is the size-normalised counts.
  The ATAC input and target are binary.
* **Training.** Adam, learning rate 0.01; the learning rate is reduced and
  training stops early on the validation loss; batch normalisation and
  gradient clipping are used for stability.

Choices where the paper gives no value (documented, not reverse-engineered):

* batch normalisation after every hidden (non-latent, non-output) linear layer,
  before its PReLU;
* gradient-norm clipping at 5;
* a constant learning rate. The paper mentions reducing it, but the reference
  runs' logs show it stayed at 0.01 throughout (``lr_patience`` enables
  ``ReduceLROnPlateau``);
* early-stopping patience 25 and batch size 64, from the scTranslation config;
* at most 500 epochs, keeping the weights with the lowest validation loss;
* losses averaged over entries;
* PyTorch default initialisation; one PReLU slope per layer;
* the NB mean is kept in [1e-5, 1e6] and the dispersion in [1e-4, 1e4]. Without
  this, the dispersion collapses to 0 and the ``logΓ(y+θ)`` gradient diverges
  within the first epochs, the standard safeguard in NB autoencoders;
* standardisation statistics and clip limits taken from the training cells and
  stored in the model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import TranslationModel, register
from ._nn import BlockLinear, ChromLinear, _Batches, _EarlyStop, _epoch_batches, _mean, _nullcontext
from ._prep import chrom_blocks, size_normalize_matrix as size_normalize


# --------------------------------------------------------------- data transforms
@dataclass
class RnaScaler:
    """log1p of size-normalised counts, standardised per gene and clipped at quantiles of the whole matrix."""

    mean: np.ndarray
    std: np.ndarray
    lo: float
    hi: float

    @classmethod
    def fit(cls, normed: sp.csr_matrix, clip: float = 0.005) -> "RnaScaler":
        L = np.log1p(normed.toarray().astype(np.float64))
        mean = L.mean(0)
        std = L.std(0)
        std[std == 0] = 1.0
        Z = (L - mean) / std
        lo, hi = np.quantile(Z, [clip, 1 - clip])
        return cls(mean, std, float(lo), float(hi))

    def transform(self, normed: sp.csr_matrix) -> np.ndarray:
        L = np.log1p(normed.toarray().astype(np.float64))
        return np.clip((L - self.mean) / self.std, self.lo, self.hi).astype(np.float32)


def nb_nll(y, mu, theta, eps: float = 1e-8):
    """Negative-binomial negative log-likelihood of ``y`` with mean ``mu`` and dispersion ``theta``.

    The paper's ``L_NB(y; ŷ, θ)``, element-wise:
    ``−θ(log(θ+ε) − log(θ+ŷ)) − y(log(ŷ+ε) − log(θ+ŷ)) − logΓ(y+θ) + logΓ(y+1) + logΓ(θ+ε)``.
    """
    log_theta_mu = torch.log(theta + mu + eps)
    return (-theta * (torch.log(theta + eps) - log_theta_mu)
            - y * (torch.log(mu + eps) - log_theta_mu)
            - torch.lgamma(y + theta) + torch.lgamma(y + 1.0) + torch.lgamma(theta + eps))


# ----------------------------------------------------------------------- network
def _hidden(din, dout):
    """Linear -> BatchNorm -> PReLU."""
    return nn.Sequential(nn.Linear(din, dout), nn.BatchNorm1d(dout), nn.PReLU())


class BabelNet(nn.Module):
    def __init__(self, n_genes: int, chroms, cfg: "BabelConfig"):
        super().__init__()
        n, H, L = len(chroms), cfg.rna_hidden, cfg.latent_dim
        self.chroms = list(chroms)
        # RNA encoder / decoder
        self.rna_enc = nn.Sequential(_hidden(n_genes, H), nn.Linear(H, L), nn.PReLU())
        self.rna_dec = _hidden(L, H)
        self.rna_mu = nn.Linear(H, n_genes)
        self.rna_theta = nn.Linear(H, n_genes)
        # ATAC encoder: per chromosome peaks -> 32 -> 16, concatenate, -> latent
        a1, a2 = cfg.atac_chrom_hidden, cfg.atac_chrom_latent
        self.atac_enc1 = ChromLinear(chroms, a1, "in", init="default")
        self.atac_enc1_post = nn.Sequential(nn.BatchNorm1d(n * a1), nn.PReLU())
        self.atac_enc2 = BlockLinear(n, a1, a2)
        self.atac_enc2_post = nn.Sequential(nn.BatchNorm1d(n * a2), nn.PReLU())
        self.atac_enc3 = nn.Sequential(nn.Linear(n * a2, L), nn.PReLU())
        # ATAC decoder: latent -> 16*C, per chromosome 16 -> 32 -> peaks
        self.atac_dec1 = _hidden(L, n * a2)
        self.atac_dec2 = BlockLinear(n, a2, a1)
        self.atac_dec2_post = nn.Sequential(nn.BatchNorm1d(n * a1), nn.PReLU())
        self.atac_dec3 = ChromLinear(chroms, a1, "out", init="default")
        self.max_log_mu = cfg.max_log_mu

    def encode_rna(self, x):
        return self.rna_enc(x)

    def encode_atac(self, x):
        h = self.atac_enc1_post(self.atac_enc1(x))
        h = self.atac_enc2_post(self.atac_enc2(h))
        return self.atac_enc3(h)

    def decode_rna(self, z):
        h = self.rna_dec(z)
        mu = torch.clamp(torch.exp(torch.clamp(self.rna_mu(h), max=self.max_log_mu)), 1e-5, 1e6)
        # the dispersion is kept in [1e-4, 1e4]: as theta -> 0 the gradient of
        # logGamma(y + theta) at y = 0 diverges (digamma(0)) and training turns NaN
        theta = torch.clamp(F.softplus(self.rna_theta(h)), 1e-4, 1e4)
        return mu, theta

    def decode_atac(self, z):
        h = self.atac_dec1(z)
        h = self.atac_dec2_post(self.atac_dec2(h))
        return torch.sigmoid(self.atac_dec3(h))


# ------------------------------------------------------------------------ config
@dataclass
class BabelConfig:
    """Hyper-parameters. Paper values: latent 16, RNA hidden 64, ATAC 32/16 per chromosome,
    β = 1.33, Adam, learning rate 0.01. scTranslation config: batch 64, early stop 25."""

    latent_dim: int = 16
    rna_hidden: int = 64
    atac_chrom_hidden: int = 32
    atac_chrom_latent: int = 16
    beta: float = 1.33
    lr: float = 0.01
    batch_size: int = 64
    max_epochs: int = 500
    patience: int = 25
    lr_patience: Optional[int] = None  # None: constant learning rate (as the reference ran)
    lr_factor: float = 0.1
    grad_clip: float = 5.0
    clip_quantile: float = 0.005
    max_log_mu: float = 20.0
    shuffle: bool = True
    # execution only
    amp: bool = False
    eval_batch_size: int = 512


FAST_OVERRIDES = dict(batch_size=256, amp=True, eval_batch_size=2048)


# ------------------------------------------------------------------------- model
@register("babel")
class Babel(TranslationModel):
    """BABEL translator (both directions are trained jointly).

    Expects **raw RNA counts** (e.g. the highly variable genes) and binary or
    count ATAC peaks (binarised internally); see
    ``epi.translate.prepare(..., method="babel")``. RNA→ATAC predictions are
    peak probabilities in [0, 1]. ATAC→RNA predictions are the negative-binomial
    means, in size-normalised counts; ``rna_target_sum`` is the per-cell total
    they are scaled to (the training median). The measured counts can be put on
    the same scale with :func:`epione.translate.size_normalize`.

    Parameters
    ----------
    fast
        bf16 autocast on CUDA, batch 256. Validated separately; see
        ``tests/parity/translate_babel/MANIFEST.md``.
    **kwargs
        Any :class:`BabelConfig` field.
    """

    method = "babel"
    directions = ("rna2atac", "atac2rna")
    default_sparsify = None  # predictions are probabilities / NB means, kept dense

    def __init__(self, *, fast: bool = False, **kwargs):
        base = dict(FAST_OVERRIDES) if fast else {}
        base.update(kwargs)
        self.config = BabelConfig(**base)
        self.fast = fast
        self.net: Optional[BabelNet] = None
        self.history: dict = {}

    # -- data ----------------------------------------------------------------
    @staticmethod
    def _binary(X):
        X = sp.csr_matrix(X, dtype=np.float32, copy=True)
        X.data = (X.data > 0).astype(np.float32)
        X.eliminate_zeros()
        return X

    def _rna_input(self, X):
        normed, _ = size_normalize(X)
        return self.scaler.transform(normed)

    # -- training --------------------------------------------------------------
    def _fit(self, rna, atac, train_idx, val_idx, device, seed, verbose):
        cfg = self.config
        torch.manual_seed(seed)
        np.random.seed(seed)
        gen = torch.Generator().manual_seed(seed)
        self.chroms = chrom_blocks(atac)
        counts = sp.csr_matrix(rna.X, dtype=np.float32)
        if counts.nnz and not np.allclose(counts.data, np.round(counts.data)):
            import warnings
            warnings.warn("BABEL expects raw RNA counts; rna.X is not integer-valued")
        # each split is size-normalised to its own median, as the reference does
        tr_norm, self.rna_target_sum = size_normalize(counts[train_idx])
        va_norm, _ = size_normalize(counts[val_idx])
        self.scaler = RnaScaler.fit(tr_norm, cfg.clip_quantile)
        # stacked [train; validation] matrices
        Xin_s = np.vstack([self.scaler.transform(tr_norm), self.scaler.transform(va_norm)])
        Xtg = sp.vstack([tr_norm, va_norm]).tocsr()
        Xa = self._binary(atac.X)[np.r_[train_idx, val_idx]]
        tr = np.arange(len(train_idx))
        va = len(train_idx) + np.arange(len(val_idx))

        net = BabelNet(rna.n_vars, self.chroms, cfg).to(device)
        self.net = net
        data = _Batches(Xin_s, Xtg, Xa, device=device)
        use_amp = cfg.amp and device.type == "cuda"
        autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else _nullcontext
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        sched = (None if cfg.lr_patience is None else
                 torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=cfg.lr_factor, patience=cfg.lr_patience))
        stopper = _EarlyStop(cfg.patience)
        beta = cfg.beta

        def losses(xin, ytg, a):
            with autocast():
                zr, za = net.encode_rna(xin), net.encode_atac(a)
                mu_rr, th_rr = net.decode_rna(zr)
                mu_ar, th_ar = net.decode_rna(za)
                p_aa, p_ra = net.decode_atac(za), net.decode_atac(zr)
            with torch.autocast(device_type=device.type, enabled=False):
                nb_rr = nb_nll(ytg, mu_rr.float(), th_rr.float()).mean()
                nb_ar = nb_nll(ytg, mu_ar.float(), th_ar.float()).mean()
                bce_aa = F.binary_cross_entropy(p_aa.float(), a)
                bce_ra = F.binary_cross_entropy(p_ra.float(), a)
            total = nb_rr + beta * bce_aa + beta * bce_ra + nb_ar
            return total, (nb_rr, bce_aa, bce_ra, nb_ar)

        hist = {"train": [], "val": [], "lr": [], "val_parts": []}
        t0 = time.time()
        for epoch in range(cfg.max_epochs):
            net.train()
            tl = []
            for xin, ytg, a in data.iterate(_epoch_batches(tr, cfg.batch_size, gen, cfg.shuffle)):
                loss, _ = losses(xin, ytg, a)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                opt.step()
                tl.append(loss.detach())
            net.eval()
            vl, parts = [], []
            with torch.no_grad():
                for xin, ytg, a in data.iterate(_epoch_batches(va, cfg.batch_size, gen, False)):
                    loss, p = losses(xin, ytg, a)
                    vl.append(loss)
                    parts.append(torch.stack(p))
            v = _mean(vl)
            hist["train"].append(_mean(tl))
            hist["val"].append(v)
            hist["val_parts"].append(torch.stack(parts).mean(0).float().cpu().numpy().tolist())
            hist["lr"].append(opt.param_groups[0]["lr"])
            if sched is not None:
                sched.step(v)
            stopper(v, net)
            if verbose and epoch % 10 == 0:
                print(f"[BABEL] epoch {epoch}: train {hist['train'][-1]:.4f} val {v:.4f} lr {hist['lr'][-1]:.2e}")
            if stopper.stop:
                break
        net.load_state_dict(stopper.state)  # best validation weights
        hist["train_seconds"] = time.time() - t0
        hist["best_epoch"] = int(np.argmin(hist["val"]))
        self.history = hist
        net.eval()

    # -- inference ---------------------------------------------------------------
    @torch.no_grad()
    def _predict(self, X: sp.csr_matrix, direction: str, device) -> np.ndarray:
        net = self.net.to(device).eval()
        Xin = self._rna_input(X) if direction == "rna2atac" else self._binary(X)
        out = []
        bs = self.config.eval_batch_size
        for i in range(0, X.shape[0], bs):
            blk = Xin[i:i + bs]
            x = torch.from_numpy(blk.toarray() if sp.issparse(blk) else blk).to(device)
            if direction == "rna2atac":
                y = net.decode_atac(net.encode_rna(x))
            else:
                y, _ = net.decode_rna(net.encode_atac(x))
            out.append(y.float().cpu().numpy())
        return np.vstack(out)

    def _state(self):
        return {"config": self.config.__dict__, "chroms": self.chroms, "fast": self.fast,
                "rna_target_sum": self.rna_target_sum,
                "scaler": {k: getattr(self.scaler, k) for k in ("mean", "std", "lo", "hi")},
                "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()}}

    @classmethod
    def _from_state(cls, state, n_genes):
        obj = cls(**state["config"])
        obj.fast = state["fast"]
        obj.chroms = state["chroms"]
        obj.rna_target_sum = state["rna_target_sum"]
        obj.scaler = RnaScaler(**state["scaler"])
        obj.net = BabelNet(n_genes, obj.chroms, obj.config)
        obj.net.load_state_dict(state["state_dict"])
        obj.net.eval()
        return obj
