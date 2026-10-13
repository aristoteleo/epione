"""JAMIE: joint variational autoencoders for multi-modal imputation and embedding.

Port of JAMIE (Cohen Kalafut, Huang & Wang, *Nat Mach Intell* 2023;
github.com/daifengwanglab/JAMIE) into Epione, in the configuration the
scTranslation benchmark runs.

* Each modality is reduced to 512 principal components, fitted on the training
  cells and standardised by the global mean and SD of the training scores.
* A variational encoder-decoder per modality (``edModelVar``):
  512 → 1024 → 512 (BatchNorm, LeakyReLU), a 32-d mean and log-variance, and
  the mirrored decoder. The two latents are combined through the cell
  correspondence P (identity for paired cells).
* The loss sums four terms: a KL term with sigmoid annealing, reconstruction
  in PCA space, the Euclidean alignment of each embedding to the combined one,
  and an F-term tying the combined latents through UnionCom's Prime-Dual
  correspondence F, computed from geodesic (kNN shortest-path) distances.
* Translation is ``decoder_to(μ(encoder_from(x)))``, mapped back through the
  inverse standardisation and PCA, so it lands in the feature space the model
  was trained on (raw counts for RNA, z-scores for ATAC in the reference setting).

Adapted from JAMIE, Copyright (C) 2022 daifengwanglab, distributed under the GNU
General Public License v3 (the same licence as Epione). It is modified here: the
UnionCom base class is folded in, and geodesic distances use Dijkstra instead of
Floyd–Warshall (the same shortest paths).

Prime-Dual matching and the geodesic distance routine are adapted from UnionCom
(Cao, Bai, Hong & Wan), distributed under the MIT License:

    MIT License

    Copyright (c) 2020 caokai1073

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

import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from ._base import TranslationModel, register
from ._nn import _nullcontext


# --------------------------------------------------------- UnionCom pieces
def geodesic_distances(X, kmax: int = 40) -> np.ndarray:
    """Shortest-path distances on a Euclidean kNN graph (UnionCom's ``geodesic_distances``).

    k starts at 5 and grows by 2 until the graph is connected or exceeds
    ``max(kmax, 0.01 n)``; unreachable pairs get twice the largest finite
    distance. Dijkstra replaces Floyd–Warshall (same shortest paths).
    """
    from scipy.sparse.csgraph import connected_components, dijkstra
    from sklearn.neighbors import NearestNeighbors

    kmin = 5

    def graph(k):
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1).fit(X)
        return nbrs.kneighbors_graph(X, mode="distance")

    knn = graph(kmin)
    while connected_components(knn, directed=False)[0] != 1:
        if kmin > np.max((kmax, 0.01 * X.shape[0])):
            break
        kmin += 2
        knn = graph(kmin)
    dist = dijkstra(knn, directed=False)
    finite = dist[np.isfinite(dist)]
    dist_max = np.nanmax(finite)
    dist[dist > dist_max] = 2 * dist_max
    return dist


def prime_dual(Kx: np.ndarray, Ky: np.ndarray, dx: int, dy: int, *, device, epoch_pd: int = 2000,
               rho: float = 10.0, epsilon: float = 0.01, delay: int = 0) -> torch.Tensor:
    """UnionCom's Prime-Dual + Adam search for the soft correspondence F (JAMIE's override)."""
    N = int(max(Kx.shape[0], Ky.shape[0]))
    Kx = torch.from_numpy(Kx / N).float().to(device)
    Ky = torch.from_numpy(Ky / N).float().to(device)
    a = np.sqrt(dy / dx)
    m, n = Kx.shape[0], Ky.shape[0]
    F = torch.zeros((m, n), device=device)
    Im = torch.ones((m, 1), device=device)
    In = torch.ones((n, 1), device=device)
    Lambda = torch.zeros((n, 1), device=device)
    Mu = torch.zeros((m, 1), device=device)
    S = torch.zeros((n, 1), device=device)
    pho1, pho2, delta = 0.9, 0.999, 10e-8
    m1 = torch.zeros((m, n), device=device)
    m2 = torch.zeros((m, n), device=device)
    for i in range(1, epoch_pd + 1):
        FKy = F @ Ky
        # rho * F @ ones(n, n) == rho * (row sums of F) broadcast over columns
        grad = (4 * FKy @ (F.T @ FKy) - 4 * a * (Kx @ FKy) + Mu @ In.T + Im @ Lambda.T
                + rho * (F.sum(1, keepdim=True).expand(m, n) + Im @ (Im.T @ F + (S - 2 * In).T)))
        m1 = pho1 * m1 + (1 - pho1) * grad
        m2 = pho2 * m2 + (1 - pho2) * grad * grad
        g = (m1 / (1 - pho1 ** i)) / (torch.sqrt(m2 / (1 - pho2 ** i)) + delta)
        F_tmp = torch.clamp(F - g, min=0)
        F = (1 - epsilon) * F + epsilon * F_tmp
        grad_s = Lambda + rho * (F.T @ Im - In + S)
        S = (1 - epsilon) * S + epsilon * torch.clamp(S - grad_s, min=0)
        Mu = Mu + epsilon * (F @ In - Im)
        Lambda = Lambda + epsilon * (F.T @ Im - In + S)
        if i >= delay:
            a = torch.trace(Kx @ (F @ Ky @ F.T)) / torch.trace(Kx @ Kx)
    return F


# ----------------------------------------------------------------- network
class EDModelVar(nn.Module):
    """JAMIE's ``edModelVar``: one variational encoder-decoder per modality."""

    def __init__(self, input_dim, output_dim, dropout: float = 0.0):
        super().__init__()
        self.num_modalities = len(input_dim)
        mk = lambda dims, last_plain=False: _mlp(dims, dropout, last_plain)
        self.encoders = nn.ModuleList([mk([d, 2 * d, d]) for d in input_dim])
        self.fc_mus = nn.ModuleList([nn.Linear(d, output_dim) for d in input_dim])
        self.fc_vars = nn.ModuleList([nn.Linear(d, output_dim) for d in input_dim])
        self.decoders = nn.ModuleList([mk([output_dim, d, 2 * d, d], last_plain=True) for d in input_dim])
        self.sigma = nn.Parameter(torch.rand(self.num_modalities))

    def refactor(self, X, index=None):
        index = range(self.num_modalities) if index is None else index
        zs, mus, logvars = [], [], []
        for x, i in zip(X, index):
            mu, logvar = self.fc_mus[i](x), self.fc_vars[i](x)
            if self.training:
                std = torch.exp(logvar / 2) + 1e-7
                zs.append(torch.distributions.Normal(mu, std).rsample())
            else:
                zs.append(mu)
            mus.append(mu)
            logvars.append(logvar)
        return zs, mus, logvars

    def combine(self, X, corr):
        out = []
        for i in range(self.num_modalities):
            j = (i + 1) % 2
            C = corr if i == 0 else corr.T
            out.append((self.sigma[i] * X[i] + self.sigma[j] * (C @ X[j]))
                       / (self.sigma[i] + self.sigma[j] * corr.sum(j).reshape(-1, 1)))
        return out

    def forward(self, *X, corr):
        zs, mus, logvars = self.refactor([self.encoders[i](X[i]) for i in range(self.num_modalities)])
        combined = self.combine(zs, corr)
        X_hat = [self.decoders[i](combined[i]) for i in range(self.num_modalities)]
        return zs, combined, X_hat, mus, logvars

    def impute(self, X, from_mod, to_mod):
        z = self.refactor([self.encoders[from_mod](X)], [from_mod])[0][0]
        return self.decoders[to_mod](z)


def _mlp(dims, dropout, last_plain):
    layers = []
    for k in range(len(dims) - 1):
        layers.append(nn.Linear(dims[k], dims[k + 1]))
        if last_plain and k == len(dims) - 2:
            break
        layers += [nn.BatchNorm1d(dims[k + 1]), nn.LeakyReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


@dataclass
class _Standardised:
    """PCA (optional) followed by subtraction of the global mean and division by the global SD of the fitted scores."""

    pca: object
    mean: float
    std: float

    def transform(self, X):
        out = self.pca.transform(X) if self.pca is not None else X
        out = (out - self.mean) / self.std
        return np.nan_to_num(out)

    def inverse(self, Z):
        out = Z * self.std + self.mean
        return self.pca.inverse_transform(out) if self.pca is not None else out


# ------------------------------------------------------------------ config
@dataclass
class JAMIEConfig:
    """Hyper-parameters; defaults are the scTranslation configuration of JAMIE."""

    output_dim: int = 32
    pca_dim: list = field(default_factory=lambda: [512, 512])
    dropout: float = 0.0
    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 10000
    min_epochs: int = 500
    min_increment: float = 1e-8
    max_steps_without_increment: int = 500
    loss_weights: list = field(default_factory=lambda: [1, 1, 1, 1])
    use_f_tilde: bool = True
    kmax: int = 40
    epoch_pd: int = 2000
    rho: float = 10.0
    epsilon: float = 0.01
    # execution only
    amp: bool = False
    eval_batch_size: int = 1024


FAST_OVERRIDES = dict(batch_size=512, amp=True)


# ------------------------------------------------------------------- model
@register("jamie")
class JAMIE(TranslationModel):
    """JAMIE translator (both directions from one jointly trained model).

    Expects RNA and ATAC in the space they should be predicted in; see
    ``epi.translate.prepare(..., method="jamie")``, which gives raw RNA counts
    and z-scored ATAC as scTranslation runs it. The validation cells are not
    used, matching the reference; training stops on the training loss.

    Parameters
    ----------
    fast
        bf16 autocast on CUDA and batch 512. Validated separately; see
        ``tests/parity/translate_jamie/MANIFEST.md``.
    **kwargs
        Any :class:`JAMIEConfig` field.
    """

    method = "jamie"
    directions = ("rna2atac", "atac2rna")
    default_sparsify = None

    def __init__(self, *, fast: bool = False, **kwargs):
        base = dict(FAST_OVERRIDES) if fast else {}
        base.update(kwargs)
        self.config = JAMIEConfig(**base)
        self.fast = fast
        self.net: Optional[EDModelVar] = None
        self.history: dict = {}

    def _fit(self, rna, atac, train_idx, val_idx, device, seed, verbose):
        from sklearn.decomposition import PCA

        cfg = self.config
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        t0 = time.time()
        data = [np.asarray(m[train_idx].toarray() if sp.issparse(m) else m[train_idx], dtype=np.float64)
                for m in (rna.X, atac.X)]
        n = data[0].shape[0]
        # correspondence F (Prime-Dual on geodesic distances), as JAMIE computes without a match file
        if cfg.use_f_tilde:
            dists = [geodesic_distances(d, cfg.kmax) for d in data]
            F = prime_dual(dists[0], dists[1], data[0].shape[1], data[1].shape[1], device=device,
                           epoch_pd=cfg.epoch_pd, rho=cfg.rho, epsilon=cfg.epsilon)
            del dists
        else:
            F = torch.zeros((n, n), device=device)
        t_match = time.time() - t0
        # PCA + global standardisation
        self.pre = []
        Z = []
        for dim, d in zip(cfg.pca_dim, data):
            dim = min(dim, *d.shape)
            pca = PCA(n_components=dim)
            s = pca.fit_transform(d)
            pre = _Standardised(pca, float(s.mean()), float(s.std()))
            self.pre.append(pre)
            Z.append(torch.from_numpy(pre.transform(d)).float().to(device))
        del data
        net = EDModelVar([z.shape[1] for z in Z], cfg.output_dim, dropout=cfg.dropout).to(device)
        self.net = net
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        P = torch.eye(n, device=device)
        n_batches = max(1, int(n / cfg.batch_size))
        bs = cfg.batch_size if int(n / cfg.batch_size) > 0 else n
        rep = min(z.shape[1] for z in Z) < bs
        ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if (cfg.amp and device.type == "cuda") else _nullcontext
        w = cfg.loss_weights
        c = (cfg.min_epochs / 2) if cfg.min_epochs > 0 else (cfg.max_epochs / 2)
        best_running, streak, hist = np.inf, 0, {"loss": [], "best_batch": []}
        net.train()
        for epoch in range(cfg.max_epochs):
            best_batch, epoch_loss = np.inf, 0.0
            kl_anneal = 1 / (1 + np.exp(-5 * (epoch - c) / c))
            for _ in range(n_batches):
                idx = np.random.choice(n, bs, replace=rep)
                it = torch.from_numpy(idx).to(device)
                x = [z[it] for z in Z]
                Pb = P[it][:, it]
                Pb = Pb / torch.where(Pb.sum(1) == 0, torch.ones_like(Pb.sum(1)), Pb.sum(1))[:, None]
                Fb = F[it][:, it]
                Fs = Fb.sum(1)
                Fb = Fb / torch.where(Fs == 0, torch.ones_like(Fs), Fs)[:, None]
                with ac():
                    emb, comb, rec, mus, logvars = net(*x, corr=Pb)
                with torch.autocast(device_type=device.type, enabled=False):
                    emb = [e.float() for e in emb]
                    comb = [e.float() for e in comb]
                    # JAMIE's refactor returns the *last* modality's log-variance tensor
                    # instead of the list, so logvars[i] is row i of that tensor, broadcast
                    # against modality i's means. Reproduced as the reference computes it.
                    lv_last = logvars[-1].float()
                    kl = sum(-0.5 * torch.mean(1 + lv_last[i] - mus[i].float().square() - lv_last[i].exp(), axis=1).mean()
                             for i in range(2))
                    rec_l = sum((r.float() - xi).square().mean(1).mean() for r, xi in zip(rec, x))
                    # euclidean "sim/diff": the loss uses the diagonal of the distance matrix
                    cos_l = sum(torch.diag(torch.cdist(emb[i], comb[i], p=2).square()).mean() / emb[i].shape[1]
                                for i in range(2))
                    f_l = (comb[0] - Fb @ comb[1]).square().mean(1).mean()
                    loss = (w[0] * 32e-3 * kl_anneal * kl + w[1] * rec_l + w[2] * 32 * cos_l + w[3] * f_l)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1)
                opt.step()
                lv_ = float(loss.detach())
                epoch_loss += lv_ / n_batches
                best_batch = min(best_batch, lv_)
            hist["loss"].append(epoch_loss)
            hist["best_batch"].append(best_batch)
            if verbose and (epoch + 1) % 100 == 0:
                print(f"[JAMIE] epoch {epoch + 1}: loss {epoch_loss:.4f}")
            if epoch > cfg.min_epochs:
                if best_running - best_batch > cfg.min_increment:
                    best_running, streak = best_batch, 0
                else:
                    streak += 1
                if streak >= cfg.max_steps_without_increment:
                    break
        net.eval()
        hist["epochs"] = len(hist["loss"])
        hist["match_seconds"] = t_match
        hist["train_seconds"] = time.time() - t0
        self.history = hist

    @torch.no_grad()
    def _predict(self, X: sp.csr_matrix, direction: str, device) -> np.ndarray:
        frm, to = (0, 1) if direction == "rna2atac" else (1, 0)
        net = self.net.to(device).eval()
        out = []
        bs = self.config.eval_batch_size
        for i in range(0, X.shape[0], bs):
            blk = X[i:i + bs]
            blk = blk.toarray() if sp.issparse(blk) else np.asarray(blk)
            z = torch.from_numpy(self.pre[frm].transform(blk.astype(np.float64))).float().to(device)
            dec = net.impute(z, frm, to).float().cpu().numpy()
            out.append(np.asarray(self.pre[to].inverse(dec), dtype=np.float32))
        return np.vstack(out)

    def _state(self):
        return {"config": self.config.__dict__, "fast": self.fast, "pre": self.pre,
                "dims": [p.pca.n_components_ for p in self.pre],
                "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()}}

    @classmethod
    def _from_state(cls, state, n_genes):
        obj = cls(**state["config"])
        obj.fast = state["fast"]
        obj.pre = state["pre"]
        obj.net = EDModelVar(state["dims"], obj.config.output_dim, dropout=obj.config.dropout)
        obj.net.load_state_dict(state["state_dict"])
        obj.net.eval()
        return obj
