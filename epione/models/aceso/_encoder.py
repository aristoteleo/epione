"""Stage 1 — state encoder: scGPT on panel RNA, distilled onto the chromatin state z.

Model construction, weight loading and input format follow the official scGPT cell-embedding
path (``scgpt.tasks.cell_emb.embed_data`` / ``get_batch_cell_embeddings``, bowang-lab/scGPT):
gene tokens with ``<cls>`` prepended, per-cell quantile binning (official ``DataCollator``,
51 bins, max length 1200), cell state = ``<cls>`` output of ``model._encode``. A linear head maps
it to the dimension of z; the output is L2-normalised.

Training (:func:`train_encoder`): full fine-tuning as in the official scGPT tutorials (Adam lr
1e-4, StepLR gamma 0.9 per epoch, AMP, grad clip 1); loss ``(1 - cos) + symmetric InfoNCE`` (tau
0.07) against L2-normalised z; one panel per batch from a :class:`PanelSampler`; cells with no
detected panel gene are dropped from the batch (they carry no input); early stopping on held-out
cells evaluated on the target panel plus fixed random panels.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from ._panel import PanelSampler, PanelSpace


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


class StateEncoder:
    """scGPT + linear head -> L2-normalised state. Wraps an ``nn.Module`` (``self.net``)."""

    def __init__(self, model_dir: str, z_dim: int, device: str = "cuda", load_weights: bool = True):
        torch, nn, F = _torch()
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab
        from scgpt.data_collator import DataCollator
        from scgpt.utils import load_pretrained
        self.vocab = GeneVocab.from_file(os.path.join(model_dir, "vocab.json"))
        for s in ("<pad>", "<cls>", "<eoc>"):
            if s not in self.vocab:
                self.vocab.append_token(s)
        self.cfg = json.load(open(os.path.join(model_dir, "args.json")))
        c = self.cfg
        scgpt = TransformerModel(ntoken=len(self.vocab), d_model=c["embsize"], nhead=c["nheads"], d_hid=c["d_hid"],
                                 nlayers=c["nlayers"], nlayers_cls=c["n_layers_cls"], n_cls=1, vocab=self.vocab,
                                 dropout=c["dropout"], pad_token=c["pad_token"], pad_value=c["pad_value"], do_mvc=True,
                                 do_dab=False, use_batch_labels=False, domain_spec_batchnorm=False,
                                 explicit_zero_prob=False, use_fast_transformer=False, pre_norm=False)
        if load_weights:
            load_pretrained(scgpt, torch.load(os.path.join(model_dir, "best_model.pt"), map_location="cpu"), verbose=False)
        self.net = nn.ModuleDict({"scgpt": scgpt, "head": nn.Linear(c["embsize"], z_dim)}).to(device)
        self.collator = DataCollator(do_padding=True, pad_token_id=self.vocab[c["pad_token"]], pad_value=c["pad_value"],
                                     do_mlm=False, do_binning=True, max_length=1200, sampling=True, keep_first_n_tokens=1)
        self.pad_id, self.cls_id, self.device = self.vocab[c["pad_token"]], self.vocab["<cls>"], device

    # ---- input
    def tokenize(self, M: sp.csr_matrix, vocab_ids: np.ndarray) -> dict:
        """Counts (cells x panel genes, CSR, no all-zero rows) -> official collated batch."""
        torch = _torch()[0]
        ex = []
        for i in range(M.shape[0]):
            s, e = M.indptr[i], M.indptr[i + 1]
            nz, v = M.indices[s:e], M.data[s:e]; k = v > 0
            ex.append({"genes": torch.from_numpy(np.insert(vocab_ids[nz[k]], 0, self.cls_id)).long(),
                       "expressions": torch.from_numpy(np.insert(v[k], 0, self.cfg["pad_value"])).float()})
        return self.collator(ex)

    # ---- forward
    def __call__(self, batch: dict):
        torch, nn, F = _torch()
        g = batch["gene"].to(self.device, non_blocking=True)
        h = self.net["scgpt"]._encode(g, batch["expr"].to(self.device, non_blocking=True),
                                      src_key_padding_mask=g.eq(self.pad_id))[:, 0, :]
        return F.normalize(self.net["head"](h).float(), dim=1)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


@dataclass
class EncoderConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-4
    gamma: float = 0.9
    tau: float = 0.07
    patience: int = 5
    max_steps: int = 0
    seed: int = 0


def alignment_loss(p, z, tau: float):
    """(1 - cos) + symmetric InfoNCE between L2-normalised predictions and targets."""
    torch, nn, F = _torch()
    logits = p @ z.T / tau; lab = torch.arange(len(p), device=p.device)
    return (1 - (p * z).sum(1)).mean() + 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))


def _drop_empty(M: sp.csr_matrix, rows: np.ndarray):
    M = sp.csr_matrix(M); M.eliminate_zeros(); ok = np.diff(M.indptr) > 0
    return M[ok], rows[ok]


def predict_states(enc: StateEncoder, X: sp.spmatrix, rows: np.ndarray, cols: np.ndarray, vocab_ids: np.ndarray,
                   depths: Optional[np.ndarray] = None, seed: int = 0, batch_size: int = 256) -> np.ndarray:
    """States for ``rows`` of ``X`` read on panel ``cols`` (optionally thinned to ``depths``);
    NaN rows for cells with no detected panel gene."""
    torch = _torch()[0]
    from ._panel import thin_counts
    M = sp.csr_matrix(X)[rows][:, cols]
    if depths is not None:
        M = thin_counts(M, depths, np.random.default_rng(seed))
    M, kept = _drop_empty(M, np.arange(len(rows)))
    out = np.full((len(rows), enc.net["head"].out_features), np.nan, np.float32)
    enc.net.eval()
    with torch.no_grad():
        for s in range(0, len(kept), batch_size):
            with torch.autocast("cuda", dtype=torch.float16):
                out[kept[s:s + batch_size]] = enc(enc.tokenize(M[s:s + batch_size], vocab_ids[cols])).cpu().numpy()
    return out


def train_encoder(enc: StateEncoder, X: sp.spmatrix, Z: np.ndarray, space: PanelSpace, sampler: PanelSampler,
                  train_rows: np.ndarray, val_rows: np.ndarray, out_dir: str, config: EncoderConfig = EncoderConfig(),
                  val_panels: Optional[Sequence] = None, log: Callable = print):
    """Panel-agnostic distillation of ``Z`` (cells x d, rows of ``X``) into ``enc``.

    ``val_panels``: list of ``(name, cols, depth_scale)``; default = target panel + 3 random panels
    (150, 400, 800 genes) drawn once with a fixed seed (or only the panels the sampler can produce).
    Writes ``best.pt`` (encoder state + config) and ``train.log``.
    """
    torch, nn, F = _torch()
    c = config
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(c.seed)
    X = sp.csr_matrix(X, dtype=np.float32)
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True); Zt = torch.from_numpy(Zn.astype(np.float32))
    if val_panels is None:
        g = np.random.default_rng(7)
        rand = [(f"random{n}", np.sort(g.choice(space.pool, n, replace=False)), n / len(space.target)) for n in (150, 400, 800)]
        val_panels = {"mixed": [("target", space.target, 1.0)] + rand, "fixed": [("target", space.target, 1.0)],
                      "random": rand}[sampler.mode]
    params = list(enc.net.parameters())
    opt = torch.optim.Adam(params, lr=c.lr, eps=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=c.gamma)
    scaler = torch.cuda.amp.GradScaler()
    best, bad, step = np.inf, 0, 0
    flog = open(os.path.join(out_dir, "train.log"), "a")
    vocab_ids = space.vocab_ids
    for ep in range(c.epochs):
        enc.net.train(); t0 = time.time(); losses = []
        rng = np.random.default_rng(100 + ep)
        perm = rng.permutation(train_rows)
        for s in range(0, len(perm) - c.batch_size + 1, c.batch_size):
            rows = perm[s:s + c.batch_size]
            cols, scale = sampler.draw(rng)
            M, rows = _drop_empty(sampler.counts(X, rows, cols, scale, rng), rows)
            if len(rows) < 2:
                continue
            with torch.autocast("cuda", dtype=torch.float16):
                p = enc(enc.tokenize(M, vocab_ids[cols]))
            loss = alignment_loss(p, Zt[rows].to(enc.device), c.tau)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update()
            losses.append(loss.item()); step += 1
            if c.max_steps and step >= c.max_steps:
                break
        sched.step()
        vres = {}
        for name, cols, scale in val_panels:
            P = predict_states(enc, X, val_rows, cols, vocab_ids, depths=np.random.default_rng(1).choice(
                sampler.target_depths, len(val_rows)) * scale, seed=1)
            m = ~np.isnan(P[:, 0])
            pt = torch.from_numpy(P[m]).to(enc.device)
            vres[name] = (alignment_loss(pt, Zt[val_rows[m]].to(enc.device), c.tau).item(),
                          float((P[m] * Zn[val_rows[m]]).sum(1).mean()))
        vl = float(np.mean([v[0] for v in vres.values()]))
        rec = dict(epoch=ep, step=step, train_loss=float(np.mean(losses)), val_loss=vl,
                   val_cos={k: round(v[1], 4) for k, v in vres.items()}, s=round(time.time() - t0, 1))
        log(json.dumps(rec)); flog.write(json.dumps(rec) + "\n"); flog.flush()
        if vl < best:
            best, bad = vl, 0
            torch.save({"encoder": enc.state_dict(), "config": asdict(c), "mode": sampler.mode}, os.path.join(out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= c.patience:
                break
        if c.max_steps and step >= c.max_steps:
            break
    enc.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location="cpu", weights_only=False)["encoder"])
    return enc
