"""Stage 0 — the chromatin state space: EpiAgent loading, CLS embedding, UFE fine-tuning.

:func:`finetune_ufe` is the official unsupervised fine-tuning recipe
(``epiagent.train.fine_tune_epiagent_for_UFE`` @ 9c4b790): CCA + SR losses, full fine-tuning,
AdamW(lr 1e-4, weight decay 0.01), Noam LambdaLR ``min((s+1)^-0.5, (s+1)*warmup^-1.5)`` with
warmup 10,000, fp16 autocast + GradScaler, batch 5, max length 8,192, alpha_for_CCA 1.

Changes from the official loop that do not change the arithmetic (profiled, ~30% host overhead):
CCA scores kept on the GPU (the official code converts ~1e5 floats to a Python list every step
for logging); loss scalars synchronised only at log steps; fused AdamW; the batch is built on the
GPU from a GPU-resident token store (same content as ``CellDatasetForUFE`` + ``collate_fn_for_UFE``;
negatives drawn with torch's RNG). Additions: checkpoint/resume and a held-out gate with
early stopping (rule fixed in :class:`UFEConfig`).

Deviation (cross-species, stated): CCA negatives are drawn only from ``valid_ccre`` (e.g. cCREs
that lifted to the target genome) — unobservable cCREs are not "inaccessible".
"""
from __future__ import annotations

import json
import os
import time
import types
from dataclasses import dataclass, asdict
from typing import Callable, Optional, Sequence

import numpy as np

N_CCRE = 1355445
VOCAB = N_CCRE + 4
MAX_LEN = 8192


def load_epiagent(weights: str, device: str = "cuda"):
    """Pretrained EpiAgent (18 layers, d 512, 8 heads, flash-attn), strict weight loading."""
    import torch
    from epiagent.model import EpiAgent
    m = EpiAgent(vocab_size=VOCAB, num_layers=18, embedding_dim=512, num_attention_heads=8,
                 max_rank_embeddings=MAX_LEN, use_flash_attn=True,
                 pos_weight_for_RLM=torch.tensor(1.), pos_weight_for_CCA=torch.tensor(1.))
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:   # a finetune_ufe last.pt checkpoint
        sd = sd["model"]
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def embed_cells(model, sentences: Sequence, batch_size: int = 16, device: str = "cuda") -> np.ndarray:
    """CLS embeddings (as ``epiagent.inference.infer_cell_embeddings``, without its per-batch
    ``empty_cache``); sentences truncated to the first 8,190 tokens (``is_random=False``)."""
    import torch
    from torch.cuda.amp import autocast
    from epiagent.dataset import CellDataset, collate_fn
    was = model.training; model.eval()
    dl = torch.utils.data.DataLoader(CellDataset(list(sentences), max_length=MAX_LEN, is_random=False),
                                     batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    out = []
    with torch.no_grad():
        for b in dl:
            with autocast():
                out.append(model(b.to(device), return_transformer_output=True)["transformer_outputs"][:, 0, :].float().cpu())
    model.train(was)
    return torch.cat(out).numpy()


@dataclass
class UFEConfig:
    num_steps: int = 200_000
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup: int = 10_000
    batch_size: int = 5
    alpha_for_cca: float = 1.0
    log_steps: int = 500
    ckpt_every: int = 5_000
    gate_every: int = 20_000
    min_steps: int = 60_000      # early stop only after this many steps
    patience: int = 2            # ... when `patience` gates in a row fail to beat the earlier best
    min_gain: float = 0.005      # ... by at least this much
    seed: int = 0


def _cca_loss_tensor(self, cell_embeddings, ex_cell_ccre_ids, ex_cell_ccre_accessibility):
    """official ``EpiAgent.CCA_loss`` returning the scores as a detached GPU tensor"""
    import torch
    rep = torch.tensor([len(i) for i in ex_cell_ccre_ids], device=cell_embeddings.device)
    e = torch.repeat_interleave(cell_embeddings, rep, dim=0)
    ids = torch.cat(ex_cell_ccre_ids, dim=0).to(cell_embeddings.device)
    acc = ex_cell_ccre_accessibility.view(-1, 1).to(cell_embeddings.device)
    s = self.fc2_for_CCA(self.layer_norm_for_CCA(self.gelu(self.fc1_for_CCA(torch.cat((e, self.cCRE_embedding(ids)), -1)))))
    return self.criterion_CCA(s, acc), s.view(-1).detach()


class _GPUBatches:
    """GPU-resident token store and UFE batch builder (content of CellDatasetForUFE + collate)."""

    def __init__(self, sentences: Sequence, valid_ccre: Optional[np.ndarray], device, seed: int):
        import torch
        toks = [np.asarray(json.loads(s) if isinstance(s, str) else s, dtype=np.int32) for s in sentences]
        self.off = np.concatenate([[0], np.cumsum([len(t) for t in toks])])
        self.tok = torch.from_numpy(np.concatenate(toks)).to(device)
        self.valid = torch.zeros(N_CCRE, dtype=torch.bool, device=device)
        self.valid[torch.as_tensor(np.arange(N_CCRE) if valid_ccre is None else valid_ccre, device=device)] = True
        self.gen = torch.Generator(device=device); self.gen.manual_seed(seed)
        self.device, self.n = device, len(toks)

    def batch(self, idx, alpha: float):
        import torch
        dev = self.device
        signals = torch.zeros(len(idx), N_CCRE, device=dev)
        cells, ids, accs = [], [], []
        for k, i in enumerate(idx):
            c = self.tok[self.off[i]:self.off[i + 1]].long()
            signals[k, c - 4] = 1.0
            pool = self.valid.clone(); pool[c - 4] = False
            cand = torch.nonzero(pool).squeeze(1) + 4
            n = int(len(c) * alpha)
            neg = cand[torch.randperm(len(cand), device=dev, generator=self.gen)[:n]] if n <= len(cand) \
                else cand[torch.randint(len(cand), (n,), device=dev, generator=self.gen)]
            ids.append(torch.cat([c, neg]))
            accs.append(torch.cat([torch.ones(len(c), device=dev), torch.zeros(len(neg), device=dev)]))
            cells.append(c[:MAX_LEN - 2])
        x = torch.zeros(len(idx), max(len(c) for c in cells) + 2, dtype=torch.long, device=dev)
        for k, c in enumerate(cells):
            x[k, 0] = 1; x[k, 1:1 + len(c)] = c; x[k, 1 + len(c)] = 2
        return x, signals, ids, torch.cat(accs)


def finetune_ufe(model, sentences: Sequence, out_dir: str, *, valid_ccre: Optional[np.ndarray] = None,
                 gate: Optional[Callable] = None, config: UFEConfig = UFEConfig(), resume: bool = True,
                 device: str = "cuda", log: Callable = print):
    """Official UFE fine-tuning of ``model`` on ``sentences`` (training cells only).

    ``valid_ccre``: 0-based cCRE indices allowed as CCA negatives (``None`` = all, official).
    ``gate``: ``callable(model) -> float`` evaluated every ``gate_every`` steps (higher is better;
    e.g. held-out kNN cell-type transfer); drives ``best.pth`` and early stopping.
    Writes ``last.pt`` (resumable), ``best.pth``, ``final.pth``, ``train.log``, ``gate.tsv``.
    """
    import torch
    from torch.cuda.amp import autocast, GradScaler
    from torch.optim.lr_scheduler import LambdaLR
    c = config
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(c.seed); np.random.seed(c.seed)
    model.criterion_CCA.pos_weight = torch.tensor(1.)
    model.CCA_loss = types.MethodType(_cca_loss_tensor, model)
    data = _GPUBatches(sentences, valid_ccre, device, c.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay, fused=True)
    sched = LambdaLR(opt, lr_lambda=lambda s: min((s + 1) ** -0.5, (s + 1) * (c.warmup ** -1.5)))
    scaler = GradScaler()
    step, gates, ck = 0, [], os.path.join(out_dir, "last.pt")
    if resume and os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"]); step, gates = s["step"], s["gates"]
        data.gen.set_state(s["gen"]); torch.set_rng_state(s["torch_rng"])
        log(f"resumed at step {step}")

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "gates": gates, "gen": data.gen.get_state(),
                    "torch_rng": torch.get_rng_state(), "config": asdict(c)}, ck)

    def run_gate():
        nonlocal gates
        t0 = time.time(); g = float(gate(model)); gates.append((step, g))
        rec = dict(step=step, gate=g, s=round(time.time() - t0)); log(json.dumps(rec)); flog.write(json.dumps(rec) + "\n"); flog.flush()
        with open(os.path.join(out_dir, "gate.tsv"), "w") as fh:
            fh.write("step\tgate\n" + "".join(f"{a}\t{b}\n" for a, b in gates))
        if g >= max(x[1] for x in gates):
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
        recent = [x[1] for x in gates[-c.patience:]]
        prior = max(x[1] for x in gates[:-c.patience]) if len(gates) > c.patience else -np.inf
        return step >= c.min_steps and len(gates) > c.patience and max(recent) < prior + c.min_gain

    flog = open(os.path.join(out_dir, "train.log"), "a")
    if gate is not None and step == 0:
        run_gate()
    model.train()
    acc_cca = torch.zeros((), device=device); acc_sr = torch.zeros((), device=device)
    t0, done = time.time(), False
    while not done:
        perm = torch.randperm(data.n).numpy()
        for s0 in range(0, len(perm), c.batch_size):
            x, signals, ids, acc = data.batch(perm[s0:s0 + c.batch_size], c.alpha_for_cca)
            opt.zero_grad(set_to_none=True)
            with autocast():
                o = model(input_ids=x, return_transformer_output=True, calculate_rlm_loss=False, calculate_cca_loss=True,
                          calculate_sr_loss=True, ex_cell_ccre_ids=ids, ex_cell_ccre_accessibility=acc, signals=signals)
                loss = o["cca_loss"] + o["sr_loss"]
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            step += 1
            acc_cca += o["cca_loss"].detach(); acc_sr += o["sr_loss"].detach()
            if step % c.log_steps == 0:
                rec = dict(step=step, total=(acc_cca + acc_sr).item() / c.log_steps, cca=acc_cca.item() / c.log_steps,
                           sr=acc_sr.item() / c.log_steps, lr=sched.get_last_lr()[0],
                           s_per_step=round((time.time() - t0) / c.log_steps, 3))
                log(json.dumps(rec)); flog.write(json.dumps(rec) + "\n"); flog.flush()
                acc_cca.zero_(); acc_sr.zero_(); t0 = time.time()
            if step % c.ckpt_every == 0:
                save()
            if gate is not None and step % c.gate_every == 0:
                done = run_gate(); save(); t0 = time.time()
                if done:
                    log(f"early stop at step {step}")
            if step >= c.num_steps or done:
                done = True
                break
    torch.save(model.state_dict(), os.path.join(out_dir, "final.pth")); save()
    return model


def knn_gate(sentences: Sequence, labels: Sequence, train_idx: Sequence, test_idx: Sequence, k: int = 15,
             min_class: int = 20, device: str = "cuda") -> Callable:
    """Gate for :func:`finetune_ufe`: kNN(k) cell-type transfer from ``train_idx`` to ``test_idx``
    on L2-normalised CLS embeddings (classes with < ``min_class`` training cells dropped)."""
    from sklearn.neighbors import KNeighborsClassifier
    y = np.asarray(labels).astype(str); tr, te = np.asarray(train_idx), np.asarray(test_idx)
    vc = _value_counts(y[tr]); ok = set(c for c, n in vc.items() if n >= min_class)
    mtr, mte = np.isin(y[tr], list(ok)), np.isin(y[te], list(ok))

    def gate(model):
        E = embed_cells(model, [sentences[i] for i in np.concatenate([tr, te])], device=device)
        E /= np.linalg.norm(E, axis=1, keepdims=True); Etr, Ete = E[:len(tr)], E[len(tr):]
        return float(np.mean(KNeighborsClassifier(k).fit(Etr[mtr], y[tr][mtr]).predict(Ete[mte]) == y[te][mte]))
    return gate


def _value_counts(x):
    u, n = np.unique(x, return_counts=True)
    return dict(zip(u, n))
