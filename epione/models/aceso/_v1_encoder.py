"""ACESO v1 stage 1: panel-agnostic state encoder (component B).

scGPT (official code @ cebd6fa, whole-human checkpoint) is fine-tuned to map *any* targeted gene
panel of a cell to its chromatin state z (EpiAgent embedding; ``target``). Model construction, gene
tokens, per-cell binning and ``<cls>`` readout are those of ``align_scgpt.py`` /
``scgpt.tasks.cell_emb`` (official ``DataCollator``).

Panel-agnostic training
-----------------------
Every batch draws one panel (all cells of a dataset share a panel):

* with probability ``p_merfish``: the MERFISH heart panel (480 genes with 1:1 human ortholog in the
  vocabulary), counts thinned to the MERFISH E7.8-E8.75 depth distribution;
* otherwise a random panel: size ~ log-uniform[``min_genes``, ``max_genes``], genes uniform from the
  top ``pool`` highly variable genes (training samples; ``rna/gex_orth_genes.tsv``), counts thinned to
  MERFISH depth x (size / 480) x log-uniform[0.5, 2] (per-gene detection held fixed, depth scales with
  panel size, platform efficiency varies 4x).

Modes: ``panel_mode`` mixed (default, ACESO) | fixed (MERFISH panel only = "-B" ablation) | random
(never the MERFISH panel = zero-shot panel-generalisation test). The same sampling, written for any
reference and any platform, is :mod:`._panel`.

Loss, optimiser, split
----------------------
As ``align_scgpt.py``: (1-cos) + symmetric InfoNCE(tau 0.07); Adam 1e-4, StepLR 0.9/epoch, AMP, grad
clip 1; test samples E7.5_rep2/E8.5_rep2/E8.75_rep2 never used; 5% of the rest = validation (MERFISH
panel + 3 fixed random panels), early stopping patience 5.

Evaluation
----------
On the test samples for: the MERFISH panel, held-out random panels (sizes 100, 250, 500, 1000; fixed
seeds; drawn from the same pool), and the full ortholog transcriptome (upper bound of the RNA side).
Metrics: cosine to true z, kNN(15) cross-modal cell-type transfer (predicted test z against true
train z), kNN in predicted space.

Outputs
-------
In ``out``: ``best.pt``, ``z_pred_merfish_depth.npy`` (MERFISH panel, all reference cells; input to
stage 2), ``metrics.tsv``, ``train.log``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ._sites import register_site, require_sites, site_path

TEST = ["E7.5_rep2", "E8.5_rep2", "E8.75_rep2"]
# The target-platform h5ad, whose obs carries the depth distribution the counts are thinned to.
# There is no default: a wrong path here would silently train on another platform's depth.


@dataclass
class V1EncoderConfig:
    """Flags of ``aceso_stage1.py``, plus the paths the script hardcoded.

    ``out``: output directory; ``target``: .h5ad carrying the chromatin state z; ``target_key``: its
    ``obsm`` key; ``panel_mode``: mixed | fixed | random (module docstring); ``p_merfish``:
    probability of drawing the MERFISH panel in mixed mode; ``min_genes`` / ``max_genes``: log-uniform
    size range of a random panel; ``pool``: how many top highly variable genes random panels are drawn
    from; ``tau``: InfoNCE temperature; ``patience``: early-stopping patience in epochs;
    ``max_steps``: stop after this many optimiser steps (0 = no limit). ``root``, ``scgpt_model`` and
    ``scgpt_src`` override the sites of the same name (:mod:`._sites`); ``merfish_h5ad`` is the
    MERFISH h5ad whose ``obs`` holds the time and per-cell depth the thinning uses.
    """
    out: Optional[str] = None
    target: Optional[str] = None
    target_key: str = "X_epiagent_zeroshot"
    panel_mode: str = "mixed"
    p_merfish: float = 0.5
    min_genes: int = 100
    max_genes: int = 1000
    pool: int = 5000
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-4
    tau: float = 0.07
    patience: int = 5
    workers: int = 6
    max_steps: int = 0
    root: Optional[str] = None
    scgpt_model: Optional[str] = None
    scgpt_src: Optional[str] = None
    merfish_h5ad: Optional[str] = None


def train_v1_encoder(config: V1EncoderConfig = V1EncoderConfig(), *, log=print):
    """Fine-tune scGPT onto z and write ``best.pt`` / ``z_pred_merfish_depth.npy`` / ``metrics.tsv``."""
    c = config
    if not c.out or not c.target:
        raise ValueError("train_v1_encoder needs config.out (output directory) and config.target (the z .h5ad)")
    if c.root:
        os.environ["EPIONE_ROOT"] = c.root          # the project root the sites table's "{R}" defaults hang off
    if c.scgpt_model:
        register_site("scgpt_model", c.scgpt_model)
    if c.scgpt_src:
        register_site("scgpt_src", c.scgpt_src)
    register_site("merfish_h5ad", [c.merfish_h5ad] if c.merfish_h5ad else [], env="ACESO_MERFISH_H5AD")
    require_sites("root", "scgpt_model", "scgpt_src", "merfish_h5ad")

    import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
    import scipy.sparse as sp, anndata as ad
    sys.path.insert(0, site_path("scgpt_src"))
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.data_collator import DataCollator
    from scgpt.utils import load_pretrained
    from sklearn.neighbors import KNeighborsClassifier

    R = site_path("root") + "/gse205117"
    MD = site_path("scgpt_model")
    MER = site_path("merfish_h5ad")
    os.makedirs(c.out, exist_ok=True)
    torch.manual_seed(0); rng = np.random.default_rng(0)
    dev = torch.device("cuda")

    # ---- data
    cells = pd.read_csv(f"{R}/rna/cells.tsv.gz", sep="\t", index_col=0)
    G = sp.load_npz(f"{R}/rna/gex_orth_counts.npz").tocsr().astype(np.float32)
    genes = pd.read_csv(f"{R}/rna/gex_orth_genes.tsv", sep="\t")
    gene_ids = genes["vocab_id"].to_numpy(np.int64)
    MER_PANEL = np.flatnonzero(genes["in_merfish_panel"].to_numpy())
    POOL = np.flatnonzero(genes["hvg_rank"].to_numpy() <= c.pool)
    vocab = GeneVocab.from_file(f"{MD}/vocab.json")
    for s in ["<pad>", "<cls>", "<eoc>"]:
        if s not in vocab: vocab.append_token(s)
    assert all(vocab[h] == v for h, v in zip(genes["human"], gene_ids))
    cfg = json.load(open(f"{MD}/args.json"))
    t = ad.read_h5ad(c.target)
    Z = np.asarray(t[cells.index].obsm[c.target_key], dtype=np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    mo = ad.read_h5ad(MER, backed="r").obs[["time", "nCounts"]]
    DEPTH = mo.nCounts.to_numpy()[(mo.time <= 8.75).to_numpy()].astype(np.float64)
    log(f"cells {G.shape[0]}, ortholog genes {G.shape[1]}, MERFISH panel {len(MER_PANEL)}, pool {len(POOL)}, "
        f"z dim {Z.shape[1]}, mode {c.panel_mode}")

    is_test = cells["sample"].isin(TEST).to_numpy()
    tr_all = np.flatnonzero(~is_test)
    val = rng.choice(tr_all, int(0.05 * len(tr_all)), replace=False)
    train = np.setdiff1d(tr_all, val); test = np.flatnonzero(is_test)

    def draw_panel(g):
        """one panel for a batch -> (gene columns, depth scale relative to the MERFISH panel)"""
        if c.panel_mode == "fixed" or (c.panel_mode == "mixed" and g.random() < c.p_merfish):
            return MER_PANEL, 1.0
        n = int(np.exp(g.uniform(np.log(c.min_genes), np.log(c.max_genes))))
        return (np.sort(g.choice(POOL, n, replace=False)),
                n / len(MER_PANEL) * float(np.exp(g.uniform(np.log(.5), np.log(2)))))

    def panel_counts(rows, cols, scale, g):
        """counts of `rows` on panel `cols`, binomially thinned to MERFISH-like depth x scale"""
        M = G[rows][:, cols].tocsr(); tot = np.asarray(M.sum(1)).ravel()
        p = np.minimum(1.0, g.choice(DEPTH, len(rows)) * scale / np.maximum(tot, 1))
        M.data = g.binomial(M.data.astype(np.int64), np.repeat(p, np.diff(M.indptr))).astype(np.float32)
        return M

    coll = DataCollator(do_padding=True, pad_token_id=vocab[cfg["pad_token"]], pad_value=cfg["pad_value"], do_mlm=False,
                        do_binning=True, max_length=1200, sampling=True, keep_first_n_tokens=1)

    def tokenize(M, cols):
        ex = []
        for i in range(M.shape[0]):
            s, e = M.indptr[i], M.indptr[i + 1]
            nz, v = M.indices[s:e], M.data[s:e]; k = v > 0
            ex.append({"genes": torch.from_numpy(np.insert(gene_ids[cols][nz[k]], 0, vocab["<cls>"])).long(),
                       "expressions": torch.from_numpy(np.insert(v[k], 0, cfg["pad_value"])).float()})
        return coll(ex)

    class TrainCollate:
        """draws one panel per batch inside the worker (independent RNG per worker and epoch)"""
        def __init__(self, seed): self.seed = seed; self.g = None

        def __call__(self, idx):
            if self.g is None:
                w = torch.utils.data.get_worker_info()
                self.g = np.random.default_rng([self.seed, w.id if w else 0])
            idx = np.asarray(idx); cols, scale = draw_panel(self.g)
            M = panel_counts(idx, cols, scale, self.g); M.eliminate_zeros()
            keep = np.diff(M.indptr) > 0                   # cells with no detected panel gene carry no input
            out = tokenize(M[keep], cols); out["id"] = torch.from_numpy(idx[keep]); return out

    def fixed_loader(rows, cols, scale, seed):
        """deterministic evaluation batches for a given panel (cells with no detected gene skipped)"""
        M = panel_counts(rows, cols, scale, np.random.default_rng(seed)); M.eliminate_zeros()
        ok = np.diff(M.indptr) > 0; M, rows = M[ok], rows[ok]
        for s in range(0, len(rows), 256):
            out = tokenize(M[s:s + 256], cols); out["id"] = torch.from_numpy(rows[s:s + 256]); yield out

    # ---- model (as scgpt.tasks.cell_emb.embed_data)
    model = TransformerModel(ntoken=len(vocab), d_model=cfg["embsize"], nhead=cfg["nheads"], d_hid=cfg["d_hid"],
                             nlayers=cfg["nlayers"], nlayers_cls=cfg["n_layers_cls"], n_cls=1, vocab=vocab,
                             dropout=cfg["dropout"], pad_token=cfg["pad_token"], pad_value=cfg["pad_value"], do_mvc=True,
                             do_dab=False, use_batch_labels=False, domain_spec_batchnorm=False, explicit_zero_prob=False,
                             use_fast_transformer=False, pre_norm=False)
    load_pretrained(model, torch.load(f"{MD}/best_model.pt", map_location="cpu"), verbose=False)
    head = nn.Linear(cfg["embsize"], Z.shape[1])
    model.to(dev); head.to(dev)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=c.lr, eps=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler()
    Zt = torch.from_numpy(Z)

    def embed(b):
        g = b["gene"].to(dev)
        h = model._encode(g, b["expr"].to(dev), src_key_padding_mask=g.eq(vocab[cfg["pad_token"]]))[:, 0, :]
        return F.normalize(head(h).float(), dim=1)

    def loss_fn(p, z):
        logits = p @ z.T / c.tau; lab = torch.arange(len(p), device=dev)
        return (1 - (p * z).sum(1)).mean() + 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))

    @torch.no_grad()
    def predict(batches, rows):
        """predictions aligned to `rows` (global cell indices); NaN where the cell had no input"""
        model.eval(); head.eval(); out = np.full((len(rows), Z.shape[1]), np.nan, np.float32)
        pos = {int(r): i for i, r in enumerate(rows)}
        for b in batches:
            with torch.autocast("cuda", dtype=torch.float16):
                p = embed(b)
            out[[pos[int(i)] for i in b["id"]]] = p.cpu().numpy()
        return out

    vg = np.random.default_rng(7)
    VAL_PANELS = [("merfish", MER_PANEL, 1.0)] + [
        (f"random{i}", *draw_panel_fixed) for i, draw_panel_fixed in enumerate(
            [(np.sort(vg.choice(POOL, n, replace=False)), n / len(MER_PANEL)) for n in (150, 400, 800)])]
    if c.panel_mode == "fixed": VAL_PANELS = VAL_PANELS[:1]
    if c.panel_mode == "random": VAL_PANELS = VAL_PANELS[1:]

    best, bad, step, flog = np.inf, 0, 0, open(f"{c.out}/train.log", "a")
    for ep in range(c.epochs):
        model.train(); head.train(); t0 = time.time(); tl = []
        dl = torch.utils.data.DataLoader(train, batch_size=c.batch_size, shuffle=True, drop_last=True,
                                         num_workers=c.workers, collate_fn=TrainCollate(100 + ep))
        for b in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                p = embed(b)
            loss = loss_fn(p, Zt[b["id"]].to(dev))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update()
            tl.append(loss.item()); step += 1
            if c.max_steps and step >= c.max_steps: break
        sched.step()
        vres = {}
        for name, cols, scale in VAL_PANELS:
            pv = predict(fixed_loader(val, cols, scale, 1), val); m = ~np.isnan(pv[:, 0])
            vres[name] = (loss_fn(torch.from_numpy(pv[m]).to(dev), Zt[val[m]].to(dev)).item(),
                          float((pv[m] * Z[val[m]]).sum(1).mean()))
        vl = float(np.mean([v[0] for v in vres.values()]))
        msg = json.dumps(dict(epoch=ep, step=step, train_loss=float(np.mean(tl)), val_loss=vl,
                              val_cos={k: round(v[1], 4) for k, v in vres.items()}, s=round(time.time() - t0, 1),
                              gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1)))
        log(msg); flog.write(msg + "\n"); flog.flush()
        if vl < best:
            best, bad = vl, 0
            torch.save({"model": model.state_dict(), "head": head.state_dict(), "args": asdict(c)}, f"{c.out}/best.pt")
        else:
            bad += 1
            if bad >= c.patience: break
        if c.max_steps and step >= c.max_steps: break

    # ---- evaluation with the best checkpoint
    ck = torch.load(f"{c.out}/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); head.load_state_dict(ck["head"])
    allc = np.arange(G.shape[0])
    P_mer = predict(fixed_loader(allc, MER_PANEL, 1.0, 0), allc)
    np.save(f"{c.out}/z_pred_merfish_depth.npy", P_mer)
    y = cells["celltype"].astype(str).to_numpy()
    vc = pd.Series(y[tr_all]).value_counts(); ok = np.isin(y, vc.index[vc >= 20])
    te, trn = test[ok[test]], tr_all[ok[tr_all]]
    knn_true = KNeighborsClassifier(15).fit(Z[trn], y[trn])
    eg = np.random.default_rng(2024)
    EVAL = [("merfish_panel", MER_PANEL, 1.0)]
    for n in (100, 250, 500, 1000):
        for r in range(3):
            cols = np.sort(eg.choice(POOL, n, replace=False)); EVAL.append((f"random_{n}_{r}", cols, n / len(MER_PANEL)))
    EVAL.append(("full_transcriptome_unthinned", np.arange(G.shape[1]), None))
    rows = []
    for name, cols, scale in EVAL:
        idx = np.concatenate([trn, te])
        if scale is None:   # full depth, top-1199 expressed genes kept by the collator's sampling
            M = G[idx]
            P = predict((dict(**tokenize(M[s:s + 256], cols), id=torch.from_numpy(idx[s:s + 256]))
                         for s in range(0, len(idx), 256)), idx)
        else:
            P = predict(fixed_loader(idx, cols, scale, 3), idx)
        Ptr, Pte = P[:len(trn)], P[len(trn):]
        mtr, mte = ~np.isnan(Ptr[:, 0]), ~np.isnan(Pte[:, 0])
        rows.append(dict(panel=name, n_genes=len(cols), overlap_merfish=int(np.isin(cols, MER_PANEL).sum()),
                         frac_test_cells_with_input=float(mte.mean()),
                         test_cos=float((Pte[mte] * Z[te][mte]).sum(1).mean()),
                         knn_cross_modal_acc=float(np.mean(knn_true.predict(Pte[mte]) == y[te][mte])),
                         knn_pred_space_acc=float(np.mean(KNeighborsClassifier(15).fit(Ptr[mtr], y[trn][mtr])
                                                          .predict(Pte[mte]) == y[te][mte]))))
        log(rows[-1])
    rows.append(dict(panel="true_epiagent_z (upper bound)", n_genes=0, overlap_merfish=0,
                     frac_test_cells_with_input=1.0, test_cos=1.0,
                     knn_cross_modal_acc=float(np.mean(knn_true.predict(Z[te]) == y[te])),
                     knn_pred_space_acc=float("nan")))
    df = pd.DataFrame(rows); df.to_csv(f"{c.out}/metrics.tsv", sep="\t", index=False)
    summ = df.assign(group=df.panel.str.replace(r"_\d+$", "", regex=True)).groupby("group", sort=False)[
        ["test_cos", "knn_cross_modal_acc", "knn_pred_space_acc"]].mean()
    log(summ.round(3).to_string())
    return df


def main(argv=None):
    """The original script's command line: same flags, same types, same defaults."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--target-key", default="X_epiagent_zeroshot")
    ap.add_argument("--panel-mode", choices=["mixed", "fixed", "random"], default="mixed")
    ap.add_argument("--p-merfish", type=float, default=0.5)
    ap.add_argument("--min-genes", type=int, default=100)
    ap.add_argument("--max-genes", type=int, default=1000)
    ap.add_argument("--pool", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args(argv)
    return train_v1_encoder(V1EncoderConfig(**vars(a)))


if __name__ == "__main__":  # pragma: no cover
    main()
