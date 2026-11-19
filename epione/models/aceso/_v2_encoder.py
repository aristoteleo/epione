"""ACESO v2 stage 1 -- scGPT -> cell state z, trained on arbitrary gene subsets.

This is the encoder the foundation-scale model is built on. It reads ``data/train_v2/`` and nothing
else, so the trainer has no knowledge of HDMA, ENCODE or GEO layouts -- the store is the contract.

The store is the contract
-------------------------
``cells.tsv``
    the row order for everything, with ``split`` by SAMPLE;
``rna.h5ad``
    cells x scGPT-vocab genes, int32 counts, symbol-keyed;
``z_epiagent.npy``
    cells x 512, NaN rows for cells with no z (masked, never dropped).

:mod:`._store` is what writes that layout, and the names and column meanings above are the whole
interface between the corpus builders and this trainer.

What changed from v1
--------------------
v1 is :mod:`._v1_encoder` (``gse205117/scgpt/aceso_stage1.py``).

**Changed: the input is an arbitrary gene subset, and no vendor panel is trained on.** v1 locked 50%
of batches to the MERFISH 480 panel and drew the rest uniformly from the top-5,000 HVGs. Both are
wrong at this scale: the 480 is organ-specific, so weighting it writes a cardiac prior; and a library
of real panels is a *narrower* sample of subset space than uniform random, not a broader one. So
subsets are 50% uniform / 50% structured, and the vendor panels move entirely to held-out evaluation,
which is what makes "any gene set" a testable claim.

**Changed: two views per cell, with the wide view as a stop-grad teacher.** A subset-only objective
can only teach invariance to missing genes, never *selectivity* about which genes carry the state --
and invariance is the wrong thing to learn where information is genuinely absent. This is the
measured bottleneck: E1-E3 found the thinned input itself carries chamber 0.6635 and an RNA-fitted
scVI latent keeps 0.6787, while v1's z kept only 0.3144-0.3670, with the retention heads already at
the ceiling. E2's subspace distillation moved z's centred effective rank 8.57 -> 18.00 at no
panel-agnosticism cost, which is why the teacher term is here.

What deliberately did not change
--------------------------------
**``max_len`` 1200 and random truncation.** Both are the checkpoint's own pretraining settings
(``args.json``: ``max_seq_len 1200``, ``trunc_by_sample True``, ``n_hvg None``), so scGPT's
pretraining task already *is* "infer cell state from a random subset" -- which is why
panel-agnostic training worked in v1 at all. Raising the length or making truncation deterministic
would both be departures from the pretrained regime, and go in as pre-registered ablations if at
all. The wide view is therefore **not** "the full transcriptome": median detected genes are
1,484-2,458, so it is a wide *random* view. It is named ``z_wide`` for that reason.

**The loss form.** ``(1 - cos) + symmetric InfoNCE(tau 0.07)``, Adam 1e-4, StepLR 0.9/epoch, AMP,
grad clip 1 -- v1's, so the two generations stay comparable::

    L = L(z_sub, z_target) + L(z_wide, z_target) + lambda * L(z_sub, sg[z_wide])

``lambda`` is zeroed per cell where no wide view exists beyond the subset itself (the GSE205117
MERFISH cells, whose 480 genes are the whole measurement).

Running it
----------
::

    python -m epione.models.aceso._v2_encoder --out runs/s1_v2 --store data/train_v2
    torchrun --standalone --nproc_per_node=4 -m epione.models.aceso._v2_encoder --out runs/s1_v2

or, as a function, ``train_v2_encoder(V2EncoderConfig(out="runs/s1_v2"))``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class V2EncoderConfig:
    """Every flag of the stage-1 v2 trainer, with the script's own defaults.

    The paths (``out``, ``store``, ``resume``) default to ``None`` and are resolved inside
    :func:`train_v2_encoder` against the project root from :mod:`._sites`; a relative path is taken
    relative to that root, an absolute one is used as given. ``store`` ``None`` means
    ``<root>/data/train_v2``, the CLI's default.

    The non-obvious ones:

    ``max_len``
        the checkpoint's pretraining length (``args.json``: ``max_seq_len 1200``,
        ``trunc_by_sample True``). It bounds the *wide* view too, so that view is a wide random
        view of the transcriptome rather than all of it, and it is kept at the pretrained value on
        purpose -- see the module docstring.
    ``lam``
        the teacher weight: how much of the loss is the subset view chasing the stop-grad wide view,
        ``lambda * L(z_sub, sg[z_wide])``. 1.0 weights the teacher term as heavily as either of the
        two terms that align a view to the EpiAgent z.
    ``min_genes``, ``p_structured``
        the subset sampler. Sizes are log-uniform on ``[min_genes, max_len]``; with probability
        ``1 - p_structured`` the genes are drawn uniformly over the in-vocabulary pool, otherwise
        the subset is grown from 1-19 seeds in the gene-index neighbourhood, so it is concentrated
        on a few programmes and often mismatched to the cells it is read on -- the realistic
        deployment case. No vendor panel is in this mixture; the panels are held-out evaluation.
    ``organ_alpha``, ``organ_cap``
        organ weighting, ``p_organ ~ n_organ^alpha`` with a ceiling of ``cap`` times uniform on any
        stratum's per-cell weight. The corpus is 59% heart, so uniform cell sampling specialises the
        encoder to one organ; see :func:`organ_weights` for the measured strata.
    ``tau``
        InfoNCE temperature, v1's 0.07.
    ``no_ram``
        read the backed h5ad instead of loading ``csr/`` into RAM.
    """

    out: Optional[str] = None
    store: Optional[str] = None
    max_len: int = 1200
    z_dim: int = 512
    batch: int = 32
    lr: float = 1e-4
    lam: float = 1.0
    tau: float = 0.07
    min_genes: int = 50
    p_structured: float = 0.5
    steps: int = 200000
    ckpt_every: int = 500
    patience: int = 30
    warmup: int = 0
    val_batches: int = 20
    resume: Optional[str] = None
    time_budget: Optional[float] = None
    workers: int = 6
    organ_alpha: float = 0.5
    organ_cap: float = 10.0
    no_ram: bool = False
    seed: int = 0


def organ_weights(O, idx, alpha, cap, *, log=print):
    """Per-cell sampling weights that stop the encoder from becoming a heart model.

    THE CORPUS IS 59% HEART. Measured over the 150 tokenised samples, 2,025,634 cells: ENCODE adult
    heart alone is 1,074,434 cells (53.0%) in 64 samples, and with HDMA fetal heart it is 1,194,688
    (59%) in 72 of 150 samples. Twenty-two organ strata would be 4.5% each if balanced. Sampling
    cells uniformly therefore spends 59% of stage 1's gradient on one organ, and stage 1 is where
    scGPT is fine-tuned -- so uniform sampling does not merely under-train the other tissues, it
    specialises the encoder to heart. That is the opposite of the corpus's reason for existing.

    THE FIX IS THE SAMPLER, NOT THE DATASET. Every heart sample stays in the corpus: stage 2 and the
    heart figures want exactly that depth, and discarding data to fix a sampling problem would throw
    away the paper's subject. `p_organ ~ n_organ^alpha` is the same temperature used for language
    sampling in multilingual pretraining, which is the identical problem.

    ADULT AND FETAL HEART ARE SEPARATE STRATA. ENCODE heart is adult and HDMA Heart is fetal; the
    congenital question lives on the developmental side, so pooling them would let 1.07 M adult cells
    swamp the 120 k fetal ones inside a single "heart" stratum.

    WHY A CAP ON TOP OF alpha. At alpha 0.5 the 612-cell HDMA:Adr stratum would be revisited 50 times
    over a 200,000-step run, which fits 612 cells rather than learning from them. The cap bounds any
    stratum's per-cell weight at `cap` times uniform; at alpha 0.5, cap 10 the worst case is 32
    passes and heart drops 59% -> 27%, with 19 of 22 strata between 1.3% and 7%.
    """
    key = (O["corpus"].astype(str) + ":" + O["organ"].astype(str).str.lower()).to_numpy()[idx]
    uniq, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv).astype(float)
    p = n ** alpha
    p /= p.sum()
    tot = n.sum()
    over = (p / n) / (1.0 / tot) > cap            # per-cell weight relative to uniform
    if over.any():
        p[over] = cap * n[over] / tot
        p[~over] *= (1.0 - p[over].sum()) / p[~over].sum()
    # Returned as STRATA rather than a per-cell weight vector. `rng.choice(p=w)` on a 2.3e6 pool
    # costs 34.2 ms per step measured, against ~0 ms unweighted, and it is called in every worker on
    # every step. The weights are constant inside a stratum, so drawing the per-stratum counts from
    # a multinomial and then sampling uniformly within each stratum is O(batch) instead of O(cells)
    # -- and it is also the better estimator: the stratum proportions then hold exactly in every
    # step, not merely in expectation.
    strata = [np.asarray(idx)[inv == i] for i in range(len(uniq))]
    order = np.argsort(-p)
    log(f"  organ sampling: alpha {alpha}, cap {cap}x uniform, {len(uniq)} strata")
    for i in order[:6]:
        log(f"    {uniq[i]:24s} {n[i]:>9,.0f} cells  {p[i]*100:5.2f}% of draws "
            f"({p[i] / (n[i] / tot):5.2f}x uniform)")
    if len(order) > 6:
        log(f"    ... {len(order)-6} more strata, smallest share {p[order[-1]]*100:.2f}%")
    heart = p[[i for i, u in enumerate(uniq) if "heart" in u]].sum()
    log(f"    heart share: {heart*100:.1f}% (uniform over cells would be "
        f"{n[[i for i, u in enumerate(uniq) if 'heart' in u]].sum()/tot*100:.1f}%)")
    return strata, p


def train_v2_encoder(config: V2EncoderConfig = V2EncoderConfig(), *, log=None):
    """Train stage 1 v2. ``log`` overrides the sink of the trainer's own rank-0 ``log()`` helper.

    Writes ``last.pt``, ``best.pt``, ``ckpt_step*.pt``, ``heartbeat.json`` and ``tb/`` into ``out``;
    the checkpoint keys are the ones the v2 stage-2 trainer reads (``model``, ``head``, ``opt``,
    ``scaler``, ``sched``, ``step``, ``best``, ``val``, ``train``, ``args``).
    """
    import pandas as pd
    import scipy.sparse as sp
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from ._sites import require_sites, site_path

    c = config
    _sink = log if log is not None else (lambda *x: print(*x, flush=True))

    # The anchor every other path hangs off, resolved the same way `_sites` resolves the rest: the
    # first candidate that EXISTS -- `$EPIONE_ROOT`, then `$EPIONE_ACESO_SITES`, then the cwd, which
    # is where the two per-site roots the script used to inline now live.
    R = Path(site_path("root"))
    # Resolved by existence rather than by a per-site default: on Marlowe scGPT lives inside the
    # project tree, on Sherlock it lives in 25_scllm. `require_sites` prints what it picked, so the
    # run log records the paths it actually used rather than the ones it hoped for.
    MD = Path(site_path("scgpt_model"))

    if not c.out:
        raise ValueError("`out` is required: the run directory for the checkpoints and the heartbeat")
    _store = c.store if c.store else "data/train_v2"
    OUT = Path(c.out) if Path(c.out).is_absolute() else R / c.out
    STORE = Path(_store) if Path(_store).is_absolute() else R / _store
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- distributed
    LOCAL = int(os.environ.get("LOCAL_RANK", 0))
    RANK = int(os.environ.get("RANK", 0))
    WORLD = int(os.environ.get("WORLD_SIZE", 1))
    DDP = WORLD > 1
    if DDP:
        # A LONGER WATCHDOG THAN THE 10-MINUTE DEFAULT. On 2026-11-12 all four ranks aborted together on
        # collective SeqNum=7 with nothing logged after "training:": every rank was still inside its first
        # DataLoader batch, so none reached the collective and the watchdog took the job down. A single
        # 32-row read costs 0.17 s on its own; what is slow is 24 worker processes opening the same 8 GB
        # h5ad at once. The timeout is a safety net against a hung rank, not a throughput budget, so it
        # should not be shorter than a cold first batch.
        from datetime import timedelta
        torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=90))
        torch.cuda.set_device(LOCAL)
    dev = torch.device(f"cuda:{LOCAL}" if torch.cuda.is_available() else "cpu")
    # Different subset draws per rank: with the same seed every rank would see the same gene set every
    # step, which turns a 4-GPU run into a 4x larger batch of ONE subset instead of four.
    rng = np.random.default_rng(c.seed + 1000 * RANK)
    torch.manual_seed(c.seed)

    def log(*x):
        if RANK == 0:
            _sink(*x)

    # Every site-dependent path is resolved and printed BEFORE the allocation is used. A wrong path then
    # costs seconds instead of the 51 minutes four idle H100s cost on 2026-11-12, when a hard-coded
    # Sherlock path failed per sample inside a driver that kept reporting progress.
    if RANK == 0:
        require_sites("root", "scgpt_model", "scgpt_src")
    else:
        site_path("scgpt_model"), site_path("scgpt_src")

    # ---------------------------------------------------------------- store
    log(f"store {STORE}")
    O = pd.read_csv(STORE / "cells.tsv", sep="\t")
    import anndata as ad
    A = ad.read_h5ad(STORE / "rna.h5ad", backed="r")
    genes = A.var_names.astype(str).to_numpy()
    zp = STORE / "z_epiagent.npy"
    if not zp.exists():
        sys.exit(f"missing {zp}. Run data/epiagent_z.py --embed --assemble first; the z target is the "
                 f"whole objective and there is nothing to train against without it.")
    Z = np.load(zp)
    assert len(Z) == len(O), (Z.shape, len(O))

    # CONTENT, NOT JUST SHAPE. On 2026-11-12 this trainer ran 1.5 h on a store whose CSR `indptr` had been
    # cast to int32: the corpus holds 3,409,034,954 nonzeros, int32 wraps at 2,147,483,647, and h5py reads
    # a negative start as an offset from the END of the array -- so every row past 1,241,158 returned a
    # real-looking count vector belonging to a DIFFERENT cell, 37.1% of the training split paired with the
    # wrong RNA. Every check in place at the time passed, because they all checked shapes. `n_umi` was
    # computed by the store builder from the correct matrix, so comparing it against the row sums the
    # trainer actually reads is a direct test of whether the bytes line up with the labels.
    # (the row-content check now runs on the in-RAM CSR below, which is faster and does
    #  not reopen the h5ad)

    # THE SPLIT IS BY SAMPLE, taken from splits.json rather than from any per-cell column, so a validation
    # cell shares no donor, no library and no batch with a training cell. A cell without a finite z is out
    # of both: the objective is L(z_sub, z_target) and there is nothing to align to.
    SPL = json.load(open(STORE / "splits.json"))
    fin = np.isfinite(Z[:, 0])
    _samp = O["sample"].astype(str).to_numpy()
    tr_idx = np.where(fin & np.isin(_samp, SPL["train"]))[0]
    va_idx = np.where(fin & np.isin(_samp, SPL["val"]))[0]
    log(f"  {len(O):,} cells, {len(genes):,} genes, {int(fin.sum()):,} with z ({fin.mean()*100:.1f}%)")
    log(f"  train {len(tr_idx):,} cells / {len(SPL['train'])} samples   val {len(va_idx):,} / "
        f"{len(SPL['val'])}   test held out: {sorted(SPL['test'])}")
    if len(tr_idx) < 20_000 or len(va_idx) < 1_000:
        sys.exit(f"train {len(tr_idx):,} / val {len(va_idx):,} cells is not a usable split")

    tr_strata = None
    if c.organ_alpha < 1.0 and {"corpus", "organ"} <= set(O.columns):
        tr_strata = organ_weights(O, tr_idx, c.organ_alpha, c.organ_cap, log=log)
    elif c.organ_alpha < 1.0:
        log("  !! cells.tsv has no corpus/organ column; falling back to uniform cell sampling")

    # ---------------------------------------------------------------- the training read path
    # aceso/TRAIN_IO_DESIGN.md: raw CSR read into RAM, with a contract checked BEFORE the allocation is
    # used. Four of the five checks are shape; the fifth is the only one that would have caught the
    # int32 `indptr` wrap that paired 37.1% of this corpus with the wrong RNA.
    CSR = None
    _csr = STORE / "csr"
    if (_csr / "indptr.npy").exists() and not c.no_ram:
        import json as _json
        _m = _json.load(open(_csr / "meta.json"))
        _h5 = STORE / "rna.h5ad"
        _st = _h5.stat()
        if _m.get("src_mtime") and abs(_m["src_mtime"] - _st.st_mtime) > 1:
            sys.exit(f"REFUSING: {_csr} was built from an rna.h5ad modified at {_m['src_mtime']} but the "
                     f"file on disk is {_st.st_mtime}. Rebuild with data/h5ad_to_memmap.py rather than "
                     f"train on a stale copy of the corpus.")
        _t = time.time()
        _ip = np.load(_csr / "indptr.npy")
        if _ip.dtype != np.int64:
            sys.exit(f"REFUSING: indptr is {_ip.dtype}, must be int64 -- int32 wraps past 2.1e9 nonzeros")
        if (np.diff(_ip) < 0).any():
            sys.exit("REFUSING: indptr is not monotonic; it has wrapped")
        if len(_ip) - 1 != len(O):
            sys.exit(f"REFUSING: indptr implies {len(_ip)-1:,} rows, cells.tsv has {len(O):,}")
        _dat = np.load(_csr / "data.npy")
        _ind = np.load(_csr / "indices.npy")
        if not (int(_ip[-1]) == len(_dat) == len(_ind)):
            sys.exit(f"REFUSING: indptr[-1]={int(_ip[-1]):,}, data={len(_dat):,}, indices={len(_ind):,}")
        CSR = (_dat, _ind, _ip)
        log(f"  CSR in RAM: {(_dat.nbytes + _ind.nbytes + _ip.nbytes)/1e9:.1f} GB, "
            f"{len(_dat):,} nonzeros  [{time.time()-_t:.0f}s]")
        if RANK == 0 and "n_umi" in O.columns:
            _r = np.sort(np.random.default_rng(0).choice(len(O), 40, replace=False))
            _got = np.array([float(_dat[_ip[i]:_ip[i+1]].sum()) for i in _r])
            _exp = O["n_umi"].to_numpy()[_r].astype(float)
            _ok = int(np.isclose(_got, _exp, rtol=1e-3).sum())
            log(f"  row-content check: {_ok}/40 row sums match cells.tsv n_umi")
            if _ok < 40:
                sys.exit(f"REFUSING: {40-_ok}/40 rows disagree with n_umi -- the bytes do not line up "
                         f"with the labels")

    # Counts stay BACKED on disk rather than materialised per rank. At Heart-only scale (2.378e8 nnz,
    # ~1.9 GB as int32/float32) four in-memory copies would fit; at the full 2.45 M cells it is ~4.8e9
    # nnz and they would not. anndata's backed CSR supports the row slicing the sampler needs, and the
    # page cache is shared across ranks on one node, so the first epoch warms it once for all four.
    #
    # THE PARENT'S HANDLE MUST BE CLOSED BEFORE THE DATALOADER FORKS. HDF5 is not fork-safe: a child that
    # inherits an open handle and then reads deadlocks, with no error and no traceback — measured here as a
    # run that printed "training:" and then sat forever at `workers=2`, while the identical run at
    # `workers=0` completed. Each worker opens its own handle in `StepDataset._x()`; the parent needs only
    # the shape and the gene names, both already read.
    _shape, _dtype = A.shape, A.X.dtype
    log(f"  counts backed on disk: {_shape}, {_dtype}")
    A.file.close()
    del A

    # ---------------------------------------------------------------- scGPT
    sys.path.insert(0, site_path("scgpt_src"))
    from scgpt.model import TransformerModel                                        # noqa: E402
    from scgpt.tokenizer import GeneVocab                                           # noqa: E402
    from scgpt.data_collator import DataCollator                                     # noqa: E402
    from scgpt.utils import load_pretrained                                          # noqa: E402

    cfg = json.load(open(MD / "args.json"))
    # `GeneVocab.from_file` costs 129 s measured, every process and every chain-link restart. The object
    # is deterministic given the json, so it is built once and pickled next to the store. Cache miss falls
    # back to building it, so a stale or absent cache is never a failure.
    _vc = STORE / f".genevocab_{Path(MD).name}.pkl"
    import pickle
    vocab = None
    if _vc.exists():
        try:
            vocab = pickle.loads(_vc.read_bytes())
            log(f"  vocab from cache {_vc.name}")
        except Exception as e:
            log(f"  vocab cache unreadable ({e}), rebuilding")
    if vocab is None:
        t0 = time.time()
        vocab = GeneVocab.from_file(str(MD / "vocab.json"))
        for s in ["<pad>", "<cls>", "<eoc>"]:
            if s not in vocab:
                vocab.append_token(s)
        log(f"  vocab built in {time.time()-t0:.0f}s")
        if RANK == 0:
            try:
                _vc.write_bytes(pickle.dumps(vocab))
            except Exception as e:
                log(f"  could not cache vocab: {e}")
    for s in ["<pad>", "<cls>", "<eoc>"]:
        if s not in vocab:
            vocab.append_token(s)
    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes], np.int64)
    in_vocab = np.array([g in vocab for g in genes])
    log(f"  {in_vocab.sum():,}/{len(genes):,} store genes map into scGPT's vocabulary")

    coll = DataCollator(do_padding=True, pad_token_id=vocab[cfg["pad_token"]], pad_value=cfg["pad_value"],
                        do_mlm=False, do_binning=True, max_length=c.max_len, sampling=True,
                        keep_first_n_tokens=1)
    model = TransformerModel(ntoken=len(vocab), d_model=cfg["embsize"], nhead=cfg["nheads"],
                             d_hid=cfg["d_hid"], nlayers=cfg["nlayers"], nlayers_cls=cfg["n_layers_cls"],
                             n_cls=1, vocab=vocab, dropout=cfg["dropout"], pad_token=cfg["pad_token"],
                             # do_mvc=False, unlike v1: the MVC decoder is never used by `_encode`, and
                             # leaving it on forces DDP into `find_unused_parameters=True`, which walks
                             # the whole graph on every backward pass for nothing.
                             pad_value=cfg["pad_value"], do_mvc=False, do_dab=False,
                             use_batch_labels=False, domain_spec_batchnorm=False,
                             explicit_zero_prob=False, use_fast_transformer=False, pre_norm=False)
    load_pretrained(model, torch.load(MD / "best_model.pt", map_location="cpu"), verbose=False)
    head = nn.Linear(cfg["embsize"], c.z_dim)
    model.to(dev); head.to(dev)

    # Nested, not module-level, only because the module must import with no torch installed: the base
    # class is `nn.Module`, which exists only once torch has been imported inside this function.
    class TwoView(nn.Module):
        """Encoder and read-out head as ONE module, because DDP counts forward passes, not tensors.

        The first version wrapped the two separately and then ran the encoder as `model.module._encode`.
        That is two faults at once. Reaching through `.module` bypasses DDP's forward hook, so the
        encoder's gradients were never all-reduced and the four ranks would have drifted into four
        different encoders while the loss curve looked normal. And the head, which *was* wrapped, was
        called twice per backward -- once per view -- which re-arms its reducer on the second forward
        while the first view's autograd graph is still outstanding, so the all-reduce belonging to that
        graph never fires. Every rank then sits at 100% GPU inside NCCL's busy-wait for a collective no
        rank will reach. That is the stall that ate every Marlowe run today.

        One module, one forward per step, both views inside it.
        """

        def __init__(self, enc, head, pad_id):
            super().__init__()
            self.enc, self.head, self.pad_id = enc, head, pad_id

        def forward(self, g1, e1, g2, e2):
            out = []
            for g, e in ((g1, e1), (g2, e2)):
                h = self.enc._encode(g, e, src_key_padding_mask=g.eq(self.pad_id))[:, 0, :]
                out.append(F.normalize(self.head(h).float(), dim=1))
            return out[0], out[1]

    # FREEZE WHAT `_encode` NEVER TOUCHES. scGPT's TransformerModel carries an expression decoder and a
    # classifier head alongside the encoder, and `_encode` -- the only path this stage uses -- runs just
    # `encoder -> value_encoder -> bn -> transformer_encoder`. Those 16 leftover parameters get no
    # gradient, which DDP reports as "Expected to have finished reduction in the prior iteration".
    # Freezing is the honest fix: `find_unused_parameters=True` would pay for a full graph traversal on
    # every backward pass to rediscover the same dead heads, and they would still sit in Adam's state.
    # This was invisible while the encoder was called through `.module`, because DDP saw no forward at all.
    _USED = ("encoder", "value_encoder", "bn", "dsbn", "transformer_encoder")
    _froz = [nm for nm, p in model.named_parameters() if nm.split(".")[0] not in _USED]
    for nm, p in model.named_parameters():
        if nm.split(".")[0] not in _USED:
            p.requires_grad_(False)
    log(f"  frozen {len(_froz)} parameters outside the _encode path: "
        f"{sorted({nm.split('.')[0] for nm in _froz})}")

    net = TwoView(model, head, vocab[cfg["pad_token"]]).to(dev)
    core = net
    if DDP:
        # No `find_unused_parameters`: with do_mvc=False every parameter takes gradient in both views,
        # so the extra graph traversal it costs would buy nothing. `broadcast_buffers=False` because
        # there are no buffers that must stay in lockstep (no BN in this path).
        net = nn.parallel.DistributedDataParallel(net, device_ids=[LOCAL], broadcast_buffers=False,
                                                  gradient_as_bucket_view=True)

    # LEARNING RATE AND THE WORLD SIZE. DDP averages the gradient across ranks, so batch 32 on 4 ranks is
    # an effective batch of 128 cells where v1 had 32 -- and on top of that `align()` gathers the InfoNCE
    # negatives, so the contrastive term sees 128 instead of 32. Both push the same way: the gradient is
    # less noisy, so the step can be larger. Scaled by sqrt(WORLD), not linearly, because Adam already
    # normalises by the gradient magnitude. An explicit `--lr` is taken as final.
    # As a function there is no argv to read, so "explicit" also covers an `lr` that differs from the
    # field's default; via the CLI the argv test alone decides, exactly as before.
    _lr_given = (c.lr != V2EncoderConfig.lr
                 or any(x == "--lr" or x.startswith("--lr=") for x in sys.argv[1:]))
    _lr = c.lr if _lr_given else c.lr * WORLD ** 0.5
    log(f"  optimizer: lr {_lr:.3g} (world {WORLD}, base {c.lr:.3g}"
        f"{'; --lr given, not scaled' if _lr_given else f', scale {WORLD ** 0.5:.3g}'}), "
        f"effective batch {c.batch * WORLD} cells, InfoNCE negatives {c.batch * WORLD}")
    opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=_lr)
    # THE SCHEDULE'S UNIT IS AN EPOCH, AND AN EPOCH IS NOT AN EVALUATION. v1 decayed 0.9 per epoch, and
    # `StepLR(opt, 1, 0.9)` reproduces that only if `sched.step()` is called once per epoch. It was being
    # called from the evaluation block, i.e. every `--ckpt-every` = 500 steps, so the rate fell 0.9x every
    # 500 steps: by step 15,000 it was 4% of its initial value and by step 20,000 1.6%. The run then sat
    # at val 12.41 and early stopping -- correctly -- called it a plateau. The plateau was the schedule,
    # not the model. Over the full 200,000 steps that schedule reaches 2e-4 * 0.9^400 ~ 1e-22.
    #
    # So the schedule counts TRAINING STEPS and decays once per pass over the trainable cells, which is
    # what "per epoch" means under stratified sampling too: the sampler changes which cells are drawn,
    # not how many are drawn per step.
    STEPS_PER_EPOCH = max(1, int(np.ceil(len(tr_idx) / (c.batch * WORLD))))
    log(f"  schedule: 0.9x every {STEPS_PER_EPOCH:,} steps (one pass over {len(tr_idx):,} cells at "
        f"{c.batch * WORLD} per step); {c.steps / STEPS_PER_EPOCH:.1f} epochs in {c.steps:,} steps, "
        f"final lr {_lr * 0.9 ** (c.steps // STEPS_PER_EPOCH):.3g}")
    # WARMUP, WHICH v1 DID NOT HAVE AND THE CHECKPOINT'S OWN RECIPE DID. `args.json` records that this
    # scGPT checkpoint was pretrained with `warmup_ratio_or_step: 10000`. Here the encoder is pretrained
    # and the 512-d read-out head is randomly initialised, so step 1 sends a random head's gradient into
    # pretrained weights at the full rate -- the case warmup exists for. Written as one LambdaLR rather
    # than a SequentialLR so the warmup and the decay are the same function of the step count, and so
    # `--warmup 0` reproduces the previous schedule exactly.
    _W = max(0, c.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (min(1.0, (s + 1) / _W) if _W else 1.0) * 0.9 ** (s // STEPS_PER_EPOCH))
    if _W:
        log(f"  warmup: linear over {_W:,} steps ({_W / STEPS_PER_EPOCH * 100:.1f}% of one epoch)")
    # bf16 rather than fp16 on H100: same throughput, wider dynamic range, and no loss scaling, so the
    # GradScaler and its skipped-step bookkeeping disappear. Kept behind a capability check because an
    # older card would fall back to fp32 rather than fail.
    BF16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    AMP_DTYPE = torch.bfloat16 if BF16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda" and not BF16))

    # A scalar log that can be watched from outside the job. Optional on purpose: a missing tensorboard
    # must not stop a 2-day run, and the JSON lines in train.log remain the source of truth.
    TB = None
    if RANK == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            TB = SummaryWriter(str(OUT / "tb"))
            log(f"  tensorboard -> {OUT}/tb")
        except Exception as e:
            log(f"  tensorboard unavailable ({type(e).__name__}); train.log still records every step")

    step0, best = 0, float("inf")
    bad_evals = 0
    if c.resume and Path(c.resume).exists():
        ck = torch.load(c.resume, map_location="cpu", weights_only=False)
        core.enc.load_state_dict(ck["model"])
        core.head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])      # or the resume restarts at the full learning rate
        step0, best = ck["step"], ck["best"]
        log(f"resumed {c.resume} at step {step0}, best {best:.4f}")

    # ---------------------------------------------------------------- subset sampler
    def draw_subset(hi):
        """An arbitrary gene subset. Uniform half covers the space; structured half is modular and biased.

        The structured half grows from a few seeds, which is what makes a subset *concentrated on a few
        programmes* and often mismatched to the cells it is applied to -- a cardiac panel read on brain is
        the realistic deployment case, and it has to be drawn on purpose rather than left to chance.
        """
        n = int(np.exp(rng.uniform(np.log(c.min_genes), np.log(hi))))
        pool = np.flatnonzero(in_vocab)
        if rng.random() >= c.p_structured:
            return rng.choice(pool, size=min(n, len(pool)), replace=False)
        k = int(rng.integers(1, 20))
        seeds = rng.choice(pool, size=min(k, len(pool)), replace=False)
        # expand around the seeds in the gene-index neighbourhood as a cheap stand-in for a co-expression
        # graph; a real graph is a pre-registered upgrade, and the point here is concentration, not biology
        out = set(int(s) for s in seeds)
        while len(out) < n:
            s = int(rng.choice(seeds))
            w = int(rng.integers(1, 400))
            lo, hi_ = max(0, s - w), min(len(genes), s + w)
            cand = pool[(pool >= lo) & (pool < hi_)]
            if len(cand) == 0:
                out.update(int(x) for x in rng.choice(pool, size=min(n - len(out), len(pool)), replace=False))
                break
            out.update(int(x) for x in cand[:max(1, n - len(out))])
        return np.fromiter(out, np.int64, len(out))[:n]

    class StepDataset(torch.utils.data.IterableDataset):
        """Yields fully tokenised training steps, so the GPU never waits on Python.

        THIS IS THE BOTTLENECK THE WHOLE CLASS EXISTS FOR. Tokenising a step means a Python loop over the
        batch's cells building one dict each, twice over (subset view and wide view) -- 2 x batch
        iterations per step, plus the collator's padding. Done on the training process that is dead time
        on four H100s. Done in `num_workers` processes it overlaps with the backward pass and disappears.

        Each worker opens its OWN h5py handle. A backed AnnData handle inherited across `fork` is not
        safe to read from concurrently -- the failure is silent wrong data, not an error -- so the handle
        is created in the worker, keyed by worker id.
        """

        def __init__(self, rows_pool, z, gene_ids, in_vocab, store, coll, vocab, cfg, max_len,
                     batch, min_genes, p_structured, seed, strata=None):
            self.pool, self.z = rows_pool, z
            self.strata, self.strata_p = strata if strata else (None, None)
            self.gene_ids, self.in_vocab = gene_ids, in_vocab
            self.store, self.coll, self.vocab, self.cfg = store, coll, vocab, cfg
            self.max_len, self.batch = max_len, batch
            # Draw 25% more rows than the batch needs, so the cells lost to an empty view are covered
            # without a redraw. A redraw is still the fallback when even the over-draw comes up short.
            self.over = batch + max(4, batch // 4)
            self.min_genes, self.p_structured, self.seed = min_genes, p_structured, seed
            self.n_genes = len(gene_ids)
            self._X = None
            self._mm = None

        def _x(self):
            """The CSR arrays, loaded into RAM when `csr/` exists, else the backed h5ad.

            WHY MEMMAP. The h5ad path is what stalled this trainer twice: 16-24 worker processes opening
            the same 8 GB file put every rank to sleep inside its first batch, with the GPUs showing 100%
            because NCCL was spinning on a collective none of them reached. HDF5 brings file locking, gzip
            chunks that decompress 104,036 elements to serve a 1,600-nonzero row, and a handle that is not
            fork-safe. The arrays are read ONCE in the parent, before the DataLoader forks, so each worker
            inherits them copy-on-write and -- being read-only -- never duplicates a page. One copy per
            rank, not per worker. See aceso/TRAIN_IO_DESIGN.md for why in-RAM beats memmap here: the
            corpus is 26 GB against 1.15 TB allocated, and a memmap is cold on first touch, which is
            exactly the moment both stalls happened.
            """
            if self._X is None:
                if CSR is not None:
                    self._mm = CSR                 # inherited through fork, read-only, never copied
                    self._X = "memmap"
                else:
                    import anndata as ad
                    self._X = ad.read_h5ad(self.store / "rna.h5ad", backed="r").X
            return self._X

        def _rows(self, rows):
            """-> csr_matrix of the given rows, from whichever backing is in use."""
            X = self._x()
            if X != "memmap":
                return sp.csr_matrix(X[rows])
            D, I, P = self._mm
            lens = (P[rows + 1] - P[rows]).astype(np.int64)
            tot = int(lens.sum())
            dat = np.empty(tot, np.float32); ind = np.empty(tot, np.int32)
            o = 0
            for r, L in zip(rows, lens):
                a_ = int(P[r]); dat[o:o + L] = D[a_:a_ + L]; ind[o:o + L] = I[a_:a_ + L]; o += int(L)
            indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
            return sp.csr_matrix((dat, ind, indptr), shape=(len(rows), self.n_genes))

        def _subset(self, rng, hi):
            n = int(np.exp(rng.uniform(np.log(self.min_genes), np.log(hi))))
            pool = np.flatnonzero(self.in_vocab)
            if rng.random() >= self.p_structured:
                return rng.choice(pool, size=min(n, len(pool)), replace=False)
            # structured: grow from a few seeds so the subset is concentrated on a few programmes and is
            # often mismatched to the cells it is read on -- the realistic deployment case
            k = int(rng.integers(1, 20))
            seeds = rng.choice(pool, size=min(k, len(pool)), replace=False)
            # THIS LOOP USED TO BE ABLE TO SPIN FOREVER, and that is what stopped every Marlowe run.
            # The windows are at most 2x399 genes wide, `n` reaches 1200, and the loop only exits when
            # `out` reaches n -- so with few seeds whose windows overlap, every further iteration
            # re-adds genes already in the set and the condition is never met. The worker pins a core at
            # 100%, stops yielding batches, the DataLoader blocks on that worker's turn, and all four
            # ranks sit in the all-reduce behind it. It takes a few steps to show because the prefetch
            # queue filled during startup has to drain first: steps 1-10 came out in 2 s, then nothing.
            out = set(int(s) for s in seeds)
            stuck = 0
            while len(out) < n and stuck < 20:
                before = len(out)
                s = int(rng.choice(seeds)); w = int(rng.integers(1, 400))
                cand = pool[(pool >= max(0, s - w)) & (pool < s + w)]
                if len(cand):
                    out.update(int(x) for x in cand[:max(1, n - len(out))])
                stuck = 0 if len(out) > before else stuck + 1
            if len(out) < n:
                # The seeds' neighbourhoods are exhausted. Top up at random rather than spin: a subset
                # that is mostly local with a random tail is still a structured subset, and a subset
                # that never returns is not a subset at all.
                have = np.fromiter(out, np.int64, len(out))
                rest = pool[~np.isin(pool, have)]
                if len(rest):
                    out.update(int(x) for x in rng.choice(rest, size=min(n - len(out), len(rest)),
                                                          replace=False))
            return np.fromiter(out, np.int64, len(out))[:n]

        def _tok(self, M, cols):
            ex, keep = [], []
            for i in range(M.shape[0]):
                lo, hi = M.indptr[i], M.indptr[i + 1]
                v = M.data[lo:hi]; nz = v > 0
                if not nz.any():
                    continue
                g = self.gene_ids[cols[M.indices[lo:hi][nz]]]
                ex.append({"genes": torch.from_numpy(np.insert(g, 0, self.vocab["<cls>"])).long(),
                           "expressions": torch.from_numpy(
                               np.insert(v[nz], 0, self.cfg["pad_value"])).float()})
                keep.append(i)
            if not ex:
                return None, None
            bt = self.coll(ex)
            return bt, np.asarray(keep)

        def __iter__(self):
            wi = torch.utils.data.get_worker_info()
            wid = 0 if wi is None else wi.id
            rng = np.random.default_rng(self.seed + 7919 * wid)
            self._x()
            ALL = np.flatnonzero(self.in_vocab)
            while True:
                if self.strata is None:
                    rows = np.sort(rng.choice(self.pool, size=self.over, replace=False))
                else:
                    # Stratified: the step's organ composition is exact, not just right on average.
                    k = rng.multinomial(self.over, self.strata_p)
                    rows = np.sort(np.concatenate([
                        rng.choice(g, size=min(c, len(g)), replace=False)
                        for g, c in zip(self.strata, k) if c]))
                Mall = self._rows(rows)
                sub = self._subset(rng, self.max_len)
                b1, k1 = self._tok(Mall[:, sub].tocsr(), sub)
                b2, k2 = self._tok(Mall[:, ALL].tocsr(), ALL)
                if b1 is None or b2 is None:
                    continue
                # EVERY BATCH IS EXACTLY `self.batch` CELLS, and that is a hard requirement, not tidiness.
                # `align()` all_gathers each rank's z to build the cross-rank InfoNCE negatives, and
                # all_gather needs the same shape on every rank. A cell is dropped here whenever it has
                # no nonzero gene in one of the two views, which happens at a rate that differs batch to
                # batch and therefore rank to rank -- so a variable batch desynchronises the collective
                # and the job hangs. Rows are over-drawn and then cut back at random (not truncated: the
                # row indices are sorted and cells.tsv is ordered by sample, so keeping the first k would
                # bias every batch toward the samples that happen to sit early in the store).
                common = np.intersect1d(k1, k2)
                if len(common) < self.batch:
                    continue
                if len(common) > self.batch:
                    common = np.sort(rng.choice(common, self.batch, replace=False))
                i1 = np.searchsorted(k1, common); i2 = np.searchsorted(k2, common)
                yield dict(g1=b1["gene"][i1], e1=b1["expr"][i1],
                           g2=b2["gene"][i2], e2=b2["expr"][i2],
                           zt=torch.from_numpy(self.z[rows[common]]).float(),
                           nsub=len(sub))

    def run_views(b):
        """Both views through the DDP-wrapped encoder+head, returning L2-normalised z.

        One call, so one forward pass per backward pass. See `TwoView` for why that matters.
        """
        return net(b["g1"].to(dev, non_blocking=True), b["e1"].to(dev, non_blocking=True),
                   b["g2"].to(dev, non_blocking=True), b["e2"].to(dev, non_blocking=True))

    class _Gather(torch.autograd.Function):
        """all_gather that passes gradient back to the owning rank.

        `torch.distributed.all_gather` is not differentiable, so using it naively would silently drop the
        gradient from every other rank's negatives -- the loss would look right and the model would learn
        from its own 32 only.
        """

        @staticmethod
        def forward(ctx, x):
            out = [torch.zeros_like(x) for _ in range(WORLD)]
            torch.distributed.all_gather(out, x.contiguous())
            ctx.save_for_backward(x)
            return torch.cat(out, 0)

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return g[RANK * x.shape[0]:(RANK + 1) * x.shape[0]]

    def align(p, q, tau):
        """(1 - cos) + symmetric InfoNCE, with the negatives GATHERED ACROSS RANKS.

        The cosine term is per-cell and local. The InfoNCE term is not: with batch 32 per rank, a local
        softmax sees 32 negatives, while gathering gives 32 x WORLD. That is a quality change as much as a
        speed one -- contrastive objectives sharpen with the number of negatives -- so 4 GPUs buy more than
        4x throughput here. v1 was single-GPU and therefore always had 32.
        """
        p = F.normalize(p, dim=1); q = F.normalize(q, dim=1)
        cos = (1 - (p * q).sum(1)).mean()
        if DDP:
            pg, qg = _Gather.apply(p), _Gather.apply(q)
            off = RANK * p.shape[0]
            lg = p @ qg.T / tau
            lg2 = q @ pg.T / tau
            lab = torch.arange(p.shape[0], device=p.device) + off
            return cos + 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg2, lab))
        lg = p @ q.T / tau
        lab = torch.arange(len(p), device=p.device)
        return cos + 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab))

    ds = StepDataset(tr_idx, Z, gene_ids, in_vocab, STORE, coll, vocab, cfg, c.max_len,
                     c.batch, c.min_genes, c.p_structured, c.seed + 1000 * RANK, strata=tr_strata)
    # VALIDATION IS ON HELD-OUT SAMPLES, not held-out cells: `va_idx` comes from splits.json, which
    # splits by sample, so a val cell shares no donor, no batch and no library with any training cell.
    # Unstratified on purpose -- the strata reweight what the model SEES, and the question validation
    # answers is how it does on the corpus as it actually is.
    vds = StepDataset(va_idx, Z, gene_ids, in_vocab, STORE, coll, vocab, cfg, c.max_len,
                      c.batch, c.min_genes, c.p_structured, c.seed + 50_000 + 1000 * RANK, strata=None)

    @torch.no_grad()
    def validate(n_batches):
        """Mean of the same loss on held-out samples. EVERY rank runs it and the result is all-reduced.

        Rank 0 doing this alone is what a careless version would do, and it is the shape of the failure
        that already cost this run twice: one rank busy while the others wait at the next collective
        until the NCCL watchdog fires. Here every rank evaluates its own shard of batches and the mean
        is reduced, so there is no skew to time out on.
        """
        net.eval()
        it = iter(torch.utils.data.DataLoader(vds, batch_size=None, num_workers=2,
                                              persistent_workers=False))
        tot, n = 0.0, 0
        for _ in range(n_batches):
            try:
                b = next(it)
            except StopIteration:
                break
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=dev.type == "cuda"):
                zs, zw = run_views(b)
                zt = F.normalize(b["zt"].to(dev), dim=1)
                L = align(zs, zt, c.tau) + align(zw, zt, c.tau) + c.lam * align(zs, zw.detach(), c.tau)
            tot += float(L); n += 1
        net.train()
        v = torch.tensor([tot, float(n)], device=dev)
        if DDP:
            torch.distributed.all_reduce(v)
        return float(v[0] / max(float(v[1]), 1.0))

    # `num_workers` is the knob that decides whether the GPUs are busy. 6 per rank x 4 ranks = 24 of the
    # node's cores on tokenisation, which at the measured per-step cost keeps four H100s fed; raise it if
    # `step/s` scales below linear in WORLD, lower it if the node's cores are contended.
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=c.workers,
                                     pin_memory=True, persistent_workers=c.workers > 0,
                                     prefetch_factor=4 if c.workers > 0 else None)

    t_start = time.time()
    log(f"\ntraining: max_len {c.max_len} (pretrained length), batch {c.batch}, lam {c.lam}, "
        f"world {WORLD}, workers {c.workers}, amp {'bf16' if BF16 else 'fp16'}")
    hist, step = [], step0 - 1
    io_s = 0.0          # seconds spent waiting on the DataLoader, the other half of every step
    it = iter(dl)
    for step in range(step0, c.steps):
        if c.time_budget and time.time() - t_start > c.time_budget:
            log(f"time budget reached at step {step}; checkpointing and exiting for the next chain link")
            break
        # DDP IS SYNCHRONOUS, SO THE SLOWEST RANK SETS THE PACE. Every step ends in an all-reduce that no
        # rank leaves until all four arrive, so a rank starved of batches does not cost its own GPU --
        # it costs all four. `io` below is the share of wall-clock spent inside `next(it)`: if it is
        # large the fix is workers or tokenisation cost, and if it is small the GPUs are the limit.
        _t0 = time.time()
        b = next(it)
        io_s += time.time() - _t0
        net.train()
        zt = b["zt"].to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=dev.type == "cuda"):
            z_sub, z_wide = run_views(b)
            loss = (align(z_sub, zt, c.tau)
                    + align(z_wide, zt, c.tau)
                    + c.lam * align(z_sub, z_wide.detach(), c.tau))
        opt.zero_grad(set_to_none=True)
        params = [q for q in net.parameters() if q.requires_grad]
        if BF16:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
        sched.step()                     # per TRAINING STEP; StepLR's period is one epoch, see above
        hist.append(float(loss))

        # THE FIRST POINT GOES IN IMMEDIATELY, then every 10 steps for the first hundred, then every
        # hundred. Logging only at step 100 means an empty TensorBoard for the first several minutes,
        # which is indistinguishable from a hung job -- and this run has now hung three times, so that
        # ambiguity has cost more than the log lines ever will.
        _d = step + 1 - step0
        if _d == 1 or (_d < 100 and _d % 10 == 0) or (step + 1) % 100 == 0:
            el = time.time() - t_start
            sps = _d / max(el, 1e-9)
            ml = float(np.mean(hist[-100:]))
            log(f"  step {step+1}  loss {ml:.4f}  |subset| {b['nsub']}  "
                f"{sps:.2f} step/s  io {io_s/max(el,1e-9)*100:.0f}%  "
                f"lr {opt.param_groups[0]['lr']:.2e}  {el:.0f}s")
            if RANK == 0:
                if TB is not None:
                    TB.add_scalar("loss/train", ml, step + 1)
                    TB.add_scalar("lr", opt.param_groups[0]["lr"], step + 1)
                    TB.add_scalar("throughput/step_per_s", sps, step + 1)
                    TB.add_scalar("throughput/io_fraction", io_s / max(el, 1e-9), step + 1)
                    TB.add_scalar("subset/n_genes", float(b["nsub"]), step + 1)
                    TB.flush()
                # A heartbeat an OUTSIDE watcher can read. TensorBoard cannot report that this job has
                # stalled, because a stall takes the writer down with it -- twice already this run died
                # with nothing after the launch banner. A file whose mtime stops advancing can be noticed
                # by something that is still alive.
                (OUT / "heartbeat.json").write_text(json.dumps(
                    dict(step=step + 1, loss=ml, step_per_s=sps, t=time.time(),
                         elapsed_s=round(el))) + "\n")
        if (step + 1) % c.ckpt_every == 0:
            tr_loss = float(np.mean(hist[-c.ckpt_every:]))
            val = validate(c.val_batches)
            improved = val < best - 1e-4
            if improved:
                best, bad_evals = val, 0
            else:
                bad_evals += 1
            if RANK == 0:
                st = dict(model=core.enc.state_dict(),
                          head=core.head.state_dict(),
                          opt=opt.state_dict(), scaler=scaler.state_dict(), sched=sched.state_dict(),
                          step=step + 1, best=best, val=val, train=tr_loss, args=asdict(c))
                # last.pt always, best.pt only when validation improved. `best` now tracks the HELD-OUT
                # loss; it used to track a running mean of the training loss, which cannot tell you the
                # model has started overfitting -- it goes down either way.
                torch.save(st, OUT / "last.pt.tmp"); os.replace(OUT / "last.pt.tmp", OUT / "last.pt")
                if improved:
                    torch.save(st, OUT / "best.pt.tmp"); os.replace(OUT / "best.pt.tmp", OUT / "best.pt")
                if (step + 1) % (c.ckpt_every * 10) == 0:
                    torch.save(st, OUT / f"ckpt_step{step+1}.pt")
                log(f"  step {step+1}  train {tr_loss:.4f}  val {val:.4f}"
                    f"{'  *best*' if improved else f'  (no gain x{bad_evals})'}")
                if TB is not None:
                    TB.add_scalar("loss/val", val, step + 1); TB.flush()
            if bad_evals >= c.patience:
                log(f"\nEARLY STOP at step {step+1}: validation has not improved for {c.patience} "
                    f"evaluations (best {best:.4f}). Releasing the allocation rather than spending it "
                    f"on steps that are not buying generalisation.")
                break

    if RANK == 0:
        torch.save(dict(model=core.enc.state_dict(),
                        head=core.head.state_dict(),
                        opt=opt.state_dict(), scaler=scaler.state_dict(), sched=sched.state_dict(),
                        step=step + 1, best=best, args=asdict(c)), OUT / f"ckpt_step{step+1}.pt")
        log(f"\nstopped at step {step+1}; checkpoints in {OUT}")
    if DDP:
        torch.distributed.destroy_process_group()
    return OUT


def main(argv=None):
    """The script's own command line, unchanged: ``--out`` and ``--store`` relative to the project root."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default="data/train_v2")
    ap.add_argument("--max-len", type=int, default=1200, help="the checkpoint's pretraining length")
    ap.add_argument("--z-dim", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--min-genes", type=int, default=50)
    ap.add_argument("--p-structured", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--ckpt-every", type=int, default=500,
                    help="was 2000; two runs died before reaching it and left nothing to resume from. "
                         "At ~1.3 step/s this is ~6 min of work at risk instead of ~25.")
    ap.add_argument("--patience", type=int, default=30,
                    help="stop after this many consecutive evaluations without a new best VALIDATION "
                         "loss. 10 was too tight: at --ckpt-every 500 that is 5,000 steps, which on a "
                         "1.72M-cell corpus is 0.37 of one pass -- short enough that ordinary "
                         "validation noise reads as convergence. 30 evaluations is 15,000 steps, about "
                         "1.1 passes.")
    ap.add_argument("--warmup", type=int, default=0,
                    help="linear learning-rate warmup over this many steps. MEASURED AND IT DOES NOT "
                         "MATTER HERE: 0 vs 1,000 steps, best validation over 4,000 steps was 12.8032 "
                         "vs 12.7935, a difference of 0.0097 against a between-evaluation fluctuation "
                         "of +/-0.2. Warmup protects randomly initialised weights from a large first "
                         "step; here the encoder is pretrained on this very task shape (`trunc_by_sample`), "
                         "the new parameters are one linear layer, and the gradient is clipped at 1.0. "
                         "scGPT itself used 10,000 (args.json `warmup_ratio_or_step`) because its whole "
                         "network was random. Default 0, which is also what v1 used.")
    ap.add_argument("--val-batches", type=int, default=20,
                    help="batches per rank per evaluation; 20 x 32 x 4 ranks = 2,560 held-out cells")
    ap.add_argument("--resume")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="seconds; checkpoint and exit before the wall clock kills us")
    ap.add_argument("--workers", type=int, default=6,
                    help="tokenisation workers PER RANK; this is the knob that decides GPU utilisation")
    ap.add_argument("--organ-alpha", type=float, default=0.5,
                    help="organ sampling temperature: p_organ ~ n_organ^alpha. 1.0 = uniform over cells "
                         "(59%% heart, measured), 0.0 = one share per organ. See organ_weights()")
    ap.add_argument("--organ-cap", type=float, default=10.0,
                    help="ceiling on any organ's per-cell weight, in multiples of uniform; bounds how "
                         "often a tiny stratum is revisited")
    ap.add_argument("--no-ram", action="store_true",
                    help="read the backed h5ad instead of loading csr/ into RAM (for a machine where "
                         "the corpus does not fit; see aceso/TRAIN_IO_DESIGN.md)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return train_v2_encoder(V2EncoderConfig(**vars(a)))


if __name__ == "__main__":  # pragma: no cover
    main()
