"""ACESO v2 stage 2: z + DNA -> per-cell ATAC (32 bp bins) and per-gene RNA.

Reads ``data/train_v2/`` -- the store :mod:`._v2_encoder` is trained on -- plus a stage-1 checkpoint,
and fits two heads that share one trunk. Both are LINEAR in the trunk features and conditioned on the
cell state ``z`` through a bilinear form. The trunk itself is
:class:`._alphagenome.AlphaGenomeScooby`: AlphaGenome frozen, LoRA + LoCon adapters on it, and
scooby's ``cell_state_to_conv`` turning ``z`` into a per-cell 1x1 convolution.

Why linear, and why that is not a compromise
--------------------------------------------
The decoder search is closed (``aceso-campaign-closed``): FiLM lost 48/48, cross-cell attention lost,
MoLoRA's gate collapsed (per-cell entropy 1.351 of ln 4 = 1.386) and lost on every metric (SC1 0.8309
vs 0.8251, r_diff 0.372 vs 0.310), VQ-MoLoRA repaired the gate statistics but not the conditioning,
and R6's affine gate was refuted under a pre-registered paired criterion. The reason is structural
rather than empirical -- Jayakumar et al. (ICLR 2020) show FiLM, gating, attention and hypernetworks
are one bilinear operator refactored, so with a frozen trunk and a fixed ``z`` a linear read-out
already attains the function class. XChrom chose the same read-out. Capacity belongs in the trunk.

**One caveat that must not be dropped.** That argument was derived with a FROZEN trunk. If the trunk
is fine-tuned here it does not carry over automatically, and "linear is provably sufficient" may not
be repeated for the trained-trunk regime without re-deriving it. ``freeze_trunk`` is the default for
exactly this reason.

The RNA head needs no BAM, and a coverage head would be worse
-------------------------------------------------------------
HDMA is SHARE-seq and ENCODE is 10x; both are 3'-biased and UMI-deduplicated, so a "coverage track"
from them is a spike near the 3' end plus library artefacts -- fitting that teaches 10x chemistry, not
regulation. Borzoi predicts coverage because CAGE and RNA-seq are full-length unbiased assays. For 3'
scRNA the informative quantity is per-gene expression in that cell state, which is exactly what the
count matrix holds. So::

    m_g    fixed, strand-aware exonic mask over the window's bins for gene g
    h_g    = sum_b m_g[b] * H[b] / sum_b m_g[b]
    lam_g  = <W_rna(z), h_g> + b_rna(z)
    loss   Poisson NLL against the observed UMI for g in that cell, size-factor offset, summed over
           the window's genes and averaged over cells (see :func:`poisson_nll` for why summed)

What is given up: no base-resolution RNA, hence no isoform read-out. Variant effects survive --
perturb the sequence and read ``d lam_g``, which is a direct eQTL prediction and closes the loop with
the ten eQTL columns of Fig 4b.

The ATAC head
-------------
Unchanged from v1 in form: per 32 bp bin, Poisson NLL against the cell's fragment counts in that bin.
Bin resolution is real here because fragments carry exact coordinates -- which is also why the ATAC
target is read from the tabix fragments rather than from any binned matrix.

Running it
----------
::

    python -m epione.models.aceso._v2_decoder --out runs/s2_v2 --stage1 runs/s1_v2/ckpt_step20000.pt
    torchrun --standalone --nproc_per_node=4 -m epione.models.aceso._v2_decoder \\
        --out runs/s2_v2 --stage1 ...

or from Python, with :func:`train_v2_decoder` and :class:`V2DecoderConfig`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# The checkpoint records `args` as the argparse namespace of the run that wrote it, and the released
# weights (`aceso/runs/s2_C/final_step7937.pt`) are read back by the figure code. These are its keys,
# in the order the flags were declared, so a checkpoint written here and one written by the original
# script stay interchangeable.
_ARG_FIELDS = ("out", "store", "stage1", "z", "z_bank", "axes", "within_weight", "within_frac",
               "within_pool", "triples", "heads", "window", "bin", "z_dim", "cells_per_step", "lr",
               "w_rna", "freeze_trunk", "steps", "workers", "val_windows", "ckpt_every", "resume",
               "init_from", "time_budget", "dense_dir", "pair_frac", "axis_frac", "w_dense",
               "w_contrast", "rna_trunk_scale", "snap_every", "warmup", "organism", "ag_weights",
               "seed", "lr_given")


@dataclass
class V2DecoderConfig:
    """Everything stage 2 reads, one field per command-line flag of the original script.

    Relative paths are taken under the project root (``root``, resolved by :mod:`._sites`); absolute
    ones are used as given. The non-obvious fields:

    ``heads``
        comma-separated subset of ``atac,rna``. Dropping ``rna`` keeps the gene read-out in the
        module (DDP needs a static graph) but zero-weights its loss.
    ``z``, ``z_bank``, ``stage1``
        the conditioning input. ``z`` is one per-cell ``.npy`` aligned to ``cells.tsv``; ``z_bank``
        is a comma-separated list of z files to SAMPLE AMONG, one view drawn per training step.
        Measured 2026-11-12 on the full store: a real deployment panel carries only 1-26% of a
        cell's UMIs and the flagship MERFISH 480 carries 7.9% (3.2% of its detected genes), so a
        wide-view z and a panel-view z are not the same conditioning variable -- about 12x apart in
        input information for the 480. Training on wide z alone and deploying on panel z is a silent
        train/deploy mismatch: both losses stay healthy while cell-type resolution collapses,
        because the decoder learns to trust a sharpness in z that the panel cannot supply.
    ``axes``
        named within-type RNA axes to condition on alongside z, an ``.npz`` with ``score``, ``mask``,
        ``names``; the cell-state embedding is then (z, axes) rather than z alone. Measured on the
        patient cohort, a 0.73 SD group shift along the chamber axis arrives at z as a 0.0013 cosine
        wiggle inside a 512-d vector whose principal structure is cell type, and the decoder
        faithfully passes on something buried -- within/across separation 0.3%. A ridge probe shows
        the information is still in z (r 0.347 within CM), so what fails is scale, not presence.
        ``"none"`` conditions on z alone, which is the configuration that measurement refuted.
    ``within_weight``, ``within_frac``, ``within_pool``, ``triples``
        stage B. ``within_weight`` > 0 adds v1's within-type contrast (:func:`within_kl`);
        ``within_frac`` of steps draw their cells as two halves of ONE (organ, cell type, sample) at
        two ends of one axis, the rest stay random, which is what keeps the shared profile from being
        torn up; ``within_pool`` bins are pooled before the binomial split; ``triples`` is the gated
        triple table.
    ``dense_dir``, ``pair_frac``, ``w_dense``, ``w_contrast``
        stage B v2. A directory of dense group pseudobulk tracks (built with ``DENSE_BUILD=1``) gives
        thousands of cells behind each target profile instead of ~1 insertion per cell. ``pair_frac``
        of steps are drawn as two cell-type groups with dense targets; ``w_dense`` weights the
        group-mean profile term and ``w_contrast`` the between-group split term
        (:func:`dense_split_kl`), each rescaled against the ATAC loss.
    ``axis_frac``
        share of steps drawn as a within-type axis triple (cycle, pcw).
    ``rna_trunk_scale``
        gradient of the RNA loss that reaches the shared trunk, as a factor; the RNA read-out itself
        always gets the full gradient.
    ``freeze_trunk``
        default True, and flipping it also invalidates the "a linear read-out is provably sufficient"
        argument in this module's docstring. That is a modelling decision, not a flag to flip idly.
    ``organism``
        AlphaGenome organism index: 0 human, 1 mouse. The v2 corpus is human; :mod:`._alphagenome`
        hard-codes 1 because v1 was mouse.
    ``init_from``
        warm start the WEIGHTS only -- step, optimiser and scheduler are reset. Stage B is a
        fine-tune, not a continuation: inheriting stage A's step count would end the run before it
        began, and inheriting its scheduler would start at the tail of a decay fitted to a different
        horizon. ``resume``/``last.pt`` remain the path for an actual interrupted run.
    ``time_budget``
        seconds; the loop writes ``last.pt`` itself when the budget expires rather than trusting the
        last periodic checkpoint, because the next link in the chain resumes from it.
    ``snap_every``
        keep an immutable ``step<N>.pt`` every N steps, alongside the rolling ``last.pt``/``best.pt``.
    ``lr_given``
        True when ``--lr`` was passed explicitly, in which case the learning rate is taken as final
        and not scaled by ``sqrt(world size)``.
    ``root``, ``scooby_src``, ``hg38_fa``, ``folds_bed``, ``exon_masks``
        resources outside the package; ``None`` resolves them through :func:`._sites.site_path`.
    """

    out: Optional[str] = None
    store: str = "data/train_v2"
    stage1: Optional[str] = None
    z: Optional[str] = None
    z_bank: Optional[str] = None
    axes: str = "data/train_v2/axes.npz"
    within_weight: float = 0.0
    within_frac: float = 0.5
    within_pool: int = 8
    triples: str = "data/train_v2/within_triples.json"
    heads: str = "atac,rna"
    window: int = 524288
    bin: int = 32
    z_dim: int = 512
    cells_per_step: int = 16
    lr: float = 1e-4
    w_rna: float = 1.0
    freeze_trunk: bool = True
    steps: int = 100000
    workers: int = 6
    val_windows: int = 64
    ckpt_every: int = 2000
    resume: Optional[str] = None
    init_from: Optional[str] = None
    time_budget: Optional[float] = None
    dense_dir: Optional[str] = None
    pair_frac: float = 0.5
    axis_frac: float = 0.2
    w_dense: float = 1.0
    w_contrast: float = 1.0
    rna_trunk_scale: float = 1.0
    snap_every: int = 1000
    warmup: Optional[int] = None
    organism: int = 0
    ag_weights: Optional[str] = None
    seed: int = 0
    lr_given: bool = False
    # resolved through `._sites` when left at None; not part of the checkpoint's `args`
    root: Optional[str] = None
    scooby_src: Optional[str] = None
    hg38_fa: Optional[str] = None
    folds_bed: Optional[str] = None
    exon_masks: Optional[str] = None


def make_model(z_dim, device, organism, ckpt=None, train=True, rna_head=True, freeze_trunk=True):
    """`AlphaGenomeScooby` with a second, gene-level head. Nothing in the ATAC path is re-derived.

    The v1 decoder IS `epione.models.aceso._alphagenome.AlphaGenomeScooby`: AlphaGenome's trunk frozen,
    LoRA + LoCon adapters on it, and scooby's `cell_state_to_conv` producing a per-cell 1x1 convolution
    from z. `forward_seq_to_emb` returns [B, 1536, 6144] -- the 32 bp trunk embedding over the 196,608 bp
    target -- which is exactly the `H` the RNA head needs, so both heads share one trunk pass.

    ORGANISM INDEX. `_alphagenome.py` hard-codes `MOUSE = 1` because v1 trained on mouse embryos. The v2
    corpus is human, and AlphaGenome's organism embedding is added to the trunk
    (`trunk + ag.organism_embed(idx)`), so leaving it at 1 would condition every human window on the
    mouse organism. `alphagenome_pytorch` documents `organism_idx = 0  # 0=human, 1=mouse`.
    """
    import torch
    import torch.nn as nn

    from ._decoder import build_decoder
    from ._alphagenome import AlphaGenomeScooby
    from ._sites import site_path

    class ACESOv2(AlphaGenomeScooby):
        """ATAC head verbatim from the parent; RNA head added; organism index configurable."""

        def __init__(self, ag, cell_emb_dim, organism=0):
            super().__init__(ag, cell_emb_dim)
            self.organism = organism
            E = self.embedding_dim
            # Same shape as `cell_state_to_conv` but one output per cell rather than per bin: the RNA
            # read-out is a scalar log-rate per gene, so it needs (E + 1) weights, not (E + 1) x bins.
            self.cell_state_to_gene = nn.Sequential(
                nn.Linear(cell_emb_dim, 128), nn.GELU(), nn.Linear(128, 256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, 1024), nn.GELU(), nn.Dropout(0.2), nn.Linear(1024, E + 1))
            for m in self.cell_state_to_gene:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
            nn.init.zeros_(self.cell_state_to_gene[-1].bias)

        def _org(self, B, device):
            return torch.full((B,), self.organism, dtype=torch.long, device=device)

        def forward_genes(self, seq_emb, cell_emb, masks):
            """seq_emb [B, E, L] ; cell_emb [B, n_cells, z] ; masks [B, G, L] -> log-rate [B, n_cells, G]

            Exonic mean-pooling first, then the bilinear read-out. Pooling before the read-out is what
            keeps the trunk pass shared: the gene features h_g do not depend on the cell.
            """
            p = self.cell_state_to_gene(cell_emb)                      # [B, n, E+1]
            if masks is None or masks.shape[1] == 0:
                # No annotated gene overlaps this window. Returning early would leave
                # `cell_state_to_gene` out of the autograd graph, and DDP then hangs waiting for a
                # bucket that never arrives (or needs find_unused_parameters, which costs a full
                # parameter scan every step). A zero-weighted term keeps the graph static.
                return p.sum() * 0.0
            w = masks / masks.sum(-1, keepdim=True).clamp(min=1.0)
            hg = torch.einsum("bgl,bel->bge", w, seq_emb)              # [B, G, E] exon-pooled
            # bias is per cell, broadcast over the window's genes: [B, n, 1] against [B, n, G]
            return torch.einsum("bge,bne->bng", hg, p[..., :-1]) + p[..., -1:]

        def forward(self, sequence, cell_emb, masks=None):
            """-> (atac rate [B, bins, n_cells], rna log-rate [B, n_cells, G])

            ONE trunk pass feeds both heads, which is the point of pooling the gene features before
            the read-out: `h_g` does not depend on the cell, so the expensive part is shared.

            The parent's `forward` returned the ATAC rate alone. It is overridden rather than
            supplemented because DDP only installs its gradient hooks on the wrapped module's
            `forward`: reaching past it to call the heads separately accumulates gradients that are
            never all-reduced, so every rank would drift onto its own weights -- silently.
            """
            seq_emb = self.forward_seq_to_emb(sequence)
            w, b = self.forward_cell_embs_only(cell_emb)
            atac = self.forward_convs_on_emb(seq_emb, w, b)
            g = getattr(self, "rna_trunk_scale", 1.0)
            if g != 1.0:
                # value unchanged, d(RNA loss)/d(trunk) multiplied by g: the gene head learns at full
                # rate while the shared adapters stay shaped by accessibility
                seq_emb = seq_emb.detach() + g * (seq_emb - seq_emb.detach())
            return atac, self.forward_genes(seq_emb, cell_emb, masks)

    if ckpt is None:
        # `load_alphagenome_scooby` falls back to `hf_hub_download` when given no path. That is the
        # right default on a login node and the wrong one on a compute node without egress, so the
        # staged copy wins when it exists.
        staged = site_path("alphagenome_weights", required=False)
        if staged:
            ckpt = str(staged)
    core = build_decoder(z_dim, device=device, checkpoint=ckpt, train=train, backbone="alphagenome")
    model = ACESOv2(core.ag, z_dim, organism=organism).to(device)
    model.cell_state_to_conv.load_state_dict(core.cell_state_to_conv.state_dict())
    for n_, p_ in model.named_parameters():
        # ACESO_HEADS_ONLY=1: light post-training. The LoRA / LoCon adapters keep their weights; only the two cell-state
        # read-outs train, so no gradient flows back through the trunk.
        adapter = ("lora_" in n_ or "locon_" in n_) and not os.environ.get("ACESO_HEADS_ONLY")
        head = (adapter or n_.startswith("cell_state_to_conv")
                or (rna_head and n_.startswith("cell_state_to_gene")))
        # `freeze_trunk=False` also invalidates the "a linear read-out is provably sufficient"
        # argument in this module's docstring; that is a modelling decision, not a flag to flip idly.
        p_.requires_grad = train and (head or not freeze_trunk)
    return model


def within_kl(out, y, half, pool):
    """KL(observed split || predicted split) per insertion between the two halves of one cell type.

    Ported from v1 (`gse205117/scooby/train_aceso_decoder.py`). Each half's prediction is rescaled to
    its OWN observed window total before the split is taken, so only the predicted shape difference
    between the halves enters and depth does not. This is the only term in the stack that is zero
    when the two halves are predicted identically -- the Poisson-multinomial loss normalises across
    genomic positions and is indifferent to it.
    """
    P = out[0].float(); T = y[0].float()
    h = half.to(P.device)
    pe = P[:, h == 0].sum(1).reshape(-1, pool).sum(1); pl = P[:, h == 1].sum(1).reshape(-1, pool).sum(1)
    te = T[:, h == 0].sum(1).reshape(-1, pool).sum(1); tl = T[:, h == 1].sum(1).reshape(-1, pool).sum(1)
    n = te + tl
    if n.sum() <= 0:
        return out.sum() * 0
    mue = pe * (te.sum() / pe.sum().clamp_min(1e-8)); mul = pl * (tl.sum() / pl.sum().clamp_min(1e-8))
    p = (mul / (mul + mue).clamp_min(1e-12)).clamp(1e-6, 1 - 1e-6)
    q = (tl / n.clamp_min(1e-12)).clamp(1e-6, 1 - 1e-6)
    kl = tl * (q.log() - p.log()) + te * ((1 - q).log() - (1 - p).log())
    return kl.sum() / n.sum()


def dense_split_kl(PA, PB, TA, TB, pool):
    """Between-group contrast against dense tracks. All [bins]. Each group's prediction is rescaled
    to its own observed total, so depth drops out and only WHERE the two groups differ is scored."""
    f = lambda v: v.reshape(-1, pool).sum(1)
    pe, pl, te, tl = f(PA), f(PB), f(TA), f(TB)
    n = te + tl
    if float(n.sum()) <= 0:
        return PA.sum() * 0
    mue = pe * (te.sum() / pe.sum().clamp_min(1e-8)); mul = pl * (tl.sum() / pl.sum().clamp_min(1e-8))
    p = (mul / (mul + mue).clamp_min(1e-12)).clamp(1e-6, 1 - 1e-6)
    q = (tl / n.clamp_min(1e-12)).clamp(1e-6, 1 - 1e-6)
    kl = tl * (q.log() - p.log()) + te * ((1 - q).log() - (1 - p).log())
    return kl.sum() / n.sum()


def poisson_nll(log_rate, target, offset=None):
    """Poisson NLL, summed over genes and averaged over cells. `offset` is the per-cell size factor.

    Written out rather than taken from torch so the size-factor offset enters the rate and not the
    target: scaling the target instead would change the variance model, which is the whole point of
    using Poisson.

    WHY SUM OVER GENES AND NOT MEAN. Measured on the corpus: a 196,608 bp window holds 2.06 annotated
    genes on average (median 1, max 30, and 24.5% of windows hold none), because 24,410 genes spread
    over 3.1 Gb is one or two per window -- that is genome structure, not a property of this corpus.
    A mean over all [cells x genes] elements therefore divides by G, which weights each gene of a
    30-gene window at 1/30 of the single gene of a 1-gene window: the windows carrying the most
    information would contribute the least per gene. Summing over genes makes a window's RNA term
    proportional to how many genes it actually constrains, and leaves `--w-rna` as the one knob that
    sets the trade-off against ATAC rather than having it diluted by window composition.

    The per-step RNA signal is thin either way -- a median of 1.6 nonzero counts per step against the
    ATAC head's 98,304 target bins -- which is why this normalisation is worth getting right rather
    than averaging over. The zeros are data too: for a Poisson rate, "not detected in this cell" is a
    constraint, which is what makes ~3.3e6 observations over a full run enough to fit a head whose
    parameters are shared across all genes.
    """
    if offset is not None:
        log_rate = log_rate + offset
    rate = log_rate.clamp(max=20).exp()
    return (rate - target * log_rate).sum(-1).mean()


def make_dataset_class(sb):
    """`ScoobyATACDataset` + the RNA head's three extra payloads, built once `sb` is importable.

    WHY A SUBCLASS AND NOT A WRAPPER. The parent draws the step's cells inside `__getitem__` and
    returns only their embeddings, not their indices -- but the RNA target has to be read for exactly
    those cells, and the exonic masks have to be built for exactly the window the parent used after
    `shift_aug`. Both are re-derived from `_get_rng(idx)`, which is `default_rng((seed, idx))` and
    carries no state, so mirroring the parent's draw order reproduces its choices exactly. That is
    asserted against the embeddings the parent returned rather than trusted, because a silent
    divergence here would pair each cell's ATAC with another cell's RNA.
    """
    import torch

    class ACESOv2Dataset(sb.ScoobyATACDataset):

        def __init__(self, *args, rna_h5ad, exon_masks, log_size, bank=None, **kw):
            super().__init__(*args, **kw)
            # Several z views of the SAME cells, one drawn per step. `self.embedding` stays the first
            # view so the parent's own bookkeeping and the re-derivation assertion below are unchanged.
            self.bank = bank
            self._rna_path = str(rna_h5ad)
            self._rna_rows_idx = None          # set by `set_row_map`: global cell -> rna.h5ad row
            self._em = exon_masks
            self._logsize = np.asarray(log_size, np.float32)
            self._h5 = None

        def set_row_map(self, rows):
            self._row2local = {int(g): i for i, g in enumerate(np.asarray(rows))}
            """`rows[i]` is the `rna.h5ad` row for global cell index i (reader order)."""
            self._rna_rows_idx = np.asarray(rows, np.int64)
            return self

        def __getstate__(self):
            s = super().__getstate__()
            s["_h5"] = None                    # h5py handles are not fork-safe: each worker opens its own
            return s

        def _rna(self, cells, cols):
            import h5py
            if self._h5 is None:
                self._h5 = h5py.File(self._rna_path, "r")
            out = np.zeros((len(cells), len(cols)), np.float32)
            if len(cols) == 0:
                return out
            # CSR row slices are column-sorted, so the window's columns are picked out by one
            # `searchsorted` per cell rather than a Python loop over that cell's ~2,000 nonzeros.
            srt = np.argsort(cols, kind="stable")
            cs = np.asarray(cols)[srt]
            ip = self._h5["X/indptr"]
            dat, ind = self._h5["X/data"], self._h5["X/indices"]
            for k, c in enumerate(cells):
                r = int(self._rna_rows_idx[c])
                s, e = int(ip[r]), int(ip[r + 1])
                if e <= s:
                    continue
                gi, gv = ind[s:e], dat[s:e]
                pos = np.searchsorted(cs, gi)
                ok = (pos < len(cs)) & (cs[np.clip(pos, 0, len(cs) - 1)] == gi)
                if ok.any():
                    out[k, srt[pos[ok]]] = gv[ok]
            return out

        def _getitem_paired(self, idx, cells, half):
            """The parent's body with the cell draw replaced. Everything else -- shift, flank,
            zero-padding off the contig, pseudobulk, reverse-complement -- is mirrored line for line
            so a within step differs from an ordinary step in the cells and in nothing else."""
            fasta = self._ensure_open()
            rng = self._get_rng(idx)
            row = self.intervals.iloc[idx]
            chrom, t_start, t_end = str(row["chrom"]), int(row["start"]), int(row["end"])
            shift = int(rng.integers(-self.shift_aug, self.shift_aug + 1)) if self.shift_aug else 0
            t_start += shift; t_end += shift
            i_start, i_end = t_start - self.flank, t_end + self.flank
            clen = fasta.get_reference_length(chrom)
            fs, fe = max(0, i_start), min(clen, i_end)
            seq = fasta.fetch(chrom, fs, fe) if fe > fs else ""
            onehot = np.zeros((self.input_length, 4), dtype=np.float32)
            if seq:
                from ..scooby._dataset import one_hot_dna as _oh
                onehot[fs - i_start: fs - i_start + len(seq)] = _oh(seq)
            groups = [np.append(self.neighbors[c].nonzero()[1], c) for c in cells]
            targets = self.reader.pseudobulk(chrom, t_start, t_end, groups,
                                             bin_size=self.bin_size, scale=self.scale)
            emb = self.embedding[cells]
            rc = False
            if self.rc_aug and bool(rng.integers(0, 2)):
                rc = True
                onehot = onehot[::-1, ::-1].copy(); targets = targets[:, ::-1].copy()
            return (torch.from_numpy(onehot), rc, torch.from_numpy(np.ascontiguousarray(targets.T)),
                    torch.from_numpy(np.ascontiguousarray(emb)),
                    torch.from_numpy(half))

        def set_triples(self, triples, frac, seed):
            """Stage B. `triples` are (organ, cell type, sample, early rows, late rows) built by
            aceso/make_within_triples.py, which already applied the three batch gates. A fraction
            `frac` of items draw their cells as two halves of ONE triple instead of at random; the
            rest are untouched, which is what keeps the shared profile intact."""
            self._triples = triples
            self._wfrac = frac
            self._wseed = seed
            return self

        def set_dense(self, track, n_cells, members, pairs, pair_frac, seed):
            """Stage B v2. `track[w, g]` = insertions of ALL cells of group g in window w (32 bp bins);
            `members[g]` = reader rows of that group; `pairs` = (gA, gB) to contrast."""
            self._dtrack, self._dn, self._dmem, self._dpairs = track, np.asarray(n_cells, np.float32), members, pairs
            self._pfrac, self._wseed = pair_frac, seed
            return self

        def _dense(self, idx, info, rc):
            out = np.zeros((2, self.target_length // self.bin_size), np.float32)
            if info is not None and getattr(self, "_dtrack", None) is not None:
                for k, g in enumerate(info):
                    out[k] = np.asarray(self._dtrack[idx, g], np.float32) / self._dn[g] * self.scale
                if rc:
                    out = out[:, ::-1].copy()
            return out

        def _paired(self, idx, n):
            r = np.random.default_rng((getattr(self, "_wseed", 0), 7_000_003, idx))
            u = r.random(); h = n // 2
            pf = getattr(self, "_pfrac", 0.0) if getattr(self, "_dpairs", None) else 0.0
            if u < pf:
                gA, gB = self._dpairs[int(r.integers(len(self._dpairs)))]
                if r.random() < 0.5:
                    gA, gB = gB, gA
                ee = r.choice(self._dmem[gA], h, replace=len(self._dmem[gA]) < h)
                ll = r.choice(self._dmem[gB], h, replace=len(self._dmem[gB]) < h)
                return np.concatenate([ee, ll]), np.array([0] * h + [1] * h, np.int64), (int(gA), int(gB))
            if not getattr(self, "_triples", None) or self._wfrac <= 0:
                return None
            if u >= pf + self._wfrac:
                return None
            t = self._triples[int(r.integers(len(self._triples)))]
            h = n // 2
            e, l = np.asarray(t["early"]), np.asarray(t["late"])
            if len(e) < h or len(l) < h:
                return None
            g2l = self._row2local                      # global cell row -> reader row
            ee = [g2l[q] for q in r.choice(e, h, replace=False) if q in g2l]
            ll = [g2l[q] for q in r.choice(l, h, replace=False) if q in g2l]
            if len(ee) < h or len(ll) < h:
                return None
            return np.array(ee + ll), np.array([0] * h + [1] * h, np.int64), None

        def __getitem__(self, idx):
            pair = self._paired(idx, self.cell_sample_size)
            info = None
            if pair is None:
                x, rc, y, emb = super().__getitem__(idx)
                half = torch.full((self.cell_sample_size,), -1, dtype=torch.long)
                paired = None
            else:
                paired, info = pair[0], pair[2]
                x, rc, y, emb, half = self._getitem_paired(idx, pair[0], pair[1])
            rng = self._get_rng(idx)
            row = self.intervals.iloc[idx]
            t_start, t_end = int(row["start"]), int(row["end"])
            if self.shift_aug:                 # mirror the parent's draw order, shift first
                sh = int(rng.integers(-self.shift_aug, self.shift_aug + 1))
                t_start += sh; t_end += sh
            if paired is not None:
                # a within step chose its own cells; nothing to re-derive, and the parent's draw
                # order was already consumed above so `rng` stays in step for the rest
                cells = paired
                if self.cell_weights is not None:
                    rng.choice(self.reader.n_cells, size=self.cell_sample_size, p=self.cell_weights)
                else:
                    rng.choice(self.reader.n_cells, size=self.cell_sample_size)
            else:
                if self.cell_weights is not None:
                    cells = rng.choice(self.reader.n_cells, size=self.cell_sample_size, p=self.cell_weights)
                else:
                    cells = rng.choice(self.reader.n_cells, size=self.cell_sample_size)
                assert np.array_equal(np.asarray(emb), self.embedding[cells]), \
                    "the re-derived cell draw diverged from the parent's; RNA would pair with the wrong cells"
            if self.bank is not None:
                # One view per step, not per cell: a step's 16 cells must share a conditioning regime,
                # because the read-out is a per-cell linear map of z and mixing widths inside a step
                # would ask it to be calibrated for two input distributions at once.
                k = int(rng.integers(len(self.bank)))
                emb = torch.from_numpy(np.ascontiguousarray(self.bank[k][cells]))
            cols, M = self._em.window(str(row["chrom"]), t_start, t_end)
            if bool(rc):
                M = np.ascontiguousarray(M[:, ::-1])   # the parent reversed the ATAC bins; match them
            cnt = self._rna(cells, cols)
            return (x, rc, y, emb, torch.from_numpy(M), torch.from_numpy(cnt),
                    torch.from_numpy(self._logsize[cells]), half,
                    torch.from_numpy(self._dense(idx, info, bool(rc))),
                    2 if info is not None else (1 if paired is not None else 0))

    return ACESOv2Dataset


def train_v2_decoder(config: V2DecoderConfig = V2DecoderConfig(), *, log=None):
    """Train the v2 decoder. `config` holds what the original script took on the command line."""
    import pandas as pd
    import torch

    from ._sites import require_sites, site_path

    c = config
    # what the original stored as `vars(args)`; built before anything is resolved, so a checkpoint
    # written from a config records the config and not the machine it ran on
    ARGS = {k: getattr(c, k) for k in _ARG_FIELDS}

    # The anchor every other path hangs off, resolved the way `_sites` resolves the rest: the first
    # candidate that EXISTS ($EPIONE_ROOT, then $EPIONE_ACESO_SITES, then the cwd).
    R = Path(c.root or site_path("root"))

    if not c.out:
        sys.exit("stage 2 writes every checkpoint under `out`; it has no default (--out on the CLI)")
    OUT = Path(c.out) if Path(c.out).is_absolute() else R / c.out
    STORE = Path(c.store) if Path(c.store).is_absolute() else R / c.store
    OUT.mkdir(parents=True, exist_ok=True)
    HEADS = set(c.heads.split(","))

    LOCAL = int(os.environ.get("LOCAL_RANK", 0))
    RANK = int(os.environ.get("RANK", 0))
    WORLD = int(os.environ.get("WORLD_SIZE", 1))
    DDP = WORLD > 1
    if DDP:
        # VALIDATION RUNS ON RANK 0 WHILE THE OTHER THREE WAIT AT A BARRIER, so the watchdog has to
        # outlast a whole validation pass, not a single collective. NCCL's 10-minute default does not:
        # `val()` walks every held-out window through the trunk one window at a time, and a barrier that
        # expires takes the job down with a timeout that looks like a hang rather than like a slow
        # evaluation. This is the same failure that killed two stage-1 runs.
        from datetime import timedelta
        torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=90))
        torch.cuda.set_device(LOCAL)
    dev = torch.device(f"cuda:{LOCAL}" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(c.seed + 1000 * RANK)
    torch.manual_seed(c.seed)

    if log is None:
        def log(*x):
            if RANK == 0:
                print(*x, flush=True)

    log(json.dumps(ARGS, indent=1))
    # resolved and printed before the allocation does any work (see `._sites`)
    if RANK == 0:
        require_sites("scooby_src", "hg38_fa", "folds_bed", "exon_masks", "alphagenome_weights")
    O = pd.read_csv(STORE / "cells.tsv", sep="\t")
    log(f"store {STORE}: {len(O):,} cells, splits {O.split.value_counts().to_dict()}")

    BANK = None
    if c.z_bank:
        paths = [q.strip() for q in c.z_bank.split(",") if q.strip()]
        BANK = [np.load(q if Path(q).is_absolute() else R / q,
                        mmap_mode="r" if os.environ.get("DENSE_BUILD") else None) for q in paths]
        for q, B in zip(paths, BANK):
            fin = np.isfinite(B[:, 0])
            log(f"  z bank: {Path(q).name:34s} {B.shape}  {fin.sum():,} finite ({fin.mean()*100:.1f}%)")
        if len({B.shape for B in BANK}) != 1:
            sys.exit(f"z bank entries disagree in shape: {[B.shape for B in BANK]}")
        # A cell is usable only if EVERY view gives it a z. A panel view is NaN wherever the cell has
        # no counts among the panel's genes, and those NaN patterns differ between views -- taking the
        # union per step would silently feed NaN conditioning on the steps that drew that view.
        Z = BANK[0]
    elif c.z:
        Z = np.load(c.z if Path(c.z).is_absolute() else R / c.z)
    elif (STORE / "z_epiagent.npy").exists() and not c.stage1:
        log("!! using z_epiagent.npy as the conditioning input.\n"
            "   That is the TARGET of stage 1, not its output -- training stage 2 on it measures the\n"
            "   decoder with a perfect encoder, which is a diagnostic and NOT the deployable model.\n"
            "   Pass --stage1 <ckpt> for the real chain.")
        Z = np.load(STORE / "z_epiagent.npy")
    else:
        sys.exit("stage 2 conditions on z. Produce it from a stage-1 checkpoint first --\n"
                 "  python aceso/stage1_infer.py --ckpt <stage1 ckpt> --view wide\n"
                 "then pass --z data/train_v2/z_stage1_wide.npy. (Running the encoder inside this\n"
                 "loop is deliberately not done: see stage1_infer.py's docstring.)")
    assert len(Z) == len(O), (Z.shape, len(O))
    log(f"  z: {Z.shape}, {np.isfinite(Z[:, 0]).sum():,} finite rows")

    # ------------------------------------------------------------------ data path
    # Reused from `epione.models.scooby`, which is the verified upstream path v1 trained on:
    # `FragmentInsertionReader` turns a tabix fragments file into per-cell insertion counts in a
    # window, and `ScoobyATACDataset` pairs that with one-hot DNA. Nothing about windows, DNA or
    # Tn5 counting is re-derived here.
    import scipy.sparse as sp
    # upstream scooby itself, for `poisson_multinomial_torch`: the ATAC loss is taken from the
    # reference implementation rather than rewritten, so the v1 and v2 ATAC numbers stay comparable
    sys.path.insert(0, c.scooby_src or site_path("scooby_src"))
    from .. import scooby as sb

    FA = str(c.hg38_fa or site_path("hg38_fa"))
    FOLDS = Path(c.folds_bed or site_path("folds_bed"))
    for q in (FA, FOLDS):
        if not Path(q).exists():
            sys.exit(f"missing {q}")

    # Many samples' fragment files behind ONE global cell index, cells stacked in sample order. The
    # factory is upstream's (`epione.models.aceso._decoder.MultiFragmentReader`) -- v1 already needed
    # it for two datasets and the only v2 difference is 173 readers instead of 2. Stacking keeps a
    # single global row index so `z` and the insertion matrix stay row-aligned, which is the one
    # failure that trains happily and yields a worthless model.
    from ._decoder import MultiFragmentReader

    def build_reader(O, store):
        """One reader per sample, in the order `O` lists them; cell ids are "<sample>#<barcode>"."""
        readers, used, names = [], [], []
        for smp, d in O.groupby("sample", sort=False):
            f = store / "fragments" / f"{smp}.fragments.tsv.gz"
            if not Path(str(f) + ".tbi").exists():
                continue                       # no fragments for this sample -> it cannot supply ATAC
            readers.append(sb.FragmentInsertionReader(str(f), [i.split("#", 1)[1] for i in d.cell_id]))
            names += list(d.cell_id)           # global index keeps the "<sample>#<barcode>" ids
            used.append(smp)
        return MultiFragmentReader(readers, names=names), used

    # Cells must be ordered BY SAMPLE, because `MultiFragmentReader` stacks one reader per sample and
    # its global row i must be the same cell as `Ztr` row i. Positions in the original frame are what
    # index `Z`, so the sort is carried out on positions and applied to both together -- never to one
    # of them alone.
    _fin = np.isfinite(Z[:, 0])
    if BANK is not None:
        for B in BANK[1:]:
            _fin &= np.isfinite(B[:, 0])
        log(f"  cells with a z in ALL {len(BANK)} views: {_fin.sum():,} "
            f"({_fin.mean()*100:.1f}% of the store)")
    keep = np.flatnonzero(_fin & O["split"].eq("train").to_numpy())
    keep = keep[np.argsort(O["sample"].to_numpy()[keep], kind="stable")]
    reader, used = build_reader(O.loc[keep].reset_index(drop=True), STORE)
    keep = keep[np.isin(O["sample"].to_numpy()[keep], used)]     # drop samples with no fragments
    Otr, Ztr = O.loc[keep].reset_index(drop=True), Z[keep]
    BANKtr = [B[keep] for B in BANK] if (BANK is not None and not os.environ.get("DENSE_BUILD")) else None

    # ---- named within-type axes, appended to every conditioning vector ------------------------
    # Appended to z rather than threaded through the dataset: `cell_state_to_conv` and
    # `cell_state_to_gene` read one cell-state vector, so widening that vector IS the conditioning,
    # and every z view in the bank gets the same axes without the dataset learning a new argument.
    #
    # TWO CHANNELS PER AXIS, NOT ONE. The design says 512 + k; this is 512 + 2k, because an axis is
    # tissue-scoped and `score = 0` has two meanings -- "defined here, and sitting at the tissue
    # mean" and "not defined here at all". A single channel collapses them, which would tell the
    # model that a liver cell is an average-chamber heart cell. The mask rides alongside so
    # "undefined" is a state the model can see.
    AXN = []
    if c.axes and c.axes.lower() != "none":
        _ap = Path(c.axes) if Path(c.axes).is_absolute() else R / c.axes
        if not _ap.exists():
            sys.exit(f"missing {_ap}. Run aceso/make_axes.py, or pass --axes none to condition on z "
                     f"alone -- which is the configuration the within-type measurement already "
                     f"refuted, so it should be a deliberate choice.")
        _az = np.load(_ap, allow_pickle=True)
        AXN = [str(q) for q in _az["names"]]
        _sc, _mk = _az["score"].astype(np.float32), _az["mask"].astype(np.float32)
        if len(_sc) != len(O):
            sys.exit(f"axes has {len(_sc):,} rows, cells.tsv has {len(O):,}; they must be the same "
                     f"cells in the same order")
        _sc, _mk = _sc[keep], _mk[keep]
        _aug = np.hstack([_sc * _mk, _mk]).astype(np.float32)
        Ztr = np.hstack([Ztr, _aug]).astype(np.float32)
        if BANKtr is not None:
            BANKtr = [np.hstack([B, _aug]).astype(np.float32) for B in BANKtr]
        for i, nm in enumerate(AXN):
            d = _mk[:, i] > 0
            log(f"  axis {nm:9s} defined in {int(d.sum()):>9,} of {len(_mk):,} training cells "
                f"({d.mean()*100:5.1f}%), score {_sc[d, i].min():+.2f} .. {_sc[d, i].max():+.2f}, "
                f"sd {_sc[d, i].std():.3f}")
    Z_DIM = c.z_dim + 2 * len(AXN)
    log(f"  cell state: {c.z_dim} z + {2 * len(AXN)} axis channels "
        f"({len(AXN)} axes x score,mask) = {Z_DIM}")
    assert reader.barcodes == list(Otr.cell_id), "reader order does not match the z rows"
    log(f"  reader: {reader.n_cells:,} cells over {len(used)} samples with fragments")

    folds = sb.read_fold_bed(str(FOLDS))
    fcol = [q for q in folds.columns if q not in ("chrom", "start", "end")][0]
    train_iv = folds[~folds[fcol].isin(["fold3", "fold4"])].reset_index(drop=True)
    val_iv = folds[folds[fcol] == "fold4"].sample(n=min(c.val_windows, (folds[fcol] == "fold4").sum()),
                                                  random_state=0).reset_index(drop=True)
    log(f"  windows: train {len(train_iv):,}  val {len(val_iv)}  "
        f"(fold3 held out entirely for test)")

    # ------------------------------------------------------------------ cell-type groups (stage B v2)
    # A group is every training cell of one (corpus, organ, cell type). Its summed insertions are the
    # dense target: thousands of cells behind each profile instead of ~1 insertion per cell.
    _lab = Otr.cell_type.astype(str).ne("unlabelled").to_numpy()
    _key = (Otr.corpus.astype(str) + "|" + Otr.organ.astype(str) + "|" + Otr.cell_type.astype(str)).to_numpy()
    _u, _inv, _cnt = np.unique(_key[_lab], return_inverse=True, return_counts=True)
    GROUPS = [u for u, n in zip(_u, _cnt) if n >= int(os.environ.get("ACESO_MIN_GROUP", 300))]
    _gid = {g: i for i, g in enumerate(GROUPS)}
    GMEM = [np.flatnonzero(_lab & (_key == g)) for g in GROUPS]
    GN = np.array([len(m) for m in GMEM])
    log(f"  groups: {len(GROUPS)} (corpus x organ x cell type, >= 300 training cells); "
        f"{int(GN.sum()):,} cells; median {int(np.median(GN)):,} per group")
    if os.environ.get("DENSE_BUILD"):
        DD = Path(c.dense_dir); DD.mkdir(parents=True, exist_ok=True)
        si, sn = [int(v) for v in os.environ["DENSE_BUILD"].split("/")]
        Gm = sp.csr_matrix((np.ones(int(GN.sum()), np.float32),
                            (np.concatenate([np.full(len(m), i) for i, m in enumerate(GMEM)]), np.concatenate(GMEM))),
                           shape=(len(GROUPS), reader.n_cells))
        NBIN = sb.TARGET_LENGTH // c.bin
        if si == 0:
            pd.DataFrame(dict(group=GROUPS, n_cells=GN)).to_csv(DD / "groups.tsv", sep="\t", index=False)
        for nm, iv in (("val", val_iv), ("train", train_iv)):
            rows = np.arange(len(iv))[si::sn]
            out = np.zeros((len(rows), len(GROUPS), NBIN), np.uint16)
            t0 = time.time()
            for k, w in enumerate(rows):
                r_ = iv.iloc[w]
                M_ = reader.window_insertions(str(r_["chrom"]), int(r_["start"]), int(r_["end"]), dtype=np.float32)
                S_ = np.asarray((Gm @ M_).todense())[:, :NBIN * c.bin].reshape(len(GROUPS), NBIN, c.bin).sum(-1)
                out[k] = np.clip(S_, 0, 65535).astype(np.uint16)
                if k % 50 == 0:
                    print(f"  [{nm} shard {si}/{sn}] {k}/{len(rows)}  {(time.time()-t0)/(k+1):.2f} s/window", flush=True)
            np.save(DD / f"{nm}.part{si}of{sn}.npy", out); np.save(DD / f"{nm}.rows{si}of{sn}.npy", rows)
        print(f"DENSE shard {si}/{sn} done", flush=True)
        sys.exit(0)

    # ------------------------------------------------------------------ RNA targets
    # `n_umi` is already a column of cells.tsv, so the size factor costs no pass over rna.h5ad. It
    # enters the Poisson rate as an offset (see `poisson_nll`): without it the head would have to
    # spend capacity predicting library depth, which is a property of the droplet, not of the locus.
    import h5py
    with h5py.File(STORE / "rna.h5ad") as f:
        genes = [x.decode() if isinstance(x, bytes) else str(x) for x in f["var/_index"][:]]
    from ._exon import ExonMasks
    em = ExonMasks(gene_list=genes, bin_size=c.bin)
    n_cov = int(em.keep.sum())
    log(f"  exon masks: {len(genes):,} store genes, {n_cov:,} with a GENCODE locus "
        f"({n_cov / len(genes) * 100:.1f}%)")

    umi = O["n_umi"].to_numpy(np.float64)[keep]
    log_size = np.log(np.clip(umi, 1, None) / np.median(umi[umi > 0])).astype(np.float32)

    no_nb = sp.csr_matrix((reader.n_cells, reader.n_cells), dtype=np.float32)
    DS = make_dataset_class(sb)
    common = dict(neighbors=no_nb, cell_sample_size=c.cells_per_step, input_length=c.window,
                  target_length=sb.TARGET_LENGTH, bin_size=c.bin, scale=sb.ATAC_SCALE,
                  rna_h5ad=STORE / "rna.h5ad", exon_masks=em, log_size=log_size, bank=BANKtr)
    ds = DS(reader, FA, train_iv, Ztr, shift_aug=3, seed=c.seed, **common).set_row_map(keep)
    if c.within_weight > 0:
        tp = Path(c.triples) if Path(c.triples).is_absolute() else R / c.triples
        TR = json.load(open(tp))
        tri = [t for t in TR["triples"] if t["axis"] != "pc1"]     # PC1 of z is circular: it rewards amplifying z itself
        import collections as _c
        byax = _c.Counter(t["axis"] for t in tri)
        byorg = _c.Counter(t["organ"] for t in tri)
        log(f"  stage B: {len(tri)} within-type triples from {tp.name}; axes {dict(byax)}")
        log(f"           organs {dict(byorg)}")
        log(f"           pcw contributed {byax.get('pcw', 0)} -- 0 means it did not survive the "
            f"permutation gate on this corpus and was dropped rather than trained on")
        ds = ds.set_triples(tri, c.within_frac, c.seed)
    else:
        log("  stage A only (--within-weight 0)")
    vds = DS(reader, FA, val_iv, Ztr, shift_aug=0, seed=c.seed + 7, **common).set_row_map(keep)
    DTR = DVA = None; VG = []
    if c.dense_dir:
        DD = Path(c.dense_dir)
        gt = pd.read_csv(DD / "groups.tsv", sep="\t")
        assert list(gt.group) == GROUPS and list(gt.n_cells) == list(GN), \
            "dense tracks were built for a different group table"
        DTR = np.load(DD / "train.npy", mmap_mode="r"); DVA = np.load(DD / "val.npy", mmap_mode="r")
        assert DTR.shape[:2] == (len(train_iv), len(GROUPS)) and DVA.shape[:2] == (len(val_iv), len(GROUPS))
        parts = [g.split("|") for g in GROUPS]
        PAIRS_ORG = [(i, j) for i in range(len(GROUPS)) for j in range(i + 1, len(GROUPS))
                     if parts[i][0] == parts[j][0] and parts[i][1] == parts[j][1]]        # same organ, other type
        PAIRS_TYP = [(i, j) for i in range(len(GROUPS)) for j in range(i + 1, len(GROUPS))
                     if parts[i][0] == parts[j][0] and parts[i][2] == parts[j][2]]        # same type, other organ
        # drawn half and half, so the handful of cross-organ pairs is not swamped by within-organ ones
        k_ = max(len(PAIRS_ORG), len(PAIRS_TYP))
        PAIRS = ((PAIRS_ORG * (k_ // max(len(PAIRS_ORG), 1) + 1))[:k_]
                 + (PAIRS_TYP * (k_ // max(len(PAIRS_TYP), 1) + 1))[:k_])
        log(f"  stage B v2: {len(PAIRS_ORG)} same-organ/other-type pairs, {len(PAIRS_TYP)} same-type/other-organ pairs; "
            f"pair steps {c.pair_frac:.2f}, axis steps {c.within_frac if c.within_weight > 0 else 0:.2f}")
        ds = ds.set_dense(DTR, GN, GMEM, PAIRS, c.pair_frac, c.seed)
        # validation groups: fixed, half fetal half adult, the largest of each
        _o = np.argsort(-GN)
        VG = [int(i) for i in _o if parts[i][0] == "hdma"][:10] + [int(i) for i in _o if parts[i][0] == "encode"][:10]
        if not VG:
            # a store from another corpus (e.g. data/train_efe): its ten largest groups; they are
            # reported under the first slot, `val_type_spec_fetal`
            VG = [int(i) for i in _o][:10]
        _vr = np.random.default_rng(c.seed + 99)
        VCE = torch.from_numpy(np.concatenate(
            [np.asarray(Ztr)[np.sort(_vr.choice(GMEM[g], 16, replace=False))] for g in VG])).float()
        log(f"  validation groups: {[GROUPS[g] for g in VG]}")
    log(f"  dataset ready: {len(ds):,} train windows x {c.cells_per_step} cells per step")

    # ------------------------------------------------------------------ model
    model = make_model(Z_DIM, dev, c.organism, ckpt=c.ag_weights, train=True,
                       rna_head="rna" in HEADS, freeze_trunk=c.freeze_trunk)
    model.rna_trunk_scale = c.rna_trunk_scale
    LAST_DENSE = {}
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"  model: {n_all / 1e6:.1f}M params, {n_tr / 1e6:.2f}M trainable "
        f"({n_tr / n_all * 100:.2f}%); trunk {'frozen' if c.freeze_trunk else 'TRAINED'}")

    net = model
    if DDP:
        # NO `static_graph`, BECAUSE IT CANNOT BE COMBINED WITH `no_sync()`, WHICH THIS LOOP NEEDS.
        # Measured in aceso/tests/test_ddp_accum.py on this torch (2 ranks, accum 4, gloo), against a
        # single-process reference gradient:
        #     static_graph=True  no_sync=True   CRASHED  expect_autograd_hooks_ INTERNAL ASSERT
        #                                                FAILED (reducer.cpp:1603)
        #     static_graph=True  no_sync=False  max|diff| 1.19e-07  OK
        #     static_graph=False no_sync=True   max|diff| 5.96e-08  OK
        #     static_graph=False no_sync=False  max|diff| 1.19e-07  OK
        # static_graph installs its delayed all-reduce on the FIRST forward and then sets a flag so it
        # never installs it again -- and with accum 8 that first forward is inside `no_sync()`.
        # The reason originally given for static_graph does not hold either: the AlphaGenome trunk
        # checkpoints with `use_reentrant=False`, which is the mode that works with DDP *without*
        # static_graph. Dropping it keeps the 7-in-8 saved all-reduces, which is the larger effect.
        # `find_unused_parameters` stays off: `forward_genes` keeps `cell_state_to_gene` in the graph
        # even on a window with no annotated gene, and frozen parameters are invisible to DDP.
        net = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL], broadcast_buffers=False, gradient_as_bucket_view=True)
        log(f"  DDP: world {WORLD}, grad accumulation via no_sync, buckets as views")

    from ._decoder import DecoderConfig, param_groups, make_optimizer, set_stabilise, trainable_state
    from scooby.utils.utils import poisson_multinomial_torch

    # LEARNING RATE AND THE WORLD SIZE. The gradient is averaged across ranks, so `accum 8` on 4 ranks
    # is an effective batch of 32 windows where v1 trained on 8. v1's lr (2e-4, cell-state 4e-4) was
    # tuned for 8. Keeping it would quietly shrink the effective step by the batch ratio, so it is
    # scaled by sqrt(WORLD) -- sqrt rather than linear because Adam already normalises by the gradient
    # magnitude, which is where linear scaling's assumption (SGD) breaks. `--lr` set explicitly on the
    # command line is taken as final and not touched.
    lr_scale = 1.0 if c.lr_given else WORLD ** 0.5
    dcfg = DecoderConfig(max_steps=c.steps, lr=c.lr * lr_scale, cellstate_lr=c.lr * 2 * lr_scale,
                         seed=c.seed, **({"warmup": c.warmup} if c.warmup is not None else {}))
    log(f"  optimizer: lr {dcfg.lr:.3g}, cell-state lr {dcfg.cellstate_lr:.3g} "
        f"(world {WORLD}, scale {lr_scale:.3g}{'; --lr given, not scaled' if c.lr_given else ''}), "
        f"warmup {dcfg.warmup} steps, effective batch {dcfg.accum * WORLD} windows")
    opt, sched = make_optimizer(model, param_groups(model, dcfg.lr, dcfg.cellstate_lr, dcfg.weight_decay),
                                dcfg, c.steps)

    step, t_start = 0, time.time()
    if c.init_from and not (OUT / "last.pt").exists():
        _s = torch.load(c.init_from, map_location="cpu", weights_only=False)
        # More named axes than the checkpoint was trained with: its state input is [z, k scores, k masks]; the
        # new layout is [z, K scores, K masks]. Old columns are moved to their new positions and the columns of
        # the added axes start at zero, so the warm-started model computes exactly what the checkpoint did.
        _msd = model.state_dict(); _ko = (int(_s.get("z_dim", Z_DIM)) - c.z_dim) // 2; _kn = len(AXN)
        if _kn > _ko:
            for _k, _v in list(_s["model"].items()):
                if _k in _msd and _v.ndim == 2 and _v.shape[1] == c.z_dim + 2 * _ko and _msd[_k].shape[1] == Z_DIM:
                    _w = torch.zeros(_v.shape[0], Z_DIM, dtype=_v.dtype); _w[:, :c.z_dim + _ko] = _v[:, :c.z_dim + _ko]
                    _w[:, c.z_dim + _kn:c.z_dim + _kn + _ko] = _v[:, c.z_dim + _ko:]; _s["model"][_k] = _w
                    log(f"  {_k}: state input widened {tuple(_v.shape)} -> {tuple(_w.shape)} ({_kn - _ko} new axis)")
        model.load_state_dict(_s["model"], strict=False)
        log(f"  warm start from {Path(c.init_from).name} (step {_s.get('step')}), "
            f"weights only; step/opt/sched reset")
    ck = OUT / "last.pt"
    if c.resume or ck.exists():
        src = Path(c.resume) if c.resume else ck
        s = torch.load(src, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"], strict=False)
        opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"]); step = s["step"]
        log(f"  resumed from {src.name} at step {step:,}")

    # Each rank walks a different shuffle of the same windows. Windows are the unit of work and there
    # are 10,116 of them, so sharding by rank (DistributedSampler) would cap an epoch at 2,529 steps
    # per rank; re-shuffling instead keeps every rank drawing from all of them, which is what the
    # cross-rank gradient average wants.
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=c.workers,
                                     worker_init_fn=sb.worker_init, persistent_workers=c.workers > 0,
                                     prefetch_factor=4 if c.workers else None,
                                     generator=torch.Generator().manual_seed(c.seed + 1000 * RANK))
    vdl = torch.utils.data.DataLoader(vds, batch_size=1, shuffle=False, num_workers=2,
                                      worker_init_fn=sb.worker_init)

    def val():
        """Pooled Pearson on log1p for ATAC, and on log1p rate for RNA. Rank 0 only."""
        from scipy import stats
        model.eval()
        PA, TA, PR, TR = [], [], [], []
        GP, GT = [], []
        with torch.no_grad():
            for wi, (x, rc, y, ce, M, cnt, ls, _h, _d, _m) in enumerate(vdl):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    at, rn = model(x.permute(0, 2, 1).to(dev), ce.to(dev), M.to(dev))
                    if VG:
                        ag = model(x.permute(0, 2, 1).to(dev), VCE[None].to(dev), None)[0]
                if VG:
                    GP.append(ag.float()[0].reshape(-1, len(VG), 16).mean(-1).cpu().numpy())
                    GT.append((np.asarray(DVA[wi][VG], np.float32) / GN[VG][:, None]).T)
                PA.append(at.float().squeeze(0).cpu().numpy()); TA.append(y.squeeze(0).numpy())
                if M.shape[1]:
                    lam = (rn.float() + ls.to(dev)[:, :, None]).clamp(max=20).exp()
                    PR.append(lam.squeeze(0).cpu().numpy().ravel())
                    TR.append(cnt.squeeze(0).numpy().ravel())
        PA, TA = np.concatenate(PA), np.concatenate(TA)
        # SPECIFICITY: both centred per bin across cells, so a prediction that ignores the cell
        # state scores 0 however good its pooled r. The zero line for pooled r on this corpus is
        # 0.14448 -- an oracle per-bin mean across cells, measured in
        # paper/fig5/results/constitutive/free_profile.json -- so pooled r below that is worse than
        # carrying no cell information at all.
        cp = np.log1p(PA) - np.log1p(PA).mean(1, keepdims=True)
        ct = np.log1p(TA) - np.log1p(TA).mean(1, keepdims=True)
        r_spec = (float(stats.pearsonr(cp.ravel(), ct.ravel())[0]) if cp.std() > 1e-12 else 0.0)
        tot = np.log1p(TA).mean(1)
        pk = tot >= np.quantile(tot, 0.99)
        r_spec_pk = (float(stats.pearsonr(cp[pk].ravel(), ct[pk].ravel())[0])
                     if cp[pk].std() > 1e-12 else 0.0)
        r_atac = float(stats.pearsonr(np.log1p(PA).ravel(), np.log1p(TA).ravel())[0])
        r_cell = float(np.nanmean([stats.pearsonr(np.log1p(PA[:, j]), np.log1p(TA[:, j]))[0]
                                   for j in range(PA.shape[1])]))
        r_rna = (float(stats.pearsonr(np.log1p(np.concatenate(PR)), np.log1p(np.concatenate(TR)))[0])
                 if PR else float("nan"))
        model.train()
        nonlocal LAST_DENSE
        LAST_DENSE = {}
        if GP:
            # between-cell-type specificity against the dense tracks, fetal and adult groups separately:
            # each group rescaled to equal depth, peaks = top 1% bins, centred per bin across groups
            # and per group across bins (a state-blind prediction scores 0)
            P_, T_ = np.concatenate(GP), np.concatenate(GT)
            for nm_, sl in (("fetal", slice(0, 10)), ("adult", slice(10, 20))):
                p_, t_ = P_[:, sl], T_[:, sl]
                if p_.shape[1] < 3:
                    continue
                p_ = p_ / (p_.sum(0) / p_.sum(0).mean()); t_ = t_ / (t_.sum(0) / t_.sum(0).mean())
                lp_, lt_ = np.log1p(p_), np.log1p(t_ * sb.ATAC_SCALE)
                pk_ = lt_.mean(1) >= np.quantile(lt_.mean(1), 0.99)
                c_ = lambda v: ((v[pk_] - v[pk_].mean(1, keepdims=True))
                                - (v[pk_] - v[pk_].mean(1, keepdims=True)).mean(0, keepdims=True))
                LAST_DENSE[f"val_type_spec_{nm_}"] = round(float(stats.pearsonr(c_(lp_).ravel(), c_(lt_).ravel())[0]), 5)
                LAST_DENSE[f"val_type_amp_{nm_}"] = round(float(c_(lp_).std() / max(c_(lt_).std(), 1e-12)), 4)
        return r_atac, r_cell, r_rna, r_spec, r_spec_pk

    log(f"\n=== training: {c.steps:,} steps, accum {dcfg.accum}, "
        f"effective batch {dcfg.accum * WORLD} windows ===")
    flog = open(OUT / "train.log", "a") if RANK == 0 else None
    TB = None
    if RANK == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            TB = SummaryWriter(str(OUT / "tb"))
            log(f"  tensorboard -> {OUT}/tb")
        except Exception as e:                                             # noqa: BLE001
            log(f"  tensorboard unavailable ({type(e).__name__})")
    micro, t_last = 0, time.time()
    best_spec = [-1.0]
    acc = dict(atac=0.0, rna=0.0, within=0.0, paired=0, n=0, raw_dense=0.0, raw_con=0.0, n2=0)
    while step < c.steps:
        for x, rc, y, ce, M, cnt, ls, half, dense, mode in dl:
            set_stabilise(model, step < dcfg.warmup)
            final = (micro + 1) % dcfg.accum == 0
            # Skipping the all-reduce on the first accum-1 micro-steps is not an optimisation of the
            # same computation -- it IS the computation: with accum 8 it removes 7 of every 8
            # gradient syncs, which on 4 ranks is where most of the DDP overhead would otherwise be.
            ctx = torch.enable_grad() if (final or not DDP) else net.no_sync()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    at, rn = net(x.permute(0, 2, 1).to(dev, non_blocking=True),
                                 ce.to(dev, non_blocking=True), M.to(dev, non_blocking=True))
                l_atac = poisson_multinomial_torch(at.float(), y.to(dev).float(), total_weight=0.2)
                l_rna = (poisson_nll(rn.float(), cnt.to(dev).float(), offset=ls.to(dev)[:, :, None])
                         if ("rna" in HEADS and M.shape[1]) else rn.float().sum() * 0.0)
                l_within = at.float().sum() * 0.0
                l_dense = at.float().sum() * 0.0; l_con = at.float().sum() * 0.0
                if int(mode) == 2:
                    h_ = half[0].to(dev); A_ = at.float()[0]; D_ = dense[0].to(dev).float()
                    PA_, PB_ = A_[:, h_ == 0].mean(1), A_[:, h_ == 1].mean(1)
                    ld = poisson_multinomial_torch(torch.stack([PA_, PB_], 1)[None],
                                                   torch.stack([D_[0], D_[1]], 1)[None], total_weight=0.2)
                    lk = dense_split_kl(PA_, PB_, D_[0], D_[1], c.within_pool)
                    l_dense = c.w_dense * (l_atac.detach() / ld.detach().clamp_min(1e-8)) * ld
                    l_con = c.w_contrast * (l_atac.detach() / lk.detach().clamp_min(1e-6)) * lk
                    acc["raw_dense"] += float(ld); acc["raw_con"] += float(lk); acc["n2"] += 1
                if c.within_weight > 0 and int(mode) == 1:
                    # `at` is already [B, bins, cells], which is v1's `out [1, bins, cells]`;
                    # `y` has the same shape because the ATAC loss consumes them together. No
                    # permute -- an earlier version transposed both and indexed 6,144 bins with a
                    # 32-long cell mask.
                    lw = within_kl(at.float(), y.to(dev).float(),
                                   half[0] if half.dim() == 2 else half, c.within_pool)
                    # scaled against l_atac, NOT the total: v2 added an RNA term v1 never had, and
                    # dividing by the total would shrink the contrast by whatever the RNA head costs
                    l_within = c.within_weight * (l_atac.detach() / lw.detach().clamp_min(1e-6)) * lw
                loss = (l_atac + c.w_rna * l_rna + l_within + l_dense + l_con) / dcfg.accum
                loss.backward()
            acc["atac"] += float(l_atac); acc["rna"] += float(l_rna)
            acc["within"] += float(l_within); acc["n"] += 1
            acc["paired"] += int(bool((half >= 0).all()))
            micro += 1
            if not final:
                continue
            # two clips, not one: the gene read-out's gradient is ~1e4 times the accessibility terms',
            # and a shared norm would rescale the ATAC update by how many genes the window holds
            torch.nn.utils.clip_grad_norm_([p for n_, p in model.named_parameters()
                                            if p.requires_grad and not n_.startswith("cell_state_to_gene")], 1.0)
            torch.nn.utils.clip_grad_norm_([p for n_, p in model.named_parameters()
                                            if p.requires_grad and n_.startswith("cell_state_to_gene")], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1

            if step % 20 == 0 and RANK == 0:
                m = max(acc["n"], 1)
                rec = dict(step=step, atac=float(f'{acc["atac"]/m:.3e}'),
                           rna=round(acc["rna"] / m, 5),
                           within=float(f'{acc["within"]/m:.3e}'),
                           within_frac_seen=round(acc["paired"] / m, 3),
                           pair_frac_seen=round(acc["n2"] / m, 3),
                           dense_raw=float(f'{acc["raw_dense"]/max(acc["n2"],1):.3e}'),
                           contrast_kl=round(acc["raw_con"] / max(acc["n2"], 1), 5),
                           lr=sched.get_last_lr()[0], s_per_step=round((time.time() - t_last) / 20, 2))
                log(json.dumps(rec)); flog.write(json.dumps(rec) + "\n"); flog.flush()
                if TB is not None:
                    for k, v in rec.items():
                        if k != "step" and isinstance(v, (int, float)):
                            TB.add_scalar(f"train/{k}", v, step)
                    TB.flush()
                acc = dict(atac=0.0, rna=0.0, within=0.0, paired=0, n=0, raw_dense=0.0, raw_con=0.0, n2=0)
                t_last = time.time()
            if step % c.ckpt_every == 0 or step >= c.steps:
                if RANK == 0:
                    r_atac, r_cell, r_rna, r_spec, r_spec_pk = val()
                    rec = dict(step=step, val_r_atac=round(r_atac, 5), val_r_atac_per_cell=round(r_cell, 5),
                               val_r_rna=round(r_rna, 5), val_spec_r=round(r_spec, 5),
                               val_spec_r_peaks=round(r_spec_pk, 5), **LAST_DENSE)
                    if LAST_DENSE:
                        # best.pt follows what stage B is for: cell-type specificity, mean of fetal and adult
                        r_spec_pk = float(np.mean([v for k_, v in LAST_DENSE.items() if "spec" in k_]))
                        rec["select"] = round(r_spec_pk, 5)
                    log(json.dumps(rec)); flog.write(json.dumps(rec) + "\n"); flog.flush()
                    if c.snap_every and step % c.snap_every == 0:
                        torch.save({"model": trainable_state(model), "step": step, "args": ARGS,
                                    "z_dim": Z_DIM, "axes": AXN, "val": rec}, OUT / f"step{step}.pt")
                    # BEST IS SELECTED ON val_spec_r_peaks, not on pooled r. Pooled r is 87%
                    # determined by having any non-zero conditioning vector at all (ablation:
                    # zero 0.0685, globalmean 0.2064, real 0.2384), so selecting on it would pick a
                    # checkpoint for a property stage B does not change.
                    tmp = OUT / "last.pt.tmp"
                    # `z_dim` and `axes` are recorded because they are no longer derivable from
                    # args: the conditioning vector is 512 + 2k wide, and a loader that rebuilds the
                    # model at 512 fails with a shape error at best and loads a misaligned
                    # hypernetwork at worst. The figures rebuild this model months from now.
                    torch.save({"model": trainable_state(model), "opt": opt.state_dict(),
                                "sched": sched.state_dict(), "step": step, "args": ARGS,
                                "z_dim": Z_DIM, "axes": AXN, "val": rec}, tmp)
                    os.replace(tmp, ck)        # atomic, so a crash never leaves a half-written ckpt
                    if r_spec_pk > best_spec[0]:
                        best_spec[0] = r_spec_pk
                        torch.save({"model": trainable_state(model), "opt": opt.state_dict(),
                                    "sched": sched.state_dict(), "step": step, "args": ARGS,
                                    "z_dim": Z_DIM, "axes": AXN, "val": rec}, OUT / "best.pt.tmp")
                        os.replace(OUT / "best.pt.tmp", OUT / "best.pt")
                        log(f"  new best val_spec_r_peaks {r_spec_pk:.5f} -> best.pt")
                if DDP:
                    torch.distributed.barrier()
            if c.time_budget and time.time() - t_start > c.time_budget:
                # Save here rather than trusting the last periodic checkpoint: the budget can expire
                # anywhere in the `ckpt_every` window, and the next link in the chain resumes from
                # `last.pt`, so whatever is not written here is simply recomputed.
                if RANK == 0:
                    # BEST IS SELECTED ON val_spec_r_peaks, not on pooled r. Pooled r is 87%
                    # determined by having any non-zero conditioning vector at all (ablation:
                    # zero 0.0685, globalmean 0.2064, real 0.2384), so selecting on it would pick a
                    # checkpoint for a property stage B does not change.
                    tmp = OUT / "last.pt.tmp"
                    torch.save({"model": trainable_state(model), "opt": opt.state_dict(),
                                "sched": sched.state_dict(), "step": step, "args": ARGS,
                                "z_dim": Z_DIM, "axes": AXN}, tmp)
                    os.replace(tmp, ck)
                if DDP:
                    torch.distributed.barrier()
                log(f"time budget reached at step {step:,}; checkpoint is in {ck}")
                step = c.steps
            if step >= c.steps:
                break
    if RANK == 0:
        flog.close()
    log("DONE")


def main(argv=None):
    """The original script's command line, flag for flag."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default="data/train_v2")
    ap.add_argument("--stage1", help="stage-1 checkpoint; or --z to use a precomputed z")
    ap.add_argument("--z", help="precomputed per-cell z (.npy), rows aligned to cells.tsv")
    ap.add_argument("--z-bank",
                    help="comma-separated z files to SAMPLE AMONG, one drawn per training step.\n"
                         "Measured 2026-11-12 on the full store: a real deployment panel carries only "
                         "1-26%% of a cell's UMIs and the flagship MERFISH 480 carries 7.9%% (3.2%% of its "
                         "detected genes). So a wide-view z and a panel-view z are not the same "
                         "conditioning variable -- about 12x apart in input information for the 480. "
                         "Training on wide z alone and deploying on panel z is a silent train/deploy "
                         "mismatch: both losses stay healthy while cell-type resolution collapses, "
                         "because the decoder learns to trust a sharpness in z that the panel cannot "
                         "supply. Build the bank with aceso/stage1_infer.py --view/--panel.")
    ap.add_argument("--axes", default="data/train_v2/axes.npz",
                    help="named within-type RNA axes to condition on alongside z, as an .npz with "
                         "`score`, `mask`, `names` (aceso/make_axes.py). The cell-state embedding is "
                         "then (z, axes) rather than z alone. WHY: measured on the patient cohort, a "
                         "0.73 SD group shift along the chamber axis arrives at z as a 0.0013 cosine "
                         "wiggle inside a 512-d vector whose principal structure is cell type, and the "
                         "decoder faithfully passes on something buried -- within/across separation "
                         "0.3%%. A ridge probe shows the information is still in z (r 0.347 within CM), "
                         "so what fails is scale, not presence. Giving the axis its own input is what "
                         "Bolero does for the AP-1 activity axis. See aceso/ARCHITECTURE_V2.md for the "
                         "pre-registered PASS line (within/across >= 3%%, across not degraded >10%%). "
                         "Pass 'none' to condition on z alone.")
    ap.add_argument("--within-weight", type=float, default=0.0,
                    help="stage B. Adds the within-type contrast of v1's `--within-weight`, ported from "
                         "gse205117/scooby/train_aceso_decoder.py:within_kl. 0 = stage A only.")
    ap.add_argument("--within-frac", type=float, default=0.5,
                    help="fraction of steps whose cells are drawn as two halves of ONE (organ, cell "
                         "type, sample) at two ends of one axis; the rest stay random, which is what "
                         "keeps the shared profile from being torn up")
    ap.add_argument("--within-pool", type=int, default=8, help="bins pooled before the binomial split")
    ap.add_argument("--triples", default="data/train_v2/within_triples.json",
                    help="built by aceso/make_within_triples.py, which applies the three batch gates. "
                         "pcw did not survive gate 2 on this corpus: 0 of 353 candidate contrasts were "
                         "separable from a permutation of their sample labels, because 74%% of "
                         "(organ, cell type, PCW) cells hold a single sample. Using it would have "
                         "trained the model on batch.")
    ap.add_argument("--heads", default="atac,rna")
    ap.add_argument("--window", type=int, default=524288, help="bp per training window")
    ap.add_argument("--bin", type=int, default=32)
    ap.add_argument("--z-dim", type=int, default=512)
    ap.add_argument("--cells-per-step", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-rna", type=float, default=1.0, help="weight on the RNA head's loss")
    ap.add_argument("--freeze-trunk", action="store_true", default=True)
    ap.add_argument("--no-freeze-trunk", dest="freeze_trunk", action="store_false")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=6, help="DataLoader workers per rank")
    ap.add_argument("--val-windows", type=int, default=64)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--resume")
    ap.add_argument("--init-from",
                    help="warm start the WEIGHTS only -- step, optimiser and scheduler are reset. Stage "
                         "B is a fine-tune, not a continuation: inheriting stage A's step count would "
                         "end the run before it began, and inheriting its scheduler would start at the "
                         "tail of a decay fitted to a different horizon. `--resume`/last.pt remain the "
                         "path for an actual interrupted run.")
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--dense-dir", default=None,
                    help="stage B v2: directory of dense group pseudobulk tracks (built with DENSE_BUILD=1)")
    ap.add_argument("--pair-frac", type=float, default=0.5,
                    help="share of steps drawn as two cell-type groups with dense targets")
    ap.add_argument("--axis-frac", type=float, default=0.2,
                    help="share of steps drawn as a within-type axis triple (cycle, pcw)")
    ap.add_argument("--w-dense", type=float, default=1.0,
                    help="weight of the group-mean profile term against the dense track")
    ap.add_argument("--w-contrast", type=float, default=1.0,
                    help="weight of the between-group split term against the dense tracks")
    ap.add_argument("--rna-trunk-scale", type=float, default=1.0,
                    help="gradient of the RNA loss that reaches the shared trunk, as a factor; the RNA "
                         "read-out itself always gets the full gradient")
    ap.add_argument("--snap-every", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=None, help="LR warm-up steps; default = DecoderConfig (1000)")
    ap.add_argument("--organism", type=int, default=0,
                    help="AlphaGenome organism index: 0 human, 1 mouse. The v2 corpus is human; "
                         "_alphagenome.py hard-codes 1 because v1 was mouse")
    ap.add_argument("--ag-weights", default=os.environ.get("ACESO_AG_WEIGHTS"),
                    help="AlphaGenome trunk safetensors. Defaults to $ACESO_AG_WEIGHTS, then "
                         "external/alphagenome/model_all_folds.safetensors, and only then to a "
                         "huggingface_hub download -- which needs network access the compute nodes may "
                         "not have, and which is why the file is staged explicitly")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    _av = list(sys.argv[1:] if argv is None else argv)
    a.lr_given = any(x == "--lr" or x.startswith("--lr=") for x in _av)
    return train_v2_decoder(V2DecoderConfig(**vars(a)))


if __name__ == "__main__":  # pragma: no cover
    main()
