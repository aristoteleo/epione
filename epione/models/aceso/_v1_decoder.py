"""Stage 2 of the released ACESO v1 model — the decoder training run, ported verbatim.

This is ``gse205117/scooby/train_aceso_decoder_axes.py``, the script that trained the released v1
decoder ``ag_scooby_axesw_ft160k_mixed/step4000.pt`` (the model every Fig 3 panel uses). The code, the
numerics, the log fields and the checkpoint format are the script's; what changed is the entry point --
the module-level body is :func:`train_v1_decoder`, driven by :class:`V1DecoderConfig`, :func:`main`
rebuilds the same command line, and the paths the script hard-coded are resolved through :mod:`._sites`,
which records them in the training log before a GPU is taken.

Decoder
-------
``decoder="molora"`` (default) is the state-conditioned mixture of LoRA experts on Borzoi
(:mod:`._molora`): scooby's model is wrapped by ``AcesoDecoder`` -- a shared lower trunk (upstream
LoRA), K expert-adapted upper trunks (the last ``expert_blocks`` transformer blocks, the U-net
upsampling path and the final convolution), a gate pi(z) over the experts, and the upstream
``cell_state_to_conv`` read-out on the gate-mixed embedding. ``decoder="film"`` is a per-cell
non-linear FiLM read-out on scooby's shared trunk (:mod:`._filmhead`); ``decoder="scooby"`` is plain
scooby, the D0 baseline. Everything else -- data, folds, targets, loss, optimiser, schedule,
stabilisation, validation -- is ``train_scooby.py``'s, i.e. upstream scooby's (see the inherited
settings below).

The expert load-balancing term (``balance``) is added to the loss; the gate is trained with the
cell-state learning rate. Logged in addition, for ``molora`` only: the mean gate per expert and the mean
per-cell gate entropy over the last 20 steps. ``balance_mode="bias"`` is auxiliary-loss-free expert
balancing (Wang et al. 2024) instead of the CV^2 penalty, whose weight 0.01 collapsed the gate to
uniform (run ``flz_molora_lsi_mixed``).

Backbone
--------
``backbone="borzoi"`` (default, ``johahi/borzoi-replicate-0``) or ``"flashzoi"``
(``johahi/flashzoi-replicate-0``; flash_attn MHA + RoPE, attention LoRA on the q/v rows of the fused
Wqkv, see :mod:`._backbone`), or ``"alphagenome"`` (alphagenome-pytorch, ``envs/alphagenome``, Python
3.12; mouse organism index; 1 bp embeddings pooled to 32 bp; :mod:`._alphagenome`).

Gate
----
``gate="prototype"`` (default; cosine to k-means-initialised prototypes over ``gate_tau``) or
``gate="mlp"``. Gate statistics are logged for ``molora`` only.

Specificity term
----------------
``spec_weight`` w > 0 adds a cross-cell specificity term to the loss (the correlation loss across cell
types of Cochran et al., *Genome Biol* 2024, and the cosine term of CREsted's ``CosineMSELog``, applied
across the cells of a sequence): predictions and targets summed over ``spec_pool`` bins (32 = 1,024 bp),
log1p, centred over the 64 cells of the sequence, Pearson r across cells per pooled bin (bins where some
cell has an insertion); the term is 1 - mean r, scaled each step to w x the Poisson-multinomial loss
(stop-gradient ratio; magnitude matching as CREsted's dynamic weight). Targets are single cells (no
neighbours), as for the main loss. The same r on the validation sequences is logged as ``val_spec_r``
(checkpoint selection unchanged: ``val_pearson_log``).

Classifier-free guidance
------------------------
``cfg_drop`` p > 0 is condition dropout (Ho & Salimans 2022): during training each sampled cell's state
is replaced, with probability p, by a learned null state (parameter ``null_cell_emb``, zero-initialised,
saved with the trainable state), so the model also learns the unconditional profile. At inference
log lambda_w = log lambda_null + w (log lambda_z - log lambda_null); w = 1 is the conditional model.
Validation is conditional (no dropout); in addition ``val_pearson_null`` is logged (the null state for
every cell).

Within-cell-type contrast
-------------------------
``within_weight`` w > 0 (2026-10-17). A fraction ``within_frac`` of the training sequences draw their 64
cells as two halves of one reference cell type at two different stages (cell type and stage pair uniform
over the (type, early stage, late stage) triples with >= 32 cells per stage; other sequences: 64 random
cells as before). Main and specificity losses are unchanged (all 64 cells). Added term on the structured
sequences: the two halves are pooled (metacells of 32 cells) and summed over ``within_pool`` bins; given
the observed insertions of both halves in a pooled bin, the late half's count is binomial with
p = mu_L / (mu_L + mu_E), mu_g = predicted pooled rate of half g rescaled to that half's observed total
over the window (so only the predicted *shape difference* between the halves enters, not depth); the
term is KL(observed split || p) = the binomial NLL minus its value at the observed split, averaged per
insertion; scaled each step to w x the Poisson-multinomial loss (stop-gradient ratio, as
``spec_weight``). Motivation: ``merfish_frag/13_within_type_diagnostics.py`` (z neighbours carry
within-type change; the decoder does not).

Axis contrast
-------------
``within_axes`` (R3, ``conditioning_pilot/r3_axis_contrast``) is ``<tsv>:<col>[,<col>...]``: it adds
within-type contrast triples along each named RNA axis (per-type median split, low vs high) next to the
stage triples; the axis is drawn uniformly, then a triple of that axis. ``within_axes_stage_share`` (R3b)
is the probability that a within item uses the stage triples, the remaining mass being split uniformly
over the ``within_axes`` groups; the default -1 is uniform over all groups (R3 arm AX).

Warm starts and checkpoints
---------------------------
``init_from`` starts from a trained checkpoint's weights (gate prototypes / bias / BN included) with a
fresh optimiser and schedule (fine-tuning), unlike ``resume``. ``save_every`` n also keeps ``step<n>.pt``
(weights only)::

    srun --overlap --jobid=<gpu> -n 1 -c 8 /scratch/users/steorra/envs/scgpt/bin/python -I \\
        -m epione.models.aceso._v1_decoder --out runs/acesodec_<z> --embedding <parquet> [--experts 4]

Inherited scooby settings
-------------------------
The text below documents the shared scooby settings (from ``train_scooby.py``). scooby on GSE205117
(mouse E7.5-E8.75 10x Multiome), ATAC, single-cell targets.

Follows upstream scooby (gagneurlab/scooby @ dbfe6cd, ``scripts/train_multiome.py``; Hingerl, Martens et
al., *Nat Methods* 2025, Methods) through the epione scooby bridge (targets / dataset, verified bit-exact
against upstream's ``onTheFlyMultiomeDataset``; :mod:`epione.models.scooby`):

model
    ``Scooby.from_pretrained('johahi/borzoi-replicate-0', embedding_dim 1920,
    return_center_bins_only, disable_cache, use_transform_borzoi_emb False)`` + upstream ``get_lora``
    (LoRA on conv_layer / to_q / to_v / transformer FF, ``cell_state_to_conv`` trainable)
data
    524,288 bp input, 196,608 bp target (6,144 x 32 bp), ATAC insertions x 0.05; 64 cells sampled per
    sequence; no neighbours (true single cell, as upstream); reverse-complement augmentation; +-3 bp
    shift augmentation
loss
    upstream ``poisson_multinomial_torch``, total_weight 0.2
optim
    AdamW, upstream ``add_weight_decay`` groups; lr 2e-4 (LoRA etc.) and 4e-4 (``cell_state_to_conv``) as
    in the paper Methods (the released ``config_multiome.yaml`` has lr 1e-4); weight decay 1e-6; grad
    clip 1.0; 1,000 warm-up steps then linear decay to 0; bf16 autocast; effective batch 8 sequences
    (paper) = gradient accumulation over 8 sequences on one GPU
stabilise
    paper: "froze the batch normalization layers ... and disabled dropout within Borzoi" at the start,
    then unfroze -- implemented as BN eval + dropout off during the warm-up steps
folds
    ``scooby_bench/mm10_folds.bed`` (contiguous 196,608 bp blocks, 8 folds); test fold3, val fold4
    (upstream's choice of indices)
eval
    upstream-style validation Pearson on val sequences every ``eval_every`` steps; best checkpoint kept

Deviations forced by the data / goal (not simplifications)
----------------------------------------------------------
n_tracks 1 (ATAC only)
    GEO has no BAMs, so per-cell stranded RNA coverage cannot be built;
cell embedding = ``embedding``
    the ACESO stage-1 state (scGPT-predicted z from the MERFISH panel,
    ``scgpt/to_scooby_embedding.py``), or, for the scooby-original control, Poisson-MultiVI (author's
    poissonmultivi branch, default parameters, batch = sample) trained on the 500-gene MERFISH panel +
    ATAC (reference) jointly with the MERFISH cells as RNA-only (``scooby/poisson_multivi.py``);
+-3 bp shift
    applied to sequence and target together (bridge semantics; upstream shifts the sequence only -- both
    are within one 32 bp bin).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _flush_print(*msg):
    """Default log sink: ``print(..., flush=True)``, the original script's behaviour."""
    print(*msg, flush=True)


@dataclass
class V1DecoderConfig:
    """One field per flag of ``train_aceso_decoder_axes.py``, same names (``-`` -> ``_``) and defaults.

    ``vars(config)`` is what the checkpoint stores under ``"molora"``, exactly as the script stored
    ``vars(args)``, so tools that turn that dict back into a command line (for instance
    ``scooby/heart/prepare_continue_12000.py``) keep working. The path fields below are therefore left
    ``None`` by default and resolved into local variables, never written back.

    The fields that are not self-explanatory
    ----------------------------------------
    ``decoder``
        ``"molora"`` = state-conditioned mixture of LoRA experts on Borzoi (:mod:`._molora`, the
        released v1 decoder); ``"film"`` = per-cell non-linear FiLM read-out on scooby's shared trunk
        (:mod:`._filmhead`); ``"scooby"`` = plain scooby (baseline D0).
    ``backbone``
        ``"borzoi"``, ``"flashzoi"`` (flash_attn MHA + RoPE; :mod:`._backbone`) or ``"alphagenome"``
        (:mod:`._alphagenome`; needs the ``alphagenome`` environment).
    ``gate``
        ``"prototype"`` = cosine to k-means-initialised prototypes over ``gate_tau``; ``"mlp"`` = an MLP
        head, which starts uniform and receives almost no gradient while the experts are identical.
    ``balance``
        weight of the expert load-balancing term added to the loss (CV^2 penalty under
        ``balance_mode="aux"``; the gate bias update rate is ``bias_rate`` under ``"bias"``).
    ``spec_weight``
        w > 0 adds the cross-cell specificity term (1 - mean Pearson r across the 64 cells of a sequence
        per ``spec_pool``-bin block, scaled each step to w x the main loss by a stop-gradient ratio).
    ``within_weight``, ``within_frac``, ``within_pool``
        w > 0 adds the within-cell-type contrast: a fraction ``within_frac`` of the sequences draw their
        64 cells as two stage halves of one cell type, and their depth-free binomial KL over
        ``within_pool``-bin blocks is scaled to w x the main loss.
    ``within_axes``, ``within_axes_stage_share``
        R3: ``<tsv>:<col>[,<col>...]`` adds within-type triples along named RNA axes (per-type median
        split) next to the stage triples; ``within_axes_stage_share`` p is the probability of drawing a
        stage triple, the rest being uniform over the axis groups (-1 = uniform over all groups).
    ``cfg_drop``
        p > 0 replaces each sampled cell's state by the learned null state with probability p
        (classifier-free guidance); validation stays conditional and logs ``val_pearson_null`` too.
    ``init_from``
        start from a trained checkpoint's weights with a fresh optimiser and schedule (fine-tuning),
        unlike ``resume``, which continues the run.
    ``save_every``
        n > 0 also keeps ``step<n>.pt`` (weights only) at every multiple of n.

    Paths
    -----
    ``root``, ``scooby_src``, ``mm10_fa``, ``folds_bed``, ``alphagenome_weights``, ``data_dir``,
    ``cells_tsv`` and ``resume_schedule_dir`` default to ``None`` and are then resolved by
    :mod:`._sites` (``require_sites`` prints the table into the training log). ``embedding`` defaults to
    ``<root>/gse205117/scooby/embedding.parquet`` and is written back, so the checkpoint records the
    parquet that was actually read.
    """

    out: str = ""                        # required on the command line; "" is not a usable output directory
    epochs: int = 40
    max_steps: int = 0                   # stop after N optimizer steps (smoke test / budget)
    accum: int = 8
    lr: float = 2e-4
    cellstate_lr: float = 4e-4
    wd: float = 1e-6
    warmup: int = 1000
    eval_every: int = 250
    patience: int = 0                    # consecutive evaluations without a new best val Pearson; 0 = off
    min_delta: float = 0.0
    val_seqs: int = 32
    workers: int = 6
    resume: Optional[str] = None
    resume_restart_schedule: bool = False
    resume_warmup: int = 100             # warmup steps for an explicit resumed LR restart
    embedding: Optional[str] = None      # default <root>/gse205117/scooby/embedding.parquet
    extra_cells: Optional[str] = None    # TSV of training cells from OTHER datasets (sample, celltype, frag)
    auto_resume: bool = False            # resume from <out>/last.pt when it exists; exit at once if <out>/DONE
    data_seed: int = 0                   # training order = permutation from (data seed, epoch)
    decoder: str = "molora"              # molora | film | scooby
    backbone: str = "borzoi"             # borzoi | flashzoi | alphagenome
    film_hidden: int = 256
    experts: int = 4
    expert_blocks: int = 2
    low_expert_blocks: int = 0           # lower-trunk conditioning: blocks below the expert part
    expert_conv_tower: bool = False      # lower-trunk conditioning: expert Locon on the conv tower
    balance: float = 0.01
    gate: str = "prototype"              # prototype | mlp
    gate_tau: float = 0.1
    # ---- gate repair (all off by default; see the "Gate repair" paragraph of :mod:`._molora`)
    gate_topk: int = 0                   # hard top-k routing with a straight-through estimator (0 = dense)
    proto_sep: float = 0.0               # weight of the prototype separation penalty
    proto_sep_margin: float = 0.0        # cosine above which a prototype pair starts to be penalised
    gate_tau_end: float = 0.0            # anneal gate_tau -> this over gate_tau_steps (0 = no annealing)
    gate_tau_steps: int = 0              # length of the annealing (0 = max_steps, else 4000)
    reinit_prototypes: bool = False      # after init_from, re-run the k-means prototype initialisation
    reset_gate_bias: bool = False        # after init_from, reset the load-balancing gate bias to zero
    codebook_init: Optional[str] = None  # after init_from, load the K x d codebook from this .npy
    # ---- VQ-MoLoRA (P1 of paper/fig4/sharpness/ARCHITECTURE_SURVEY.md; all off by default)
    affine_scale: float = 1.0            # gate-mode affine: scale on the head output before the 1/K offset
    gate_mode: str = "softmax"           # softmax | vq (one hard code per cell) | affine
    codebook_ema: float = 0.99           # EMA decay of the VQ codebook update (0 = train the codes)
    commit_weight: float = 0.0           # VQ commitment term beta
    dead_code_restart: int = 0           # restart codes under dead_code_thresh every N steps (0 = off)
    dead_code_thresh: float = 0.02
    learn_tau: bool = False              # learnable gate log-temperature (clamped to [0.01, 1])
    rotation_trick: bool = False         # Fifty et al. (ICLR 2025) rotation+rescale
    code_grouped_batches: int = 0        # VQ: keep only the cells of N codes of each sampled batch (0 = off)
    code_group_mode: str = "top"         # top | mixed
    grow_experts: bool = False           # init_from with more experts than the checkpoint: tile them
    limoe_weight: float = 0.0            # LIMoE entropy pair in place of the load-balancing bias
    balance_mode: str = "aux"            # aux | bias
    bias_rate: float = 1e-3
    spec_weight: float = 0.0
    spec_pool: int = 32
    cfg_drop: float = 0.0
    within_weight: float = 0.0
    within_frac: float = 0.5
    within_pool: int = 32
    within_min_cells: int = 32
    within_axes_stage_share: float = -1.0
    within_axes: str = ""
    init_from: Optional[str] = None
    save_every: int = 0
    # ---- resources; None = resolve through :mod:`._sites` (see the class docstring)
    root: Optional[str] = None
    scooby_src: Optional[str] = None
    mm10_fa: Optional[str] = None
    folds_bed: Optional[str] = None
    alphagenome_weights: Optional[str] = None
    data_dir: Optional[str] = None
    cells_tsv: Optional[str] = None
    resume_schedule_dir: Optional[str] = None


def train_v1_decoder(config: V1DecoderConfig = V1DecoderConfig(), *, log=print):
    """Train the ACESO v1 decoder. The body is the script's, statement for statement."""
    import pandas as pd
    import torch
    import torch.nn as nn
    from scipy import stats

    from ._sites import require_sites, site_path

    c = config
    emit = _flush_print if log is print else log          # the script's print(..., flush=True)
    if c.limoe_weight > 0:          # the bias cancels a shared preference in the dense weights: a confound under
        c.bias_rate = 0.0           # hard routing, and the LIMoE pair is the replacement (survey P1)
        emit("LIMoE entropy pair on: the load-balancing gate bias is frozen (bias_rate = 0)")
    require_sites("root", "scooby_src", "mm10_fa", "folds_bed_mm10")
    R = (c.root or site_path("root")) + "/gse205117"
    SCOOBY_SRC = c.scooby_src or site_path("scooby_src")
    FA = c.mm10_fa or site_path("mm10_fa")
    FOLDS_BED = c.folds_bed or site_path("folds_bed_mm10")
    DATA = c.data_dir or f"{R}/data"
    CELLS_TSV = c.cells_tsv or f"{R}/rna/cells.tsv.gz"
    RESUME_SCHEDULE_DIR = c.resume_schedule_dir or f"{R}/scooby"
    c.embedding = c.embedding or f"{R}/scooby/embedding.parquet"          # recorded in the checkpoint
    AG_WEIGHTS = ((c.alphagenome_weights or site_path("alphagenome_weights", required=False))
                  if c.backbone == "alphagenome" else None)
    if c.backbone == "alphagenome":   # None = the pinned HF copy _alphagenome downloads, as in the script
        emit(f"  alphagenome_weights    {AG_WEIGHTS}")
    sys.path.insert(0, SCOOBY_SRC)
    from .. import scooby as sb
    from scooby.utils.utils import poisson_multinomial_torch
    from ._decoder import build_decoder
    from ._filmhead import FiLMConfig, build_film_decoder
    from ._molora import MoLoRAConfig, build_aceso_decoder

    os.makedirs(c.out, exist_ok=True)
    torch.manual_seed(0)
    dev = torch.device("cuda")

    class MultiFragmentReader(sb.FragmentInsertionReader):
        """Several samples' fragment files behind one global cell index (cells stacked in sample order),
        so 64 cells are sampled from all cells as in upstream's merged insertion matrix."""

        def __init__(self, readers):
            self.readers = readers
            self.barcodes = [b for r in readers for b in r.prefixed]

        @property
        def n_cells(self):
            return len(self.barcodes)

        def reopen(self):
            for r in self.readers: r.reopen()
            return self

        def window_insertions(self, chrom, start, end, *, dtype=np.float32):
            import scipy.sparse as sp
            return sp.vstack([r.window_insertions(chrom, start, end, dtype=dtype) for r in self.readers]).tocsr()

    def build_reader(cells):
        """One FragmentInsertionReader per sample, stacked in the order the samples appear in `cells`.

        The fragments file is `cells['frag']` when that column carries a path for the sample (cells added
        through --extra-cells, whose files live outside {R}/data), otherwise the GSE205117 file found by
        suffix in {R}/data -- the original behaviour, unchanged when no --extra-cells table is given.
        """
        readers = []
        for s, d in cells.groupby("sample", sort=False):
            f = d["frag"].iloc[0] if "frag" in d.columns and isinstance(d["frag"].iloc[0], str) else None
            if f is None:
                f = f"{DATA}/" + [x for x in os.listdir(DATA) if x.endswith(f"_{s}_ATAC_fragments.tsv.gz")][0]
            r = sb.FragmentInsertionReader(f, [i.split("#", 1)[1] for i in d.index])
            r.prefixed = list(d.index)
            readers.append(r)
        return MultiFragmentReader(readers)

    emb = pd.read_parquet(c.embedding)                             # obs_names, embedding (reference + MERFISH)
    cells = pd.read_csv(CELLS_TSV, sep="\t", index_col=0)
    if c.extra_cells:
        # additional training samples behind the same global cell index (e.g. the GSE280587 heart multiome,
        # scooby/heart/h3_extra_cells.py). Columns: sample, celltype, frag (path to the fragments file).
        ex = pd.read_csv(c.extra_cells, sep="\t", index_col=0)
        assert not len(cells.index.intersection(ex.index)), "--extra-cells repeats a GSE205117 cell id"
        assert {"sample", "celltype", "frag"} <= set(ex.columns), sorted(ex.columns)
        assert not set(ex["sample"]) & set(cells["sample"]), "--extra-cells reuses a GSE205117 sample name"
        cells = pd.concat([cells, ex])
        emit(f"--extra-cells {c.extra_cells}: +{len(ex)} cells in {ex['sample'].nunique()} samples")
    cells = cells.loc[[x for x in emb.obs_names if x in cells.index]]
    order = cells.sort_values("sample", kind="stable").index             # reader stacks samples in this order
    cells = cells.loc[order]
    reader = build_reader(cells)
    assert reader.barcodes == list(cells.index)
    E = np.vstack(emb.set_index("obs_names").loc[cells.index, "embedding"].values).astype(np.float32)
    folds = sb.read_fold_bed(FOLDS_BED)
    fcol = [x for x in folds.columns if x not in ("chrom", "start", "end")][0]
    train_iv = folds[~folds[fcol].isin(["fold3", "fold4"])].reset_index(drop=True)
    val_iv = folds[folds[fcol] == "fold4"].sample(n=min(c.val_seqs, (folds[fcol] == "fold4").sum()),
                                                  random_state=0).reset_index(drop=True)
    import scipy.sparse as sp
    no_nb = sp.csr_matrix((reader.n_cells, reader.n_cells), dtype=np.float32)

    class WithinTypeDataset(sb.ScoobyATACDataset):
        """--within-weight: with probability --within-frac an item's 64 cells are two halves (early / late stage) of one
        cell type; returns (x, rc, y, ce, half) with half = 0 early / 1 late / -1 (random cells). Items not structured
        are exactly the parent class's (same RNG stream)."""

        def __init__(self, *args, celltype, stage, frac, min_cells, batch=None, axes=None, stage_share=-1.0, **kw):
            self.stage_share = stage_share
            """`batch`: optional per-cell group (the source dataset). Both halves of a pair are then drawn from
            the SAME group, so the early/late contrast never becomes a cross-dataset contrast -- otherwise the
            term would ask the model to reproduce the batch effect between GSE205117 and GSE280587 at stages
            present in only one of them. With batch=None (the default) the triples are exactly as before."""
            super().__init__(*args, **kw)
            self.frac, h = frac, self.cell_sample_size // 2
            self.half = h
            batch = np.full(len(celltype), "", object) if batch is None else np.asarray(batch, object)
            idx = pd.DataFrame(dict(t=celltype, b=batch, s=stage)).groupby(["t", "b", "s"]).indices
            self.triples = []
            for t in sorted(set(celltype)):
                for b in sorted(set(batch)):
                    st = sorted([s for (tt, bb, s), ix in idx.items()
                                 if tt == t and bb == b and len(ix) >= max(min_cells, h)], key=float)
                    self.triples += [(idx[(t, b, x)], idx[(t, b, y)]) for i, x in enumerate(st) for y in st[i + 1:]]
            # --within-axes: one extra triple group per named axis (per cell type x batch, median split low / high)
            self.triple_groups = [self.triples]
            for name, lab in (axes or {}).items():
                g = []
                for (t, b), ix in pd.DataFrame(dict(t=celltype, b=batch)).groupby(["t", "b"]).indices.items():
                    v = np.asarray(lab, float)[ix]; ok = np.isfinite(v)
                    if ok.sum() < 2 * max(min_cells, h):
                        continue
                    med = np.median(v[ok]); lo, hi = ix[ok & (v <= med)], ix[ok & (v > med)]
                    if len(lo) >= max(min_cells, h) and len(hi) >= max(min_cells, h):
                        g.append((lo, hi))
                emit(f"within-axes: {name}: {len(g)} (cell type, low, high) triples")
                if g:
                    self.triple_groups.append(g)

        def __getitem__(self, i):
            torch_ = torch
            sel = np.random.default_rng((self._seed, i, 7919))
            if self.frac <= 0 or sel.random() >= self.frac:
                return (*super().__getitem__(i), torch_.full((self.cell_sample_size,), -1, dtype=torch_.int8))
            if len(self.triple_groups) > 1 and self.stage_share >= 0:
                grp = (self.triple_groups[0] if sel.random() < self.stage_share
                       else self.triple_groups[1 + sel.integers(len(self.triple_groups) - 1)])
            else:
                grp = (self.triple_groups[sel.integers(len(self.triple_groups))]
                       if len(self.triple_groups) > 1 else self.triples)
            early, late = grp[sel.integers(len(grp))]
            cells = np.r_[sel.choice(early, self.half, replace=False), sel.choice(late, self.half, replace=False)]
            fasta = self._ensure_open(); rng = self._get_rng(i)               # as the parent class from here on
            row = self.intervals.iloc[i]
            chrom, t_start, t_end = str(row["chrom"]), int(row["start"]), int(row["end"])
            shift = int(rng.integers(-self.shift_aug, self.shift_aug + 1)) if self.shift_aug else 0
            t_start += shift; t_end += shift
            i_start, i_end = t_start - self.flank, t_end + self.flank
            chrom_len = fasta.get_reference_length(chrom)
            fs, fe = max(0, i_start), min(chrom_len, i_end)
            seq = fasta.fetch(chrom, fs, fe) if fe > fs else ""
            onehot = np.zeros((self.input_length, 4), dtype=np.float32)
            if seq:
                onehot[fs - i_start: fs - i_start + len(seq)] = _onehot(seq)
            groups = [np.append(self.neighbors[x].nonzero()[1], x) for x in cells]
            targets = self.reader.pseudobulk(chrom, t_start, t_end, groups, bin_size=self.bin_size, scale=self.scale)
            emb_ = self.embedding[cells]
            rc = False
            if self.rc_aug and bool(rng.integers(0, 2)):
                rc = True; onehot = onehot[::-1, ::-1].copy(); targets = targets[:, ::-1].copy()
            half = np.r_[np.zeros(self.half, np.int8), np.ones(self.half, np.int8)]
            return (torch_.from_numpy(onehot), torch_.tensor(rc), torch_.from_numpy(np.ascontiguousarray(targets.T)),
                    torch_.from_numpy(np.ascontiguousarray(emb_)), torch_.from_numpy(half))

    from ..scooby._dataset import one_hot_dna as _onehot
    if c.within_weight > 0:
        ds = WithinTypeDataset(reader, FA, train_iv, E, neighbors=no_nb, cell_sample_size=64, rc_aug=True,
                               shift_aug=3, seed=0,
                               celltype=cells.celltype.astype(str).to_numpy(),
                               stage=cells["sample"].str.split("_").str[0].str.lstrip("E").to_numpy(),
                               batch=(cells["dataset"].fillna("reference").astype(str).to_numpy()
                                      if "dataset" in cells.columns else None),
                               frac=c.within_frac, min_cells=c.within_min_cells,
                               axes=(lambda p, cs: {col: pd.read_csv(p, sep="\t", index_col=0)
                                                    .reindex(cells.index)[col].to_numpy(float)
                                                    for col in cs})(
                                   *((c.within_axes.split(":")[0], c.within_axes.split(":")[1].split(","))
                                     if c.within_axes else ("", []))) if c.within_axes else None,
                               stage_share=c.within_axes_stage_share)
        emit(f"within-type: {len(ds.triples)} (cell type, early stage, late stage) triples, "
             f"fraction {c.within_frac}")
    else:
        ds = sb.ScoobyATACDataset(reader, FA, train_iv, E, neighbors=no_nb, cell_sample_size=64, rc_aug=True,
                                  shift_aug=3, seed=0)
    vds = sb.ScoobyATACDataset(reader, FA, val_iv, E, neighbors=no_nb, cell_sample_size=64, rc_aug=False,
                               shift_aug=0, seed=1)

    class EpochSampler(torch.utils.data.Sampler):
        """Training order = permutation drawn from (--data-seed, epoch) only: the same in every run (any decoder,
        backbone, worker count) and after a resume, which restarts at (epoch, position) of the checkpoint."""

        def __init__(self, n, seed):
            self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

        def __iter__(self):
            return iter(np.random.default_rng((self.seed, self.epoch)).permutation(self.n)[self.start:].tolist())

        def __len__(self):
            return self.n - self.start

    sampler = EpochSampler(len(ds), c.data_seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, sampler=sampler, num_workers=c.workers,
                                     worker_init_fn=sb.worker_init, persistent_workers=True, prefetch_factor=4)
    vdl = torch.utils.data.DataLoader(vds, batch_size=1, shuffle=False, num_workers=2, worker_init_fn=sb.worker_init)
    emit(f"cells {reader.n_cells}, emb dim {E.shape[1]}, train seqs {len(ds)}, val seqs {len(vds)}")

    if c.decoder == "molora":
        model = build_aceso_decoder(E.shape[1], MoLoRAConfig(n_experts=c.experts, n_expert_blocks=c.expert_blocks,
                                                             balance_weight=c.balance, gate=c.gate,
                                                             gate_tau=c.gate_tau,
                                                             balance=c.balance_mode, bias_rate=c.bias_rate,
                                                             n_low_expert_blocks=c.low_expert_blocks,
                                                             expert_conv_tower=c.expert_conv_tower,
                                                             gate_topk=c.gate_topk, proto_sep_weight=c.proto_sep,
                                                             proto_sep_margin=c.proto_sep_margin,
                                                             gate_mode=c.gate_mode, affine_scale=c.affine_scale,
                                                             codebook_ema=c.codebook_ema,
                                                             commit_weight=c.commit_weight,
                                                             dead_code_restart=c.dead_code_restart,
                                                             dead_code_thresh=c.dead_code_thresh,
                                                             learn_tau=c.learn_tau,
                                                             rotation_trick=c.rotation_trick),
                                    device=dev, backbone=c.backbone, checkpoint=AG_WEIGHTS)
        if c.gate == "prototype" and not c.codebook_init:
            # k-means of the cell states the decoder is conditioned on (a resume overwrites). Skipped when
            # --codebook-init is given: that path replaces every prototype a few lines further down, so the fit
            # has no effect on the model -- and at K = 16 over 89,593 states it costs ~8 minutes of start-up.
            sizes = model.cell_state_to_gate.init_kmeans(E, seed=0)
            emit(f"prototype gate: k-means cluster sizes {sizes.tolist()}")
    elif c.decoder == "film":
        model = build_film_decoder(E.shape[1], FiLMConfig(hidden=c.film_hidden), device=dev, backbone=c.backbone,
                                   checkpoint=AG_WEIGHTS)
    else:                                       # plain scooby (D0) on the chosen backbone
        model = build_decoder(E.shape[1], device=dev, backbone=c.backbone, checkpoint=AG_WEIGHTS)
    if c.cfg_drop > 0:                          # before the optimiser groups and any resume
        model.register_parameter("null_cell_emb", nn.Parameter(torch.zeros(E.shape[1], device=dev)))
    emit(f"trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
         f"(experts {sum(p.numel() for n, p in model.named_parameters() if 'expert_lora' in n or 'expert_locon' in n):,})")
    from ._decoder import param_groups, trainable_state
    groups = param_groups(model, c.lr, c.cellstate_lr, c.wd)
    opt = torch.optim.AdamW(groups)
    requested_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = math.ceil(len(ds) / c.accum)
    total = c.epochs * steps_per_epoch if not c.max_steps else c.max_steps
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-7, total_iters=c.warmup),
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=max(total - c.warmup, 1))],
        milestones=[c.warmup])
    schedule_config = dict(origin_step=0, total_steps=total, warmup=c.warmup, peak_lrs=requested_lrs)
    step, best, t0 = 0, -1.0, time.time()
    best_step, stopped_early = 0, False
    if c.auto_resume:
        if os.path.exists(f"{c.out}/DONE"):
            emit("already finished:", open(f"{c.out}/DONE").read().strip()); sys.exit(0)
        if os.path.exists(f"{c.out}/last.pt"):
            c.resume = f"{c.out}/last.pt"
    if c.init_from and not c.resume:                                   # fine-tune from trained weights, fresh optimiser
        ck = torch.load(c.init_from, map_location="cpu", weights_only=False)
        st = ck["model"]
        if c.decoder == "molora" and getattr(model, "deep", False):
            from ._molora import deep_state_dict                        # lower-trunk conditioning: adapters moved down
            st = deep_state_dict(st, model)
            emit(f"lower-trunk conditioning: remapped {sum(k not in ck['model'] for k in st):,} checkpoint keys")
        if c.decoder == "molora" and c.grow_experts:
            from ._molora import grow_experts_state_dict
            k_old = int(ck["molora"].get("experts", c.experts))
            st = grow_experts_state_dict(st, model)
            emit(f"grow experts: {k_old} -> {c.experts}; new expert j starts as a copy of old expert j % {k_old}")
        res = model.load_state_dict(st, strict=False)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        new_deep = set()
        if c.decoder == "molora" and getattr(model, "deep", False):
            # the lower-trunk adapters do not exist in the shallow warm-start checkpoint; they are zero-initialised
            # (expert_locon_up / expert_lora_B = 0), so step 0 still reproduces the warm-start model exactly
            low = {n for n, _ in model.named_parameters() if "expert_locon" in n or "expert_lora" in n}
            new_deep = {n for n in low & set(res.missing_keys)}
        gate_swap = set()
        if c.decoder == "molora":
            # R6: --gate mlp warm-started from a prototype-gate checkpoint (or vice versa). The gate head is the only
            # thing that differs; the MLP's last layer is zero-initialised, so with --gate-mode affine the model starts
            # at c = 1/K (the uniform mixture of the warm start's experts) and with softmax at pi = uniform -- the two
            # arms therefore start from the *same* function. Declared, printed, and restricted to gate tensors only.
            ck_gate = {k for k in ck["model"] if k.startswith("cell_state_to_gate.")}
            md_gate = {n_ for n_, _ in model.named_parameters() if n_.startswith("cell_state_to_gate.")}
            if ck_gate != md_gate:
                gate_swap = md_gate
                res_unexpected_gate = [k for k in res.unexpected_keys if k.startswith("cell_state_to_gate.")]
                emit(f"gate swap: checkpoint {sorted(ck_gate)[:3]} -> model {sorted(md_gate)[:3]}; "
                     f"{len(res_unexpected_gate)} checkpoint gate tensors dropped, "
                     f"{len(md_gate & set(res.missing_keys))} model gate tensors freshly initialised")
                res = type(res)([k for k in res.missing_keys],
                                [k for k in res.unexpected_keys if not k.startswith("cell_state_to_gate.")])
            new_deep |= gate_swap
        if c.decoder == "molora" and (c.gate_mode == "vq" or c.learn_tau):
            # the quantised gate adds parameters/buffers (log_tau, EMA statistics) that a softmax checkpoint has not
            gate_new = ({n for n, _ in model.named_parameters() if n.startswith("cell_state_to_gate.")}
                        & set(res.missing_keys))
            if gate_new:
                emit(f"quantised gate: {len(gate_new)} new gate tensors {sorted(gate_new)}")
            new_deep |= gate_new
            emit(f"lower-trunk conditioning: {len(new_deep)} new zero-initialised expert tensors "
                 f"({sum(p.numel() for n, p in model.named_parameters() if n in new_deep):,} parameters)")
        assert not res.unexpected_keys and not (trainable - new_deep) & set(res.missing_keys), \
            f"init checkpoint does not match the model: unexpected {res.unexpected_keys[:3]}, " \
            f"missing {sorted((trainable - new_deep) & set(res.missing_keys))[:3]}"
        if c.decoder == "molora":
            assert "gate_bias" in ck["model"] or gate_swap
            if c.codebook_init:          # a codebook built outside (e.g. the union codebook of paper/fig4/sharpness)
                P = np.load(c.codebook_init).astype(np.float32)
                g = model.cell_state_to_gate
                assert P.shape == tuple(g.prototypes.shape), (P.shape, tuple(g.prototypes.shape))
                # set_codebook (not a direct write to g.prototypes) also re-seeds the EMA statistics and the load
                # history: without that, codebook_update's `prototypes = ema_sum / ema_count` pulls the codes back to
                # the constructed ones within a few steps (under --grow-experts: the K=4 checkpoint's prototypes tiled
                # to K, i.e. a few distinct directions). Measured: 16 of 16 codes live at step 20, 2 of 16 from step
                # 40 on (heart/runs/flz_heart_emaseed_abandoned_step160). The same reseed now happens inside
                # QuantisedGate.init_kmeans, so --reinit-prototypes is safe too.
                model.set_codebook(P)
                lab = np.argmax((E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)) @
                                (P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)).T, 1)
                emit(f"codebook initialised from {c.codebook_init}: training occupancy "
                     f"{np.round(np.bincount(lab, minlength=len(P)) / len(lab), 4).tolist()}")
            elif c.reinit_prototypes:    # gate repair: after the load, so the checkpoint's collapsed prototypes are dropped
                sizes = model.cell_state_to_gate.init_kmeans(E, seed=0)
                emit(f"gate repair: prototypes re-initialised by k-means, cluster sizes {sizes.tolist()}")
            if c.reset_gate_bias:
                model.gate_bias.zero_()
                emit("gate repair: load-balancing gate bias reset to zero")
        emit(f"initialised from {c.init_from} (step {ck['step']}, best {ck['best']:.4f})")
    if c.resume:
        emit(f"resuming from {c.resume}")
        ck = torch.load(c.resume, map_location="cpu", weights_only=False)
        res = model.load_state_dict(ck["model"], strict=False)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert not res.unexpected_keys and not trainable & set(res.missing_keys), \
            f"checkpoint does not match the model: unexpected {res.unexpected_keys[:3]}, " \
            f"missing {sorted(trainable & set(res.missing_keys))[:3]}"
        sys.path.insert(0, RESUME_SCHEDULE_DIR)
        from resume_schedule import restore as restore_schedule
        sched, schedule_config = restore_schedule(opt, sched, ck, total,
                                                  c.resume_warmup if c.resume_restart_schedule else c.warmup,
                                                  requested_lrs, c.resume_restart_schedule)
        emit("resumed LR schedule:", json.dumps(schedule_config), "current LR", sched.get_last_lr())
        step, best = ck["step"], ck["best"]
        best_step = ck.get("best_step", step)          # checkpoints before --patience: count patience from the resume point
        sampler.epoch, sampler.start = ck["epoch"], ck["pos"]
        if sampler.start >= len(ds):
            sampler.epoch, sampler.start = sampler.epoch + 1, 0
        torch.set_rng_state(ck["torch_rng"]); torch.cuda.set_rng_state(ck["cuda_rng"])      # dropout masks
        emit(f"resumed at step {step} (best val {best:.4f}), epoch {sampler.epoch} position {sampler.start}")

    def set_stabilise(on):
        """BN layers of Borzoi in eval mode and dropout off (on=True), or normal training mode."""
        model.train()
        if on:
            for mod in model.modules():
                if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Dropout)):
                    mod.eval()

    def spec_r(out, y, pool):
        """mean over pooled bins of the Pearson r across cells between log1p(pooled prediction) and log1p(pooled target);
        out / y [1, bins, cells]. Bins where no cell has an insertion are skipped."""
        P = torch.log1p(out[0].float().reshape(-1, pool, out.shape[-1]).sum(1))
        T = torch.log1p(y[0].float().reshape(-1, pool, y.shape[-1]).sum(1))
        P = P - P.mean(1, keepdim=True); T = T - T.mean(1, keepdim=True)
        ok = T.abs().sum(1) > 0
        r = (P * T).sum(1)[ok] / (P[ok].norm(dim=1) * T[ok].norm(dim=1) + 1e-8)
        return r.mean() if ok.any() else out.sum() * 0

    def within_kl(out, y, half, pool):
        """KL(observed split || predicted split) per insertion between the late (half 1) and early (half 0) metacells;
        out / y [1, bins, cells]; each half's prediction rescaled to its observed window total (depth-free)."""
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

    def with_null(ce, p):
        """ce [1, n, d]: each cell's state replaced by the null state with probability p (p = 1: all)"""
        m = torch.rand(ce.shape[:-1], device=ce.device) < p
        return torch.where(m[..., None], model.null_cell_emb.to(ce.dtype), ce)

    @torch.no_grad()
    def validate():
        model.eval(); P, T, SR, P0 = [], [], [], []
        for x, rc, y, ce in vdl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x.permute(0, 2, 1).to(dev), ce.to(dev)).float()
            SR.append(float(spec_r(out, y.to(dev), c.spec_pool)))
            if c.cfg_drop > 0:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o0 = model(x.permute(0, 2, 1).to(dev), with_null(ce.to(dev), 1.0)).float()
                P0.append(o0.squeeze(0).cpu().numpy())
            P.append(out.squeeze(0).cpu().numpy()); T.append(y.squeeze(0).numpy())
        P = np.concatenate(P); T = np.concatenate(T)                    # bins x cells
        r_all = stats.pearsonr(np.log1p(P).ravel(), np.log1p(T).ravel())[0]
        r_cell = np.nanmean([stats.pearsonr(np.log1p(P[:, j]), np.log1p(T[:, j]))[0] for j in range(P.shape[1])])
        r_null = float(stats.pearsonr(np.log1p(np.concatenate(P0)).ravel(), np.log1p(T).ravel())[0]) if P0 else None
        return float(r_all), float(r_cell), float(np.mean(SR)), r_null

    logf = open(f"{c.out}/train.log", "a")
    # gate repair: linear temperature annealing --gate-tau -> --gate-tau-end over the first tau_steps optimizer steps
    tau_steps = c.gate_tau_steps or (c.max_steps if c.max_steps else 4000)
    anneal_tau = c.decoder == "molora" and c.gate == "prototype" and c.gate_tau_end > 0

    def set_tau(s):
        if not anneal_tau:
            return c.gate_tau
        f = min(max(s / max(tau_steps, 1), 0.0), 1.0)
        tau = c.gate_tau + f * (c.gate_tau_end - c.gate_tau)
        model.set_gate_tau(tau)
        return tau

    cur_tau = set_tau(step)
    if anneal_tau:
        emit(f"gate repair: annealing tau {c.gate_tau} -> {c.gate_tau_end} over {tau_steps} steps")

    def new_log_sums():
        # Match Python float (double) accumulation, without reading CUDA scalars.
        # All values added to these views must be detached from the training graph.
        return torch.zeros(9, device=dev, dtype=torch.float64).unbind()

    ent_sum, maxpi_sum, proto_sum, limoe_sum, commit_sum, active_sum, kept_sum, spec_sum, within_sum = new_log_sums()
    gate_sum, n_gate = torch.zeros(c.experts, device=dev), 0
    is_vq = c.decoder == "molora" and c.gate_mode == "vq"
    n_kept = 0
    # PREREG A8: cells actually passed through each expert over the WHOLE run (not the 20-step logging window),
    # so a within-cell-type code that stays under 1,000 trained cells can be reported as under-trained.
    trained_codes = torch.zeros(c.experts, dtype=torch.int64, device=dev)
    if c.resume:                                   # keep the count across a requeue / --auto-resume
        _tc = torch.load(c.resume, map_location="cpu", weights_only=False).get("trained_codes")
        if _tc is not None and len(_tc) == c.experts:
            trained_codes = torch.as_tensor(np.asarray(_tc, dtype=np.int64), device=dev).clone()
    marg_acc, marg_n = torch.zeros(c.experts, device=dev), 0        # soft pi marginal of the current optimizer step

    def code_group(ce, y, n_codes, half=None, min_half=8):
        """--code-grouped-batches: keep only the cells of the ``n_codes`` most frequent codes of this draw, so the
        sequence path runs ``n_codes`` experts instead of all K (deep experts make the cost of running all K
        prohibitive: K = 16 deep passes OOM an 80 GB H100). ce [1, n, d], y [1, bins, n], half [n] or None
        (the caller passes ``batch[4][0]``, i.e. the batch dimension is already squeezed off).
        Within-type batches are grouped too, and their within-type term is dropped (half set to -1) when grouping
        leaves either stage half with fewer than ``min_half`` cells."""
        with torch.no_grad():
            # the load used for the dead-code test and for codebook_utilisation is measured HERE, on the full draw:
            # grouping drops exactly the deliberately rare codes, so measuring it afterwards would mark them dead
            codes = model.observe_load(ce)
            if codes is None:
                codes = model.cell_state_to_gate.assign(ce.reshape(-1, ce.shape[-1]))[1]
        uniq, cnt = torch.unique(codes, return_counts=True)
        if c.code_group_mode == "mixed" and len(uniq) > n_codes:
            # "top" keeps the n_codes most frequent codes of the draw. With cells drawn uniformly from the training
            # set that is always the largest LINEAGE codes: cardiomyocytes are ~4.8% of the cells spread over 8
            # within-cell-type codes, so those codes get 0-2 cells per 64-cell draw and are never in the top n, i.e.
            # the very experts the union codebook exists for receive almost no gradient from the main loss.
            # "mixed" keeps the single most frequent code plus n_codes - 1 drawn UNIFORMLY over the other codes
            # present, so every code present is selected at rate ~1/|present| instead of ~0, at the same memory
            # (n_codes active experts).
            order = cnt.argsort(descending=True)
            top, rest = order[:1], order[1:]
            pick = rest[torch.randperm(len(rest), device=rest.device)[:n_codes - 1]]
            keep_codes = uniq[torch.cat([top, pick])]
        else:
            keep_codes = uniq[cnt.argsort(descending=True)[:n_codes]]
        m = torch.isin(codes, keep_codes)
        if int(m.sum()) < 2:
            return ce, y, half, 1.0
        h = None
        if half is not None:
            h = half[m.cpu()] if half.shape[-1] == m.shape[0] else half
            if min(int((h == 0).sum()), int((h == 1).sum())) < min_half:
                h = torch.full_like(h, -1)                 # not enough cells left: skip the within-type term
        return ce[:, m], y[:, :, m], h, m.float().mean()

    step_gate, n_step_gate, n_spec = torch.zeros(c.experts, device=dev), 0, 0
    n_within = 0
    done = False
    while not done:
        for j, batch in enumerate(dl):
            x, rc, y, ce = batch[:4]
            half = batch[4][0] if len(batch) > 4 else None
            k = sampler.start + j                                   # position in this epoch's permutation
            set_stabilise(step < c.warmup)
            x = x.permute(0, 2, 1).to(dev, non_blocking=True); y = y.to(dev, non_blocking=True); ce = ce.to(dev)
            if c.cfg_drop > 0:
                ce = with_null(ce, c.cfg_drop)
            if is_vq and c.code_grouped_batches:        # every batch, within-type ones included (deep K passes OOM)
                ce, y, half, kept = code_group(ce, y, c.code_grouped_batches, half)
                kept_sum += kept; n_kept += 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, ce)
            loss = poisson_multinomial_torch(out.float(), y.float(), total_weight=0.2) / c.accum
            main = loss.detach()
            if c.spec_weight > 0:
                ls = 1 - spec_r(out, y, c.spec_pool)
                loss = loss + c.spec_weight * (loss.detach() / ls.detach().clamp_min(1e-8)) * ls
                spec_sum += ls.detach(); n_spec += 1
            if c.within_weight > 0 and half is not None and bool((half >= 0).all()):
                lw = within_kl(out, y, half, c.within_pool)
                loss = loss + c.within_weight * (main / lw.detach().clamp_min(1e-6)) * lw
                within_sum += lw.detach(); n_within += 1
            if c.decoder == "molora":
                loss = loss + model.balance_loss() / c.accum
                if c.proto_sep > 0:                                   # gate repair: prototype separation penalty
                    lp = model.prototype_penalty()
                    loss = loss + lp / c.accum
                    proto_sum += lp.detach()
                if c.limoe_weight > 0:
                    # global-batch scope (Qiu et al. 2025): the marginal is the mean soft pi over ALL micro-batches of
                    # this optimizer step; only this micro-batch's share carries gradient, which is the correct partial
                    # derivative of the global term. Scaled as main.detach() x (H_cell - H_marg)/ln K -- the same
                    # stop-gradient-ratio convention as --spec-weight, but with ln K as the normaliser because the
                    # LIMoE term changes sign and passes through zero, so dividing by its own value is unusable.
                    pm = model.last_gate.reshape(-1, c.experts).float().mean(0)
                    lm = model.limoe_loss(marginal=(marg_acc + pm) / (marg_n + 1))
                    loss = loss + c.limoe_weight * main * lm / c.accum
                    marg_acc = marg_acc + pm.detach(); marg_n += 1
                    limoe_sum += lm.detach()
                if is_vq:
                    codes = model.last_codes.detach().reshape(-1)
                    # Fixed-size scatter avoids both a host copy and bincount's
                    # device-to-host synchronization to determine its output size.
                    code_counts = torch.zeros_like(trained_codes).scatter_add_(0, codes, torch.ones_like(codes))
                    active_sum += (code_counts > 0).sum()
                    trained_codes += code_counts
                    commit_sum += model.cell_state_to_gate.commitment().detach()
                g = (model.last_gate_hard if is_vq else model.last_gate).detach().reshape(-1, c.experts)
                gate_sum += g.mean(0); ent_sum += -(g * g.clamp_min(1e-9).log()).sum(1).mean(); n_gate += 1
                maxpi_sum += g.max(1).values.mean()
                step_gate += g.mean(0); n_step_gate += 1
            loss.backward()
            if (k + 1) % c.accum: continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            cur_tau = set_tau(step)
            if c.decoder == "molora":
                model.update_gate_bias(step_gate / max(n_step_gate, 1)); step_gate.zero_(); n_step_gate = 0
                marg_acc = torch.zeros_like(marg_acc); marg_n = 0
                if is_vq:
                    model.codebook_update(step)
            if step % 20 == 0:
                # One packed read per logging window, instead of scalar reads on
                # every micro-batch. The counters never participate in the loss.
                packed = torch.cat((torch.stack((ent_sum, maxpi_sum, proto_sum, limoe_sum, commit_sum,
                                                  active_sum, kept_sum, spec_sum, within_sum)),
                                    (gate_sum / max(n_gate, 1)).double(), model.gate_bias.double()
                                    if c.decoder == "molora" else torch.zeros_like(gate_sum, dtype=torch.float64)))
                values = packed.cpu().tolist()
                ent_sum, maxpi_sum, proto_sum, limoe_sum, commit_sum, active_sum, kept_sum, spec_sum, within_sum = \
                    values[:9]
                gate_values, bias_values = values[9:9 + c.experts], values[9 + c.experts:]
                msg = json.dumps(dict(step=step, loss=round(loss.item() * c.accum, 5), lr=sched.get_last_lr()[0],
                                      s_per_step=round((time.time() - t0) / 20, 2),
                                      gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                                      **(dict(gate=[round(v, 3) for v in gate_values],
                                              gate_entropy=round(ent_sum / max(n_gate, 1), 3),
                                              gate_maxpi=round(maxpi_sum / max(n_gate, 1), 3),
                                              gate_bias=[round(v, 3) for v in bias_values])
                                        if c.decoder == "molora" else {}),
                                      **(dict(gate_tau=round(model.gate_tau(), 4))
                                         if anneal_tau or (is_vq and c.learn_tau) else {}),
                                      **(dict(proto_sep=round(proto_sum / max(n_gate, 1), 6)) if c.proto_sep > 0 else {}),
                                      **(dict(limoe=round(limoe_sum / max(n_gate, 1), 4)) if c.limoe_weight > 0 else {}),
                                      **(dict(codebook_perplexity=round(model.cell_state_to_gate.perplexity(), 2),
                                              # utilisation = share of experts that received GRADIENT (the number that
                                              # matters for the rare within-cell-type experts); *_pop = share of codes
                                              # occupied in the full pre-grouping draws, i.e. the codebook's own spread
                                              codebook_utilisation=round(model.cell_state_to_gate.utilisation(), 3),
                                              codebook_utilisation_pop=round(model.cell_state_to_gate.utilisation_full(), 3),
                                              codebook_perplexity_pop=round(model.cell_state_to_gate.perplexity_full(), 2),
                                              dead_restarts=model.cell_state_to_gate.restarts,
                                              commit=round(commit_sum / max(n_gate, 1), 5),
                                              n_active=round(active_sum / max(n_gate, 1), 2),
                                              **(dict(kept_frac=round(kept_sum / max(n_kept, 1), 3)) if n_kept else {}))
                                        if is_vq else {}),
                                      **(dict(spec_loss=round(spec_sum / max(n_spec, 1), 4)) if c.spec_weight > 0 else {}),
                                      **(dict(within_kl=round(within_sum / max(n_within, 1), 5), within_n=n_within)
                                         if c.within_weight > 0 else {})))
                if c.decoder == "molora" and ent_sum / max(n_gate, 1) > 0.9 * math.log(c.experts):
                    emit(f"WARNING step {step}: per-cell gate entropy {ent_sum / max(n_gate, 1):.3f} > 0.9 ln K "
                         f"-- gate near uniform")
                gate_sum.zero_(); n_gate, n_spec, n_within, n_kept = 0, 0, 0, 0
                ent_sum, maxpi_sum, proto_sum, limoe_sum, commit_sum, active_sum, kept_sum, spec_sum, within_sum = \
                    new_log_sums()
                if is_vq:                      # perplexity / utilisation over the window just logged, not since step 0
                    model.cell_state_to_gate.reset_load()
                emit(msg); logf.write(msg + "\n"); logf.flush(); t0 = time.time()
            if step % c.eval_every == 0 or step == total:
                # Save completed optimizer work before validation can fail.
                state = {"model": trainable_state(model),
                         "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "best": best,
                         "best_step": best_step,
                         "schedule_config": schedule_config,
                         "emb_dim": E.shape[1], "molora": vars(c), "epoch": sampler.epoch, "pos": k + 1,
                         # PREREG A8 (VQ gate only; zeros otherwise)
                         "trained_codes": trained_codes.cpu().numpy().copy(),
                         "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}
                torch.save(state, f"{c.out}/last.pt.tmp")
                os.replace(f"{c.out}/last.pt.tmp", f"{c.out}/last.pt")            # atomic: preemption-safe
                r_all, r_cell, r_spec, r_null = validate()
                msg = json.dumps(dict(step=step, val_pearson_log=r_all, val_pearson_per_cell=r_cell, val_spec_r=r_spec,
                                      **(dict(val_pearson_null=r_null) if r_null is not None else {})))
                emit(msg); logf.write(msg + "\n"); logf.flush()
                state.update(best=max(best, r_all),
                             best_step=step if r_all > best + c.min_delta else best_step)
                torch.save(state, f"{c.out}/last.pt.tmp"); os.replace(f"{c.out}/last.pt.tmp", f"{c.out}/last.pt")
                if c.save_every and step % c.save_every == 0:
                    torch.save({x: state[x] for x in ("model", "step", "best", "emb_dim", "molora")},
                               f"{c.out}/step{step}.pt")
                if r_all > best + c.min_delta:
                    best_step = step
                if r_all > best:
                    best = r_all
                    torch.save(state, f"{c.out}/best.pt.tmp"); os.replace(f"{c.out}/best.pt.tmp", f"{c.out}/best.pt")
                if c.patience and step - best_step >= c.patience * c.eval_every:
                    emit(f"early stop at step {step}: no new best (> {c.min_delta}) for {c.patience} evaluations "
                         f"since step {best_step}")
                    stopped_early = done = True; break
            if step >= total:
                done = True; break
        else:                                                       # epoch exhausted: next permutation from its start
            sampler.epoch, sampler.start = sampler.epoch + 1, 0
    emit("finished", step, "steps; best val", best)
    if is_vq:       # PREREG A8: per-expert trained-cell counts over the whole run
        trained_codes = trained_codes.cpu().numpy()
        pd.DataFrame(dict(code=np.arange(c.experts), trained_cells=trained_codes,
                          under_trained=trained_codes < 1000)).to_csv(f"{c.out}/trained_codes.tsv", sep="\t",
                                                                      index=False)
        emit("trained cells per code:", trained_codes.tolist())
    open(f"{c.out}/DONE", "w").write(f"{step} steps; best val {best}"
                                     + (f"; early stop (patience {c.patience}, best step {best_step})"
                                        if stopped_early else "") + "\n")


def main(argv=None):
    """The script's command line, flag for flag."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke test / budget)")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cellstate-lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping: stop after this many consecutive evaluations without a new best validation "
                         "Pearson (more than --min-delta above the best); 0 = off. Resumable (best step kept in the "
                         "checkpoint)")
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--val-seqs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--resume-restart-schedule", action="store_true",
                    help="continue a completed budget with preserved optimizer moments and a fresh LR schedule over "
                         "the remaining steps")
    ap.add_argument("--resume-warmup", type=int, default=100,
                    help="warmup steps for an explicit resumed LR restart; original warmup is unchanged")
    ap.add_argument("--embedding", default=None,
                    help="cell embedding parquet (obs_names, embedding); default <root>/gse205117/scooby/embedding.parquet")
    ap.add_argument("--extra-cells", default=None,
                    help="TSV (index = <sample>#<barcode>) of training cells from OTHER datasets, appended to "
                         "rna/cells.tsv.gz behind the same global cell index; columns sample, celltype, frag "
                         "(path to that sample's tabix-indexed fragments file). Default off: without it the cell "
                         "table, the reader and the embedding rows are exactly as before")
    ap.add_argument("--auto-resume", action="store_true",
                    help="resume from <out>/last.pt when it exists (preemptible / requeued jobs); exit at once if "
                         "<out>/DONE")
    ap.add_argument("--data-seed", type=int, default=0, help="training order = permutation from (data seed, epoch)")
    ap.add_argument("--decoder", choices=["molora", "film", "scooby"], default="molora")
    ap.add_argument("--backbone", choices=["borzoi", "flashzoi", "alphagenome"], default="borzoi")
    ap.add_argument("--film-hidden", type=int, default=256)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--expert-blocks", type=int, default=2)
    ap.add_argument("--low-expert-blocks", type=int, default=0,
                    help="lower-trunk conditioning: transformer blocks below the expert part that also get expert LoRA "
                         "(zero-initialised; the lower trunk then runs K times per window)")
    ap.add_argument("--expert-conv-tower", action="store_true",
                    help="lower-trunk conditioning: expert Locon on conv_dna / res_tower / unet1 / horizontal convs "
                         "(zero-initialised; the whole trunk then runs K times per window)")
    ap.add_argument("--balance", type=float, default=0.01)
    ap.add_argument("--gate", choices=["prototype", "mlp"], default="prototype")
    ap.add_argument("--gate-tau", type=float, default=0.1)
    # ---- gate repair (all off by default; see the "Gate repair" paragraph of epione/models/aceso/_molora.py)
    ap.add_argument("--gate-topk", type=int, default=0,
                    help="hard top-k routing with a straight-through estimator instead of the dense softmax (0 = dense)")
    ap.add_argument("--proto-sep", type=float, default=0.0,
                    help="weight of the prototype separation penalty (keeps the K prototypes from collapsing)")
    ap.add_argument("--proto-sep-margin", type=float, default=0.0,
                    help="cosine above which a prototype pair starts to be penalised")
    ap.add_argument("--gate-tau-end", type=float, default=0.0,
                    help="anneal the prototype gate temperature linearly from --gate-tau to this over --gate-tau-steps "
                         "optimizer steps (0 = no annealing)")
    ap.add_argument("--gate-tau-steps", type=int, default=0,
                    help="length of the temperature annealing in optimizer steps (0 = --max-steps, else 4000)")
    ap.add_argument("--reinit-prototypes", action="store_true",
                    help="after --init-from, re-run the k-means prototype initialisation on the training cell states")
    ap.add_argument("--reset-gate-bias", action="store_true",
                    help="after --init-from, reset the load-balancing gate bias to zero")
    ap.add_argument("--codebook-init", default=None,
                    help="after --init-from, load the K x d codebook from this .npy (takes precedence over "
                         "--reinit-prototypes; used for the union codebook of paper/fig4/sharpness, whose K-means "
                         "scope was chosen by the CPU protocol s8/s8b/s8c)")
    # ---- VQ-MoLoRA (P1 of paper/fig4/sharpness/ARCHITECTURE_SURVEY.md; all off by default)
    ap.add_argument("--affine-scale", type=float, default=1.0,
                    help="gate-mode affine: scale on the head output before the 1/K offset")
    ap.add_argument("--gate-mode", choices=["softmax", "vq", "affine"], default="softmax",
                    help="vq: one hard code per cell (argmax cosine), straight-through; only the codes present in a "
                         "batch are run, so K > 4 stays affordable")
    ap.add_argument("--codebook-ema", type=float, default=0.99,
                    help="EMA decay of the VQ codebook update (0 = train the codes by gradient instead)")
    ap.add_argument("--commit-weight", type=float, default=0.0, help="VQ commitment term beta")
    ap.add_argument("--dead-code-restart", type=int, default=0,
                    help="restart codes whose load is below --dead-code-thresh every N optimizer steps (0 = off)")
    ap.add_argument("--dead-code-thresh", type=float, default=0.02)
    ap.add_argument("--learn-tau", action="store_true", help="learnable gate log-temperature (clamped to [0.01, 1])")
    ap.add_argument("--rotation-trick", action="store_true",
                    help="Fifty et al. (ICLR 2025) rotation+rescale instead of plain straight-through")
    ap.add_argument("--code-grouped-batches", type=int, default=0,
                    help="VQ: keep only the cells of N codes of each sampled batch, so the sequence path runs N "
                         "experts instead of all K (0 = off)")
    ap.add_argument("--code-group-mode", choices=["top", "mixed"], default="top",
                    help="which N codes --code-grouped-batches keeps: 'top' = the N most frequent of the draw "
                         "(default, but with uniform cell sampling those are always the large lineage codes, so the "
                         "rare within-cell-type experts get almost no gradient); 'mixed' = the most frequent code "
                         "plus N-1 drawn uniformly over the other codes present in the draw")
    ap.add_argument("--grow-experts", action="store_true",
                    help="--init-from with more experts than the checkpoint: tile the checkpoint's experts so new "
                         "expert j starts as a copy of old expert j %% K_old (use with --reinit-prototypes)")
    ap.add_argument("--limoe-weight", type=float, default=0.0,
                    help="LIMoE entropy pair (minimise per-cell routing entropy, maximise the batch-marginal entropy) "
                         "over the whole optimizer step, in place of the load-balancing bias")
    ap.add_argument("--balance-mode", choices=["aux", "bias"], default="aux")
    ap.add_argument("--bias-rate", type=float, default=1e-3)
    ap.add_argument("--spec-weight", type=float, default=0.0)
    ap.add_argument("--spec-pool", type=int, default=32)
    ap.add_argument("--cfg-drop", type=float, default=0.0)
    ap.add_argument("--within-weight", type=float, default=0.0)
    ap.add_argument("--within-frac", type=float, default=0.5)
    ap.add_argument("--within-pool", type=int, default=32)
    ap.add_argument("--within-min-cells", type=int, default=32)
    ap.add_argument("--within-axes-stage-share", type=float, default=-1.0,
                    help="R3b: probability that a within item uses the stage triples; the remaining mass is split "
                         "uniformly over the --within-axes groups. Default -1 = uniform over all groups (R3 arm AX)")
    ap.add_argument("--within-axes", default="",
                    help="R3 (conditioning_pilot/r3_axis_contrast): <tsv>:<col>[,<col>...]; adds within-type contrast "
                         "triples along each named RNA axis (per-type median split, low vs high) next to the stage "
                         "triples; the axis is drawn uniformly, then a triple of that axis")
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--save-every", type=int, default=0)
    a = ap.parse_args(argv)
    return train_v1_decoder(V1DecoderConfig(**vars(a)))


if __name__ == "__main__":  # pragma: no cover
    main()
