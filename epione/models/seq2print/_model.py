"""The seq2PRINT model: DNA -> multi-scale footprints + coverage.

seq2PRINT (scPrinter, Buenrostro lab) is a dilated CNN in the
ChromBPNet/BPNet family that reads 1,840 bp of sequence and predicts, for
the central 800 bp, PRINT's footprint z-score at each of 99 scales, plus
the log total insertions of the 1 kb window. Fine-tuned per cell state
with LoRA, it becomes a single-cell sequence-to-footprint model.

This module assembles it. The building blocks are scPrinter's own
(:mod:`._modules`, vendored verbatim) and the assembled model has
scPrinter's module layout, so a ``state_dict`` saved by either side loads
into the other and the forward pass is identical. What changed is the
assembly:

* :func:`build_seq2print` replaces scPrinter's JSON-config route
  (``seq_model_config`` -> ``construct_model_from_config``) with keyword
  arguments whose defaults are scPrinter's published configuration.
* :meth:`Seq2PRINT.add_lora` attaches cell-state LoRA in one loop instead
  of five copy-pasted blocks, in the same order and with the same
  embedding rescaling, so the result is still state-dict compatible.
"""
from __future__ import annotations

import copy
import math
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch import nn

from ._modules import (
    BiasAdjustedFootprintsHead,
    Conv1dLoRA,
    Conv1dWrapper,
    DilatedCNN,
    DNA_CNN,
    Footprints_head,
)

__all__ = ["LORA_TARGETS", "Seq2PRINT", "build_seq2print"]

#: Number of footprint scales seq2PRINT predicts (radii 2..100).
N_SCALES = 99

#: Layers LoRA can wrap, in the order scPrinter wraps them. The order
#: matters: each wrap rescales the shared cell embedding.
LORA_TARGETS = ("dna_cnn", "dilated", "pff", "profile", "count")


def _rescale_embedding(model, rank: int, A_embedding, B_embedding) -> float:
    """scPrinter ``adjust_embedding_scale``: rescale the shared cell
    embedding so the LoRA update ``A @ B`` starts at unit scale.

    Note this mutates the embedding every time a layer is wrapped, so its
    final scale depends on how many layers were wrapped -- scPrinter's
    behaviour, kept for compatibility.
    """
    emb, cov = model.embeddings, model.coverages
    with torch.no_grad():
        emb.eval(); A_embedding.eval(); B_embedding.eval()
        n = min(100, emb.weight.shape[0])
        cells = emb(torch.arange(n).long())
        if model.coverage_in_lora:
            cells = torch.cat([cells, cov(torch.arange(n).long())], dim=-1)
        A = A_embedding(cells).reshape((n, -1, rank))
        B = B_embedding(cells).reshape((n, rank, -1))
        std = torch.bmm(A, B).std()
        factor = math.sqrt(1 / std)
        emb.weight.data[...] *= factor
        if model.coverage_in_lora:
            cov.weight.data[...] *= factor
    return factor


class Seq2PRINT(nn.Module):
    """seq2PRINT. Build with :func:`build_seq2print`.

    Attributes (scPrinter's names -- they are the state-dict keys)
    ----------------------------------------------------------------
    dna_cnn_model, hidden_layer_model, profile_cnn_model
        First convolution, dilated tower, footprint/count head.
    dna_len, output_len
        Input sequence length and predicted window length.
    embeddings, coverages
        Frozen per-cell-state embedding / coverage covariates once
        :meth:`add_lora` has been called, else ``None``.
    """

    def __init__(self, dna_cnn_model, hidden_layer_model, profile_cnn_model,
                 dna_len: int, output_len: int) -> None:
        super().__init__()
        self.dna_cnn_model = dna_cnn_model
        self.hidden_layer_model = hidden_layer_model
        self.profile_cnn_model = profile_cnn_model
        self.dna_len = int(dna_len)
        self.output_len = int(output_len)
        self.embeddings = None
        self.coverages = None
        self.coverage_in_lora = False

    # -- LoRA --------------------------------------------------------------

    def add_lora(
        self,
        embeddings: np.ndarray,
        *,
        coverages: Optional[np.ndarray] = None,
        targets: Iterable[str] = LORA_TARGETS,
        rank: int = 32,
        hidden_dim: Optional[int] = 256,
        n_lora_layers: int = 0,
        coverage_in_lora: bool = False,
        freeze_backbone: bool = True,
    ) -> "Seq2PRINT":
        """Condition the model on cell state with LoRA.

        Each targeted convolution gets a low-rank weight update produced
        by a small network from the cell state's embedding, so one set
        of backbone weights serves every cell state.

        Parameters
        ----------
        embeddings
            ``[n_states, d]`` cell-state embedding (e.g. mean LSI of each
            group). Frozen.
        coverages
            ``[n_states, k]`` coverage covariates (scPrinter uses z-scored
            log depth genome-wide and in peaks). When given, the head is
            wrapped so predictions can absorb depth differences.
        targets
            Which of :data:`LORA_TARGETS` to wrap. Default: all, as
            scPrinter does.
        rank, hidden_dim, n_lora_layers
            LoRA rank and the width/depth of the embedding -> LoRA
            network. Defaults are scPrinter's (32, 256, 0). The count
            head always uses rank 1, as in scPrinter.
        coverage_in_lora
            Also feed the coverages into the LoRA networks.
        freeze_backbone
            Freeze every existing weight before wrapping, so only the
            LoRA networks (and the coverage adjustment) train -- what
            scPrinter does when fine-tuning a bulk model per cell state.
        """
        targets = tuple(targets)
        unknown = set(targets) - set(LORA_TARGETS)
        if unknown:
            raise ValueError(f"unknown LoRA targets {sorted(unknown)}; "
                             f"choose from {LORA_TARGETS}")
        if self.embeddings is not None:
            raise RuntimeError("LoRA is already attached")
        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

        if coverages is not None:
            self.coverages = nn.Embedding(*coverages.shape)
            self.coverages.weight.data = torch.from_numpy(np.asarray(coverages)).float()
            self.coverages.weight.requires_grad = False
        self.embeddings = nn.Embedding(*embeddings.shape)
        self.embeddings.weight.data = torch.from_numpy(np.asarray(embeddings)).float()
        self.embeddings.weight.requires_grad = False
        self.coverage_in_lora = bool(coverage_in_lora)
        dim = embeddings.shape[-1] + (coverages.shape[-1] if coverage_in_lora else 0)

        def wrap(owner, attr, r):
            layer = Conv1dLoRA(getattr(owner, attr), A_embedding_dim=dim,
                               B_embedding_dim=dim, r=r, hidden_dim=hidden_dim,
                               n_layers=n_lora_layers)
            setattr(owner, attr, layer)
            _rescale_embedding(self, r, layer.A_embedding, layer.B_embedding)
            layer.reset_B()

        # Same order as scPrinter: dna conv, then per hidden layer (conv1,
        # conv2), then the footprint head, then the count head.
        if "dna_cnn" in targets:
            wrap(self.dna_cnn_model, "conv", rank)
        for block in self.hidden_layer_model.layers:
            if "dilated" in targets:
                wrap(block.module, "conv1", rank)
            if "pff" in targets:
                wrap(block.module, "conv2", rank)
        if "profile" in targets:
            wrap(self.profile_cnn_model, "conv_layer", rank)
        if "count" in targets:
            wrap(self.profile_cnn_model, "linear", 1)

        if self.coverages is not None:
            self.profile_cnn_model = BiasAdjustedFootprintsHead(
                self.profile_cnn_model, self.coverages.weight.data.shape[1])
        return self

    @property
    def n_states(self) -> int:
        return 0 if self.embeddings is None else self.embeddings.weight.shape[0]

    def collapse(self, state: int) -> "Seq2PRINT":
        """A plain (LoRA-free) copy of the model for one cell state.

        Every LoRA layer is replaced by a convolution whose weight already
        includes that state's low-rank update, and a coverage-adjusted
        head by its collapsed form -- scPrinter's ``seq2PRINT.collapse``.
        The copy predicts what the LoRA model predicts for ``state`` but
        takes no ``cells`` argument, which is what attribution methods
        need. Gradients are switched on for all its parameters, as in
        scPrinter.
        """
        if self.embeddings is None:
            raise ValueError("collapse needs a LoRA model (call add_lora first)")
        clone = copy.deepcopy(self)
        dev = self.embeddings.weight.device
        cell = torch.tensor([int(state)], dtype=torch.long, device=dev)
        with torch.no_grad():
            A = B = self.embeddings(cell)
            cov = self.coverages(cell) if self.coverages is not None else None
            if self.coverage_in_lora:
                A = B = torch.cat([A, cov], dim=-1)

        def fold(owner, attr):
            layer = getattr(owner, attr)
            if isinstance(layer, Conv1dLoRA):
                setattr(owner, attr, layer.collapse_layer(A, B))

        fold(clone.dna_cnn_model, "conv")
        for block in clone.hidden_layer_model.layers:
            fold(block.module, "conv1")
            fold(block.module, "conv2")
        head = clone.profile_cnn_model
        if isinstance(head, BiasAdjustedFootprintsHead):
            head.collapse_layer(cov)
            head = head.footprints_head
        fold(head, "conv_layer")
        fold(head, "linear")
        clone.embeddings = clone.coverages = None
        clone.coverage_in_lora = False
        for p in clone.parameters():
            p.requires_grad = True
        return clone

    # -- forward -------------------------------------------------------------

    def forward(self, X, cells=None, output_len=None, modes=None):
        """Predict footprints and coverage.

        Parameters
        ----------
        X
            ``[batch, 4, dna_len]`` one-hot sequence (ACGT).
        cells
            ``[batch]`` cell-state indices; required once LoRA is attached.
        output_len
            Width to predict; default :attr:`output_len`.
        modes
            Indices (0..98, i.e. radius - 2) of the scales to predict;
            default all.

        Returns
        -------
        ``(footprints [batch, n_modes, output_len], log_coverage [batch])``
        """
        if output_len is None:
            output_len = self.output_len
        A_cells = B_cells = coverages = None
        if cells is not None and self.embeddings is not None:
            A_cells = self.embeddings(cells)
            B_cells = self.embeddings(cells)
            if self.coverages is not None:
                coverages = self.coverages(cells)
            if self.coverage_in_lora:
                A_cells = torch.cat([A_cells, coverages], dim=-1)
                B_cells = torch.cat([B_cells, coverages], dim=-1)
        X = self.dna_cnn_model(X, A_cells=A_cells, B_cells=B_cells)
        X = self.hidden_layer_model(X, A_cells=A_cells, B_cells=B_cells)
        return self.profile_cnn_model(
            X, A_cells=A_cells, B_cells=B_cells, coverages=coverages,
            output_len=output_len, modes=modes,
        )


def build_seq2print(
    *,
    n_filters: int = 1024,
    n_layers: int = 8,
    kernel_size: int = 3,
    head_kernel_size: int = 1,
    groups: int = 8,
    bottleneck_factor: float = 1.0,
    dilation_base: int = 1,
    activation: str = "gelu",
    batch_norm: bool = True,
    batch_norm_momentum: float = 0.1,
    rezero: bool = False,
    n_inception_layers: Optional[int] = None,
    inception_layers_after: bool = True,
    inception_version: int = 2,
    output_len: int = 800,
) -> Seq2PRINT:
    """Build a seq2PRINT model.

    Defaults are scPrinter's published configuration (``seq_model_config``
    in ``scprinter/tools.py``): 1,024 filters, 8 dilated inception-v2
    blocks with dilations 2..256, GELU, batch norm, 800 bp predicted from
    1,840 bp of sequence.

    Returns
    -------
    :class:`Seq2PRINT`, with ``.dna_len`` set to the input length the
    receptive field implies.
    """
    act = {"gelu": nn.GELU(), "relu": nn.ReLU()}.get(activation)
    if act is None:
        raise ValueError("activation must be 'gelu' or 'relu'")
    if n_inception_layers is None:
        n_inception_layers = n_layers
    inception = ([False] * (n_layers - n_inception_layers) + [True] * n_inception_layers
                 if inception_layers_after else
                 [True] * n_inception_layers + [False] * (n_layers - n_inception_layers))
    dilation = lambda i: 2 ** (i + dilation_base)  # noqa: E731

    dna_cnn = DNA_CNN(n_filters=n_filters)
    hidden = DilatedCNN(
        n_filters=n_filters, bottleneck=int(n_filters * bottleneck_factor),
        n_layers=n_layers, kernel_size=kernel_size, groups=groups,
        activation=act, batch_norm=batch_norm, residual=True, rezero=rezero,
        dilation_func=dilation, batch_norm_momentum=batch_norm_momentum,
        inception=inception, inception_version=inception_version,
    )
    head = Footprints_head(n_filters, kernel_size=head_kernel_size, n_scales=N_SCALES)

    dna_len = output_len + dna_cnn.conv.weight.shape[2] - 1
    for i in range(n_layers):
        dna_len += 2 * (kernel_size // 2) * dilation(i)
    return Seq2PRINT(dna_cnn, hidden, head, dna_len=dna_len, output_len=output_len)
