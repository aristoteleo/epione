"""BPNet and ChromBPNet networks in PyTorch.

A line-by-line translation of the official Keras definitions in
kundajelab/chrombpnet (MIT licence, Copyright (c) 2022 Kundaje Lab):
``chrombpnet/training/models/bpnet_model.py`` and
``chrombpnet_with_bias_model.py`` at commit ``09938fd``.

Layout and conventions follow the Keras model exactly, so official ``.h5``
weights map onto these modules one to one (see :mod:`._keras`):

* input is one-hot DNA, ``(N, 4, L)`` here (torch is channels-first;
  Keras is ``(N, L, 4)``), bases ordered A, C, G, T;
* a 21-bp ``valid`` convolution, then ``n_dil_layers`` residual blocks of a
  3-bp ``valid`` convolution with dilation ``2**i`` (i = 1..n) whose input
  is centre-cropped to the block output before the addition;
* profile head: one 75-bp ``valid`` convolution, centre-cropped to
  ``output_len`` logits;
* counts head: global average pool over length, then a dense layer giving
  ``log(1 + total counts)``.

ChromBPNet adds a frozen bias model to an accessibility ("no-bias") model:
profile logits add, and the log-counts combine with ``logsumexp``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

__all__ = ["BPNet", "ChromBPNet", "glorot_init_"]

#: Official defaults (``bpnet_model.py``).
CONV1_KERNEL = 21
PROFILE_KERNEL = 75


def glorot_init_(module: nn.Module) -> nn.Module:
    """Keras initialisation: Glorot-uniform kernels, zero biases.

    PyTorch's default (Kaiming-uniform with a fan-in bias) differs; training
    from scratch uses the Keras one so the starting point matches upstream.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module


def _center_crop(x: torch.Tensor, length: int) -> torch.Tensor:
    """Keras ``Cropping1D(((L - length) // 2,) * 2)`` on a ``(N, C, L)``
    tensor. The callers assert the difference is even, as upstream does."""
    trim = (x.shape[-1] - length) // 2
    return x[..., trim:trim + length] if trim else x


class BPNet(nn.Module):
    """The BPNet architecture used for both ChromBPNet sub-models.

    Parameters
    ----------
    filters
        Channels in every convolution (official: 128 for the bias model,
        512 for the accessibility model).
    n_dil_layers
        Number of dilated residual blocks (official: 4 bias, 8 ChromBPNet).
    input_len, output_len
        Sequence length in and profile length out (official 2114 and 1000).
    precise_profile
        Run the 75-bp profile convolution in float64. It is the widest
        reduction in the network; on CPU, float32 there accounts for nearly
        all the difference from the official Keras outputs. ``'auto'``
        (default): float64 on CPU, native precision on GPU, where both
        TensorFlow and PyTorch run convolutions in TF32 by default and a
        float64 pass costs ~70% of the training step for no parity gain.
    """

    def __init__(
        self,
        filters: int = 512,
        n_dil_layers: int = 8,
        input_len: int = 2114,
        output_len: int = 1000,
        conv1_kernel_size: int = CONV1_KERNEL,
        profile_kernel_size: int = PROFILE_KERNEL,
        precise_profile="auto",
    ) -> None:
        super().__init__()
        if precise_profile not in (True, False, "auto"):
            raise ValueError("precise_profile must be True, False or 'auto'")
        self.precise_profile = precise_profile
        self.filters = int(filters)
        self.n_dil_layers = int(n_dil_layers)
        self.input_len = int(input_len)
        self.output_len = int(output_len)

        self.conv1 = nn.Conv1d(4, filters, conv1_kernel_size)
        self.relu1 = nn.ReLU()
        self.dil_convs = nn.ModuleList(
            nn.Conv1d(filters, filters, 3, dilation=2 ** i)
            for i in range(1, n_dil_layers + 1)
        )
        # One ReLU module per layer (not a shared/functional one) so that
        # DeepLIFT implementations can attach per-layer hooks.
        self.dil_relus = nn.ModuleList(nn.ReLU() for _ in range(n_dil_layers))
        self.prof_conv = nn.Conv1d(filters, 1, profile_kernel_size)
        self.count_dense = nn.Linear(filters, 1)

        # Shape checks the Keras model makes with asserts.
        L = self.input_len - (conv1_kernel_size - 1)
        for i in range(1, n_dil_layers + 1):
            L_new = L - 2 * 2 ** i
            if (L - L_new) % 2:
                raise ValueError("asymmetric crop in dilated stack")
            L = L_new
        prof_len = L - (profile_kernel_size - 1)
        if prof_len % 2 or prof_len < self.output_len:
            raise ValueError(
                f"input_len={input_len} gives a {prof_len}-bp profile; need an "
                f"even length >= output_len={output_len}"
            )
        self._prof_len = prof_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``x``: ``(N, 4, input_len)`` one-hot.

        Returns ``(profile_logits (N, output_len), log_counts (N, 1))``,
        the two outputs of the Keras model in the same order.
        """
        h = self.relu1(self.conv1(x))
        for conv, relu in zip(self.dil_convs, self.dil_relus):
            c = relu(conv(h))
            h = c + _center_crop(h, c.shape[-1])
        precise = (self.precise_profile is True
                   or (self.precise_profile == "auto" and h.device.type == "cpu"))
        if precise:
            # The profile head sums 75 x filters products per output base;
            # in float32 that alone adds ~1e-4 to the logits on CPU, ~30x
            # the Keras/Eigen rounding (see tests/parity/chrombpnet). One
            # output channel makes a float64 pass cheap next to the trunk.
            w = self.prof_conv.weight
            prof = torch.nn.functional.conv1d(
                h.double(), w.double(), self.prof_conv.bias.double()).to(h.dtype)
        else:
            prof = self.prof_conv(h)  # (N, 1, prof_len)
        # Keras: cropsize = int(L/2) - int(out/2), applied on both sides.
        crop = prof.shape[-1] // 2 - self.output_len // 2
        prof = prof[:, 0, crop:crop + self.output_len]
        counts = self.count_dense(h.mean(dim=-1))
        return prof, counts

    @property
    def receptive_field(self) -> int:
        return self.input_len - self._prof_len + 1


class ChromBPNet(nn.Module):
    """Bias-factorised ChromBPNet: frozen ``bias`` + trainable ``accessibility``.

    ``forward`` returns the full model's outputs (what the official
    ``model.chrombpnet.*.h5`` predicts). Use :attr:`accessibility` alone for
    bias-corrected predictions, as upstream does with ``chrombpnet_nobias``.
    """

    def __init__(self, bias: BPNet, accessibility: BPNet) -> None:
        super().__init__()
        if bias.output_len != accessibility.output_len:
            raise ValueError("bias and accessibility models need the same output_len")
        if bias.input_len != accessibility.input_len:
            raise ValueError("bias and accessibility models need the same input_len")
        self.bias = bias
        self.accessibility = accessibility
        self.freeze_bias()

    def freeze_bias(self) -> None:
        for p in self.bias.parameters():
            p.requires_grad_(False)

    @property
    def input_len(self) -> int:
        return self.accessibility.input_len

    @property
    def output_len(self) -> int:
        return self.accessibility.output_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        acc_prof, acc_counts = self.accessibility(x)
        bias_prof, bias_counts = self.bias(x)
        prof = acc_prof + bias_prof
        counts = torch.logsumexp(torch.cat([acc_counts, bias_counts], dim=-1),
                                 dim=-1, keepdim=True)
        return prof, counts

    def train(self, mode: bool = True):
        # The bias model stays in eval mode and frozen, as in upstream
        # (``layers[i].trainable = False``).
        super().train(mode)
        self.bias.eval()
        return self


def build_bpnet(kind: str = "chrombpnet", **kwargs) -> BPNet:
    """BPNet with the official defaults for ``kind`` = ``'bias'`` (128 x 4)
    or ``'chrombpnet'`` (512 x 8); keyword arguments override them."""
    defaults = {"bias": dict(filters=128, n_dil_layers=4),
                "chrombpnet": dict(filters=512, n_dil_layers=8)}[kind]
    defaults.update(kwargs)
    return BPNet(**defaults)


def one_hot(seqs, dtype=torch.float32) -> torch.Tensor:
    """One-hot encode strings to ``(N, 4, L)`` with upstream's convention:
    A, C, G, T channels; any other character (N) is all zeros."""
    import numpy as np

    from ._data import one_hot_encode
    return torch.as_tensor(np.transpose(one_hot_encode(seqs), (0, 2, 1)), dtype=dtype)


def predict(model: nn.Module, x, *, batch_size: int = 64, device: Optional[str] = None
            ) -> Tuple["np.ndarray", "np.ndarray"]:  # noqa: F821
    """Run ``model`` over one-hot ``x`` (``(N, 4, L)`` array/tensor or a
    list of strings) in batches; returns numpy ``(profile_logits, log_counts)``."""
    import numpy as np

    if isinstance(x, (list, tuple)) and x and isinstance(x[0], str):
        x = one_hot(x)
    x = torch.as_tensor(x, dtype=torch.float32)
    dev = torch.device(device) if device else next(model.parameters()).device
    was_training = model.training
    model.eval()
    profs, counts = [], []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            p, c = model(x[i:i + batch_size].to(dev))
            profs.append(p.float().cpu().numpy())
            counts.append(c.float().cpu().numpy())
    model.train(was_training)
    return np.concatenate(profs), np.concatenate(counts)
