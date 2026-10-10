"""Loading upstream scooby models reliably.

The model itself is upstream's — this module only constructs it, because
doing so correctly is not obvious under current dependency versions.

``Scooby.from_pretrained`` inherits transformers' meta-device
initialisation: the model is built on the ``meta`` device and then
populated from the checkpoint's state dict. Borzoi's relative-position
tables are registered with ``persistent=False``, so they are absent from
every checkpoint and nothing materialises them. They stay on ``meta``,
and the next ``.to(device)`` fails with

    NotImplementedError: Cannot copy out of meta tensor; no data!

Neither ``low_cpu_mem_usage=False`` nor ``_fast_init=False`` avoids it
(checked against transformers 4.56.2 / borzoi-pytorch 0.4.2). The
tables are deterministic — ``get_positional_embed(4096,
num_rel_pos_features)`` — so :func:`materialize_meta_buffers` simply
recomputes them, which is version-independent in a way that pinning
transformers would not be.
"""
from __future__ import annotations

from typing import Any, Optional

__all__ = ["load_scooby", "materialize_meta_buffers"]

# Borzoi hard-codes this sequence length when building the table; see
# ``pytorch_borzoi_transformer.Attention.__init__``.
_REL_POS_SEQ_LEN = 4096


def materialize_meta_buffers(model, device: Optional[Any] = None) -> int:
    """Recompute Borzoi positional buffers left on the meta device.

    Parameters
    ----------
    model
        A Borzoi-derived module, possibly already wrapped by PEFT.
    device
        Device to build the tables on. Defaults to CPU; the caller
        usually moves the whole model afterwards.

    Returns
    -------
    int
        Number of buffers rebuilt. Zero means the load already produced
        a complete model and nothing was needed.
    """
    import torch
    from borzoi_pytorch.pytorch_borzoi_transformer import get_positional_embed

    device = torch.device(device) if device is not None else torch.device("cpu")
    n = 0
    for module in model.modules():
        buf = module._buffers.get("positions")
        if buf is None or not buf.is_meta:
            continue
        features = getattr(module, "num_rel_pos_features", None)
        if features is None:
            raise RuntimeError(
                "found a meta 'positions' buffer on a module without "
                f"num_rel_pos_features ({type(module).__name__}); the "
                "upstream attention layout has changed and this helper "
                "needs updating"
            )
        module._buffers["positions"] = get_positional_embed(
            _REL_POS_SEQ_LEN, features, device
        )
        n += 1

    remaining = [name for name, b in model.named_buffers() if b.is_meta]
    if remaining:
        raise RuntimeError(
            f"{len(remaining)} buffer(s) still on meta after "
            f"rematerialisation, e.g. {remaining[:3]}"
        )
    return n


def load_scooby(
    checkpoint: str,
    *,
    cell_emb_dim: int,
    n_tracks: int = 1,
    embedding_dim: int = 1920,
    lora: bool = True,
    train: bool = True,
    device: Optional[Any] = None,
    **kwargs,
):
    """Build an upstream ``Scooby`` on a pretrained Borzoi backbone.

    Parameters
    ----------
    checkpoint
        A Borzoi checkpoint to start from, e.g.
        ``'johahi/borzoi-replicate-0'``, or a scooby checkpoint to
        resume. The cell-state head is newly initialised either way
        unless the checkpoint already carries one of matching shape.
    cell_emb_dim
        Width of the cell embedding the decoder is conditioned on. Must
        match the embedding passed to the dataset.
    n_tracks
        Output tracks per cell: 1 for ATAC alone, 3 for upstream's
        multiome layout (RNA +, RNA -, ATAC).
    embedding_dim
        Borzoi trunk width. 1920 for the released checkpoints.
    lora
        Apply upstream's LoRA configuration.
    train
        Passed to upstream ``get_lora``; ``True`` unfreezes the LoRA
        parameters and the cell-state head.
    device
        Device to move the model to once buffers are materialised.
    **kwargs
        Forwarded to ``Scooby.from_pretrained``.

    Returns
    -------
    The model, ready for ``.to(device)``-free use.

    Notes
    -----
    ``scooby`` is not a dependency of Epione; put its source on
    ``sys.path`` or pip-install it first. Its ``scooby.data`` module
    imports ``pybigtools`` (a Rust build) which this bridge does not
    need — stub it if the install is awkward.
    """
    import torch

    try:
        from scooby.modeling import Scooby
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "upstream scooby is required to build the model. Clone "
            "https://github.com/gagneurlab/scooby and put it on sys.path."
        ) from exc

    kwargs.setdefault("return_center_bins_only", True)
    kwargs.setdefault("disable_cache", True)
    kwargs.setdefault("use_transform_borzoi_emb", False)

    model = Scooby.from_pretrained(
        checkpoint,
        cell_emb_dim=cell_emb_dim,
        embedding_dim=embedding_dim,
        n_tracks=n_tracks,
        **kwargs,
    )
    materialize_meta_buffers(model)

    if lora:
        from scooby.utils.utils import get_lora

        model = get_lora(model, train=train)
        # PEFT wrapping can reintroduce meta buffers on copied modules.
        materialize_meta_buffers(model)

    if device is not None:
        model = model.to(torch.device(device))
    return model
