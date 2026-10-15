"""Registry loaders for families whose own modules do not carry one.

Each is called as ``loader(entry, **kwargs)`` by
:func:`epione.models.load_pretrained` and fetches its files through
``entry.fetch`` (checksum-verified, cached, offline-aware).
"""
from __future__ import annotations

from pathlib import Path


def _device(model, device):
    return model.to(device) if device is not None else model


# -- PRINT (scPrinter's Zenodo weights) --------------------------------------

def load_print(entry, *, device=None):
    """``print_dispersion`` -> :class:`epione.tl.DispersionModel`;
    ``print_tn5_bias`` / ``print_tf_binding`` -> the TorchScript module."""
    path = entry.fetch()
    if path.suffix == ".h5":
        from epione.tl._print import DispersionModel
        return DispersionModel.load(path)
    import torch
    return torch.jit.load(str(path), map_location=device or "cpu").eval()


# -- HINT-ATAC (RGT) -----------------------------------------------------------

def load_hint(entry, *, genome=None, model=None):
    """``hint_atac_hmm_*`` -> :class:`epione.tl.HintHMM`.
    ``hint_atac_bias_tables`` -> :class:`epione.tl.HintBias` when
    ``genome`` is given, else the ``(forward, reverse)`` k-mer tables."""
    from epione.tl import _hint

    npz = [f for f in entry.files if f.endswith(".npz")]
    if npz:
        return _hint._hmm_from_npz(entry.fetch(npz[0]), name=f"atac_{model}" if model else None)
    fwd = next(f for f in entry.files if f.endswith("_F.txt"))
    rev = next(f for f in entry.files if f.endswith("_R.txt"))
    pf, pr = entry.fetch(fwd), entry.fetch(rev)
    if genome is not None:
        return _hint.HintBias(genome, table_f=pf, table_r=pr)
    return _hint.load_hint_bias_table(pf, pr)


# -- Borzoi / scooby (Hugging Face, transformers layout) ------------------------

def _hf_dir(entry) -> Path:
    paths = [entry.fetch(f) for f in entry.files]
    return paths[0].parent


def load_borzoi(entry, *, device=None, **kwargs):
    """A Borzoi backbone (``borzoi_pytorch.Borzoi``). Needs
    ``borzoi-pytorch``. For fine-tuning with scooby pass
    ``epi.models.fetch_pretrained(name)['config.json'].parent`` to
    :func:`epione.models.scooby.load_scooby`."""
    try:
        from borzoi_pytorch import Borzoi
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError("Borzoi weights need `pip install borzoi-pytorch`") from exc
    from epione.models.scooby._model import materialize_meta_buffers

    model = Borzoi.from_pretrained(str(_hf_dir(entry)), **kwargs)
    materialize_meta_buffers(model)
    return _device(model.eval(), device)


def load_scooby_checkpoint(entry, *, cell_emb_dim: int, n_tracks: int, device=None,
                           lora: bool = False, train: bool = False, **kwargs):
    """A published scooby model, built with upstream's inference settings
    (``use_transform_borzoi_emb=True``, merged weights, no fresh LoRA).
    Needs upstream scooby on ``sys.path`` (see
    :func:`epione.models.scooby.load_scooby`)."""
    from epione.models.scooby._model import load_scooby

    kwargs.setdefault("use_transform_borzoi_emb", True)
    model = load_scooby(str(_hf_dir(entry)), cell_emb_dim=cell_emb_dim, n_tracks=n_tracks,
                        lora=lora, train=train, device=device, **kwargs)
    return model if train else model.eval()


# -- families with user-trained checkpoints only -------------------------------

def load_translate(entry, *, file=None):
    """An :mod:`epione.translate` model saved with ``model.save(path)``.

    The file is a torch pickle: register only files you trust (the
    checksum then guarantees it is the one you registered)."""
    from epione.translate import load
    return load(str(entry.fetch(file)))


def load_higashi(entry, *, device=None, verbose: bool = False, return_config: bool = False):
    """A Higashi checkpoint written by :mod:`epione.single.hic` (torch
    pickle; trusted files only). Returns the model (and its config)."""
    from epione.single.hic._higashi_api import _load
    model, cfg = _load(str(entry.fetch()), device, verbose)
    return (model, cfg) if return_config else model
