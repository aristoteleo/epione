"""Pretrained seq2PRINT checkpoints hosted on Hugging Face.

scPrinter publishes no seq2PRINT weights, so models are trained per
dataset; this registry lets a trained one be reused instead of retrained.
Each checkpoint is two files on the Hub:

* ``model.safetensors`` -- the weights. safetensors loads without
  executing code (unlike a pickled ``.pt``), and the layout is
  scPrinter's, so the file also loads into scPrinter's ``seq2PRINT``;
* ``manifest.json`` -- everything needed to rebuild and judge the model:
  architecture and LoRA arguments, the data and peaks it was trained on,
  the fold, the training settings, held-out metrics and the Epione commit.

Downloads go through pooch with a pinned Hub revision and an md5 per file,
the same mechanism as PRINT's weights (:func:`epione.tl.fetch_print_model`),
so no Hugging Face client is required and a later push to the repository
cannot silently change what a tutorial loads. The checkpoints are entries
of the unified registry (:func:`epione.models.list_pretrained`,
``epi.models.load_pretrained('seq2print_p2s1_fold0_bulk')``); this module's
:func:`load_pretrained` and :data:`PRETRAINED` are the family-local view.
Set ``EPIONE_MODELS_DIR`` (or ``EPIONE_DATA_DIR``) to choose the cache.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

__all__ = ["PRETRAINED", "load_pretrained", "save_pretrained"]

HF_REPO = "starlitnightly/epione-seq2print"


def _legacy_view() -> Dict[str, dict]:
    """``{key: {"revision", "files"}}`` built from the registry manifest."""
    from .._manifest import BUILTIN_ENTRIES

    return {e["source"]["subfolder"]: {"revision": e["source"]["revision"],
                                       "files": dict(e["files"])}
            for e in BUILTIN_ENTRIES if e["family"] == "seq2print"}


#: name -> {"revision": Hub commit, "files": {filename: md5}}.
#: A view of :mod:`epione.models.registry` (entries ``seq2print_<name>``),
#: kept for compatibility; register new checkpoints with
#: :func:`epione.models.register` (or add them to ``_manifest.py``).
#: Keys added here still load through :func:`load_pretrained`.
PRETRAINED: Dict[str, dict] = _legacy_view()


def _cache_dir():
    from ..registry import pretrained_cache_dir

    return pretrained_cache_dir("seq2print")


def _entry(name: str):
    from ..registry import PretrainedEntry

    e = PRETRAINED[name]
    return PretrainedEntry(name=name, family="seq2print", files=dict(e["files"]),
                           source={"type": "huggingface", "repo": HF_REPO,
                                   "revision": e["revision"], "subfolder": name,
                                   "layout": "flat"})


def _fetch(name: str, filename: str) -> str:
    """Download one file of a checkpoint at its pinned revision
    (checksum-verified, cached; see :mod:`epione.models.registry`)."""
    return str(_entry(name).fetch(filename))


def _build(manifest: dict):
    from ._model import LORA_TARGETS, build_seq2print

    model = build_seq2print(**manifest["architecture"])
    lora = manifest.get("lora")
    if lora:
        # Shapes only; the real (rescaled) embeddings come from the weights.
        emb = np.zeros(tuple(lora["embedding_shape"]), np.float32)
        cov = (np.zeros(tuple(lora["coverage_shape"]), np.float32)
               if lora.get("coverage_shape") else None)
        model.add_lora(emb, coverages=cov, rank=lora["rank"],
                       hidden_dim=lora["hidden_dim"],
                       n_lora_layers=lora.get("n_lora_layers", 0),
                       targets=tuple(lora.get("targets", LORA_TARGETS)),
                       coverage_in_lora=lora.get("coverage_in_lora", False))
    return model


def load_pretrained(name: str, *, return_manifest: bool = False):
    """Load a published seq2PRINT checkpoint.

    Parameters
    ----------
    name
        A key of :data:`PRETRAINED`.
    return_manifest
        Also return the manifest dict (data, fold, metrics, history).

    Returns
    -------
    The model in eval mode (and the manifest).
    """
    if name not in PRETRAINED:
        raise KeyError(f"unknown checkpoint {name!r}; available: {sorted(PRETRAINED)}")
    return _load_files(_fetch(name, "manifest.json"), _fetch(name, "model.safetensors"),
                       return_manifest)


def _load_files(manifest_path, weights_path, return_manifest=False):
    from safetensors.torch import load_model

    manifest = json.loads(Path(manifest_path).read_text())
    model = _build(manifest)
    load_model(model, str(weights_path), strict=True)
    model.eval()
    return (model, manifest) if return_manifest else model


def _registry_load(entry, *, return_manifest: bool = False, device=None):
    """Registry loader (``epi.models.load_pretrained('seq2print_...')``)."""
    out = _load_files(entry.fetch("manifest.json"), entry.fetch("model.safetensors"),
                      return_manifest)
    if device is not None:
        m = out[0] if return_manifest else out
        m.to(device)
    return out


def save_pretrained(model, directory, *, manifest: dict) -> Dict[str, str]:
    """Write ``model.safetensors`` + ``manifest.json`` for publication.

    ``manifest`` must hold ``architecture`` (the :func:`build_seq2print`
    keyword arguments) and, for a LoRA model, ``lora`` (the
    :meth:`~._model.Seq2PRINT.add_lora` arguments); shapes of the
    embedding / coverage tables are added here. The round trip is checked:
    the saved files are reloaded and must reproduce the model's outputs.

    Returns
    -------
    ``{filename: "md5:..."}`` -- the ``files`` of a registry entry
    (:func:`epione.models.register`).
    """
    import hashlib

    import torch
    from safetensors.torch import load_model, save_model

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(json.dumps(manifest))          # deep copy, JSON-safe
    if model.embeddings is not None:
        manifest.setdefault("lora", {})
        manifest["lora"]["embedding_shape"] = list(model.embeddings.weight.shape)
        manifest["lora"]["coverage_shape"] = (list(model.coverages.weight.shape)
                                              if model.coverages is not None else None)
    model = model.cpu().eval()
    save_model(model, str(directory / "model.safetensors"))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Round trip: rebuild from the manifest and compare outputs.
    clone = _build(manifest)
    load_model(clone, str(directory / "model.safetensors"), strict=True)
    clone.eval()
    x = torch.nn.functional.one_hot(
        torch.randint(0, 4, (2, model.dna_len), generator=torch.Generator().manual_seed(0)), 4
    ).permute(0, 2, 1).float()
    cells = torch.zeros(2, dtype=torch.long) if model.embeddings is not None else None
    with torch.no_grad():
        a, b = model(x, cells), clone(x, cells)
    if not (torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])):
        raise RuntimeError("saved checkpoint does not reproduce the model")

    return {f: "md5:" + hashlib.md5((directory / f).read_bytes()).hexdigest()
            for f in ("model.safetensors", "manifest.json")}
