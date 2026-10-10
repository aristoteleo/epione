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
cannot silently change what a tutorial loads. Set ``EPIONE_DATA_DIR`` to
choose the cache.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

__all__ = ["PRETRAINED", "load_pretrained", "save_pretrained"]

HF_REPO = "starlitnightly/epione-seq2print"

#: name -> {"revision": Hub commit, "files": {filename: md5}}.
#: Filled in when a checkpoint is published; see :func:`save_pretrained`.
PRETRAINED: Dict[str, dict] = {}


def _cache_dir():
    import os

    import pooch

    d = os.environ.get("EPIONE_DATA_DIR")
    return Path(d) / "seq2print" if d else pooch.os_cache("epione") / "seq2print"


def _fetch(name: str, filename: str) -> str:
    """Download one file of a checkpoint at its pinned revision."""
    import pooch

    entry = PRETRAINED[name]
    rev = entry["revision"]
    return pooch.retrieve(
        f"https://huggingface.co/{HF_REPO}/resolve/{rev}/{name}/{filename}",
        known_hash=entry["files"][filename],
        fname=f"{name}__{rev[:12]}__{filename}",
        path=_cache_dir(),
    )


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
    from safetensors.torch import load_model

    manifest = json.loads(Path(_fetch(name, "manifest.json")).read_text())
    model = _build(manifest)
    load_model(model, _fetch(name, "model.safetensors"), strict=True)
    model.eval()
    return (model, manifest) if return_manifest else model


def save_pretrained(model, directory, *, manifest: dict) -> Dict[str, str]:
    """Write ``model.safetensors`` + ``manifest.json`` for publication.

    ``manifest`` must hold ``architecture`` (the :func:`build_seq2print`
    keyword arguments) and, for a LoRA model, ``lora`` (the
    :meth:`~._model.Seq2PRINT.add_lora` arguments); shapes of the
    embedding / coverage tables are added here. The round trip is checked:
    the saved files are reloaded and must reproduce the model's outputs.

    Returns
    -------
    ``{filename: "md5:..."}`` for the registry.
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
