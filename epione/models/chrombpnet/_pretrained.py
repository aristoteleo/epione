"""Official pretrained ChromBPNet models from the ENCODE portal.

ENCODE releases ChromBPNet models (5 folds each) as one tar per cell type.
:func:`load_pretrained` downloads the tar once through pooch, pinned by
ENCODE accession **and md5**, extracts only the requested ``.h5`` file and
loads it with :func:`~._keras.load_keras_h5` (h5py; no TensorFlow).

The md5 pin means a later change on the portal fails loudly instead of
silently changing a tutorial's model. The releases are entries of the
unified registry (``epi.models.list_pretrained('chrombpnet')``,
``epi.models.load_pretrained('chrombpnet_encode_k562', kind='bias')``);
this module's :func:`load_pretrained` / :data:`PRETRAINED` are the
family-local view and accept the same names. Set ``EPIONE_MODELS_DIR`` (or
``EPIONE_DATA_DIR``) to choose the cache.
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Dict

__all__ = ["PRETRAINED", "load_pretrained", "pretrained_path"]

ENCODE_URL = "https://www.encodeproject.org/files/{acc}/@@download/{acc}.tar.gz"


def _legacy_view() -> Dict[str, dict]:
    from .._manifest import BUILTIN_ENTRIES

    out = {}
    for e in BUILTIN_ENTRIES:
        if e["family"] == "chrombpnet":
            x = e["extra"]
            out[x["key"]] = {"accession": e["source"]["accession"],
                             "md5": e["files"][f"{e['source']['accession']}.tar.gz"][4:],
                             "experiment": x["experiment"], "annotation": x["annotation"],
                             "biosample": e["biosample"]}
    return out


#: name -> ENCODE ``models`` file accession, md5, and the experiment id used
#: in the member file names. ATAC-seq, GRCh38, ENCODE folds (= seq2print FOLDS).
#: A view of the registry entries ``chrombpnet_<name>``; keys added here
#: still load.
PRETRAINED: Dict[str, dict] = _legacy_view()

#: ``kind`` -> member-name stem in the ENCODE tar.
KINDS = {"chrombpnet": "model.chrombpnet", "nobias": "model.chrombpnet_nobias",
         "bias": "model.bias_scaled"}


def _cache_dir() -> Path:
    from ..registry import pretrained_cache_dir

    return pretrained_cache_dir("chrombpnet")


def _entry(name: str):
    """Registry entry for a :data:`PRETRAINED` key or a registry name."""
    from ..registry import PretrainedEntry, get_pretrained

    if name in PRETRAINED:
        e = PRETRAINED[name]
        return PretrainedEntry(
            name=f"chrombpnet_{name}", family="chrombpnet",
            source={"type": "encode", "accession": e["accession"]},
            files={f"{e['accession']}.tar.gz": f"md5:{e['md5']}"},
            biosample=e.get("biosample", ""),
            extra={"key": name, "experiment": e["experiment"],
                   "annotation": e.get("annotation", "")})
    try:
        entry = get_pretrained(name)
    except KeyError:
        entry = None
    if entry is None or entry.family != "chrombpnet":
        raise KeyError(f"unknown model {name!r}; available: {sorted(PRETRAINED)}")
    return entry


def _member_path(entry, kind: str, fold: int) -> Path:
    """Extract (once) and return one ``.h5`` of an ENCODE release tar."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    exp = entry.extra["experiment"]
    member = f"{KINDS[kind]}.fold_{fold}.{exp}.h5"
    out = _cache_dir() / entry.extra.get("key", entry.name) / f"fold_{fold}" / member
    if out.exists():
        return out
    tar = entry.fetch()
    with tarfile.open(tar, "r:gz") as tf:
        hits = [m for m in tf.getmembers() if m.name.endswith("/" + member) or m.name == member]
        if not hits:
            raise FileNotFoundError(f"{member} not in {tar.name}")
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + ".part")
        with tf.extractfile(hits[0]) as src, open(part, "wb") as dst:
            dst.write(src.read())
        part.replace(out)
    return out


def pretrained_path(name: str, kind: str = "chrombpnet", fold: int = 0) -> Path:
    """Local path of one ``.h5`` from an ENCODE ChromBPNet release,
    downloading (md5-checked) and extracting it on first use."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    return _member_path(_entry(name), kind, fold)


def load_pretrained(name: str, kind: str = "chrombpnet", fold: int = 0):
    """Load an official ENCODE ChromBPNet model into PyTorch.

    ``kind``: ``'chrombpnet'`` (full model, :class:`ChromBPNet`),
    ``'nobias'`` (bias-corrected accessibility :class:`BPNet`, the one to
    use for biological discovery) or ``'bias'`` (scaled Tn5 bias
    :class:`BPNet`).
    """
    from ._keras import load_keras_h5

    return load_keras_h5(pretrained_path(name, kind, fold))


def _registry_load(entry, *, kind: str = "chrombpnet", fold: int = 0, device=None):
    """Registry loader. An ENCODE release tar is opened as above; an entry
    of your own ``.h5`` files maps ``kind`` to a file through
    ``extra['kinds']`` (or has a single ``.h5``)."""
    from ._keras import load_keras_h5

    if entry.source["type"] == "encode":
        path = _member_path(entry, kind, fold)
    else:
        kinds = entry.extra.get("kinds")
        if kinds:
            if kind not in kinds:
                raise ValueError(f"{entry.name} has kinds {sorted(kinds)}, not {kind!r}")
            path = entry.fetch(kinds[kind])
        else:
            path = entry.fetch()
    model = load_keras_h5(path)
    return model.to(device) if device is not None else model
