"""Official pretrained ChromBPNet models from the ENCODE portal.

ENCODE releases ChromBPNet models (5 folds each) as one tar per cell type.
:func:`load_pretrained` downloads the tar once through pooch, pinned by
ENCODE accession **and md5**, extracts only the requested ``.h5`` file and
loads it with :func:`~._keras.load_keras_h5` (h5py; no TensorFlow).

The md5 pin means a later change on the portal fails loudly instead of
silently changing a tutorial's model. Set ``EPIONE_DATA_DIR`` to choose the
cache.
"""
from __future__ import annotations

import os
import tarfile
from pathlib import Path
from typing import Dict

__all__ = ["PRETRAINED", "load_pretrained", "pretrained_path"]

ENCODE_URL = "https://www.encodeproject.org/files/{acc}/@@download/{acc}.tar.gz"

#: name -> ENCODE ``models`` file accession, md5, and the experiment id used
#: in the member file names. ATAC-seq, GRCh38, ENCODE folds (= seq2print FOLDS).
PRETRAINED: Dict[str, dict] = {
    "encode_gm12878": {"accession": "ENCFF772UBB", "md5": "47a4549c40d25bb2440f4fc9b07eec21",
                       "experiment": "ENCSR637XSC", "annotation": "ENCSR081YFO",
                       "biosample": "GM12878"},
    "encode_k562": {"accession": "ENCFF008TMU", "md5": "a1b9870fcb52bba03ea5ef71c8acf897",
                    "experiment": "ENCSR868FGK", "annotation": "ENCSR893SUD",
                    "biosample": "K562"},
    "encode_hepg2": {"accession": "ENCFF839MZM", "md5": "952774ac9f31b5765b224fc85540dabb",
                     "experiment": "ENCSR291GJU", "annotation": "ENCSR476FKK",
                     "biosample": "HepG2"},
    "encode_imr90": {"accession": "ENCFF419JCT", "md5": "eaf764676ca1000ae5d296a6ce4764d1",
                     "experiment": "ENCSR200OML", "annotation": "ENCSR585BFP",
                     "biosample": "IMR-90"},
    "encode_tcell": {"accession": "ENCFF358NCZ", "md5": "6c1471fcc3f8606678a044686adb09bd",
                     "experiment": "ENCSR102RXG", "annotation": "ENCSR752EUR",
                     "biosample": "T-cell"},
}

#: ``kind`` -> member-name stem in the ENCODE tar.
KINDS = {"chrombpnet": "model.chrombpnet", "nobias": "model.chrombpnet_nobias",
         "bias": "model.bias_scaled"}


def _cache_dir() -> Path:
    import pooch

    d = os.environ.get("EPIONE_DATA_DIR")
    return Path(d) / "chrombpnet" if d else Path(pooch.os_cache("epione")) / "chrombpnet"


def pretrained_path(name: str, kind: str = "chrombpnet", fold: int = 0) -> Path:
    """Local path of one ``.h5`` from an ENCODE ChromBPNet release,
    downloading (md5-checked) and extracting it on first use."""
    import pooch

    if name not in PRETRAINED:
        raise KeyError(f"unknown model {name!r}; available: {sorted(PRETRAINED)}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    e = PRETRAINED[name]
    cache = _cache_dir()
    member = f"{KINDS[kind]}.fold_{fold}.{e['experiment']}.h5"
    out = cache / name / f"fold_{fold}" / member
    if out.exists():
        return out
    tar = pooch.retrieve(ENCODE_URL.format(acc=e["accession"]), known_hash=f"md5:{e['md5']}",
                         fname=f"{e['accession']}.tar.gz", path=cache, progressbar=False)
    with tarfile.open(tar, "r:gz") as tf:
        hits = [m for m in tf.getmembers() if m.name.endswith("/" + member) or m.name == member]
        if not hits:
            raise FileNotFoundError(f"{member} not in {e['accession']}")
        out.parent.mkdir(parents=True, exist_ok=True)
        with tf.extractfile(hits[0]) as src, open(out, "wb") as dst:
            dst.write(src.read())
    return out


def load_pretrained(name: str, kind: str = "chrombpnet", fold: int = 0):
    """Load an official ENCODE ChromBPNet model into PyTorch.

    ``kind``: ``'chrombpnet'`` (full model, :class:`ChromBPNet`),
    ``'nobias'`` (bias-corrected accessibility :class:`BPNet`, the one to
    use for biological discovery) or ``'bias'`` (scaled Tn5 bias
    :class:`BPNet`).
    """
    from ._keras import load_keras_h5

    return load_keras_h5(pretrained_path(name, kind, fold))
