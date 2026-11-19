"""Where the resources ACESO needs live on this machine, resolved by existence and checked at start-up.

Training ACESO is not self-contained: it reads a genome FASTA, a fold table, the scGPT checkpoint and
source, the scooby source, AlphaGenome weights, and (for stage 0) EpiAgent's cCRE files. Those live
outside the package and in a different place on every machine, which is a reliable source of long
silent failures -- a wrong interpreter path once kept four GPUs busy for 51 minutes while every
subprocess died and the driver's log looked healthy.

Three properties follow from that:

one place
    the mapping from a logical name to its candidates is here, so a new machine is one edit;
resolve by existence
    the first candidate that exists wins. Guessing the site from a hostname is itself a guess that is
    wrong exactly when a filesystem is mounted somewhere unexpected;
fail at start-up
    :func:`require` is called before any GPU is taken, so a missing path costs seconds.

Environment variables are the first candidate for every name, because an unset variable falls back
silently and a stale one fails per item; the existence check turns both into one named error.

    from epione.models.aceso import site_path, require_sites
    require_sites("hg38_fa", "scgpt_model")          # at the top, before any work
    fasta = site_path("hg38_fa")

Candidates are read, in order, from ``$EPIONE_ACESO_SITES`` (a JSON file mapping name -> path or list
of paths), the name's own environment variable, and the built-in defaults, which are relative to
``$EPIONE_ROOT`` (itself resolved the same way) and so need no per-site entry for anything that lives
inside the project tree. :func:`register_site` adds a name at run time.
"""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# name -> (environment variable, [candidates; "{R}" is the project root, first existing wins])
_SPEC: Dict[str, Tuple[Optional[str], List[str]]] = {
    "root": ("EPIONE_ROOT", []),
    # stage 0 -- EpiAgent defines the chromatin state space z
    "epiagent_python": ("EPIAGENT_PYTHON", []),
    "epiagent_weights": ("EPIAGENT_WEIGHTS", ["{R}/external/epiagent/pretrained_EpiAgent.pth"]),
    "ccre_bed": ("EPIAGENT_CCRE_BED", ["{R}/data/cCRE.bed"]),
    "ccre_df_npy": ("EPIAGENT_CCRE_DF", ["{R}/data/cCRE_document_frequency.npy"]),
    # stage 1 -- scGPT reads the RNA panel
    "scgpt_model": ("SCGPT_MODEL_DIR", ["{R}/external/scgpt"]),
    "scgpt_src": ("SCGPT_SRC", ["{R}/external/scGPT-src"]),
    # stage 2 -- sequence trunk and its data
    "scooby_src": ("SCOOBY_SRC", ["{R}/external/scooby-dbfe6cd"]),
    "alphagenome_weights": ("ACESO_AG_WEIGHTS", ["{R}/external/alphagenome/model_all_folds.safetensors"]),
    "hg38_fa": ("HG38_FA", ["{R}/genome/hg38/hg38.fa"]),
    "mm10_fa": ("MM10_FA", ["{R}/genome/mm10/mm10.fa"]),
    "folds_bed": ("FOLDS_BED", ["{R}/scooby_bench/hg38_folds.bed"]),
    "folds_bed_mm10": ("FOLDS_BED_MM10", ["{R}/scooby_bench/mm10_folds.bed"]),
    "exon_masks": ("EXON_MASKS", ["{R}/data/exon_masks/exons_hg38.npz"]),
    # tools called as subprocesses; taking the neighbour of sys.executable is a guess that has failed
    "bedtools": ("BEDTOOLS", []),
    "tabix": ("TABIX", []),
}

_cache: Dict[str, Optional[str]] = {}


def _sites_file() -> Dict[str, object]:
    q = os.environ.get("EPIONE_ACESO_SITES")
    if q and Path(q).exists():
        return json.loads(Path(q).read_text())
    return {}


def _root() -> str:
    """The project root every in-tree default hangs off: ``$EPIONE_ROOT``, a sites file, or the cwd."""
    q = os.environ.get("EPIONE_ROOT") or _sites_file().get("root")
    if isinstance(q, (list, tuple)):
        q = next((c for c in q if Path(c).exists()), None)
    return str(q) if q and Path(str(q)).exists() else os.getcwd()


def register_site(name: str, candidates, env: Optional[str] = None) -> None:
    """Add or replace a logical name. ``candidates`` is a path or a list of them, "{R}" = project root."""
    if isinstance(candidates, (str, Path)):
        candidates = [str(candidates)]
    _SPEC[name] = (env, [str(c) for c in candidates])
    _cache.pop(name, None)


def site_path(name: str, required: bool = True) -> Optional[str]:
    """Resolve a logical name to the first candidate that exists; ``None`` if absent and not required."""
    if name in _cache:
        return _cache[name]
    if name not in _SPEC:
        raise KeyError(f"unknown resource {name!r}; known: {sorted(_SPEC)}")
    if name == "root":
        _cache[name] = _root()
        return _cache[name]
    env, cands = _SPEC[name]
    extra = _sites_file().get(name, [])
    extra = [extra] if isinstance(extra, str) else list(extra)
    tried = []
    for q in extra + ([os.environ.get(env)] if env else []) + list(cands):
        if not q:
            continue
        q = str(q).replace("{R}", _root())
        tried.append(q)
        if Path(q).exists():
            _cache[name] = q
            return q
    if required:
        hint = f"set ${env} or list it in $EPIONE_ACESO_SITES" if env else "list it in $EPIONE_ACESO_SITES"
        raise FileNotFoundError(f"cannot resolve {name!r} on {socket.gethostname()} (root {_root()}).\n"
                                f"  tried: {tried}\n"
                                f"  {hint}.")
    _cache[name] = None
    return None


def require_sites(*names: str, verbose: bool = True) -> Dict[str, Optional[str]]:
    """Resolve every name now and print the table, so the training log records what was actually used."""
    out = {n: site_path(n) for n in names}
    if verbose:
        print(f"[aceso] host {socket.gethostname()}  root {_root()}", flush=True)
        for n, q in out.items():
            print(f"  {n:22s} {q}", flush=True)
    return out
