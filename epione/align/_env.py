"""Env / tool resolution helpers for epione.align.

Jupyter kernels frequently inherit a ``PATH`` that doesn't include
their own conda env's ``bin/`` (e.g. when the kernelspec was
registered from a shell without the env activated). A bare
``shutil.which('bowtie2')`` therefore fails even though the tool is
installed next to ``sys.executable``. This module handles that by:

1. Checking ``shutil.which(name)`` first (the classic lookup).
2. Falling back to ``Path(sys.executable).parent / name`` — the
   active Python env's own ``bin/``.
3. Prepending that same directory to ``PATH`` in :func:`build_env`
   so subprocesses launched with ``env=build_env()`` inherit a
   working PATH without the caller having to patch it manually.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union


_INSTALL_HINTS = {
    "bowtie2": "mamba install -c bioconda -y bowtie2",
    "samtools": "mamba install -c bioconda -y samtools",
    "STAR": "mamba install -c bioconda -y star",
    "macs2": "mamba install -c bioconda -y macs2",
    "bedtools": "mamba install -c bioconda -y bedtools",
    "featureCounts": "mamba install -c bioconda -y subread",
    "htseq-count": "pip install HTSeq",
    "tabix": "mamba install -c bioconda -y tabix",
    "bgzip": "mamba install -c bioconda -y tabix",
    "findMotifsGenome.pl": "mamba install -c bioconda -y homer",
    "fastp": "mamba install -c bioconda -y fastp",
}


def _env_bin_candidate(name: str) -> Optional[str]:
    """Return ``<sys.executable>/../<name>`` (or .exe on Windows) if
    it exists and is executable, else ``None``."""
    parent = Path(sys.executable).parent
    for cand in (parent / name, parent / f"{name}.exe"):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _available_installer() -> Optional[str]:
    for env_var in ("MAMBA_EXE", "CONDA_EXE", "MICROMAMBA_EXE"):
        cand = os.environ.get(env_var)
        if cand and Path(cand).exists():
            return cand
    for name in ("mamba", "conda", "micromamba"):
        p = shutil.which(name)
        if p:
            return p
    return None


def _install_tool(name: str) -> bool:
    hint = _INSTALL_HINTS.get(name)
    if not hint:
        return False
    installer = _available_installer()
    if installer is None:
        return False
    # Hint has the form "mamba install -c bioconda -y <pkg>". Swap the
    # leading binary for whichever installer we actually found.
    tokens = hint.split()
    tokens[0] = installer
    try:
        subprocess.check_call(tokens)
        return True
    except Exception:
        return False


def resolve_executable(
    name: str,
    explicit: Optional[str] = None,
    auto_install: bool = False,
) -> str:
    """Resolve a CLI executable, checking PATH and the active env bin.

    Order of lookup:
        1. ``explicit`` path (if given and executable).
        2. ``shutil.which(name)``.
        3. ``<sys.executable>/../<name>``.
        4. If ``auto_install`` is True and an installer (mamba /
           conda / micromamba) is available, install the tool, then
           retry steps 2-3.
    Raises :class:`FileNotFoundError` if none of the above resolve.
    """
    if explicit:
        p = Path(explicit)
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
        raise FileNotFoundError(f"Executable not found or not executable: {explicit}")

    path = shutil.which(name) or _env_bin_candidate(name)
    if path:
        return path

    if auto_install and _install_tool(name):
        path = shutil.which(name) or _env_bin_candidate(name)
        if path:
            return path

    hint = _INSTALL_HINTS.get(name, "Install it manually and re-run.")
    raise FileNotFoundError(
        f"'{name}' not found on PATH or in {Path(sys.executable).parent}. {hint}"
    )


def tool_path(name: str, *, auto_install: bool = False) -> str:
    """Thin wrapper around :func:`resolve_executable` — returns the
    absolute path of the named tool."""
    return resolve_executable(name, auto_install=auto_install)


def check_tools(
    names: Sequence[str],
    *,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """Return ``{name: path_or_None}`` — a one-line readiness check.

    Set ``verbose=False`` to suppress the ✓ / ✗ stdout table.
    """
    out: Dict[str, Optional[str]] = {}
    for n in names:
        try:
            out[n] = tool_path(n, auto_install=False)
        except FileNotFoundError:
            out[n] = None
    if verbose:
        for n, p in out.items():
            status = "✓" if p else "✗"
            print(f"  {status}  {n:25s}  {p or 'NOT FOUND'}")
    return out


def build_env(
    extra_paths: Optional[Iterable[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build a subprocess environment with the active env bin on PATH.

    This is the companion of :func:`resolve_executable`: even if the
    resolver hands back a full path for a tool, any nested
    subprocesses that fork-exec bare names still need PATH to include
    ``<sys.executable>/../``. Prepend it unconditionally.
    """
    env = os.environ.copy()
    env_bin = str(Path(sys.executable).parent)
    parts: List[str] = [env_bin]
    if extra_paths:
        parts.extend([p for p in extra_paths if p])
    parts.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(parts)
    if extra_env:
        env.update(extra_env)
    return env


def run_cmd(
    cmd: Sequence[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Union[str, Path]] = None,
    check: bool = True,
) -> None:
    """Thin wrapper over :func:`subprocess.run` that defaults ``env``
    to :func:`build_env` — so every command sees the active env bin
    on PATH without the caller having to remember it."""
    p = subprocess.run(
        list(cmd),
        env=env or build_env(),
        cwd=str(cwd) if cwd else None,
    )
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


ATAC_TOOLS = ("bowtie2", "samtools", "bedtools", "macs2", "tabix", "bgzip", "featureCounts")
RNA_TOOLS = ("STAR", "samtools", "htseq-count", "featureCounts")
MOTIF_TOOLS = ("findMotifsGenome.pl",)


__all__ = [
    "resolve_executable",
    "tool_path",
    "check_tools",
    "build_env",
    "run_cmd",
    "ensure_dir",
    "ATAC_TOOLS",
    "RNA_TOOLS",
    "MOTIF_TOOLS",
]
