"""Minimal env / tool-resolution helpers for the epione bulk pipeline.

Mirrors the ``omicverse.alignment._cli_utils`` API (``resolve_executable``,
``build_env``, ``run_cmd``, ``_install_tool``) so notebooks can locate
the right ``bowtie2`` / ``samtools`` / ``STAR`` / ``macs2`` / ``bedtools`` /
``featureCounts`` / ``htseq-count`` / ``findMotifsGenome.pl`` binary
regardless of which conda env the user is running in, and auto-install
missing tools via ``mamba`` / ``conda`` on request.

This module prefers to re-export the omicverse helpers when that
package is importable (same repo during development), and falls back
to a lightweight local implementation otherwise.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

try:
    # Re-use the well-tested omicverse implementation when available —
    # it already handles conda-meta state parsing, mamba/micromamba
    # detection, and the INSTALL_HINTS map for the common tools.
    from omicverse.alignment._cli_utils import (  # type: ignore
        resolve_executable,
        build_env,
        run_cmd,
        ensure_dir,
        _install_tool,
        _available_installer,
    )
    _HAS_OMICVERSE = True
except Exception:  # pragma: no cover
    _HAS_OMICVERSE = False

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

    def ensure_dir(p: Union[str, Path]) -> Path:
        p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

    def build_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        env = os.environ.copy()
        if extra:
            env.update(extra)
        return env

    def run_cmd(cmd: Sequence[str], env: Optional[dict] = None,
                cwd: Optional[Union[str, Path]] = None, check: bool = True) -> None:
        p = subprocess.run(list(cmd), env=env or os.environ.copy(),
                            cwd=str(cwd) if cwd else None)
        if check and p.returncode != 0:
            raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")

    def _available_installer() -> Optional[str]:
        for name in ("mamba", "conda", "micromamba"):
            if shutil.which(name):
                return name
        return None

    def _install_tool(name: str) -> bool:
        hint = _INSTALL_HINTS.get(name)
        if not hint:
            return False
        try:
            subprocess.check_call(hint.split())
            return True
        except Exception:
            return False

    def resolve_executable(name: str, explicit: Optional[str] = None,
                           auto_install: bool = False) -> str:
        if explicit and shutil.which(explicit):
            return explicit
        path = shutil.which(name)
        if path:
            return path
        if auto_install and _install_tool(name):
            path = shutil.which(name)
            if path:
                return path
        raise FileNotFoundError(
            f"Could not find '{name}' on PATH. "
            + (_INSTALL_HINTS.get(name, f"Install it manually and re-run.")
               if _INSTALL_HINTS else f"Install it manually and re-run.")
        )


# ---------------------------------------------------------------------------
# epione-specific resolvers (one per tool we use in bulk notebooks)
# ---------------------------------------------------------------------------

def tool_path(name: str, *, auto_install: bool = False) -> str:
    """Return the absolute path of a bulk-pipeline tool.

    ``auto_install=True`` asks the resolver to run the conda install
    hint for this tool if it's missing — handy for one-shot setup
    cells in notebooks. False by default so tutorials don't
    silently trigger long installs.
    """
    return resolve_executable(name, auto_install=auto_install)


def check_tools(names: Sequence[str], *, verbose: bool = True) -> Dict[str, Optional[str]]:
    """Return ``{name: path_or_None}`` — handy one-line readiness check."""
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


# Common tool bundles for the paper's ATAC/ChIP/CUT&RUN/RNA pipelines
ATAC_TOOLS = (
    "bowtie2", "samtools", "bedtools", "macs2",
    "tabix", "bgzip", "featureCounts",
)
RNA_TOOLS = (
    "STAR", "samtools", "htseq-count", "featureCounts",
)
MOTIF_TOOLS = (
    "findMotifsGenome.pl",
)


__all__ = [
    "resolve_executable", "build_env", "run_cmd", "ensure_dir",
    "tool_path", "check_tools",
    "ATAC_TOOLS", "RNA_TOOLS", "MOTIF_TOOLS",
]
