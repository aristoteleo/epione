from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Union


class CommandError(RuntimeError):
    pass


def run(
    cmd: Sequence[str],
    cwd: Optional[Union[str, Path]] = None,
    stream: bool = True,
) -> None:
    if stream:
        proc = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None)
        if proc.returncode != 0:
            raise CommandError(f"Command failed: {' '.join(map(str, cmd))}")
        return

    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise CommandError(
            f"Command failed: {' '.join(map(str, cmd))}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def resolve_tool(name: str, explicit: Optional[str] = None) -> str:
    if explicit:
        if shutil.which(explicit) is None:
            raise FileNotFoundError(f"Requested binary '{explicit}' for '{name}' not found in PATH.")
        return explicit

    try:
        from ._env import tool_path

        return tool_path(name)
    except Exception:
        path = shutil.which(name)
        if path is None:
            raise FileNotFoundError(
                f"Required binary '{name}' not found in PATH. Please install it and try again."
            )
        return path


def run_pipe(commands: List[List[str]], cwd: Optional[Union[str, Path]] = None) -> None:
    procs = []
    prev_stdout = None
    try:
        for i, cmd in enumerate(commands):
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdin=prev_stdout,
                stdout=subprocess.PIPE if i < len(commands) - 1 else None,
            )
            procs.append(proc)
            if prev_stdout is not None:
                prev_stdout.close()
            prev_stdout = proc.stdout

        for proc in procs:
            proc.wait()

        failed = [cmd for cmd, proc in zip(commands, procs) if proc.returncode != 0]
        if failed:
            raise CommandError(
                "Pipeline failed:\n" + "\n".join("  " + " ".join(cmd) for cmd in failed)
            )
    finally:
        for proc in procs:
            if proc.stdout is not None:
                proc.stdout.close()


def remove_if_exists(*paths: Union[str, Path]) -> None:
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass
