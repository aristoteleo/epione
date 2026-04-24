"""Tests for epione.align._env — the tool-resolver the upstream
pipelines rely on.

Covers the Jupyter-kernel-without-activated-env bug that was fixed in
PR #11: ``shutil.which(name)`` alone misses tools co-installed next to
``sys.executable`` when PATH wasn't updated by the kernel.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest


def test_resolve_executable_uses_path_first(tmp_path, monkeypatch):
    """When the tool is on PATH, resolve_executable should return that."""
    from epione.align._env import resolve_executable

    # Make a fake 'foo' binary in a dir, then prepend it to PATH.
    foo = tmp_path / "foo"
    foo.write_text("#!/bin/sh\necho ok\n")
    foo.chmod(foo.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH','')}")

    p = resolve_executable("foo")
    assert p == str(foo)


def test_resolve_executable_falls_back_to_sys_executable_parent(tmp_path, monkeypatch):
    """Even when PATH has no such binary, if one sits next to
    ``sys.executable`` the resolver should find it. Simulates the
    Jupyter-kernel situation where PATH was inherited without the
    conda env's ``bin/``."""
    from epione.align import _env as env_mod

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python-fake"
    fake_python.write_text("")
    tool = bin_dir / "faketool"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)

    # Point sys.executable at our fake python, strip PATH so shutil.which
    # will miss.
    monkeypatch.setattr(sys, "executable", str(fake_python))
    monkeypatch.setenv("PATH", "/nonexistent")

    p = env_mod.resolve_executable("faketool")
    assert p == str(tool)


def test_resolve_executable_raises_when_missing(monkeypatch):
    from epione.align._env import resolve_executable

    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(FileNotFoundError):
        resolve_executable("definitely_not_a_real_tool_xyzzy")


def test_check_tools_returns_mapping(tmp_path, monkeypatch, capsys):
    from epione.align._env import check_tools

    foo = tmp_path / "foo"
    foo.write_text("#!/bin/sh\n")
    foo.chmod(foo.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(tmp_path))

    out = check_tools(["foo", "missing_tool_xyzzy"], verbose=False)
    assert out["foo"] == str(foo)
    assert out["missing_tool_xyzzy"] is None


def test_build_env_prepends_active_env_bin():
    """``build_env`` must put ``<sys.executable>/../`` at the front of
    PATH so subprocesses inherit it even if the caller's PATH missed it."""
    from epione.align._env import build_env

    env = build_env()
    env_bin = str(Path(sys.executable).parent)
    assert env["PATH"].split(os.pathsep)[0] == env_bin


def test_build_env_respects_extra_env_and_paths():
    from epione.align._env import build_env

    env = build_env(extra_paths=["/extra"], extra_env={"MYFLAG": "1"})
    parts = env["PATH"].split(os.pathsep)
    assert "/extra" in parts
    assert env["MYFLAG"] == "1"


def test_bulk_env_mirrors_align_env_api():
    """epione.bulk._env and epione.align._env should expose the same
    surface — they're pure-Python duplicates of the same algorithm."""
    from epione.align import _env as align_env
    from epione.bulk import _env as bulk_env

    for sym in ("resolve_executable", "tool_path", "check_tools",
                "build_env", "run_cmd", "ATAC_TOOLS", "RNA_TOOLS",
                "MOTIF_TOOLS"):
        assert hasattr(align_env, sym), f"align missing {sym}"
        assert hasattr(bulk_env, sym),  f"bulk missing {sym}"
