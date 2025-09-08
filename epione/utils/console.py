import os
import sys
from typing import Optional, List
from contextlib import contextmanager

# Simple, dependency-free colored console with verbosity levels.

# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_COLORS = {
    "grey": "\033[90m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[97m",
}

_DEFAULT_VERBOSITY = int(os.environ.get("EPIONE_VERBOSITY", "2"))
_verbosity = max(0, min(4, _DEFAULT_VERBOSITY))  # 0=silent, otherwise print
_force_color: Optional[bool] = None  # None=auto, True=force, False=disable
_tree_stack: List[bool] = []  # For tree rendering; each bool indicates ancestor has more siblings


def _supports_color() -> bool:
    if _force_color is not None:
        return _force_color
    if os.environ.get("NO_COLOR") is not None:
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _colorize(text: str, color: Optional[str] = None, bold: bool = False, dim: bool = False) -> str:
    if not _supports_color() or color not in _COLORS:
        return text
    prefix = ""
    if bold:
        prefix += _BOLD
    if dim:
        prefix += _DIM
    return f"{prefix}{_COLORS[color]}{text}{_RESET}"


def set_verbosity(level: int) -> None:
    """Set global verbosity (0-4).

    - 0: silent
    - 1: level1 only (high-level)
    - 2: level1-2 (default)
    - 3: level1-3 (verbose)
    - 4: level1-4 (debug)
    """
    global _verbosity
    _verbosity = max(0, min(4, int(level)))


def get_verbosity() -> int:
    return _verbosity


def enable_color(enable: Optional[bool]) -> None:
    """Force color enable/disable. Pass None to restore auto-detection."""
    global _force_color
    _force_color = enable


def _print(level: int, message: str, *, color: Optional[str] = None, bold: bool = False, dim: bool = False, prefix: str = "", marker: Optional[str] = None) -> None:
    # Level now represents indentation depth (1-based). Verbosity 0 hides all.
    if _verbosity == 0:
        return
    depth = max(1, int(level))
    extra = "  " * (depth - 1)  # relative indent placed BEFORE the tree marker

    if _tree_stack:
        # Draw ancestors except the immediate parent level
        parts = ["│  " if more else "   " for more in _tree_stack[:-1]]
        used_marker = marker if marker is not None else "└─ "
        base = "".join(parts) + used_marker
        text = f"{extra}{base}{prefix}{message}"
    else:
        # Simple indentation when no tree context is active
        used_marker = marker if marker is not None else "└─ "
        text = f"{extra}{used_marker}{prefix}{message}"
    print(_colorize(text, color=color, bold=bold, dim=dim))


# Levelled output (level1..level4)
def level1(message: str) -> None:
    """High-level progress. Visible at verbosity >=1."""
    _print(1, message, color="cyan", bold=True)


def level2(message: str) -> None:
    """Normal progress/info. Visible at verbosity >=2."""
    _print(2, message, color="blue")


def level3(message: str) -> None:
    """Detailed info. Visible at verbosity >=3."""
    _print(3, message, color="grey")


def level4(message: str) -> None:
    """Debug details. Visible at verbosity >=4."""
    _print(4, message, color="magenta", dim=True)


# Semantics: info/success/warn/error with colors, mapped to sensible levels
def info(message: str, level: int = 2) -> None:
    _print(level, message, color="blue")


def success(message: str, level: int = 2) -> None:
    _print(level, message, color="green", bold=True)


def warn(message: str, level: int = 1) -> None:
    _print(level, f"⚠️  {message}", color="yellow", bold=True)


def error(message: str, level: int = 1) -> None:
    _print(level, f"❌ {message}", color="red", bold=True)


# Tree-aware printing
def node(message: str, *, last: bool = True, level: int = 1, color: Optional[str] = None, bold: bool = False, dim: bool = False) -> None:
    """Print a tree node using current group context.

    - last=True uses '└─', otherwise '├─'.
    - When no group context is active, falls back to simple indentation.
    """
    marker = "└─ " if last else "├─ "
    _print(level, message, color=color, bold=bold, dim=dim, marker=marker)


@contextmanager
def group(*, last: bool = False):
    """Create a tree group context for children.

    - last=False means current group has following siblings; children lines will show '│ ' under this group.
    - last=True means no following siblings; children lines will show blank under this group.
    """
    _tree_stack.append(not last)
    try:
        yield
    finally:
        _tree_stack.pop()


@contextmanager
def group_node(message: str, *, last: bool = True, level: int = 1, color: Optional[str] = None, bold: bool = False, dim: bool = False):
    """Print a node and create a group context for its children in one go."""
    node(message, last=last, level=level, color=color, bold=bold, dim=dim)
    _tree_stack.append(not last)
    try:
        yield
    finally:
        _tree_stack.pop()


# Convenience alias
set_level = set_verbosity
