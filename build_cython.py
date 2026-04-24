#!/usr/bin/env python
"""Rebuild every Cython extension under ``epione/utils/`` in place.

epione ships three Cython modules used by the footprint / bindetect /
ATAC-correct machinery:

- ``_footprint_cython.pyx`` — per-bp footprint scoring kernels
- ``signals.pyx``            — `bindetect` signal-processing inner loops
- ``sequences.pyx``          — `atacorrect` k-mer / bias scoring

The shipped ``.so`` artefacts were compiled against NumPy 1.x. On a
NumPy 2.x environment the old artefacts fail to import
(``numpy.core.multiarray failed to import``) because the Cython-
generated binding calls the NumPy 1.x C ABI. Run this script after
upgrading numpy so the ``.so`` files are rebuilt against whichever
numpy headers the current env exposes::

    python build_cython.py

Requires ``cython`` + a C compiler.
"""
import os
import subprocess
import sys


PYX_FILES = [
    "_footprint_cython.pyx",
    "signals.pyx",
    "sequences.pyx",
]


def build_one(pyx: str, utils_dir: str) -> bool:
    """Cythonize + compile one .pyx file in-place."""
    mod = pyx[:-4]  # strip ".pyx"
    os.chdir(utils_dir)

    # 1. .pyx -> .c
    r = subprocess.run(
        [sys.executable, "-m", "cython", "-3", "--fast-fail", pyx],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"[{pyx}] cython failed:\n{r.stderr}")
        return False

    # 2. .c -> .so via a throwaway setuptools Extension, forced in-place.
    r2 = subprocess.run(
        [sys.executable, "-c", f"""
from setuptools import setup, Extension
import numpy as np
ext = Extension(
    {mod!r}, [{mod + '.c'!r}],
    include_dirs=[np.get_include()],
    extra_compile_args=['-O3', '-ffast-math'],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
)
setup(
    name='_',
    ext_modules=[ext],
    script_args=['build_ext', '--inplace', '--force'],
)
"""],
        capture_output=True, text=True,
    )
    if r2.returncode != 0:
        print(f"[{pyx}] compile failed:\n{r2.stdout}\n{r2.stderr[-600:]}")
        return False

    so = next((f for f in os.listdir('.')
               if f.startswith(mod + '.cpython') and f.endswith('.so')),
              None)
    print(f"[{pyx}] rebuilt → {so}")
    return True


def main() -> int:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    utils_dir = os.path.join(repo_root, "epione", "utils")
    original = os.getcwd()
    ok = True
    try:
        for pyx in PYX_FILES:
            ok &= build_one(pyx, utils_dir)
    finally:
        os.chdir(original)
    # Clean setuptools build/ debris.
    import shutil
    for d in (os.path.join(utils_dir, "build"),):
        if os.path.exists(d):
            shutil.rmtree(d)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
