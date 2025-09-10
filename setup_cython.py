#!/usr/bin/env python

"""
Setup script for compiling Cython extensions for epione footprint analysis

Run this to compile the Cython optimized functions:
    python setup_cython.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the Cython extension
extensions = [
    Extension(
        "_footprint_cython",
        ["epione/utils/_footprint_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],
        language="c"
    )
]

# Build the extensions
setup(
    name="epione-footprint-cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'nonecheck': False
        }
    ),
    zip_safe=False
)