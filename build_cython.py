#!/usr/bin/env python

"""
Simple script to build Cython extension for footprint calculations
"""

import os
import subprocess
import sys
import numpy as np

def build_cython_extension():
    """Build the Cython extension in place"""
    
    # Change to the utils directory
    utils_dir = os.path.join(os.path.dirname(__file__), 'epione', 'utils')
    original_dir = os.getcwd()
    
    try:
        os.chdir(utils_dir)
        
        # Build command
        cmd = [
            sys.executable, 
            "-c",
            f"""
import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext = Extension(
    '_footprint_cython',
    ['_footprint_cython.pyx'],
    include_dirs=[np.get_include()],
    extra_compile_args=['-O3', '-ffast-math']
)

setup(
    ext_modules=cythonize([ext], 
                         compiler_directives={{'language_level': 3, 
                                              'boundscheck': False, 
                                              'wraparound': False,
                                              'cdivision': True}})
)
"""
        ]
        
        # Run the build
        result = subprocess.run(cmd + ["build_ext", "--inplace"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Cython extension built successfully!")
            
            # Check if .so file was created
            so_files = [f for f in os.listdir('.') if f.endswith('.so') and 'footprint_cython' in f]
            if so_files:
                print(f"Created: {so_files[0]}")
            else:
                print("Warning: No .so file found")
        else:
            print("Build failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    build_cython_extension()