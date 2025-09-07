import os
import sys
from datetime import datetime
from pathlib import Path

try:
    # Python 3.8+
    from importlib.metadata import version as pkg_version
except Exception:  # pragma: no cover
    from importlib_metadata import version as pkg_version  # type: ignore

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

# -- Project information -----------------------------------------------------
project = "epione"
author = "EpiOne contributors"
copyright = f"{datetime.now():%Y}, {author}."
try:
    release = pkg_version("epione")
except Exception:
    release = "0.0.0"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

# Configure autodoc to mock missing imports
autodoc_mock_imports = ["epione", "pyBigWig"]
autosummary_mock_imports = ["epione", "pyBigWig"]

autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
    "substitution",
    "linkify",
]

# Disable notebook execution during build
nb_execution_mode = "off"

source_suffix = {
    ".rst": None,
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = project
html_static_path = ["_static"]
html_css_files = ["css/override.css"]
html_show_sphinx = False

# Configure logo
html_logo = "_static/logo.png"
