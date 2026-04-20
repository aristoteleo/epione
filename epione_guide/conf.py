import os
import sys
from datetime import datetime
from pathlib import Path

try:
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
repository_url = "https://github.com/aristoteleo/epione"
default_github_ref = "main"

html_context = {
    "display_github": True,
    "github_user": "aristoteleo",
    "github_repo": project,
    "github_version": default_github_ref,
    "conf_py_path": "/epione_guide/",
}

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "sphinx_copybutton",
    "sphinx_design",
]

# Mock heavy optional C-extension / binary deps that aren't installed in
# the RTD build env. The ``epione`` package itself is NOT mocked — autodoc
# needs the real module to resolve docstrings and autosummary entries.
autodoc_mock_imports = [
    "pyBigWig",
    "point_reducer_cy",            # .pyx extension used by rph_kmeans
    "epione.external.rph_kmeans._point_reducer_cy",
]
autosummary_mock_imports = autodoc_mock_imports

autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]

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
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]
needs_sphinx = "4.0"
nitpicky = False

# -- extlinks / intersphinx ---------------------------------------------------
extlinks = {
    "issue": (f"{repository_url}/issues/%s", "#%s"),
    "pr": (f"{repository_url}/pull/%s", "#%s"),
    "ghuser": ("https://github.com/%s", "@%s"),
}

intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}

# -- HTML / sphinx_book_theme -------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = project
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"

html_theme_options = {
    "repository_url": repository_url,
    "repository_branch": default_github_ref,
    "path_to_docs": "epione_guide",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "navigation_with_keys": True,
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
}

# Match omicverse_guide's Pygments defaults (the custom.css token palette
# overrides colours regardless, but leave the fallback style sane).
pygments_style = "tango"
pygments_dark_style = "monokai"

html_static_path = ["_static"]
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
    "css/custom.css",
    "css/icons.css",
    "css/override.css",
]
html_show_sphinx = False
