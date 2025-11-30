# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib  # type: ignore[import]


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
PYPROJECT = ROOT_DIR / "pyproject.toml"

# Ensure the source directory is on sys.path
sys.path.insert(0, str(SRC_DIR))


def _load_version(default: str = "0.0.0") -> str:
    """Load the project version from pyproject.toml.

    This keeps Sphinx/Read the Docs in sync with the package metadata.
    """
    try:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        return str(data.get("project", {}).get("version", default))
    except Exception:
        return default


# -- Project information -----------------------------------------------------

project = "Moltres"
copyright = "2024, Odos Matthews"
author = "Odos Matthews"
release = _load_version("0.0.0")
version = release


# -- General configuration ---------------------------------------------------

# Explicitly set the master document so Sphinx/RTD use index.rst as the root.
master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # For Google/NumPy style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.todo",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Avoid clashing with Sphinx's root `index` document.
    # `docs/index.md` is included via `moltres-design-notes.md` instead.
    "index.md",
]

# MyST configuration for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
]
myst_heading_anchors = 3


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"  # Read the Docs theme
html_static_path = ["_static"]

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "titles_only": False,
}

# html_logo and html_favicon can be set here if you add assets to docs/_static/
# html_logo = "_static/moltres-logo.png"
# html_favicon = "_static/moltres-favicon.ico"


# -- Extension configuration -------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show_inheritance": True,
}
autodoc_mock_imports = [
    # Core optional dependencies that should not be required to build docs
    "sqlalchemy",
    "pandas",
    "polars",
    "aiofiles",
    # Database/engine integrations that can cause import-time side effects
    "duckdb_engine",
]

# Autosummary settings
autosummary_generate = True

# Napoleon settings (for docstring parsing)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
}

# TODO extension
todo_include_todos = False
