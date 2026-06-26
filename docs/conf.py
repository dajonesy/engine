"""Sphinx configuration for the clifford documentation.

This file lives in ``<project_root>/docs/conf.py``.  Sphinx reads it
when you run ``make html``.
"""

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Add the project root (one level above this docs/ folder) to sys.path so
# that ``import clifford`` works during the documentation build.
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------
project   = 'Clifford'
author    = 'D. A. Jones'
copyright = f'{datetime.now().year}, {author}'
release   = '0.1'


# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # pulls docstrings from your modules
    'sphinx.ext.autosummary',   # auto-generates summary tables and stub pages
    'sphinx.ext.napoleon',      # understands numpy- and google-style docstrings
    'sphinx.ext.viewcode',      # adds [source] links next to each function
    'sphinx.ext.intersphinx',   # cross-links to numpy/python/numba docs
    'sphinx.ext.mathjax',       # renders the LaTeX in your docstrings
    'myst_nb',                  # execute and render Jupyter notebooks
]

nb_execution_mode = "cache"     # re-execute only when notebook source changes
nb_execution_timeout = 600      # seconds per notebook

# Have autosummary build per-module rst stubs automatically.
autosummary_generate = True

# Default options for ``.. autoclass::`` / ``.. automodule::``.
autodoc_default_options = {
    'members':         True,    # include all documented members
    'undoc-members':   False,   # skip ones without docstrings
    'show-inheritance': True,
    'member-order':    'bysource',   # preserve the order in the .py file
}

# Napoleon: numpy style on, google style off (you use numpy).
napoleon_numpy_docstring  = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True  # render `Examples:` as a callout
napoleon_use_admonition_for_notes    = True  # render `Notes:` as a callout

# Intersphinx: clickable cross-references to numpy / python / numba docs.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable/', None),
    'numba':  ('https://numba.readthedocs.io/en/stable/', None),
}

templates_path   = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']


# -- HTML output -------------------------------------------------------------
html_theme        = 'furo'
html_static_path  = ['_static']
html_title        = f'{project} {release}'
