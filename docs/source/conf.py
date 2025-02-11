# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import pathlib


# this path is pointing to project/docs/source
CURRENT_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
GARPAR_PATH = CURRENT_PATH.parent.parent

sys.path.insert(0, str(GARPAR_PATH))

import garpar


# -- Project information -----------------------------------------------------

project = 'Garpar'
copyright = '2020, QuatroPe'
author = 'QuatroPe'

# The full version, including alpha/beta/rc tags
release = garpar.__version__
version = garpar.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'nbsphinx']

exclude_patterns = ['_build', 'source/.ipynb_checkpoints/*']

numpydoc_class_members_toctree = False

nbsphinx_execute = 'never'



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}

# =============================================================================
# PREPROCESS RST
# =============================================================================

html_logo = "_static/logo.png"

html_css_files = [
    'custom.css',
]

html_theme_options = {}


# =============================================================================
# INJECT README INTO THE RESTRUCTURED TEXT
# =============================================================================

import m2r2

DYNAMIC_RST = {
    "README.md": "README.rst",
    "CHANGELOG.md": "CHANGELOG.rst",
}

for md_name, rst_name in DYNAMIC_RST.items():
    md_path = GARPAR_PATH / md_name
    with open(md_path) as fp:
        readme_md = fp.read().split("<!-- BODY -->", 1)[-1]

    rst_path = CURRENT_PATH / "_dynamic" / rst_name

    with open(rst_path, "w") as fp:
        fp.write(".. FILE AUTO GENERATED !! \n")
        fp.write(m2r2.convert(readme_md))
        print(f"{md_path} -> {rst_path} regenerated!")
