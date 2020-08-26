# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../"))

# imports for RTD


# specify the master doc, otherwise the build at read the docs fails
master_doc = "index"


# -- Project information -----------------------------------------------------

project = "q-optimize"

author = "Nicolas Wittler, \
        Federico Roy, \
        Kevin Pack, \
        Anurag Saha Roy, \
        Max Werninghaus, \
        Daniel J Egger, \
        Stefan Filipp, \
        Frank K Wilhelm, \
        Shai Machnes"
license = "Apache 2.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosummary",
    "autoapi.extension",
    "sphinx_rtd_theme",
    "sphinx.ext.graphviz",
]

# autoapi config
autoapi_dirs = ["../c3po"]
autoapi_file_patterns = ["*.py"]
autoapi_ignore = ["*logs*", "__pycache__"]
autoapi_member_order = "bysource"
autoapi_keep_files = True
autoapi_add_toctree_entry = True
autoapi_options = [
    "members",
    "inherited-members",
    "special-members",
    "show-inheritance",
    "special-members",
    "imported-members",
    "show-inheritance-diagram",
]

autosummary_generate = True

# Sort members by type
autodoc_member_order = "groupwise"


# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
