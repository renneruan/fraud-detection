# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Fraud Detection"
copyright = "2025, Rennê Oliveira"
author = "Rennê Oliveira"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []

language = "pt_BR"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


def skip_data_class_members(app, what, name, obj, skip, options):
    # Skip dataclass attributes, including class variables
    if hasattr(obj, "__dataclass_fields__") and what == "attribute":
        return True  # Skip the attribute from documentation

    # You can add other conditions for specific modules if needed
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_data_class_members)
