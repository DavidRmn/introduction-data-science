# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'idstools'
copyright = '2024, DavidRmn'
author = 'DavidRmn'

import os
import sys
sys.path.insert(0, os.path.abspath('../src/idstools'))
sys.path.insert(0, os.path.abspath('../src/idstools/_config'))
sys.path.insert(0, os.path.abspath('../src/idstools/_data_models'))
sys.path.insert(0, os.path.abspath('../src/idstools/_helpers'))
sys.path.insert(0, os.path.abspath('../src/idstools/_objects'))
sys.path.insert(0, os.path.abspath('../src/idstools/_transformer'))
sys.path.insert(0, os.path.abspath('../src/idstools/data_explorer'))
sys.path.insert(0, os.path.abspath('../src/idstools/data_preparation'))
sys.path.insert(0, os.path.abspath('../src/idstools/model_optimization'))
sys.path.insert(0, os.path.abspath('../src/idstools/wrapper'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'myst_parser']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'classic'
html_static_path = ['_static']