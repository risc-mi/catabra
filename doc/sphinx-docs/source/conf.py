# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from urllib.parse import quote

sys.path.insert(0, os.path.abspath('../../../catabra/'))
sys.path.insert(0, os.path.abspath('_ext'))

project = 'CaTabRa'
copyright = '2023, RISC Software GmbH'
author = 'RISC Software GmbH'
github_url = 'https://github.com/risc-mi/catabra/'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
    'myst_parser',
    'nbsphinx',
    'nbsphinx_link',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# ----------------------------------------------------------------------------------------------------------------------
# -- autodoc configurations --------------------------------------------------------------------------------------------
add_module_names = False
autodoc_member_order = 'bysource'


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = quote(info['module'].replace('.', '/'))
    filename = 'catabra/' + filename
    if "fullname" in info:
        anchor = info["fullname"]
        anchor = "#:~:text=" + quote(anchor.split(".")[-1])
    else:
        anchor = ""

    # github
    result = github_url + "blob/main/%s.py%s" % (filename, anchor)
    return result
