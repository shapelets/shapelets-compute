#! /usr/bin/env python3
# Copyright (c) 2021 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

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

file_loc = os.path.split(__file__)[0]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(file_loc), '.')))

import shapelets

project = 'Shapelets Compute'
copyright = '2021, Shapelets'
author = 'Shapelets'

# The full version, including alpha/beta/rc tags
version = shapelets.__version__
release = version

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx.ext.mathjax',  
    'sphinx_tabs.tabs'
]

intersphinx_mapping = {'python': ('https://docs.python.org/3/', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
                       'matplotlib': ('https://matplotlib.org/stable', None)}

imgmath_image_format = 'svg'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = '.rst'
autosectionlabel_prefix_document = True

master_doc = 'contents'

autodoc_type_aliases = {
    'ArrayLike': 'ArrayLike',
    'ShapeletsArray': 'ShapeletsArray',
    'DataTypeLike': 'DataTypeLike',
    'ShapeLike': 'ShapeLike',
    'DistanceType': 'DistanceType',
    'NormType': 'NormType',
    'DeviceInfo': 'DeviceInfo',
    'DeviceMemory': 'DeviceMemory',
    'ShapeletsRandomEngine': 'ShapeletsRandomEngine'
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "logo_link": "https://shapelets.io",
    "icon_links_label": "Quick Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/shapelets/shapelets-compute",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/shapelets",
            "icon": "fab fa-twitter-square",
        },
        {
            "name": "LinkedIn",
            "url": "https://www.linkedin.com/company/shapelets",
            "icon": "fab fa-linkedin"
        }
    ],
    "favicons": [
      {
         "rel": "icon",
         "sizes": "256x256",
         "href": "https://shapelets.io/static/images/favicon.ico",
      },
      {
         "rel": "apple-touch-icon",
         "sizes": "256x256",
         "href": "https://shapelets.io/static/images/favicon.ico"
      },
   ],
   "show_prev_next": False,
   "use_edit_page_button": False,
   "navigation_with_keys": True,
   "show_toc_level": 1,
   "collapse_navigation": True,
   "navigation_depth": 2,
#    "navbar_align": "right",
#    "sidebarwidth" : 470,
#    "sidebar_includehidden" : True
   "google_analytics_id": "UA-124462518-1",
   # "**": ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]
   # "external_links": [
   #   {"name": "link-one-name", "url": "https://<link-one>"},
   #   {"name": "link-two-name", "url": "https://<link-two>"}
   # ]
}

html_title = "%s v%s Manual" % (project, version)
html_short_title = project
html_baseurl = "https://shapelets.io/docs/compute"
html_logo = '_static/shapeletslogo.png'
html_css_files = ['shapelets.css']
html_static_path = ['_static']

html_additional_pages = {
    'index': 'indexcontent.html',
}

html_last_updated_fmt = '%b %d, %Y'
html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = '.html'

plot_html_show_formats = False
plot_html_show_source_link = False
plot_rcparams = {
    'font.size': 20,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Lato',
    'savefig.pad_inches': 0.1
}

htmlhelp_basename = 'shapelets'
