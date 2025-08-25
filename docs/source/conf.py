import os
import sys

sys.path.insert(0, os.path.abspath('../../'))  # so Sphinx finds bahamas

def setup(app):    
    app.add_css_file("favicon_switcher.css")

project = 'bahamas'
author = 'Federico Pozzoli'
release = '0.1.2'

extensions = [
    'myst_parser',  # Add myst_parser extension
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # if you use NumPy or Google-style docstrings
]

html_theme = 'furo'
html_static_path = ['_static']
html_logo = "_static/bahamas_logo.jpg"

# Optional: configure myst_parser
myst_enable_extensions = ["amsmath", "dollarmath"]  # Enable any extensions you need


