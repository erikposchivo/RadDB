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
import shutil
import sys
import inspect
import raddb

# sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.join(os.path.abspath("../.."), "raddb"))
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# -- Project information -----------------------------------------------------

project = "raddb"
copyright = "Gionata Ghiggi"
author = "Gionata Ghiggi"


# -- Copy Jupyter Notebook Tutorials -----------------------------------------
_source_dir = os.path.abspath(os.path.dirname(__file__))
_root_dir = os.path.dirname(os.path.dirname(_source_dir))
_tutorials_dir = os.path.join(_source_dir, "tutorials")
os.makedirs(_tutorials_dir, exist_ok=True)
for _filename in [
    "01_archiving.ipynb",
    "02_opening_and_filtering.ipynb",
    "03_area_of_interest.ipynb",
    "04_plots.ipynb",
    "05_demo_pipeline.ipynb",
]:
    shutil.copyfile(os.path.join(_root_dir, "tutorial", _filename), os.path.join(_tutorials_dir, _filename))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    # "sphinx_design",
    # "sphinx_gallery.gen_gallery",
    # "sphinx.ext.autosectionlabel",
    "sphinx_mdinclude",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "myst_parser",
    "nbsphinx",
    "sphinxcontrib.youtube",
]

# Set up mapping for other projects' docs
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "xradar": ("https://docs.openradarscience.org/projects/xradar/en/stable/", None),
    "pyart": ("https://arm-doe.github.io/pyart/", None),
    "fsspec": ("https://filesystem-spec.readthedocs.io/en/stable/", None),
}
always_document_param_types = True

# Warn when a reference is not found in docstrings
nitpicky = True
nitpick_ignore = [
    ("py:class", "optional"),
    ("py:class", "array-like"),
    ("py:class", "file-like object"),
    # For traitlets docstrings
    ("py:class", "All"),
    ("py:class", "t.Any"),
    ("py:class", "t.Iterable"),
]
nitpick_ignore_regex = [
    ("py:class", r".*[cC]allable"),
]

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# For a class, combine class and __init__ docstrings
autoclass_content = "both"

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Render the notebooks as archived: never execute them at build time.
nbsphinx_execute = "never"

# # Controlling automatically generating summary tables in the docs
# autosummary_generate = True
# autosummary_ignore_module_all = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_title = "RadDB"
html_theme_options = {
    "repository_url": "https://github.com/ltelab/raddb",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_edit_page_button": True,
    # "use_source_button": True,
    "use_issues_button": True,
    # "use_repository_button": True,
    "use_download_button": True,
    # "use_sidenotes": True,
    "show_toc_level": 2,
    "navigation_with_keys": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]


# -- Automatically run apidoc to generate rst from code ----------------------
# https://github.com/readthedocs/readthedocs.org/issues/1139
def run_apidoc(_):
    from sphinx.ext.apidoc import main

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    cur_dir = os.path.abspath(os.path.dirname(__file__))

    module_dir = os.path.join(cur_dir, "..", "..", "raddb")
    output_dir = os.path.join(cur_dir, "api")
    exclude = [os.path.join(module_dir, "tests")]
    main(["-f", "-o", output_dir, module_dir, *exclude])


def setup(app):
    app.connect("builder-inited", run_apidoc)


# Function to resolve source code links for `linkcode`
# adapted from NumPy, Pandas implementations
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        try:  # property
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        except (AttributeError, TypeError):
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        try:  # property
            source, lineno = inspect.getsourcelines(obj.fget)
        except (AttributeError, TypeError):
            lineno = None
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(raddb.__file__))

    if "+" in raddb.__version__:
        return f"https://github.com/ltelab/raddb/blob/main/raddb/{fn}{linespec}"
    else:
        return f"https://github.com/ltelab/raddb/blob/" f"v{raddb.__version__}/raddb/{fn}{linespec}"
