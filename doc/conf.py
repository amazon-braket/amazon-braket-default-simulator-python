"""Sphinx configuration."""
import datetime

import pkg_resources

# Sphinx configuration below.
project = "amazon-braket-default-simulator"
version = pkg_resources.require(project)[0].version
release = version
copyright = "{}, Amazon.com".format(datetime.datetime.now().year)

extensions = [
    "sphinxcontrib.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
]

source_suffix = ".rst"
master_doc = "index"

autoclass_content = "both"
autodoc_member_order = "bysource"
default_role = "py:obj"

html_theme = "sphinx_rtd_theme"
htmlhelp_basename = "{}doc".format(project)

napoleon_use_rtype = False

apidoc_module_dir = "../src/braket"
apidoc_output_dir = "_apidoc"
apidoc_excluded_paths = ["../test"]
apidoc_separate_modules = True
apidoc_module_first = True
apidoc_extra_args = ["-f", "--implicit-namespaces", "-H", "API Reference"]


# -- Options for MathJax output -------------------------------------------

mathjax_config = {
    "TeX": {
        "Macros": {
            "bra": [r"{\langle #1 |}", 1],
            "ket": [r"{| #1 \rangle}", 1],
            "expectation": [r"{\langle #1 \rangle_#2}", 2],
            "variance": [r"{\mathrm{Var}_#2 \left( #1 \right)}", 2],
        }
    }
}
