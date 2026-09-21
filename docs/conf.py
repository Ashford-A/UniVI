"""Sphinx configuration for the UniVI documentation (Read the Docs)."""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # document the source tree without installing heavy dependencies

project = "UniVI"
author = "Andrew J. Ashford and contributors"
copyright = "2026, Andrew J. Ashford and contributors"
release = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
                    re.MULTILINE).group(1)
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
]

master_doc = "index"
exclude_patterns = ["_build", "**.ipynb_checkpoints", "requirements.txt"]

# --- MyST / notebooks -------------------------------------------------------
# Notebooks are rendered with whatever outputs are saved in them; Read the Docs
# never executes them (no GPU, no data download).
nb_execution_mode = "off"
nb_merge_streams = True
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath", "attrs_inline"]
myst_heading_anchors = 3

# --- API reference -----------------------------------------------------------
# Heavy runtime dependencies are mocked so the docs build without PyTorch.
autodoc_mock_imports = [
    "torch", "scanpy", "anndata", "sklearn", "scipy", "h5py", "igraph", "leidenalg",
    "joblib", "matplotlib", "seaborn", "tqdm", "openpyxl", "yaml",
]
autosummary_generate = True
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": False}
napoleon_numpy_docstring = True
napoleon_google_docstring = True
suppress_warnings = ["myst.header", "autosummary.import_cycle"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}
if os.environ.get("UNIVI_DOCS_OFFLINE"):
    intersphinx_mapping = {}

# --- HTML -------------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "UniVI"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "repository_url": "https://github.com/Ashford-A/UniVI",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
    "show_toc_level": 2,
    "logo": {"text": "UniVI"},
}
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
